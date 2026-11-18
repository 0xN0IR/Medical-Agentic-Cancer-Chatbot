from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

from app.config import settings
from app.knowledge import search_community_kb

SYSTEM_PROMPT = """You are a compassionate, evidence-aware Cancer Support & Research Assistant.

Your knowledge architecture has two complementary tools:
1. `search_community_discussions`: Searches a curated peer-support knowledge base of real Breast Cancer community discussions and replies. Call this first for breast cancer questions, symptoms, treatments, side effects, or general cancer support queries where breast cancer may apply.
2. `search_medical_web`: Searches trusted medical and oncology sources on the web (via Tavily). Call this tool DIRECTLY (without calling `search_community_discussions` first) whenever the user explicitly asks about a non-breast cancer type (e.g., lung, colorectal, prostate, ovarian, leukemia, lymphoma), or as a fallback when `search_community_discussions` returns `NO_RELEVANT_COMMUNITY_MATCHES`.

Response & Safety Rules:
- Clearly distinguish between **peer patient experiences** (from the community discussions) and **general medical information** (from trusted clinical web sources).
- Cite the discussion titles/IDs or web sources you relied on so the user can explore further.
- Never diagnose a condition, prescribe medication, or tell a user to start/stop a treatment.
- Always remind the user to consult their oncologist or healthcare team for personalized medical decisions.
- Be warm, clear, structured, and easy to read on both web and mobile screens."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def search_community_discussions(query: str) -> str:
    """
    Search the Breast Cancer Community Knowledge Base (discussions and peer replies).
    Use this first for breast cancer topics, patient experiences, side effects, and peer support.
    """
    result = search_community_kb(query=query, top_k_discussions=3, top_k_replies=6)
    if not result["has_sufficient_context"]:
        return (
            "NO_RELEVANT_COMMUNITY_MATCHES: The breast cancer community database did not have "
            "a strong match for this query (best similarity score: "
            f"{result['best_score']:.2f}). Use `search_medical_web` instead."
        )

    sources_summary = "\n".join(
        f"- [{s['question_id']}] {s['title']} ({s['category']}) -> {s['source_url']}"
        for s in result["sources"]
    )
    return (
        f"MATCHED COMMUNITY DISCUSSIONS:\n{sources_summary}\n\n"
        f"FULL THREAD CONTEXT:\n{result['context_text']}"
    )


@tool
def search_medical_web(query: str) -> str:
    """
    Search trusted medical and oncology sources on the web using Tavily.
    Use this directly for non-breast cancer types (lung, colon, prostate, blood cancers, etc.),
    official clinical guidelines, or when community discussions lack sufficient context.
    """
    if not settings.tavily_api_key:
        return "WEB_SEARCH_UNAVAILABLE: TAVILY_API_KEY is not configured in .env."

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=f"{query} oncology cancer medical information",
            search_depth="basic",
            max_results=4,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return f"WEB_SEARCH_ERROR: Tavily search failed ({exc}). Answer using general established medical knowledge with appropriate caveats."

    results = response.get("results", [])
    if not results:
        return "NO_WEB_RESULTS: No relevant web articles found."

    formatted = []
    for idx, item in enumerate(results, start=1):
        formatted.append(
            f"[{idx}] {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Summary: {item.get('content', '')}"
        )
    return "\n\n".join(formatted)

TOOLS = [search_community_discussions, search_medical_web]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _build_llm():
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in your .env file.")
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.gemini_api_key,
        thinking_level="low",
        timeout=20,
        max_retries=1,
    )
    return llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Invoke Gemini with bound community + medical web tools."""
    llm_with_tools = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Execute any tool calls requested by Gemini."""
    last_message = state["messages"][-1]
    tool_outputs: list[BaseMessage] = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                output = f"Error: Unknown tool '{call['name']}'."
            else:
                output = tool_fn.invoke(call["args"])
            tool_outputs.append(
                ToolMessage(content=str(output), tool_call_id=call["id"])
            )

    return {"messages": tool_outputs}


def should_continue(state: AgentState) -> str:
    """Route to 'tools' if Gemini requested a tool call, otherwise finish."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_agent_graph():
    """Compile the LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


agent_graph = build_agent_graph()


def run_agent(user_query: str, chat_history: list[BaseMessage] | None = None) -> dict[str, Any]:
    """
    Run a single user turn through the LangGraph agent and return:
      - answer: final markdown response
      - tools_used: list of tools invoked during reasoning
    """
    messages: list[BaseMessage] = list(chat_history) if chat_history else []
    messages.append(HumanMessage(content=user_query))

    final_state = agent_graph.invoke({"messages": messages})
    all_messages = final_state["messages"]

    tools_used: list[str] = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call["name"] in TOOLS_BY_NAME and call["name"] not in tools_used:
                    tools_used.append(call["name"])

    final_answer = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Handle string or list-of-blocks content from Gemini
            if isinstance(msg.content, str):
                final_answer = msg.content
            elif isinstance(msg.content, list):
                final_answer = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in msg.content
                )
            break

    return {
        "answer": final_answer,
        "tools_used": tools_used,
    }