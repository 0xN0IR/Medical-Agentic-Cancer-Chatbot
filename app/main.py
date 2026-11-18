import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chat_history.db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def init_db() -> None:
    """Create the SQLite table for multi-turn conversation memory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tools_used TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages (session_id, id)"
        )
        conn.commit()


def load_session_history(session_id: str, limit: int = 10) -> list[BaseMessage]:
    """Load recent conversation turns as LangChain messages."""
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) sub ORDER BY id ASC
            """,
            (session_id, limit),
        ).fetchall()

    messages: list[BaseMessage] = []
    for role, content in rows:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def save_turn(session_id: str, user_msg: str, assistant_msg: str, tools_used: list[str]) -> None:
    """Save a completed user + assistant exchange into SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, tools_used) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_msg, ""),
        )
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, tools_used) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", assistant_msg, ",".join(tools_used)),
        )
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Cancer Support & Research Agent API",
    description="Agentic Discussion-First RAG backend for Breast Cancer Community & Oncology Support",
    version="1.0.0",
    lifespan=lifespan,
)

origins = (
    ["*"]
    if settings.allowed_origins.strip() == "*"
    else [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None


def verify_api_token(authorization: str | None = Header(default=None)) -> None:
    """Optional Bearer token check when API_TOKEN is configured in .env."""
    configured = settings.api_token.strip()
    if not configured or configured == "change-this-to-a-secret-token-for-your-app":
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != configured:
        raise HTTPException(status_code=403, detail="Invalid API token.")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tools_used: list[str]


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "chat_model": settings.gemini_chat_model,
        "embedding_model": settings.gemini_embedding_model,
        "pinecone_index": settings.pinecone_index_name,
    }


@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    if templates and (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse("<h2>templates/index.html not found. Visit <a href='/docs'>/docs</a>.</h2>")

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_token)])
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_token)])
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    session_id = (payload.session_id or "").strip() or str(uuid.uuid4())
    history = load_session_history(session_id=session_id, limit=10)

    try:
        result = run_agent(user_query=payload.message.strip(), chat_history=history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    save_turn(
        session_id=session_id,
        user_msg=payload.message.strip(),
        assistant_msg=result["answer"],
        tools_used=result["tools_used"],
    )

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        tools_used=result["tools_used"],
    )


@app.get("/api/history/{session_id}", dependencies=[Depends(verify_api_token)])
def get_history(session_id: str) -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"session_id": session_id, "messages": []}
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content, tools_used, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": role,
                "content": content,
                "tools_used": [t for t in tools.split(",") if t],
                "created_at": created_at,
            }
            for role, content, tools, created_at in rows
        ],
    }


@app.delete("/api/history/{session_id}", dependencies=[Depends(verify_api_token)])
def clear_history(session_id: str) -> dict[str, str]:
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.commit()
    return {"status": "cleared", "session_id": session_id}