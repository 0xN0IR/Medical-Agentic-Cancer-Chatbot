import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

DISCUSSIONS_NAMESPACE = "discussions"
REPLIES_NAMESPACE = "replies"

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@lru_cache(maxsize=1)
def get_clients() -> tuple[GoogleGenerativeAIEmbeddings, Pinecone]:
    """Create and cache Gemini Embeddings and Pinecone clients."""
    if not settings.gemini_api_key or not settings.pinecone_api_key:
        raise ValueError(
            "GEMINI_API_KEY and PINECONE_API_KEY must be set in your .env file."
        )
    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return embedder, pinecone_client


def ensure_index():
    """Create the Pinecone serverless index automatically if it does not exist yet."""
    _, pc = get_clients()
    index_name = settings.pinecone_index_name

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into 1536-dim Gemini embeddings."""
    embedder, _ = get_clients()
    return embedder.embed_documents(texts, output_dimensionality=1536)

@lru_cache(maxsize=1)
def load_local_store() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Load all local processed JSON files into lookup dictionaries:
      - discussions_by_id[question_id] -> full discussion dict
      - replies_by_id[answer_id] -> full reply dict (with parent question_id)
    Full patient text stays local and is never stored in Pinecone metadata.
    """
    discussions_by_id: dict[str, dict[str, Any]] = {}
    replies_by_id: dict[str, dict[str, Any]] = {}

    if not PROCESSED_DATA_DIR.exists():
        return discussions_by_id, replies_by_id

    for json_file in sorted(PROCESSED_DATA_DIR.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            records = json.load(f)

        for disc in records:
            qid = disc["question_id"]
            discussions_by_id[qid] = {
                "question_id": qid,
                "category": disc.get("category", ""),
                "discussion_title": disc.get("discussion_title", ""),
                "source_url": disc.get("source_url", ""),
                "original_post_text": disc.get("original_post", {}).get("text", ""),
                "answers_count": disc.get("answers_count", 0),
                "local_file": json_file.name,
            }
            for ans in disc.get("answers", []):
                aid = ans["answer_id"]
                replies_by_id[aid] = {
                    "answer_id": aid,
                    "question_id": qid,
                    "text": ans.get("text", ""),
                    "local_file": json_file.name,
                }

    return discussions_by_id, replies_by_id


def search_community_kb(
    query: str,
    top_k_discussions: int = 3,
    top_k_replies: int = 6,
    min_score: float = 0.30,
) -> dict[str, Any]:
    """
    Discussion-First RAG Retrieval:
      1. Search the 'discussions' namespace for relevant parent threads.
      2. Search the 'replies' namespace filtered strictly to those parent question_ids.
      3. Hydrate full patient stories and source URLs from local JSON files.
    """
    index = ensure_index()
    discussions_by_id, replies_by_id = load_local_store()

    query_vector = embed_texts([query])[0]

    # Step 1: Retrieve relevant original discussions
    disc_results = index.query(
        namespace=DISCUSSIONS_NAMESPACE,
        vector=query_vector,
        top_k=top_k_discussions * 2,  # over-fetch slightly in case of multi-chunk threads
        include_metadata=True,
    )

    matched_qids: list[str] = []
    best_disc_score = 0.0
    for match in disc_results.get("matches", []):
        score = float(match.get("score", 0.0))
        best_disc_score = max(best_disc_score, score)
        best_disc_score = score
        if score < min_score:
            continue
        qid = match.get("metadata", {}).get("question_id")
        if qid and qid not in matched_qids:
            matched_qids.append(qid)
        if len(matched_qids) >= top_k_discussions:
            break

    if not matched_qids:
        return {
            "context_text": "",
            "sources": [],
            "has_sufficient_context": False,
            "best_score": best_disc_score,
        }

    # Step 2: Retrieve relevant replies ONLY from those matched discussions
    reply_results = index.query(
        namespace=REPLIES_NAMESPACE,
        vector=query_vector,
        top_k=top_k_replies * 2,
        filter={"question_id": {"$in": matched_qids}},
        include_metadata=True,
    )

    replies_by_parent: dict[str, list[dict[str, Any]]] = {qid: [] for qid in matched_qids}
    seen_reply_ids: set[str] = set()

    for match in reply_results.get("matches", []):
        meta = match.get("metadata", {})
        qid = meta.get("question_id")
        aid = meta.get("reply_id")
        if not qid or not aid or aid in seen_reply_ids:
            continue
        if qid in replies_by_parent and aid in replies_by_id:
            seen_reply_ids.add(aid)
            replies_by_parent[qid].append(replies_by_id[aid])
            if len(seen_reply_ids) >= top_k_replies:
                break

    # Step 3: Assemble grounded context & structured citations from local JSON
    context_blocks: list[str] = []
    sources: list[dict[str, str]] = []

    for qid in matched_qids:
        disc = discussions_by_id.get(qid)
        if not disc:
            continue

        sources.append(
            {
                "question_id": qid,
                "title": disc["discussion_title"],
                "category": disc["category"],
                "source_url": disc["source_url"],
            }
        )

        block_lines = [
            f"=== DISCUSSION [{qid}] ({disc['category']}) ===",
            f"Title: {disc['discussion_title']}",
            f"Original Post: {disc['original_post_text']}",
            "Relevant Community Replies:",
        ]

        thread_replies = replies_by_parent.get(qid, [])
        if thread_replies:
            for r in thread_replies:
                block_lines.append(f"- [{r['answer_id']}]: {r['text']}")
        else:
            block_lines.append("- (No specific replies matched above threshold)")

        context_blocks.append("\n".join(block_lines))

    return {
        "context_text": "\n\n".join(context_blocks),
        "sources": sources,
        "has_sufficient_context": len(sources) > 0,
        "best_score": best_disc_score,
    }