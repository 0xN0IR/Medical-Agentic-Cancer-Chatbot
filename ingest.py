import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from app.knowledge import (
    DISCUSSIONS_NAMESPACE,
    PROCESSED_DATA_DIR,
    REPLIES_NAMESPACE,
    embed_texts,
    ensure_index,
)

# Regex patterns to mask remaining external URLs and off-platform contact prompts
URL_RE = re.compile(r"https?://[^\s<>\"')]+|www\.[^\s<>\"')]+", re.IGNORECASE)
PM_SOLICIT_RE = re.compile(
    r"\b(?:please\s+)?(?:send\s+me\s+a\s+)?(?:pm|private\s+message|dm|email\s+me|message\s+me|call\s+me|text\s+me)\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)


def sanitize_for_embedding(text: str) -> str:
    """
    Mask external URLs and off-platform contact solicitations before sending text
    to OpenAI's embedding API. Local JSON files on disk remain untouched.
    """
    cleaned = URL_RE.sub("[url redacted]", text)
    cleaned = PM_SOLICIT_RE.sub("[contact solicitation redacted]", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Split long posts/replies into overlapping character chunks at word boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start + 200:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_records(limit_discussions: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Prepare discussion and reply records with sanitized embedding text and
    minimal, privacy-safe Pinecone metadata (zero raw patient text in metadata).
    """
    discussion_records: list[dict[str, Any]] = []
    reply_records: list[dict[str, Any]] = []
    processed_count = 0

    for json_file in sorted(PROCESSED_DATA_DIR.glob("*.json")):
        category_id = json_file.name.split("_", 1)[0]  # e.g., "01" -> "cat01"
        cat_code = f"cat{category_id}"

        with json_file.open("r", encoding="utf-8") as f:
            discussions = json.load(f)

        for disc in discussions:
            if limit_discussions is not None and processed_count >= limit_discussions:
                return discussion_records, reply_records

            qid = disc["question_id"]
            category_name = disc.get("category", "")
            title = disc.get("discussion_title", "").strip()
            raw_post = disc.get("original_post", {}).get("text", "").strip()
            answers = disc.get("answers", [])
            answers_count = disc.get("answers_count", len(answers))
            is_truncated = answers_count >= 50  # Scraper ceiling documented in audit

            clean_title = sanitize_for_embedding(title)
            clean_post = sanitize_for_embedding(raw_post)
            full_disc_text = f"Category: {category_name} | Title: {clean_title} | Post: {clean_post}"

            disc_chunks = chunk_text(full_disc_text)
            for idx, chunk in enumerate(disc_chunks):
                discussion_records.append(
                    {
                        "id": f"{qid}_c{idx:02d}",
                        "text_to_embed": chunk,
                        "metadata": {
                            "record_type": "discussion",
                            "question_id": qid,
                            "category_id": cat_code,
                            "category_name": category_name,
                            "chunk_index": idx,
                            "total_chunks": len(disc_chunks),
                            "answers_count": int(answers_count),
                            "is_truncated": bool(is_truncated),
                            "local_file": json_file.name,
                        },
                    }
                )

            for ans in answers:
                aid = ans["answer_id"]
                raw_reply = ans.get("text", "").strip()
                clean_reply = sanitize_for_embedding(raw_reply)
                if not clean_reply:
                    continue

                # Prefix reply with parent thread title so short replies retain context
                reply_context_text = f"Thread: {clean_title} | Reply: {clean_reply}"
                reply_chunks = chunk_text(reply_context_text)

                for idx, chunk in enumerate(reply_chunks):
                    reply_records.append(
                        {
                            "id": f"{aid}_c{idx:02d}",
                            "text_to_embed": chunk,
                            "metadata": {
                                "record_type": "reply",
                                "question_id": qid,
                                "reply_id": aid,
                                "category_id": cat_code,
                                "category_name": category_name,
                                "chunk_index": idx,
                                "total_chunks": len(reply_chunks),
                                "local_file": json_file.name,
                            },
                        }
                    )

            processed_count += 1

    return discussion_records, reply_records


def upsert_batches(index, records: list[dict[str, Any]], namespace: str, batch_size: int = 40) -> None:
    """Embed text in rate-limit-safe batches (Gemini free tier = 100 items/min) and upsert to Pinecone."""
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        texts = [item["text_to_embed"] for item in batch]

        while True:
            try:
                vectors = embed_texts(texts)
                break
            except Exception as err:
                if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                    print("Rate limit reached (100/min free tier). Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    raise

        pinecone_payload = [
            {
                "id": item["id"],
                "values": vec,
                "metadata": item["metadata"],
            }
            for item, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=pinecone_payload, namespace=namespace)
        done = min(start + batch_size, total)
        print(f"[{namespace}] Upserted {done} / {total} chunks")

        if done < total:
            time.sleep(30)  # Stay safely under Gemini's 100 embeddings/minute free limit

def main():
    parser = argparse.ArgumentParser(description="Ingest Breast Cancer Community JSON into Pinecone.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of discussions to ingest (e.g., --limit 5 for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and count chunks locally without calling OpenAI or Pinecone APIs.",
    )
    args = parser.parse_args()

    disc_records, reply_records = build_records(limit_discussions=args.limit)
    print(f"Prepared {len(disc_records)} discussion chunks and {len(reply_records)} reply chunks.")

    if args.dry_run:
        print("Dry run complete — zero external API calls made.")
        return

    index = ensure_index()
    print(f"Connected to Pinecone index: {index}")

    upsert_batches(index, disc_records, namespace=DISCUSSIONS_NAMESPACE)
    upsert_batches(index, reply_records, namespace=REPLIES_NAMESPACE)
    print("Ingestion complete!")


if __name__ == "__main__":
    main()