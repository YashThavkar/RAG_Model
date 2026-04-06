from __future__ import annotations

"""
Glue between stages: ingest/index once, answer queries many times.

Scripts and the API should call into here instead of re-wiring loader → chunk → embed by hand.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.chunking import chunk_documents, save_chunks_jsonl
from app.text_utils import flatten_for_output
from app.embeddings import build_vector_index, get_model, save_vector_index
from app.evaluator import run_full_evaluation
from app.generator import generate_answer
from app.loader import default_raw_paths, load_documents
from app.preprocess import preprocess_documents
from app.retriever import get_query_retriever


def ingest_and_index(
    raw_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Full rebuild: raw PDFs → cleaned units → chunks.jsonl → FAISS + meta.json."""
    raw_paths = raw_paths or default_raw_paths()
    missing = [p for p in raw_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw documents: "
            + ", ".join(str(p) for p in missing)
            + " — run setup_data.py or copy PDFs into data/raw/."
        )

    units = load_documents(raw_paths)
    cleaned = preprocess_documents(units)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with (config.DATA_PROCESSED / "cleaned_units.jsonl").open("w", encoding="utf-8") as f:
        for u in cleaned:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    chunks = chunk_documents(cleaned)
    save_chunks_jsonl(chunks)

    model = get_model()
    index, meta_chunks = build_vector_index(chunks, model=model)
    save_vector_index(index, meta_chunks)

    return {
        "num_units": len(cleaned),
        "num_chunks": len(chunks),
        "chunks_path": str(config.CHUNKS_JSONL),
        "index_path": str(config.FAISS_INDEX_FILE),
    }


def answer_query(
    query: str,
    top_k: int | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end query: retrieve → generate (raw newlines in prompt) → flatten for API/log."""
    top_k = top_k or config.TOP_K
    retriever = get_query_retriever()
    retrieved = retriever.retrieve_with_scores(query, top_k=top_k)
    # generate_answer sees original chunk_text; users see flattened strings below.
    answer = flatten_for_output(generate_answer(query, retrieved))

    retrieved_out = [
        {**r, "chunk_text": flatten_for_output(r.get("chunk_text", ""))} for r in retrieved
    ]

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
        "retrieved_locations": [
            {
                "source_file": r.get("source_file"),
                "page_number": r.get("page_number"),
                "chunk_id": r.get("chunk_id"),
            }
            for r in retrieved
        ],
        "source_files": list({r["source_file"] for r in retrieved}),
        "scores": [r.get("score") for r in retrieved],
        "retrieval": "hybrid" if config.USE_HYBRID_RETRIEVAL else "vector",
        "answer": answer,
    }
    log_path = log_path or config.QUERY_LOGS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Response payload for HTTP/CLI: human-friendly previews + full flattened chunks.
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "source_file": r["source_file"],
                "page_number": r.get("page_number"),
                "section_heading": r.get("section_heading"),
                "score": r.get("score"),
                "preview": r["chunk_text"][:280]
                + ("…" if len(r["chunk_text"]) > 280 else ""),
            }
            for r in retrieved_out
        ],
        "retrieved": retrieved_out,
    }


def run_evaluation() -> dict[str, Any]:
    return run_full_evaluation()
