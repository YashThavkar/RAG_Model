from __future__ import annotations

"""
Offline checks against data/eval/test_queries.json.

Not a gold benchmark — lightweight signals: did we pull the right PDF (hit@k),
do reference keywords appear in top chunks, rough overlap between answer and refs,
and a naive “unsupported words” style hallucination hint.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.chunking import load_chunks_jsonl
from app.generator import generate_answer
from app.retriever import BM25Retriever, get_query_retriever
from app.text_utils import flatten_for_output


def _query_refs(q: dict[str, Any]) -> str:
    return str(q.get("refs") or q.get("gold_answer_hint") or "")


def load_test_queries(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = path or (config.DATA_EVAL / "test_queries.json")
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"] if isinstance(data, dict) else data


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def retrieval_hit_at_k(
    retrieved: list[dict[str, Any]],
    expected_source: str,
    k: int,
) -> int:
    top = retrieved[:k]
    for r in top:
        if r.get("source_file") == expected_source:
            return 1
    return 0


def chunk_refs_match(retrieved: list[dict[str, Any]], refs: str) -> int:
    if not refs:
        return 0
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]{4,}", refs.lower()) if len(t) > 4]
    if not tokens:
        return 0
    joined = " ".join(_normalize(c.get("chunk_text", "")) for c in retrieved[:5])
    hits = sum(1 for t in tokens if t in joined)
    return 1 if hits >= max(1, len(tokens) // 3) else 0


def score_answer_correctness(answer: str, refs: str, context_text: str) -> int:
    a = _normalize(answer)
    if "not available in the provided" in a or "not available in the pr" in a:
        return 0 if refs else 1
    if a.startswith("relevant excerpt"):
        return 1
    h = _normalize(refs)
    ctx = _normalize(context_text)
    ref_words = set(re.findall(r"[a-z]{5,}", h))
    answer_words = set(re.findall(r"[a-z]{5,}", a))
    overlap = ref_words & answer_words
    in_ctx = sum(1 for w in answer_words if w in ctx and len(w) > 5)
    if len(overlap) >= 3 and in_ctx >= 4:
        return 2
    if len(overlap) >= 1 or in_ctx >= 2:
        return 1
    return 0


def score_hallucination(answer: str, context_text: str) -> int:
    a = _normalize(answer)
    if "not available" in a or a.startswith("relevant excerpt"):
        return 0
    ctx = _normalize(context_text)
    numbers = set(re.findall(r"\b\d{3,}\b", answer))
    ctx_nums = set(re.findall(r"\b\d{3,}\b", context_text))
    stray_nums = numbers - ctx_nums
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    unsupported = 0
    for sent in sentences[:6]:
        sn = _normalize(sent)
        words = [w for w in re.findall(r"[a-z]{6,}", sn) if w not in ("because", "however", "therefore")]
        if not words:
            continue
        ok = sum(1 for w in words if w in ctx)
        if ok < max(1, len(words) // 4):
            unsupported += 1
    if unsupported >= 3 or len(stray_nums) >= 2:
        return 2
    if unsupported >= 1 or len(stray_nums) >= 1:
        return 1
    return 0


def evaluate_retrieval(
    test_queries: list[dict[str, Any]],
    retriever_fn,
    k: int | None = None,
) -> list[dict[str, Any]]:
    k = k or config.TOP_K
    rows: list[dict[str, Any]] = []
    for q in test_queries:
        query = q["query"]
        expected = q.get("expected_source", "")
        retrieved = retriever_fn(query, top_k=k)
        hit = retrieval_hit_at_k(retrieved, expected, k)
        rhit = chunk_refs_match(retrieved, _query_refs(q))
        rows.append(
            {
                "query": query,
                "expected_source": expected,
                "retrieval_hit_at_k": hit,
                "refs_in_top_chunks": rhit,
                "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
                "scores": [r.get("score") for r in retrieved],
            }
        )
    return rows


def evaluate_generation(
    test_queries: list[dict[str, Any]],
    retrieve_fn,
    k: int | None = None,
) -> list[dict[str, Any]]:
    k = k or config.TOP_K
    rows: list[dict[str, Any]] = []
    for q in test_queries:
        query = q["query"]
        refs = _query_refs(q)
        retrieved = retrieve_fn(query, top_k=k)
        ctx = "\n".join(c.get("chunk_text", "") for c in retrieved)
        answer = flatten_for_output(generate_answer(query, retrieved))
        corr = score_answer_correctness(answer, refs, ctx)
        hall = score_hallucination(answer, ctx)
        rows.append(
            {
                "query": query,
                "answer": answer,
                "answer_correctness": corr,
                "hallucination": hall,
                "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
            }
        )
    return rows


def summarize_results(
    retrieval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    label: str = "vector",
) -> dict[str, Any]:
    n = max(len(retrieval_rows), 1)
    hit = sum(r["retrieval_hit_at_k"] for r in retrieval_rows) / n
    rhit = sum(r.get("refs_in_top_chunks", 0) for r in retrieval_rows) / n
    gen_n = len(generation_rows)
    out: dict[str, Any] = {
        "label": label,
        "retrieval_hit_at_k_mean": round(hit, 4),
        "refs_match_rate": round(rhit, 4),
        "n_queries": n,
    }
    if gen_n:
        out["answer_correctness_mean_0_2"] = round(
            sum(r["answer_correctness"] for r in generation_rows) / gen_n, 4
        )
        out["hallucination_mean_0_2"] = round(
            sum(r["hallucination"] for r in generation_rows) / gen_n, 4
        )
    return out


def run_full_evaluation(
    test_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare primary retriever vs BM25-only; write JSON summary to outputs/eval_results.json."""
    test_path = test_path or (config.DATA_EVAL / "test_queries.json")
    output_path = output_path or config.EVAL_RESULTS
    queries = load_test_queries(test_path)
    chunks = load_chunks_jsonl()
    if not chunks:
        raise RuntimeError("No chunks found. Run build_index first.")

    qret = get_query_retriever()

    def vec(q: str, top_k: int | None = None):
        return qret.retrieve_with_scores(q, top_k=top_k)

    bret = BM25Retriever(chunks)

    def bm25(q: str, top_k: int | None = None):
        return bret.retrieve_with_scores(q, top_k=top_k)

    r_vec = evaluate_retrieval(queries, vec)
    r_bm25 = evaluate_retrieval(queries, bm25)
    g_vec = evaluate_generation(queries, vec)

    primary_label = "hybrid" if config.USE_HYBRID_RETRIEVAL else "vector"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vector_retrieval": summarize_results(r_vec, g_vec, primary_label),
        "bm25_retrieval_only": summarize_results(r_bm25, [], "bm25"),
        "per_query": [
            {
                "query": q["query"],
                "vector_hit": rv["retrieval_hit_at_k"],
                "bm25_hit": rb["retrieval_hit_at_k"],
                "answer_correctness": gv["answer_correctness"],
                "hallucination": gv["hallucination"],
            }
            for q, rv, rb, gv in zip(
                queries, r_vec, r_bm25, g_vec, strict=False
            )
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
