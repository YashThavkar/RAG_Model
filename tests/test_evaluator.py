from __future__ import annotations

"""Pure helper tests for hit@k and heuristic scoring (no live retrieval)."""

from app.evaluator import (
    retrieval_hit_at_k,
    score_answer_correctness,
    score_hallucination,
)


def test_retrieval_hit_at_k():
    r = [
        {"source_file": "a.pdf", "chunk_id": "1"},
        {"source_file": "b.docx", "chunk_id": "2"},
    ]
    assert retrieval_hit_at_k(r, "b.docx", k=2) == 1
    assert retrieval_hit_at_k(r, "c.pdf", k=2) == 0


def test_answer_scoring():
    ctx = "The model uses self-attention and feed-forward layers."
    ans = "The architecture relies on self-attention mechanisms."
    assert score_answer_correctness(ans, "self-attention feed-forward", ctx) >= 1
    assert score_hallucination(ans, ctx) <= 1
