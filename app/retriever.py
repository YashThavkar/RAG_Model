from __future__ import annotations

"""
Retrieval: dense (FAISS + bi-encoder), sparse (BM25), and hybrid fusion.

VectorRetriever embeds the query and does nearest-neighbor search.
BM25Retriever is a keyword baseline. HybridRetriever merges both rank lists (RRF),
optionally reranks the top pool with a cross-encoder, and boosts “phrase-shaped” queries
by scanning the corpus for ordered token matches (see lexical_phrase_chunk_ids).
"""

import re
from collections import defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.embeddings import embed_texts, get_model, load_vector_index

# Loaded on first rerank only — keeps import-time and vector-only paths light.
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
    return _cross_encoder


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


def _lexical_ordered_terms_match(
    query: str, text: str, *, max_span: int | None = None
) -> bool:
    """True if all query tokens appear in order within max_span characters (lowercased)."""
    max_span = max_span if max_span is not None else config.LEXICAL_PHRASE_MAX_SPAN
    terms = _tokenize(query)
    if len(terms) < 2:
        return False
    tl = text.lower()
    start = 0
    first_idx: int | None = None
    last_idx: int | None = None
    for t in terms:
        i = tl.find(t, start)
        if i < 0:
            return False
        if first_idx is None:
            first_idx = i
        last_idx = i + len(t)
        start = i + 1
    assert first_idx is not None and last_idx is not None
    return (last_idx - first_idx) <= max_span


def lexical_phrase_chunk_ids(query: str, chunks: list[dict[str, Any]]) -> list[str]:
    """
    Chunks where the query matches lexically in a phrase-like way (ordered tokens, short window).
    Catches 'track, document' vs BM25 on separate rare tokens; skips very short/generic queries.
    """
    if not config.USE_LEXICAL_PHRASE_PRIORITY:
        return []
    terms = _tokenize(query)
    out: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        t = c.get("chunk_text", "")
        ok = False
        if len(terms) >= 2 and sum(len(x) for x in terms) >= 8:
            ok = _lexical_ordered_terms_match(query, t)
        elif len(terms) == 1 and len(terms[0]) >= 8:
            ok = terms[0] in t.lower()
        if not ok:
            continue
        cid = c["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _merge_phrase_priority_vector(
    query: str,
    results: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    final_n: int,
) -> list[dict[str, Any]]:
    phrase_ids = lexical_phrase_chunk_ids(query, chunks)
    if not phrase_ids:
        return results[:final_n]
    by_id = {c["chunk_id"]: c for c in chunks}
    pid_set = set(phrase_ids)
    head: list[dict[str, Any]] = []
    for pid in phrase_ids:
        row = next((r for r in results if r["chunk_id"] == pid), None)
        if row is not None:
            r = row.copy()
        else:
            base = by_id.get(pid)
            if base is None:
                continue
            r = _chunk_row(base, 1.0)
        r["retrieval_mode"] = "lexical_phrase"
        r["score"] = max(float(r.get("score", 0.0)), 1.0)
        head.append(r)
    tail = [r for r in results if r["chunk_id"] not in pid_set]
    return (head + tail)[:final_n]


def _dedupe_results(
    results: list[dict[str, Any]], max_same_doc_streak: int = 3
) -> list[dict[str, Any]]:
    """Avoid returning the same passage many times when the index has near-duplicates."""
    out: list[dict[str, Any]] = []
    prev_text = None
    streak = 0
    for r in results:
        t = r.get("chunk_text", "")[:200]
        if t == prev_text:
            streak += 1
            if streak > max_same_doc_streak:
                continue
        else:
            streak = 1
            prev_text = t
        out.append(r)
    return out


def _chunk_row(c: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "chunk_id": c["chunk_id"],
        "score": score,
        "source_file": c["source_file"],
        "document_type": c["document_type"],
        "page_number": c.get("page_number"),
        "section_heading": c.get("section_heading"),
        "chunk_text": c["chunk_text"],
    }


class VectorRetriever:
    """Bi-encoder similarity search over the FAISS index built at ingest time."""

    def __init__(
        self,
        index_path=None,
        meta_path=None,
        model_name: str | None = None,
    ):
        self.index, self.chunks = load_vector_index(index_path, meta_path)
        self.model = get_model(model_name)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        *,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or config.TOP_K
        final_n = max_results if max_results is not None else top_k
        final_n = min(max(1, final_n), len(self.chunks))
        # Ask FAISS for a few extra neighbors, then threshold/dedupe/phrase-merge down to final_n.
        search_n = min(max(final_n * 2, final_n), len(self.chunks))
        q = embed_texts([query], model=self.model)[0:1]
        scores, idxs = self.index.search(q, search_n)
        raw: list[dict[str, Any]] = []
        for score, i in zip(scores[0], idxs[0], strict=False):
            if i < 0:
                continue
            c = self.chunks[i]
            if float(score) < config.SIMILARITY_THRESHOLD:
                continue
            raw.append(_chunk_row(c, float(score)))
        raw = _dedupe_results(raw)
        return _merge_phrase_priority_vector(query, raw, self.chunks, final_n)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_with_scores(query, top_k=top_k)


class BM25Retriever:
    """Bag-of-words scoring — great for exact terminology, weak on paraphrases."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self._corpus = [_tokenize(c["chunk_text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        *,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or config.TOP_K
        final_n = max_results if max_results is not None else top_k
        final_n = min(max(1, final_n), len(self.chunks))
        search_n = min(max(final_n * 2, final_n), len(self.chunks))
        q = _tokenize(query)
        scores = self._bm25.get_scores(q)
        order = np.argsort(scores)[::-1][:search_n]
        raw: list[dict[str, Any]] = []
        for i in order:
            s = float(scores[i])
            c = self.chunks[i]
            raw.append(_chunk_row(c, s))
        raw = _dedupe_results(raw)
        return raw[:final_n]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_with_scores(query, top_k=top_k)


def _reciprocal_rank_fusion(
    ranked_ids: list[list[str]], k: int
) -> dict[str, float]:
    """Merge ordered lists without normalizing incompatible score scales (classic RRF formula)."""
    scores: dict[str, float] = defaultdict(float)
    for ids in ranked_ids:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] += 1.0 / (k + rank)
    return scores


def _rerank_cross_encoder(
    query: str, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    model = _get_cross_encoder()
    pairs = [[query, c["chunk_text"]] for c in chunks]
    rel_scores = model.predict(pairs, show_progress_bar=False)
    order = np.argsort(rel_scores)[::-1]
    out: list[dict[str, Any]] = []
    for idx in order:
        c = chunks[int(idx)].copy()
        c["score"] = float(rel_scores[int(idx)])
        c["retrieval_mode"] = "hybrid_rerank"
        out.append(c)
    return out


class HybridRetriever:
    """Dense + BM25 with reciprocal rank fusion; optional cross-encoder rerank."""

    def __init__(
        self,
        index_path=None,
        meta_path=None,
        model_name: str | None = None,
    ):
        self._vec = VectorRetriever(index_path, meta_path, model_name)
        self.chunks = self._vec.chunks
        self._bm25 = BM25Retriever(self.chunks)
        self._by_id = {c["chunk_id"]: c for c in self.chunks}

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        top_k = top_k or config.TOP_K
        # How many candidates we pull from each channel before fusion/rerank (capped by corpus size).
        pool = min(
            max(config.HYBRID_CANDIDATE_POOL, config.RERANK_POOL_SIZE, top_k * 4),
            len(self.chunks),
        )

        # Full-corpus scan for short “phrase-like” queries (e.g. two rare tokens in order).
        phrase_ids = lexical_phrase_chunk_ids(query, self.chunks)

        vec_hits = self._vec.retrieve_with_scores(query, max_results=pool)
        bm25_hits = self._bm25.retrieve_with_scores(query, max_results=pool)

        vec_ids = [h["chunk_id"] for h in vec_hits]
        bm25_ids = [h["chunk_id"] for h in bm25_hits]

        rrf = _reciprocal_rank_fusion([vec_ids, bm25_ids], config.RRF_K)
        by_rrf = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
        # Lexical hits (e.g. exact Annex III row titles) can be #1 for BM25 but low for dense;
        # pure RRF may push them below the rerank cut. Surface top BM25 ids first, then RRF tail.
        bm25_prio = [h["chunk_id"] for h in bm25_hits[:20]]
        merged_ids: list[str] = []
        seen: set[str] = set()
        for cid in phrase_ids + bm25_prio + by_rrf:
            if cid in seen:
                continue
            seen.add(cid)
            merged_ids.append(cid)

        max_take = max(
            config.HYBRID_CANDIDATE_POOL,
            config.RERANK_POOL_SIZE,
            top_k * 5,
        )
        merged_ids = merged_ids[: min(max_take, len(merged_ids))]

        candidates: list[dict[str, Any]] = []
        for cid in merged_ids:
            base = self._by_id.get(cid)
            if base is None:
                continue
            row = _chunk_row(base, float(rrf.get(cid, 0.01)))
            if cid in phrase_ids:
                row["retrieval_mode"] = "lexical_phrase+rrf"
                row["score"] = max(float(row["score"]), 2.0)
            else:
                row["retrieval_mode"] = "hybrid_rrf"
            candidates.append(row)

        phrase_set = set(phrase_ids)

        if config.RERANK_WITH_CROSS_ENCODER and len(candidates) > 1:
            rerank_n = min(len(candidates), max_take)
            to_score = candidates[:rerank_n]
            reranked = _rerank_cross_encoder(query, to_score)
            head = [r for r in reranked if r["chunk_id"] in phrase_set]
            tail = [r for r in reranked if r["chunk_id"] not in phrase_set]
            return (head + tail)[:top_k]

        head = [c for c in candidates if c["chunk_id"] in phrase_set]
        tail = [c for c in candidates if c["chunk_id"] not in phrase_set]
        return (head + tail)[:top_k]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.retrieve_with_scores(query, top_k=top_k)


def get_query_retriever():
    """Factory used by pipeline/API so switching hybrid on/off is config-only."""
    if config.USE_HYBRID_RETRIEVAL:
        return HybridRetriever()
    return VectorRetriever()