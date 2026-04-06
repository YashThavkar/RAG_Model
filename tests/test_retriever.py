from __future__ import annotations

"""BM25 ordering, lexical phrase helper, and (optional) FAISS round-trip integration."""

import pytest

from app import config
from app.chunking import chunk_document, save_chunks_jsonl
from app.embeddings import build_vector_index, get_model, save_vector_index
from app.retriever import (
    BM25Retriever,
    VectorRetriever,
    _lexical_ordered_terms_match,
    lexical_phrase_chunk_ids,
)


def test_lexical_ordered_terms_match():
    eu = (
        "• Track, document and report serious incidents to the AI Office "
        "without undue delay."
    )
    assert _lexical_ordered_terms_match("track, document", eu)
    assert not _lexical_ordered_terms_match("track, document", "document first then track later")


def test_lexical_phrase_prefers_eu_chunk_over_attention():
    chunks = [
        {
            "chunk_id": "att_0",
            "chunk_text": "We compute attention using queries, keys, and values in this document.",
        },
        {
            "chunk_id": "eu_0",
            "chunk_text": (
                "• Track, document and report serious incidents and possible "
                "corrective measures to the AI Office."
            ),
        },
    ]
    ids = lexical_phrase_chunk_ids("track, document", chunks)
    assert ids and ids[0] == "eu_0"


def test_bm25_retrieval_order():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d",
            "source_file": "f.pdf",
            "document_type": "pdf",
            "page_number": 1,
            "section_heading": None,
            "chunk_text": "The cat sat on the mat.",
            "token_count": 6,
        },
        {
            "chunk_id": "c2",
            "doc_id": "d",
            "source_file": "f.pdf",
            "document_type": "pdf",
            "page_number": 1,
            "section_heading": None,
            "chunk_text": "Transformer self-attention replaces recurrence.",
            "token_count": 5,
        },
    ]
    r = BM25Retriever(chunks)
    out = r.retrieve("self-attention transformer", top_k=2)
    assert out[0]["chunk_id"] == "c2"


@pytest.mark.integration
def test_vector_retriever_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAISS_INDEX_FILE", tmp_path / "index.faiss")
    monkeypatch.setattr(config, "FAISS_META_FILE", tmp_path / "meta.json")
    doc = {
        "doc_id": "d1",
        "source_file": "x.pdf",
        "document_type": "pdf",
        "page_number": 1,
        "section_heading": None,
        "raw_text": "Scaled dot-product attention uses softmax over keys and values.",
    }
    chunks = chunk_document(doc, chunk_target_chars=800, overlap_chars=50)
    model = get_model()
    index, meta = build_vector_index(chunks, model=model)
    save_vector_index(index, meta)
    vr = VectorRetriever(index_path=config.FAISS_INDEX_FILE, meta_path=config.FAISS_META_FILE)
    hits = vr.retrieve("dot-product attention softmax", top_k=2)
    assert hits and "attention" in hits[0]["chunk_text"].lower()
