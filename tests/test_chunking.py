from __future__ import annotations

"""Chunk schema, overlap, and multi-chunk behaviour on synthetic long text."""

from app.chunking import chunk_document
from app.preprocess import preprocess_document


def test_chunk_schema_and_overlap():
    doc = {
        "doc_id": "d1",
        "source_file": "x.pdf",
        "document_type": "pdf",
        "page_number": 1,
        "section_heading": None,
        "raw_text": "\n\n".join(["Para one " + ("word " * 200)] * 3),
    }
    chunks = chunk_document(doc, chunk_target_chars=500, overlap_chars=80)
    assert len(chunks) >= 2
    for c in chunks:
        assert "chunk_id" in c
        assert "chunk_text" in c
        assert c["source_file"] == "x.pdf"
        assert c["token_count"] > 0


def test_attention_paper_preprocess_merges_section_lines():
    raw = "a@b.co\nAbstract\nThe abstract.\n1\nIntroduction\nIntro body."
    doc = preprocess_document(
        {
            "doc_id": "att",
            "source_file": "Attention_is_all_you_need.pdf",
            "document_type": "pdf",
            "page_number": 1,
            "section_heading": None,
            "raw_text": raw,
        }
    )
    assert "1 Introduction" in doc["raw_text"]
    assert "3.2.1 Scaled Dot-Product Attention" in preprocess_document(
        {
            "doc_id": "att",
            "source_file": "Attention_is_all_you_need.pdf",
            "document_type": "pdf",
            "page_number": 4,
            "section_heading": None,
            "raw_text": "para\n3.2.1\nScaled Dot-Product Attention\nWe call our attention.",
        }
    )["raw_text"]


def test_attention_paper_chunk_starts_without_overlap_prefix():
    pre = ("before " * 80).strip()
    raw = f"{pre}\n1\nIntroduction\nStart of intro body."
    doc = preprocess_document(
        {
            "doc_id": "att",
            "source_file": "Attention_is_all_you_need.pdf",
            "document_type": "pdf",
            "page_number": 2,
            "section_heading": None,
            "raw_text": raw,
        }
    )
    chunks = chunk_document(doc, chunk_target_chars=300, overlap_chars=60)
    assert any(c["chunk_text"].lstrip().startswith("1 Introduction") for c in chunks)
