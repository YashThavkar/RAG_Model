from __future__ import annotations

"""
Turn each cleaned document unit into many smaller chunks for embedding and search.

Strategy: split on blank lines first, then break oversized paragraphs on sentence boundaries.
We carry overlap between consecutive chunks so facts on boundaries aren’t lost, with
special cases for EU annex headings and Attention-paper section starts so overlap does not
prefix the previous section’s tail onto a new heading.
"""

import json
import re
from pathlib import Path
from typing import Any

from app import config
from app.preprocess import EU_ACT_ROW_HEADINGS


def split_into_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _approx_token_count(text: str) -> int:
    return len(text.split())


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Greedy pack sentences until max_chars; avoids one giant unbreakable paragraph."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip() if buf else s
        else:
            if buf:
                chunks.append(buf)
            buf = s if len(s) <= max_chars else s[:max_chars]
    if buf:
        chunks.append(buf)
    return chunks


def _piece_starts_eu_annex_heading(piece: str, source_file: str) -> bool:
    """Avoid cross-chunk overlap into a new Annex III row — overlap would prefix the *previous* section."""
    if "EU_AI_Act" not in (source_file or ""):
        return False
    s = piece.lstrip()
    return any(s.startswith(f"{h}:") or s.startswith(f"{h} :") for h in EU_ACT_ROW_HEADINGS)


def _piece_starts_gpai_block_eu_summary(piece: str, source_file: str) -> bool:
    """In the EU summary PDF, GPAI follows Annex III — start a new chunk so embeddings are not mixed."""
    if "EU_AI_Act" not in (source_file or ""):
        return False
    return piece.lstrip().startswith("General purpose AI (GPAI)")


_ATTENTION_SECTION_HEAD_RE = re.compile(
    r"^(?:"
    r"(?:[1-9]\d?|\d+\.\d+(?:\.\d+)?)\s+[A-Za-z]"
    r"|Abstract\b"
    r"|Attention Is All You Need\b"
    r"|Acknowledgements\b"
    r"|References\b"
    r"|(?:Figure|Table)\s+\d+[:.])",
    re.IGNORECASE,
)


def _piece_starts_attention_section(piece: str, source_file: str) -> bool:
    """After preprocess, numbered headings / abstract / refs / captions start clean chunks (no overlap in)."""
    if config.PDF_ATTENTION_NAME not in (source_file or ""):
        return False
    return bool(_ATTENTION_SECTION_HEAD_RE.match(piece.lstrip()))


def _overlap_prefix(prev_text: str, overlap_chars: int) -> str:
    """Take the tail of the previous chunk, trimmed to a word boundary, to prefix the next."""
    if overlap_chars <= 0 or not prev_text:
        return ""
    tail = prev_text[-overlap_chars:]
    cut = tail.find(" ")
    return tail[cut + 1 :].strip() if cut != -1 else tail.strip()


def chunk_document(
    doc: dict[str, Any],
    chunk_target_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Produce ordered chunks with metadata; chunk_id includes page so ids are unique corpus-wide."""
    chunk_target_chars = chunk_target_chars or config.CHUNK_TARGET_CHARS
    overlap_chars = overlap_chars or config.CHUNK_OVERLAP_CHARS
    doc_id = doc["doc_id"]
    source_file = doc["source_file"]
    document_type = doc["document_type"]
    page_number = doc.get("page_number")
    section_heading = doc.get("section_heading")

    paragraphs = split_into_paragraphs(doc["raw_text"])
    if not paragraphs:
        return []

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_target_chars:
            pieces.append(para)
        else:
            pieces.extend(_split_long_paragraph(para, chunk_target_chars))

    chunks: list[dict[str, Any]] = []
    buf = ""
    prev_chunk_text = ""
    chunk_idx = 0

    def emit():
        nonlocal buf, prev_chunk_text, chunk_idx
        text = buf.strip()
        if not text:
            buf = ""
            return
        # Page in the id prevents collisions when each PDF page used to restart chunk_idx at 0.
        pg = page_number if page_number is not None else 0
        cid = f"{doc_id}_p{pg}_c{chunk_idx:04d}"
        chunk_idx += 1
        chunks.append(
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "source_file": source_file,
                "document_type": document_type,
                "page_number": page_number,
                "section_heading": section_heading,
                "chunk_text": text,
                "token_count": _approx_token_count(text),
            }
        )
        prev_chunk_text = text
        buf = ""

    for piece in pieces:
        # Starting a new buffer: optionally prepend overlap from the chunk we just finished.
        prefix = _overlap_prefix(prev_chunk_text, overlap_chars) if not buf else ""
        if prefix and _piece_starts_eu_annex_heading(piece, source_file):
            prefix = ""
        if prefix and _piece_starts_attention_section(piece, source_file):
            prefix = ""
        candidate = f"{prefix}\n\n{piece}".strip() if prefix else piece
        if not buf:
            buf = candidate
        elif _piece_starts_gpai_block_eu_summary(piece, source_file) and buf.strip():
            emit()
            buf = piece
        elif _piece_starts_attention_section(piece, source_file) and buf.strip():
            emit()
            buf = piece
        elif len(buf) + 2 + len(piece) <= chunk_target_chars:
            buf = f"{buf}\n\n{piece}"
        else:
            emit()
            op = _overlap_prefix(prev_chunk_text, overlap_chars)
            if _piece_starts_eu_annex_heading(piece, source_file):
                op = ""
            if _piece_starts_attention_section(piece, source_file):
                op = ""
            buf = f"{op}\n\n{piece}".strip() if op else piece
    emit()
    return chunks


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def save_chunks_jsonl(chunks: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or config.CHUNKS_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


def load_chunks_jsonl(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or config.CHUNKS_JSONL
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
