from __future__ import annotations

"""
Read raw documents and turn them into uniform dicts the rest of the pipeline expects.

PDFs: one record per page (PyMuPDF). DOCX: one record per body paragraph, with optional heading trail.
No chunking or cleaning here — that’s preprocess + chunking.
"""

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from app import config


def _doc_id_from_path(path: Path) -> str:
    # Stable id from filename so chunk_ids and indexes don’t care about full paths.
    stem = path.stem.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", stem) or "doc"


def load_pdf(path: str | Path) -> list[dict[str, Any]]:
    """Extract plain text per page; empty pages are skipped."""
    path = Path(path)
    doc_id = _doc_id_from_path(path)
    out: list[dict[str, Any]] = []
    with fitz.open(path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            raw_text = page.get_text("text") or ""
            # Drop blank lines so we don’t store huge runs of whitespace.
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if not lines:
                continue
            text = "\n".join(lines)
            out.append(
                {
                    "doc_id": doc_id,
                    "source_file": path.name,
                    "document_type": "pdf",
                    "page_number": page_index + 1,
                    "section_heading": None,
                    "raw_text": text,
                }
            )
    return out


def load_docx(path: str | Path) -> list[dict[str, Any]]:
    """Walk paragraphs; headings update context, body text becomes one unit each."""
    path = Path(path)
    doc_id = _doc_id_from_path(path)
    document = DocxDocument(path)
    out: list[dict[str, Any]] = []
    # Word has no real pages here — we use a counter so every unit has a sortable id.
    page_proxy = 1
    current_heading: str | None = None

    for para in document.paragraphs:
        style = para.style.name if para.style else ""
        text = (para.text or "").strip()
        if not text:
            continue
        if style.startswith("Heading") or style == "Title":
            current_heading = text
            continue
        out.append(
            {
                "doc_id": doc_id,
                "source_file": path.name,
                "document_type": "docx",
                "page_number": page_proxy,
                "section_heading": current_heading,
                "raw_text": text,
            }
        )
        page_proxy += 1

    return out


def load_documents(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Dispatch by extension; raises if a path is missing or unsupported."""
    units: list[dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing document: {p}")
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            units.extend(load_pdf(p))
        elif suffix == ".docx":
            units.extend(load_docx(p))
        else:
            raise ValueError(f"Unsupported format: {p}")
    return units


def default_raw_paths() -> list[Path]:
    return [
        config.DATA_RAW / config.PDF_ATTENTION_NAME,
        config.DATA_RAW / config.PDF_EU_ACT_NAME,
    ]
