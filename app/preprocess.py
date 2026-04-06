from __future__ import annotations

"""
Text cleanup after load, before chunking.

Generic steps apply to every doc; EU AI Act PDFs get extra splits so annex rows and bullets
become separate paragraphs. The Attention paper PDF gets splits at section / subsection /
abstract / figure lines for the same reason.
"""

import re
import unicodedata
from collections import Counter
from typing import Any

from app import config


def normalize_unicode(text: str) -> str:
    # NFKC folds weird compatibility characters (smart quotes, ligatures) toward normal forms.
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    # Normalize newlines and squash runs of spaces; trim outer whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Annex III–style row titles as they appear in the EU_AI_Act summary PDF (exact prefixes before ':').
EU_ACT_ROW_HEADINGS = (
    "Non-banned biometrics",
    "Critical infrastructure",
    "Education and vocational training",
    "Employment, workers management and access to self-employment",
    "Access to and enjoyment of essential public and private services",
    "Law enforcement",
    "Migration, asylum and border control management",
    "Administration of justice and democratic processes",
)
_EU_ACT_HEADING_ALT = "|".join(re.escape(h) for h in EU_ACT_ROW_HEADINGS)


def split_eu_act_category_lines(text: str) -> str:
    """
    Insert paragraph breaks before Annex III–style category headings so each row can
    be chunked separately. PDF text often has 'Law enforcement: AI systems...' on one
    line (no newline after the colon), so a generic 'heading then newline' pattern fails.
    """
    # New line already present before heading — ensure a paragraph break (not triple).
    text = re.sub(
        rf"(?<!\n\n)(?<=\n)(\s*[•·]?\s*)({_EU_ACT_HEADING_ALT}):",
        r"\n\n\1\2:",
        text,
    )
    # Heading glued to prior sentence: "...insurance.\nLaw enforcement: ..." or "...insurance. Law enforcement: ..."
    text = re.sub(
        rf"(?<=[.!?])\s+({_EU_ACT_HEADING_ALT}):",
        r"\n\n\1:",
        text,
    )
    # Major block after Annex III rows in this summary PDF — keep out of the last Annex III chunk.
    text = re.sub(
        r"(?<!\n\n)(?<=\n)(General purpose AI \(GPAI\))",
        r"\n\n\1",
        text,
    )
    return text


def split_eu_inline_bullets(text: str) -> str:
    """
    Break merged bullet runs like '...text. • Next obligation' into separate paragraphs
    so each bullet can be chunked and retrieved on its own (e.g. 'Track, document...').
    """
    return re.sub(r"(?<!\n\n)(?<=[^\n•])\s+(?=•\s)", r"\n\n", text)


# Section titles for "Attention Is All You Need" as they appear after digit+newline in the arXiv PDF.
_ATTENTION_MAIN_SECTION_TITLES = (
    "Introduction",
    "Background",
    "Model Architecture",
    "Why Self-Attention",
    "Training",
    "Results",
    "Conclusion",
)
_ATTENTION_MAIN_TITLE_ALT = "|".join(re.escape(t) for t in _ATTENTION_MAIN_SECTION_TITLES)


def split_attention_paper_structure(text: str) -> str:
    """
    Insert paragraph breaks where this PDF uses single newlines between structural lines
    (section number, title, abstract, figures), so generic \\n\\n paragraph splitting works.
    """
    # Title line after the permission boilerplate.
    text = re.sub(
        r"(?<=[.!?])\s*\n(Attention Is All You Need)\s*\n",
        r"\n\n\1\n\n",
        text,
    )
    # Abstract after author / email lines.
    text = re.sub(
        r"(?<=[a-zA-Z0-9@.!])\n(Abstract)\s*\n",
        r"\n\n\1\n\n",
        text,
    )
    # Subsections: "3.2.1\\nScaled Dot-Product...\\n" — (?m)^ so page breaks (line starts with "3.2.1") match.
    text = re.sub(
        r"(?m)^(\d+\.\d+(?:\.\d+)?)\s*\n([A-Z][^\n]{2,160})\n",
        r"\1 \2\n\n",
        text,
    )
    # Main sections: "1\\nIntroduction\\n" — allowlisted titles; (?m)^ covers start-of-page units.
    text = re.sub(
        rf"(?m)^([1-9]\d?)\s*\n({_ATTENTION_MAIN_TITLE_ALT})\n",
        r"\1 \2\n\n",
        text,
    )
    # One newline after body text still leaves the heading in the same paragraph — force a break before it.
    text = re.sub(
        r"(?<!\n\n)(?<=[^\n])\n(?=(?:[1-9]\d?|\d+\.\d+(?:\.\d+)?) [A-Z])",
        r"\n\n",
        text,
    )
    text = re.sub(
        r"(?<!\n\n)(?<=[^\n])\n(Acknowledgements)\s*\n",
        r"\n\n\1\n\n",
        text,
    )
    text = re.sub(
        r"(?<!\n\n)(?<=[^\n])\n(References)\s*\n(?=\[\d+\])",
        r"\n\n\1\n\n",
        text,
    )
    text = re.sub(
        r"(?<!\n\n)(?<=[^\n])\n(?=(Figure|Table)\s+\d+[:.])",
        r"\n\n",
        text,
        flags=re.IGNORECASE,
    )
    return text


def remove_repeated_headers(text: str, min_line_occurrences: int = 8) -> str:
    """Strip lines that look like repeating headers/footers (common in PDFs)."""
    lines = text.split("\n")
    if len(lines) < 3:
        return text
    stripped = [ln.strip() for ln in lines if ln.strip()]
    if not stripped:
        return text
    counts = Counter(stripped)
    n = len(stripped)
    drop = {
        ln
        for ln, c in counts.items()
        if c >= min_line_occurrences and len(ln) < 120 and c / n > 0.15
    }
    if not drop:
        return text
    kept: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s and s in drop:
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def preprocess_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the unit with `raw_text` cleaned; other fields unchanged."""
    t = doc.get("raw_text") or ""
    t = normalize_unicode(t)
    t = clean_whitespace(t)
    t = remove_repeated_headers(t)
    # EU summary PDF: structured headings/bullets — worth special-casing for RAG quality.
    sf = doc.get("source_file", "")
    if "EU_AI_Act" in sf:
        t = split_eu_act_category_lines(t)
        t = split_eu_inline_bullets(t)
        t = clean_whitespace(t)
    elif config.PDF_ATTENTION_NAME in sf:
        t = split_attention_paper_structure(t)
        t = clean_whitespace(t)
    out = {**doc, "raw_text": t}
    return out


def preprocess_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [preprocess_document(d) for d in documents]
