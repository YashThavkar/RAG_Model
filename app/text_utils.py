from __future__ import annotations

"""
Small string helpers for API / JSON output.

Chunk text keeps newlines while the LLM reads it; we normalize only what we ship outward
so logs and browser JSON stay readable on one line.
"""

import re


def flatten_for_output(s: str | None) -> str:
    """Turn multi-line text into a single line for JSON logs and the web UI."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())
