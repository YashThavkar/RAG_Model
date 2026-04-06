from __future__ import annotations

"""
LLM step: turn (question + retrieved chunks) into a natural-language answer.

Deliberately dumb about retrieval — it only formats context and calls the chat API.
If there’s no API key, we still return something useful by quoting the top chunk.
"""

from typing import Any

from openai import OpenAI

from app import config

SYSTEM_INSTRUCTION = """Answer using only the provided context. If the answer is not there, say exactly: "The answer is not available in the provided documents."
Do not invent facts. For every fact you take from the context, say which file and page it comes from (the context lines are labeled with file name and page number).
When several excerpts are about different sub-topics, use the one whose title or opening lines match the subject of the question (e.g. the same section heading or definition), not a neighbouring section."""


def _chunk_location_label(ch: dict[str, Any]) -> str:
    src = ch.get("source_file", "unknown")
    pg = ch.get("page_number")
    if pg is not None:
        return f"{src} — page {pg}"
    return src


def format_context_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    """One labeled block per chunk so the model can cite file + page."""
    parts: list[str] = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        loc = _chunk_location_label(ch)
        cid = ch.get("chunk_id", "")
        parts.append(f"[{i}] {loc} (chunk_id: {cid})\n{ch.get('chunk_text', '')}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    ctx = format_context_block(retrieved_chunks)
    return f"""Question:
{query}

Context:
{ctx}

Answer:"""


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    if not retrieved_chunks:
        return "The answer is not available in the provided documents."

    # No key → skip network; demo mode still shows what retrieval found.
    api_key = config.OPENAI_API_KEY
    model = model or config.OPENAI_CHAT_MODEL

    if not api_key:
        return _answer_without_llm(retrieved_chunks)

    client = client or OpenAI(api_key=api_key, base_url=config.OPENAI_BASE_URL)
    user_content = build_prompt(query, retrieved_chunks)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()


def _answer_without_llm(retrieved_chunks: list[dict[str, Any]]) -> str:
    top = retrieved_chunks[0]
    loc = _chunk_location_label(top)
    preview = top.get("chunk_text", "")[:400].replace("\n", " ")
    return f"Relevant excerpt ({loc}): {preview}..."
