from __future__ import annotations

"""
Dense vectors for chunks and queries.

We L2-normalize embeddings so FAISS inner product search lines up with cosine similarity.
Index is brute-force (IndexFlatIP) — fine for small corpora; swap for IVF/HNSW if you scale up.
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


def get_model(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or config.EMBEDDING_MODEL_NAME
    return SentenceTransformer(name)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    # normalize_embeddings=True → unit vectors → dot product == cosine similarity.
    model = model or get_model()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_vector_index(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(texts, model=model)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search: compare query to every vector
    index.add(vectors)
    return index, chunks


def save_vector_index(
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Path, Path]:
    index_path = index_path or config.FAISS_INDEX_FILE
    meta_path = meta_path or config.FAISS_META_FILE
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    # meta.json order MUST match FAISS row order (row i ↔ chunks[i]).
    serializable = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "source_file": c["source_file"],
            "document_type": c["document_type"],
            "page_number": c.get("page_number"),
            "section_heading": c.get("section_heading"),
            "chunk_text": c["chunk_text"],
            "token_count": c.get("token_count"),
        }
        for c in chunks
    ]
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)
    return index_path, meta_path


def load_vector_index(
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    index_path = index_path or config.FAISS_INDEX_FILE
    meta_path = meta_path or config.FAISS_META_FILE
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing vector index — run build_index.py.")
    index = faiss.read_index(str(index_path))
    with meta_path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
