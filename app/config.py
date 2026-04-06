from __future__ import annotations

"""
Single source of truth for paths and tunables.

Anything you might change between dev / prod (model names, chunk size, API keys)
lives here and reads from the environment so you don’t hunt through the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Pull variables from a local .env file if present (never commit real secrets).
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data layout (everything under the repo root) ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
OUTPUTS = PROJECT_ROOT / "outputs"
VECTOR_INDEX_DIR = DATA_PROCESSED / "vector_index"
CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"
CLEANED_JSONL = DATA_PROCESSED / "cleaned_units.jsonl"
QUERY_LOGS = OUTPUTS / "query_logs.jsonl"
EVAL_RESULTS = OUTPUTS / "eval_results.json"

# Default filenames the loader expects under DATA_RAW (see loader.default_raw_paths).
PDF_ATTENTION_NAME = "Attention_is_all_you_need.pdf"
PDF_EU_ACT_NAME = "EU_AI_Act_Doc.pdf"

# Sentence-transformers model id; changing it requires re-running build_index.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-mpnet-base-v2",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# --- Chunking & retrieval (rebuild index after changing chunk / embedding settings) ---
# Smaller chunks → finer retrieval (especially for list-style EU Act PDFs).
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
# Drop FAISS hits below this inner-product score (0 = keep all).
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))

# Hybrid = dense + BM25 merged with RRF; can turn off to debug dense-only.
USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "1").lower() in ("1", "true", "yes")
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_CANDIDATE_POOL = int(os.getenv("HYBRID_CANDIDATE_POOL", "48"))

# Second-stage (query, passage) scorer — slower but often better ordering than cosine alone.
RERANK_WITH_CROSS_ENCODER = os.getenv("RERANK_WITH_CROSS_ENCODER", "1").lower() in (
    "1",
    "true",
    "yes",
)
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", "24"))
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Promote chunks where query terms appear in order within a short span (helps short / typo queries).
USE_LEXICAL_PHRASE_PRIORITY = os.getenv(
    "USE_LEXICAL_PHRASE_PRIORITY", "1"
).lower() in ("1", "true", "yes")
LEXICAL_PHRASE_MAX_SPAN = int(os.getenv("LEXICAL_PHRASE_MAX_SPAN", "160"))

# Built by embeddings.save_vector_index; must stay in sync with meta.json row order.
FAISS_INDEX_FILE = VECTOR_INDEX_DIR / "index.faiss"
FAISS_META_FILE = VECTOR_INDEX_DIR / "meta.json"
