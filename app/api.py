from __future__ import annotations

"""
HTTP front door: JSON API for ingest + query, plus the RAGBox HTML shell.

Business logic stays in pipeline — this file validates input and serves the UI.
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.pipeline import answer_query, ingest_and_index

app = FastAPI(title="RAGBox", version="1.1.0")
UI_FILE = Path(__file__).resolve().parent / "static" / "index.html"
ALLOWED_SUFFIX = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def _raw_library() -> list[dict]:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(config.DATA_RAW.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX:
            rows.append(
                {
                    "name": p.name,
                    "kind": p.suffix.lower().lstrip("."),
                    "bytes": p.stat().st_size,
                }
            )
    return rows


@app.get("/")
def index() -> FileResponse:
    if not UI_FILE.exists():
        raise HTTPException(status_code=500, detail="UI file missing.")
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/library")
def library() -> dict:
    return {
        "files": _raw_library(),
        "indexed": config.FAISS_INDEX_FILE.exists(),
        "chunks_path": str(config.CHUNKS_JSONL),
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Drop at least one PDF or Word file.")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        name = Path(upload.filename or "document").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIX:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file: {name}. Use .pdf or .docx.",
            )
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"{name} is over 20 MB.")
        dest = config.DATA_RAW / name
        dest.write_bytes(payload)
        saved.append(name)

    try:
        info = ingest_and_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"saved": saved, "library": _raw_library(), **info}


@app.post("/query")
def post_query(body: QueryBody) -> dict:
    try:
        top_k = body.top_k or config.TOP_K
        return answer_query(body.query, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e) + " Drop documents in the UI or run scripts/build_index.py.",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
