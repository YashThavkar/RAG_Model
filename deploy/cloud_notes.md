# Deploy

Ship `data/processed/` (chunks + FAISS) in the image or run `build_index` on startup.

Set `OPENAI_API_KEY` on the host. No key = answers are just the pulled chunk text.

Examples: Render / Fly / Cloud Run / Azure — run `uvicorn app.api:app --host 0.0.0.0 --port $PORT` (or 8000 locally).

```bash
docker build -f deploy/Dockerfile -t rag-doc-qa .
docker run -p 8000:8000 -e OPENAI_API_KEY=... rag-doc-qa
```
