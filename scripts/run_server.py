from __future__ import annotations

"""Dev entrypoint: uvicorn app.api:app — prints URLs (open manually; Ctrl+click in many terminals)."""

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Start the FastAPI server (CeDAR web UI + /query API).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Restart on code changes (dev only).")
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print()
    print("  CeDAR API is starting.")
    print(f"  Open in your browser (Ctrl+click the URL in many terminals):")
    print()
    print(f"  {url}")
    print()
    print("  API docs: ", f"http://{args.host}:{args.port}/docs")
    print()
    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
