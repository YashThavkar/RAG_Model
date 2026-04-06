from __future__ import annotations

"""Quick terminal Q&A; appends one JSON line to outputs/query_logs.jsonl via pipeline.answer_query."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import answer_query


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", help="Question text")
    p.add_argument(
        "-q",
        "--stdin",
        action="store_true",
        help="Read query from stdin",
    )
    args = p.parse_args()
    q = args.query
    if args.stdin or not q:
        q = sys.stdin.read().strip()
    if not q:
        p.error("Provide a query string or use --stdin")
    out = answer_query(q)
    print(json.dumps({k: out[k] for k in ("query", "answer", "sources")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
