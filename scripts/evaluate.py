from __future__ import annotations

"""Run test_queries.json against the current index; prints summary and writes outputs/eval_results.json."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluator import run_full_evaluation


def main() -> None:
    summary = run_full_evaluation()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
