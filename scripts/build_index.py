from __future__ import annotations

"""Rebuild chunks + FAISS from whatever is in data/raw/ (run after changing PDFs or chunk settings)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import ingest_and_index


def main() -> None:
    info = ingest_and_index()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
