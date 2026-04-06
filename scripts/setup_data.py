from __future__ import annotations

"""Copy the two expected PDFs into data/raw/ from CLI paths or SOURCE_* env vars."""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--attention-src",
        default=os.environ.get("SOURCE_ATTENTION_PDF", ""),
        help="Attention PDF path",
    )
    p.add_argument(
        "--eu-act-src",
        default=os.environ.get("SOURCE_EU_AI_ACT_PDF", ""),
        help="EU AI Act PDF path",
    )
    args = p.parse_args()

    dest_dir = config.DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_att = dest_dir / config.PDF_ATTENTION_NAME
    dest_eu = dest_dir / config.PDF_EU_ACT_NAME

    if args.attention_src:
        src = Path(args.attention_src).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Attention PDF not found: {src}")
        shutil.copy2(src, dest_att)
        print(f"Copied -> {dest_att}")
    elif not dest_att.is_file():
        raise FileNotFoundError(
            f"Missing {dest_att}. Copy the company Attention PDF there, or run:\n"
            f"  python scripts/setup_data.py --attention-src <path>"
        )

    if args.eu_act_src:
        src = Path(args.eu_act_src).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"EU AI Act PDF not found: {src}")
        shutil.copy2(src, dest_eu)
        print(f"Copied -> {dest_eu}")
    elif not dest_eu.is_file():
        raise FileNotFoundError(
            f"Missing {dest_eu}. Copy the company EU AI Act PDF there, or run:\n"
            f"  python scripts/setup_data.py --eu-act-src <path>"
        )

    if not args.attention_src and not args.eu_act_src:
        print("OK: both files present:", dest_att, dest_eu)


if __name__ == "__main__":
    main()
