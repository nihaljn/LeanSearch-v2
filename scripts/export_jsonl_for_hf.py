"""Dump the `data` list from a built cuVS metadata.pkl into a single JSONL
file, suitable for publishing as the HuggingFace `*-jsonl` dataset.

Usage:
    python scripts/export_jsonl_for_hf.py <cuvs_db_dir>/metadata.pkl out.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_pkl", type=str)
    parser.add_argument("output_jsonl", type=str)
    args = parser.parse_args()

    with open(args.metadata_pkl, "rb") as f:
        meta = pickle.load(f)
    rows = meta["data"]
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
