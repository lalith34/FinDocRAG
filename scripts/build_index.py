"""Build the full corpus: download 10-Ks, chunk with both strategies, embed,
and persist the vector stores.

    python -m scripts.build_index                 # ingest + build both strategies
    python -m scripts.build_index --no-ingest     # reuse already-downloaded text
    python -m scripts.build_index --strategies fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.pipeline import build_indexes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ingest", action="store_true", help="skip EDGAR download, reuse data/processed")
    ap.add_argument("--strategies", nargs="+", default=list(config.STRATEGIES))
    args = ap.parse_args()

    summary = build_indexes(strategies=args.strategies, do_ingest=not args.no_ingest)

    print("\n=== Build summary ===")
    for strategy, stats in summary.items():
        print(f"  {strategy}: {stats['chunks']} chunks, avg {stats['avg_tokens']} tokens/chunk")


if __name__ == "__main__":
    main()
