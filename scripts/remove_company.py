"""Remove one or more tickers from the corpus, including all of their data.

    python -m scripts.remove_company TSLA
    python -m scripts.remove_company TSLA META

For each ticker: delete its vectors from every Pinecone namespace, prune the local
BM25 snapshots, drop its processed/raw/tree files, and unregister it. No ghost data
is left behind. Failures are reported per ticker.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import remove_company  # noqa: E402
from src.telemetry import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+", help="ticker symbol(s) to remove")
    args = ap.parse_args()

    setup_logging()
    failures = 0
    for t in args.tickers:
        print(f"\n=== remove {t.upper()} ===", flush=True)
        res = remove_company(t)
        print(("✓ " if res.ok else "✗ ") + res.message, flush=True)
        failures += 0 if res.ok else 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
