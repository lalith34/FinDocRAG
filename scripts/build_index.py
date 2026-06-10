"""Build the corpus: download 10-Ks (incremental by EDGAR accession), chunk,
embed, and upsert to Pinecone. Unchanged filings are skipped unless --force.

    python -m scripts.build_index                 # incremental ingest + build
    python -m scripts.build_index --force         # rebuild from CACHED raw HTML
    python -m scripts.build_index --refresh-raw   # re-download raw HTML from EDGAR
    python -m scripts.build_index --no-ingest     # reuse already-cleaned text
    python -m scripts.build_index --strategies fixed

--force rebuilds chunks/embeddings by re-cleaning the cached data/raw HTML, so it
is reproducible. Pass --refresh-raw to actually re-fetch the filings from EDGAR
(the only non-deterministic step).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.pipeline import build_indexes  # noqa: E402
from src.telemetry import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ingest", action="store_true", help="skip EDGAR download, reuse data/processed")
    ap.add_argument("--force", action="store_true", help="rebuild all tickers from cached raw HTML, even if filings are unchanged")
    ap.add_argument("--refresh-raw", action="store_true", help="re-download raw HTML from EDGAR (non-deterministic; default reuses data/raw)")
    ap.add_argument("--strategies", nargs="+", default=list(config.STRATEGIES))
    args = ap.parse_args()

    setup_logging()
    summary = build_indexes(
        strategies=args.strategies,
        do_ingest=not args.no_ingest,
        force=args.force,
        refresh_raw=args.refresh_raw,
    )

    print("\n=== Build summary ===")
    for strategy, stats in summary.items():
        rebuilt = ", ".join(stats["rebuilt_tickers"]) or "none"
        print(
            f"  {strategy}: {stats['chunks']} chunks, avg {stats['avg_tokens']} "
            f"tokens/chunk (rebuilt: {rebuilt})"
        )


if __name__ == "__main__":
    main()
