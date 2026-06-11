"""Add one or more tickers to the live corpus, end-to-end.

    python -m scripts.add_company TSLA
    python -m scripts.add_company TSLA META
    python -m scripts.add_company AAPL --rebuild   # force re-ingest/rebuild

Each ticker is validated against SEC EDGAR, checked for a 10-K, registered with an
auto-derived name alias, then ingested/chunked/embedded/upserted — only that
ticker; the rest of the corpus is untouched. Failures are reported per ticker and
do not stop the others.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import add_company  # noqa: E402
from src.telemetry import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+", help="ticker symbol(s), e.g. TSLA META")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-ingest and rebuild even if the company is already present")
    args = ap.parse_args()

    setup_logging()
    failures = 0
    for t in args.tickers:
        print(f"\n=== add {t.upper()} ===", flush=True)
        res = add_company(t, rebuild=args.rebuild)
        print(("✓ " if res.ok else "✗ ") + res.message, flush=True)
        if res.ok and res.chunks:
            print("   index: " + ", ".join(f"{s}={n} chunks" for s, n in res.chunks.items()),
                  flush=True)
        failures += 0 if res.ok else 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
