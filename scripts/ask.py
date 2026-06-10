"""Quick CLI to ask a question without launching the UI.

    python -m scripts.ask "What were Apple's total net sales last year?"
    python -m scripts.ask --strategy fixed --no-rerank "NVIDIA data center revenue?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", default="semantic", choices=list(config.STRATEGIES))
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    pipe = RAGPipeline(strategy=args.strategy)
    result = pipe.answer(args.question, use_rerank=not args.no_rerank, top_k=args.top_k)

    print("\n" + "=" * 70)
    print(result.answer.text)
    print("=" * 70)
    if not result.answer.refused and result.answer.sources:
        print("\nSources:")
        for s in result.answer.sources:
            print(f"  [{s.n}] {s.company} ({s.ticker})  {s.chunk_id}")
            print(f"      {s.source_url}")


if __name__ == "__main__":
    main()
