"""Evaluation harness: the two deliverables from the project brief.

  1. Chunking strategy comparison (fixed vs semantic) on retrieval quality.
  2. Reranking impact analysis (hybrid retrieval with vs without the reranker).

Plus an optional LLM-judge pass on the generated answers (faithfulness +
relevance) and a check that the refusal path fires on an unanswerable query.

Retrieval relevance is judged with lightweight labels in eval/queries.json: a
retrieved chunk counts as relevant if it comes from an expected ticker AND
contains an expected keyword. Reported metrics: Hit@k, MRR, Precision@k.

Run after building indexes:
    python -m scripts.evaluate            # full run incl. LLM judge
    python -m scripts.evaluate --no-judge # retrieval/rerank metrics only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import embeddings  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.generate import generate  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

TOP_K = config.TOP_K


# --- labelling ---------------------------------------------------------------
def is_relevant(chunk: Chunk, query: dict) -> bool:
    if chunk.ticker not in query["expected_tickers"]:
        return False
    text = chunk.text.lower()
    return any(kw.lower() in text for kw in query["must_contain"])


def rank_metrics(chunks: list[Chunk], query: dict, k: int = TOP_K) -> dict:
    top = chunks[:k]
    rels = [is_relevant(c, query) for c in top]
    hit = 1.0 if any(rels) else 0.0
    mrr = 0.0
    for rank, r in enumerate(rels):
        if r:
            mrr = 1.0 / (rank + 1)
            break
    precision = sum(rels) / k
    return {"hit": hit, "mrr": mrr, "precision": precision}


def avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(1, len(rows))


# --- LLM judge ---------------------------------------------------------------
_JUDGE_SYS = (
    "You are evaluating a RAG answer. You are given a QUESTION, the SOURCES the "
    "system was shown, and its ANSWER. Score two things on 0-1:\n"
    "- faithfulness: 1 if every claim in the ANSWER is supported by the SOURCES "
    "(no invented facts), else 0.\n"
    "- relevance: 1 if the ANSWER actually addresses the QUESTION, else 0.\n"
    'Respond ONLY as JSON: {"faithfulness": 0 or 1, "relevance": 0 or 1}.'
)


def judge_answer(question: str, answer_text: str, sources_text: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _JUDGE_SYS},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nSOURCES:\n{sources_text}\n\nANSWER:\n{answer_text}",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM faithfulness judge")
    ap.add_argument("--strategies", nargs="+", default=list(config.STRATEGIES))
    args = ap.parse_args()

    queries = json.loads((config.EVAL_DIR / "queries.json").read_text())
    answerable = [q for q in queries if not q["expect_refusal"]]
    refusal_qs = [q for q in queries if q["expect_refusal"]]

    # retrieval + rerank metrics per strategy ---------------------------------
    chunk_rows = {}   # strategy -> retrieval-only metric rows
    rerank_rows = {}  # strategy -> reranked metric rows

    for strategy in args.strategies:
        store = VectorStore.load(strategy)
        base, reranked = [], []
        for q in answerable:
            qvec = embeddings.embed_query(q["question"])
            pool = [store.get(i) for i, _ in store.hybrid(q["question"], qvec, config.RETRIEVE_K)]
            base.append(rank_metrics(pool, q, TOP_K))
            rr = [c for c, _ in rerank(q["question"], pool, TOP_K)]
            reranked.append(rank_metrics(rr, q, TOP_K))
            print(f"  [{strategy}] {q['id']} done", flush=True)
        chunk_rows[strategy] = base
        rerank_rows[strategy] = reranked

    # generation faithfulness + refusal check ---------------------------------
    gen_rows = []
    refusal_results = []
    if not args.no_judge:
        gen_strategy = args.strategies[-1]  # judge on the richer/last strategy
        store = VectorStore.load(gen_strategy)
        for q in answerable:
            qvec = embeddings.embed_query(q["question"])
            pool = [store.get(i) for i, _ in store.hybrid(q["question"], qvec, config.RETRIEVE_K)]
            top = [c for c, _ in rerank(q["question"], pool, TOP_K)]
            ans = generate(q["question"], top)
            src_text = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(top))
            verdict = judge_answer(q["question"], ans.text, src_text)
            gen_rows.append(verdict)
            print(f"  [gen] {q['id']} faithful={verdict.get('faithfulness')} "
                  f"relevant={verdict.get('relevance')}", flush=True)

        for q in refusal_qs:
            qvec = embeddings.embed_query(q["question"])
            pool = [store.get(i) for i, _ in store.hybrid(q["question"], qvec, config.RETRIEVE_K)]
            top = [c for c, _ in rerank(q["question"], pool, TOP_K)]
            ans = generate(q["question"], top)
            refusal_results.append((q["id"], ans.refused))
            print(f"  [refusal] {q['id']} refused={ans.refused}", flush=True)

    write_report(args, chunk_rows, rerank_rows, gen_rows, refusal_results)


def write_report(args, chunk_rows, rerank_rows, gen_rows, refusal_results):
    lines = ["# Financial RAG — Evaluation Report", ""]

    # corpus summary
    lines += ["## Corpus", "", "| Ticker | Company | Form | Filed | Chars |", "|---|---|---|---|---|"]
    for meta_path in sorted(config.PROCESSED_DIR.glob("*.meta.json")):
        m = json.loads(meta_path.read_text())
        lines.append(
            f"| {m['ticker']} | {m['company']} | {m['form']} | {m['filing_date']} | {m['char_count']:,} |"
        )
    lines.append("")

    # index stats
    lines += ["## Index", "", "| Strategy | Chunks | Avg tokens/chunk |", "|---|---|---|"]
    for strategy in args.strategies:
        raw = json.loads((config.INDEX_DIR / strategy / "chunks.json").read_text())
        avg_tok = round(sum(c["token_count"] for c in raw) / max(1, len(raw)), 1)
        lines.append(f"| {strategy} | {len(raw)} | {avg_tok} |")
    lines.append("")

    # chunking comparison
    lines += [
        "## 1. Chunking strategy comparison (hybrid retrieval, no rerank)",
        "",
        f"Metrics @k={TOP_K} over {len(chunk_rows[args.strategies[0]])} answerable queries.",
        "",
        "| Strategy | Hit@k | MRR | Precision@k |",
        "|---|---|---|---|",
    ]
    for strategy in args.strategies:
        r = chunk_rows[strategy]
        lines.append(
            f"| {strategy} | {avg(r,'hit'):.2f} | {avg(r,'mrr'):.3f} | {avg(r,'precision'):.3f} |"
        )
    lines.append("")

    # reranking impact
    lines += [
        "## 2. Reranking impact (hybrid retrieval, with vs without reranker)",
        "",
        "| Strategy | Variant | Hit@k | MRR | Precision@k |",
        "|---|---|---|---|---|",
    ]
    for strategy in args.strategies:
        b, rr = chunk_rows[strategy], rerank_rows[strategy]
        lines.append(
            f"| {strategy} | retrieval-only | {avg(b,'hit'):.2f} | {avg(b,'mrr'):.3f} | {avg(b,'precision'):.3f} |"
        )
        lines.append(
            f"| {strategy} | + rerank | {avg(rr,'hit'):.2f} | {avg(rr,'mrr'):.3f} | {avg(rr,'precision'):.3f} |"
        )
    lines.append("")

    # generation
    if gen_rows:
        faith = sum(r.get("faithfulness", 0) for r in gen_rows) / len(gen_rows)
        rele = sum(r.get("relevance", 0) for r in gen_rows) / len(gen_rows)
        lines += [
            "## 3. Generation quality (LLM judge, full pipeline)",
            "",
            f"- Faithfulness: **{faith*100:.0f}%**",
            f"- Relevance: **{rele*100:.0f}%**",
            "",
        ]
    if refusal_results:
        ok = sum(1 for _, refused in refusal_results if refused)
        lines += [
            "## 4. Refusal path",
            "",
            f"- Unanswerable queries correctly refused: **{ok}/{len(refusal_results)}**",
            "",
        ]

    out = config.EVAL_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written -> {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
