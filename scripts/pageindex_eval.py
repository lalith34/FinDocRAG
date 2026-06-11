"""PageIndex (vectorless tree navigation) vs Vector DB (Pinecone hybrid+rerank),
benchmarked over the FULL set of MSFT questions in the grounded question bank.

Three things over the earlier 12-question prototype:
  (a) all 30 MSFT questions (every category), for tighter numbers.
  (b) both arms now run through the same productionised retrievers (src.pageindex
      and the Pinecone store), not a one-off script.
  (c) RAGAS framework scoring of BOTH arms — faithfulness, answer relevancy and
      context precision (reference-free) over all 30, plus reference-based context
      precision and context recall over a curated subset whose `reference` answers
      were verified against the filing text.

The per-question pipeline pass (retrieve -> generate -> custom judge) is cached to
a JSONL so re-scoring with RAGAS never re-spends those tokens.

    python -m scripts.pageindex_eval                 # full run, both arms + RAGAS
    python -m scripts.pageindex_eval --no-ragas      # skip the RAGAS pass
    python -m scripts.pageindex_eval --fresh         # ignore the cache, rerun
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import embeddings, pageindex  # noqa: E402
from src.eval_utils import rank_metrics  # noqa: E402
from src.generate import generate  # noqa: E402
from src.reliability import anthropic_retry, make_anthropic_client  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.vectorstore import PineconeStore  # noqa: E402

TICKER = "MSFT"
TOP_K = config.TOP_K
JUDGE_MODEL = config.JUDGE_MODEL    # custom faithfulness/relevance judge (Claude)
RAGAS_MODEL = config.JUDGE_MODEL    # RAGAS judge (Claude)
BENCH_CKPT = config.LOGS_DIR / "pageindex_bench.jsonl"

# Reference answers verified against data/processed/MSFT.txt (FY2025 10-K), used
# only for RAGAS's reference-based metrics. Kept here rather than in the shared
# question bank so the curated bank stays a pure retrieval/keyword spec.
REFERENCES = {
    "qb006": "Azure and other cloud services revenue grew 34% in fiscal 2025.",
    "qb007": "Intelligent Cloud operating income increased $6.8 billion, or 18%.",
    "qb008": "Microsoft Cloud revenue increased 23% to $168.9 billion.",
    "qb009": "LinkedIn revenue increased 9% (up $1.4 billion), with growth across all lines of business.",
    "qb010": "Productivity and Business Processes operating income increased $10.1 billion, or 17%, led by Microsoft 365 Commercial cloud.",
    "qb031": "Azure and other cloud services grew 34%, driving Intelligent Cloud's server products and cloud services revenue up 23%.",
    "qb056": "Microsoft reports three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
    "qb058": "Total revenue increased $36.6 billion, or 15%, and operating income increased $19.1 billion, or 17%, with growth across every segment.",
    "qb131": "Microsoft discloses cybersecurity risks from sophisticated cyberattacks and security breaches that could disrupt operations, expose data, and harm its reputation.",
}

_client = None


def _cli():
    global _client
    if _client is None:
        _client = make_anthropic_client()
    return _client


# --- custom judge ------------------------------------------------------------
_JUDGE_SYS = (
    "You are evaluating a RAG answer. Given a QUESTION, the SOURCES shown, and the "
    "ANSWER, score on 0-1: faithfulness (every claim supported by SOURCES) and "
    'relevance (answer addresses the QUESTION). Respond ONLY as JSON: '
    '{"faithfulness": 0 or 1, "relevance": 0 or 1}.'
)


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "enum": [0, 1]},
        "relevance": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["faithfulness", "relevance"],
    "additionalProperties": False,
}


@anthropic_retry
def judge(question: str, answer: str, sources: str) -> dict:
    resp = _cli().messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        system=_JUDGE_SYS,
        messages=[
            {"role": "user", "content": f"QUESTION:\n{question}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer}"},
        ],
        output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        return {"faithfulness": 0, "relevance": 0, "_parse_error": True}


def _must_ok(answer: str, q: dict) -> bool:
    a = answer.lower()
    return all(kw.lower() in a for kw in q.get("must_contain", []))


# --- the two retrieval arms --------------------------------------------------
def vector_arm(store: PineconeStore, question: str):
    qv = embeddings.embed_query(question)
    pool = [c for c, _ in store.hybrid(question, qv, config.RETRIEVE_K, ticker=TICKER)]
    top = [c for c, _ in rerank(question, pool, TOP_K)]
    return top, "", ""


def pageindex_arm(retr: pageindex.PageIndexRetriever, question: str):
    return retr.retrieve(question, TICKER, TOP_K)


def run_row(arm: str, retrieve_fn, q: dict) -> dict:
    t0 = time.perf_counter()
    chunks, path, reasoning = retrieve_fn(q["question"])
    t1 = time.perf_counter()
    ans = generate(q["question"], chunks)
    t2 = time.perf_counter()
    m = rank_metrics(chunks, q, TOP_K)
    src_text = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    verdict = judge(q["question"], ans.text, src_text)
    row = {
        "arm": arm,
        "id": q["id"],
        "category": q.get("category", ""),
        "question": q["question"],
        "hit": m["hit"],
        "mrr": m["mrr"],
        "precision": m["precision"],
        "faithfulness": verdict.get("faithfulness"),
        "relevance": verdict.get("relevance"),
        "must_contain": 1.0 if _must_ok(ans.text, q) else 0.0,
        "retrieve_ms": round((t1 - t0) * 1000, 1),
        "gen_ms": round((t2 - t1) * 1000, 1),
        "path": path,
        "reasoning": reasoning,
        # carried for the RAGAS pass so it reuses these instead of regenerating:
        "response": ans.text,
        "retrieved_contexts": [c.text for c in chunks],
    }
    if q["id"] in REFERENCES:
        row["reference"] = REFERENCES[q["id"]]
    return row


# --- benchmark with a resumable cache ----------------------------------------
def _load_ckpt(path: Path) -> dict[tuple[str, str], dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["id"])] = r
    return done


def benchmark(questions, store, retr, *, fresh=False, pace=0.0) -> list[dict]:
    if fresh:
        BENCH_CKPT.unlink(missing_ok=True)
    done = _load_ckpt(BENCH_CKPT)
    if done:
        print(f"  resuming: {len(done)} rows cached in {BENCH_CKPT.name}", flush=True)
    arms = {"vector": lambda x: vector_arm(store, x),
            "pageindex": lambda x: pageindex_arm(retr, x)}
    rows = []
    with BENCH_CKPT.open("a", encoding="utf-8") as fh:
        for q in questions:
            for arm, fn in arms.items():
                if (arm, q["id"]) in done:
                    rows.append(done[(arm, q["id"])])
                    continue
                row = run_row(arm, fn, q)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                print(f"  [{arm}] {q['id']} hit={row['hit']:.0f} "
                      f"faith={row['faithfulness']} must={row['must_contain']:.0f}"
                      + (f" path={row['path']}" if arm == "pageindex" else ""),
                      flush=True)
                if pace:
                    time.sleep(pace)
    return rows


# --- RAGAS pass over the cached rows (both arms) ------------------------------
def score_ragas(rows: list[dict]) -> dict:
    import ragas
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAIEmbeddings
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=RAGAS_MODEL, api_key=config.ANTHROPIC_API_KEY, max_tokens=1024, timeout=120
        )
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY)
    )
    rc = RunConfig(max_workers=3, timeout=180, max_retries=10, max_wait=60)

    def _means(result, metrics):
        df = result.to_pandas()
        return {m.name: float(df[m.name].dropna().mean()) for m in metrics}

    out = {"_version": ragas.__version__, "n": len(rows)}
    free = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()]
    ds = EvaluationDataset.from_list(
        [{"user_input": r["question"], "response": r["response"],
          "retrieved_contexts": r["retrieved_contexts"]} for r in rows]
    )
    out.update(_means(ragas_evaluate(ds, metrics=free, llm=llm, embeddings=emb, run_config=rc), free))

    ref_rows = [r for r in rows if r.get("reference")]
    if ref_rows:
        ref_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
        ds_ref = EvaluationDataset.from_list(
            [{"user_input": r["question"], "response": r["response"],
              "retrieved_contexts": r["retrieved_contexts"], "reference": r["reference"]}
             for r in ref_rows]
        )
        out["n_ref"] = len(ref_rows)
        out.update(_means(
            ragas_evaluate(ds_ref, metrics=ref_metrics, llm=llm, embeddings=emb, run_config=rc),
            ref_metrics))
    return out


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore the bench cache")
    ap.add_argument("--rebuild-tree", action="store_true")
    ap.add_argument("--pace", type=float, default=0.0)
    args = ap.parse_args()

    if args.rebuild_tree:
        pageindex.build_tree(TICKER, rebuild=True, verbose=True)

    store = PineconeStore("fixed")
    retr = pageindex.PageIndexRetriever()

    bank = json.loads((config.EVAL_DIR / "question_bank.json").read_text())
    msft_qs = [q for q in bank if q["expected_tickers"] == [TICKER]]
    print(f"Benchmarking {len(msft_qs)} MSFT questions × 2 arms "
          f"(vector, pageindex) ...\n", flush=True)

    rows = benchmark(msft_qs, store, retr, fresh=args.fresh, pace=args.pace)

    ragas = {}
    if not args.no_ragas:
        print("\n  scoring both arms with RAGAS ...", flush=True)
        ragas = {
            "vector": score_ragas([r for r in rows if r["arm"] == "vector"]),
            "pageindex": score_ragas([r for r in rows if r["arm"] == "pageindex"]),
        }

    write_report(msft_qs, rows, ragas)


def write_report(questions, rows, ragas):
    vec = [r for r in rows if r["arm"] == "vector"]
    pi = [r for r in rows if r["arm"] == "pageindex"]
    n = len(questions)
    tree = pageindex.build_tree(TICKER)
    n_leaves = sum(len(v) for v in tree.values())

    L = ["# PageIndex (vectorless) vs Vector DB — MSFT 10-K (full bank)", ""]
    L += [
        f"_Head-to-head over **all {n} MSFT questions** in `question_bank.json` "
        f"(every category). Both arms share the generator (`{config.CHAT_MODEL}`) and "
        f"TOP_K={TOP_K}; only retrieval differs._",
        "",
        "- **Vector** — Pinecone dense + BM25 hybrid → ONNX cross-encoder rerank (production path).",
        f"- **PageIndex** — an LLM reasons over a {n_leaves}-leaf tree of the filing's "
        f"{len(tree)} 10-K Items and navigates to leaves. No embeddings, no ANN search.",
        "",
        "## 1. Aggregate (n = %d per arm)" % n,
        "",
        "| Metric | Vector | PageIndex |",
        "|---|---|---|",
        f"| Hit@{TOP_K} | {_mean(vec,'hit'):.2f} | {_mean(pi,'hit'):.2f} |",
        f"| MRR | {_mean(vec,'mrr'):.3f} | {_mean(pi,'mrr'):.3f} |",
        f"| Precision@{TOP_K} | {_mean(vec,'precision'):.3f} | {_mean(pi,'precision'):.3f} |",
        f"| Faithfulness (custom judge) | {_mean(vec,'faithfulness')*100:.0f}% | {_mean(pi,'faithfulness')*100:.0f}% |",
        f"| Relevance (custom judge) | {_mean(vec,'relevance')*100:.0f}% | {_mean(pi,'relevance')*100:.0f}% |",
        f"| must_contain in answer | {_mean(vec,'must_contain')*100:.0f}% | {_mean(pi,'must_contain')*100:.0f}% |",
        f"| Retrieval latency (ms) | {_mean(vec,'retrieve_ms'):.0f} | {_mean(pi,'retrieve_ms'):.0f} |",
        "",
        "Precision@k is mechanically lower for PageIndex because it deliberately "
        "returns fewer, more targeted leaves (often 2–3) rather than filling all "
        f"{TOP_K} slots — read it together with MRR, which rewards ranking the best "
        "leaf first.",
        "",
    ]

    # per-category retrieval
    cats = sorted({q["category"] for q in questions})
    L += ["## 2. Per-category Hit@%d / MRR" % TOP_K, "",
          "| Category | n | Vector Hit | Vector MRR | PageIndex Hit | PageIndex MRR |",
          "|---|---|---|---|---|---|"]
    for cat in cats:
        cid = {q["id"] for q in questions if q["category"] == cat}
        v = [r for r in vec if r["id"] in cid]
        p = [r for r in pi if r["id"] in cid]
        L.append(f"| {cat} | {len(cid)} | {_mean(v,'hit'):.2f} | {_mean(v,'mrr'):.3f} | "
                 f"{_mean(p,'hit'):.2f} | {_mean(p,'mrr'):.3f} |")
    L.append("")

    # RAGAS
    if ragas:
        def pct(d, k):
            v = d.get(k)
            return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "—"
        rv, rp = ragas["vector"], ragas["pageindex"]
        L += [
            "## 3. RAGAS framework scores (both arms)", "",
            f"Reference-free metrics over all {rv.get('n', 0)} questions; reference-based "
            f"metrics over the {rv.get('n_ref', 0)} questions with a filing-verified "
            f"`reference`. RAGAS judge: `{RAGAS_MODEL}` (RAGAS {rv.get('_version','?')}).",
            "",
            "| Metric | Needs reference? | Vector | PageIndex |",
            "|---|---|---|---|",
            f"| Faithfulness | no | {pct(rv,'faithfulness')} | {pct(rp,'faithfulness')} |",
            f"| Answer relevancy | no | {pct(rv,'answer_relevancy')} | {pct(rp,'answer_relevancy')} |",
            f"| Context precision | no | {pct(rv,'llm_context_precision_without_reference')} | {pct(rp,'llm_context_precision_without_reference')} |",
            f"| Context precision (vs reference) | yes | {pct(rv,'llm_context_precision_with_reference')} | {pct(rp,'llm_context_precision_with_reference')} |",
            f"| Context recall (vs reference) | yes | {pct(rv,'context_recall')} | {pct(rp,'context_recall')} |",
            "",
        ]

    # per-question table with paths
    L += ["## 4. Per-question (PageIndex shows the path it took)", "",
          "| id | category | vec hit | vec faith | pi hit | pi faith | PageIndex path |",
          "|---|---|---|---|---|---|---|"]
    byk = {(r["arm"], r["id"]): r for r in rows}
    for q in questions:
        v = byk[("vector", q["id"])]
        p = byk[("pageindex", q["id"])]
        L.append(f"| {q['id']} | {q['category']} | {v['hit']:.0f} | {v['faithfulness']} | "
                 f"{p['hit']:.0f} | {p['faithfulness']} | `{p['path']}` |")
    L.append("")

    out = config.EVAL_DIR / "pageindex_msft_report.md"
    out.write_text("\n".join(L))
    print(f"\nReport -> {out}\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
