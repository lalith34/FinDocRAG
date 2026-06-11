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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import embeddings  # noqa: E402
from src.eval_utils import is_relevant, rank_metrics  # noqa: E402
from src.generate import generate  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.reliability import anthropic_retry, make_anthropic_client  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.vectorstore import PineconeStore  # noqa: E402

TOP_K = config.TOP_K


def avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(1, len(rows))


def gold_chunks(store: PineconeStore, query: dict, k: int) -> list:
    """The chunks the relevance labels say SHOULD be retrieved for this query.
    Feeding these straight to the generator tests generation in isolation from
    retrieval (deck S2 §12: "give the LLM the correct chunks, skip retrieval — if
    the answer is wrong, the prompt is the bug"). Stable order by chunk_id."""
    rel = sorted(
        (c for c in store.chunks if is_relevant(c, query)), key=lambda c: c.chunk_id
    )
    return rel[:k]


# --- RAGAS framework eval (deck S2 §12) --------------------------------------
# Runs *alongside* the hand-rolled judge, not instead of it. The pipeline pass
# (hybrid -> rerank -> generate) is cached to a JSONL so re-scoring with RAGAS
# never re-spends those tokens; RAGAS then scores faithfulness, answer relevancy,
# and context precision/recall with its own LLM-as-judge.
def _ragas_ckpt_path(queries_name: str, strategy: str) -> Path:
    stem = Path(queries_name).stem
    return config.LOGS_DIR / f"ragas_inputs_{stem}_{strategy}.jsonl"


def build_ragas_rows(pipe, answerable, ckpt: Path, *, fresh=False, pace=0.0) -> list[dict]:
    if fresh:
        ckpt.unlink(missing_ok=True)
    done = _load_judge_ckpt(ckpt)  # same id-keyed JSONL resume loader
    rows = []
    with ckpt.open("a", encoding="utf-8") as fh:
        for q in answerable:
            if q["id"] in done:
                rows.append(done[q["id"]])
                continue
            # Go through the production pipeline (router + COMPARISON per-ticker
            # quota), not a flat hybrid->rerank top-k, so the scored answer is the
            # one the app would actually produce.
            res = pipe.answer(q["question"])
            top = res.retrieved
            row = {
                "id": q["id"],
                "user_input": q["question"],
                "response": res.answer.text,
                "retrieved_contexts": [c.text for c in top],
            }
            if q.get("reference"):
                row["reference"] = q["reference"]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"  [ragas-gen] {q['id']} done", flush=True)
            if pace:
                time.sleep(pace)
    return rows


def score_ragas(rows: list[dict]) -> dict:
    """Score cached rows with RAGAS. Reference-free metrics (faithfulness, answer
    relevancy, reference-free context precision) run over every row; the
    reference-based metrics (context recall, reference context precision) run only
    over the subset of rows that carry a `reference`."""
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

    # Judge on Claude (JUDGE_MODEL); keep OpenAI only for the embeddings the
    # answer-relevancy metric needs (Anthropic has no embeddings endpoint).
    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=config.JUDGE_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            # RAGAS's context-precision/recall prompts reason over all retrieved
            # chunks; 1024 truncated some verdicts (LLMDidNotFinishException), so
            # give the judge ample room.
            max_tokens=4096,
            timeout=120,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY)
    )
    # Low concurrency + generous retries to ride out per-model rate limits.
    rc = RunConfig(max_workers=4, timeout=180, max_retries=10, max_wait=60)

    def _means(result, metrics) -> dict:
        df = result.to_pandas()
        return {m.name: float(df[m.name].dropna().mean()) for m in metrics}

    scores = {"_version": ragas.__version__, "n": len(rows)}
    free = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()]
    ds = EvaluationDataset.from_list(
        [
            {k: r[k] for k in ("user_input", "response", "retrieved_contexts")}
            for r in rows
        ]
    )
    scores.update(_means(ragas_evaluate(ds, metrics=free, llm=llm, embeddings=emb, run_config=rc), free))

    ref_rows = [r for r in rows if r.get("reference")]
    if ref_rows:
        ref_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
        ds_ref = EvaluationDataset.from_list(
            [
                {k: r[k] for k in ("user_input", "response", "retrieved_contexts", "reference")}
                for r in ref_rows
            ]
        )
        scores["n_ref"] = len(ref_rows)
        scores.update(
            _means(ragas_evaluate(ds_ref, metrics=ref_metrics, llm=llm, embeddings=emb, run_config=rc), ref_metrics)
        )
    return scores


# --- LLM judge ---------------------------------------------------------------
_JUDGE_SYS = (
    "You are evaluating a RAG answer. You are given a QUESTION, the SOURCES the "
    "system was shown, and its ANSWER. Score two things on 0-1:\n"
    "- faithfulness: 1 if every claim in the ANSWER is supported by the SOURCES "
    "(no invented facts), else 0.\n"
    "- relevance: 1 if the ANSWER actually addresses the QUESTION, else 0.\n"
    'Respond ONLY as JSON: {"faithfulness": 0 or 1, "relevance": 0 or 1}.'
)


# Structured-output schema: makes the judge return schema-valid JSON every time
# instead of occasional prose that crashes a parse mid-run.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "enum": [0, 1]},
        "relevance": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["faithfulness", "relevance"],
    "additionalProperties": False,
}


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of a model reply, tolerating stray prose or
    ```json fences around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def judge_answer(question: str, answer_text: str, sources_text: str) -> dict:
    # The judge runs on JUDGE_MODEL (a different, lighter Claude than generation),
    # so it draws from a separate per-model rate-limit bucket and is not the same
    # model it is grading. output_config pins the reply to valid JSON.
    client = make_anthropic_client()

    @anthropic_retry
    def _call():
        return client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=256,
            system=_JUDGE_SYS,
            messages=[
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nSOURCES:\n{sources_text}\n\nANSWER:\n{answer_text}",
                },
            ],
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )

    resp = _call()
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        # A single unparseable verdict (e.g. a safety refusal) must not nuke a long
        # run; flag it so it's visible in the report rather than silently scored.
        print(f"  [judge] WARN unparseable verdict (stop={resp.stop_reason!r}); "
              "defaulting to 0/0", flush=True)
        return {"faithfulness": 0, "relevance": 0, "_parse_error": True}


# --- judge checkpoint --------------------------------------------------------
# The judge pass is the only expensive, rate-limited part of the eval and it can
# run for many minutes against a 30K TPM ceiling. We append each verdict to a
# JSONL checkpoint as soon as it is computed and resume from it on the next run,
# so a crash (or Ctrl-C) never throws away completed work — only the in-flight
# question is repeated.
def _judge_ckpt_path(queries_name: str, strategy: str) -> Path:
    stem = Path(queries_name).stem
    return config.LOGS_DIR / f"judge_{stem}_{strategy}.jsonl"


def _load_judge_ckpt(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM faithfulness judge")
    ap.add_argument("--strategies", nargs="+", default=list(config.STRATEGIES))
    ap.add_argument(
        "--queries",
        default="queries.json",
        help="query/label file under eval/ (e.g. question_bank.json for the "
        "150-question grounded bank). Default: the curated queries.json.",
    )
    ap.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help="seconds to sleep between judged questions; spreads gpt-4o token "
        "use under a tight per-minute rate limit (0 = as fast as retries allow).",
    )
    ap.add_argument(
        "--fresh-judge",
        action="store_true",
        help="ignore any saved judge checkpoint and re-score every question.",
    )
    ap.add_argument(
        "--isolate-generation",
        action="store_true",
        help="also judge generation on the gold (label-relevant) chunks, skipping "
        "retrieval, to separate retrieval failures from generation failures.",
    )
    ap.add_argument(
        "--ragas",
        action="store_true",
        help="also score generation with the RAGAS framework (faithfulness, answer "
        "relevancy, context precision/recall) alongside the custom judge. Requires "
        "`pip install -r requirements-eval.txt`.",
    )
    args = ap.parse_args()

    queries = json.loads((config.EVAL_DIR / args.queries).read_text())
    answerable = [q for q in queries if not q["expect_refusal"]]
    refusal_qs = [q for q in queries if q["expect_refusal"]]

    # retrieval + rerank metrics per strategy ---------------------------------
    chunk_rows = {}   # strategy -> retrieval-only metric rows
    rerank_rows = {}  # strategy -> reranked metric rows

    for strategy in args.strategies:
        store = PineconeStore(strategy)
        base, reranked = [], []
        for q in answerable:
            qvec = embeddings.embed_query(q["question"])
            pool = [c for c, _ in store.hybrid(q["question"], qvec, config.RETRIEVE_K)]
            base.append(rank_metrics(pool, q, TOP_K))
            rr = [c for c, _ in rerank(q["question"], pool, TOP_K)]
            reranked.append(rank_metrics(rr, q, TOP_K))
            print(f"  [{strategy}] {q['id']} done", flush=True)
        chunk_rows[strategy] = base
        rerank_rows[strategy] = reranked

    # generation faithfulness + refusal check ---------------------------------
    gen_rows = []
    iso_rows = []
    refusal_results = []
    if not args.no_judge:
        gen_strategy = args.strategies[-1]  # judge on the richer/last strategy
        store = PineconeStore(gen_strategy)  # used by the §3b isolation pass
        # Judge the answer the *app* produces: route through RAGPipeline so the
        # router + COMPARISON per-ticker quota are exercised, not a flat top-k.
        pipe = RAGPipeline(gen_strategy)

        ckpt = _judge_ckpt_path(args.queries, gen_strategy)
        if args.fresh_judge:
            ckpt.unlink(missing_ok=True)
        done = _load_judge_ckpt(ckpt)
        if done:
            print(f"  [gen] resuming: {len(done)} verdicts loaded from {ckpt.name}",
                  flush=True)

        # Append-mode so each verdict is durably flushed before the next call;
        # an interrupted run resumes here instead of re-spending tokens.
        with ckpt.open("a", encoding="utf-8") as fh:
            for q in answerable:
                if q["id"] in done:
                    gen_rows.append(done[q["id"]])
                    continue
                res = pipe.answer(q["question"])
                top = res.retrieved
                ans = res.answer
                src_text = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(top))
                verdict = judge_answer(q["question"], ans.text, src_text)
                rec = {"id": q["id"], **verdict}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                gen_rows.append(rec)
                print(f"  [gen] {q['id']} faithful={verdict.get('faithfulness')} "
                      f"relevant={verdict.get('relevance')}", flush=True)
                if args.pace:
                    time.sleep(args.pace)

        for q in refusal_qs:
            res = pipe.answer(q["question"])
            refusal_results.append((q["id"], res.answer.refused))
            print(f"  [refusal] {q['id']} refused={res.answer.refused}", flush=True)

        # generation in isolation: feed the gold (label-relevant) chunks directly,
        # bypassing retrieval, so a low score here points at the prompt/model and a
        # high score here (with a low full-pipeline score) points at retrieval.
        if args.isolate_generation:
            iso_ckpt = _judge_ckpt_path(args.queries, gen_strategy).with_name(
                f"judge_iso_{Path(args.queries).stem}_{gen_strategy}.jsonl"
            )
            if args.fresh_judge:
                iso_ckpt.unlink(missing_ok=True)
            iso_done = _load_judge_ckpt(iso_ckpt)
            with iso_ckpt.open("a", encoding="utf-8") as fh:
                for q in answerable:
                    if q["id"] in iso_done:
                        iso_rows.append(iso_done[q["id"]])
                        continue
                    gold = gold_chunks(store, q, TOP_K)
                    if not gold:
                        print(f"  [iso] {q['id']} no gold chunks; skipped", flush=True)
                        continue
                    ans = generate(q["question"], gold)
                    src_text = "\n\n".join(
                        f"[{i+1}] {c.text}" for i, c in enumerate(gold)
                    )
                    verdict = judge_answer(q["question"], ans.text, src_text)
                    rec = {"id": q["id"], "n_gold": len(gold), **verdict}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    iso_rows.append(rec)
                    print(f"  [iso] {q['id']} faithful={verdict.get('faithfulness')} "
                          f"relevant={verdict.get('relevance')}", flush=True)
                    if args.pace:
                        time.sleep(args.pace)

    # RAGAS framework eval (independent of the custom judge; runs even with
    # --no-judge). Own resumable cache for the pipeline pass.
    ragas_scores = {}
    if args.ragas:
        gen_strategy = args.strategies[-1]
        rpipe = RAGPipeline(gen_strategy)
        rows = build_ragas_rows(
            rpipe,
            answerable,
            _ragas_ckpt_path(args.queries, gen_strategy),
            fresh=args.fresh_judge,
            pace=args.pace,
        )
        print("  [ragas] scoring with RAGAS ...", flush=True)
        ragas_scores = score_ragas(rows)

    write_report(args, answerable, chunk_rows, rerank_rows, gen_rows, iso_rows,
                 refusal_results, ragas_scores)


def write_report(args, answerable, chunk_rows, rerank_rows, gen_rows, iso_rows,
                 refusal_results, ragas_scores=None):
    lines = ["# Financial RAG — Evaluation Report", ""]
    lines += [f"_Query set: `{args.queries}` ({len(answerable)} answerable queries)._", ""]

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
        "| Strategy | Hit@k | MRR | NDCG@k | Precision@k |",
        "|---|---|---|---|---|",
    ]
    for strategy in args.strategies:
        r = chunk_rows[strategy]
        lines.append(
            f"| {strategy} | {avg(r,'hit'):.2f} | {avg(r,'mrr'):.3f} | "
            f"{avg(r,'ndcg'):.3f} | {avg(r,'precision'):.3f} |"
        )
    lines.append("")

    # reranking impact
    lines += [
        "## 2. Reranking impact (hybrid retrieval, with vs without reranker)",
        "",
        "| Strategy | Variant | Hit@k | MRR | NDCG@k | Precision@k |",
        "|---|---|---|---|---|---|",
    ]
    for strategy in args.strategies:
        b, rr = chunk_rows[strategy], rerank_rows[strategy]
        lines.append(
            f"| {strategy} | retrieval-only | {avg(b,'hit'):.2f} | {avg(b,'mrr'):.3f} | "
            f"{avg(b,'ndcg'):.3f} | {avg(b,'precision'):.3f} |"
        )
        lines.append(
            f"| {strategy} | + rerank | {avg(rr,'hit'):.2f} | {avg(rr,'mrr'):.3f} | "
            f"{avg(rr,'ndcg'):.3f} | {avg(rr,'precision'):.3f} |"
        )
    lines.append("")

    # per-category breakdown (only when the query set carries category labels,
    # e.g. the grounded question bank). Reported on the reranked variant since
    # that is the production retrieval path.
    if any("category" in q for q in answerable):
        cats = [q.get("category", "(uncategorized)") for q in answerable]
        lines += [
            "## Per-category retrieval (+rerank)",
            "",
            "| Category | Strategy | n | Hit@k | MRR | NDCG@k | Precision@k |",
            "|---|---|---|---|---|---|---|",
        ]
        for category in sorted(set(cats)):
            idx = [i for i, c in enumerate(cats) if c == category]
            for strategy in args.strategies:
                rr = [rerank_rows[strategy][i] for i in idx]
                lines.append(
                    f"| {category} | {strategy} | {len(rr)} | {avg(rr,'hit'):.2f} | "
                    f"{avg(rr,'mrr'):.3f} | {avg(rr,'ndcg'):.3f} | {avg(rr,'precision'):.3f} |"
                )
        lines.append("")

    # generation
    if gen_rows:
        faith = sum(r.get("faithfulness", 0) for r in gen_rows) / len(gen_rows)
        rele = sum(r.get("relevance", 0) for r in gen_rows) / len(gen_rows)
        n_parse_err = sum(1 for r in gen_rows if r.get("_parse_error"))
        lines += [
            "## 3. Generation quality (LLM judge, full pipeline)",
            "",
            f"- Faithfulness: **{faith*100:.0f}%**",
            f"- Relevance: **{rele*100:.0f}%**",
        ]
        if n_parse_err:
            lines.append(
                f"- ⚠️ {n_parse_err} verdict(s) defaulted to 0/0 (judge returned "
                "no parseable JSON) — excluded reasoning, counted as failures above."
            )
        lines.append("")
    if iso_rows:
        iso_faith = sum(r.get("faithfulness", 0) for r in iso_rows) / len(iso_rows)
        iso_rele = sum(r.get("relevance", 0) for r in iso_rows) / len(iso_rows)
        lines += [
            "## 3b. Generation in isolation (gold chunks fed directly)",
            "",
            f"Generator scored on the {len(iso_rows)} queries with label-relevant "
            "chunks, bypassing retrieval. Compare with §3: if isolation scores high "
            "but the full pipeline scores low, the bug is in retrieval; if both are "
            "low, the bug is in the prompt/model.",
            "",
            f"- Faithfulness (gold chunks): **{iso_faith*100:.0f}%**",
            f"- Relevance (gold chunks): **{iso_rele*100:.0f}%**",
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

    if ragas_scores:
        def _pct(key):
            v = ragas_scores.get(key)
            return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "—"

        lines += [
            "## 5. RAGAS metrics (framework eval)",
            "",
            f"Scored over {ragas_scores.get('n', 0)} answerable queries with RAGAS "
            f"`{ragas_scores.get('_version', '?')}` (judge: {config.JUDGE_MODEL}), "
            "run alongside the custom judge in §3.",
            "",
            "| Metric | Score | Needs reference? |",
            "|---|---|---|",
            f"| Faithfulness | {_pct('faithfulness')} | no |",
            f"| Answer relevancy | {_pct('answer_relevancy')} | no |",
            f"| Context precision | {_pct('llm_context_precision_without_reference')} | no |",
        ]
        if "n_ref" in ragas_scores:
            n_ref = ragas_scores["n_ref"]
            lines += [
                f"| Context precision (vs reference) | {_pct('llm_context_precision_with_reference')} | yes (n={n_ref}) |",
                f"| Context recall (vs reference) | {_pct('context_recall')} | yes (n={n_ref}) |",
            ]
        else:
            lines += [
                "",
                "_Context recall and reference-based precision need a `reference` field "
                "on queries; none present, so those metrics were skipped._",
            ]
        lines.append("")

    # Curated default keeps its historical name; any other query set writes a
    # sibling report so it never clobbers REPORT.md.
    stem = Path(args.queries).stem
    report_name = "REPORT.md" if stem == "queries" else f"{stem}_report.md"
    out = config.EVAL_DIR / report_name
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written -> {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
