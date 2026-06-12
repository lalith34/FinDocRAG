# Failure Analysis — Adversarial Hard Set

Companion to [REPORT.md](REPORT.md) (representative 29-query set) and
[hard_set_report.md](hard_set_report.md) (this hard set). Generator:
`openai/gpt-oss-120b` via Nebius Token Factory; judge: `claude-sonnet-4-6`.

## How the hard set was built

Rather than guessing which questions are hard, the set was **mined from
evidence** across prior eval runs:

- **20 questions** from the 150-question bank where the LLM judge scored the
  pipeline's answer 0 on faithfulness or relevance;
- **5 questions** where the production pipeline's logged traces
  (`logs/queries.jsonl`) show the top-5 context contained **no label-relevant
  chunk** (or the first relevant chunk sat at rank 4);
- **4 questions** from the analyst set that failed its judge pass;
- the **3 refusal traps** retained from the curated set.

Every answerable question in `hard_set.json` is therefore a *documented prior
failure*. Scores on it are structurally depressed by selection — that is the
point: it measures the system's worst case, not its expected case.

## The gap

| Query set | n | Faithfulness | Relevance | Refusals |
|---|---|---|---|---|
| Curated representative (`queries.json`) | 29 | **97%** | 90% | 3/3 |
| Full question bank (`question_bank.json`) | 150 | **93%** | 94% | — |
| **Adversarial hard set (`hard_set.json`)** | 29 | **69%** | 72% | 3/3 |

Two observations frame everything below. First, ~20 of the 29 prior failures
**pass on re-run** — most "failures" are borderline cases near the judge's
threshold, not systematic breakage. Second, the refusal path holds even under
adversarial selection (3/3): the system fails by *underperforming*, not by
hallucinating freely.

## Failure modes (from the 17 questions that still fail)

**1. Computation-required questions — the largest faithfulness bucket.**
Questions like "rank the five companies by net income" (H2), "calculate the
largest revenue bucket's share" (H4), "what share of net sales did each
geographic segment represent?" (qb104, qb123), "which segment had the highest
operating margin?" (M5). The grounding prompt deliberately forbids computing
numbers not shown in the sources; 10-Ks rarely print percentages, margins, or
rankings directly. The model either computes anyway (judged unfaithful) or
reports raw figures without the asked-for ratio (judged irrelevant). This is a
**design trade-off surfacing**, not a retrieval bug: strict no-arithmetic
grounding buys 93–97% faithfulness on representative sets at the cost of
analytical questions. *Mitigation: a verified-calculation step (extract figures
with citations → compute outside the LLM → render), or relax the rule to allow
arithmetic over explicitly cited inputs.*

**2. Terse lexical line-item queries.** "Third-party seller services revenue",
"Advertising services revenue", "Google Search and other revenue" (qb012/014/018).
Three-word queries give the cross-encoder almost no signal, and the relevant
evidence is a pipe-flattened table row that looks like noise next to prose
chunks. The answer often lands on an adjacent line item or a total. *Mitigation:
let the LEXICAL route skip or down-weight the reranker (it already boosts BM25),
or expand terse queries with the detected ticker + "revenue table".*

**3. Year-over-year "and why" questions.** "How did Apple's effective tax rate
change year over year, and why?" (qb105), "How did Gaming revenue trend?"
(qb048), "What drove the change in Services revenue?" (qb027 — retrieval miss).
The figure lives in a financial-statements table while the *driver narrative*
lives in a different MD&A chunk; top-5 rarely holds both halves. *Mitigation:
section-aware retrieval (pull one chunk from Item 7 MD&A and one from Item 8
statements for trend questions), or raise k for this query shape.*

**4. Broad synthesis questions.** "Overview of Amazon's major investment and
growth areas" (qb065), "Why does Apple believe its ecosystem keeps customers
loyal?" (qb076), "What regulatory and antitrust risks does Alphabet report?"
(qb142). The answer is spread across many chunks; with 5 chunks the model
generalizes beyond its context to sound complete, and the judge flags the
unsupported glue. *Mitigation: detect breadth (no specific metric named) and
retrieve wider/shallower, or answer as a cited bullet list strictly bounded by
the chunks.*

**5. Reranker demotions under adversarial selection.** On this set the
cross-encoder *lowers* Hit@5 (semantic 0.93 → 0.76; fixed 0.93 → 0.83) — the
mirror image of the curated set, where it helps (0.86 → 0.93). The pattern: when
base retrieval is already strong, the reranker's text-similarity prior demotes
correct-but-ugly chunks (flattened tables, XBRL residue) below fluent-but-wrong
prose. The COMPARISON route already skips reranking for exactly this reason;
the LEXICAL route is the next candidate. *General lesson: reranking helps hard
queries and can hurt easy ones — measure per route, not in aggregate.*

## What this means for the headline numbers

The representative reports remain the honest expected-case measure (93–97%
faithfulness). The hard set bounds the worst case (69%) and localizes the
headroom: **analytical/computational questions and terse table lookups**, not
general grounding. The refusal path and citation discipline hold in both
regimes.

---

_Regenerate either report: `python -m scripts.evaluate --model openai/gpt-oss-120b`
(representative) / add `--queries hard_set.json` (adversarial). Judge verdicts
are checkpointed under `logs/`, so re-runs are free until the question set
changes._
