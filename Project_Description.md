# 📊 Project Description — FinDocRAG

> **Financial Document Intelligence Pipeline** — cited, grounded Q&A over SEC 10-K filings.

| | |
|---|---|
| **Program** | The Gen Academy · Week 2 Project |
| **Use case** | #2 — Financial Document Intelligence Pipeline |
| **Build track** | Track 2 — code-heavy (LangChain + LangGraph, Python) |
| **Repo** | https://github.com/lalith34/FinDocRAG |

---

## Part 1 — The Primer: One-liner

> **My RAG app helps equity research analysts answer financial-performance, risk, and strategy questions from the latest SEC 10-K filings of five mega-cap tech companies (AAPL, MSFT, GOOGL, AMZN, NVDA — extensible to any US 10-K filer at runtime) in a Streamlit chat app with 85% faithfulness and 95% relevance (LLM-judged, 20-question analyst set).**

### Checking the three rules

| Rule | How this project meets it |
|---|---|
| **1 · Name the corpus specifically** | Five latest 10-K filings (~1.5M chars of cleaned text, ~477–546 chunks per strategy), pulled live from SEC EDGAR; registry-backed, so tickers can be added/removed without code changes. |
| **2 · Faithfulness, not just relevance** | Measured by an LLM judge (`claude-sonnet-4-6`, a *different* model from the generator) over the full pipeline: **85% faithfulness / 95% relevance** on the 20-question analyst set (`eval/analyst_questions_report.md`); **97% faithfulness** on the broader 29-query set (`eval/REPORT.md`). |
| **3 · Latency is a first-class constraint** | Target ceiling **≤ 15 s** end-to-end. Measured from telemetry (199 logged queries): **median 7.6 s, p90 15.4 s** (retrieval + ONNX rerank + generation). |

---

## Part 2 — The Framework

Every layer of the RAG stack, one decision per row — nothing hand-waved.

| # | Field | Decision (1–2 sentences) |
|---|---|---|
| 1 | **Use case** | Equity research analysts ask factual questions about company financials, risks, and strategy ("What was Microsoft's cloud revenue?", "Compare Apple and NVIDIA's supply-chain risks") and get cited, grounded answers in a Streamlit chatbot (plus a CLI for scripting). |
| 2 | **Corpus** | Latest SEC 10-K filings (HTML from EDGAR) for AAPL, MSFT, GOOGL, AMZN, NVDA — ~1.5M characters of cleaned English text, ~477 fixed / ~546 semantic chunks. Source of truth is SEC EDGAR; the company list is a runtime-editable JSON registry (`data/companies.json`), so any US-listed 10-K filer can be added or removed live. |
| 3 | **Ingestion + cleaning** | EDGAR submissions API → latest 10-K primary HTML, cached raw on disk. Cleaning strips script/style/head, decodes entities, flattens tables to pipe-delimited rows (keeps dollar figures attached to their labels), and strips inline-XBRL tagging noise; the cleaner is deterministic, so processed text is byte-stable. |
| 4 | **Ingestion + freshness** | Incremental rebuilds: a ticker is skipped when its cached EDGAR accession number already matches the latest filing, and only changed tickers are re-chunked/re-embedded/re-upserted. 10-Ks are annual, so freshness SLA = re-run `scripts/build_index.py` after each filing season (or on demand when adding a ticker). |
| 5 | **Chunking + embedding** | Two strategies compared head-to-head — **fixed** (800-token windows, 120 overlap) vs **semantic** (sentence-embedding breakpoints at the 90th-percentile cosine-distance gap, 120–1000 token guards) — with every chunk tagged with its 10-K section (e.g. "Item 1A — Risk Factors"). Embeddings: OpenAI `text-embedding-3-small` (1536-dim), because 800-token chunks match the model's capacity and an on-disk embed cache makes runs reproducible. |
| 6 | **Retrieve** | **Hybrid**: Pinecone serverless (cosine, one namespace per chunking strategy) for dense + local BM25 for sparse, fused with weighted Reciprocal Rank Fusion (RRF k=60) — 20 candidates pulled, **top-k = 5** to the generator. A rule-based router sets dense/sparse weights per query type and scopes single-company questions to that ticker; an alternative **vectorless PageIndex retriever** (LLM navigates the filing's section tree) is selectable per query. |
| 7 | **Rerank** | Local ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed — no torch, runs on Intel macOS) reorders the 20-candidate pool to top-5. Measured impact: semantic strategy Hit@5 0.95→1.00, MRR 0.860→0.912, NDCG 0.877→0.923. |
| 8 | **Generate + cite** | Multi-provider generation (default `claude-opus-4-8`; Anthropic, OpenAI, or **Nebius Token Factory** open-source models like `meta-llama/Llama-3.3-70B-Instruct` selectable per query from a dropdown — Nebius satisfies the handout's Token Factory requirement). Cite-every-claim `[n]` prompt over structured `<source>` context with lost-in-the-middle bookend ordering; every answer carries section-level source cards. |
| 9 | **The "I don't know" path** | Designed first: the prompt mandates the exact refusal "I could not find this in the filings I have indexed" when no source supports an answer, and deterministic guardrails screen input (prompt injection, investment-advice solicitation, length cap) before retrieval and audit output (dangling-citation check, standing not-investment-advice disclaimer) after. |
| 10 | **Evaluation** | LLM-as-judge (`claude-sonnet-4-6`, separate from the generator) + retrieval metrics (Hit@k, MRR, NDCG@k, Precision@k) over labeled query sets, plus an opt-in RAGAS second opinion and a generation-isolation mode to localize failures to retrieval vs prompt. Results: **85% faithfulness / 95% relevance** (analyst set), **97% faithfulness** (29-query set); 136 unit tests + CI. |
| 11 | **Latency** | Ceiling ≤ 15 s; measured median **7.6 s**, p90 **15.4 s** end-to-end (199 telemetry-logged queries). Kept in budget by a rule-based (no-LLM) router and guardrails, local ONNX reranking, and an embed cache. |

---

## Build track

**Track 2 — code-heavy (LangChain + LangGraph, Python).**

The query flow is a compiled **LangGraph `StateGraph`** (`src/graph.py`):

```
input-guard → route → retrieve (vector / per-company comparison / PageIndex)
            → rerank → generate → citation-audit
```

with the routing decisions expressed as conditional edges. All generation runs through **LangChain chat models** (`ChatAnthropic` / `ChatOpenAI`); LangChain also powers the RAGAS evaluation harness. The domain primitives the track teaches (loaders, splitters, retrievers) are custom implementations invoked as graph nodes — full control over chunking/rerank logic and evals-as-code, exactly the Track 2 tradeoff the handout describes.

> **Nebius Token Factory requirement** — generation can run on Nebius-served open-source models (`meta-llama/Llama-3.3-70B-Instruct`, `openai/gpt-oss-120b`) via its OpenAI-compatible API: set `NEBIUS_API_KEY` and pick the model from the answer-model dropdown (or `scripts/ask.py --model`). The shipped `eval/REPORT.md` was generated with `openai/gpt-oss-120b` via Nebius.

---

## Deliverables mapping (Project 2 requirements)

| Requirement | Status | Where |
|---|:---:|---|
| Working financial RAG pipeline | ✅ | `app.py` (Streamlit) + `scripts/ask.py` (CLI) |
| Two chunking strategies compared | ✅ | `eval/REPORT.md` §1 · `eval/analyst_questions_report.md` §1 |
| Reranking step + measured improvement | ✅ | `eval/REPORT.md` §2 · `eval/analyst_questions_report.md` §2 |
| Nebius Token Factory model call | ✅ | `src/generate.py` provider wiring; eval report generated via Nebius |
| Chatbot UI (bonus add-on) | ✅ | Streamlit chat with model dropdown, source cards, live cost/latency |

---

## Results at a glance

| Metric | Value |
|---|---|
| Chunking winner (Hit@5, hybrid, no rerank) | **semantic 0.97** vs fixed 0.86 |
| Rerank impact (fixed, Hit@5) | 0.86 → **0.93** |
| Faithfulness (LLM judge) | **97%** (29-query set) · 85% (analyst set) |
| Relevance (LLM judge) | 90% (29-query set) · **95%** (analyst set) |
| Refusals on unanswerable queries | **3/3** correct |
| Latency (199 logged queries) | median **7.6 s** · p90 15.4 s |
