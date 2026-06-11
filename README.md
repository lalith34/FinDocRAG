# FinDocRAG — Financial Document Intelligence (RAG Pipeline)

> My RAG app helps **investment analysts** answer **factual questions about
> company financials & disclosures** from the **latest SEC 10-K filings of AAPL,
> NVDA, MSFT, GOOGL, AMZN** in a **Streamlit chatbot**, with **cited, grounded
> answers and a designed refusal path**.

A Week 2 RAG project implementing the full stack: ingest → clean → chunk (two
strategies) → embed → store → hybrid retrieve → rerank → generate cited answers,
plus an evaluation report and a chatbot UI.

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Language / runtime** | Python 3 | ecosystem for RAG, NLP, and data tooling |
| **UI** | Streamlit | a chat interface in pure Python — answer-model dropdown, sources, live cost/latency |
| **Ingestion** | SEC EDGAR API + BeautifulSoup / lxml | pull the latest 10-K and clean it (table-aware HTML → pipe-delimited rows) |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536-dim) | strong, stable dense vectors; on-disk cache makes builds reproducible & free once warm |
| **Vector DB** | Pinecone (serverless, cosine) | managed dense retrieval; one namespace per chunking strategy |
| **Sparse + hybrid** | BM25 (`rank_bm25`) + Reciprocal Rank Fusion | catches exact matches (tickers, line items) dense search misses; RRF fuses both |
| **Reranker** | ONNX cross-encoder via `fastembed` (no torch) | precise top-k reorder that runs locally — no GPU, works on Intel macOS |
| **Generation** | Anthropic (default `claude-opus-4-8`) **or** OpenAI (`gpt-4o`, …) | multi-provider, switchable per query; cited, grounded answers with a refusal path |
| **Guardrails** | Custom rule-based (no extra LLM call) | input injection/advice/length screening + output citation auditing |
| **Tokenization** | `tiktoken` | token-accurate chunk sizing and batching |
| **Evaluation** | Custom harness (Hit@k, MRR, NDCG, Precision) + LLM judge + RAGAS | chunking comparison + reranking impact, the graded deliverables |
| **Reliability** | `tenacity` | retry/backoff around SEC, OpenAI, and Pinecone calls |
| **Testing / CI** | `pytest` + GitHub Actions | 136 offline unit tests on every push |

> The **Framework** table below maps the same pieces onto the RAG pipeline stages
> (use case → corpus → ingest → chunk → embed → retrieve → rerank → generate →
> guardrails).

## Framework

| Layer | Choice |
|---|---|
| **Use case** | Analysts asking factual questions over annual filings, in a chat UI |
| **Corpus** | Latest `10-K`s, auto-pulled from SEC EDGAR. Ships with 5 mega-caps; any US-listed company with a 10-K can be added/removed at runtime (registry-backed, no code edits) |
| **Ingestion + cleaning** | EDGAR submissions API → primary doc HTML → strip script/style, decode entities, **flatten tables to pipe-delimited rows** (table-aware) |
| **Chunking** | `fixed` (800-tok window, 120 overlap) **vs** `semantic` (sentence-grouped, break at 90th-pct embedding-distance) — compared in the eval |
| **Embedding** | OpenAI `text-embedding-3-small` (1536-dim) with on-disk cache |
| **Retrieve** | **Pinecone** serverless index (dense cosine, one namespace per strategy) + local BM25 sparse, fused with RRF; top-k=5 from a 20-candidate pool |
| **Rerank** | Local **ONNX cross-encoder** (`fastembed`, no torch) reranks the candidate pool; impact measured in the eval |
| **Generate** | Multi-provider — Anthropic (default `claude-opus-4-8`) or OpenAI (`gpt-4o`, …), chosen per query; cite-everything prompt, **explicit refusal** when context is insufficient. The chatbot lets you pick the answer model. |
| **Guardrails** | Deterministic, rule-based (no extra LLM call) — **input**: blocks prompt-injection / instruction-extraction, deflects investment-advice solicitation, caps query length; **output**: audits every `[n]` citation against real sources (flags invented references + ungrounded answers), appends a not-advice disclaimer. See [src/guardrails.py](src/guardrails.py). |

## Setup

```bash
cd Week_2_Fintech_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your OPENAI_API_KEY and PINECONE_API_KEY
```

Required keys (see [.env.example](.env.example)): `OPENAI_API_KEY` (embeddings +
generation), `PINECONE_API_KEY` (dense vector store). The reranker runs locally
from a one-time ~90MB ONNX model download, no key needed.

## Run

```bash
# 1. Download filings, chunk both ways, embed, build indexes
python -m scripts.build_index

# 2. Produce the deliverable: chunking comparison + reranking impact report
python -m scripts.evaluate            # add --no-judge to skip LLM grading
#   -> writes eval/REPORT.md
#   --isolate-generation  adds a retrieval-vs-generation diagnostic (report §3b)
#   --ragas               adds RAGAS framework scores (report §5);
#                         first: pip install -r requirements-eval.txt

# 3a. Ask from the CLI
python -m scripts.ask "What were Apple's total net sales last year?"

# 3b. Or launch the chatbot UI
streamlit run app.py
```

### Manage the corpus (add/remove companies)

The corpus is data, not code — companies live in a registry (`data/companies.json`),
so adding one is a single command, not an edit + full rebuild. Each add is validated
against SEC EDGAR (unknown symbol or a foreign filer with no 10-K is rejected with a
clear message), then only that ticker is ingested, chunked, embedded and indexed; the
rest of the corpus is untouched.

```bash
python -m scripts.add_company TSLA META   # validate -> download 10-K -> index
python -m scripts.remove_company TSLA      # delete vectors + snapshots + files
```

The Streamlit sidebar's **➕ Manage corpus** panel does the same thing live (handy for
a demo): type a ticker, click **Add company**, then immediately ask about it.

## Layout

```
config.py             # all knobs: models, chunk sizes, top-k, companies
src/
  ingest.py           # EDGAR download + table-aware HTML cleaning
  chunking.py         # fixed-size and semantic chunkers
  embeddings.py       # OpenAI embeddings + batching + cache
  vectorstore.py      # Pinecone dense + local BM25 sparse, fused with RRF (hybrid)
  rerank.py           # local ONNX cross-encoder reranker
  generate.py         # cited generation + refusal path
  pipeline.py         # build_indexes() + RAGPipeline.answer()
scripts/
  build_index.py      # build the corpus
  evaluate.py         # the comparison & impact report
  ask.py              # one-off CLI query
eval/
  queries.json        # labelled test queries (refusal + reference fields)
  REPORT.md           # generated
requirements-eval.txt # optional RAGAS deps (kept out of the lean core install)
app.py                # Streamlit chatbot
```

## Design notes

- **Hybrid over pure dense.** Pure dense retrieval misses exact tokens that
  matter in filings — ticker symbols, line-item names like *"stock-based
  compensation"*. BM25 catches those; RRF fuses the two rankings.
- **Refusal first.** The generator is instructed to return a fixed refusal
  string when the retrieved context can't support an answer, and the eval
  verifies it fires on an unanswerable query.
- **Two chunkers, one interface.** Both emit the same `Chunk` record, so the
  store/retriever/generator never care which strategy produced the data — the
  eval just builds both indexes and compares.
```
