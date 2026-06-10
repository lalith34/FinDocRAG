# Financial Document Intelligence — RAG Pipeline

> My RAG app helps **investment analysts** answer **factual questions about
> company financials & disclosures** from the **latest SEC 10-K filings of AAPL,
> NVDA, MSFT, GOOGL, AMZN** in a **Streamlit chatbot**, with **cited, grounded
> answers and a designed refusal path**.

A Week 2 RAG project implementing the full stack: ingest → clean → chunk (two
strategies) → embed → store → hybrid retrieve → rerank → generate cited answers,
plus an evaluation report and a chatbot UI.

## Framework

| Layer | Choice |
|---|---|
| **Use case** | Analysts asking factual questions over annual filings, in a chat UI |
| **Corpus** | Latest `10-K` for 5 mega-caps, auto-pulled from SEC EDGAR |
| **Ingestion + cleaning** | EDGAR submissions API → primary doc HTML → strip script/style, decode entities, **flatten tables to pipe-delimited rows** (table-aware) |
| **Chunking** | `fixed` (800-tok window, 120 overlap) **vs** `semantic` (sentence-grouped, break at 90th-pct embedding-distance) — compared in the eval |
| **Embedding** | OpenAI `text-embedding-3-small` (1536-dim) with on-disk cache |
| **Retrieve** | Local numpy store; **hybrid** = dense cosine + BM25 sparse, fused with RRF; top-k=5 from a 20-candidate pool |
| **Rerank** | LLM listwise reranker (one call/query); impact measured in the eval |
| **Generate** | `gpt-4o-mini`, cite-everything prompt, **explicit refusal** when context is insufficient |

## Setup

```bash
cd Week_2_Fintech_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your OPENAI_API_KEY
```

## Run

```bash
# 1. Download filings, chunk both ways, embed, build indexes
python -m scripts.build_index

# 2. Produce the deliverable: chunking comparison + reranking impact report
python -m scripts.evaluate            # add --no-judge to skip LLM grading
#   -> writes eval/REPORT.md

# 3a. Ask from the CLI
python -m scripts.ask "What were Apple's total net sales last year?"

# 3b. Or launch the chatbot UI
streamlit run app.py
```

## Layout

```
config.py             # all knobs: models, chunk sizes, top-k, companies
src/
  ingest.py           # EDGAR download + table-aware HTML cleaning
  chunking.py         # fixed-size and semantic chunkers
  embeddings.py       # OpenAI embeddings + batching + cache
  vectorstore.py      # dense / sparse(BM25) / hybrid(RRF) retrieval
  rerank.py           # LLM listwise reranker
  generate.py         # cited generation + refusal path
  pipeline.py         # build_indexes() + RAGPipeline.answer()
scripts/
  build_index.py      # build the corpus
  evaluate.py         # the comparison & impact report
  ask.py              # one-off CLI query
eval/
  queries.json        # labelled test queries (incl. an unanswerable one)
  REPORT.md           # generated
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
