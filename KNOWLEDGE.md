# Project Knowledge — Financial Document Intelligence (RAG)

A developer's map of the codebase: what each piece does, how data flows, where
the important decisions live, and the design constraints that explain *why* the
code looks the way it does. Pairs with [README.md](README.md) (user-facing) and
[Project_Description.md](Project_Description.md) (the brief).

---

## 1. What this is

A Retrieval-Augmented Generation pipeline that answers factual questions about
company financials from the **latest SEC 10-K filings** of five mega-caps (AAPL,
NVDA, MSFT, GOOGL, AMZN), with **cited, grounded answers** and an explicit
**refusal path** when the filings don't support an answer. Delivered as a CLI and
a Streamlit chatbot, plus an evaluation harness that produces the two required
deliverables: a **chunking-strategy comparison** and a **reranking-impact
analysis**.

---

## 2. End-to-end data flow

```
                    BUILD (scripts/build_index.py -> pipeline.build_indexes)
EDGAR 10-K HTML ──ingest──► clean text ──chunk──► chunks ──embed──► Pinecone
   (data/raw)            (data/processed)    (fixed|semantic)  (vectors, 1 ns/strategy)
                                                  │
                                                  └──► data/index/<strategy>/chunks.json
                                                       (BM25 + eval source of truth)

                    QUERY (pipeline.RAGPipeline.answer)
question ──router──► retrieve (hybrid: Pinecone dense + BM25 sparse, RRF) ──►
          rerank (ONNX cross-encoder) ──► generate (gpt-4o, cited) ──► Answer
                                                                       + QueryTrace
```

Build is **incremental**: ingest skips a ticker whose latest EDGAR accession
already matches what's processed, and only changed tickers are re-chunked /
re-embedded / re-upserted.

---

## 3. Where the key things happen (quick index)

| Question | Answer | Location |
|---|---|---|
| Where is text **embedded**? | OpenAI `text-embedding-3-small`, 1536-dim, disk-cached | [src/embeddings.py](src/embeddings.py) |
| Where are **vectors stored**? | Pinecone serverless, cosine, **one namespace per strategy** | [src/vectorstore.py](src/vectorstore.py) `PineconeStore` |
| What is the **BM25 / eval** source of truth? | local `data/index/<strategy>/chunks.json` (text + metadata, no vectors) | `PineconeStore._load_chunks` |
| What **chunking** strategies exist? | `fixed` (token window) and `semantic` (breakpoint) | [src/chunking.py](src/chunking.py) |
| How is **retrieval** done? | hybrid dense + BM25, fused with RRF, weights set by router | `PineconeStore.hybrid` + [src/router.py](src/router.py) |
| Where is **reranking**? | local ONNX cross-encoder (fastembed, no torch) | [src/rerank.py](src/rerank.py) |
| Where is the **answer generated**? | gpt-4o (seeded), cite-everything prompt, refusal path | [src/generate.py](src/generate.py) |
| Where is the **model chosen**? | `config.CHAT_MODELS`; UI dropdown → `answer(model=…)` → `generate` | [config.py](config.py), [app.py](app.py) |

---

## 4. Embedding — `src/embeddings.py`

- Model: `text-embedding-3-small` (1536-dim). The name pins the version (OpenAI
  doesn't rotate dated snapshots for it).
- **On-disk cache** at `data/embed_cache.pkl`, keyed `sha1(model::text)`. So the
  same sentence embedded during *semantic chunking* is free when the *chunk* is
  later indexed, and re-running a build pays nothing once warm → embeddings are
  fully reproducible.
- Token-aware batching (≤128 items, ≤250k tokens/request) using `tiktoken`.
- **Three call sites:** (1) indexing chunk text ([pipeline.py:83](src/pipeline.py));
  (2) embedding sentences for semantic breakpoints ([chunking.py:110](src/chunking.py));
  (3) `embed_query()` for the dense side at query time ([pipeline.py:129](src/pipeline.py)).

## 5. Vector store — `src/vectorstore.py`

- **Pinecone serverless** index `fintech-10k` (cosine, AWS `us-east-1`), created
  on first use. **One namespace per chunking strategy** so `fixed` and `semantic`
  vectors never mix.
- Vectors live **only** in Pinecone. The per-strategy `chunks.json` snapshot is
  the local source of truth for **BM25** and the **eval** (chunk text + metadata).
- `delete_ticker` uses id-prefix listing (`<TICKER>-<strategy>-NNNN`) because
  serverless has no delete-by-metadata-filter — this is how incremental rebuilds
  replace one company without touching the rest.
- **Three retrievers:**
  - `dense(qvec, k, ticker=…)` — Pinecone query, optional `$eq` ticker filter,
    re-sorted `(score desc, id)` for reproducibility (Pinecone is approximate and
    doesn't guarantee tie order).
  - `sparse(query, k, ticker=…)` — BM25 over local chunks; stable sort; mirrors
    the dense ticker filter so a single-company question doesn't pull other
    companies' chunks into fusion.
  - `hybrid(...)` — pulls a pool from both, fuses with `rrf_fuse`.
- `rrf_fuse` — weighted Reciprocal Rank Fusion (`score += w / (rrf_k + rank + 1)`),
  ties broken by `chunk_id` for a stable order. `rrf_k = 60`.

## 6. Chunking — `src/chunking.py`

Both strategies emit the same `Chunk` dataclass, so everything downstream is
strategy-agnostic. `chunk_id = "<TICKER>-<strategy>-NNNN"`.

Every chunk is also tagged with its **10-K section** (`Chunk.section`, e.g.
"Item 1A — Risk Factors"). Sections are detected **once on the original document**
by `_section_spans` (an `Item N.` header regex that excludes pipe-flattened
table-of-contents rows via a `(?!\|)` lookahead), then each chunk is labelled by
the boundary in force at its start offset — found with a forward-moving
`text.find` on the chunk's opening text. Doing this on the source text (not the
chunk text) is what makes it robust: the semantic chunker joins sentences with
spaces, which would otherwise hide a mid-document `\nITEM 7.` header from any scan
of the chunk itself. The section rides into Pinecone metadata, `chunks.json`, the
`<source>` prompt element, and the CLI/UI source cards.

- **fixed** — 800-token window, 120 overlap. Cheap, deterministic, can split
  mid-thought.
- **semantic** — split sentences, embed them, break where adjacent-sentence
  cosine distance crosses the 90th percentile, gated by min/max token guards
  (120–1000) so chunks stay embeddable. A hard 500-token sentence ceiling stops a
  giant flattened table from blowing the embed limit.

> **Note — `Chunk.meta` is vestigial:** the field exists but is never assigned
> (`chunk_document` doesn't set it) and never read (`upsert` builds Pinecone
> metadata from explicit fields). Always `{}`. Safe to remove if touched.

## 7. Router — `src/router.py`

Deterministic, rule-based, pure function of the query string (an LLM classifier
would reintroduce nondeterminism and add latency/cost to every query). Returns a
`Route` that the pipeline acts on:

| Route | Trigger | Effect |
|---|---|---|
| `SMALLTALK` | greeting / <3 chars | skip retrieval, canned reply |
| `CORPUS` | meta-question about what's indexed, no company named (e.g. "what is in the corpus?", "which companies do you have?") | skip retrieval, list all filings from corpus metadata |
| `COMPARISON` | ≥2 companies named | per-company dense quota (no rerank) |
| `LEXICAL` | exact-match signals dominate | dense=1, sparse=2 |
| `SEMANTIC` | conceptual cues dominate | dense=2, sparse=1 |
| `HYBRID` | balanced default | dense=1, sparse=1 |

A **single** named company scopes retrieval to that ticker (filter on both dense
and sparse) so near-identical chunks from other filings can't crowd out the
answer. The lexical/semantic decision needs a margin of ≥2 to flip off the robust
hybrid default.

> **Note — `Route.reason`** is populated but only consumed by router tests; it is
> not logged into `QueryTrace`. Debug-only.

## 8. Rerank — `src/rerank.py`

Local **ONNX cross-encoder** (`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed`, no
torch — works on Intel macOS). Reorders the ~20-candidate hybrid pool down to
top-k by true query relevance. **Degrades gracefully:** if the model can't load,
it falls back to the original retrieval order. Tie-broken by `chunk_id` for
reproducibility. Skipped on COMPARISON queries (the per-company dense path beats
the reranker, which tends to bury each company's income-statement table under
XBRL noise on multi-company queries).

## 9. Generate — `src/generate.py`

- Model: `gpt-4o-2024-08-06` by default (dated snapshot + `temperature=0` +
  `seed=7` for reproducibility). The UI can pick a different model via
  `config.CHAT_MODELS`; `generate(query, chunks, model=…)` threads it through.
- The system prompt is the heart of grounding: cite every claim with `[n]`,
  quote figures exactly, **10-Ks are annual** (never fabricate quarters), map
  numbers to fiscal-year columns only when the header says so, and **refuse**
  (exact `config.REFUSAL_TEXT`) only when no source addresses the subject.
- **Context layout (lost-in-the-middle, deck S2 §9):** `_order_for_context`
  reorders the relevance-ranked top-k into a bookend — strongest chunk first,
  second-strongest last, weaker ones buried in the middle where long-context
  models attend least (`[r1,r3,r5,r4,r2]`). Sources are wrapped in structured
  `<source id="N" company=… ticker=… section=…>` elements inside `<context>`,
  with the question in `<question>` (structured context > plain prose). The
  reorder is **skipped for COMPARISON queries** (`reorder=False`), whose chunks
  are grouped per company rather than globally ranked.
- Returns `Answer(text, sources, refused, usage)`; each `Source` now carries its
  `section`; `usage` carries prompt/completion tokens and `system_fingerprint`
  (reproducibility provenance).

## 10. Ingestion — `src/ingest.py`

- EDGAR submissions API → latest 10-K primary doc HTML.
- **Raw HTML cached** under `data/raw` is the source of truth. Re-cleaning reuses
  it (no network), so processed text is byte-for-byte reproducible — `_clean_html`
  is deterministic, the download is not.
- Cleaning: strip script/style/head, decode entities, **flatten tables to
  pipe-delimited rows** (keeps dollar figures attached to their labels), then
  **strip inline-XBRL tagging noise** (qualified names, lone CIKs, ISO dates,
  durations, taxonomy URLs) using *unambiguous* single-token shapes so real
  tables and prose survive.
- Incremental: skip when the cached accession matches the latest EDGAR accession.

## 11. Cross-cutting

- **Reliability** ([src/reliability.py](src/reliability.py)) — tenacity
  retry/backoff for SEC (5 attempts), OpenAI (8 attempts ≈ 3 min, to ride out a
  30K-TPM rate-limit burst during eval), and Pinecone (5). OpenAI clients use
  `max_retries=0` so tenacity owns the policy.
- **Telemetry** ([src/telemetry.py](src/telemetry.py)) — JSON logging; per-query
  `QueryTrace` (route, timings, tokens, est. cost, model, system_fingerprint)
  appended to `logs/queries.jsonl`; UI thumbs feedback to `logs/feedback.jsonl`.
- **Eval labels** ([src/eval_utils.py](src/eval_utils.py)) — a chunk is
  "relevant" if it's from an expected ticker AND contains an expected keyword;
  yields Hit@k, MRR, **NDCG@k**, Precision@k. NDCG rewards ranking relevant
  chunks higher, not just retrieving them. Shared by the harness and unit tests.

## 12. Scripts & app

| Entry point | Purpose |
|---|---|
| `python -m scripts.build_index` | ingest → chunk → embed → upsert (`--force`, `--refresh-raw`, `--no-ingest`, `--strategies`) |
| `python -m scripts.evaluate` | the two deliverables + LLM-judge + refusal check → `eval/REPORT.md` (`--queries`, `--no-judge`, `--pace`, `--fresh-judge`, `--isolate-generation`). `--isolate-generation` adds §3b: feeds the gold (label-relevant) chunks straight to the generator, bypassing retrieval, so a low score there points at the prompt and a high score there with a low §3 points at retrieval (deck S2 §12 two-stage diagnostic). |
| `python -m scripts.ask "..."` | one-off CLI query (`--strategy`, `--no-rerank`, `--top-k`) |
| `python -m scripts.smoke_test` | offline path check (no API key): ingest, both chunkers w/ fake embedder, BM25, RRF |
| `streamlit run app.py` | chatbot: answer-model dropdown, rerank toggle, top-k slider |

## 13. Config knobs — `config.py`

Corpus (`COMPANIES`), models (`EMBED_MODEL`, `CHAT_MODEL`, `CHAT_MODELS`,
`GEN_SEED`, `MODEL_PRICES`), Pinecone (`PINECONE_INDEX_NAME`, cloud/region),
reranker (`RERANK_ONNX_MODEL`), chunking (`FIXED_*`, `SEMANTIC_*`), retrieval
(`TOP_K=5`, `RETRIEVE_K=20`, `RRF_K=60`), and `REFUSAL_TEXT`. Secrets via `.env`
(`OPENAI_API_KEY`, `PINECONE_API_KEY`).

## 14. Reproducibility design (the through-line)

Almost every "why" in this codebase traces to determinism:
- Pinned embed + chat model snapshots; `temperature=0`, `seed`, logged
  `system_fingerprint`.
- Stable tie-breaking (by `chunk_id`) in dense, sparse, RRF, and rerank — because
  Pinecone/BM25/cross-encoder don't guarantee order for equal scores.
- Rule-based router instead of an LLM classifier.
- Cached raw HTML + deterministic cleaner → byte-stable processed text.
- On-disk embed cache → identical vectors run to run.

## 15. Tests — `tests/`

`pytest` covers chunking (incl. **section detection + carry-forward**),
determinism, eval labels (incl. **NDCG**), generation (**lost-in-the-middle
ordering** in `test_generate.py`), ingest cleaning, ingest reproducibility, router
decisions, and RRF fusion. The expensive Pinecone/LLM paths are exercised by the
online eval, not unit tests. CI: `.github/workflows/ci.yml`.

## 16. Known minor cleanup items

1. `Chunk.meta` — declared, never set or read (§6). Removable.
2. `Route.reason` — set but only tested, not logged into `QueryTrace` (§7).
3. (Resolved) README architecture drift — synced to Pinecone + ONNX in commit
   `5a1b374`.
