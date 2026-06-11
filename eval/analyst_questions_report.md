# Financial RAG — Evaluation Report

_Query set: `analyst_questions.json` (20 answerable queries)._

## Corpus

| Ticker | Company | Form | Filed | Chars |
|---|---|---|---|---|
| AAPL | Apple Inc. | 10-K | 2025-10-31 | 210,409 |
| AMZN | Amazon.com, Inc. | 10-K | 2026-02-06 | 287,564 |
| GOOGL | Alphabet Inc. | 10-K | 2026-02-05 | 350,275 |
| MSFT | Microsoft Corporation | 10-K | 2025-07-30 | 319,482 |
| NVDA | NVIDIA Corporation | 10-K | 2026-02-25 | 344,323 |

## Index

| Strategy | Chunks | Avg tokens/chunk |
|---|---|---|
| fixed | 477 | 795.7 |
| semantic | 546 | 591.5 |

## 1. Chunking strategy comparison (hybrid retrieval, no rerank)

Metrics @k=5 over 20 answerable queries.

| Strategy | Hit@k | MRR | NDCG@k | Precision@k |
|---|---|---|---|---|
| fixed | 0.95 | 0.860 | 0.875 | 0.780 |
| semantic | 0.95 | 0.860 | 0.877 | 0.760 |

## 2. Reranking impact (hybrid retrieval, with vs without reranker)

| Strategy | Variant | Hit@k | MRR | NDCG@k | Precision@k |
|---|---|---|---|---|---|
| fixed | retrieval-only | 0.95 | 0.860 | 0.875 | 0.780 |
| fixed | + rerank | 0.95 | 0.900 | 0.901 | 0.720 |
| semantic | retrieval-only | 0.95 | 0.860 | 0.877 | 0.760 |
| semantic | + rerank | 1.00 | 0.912 | 0.923 | 0.780 |

## Per-category retrieval (+rerank)

| Category | Strategy | n | Hit@k | MRR | NDCG@k | Precision@k |
|---|---|---|---|---|---|---|
| Advanced | fixed | 5 | 1.00 | 1.000 | 0.997 | 0.800 |
| Advanced | semantic | 5 | 1.00 | 1.000 | 0.981 | 0.800 |
| Basic | fixed | 5 | 1.00 | 1.000 | 0.972 | 0.840 |
| Basic | semantic | 5 | 1.00 | 1.000 | 0.990 | 0.840 |
| High | fixed | 5 | 0.80 | 0.800 | 0.759 | 0.520 |
| High | semantic | 5 | 1.00 | 0.850 | 0.850 | 0.680 |
| Medium | fixed | 5 | 1.00 | 0.800 | 0.875 | 0.720 |
| Medium | semantic | 5 | 1.00 | 0.800 | 0.871 | 0.800 |

## 3. Generation quality (LLM judge, full pipeline)

- Faithfulness: **85%**
- Relevance: **95%**

## 3b. Generation in isolation (gold chunks fed directly)

Generator scored on the 20 queries with label-relevant chunks, bypassing retrieval. Compare with §3: if isolation scores high but the full pipeline scores low, the bug is in retrieval; if both are low, the bug is in the prompt/model.

- Faithfulness (gold chunks): **100%**
- Relevance (gold chunks): **100%**

## 5. RAGAS metrics (framework eval)

Scored over 20 answerable queries with RAGAS `0.2.15` (judge: claude-sonnet-4-6), run alongside the custom judge in §3.

| Metric | Score | Needs reference? |
|---|---|---|
| Faithfulness | 92% | no |
| Answer relevancy | 49% | no |
| Context precision | 75% | no |
| Context precision (vs reference) | 34% | yes (n=20) |
| Context recall (vs reference) | 63% | yes (n=20) |
