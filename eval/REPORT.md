# Financial RAG — Evaluation Report

## Corpus

| Ticker | Company | Form | Filed | Chars |
|---|---|---|---|---|
| AAPL | Apple Inc. | 10-K | 2025-10-31 | 210,370 |
| AMZN | Amazon.com, Inc. | 10-K | 2026-02-06 | 287,545 |
| GOOGL | Alphabet Inc. | 10-K | 2026-02-05 | 350,262 |
| MSFT | Microsoft Corporation | 10-K | 2025-07-30 | 319,458 |
| NVDA | NVIDIA Corporation | 10-K | 2026-02-25 | 344,309 |

## Index

| Strategy | Chunks | Avg tokens/chunk |
|---|---|---|
| fixed | 477 | 795.6 |
| semantic | 546 | 591.4 |

## 1. Chunking strategy comparison (hybrid retrieval, no rerank)

Metrics @k=5 over 29 answerable queries.

| Strategy | Hit@k | MRR | Precision@k |
|---|---|---|---|
| fixed | 0.86 | 0.794 | 0.648 |
| semantic | 0.97 | 0.886 | 0.662 |

## 2. Reranking impact (hybrid retrieval, with vs without reranker)

| Strategy | Variant | Hit@k | MRR | Precision@k |
|---|---|---|---|---|
| fixed | retrieval-only | 0.86 | 0.794 | 0.648 |
| fixed | + rerank | 0.97 | 0.860 | 0.690 |
| semantic | retrieval-only | 0.97 | 0.886 | 0.662 |
| semantic | + rerank | 0.97 | 0.897 | 0.655 |

## 3. Generation quality (LLM judge, full pipeline)

- Faithfulness: **97%**
- Relevance: **93%**

## 4. Refusal path

- Unanswerable queries correctly refused: **3/3**
