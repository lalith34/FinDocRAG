# Financial RAG — Evaluation Report

## Corpus

| Ticker | Company | Form | Filed | Chars |
|---|---|---|---|---|
| AAPL | Apple Inc. | 10-K | 2025-10-31 | 224,234 |
| AMZN | Amazon.com, Inc. | 10-K | 2026-02-06 | 316,345 |
| GOOGL | Alphabet Inc. | 10-K | 2026-02-05 | 389,842 |
| MSFT | Microsoft Corporation | 10-K | 2025-07-30 | 353,412 |
| NVDA | NVIDIA Corporation | 10-K | 2026-02-25 | 364,454 |

## Index

| Strategy | Chunks | Avg tokens/chunk |
|---|---|---|
| fixed | 556 | 797.0 |
| semantic | 603 | 625.3 |

## 1. Chunking strategy comparison (hybrid retrieval, no rerank)

Metrics @k=5 over 11 answerable queries.

| Strategy | Hit@k | MRR | Precision@k |
|---|---|---|---|
| fixed | 1.00 | 0.955 | 0.745 |
| semantic | 1.00 | 0.927 | 0.691 |

## 2. Reranking impact (hybrid retrieval, with vs without reranker)

| Strategy | Variant | Hit@k | MRR | Precision@k |
|---|---|---|---|---|
| fixed | retrieval-only | 1.00 | 0.955 | 0.745 |
| fixed | + rerank | 1.00 | 0.955 | 0.800 |
| semantic | retrieval-only | 1.00 | 0.927 | 0.691 |
| semantic | + rerank | 1.00 | 1.000 | 0.855 |

## 3. Generation quality (LLM judge, full pipeline)

- Faithfulness: **100%**
- Relevance: **100%**

## 4. Refusal path

- Unanswerable queries correctly refused: **1/1**
