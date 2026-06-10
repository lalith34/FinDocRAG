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

Metrics @k=5 over 29 answerable queries.

| Strategy | Hit@k | MRR | Precision@k |
|---|---|---|---|
| fixed | 0.90 | 0.813 | 0.648 |
| semantic | 0.93 | 0.852 | 0.648 |

## 2. Reranking impact (hybrid retrieval, with vs without reranker)

| Strategy | Variant | Hit@k | MRR | Precision@k |
|---|---|---|---|---|
| fixed | retrieval-only | 0.90 | 0.813 | 0.648 |
| fixed | + rerank | 0.97 | 0.871 | 0.669 |
| semantic | retrieval-only | 0.93 | 0.852 | 0.648 |
| semantic | + rerank | 0.97 | 0.897 | 0.662 |

## 3. Generation quality (LLM judge, full pipeline)

- Faithfulness: **97%**
- Relevance: **93%**

## 4. Refusal path

- Unanswerable queries correctly refused: **3/3**
