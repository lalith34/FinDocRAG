# Financial RAG — Evaluation Report

_Query set: `question_bank.json` (150 answerable queries)._

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

Metrics @k=5 over 150 answerable queries.

| Strategy | Hit@k | MRR | Precision@k |
|---|---|---|---|
| fixed | 0.96 | 0.866 | 0.699 |
| semantic | 0.97 | 0.860 | 0.635 |

## 2. Reranking impact (hybrid retrieval, with vs without reranker)

| Strategy | Variant | Hit@k | MRR | Precision@k |
|---|---|---|---|---|
| fixed | retrieval-only | 0.96 | 0.866 | 0.699 |
| fixed | + rerank | 0.92 | 0.873 | 0.683 |
| semantic | retrieval-only | 0.97 | 0.860 | 0.635 |
| semantic | + rerank | 0.94 | 0.831 | 0.639 |

## Per-category retrieval (+rerank)

| Category | Strategy | n | Hit@k | MRR | Precision@k |
|---|---|---|---|---|---|
| analytics | fixed | 25 | 0.88 | 0.793 | 0.576 |
| analytics | semantic | 25 | 0.88 | 0.783 | 0.512 |
| comprehensive | fixed | 25 | 0.88 | 0.833 | 0.648 |
| comprehensive | semantic | 25 | 0.96 | 0.808 | 0.616 |
| hybrid | fixed | 25 | 0.92 | 0.920 | 0.744 |
| hybrid | semantic | 25 | 0.96 | 0.853 | 0.704 |
| lexical | fixed | 25 | 0.96 | 0.893 | 0.728 |
| lexical | semantic | 25 | 0.96 | 0.880 | 0.720 |
| problems | fixed | 25 | 1.00 | 0.940 | 0.784 |
| problems | semantic | 25 | 1.00 | 0.921 | 0.688 |
| semantic | fixed | 25 | 0.88 | 0.860 | 0.616 |
| semantic | semantic | 25 | 0.88 | 0.740 | 0.592 |
