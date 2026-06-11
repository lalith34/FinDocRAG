# PageIndex (vectorless) vs Vector DB — MSFT 10-K (full bank)

_Head-to-head over **all 30 MSFT questions** in `question_bank.json` (every category). Both arms share the generator (`gpt-4o-2024-08-06`) and TOP_K=5; only retrieval differs._

- **Vector** — Pinecone dense + BM25 hybrid → ONNX cross-encoder rerank (production path).
- **PageIndex** — an LLM reasons over a 101-leaf tree of the filing's 11 10-K Items and navigates to leaves. No embeddings, no ANN search.

## 1. Aggregate (n = 30 per arm)

| Metric | Vector | PageIndex |
|---|---|---|
| Hit@5 | 1.00 | 0.97 |
| MRR | 0.918 | 0.911 |
| Precision@5 | 0.820 | 0.520 |
| Faithfulness (custom judge) | 97% | 97% |
| Relevance (custom judge) | 100% | 100% |
| must_contain in answer | 93% | 93% |
| Retrieval latency (ms) | 3676 | 5779 |

Precision@k is mechanically lower for PageIndex because it deliberately returns fewer, more targeted leaves (often 2–3) rather than filling all 5 slots — read it together with MRR, which rewards ranking the best leaf first.

## 2. Per-category Hit@5 / MRR

| Category | n | Vector Hit | Vector MRR | PageIndex Hit | PageIndex MRR |
|---|---|---|---|---|---|
| analytics | 5 | 1.00 | 0.767 | 1.00 | 0.900 |
| comprehensive | 5 | 1.00 | 1.000 | 1.00 | 1.000 |
| hybrid | 5 | 1.00 | 1.000 | 0.80 | 0.700 |
| lexical | 5 | 1.00 | 1.000 | 1.00 | 1.000 |
| problems | 5 | 1.00 | 0.900 | 1.00 | 1.000 |
| semantic | 5 | 1.00 | 0.840 | 1.00 | 0.867 |

## 3. RAGAS framework scores (both arms)

Reference-free metrics over all 30 questions; reference-based metrics over the 9 questions with a filing-verified `reference`. RAGAS judge: `gpt-4o-mini` (RAGAS 0.2.15).

| Metric | Needs reference? | Vector | PageIndex |
|---|---|---|---|
| Faithfulness | no | 94% | 96% |
| Answer relevancy | no | 84% | 85% |
| Context precision | no | 90% | 94% |
| Context precision (vs reference) | yes | 94% | 95% |
| Context recall (vs reference) | yes | 100% | 74% |

## 4. Per-question (PageIndex shows the path it took)

| id | category | vec hit | vec faith | pi hit | pi faith | PageIndex path |
|---|---|---|---|---|---|---|
| qb006 | lexical | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0040, MSFT-fixed-0042, MSFT-fixed-0088` |
| qb007 | lexical | 1 | 1 | 1 | 1 | `Item 8 — Financial Statements → MSFT-fixed-0086, MSFT-fixed-0087` |
| qb008 | lexical | 1 | 1 | 1 | 1 | `Item 8 — Financial Statements → MSFT-fixed-0088` |
| qb009 | lexical | 1 | 1 | 1 | 1 | `Item 8 — Financial Statements → MSFT-fixed-0086, MSFT-fixed-0087` |
| qb010 | lexical | 1 | 1 | 1 | 1 | `Item 8 — Financial Statements → MSFT-fixed-0086, MSFT-fixed-0087` |
| qb031 | hybrid | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0005, MSFT-fixed-0006, MSFT-fixed-0007` |
| qb032 | hybrid | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0004, MSFT-fixed-0007` |
| qb033 | hybrid | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0042, MSFT-fixed-0043, MSFT-fixed-0040` |
| qb034 | hybrid | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0004, MSFT-fixed-0005` |
| qb035 | hybrid | 1 | 1 | 0 | 1 | `Item 1 — Business → MSFT-fixed-0005, MSFT-fixed-0006, MSFT-fixed-0007` |
| qb056 | comprehensive | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0004, MSFT-fixed-0005, MSFT-fixed-0006` |
| qb057 | comprehensive | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0003, MSFT-fixed-0004, MSFT-fixed-0005, MSFT-fixed-0006, MSFT-fixed-0007` |
| qb058 | comprehensive | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0040, MSFT-fixed-0041, MSFT-fixed-0042` |
| qb059 | comprehensive | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0003, MSFT-fixed-0004, MSFT-fixed-0005, MSFT-fixed-0006, MSFT-fixed-0011` |
| qb060 | comprehensive | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0003, MSFT-fixed-0004, MSFT-fixed-0005, MSFT-fixed-0007` |
| qb081 | semantic | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0003, MSFT-fixed-0004, MSFT-fixed-0005, MSFT-fixed-0007` |
| qb082 | semantic | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0003, MSFT-fixed-0005, MSFT-fixed-0007` |
| qb083 | semantic | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0006, MSFT-fixed-0007` |
| qb084 | semantic | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0024, MSFT-fixed-0025, MSFT-fixed-0030` |
| qb085 | semantic | 1 | 1 | 1 | 1 | `Item 1 — Business → MSFT-fixed-0011, MSFT-fixed-0012` |
| qb106 | analytics | 1 | 1 | 1 | 1 | `Item 8 — Financial Statements → MSFT-fixed-0086, MSFT-fixed-0087` |
| qb107 | analytics | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0040, MSFT-fixed-0041, MSFT-fixed-0042` |
| qb108 | analytics | 1 | 0 | 1 | 0 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0040, MSFT-fixed-0041, MSFT-fixed-0042, MSFT-fixed-0086, MSFT-fixed-0087` |
| qb109 | analytics | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0040, MSFT-fixed-0042, MSFT-fixed-0088` |
| qb110 | analytics | 1 | 1 | 1 | 1 | `Item 7 — Management's Discussion and Analysis → MSFT-fixed-0037, MSFT-fixed-0040, MSFT-fixed-0041, MSFT-fixed-0042` |
| qb131 | problems | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0019, MSFT-fixed-0020, MSFT-fixed-0021, MSFT-fixed-0022, MSFT-fixed-0023` |
| qb132 | problems | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0024, MSFT-fixed-0025, MSFT-fixed-0030` |
| qb133 | problems | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0025` |
| qb134 | problems | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0026, MSFT-fixed-0027, MSFT-fixed-0029` |
| qb135 | problems | 1 | 1 | 1 | 1 | `Item 1A — Risk Factors → MSFT-fixed-0016, MSFT-fixed-0017, MSFT-fixed-0018` |
