Below is a **20-question RAG evaluation set**: **5 Basic, 5 Medium, 5 High, 5 Advanced**. I included an **expected retrieval check / answer key** so you can validate whether the RAG app retrieved the right chunks. Amounts below are shown in **USD billions** for readability; the SEC tables generally report values in **millions**.

For cross-company questions, the app should mention that the companies have different fiscal year ends: MSFT ends June 30, Apple ends late September, Amazon and Alphabet end December 31, and NVIDIA ends late January.

## Basic — direct lookup questions

**B1 — MSFT**
**Question:** What were Microsoft’s total revenue, operating income, and net income for fiscal year 2025?
**Expected retrieval check:** Revenue **$281.724B**, operating income **$128.528B**, net income **$101.832B**. 

**B2 — AMZN**
**Question:** What were Amazon’s 2025 net sales by segment: North America, International, AWS, and consolidated total?
**Expected retrieval check:** North America **$426.305B**, International **$161.894B**, AWS **$128.725B**, consolidated **$716.924B**. ([SEC][1])

**B3 — GOOG**
**Question:** What were Alphabet’s 2025 revenues by segment?
**Expected retrieval check:** Google Services **$342.721B**, Google Cloud **$58.705B**, Other Bets **$1.537B**, hedging loss **$0.127B**, total revenues **$402.836B**. ([SEC][2])

**B4 — AAPL**
**Question:** What were Apple’s 2025 net sales by product category?
**Expected retrieval check:** iPhone **$209.586B**, Mac **$33.708B**, iPad **$28.023B**, Wearables, Home and Accessories **$35.686B**, Services **$109.158B**, total net sales **$416.161B**. ([SEC][3])

**B5 — NVDA**
**Question:** What were NVIDIA’s fiscal 2026 revenues by reportable segment?
**Expected retrieval check:** Compute & Networking **$193.479B**, Graphics **$22.459B**, total revenue **$215.938B**. ([SEC][4])

## Medium — comparison or light calculation

**M1 — MSFT, AMZN, GOOG**
**Question:** Compare Microsoft Intelligent Cloud, Amazon AWS, and Google Cloud by revenue and operating income. Which had the highest revenue?
**Expected retrieval check:** AWS had the highest revenue at **$128.725B**. Microsoft Intelligent Cloud revenue was **$106.265B** and Google Cloud revenue was **$58.705B**. Operating income was AWS **$45.606B**, Microsoft Intelligent Cloud **$44.589B**, and Google Cloud **$13.910B**. 

**M2 — AAPL**
**Question:** Which Apple product category grew the fastest year over year in 2025, and which category declined?
**Expected retrieval check:** Services grew fastest at **14%**; Wearables, Home and Accessories declined **4%**. ([SEC][3])

**M3 — NVDA**
**Question:** How much did NVIDIA Data Center revenue increase from fiscal 2025 to fiscal 2026, and how does that compare with Gaming revenue growth?
**Expected retrieval check:** Data Center increased by **$78.551B** from **$115.186B** to **$193.737B**. Gaming increased by **$4.692B** from **$11.350B** to **$16.042B**. ([SEC][4])

**M4 — AMZN, GOOG**
**Question:** Compare AWS and Google Cloud in 2025 by revenue and operating income. Which was larger, and by how much?
**Expected retrieval check:** AWS revenue exceeded Google Cloud revenue by **$70.020B**. AWS operating income exceeded Google Cloud operating income by **$31.696B**. ([SEC][1])

**M5 — MSFT**
**Question:** Which Microsoft reportable segment had the highest operating margin in fiscal 2025? Calculate operating income divided by revenue for each segment.
**Expected retrieval check:** Productivity and Business Processes had the highest margin, about **57.8%**; Intelligent Cloud was about **42.0%**; More Personal Computing was about **25.9%**. 

## High — 4- or 5-ticker multi-document questions

**H1 — all 5 tickers**
**Question:** Rank MSFT, AMZN, GOOG, AAPL, and NVDA by total revenue in their latest annual filings.
**Expected retrieval check:** AMZN **$716.924B**, AAPL **$416.161B**, GOOG **$402.836B**, MSFT **$281.724B**, NVDA **$215.938B**. ([SEC][1])

**H2 — all 5 tickers**
**Question:** Rank the five companies by net income. Does the ranking match the revenue ranking?
**Expected retrieval check:** Net income ranking: GOOG **$132.170B**, NVDA **$120.067B**, AAPL **$112.010B**, MSFT **$101.832B**, AMZN **$77.670B**. The ranking does **not** match revenue ranking; Amazon has the highest revenue but the lowest net income among these five. ([SEC][2])

**H3 — MSFT, AMZN, GOOG, NVDA**
**Question:** Compare the companies’ disclosed cloud or AI-infrastructure-related revenue buckets: Microsoft Cloud, AWS, Google Cloud, and NVIDIA Data Center. Rank them by 2025/2026 size and by year-over-year growth rate.
**Expected retrieval check:** By size: NVIDIA Data Center **$193.737B**, Microsoft Cloud **$168.9B**, AWS **$128.725B**, Google Cloud **$58.705B**. By growth: NVIDIA Data Center about **68.2%**, Google Cloud about **35.8%**, Microsoft Cloud about **22.7%**, AWS about **19.7%**. The answer should state these buckets are **not perfectly comparable** because they are different disclosures. ([SEC][4])

**H4 — all 5 tickers**
**Question:** For each company, identify its largest disclosed revenue bucket and calculate what percentage of total revenue it represents.
**Expected retrieval check:** MSFT Productivity and Business Processes ≈ **42.9%** of revenue; AMZN North America ≈ **59.5%**; GOOG Google Services ≈ **85.1%**; AAPL iPhone ≈ **50.4%**; NVDA Data Center ≈ **89.7%**. 

**H5 — all 5 tickers**
**Question:** What reportable segment structure does each company use in the filing? Summarize the segment names for all five companies.
**Expected retrieval check:** MSFT: Productivity and Business Processes, Intelligent Cloud, More Personal Computing. AMZN: North America, International, AWS. GOOG: Google Services, Google Cloud, Other Bets. AAPL: Americas, Europe, Greater China, Japan, Rest of Asia Pacific, Corporate. NVDA: Compute & Networking and Graphics. 

## Advanced — multi-hop, calculation, synthesis

**A1 — all 5 tickers**
**Question:** Calculate and rank operating margin for all five companies using operating income divided by total revenue.
**Expected retrieval check:** NVDA ≈ **60.4%**, MSFT ≈ **45.6%**, GOOG ≈ **32.0%**, AAPL ≈ **32.0%**, AMZN ≈ **11.2%**. ([SEC][4])

**A2 — all 5 tickers**
**Question:** Calculate and rank net profit margin for all five companies using net income divided by total revenue.
**Expected retrieval check:** NVDA ≈ **55.6%**, MSFT ≈ **36.1%**, GOOG ≈ **32.8%**, AAPL ≈ **26.9%**, AMZN ≈ **10.8%**. ([SEC][4])

**A3 — all 5 tickers**
**Question:** Which company had net income greater than operating income, and what filing items explain that result?
**Expected retrieval check:** Alphabet/GOOG is the key case: operating income was **$129.039B** while net income was **$132.170B**. The answer should retrieve Other income details, including **$29.787B** of other income and a **$24.080B** gain on equity securities, before considering tax effects. ([SEC][2])

**A4 — all 5 tickers**
**Question:** Compare each company’s major services, cloud, or AI-infrastructure-related growth bucket: MSFT Microsoft Cloud, AMZN AWS, GOOG Google Cloud, AAPL Services, and NVDA Data Center. Rank by year-over-year growth rate and note comparability limitations.
**Expected retrieval check:** NVDA Data Center ≈ **68.2%**, GOOG Google Cloud ≈ **35.8%**, MSFT Microsoft Cloud ≈ **22.7%**, AMZN AWS ≈ **19.7%**, AAPL Services ≈ **13.5%**. The response should explicitly say these are **management-disclosed categories, not identical business lines**. ([SEC][4])

**A5 — all 5 tickers**
**Question:** Using only the filings, write a comparative paragraph identifying each company’s main 2025/2026 revenue or profit driver and one caveat or risk signal from the same filing.
**Expected retrieval check:** A strong answer should mention: MSFT’s Microsoft Cloud and segment strength; AMZN’s North America scale plus AWS profitability; GOOG’s Google Services scale and Google Cloud growth, with Other Bets losses; AAPL’s iPhone scale and Services growth/high margin; NVDA’s Data Center/Blackwell-driven growth and customer concentration. 

[1]: https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm "amzn-20251231"
[2]: https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm "goog-20251231"
[3]: https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm "aapl-20250927"
[4]: https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm "nvda-20260125"
[5]: https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm "msft-20250630"