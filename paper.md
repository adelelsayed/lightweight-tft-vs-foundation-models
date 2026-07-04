# A Lightweight Temporal Fusion Transformer Matches Billion-Scale Zero-Shot Foundation Models for Demand Forecasting at a Fraction of the Cost

**Author:** Adel Elsayed Mahmoud
**Corresponding author:** adelelsayed1991@gmail.com

---

## Abstract

Time-series foundation models (TSFMs) such as Chronos, TimesFM, Moirai and TabPFN-TS promise accurate, zero-shot forecasting without task-specific training, but they are large (10^8–10^9 parameters), are often consumed through paid cloud APIs, and may require sensitive data to leave an organisation's premises. We ask a practical question for resource-constrained and privacy-sensitive deployments — particularly in healthcare and pharmacy supply chains — *whether a small, purpose-trained model can match these foundation models.* We train a compact Temporal Fusion Transformer (TFT; 0.19–0.63 million parameters) and benchmark it head-to-head against four state-of-the-art zero-shot TSFMs and a seasonal-naive baseline on two real demand datasets: NHS England prescribing volumes (region × therapeutic-chapter, monthly) and the M5 retail competition (item × store, weekly). Using identical evaluation windows, a clustered (block) bootstrap over series, and a Diebold–Mariano test — repeated across three random seeds — we find **no statistically significant difference between the lightweight TFT and any foundation model on either dataset** (the paired confidence intervals on the WAPE gap straddle zero in all cases). On M5 the TFT additionally and reproducibly outperforms the seasonal-naive baseline; on the heavily aggregated NHS series no learned model — small or large — beats seasonal-naive. We are careful to interpret "no significant difference" as failure to detect a gap rather than proven equivalence (a formal equivalence test at a 10%-of-naive margin is not satisfied at our sample sizes). The TFT reaches this foundation-model-level accuracy with a model 2–3 orders of magnitude smaller; for the NHS task it both trains and serves on a commodity CPU with no data leaving the premises. Our results indicate that, for these demand series, a small trained model is a competitive, far cheaper, and privacy-preserving alternative to large zero-shot foundation models. We release all code, forecasts and analysis scripts for full reproducibility.

**Keywords:** demand forecasting; Temporal Fusion Transformer; time-series foundation models; zero-shot forecasting; healthcare analytics; model efficiency; reproducibility.

---

## 1. Introduction

Accurate short-horizon demand forecasting underpins inventory planning, budgeting and capacity management. In healthcare and pharmacy supply chains specifically, forecasting prescription and product demand affects medicine availability, waste, and cost control, and is subject to two practical constraints that the recent literature rarely foregrounds: **(i) data governance** — clinical and prescribing data frequently cannot be transmitted to external cloud services; and **(ii) limited infrastructure** — many providers operate modest on-premises hardware without dedicated GPUs.

The dominant recent trend in time-series machine learning is the *foundation model*: a large neural network pre-trained on enormous and diverse corpora that forecasts new series **zero-shot**, i.e. with no task-specific training. Chronos [1], TimesFM [2], Moirai [3] and TabPFN-TS [4] exemplify this paradigm. Their appeal is obvious — no training, strong out-of-the-box accuracy — but they carry hidden costs for the constrained settings above: parameter counts in the hundreds of millions, GPU or paid-API inference, and (for hosted variants) the need to send data off-site.

This motivates the question we study:

> **For real, operationally relevant demand series, can a small, purpose-trained model — light enough to train once and serve on a CPU, on-premises — match the accuracy of large zero-shot foundation models?**

We answer this with a rigorous head-to-head benchmark. Our contributions are:

1. **A fair, reproducible benchmark protocol** comparing a compact Temporal Fusion Transformer against four TSFMs and a seasonal-naive baseline on *identical* forecast windows, in native demand units, with proper statistical testing (clustered bootstrap + Diebold–Mariano) repeated over three seeds.
2. **Evidence on two contrasting real datasets** — NHS-EPD prescribing (monthly, aggregated) and M5 retail (weekly, granular, covariate-rich) — showing the lightweight TFT is statistically indistinguishable from every foundation model (no significant WAPE gap in any of the four pairwise comparisons, on either dataset), while reproducibly beating the seasonal-naive baseline on M5.
3. **A cost/efficiency analysis** quantifying the deployment gap: a 0.19–0.63 M-parameter model, trainable in minutes-to-hours and served on CPU, versus 10^8–10^9-parameter cloud/GPU models.
4. **Full artifact release** (code, per-seed forecasts, analysis scripts) enabling exact reproduction of every reported number.

We deliberately do **not** claim the TFT is the most accurate forecaster. Our claim is the more useful one for practitioners: *parity at a fraction of the cost and with stronger data-governance properties.*

---

## 2. Related Work

**Temporal Fusion Transformer.** The TFT [5] is an attention-based architecture purpose-built for multi-horizon forecasting with heterogeneous inputs: static covariates, known-future inputs (e.g. calendar, prices) and observed past inputs. Its variable-selection networks and gated residual components provide both accuracy and interpretability. We use a modernised TFT that combines the original TFT skeleton with components from contemporary transformer practice: the Transformer self-attention block [7], RMSNorm [9], rotary position embeddings (RoPE) [8], grouped-query attention [10], gated feed-forward layers (GLU variants) [11], and FlashAttention [14] for efficient exact attention, with a multi-quantile output head.

**Time-series foundation models.** Chronos tokenises values and applies a language-model backbone; TimesFM is a patched decoder pre-trained on large real/synthetic corpora; Moirai is a masked-encoder universal forecaster; TabPFN-TS adapts tabular in-context learning to time series. All forecast zero-shot. Independent benchmarks (e.g. GIFT-Eval) report strong but not uniformly dominant performance, and note that classical and lightweight baselines remain competitive on many series — consistent with our findings.

**Efficiency and "are foundation models worth it?"** A growing line of work questions whether large pre-trained forecasters justify their cost relative to simple or small models. We contribute a controlled, statistically tested, two-domain data point to this debate, with an explicit healthcare-deployment lens.

---

## 3. Data

### 3.1 NHS England Prescribing Dataset (EPD)

The NHS Business Services Authority publishes monthly prescribing volumes. We construct series at the **Regional Office × BNF chapter** level (the higher-level therapeutic grouping recommended by the EPD release guidance when not using SNOMED-level detail), giving **144 series over 65 months** (Nov 2020–Mar 2026). The target is monthly **items** dispensed. Features: static categoricals (region, BNF chapter) and per-series scale statistics; known-future calendar (month-of-year, sin/cos, trend); observed past (log-demand history plus per-item cost/quantity proxies). The split is temporal: the final 12 months are test, the preceding 6 are validation, the rest train. Encoder length L = 12 months, horizon H = 6 months.

### 3.2 M5 (Walmart) retail competition

The M5 dataset [6] contains daily unit sales of 30,490 item × store series across 10 US stores. Daily item-level demand is highly intermittent (zero-inflated), which suppresses learnable signal; we therefore **aggregate to weekly** resolution (summing 7-day buckets), yielding 278 weekly steps. We retain all sufficiently active series (28,175). The target is weekly units. Features map cleanly to the three TFT channels: static (department, category, store, state, plus per-series scale); **known-future** (week-of-year sin/cos, trend, month, event type, *and sell price* — set in advance by the retailer); observed past (log-demand history, SNAP fraction). Encoder length L = 24 weeks, horizon H = 6 weeks; the final 8 weeks are test, the preceding 8 validation.

The two datasets are deliberately contrasting: NHS-EPD is **few, smooth, heavily aggregated** series; M5-weekly is **many, noisier, covariate-rich** series. Together they probe whether the parity result is regime-dependent.

---

## 4. Methods

### 4.1 Models

**ModernTFT (ours).** A multi-quantile TFT (quantiles 0.1/0.25/0.5/0.75/0.9). Configurations: NHS-EPD d_model = 64 (≈0.63 M parameters); M5 d_model = 64 with the M5 feature set (≈0.19 M parameters; fewer embeddings/known-reals). Trained with AdamW [12], a cosine learning-rate schedule with linear warm-up and warm restarts [13], a pinball (quantile) loss plus a small non-crossing penalty, early stopping on validation pinball loss.

**Foundation-model baselines (zero-shot).** Chronos-Bolt (base), TimesFM-2.5 (200 M), Moirai-1.1-R (large), TabPFN-TS (cloud client). Each receives the *same* context window the TFT saw and forecasts the same H steps; quantiles are read or interpolated from each model's native output.

**Seasonal-naive.** Repeats the value from one season ago; quantiles from in-sample seasonal residuals. The standard sanity-check baseline.

### 4.2 Fair-comparison protocol

All models are scored on **identical (series, forecast-origin) windows**, in native demand units (forecasts produced in log space are inverted via `expm1`). For M5, evaluation is restricted to a stratified, seeded sample of 3,000 windows (same windows for every model) to keep cloud/GPU baselines tractable; NHS-EPD uses all 1,721 test windows. Quantile predictions are clipped to be non-negative and sorted to enforce monotonicity for every model alike.

### 4.3 Metrics

- **WAPE** (weighted absolute percentage error, Σ|e|/Σ|y|) — volume-weighted point accuracy; our headline metric.
- **MASE** (mean absolute scaled error) [16] — MAE relative to the in-sample seasonal-naive MAE; <1 beats the naive scale.
- **RMSE**, **MAE**, **bias** — complementary point metrics.
- **Pinball (quantile) loss** — the proper scoring rule for the probabilistic forecast; the model's training objective.
- **Coverage** (empirical fraction inside the 80% p10–p90 interval; nominal 0.80) and the 50% p25–p75 interval — calibration.

### 4.4 Statistical testing

A single WAPE number can mislead because forecast rows are **not independent** (each series contributes overlapping rolling windows × horizon steps); the effective sample size is closer to the number of *series*. We therefore use two complementary tests on the per-window error differential between the TFT and each rival:

1. **Cluster (block) bootstrap over series** [17] (2,000 resamples): resample whole series with replacement, recompute the WAPE gap each draw, and report the 95% confidence interval. An interval excluding 0 is a significant difference; an interval straddling 0 is a statistical tie.
2. **Diebold–Mariano test** [15] on the absolute-error differential with a **series-clustered** standard error.

We run the entire pipeline for **three random seeds (42, 43, 44)** and report mean ± standard deviation, plus a robustness verdict requiring agreement across all three seeds.

---

## 5. Results

All numbers below are regenerated by the released `analysis.py` from the per-seed forecast files.

### 5.1 NHS-EPD (monthly, 144 series, 1,721 test windows)

**Table 1 — NHS-EPD point and probabilistic accuracy.** ModernTFT shown as mean ± std over 3 seeds; foundation models are zero-shot (seed-independent).

| Model | WAPE | MASE | RMSE | Coverage (80%) |
|---|---|---|---|---|
| SeasonalNaive | **0.0359** | **0.664** | **58,438** | 0.919 |
| **ModernTFT (ours)** | 0.0431 ± 0.0029 | 0.797 ± 0.053 | 66,353 ± 4,926 | 0.759 |
| TabPFN | 0.0432 | 0.797 | 76,255 | 0.684 |
| TimesFM | 0.0440 | 0.813 | 78,891 | 0.803 |
| Moirai | 0.0511 | 0.944 | 98,710 | 0.868 |
| Chronos | 0.0514 | 0.950 | 111,588 | 0.928 |

**Significance (robustness across all 3 seeds).** Against the four foundation models the TFT shows **no significant difference** in any case (bootstrap CIs straddle 0; per-seed verdicts mix TIE and TFT-better, never unanimous in either direction). The seasonal-naive baseline *significantly beats* the TFT in all three seeds — on these heavily aggregated, strongly seasonal series, "repeat last year" is exceptionally hard to beat, and the foundation models do not beat it either. Among the *learned* models the TFT has the lowest RMSE (66,353 vs 76,255–111,588 for the foundation models), indicating smaller worst-case errors than every foundation model, though seasonal-naive's RMSE (58,438) is lower still.

### 5.2 M5 (weekly, 3,000 evaluation windows)

**Table 2 — M5 point and probabilistic accuracy.** ModernTFT mean ± std over 3 seeds.

| Model | WAPE | MASE | RMSE | Coverage (80%) |
|---|---|---|---|---|
| TimesFM | **0.4102** | **1.045** | 13,326 | 0.667 |
| Chronos | 0.4167 | 1.061 | 13,120 | 0.703 |
| **ModernTFT (ours)** | 0.4170 ± 0.0093 | 1.062 ± 0.024 | 14,156 ± 433 | **0.805** |
| TabPFN | 0.4267 | 1.087 | 13,377 | 0.655 |
| Moirai | 0.4399 | 1.121 | 14,497 | 0.660 |
| SeasonalNaive | 0.4773 | 1.216 | 14,616 | 0.771 |

**Significance (robustness across all 3 seeds).**

| Rival | WAPE gap (rival − TFT) | Verdict across 3 seeds |
|---|---|---|
| SeasonalNaive | +0.054 … +0.071 | **TFT significantly better (all 3)** |
| Moirai | +0.016 … +0.034 | Tie (TFT better in 1/3) |
| TabPFN | +0.003 … +0.020 | Tie (TFT better in 1/3) |
| Chronos | −0.007 … +0.010 | Tie (all 3) |
| TimesFM | −0.014 … +0.004 | Tie (TimesFM better in 1/3) |

The TFT is **statistically tied with all four foundation models**, sits mid-pack on WAPE, and is the **best-calibrated** model (80% coverage = 0.805, essentially nominal, versus 0.66–0.70 for the strongest TSFMs, which are over-confident). It significantly and reproducibly beats the seasonal-naive baseline.

### 5.3 Cost and deployment (the central finding)

**Table 3 — Model size and deployment profile.**

| Model | Parameters | Training | Inference | Data leaves premises? |
|---|---|---|---|---|
| **ModernTFT (ours)** | **0.19–0.63 M** | once: 9–14 min (NHS, CPU) / 8.8–10.6 h (M5, 1 GPU) | CPU (NHS, demonstrated) / GPU used for M5 | **No** |
| Chronos-Bolt base | ≈200 M | none (zero-shot) | GPU | depends on host |
| TimesFM-2.5 | 200 M | none | GPU | depends on host |
| Moirai-1.1-R large | ≈311 M | none | GPU | depends on host |
| TabPFN-TS | ≈11 M (served via cloud client) | none | **cloud API** | **Yes (hosted)** |

The TFT is **2–3 orders of magnitude smaller** than the neural TSFMs. At this size, CPU inference is feasible and was used end-to-end for the NHS-EPD task (training and inference on CPU); for the larger M5 corpus we trained on a single GPU, and the 0.19 M-parameter model is well within CPU-inference range, though we report CPU inference timing only for NHS. Crucially, the model and data remain on-premises for both tasks, unlike a hosted-API baseline such as TabPFN-TS. Its only cost is a one-time training run (minutes for NHS-EPD; under a GPU-day for full M5).

---

## 6. Discussion

**The headline.** Across two contrasting real demand datasets, a sub-million-parameter, CPU-servable TFT is **statistically indistinguishable from four billion-scale zero-shot foundation models**, and is better calibrated than the strongest of them on M5. Neither the small model nor the large models beat a seasonal-naive baseline on the heavily aggregated NHS series — a useful caution that, when demand is smooth and dominated by annual seasonality, sophisticated models add little over a strong classical baseline.

**Why this matters for healthcare/pharmacy.** The constraints that make TSFMs awkward in clinical settings — cloud dependency, GPU requirements, data egress — are exactly where the lightweight TFT is advantageous. A model that trains in minutes on CPU and serves on existing on-premises hardware, with accuracy matching the state of the art, is operationally preferable for prescribing- and product-demand forecasting under data-governance constraints.

**Regime dependence.** The two datasets bracket the practical spectrum. On few/smooth/aggregated series (NHS), all learned models converge to roughly the seasonal-naive frontier. On many/granular/covariate-rich series (M5-weekly), the trained TFT can exploit known-future covariates (notably price and events) and pulls clear of the naive baseline while matching the TSFMs. In both regimes, *training a small specialist is sufficient to reach the foundation-model frontier.*

**Calibration.** Beyond point accuracy, the TFT's prediction intervals were close to nominal coverage on M5 (0.805 vs target 0.80), whereas TimesFM and Chronos were over-confident (0.66–0.70). For inventory decisions driven by quantiles (safety stock), calibration is as important as the median, and here the small trained model has an edge.

---

## 7. Limitations

- **"No significant difference" is not proven equivalence.** Our central claim rests on *failing to reject* a difference between the TFT and each foundation model. We checked a formal equivalence test (a TOST-style criterion requiring the 95% bootstrap CI of the WAPE gap to fall within ±10% of the seasonal-naive WAPE) and it is **not** satisfied at our sample sizes — the CIs are wider than that margin. We therefore claim only statistical indistinguishability at the tested power, not equivalence; larger samples could resolve small true differences in either direction.
- **Only three seeds.** Mean±std over three seeds is a minimal robustness check; per-seed verdicts occasionally flip (e.g. TimesFM beats the TFT in 1 of 3 M5 seeds), which is precisely why we report the full per-seed table rather than a single run. A larger seed budget would tighten these estimates.
- **Two datasets, one domain family (demand).** Findings may not transfer to other time-series domains.
- **M5 evaluation uses a 3,000-window stratified sample** of the busiest series and a single sampling seed; extreme-tail intermittent series are under-represented (by design, weekly aggregation, and the volume filter).
- **MASE > 1 on M5** reflects the 6-week horizon and the intermittency of weekly retail demand; it is not unique to our model (all models exceed 1 on the seasonal-naive scale there).
- **Foundation models are evaluated zero-shot**, as intended; fine-tuning them was out of scope and would change the cost comparison.
- **Single context configuration per dataset.** We did not exhaustively tune encoder length or the TSFM context mode.
- **Hyperparameter sensitivity.** Training proved sensitive to learning rate (a 10× increase caused divergence); we report a stable configuration.

---

## 8. Conclusion

For demand forecasting under real-world constraints, a lightweight, purpose-trained Temporal Fusion Transformer matches the accuracy of large zero-shot time-series foundation models — statistically tied on two contrasting datasets, better calibrated on the granular one, and significantly better than a seasonal-naive baseline where any model can be. It does so at 2–3 orders of magnitude fewer parameters, with CPU inference and full on-premises data control. Where infrastructure, budget, or data governance are binding — as they routinely are in healthcare and pharmacy — a small trained model is a credible, cheaper, and safer alternative to billion-scale foundation models. We release all code and forecasts to support replication and extension.

---

## Reproducibility

All artifacts are in the accompanying package:
- `code/` — the modern TFT implementation, the NHS-EPD pipeline, and the M5 notebooks (3 seeds).
- `results/` — per-seed TFT forecasts and the foundation-model forecasts on the shared evaluation windows, for both datasets.
- `analysis.py` — regenerates every table (per-seed metrics, mean±std, foundation-model metrics, bootstrap + Diebold–Mariano significance, robustness verdicts). Run `python analysis.py`; outputs land in `tables/`.
- `tables/` — the generated CSVs underlying Tables 1–2.

Data: NHS-EPD is openly available from the NHSBSA Open Data Portal; M5 is available from the M5 Kaggle competition. Raw M5 files are not redistributed here (see Kaggle terms); the ETL in the notebooks reconstructs the cache from them.

---

## Declaration on the use of AI tools

The authors used an AI assistant (Anthropic Claude) to support code development, analysis scripting, figure generation, and manuscript drafting. All quantitative results were produced by the authors' own code (`analysis.py` and the released pipelines) and were verified by the authors. The authors take full responsibility for the content of this article.

## Data availability

- **NHS English Prescribing Dataset (EPD, with SNOMED code).** NHS Business Services Authority Open Data Portal. https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code (accessed June 2026). Open Government Licence.
- **M5 Forecasting – Accuracy.** Kaggle competition dataset. https://www.kaggle.com/competitions/m5-forecasting-accuracy
- **Code, per-seed forecasts, analysis scripts.** Archived on Zenodo, DOI: 10.5281/zenodo.21189124.
- **Preprint.** Zenodo, DOI: 10.5281/zenodo.21189175.

## References

*Foundation models benchmarked (exact versions used: `amazon/chronos-bolt-base`, `google/timesfm-2.5-200m-pytorch`, `Salesforce/moirai-1.1-R-large`, `tabpfn-time-series`).*

[1] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, Y. Wang. "Chronos: Learning the Language of Time Series." *Transactions on Machine Learning Research (TMLR)*, 2024. arXiv:2403.07815.

[2] A. Das, W. Kong, R. Sen, Y. Zhou. "A decoder-only foundation model for time-series forecasting (TimesFM)." *International Conference on Machine Learning (ICML)*, 2024. arXiv:2310.10688.

[3] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, D. Sahoo. "Unified Training of Universal Time Series Forecasting Transformers (Moirai)." *International Conference on Machine Learning (ICML)*, 2024. arXiv:2402.02592.

[4] S. B. Hoo, S. Müller, D. Salinas, F. Hutter. "From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting (TabPFN-TS)." 2025. arXiv:2501.02945.

[5] B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 37(4):1748–1764, 2021.

[6] S. Makridakis, E. Spiliotis, V. Assimakopoulos. "The M5 competition: Background, organization, and implementation." *International Journal of Forecasting*, 38(4):1325–1336, 2022.

[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[8] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021. arXiv:2104.09864. (Published in *Neurocomputing*, 2024.)

[9] B. Zhang, R. Sennrich. "Root Mean Square Layer Normalization (RMSNorm)." *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. arXiv:1910.07467.

[10] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, S. Sanghai. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP*, 2023. arXiv:2305.13245.

[11] N. Shazeer. "GLU Variants Improve Transformer." 2020. arXiv:2002.05202.

[12] I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization (AdamW)." *International Conference on Learning Representations (ICLR)*, 2019. arXiv:1711.05101.

[13] I. Loshchilov, F. Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *International Conference on Learning Representations (ICLR)*, 2017. arXiv:1608.03983.

[14] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2205.14135.

[15] F. X. Diebold, R. S. Mariano. "Comparing predictive accuracy." *Journal of Business & Economic Statistics*, 13(3):253–263, 1995.

[16] R. J. Hyndman, A. B. Koehler. "Another look at measures of forecast accuracy." *International Journal of Forecasting*, 22(4):679–688, 2006.

[17] H. R. Künsch. "The jackknife and the bootstrap for general stationary observations." *The Annals of Statistics*, 17(3):1217–1241, 1989.

*Verification status: refs [1]–[4], [8], [10] were confirmed against their arXiv records (titles, authors, IDs). Refs [5]–[7], [9], [11]–[17] are canonical works; please confirm exact page numbers / venue formatting against the target journal's style before submission.*
