#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis.py — regenerates every number in the paper from the copied artifacts.

Produces, for BOTH datasets (NHS-EPD and M5-weekly):
  * per-seed ModernTFT metrics + mean/std across seeds 42/43/44
  * foundation-model (+ seasonal-naive) metrics on the shared eval windows
  * cluster-bootstrap (over series) + Diebold-Mariano significance, per seed
  * a robustness verdict (does the conclusion hold across all 3 seeds?)

Outputs CSVs into tables/ . Run:  python analysis.py
"""
import os, json, math
import numpy as np, pandas as pd
from math import erfc, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
QCOLS = ["p10", "p25", "p50", "p75", "p90"]; QS = [.1, .25, .5, .75, .9]
SEEDS = [42, 43, 44]
RNG_SEED = 0

def wape(a, f): return np.sum(np.abs(a - f)) / np.sum(np.abs(a))
def pinball(a, p, q): e = a - p; return np.mean(np.maximum(q * e, (q - 1) * e))

def load_uid(path):
    d = pd.read_csv(path)
    d["uid"] = (d.series_id.astype(str) + "_" +
                d.year_month.astype(str) + "_s" + d.step.astype(str))
    return d.set_index("uid")

def metrics(d, mase_scale):
    a = d.target.values; f = d.p50.values; e = a - f
    return dict(
        WAPE=wape(a, f), MASE=np.mean(np.abs(e)) / mase_scale,
        MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), bias=np.mean(f - a),
        pinball=np.mean([pinball(a, d[c].values, q) for c, q in zip(QCOLS, QS)]),
        cov_10_90=np.mean((a >= d.p10) & (a <= d.p90)),
        cov_25_75=np.mean((a >= d.p25) & (a <= d.p75)),
    )

def significance(td, fd, rng, n_boot=2000):
    a = td.target.values; ft = td.p50.values; ff = fd.p50.values; sid = td.series_id.values
    gap = wape(a, ff) - wape(a, ft)                       # >0 => TFT better
    uq = np.unique(sid); by = {u: np.where(sid == u)[0] for u in uq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pk = rng.choice(uq, len(uq), replace=True)
        idx = np.concatenate([by[u] for u in pk])
        boot[b] = wape(a[idx], ff[idx]) - wape(a[idx], ft[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = np.abs(a - ff) - np.abs(a - ft); db = d.mean()
    cl = np.array([d[by[u]].sum() for u in uq]); n = len(d); G = len(uq)
    var = np.sum((cl - n * db / G) ** 2) / n ** 2 * (G / (G - 1))
    se = math.sqrt(var) if var > 0 else float("nan")
    dm = db / se if se and se > 0 else float("nan")
    p = erfc(abs(dm) / sqrt(2)) if np.isfinite(dm) else float("nan")
    verdict = "TFT" if lo > 0 else ("rival" if hi < 0 else "TIE")
    return dict(gap=gap, ci_lo=lo, ci_hi=hi, DM=dm, DM_p=p, n_series=G, verdict=verdict)

def run_dataset(name, tft_paths, fm_paths, mase_scale, out_prefix):
    print("\n" + "#" * 80 + f"\n# {name}\n" + "#" * 80)
    tft = {s: load_uid(tft_paths[s]) for s in SEEDS}
    fms = {m: load_uid(p) for m, p in fm_paths.items()}
    common = set(tft[SEEDS[0]].index)
    for d in list(tft.values()) + list(fms.values()):
        common &= set(d.index)
    common = sorted(common)
    for s in SEEDS: tft[s] = tft[s].loc[common]
    for m in fms: fms[m] = fms[m].loc[common]
    n_windows = len(common) // int(common[0].count("_s") and tft[SEEDS[0]].step.max())
    print(f"common rows: {len(common)} | series: {tft[SEEDS[0]].series_id.nunique()} | H={tft[SEEDS[0]].step.max()}")

    # (1) multi-seed TFT
    tr = pd.DataFrame({s: metrics(tft[s], mase_scale) for s in SEEDS}).T
    tr.index.name = "seed"
    print("\n[TFT per seed]"); print(tr.round(4).to_string())
    summ = pd.DataFrame({"mean": tr.mean(), "std": tr.std(ddof=1)}).T
    print("[TFT mean/std]"); print(summ.round(4).to_string())
    tr.to_csv(f"{HERE}/tables/{out_prefix}_tft_per_seed.csv")
    summ.to_csv(f"{HERE}/tables/{out_prefix}_tft_meanstd.csv")

    # (2) FMs
    fmm = pd.DataFrame({m: metrics(fms[m], mase_scale) for m in fms}).T.sort_values("WAPE")
    print("\n[Foundation models]"); print(fmm.round(4).to_string())
    fmm.to_csv(f"{HERE}/tables/{out_prefix}_fm_metrics.csv")

    # (3) significance per seed, (4) robustness
    rng = np.random.default_rng(RNG_SEED); rows = []
    for s in SEEDS:
        for m in fms:
            r = significance(tft[s], fms[m], rng); r.update(seed=s, rival=m); rows.append(r)
    sg = pd.DataFrame(rows)[["seed", "rival", "gap", "ci_lo", "ci_hi", "DM", "DM_p", "n_series", "verdict"]]
    print("\n[Significance per seed]"); print(sg.round(4).to_string(index=False))
    sg.to_csv(f"{HERE}/tables/{out_prefix}_significance.csv", index=False)

    print("\n[Robustness across all 3 seeds]")
    rob = []
    for m in fms:
        sub = sg[sg.rival == m]
        allwin = (sub.verdict == "TFT").all()
        anylose = (sub.verdict == "rival").any()
        v = ("TFT beats in ALL 3" if allwin else
             "rival beats in >=1" if anylose else "TIE (not unanimous TFT)")
        rob.append(dict(rival=m, gap_min=sub.gap.min(), gap_max=sub.gap.max(),
                        DM_p_max=sub.DM_p.max(), verdict=v))
        print(f"  {m:14s}: gap [{sub.gap.min():+.4f},{sub.gap.max():+.4f}] DM_p_max={sub.DM_p.max():.3f} -> {v}")
    pd.DataFrame(rob).to_csv(f"{HERE}/tables/{out_prefix}_robustness.csv", index=False)
    return tr, fmm, sg

def main():
    os.makedirs(f"{HERE}/tables", exist_ok=True)
    R = f"{HERE}/results"

    # ---- NHS-EPD: MASE scale from a stored comparison (seasonal-naive in-sample MAE) ----
    cs = pd.read_csv(f"{R}/nhs/seed43/comparison_summary.csv", index_col=0)
    nhs_scale = float(cs.loc["TabPFN", "MAE"] / cs.loc["TabPFN", "MASE"])
    run_dataset(
        "NHS-EPD (monthly, region x BNF-chapter, H=6)",
        {s: f"{R}/nhs/tft_forecasts/seed{s}.csv" for s in SEEDS},
        {m: f"{R}/nhs/fm_forecasts/forecasts_{m}.csv"
         for m in ["SeasonalNaive", "Chronos", "TimesFM", "Moirai", "TabPFN"]},
        nhs_scale, "nhs")

    # ---- M5 weekly: MASE scale from stored comparison ----
    cs = pd.read_csv(f"{R}/m5/fm_bench_seed42/comparison_summary.csv", index_col=0)
    m5_scale = float(cs.loc["ModernTFT", "MAE"] / cs.loc["ModernTFT", "MASE"])
    run_dataset(
        "M5 (weekly, item x store, H=6)",
        {s: f"{R}/m5/tft_forecasts/seed{42 if s==42 else s}.csv" for s in SEEDS},
        {m: f"{R}/m5/fm_bench_seed42/forecasts_{m}.csv"
         for m in ["SeasonalNaive", "Chronos", "TimesFM", "Moirai", "TabPFN"]},
        m5_scale, "m5")

    print("\nDone. Tables written to tables/ .")

if __name__ == "__main__":
    main()
