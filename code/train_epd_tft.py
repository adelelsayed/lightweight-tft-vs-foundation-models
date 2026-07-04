"""
train_epd_tft.py
================

NHS England Prescribing Dataset (EPD) monthly drug-demand forecasting with the
ModernTFT model (see modern_tft.py), wrapped in the same trainer idiom as the
BirdCLEF / MIMIC pipeline: a CONFIG dict + a single Trainer class that owns
datasets, loaders, model, loss, optimizer, SequentialLR (warmup -> cosine) with
a warm restart, AMP, gradient accumulation, history JSON, checkpointing, plots,
and a synthetic fallback so the whole thing is runnable before the (open) raw
files are downloaded.

WHY EPD INSTEAD OF MIMIC
------------------------
MIMIC's per-patient date scramble destroys the cross-series calendar axis, which
is exactly the signal a stock/demand forecaster needs. EPD keeps real calendar
months, so this is a genuine aggregate-demand problem and a much better fit for a
TFT: it actually has static / known-future / observed-past covariates to select.

The ModernTFT model is data-agnostic (it consumes a `batch` dict described by
TFTConfig), so it is reused UNCHANGED. Everything below is the data layer.

KEY DESIGN CHANGES vs the MIMIC pipeline
----------------------------------------
  series          one ICU stay (relative time)      -> one (ICB x BNF-chemical) monthly series (real calendar)
  time axis       hours since intime                 -> global month index (shared across all series)
  target          a vital (z-scored)                 -> ITEMS dispensed, log1p then z-scored (count demand)
  static_cat      gender/careunit/admtype/race       -> region / ICB / BNF-chapter / BNF-chemical
  static_real     age                                -> series log-mean & log-std of demand (scale + volatility)
  known (future)  hour-of-day, elapsed frac          -> month-of-year (12), sin/cos month, linear time-trend
  observed (past) vitals incl. target                -> log-demand (target) + cost/item + qty/item + ADQ/item
  split           random by stay                     -> TEMPORAL (train past / val & test future)  [no leakage]
  metric denorm   affine z -> physical               -> expm1 of (z * std + mean) ; adds WAPE (demand metric)

DATA CONTRACT (what the dataset yields, matching ModernTFT.forward):
    static_cat   : (B, n_static_cat)       long    region, ICB, chapter, chemical
    static_real  : (B, n_static_real)      float   series log-mean & log-std demand
    known_cat    : (B, T, 1)               long    month-of-year (0..11)
    known_real   : (B, T, 3)               float   sin(month), cos(month), time-trend
    observed_real: (B, L, n_observed_real) float   [log-demand(target), cost/item, qty/item, adq/item]
    target       : (B, H)                  float   future log-demand (z-scored, channel 0)
  where T = L (encoder/past months) + H (decoder/horizon months).
"""

from __future__ import annotations
import os, sys, json, time, math, glob
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from modern_tft import ModernTFT, TFTConfig

root = os.environ.get("EPD_ROOT", "/home/claude/epd_run")


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # ── Paths ───────────────────────────────────────────────────────────
    # Point `epd_raw` at a directory of monthly EPD CSVs (or .csv.gz), e.g.
    # EPD_202401.csv ... downloaded from the NHSBSA Open Data Portal:
    #   https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code
    # (open data, no credentialing — unlike MIMIC). The ETL streams them in
    # chunks; aggregated monthly series are tiny.
    "epd_raw":        os.path.join(root, "raw"),
    "cache_dir":      os.path.join(root, "cache"),
    "splits_dir":     os.path.join(root, "cache", "splits"),
    "logdir":         os.path.join(root, "tft_epd_forecast", "logs"),
    "resume":         os.path.join(root, "tft_epd_forecast", "checkpoints", "best.pth"),
    "submission_csv": os.path.join(root, "tft_epd_forecast", "forecasts.csv"),

    # ── Series definition / cohort ──────────────────────────────────────
    "series_keys":    ["ICB_CODE", "BNF_CHEMICAL_SUBSTANCE_CODE"],  # one series per (ICB, chemical)
    "target_col":     "ITEMS",            # demand target (number of prescription items)
    "bnf_chapter_filter": None,           # e.g. "04: Central Nervous System" to focus one chapter; None = all
    "min_active_months": 36,              # drop series with too little history to window
    "top_k_series":   None,               # keep only the K highest-volume series (None = all)

    # ── Task / windowing (MONTHLY) ──────────────────────────────────────
    "encoder_len":    24,                 # L months of history (2 seasonal cycles)
    "horizon":        6,                  # H months ahead
    "stride":         1,                  # window stride (months)
    # Temporal split: the LAST `test_months` of calendar are test horizons,
    # the `val_months` before that are val horizons, everything earlier is train.
    "val_months":     6,
    "test_months":    12,

    # ── Model (ModernTFT) ───────────────────────────────────────────────
    "d_model":        64,
    "n_heads":        4,
    "n_kv_heads":     1,
    "n_blocks":       2,
    "dropout":        0.1,
    "quantiles":      (0.1, 0.5, 0.9),

    # ── Optim ───────────────────────────────────────────────────────────
    "batch_size":     256,
    "num_workers":    4,
    "lr":             3e-4,
    "weight_decay":   0.1,
    "feature_lr_mult": 0.3,               # slow group (embeds/LSTM/VSN/GRN) LR multiplier
    "num_epochs":     60,
    "warmup":         5,
    "restart_epoch":  40,
    "restart_lr":     5e-5,
    "accumulation":   1,
    "use_amp":        True,
    "grad_clip":      1.0,

    # ── Ramped aux loss (quantile non-crossing penalty) ─────────────────
    "aux_weight_max":      0.1,
    "aux_ramp_start_epoch": 5,
    "aux_ramp_end_epoch":   20,

    # ── Misc ────────────────────────────────────────────────────────────
    "train_metric_stride": 4,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "is_train":       True,
    "seed":           42,
}

# Raw EPD columns we read (matches the NHSBSA schema in the sample record).
EPD_USECOLS = [
    "YEAR_MONTH", "REGIONAL_OFFICE_CODE", "ICB_CODE",
    "BNF_CHAPTER_PLUS_CODE", "BNF_CHEMICAL_SUBSTANCE_CODE",
    "ITEMS", "TOTAL_QUANTITY", "NIC", "ACTUAL_COST", "ADQ_USAGE", "UNIDENTIFIED",
]
# Static categoricals describing a series (parents of the series keys included).
STATIC_CAT_COLS = ["REGIONAL_OFFICE_CODE", "ICB_CODE",
                   "BNF_CHAPTER_PLUS_CODE", "BNF_CHEMICAL_SUBSTANCE_CODE"]
# Raw monthly channels we aggregate (sum) per series-month.
AGG_COLS = ["ITEMS", "TOTAL_QUANTITY", "NIC", "ACTUAL_COST", "ADQ_USAGE"]


# =============================================================================
# Small utilities (mirror the BirdCLEF helpers)
# =============================================================================
class TeeFile:
    """Write to stdout and a log file simultaneously (for tqdm `file=`)."""
    def __init__(self, stream, path):
        self.stream = stream
        self.log = open(path, "a")
    def write(self, data):
        self.stream.write(data); self.log.write(data)
    def flush(self):
        self.stream.flush(); self.log.flush()


# tqdm is optional; degrade to a no-op wrapper if missing.
try:
    from tqdm.auto import tqdm
except Exception:                                       # pragma: no cover
    def tqdm(it=None, **k):
        return it if it is not None else []


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = math.inf; self.count = 0; self.stop = False
    def step(self, value):                 # lower is better
        if value < self.best - self.min_delta:
            self.best = value; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        return self.stop


def parse_year_month(v) -> int:
    """EPD YEAR_MONTH -> absolute month ordinal (year*12 + month-1).
    Handles 'YYYY-MM', 'YYYYMM', and integer 202603."""
    s = str(v).strip()
    if "-" in s:
        y, m = s.split("-")[:2]
        y, m = int(y), int(m)
    else:
        iv = int(float(s)); y, m = iv // 100, iv % 100
    return y * 12 + (m - 1)


def ord_to_label(ordm: int) -> str:
    y, m = divmod(ordm, 12)
    return f"{y}-{m + 1:02d}"


# =============================================================================
# EPD  ->  cache  (real ETL + synthetic fallback)
# =============================================================================
def _list_epd_files(raw_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")) +
                   glob.glob(os.path.join(raw_dir, "*.csv.gz")))
    return files


def build_epd_cache(cfg):
    """Real ETL from raw monthly EPD CSVs. Streams each file in chunks, filters
    UNIDENTIFIED rows (+ optional BNF chapter), and sums the demand channels per
    (series, month). Aggregated monthly series are small, so this is light."""
    raw = cfg["epd_raw"]; files = _list_epd_files(raw)
    if not files:
        raise FileNotFoundError(
            f"No EPD CSVs under {raw}. Download monthly files from the NHSBSA "
            f"Open Data Portal, or call prepare_synthetic_epd_cache(CONFIG).")

    keys = cfg["series_keys"]; chap = cfg.get("bnf_chapter_filter")
    parts = []
    for fp in files:
        reader = pd.read_csv(fp, usecols=lambda c: c in EPD_USECOLS,
                             chunksize=1_000_000, dtype={"YEAR_MONTH": str})
        for chunk in tqdm(reader, desc=f"{os.path.basename(fp)}"):
            chunk = chunk[chunk["UNIDENTIFIED"].astype(str).str.upper() == "N"]
            if chap is not None:
                chunk = chunk[chunk["BNF_CHAPTER_PLUS_CODE"] == chap]
            if chunk.empty:
                continue
            chunk["ym"] = chunk["YEAR_MONTH"].map(parse_year_month)
            # keep the static descriptors alongside the keys (region/chapter ride along)
            gb_cols = list(dict.fromkeys(keys + STATIC_CAT_COLS + ["ym"]))
            agg = chunk.groupby(gb_cols, observed=True)[AGG_COLS].sum().reset_index()
            parts.append(agg)

    long = pd.concat(parts, ignore_index=True)
    # collapse duplicates that spanned chunks/files
    gb_cols = list(dict.fromkeys(keys + STATIC_CAT_COLS + ["ym"]))
    long = long.groupby(gb_cols, observed=True)[AGG_COLS].sum().reset_index()
    return _finalize_epd_cache(cfg, long)


def prepare_synthetic_epd_cache(cfg, n_icb=12, n_chem=40, n_months=96, seed=0):
    """Fabricate EPD-shaped monthly demand (trend + 12-month seasonality + noise +
    a price channel) so the full pipeline runs without the raw files. Calendar is
    real (a global month grid), exactly like the true data."""
    rng = np.random.default_rng(seed)
    base_ord = 2016 * 12 + 0                       # global index 0 == Jan 2016
    regions = [f"Y{60+i}" for i in range(4)]
    chapters = ["04: Central Nervous System", "02: Cardiovascular System",
                "05: Infections", "03: Respiratory System"]
    rows = []
    for i in range(n_icb):
        icb = f"ICB{i:02d}"; region = regions[i % len(regions)]
        for j in range(n_chem):
            chem = f"CHEM{j:04d}"; chapter = chapters[j % len(chapters)]
            level = rng.uniform(80, 4000)
            trend = rng.normal(0, level * 0.0015, 1)[0]
            seas_amp = rng.uniform(0.05, 0.30) * level
            phase = rng.uniform(0, 2 * np.pi)
            price = rng.uniform(1.5, 40.0)
            start = int(rng.integers(0, n_months // 4))   # staggered series births
            for t in range(start, n_months):
                ordm = base_ord + t
                moy = ordm % 12
                seas = seas_amp * np.sin(2 * np.pi * moy / 12 + phase)
                mean = max(0.0, level + trend * (t - start) + seas)
                items = max(0.0, rng.normal(mean, mean * 0.08 + 1.0))
                items = float(np.round(items))
                qty = items * rng.uniform(20, 60)
                cost = items * price * rng.uniform(0.9, 1.1)
                adq = items * rng.uniform(15, 45)
                rows.append((icb, chem, region, chapter, ordm,
                             items, qty, cost * 0.95, cost, adq))
    long = pd.DataFrame(rows, columns=cfg["series_keys"] + STATIC_CAT_COLS[:1] +
                        ["BNF_CHAPTER_PLUS_CODE", "ym"] + AGG_COLS)
    # reorder to canonical column layout
    long = long.rename(columns={cfg["series_keys"][0]: "ICB_CODE",
                                cfg["series_keys"][1]: "BNF_CHEMICAL_SUBSTANCE_CODE"})
    return _finalize_epd_cache(cfg, long)


def _finalize_epd_cache(cfg, long: pd.DataFrame):
    """Pivot the long (series, month) table onto a shared global month grid,
    encode statics, log1p+standardize the target & observed covariates on TRAIN
    months only, compute the temporal split, and write the cache + stats."""
    keys = cfg["series_keys"]
    L, H, stride = cfg["encoder_len"], cfg["horizon"], cfg["stride"]
    T = L + H

    # ---- global calendar grid -----------------------------------------------
    ym_min, ym_max = int(long["ym"].min()), int(long["ym"].max())
    n_months = ym_max - ym_min + 1
    base_year, base_moy = divmod(ym_min, 12)         # calendar of global idx 0
    long["gidx"] = long["ym"] - ym_min               # 0..n_months-1

    # temporal split boundaries (by the calendar month being forecast)
    test_start = n_months - cfg["test_months"]
    val_start = test_start - cfg["val_months"]
    assert val_start - (L + H) > 0, "Not enough history before the val split."

    # ---- one row per series with its static descriptors ---------------------
    series_meta = (long.sort_values("gidx")
                       .groupby(keys, observed=True)
                       .agg(REGIONAL_OFFICE_CODE=("REGIONAL_OFFICE_CODE", "last"),
                            BNF_CHAPTER_PLUS_CODE=("BNF_CHAPTER_PLUS_CODE", "last"),
                            first_g=("gidx", "min"), last_g=("gidx", "max"),
                            n_active=("gidx", "nunique"))
                       .reset_index())
    series_meta = series_meta[series_meta["n_active"] >= cfg["min_active_months"]]

    # rank by volume; optionally keep only the top-K busiest series
    vol = long.groupby(keys, observed=True)["ITEMS"].sum().rename("vol")
    series_meta = series_meta.merge(vol, on=keys, how="left")
    series_meta = series_meta.sort_values("vol", ascending=False)
    if cfg.get("top_k_series"):
        series_meta = series_meta.head(int(cfg["top_k_series"]))
    series_meta = series_meta.reset_index(drop=True)
    series_meta["series_id"] = np.arange(len(series_meta))
    keep = set(map(tuple, series_meta[keys].itertuples(index=False, name=None)))

    # ---- dense per-series channel arrays on the global grid -----------------
    long_k = long.set_index(keys)
    key_to_sid = {tuple(r[keys].values): int(r["series_id"])
                  for _, r in series_meta.iterrows()}

    # accumulate per (series, channel) on the full month grid
    raw_series: Dict[int, np.ndarray] = {
        sid: np.full((n_months, len(AGG_COLS)), np.nan, np.float64)
        for sid in series_meta["series_id"]}
    active_span: Dict[int, Tuple[int, int]] = {}

    gsub = long.groupby(keys, observed=True)
    for kv, g in gsub:
        kv = kv if isinstance(kv, tuple) else (kv,)
        if kv not in keep:
            continue
        sid = key_to_sid[kv]
        gi = g["gidx"].to_numpy()
        raw_series[sid][gi] = g[AGG_COLS].to_numpy(np.float64)
        active_span[sid] = (int(gi.min()), int(gi.max()))

    # ---- derive channels: target=log1p(items); per-item cost/qty/adq --------
    # train mask = months strictly before the val split (no leakage in stats)
    train_g = np.arange(n_months) < val_start

    def per_item(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        return r

    obs_train_pool = {c: [] for c in ["logdemand", "cost_pi", "qty_pi", "adq_pi"]}
    derived: Dict[int, np.ndarray] = {}
    static_scale = {}
    for sid, arr in raw_series.items():
        items = arr[:, 0]
        # demand: a NaN month inside the active span is genuine zero demand
        a0, a1 = active_span.get(sid, (0, n_months - 1))
        span = np.zeros(n_months, bool); span[a0:a1 + 1] = True
        items_filled = np.where(span & np.isnan(items), 0.0, items)
        logdemand = np.log1p(np.where(np.isnan(items_filled), 0.0, items_filled))

        cost_pi = per_item(arr[:, 3], items)         # ACTUAL_COST / ITEMS
        qty_pi = per_item(arr[:, 1], items)          # TOTAL_QUANTITY / ITEMS
        adq_pi = per_item(arr[:, 4], items)          # ADQ_USAGE / ITEMS
        # fill undefined per-item channels within span (ffill/bfill/series median)
        def fill(x):
            s = pd.Series(x).where(span)
            s = s.ffill().bfill()
            s = s.fillna(s.median() if np.isfinite(s.median()) else 0.0).fillna(0.0)
            return s.to_numpy(np.float64)
        cost_pi, qty_pi, adq_pi = fill(cost_pi), fill(qty_pi), fill(adq_pi)

        d = np.stack([logdemand, cost_pi, qty_pi, adq_pi], axis=1)  # (n_months, 4)
        derived[sid] = d

        # static scale features: series log-mean & log-std demand (train span)
        tr = logdemand[train_g & span]
        tr = tr if tr.size else logdemand[span]
        static_scale[sid] = (float(np.mean(tr)) if tr.size else 0.0,
                             float(np.std(tr)) if tr.size else 0.0)

        m = train_g & span
        if m.any():
            obs_train_pool["logdemand"].append(logdemand[m])
            obs_train_pool["cost_pi"].append(cost_pi[m])
            obs_train_pool["qty_pi"].append(qty_pi[m])
            obs_train_pool["adq_pi"].append(adq_pi[m])

    # ---- standardization stats (TRAIN months only) --------------------------
    def mean_std(chunks):
        x = np.concatenate(chunks) if chunks else np.zeros(1)
        mu, sd = float(np.mean(x)), float(np.std(x))
        return mu, (sd if sd > 1e-6 else 1.0)

    obs_names = ["logdemand", "cost_pi", "qty_pi", "adq_pi"]
    obs_mean, obs_std = {}, {}
    for nm in obs_names:
        obs_mean[nm], obs_std[nm] = mean_std(obs_train_pool[nm])
    t_mean, t_std = obs_mean["logdemand"], obs_std["logdemand"]   # target == channel 0

    # static-real (series scale) stats over train
    sc = np.array(list(static_scale.values()), float) if static_scale else np.zeros((1, 2))
    sc_mean, sc_std = sc.mean(0), np.where(sc.std(0) < 1e-6, 1.0, sc.std(0))

    # ---- standardize & write per-series observed arrays ---------------------
    os.makedirs(os.path.join(cfg["cache_dir"], "series"), exist_ok=True)
    means = np.array([obs_mean[n] for n in obs_names])
    stds = np.array([obs_std[n] for n in obs_names])
    for sid, d in derived.items():
        z = (d - means) / stds
        np.save(os.path.join(cfg["cache_dir"], "series", f"{sid}.npy"),
                z.astype(np.float32))

    # ---- encode static categoricals -----------------------------------------
    cardinalities, maps = [], {}
    for c in STATIC_CAT_COLS:
        vocab = sorted(series_meta[c].fillna("UNK").astype(str).unique().tolist())
        m = {v: i for i, v in enumerate(vocab)}
        series_meta[c + "_id"] = series_meta[c].fillna("UNK").astype(str).map(m).astype(int)
        cardinalities.append(len(vocab)); maps[c] = m

    # static reals (standardized)
    sm = series_meta["series_id"].map(lambda s: static_scale[s][0]).to_numpy(float)
    sv = series_meta["series_id"].map(lambda s: static_scale[s][1]).to_numpy(float)
    series_meta["scale_mean_z"] = (sm - sc_mean[0]) / sc_std[0]
    series_meta["scale_std_z"] = (sv - sc_mean[1]) / sc_std[1]
    series_meta["active_start"] = series_meta["series_id"].map(lambda s: active_span[s][0])
    series_meta["active_end"] = series_meta["series_id"].map(lambda s: active_span[s][1])

    stats = {
        "series_keys": keys, "target_col": cfg["target_col"], "log_transform": True,
        "t_mean": t_mean, "t_std": t_std,
        "obs_names": obs_names, "obs_mean": obs_mean, "obs_std": obs_std,
        "static_cat_cols": STATIC_CAT_COLS, "static_cat_cardinalities": cardinalities,
        "static_cat_maps": maps,
        "scale_mean": sc_mean.tolist(), "scale_std": sc_std.tolist(),
        "known_cat_cardinalities": [12],          # month-of-year
        "n_known_real": 3,                         # sin, cos, trend
        "n_static_real": 2,                        # series log-mean & log-std
        "n_observed_real": len(obs_names),
        "target_idx": 0,
        "encoder_len": L, "horizon": H, "stride": stride,
        "n_months": int(n_months), "base_year": int(base_year), "base_moy": int(base_moy),
        "val_start": int(val_start), "test_start": int(test_start),
    }
    _save_epd_cache(cfg["cache_dir"], cfg["splits_dir"], series_meta, stats)
    return stats


def _save_epd_cache(cache_dir, splits_dir, series_meta, stats):
    os.makedirs(cache_dir, exist_ok=True); os.makedirs(splits_dir, exist_ok=True)
    series_meta.to_parquet(os.path.join(cache_dir, "static.parquet"))
    with open(os.path.join(cache_dir, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    L, H, stride = stats["encoder_len"], stats["horizon"], stats["stride"]
    T = L + H
    val_start, test_start = stats["val_start"], stats["test_start"]

    def split_of(last_g):                 # by the calendar month of the LAST horizon step
        if last_g < val_start:  return "train"
        if last_g < test_start: return "val"
        return "test"

    rows = {"train": [], "val": [], "test": []}
    for _, r in series_meta.iterrows():
        sid = int(r["series_id"]); a0, a1 = int(r["active_start"]), int(r["active_end"])
        # only windows fully inside the series' active calendar span
        for t0 in range(a0, a1 - T + 2, stride):
            if t0 < 0 or t0 + T - 1 > a1:
                continue
            rows[split_of(t0 + T - 1)].append((sid, t0))
    for split, rws in rows.items():
        pd.DataFrame(rws, columns=["series_id", "t0"]).to_csv(
            os.path.join(splits_dir, f"windows_{split}.csv"), index=False)
    n = {k: len(v) for k, v in rows.items()}
    print(f"[cache] {len(series_meta)} series · {stats['n_months']} months · "
          f"windows train/val/test = {n['train']}/{n['val']}/{n['test']}")


# =============================================================================
# Dataset
# =============================================================================
class EPDForecastDataset(Dataset):
    """One (encoder, decoder, target) window per index, in the exact key layout
    ModernTFT.forward expects. Series arrays are pre-standardized in the cache and
    indexed by the GLOBAL month grid, so known-future calendar features are real."""

    def __init__(self, cache_dir, splits_dir, split, stats):
        self.cache_dir = cache_dir; self.stats = stats
        self.L = stats["encoder_len"]; self.H = stats["horizon"]; self.T = self.L + self.H
        self.target_idx = stats["target_idx"]
        self.base_moy = stats["base_moy"]; self.n_months = stats["n_months"]

        self.windows = pd.read_csv(os.path.join(splits_dir, f"windows_{split}.csv"))
        st = pd.read_parquet(os.path.join(cache_dir, "static.parquet")).set_index("series_id")
        self.cat_cols = [c + "_id" for c in stats["static_cat_cols"]]
        self.static = st
        self._series_cache: Dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.windows)

    def _series(self, sid):
        a = self._series_cache.get(sid)
        if a is None:
            a = np.load(os.path.join(self.cache_dir, "series", f"{sid}.npy"))
            self._series_cache[sid] = a
        return a

    def __getitem__(self, i):
        sid = int(self.windows.iloc[i]["series_id"]); t0 = int(self.windows.iloc[i]["t0"])
        s = self._series(sid)
        L, H, T = self.L, self.H, self.T
        enc = s[t0:t0 + L]                                # (L, n_obs)
        target = s[t0 + L:t0 + T, self.target_idx]        # (H,)  z-scored log-demand

        row = self.static.loc[sid]
        static_cat = np.array([row[c] for c in self.cat_cols], dtype=np.int64)
        static_real = np.array([row["scale_mean_z"], row["scale_std_z"]], dtype=np.float32)

        # known-future calendar features from the absolute global month index
        g = t0 + np.arange(T)                             # (T,) global month indices
        moy = (self.base_moy + g) % 12                    # month-of-year 0..11
        ang = 2 * np.pi * moy / 12.0
        trend = (g / max(1, self.n_months - 1)).astype(np.float32)
        known_real = np.stack([np.sin(ang), np.cos(ang), trend], axis=-1).astype(np.float32)

        return {
            "static_cat":    torch.from_numpy(static_cat),
            "static_real":   torch.from_numpy(static_real),
            "known_cat":     torch.from_numpy(moy.astype(np.int64))[:, None],   # (T,1)
            "known_real":    torch.from_numpy(known_real),                      # (T,3)
            "observed_real": torch.from_numpy(enc.astype(np.float32)),          # (L,n_obs)
            "target":        torch.from_numpy(target.astype(np.float32)),       # (H,)
            "series_id":     sid,
            "t0":            t0,
            "horizon_start_g": t0 + L,
            "sample_name":   f"{sid}_{t0}",
        }


def make_datasets(cfg, stats) -> Dict[str, Dataset]:
    out = {}
    for split in ("train", "val", "test"):
        wpath = os.path.join(cfg["splits_dir"], f"windows_{split}.csv")
        if os.path.exists(wpath) and len(pd.read_csv(wpath)):
            out[split] = EPDForecastDataset(cfg["cache_dir"], cfg["splits_dir"], split, stats)
    return out


def _make_loaders(datasets, batch_size, num_workers, cfg):
    """Returns (loaders, None). None kept for API parity with the BirdCLEF
    trainer (curriculum/domain-sampler hook); unused here."""
    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=(cfg["device"] != "cpu"),
            drop_last=(split == "train"), persistent_workers=(num_workers > 0),
        )
    return loaders, None


# =============================================================================
# Loss  (pinball + ramped non-crossing penalty)  -- unchanged from MIMIC
# =============================================================================
class CombinedQuantileLoss(nn.Module):
    def __init__(self, quantiles, aux_weight=0.0):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))
        self.aux_weight = float(aux_weight)
    def set_aux_weight(self, w: float):
        self.aux_weight = float(w)
    def forward(self, preds, target):
        e = target.unsqueeze(-1) - preds
        pinball = torch.maximum(self.q * e, (self.q - 1.0) * e).mean()
        cross = F.relu(preds[..., :-1] - preds[..., 1:]).mean()      # non-crossing
        return pinball + self.aux_weight * cross


# =============================================================================
# Forecast metrics (numpy). Point metrics & sharpness in ITEM units (expm1);
# coverage & pinball are computed in standardized space (monotonic-invariant).
# Adds WAPE, the standard demand/stock forecasting error.
# =============================================================================
def forecast_metrics(preds, target, quantiles, inv=None):
    q = np.asarray(quantiles)
    med_i = int(np.argmin(np.abs(q - 0.5)))
    lo_i, hi_i = int(np.argmin(q)), int(np.argmax(q))

    cov = float(np.mean((target >= preds[..., lo_i]) & (target <= preds[..., hi_i])))
    eq = target[..., None] - preds
    pinball = float(np.mean(np.maximum(q * eq, (q - 1.0) * eq)))

    if inv is None:
        inv = lambda z: z
    med_u = inv(preds[..., med_i]); tgt_u = inv(target)
    lo_u, hi_u = inv(preds[..., lo_i]), inv(preds[..., hi_i])
    e = tgt_u - med_u
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sharp = float(np.mean(hi_u - lo_u))
    wape = float(np.sum(np.abs(e)) / (np.sum(np.abs(tgt_u)) + 1e-8))
    return {"qloss": pinball, "mae": mae, "rmse": rmse,
            "coverage": cov, "sharpness": sharp, "wape": wape}


# =============================================================================
# Trainer
# =============================================================================
class EPDForecastTrainer:

    def __init__(self, config: dict):
        self.cfg = config
        torch.manual_seed(config.get("seed", 0)); np.random.seed(config.get("seed", 0))
        self._setup_dirs(); self._setup_logging()
        self.is_train = self.cfg["is_train"]

        print(f"\n{'='*70}\nNHS EPD · ModernTFT Demand Forecaster\n{'='*70}\n")

        with open(os.path.join(config["cache_dir"], "feature_stats.json")) as f:
            self.stats = json.load(f)
        self.quantiles = tuple(config["quantiles"])
        self.target_idx = self.stats["target_idx"]
        self.t_mean = self.stats["t_mean"]; self.t_std = self.stats["t_std"]
        # inverse transform: standardized log-demand -> item counts
        self.inv = lambda z: np.expm1(z * self.t_std + self.t_mean)

        print("Building datasets...")
        self.datasets = make_datasets(config, self.stats)
        self.loaders, self.domain_sampler = _make_loaders(
            self.datasets, batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4), cfg=config)
        for k, ld in self.loaders.items():
            print(f"  {k:5s}: {len(ld.dataset):>8,} windows  ({len(ld)} batches)")

        print("Building model...")
        tcfg = TFTConfig(
            static_cat_cardinalities=tuple(self.stats["static_cat_cardinalities"]),
            n_static_real=self.stats["n_static_real"],
            known_cat_cardinalities=tuple(self.stats["known_cat_cardinalities"]),
            n_known_real=self.stats["n_known_real"],
            n_observed_real=self.stats["n_observed_real"],
            d_model=config["d_model"], n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"], n_blocks=config["n_blocks"],
            dropout=config["dropout"], quantiles=self.quantiles,
            max_len=config["encoder_len"] + config["horizon"] + 8,
        )
        self.tcfg = tcfg
        self.model = ModernTFT(tcfg).to(config["device"])
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  ModernTFT: {n_params:,} params | d_model={tcfg.d_model} "
              f"d_ff={tcfg.d_ff} blocks={tcfg.n_blocks} kv_heads={tcfg.n_kv_heads} "
              f"| static_card={self.stats['static_cat_cardinalities']}")

        self.criterion = CombinedQuantileLoss(self.quantiles, aux_weight=0.0).to(config["device"])

        # optimiser: slow feature stack / fast attention+head (unchanged idiom)
        SLOW = ("emb", "lstm", "vsn_", "grn_", "enrich")
        slow, fast = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (slow if any(s in n for s in SLOW) else fast).append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": slow, "lr": config["lr"] * config["feature_lr_mult"]},
            {"params": fast, "lr": config["lr"]},
        ], weight_decay=config["weight_decay"], betas=(0.9, 0.95))

        warmup = config.get("warmup", 5); restart_epoch = config.get("restart_epoch", 50)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, eta_min=1e-6, T_max=max(1, restart_epoch - warmup)),
            ], milestones=[warmup])
        self._restart_epoch = restart_epoch
        self._restart_lr = config.get("restart_lr", 3e-5)
        self._restart_epochs_remaining = config["num_epochs"] - restart_epoch
        self._phase2_scheduler = None

        self.scaler = (torch.amp.GradScaler("cuda")
                       if config.get("use_amp", True) and config["device"] != "cpu" else None)

        self.current_epoch = 0
        self.best_val = math.inf
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 15),
            min_delta=config.get("early_stopping_min_delta", 1e-4))
        self.history = self._load_history()

        resume = config.get("resume", "")
        if resume and Path(resume).exists():
            self._load_checkpoint(resume)

    # ------------------------------------------------------------------ #
    def _setup_dirs(self):
        cfg = self.cfg
        for d in [cfg["logdir"], os.path.join(cfg["logdir"], "figures"),
                  str(Path(cfg.get("resume", os.path.join(cfg["logdir"], "best.pth"))).parent)]:
            os.makedirs(d, exist_ok=True)

    def _setup_logging(self):
        logdir = self.cfg["logdir"]
        self._train_tee = TeeFile(sys.stdout, os.path.join(logdir, "train.log"))
        self._val_tee = TeeFile(sys.stdout, os.path.join(logdir, "val.log"))

    def _load_history(self) -> dict:
        path = os.path.join(self.cfg["logdir"], "history.json")
        if Path(path).exists():
            with open(path) as f: return json.load(f)
        return {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": [],
                "val_rmse": [], "val_wape": [], "val_coverage": [], "val_sharpness": [], "lr": []}

    def _save_history(self):
        with open(os.path.join(self.cfg["logdir"], "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    @staticmethod
    def _ramp(epoch, start_epoch, end_epoch, max_value):
        if epoch < start_epoch: return 0.0
        if epoch >= end_epoch:  return max_value
        return float(max_value * (epoch - start_epoch) / max(1, end_epoch - start_epoch))

    def _save_checkpoint(self, epoch, val_loss, is_best):
        ckpt = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "epoch": epoch,
                "best_val": self.best_val, "tft_config": asdict(self.tcfg),
                "stats": self.stats,
                "config": {k: v for k, v in self.cfg.items() if k != "device"}}
        torch.save(ckpt, os.path.join(self.cfg["logdir"], "latest.pth"))
        if is_best:
            best_path = self.cfg.get("resume", os.path.join(self.cfg["logdir"], "best.pth"))
            torch.save(ckpt, best_path)
            print(f"  ✓ Best model saved (val qloss {val_loss:.4f}) → {best_path}")

    def _load_checkpoint(self, path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.cfg["device"], weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt: self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: self.scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt: self.current_epoch = ckpt["epoch"] + 1
        if "best_val" in ckpt: self.best_val = ckpt["best_val"]
        print(f"  Resumed: epoch {self.current_epoch}, best val {self.best_val:.4f}")

    def _to_device(self, batch):
        return {k: (v.to(self.cfg["device"], non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch) -> dict:
        self.current_epoch = epoch
        self.model.train()
        loader = self.loaders["train"]; total = len(loader)
        accum = self.cfg.get("accumulation", 1); clip = self.cfg.get("grad_clip", 1.0)
        metric_stride = self.cfg.get("train_metric_stride", 4)

        running_loss = 0.0; all_preds, all_targets = [], []
        self.optimizer.zero_grad(set_to_none=True)
        looper = tqdm(enumerate(loader), total=total, desc=f"Train {epoch+1}", file=self._train_tee)

        for step, batch in looper:
            batch = self._to_device(batch); target = batch["target"]

            if (step % metric_stride) == 0:
                self.model.eval()
                with torch.no_grad():
                    clean = self.model(batch)["prediction"].float().cpu().numpy()
                all_preds.append(clean); all_targets.append(target.detach().cpu().numpy())
                self.model.train()

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    preds = self.model(batch)["prediction"]; loss = self.criterion(preds, target)
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0 or step == total - 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                    self.scaler.step(self.optimizer); self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                preds = self.model(batch)["prediction"]; loss = self.criterion(preds, target)
                (loss / accum).backward()
                if (step + 1) % accum == 0 or step == total - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                    self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            looper.set_postfix({"loss": f"{running_loss/(step+1):.4f}",
                                "lr": f"{self.optimizer.param_groups[1]['lr']:.1e}"})

        if all_preds:
            P = np.concatenate(all_preds, 0); Tg = np.concatenate(all_targets, 0)
            m = forecast_metrics(P, Tg, self.quantiles, self.inv); mae = m["mae"]
        else:
            mae = float("nan")
        return {"loss": running_loss / total, "mae": mae}

    @torch.no_grad()
    def audit_feature_statistics(self):
        for split in ("train", "val"):
            if split not in self.loaders: continue
            loader = self.loaders[split]; rows = []
            for i, batch in enumerate(loader):
                if i >= 20: break
                obs = batch["observed_real"].float()
                rows.append((obs.mean().item(), obs.std().item(),
                             batch["target"].float().mean().item()))
            a = np.array(rows)
            print(f"  {split}: obs mean={a[:,0].mean():+.3f} std={a[:,1].mean():.3f} "
                  f"target(z) mean={a[:,2].mean():+.3f}")

    # ------------------------------------------------------------------ #
    def _validate(self, epoch) -> dict:
        self.model.eval()
        loader = self.loaders["val"]; running_loss = 0.0
        all_preds, all_targets = [], []
        looper = tqdm(enumerate(loader), total=len(loader), desc=f"Val   {epoch+1}", file=self._val_tee)
        with torch.no_grad():
            for step, batch in looper:
                batch = self._to_device(batch); target = batch["target"]
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        preds = self.model(batch)["prediction"]; loss = self.criterion(preds, target)
                else:
                    preds = self.model(batch)["prediction"]; loss = self.criterion(preds, target)
                running_loss += loss.item()
                all_preds.append(preds.float().cpu().numpy())
                all_targets.append(target.float().cpu().numpy())
                looper.set_postfix({"loss": f"{running_loss/(step+1):.4f}"})
        P = np.concatenate(all_preds, 0); Tg = np.concatenate(all_targets, 0)
        m = forecast_metrics(P, Tg, self.quantiles, self.inv)
        m["loss"] = running_loss / len(loader); m["preds"] = P; m["targets"] = Tg
        return m

    # ------------------------------------------------------------------ #
    def train(self):
        print(f"\n{'='*70}")
        print(f"Training  |  target={self.cfg['target_col']}  |  "
              f"L={self.cfg['encoder_len']} H={self.cfg['horizon']} months  |  "
              f"epochs {self.current_epoch}→{self.cfg['num_epochs']}")
        print(f"{'='*70}\n")

        aux_max = float(self.cfg.get("aux_weight_max", 0.1))
        aux_s = int(self.cfg.get("aux_ramp_start_epoch", 5))
        aux_e = int(self.cfg.get("aux_ramp_end_epoch", 20))
        self.audit_feature_statistics()

        for epoch in range(self.current_epoch, self.cfg["num_epochs"]):
            t0 = time.time()
            aux_w = self._ramp(epoch, aux_s, aux_e, aux_max)
            self.criterion.set_aux_weight(aux_w)
            print(f"  [loss] non-crossing aux_w={aux_w:.3f}")

            train_m = self._train_epoch(epoch)
            val_m = self._validate(epoch)

            if epoch + 1 == self._restart_epoch and self._restart_epochs_remaining > 0:
                print(f"\n  >>> WARM RESTART: LR → {self._restart_lr:.1e}, "
                      f"{self._restart_epochs_remaining} epochs remaining")
                top = max(g["lr"] for g in self.optimizer.param_groups)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self._restart_lr * max(pg["lr"] / top, 0.1)
                self._phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, eta_min=1e-6, T_max=self._restart_epochs_remaining)

            if self._phase2_scheduler is not None and epoch >= self._restart_epoch:
                self._phase2_scheduler.step()
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[1]["lr"]
            print(
                f"\nEpoch {epoch+1:3d}/{self.cfg['num_epochs']}  ({time.time()-t0:.0f}s)\n"
                f"  Train | loss {train_m['loss']:.4f}  MAE {train_m['mae']:.1f} items\n"
                f"  Val   | loss {val_m['loss']:.4f}  MAE {val_m['mae']:.1f} items  "
                f"RMSE {val_m['rmse']:.1f}  WAPE {val_m['wape']:.3f}  "
                f"cover[{self.quantiles[0]:.0%}-{self.quantiles[-1]:.0%}] {val_m['coverage']:.3f}  "
                f"sharp {val_m['sharpness']:.1f}  lr {lr:.1e}")

            self.history["train_loss"].append(float(train_m["loss"]))
            self.history["train_mae"].append(float(train_m["mae"]))
            self.history["val_loss"].append(float(val_m["loss"]))
            self.history["val_mae"].append(float(val_m["mae"]))
            self.history["val_rmse"].append(float(val_m["rmse"]))
            self.history["val_wape"].append(float(val_m["wape"]))
            self.history["val_coverage"].append(float(val_m["coverage"]))
            self.history["val_sharpness"].append(float(val_m["sharpness"]))
            self.history["lr"].append(float(lr))
            self._save_history()

            is_best = val_m["loss"] < self.best_val
            if is_best: self.best_val = val_m["loss"]
            self._save_checkpoint(epoch, val_m["loss"], is_best)

            if (epoch + 1) % 5 == 0:
                try:
                    self._plot_metrics(epoch + 1)
                    self._plot_forecast_examples(val_m["preds"], val_m["targets"], epoch + 1)
                    self._plot_calibration(val_m["preds"], val_m["targets"], epoch + 1)
                except Exception as e:
                    print(f"  [plot] skipped ({e})")

            if self.early_stopping.step(val_m["loss"]):
                print(f"  Early stopping at epoch {epoch+1} (no improvement)."); break

        print(f"\n{'='*70}\nTraining complete.  Best val qloss: {self.best_val:.4f}\n{'='*70}\n")

    # ------------------------------------------------------------------ #
    def predict_test(self, model_path: Optional[str] = None) -> pd.DataFrame:
        if "test" not in self.loaders:
            raise RuntimeError("No test split found in cache.")
        if model_path is None:
            model_path = self.cfg.get("resume", os.path.join(self.cfg["logdir"], "best.pth"))
        if Path(model_path).exists():
            print(f"\nLoading best model: {model_path}")
            ckpt = torch.load(model_path, map_location=self.cfg["device"], weights_only=False)
            self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

        base_year, base_moy = self.stats["base_year"], self.stats["base_moy"]
        base_ord = base_year * 12 + base_moy
        qcols = [f"p{int(q*100)}" for q in self.quantiles]
        rows = []
        with torch.no_grad():
            for batch in tqdm(self.loaders["test"], desc="Inference"):
                sid = batch["series_id"]; hstart = batch["horizon_start_g"]
                target = batch["target"].numpy()
                batch = self._to_device(batch)
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        preds = self.model(batch)["prediction"]
                else:
                    preds = self.model(batch)["prediction"]
                preds = preds.float().cpu().numpy()                 # (B,H,Q) z-space
                preds_u = self.inv(preds)                           # -> item counts
                tgt_u = self.inv(target)
                B, H, Q = preds_u.shape
                for b in range(B):
                    for h in range(H):
                        ym = ord_to_label(base_ord + int(hstart[b]) + h)
                        rows.append([f"{int(sid[b])}_{ym}", int(sid[b]), ym, h + 1,
                                     *preds_u[b, h].tolist(), float(tgt_u[b, h])])

        sub = pd.DataFrame(rows, columns=["row_id", "series_id", "year_month", "step",
                                          *qcols, "target"])
        out_csv = self.cfg.get("submission_csv", "forecasts.csv")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"Forecasts saved → {out_csv}  ({len(sub):,} rows)")
        return sub

    # ------------------------------------------------------------------ #
    # Plots (item-unit scaling via expm1)
    # ------------------------------------------------------------------ #
    def _plot_metrics(self, epoch):
        if not HAS_MPL: return
        save_dir = os.path.join(self.cfg["logdir"], "figures", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        h = self.history; eps = list(range(1, len(h["train_loss"]) + 1))
        panels = [("loss", "train_loss", "val_loss", "Quantile loss", False),
                  ("mae", "train_mae", "val_mae", "MAE (items)", False),
                  ("wape", None, "val_wape", "WAPE", False),
                  ("coverage", None, "val_coverage", "Interval coverage", False),
                  ("sharpness", None, "val_sharpness", "Sharpness (items)", False),
                  ("lr", None, "lr", "Learning rate", True)]
        for fname, tkey, vkey, title, log_y in panels:
            fig, ax = plt.subplots(figsize=(9, 5))
            if tkey and tkey in h: ax.plot(eps, h[tkey], label="Train", linewidth=2)
            if vkey and vkey in h: ax.plot(eps, h[vkey], label="Val", linewidth=2)
            if fname == "coverage":
                ax.axhline(self.quantiles[-1] - self.quantiles[0], color="red",
                           ls="--", alpha=.7, label="nominal")
            ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(title, fontweight="bold")
            if log_y: ax.set_yscale("log")
            ax.legend(); ax.grid(True, ls="--", alpha=.5); fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=130); plt.close(fig)

    def _plot_forecast_examples(self, preds, targets, epoch, n=6):
        if not HAS_MPL: return
        save_dir = os.path.join(self.cfg["logdir"], "figures", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        q = np.asarray(self.quantiles)
        med_i = int(np.argmin(np.abs(q - 0.5))); lo_i, hi_i = int(np.argmin(q)), int(np.argmax(q))
        idx = np.linspace(0, len(preds) - 1, min(n, len(preds))).astype(int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, i in zip(axes.ravel(), idx):
            hsteps = np.arange(preds.shape[1])
            ax.fill_between(hsteps, self.inv(preds[i, :, lo_i]), self.inv(preds[i, :, hi_i]),
                            color="#5fd0c8", alpha=.25, label="p10–p90")
            ax.plot(hsteps, self.inv(preds[i, :, med_i]), color="#f0b357", lw=2, label="median")
            ax.plot(hsteps, self.inv(targets[i]), color="#222", lw=1.6, ls="--",
                    marker="o", ms=3, label="actual")
            ax.set_xlabel("horizon (months)"); ax.set_ylabel("items")
            ax.grid(True, ls="--", alpha=.4)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f"Forecast examples (epoch {epoch})", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(save_dir, "forecast_examples.png"), dpi=130); plt.close(fig)

    def _plot_calibration(self, preds, targets, epoch):
        if not HAS_MPL: return
        save_dir = os.path.join(self.cfg["logdir"], "figures", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        q = np.asarray(self.quantiles)
        emp = [(targets <= preds[..., k]).mean() for k in range(len(q))]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], color="red", ls="--", alpha=.7, label="ideal")
        ax.plot(q, emp, marker="o", color="#5fd0c8", lw=2, label="empirical")
        ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical coverage")
        ax.set_title(f"Quantile calibration (epoch {epoch})", fontweight="bold")
        ax.legend(); ax.grid(True, ls="--", alpha=.5); fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "calibration.png"), dpi=130); plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================
build_cache = os.environ.get("BUILD_CACHE", "0") == "1"
train_mode = True
test_mode = True

if __name__ == "__main__":
    if build_cache:
        if _list_epd_files(CONFIG["epd_raw"]):
            build_epd_cache(CONFIG)
        else:
            print("[cache] raw EPD not found — generating synthetic cache")
            prepare_synthetic_epd_cache(CONFIG)

    if not Path(CONFIG["cache_dir"], "feature_stats.json").exists():
        print("[cache] no cache found — generating synthetic cache")
        prepare_synthetic_epd_cache(CONFIG)

    if train_mode:
        trainer = EPDForecastTrainer(CONFIG)
        trainer.train()

    if test_mode:
        CONFIG["num_workers"] = 0
        trainer = EPDForecastTrainer({**CONFIG, "is_train": False})
        trainer.predict_test()
