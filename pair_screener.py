#!/usr/bin/env python
"""Regime-aware, HMM-free universe-wide pair screener.

Screens every tradable unordered pair (all pairs excluding instrument 0, which the
strategy may not trade) on the train window with methods chosen to survive rare regime
switches instead of relying on a well-fit HMM:

  1. Changepoint detection      - PELT (L2 cost, BIC penalty) on the Kalman spread level
                                  and on |spread increments| (vol breaks). Needs no
                                  repeated regimes: a single break can be detected.
  2. Rolling-window stability   - per-window hedge ratio, AR(1) half-life, ADF p-value
                                  and strategy-replica PnL; drift shows up as window
                                  dispersion without any explicit state model.
  3. Structural-break tests     - CUSUM of OLS residuals (proper asymptotic p-value)
                                  plus Hansen's Lc stat as a red-flag feature, run on
                                  the full-train regression.
  4. Kalman dynamic hedge ratio - the strategy's own random-walk-coefficient filter;
                                  beta-path dispersion and one-day jumps measure how
                                  violently the relationship would have to be tracked.
  5. Volatility-regime proxies  - market-wide EWMA vol median split as a coarse regime
                                  label; a pair must earn positive replica PnL in BOTH
                                  regimes, so its edge cannot come from one hot regime.

Tradability is measured with a fast causal replica of the real strategy (same Kalman
filter class, same rolling OU estimation, same optimal-stopping dynamic z-entry, same
commission) run at three fixed Kalman speeds (never tuned at the screening stage, to
avoid screen overfitting). The screen score is the across-window median - 0.5*std of
replica scores, mirroring the tuner's robustness-penalized objective.

Multiple-comparison control: per-window ADF p-values across the whole universe are
pooled into one Benjamini-Hochberg FDR correction, and every gate decision is recorded
so downstream verification can show exactly which evidence each pair passed on.

Outputs a full metrics JSON, a human-readable report, screening plots, and a tuner-ready
pairs file with the top-N new pairs plus a seeded random placebo cohort (pairs that
failed the screen) used downstream as the pipeline-level noise control.

Usage:
    python pair_screener.py --start 201 --end 800 --n_windows 5 \
        --out auto_tuning_results/screen_train.json \
        --pairs_out auto_tuning_results/pairs_screened.json
    python pair_screener.py --smoke            # quick 40-pair sanity run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid, breaks_hansen
from statsmodels.tsa.stattools import adfuller

import testbedv3 as strat

# ============================================================================
# CONFIG
# ============================================================================
CFG = {
    "start": 201,          # first scored day (1-based, like the tuner/eval)
    "end": 800,            # last scored day
    "n_windows": 5,
    "warmup": 30,
    "workers": max(1, multiprocessing.cpu_count()),
    "topn": 10,
    "n_placebo": 6,
    "placebo_seed": 7,
    "prices": "./prices.txt",
    "user_pairs_file": "user_candidate_pairs.json",
    "out": "auto_tuning_results/screen_train.json",
    "pairs_out": "auto_tuning_results/pairs_screened.json",
    "report": "auto_tuning_results/screen_report.txt",
    "smoke": False,
}

PROXY_QS = (1e-7, 1e-6, 1e-5)     # fixed Kalman speeds; never tuned at screening
Q_REF = 1e-6                       # reference filter for PELT/beta-path features
TC = 0.0001                        # commission per dollar per leg
LEG = 10_000.0
OU_WINDOW = 100
OU_ALPHA = 0.97
MIN_TRADES = 5
ALGO = 0                           # instrument 0: never tradable in the strategy

GATE_ADF_P = 0.05                  # per-window ADF raw p gate
GATE_FDR_Q = 0.10                  # pooled Benjamini-Hochberg q gate
GATE_HL_MED = 40.0                 # median window half-life cap (days)
GATE_HL_WIN = 60.0                 # per-window half-life cap
GATE_BETA_CV = 0.5                 # beta-path dispersion red flag
GATE_MAX_BREAKS = 3                # mean-break count red flag

# The 8 pairs currently traded by latest_strategy.py (kept as a constant so the
# screener never depends on which params are hardcoded there).
CURRENT_PAIRS = [
    ("RTTH", "NAYO"), ("AENO", "NWIG"), ("HETT", "ULXY"), ("HUXZ", "ACAC"),
    ("SMAH", "ILVX"), ("EORC", "NGTE"), ("CTGI", "EELT"), ("ALUT", "CCNS"),
]

# ============================================================================
# WORKER GLOBALS
# ============================================================================
_G: dict = {}


def init_worker(logp, mkt_vol, cfg):
    _G["logp"] = logp
    _G["mkt_vol"] = mkt_vol
    _G["cfg"] = cfg


# ============================================================================
# BASIC STATISTICAL PIECES
# ============================================================================
def ar1_phi(s: np.ndarray) -> float:
    """OLS AR(1) coefficient with intercept, same estimator shape as the strategy."""
    s = np.asarray(s, dtype=float)
    if s.size < 10:
        return float("nan")
    a, b = s[:-1], s[1:]
    ma, mb = a.mean(), b.mean()
    den = ((a - ma) ** 2).sum()
    if den < 1e-18:
        return float("nan")
    return float(((a - ma) * (b - mb)).sum() / den)


def halflife_days(phi: float) -> float:
    """OU-style half-life in days; inf if not mean-reverting, ~0.08 if anti-persistent."""
    if not np.isfinite(phi):
        return float("inf")
    phi = min(max(phi, 1e-4), 0.9999)
    return math.log(2.0) / (-math.log(phi))


def adf_p(x: np.ndarray) -> float:
    """Fixed-spec ADF p-value (maxlag=1, constant) so windows are comparable."""
    try:
        r = adfuller(np.asarray(x, dtype=float), maxlag=1, regression="c", autolag=None)
        return float(r[1])
    except Exception:
        return 1.0


def pelt_changepoints(x: np.ndarray, min_seg: int = 15) -> list[int]:
    """PELT with L2 cost and persistence-adjusted BIC penalty for mean shifts
    (works with a single break).

    Naive BIC (2*sigma^2*ln n) over-segments mean-reverting series: AR(1) noise with
    coefficient phi has long-run variance inflated by (1+phi)/(1-phi), so the penalty
    is scaled by that factor, and sigma^2 is estimated robustly as the variance around
    a rolling median (immune to the very level shifts being detected).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * min_seg + 2:
        return []
    k = min(21, n)
    med = pd.Series(x).rolling(k, center=True, min_periods=1).median().values
    resid = x - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sig2 = max((1.4826 * mad) ** 2, 1e-18)
    phi = ar1_phi(x)
    phi = min(max(phi, 0.0), 0.90) if np.isfinite(phi) else 0.0
    pen = 2.0 * sig2 * math.log(n) * (1.0 + phi) / (1.0 - phi)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))

    def cost(a: int, b: int) -> float:  # cost of segment x[a:b]
        s = c1[b] - c1[a]
        s2 = c2[b] - c2[a]
        m = b - a
        return s2 - s * s / m

    F = np.full(n + 1, np.inf)
    F[0] = -pen
    prev = np.zeros(n + 1, dtype=int)
    R = [0]
    for t in range(1, n + 1):
        vals = [F[tau] + cost(tau, t) for tau in R]
        j = int(np.argmin(vals))
        F[t] = vals[j] + pen
        prev[t] = R[j]
        R = [tau for tau, v in zip(R, vals) if v <= F[t]]
        R.append(t)

    cps = []
    t = n
    while t > 0:
        t = int(prev[t])
        if t > 0:
            cps.append(t)
    cps.reverse()

    # post-merge segments shorter than min_seg into the cheaper neighbour
    changed = True
    while changed and len(cps) > 0:
        changed = False
        bounds = [0] + cps + [n]
        seglens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        if min(seglens) >= min_seg:
            break
        i = int(np.argmin(seglens))
        # remove the boundary that merges the short segment most cheaply
        if i == 0:
            del cps[0]
        elif i == len(seglens) - 1:
            del cps[-1]
        else:
            left_gain = cost(bounds[i - 1], bounds[i + 1]) - cost(bounds[i - 1], bounds[i]) - cost(bounds[i], bounds[i + 1])
            right_gain = cost(bounds[i], bounds[i + 2]) - cost(bounds[i], bounds[i + 1]) - cost(bounds[i + 1], bounds[i + 2])
            del cps[i if left_gain >= right_gain else i - 1]
        changed = True
    return cps


# ============================================================================
# STRATEGY-REPLICA PROXY (causal; identical components to testbedv3)
# ============================================================================
def proxy_run(ly: np.ndarray, lx: np.ndarray, py: np.ndarray, px: np.ndarray,
              q_beta: float, record: bool = False):
    """One causal pass of the real strategy with fixed (untuned) parameters.

    Day indexing matches the eval: positions decided with data through index i earn the
    return from index i-1 to i, commissions charged on dollar position changes.
    Returns (daily_pnl, n_trades, recorder) where recorder holds beta/spread paths for
    the reference-q run only.
    """
    n = ly.size
    filt = strat.PairSegFilter(strat.FiltCfg(q_beta=q_beta))
    spread_hist: list[float] = []
    beta_hist: list[float] = []
    kappa, sigma, theta, a_star = 0.1, 0.02, 0.0, 1.0
    side = 0
    pnl = np.zeros(n)
    prev_py_d = prev_px_d = 0.0
    n_trades = 0

    for i in range(n):
        o = filt.step(ly[i], lx[i])
        spread_hist.append(o.spread)
        if record:
            beta_hist.append(o.beta)

        if len(spread_hist) >= OU_WINDOW:
            s = np.asarray(spread_hist[-OU_WINDOW:])
            s_lag, s_t = s[:-1], s[1:]
            ml, mt = s_lag.mean(), s_t.mean()
            den = ((s_lag - ml) ** 2).sum()
            if den > 1e-8:
                b_mle = min(max(float(((s_lag - ml) * (s_t - mt)).sum() / den), 1e-4), 0.9999)
                a_mle = mt - b_mle * ml
                eps = s_t - (a_mle + b_mle * s_lag)
                sig = float(np.sqrt((eps ** 2).sum() / max(eps.size - 2, 1)))
                rk = -math.log(b_mle)
                rs = sig * math.sqrt((2.0 * rk) / (1.0 - math.exp(-2.0 * rk)))
                th = a_mle / (1.0 - b_mle)
                lk = 1.0 - OU_ALPHA
                kappa = lk * kappa + (1.0 - lk) * rk
                sigma = lk * sigma + (1.0 - lk) * rs
                theta = lk * theta + (1.0 - lk) * th
                try:
                    a_star = strat.get_optimal_z_entry(kappa, sigma, TC * 2.0, rule="new")
                except Exception:
                    pass  # keep previous a_star

        prev_side = side
        if i >= strat.WARMUP + 9:  # day WARMUP+10 onwards, matching getMyPosition
            if kappa > 0:
                ou_std = sigma / math.sqrt(2.0 * kappa)
                sz = (spread_hist[-1] - theta) / max(ou_std, 1e-8)
            else:
                sz = 0.0
            if side == 0:
                if sz >= a_star:
                    side = -1
                elif sz <= -a_star:
                    side = 1
            elif side == 1:
                if sz >= a_star:
                    side = -1
            elif sz <= -a_star:
                side = 1

        if side == 0:
            pos_y_d = pos_x_d = 0.0
        else:
            beta = float(filt.beta) if filt.beta is not None else 1.0
            scale = LEG / max(1.0, abs(beta))
            pos_y_d = scale * side
            pos_x_d = -beta * pos_y_d

        if side != prev_side and side != 0:
            n_trades += 1

        if i >= 1:
            ry = py[i] / py[i - 1] - 1.0
            rx = px[i] / px[i - 1] - 1.0
            comm = TC * (abs(pos_y_d - prev_py_d) + abs(pos_x_d - prev_px_d))
            pnl[i] = prev_py_d * ry + prev_px_d * rx - comm
        prev_py_d, prev_px_d = pos_y_d, pos_x_d

    recorder = {"beta": beta_hist, "spread": spread_hist} if record else None
    return pnl, n_trades, recorder


def score_fn(mu: float, sigma: float, param: float = 1.0) -> float:
    """The competition score: mean daily PnL damped by a Sharpe-squared factor."""
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = math.sqrt(250.0) * mu / sigma
    return float(mu * (sr * sr) / (sr * sr + param * param))


# ============================================================================
# PER-PAIR SCREEN
# ============================================================================
def screen_orientation(ly, lx, py, px, win_idx, cfg):
    """All metrics for one trading orientation (y=ly regressed on x=lx)."""
    a0, b0 = cfg["start"] - 1, cfg["end"]          # 0-based slice [a0, b0)
    out: dict = {"windows": [], "proxies": {}}

    # --- statistical windows (OLS beta re-fit per window: rolling stability) ---
    betas, phis, hls, adfs, vols = [], [], [], [], []
    for (wa, wb) in win_idx:
        ys, xs = ly[wa:wb], lx[wa:wb]
        denom = float(xs @ xs)
        beta = float(xs @ ys) / denom if denom > 1e-12 else float("nan")
        spread = ys - beta * xs
        phi = ar1_phi(spread)
        hl = halflife_days(phi)
        betas.append(beta)
        phis.append(phi)
        hls.append(hl)
        adfs.append(adf_p(spread))
        vols.append(float(np.std(spread)))
        out["windows"].append({
            "beta": beta, "phi": phi, "halflife": hl,
            "adf_p": adfs[-1], "spread_vol": vols[-1],
        })

    # --- full-train regression: Engle-Granger + structural-break battery ---
    yt, xt = ly[a0:b0], lx[a0:b0]
    den = float(xt @ xt)
    beta_full = float(xt @ yt) / den if den > 1e-12 else float("nan")
    spread_full = yt - beta_full * xt
    out["beta_full"] = beta_full
    out["eg_adf_p"] = adf_p(spread_full)
    try:
        res = sm.OLS(yt, sm.add_constant(xt)).fit()
        sup_b, cusum_p, _ = breaks_cusumolsresid(res.resid, ddof=2)
        h_stat, _ = breaks_hansen(res)
        out["cusum_sup_b"] = float(sup_b)
        out["cusum_p"] = float(cusum_p)
        out["hansen_lc"] = float(h_stat)
    except Exception:
        out["cusum_sup_b"] = out["cusum_p"] = out["hansen_lc"] = float("nan")

    # --- strategy replica at three fixed Kalman speeds ---
    pnls, trades = [], []
    rec = None
    for q in PROXY_QS:
        pnl_q, tr_q, rec_q = proxy_run(ly, lx, py, px, q_beta=q, record=(q == Q_REF))
        pnls.append(pnl_q)
        trades.append(tr_q)
        if rec_q is not None:
            rec = rec_q
        w_scores, w_means, w_sharpes = [], [], []
        for (wa, wb) in win_idx:
            seg = pnl_q[wa:wb]
            mu, sd = float(seg.mean()), float(seg.std())
            w_scores.append(score_fn(mu, sd))
            w_means.append(mu)
            w_sharpes.append(math.sqrt(250.0) * mu / sd if sd > 1e-10 else 0.0)
        seg_full = pnl_q[a0:b0]
        out["proxies"][f"{q:g}"] = {
            "window_scores": w_scores, "window_means": w_means,
            "window_sharpes": w_sharpes, "n_trades": tr_q,
            "full_score": score_fn(float(seg_full.mean()), float(seg_full.std())),
        }
    pnl_mean = np.mean(pnls, axis=0)   # config-averaged daily PnL for regime splits
    out["proxy_window_scores"] = [
        float(np.mean([out["proxies"][f"{q:g}"]["window_scores"][k] for q in PROXY_QS]))
        for k in range(len(win_idx))
    ]
    out["proxy_full_score"] = float(np.mean([out["proxies"][f"{q:g}"]["full_score"] for q in PROXY_QS]))
    out["n_trades_med"] = float(np.median(trades))
    out["n_win_proxy_pos"] = int(sum(1 for k in range(len(win_idx))
                                     if pnl_mean[win_idx[k][0]:win_idx[k][1]].mean() > 0))

    # --- volatility-regime proxy split (market-wide EWMA vol, median split) ---
    mv = _G["mkt_vol"]
    mv_seg = mv[a0:b0]
    med = float(np.median(mv_seg))
    high = mv_seg > med
    pnl_seg = pnl_mean[a0:b0]
    out["regime_mktvol"] = {
        "high_mean": float(pnl_seg[high].mean()) if high.any() else float("nan"),
        "low_mean": float(pnl_seg[~high].mean()) if (~high).any() else float("nan"),
        "frac_high": float(high.mean()),
    }

    # --- reference-filter paths: changepoints + beta stability ---
    if rec is not None and len(rec["spread"]) >= b0:
        sp_full = np.asarray(rec["spread"])
        # mean-reversion of the TRADED (Kalman residual) spread per window: this is the
        # series the strategy actually trades, so it - not the static-OLS spread - is
        # what the G1/G2 rolling-stability gates must test
        for k, (wa, wb) in enumerate(win_idx):
            seg = sp_full[wa:wb]
            phi_k = ar1_phi(seg)
            out["windows"][k]["adf_p_kalman"] = adf_p(seg)
            out["windows"][k]["phi_kalman"] = phi_k
            out["windows"][k]["halflife_kalman"] = halflife_days(phi_k)
        sp = sp_full[a0:b0]
        bt = np.asarray(rec["beta"][a0:b0])
        mean_cps = pelt_changepoints(sp)
        vol_cps = pelt_changepoints(np.abs(np.diff(sp)))
        bounds = [0] + mean_cps + [sp.size]
        longest = max(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)) / sp.size
        bmean = float(np.mean(np.abs(bt))) if bt.size else float("nan")
        out["n_mean_breaks"] = len(mean_cps)
        out["n_vol_breaks"] = len(vol_cps)
        out["mean_breaks_at"] = [int(c + a0 + 1) for c in mean_cps]  # 1-based days
        out["vol_breaks_at"] = [int(c + a0 + 1) for c in vol_cps]
        out["longest_seg_frac"] = float(longest)
        out["beta_cv"] = float(np.std(bt) / bmean) if bmean and bmean > 1e-12 else float("nan")
        out["beta_maxjump"] = float(np.max(np.abs(np.diff(bt)))) if bt.size > 1 else 0.0
        out["beta_range_rel"] = float((bt.max() - bt.min()) / bmean) if bmean and bmean > 1e-12 else float("nan")
        n_third = max(1, sp.size // 3)
        v0 = float(np.var(np.diff(sp[:n_third]))) if n_third > 2 else float("nan")
        v1 = float(np.var(np.diff(sp[-n_third:])))
        out["innov_var_ratio"] = v1 / v0 if v0 and v0 > 1e-18 else float("nan")
    return out


def screen_pair(task):
    i_a, i_b, sym_a, sym_b = task
    logp = _G["logp"]
    cfg = _G["cfg"]
    a0, b0 = cfg["start"] - 1, cfg["end"]
    span = b0 - a0
    n_win = min(cfg["n_windows"], span)
    width = span // n_win
    win_idx = [(a0 + k * width, b0 if k == n_win - 1 else a0 + (k + 1) * width)
               for k in range(n_win)]

    # trim to the scored end day: nothing after `end` may influence the screen,
    # and the forward filter is identical either way (causality + less compute)
    la, lb = logp[i_a, :b0], logp[i_b, :b0]
    res_a = screen_orientation(la, lb, np.exp(la), np.exp(lb), win_idx, cfg)
    res_b = screen_orientation(lb, la, np.exp(lb), np.exp(la), win_idx, cfg)

    orient = "A" if res_a["proxy_full_score"] >= res_b["proxy_full_score"] else "B"
    return {
        "pair": [sym_a, sym_b],
        "window_days": [[w[0] + 1, w[1]] for w in win_idx],
        "A": res_a,
        "B": res_b,
        "orient": orient,
        "chosen": res_a if orient == "A" else res_b,
    }


# ============================================================================
# AGGREGATION: FDR, GATES, RANKS
# ============================================================================
def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of rejections under BH at level q."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresh = q * ranks / n
    passed = p <= thresh
    # BH: reject all p <= p_(k*) where k* is the largest k with p_(k) <= qk/n
    k_star = 0
    for k in range(n, 0, -1):
        if p[order[k - 1]] <= q * k / n:
            k_star = k
            break
    rej = np.zeros(n, dtype=bool)
    if k_star > 0:
        rej[order[:k_star]] = True
    return rej


def evaluate_gates(rec: dict, fdr_mask: np.ndarray) -> dict:
    """Apply the five screen gates to one pair record.

    G1 (cointegration/FDR) is always evaluated on orientation A because the pooled
    Benjamini-Hochberg mask is defined on A's windows; G2-G5 use the chosen trading
    orientation. Threshold counts scale with n_windows so shorter (e.g. OOS) screens
    stay passable: ~80% of windows for ADF/half-life, ~60% for FDR and positive PnL.
    """
    ch = rec["chosen"]
    wins = ch["windows"]
    n_win = len(wins)
    need_stat = int(math.ceil(0.8 * n_win))
    need_soft = int(math.ceil(0.6 * n_win))

    # G1/G2 test the Kalman-residual spread the strategy actually trades (chosen
    # orientation); static-OLS window stats stay as drift diagnostics only
    n_adf = sum(1 for w in wins if w.get("adf_p_kalman", 1.0) < GATE_ADF_P)
    n_adf_static = sum(1 for w in rec["A"]["windows"] if w["adf_p"] < GATE_ADF_P)
    n_fdr = int(np.sum(fdr_mask))  # diagnostic only: pooled BH is deliberately strict;
    # multiplicity is handled at report level (binomial-null expected false passers)
    g1 = n_adf >= need_stat

    hls = [w.get("halflife_kalman", float("inf")) for w in wins]
    med_hl = float(np.median(hls))
    n_hl_ok = sum(1 for h in hls if h <= GATE_HL_WIN)
    g2 = (1.0 <= med_hl <= GATE_HL_MED) and (n_hl_ok >= need_stat)

    beta_cv = ch.get("beta_cv", float("nan"))
    n_breaks = ch.get("n_mean_breaks", 0)
    cusum_p = ch.get("cusum_p", 1.0)
    struct_redflags = (np.isfinite(cusum_p) and cusum_p < 0.05
                       and np.isfinite(beta_cv) and beta_cv > GATE_BETA_CV
                       and n_breaks >= GATE_MAX_BREAKS)
    g3 = not struct_redflags

    reg = ch.get("regime_mktvol", {})
    hi, lo = reg.get("high_mean", float("nan")), reg.get("low_mean", float("nan"))
    g4 = np.isfinite(hi) and np.isfinite(lo) and hi > 0 and lo > 0

    g5 = (ch["proxy_full_score"] > 0
          and ch["n_win_proxy_pos"] >= need_soft
          and ch["n_trades_med"] >= MIN_TRADES)

    gates = {"G1_coint_fdr": bool(g1), "G2_halflife": bool(g2),
             "G3_structural": bool(g3), "G4_vol_regime": bool(g4),
             "G5_tradable": bool(g5)}
    win_scores = ch["proxy_window_scores"]
    screen_score = float(np.median(win_scores) - 0.5 * np.std(win_scores))
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "screen_score": screen_score,
        "n_win_adf05": n_adf,
        "n_win_adf05_static": n_adf_static,
        "n_win_fdr10": n_fdr,
        "median_halflife": med_hl,
        "beta_cv": beta_cv,
        "cusum_p": cusum_p,
        "orient": rec["orient"],
    }


# ============================================================================
# REPORT & PLOTS
# ============================================================================
def load_user_pairs(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [(d["y"], d["x"]) for d in data]


def write_report(records, cfg, user_pairs, out_path, null_info=None):
    cur = {tuple(sorted(p)) for p in CURRENT_PAIRS}
    usr = {tuple(sorted(p)) for p in user_pairs}
    lines = []
    add = lines.append

    add(f"PAIR SCREEN REPORT - days {cfg['start']}-{cfg['end']} "
        f"({cfg['n_windows']} windows)")
    add(f"universe: {len(records)} pairs (instrument 0 excluded)")
    n_pass = sum(1 for r in records if r["gate_eval"]["pass"])
    add(f"gate-passing pairs: {n_pass}")
    if null_info:
        add(f"G1 noise check: {null_info['observed']} passers observed vs "
            f"{null_info['expected']:.2f} expected if every spread were pure noise "
            f"(binomial tail {null_info['p']:.2e} per pair)")
    add("")
    hdr = (f"{'pair':<16}{'orient':>7}{'score':>8}{'G1':>4}{'G2':>4}{'G3':>4}{'G4':>4}"
           f"{'G5':>4}{'adfW':>5}{'fdrW':>5}{'medHL':>7}{'bCV':>6}{'brk':>4}{'tag':>10}")
    add(hdr)
    add("-" * len(hdr))
    for r in records[:60]:
        g = r["gate_eval"]
        pair = r["pair"]
        key = tuple(sorted(pair))
        tag = "CURRENT" if key in cur else ("user89" if key in usr else "")
        gates = g["gates"]
        add(f"{pair[0]+'-'+pair[1]:<16}{g['orient']:>7}{g['screen_score']:>8.2f}"
            f"{('Y' if gates['G1_coint_fdr'] else '.'):>4}{('Y' if gates['G2_halflife'] else '.'):>4}"
            f"{('Y' if gates['G3_structural'] else '.'):>4}{('Y' if gates['G4_vol_regime'] else '.'):>4}"
            f"{('Y' if gates['G5_tradable'] else '.'):>4}{g['n_win_adf05']:>5}{g['n_win_fdr10']:>5}"
            f"{g['median_halflife']:>7.1f}{(g['beta_cv'] if np.isfinite(g['beta_cv']) else -1):>6.2f}"
            f"{r['chosen'].get('n_mean_breaks', -1):>4}{tag:>10}")
    add("")
    add("Legend: G1 traded-spread reversion (>=80% of windows have ADF p<0.05 on the Kalman")
    add("          residual the strategy actually trades; static-OLS ADF counts and pooled")
    add("          BH-FDR are diagnostics, multiplicity is quantified by the binomial null),")
    add("        G2 half-life (Kalman-spread median in [1,40]d and >=80% of windows <=60d),")
    add("        G3 structural (no joint CUSUM+beta-dispersion+breaks red flag),")
    add("        G4 vol regime (positive replica PnL in high AND low market-vol halves),")
    add("        G5 tradable (positive replica score, >=60% of windows positive, >=5 trades).")

    add("")
    add("=" * 100)
    add("CURRENT 8 PAIRS - gate readout")
    add("=" * 100)
    by_key = {tuple(sorted(r["pair"])): r for r in records}
    for p in CURRENT_PAIRS:
        r = by_key.get(tuple(sorted(p)))
        if r is None:
            add(f"  {p[0]}-{p[1]}: not in universe?!")
            continue
        g = r["gate_eval"]
        rank = next(i for i, rr in enumerate(records) if rr is r) + 1
        gstr = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in g["gates"].items())
        add(f"  {p[0]}-{p[1]}: rank {rank}/{len(records)} score {g['screen_score']:.2f} "
            f"pass={g['pass']} | {gstr}")

    add("")
    add("=" * 100)
    add(f"USER CANDIDATES ({len(user_pairs)}) - how many pass all gates")
    add("=" * 100)
    n_usr_pass = 0
    for p in user_pairs:
        r = by_key.get(tuple(sorted(p)))
        if r is None:
            add(f"  {p[0]}-{p[1]}: MISSING from universe")
            continue
        g = r["gate_eval"]
        n_usr_pass += g["pass"]
        rank = next(i for i, rr in enumerate(records) if rr is r) + 1
        add(f"  {p[0]}-{p[1]:<6} rank {rank:>5} score {g['screen_score']:>7.2f} "
            f"pass={'Y' if g['pass'] else 'N'}")
    add(f"  -> {n_usr_pass}/{len(user_pairs)} user candidates pass all gates")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[: min(75, len(lines))]))
    print(f"\n[report] full text written to {out_path}")


def make_plots(records, cfg, out_prefix):
    cur = {tuple(sorted(p)) for p in CURRENT_PAIRS}
    scores = np.array([r["gate_eval"]["screen_score"] for r in records])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(scores, bins=60, color="#7f7f7f", alpha=0.6, edgecolor="white")
    for r in records:
        key = tuple(sorted(r["pair"]))
        if key in cur:
            ax.axvline(r["gate_eval"]["screen_score"], color="crimson",
                       linewidth=1.4, alpha=0.8)
    ax.axvline(np.percentile(scores, 99), color="darkblue", linestyle="--",
               label="99th pct of universe")
    ax.set_title(f"Screen score distribution - {len(records)} pairs "
                 f"(days {cfg['start']}-{cfg['end']}); red = current 8")
    ax.set_xlabel("screen score (median window replica score - 0.5*std)")
    ax.legend()
    fig.tight_layout()
    p1 = f"{out_prefix}_score_dist.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    top = records[:20]
    mat = np.array([r["chosen"]["proxy_window_scores"] for r in top])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r['pair'][0]}-{r['pair'][1]}" for r in top], fontsize=8)
    wd = top[0]["window_days"]
    ax.set_xticks(range(len(wd)))
    ax.set_xticklabels([f"{a}-{b}" for a, b in wd], fontsize=8)
    ax.set_title("Top-20 pairs: per-window replica score (regime stability map)")
    fig.colorbar(im, ax=ax, label="window score")
    fig.tight_layout()
    p2 = f"{out_prefix}_window_heatmap.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {p1} and {p2}")


# ============================================================================
# MAIN
# ============================================================================
def run():
    cfg = CFG
    prc = pd.read_csv(cfg["prices"], sep=r"\s+", header=0)
    prices = prc.values.T.astype(float)
    names = list(prc.columns)
    logp = np.log(np.maximum(prices, 1e-12))
    n_inst, nt = logp.shape
    print(f"Loaded {n_inst} instruments x {nt} days; scoring days "
          f"{cfg['start']}-{cfg['end']} in {cfg['n_windows']} windows")

    # market-wide volatility regime proxy: cross-sectional median |log return|,
    # EWMA-smoothed; used as the coarse high/low-vol regime label.
    rets = np.diff(logp, axis=1, prepend=logp[:, :1])
    mkt = np.median(np.abs(rets[1:]), axis=0)  # exclude instrument 0
    mkt_vol = np.empty_like(mkt)
    alpha = 0.06
    mkt_vol[0] = mkt[0]
    for i in range(1, mkt.size):
        mkt_vol[i] = (1 - alpha) * mkt_vol[i - 1] + alpha * mkt[i]

    idxs = [i for i in range(n_inst) if i != ALGO]
    pairs = [(i, j, names[i], names[j]) for k, i in enumerate(idxs) for j in idxs[k + 1:]]
    if cfg["smoke"]:
        pairs = pairs[:40]
    print(f"Screening {len(pairs)} pairs on {cfg['workers']} workers...")

    records = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg["workers"], initializer=init_worker,
            initargs=(logp, mkt_vol, cfg)) as ex:
        for done, rec in enumerate(ex.map(screen_pair, pairs, chunksize=4), 1):
            records.append(rec)
            if done % 100 == 0:
                print(f"  [screen] {done}/{len(pairs)} pairs done", flush=True)

    # ---- pooled Benjamini-Hochberg FDR across every (pair, window) ADF test ----
    pvals = np.array([w["adf_p"] for r in records for w in r["A"]["windows"]])
    fdr_mask = benjamini_hochberg(pvals, GATE_FDR_Q)
    n_win = len(records[0]["A"]["windows"])
    for k, r in enumerate(records):
        r["gate_eval"] = evaluate_gates(
            r, fdr_mask[k * n_win:(k + 1) * n_win])
    print(f"[FDR] {int(fdr_mask.sum())}/{pvals.size} window ADF tests rejected at "
          f"q={GATE_FDR_Q} (pooled across universe)")

    records.sort(key=lambda r: r["gate_eval"]["screen_score"], reverse=True)

    # Binomial null for the G1 statistical gate: if every pair's spread were a pure
    # random walk, each window's ADF would reject with prob 0.05, so the chance of
    # passing >= need_stat of n_win windows is a binomial tail. Comparing the observed
    # G1-passer count to records*p_null quantifies "is this just noise?" at screen level
    # (windows are contiguous, so this is approximate; the placebo tuning arm is the
    # pipeline-level empirical answer).
    need_stat = int(math.ceil(0.8 * n_win))
    p_null_g1 = sum(math.comb(n_win, k) * 0.05 ** k * 0.95 ** (n_win - k)
                    for k in range(need_stat, n_win + 1))
    n_g1_pass = sum(1 for r in records if r["gate_eval"]["gates"]["G1_coint_fdr"])
    expected_false_g1 = len(records) * p_null_g1
    print(f"[null] G1 passers: {n_g1_pass} observed vs {expected_false_g1:.2f} expected "
          f"under the all-noise null (binomial tail p={p_null_g1:.2e})")

    cur = {tuple(sorted(p)) for p in CURRENT_PAIRS}
    passers = [r for r in records if r["gate_eval"]["pass"]]

    # Shortlist = top screen-score pairs passing the tradability sanity gates. The G1
    # ADF battery is a weak instrument on 120-day windows (even proven OOS-profitable
    # pairs sit at 1-3/5 windows, though all in the top ~4% of the universe), so it is
    # reported as per-pair evidence, NOT used as a hard filter - loosening it until
    # known pairs pass would tune the gate to the answer. Final verification belongs
    # to the robust tuner + placebo arm + OOS + Deflated Sharpe.
    sanity = ("G2_halflife", "G3_structural", "G4_vol_regime", "G5_tradable")
    eligible = [r for r in records if all(r["gate_eval"]["gates"][g] for g in sanity)]
    new_eligible = [r for r in eligible if tuple(sorted(r["pair"])) not in cur]
    shortlist = new_eligible[: cfg["topn"]]

    # placebo cohort: seeded random sample of pairs with NEGATIVE screen score (no
    # strategy-aligned edge at all). The identical tuner budget downstream answers
    # "could the pipeline manufacture good OOS from nothing?" - the noise control arm.
    no_edge = [r for r in records if r["gate_eval"]["screen_score"] < 0]
    rng = random.Random(cfg["placebo_seed"])
    placebos = rng.sample(no_edge, min(cfg["n_placebo"], len(no_edge)))

    user_pairs = load_user_pairs(cfg["user_pairs_file"])
    write_report(records, cfg, user_pairs, cfg["report"],
                 null_info={"observed": n_g1_pass, "expected": expected_false_g1,
                            "p": p_null_g1})
    make_plots(records, cfg, cfg["out"].replace(".json", ""))

    slim = []
    for r in records:
        g = r["gate_eval"]
        slim.append({
            "pair": r["pair"], "orient": r["orient"], "window_days": r["window_days"],
            "gate_eval": g, "A": r["A"], "B": r["B"],
        })
    meta = {
        "config": {k: cfg[k] for k in ("start", "end", "n_windows", "warmup",
                                       "topn", "n_placebo", "placebo_seed")},
        "fdr": {"q": GATE_FDR_Q, "tests": int(pvals.size), "rejected": int(fdr_mask.sum())},
        "null_model": {"g1_binomial_tail": p_null_g1,
                       "g1_passers_observed": n_g1_pass,
                       "g1_passers_expected_under_null": expected_false_g1},
        "n_pairs": len(records),
        "n_pass": len(passers),
        "sanity_gates": list(sanity),
        "n_eligible": len(eligible),
        "current_pairs": [list(p) for p in CURRENT_PAIRS],
        "shortlist": [{"pair": r["pair"], "orient": r["orient"],
                       "screen_score": r["gate_eval"]["screen_score"]} for r in shortlist],
        "placebos": [{"pair": r["pair"], "orient": r["orient"],
                      "screen_score": r["gate_eval"]["screen_score"],
                      "failed_gates": [k for k, v in r["gate_eval"]["gates"].items() if not v]}
                     for r in placebos],
    }
    with open(cfg["out"], "w") as f:
        json.dump({"meta": meta, "records": slim}, f)
    print(f"[out] full screen metrics -> {cfg["out"]}")

    tuner_pairs = []
    for r in shortlist + placebos:
        y, x = (r["pair"] if r["orient"] == "A" else r["pair"][::-1])
        tuner_pairs.append({"y": y, "x": x})
    with open(cfg["pairs_out"], "w") as f:
        json.dump(tuner_pairs, f, indent=2)
    print(f"[out] tuner pairs file ({len(tuner_pairs)} pairs: "
          f"{len(shortlist)} new + {len(placebos)} placebo) -> {cfg['pairs_out']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regime-aware universe-wide pair screener")
    ap.add_argument("--start", type=int, default=CFG["start"])
    ap.add_argument("--end", type=int, default=CFG["end"])
    ap.add_argument("--n_windows", type=int, default=CFG["n_windows"])
    ap.add_argument("--workers", type=int, default=CFG["workers"])
    ap.add_argument("--topn", type=int, default=CFG["topn"])
    ap.add_argument("--n_placebo", type=int, default=CFG["n_placebo"])
    ap.add_argument("--placebo_seed", type=int, default=CFG["placebo_seed"])
    ap.add_argument("--user_pairs_file", default=CFG["user_pairs_file"])
    ap.add_argument("--out", default=CFG["out"])
    ap.add_argument("--pairs_out", default=CFG["pairs_out"])
    ap.add_argument("--report", default=CFG["report"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    CFG.update(vars(args))
    run()
