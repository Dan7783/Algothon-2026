"""Per-pair book lab: benchmark, screen, and tune smah-style pair books.

Benchmark: each atlas pair traded ALONE with the four_sigmas ATLAS mechanics
(expanding Kalman hedge, ADF(1) gate at -2.9, 145-day z fade, OU filter),
scored with faithful eval.py accounting on train 251-500 and test 501-750.

Books: the smah_ngte template generalized to any (y, x) — raw prices into the
sticky PairSegFilter, entry/exit/stop tied to the live CUSUM h, hedge frozen
at entry, stand down on cuts, hold through adapt.

Tuning: per pair, the filter path is computed once per filter config and every
threshold combo is re-simulated cheaply on that path.  Selection uses ONLY the
train window: among the top configs by train score, keep the one with the best
worst-half score (251-375 vs 376-500), tie-broken by lower turnover.  The test
window is scored once, for the selected config only.

Screen: every non-atlas pair (instrument 0 excluded) is run through the filter
on train days only; pairs with a stationary filter spread (EG-corrected ADF),
positive beta and few cuts become extra book candidates.

Artifacts land in cusum_output/pair_books/.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import json
import time
from math import sqrt
from pathlib import Path

import numpy as np

from research.pairs_analysis.pair_seg_filter import PairSegConfig, PairSegFilter

N_INST = 51
LEG = 10_000.0
MIN_HISTORY = 80

TRAIN_START, TRAIN_END = 251, 500
TEST_START, TEST_END = 501, 750
TRAIN_MID = 375  # split-half stability boundary

ATLAS_I = [10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26]
ATLAS_J = [46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32]
ATLAS_PAIRS = list(zip(ATLAS_I, ATLAS_J))
EXTRA_SEED_PAIRS = [(10, 45)]  # SMAH/NGTE, the user's worked example

EG_ADF_5PCT = -3.34  # Engle-Granger 5% critical for an estimated hedge
SCREEN_MAX_CUTS = 3
SCREEN_TOP = 8

OUT_DIR = Path("cusum_output/pair_books")

# ---- eval.py accounting constants ----
COMM_RATE = 0.0001
POS_LIMIT = 10_000.0


def load_raw_prices(path: str = "prices.txt") -> np.ndarray:
    import pandas as pd

    return pd.read_csv(path, sep=r"\s+", header=0).to_numpy(dtype=float).T


def eval_score(mu: float, sigma: float) -> float:
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = sqrt(250.0) * mu / sigma
    frac = sr * sr / (sr * sr + 1.0)
    return mu * frac


def account_two_leg(
    sy: np.ndarray,
    sx: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    start_day: int,
    end_day: int,
) -> np.ndarray:
    """Faithful eval.py calcPL for a two-instrument book.

    ``sy[t-1]``/``sx[t-1]`` are the integer target shares decided from a
    ``t``-day history (eval applies them at day ``t``'s close).  Commission is
    deducted one day late, exactly as in eval.py.  Returns the daily PnL for
    the scored days ``start_day..end_day``.
    """
    cash = 0.0
    cy = cx = 0.0
    value = 0.0
    comm = 0.0
    pll = []
    for t in range(start_day - 1, end_day + 1):
        cpy, cpx = py[t - 1], px[t - 1]
        if t < end_day:
            limit_y = int(POS_LIMIT / cpy)
            limit_x = int(POS_LIMIT / cpx)
            ny = float(np.clip(sy[t - 1], -limit_y, limit_y))
            nx = float(np.clip(sx[t - 1], -limit_x, limit_x))
        else:
            ny, nx = cy, cx
        dy, dx = ny - cy, nx - cx
        cash -= cpy * dy + cpx * dx + comm
        comm = COMM_RATE * (abs(dy) * cpy + abs(dx) * cpx)
        cy, cx = ny, nx
        pos_value = cy * cpy + cx * cpx
        today = cash + pos_value - value
        value = cash + pos_value
        if t >= start_day:
            pll.append(today)
    return np.asarray(pll)


def window_stats(pll: np.ndarray) -> dict:
    mu = float(pll.mean())
    sd = float(pll.std())
    return {"mean": mu, "std": sd, "score": eval_score(mu, sd)}


# --------------------------------------------------------------------------
# Benchmark: four_sigmas ATLAS mechanics, one pair alone
# --------------------------------------------------------------------------


def _kalman_beta(y: np.ndarray, x: np.ndarray) -> float:
    """four_sigmas expanding intercept/slope Kalman (verbatim mechanics)."""
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    covariance = np.eye(2) * 1e-2
    process_noise = np.eye(2) * 1e-5
    measurement_noise = np.var(np.diff(y[:120] - x[:120])) + 1e-8
    for t in range(len(y)):
        covariance += process_noise
        observation = np.array([1.0, x[t]])
        innovation_variance = (
            observation @ covariance @ observation + measurement_noise
        )
        gain = covariance @ observation / innovation_variance
        theta += gain * (y[t] - observation @ theta)
        covariance -= np.outer(gain, observation) @ covariance
    return float(theta[1])


def _adf1_tstat(series: np.ndarray) -> float:
    delta = np.diff(series)
    target = delta[1:]
    design = np.column_stack((np.ones(len(target)), series[1:-1], delta[:-1]))
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ target
    error = target - design @ coefficients
    variance = float(error @ error) / max(len(target) - 3, 1)
    standard_error = np.sqrt(max(variance * inverse[1, 1], 1e-300))
    return float(coefficients[1] / standard_error)


def _ou_kappa(spread: np.ndarray) -> float:
    delta = np.diff(spread)
    lag = spread[:-1] - spread[:-1].mean()
    denominator = (lag * lag).sum()
    return float(-(lag @ delta) / denominator) if denominator > 1e-18 else 0.0


def benchmark_positions(pair: tuple[int, int], prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily integer shares for one pair under four_sigmas atlas rules alone."""
    i, j = pair
    log_prices = np.log(prices)
    nt_total = prices.shape[1]
    sy = np.zeros(nt_total)
    sx = np.zeros(nt_total)
    for t in range(130, nt_total + 1):
        yl = log_prices[i, :t]
        xl = log_prices[j, :t]
        beta = _kalman_beta(yl[: t - 1], xl[: t - 1])
        full_spread = yl - beta * xl
        if _adf1_tstat(full_spread) >= -2.9:
            continue
        recent = full_spread[-145:]
        z = (recent[-1] - recent[:-1].mean()) / (recent[:-1].std() + 1e-12)
        if _ou_kappa(full_spread[-120:]) <= 0.0 or not (0.27 <= abs(z) <= 4.0):
            continue
        direction = -np.sign(z)
        sy[t - 1] = int(direction * LEG / prices[i, t - 1])
        sx[t - 1] = int(-direction * LEG / prices[j, t - 1])
    return sy, sx


# --------------------------------------------------------------------------
# Book: smah_ngte template generalized
# --------------------------------------------------------------------------


def run_filter_path(y: np.ndarray, x: np.ndarray, cfg: PairSegConfig) -> dict:
    filt = PairSegFilter(cfg)
    T = len(y)
    beta = np.empty(T)
    spread = np.empty(T)
    z = np.empty(T)
    h = np.empty(T)
    cut = np.zeros(T, dtype=bool)
    adapting = np.zeros(T, dtype=bool)
    for t in range(T):
        o = filt.step(float(y[t]), float(x[t]))
        beta[t] = o.beta
        spread[t] = o.spread
        z[t] = o.z
        h[t] = o.h
        cut[t] = o.cut
        adapting[t] = o.adapting
    return {
        "beta": beta,
        "spread": spread,
        "z": z,
        "h": h,
        "cut": cut,
        "adapting": adapting,
    }


def rolling_level_z(spread: np.ndarray, window: int) -> np.ndarray:
    """Trailing centered z of the filter spread: history excludes today."""
    T = len(spread)
    z = np.zeros(T)
    csum = np.concatenate(([0.0], np.cumsum(spread)))
    csq = np.concatenate(([0.0], np.cumsum(spread * spread)))
    for t in range(10, T):
        lo = max(0, t - window)
        n = t - lo
        mean = (csum[t] - csum[lo]) / n
        var = (csq[t] - csq[lo]) / n - mean * mean
        sd = sqrt(var) if var > 1e-18 else 0.0
        if sd > 1e-12:
            z[t] = (spread[t] - mean) / sd
    return z


def beta_dollar_legs(side: int, beta: float) -> tuple[float, float]:
    if side == 0 or abs(beta) < 1e-12:
        return 0.0, 0.0
    ab = abs(beta)
    if ab >= 1.0:
        d_x = -side * np.sign(beta) * LEG
        d_y = side * (LEG / ab)
    else:
        d_y = side * LEG
        d_x = -side * beta * LEG
    return float(d_y), float(d_x)


def simulate_book(
    path: dict,
    py: np.ndarray,
    px: np.ndarray,
    entry_mult: float,
    exit_mult: float,
    stop_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily integer shares from a recorded filter path (smah innovation-z)."""
    T = len(py)
    beta_p, z_p, h_p = path["beta"], path["z"], path["h"]
    cut_p, adapt_p = path["cut"], path["adapting"]
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    hedge = 0.0
    for d in range(MIN_HISTORY - 1, T):
        z = z_p[d]
        h_thr = max(2.0, h_p[d])
        entry, exit_, stop = entry_mult * h_thr, exit_mult * h_thr, stop_mult * h_thr
        if cut_p[d] and side != 0:
            side = 0
        elif adapt_p[d]:
            pass  # hold frozen hedge; no new entries
        elif side == 0:
            if z >= entry:
                side, hedge = -1, beta_p[d]
            elif z <= -entry:
                side, hedge = 1, beta_p[d]
        elif abs(z) <= exit_ or abs(z) >= stop:
            side = 0
        if side == 0:
            hedge = 0.0
            continue
        d_y, d_x = beta_dollar_legs(side, hedge)
        sy[d] = int(d_y / py[d])
        sx[d] = int(d_x / px[d])
    return sy, sx


def simulate_level_book(
    path: dict,
    z_lvl: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    entry: float,
    exit_: float,
    stop: float,
    equal_legs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark-class level fade on the filter spread, in sigma units.

    Same signal family as the four_sigmas atlas (centered level stretch), but
    the hedge is the adaptive filter beta and the book stands down on cuts and
    holds frozen through adapt bursts — that is what handles changing beta.
    """
    T = len(py)
    beta_p = path["beta"]
    cut_p, adapt_p = path["cut"], path["adapting"]
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    hedge = 0.0
    for d in range(MIN_HISTORY - 1, T):
        z = z_lvl[d]
        if cut_p[d] and side != 0:
            side = 0
        elif adapt_p[d]:
            pass
        elif side == 0:
            if z >= entry:
                side, hedge = -1, beta_p[d]
            elif z <= -entry:
                side, hedge = 1, beta_p[d]
        elif abs(z) <= exit_ or abs(z) >= stop:
            side = 0
        if side == 0:
            hedge = 0.0
            continue
        if equal_legs:
            d_y, d_x = side * LEG, -side * LEG
        else:
            d_y, d_x = beta_dollar_legs(side, hedge)
        sy[d] = int(d_y / py[d])
        sx[d] = int(d_x / px[d])
    return sy, sx


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------

Q_GRID = [0.0, 3e-8, 1e-6]
K_GRID = [0.45, 1.0]
H_GRID = [6.0, 10.0]
MSL_GRID = [30, 60]
ADAPT_GRID = [8, 14]
OS_GRID = [1.0, 1.15]
# innovation-z thresholds (multiples of live CUSUM h)
ENTRY_GRID = [0.20, 0.30, 0.40]
EXIT_GRID = [0.08, 0.13, 0.20]
STOP_GRID = [0.45, 0.60, 0.90]
# level-z policy (sigma units on the centered filter spread)
LEVEL_W_GRID = [60, 145]
LEVEL_ENTRY_GRID = [0.3, 0.5, 0.75]
LEVEL_EXIT_GRID = [0.1, 0.25]
LEVEL_STOP_GRID = [3.0, 4.0]
# band policy: the benchmark's always-in fade (side = -sign(z) inside the
# band, re-evaluated daily, equal legs) on the adaptive filter spread
BAND_LO_GRID = [0.2, 0.27, 0.4]
BAND_HI = 4.0
TOP_K = 40
FINALIST_TOLERANCE = 0.95  # near-ties on train resolved toward fewer cuts


def filter_configs() -> list[PairSegConfig]:
    cfgs = []
    for q in Q_GRID:
        for k in K_GRID:
            for h in H_GRID:
                for msl in MSL_GRID:
                    for ab in ADAPT_GRID:
                        for os_ in OS_GRID:
                            cfgs.append(
                                PairSegConfig(
                                    q_beta=q,
                                    cusum_k=k,
                                    h=h,
                                    min_seg_len=msl,
                                    adapt_bars=ab,
                                    snap_overshoot=os_,
                                    target_seg=50,
                                    snap_window=8,
                                    cov_inflate=50.0,
                                )
                            )
    return cfgs


def cfg_key(cfg: PairSegConfig) -> dict:
    return {
        "q_beta": cfg.q_beta,
        "cusum_k": cfg.cusum_k,
        "h": cfg.h,
        "min_seg_len": cfg.min_seg_len,
        "adapt_bars": cfg.adapt_bars,
        "snap_overshoot": cfg.snap_overshoot,
    }


def simulate_band_book(
    path: dict,
    z_lvl: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """four_sigmas-style always-in band fade on the filter spread.

    Side is re-evaluated every day (no hysteresis), equal dollar legs — the
    benchmark's own policy, with the expanding-Kalman hedge replaced by the
    adaptive filter spread plus cut/adapt stand-down.
    """
    T = len(py)
    cut_p, adapt_p = path["cut"], path["adapting"]
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    for d in range(MIN_HISTORY - 1, T):
        z = z_lvl[d]
        if cut_p[d]:
            side = 0
        elif adapt_p[d]:
            pass  # hold current side through the re-estimation burst
        elif lo <= abs(z) <= hi:
            side = -int(np.sign(z))
        else:
            side = 0
        if side == 0:
            continue
        sy[d] = int(side * LEG / py[d])
        sx[d] = int(-side * LEG / px[d])
    return sy, sx


def _score_row(
    sy: np.ndarray,
    sx: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
) -> dict:
    train = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
    h1 = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_MID)
    h2 = account_two_leg(sy, sx, py, px, TRAIN_MID + 1, TRAIN_END)
    return {
        "train": window_stats(train),
        "half1": eval_score(h1.mean(), h1.std()),
        "half2": eval_score(h2.mean(), h2.std()),
        "turnover": float(np.abs(np.diff(sy)).sum() + np.abs(np.diff(sx)).sum()),
    }


def _resimulate(row: dict, path: dict, py: np.ndarray, px: np.ndarray):
    if row["mode"] == "innov":
        return simulate_book(path, py, px, row["entry"], row["exit"], row["stop"])
    z_lvl = rolling_level_z(path["spread"], row["window"])
    if row["mode"] == "band":
        return simulate_band_book(path, z_lvl, py, px, row["entry"], row["stop"])
    return simulate_level_book(
        path, z_lvl, py, px, row["entry"], row["exit"], row["stop"],
        row["equal_legs"],
    )


def tune_pair(pair: tuple[int, int], prices: np.ndarray) -> dict:
    i, j = pair
    py, px = prices[i], prices[j]
    rows = []
    for ci, cfg in enumerate(filter_configs()):
        path = run_filter_path(py, px, cfg)
        n_cuts = int(path["cut"].sum())
        for me in ENTRY_GRID:
            for mx in EXIT_GRID:
                for ms in STOP_GRID:
                    sy, sx = simulate_book(path, py, px, me, mx, ms)
                    rows.append(
                        {
                            "mode": "innov",
                            "cfg_index": ci,
                            "cfg": cfg_key(cfg),
                            "entry": me,
                            "exit": mx,
                            "stop": ms,
                            "path_cuts": n_cuts,
                            **_score_row(sy, sx, py, px),
                        }
                    )
        for w in LEVEL_W_GRID:
            z_lvl = rolling_level_z(path["spread"], w)
            for me in LEVEL_ENTRY_GRID:
                for mx in LEVEL_EXIT_GRID:
                    for ms in LEVEL_STOP_GRID:
                        for eq in (True, False):
                            sy, sx = simulate_level_book(
                                path, z_lvl, py, px, me, mx, ms, eq
                            )
                            rows.append(
                                {
                                    "mode": "level",
                                    "cfg_index": ci,
                                    "cfg": cfg_key(cfg),
                                    "window": w,
                                    "equal_legs": eq,
                                    "entry": me,
                                    "exit": mx,
                                    "stop": ms,
                                    "path_cuts": n_cuts,
                                    **_score_row(sy, sx, py, px),
                                }
                            )
            for lo in BAND_LO_GRID:
                sy, sx = simulate_band_book(path, z_lvl, py, px, lo, BAND_HI)
                rows.append(
                    {
                        "mode": "band",
                        "cfg_index": ci,
                        "cfg": cfg_key(cfg),
                        "window": w,
                        "equal_legs": True,
                        "entry": lo,
                        "exit": lo,
                        "stop": BAND_HI,
                        "path_cuts": n_cuts,
                        **_score_row(sy, sx, py, px),
                    }
                )
    rows.sort(key=lambda r: -r["train"]["score"])
    finalists = rows[:TOP_K]
    best_score = finalists[0]["train"]["score"]
    floor = best_score * FINALIST_TOLERANCE if best_score > 0 else best_score
    eligible = [r for r in finalists if r["train"]["score"] >= floor]
    chosen = min(
        eligible,
        key=lambda r: (
            r.get("path_cuts", 99),
            -min(r["half1"], r["half2"]),
            r["turnover"],
        ),
    )
    # Score the test window once, for the selected config only.
    cfg = filter_configs()[chosen["cfg_index"]]
    path = run_filter_path(py, px, cfg)
    sy, sx = _resimulate(chosen, path, py, px)
    test = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
    chosen = dict(chosen)
    chosen["test"] = window_stats(test)
    chosen["cuts"] = int(path["cut"].sum())
    chosen["best_train_score"] = rows[0]["train"]["score"]
    return chosen


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------


def screen_pairs(prices: np.ndarray) -> list[dict]:
    train_prices = prices[:, :TRAIN_END]
    atlas_keys = {tuple(sorted(p)) for p in ATLAS_PAIRS + EXTRA_SEED_PAIRS}
    cfg = PairSegConfig(target_seg=50, snap_window=8, cov_inflate=50.0)
    found = []
    for i in range(1, N_INST):
        for j in range(i + 1, N_INST):
            if tuple(sorted((i, j))) in atlas_keys:
                continue
            path = run_filter_path(train_prices[i], train_prices[j], cfg)
            cuts = int(path["cut"].sum())
            if cuts > SCREEN_MAX_CUTS or path["beta"][-1] <= 0.0:
                continue
            spread = train_prices[i] - path["beta"] * train_prices[j]
            adf = _adf1_tstat(spread[100:])
            if adf < EG_ADF_5PCT:
                found.append({"i": i, "j": j, "adf": adf, "cuts": cuts})
    found.sort(key=lambda r: r["adf"])
    return found[:SCREEN_TOP]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# v2: the uniform untuned pair_books policy — per-pair diagnostics and plots
# --------------------------------------------------------------------------

# Okabe-Ito colorblind-safe assignments, fixed per entity across all figures.
C_Y = "#0072B2"      # y-leg price, book equity, beta line
C_X = "#E69F00"      # x-leg price
C_BENCH = "#7f7f7f"  # benchmark equity
C_CUT = "#D55E00"    # CUSUM cut markers
C_LONG = "#009E73"   # long-spread state
C_SHORT = "#CC79A7"  # short-spread state


def simulate_v2_pair(pair: tuple[int, int], prices: np.ndarray) -> dict:
    """Drive the actual pair_books.PairBook day by day with tracing on."""
    import pair_books as pb

    i, j = pair
    book = pb.PairBook(i, j, trace=True)
    T = prices.shape[1]
    ly, lx = np.log(prices[i]), np.log(prices[j])
    for t in range(T):
        book._advance_day(
            float(ly[t]),
            float(lx[t]),
            decide=t >= pb.MIN_HISTORY - 1,
        )
    tr = book.trace
    side = np.asarray(tr["side"], dtype=int)
    py, px = prices[i], prices[j]
    sy = np.where(side != 0, side * LEG / py, 0.0).astype(int)
    sx = np.where(side != 0, -side * LEG / px, 0.0).astype(int)
    return {
        "side": side,
        "z": np.asarray(tr["z"]),
        "gate": np.asarray(tr["gate"], dtype=bool),
        "beta": np.asarray(tr["beta"]),
        "cut": np.asarray(tr["cut"], dtype=bool),
        "adapting": np.asarray(tr["adapting"], dtype=bool),
        "sy": sy,
        "sx": sx,
    }


def plot_v2_pair(
    pair: tuple[int, int],
    name: str,
    prices: np.ndarray,
    sim: dict,
    bench_sy: np.ndarray,
    bench_sx: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    from research.pairs_analysis.pair_seg_filter import _rolling_through_origin

    i, j = pair
    py, px = prices[i], prices[j]
    T = len(py)
    days = np.arange(1, T + 1)
    side, z = sim["side"], sim["z"]
    cuts = np.flatnonzero(sim["cut"])
    entries = np.flatnonzero((side != 0) & (np.roll(side, 1) == 0))
    exits = np.flatnonzero((side == 0) & (np.roll(side, 1) != 0))
    exits = exits[exits > 0]

    book_pll = account_two_leg(sim["sy"], sim["sx"], py, px, TRAIN_START, TEST_END)
    bench_pll = account_two_leg(bench_sy, bench_sx, py, px, TRAIN_START, TEST_END)
    pll_days = np.arange(TRAIN_START, TEST_END + 1)

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.2, 1.2]},
    )

    ax = axes[0]
    ax.plot(days, py / py[0], color=C_Y, lw=1.2, label=name.split("/")[0])
    ax.plot(days, px / px[0], color=C_X, lw=1.2, label=name.split("/")[1])
    for d in cuts:
        ax.axvline(d + 1, color=C_CUT, ls="--", lw=0.9, alpha=0.8)
    ax.set_ylabel("price / day 1")
    ax.set_title(f"{name}: adaptive-beta book vs four_sigmas-alone benchmark")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    raw = _rolling_through_origin(np.log(py), np.log(px), 100)
    ax.plot(days, raw, color="#444444", lw=1.6, alpha=0.7,
            label="rolling 100d OLS (log)")
    ax.plot(days, sim["beta"], color=C_Y, lw=1.3, label="filter beta (log)")
    for k, d in enumerate(cuts):
        ax.axvline(
            d + 1, color=C_CUT, ls="--", lw=0.9, alpha=0.8,
            label="CUSUM cut" if k == 0 else None,
        )
    ax.set_ylabel("hedge beta")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    zc = np.clip(z, -5.0, 5.0)
    in_long = side > 0
    in_short = side < 0
    ax.fill_between(days, -5, 5, where=in_long, color=C_LONG, alpha=0.12,
                    step="pre", label="long spread")
    ax.fill_between(days, -5, 5, where=in_short, color=C_SHORT, alpha=0.14,
                    step="pre", label="short spread")
    ax.plot(days, zc, color="#444444", lw=0.9, label="spread z (145d)")
    for lvl in (0.27, -0.27):
        ax.axhline(lvl, color="#999999", lw=0.7, ls=":")
    for lvl in (4.0, -4.0):
        ax.axhline(lvl, color=C_CUT, lw=0.7, ls=":")
    ax.scatter(entries + 1, zc[entries], marker="^", s=28, color=C_LONG,
               zorder=3, label="entry")
    ax.scatter(exits + 1, zc[exits], marker="x", s=28, color="#444444",
               zorder=3, label="exit")
    ax.set_ylabel("z / position")
    ax.set_ylim(-5.2, 5.2)
    ax.legend(fontsize=7, loc="upper left", ncol=3, frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(pll_days, np.cumsum(book_pll), color=C_Y, lw=1.5,
            label="book cumulative PnL")
    ax.plot(pll_days, np.cumsum(bench_pll), color=C_BENCH, lw=1.4, ls="--",
            label="benchmark cumulative PnL")
    ax.axvline(TRAIN_END + 0.5, color="#444444", lw=1.0, ls=":",
               label="train | test")
    ax.axhline(0.0, color="#999999", lw=0.7)
    ax.set_ylabel("cumulative $")
    ax.set_xlabel("day")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def v2_main() -> None:
    import pair_books as pb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_raw_prices()
    import pandas as pd

    tickers = list(pd.read_csv("prices.txt", sep=r"\s+", header=0).columns)
    benchmark = json.loads(
        (OUT_DIR / "benchmark.json").read_text(encoding="utf-8")
    )

    rows = {}
    for pair in pb.PAIRS:
        name = f"{tickers[pair[0]]}/{tickers[pair[1]]}"
        sim = simulate_v2_pair(pair, prices)
        py, px = prices[pair[0]], prices[pair[1]]
        train = window_stats(
            account_two_leg(sim["sy"], sim["sx"], py, px, TRAIN_START, TRAIN_END)
        )
        test = window_stats(
            account_two_leg(sim["sy"], sim["sx"], py, px, TEST_START, TEST_END)
        )
        bench_sy, bench_sx = benchmark_positions(pair, prices)
        b = benchmark[str(pair)]
        rows[str(pair)] = {
            "pair": list(pair),
            "name": name,
            "train": train,
            "test": test,
            "cuts": int(sim["cut"].sum()),
            "days_in_market": int((sim["side"] != 0).sum()),
        }
        plot_v2_pair(
            pair, name, prices, sim, bench_sy, bench_sx,
            OUT_DIR / "plots" / f"{name.replace('/', '_').lower()}.png",
        )
        print(
            f"v2 {name:>10}: train {train['score']:+7.2f} "
            f"(bench {b['train']['score']:+7.2f})  "
            f"test {test['score']:+7.2f} (bench {b['test']['score']:+7.2f})  "
            f"cuts {int(sim['cut'].sum())}  days-in {rows[str(pair)]['days_in_market']}"
        )
    (OUT_DIR / "v2_books.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'v2_books.json'} and plots/")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_raw_prices()
    import pandas as pd

    tickers = list(pd.read_csv("prices.txt", sep=r"\s+", header=0).columns)

    def name(p):
        return f"{tickers[p[0]]}/{tickers[p[1]]}"

    # ---- benchmark (cached across lab iterations) ----
    t0 = time.perf_counter()
    bench_path = OUT_DIR / "benchmark.json"
    if bench_path.exists():
        benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
        print(f"benchmark loaded from cache ({len(benchmark)} pairs)")
    else:
        benchmark = {}
        for pair in ATLAS_PAIRS + EXTRA_SEED_PAIRS:
            sy, sx = benchmark_positions(pair, prices)
            py, px = prices[pair[0]], prices[pair[1]]
            train = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
            test = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
            benchmark[str(pair)] = {
                "pair": list(pair),
                "name": name(pair),
                "train": window_stats(train),
                "test": window_stats(test),
            }
            b = benchmark[str(pair)]
            print(
                f"bench {name(pair):>10}: train {b['train']['score']:+7.2f} "
                f"(mu {b['train']['mean']:+6.1f})  test {b['test']['score']:+7.2f} "
                f"(mu {b['test']['mean']:+6.1f})"
            )
        bench_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
        print(f"benchmark done in {time.perf_counter() - t0:.0f}s")

    # ---- screen (cached) ----
    t1 = time.perf_counter()
    screen_path = OUT_DIR / "screen.json"
    if screen_path.exists():
        extras = json.loads(screen_path.read_text(encoding="utf-8"))
        print(f"screen loaded from cache ({len(extras)} pairs)")
    else:
        extras = screen_pairs(prices)
        for row in extras:
            row["name"] = name((row["i"], row["j"]))
            print(
                f"screen {row['name']:>10}: adf {row['adf']:+.2f} cuts {row['cuts']}"
            )
        screen_path.write_text(json.dumps(extras, indent=2), encoding="utf-8")
        print(f"screen done in {time.perf_counter() - t1:.0f}s")

    # ---- tune all books ----
    t2 = time.perf_counter()
    book_pairs = ATLAS_PAIRS + EXTRA_SEED_PAIRS + [
        (row["i"], row["j"]) for row in extras
    ]
    tuned = {}
    for pair in book_pairs:
        result = tune_pair(pair, prices)
        # Extra screened pairs get the same benchmark mechanics for comparison.
        key = str(pair)
        if key not in benchmark:
            sy, sx = benchmark_positions(pair, prices)
            py, px = prices[pair[0]], prices[pair[1]]
            benchmark[key] = {
                "pair": list(pair),
                "name": name(pair),
                "train": window_stats(
                    account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
                ),
                "test": window_stats(
                    account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
                ),
            }
        result["pair"] = list(pair)
        result["name"] = name(pair)
        tuned[key] = result
        b = benchmark[key]
        extra_bits = (
            f"W {result['window']} eq {result['equal_legs']}"
            if result["mode"] == "level"
            else ""
        )
        print(
            f"tuned {result['name']:>10} [{result['mode']}]: "
            f"train {result['train']['score']:+7.2f} "
            f"(bench {b['train']['score']:+7.2f})  test {result['test']['score']:+7.2f} "
            f"(bench {b['test']['score']:+7.2f})  cuts {result['cuts']} "
            f"entry {result['entry']} exit {result['exit']} stop {result['stop']} "
            f"{extra_bits}"
        )
    (OUT_DIR / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "tuned_books.json").write_text(
        json.dumps(tuned, indent=2), encoding="utf-8"
    )
    print(f"tuning done in {time.perf_counter() - t2:.0f}s")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        v2_main()
    else:
        main()
