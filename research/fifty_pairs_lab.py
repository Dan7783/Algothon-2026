"""Analyse, tune and plot the user's 50 labeled pairs (16 stable / 34 adaptive).

Stable pairs trade the four_sigmas atlas mechanics WITHOUT any ADF or OU-kappa
gate (all removed on instruction): expanding intercept/slope Kalman hedge on
log prices, centered z over a trailing window, always-in fade inside a band,
equal $10k legs.

Adaptive pairs trade the sticky CUSUM filter (defaults, log prices): centered
z on the CURRENT hedge segment (statistics reset on a cut), same band fade,
flat on cut, hold through adapt.

Tuning is SLEEVE-LEVEL only — two knobs per sleeve, (z window W, band entry
lo), small grids, selected on the train window 251-500 by the best worst-half
score (251-375 vs 376-500).  Per-pair tuning is deliberately absent: it was
shown earlier to be pure overfit.  The test window is scored once, after
selection, for reporting.

Outputs: cusum_output/fifty_pairs/{params.json, report.md, plots/<pair>.png}.
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
from pathlib import Path

import numpy as np

from research.pair_book_lab import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_MID,
    TRAIN_START,
    account_two_leg,
    eval_score,
    load_raw_prices,
    window_stats,
)
from pair_seg_filter import PairSegConfig, PairSegFilter, _rolling_through_origin

LEG = 10_000.0
MIN_HISTORY = 130
BAND_HI = 4.0

STABLE_NAMES = [
    ("AENO", "NWIG"), ("ILVX", "SMAH"), ("EORC", "NGTE"), ("ACAC", "HUXZ"),
    ("HETT", "ULXY"), ("MHRM", "SRTX"), ("EAFC", "SRTX"), ("CCNS", "SRTX"),
    ("MSDP", "ULXY"), ("FWWG", "BLBT"), ("MTNS", "BENI"), ("HRET", "CTGI"),
    ("ACIX", "ULXY"), ("DIHO", "RTTH"), ("MHRM", "RCRI"), ("HUXZ", "NWIG"),
]
ADAPTIVE_NAMES = [
    ("EORC", "SMAH"), ("EAFC", "HRET"), ("SMAH", "NGTE"), ("HETT", "SRTX"),
    ("EAFC", "EELT"), ("ULXY", "ELLT"), ("FWWG", "SMAH"), ("CUBO", "ULXY"),
    ("ACAC", "CTGI"), ("HUXZ", "CTGI"), ("MHRM", "ILVX"), ("AETS", "LSST"),
    ("NGTE", "ILVX"), ("HTRK", "RTTH"), ("NGTE", "CTGI"), ("SMAH", "CTGI"),
    ("DUCT", "FWWG"), ("BENI", "BLBT"), ("NPCK", "NAYO"), ("BLBT", "CTGI"),
    ("FARS", "BLBT"), ("DIHO", "FCSG"), ("LSST", "BENI"), ("FWWG", "FARS"),
    ("SRTX", "MDGI"), ("CTGI", "MDGI"), ("HRND", "RRES"), ("HTRK", "MSDP"),
    ("EORC", "ILVX"), ("ELLT", "IHOZ"), ("MHRM", "RRES"), ("ITPA", "ILVX"),
    ("ILVX", "HETT"), ("SRNA", "IHOZ"),
]

STABLE_W_GRID = [100, 145, 200]
ADAPTIVE_W_GRID = [60, 100, 145]
LO_GRID = [0.2, 0.27, 0.4]

OUT_DIR = Path("cusum_output/fifty_pairs")

C_Y = "#0072B2"
C_X = "#E69F00"
C_CUT = "#D55E00"
C_LONG = "#009E73"
C_SHORT = "#CC79A7"


def ticker_indices() -> dict[str, int]:
    header = Path("prices.txt").read_text(encoding="utf-8").split("\n", 1)[0]
    return {t: k for k, t in enumerate(header.split())}


# --------------------------------------------------------------------------
# Stable sleeve: four_sigmas expanding Kalman, gates removed
# --------------------------------------------------------------------------


def kalman_beta_path(ly: np.ndarray, lx: np.ndarray) -> np.ndarray:
    """beta[t] = four_sigmas expanding Kalman fitted on days < t (their lag)."""
    T = len(ly)
    beta = np.full(T, np.nan)
    if T < MIN_HISTORY:
        return beta
    theta = np.array([ly[:60].mean() - lx[:60].mean(), 1.0])
    covariance = np.eye(2) * 1e-2
    process_noise = np.eye(2) * 1e-5
    measurement_noise = np.var(np.diff(ly[:120] - lx[:120])) + 1e-8
    for t in range(1, T):
        # advance the state with day t-1 so beta[t] excludes today
        covariance += process_noise
        observation = np.array([1.0, lx[t - 1]])
        innovation_variance = (
            observation @ covariance @ observation + measurement_noise
        )
        gain = covariance @ observation / innovation_variance
        theta += gain * (ly[t - 1] - observation @ theta)
        covariance -= np.outer(gain, observation) @ covariance
        beta[t] = theta[1]
    return beta


def stable_z_path(ly: np.ndarray, lx: np.ndarray, beta: np.ndarray, window: int) -> np.ndarray:
    T = len(ly)
    z = np.zeros(T)
    for t in range(MIN_HISTORY - 1, T):
        b = beta[t]
        lo = max(0, t - window + 1)
        recent = ly[lo : t + 1] - b * lx[lo : t + 1]
        history = recent[:-1]
        sd = float(history.std())
        if sd > 1e-12:
            z[t] = (recent[-1] - float(history.mean())) / sd
    return z


def band_sim(
    z: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    lo: float,
    cut: np.ndarray | None = None,
    adapting: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Always-in band fade, equal legs; optional cut/adapt overrides."""
    T = len(py)
    sy = np.zeros(T)
    sx = np.zeros(T)
    side_path = np.zeros(T, dtype=int)
    side = 0
    for d in range(MIN_HISTORY - 1, T):
        if cut is not None and cut[d]:
            side = 0
        elif adapting is not None and adapting[d]:
            pass
        elif lo <= abs(z[d]) <= BAND_HI:
            side = -int(np.sign(z[d]))
        else:
            side = 0
        side_path[d] = side
        if side != 0:
            sy[d] = int(side * LEG / py[d])
            sx[d] = int(-side * LEG / px[d])
    return sy, sx, side_path


# --------------------------------------------------------------------------
# Adaptive sleeve: CUSUM filter, segment-aware z
# --------------------------------------------------------------------------


def filter_paths(ly: np.ndarray, lx: np.ndarray) -> dict:
    filt = PairSegFilter(PairSegConfig())
    T = len(ly)
    beta = np.empty(T)
    spread = np.empty(T)
    cut = np.zeros(T, dtype=bool)
    adapting = np.zeros(T, dtype=bool)
    for t in range(T):
        o = filt.step(float(ly[t]), float(lx[t]))
        beta[t] = o.beta
        spread[t] = o.spread
        cut[t] = o.cut
        adapting[t] = o.adapting
    return {"beta": beta, "spread": spread, "cut": cut, "adapting": adapting}


def adaptive_z_path(path: dict, window: int) -> np.ndarray:
    """Centered z of the filter spread on the current segment only."""
    spread, cut = path["spread"], path["cut"]
    T = len(spread)
    z = np.zeros(T)
    seg_start = 0
    for t in range(T):
        if cut[t]:
            seg_start = t  # statistics restart on the new hedge segment
        n = t - seg_start
        if n < 10:
            continue
        lo = max(seg_start, t - window)
        history = spread[lo:t]
        sd = float(history.std())
        if sd > 1e-12:
            z[t] = (spread[t] - float(history.mean())) / sd
    return z


# --------------------------------------------------------------------------
# Sleeve tuning: train-only, split-half stability
# --------------------------------------------------------------------------


def tune_sleeve(
    pair_sims: dict[tuple[int, float], list[tuple[np.ndarray, np.ndarray]]],
    prices: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[tuple[int, float], list[dict]]:
    """pair_sims: (W, lo) -> [(sy, sx) per pair].  Returns chosen combo."""
    table = []
    for (w, lo), sims in pair_sims.items():
        train_sum = None
        h1_sum = None
        h2_sum = None
        for (i, j), (sy, sx) in zip(pairs, sims):
            py, px = prices[i], prices[j]
            train = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
            h1 = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_MID)
            h2 = account_two_leg(sy, sx, py, px, TRAIN_MID + 1, TRAIN_END)
            train_sum = train if train_sum is None else train_sum + train
            h1_sum = h1 if h1_sum is None else h1_sum + h1
            h2_sum = h2 if h2_sum is None else h2_sum + h2
        table.append(
            {
                "window": w,
                "lo": lo,
                "train": eval_score(train_sum.mean(), train_sum.std()),
                "half1": eval_score(h1_sum.mean(), h1_sum.std()),
                "half2": eval_score(h2_sum.mean(), h2_sum.std()),
            }
        )
    table.sort(key=lambda r: -min(r["half1"], r["half2"]))
    chosen = table[0]
    return (chosen["window"], chosen["lo"]), table


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_pair(
    name: str,
    kind: str,
    py: np.ndarray,
    px: np.ndarray,
    beta: np.ndarray,
    z: np.ndarray,
    side: np.ndarray,
    lo: float,
    cut: np.ndarray | None,
    sy: np.ndarray,
    sx: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    T = len(py)
    days = np.arange(1, T + 1)
    cuts = np.flatnonzero(cut) if cut is not None else np.array([], dtype=int)
    entries = np.flatnonzero((side != 0) & (np.roll(side, 1) == 0))
    exits = np.flatnonzero((side == 0) & (np.roll(side, 1) != 0))
    exits = exits[exits > 0]
    pll = account_two_leg(sy, sx, py, px, TRAIN_START, TEST_END)
    pll_days = np.arange(TRAIN_START, TEST_END + 1)

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.2, 1.2]},
    )
    yt, xt = name.split("/")

    ax = axes[0]
    ax.plot(days, py / py[0], color=C_Y, lw=1.2, label=yt)
    ax.plot(days, px / px[0], color=C_X, lw=1.2, label=xt)
    for d in cuts:
        ax.axvline(d + 1, color=C_CUT, ls="--", lw=0.9, alpha=0.8)
    ax.set_ylabel("price / day 1")
    ax.set_title(f"{name} [{kind}]")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    raw = _rolling_through_origin(np.log(py), np.log(px), 100)
    ax.plot(days, raw, color="#444444", lw=1.6, alpha=0.7,
            label="rolling 100d OLS (log)")
    label = "filter beta" if kind == "adaptive" else "expanding Kalman beta"
    ax.plot(days, beta, color=C_Y, lw=1.3, label=label)
    for k, d in enumerate(cuts):
        ax.axvline(d + 1, color=C_CUT, ls="--", lw=0.9, alpha=0.8,
                   label="CUSUM cut" if k == 0 else None)
    ax.set_ylabel("hedge beta")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    zc = np.clip(z, -5.0, 5.0)
    ax.fill_between(days, -5, 5, where=side > 0, color=C_LONG, alpha=0.12,
                    step="pre", label="long spread")
    ax.fill_between(days, -5, 5, where=side < 0, color=C_SHORT, alpha=0.14,
                    step="pre", label="short spread")
    ax.plot(days, zc, color="#444444", lw=0.9, label="spread z")
    for lvl in (lo, -lo):
        ax.axhline(lvl, color="#999999", lw=0.7, ls=":")
    for lvl in (BAND_HI, -BAND_HI):
        ax.axhline(lvl, color=C_CUT, lw=0.7, ls=":")
    ax.scatter(entries + 1, zc[entries], marker="^", s=26, color=C_LONG,
               zorder=3, label="entry")
    ax.scatter(exits + 1, zc[exits], marker="x", s=26, color="#444444",
               zorder=3, label="exit")
    ax.set_ylabel("z / position")
    ax.set_ylim(-5.2, 5.2)
    ax.legend(fontsize=7, loc="upper left", ncol=3, frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(pll_days, np.cumsum(pll), color=C_Y, lw=1.5, label="cumulative PnL")
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_raw_prices()
    idx = ticker_indices()
    stable_pairs = [(idx[a], idx[b]) for a, b in STABLE_NAMES]
    adaptive_pairs = [(idx[a], idx[b]) for a, b in ADAPTIVE_NAMES]
    print("STABLE_PAIRS =", tuple(stable_pairs))
    print("ADAPTIVE_PAIRS =", tuple(adaptive_pairs))

    t0 = time.perf_counter()
    log_prices = np.log(prices)

    # ---- stable sleeve paths + tuning ----
    stable_beta = {
        p: kalman_beta_path(log_prices[p[0]], log_prices[p[1]])
        for p in stable_pairs
    }
    stable_sims: dict[tuple[int, float], list] = {}
    stable_z: dict[tuple[tuple[int, int], int], np.ndarray] = {}
    for w in STABLE_W_GRID:
        for p in stable_pairs:
            stable_z[(p, w)] = stable_z_path(
                log_prices[p[0]], log_prices[p[1]], stable_beta[p], w
            )
        for lo in LO_GRID:
            stable_sims[(w, lo)] = [
                band_sim(stable_z[(p, w)], prices[p[0]], prices[p[1]], lo)[:2]
                for p in stable_pairs
            ]
    stable_choice, stable_table = tune_sleeve(stable_sims, prices, stable_pairs)
    print(f"stable sleeve chosen (W, lo) = {stable_choice}")
    for r in stable_table[:4]:
        print(
            f"  W {r['window']:>3} lo {r['lo']:.2f}: train {r['train']:+8.1f} "
            f"halves {r['half1']:+8.1f}/{r['half2']:+8.1f}"
        )

    # ---- adaptive sleeve paths + tuning ----
    adaptive_path = {
        p: filter_paths(log_prices[p[0]], log_prices[p[1]])
        for p in adaptive_pairs
    }
    adaptive_sims: dict[tuple[int, float], list] = {}
    adaptive_zs: dict[tuple[tuple[int, int], int], np.ndarray] = {}
    for w in ADAPTIVE_W_GRID:
        for p in adaptive_pairs:
            adaptive_zs[(p, w)] = adaptive_z_path(adaptive_path[p], w)
        for lo in LO_GRID:
            adaptive_sims[(w, lo)] = [
                band_sim(
                    adaptive_zs[(p, w)],
                    prices[p[0]],
                    prices[p[1]],
                    lo,
                    cut=adaptive_path[p]["cut"],
                    adapting=adaptive_path[p]["adapting"],
                )[:2]
                for p in adaptive_pairs
            ]
    adaptive_choice, adaptive_table = tune_sleeve(
        adaptive_sims, prices, adaptive_pairs
    )
    print(f"adaptive sleeve chosen (W, lo) = {adaptive_choice}")
    for r in adaptive_table[:4]:
        print(
            f"  W {r['window']:>3} lo {r['lo']:.2f}: train {r['train']:+8.1f} "
            f"halves {r['half1']:+8.1f}/{r['half2']:+8.1f}"
        )

    # ---- per-pair stats + plots under the chosen sleeve params ----
    rows = []
    for kind, pairs, names, choice in (
        ("stable", stable_pairs, STABLE_NAMES, stable_choice),
        ("adaptive", adaptive_pairs, ADAPTIVE_NAMES, adaptive_choice),
    ):
        w, lo = choice
        for p, nm in zip(pairs, names):
            name = f"{nm[0]}/{nm[1]}"
            py, px = prices[p[0]], prices[p[1]]
            if kind == "stable":
                z = stable_z[(p, w)]
                beta = stable_beta[p]
                cut = None
                sy, sx, side = band_sim(z, py, px, lo)
                cuts_n = 0
            else:
                z = adaptive_zs[(p, w)]
                beta = adaptive_path[p]["beta"]
                cut = adaptive_path[p]["cut"]
                sy, sx, side = band_sim(
                    z, py, px, lo, cut=cut, adapting=adaptive_path[p]["adapting"]
                )
                cuts_n = int(cut.sum())
            train = window_stats(account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END))
            test = window_stats(account_two_leg(sy, sx, py, px, TEST_START, TEST_END))
            drift = float(np.nanstd(
                _rolling_through_origin(np.log(py), np.log(px), 100)[130:TRAIN_END]
            ))
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "pair": list(p),
                    "train": train,
                    "test": test,
                    "cuts": cuts_n,
                    "beta_drift_100d": drift,
                    "days_in": int((side != 0).sum()),
                }
            )
            plot_pair(
                name, kind, py, px, beta, z, side, lo, cut, sy, sx,
                OUT_DIR / "plots" / f"{nm[0].lower()}_{nm[1].lower()}.png",
            )
            print(
                f"{kind:>8} {name:>10}: train {train['score']:+7.2f} "
                f"test {test['score']:+7.2f} cuts {cuts_n} "
                f"days-in {rows[-1]['days_in']} drift {drift:.3f}"
            )

    params = {
        "stable": {"window": stable_choice[0], "lo": stable_choice[1]},
        "adaptive": {"window": adaptive_choice[0], "lo": adaptive_choice[1]},
        "band_hi": BAND_HI,
        "stable_pairs": [list(p) for p in stable_pairs],
        "adaptive_pairs": [list(p) for p in adaptive_pairs],
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (OUT_DIR / "per_pair.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
