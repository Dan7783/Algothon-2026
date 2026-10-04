#!/usr/bin/env python
"""
Show why EG+Lc binary detection is finicky vs continuous 'activity'.

For each pair: rolling 100d
  - binary: EG p<0.05 AND Lc stable (current method)
  - continuous: -log10(EG p) smoothed
  - hysteresis: enter p<0.05, exit only when p>0.20
  - rolling level correlation |rho|

Ranks pairs by continuous activity persistence (matches visual better).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

LOOKBACK = 100
STEP = 5
SIGNIFICANCE = 0.05
EXIT_P = 0.20  # hysteresis exit
LC_CRIT_5 = 0.47
SMOOTH = 3  # rolling median window (in steps) for continuous strength
N_WORKERS = max(1, min(20, (os.cpu_count() or 4) - 2))
OUTPUT_DIR = Path("cointegration_output/activity_vs_binary")

# Pairs that looked "always on" or interesting from prior analysis + open file
FOCUS_PAIRS = [
    ("AENO", "NWIG"),
    ("AENO", "RRES"),
    ("HRET", "MMBT"),
    ("HRET", "FCSG"),
    ("HRET", "BENI"),
    ("HRET", "ULXY"),
    ("EORC", "NGTE"),
    ("SMAH", "NGTE"),
    ("SMAH", "CTGI"),
    ("ELLT", "ILVX"),
    ("GARI", "IHOZ"),
    ("ALGO", "SMAH"),
]


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def hansen_lc_fast(y: np.ndarray, x: np.ndarray) -> float:
    n = len(y)
    Z = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta
    f = Z * e[:, None]
    L = max(1, int(np.floor(n ** (1 / 3))))
    Omega = (f.T @ f) / n
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (f[ell:].T @ f[:-ell]) / n
        Omega += w * (Gamma + Gamma.T)
    S = np.cumsum(f, axis=0)
    Vinv = np.linalg.pinv(Omega)
    return float(np.einsum("ti,ij,tj->t", S, Vinv, S).sum() / (n**2))


def process_window(payload: tuple) -> tuple[int, dict]:
    """Compute metrics for FOCUS pairs only (fast) for one window."""
    values, colnames, end, lookback, pair_idx = payload
    # pair_idx: list of (i, j, name)
    start = end - lookback
    block = values[start:end]
    out = {}
    for i, j, name in pair_idx:
        y, x = block[:, i], block[:, j]
        eg_p, strength, lc, binary, corr = 1.0, 0.0, np.nan, False, 0.0
        if np.std(y) >= 1e-12 and np.std(x) >= 1e-12:
            try:
                corr = float(np.corrcoef(y, x)[0, 1])
                _, eg_p, _ = coint(y, x)
                eg_p = float(eg_p)
                strength = float(-np.log10(max(eg_p, 1e-16)))
                if eg_p < SIGNIFICANCE:
                    lc = hansen_lc_fast(y, x)
                    binary = lc <= LC_CRIT_5
            except Exception:
                pass
        out[name] = {
            "eg_p": eg_p,
            "strength": strength,
            "lc": lc,
            "binary": binary,
            "corr": corr,
        }
    return end, out


def hysteresis_from_p(pvals: np.ndarray, enter: float = SIGNIFICANCE, exit_p: float = EXIT_P) -> np.ndarray:
    """Sticky ON: turn on at p<enter, stay on until p>exit_p."""
    on = np.zeros(len(pvals), dtype=bool)
    state = False
    for i, p in enumerate(pvals):
        if not state and p < enter:
            state = True
        elif state and p > exit_p:
            state = False
        on[i] = state
    return on


def smooth_series(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    if k <= 1:
        return x
    s = pd.Series(x).rolling(k, center=True, min_periods=1).median()
    return s.values


def run_focus_rolling(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    colnames = list(prices.columns)
    name_to_i = {c: i for i, c in enumerate(colnames)}
    pair_idx = []
    for a, b in FOCUS_PAIRS:
        if a in name_to_i and b in name_to_i:
            pair_idx.append((name_to_i[a], name_to_i[b], f"{a}-{b}"))

    values = prices.values.astype(float)
    ends = list(range(LOOKBACK, len(prices) + 1, STEP))
    print(f"Focus pairs={len(pair_idx)}, windows={len(ends)}, workers={N_WORKERS}")

    payloads = [(values, colnames, end, LOOKBACK, pair_idx) for end in ends]
    by_end = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(process_window, p) for p in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            end, out = fut.result()
            by_end[end] = out
            if i % 30 == 0 or i == len(ends):
                print(f"  {i}/{len(ends)} windows")

    series = {}
    for _, _, name in pair_idx:
        rows = []
        for end in ends:
            m = by_end[end][name]
            rows.append({"day": end, **m})
        df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
        df["strength_smooth"] = smooth_series(df["strength"].values)
        df["hysteresis"] = hysteresis_from_p(df["eg_p"].values)
        df["active_visual"] = df["strength_smooth"] >= -np.log10(SIGNIFICANCE)  # smooth p<=0.05
        series[name] = df
    return series


def activity_score(df: pd.DataFrame) -> dict:
    """How 'always on' does this pair look on continuous metrics?"""
    thr = -np.log10(SIGNIFICANCE)
    strength = df["strength_smooth"].values
    binary = df["binary"].values.astype(float)
    hyst = df["hysteresis"].values.astype(float)
    # fraction of time above significance on smoothed strength
    frac_smooth = float((strength >= thr).mean())
    frac_binary = float(binary.mean())
    frac_hyst = float(hyst.mean())
    # jumpiness of binary vs smooth
    bin_switches = int(np.sum(np.diff(binary.astype(int)) != 0))
    hyst_switches = int(np.sum(np.diff(hyst.astype(int)) != 0))
    smooth_switches = int(np.sum(np.diff((strength >= thr).astype(int)) != 0))
    mean_corr = float(np.nanmean(np.abs(df["corr"].values)))
    mean_strength = float(np.nanmean(strength))
    return {
        "frac_smooth_active": frac_smooth,
        "frac_binary_stable": frac_binary,
        "frac_hysteresis": frac_hyst,
        "binary_switches": bin_switches,
        "hyst_switches": hyst_switches,
        "smooth_switches": smooth_switches,
        "mean_abs_corr": mean_corr,
        "mean_strength": mean_strength,
        "finicky_gap": frac_smooth - frac_binary,  # visual active but binary off
    }


def plot_pair_panels(series: dict[str, pd.DataFrame], out_dir: Path) -> None:
    names = list(series.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    thr = -np.log10(SIGNIFICANCE)

    for ax, name in zip(axes, names):
        df = series[name]
        ax.fill_between(df["day"], 0, 1, where=df["binary"], color="#C44E52", alpha=0.35, transform=ax.get_xaxis_transform(), label="binary EG+Lc")
        ax.fill_between(df["day"], 0, 1, where=df["hysteresis"], color="#4C72B0", alpha=0.2, transform=ax.get_xaxis_transform(), label="hysteresis EG")
        ax.plot(df["day"], df["strength_smooth"], color="#2ca02c", lw=1.5, label="smooth -log10(p)")
        ax.axhline(thr, color="gray", ls="--", lw=0.8)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(0, max(4, df["strength_smooth"].max() * 1.05))
        sc = activity_score(df)
        ax.text(
            0.99,
            0.9,
            f"smooth ON {sc['frac_smooth_active']*100:.0f}% | binary {sc['frac_binary_stable']*100:.0f}% | "
            f"hyst {sc['frac_hysteresis']*100:.0f}% | bin switches={sc['binary_switches']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#333",
        )

    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    axes[-1].set_xlabel(f"Day (end of {LOOKBACK}d lookback)")
    fig.suptitle(
        "Binary detection is finicky — continuous strength stays elevated for some pairs\n"
        f"red=EG+Lc stable | blue=hysteresis EG (enter p<{SIGNIFICANCE}, exit p>{EXIT_P}) | green=smoothed cointegration strength",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "focus_pairs_activity_panels.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_ranking(scores: pd.DataFrame, out_dir: Path) -> None:
    s = scores.sort_values("frac_smooth_active", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(s))
    ax.barh(y - 0.2, s["frac_smooth_active"] * 100, height=0.35, label="Smooth strength ON %", color="#2ca02c")
    ax.barh(y + 0.2, s["frac_binary_stable"] * 100, height=0.35, label="Binary EG+Lc ON %", color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(s["pair"])
    ax.set_xlabel("% of rolling windows active")
    ax.set_title(
        "Visual/continuous activity vs finicky binary filter\n"
        "(pairs ranked by smoothed cointegration strength time-on)"
    )
    ax.legend(fontsize=8)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(
            101,
            i,
            f"gap={r['finicky_gap']*100:+.0f}pp | bin switches={int(r['binary_switches'])}",
            va="center",
            fontsize=7,
        )
    ax.set_xlim(0, 130)
    plt.tight_layout()
    path = out_dir / "activity_vs_binary_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_example_prices(prices: pd.DataFrame, series: dict, out_dir: Path) -> None:
    """Overlay normalized prices with smooth-active shading for top continuous pairs."""
    scores = []
    for name, df in series.items():
        sc = activity_score(df)
        sc["pair"] = name
        scores.append(sc)
    scores = pd.DataFrame(scores).sort_values("frac_smooth_active", ascending=False)
    top = scores.head(4)["pair"].tolist()

    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]
    for ax, name in zip(axes, top):
        a, b = name.split("-")
        df = series[name]
        # map lookback-end day to shade; price index is 0..n-1 with day=t meaning days 1..t used
        t = np.arange(1, len(prices) + 1)
        ya = prices[a].values / prices[a].iloc[0]
        yb = prices[b].values / prices[b].iloc[0]
        ax.plot(t, ya, label=a, lw=1.2)
        ax.plot(t, yb, label=b, lw=1.2)
        thr = -np.log10(SIGNIFICANCE)
        for _, row in df.iterrows():
            if row["strength_smooth"] >= thr:
                # shade the lookback window end point neighborhood
                ax.axvspan(row["day"] - STEP / 2, row["day"] + STEP / 2, color="#2ca02c", alpha=0.15, lw=0)
            if row["binary"]:
                ax.axvspan(row["day"] - STEP / 2, row["day"] + STEP / 2, color="#C44E52", alpha=0.25, lw=0)
        ax.set_ylabel("Price / day1")
        ax.set_title(f"{name}: green=smooth-active, red overlay=binary EG+Lc (often drops while prices still track)")
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    path = out_dir / "prices_with_activity_overlay.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days")

    series = run_focus_rolling(prices)
    # save per-pair series
    for name, df in series.items():
        df.to_csv(OUTPUT_DIR / f"series_{name.replace('-', '_')}.csv", index=False)

    scores = []
    for name, df in series.items():
        sc = activity_score(df)
        sc["pair"] = name
        scores.append(sc)
    scores_df = pd.DataFrame(scores).sort_values("finicky_gap", ascending=False)
    scores_df.to_csv(OUTPUT_DIR / "activity_scores.csv", index=False)

    print("\n=== Continuous vs binary (finicky gap = smooth ON - binary ON) ===")
    print(
        scores_df[
            ["pair", "frac_smooth_active", "frac_binary_stable", "frac_hysteresis", "finicky_gap", "binary_switches", "mean_abs_corr"]
        ].to_string(index=False)
    )

    plot_pair_panels(series, OUTPUT_DIR)
    plot_ranking(scores_df, OUTPUT_DIR)
    plot_example_prices(prices, series, OUTPUT_DIR)

    print("\nInterpretation:")
    print("  Large finicky_gap + many binary_switches => tests flicker while relationship looks active.")
    print("  Prefer smoothed strength / hysteresis for regime ID; use Lc as soft confirmation, not hard gate.")


if __name__ == "__main__":
    main()
