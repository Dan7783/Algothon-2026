#!/usr/bin/env python
"""
Rolling 100-day windows: once a pair is detected as stable_coint
(EG p<0.05 AND Hansen Lc stable), how long does that last?

Windows are evaluated in parallel (ProcessPoolExecutor).
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
LC_CRIT_5 = 0.470  # Hansen Lc, k=1
OUTPUT_DIR = Path("cointegration_output/stable_rolling_duration")
TOP_N_TIMELINE = 20
# Leave a couple cores free on this 24-thread machine
N_WORKERS = max(1, min(20, (os.cpu_count() or 4) - 2))


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def hansen_lc_fast(y: np.ndarray, x: np.ndarray) -> float:
    """Hansen Lc only (no extra dict overhead)."""
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
    quad = np.einsum("ti,ij,tj->t", S, Vinv, S)
    return float(np.sum(quad) / (n**2))


def process_window(payload: tuple) -> tuple[int, np.ndarray, list]:
    """
    Worker: evaluate all pairs for one rolling window end-day.
    Returns (end_day, stable_bool[n_pairs], hit_rows).
    """
    values, end, lookback, significance, lc_crit = payload
    start = end - lookback
    block = values[start:end]  # (T, n_inst)
    n_inst = block.shape[1]
    n_pairs = n_inst * (n_inst - 1) // 2
    stable = np.zeros(n_pairs, dtype=bool)
    hits = []

    pi = 0
    for i in range(n_inst):
        y = block[:, i]
        for j in range(i + 1, n_inst):
            x = block[:, j]
            ok = False
            pval = 1.0
            lc_val = np.nan
            if np.std(y) >= 1e-12 and np.std(x) >= 1e-12:
                try:
                    _, pval, _ = coint(y, x)
                    pval = float(pval)
                    if pval < significance:
                        lc_val = hansen_lc_fast(y, x)
                        ok = lc_val <= lc_crit
                except Exception:
                    pass
            stable[pi] = ok
            if ok:
                hits.append((end, i, j, pval, lc_val))
            pi += 1

    return end, stable, hits


def extract_spells(days: np.ndarray, on: np.ndarray) -> list[dict]:
    spells = []
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        start = i
        while i < n and on[i]:
            i += 1
        end = i - 1
        spells.append(
            {
                "start_day": int(days[start]),
                "end_day": int(days[end]),
                "duration_days": int(days[end] - days[start]) + STEP,
                "n_windows": end - start + 1,
                "censored": end == n - 1,
            }
        )
    return spells


def run_rolling_parallel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tickers = prices.columns.tolist()
    values = prices.values.astype(float)
    n_inst = len(tickers)
    pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i + 1 :]]
    ends = np.array(list(range(LOOKBACK, len(prices) + 1, STEP)))

    print(
        f"Rolling lookback={LOOKBACK}, step={STEP}, windows={len(ends)}, "
        f"pairs={len(pairs)}, workers={N_WORKERS}"
    )

    payloads = [
        (values, int(end), LOOKBACK, SIGNIFICANCE, LC_CRIT_5) for end in ends
    ]

    # Collect results in end-day order
    by_end: dict[int, tuple[np.ndarray, list]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_window, p): p[1] for p in payloads}
        for fut in as_completed(futures):
            end, stable, hits = fut.result()
            by_end[end] = (stable, hits)
            done += 1
            if done % 20 == 0 or done == len(ends):
                print(f"  completed {done}/{len(ends)} windows")

    stable_mat = np.zeros((len(pairs), len(ends)), dtype=bool)
    hit_rows = []
    for wi, end in enumerate(ends):
        stable, hits = by_end[int(end)]
        stable_mat[:, wi] = stable
        for end_d, i, j, pval, lc_val in hits:
            hit_rows.append(
                {
                    "day": end_d,
                    "pair": f"{tickers[i]}-{tickers[j]}",
                    "ticker_a": tickers[i],
                    "ticker_b": tickers[j],
                    "eg_pval": pval,
                    "lc": lc_val,
                }
            )

    spell_rows = []
    summary_rows = []
    for pi, (a, b) in enumerate(pairs):
        on = stable_mat[pi]
        if not on.any():
            continue
        pair = f"{a}-{b}"
        spells = extract_spells(ends, on)
        for s in spells:
            spell_rows.append({"pair": pair, "ticker_a": a, "ticker_b": b, **s})
        completed = [s for s in spells if not s["censored"]]
        durs = [s["duration_days"] for s in completed]
        summary_rows.append(
            {
                "pair": pair,
                "n_stable_windows": int(on.sum()),
                "pct_windows": 100.0 * on.mean(),
                "n_spells": len(spells),
                "median_spell_days": float(np.median(durs)) if durs else np.nan,
                "mean_spell_days": float(np.mean(durs)) if durs else np.nan,
                "max_spell_days": float(max(durs)) if durs else float(spells[-1]["duration_days"]),
                "total_stable_days_approx": int(on.sum() * STEP),
            }
        )

    spells_df = pd.DataFrame(spell_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["max_spell_days", "n_stable_windows", "median_spell_days"],
        ascending=[False, False, False],
    )
    hits_df = pd.DataFrame(hit_rows)
    meta = {"ends": ends, "pairs": pairs, "stable_mat": stable_mat, "tickers": tickers}
    return spells_df, summary_df, hits_df, meta


def plot_survival(spells_df: pd.DataFrame, out_dir: Path) -> None:
    completed = spells_df[~spells_df["censored"]]
    if completed.empty:
        print("No completed spells.")
        return
    durs = completed["duration_days"].values
    max_d = min(int(durs.max()), 200)
    lags = np.arange(0, max_d + 1, STEP)
    surv = np.array([(durs >= k).mean() for k in lags])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(lags, surv * 100, color="#4C72B0", lw=2)
    axes[0].fill_between(lags, 0, surv * 100, color="#4C72B0", alpha=0.2)
    axes[0].axhline(50, color="gray", ls="--", lw=1)
    if np.any(surv <= 0.5):
        half = int(lags[np.argmax(surv <= 0.5)])
        axes[0].axvline(half, color="red", ls=":", lw=1, label=f"50% survival ~{half}d")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Days after stable detection (still stable)")
    axes[0].set_ylabel("Survival (%)")
    axes[0].set_title(
        f"Stable-spell survival (n={len(durs)} completed)\n"
        f"Rolling {LOOKBACK}d lookback, step {STEP}d | EG + Lc stable"
    )
    axes[0].set_ylim(0, 105)

    axes[1].hist(
        durs,
        bins=np.arange(0, min(durs.max(), 150) + STEP, STEP),
        color="#55A868",
        edgecolor="white",
    )
    axes[1].axvline(np.median(durs), color="red", ls="--", label=f"median={np.median(durs):.0f}d")
    axes[1].axvline(np.mean(durs), color="orange", ls="--", label=f"mean={np.mean(durs):.1f}d")
    axes[1].set_xlabel("Spell duration (days)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("How long can a stable pair be traded?")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "stable_spell_survival.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_timelines(summary_df: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    ends = meta["ends"]
    pairs = meta["pairs"]
    stable_mat = meta["stable_mat"]
    pair_to_idx = {f"{a}-{b}": i for i, (a, b) in enumerate(pairs)}

    top = summary_df.head(TOP_N_TIMELINE)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.4 * len(top))))

    for row_i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        pair = row["pair"]
        pi = pair_to_idx[pair]
        on = stable_mat[pi]
        ax.plot([ends[0], ends[-1]], [row_i, row_i], color="#dddddd", lw=4, solid_capstyle="butt")
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j < len(on) and on[j]:
                j += 1
            ax.plot([ends[i], ends[j - 1]], [row_i, row_i], color="#2ca02c", lw=6, solid_capstyle="butt")
            dur = int(ends[j - 1] - ends[i]) + STEP
            ax.text((ends[i] + ends[j - 1]) / 2, row_i + 0.25, f"{dur}d", ha="center", va="bottom", fontsize=6, color="#1a5c1a")
            i = j
        ax.text(ends[-1] + 8, row_i, f"max={row['max_spell_days']:.0f}d", va="center", fontsize=7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(list(top["pair"].iloc[::-1]), fontsize=9)
    ax.set_xlabel(f"Day (end of {LOOKBACK}d lookback)")
    ax.set_xlim(ends[0] - 5, ends[-1] + 80)
    ax.set_title(
        f"Top {TOP_N_TIMELINE} pairs by max stable spell — green = tradeable window\n"
        f"(stable = EG p<{SIGNIFICANCE} and Hansen Lc stable in rolling {LOOKBACK}d)"
    )
    plt.tight_layout()
    path = out_dir / "stable_pair_timelines.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_bars(summary_df: pd.DataFrame, out_dir: Path) -> None:
    top = summary_df.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.cm.viridis(top["n_stable_windows"] / max(top["n_stable_windows"].max(), 1))
    ax.barh(top["pair"], top["max_spell_days"], color=colors, edgecolor="white")
    for i, (_, r) in enumerate(top.iterrows()):
        med = r["median_spell_days"]
        med_s = f"{med:.0f}" if np.isfinite(med) else "—"
        ax.text(r["max_spell_days"] + 1, i, f"med={med_s}d | {int(r['n_spells'])} spells", va="center", fontsize=7)
    ax.set_xlabel("Max stable spell duration (days)")
    ax.set_title(f"Longest tradeable windows after stable detection\n(rolling {LOOKBACK}d, step {STEP}d)")
    plt.tight_layout()
    path = out_dir / "longest_stable_spells.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments\n")

    spells_df, summary_df, hits_df, meta = run_rolling_parallel(prices)

    spells_df.to_csv(OUTPUT_DIR / "stable_spells.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "stable_pair_summary.csv", index=False)
    hits_df.to_csv(OUTPUT_DIR / "stable_hits_detail.csv", index=False)

    print(f"\nPairs ever stable: {len(summary_df)}")
    print(f"Total spells: {len(spells_df)}")
    completed = spells_df[~spells_df["censored"]] if len(spells_df) else spells_df
    if len(completed):
        d = completed["duration_days"]
        print(f"Completed spells: {len(completed)}")
        print(f"  median duration: {d.median():.0f} days")
        print(f"  mean duration:   {d.mean():.1f} days")
        print(f"  >=25d:  {(d >= 25).mean()*100:.0f}%")
        print(f"  >=50d:  {(d >= 50).mean()*100:.0f}%")
        print(f"  >=100d: {(d >= 100).mean()*100:.0f}%")

    if len(summary_df):
        print("\nTop pairs by max stable spell:")
        print(
            summary_df.head(15)[
                ["pair", "max_spell_days", "median_spell_days", "n_spells", "n_stable_windows", "pct_windows"]
            ].to_string(index=False)
        )
        plot_survival(spells_df, OUTPUT_DIR)
        plot_top_bars(summary_df, OUTPUT_DIR)
        plot_top_timelines(summary_df, meta, OUTPUT_DIR)


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor spawn
    main()
