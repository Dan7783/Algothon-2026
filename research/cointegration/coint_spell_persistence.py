#!/usr/bin/env python
"""
Persistence after identification: once rolling-window cointegration turns ON
using lagging data, how long does it stay ON before disappearing?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

LOOKBACK = 100
STEP = 1  # daily roll for precise spell lengths
SIGNIFICANCE = 0.05
OUTPUT_DIR = Path("cointegration_output/coint_persistence_spells_100")

# All pairs is expensive at daily step; use persistence leaders + random sample of others
# We'll compute for every pair but with STEP=5 for breadth, and STEP=1 for top pairs.
TOP_PAIRS = [
    ("ELLT", "ILVX"),
    ("AMRP", "HRET"),
    ("GARI", "IHOZ"),
    ("AENO", "RRES"),
    ("SRTX", "EAFC"),
    ("SMAH", "CTGI"),
    ("EORC", "NGTE"),
    ("HRET", "ULXY"),
    ("RTTH", "NGTE"),
    ("AENO", "NWIG"),
    ("GARI", "ULXY"),
    ("RRES", "ACAC"),
    ("ANSO", "ACIX"),
    ("CTGI", "HRND"),
    ("NWIG", "FWWG"),
]


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def rolling_sig_series(y: np.ndarray, x: np.ndarray, lookback: int = LOOKBACK, step: int = STEP) -> pd.DataFrame:
    rows = []
    for t in range(lookback, len(y), step):
        y_lb, x_lb = y[t - lookback : t], x[t - lookback : t]
        if np.std(y_lb) < 1e-12 or np.std(x_lb) < 1e-12:
            pval = 1.0
        else:
            _, pval, _ = coint(y_lb, x_lb)
        rows.append({"day": t, "pval": pval, "sig": pval < SIGNIFICANCE, "strength": -np.log10(max(pval, 1e-16))})
    return pd.DataFrame(rows)


def extract_spells(sig: np.ndarray, days: np.ndarray) -> list[dict]:
    """
    Contiguous True runs in the rolling significance series.
    Duration in calendar days = last_day - first_day + step approximation via day indices.
    Also count number of consecutive True windows.
    """
    spells = []
    i = 0
    n = len(sig)
    while i < n:
        if not sig[i]:
            i += 1
            continue
        start = i
        while i < n and sig[i]:
            i += 1
        end = i - 1  # inclusive
        # Right-censored if spell runs to end of sample (we don't know when it dies)
        censored = end == n - 1
        duration_windows = end - start + 1
        duration_days = int(days[end] - days[start]) + 1  # inclusive span of lookback-end days
        spells.append(
            {
                "start_day": int(days[start]),
                "end_day": int(days[end]),
                "duration_windows": duration_windows,
                "duration_days": duration_days,
                "censored": censored,
            }
        )
    return spells


def survival_curve(durations: np.ndarray, max_d: int | None = None) -> pd.DataFrame:
    """P(still on at lag k | turned on), using completed spell lengths (uncensored)."""
    if len(durations) == 0:
        return pd.DataFrame(columns=["lag_days", "survival"])
    max_d = max_d or int(durations.max())
    lags = np.arange(0, max_d + 1)
    surv = np.array([(durations >= k).mean() for k in lags], dtype=float)
    return pd.DataFrame({"lag_days": lags, "survival": surv})


def conditional_persistence(sig: np.ndarray, horizons: list[int]) -> dict:
    """Given ON at t, P(ON at t+h) for h in horizons (in windows)."""
    out = {}
    on_idx = np.where(sig)[0]
    for h in horizons:
        ok = on_idx[on_idx + h < len(sig)]
        if len(ok) == 0:
            out[h] = np.nan
        else:
            out[h] = float(sig[ok + h].mean())
    return out


def run_pair(prices: pd.DataFrame, a: str, b: str, step: int) -> tuple[pd.DataFrame, list[dict], dict]:
    series = rolling_sig_series(prices[a].values, prices[b].values, LOOKBACK, step)
    spells = extract_spells(series["sig"].values, series["day"].values)
    # horizons in windows -> convert label to approx days
    horizons_w = [1, 5, 10, 20, 30, 50]
    cond = conditional_persistence(series["sig"].values, horizons_w)
    meta = {
        "pair": f"{a}-{b}",
        "step": step,
        "n_windows": len(series),
        "pct_on": 100 * series["sig"].mean(),
        "n_spells": len(spells),
        "n_completed": sum(1 for s in spells if not s["censored"]),
    }
    completed = [s for s in spells if not s["censored"]]
    if completed:
        durs = np.array([s["duration_days"] for s in completed])
        meta.update(
            {
                "median_spell_days": float(np.median(durs)),
                "mean_spell_days": float(np.mean(durs)),
                "p25_spell_days": float(np.percentile(durs, 25)),
                "p75_spell_days": float(np.percentile(durs, 75)),
                "max_spell_days": float(np.max(durs)),
                "frac_spells_ge_10d": float((durs >= 10).mean()),
                "frac_spells_ge_25d": float((durs >= 25).mean()),
                "frac_spells_ge_50d": float((durs >= 50).mean()),
            }
        )
    else:
        meta.update(
            {
                "median_spell_days": np.nan,
                "mean_spell_days": np.nan,
                "p25_spell_days": np.nan,
                "p75_spell_days": np.nan,
                "max_spell_days": np.nan,
                "frac_spells_ge_10d": np.nan,
                "frac_spells_ge_25d": np.nan,
                "frac_spells_ge_50d": np.nan,
            }
        )
    for h in horizons_w:
        meta[f"p_still_on_{h * step}d"] = cond[h]
    return series, spells, meta


def plot_survival(all_spells: pd.DataFrame, out_dir: Path) -> None:
    completed = all_spells[~all_spells["censored"]]
    if completed.empty:
        return
    durs = completed["duration_days"].values
    surv = survival_curve(durs, max_d=min(100, int(durs.max())))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(surv["lag_days"], surv["survival"] * 100, color="#4C72B0", lw=2)
    axes[0].fill_between(surv["lag_days"], 0, surv["survival"] * 100, color="#4C72B0", alpha=0.2)
    axes[0].axhline(50, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Days after identification (spell still ON)")
    axes[0].set_ylabel("Survival (%)")
    axes[0].set_title(
        f"Cointegration spell survival\n"
        f"(n={len(durs)} completed spells across top pairs, {LOOKBACK}d lookback)"
    )
    axes[0].set_ylim(0, 105)

    axes[1].hist(durs, bins=np.arange(0.5, min(durs.max(), 80) + 1.5, 1), color="#55A868", edgecolor="white")
    axes[1].axvline(np.median(durs), color="red", ls="--", lw=1.5, label=f"median={np.median(durs):.0f}d")
    axes[1].axvline(np.mean(durs), color="orange", ls="--", lw=1.5, label=f"mean={np.mean(durs):.1f}d")
    axes[1].set_xlabel("Spell duration (days)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of ON spell lengths")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "spell_survival.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_conditional(summary: pd.DataFrame, out_dir: Path) -> None:
    # Average P(still on) across pairs
    cols = [c for c in summary.columns if c.startswith("p_still_on_")]
    if not cols:
        return
    lags = [int(c.replace("p_still_on_", "").replace("d", "")) for c in cols]
    means = [summary[c].mean() * 100 for c in cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, means, "o-", color="#C44E52", lw=2, markersize=7)
    for x, y in zip(lags, means):
        ax.text(x, y + 2, f"{y:.0f}%", ha="center", fontsize=9)
    ax.axhline(summary["pct_on"].mean(), color="gray", ls="--", lw=1, label=f"unconditional ON rate ({summary['pct_on'].mean():.0f}%)")
    ax.set_xlabel("Horizon after an ON day (calendar days ≈ windows×step)")
    ax.set_ylabel("P(still ON | ON today) %")
    ax.set_title("Conditional persistence of rolling cointegration")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    path = out_dir / "conditional_persistence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_example_spells(series: pd.DataFrame, spells: list[dict], pair: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series["day"], series["strength"], color="#4C72B0", lw=1.2)
    ax.axhline(-np.log10(SIGNIFICANCE), color="red", ls="--", lw=1, label="p=0.05")
    for s in spells:
        ax.axvspan(s["start_day"], s["end_day"], color="#55A868", alpha=0.25)
        ax.text(
            (s["start_day"] + s["end_day"]) / 2,
            ax.get_ylim()[1] if False else series["strength"].max() * 0.95,
            f"{s['duration_days']}d",
            ha="center",
            fontsize=7,
            color="#1a5c1a",
        )
    ax.set_title(f"{pair}: ON spells (green) after identification | {LOOKBACK}d rolling lookback")
    ax.set_xlabel("Day (end of lookback)")
    ax.set_ylabel("Strength (-log10 p)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = out_dir / f"spells_{pair.replace('-', '_')}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days | lookback={LOOKBACK}, daily roll (step=1)")
    print("Question: after cointegration turns ON, how long until it disappears?\n")

    summary_rows = []
    all_spell_rows = []

    for a, b in TOP_PAIRS:
        series, spells, meta = run_pair(prices, a, b, step=1)
        summary_rows.append(meta)
        for s in spells:
            all_spell_rows.append({"pair": f"{a}-{b}", **s})
        plot_example_spells(series, spells, f"{a}-{b}", OUTPUT_DIR)
        series.to_csv(OUTPUT_DIR / f"series_{a}_{b}.csv", index=False)
        print(
            f"{a}-{b}: ON {meta['pct_on']:.0f}% of days | "
            f"{meta['n_spells']} spells | median duration={meta['median_spell_days']:.0f}d "
            f"(p25={meta['p25_spell_days']:.0f}, p75={meta['p75_spell_days']:.0f}) | "
            f"P(on|+1d)={meta.get('p_still_on_1d', float('nan'))*100:.0f}% "
            f"P(on|+10d)={meta.get('p_still_on_10d', float('nan'))*100:.0f}% "
            f"P(on|+25d)={meta.get('p_still_on_25d', float('nan'))*100:.0f}%"
        )

    summary = pd.DataFrame(summary_rows)
    spells_df = pd.DataFrame(all_spell_rows)
    summary.to_csv(OUTPUT_DIR / "pair_spell_summary.csv", index=False)
    spells_df.to_csv(OUTPUT_DIR / "all_spells.csv", index=False)

    plot_survival(spells_df, OUTPUT_DIR)
    plot_conditional(summary, OUTPUT_DIR)

    completed = spells_df[~spells_df["censored"]]
    print("\n=== Aggregate (completed spells) ===")
    if len(completed):
        d = completed["duration_days"]
        print(f"n spells: {len(completed)}")
        print(f"median duration: {d.median():.0f} days")
        print(f"mean duration:   {d.mean():.1f} days")
        print(f"% lasting >=10d:  {(d >= 10).mean()*100:.0f}%")
        print(f"% lasting >=25d:  {(d >= 25).mean()*100:.0f}%")
        print(f"% lasting >=50d:  {(d >= 50).mean()*100:.0f}%")

    print("\n=== Mean conditional persistence across pairs ===")
    for col in [c for c in summary.columns if c.startswith("p_still_on_")]:
        print(f"  {col}: {summary[col].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
