#!/usr/bin/env python
"""
Re-identify cointegrated pairs in 100-day and 250-day windows,
then run Gregory-Hansen + Hansen Lc *within each window*.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from research.cointegration.cointegration_output.gregory_hansen_lc import (
    GH_CRIT,
    SIGNIFICANCE,
    classify,
    gregory_hansen,
    hansen_lc,
    load_prices,
)

OUTPUT_DIR = Path("cointegration_output/gh_lc_by_window")
WINDOW_SIZES = [100, 250]
MIN_WINDOW = 80  # skip tiny leftover blocks
# Cap GH cost: test all EG-significant pairs, but only GH model C/S (+ Lc)
GH_MODEL = "C/S"


def split_blocks(df: pd.DataFrame, block_size: int) -> list[tuple[int, int, pd.DataFrame]]:
    blocks = []
    n = len(df)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        if end - start < MIN_WINDOW:
            continue
        blocks.append((start, end, df.iloc[start:end].copy()))
    return blocks


def identify_eg_pairs(block: pd.DataFrame) -> list[dict]:
    """All pairs with Engle-Granger p < 0.05 inside this window."""
    tickers = block.columns.tolist()
    hits = []
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            y, x = block[a].values, block[b].values
            if np.std(y) < 1e-12 or np.std(x) < 1e-12:
                continue
            try:
                _, pval, _ = coint(y, x)
            except Exception:
                continue
            if pval < SIGNIFICANCE:
                hits.append({"ticker_a": a, "ticker_b": b, "eg_pval": float(pval)})
    hits.sort(key=lambda d: d["eg_pval"])
    return hits


def analyze_pair_in_window(y: np.ndarray, x: np.ndarray, eg_pval: float) -> dict:
    gh = gregory_hansen(y, x, model=GH_MODEL)
    lc = hansen_lc(y, x)
    eg_sig = eg_pval < SIGNIFICANCE
    label = classify(eg_sig, gh["reject_5"], lc["reject_5"])
    # refine: if GH sig and Lc unstable and not eg... already break_coint
    # if eg and gh and lc unstable -> unstable_coint from classify
    return {
        "eg_pval": eg_pval,
        "eg_sig": eg_sig,
        "gh_adf_star": gh["adf_star"],
        "gh_crit_5": gh["crit_5"],
        "gh_sig": gh["reject_5"],
        "gh_break_frac": gh["break_frac"],
        "gh_break_idx": gh["break_idx"],
        "lc": lc["Lc"],
        "lc_crit_5": lc["crit_5"],
        "lc_unstable": lc["reject_5"],
        "classification": label,
    }


def run_window_size(prices: pd.DataFrame, block_size: int) -> pd.DataFrame:
    blocks = split_blocks(prices, block_size)
    print(f"\n{'=' * 60}")
    print(f"{block_size}-DAY WINDOWS: {len(blocks)} blocks")
    print(f"{'=' * 60}")

    rows = []
    for bi, (start, end, block) in enumerate(blocks):
        print(f"\nBlock {bi + 1}: days {start + 1}-{end} ({end - start} days)")
        hits = identify_eg_pairs(block)
        print(f"  EG-significant pairs: {len(hits)}")

        for h in hits:
            a, b = h["ticker_a"], h["ticker_b"]
            y, x = block[a].values.astype(float), block[b].values.astype(float)
            res = analyze_pair_in_window(y, x, h["eg_pval"])
            break_day = None
            if res["gh_break_idx"] is not None:
                break_day = start + res["gh_break_idx"] + 1  # calendar day
            rows.append(
                {
                    "window_size": block_size,
                    "block": bi + 1,
                    "day_start": start + 1,
                    "day_end": end,
                    "n_days": end - start,
                    "pair": f"{a}-{b}",
                    "ticker_a": a,
                    "ticker_b": b,
                    **{k: v for k, v in res.items() if k != "gh_break_idx"},
                    "gh_break_day": break_day,
                }
            )

        if hits:
            block_rows = [r for r in rows if r["block"] == bi + 1 and r["window_size"] == block_size]
            vc = pd.Series([r["classification"] for r in block_rows]).value_counts()
            print("  Classifications:", dict(vc))
            print(
                f"  Lc unstable: {np.mean([r['lc_unstable'] for r in block_rows])*100:.0f}% | "
                f"GH sig: {np.mean([r['gh_sig'] for r in block_rows])*100:.0f}%"
            )

    return pd.DataFrame(rows)


def plot_class_by_block(df: pd.DataFrame, block_size: int, out_dir: Path) -> None:
    sub = df[df["window_size"] == block_size]
    if sub.empty:
        return

    blocks = sorted(sub["block"].unique())
    classes = ["stable_coint", "break_coint", "unstable_coint", "break_and_unstable", "no_coint", "mixed"]
    colors = {
        "stable_coint": "#55A868",
        "break_coint": "#4C72B0",
        "unstable_coint": "#C44E52",
        "break_and_unstable": "#8172B2",
        "no_coint": "#999999",
        "mixed": "#CCB974",
    }

    # Stacked bar: counts per block
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bottom = np.zeros(len(blocks))
    x = np.arange(len(blocks))
    for cls in classes:
        heights = []
        for b in blocks:
            heights.append(((sub["block"] == b) & (sub["classification"] == cls)).sum())
        heights = np.array(heights, dtype=float)
        if heights.sum() == 0:
            continue
        axes[0].bar(x, heights, bottom=bottom, label=cls, color=colors[cls])
        bottom += heights
    axes[0].set_xticks(x)
    labels = []
    for b in blocks:
        row = sub[sub["block"] == b].iloc[0]
        labels.append(f"B{b}\nd{row['day_start']}-{row['day_end']}")
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("# EG-identified pairs")
    axes[0].set_title(f"{block_size}d windows: GH/Lc classification of EG hits")
    axes[0].legend(fontsize=7, loc="upper right")

    # Overall pie-like bars for this window size
    vc = sub["classification"].value_counts().reindex(classes, fill_value=0)
    axes[1].barh(vc.index, vc.values, color=[colors[c] for c in vc.index])
    axes[1].set_xlabel("# pair-windows")
    total = len(sub)
    axes[1].set_title(f"{block_size}d overall (n={total} EG hits)")
    for i, (cls, val) in enumerate(vc.items()):
        if val > 0:
            axes[1].text(val + 0.5, i, f"{val} ({100*val/total:.0f}%)", va="center", fontsize=8)

    fig.suptitle(
        f"Re-identified pairs → Gregory-Hansen ({GH_MODEL}) + Hansen Lc | {block_size}-day windows",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / f"classification_{block_size}d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side 100 vs 250 classification shares."""
    classes = ["stable_coint", "unstable_coint", "break_coint", "break_and_unstable", "no_coint", "mixed"]
    colors = {
        "stable_coint": "#55A868",
        "break_coint": "#4C72B0",
        "unstable_coint": "#C44E52",
        "break_and_unstable": "#8172B2",
        "no_coint": "#999999",
        "mixed": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(2)
    width = 0.12
    for i, cls in enumerate(classes):
        shares = []
        for ws in WINDOW_SIZES:
            sub = df[df["window_size"] == ws]
            shares.append(100 * (sub["classification"] == cls).mean() if len(sub) else 0)
        ax.bar(x + (i - 2.5) * width, shares, width, label=cls, color=colors[cls])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ws}d windows\n(n={len(df[df.window_size==ws])})" for ws in WINDOW_SIZES])
    ax.set_ylabel("% of EG-identified pair-windows")
    ax.set_title("What EG 'hits' really are: 100d vs 250d windows")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    path = out_dir / "classification_100_vs_250.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_lc_gh_scatter(df: pd.DataFrame, block_size: int, out_dir: Path) -> None:
    sub = df[df["window_size"] == block_size]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(sub["lc_unstable"], "#C44E52", "#55A868")
    ax.scatter(sub["eg_pval"], sub["gh_adf_star"], c=colors, s=18, alpha=0.5)
    ax.axvline(0.05, color="gray", ls="--", lw=1)
    ax.axhline(GH_CRIT[GH_MODEL]["5%"], color="gray", ls="--", lw=1)
    ax.set_xlabel("EG p-value (identification)")
    ax.set_ylabel(f"GH ADF* ({GH_MODEL})")
    ax.set_title(
        f"{block_size}d: EG hits — green=Lc stable, red=Lc unstable\n"
        f"Below dashed line = GH finds break-coint"
    )
    plt.tight_layout()
    path = out_dir / f"eg_vs_gh_scatter_{block_size}d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments")
    print(f"Identify EG pairs (p<{SIGNIFICANCE}) per window, then GH[{GH_MODEL}] + Hansen Lc\n")

    frames = []
    for ws in WINDOW_SIZES:
        df_ws = run_window_size(prices, ws)
        frames.append(df_ws)
        df_ws.to_csv(OUTPUT_DIR / f"gh_lc_pairs_{ws}d.csv", index=False)
        plot_class_by_block(df_ws, ws, OUTPUT_DIR)
        plot_lc_gh_scatter(df_ws, ws, OUTPUT_DIR)

        print(f"\n=== {ws}d SUMMARY ===")
        print(f"Total EG hits: {len(df_ws)}")
        print(df_ws["classification"].value_counts().to_string())
        print(
            f"Lc unstable among EG hits: {df_ws['lc_unstable'].mean()*100:.1f}% | "
            f"GH significant: {df_ws['gh_sig'].mean()*100:.1f}% | "
            f"stable_coint: {(df_ws['classification']=='stable_coint').mean()*100:.1f}%"
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "gh_lc_pairs_all.csv", index=False)
    plot_comparison(df, OUTPUT_DIR)

    # Top stable pairs list
    stable = df[df["classification"] == "stable_coint"].sort_values(["window_size", "eg_pval"])
    stable.to_csv(OUTPUT_DIR / "stable_coint_pairs.csv", index=False)
    print(f"\nStable cointegrated pair-windows: {len(stable)}")
    if len(stable):
        print(stable[["window_size", "block", "day_start", "day_end", "pair", "eg_pval", "gh_adf_star", "lc"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
