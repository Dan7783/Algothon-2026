#!/usr/bin/env python
"""Bar charts of pair cointegration persistence and strength across block windows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIGNIFICANCE = 0.05
OUTPUT_ROOT = Path("cointegration_output")
TOP_N = 40


def load_block_pvalues(block_dir: Path) -> list[pd.DataFrame]:
    files = sorted(block_dir.glob("pvalues_block_*.csv"))
    if not files:
        raise FileNotFoundError(f"No p-value CSVs found in {block_dir}")
    return [pd.read_csv(f, index_col=0) for f in files]


def pair_strength_table(pval_blocks: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate pairwise cointegration stats across all windows."""
    tickers = pval_blocks[0].index.tolist()
    n_windows = len(pval_blocks)
    rows = []

    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if j <= i:
                continue
            pvals = np.array([blk.loc[a, b] for blk in pval_blocks], dtype=float)
            strength = -np.log10(np.clip(pvals, 1e-16, 1.0))
            sig_mask = pvals < SIGNIFICANCE
            n_sig = int(sig_mask.sum())

            rows.append(
                {
                    "pair": f"{a}-{b}",
                    "ticker_a": a,
                    "ticker_b": b,
                    "n_windows": n_windows,
                    "n_sig_windows": n_sig,
                    "pct_windows": 100.0 * n_sig / n_windows,
                    "mean_strength": float(strength.mean()),
                    "mean_sig_strength": float(strength[sig_mask].mean()) if n_sig else 0.0,
                    "max_strength": float(strength.max()),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["n_sig_windows", "mean_strength", "mean_sig_strength"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df


def plot_pair_bars(df: pd.DataFrame, block_size: int, n_windows: int, out_path: Path) -> None:
    plot_df = df.head(TOP_N).iloc[::-1]  # highest at top for horizontal bars

    fig, ax = plt.subplots(figsize=(12, max(8, TOP_N * 0.28)))
    colors = plt.cm.viridis(plot_df["n_sig_windows"] / n_windows)
    bars = ax.barh(
        plot_df["pair"],
        plot_df["mean_strength"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )

    for bar, n_sig, pct in zip(
        bars, plot_df["n_sig_windows"], plot_df["pct_windows"], strict=True
    ):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{n_sig}/{n_windows} ({pct:.0f}%)",
            va="center",
            fontsize=7,
            color="#333333",
        )

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=0, vmax=n_windows),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(f"Windows significant (of {n_windows})")

    ax.set_xlabel("Mean cointegration strength (-log10 p-value) across all windows")
    ax.set_ylabel("Pair")
    ax.set_title(
        f"Top {TOP_N} pairs by persistence — {block_size}-day blocks\n"
        f"(ordered by # windows with p < {SIGNIFICANCE}, then mean strength)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, max(plot_df["mean_strength"].max() * 1.25, 0.5))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def run_for_block_size(block_size: int) -> None:
    block_dir = OUTPUT_ROOT / f"block_{block_size}"
    pval_blocks = load_block_pvalues(block_dir)
    n_windows = len(pval_blocks)

    df = pair_strength_table(pval_blocks)
    csv_path = OUTPUT_ROOT / f"pair_persistence_{block_size}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df)} pairs)")

    plot_pair_bars(
        df,
        block_size,
        n_windows,
        OUTPUT_ROOT / f"pair_persistence_bars_{block_size}.png",
    )

    top = df.head(10)
    print(f"\nTop 10 pairs ({block_size}-day blocks):")
    print(
        top[["pair", "n_sig_windows", "pct_windows", "mean_strength"]].to_string(index=False)
    )


def main() -> None:
    for block_size in (50, 100):
        print(f"\n{'=' * 60}")
        print(f"{block_size}-DAY BLOCKS")
        print(f"{'=' * 60}")
        run_for_block_size(block_size)


if __name__ == "__main__":
    main()
