#!/usr/bin/env python
"""Split prices into day blocks and compute pairwise cointegration matrices."""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

SIGNIFICANCE = 0.05
OUTPUT_ROOT = Path("cointegration_output")


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def split_blocks(df: pd.DataFrame, block_size: int) -> list[tuple[int, int, pd.DataFrame]]:
    blocks = []
    n_days = len(df)
    for start in range(0, n_days, block_size):
        end = min(start + block_size, n_days)
        blocks.append((start, end, df.iloc[start:end].copy()))
    return blocks


def cointegration_matrix(block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Engle-Granger cointegration: p-values and binary significance."""
    tickers = block.columns.tolist()
    n = len(tickers)
    pvals = np.ones((n, n))
    sig = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            y = block.iloc[:, i].values
            x = block.iloc[:, j].values
            if np.std(y) < 1e-12 or np.std(x) < 1e-12:
                continue
            _, pvalue, _ = coint(y, x)
            pvals[i, j] = pvalue
            pvals[j, i] = pvalue
            is_sig = pvalue < SIGNIFICANCE
            sig[i, j] = is_sig
            sig[j, i] = is_sig

    pval_df = pd.DataFrame(pvals, index=tickers, columns=tickers)
    sig_df = pd.DataFrame(sig, index=tickers, columns=tickers)
    return pval_df, sig_df


def plot_block_matrices(
    block_idx: int,
    day_start: int,
    day_end: int,
    pval_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    tickers = pval_df.index.tolist()
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    n_sig = int(np.triu(sig_df.values, k=1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Block {block_idx + 1}: Days {day_start + 1}-{day_end} "
        f"({day_end - day_start} days) | {n_sig}/{n_pairs} pairs cointegrated (p<{SIGNIFICANCE})",
        fontsize=14,
        fontweight="bold",
    )

    log_pvals = -np.log10(pval_df.values.clip(min=1e-16))
    im0 = axes[0].imshow(log_pvals, cmap="viridis", aspect="auto")
    axes[0].set_title("-log10(p-value)")
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_yticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=90, fontsize=6)
    axes[0].set_yticklabels(tickers, fontsize=6)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sig_df.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title(f"Cointegrated (p < {SIGNIFICANCE})")
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_yticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=90, fontsize=6)
    axes[1].set_yticklabels(tickers, fontsize=6)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1])

    plt.tight_layout()
    out_path = out_dir / f"cointegration_block_{block_idx + 1:02d}_days_{day_start + 1}_{day_end}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_overview(sig_matrices: list[tuple[int, int, int, pd.DataFrame]], out_dir: Path, block_size: int) -> None:
    n_blocks = len(sig_matrices)
    cols = min(4, n_blocks)
    rows = (n_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (block_idx, day_start, day_end, sig_df) in zip(axes, sig_matrices):
        n_sig = int(np.triu(sig_df.values, k=1).sum())
        n_pairs = len(sig_df) * (len(sig_df) - 1) // 2
        ax.imshow(sig_df.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"B{block_idx + 1}: d{day_start + 1}-{day_end}\n{n_sig}/{n_pairs} pairs", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_blocks:]:
        ax.axis("off")

    fig.suptitle(
        f"Cointegration Overview — {block_size}-day blocks (p < {SIGNIFICANCE})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "cointegration_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def concatenate_block_pngs(block_pngs: list[Path], out_path: Path, block_size: int) -> None:
    """Stack all per-block PNGs vertically into one image."""
    images = [mpimg.imread(p) for p in block_pngs]
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(18, 8 * n))
    if n == 1:
        axes = [axes]

    for ax, img, png_path in zip(axes, images, block_pngs):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(png_path.stem.replace("_", " "), fontsize=10, loc="left", pad=4)

    fig.suptitle(f"All {block_size}-day cointegration blocks ({n} panels)", fontsize=16, fontweight="bold", y=1.0)
    plt.subplots_adjust(hspace=0.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved concatenated image: {out_path}")


def run_analysis(prices: pd.DataFrame, block_size: int) -> None:
    out_dir = OUTPUT_ROOT / f"block_{block_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = split_blocks(prices, block_size)
    print(f"\n{'=' * 60}")
    print(f"{block_size}-DAY BLOCKS: {len(blocks)} blocks")
    print(f"{'=' * 60}")

    sig_matrices = []
    summary_rows = []
    block_pngs: list[Path] = []

    for block_idx, (day_start, day_end, block_df) in enumerate(blocks):
        n_days = day_end - day_start
        print(f"Block {block_idx + 1}: days {day_start + 1}-{day_end} ({n_days} days)")

        pval_df, sig_df = cointegration_matrix(block_df)
        n_pairs = len(block_df.columns) * (len(block_df.columns) - 1) // 2
        n_sig = int(np.triu(sig_df.values, k=1).sum())

        summary_rows.append(
            {
                "block": block_idx + 1,
                "day_start": day_start + 1,
                "day_end": day_end,
                "n_days": n_days,
                "cointegrated_pairs": n_sig,
                "total_pairs": n_pairs,
                "pct_cointegrated": round(100 * n_sig / n_pairs, 1),
            }
        )

        pval_df.to_csv(out_dir / f"pvalues_block_{block_idx + 1:02d}.csv")
        sig_df.to_csv(out_dir / f"significance_block_{block_idx + 1:02d}.csv")

        png_path = plot_block_matrices(block_idx, day_start, day_end, pval_df, sig_df, out_dir)
        block_pngs.append(png_path)
        sig_matrices.append((block_idx, day_start, day_end, sig_df))

    plot_overview(sig_matrices, out_dir, block_size)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n=== {block_size}-day summary ===")
    print(summary.to_string(index=False))

    concat_path = OUTPUT_ROOT / f"all_blocks_{block_size}_concat.png"
    concatenate_block_pngs(block_pngs, concat_path, block_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cointegration analysis by day-block window")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Block window sizes in days (default: 50 100)",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments")

    for block_size in args.block_sizes:
        run_analysis(prices, block_size)


if __name__ == "__main__":
    main()
