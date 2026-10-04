#!/usr/bin/env python
"""
Plot prices for:
  1) Top 40 pairs by MAX Engle-Granger strength in any 100d block
  2) All STEADY pairs (significant in >= 3 of the 100d blocks)

Union of both sets. Shade significant windows; gold = strongest window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOCK_DIR = Path("cointegration_output/block_100")
OUTPUT_DIR = Path("cointegration_output/top40_max_coint_prices")
SIGNIFICANCE = 0.05
TOP_N_PEAK = 40
MIN_SIG_BLOCKS_STEADY = 3  # "steady" = cointegrated in at least this many 100d blocks


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def load_block_meta() -> pd.DataFrame:
    return pd.read_csv(BLOCK_DIR / "summary.csv")


def collect_pair_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = load_block_meta()
    pval_files = sorted(BLOCK_DIR.glob("pvalues_block_*.csv"))

    records = []
    for f in pval_files:
        block_num = int(f.stem.split("_")[-1])
        row = meta[meta["block"] == block_num].iloc[0]
        day_start, day_end = int(row["day_start"]), int(row["day_end"])
        pmat = pd.read_csv(f, index_col=0)
        tickers = pmat.index.tolist()
        for i, a in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                b = tickers[j]
                p = float(pmat.loc[a, b])
                strength = -np.log10(max(p, 1e-16))
                records.append(
                    {
                        "pair": f"{a}-{b}",
                        "ticker_a": a,
                        "ticker_b": b,
                        "block": block_num,
                        "day_start": day_start,
                        "day_end": day_end,
                        "pval": p,
                        "strength": strength,
                        "significant": p < SIGNIFICANCE,
                    }
                )

    detail = pd.DataFrame(records)
    ranking = detail.groupby(["pair", "ticker_a", "ticker_b"], as_index=False).agg(
        max_strength=("strength", "max"),
        min_pval=("pval", "min"),
        n_sig_blocks=("significant", "sum"),
        mean_strength=("strength", "mean"),
        best_block=("strength", "idxmax"),
    )
    best_rows = detail.loc[ranking["best_block"]].reset_index(drop=True)
    ranking["best_block"] = best_rows["block"].values
    ranking["best_day_start"] = best_rows["day_start"].values
    ranking["best_day_end"] = best_rows["day_end"].values
    ranking["best_pval"] = best_rows["pval"].values

    # mean strength only on significant blocks
    sig_mean = (
        detail[detail["significant"]]
        .groupby("pair")["strength"]
        .mean()
        .rename("mean_sig_strength")
    )
    ranking = ranking.merge(sig_mean, on="pair", how="left")
    ranking["mean_sig_strength"] = ranking["mean_sig_strength"].fillna(0.0)

    ranking = ranking.sort_values("max_strength", ascending=False).reset_index(drop=True)
    return ranking, detail


def select_peak_and_steady(ranking: pd.DataFrame) -> pd.DataFrame:
    """Union of top peak pairs and steady pairs."""
    peak = ranking.head(TOP_N_PEAK).copy()
    peak["in_peak_top40"] = True

    steady = ranking[ranking["n_sig_blocks"] >= MIN_SIG_BLOCKS_STEADY].copy()
    steady = steady.sort_values(
        ["n_sig_blocks", "mean_sig_strength", "max_strength"],
        ascending=[False, False, False],
    )
    steady["in_steady"] = True

    peak_pairs = set(peak["pair"])
    steady_pairs = set(steady["pair"])

    combined = ranking[ranking["pair"].isin(peak_pairs | steady_pairs)].copy()
    combined["in_peak_top40"] = combined["pair"].isin(peak_pairs)
    combined["in_steady"] = combined["pair"].isin(steady_pairs)

    def category(r):
        if r["in_peak_top40"] and r["in_steady"]:
            return "peak+steady"
        if r["in_steady"]:
            return "steady"
        return "peak"

    combined["category"] = combined.apply(category, axis=1)

    # Order: peak+steady first, then peak-only by max_strength, then steady-only by n_sig
    combined["_ord"] = combined["category"].map({"peak+steady": 0, "peak": 1, "steady": 2})
    combined = combined.sort_values(
        ["_ord", "n_sig_blocks", "max_strength"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    combined = combined.drop(columns=["_ord"])
    return combined, peak, steady


def plot_pair_stack(prices: pd.DataFrame, selected: pd.DataFrame, detail: pd.DataFrame, path: Path) -> None:
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    t = np.arange(1, len(prices) + 1)
    cat_color = {"peak": "#4C72B0", "steady": "#2ca02c", "peak+steady": "#8172B2"}

    for ax, (_, row) in zip(axes, selected.iterrows()):
        a, b = row["ticker_a"], row["ticker_b"]
        ax.plot(t, prices[a].values / prices[a].iloc[0], lw=1.1, color="#4C72B0", label=a)
        ax.plot(t, prices[b].values / prices[b].iloc[0], lw=1.1, color="#C44E52", label=b)

        sig = detail[(detail["pair"] == row["pair"]) & (detail["significant"])]
        for _, s in sig.iterrows():
            ax.axvspan(s["day_start"], s["day_end"], color="#2ca02c", alpha=0.18)
        ax.axvspan(row["best_day_start"], row["best_day_end"], color="#FFD700", alpha=0.28)

        cat = row["category"]
        ax.set_title(
            f"[{cat}] {row['pair']}  max_str={row['max_strength']:.2f}  "
            f"sig_blocks={int(row['n_sig_blocks'])}  "
            f"best=B{int(row['best_block'])} d{int(row['best_day_start'])}-{int(row['best_day_end'])}",
            fontsize=9,
            loc="left",
            color=cat_color.get(cat, "#333"),
        )
        ax.set_ylabel("P/P0", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Day")
    fig.suptitle(
        f"Peak top-{TOP_N_PEAK} (max strength) + steady pairs (sig in >={MIN_SIG_BLOCKS_STEADY} × 100d blocks)\n"
        "Green shade = significant window (p<0.05) | Gold = strongest single window",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_pair_grid(prices: pd.DataFrame, selected: pd.DataFrame, detail: pd.DataFrame, path: Path) -> None:
    n = len(selected)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(1, len(prices) + 1)

    for ax, (_, row) in zip(axes, selected.iterrows()):
        a, b = row["ticker_a"], row["ticker_b"]
        ax.plot(t, prices[a].values / prices[a].iloc[0], lw=1.0, color="#4C72B0", label=a)
        ax.plot(t, prices[b].values / prices[b].iloc[0], lw=1.0, color="#C44E52", label=b)
        sig = detail[(detail["pair"] == row["pair"]) & (detail["significant"])]
        labeled = False
        for _, s in sig.iterrows():
            ax.axvspan(s["day_start"], s["day_end"], color="#2ca02c", alpha=0.18, label="sig window" if not labeled else None)
            labeled = True
        ax.axvspan(row["best_day_start"], row["best_day_end"], color="#FFD700", alpha=0.25, label="best window")
        ax.set_title(
            f"[{row['category']}] {row['pair']}\n"
            f"max={row['max_strength']:.2f} | {int(row['n_sig_blocks'])} sig blocks",
            fontsize=7.5,
        )
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="upper left")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Peak top-{TOP_N_PEAK} + steady (≥{MIN_SIG_BLOCKS_STEADY} sig blocks) | n={n} pairs\n"
        "Green = p<0.05 windows | Gold = max-strength window",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    ranking, detail = collect_pair_scores()
    detail.to_csv(OUTPUT_DIR / "all_pairs_block_scores.csv", index=False)

    selected, peak, steady = select_peak_and_steady(ranking)
    selected.to_csv(OUTPUT_DIR / "peak_and_steady_pairs.csv", index=False)
    steady.to_csv(OUTPUT_DIR / "steady_pairs.csv", index=False)

    print(f"Peak top-{TOP_N_PEAK}: {len(peak)}")
    print(f"Steady (n_sig>={MIN_SIG_BLOCKS_STEADY}): {len(steady)}")
    print(f"Union: {len(selected)}")
    print(f"  peak+steady overlap: {(selected['category']=='peak+steady').sum()}")
    print(f"  peak only: {(selected['category']=='peak').sum()}")
    print(f"  steady only (includes EORC-NGTE etc.): {(selected['category']=='steady').sum()}")
    print("\nSteady pairs:")
    print(
        steady[["pair", "n_sig_blocks", "max_strength", "mean_sig_strength", "min_pval"]].to_string(index=False)
    )

    plot_pair_stack(
        prices,
        selected,
        detail,
        OUTPUT_DIR / "peak_and_steady_prices_stack.png",
    )
    plot_pair_grid(
        prices,
        selected,
        detail,
        OUTPUT_DIR / "peak_and_steady_prices_grid.png",
    )


if __name__ == "__main__":
    main()
