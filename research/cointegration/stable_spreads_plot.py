#!/usr/bin/env python
"""
Find pairs with the most stable cointegrating spreads and plot them.

Stability score (higher = more stable):
  - Prefer pairs significant in multiple 100d blocks
  - Low residual volatility (std of e_t / price scale) in those blocks
  - Fast mean reversion (shorter half-life, capped)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOCK_DIR = Path("cointegration_output/block_100")
OUTPUT_DIR = Path("cointegration_output/stable_spreads")
SIGNIFICANCE = 0.05
MIN_SIG_BLOCKS = 2
TOP_N = 24


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def ols_spread(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - mu - beta * x
    return e, float(mu), float(beta)


def half_life(spread: np.ndarray) -> float:
    if len(spread) < 15 or np.std(spread) < 1e-12:
        return np.nan
    lag, cur = spread[:-1], spread[1:]
    Z = np.column_stack([np.ones(len(lag)), lag])
    _, rho = np.linalg.lstsq(Z, cur, rcond=None)[0]
    if not (0 < rho < 1):
        return np.nan
    hl = -np.log(2) / np.log(rho)
    return float(hl) if 0.25 < hl < 200 else np.nan


def block_windows(prices: pd.DataFrame) -> list[tuple[int, int, int]]:
    meta = pd.read_csv(BLOCK_DIR / "summary.csv")
    return [
        (int(r.block), int(r.day_start) - 1, int(r.day_end))  # start 0-index, end exclusive-ish
        for r in meta.itertuples()
    ]


def score_pairs(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(BLOCK_DIR / "summary.csv")
    pval_files = sorted(BLOCK_DIR.glob("pvalues_block_*.csv"))
    rows = []

    for f in pval_files:
        block_num = int(f.stem.split("_")[-1])
        m = meta[meta["block"] == block_num].iloc[0]
        s0, s1 = int(m["day_start"]) - 1, int(m["day_end"])
        block = prices.iloc[s0:s1]
        pmat = pd.read_csv(f, index_col=0)
        tickers = pmat.index.tolist()

        for i, a in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                b = tickers[j]
                p = float(pmat.loc[a, b])
                if p >= SIGNIFICANCE:
                    continue
                y, x = block[a].values.astype(float), block[b].values.astype(float)
                e, mu, beta = ols_spread(y, x)
                scale = max(np.mean(np.abs(y)), 1e-8)
                resid_cv = float(np.std(e) / scale)  # lower = tighter spread
                hl = half_life(e)
                rows.append(
                    {
                        "pair": f"{a}-{b}",
                        "ticker_a": a,
                        "ticker_b": b,
                        "block": block_num,
                        "day_start": s0 + 1,
                        "day_end": s1,
                        "pval": p,
                        "strength": -np.log10(max(p, 1e-16)),
                        "resid_cv": resid_cv,
                        "half_life": hl,
                        "beta": beta,
                        "mu": mu,
                    }
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail

    # Aggregate: require multiple significant blocks
    agg = detail.groupby(["pair", "ticker_a", "ticker_b"], as_index=False).agg(
        n_sig_blocks=("block", "count"),
        mean_resid_cv=("resid_cv", "mean"),
        median_resid_cv=("resid_cv", "median"),
        mean_half_life=("half_life", "median"),
        mean_strength=("strength", "mean"),
        max_strength=("strength", "max"),
        min_pval=("pval", "min"),
    )
    agg = agg[agg["n_sig_blocks"] >= MIN_SIG_BLOCKS].copy()

    # Stability score: tight residual + enough persistence + not glacial half-life
    # hl_penalty: prefer HL in (1, 40); very long HL = wandering spread
    hl = agg["mean_half_life"].fillna(60.0).clip(1, 100)
    hl_factor = 1.0 / (1.0 + (hl / 15.0))  # ~1 when HL short, smaller when long
    # lower resid_cv is better
    tight = 1.0 / (agg["mean_resid_cv"] + 1e-6)
    persistence = agg["n_sig_blocks"] / 8.0
    agg["stability_score"] = tight * hl_factor * (0.5 + 0.5 * persistence)
    agg = agg.sort_values("stability_score", ascending=False).reset_index(drop=True)
    return agg, detail


def full_sample_spread(prices: pd.DataFrame, a: str, b: str) -> tuple[np.ndarray, float, float]:
    y, x = prices[a].values.astype(float), prices[b].values.astype(float)
    return ols_spread(y, x)


def rolling_spread(prices: pd.DataFrame, a: str, b: str, w: int = 100) -> np.ndarray:
    """Causal residual using expanding/rolling hedge — better for 'stable spread' view."""
    y, x = prices[a].values.astype(float), prices[b].values.astype(float)
    n = len(y)
    e = np.full(n, np.nan)
    for i in range(w, n):
        ee, mu, beta = ols_spread(y[i - w : i], x[i - w : i])
        e[i] = y[i] - mu - beta * x[i]
    return e


def plot_stable_spreads(prices: pd.DataFrame, ranking: pd.DataFrame, detail: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top = ranking.head(TOP_N)
    top.to_csv(OUTPUT_DIR / "most_stable_spreads_ranking.csv", index=False)

    n = len(top)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.9 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(1, len(prices) + 1)

    for ax, (_, row) in zip(axes, top.iterrows()):
        a, b = row["ticker_a"], row["ticker_b"]
        # Plot rolling residual (demeaned locally) — shows if spread stays tight
        e = rolling_spread(prices, a, b, w=100)
        ax.plot(t, e, color="#4C72B0", lw=0.9)
        ax.axhline(0, color="gray", lw=0.7)

        sig = detail[detail["pair"] == row["pair"]]
        for _, s in sig.iterrows():
            ax.axvspan(s["day_start"], s["day_end"], color="#2ca02c", alpha=0.2)

        ax.set_title(
            f"{row['pair']}  score={row['stability_score']:.0f}\n"
            f"resid_cv={row['mean_resid_cv']:.3f}  HL={row['mean_half_life']:.1f}d  "
            f"sig={int(row['n_sig_blocks'])} blocks",
            fontsize=8,
        )
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Top {TOP_N} pairs with most stable spreads "
        f"(≥{MIN_SIG_BLOCKS} sig. 100d blocks | tight residual + reasonable half-life)\n"
        "Blue = rolling 100d residual | Green = windows with EG p<0.05",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "most_stable_spreads_grid.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Stacked version with prices + spread for top 12
    top12 = top.head(12)
    fig, axes = plt.subplots(len(top12), 2, figsize=(14, 2.4 * len(top12)), sharex="col")
    for i, (_, row) in enumerate(top12.iterrows()):
        a, b = row["ticker_a"], row["ticker_b"]
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.plot(t, prices[a].values / prices[a].iloc[0], color="#4C72B0", lw=1, label=a)
        ax0.plot(t, prices[b].values / prices[b].iloc[0], color="#C44E52", lw=1, label=b)
        for _, s in detail[detail["pair"] == row["pair"]].iterrows():
            ax0.axvspan(s["day_start"], s["day_end"], color="#2ca02c", alpha=0.18)
            ax1.axvspan(s["day_start"], s["day_end"], color="#2ca02c", alpha=0.18)
        ax0.set_ylabel("P/P0", fontsize=8)
        ax0.legend(fontsize=6, loc="upper left")
        ax0.set_title(f"{row['pair']} prices", fontsize=9, loc="left")

        e = rolling_spread(prices, a, b, w=100)
        ax1.plot(t, e, color="#8172B2", lw=0.9)
        ax1.axhline(0, color="gray", lw=0.7)
        ax1.set_ylabel("spread", fontsize=8)
        ax1.set_title(
            f"rolling residual | cv={row['mean_resid_cv']:.3f} HL={row['mean_half_life']:.1f}d "
            f"sig={int(row['n_sig_blocks'])}",
            fontsize=9,
            loc="left",
        )

    axes[-1, 0].set_xlabel("Day")
    axes[-1, 1].set_xlabel("Day")
    fig.suptitle(
        "Most stable spreads — prices (left) vs rolling residual (right)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path2 = OUTPUT_DIR / "most_stable_spreads_prices_and_residual.png"
    fig.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path2}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    ranking, detail = score_pairs(prices)
    detail.to_csv(OUTPUT_DIR / "stable_spread_block_detail.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "most_stable_spreads_ranking.csv", index=False)

    print(f"Pairs with >={MIN_SIG_BLOCKS} sig blocks scored: {len(ranking)}")
    print("\nTop 15 most stable spreads:")
    print(
        ranking.head(15)[
            ["pair", "stability_score", "mean_resid_cv", "mean_half_life", "n_sig_blocks", "max_strength"]
        ].to_string(index=False)
    )
    plot_stable_spreads(prices, ranking, detail)


if __name__ == "__main__":
    main()
