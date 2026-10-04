#!/usr/bin/env python
"""
Full-sample (all days) Engle-Granger cointegration across every pair.
Score by -log10(p); plot highest-scoring pairs (prices + residual).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

OUTPUT_DIR = Path("cointegration_output/full_sample_750")
SIGNIFICANCE = 0.05
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
    return float(hl) if 0.25 < hl < 500 else np.nan


def score_all_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    tickers = prices.columns.tolist()
    n = len(tickers)
    rows = []
    total = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = tickers[i], tickers[j]
            y = prices[a].values.astype(float)
            x = prices[b].values.astype(float)
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  EG progress: {done}/{total}")

            if np.std(y) < 1e-12 or np.std(x) < 1e-12:
                continue

            _, pval, _ = coint(y, x)
            e, mu, beta = ols_spread(y, x)
            scale = max(float(np.mean(np.abs(y))), 1e-8)
            resid_cv = float(np.std(e) / scale)
            hl = half_life(e)

            ya = y / y[0]
            yb = x / x[0]
            level_corr = float(np.corrcoef(ya, yb)[0, 1])
            norm_mad = float(np.mean(np.abs(ya - yb)))

            rows.append(
                {
                    "pair": f"{a}-{b}",
                    "ticker_a": a,
                    "ticker_b": b,
                    "pval": float(pval),
                    "strength": float(-np.log10(max(pval, 1e-16))),
                    "significant": pval < SIGNIFICANCE,
                    "mu": mu,
                    "beta": beta,
                    "resid_cv": resid_cv,
                    "half_life": hl,
                    "level_corr": level_corr,
                    "norm_mad": norm_mad,
                }
            )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("strength", ascending=False).reset_index(drop=True)
    return ranking


def plot_top_pairs(prices: pd.DataFrame, ranking: pd.DataFrame) -> None:
    top = ranking.head(TOP_N).copy()
    n = len(top)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle(
        f"Full-sample EG cointegration (T={len(prices)} days) — top {n} by -log10(p)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for k, row in enumerate(top.itertuples()):
        ax = axes[k // ncols][k % ncols]
        a, b = row.ticker_a, row.ticker_b
        ya = prices[a].values / prices[a].iloc[0]
        yb = prices[b].values / prices[b].iloc[0]
        ax.plot(ya, lw=1.0, label=a)
        ax.plot(yb, lw=1.0, label=b)
        ax.set_title(
            f"{row.pair}  p={row.pval:.2e}\n"
            f"str={row.strength:.2f}  HL={row.half_life:.0f}d  corr={row.level_corr:.2f}",
            fontsize=8,
        )
        ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "top_full_sample_prices_grid.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

    # Stack: normalized prices | residual
    fig2, axes2 = plt.subplots(n, 2, figsize=(12, 2.1 * n), sharex=True)
    if n == 1:
        axes2 = np.array([axes2])
    fig2.suptitle(
        "Full-sample top pairs — normalized prices (left) vs EG residual (right)",
        fontsize=13,
        fontweight="bold",
    )

    days = np.arange(1, len(prices) + 1)
    for k, row in enumerate(top.itertuples()):
        a, b = row.ticker_a, row.ticker_b
        y = prices[a].values.astype(float)
        x = prices[b].values.astype(float)
        e, mu, beta = ols_spread(y, x)

        axp, axr = axes2[k, 0], axes2[k, 1]
        axp.plot(days, y / y[0], lw=1.0, label=a)
        axp.plot(days, x / x[0], lw=1.0, label=b)
        axp.set_ylabel(row.pair, fontsize=8)
        axp.legend(fontsize=6, loc="best")
        axp.grid(True, alpha=0.25)
        axp.set_title(
            f"p={row.pval:.2e}  beta={beta:.3f}  corr={row.level_corr:.2f}",
            fontsize=8,
            loc="left",
        )

        axr.plot(days, e, lw=0.9, color="0.25")
        axr.axhline(0, color="C3", lw=0.8, alpha=0.7)
        axr.grid(True, alpha=0.25)
        axr.set_title(
            f"resid_cv={row.resid_cv:.3f}  HL={row.half_life:.1f}d",
            fontsize=8,
            loc="left",
        )

    axes2[-1, 0].set_xlabel("day")
    axes2[-1, 1].set_xlabel("day")
    fig2.tight_layout(rect=[0, 0, 1, 0.99])
    path2 = OUTPUT_DIR / "top_full_sample_prices_and_residual.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {path2}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} instruments")
    print("Running full-sample Engle-Granger on all pairs...")

    ranking = score_all_pairs(prices)
    ranking.to_csv(OUTPUT_DIR / "full_sample_ranking.csv", index=False)

    n_sig = int(ranking["significant"].sum())
    print(f"Significant pairs (p<{SIGNIFICANCE}): {n_sig} / {len(ranking)}")
    print("\nTop 15 by strength (-log10(p)):")
    print(
        ranking.head(15)[
            ["pair", "pval", "strength", "half_life", "resid_cv", "level_corr", "norm_mad", "beta"]
        ].to_string(index=False)
    )

    plot_top_pairs(prices, ranking)
    ranking.head(TOP_N).to_csv(OUTPUT_DIR / "top_full_sample_pairs.csv", index=False)
    print(f"\nOutputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
