#!/usr/bin/env python
"""
Price paths + return stats + full ACF for the best assets
(ranked by max |ACF| across lags 1..n/4 from lagged_returns_analysis.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

INPUT_DIR = Path("cointegration_output/lagged_returns")
OUTPUT_DIR = INPUT_DIR / "best_assets"
TOP_N = 12


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def asset_stats(price: np.ndarray, returns: np.ndarray, row: pd.Series) -> dict:
    r = returns[np.isfinite(returns)]
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    sharpe = mu / sig * np.sqrt(252) if sig > 1e-12 else np.nan
    downside = r[r < 0]
    sortino = (
        mu / np.std(downside, ddof=1) * np.sqrt(252)
        if len(downside) > 5 and np.std(downside, ddof=1) > 1e-12
        else np.nan
    )
    # max drawdown on price
    peak = np.maximum.accumulate(price)
    dd = (price - peak) / peak
    return {
        "n_days": len(price),
        "total_return": float(price[-1] / price[0] - 1),
        "ann_vol": sig * np.sqrt(252),
        "ann_mean": mu * 252,
        "sharpe": sharpe,
        "sortino": sortino,
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt()),
        "max_dd": float(dd.min()),
        "acf_1": float(row["acf_1"]),
        "best_lag": int(row["best_lag"]),
        "best_acf": float(row["best_acf"]),
        "best_pval": float(row["best_pval"]),
        "n_sig_lags": int(row["n_sig_lags"]),
    }


def plot_one_asset(
    ticker: str,
    price: np.ndarray,
    returns: np.ndarray,
    acf: np.ndarray,
    lags: np.ndarray,
    stats: dict,
    path: Path,
) -> None:
    t = np.arange(1, len(price) + 1)
    ci = 1.96 / np.sqrt(len(returns))
    norm = price / price[0]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 0.9, 1.0], hspace=0.35, wspace=0.3)

    # Price
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, norm, color="#4C72B0", lw=1.2)
    ax.set_title(f"{ticker} — normalized price")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price / P0")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    sign = "momentum" if stats["best_acf"] > 0 else "reversal"
    ax.text(
        0.01,
        0.98,
        f"best ACF: lag {stats['best_lag']} = {stats['best_acf']:+.3f} ({sign})",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    # Stats box
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lines = [
        f"Total return: {stats['total_return']:+.1%}",
        f"Ann. mean:    {stats['ann_mean']:+.1%}",
        f"Ann. vol:     {stats['ann_vol']:.1%}",
        f"Sharpe:       {stats['sharpe']:.2f}",
        f"Sortino:      {stats['sortino']:.2f}",
        f"Skew / kurt:  {stats['skew']:+.2f} / {stats['kurt']:.1f}",
        f"Max drawdown: {stats['max_dd']:.1%}",
        "",
        f"ACF(1):       {stats['acf_1']:+.4f}",
        f"Best lag:     {stats['best_lag']}",
        f"Best ACF:     {stats['best_acf']:+.4f}",
        f"Best p-val:   {stats['best_pval']:.4f}",
        f"# sig lags:   {stats['n_sig_lags']}",
    ]
    ax.set_title("Statistics", fontsize=11, pad=8)
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        linespacing=1.35,
    )

    # Returns
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t[1:], returns, color="#55A868", lw=0.6, alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{ticker} — daily returns")
    ax.set_xlabel("Day")
    ax.set_ylabel("Return")

    # Return distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(returns, bins=40, color="#55A868", edgecolor="white", alpha=0.9, density=True)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(np.mean(returns), color="#C44E52", ls="--", lw=1, label="mean")
    ax.set_title("Return distribution")
    ax.set_xlabel("Return")
    ax.legend(fontsize=8)

    # Full ACF
    ax = fig.add_subplot(gs[2, :])
    colors = ["#C44E52" if abs(v) > ci else "#4C72B0" for v in acf]
    ax.bar(lags, acf, width=0.8, color=colors, align="center")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="gray", ls="--", lw=0.8)
    ax.axhline(-ci, color="gray", ls="--", lw=0.8, label=f"±1.96/√n ({ci:.3f})")
    ax.axvline(stats["best_lag"], color="#C44E52", ls=":", lw=1.2, label=f"best lag {stats['best_lag']}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title(f"{ticker} — return ACF (lags 1…{int(lags[-1])}; red bars exceed CI)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Best-asset profile: {ticker}", fontsize=13, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview_grid(
    prices: pd.DataFrame,
    top: pd.DataFrame,
    acf_mat: np.ndarray,
    tickers_order: list[str],
    lags: np.ndarray,
    path: Path,
) -> None:
    n = len(top)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)
    t = np.arange(1, len(prices) + 1)
    ticker_to_i = {tk: i for i, tk in enumerate(tickers_order)}

    for k, row in enumerate(top.itertuples()):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        p = prices[row.ticker].values.astype(float)
        ax.plot(t, p / p[0], color="#4C72B0", lw=1.0)
        ax.set_title(
            f"{row.ticker}  |  best L{int(row.best_lag)}={row.best_acf:+.3f}  |  ACF1={row.acf_1:+.3f}",
            fontsize=9,
        )
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
        # tiny ACF sparkline on twin? keep simple — annotate n_sig
        ax.text(
            0.98,
            0.05,
            f"{int(row.n_sig_lags)} sig lags",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"Top {n} assets by max |return ACF| over lags 1…{int(lags[-1])}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stats_table(top: pd.DataFrame, stats_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(top) + 1.5))
    ax.axis("off")
    headers = [
        "Ticker",
        "Total ret",
        "Ann vol",
        "Sharpe",
        "Max DD",
        "ACF(1)",
        "Best lag",
        "Best ACF",
        "#sig",
    ]
    cell = []
    for row, st in zip(top.itertuples(), stats_rows):
        cell.append(
            [
                row.ticker,
                f"{st['total_return']:+.1%}",
                f"{st['ann_vol']:.1%}",
                f"{st['sharpe']:.2f}",
                f"{st['max_dd']:.1%}",
                f"{st['acf_1']:+.3f}",
                str(st["best_lag"]),
                f"{st['best_acf']:+.3f}",
                str(st["n_sig_lags"]),
            ]
        )
    table = ax.table(cellText=cell, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (i, j), cell_obj in table.get_celld().items():
        if i == 0:
            cell_obj.set_facecolor("#4C72B0")
            cell_obj.set_text_props(color="white", weight="bold")
        elif i % 2 == 0:
            cell_obj.set_facecolor("#f2f2f2")
    ax.set_title(f"Top {len(top)} best assets — summary statistics", pad=12, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    returns = prices.pct_change().iloc[1:]
    acf_df = pd.read_csv(INPUT_DIR / "asset_return_acf.csv")
    acf_mat = np.load(INPUT_DIR / "asset_acf_matrix.npy")
    tickers_order = prices.columns.tolist()
    n_lags = acf_mat.shape[1]
    lags = np.arange(1, n_lags + 1)

    top = acf_df.head(TOP_N).copy()
    print(f"Plotting top {TOP_N} assets by max |ACF|: {', '.join(top['ticker'])}")

    stats_rows = []
    for row in top.itertuples():
        price = prices[row.ticker].values.astype(float)
        ret = returns[row.ticker].values.astype(float)
        st = asset_stats(price, ret, acf_df.loc[acf_df["ticker"] == row.ticker].iloc[0])
        stats_rows.append(st)
        i = tickers_order.index(row.ticker)
        plot_one_asset(
            row.ticker,
            price,
            ret,
            acf_mat[i],
            lags,
            st,
            OUTPUT_DIR / f"profile_{row.ticker}.png",
        )
        print(f"  wrote profile_{row.ticker}.png")

    plot_overview_grid(prices, top, acf_mat, tickers_order, lags, OUTPUT_DIR / "best_assets_price_grid.png")
    plot_stats_table(top, stats_rows, OUTPUT_DIR / "best_assets_stats_table.png")

    # also save stats csv
    out = top[["ticker", "acf_1", "best_lag", "best_acf", "best_pval", "n_sig_lags"]].copy()
    for k in ("total_return", "ann_vol", "sharpe", "sortino", "skew", "kurt", "max_dd", "ann_mean"):
        out[k] = [s[k] for s in stats_rows]
    out.to_csv(OUTPUT_DIR / "best_assets_stats.csv", index=False)

    print(f"\nWrote grid + table + {TOP_N} profiles to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
