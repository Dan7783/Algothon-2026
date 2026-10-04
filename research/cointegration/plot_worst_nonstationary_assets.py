#!/usr/bin/env python
"""
Stationarity screen + worst assets:
  1) Returns: ADF + KPSS (almost all returns are stationary; flag failures)
  2) Prices: least-stationary price series (high ADF p)
  3) Performance: worst Sharpe / total return

PNG profiles for each set.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("cointegration_output/lagged_returns/worst_nonstationary")
TOP_N = 12
SIGNIFICANCE = 0.05
ROLL = 60


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def safe_adf(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if len(x) < 30 or np.std(x) < 1e-15:
        return np.nan, np.nan, 0
    try:
        stat, pval, usedlag, *_ = adfuller(x, maxlag=None, autolag="AIC")
        return float(stat), float(pval), int(usedlag)
    except Exception:
        return np.nan, np.nan, 0


def safe_kpss(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 30 or np.std(x) < 1e-15:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, *_ = kpss(x, regression="c", nlags="auto")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def score_assets(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        r = returns[col].values.astype(float)
        p = prices[col].values.astype(float)
        adf_r, padf_r, lag_r = safe_adf(r)
        kpss_r, pkpss_r = safe_kpss(r)
        adf_p, padf_p, lag_p = safe_adf(p)
        kpss_p, pkpss_p = safe_kpss(p)

        mu = float(np.mean(r))
        sig = float(np.std(r, ddof=1))
        sharpe = mu / sig * np.sqrt(252) if sig > 1e-12 else np.nan
        peak = np.maximum.accumulate(p)
        max_dd = float(np.min((p - peak) / peak))

        rows.append(
            {
                "ticker": col,
                "adf_ret_stat": adf_r,
                "adf_ret_pval": padf_r,
                "adf_ret_lag": lag_r,
                "kpss_ret_stat": kpss_r,
                "kpss_ret_pval": pkpss_r,
                "adf_price_stat": adf_p,
                "adf_price_pval": padf_p,
                "adf_price_lag": lag_p,
                "kpss_price_pval": pkpss_p,
                "ret_adf_fail": int(np.isfinite(padf_r) and padf_r >= SIGNIFICANCE),
                "ret_kpss_reject": int(np.isfinite(pkpss_r) and pkpss_r < SIGNIFICANCE),
                "price_adf_fail": int(np.isfinite(padf_p) and padf_p >= SIGNIFICANCE),
                "total_return": float(p[-1] / p[0] - 1),
                "ann_vol": sig * np.sqrt(252),
                "sharpe": sharpe,
                "max_dd": max_dd,
                "acf_1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def plot_profile(
    ticker: str,
    price: np.ndarray,
    returns: np.ndarray,
    row: pd.Series,
    path: Path,
    title_prefix: str,
) -> None:
    t = np.arange(1, len(price) + 1)
    norm = price / price[0]
    r_s = pd.Series(returns)
    roll_mu = r_s.rolling(ROLL).mean()
    roll_sig = r_s.rolling(ROLL).std()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 0.95, 1.0], hspace=0.38, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, norm, color="#C44E52", lw=1.2)
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_title(f"{ticker} — normalized price")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price / P0")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lines = [
        f"Total return:     {row['total_return']:+.1%}",
        f"Ann. vol:         {row['ann_vol']:.1%}",
        f"Sharpe:           {row['sharpe']:.2f}",
        f"Max drawdown:     {row['max_dd']:.1%}",
        f"ACF(1) returns:   {row['acf_1']:+.4f}",
        "",
        "— Returns —",
        f"ADF p:            {row['adf_ret_pval']:.2e}",
        f"KPSS p:           {row['kpss_ret_pval']:.4f}",
        f"ADF fail / KPSS:  {int(row['ret_adf_fail'])} / {int(row['ret_kpss_reject'])}",
        "",
        "— Prices —",
        f"ADF p:            {row['adf_price_pval']:.4f}",
        f"KPSS p:           {row['kpss_price_pval']:.4f}",
        f"Price unit-root:  {'YES' if row['price_adf_fail'] else 'no'}",
    ]
    ax.set_title("Statistics", fontsize=11)
    ax.text(0.04, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=9)

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t[1:], returns, color="#4C72B0", lw=0.55, alpha=0.8)
    ax.plot(t[1:], roll_mu, color="#C44E52", lw=1.2, label=f"{ROLL}d rolling mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{ticker} — daily returns")
    ax.set_xlabel("Day")
    ax.set_ylabel("Return")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t[1:], roll_sig, color="#8172B3", lw=1.2)
    ax.set_title(f"{ROLL}d rolling vol")
    ax.set_xlabel("Day")
    ax.set_ylabel("Std")

    ax = fig.add_subplot(gs[2, :2])
    cum = np.cumprod(1 + np.nan_to_num(returns, nan=0.0))
    ax.plot(t[1:], cum, color="#4C72B0", lw=1.2)
    ax.set_title(f"{ticker} — cumulative wealth")
    ax.set_xlabel("Day")
    ax.set_ylabel("Wealth")

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(returns[np.isfinite(returns)], bins=40, color="#C44E52", edgecolor="white", density=True, alpha=0.9)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(np.nanmean(returns), color="#4C72B0", ls="--", label="mean")
    ax.set_title("Return distribution")
    ax.legend(fontsize=8)

    fig.suptitle(f"{title_prefix}: {ticker}", fontsize=13, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(prices: pd.DataFrame, subset: pd.DataFrame, path: Path, title: str, color: str) -> None:
    n = len(subset)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)
    t = np.arange(1, len(prices) + 1)
    for k, row in enumerate(subset.itertuples()):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        p = prices[row.ticker].values.astype(float)
        ax.plot(t, p / p[0], color=color, lw=1.0)
        ax.axhline(1.0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(
            f"{row.ticker}  |  ret {row.total_return:+.0%}  |  sharpe {row.sharpe:.2f}",
            fontsize=9,
        )
        ax.text(
            0.98,
            0.05,
            f"price ADF p={row.adf_price_pval:.2f}  ret ADF p={row.adf_ret_pval:.1e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_table(subset: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 0.45 * len(subset) + 1.6))
    ax.axis("off")
    headers = [
        "Ticker",
        "Ret ADF p",
        "Ret KPSS p",
        "Price ADF p",
        "Total ret",
        "Ann vol",
        "Sharpe",
        "Max DD",
    ]
    cells = [
        [
            r.ticker,
            f"{r.adf_ret_pval:.2e}",
            f"{r.kpss_ret_pval:.3f}",
            f"{r.adf_price_pval:.3f}",
            f"{r.total_return:+.1%}",
            f"{r.ann_vol:.1%}",
            f"{r.sharpe:.2f}",
            f"{r.max_dd:.1%}",
        ]
        for r in subset.itertuples()
    ]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#C44E52")
            cell.set_text_props(color="white", weight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax.set_title(title, pad=12, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview_bars(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    ax = axes[0]
    o = df.sort_values("adf_ret_pval")
    colors = ["#C44E52" if k else "#4C72B0" for k in o["ret_kpss_reject"]]
    ax.barh(o["ticker"], o["adf_ret_pval"], color=colors, height=0.7)
    ax.axvline(SIGNIFICANCE, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("ADF p-value on returns (log)")
    ax.set_title("Return ADF (red=KPSS rejects stationarity)")

    ax = axes[1]
    o = df.sort_values("adf_price_pval")
    colors = ["#C44E52" if f else "#4C72B0" for f in o["price_adf_fail"]]
    ax.barh(o["ticker"], o["adf_price_pval"], color=colors, height=0.7)
    ax.axvline(SIGNIFICANCE, color="k", ls="--", lw=1, label="5%")
    ax.set_xlabel("ADF p-value on prices")
    ax.set_title("Price ADF (red = unit root not rejected)")
    ax.legend(fontsize=8)

    ax = axes[2]
    o = df.sort_values("sharpe")
    colors = ["#C44E52" if s < 0 else "#55A868" for s in o["sharpe"]]
    ax.barh(o["ticker"], o["sharpe"], color=colors, height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sharpe")
    ax.set_title("Performance Sharpe (all assets)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dump_set(
    name: str,
    subset: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    df: pd.DataFrame,
    title_prefix: str,
    color: str,
) -> None:
    subdir = OUTPUT_DIR / name
    subdir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(subdir / "selected.csv", index=False)
    plot_grid(prices, subset, subdir / "price_grid.png", f"{title_prefix} — price grid", color)
    plot_table(subset, subdir / "stats_table.png", title_prefix)
    for row in subset.itertuples():
        full = df.loc[df["ticker"] == row.ticker].iloc[0]
        plot_profile(
            row.ticker,
            prices[row.ticker].values.astype(float),
            returns[row.ticker].values.astype(float),
            full,
            subdir / f"profile_{row.ticker}.png",
            title_prefix,
        )
    print(f"  [{name}] {len(subset)} assets -> {subdir}/")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    returns = prices.pct_change().iloc[1:]

    print("Scoring all assets (ADF/KPSS on returns + prices)...")
    df = score_assets(prices, returns)
    df.to_csv(OUTPUT_DIR / "stationarity_performance_all.csv", index=False)

    n_ret_adf_fail = int(df["ret_adf_fail"].sum())
    n_ret_kpss = int(df["ret_kpss_reject"].sum())
    n_price_fail = int(df["price_adf_fail"].sum())
    print(f"  Returns: ADF fails to reject unit root: {n_ret_adf_fail}/{len(df)}")
    print(f"  Returns: KPSS rejects stationarity:     {n_ret_kpss}/{len(df)}")
    print(f"  Prices:  ADF fails to reject unit root: {n_price_fail}/{len(df)}")

    plot_overview_bars(df, OUTPUT_DIR / "overview_ranks.png")

    # 1) Non-stationary returns (true failures) — usually just ELLT
    ret_bad = df[(df["ret_adf_fail"] == 1) | (df["ret_kpss_reject"] == 1)].sort_values(
        ["ret_kpss_reject", "adf_ret_pval"], ascending=[False, False]
    )
    if len(ret_bad) == 0:
        # fallback: highest return ADF p (still tiny) + note
        ret_bad = df.sort_values("adf_ret_pval", ascending=False).head(min(3, TOP_N))
    dump_set(
        "nonstationary_returns",
        ret_bad,
        prices,
        returns,
        df,
        "Non-stationary / borderline returns",
        "#C44E52",
    )

    # 2) Least stationary prices
    price_bad = df.sort_values("adf_price_pval", ascending=False).head(TOP_N)
    dump_set(
        "least_stationary_prices",
        price_bad,
        prices,
        returns,
        df,
        "Least-stationary prices (highest ADF p)",
        "#8172B3",
    )

    # 3) Worst performers by Sharpe
    perf_bad = df.sort_values("sharpe", ascending=True).head(TOP_N)
    dump_set(
        "worst_performance",
        perf_bad,
        prices,
        returns,
        df,
        "Worst assets by Sharpe",
        "#C44E52",
    )

    print("\nNon-stationary returns:")
    print(ret_bad[["ticker", "adf_ret_pval", "kpss_ret_pval", "total_return", "sharpe"]].to_string(index=False))
    print("\nLeast-stationary prices (top):")
    print(price_bad[["ticker", "adf_price_pval", "total_return", "sharpe"]].head(8).to_string(index=False))
    print("\nWorst Sharpe:")
    print(perf_bad[["ticker", "sharpe", "total_return", "max_dd"]].to_string(index=False))
    print(f"\nAll outputs under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
