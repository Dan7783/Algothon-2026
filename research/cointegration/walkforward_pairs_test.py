#!/usr/bin/env python
"""
Walk-forward test: if a pair is cointegrated over a 50-day lookback,
does it remain tradeable over the next 50 days?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

LOOKBACK = 50
FORWARD = 50
STEP = 50
SIGNIFICANCE = 0.05
OUTPUT_DIR = Path("cointegration_output/walkforward_50")


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """OLS hedge ratio: y ~ beta * x (no intercept, standard for log prices; with intercept for levels)."""
    x_ = np.column_stack([np.ones(len(x)), x])
    beta, _ = np.linalg.lstsq(x_, y, rcond=None)[0]
    return beta


def spread_half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life from AR(1) on spread."""
    if len(spread) < 5:
        return np.inf
    lag = spread[:-1]
    delta = np.diff(spread)
    if np.std(lag) < 1e-12:
        return np.inf
    x_ = np.column_stack([np.ones(len(lag)), lag])
    phi, _ = np.linalg.lstsq(x_, delta, rcond=None)[0]
    if phi >= 0:
        return np.inf
    hl = -np.log(2) / phi
    return float(hl) if hl > 0 else np.inf


def simple_pairs_pnl(
    y_fw: np.ndarray,
    x_fw: np.ndarray,
    beta: float,
    mu_lb: float,
    sigma_lb: float,
    entry_z: float = 2.0,
) -> float:
    """
    Out-of-sample mean-reversion PnL using lookback hedge ratio and z-score params.
    """
    spread = y_fw - beta * x_fw
    if sigma_lb < 1e-12:
        return 0.0

    z = (spread - mu_lb) / sigma_lb
    position = 0.0
    entry_spread = 0.0
    pnl = 0.0

    for i in range(1, len(spread)):
        if position == 0:
            if z[i - 1] > entry_z:
                position = -1.0
                entry_spread = spread[i - 1]
            elif z[i - 1] < -entry_z:
                position = 1.0
                entry_spread = spread[i - 1]
        else:
            # exit when z crosses 0 or at end
            if (position > 0 and z[i - 1] >= 0) or (position < 0 and z[i - 1] <= 0):
                pnl += position * (spread[i - 1] - entry_spread)
                position = 0.0

    if position != 0:
        pnl += position * (spread[-1] - entry_spread)
    return float(pnl)


def evaluate_pair_window(y_lb: np.ndarray, x_lb: np.ndarray, y_fw: np.ndarray, x_fw: np.ndarray) -> dict:
    if np.std(y_lb) < 1e-12 or np.std(x_lb) < 1e-12:
        return None

    _, lb_pval, _ = coint(y_lb, x_lb)
    lb_sig = lb_pval < SIGNIFICANCE

    beta = hedge_ratio(y_lb, x_lb)
    lb_spread = y_lb - beta * x_lb
    mu_lb, sigma_lb = lb_spread.mean(), lb_spread.std()
    fw_spread = y_fw - beta * x_fw

    _, fw_pval, _ = coint(y_fw, x_fw)
    fw_coint = fw_pval < SIGNIFICANCE

    adf_stat, adf_pval, *_ = adfuller(fw_spread, autolag="AIC")
    spread_stationary = adf_pval < SIGNIFICANCE

    hl = spread_half_life(fw_spread)
    hl_ok = hl < FORWARD  # mean-reverts within forward window

    pnl = simple_pairs_pnl(y_fw, x_fw, beta, mu_lb, sigma_lb)
    profitable = pnl > 0

    # composite "tradeable": spread stationary OR still cointegrated, with reasonable half-life
    tradeable = (fw_coint or spread_stationary) and hl_ok

    return {
        "lb_pval": lb_pval,
        "lb_sig": lb_sig,
        "fw_pval": fw_pval,
        "fw_coint": fw_coint,
        "adf_pval": adf_pval,
        "spread_stationary": spread_stationary,
        "half_life": hl,
        "hl_ok": hl_ok,
        "pnl": pnl,
        "profitable": profitable,
        "tradeable": tradeable,
        "beta": beta,
    }


def run_walkforward(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = prices.columns.tolist()
    n_days = len(prices)
    window_starts = list(range(0, n_days - LOOKBACK - FORWARD + 1, STEP))

    detail_rows = []
    for w_idx, start in enumerate(window_starts):
        lb_end = start + LOOKBACK
        fw_end = lb_end + FORWARD
        lb = prices.iloc[start:lb_end]
        fw = prices.iloc[lb_end:fw_end]

        for i, a in enumerate(tickers):
            for j, b in enumerate(tickers):
                if j <= i:
                    continue
                res = evaluate_pair_window(
                    lb[a].values, lb[b].values, fw[a].values, fw[b].values
                )
                if res is None:
                    continue
                detail_rows.append(
                    {
                        "window": w_idx + 1,
                        "lb_start_day": start + 1,
                        "lb_end_day": lb_end,
                        "fw_start_day": lb_end + 1,
                        "fw_end_day": fw_end,
                        "pair": f"{a}-{b}",
                        "ticker_a": a,
                        "ticker_b": b,
                        **res,
                    }
                )

    detail = pd.DataFrame(detail_rows)

    # Per-pair aggregation (only windows where lookback was significant)
    pair_rows = []
    for pair, grp in detail.groupby("pair"):
        lb_sig = grp[grp["lb_sig"]]
        n_lb_sig = len(lb_sig)
        n_windows = grp["window"].nunique()

        if n_lb_sig == 0:
            pair_rows.append(
                {
                    "pair": pair,
                    "ticker_a": grp["ticker_a"].iloc[0],
                    "ticker_b": grp["ticker_b"].iloc[0],
                    "n_windows": n_windows,
                    "n_lb_sig": 0,
                    "n_fw_coint": 0,
                    "n_spread_stationary": 0,
                    "n_tradeable": 0,
                    "n_profitable": 0,
                    "hit_rate_fw_coint": 0.0,
                    "hit_rate_spread_stat": 0.0,
                    "hit_rate_tradeable": 0.0,
                    "hit_rate_profitable": 0.0,
                    "mean_fw_pval_when_lb_sig": np.nan,
                    "mean_pnl_when_lb_sig": np.nan,
                }
            )
            continue

        pair_rows.append(
            {
                "pair": pair,
                "ticker_a": grp["ticker_a"].iloc[0],
                "ticker_b": grp["ticker_b"].iloc[0],
                "n_windows": n_windows,
                "n_lb_sig": n_lb_sig,
                "n_fw_coint": int(lb_sig["fw_coint"].sum()),
                "n_spread_stationary": int(lb_sig["spread_stationary"].sum()),
                "n_tradeable": int(lb_sig["tradeable"].sum()),
                "n_profitable": int(lb_sig["profitable"].sum()),
                "hit_rate_fw_coint": 100 * lb_sig["fw_coint"].mean(),
                "hit_rate_spread_stat": 100 * lb_sig["spread_stationary"].mean(),
                "hit_rate_tradeable": 100 * lb_sig["tradeable"].mean(),
                "hit_rate_profitable": 100 * lb_sig["profitable"].mean(),
                "mean_fw_pval_when_lb_sig": lb_sig["fw_pval"].mean(),
                "mean_pnl_when_lb_sig": lb_sig["pnl"].mean(),
            }
        )

    summary = pd.DataFrame(pair_rows).sort_values(
        ["n_lb_sig", "hit_rate_tradeable", "hit_rate_profitable"],
        ascending=[False, False, False],
    )
    return detail, summary


def print_overall_stats(detail: pd.DataFrame, n_windows: int) -> None:
    lb_sig = detail[detail["lb_sig"]]
    n_tests = len(lb_sig)
    n_pairs = detail["pair"].nunique()

    print(f"\nWalk-forward setup: {LOOKBACK}-day lookback -> {FORWARD}-day forward, step {STEP}")
    print(f"Windows: {n_windows} | Unique pairs: {n_pairs}")
    print(f"\n--- When lookback cointegration is significant (p<{SIGNIFICANCE}) ---")
    print(f"  Observations (pair x window): {n_tests:,}")
    if n_tests == 0:
        return

    print(f"  Still cointegrated forward:     {lb_sig['fw_coint'].mean()*100:5.1f}%")
    print(f"  Forward spread stationary (ADF): {lb_sig['spread_stationary'].mean()*100:5.1f}%")
    print(f"  Half-life < {FORWARD} days:              {lb_sig['hl_ok'].mean()*100:5.1f}%")
    print(f"  Composite 'tradeable':           {lb_sig['tradeable'].mean()*100:5.1f}%")
    print(f"  Simple MR strategy profitable:   {lb_sig['profitable'].mean()*100:5.1f}%")

    # Pairs that ever appeared as lb_sig
    ever_sig = detail.groupby("pair")["lb_sig"].any()
    n_ever = ever_sig.sum()
    print(f"\n--- Pair coverage ---")
    print(f"  Pairs cointegrated in >=1 lookback window: {n_ever} / {n_pairs}")

    good_pairs = detail.groupby("pair").apply(
        lambda g: (g["lb_sig"].sum() >= 2) and (g.loc[g["lb_sig"], "tradeable"].mean() >= 0.5)
        if g["lb_sig"].any()
        else False,
        include_groups=False,
    )
    print(f"  Pairs with >=2 lb signals AND >=50% forward tradeable: {good_pairs.sum()}")


def plot_results(summary: pd.DataFrame, detail: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lb_sig = detail[detail["lb_sig"]]

    # 1. Overall hit rates bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {
        "Forward\ncointegrated": lb_sig["fw_coint"].mean() * 100,
        "Spread\nstationary": lb_sig["spread_stationary"].mean() * 100,
        "Half-life\nOK": lb_sig["hl_ok"].mean() * 100,
        "Composite\ntradeable": lb_sig["tradeable"].mean() * 100,
        "MR strategy\nprofitable": lb_sig["profitable"].mean() * 100,
    }
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.7, label="50% baseline")
    ax.set_ylabel("Hit rate (%) when lookback p < 0.05")
    ax.set_title(f"Walk-forward: {LOOKBACK}-day lookback → {FORWARD}-day forward\n(all pair-window observations)")
    for bar, val in zip(bars, metrics.values(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, max(metrics.values()) * 1.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "overall_hit_rates.png", dpi=150)
    plt.close(fig)

    # 2. Top pairs by forward tradeability (only pairs with >=2 lb signals)
    plot_df = summary[summary["n_lb_sig"] >= 2].head(30).iloc[::-1]
    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.25)))
        colors = plt.cm.RdYlGn(plot_df["hit_rate_tradeable"] / 100)
        ax.barh(plot_df["pair"], plot_df["hit_rate_tradeable"], color=colors, edgecolor="white")
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row["hit_rate_tradeable"] + 1,
                i,
                f"{row['n_tradeable']}/{row['n_lb_sig']} lb-sig",
                va="center",
                fontsize=7,
            )
        ax.set_xlabel("Forward tradeable hit rate (%) when lookback significant")
        ax.set_title("Top pairs: lookback cointegration → forward tradeability\n(≥2 lookback signals)")
        ax.set_xlim(0, 105)
        plt.tight_layout()
        fig.savefig(out_dir / "top_pairs_tradeability.png", dpi=150)
        plt.close(fig)

    # 3. Scatter: persistence vs forward hit rate
    active = summary[summary["n_lb_sig"] >= 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        active["n_lb_sig"],
        active["hit_rate_tradeable"],
        c=active["hit_rate_profitable"],
        cmap="RdYlGn",
        s=30,
        alpha=0.7,
        vmin=0,
        vmax=100,
    )
    ax.set_xlabel("# lookback windows with p < 0.05")
    ax.set_ylabel("Forward tradeable hit rate (%)")
    ax.set_title("Pair persistence vs forward predictability")
    plt.colorbar(sc, label="Profitable hit rate (%)")
    plt.tight_layout()
    fig.savefig(out_dir / "persistence_vs_hitrate.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {out_dir}/")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    n_windows = len(range(0, len(prices) - LOOKBACK - FORWARD + 1, STEP))

    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments")
    detail, summary = run_walkforward(prices)

    detail.to_csv(OUTPUT_DIR / "walkforward_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "walkforward_pair_summary.csv", index=False)
    print(f"Saved detail ({len(detail):,} rows) and summary ({len(summary)} pairs)")

    print_overall_stats(detail, n_windows)
    plot_results(summary, detail, OUTPUT_DIR)

    top = summary[summary["n_lb_sig"] >= 2].head(15)
    print("\nTop pairs (>=2 lookback signals, by forward tradeability):")
    print(
        top[
            ["pair", "n_lb_sig", "n_tradeable", "hit_rate_tradeable", "hit_rate_profitable", "mean_pnl_when_lb_sig"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
