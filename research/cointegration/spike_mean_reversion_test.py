#!/usr/bin/env python
"""
Test: after a large return spike, do returns mean-revert next day(s)?
Cross-sectional + pooled evidence on all assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("cointegration_output/lagged_returns/spike_mean_reversion")
HORIZONS = (1, 2, 3, 5)
PERCENTILES = (95, 97, 99)  # upper spike thresholds; lower = 100-p


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def forward_return(r: np.ndarray, t: int, h: int) -> float:
    """Compound return from t+1 .. t+h inclusive (0-indexed on returns array)."""
    if t + h >= len(r):
        return np.nan
    chunk = r[t + 1 : t + 1 + h]
    if not np.all(np.isfinite(chunk)):
        return np.nan
    return float(np.prod(1.0 + chunk) - 1.0)


def analyze(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each asset, mark spike days by percentile of that asset's own return dist,
    then measure forward returns. Also pooled across all assets.
    """
    event_rows = []
    summary_rows = []

    for col in returns.columns:
        r = returns[col].values.astype(float)
        valid = r[np.isfinite(r)]
        for p in PERCENTILES:
            hi = np.nanpercentile(valid, p)
            lo = np.nanpercentile(valid, 100 - p)
            for t in range(len(r) - max(HORIZONS) - 1):
                if not np.isfinite(r[t]):
                    continue
                if r[t] >= hi:
                    side = "up_spike"
                elif r[t] <= lo:
                    side = "down_spike"
                else:
                    continue
                row = {
                    "ticker": col,
                    "t": t,
                    "side": side,
                    "percentile": p,
                    "spike_ret": float(r[t]),
                    "threshold_hi": float(hi),
                    "threshold_lo": float(lo),
                }
                for h in HORIZONS:
                    row[f"fwd_{h}d"] = forward_return(r, t, h)
                event_rows.append(row)

        # per-asset summary at 97th pct
        p = 97
        hi = np.nanpercentile(valid, p)
        lo = np.nanpercentile(valid, 100 - p)
        up_idx = np.where(r[:- max(HORIZONS) - 1] >= hi)[0]
        dn_idx = np.where(r[:- max(HORIZONS) - 1] <= lo)[0]
        def avg_fwd(idxs, h):
            vals = [forward_return(r, int(i), h) for i in idxs]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan, len(vals)

        srow = {"ticker": col, "n_up": len(up_idx), "n_down": len(dn_idx)}
        for h in HORIZONS:
            mu_u, n_u = avg_fwd(up_idx, h)
            mu_d, n_d = avg_fwd(dn_idx, h)
            srow[f"after_up_fwd_{h}d"] = mu_u
            srow[f"after_down_fwd_{h}d"] = mu_d
            # hit rate: after up spike, next day negative?
            if h == 1:
                ups = [forward_return(r, int(i), 1) for i in up_idx]
                dns = [forward_return(r, int(i), 1) for i in dn_idx]
                ups = [v for v in ups if np.isfinite(v)]
                dns = [v for v in dns if np.isfinite(v)]
                srow["after_up_neg_rate"] = float(np.mean(np.array(ups) < 0)) if ups else np.nan
                srow["after_down_pos_rate"] = float(np.mean(np.array(dns) > 0)) if dns else np.nan
        summary_rows.append(srow)

    events = pd.DataFrame(event_rows)
    per_asset = pd.DataFrame(summary_rows)

    # pooled summary by percentile x side x horizon
    pooled = []
    for p in PERCENTILES:
        sub = events[events["percentile"] == p]
        for side in ("up_spike", "down_spike"):
            s = sub[sub["side"] == side]
            for h in HORIZONS:
                col = f"fwd_{h}d"
                x = s[col].dropna().values
                if len(x) == 0:
                    continue
                pooled.append(
                    {
                        "percentile": p,
                        "side": side,
                        "horizon": h,
                        "n": len(x),
                        "mean": float(np.mean(x)),
                        "median": float(np.median(x)),
                        "std": float(np.std(x, ddof=1)),
                        "tstat": float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
                        if len(x) > 2 and np.std(x, ddof=1) > 0
                        else np.nan,
                        "hit_fade": float(np.mean(x < 0))
                        if side == "up_spike"
                        else float(np.mean(x > 0)),
                    }
                )
    pooled_df = pd.DataFrame(pooled)
    return events, per_asset, pooled_df


def plot_pooled(pooled: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, side, title, color in (
        (axes[0], "up_spike", "After UP spike → fade (short)", "#C44E52"),
        (axes[1], "down_spike", "After DOWN spike → bounce (long)", "#55A868"),
    ):
        sub = pooled[pooled["side"] == side]
        for p in PERCENTILES:
            s = sub[sub["percentile"] == p].sort_values("horizon")
            ax.plot(s["horizon"], s["mean"] * 100, "o-", label=f"p{p}/p{100-p}", lw=1.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Horizon (days ahead)")
        ax.set_ylabel("Mean forward return (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(list(HORIZONS))

    fig.suptitle("Spike mean-reversion: average forward return after extreme days (all assets pooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hit_rates(pooled: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, side, title in (
        (axes[0], "up_spike", "P(next return < 0) after UP spike"),
        (axes[1], "down_spike", "P(next return > 0) after DOWN spike"),
    ):
        sub = pooled[(pooled["side"] == side) & (pooled["horizon"] == 1)]
        ax.bar([str(p) for p in sub["percentile"]], sub["hit_fade"], color="#4C72B0", width=0.5)
        ax.axhline(0.5, color="k", ls="--", lw=0.8, label="coin flip 50%")
        for i, r in enumerate(sub.itertuples()):
            ax.text(i, r.hit_fade + 0.01, f"{r.hit_fade:.1%}\nn={r.n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Spike percentile threshold")
        ax.set_ylabel("Fade hit rate")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("How often does the fade work next day?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_spike_vs_next(events: pd.DataFrame, path: Path) -> None:
    """All assets, p97 events: spike size vs next-day return."""
    sub = events[events["percentile"] == 97].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, side, color, title in (
        (axes[0], "up_spike", "#C44E52", "UP spikes (p97): spike vs next day"),
        (axes[1], "down_spike", "#55A868", "DOWN spikes (p3): spike vs next day"),
    ):
        s = sub[sub["side"] == side]
        ax.scatter(s["spike_ret"] * 100, s["fwd_1d"] * 100, s=8, alpha=0.25, color=color)
        # binned mean
        if len(s) > 20:
            qs = np.linspace(s["spike_ret"].min(), s["spike_ret"].max(), 8)
            mids, means = [], []
            for a, b in zip(qs[:-1], qs[1:]):
                m = s[(s["spike_ret"] >= a) & (s["spike_ret"] < b)]["fwd_1d"]
                if len(m) >= 5:
                    mids.append(0.5 * (a + b) * 100)
                    means.append(m.mean() * 100)
            ax.plot(mids, means, "k-o", lw=1.5, label="binned mean")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("Spike day return (%)")
        ax.set_ylabel("Next-day return (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_asset(per_asset: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    o = per_asset.sort_values("after_up_fwd_1d")
    colors = ["#55A868" if v < 0 else "#C44E52" for v in o["after_up_fwd_1d"]]
    # green = mean reversion works (negative after up)
    ax = axes[0]
    ax.barh(o["ticker"], o["after_up_fwd_1d"] * 100, color=colors, height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean next-day return after UP spike (%)")
    ax.set_title("After UP spike (p97): green = fades (MR works)")

    o = per_asset.sort_values("after_down_fwd_1d")
    colors = ["#55A868" if v > 0 else "#C44E52" for v in o["after_down_fwd_1d"]]
    ax = axes[1]
    ax.barh(o["ticker"], o["after_down_fwd_1d"] * 100, color=colors, height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean next-day return after DOWN spike (%)")
    ax.set_title("After DOWN spike (p3): green = bounces (MR works)")

    fig.suptitle("Per-asset spike fade — next day (own-asset 97th/3rd percentile)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_climbing_vs_spike_concept(prices: pd.DataFrame, returns: pd.DataFrame, path: Path) -> None:
    """Illustrate: trending climb ≠ repeated spikes; mark spike days on a few assets."""
    # pick a climber (good sharpe / up trend) and a spiky loser
    tickers = ["MDGI", "SMAH", "BLBT", "ELLT"]
    tickers = [t for t in tickers if t in prices.columns]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.ravel()
    for ax, col in zip(axes, tickers):
        p = prices[col].values.astype(float)
        r = returns[col].values.astype(float)
        t = np.arange(1, len(p) + 1)
        ax.plot(t, p / p[0], color="#4C72B0", lw=1.1)
        hi = np.nanpercentile(r, 97)
        lo = np.nanpercentile(r, 3)
        up = np.where(r >= hi)[0]
        dn = np.where(r <= lo)[0]
        ax.scatter(up + 2, (p / p[0])[up + 1], s=18, color="#C44E52", zorder=3, label="up spike day")
        ax.scatter(dn + 2, (p / p[0])[dn + 1], s=18, color="#55A868", zorder=3, label="down spike day")
        ax.set_title(f"{col}")
        ax.set_ylabel("Price / P0")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Climbing price ≠ continuous spikes: red/green = rare extreme return days (p97 / p3)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def simple_backtest(returns: pd.DataFrame, p: int = 97) -> pd.DataFrame:
    """
    Rule: after up spike → short 1 day; after down spike → long 1 day.
    Equal-weight across assets that signal; pnl = -sign * next return for up, + for down.
    Expanding threshold using only past data (causal).
    """
    n = len(returns)
    cols = returns.columns.tolist()
    R = returns.values.astype(float)
    # causal percentile: use data up to t inclusive for threshold at t
    pnls = []
    for t in range(60, n - 1):  # warm-up 60 days
        day_pnls = []
        for j, col in enumerate(cols):
            hist = R[: t + 1, j]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 40:
                continue
            hi = np.percentile(hist, p)
            lo = np.percentile(hist, 100 - p)
            rt = R[t, j]
            nxt = R[t + 1, j]
            if not np.isfinite(rt) or not np.isfinite(nxt):
                continue
            if rt >= hi:
                day_pnls.append(-nxt)  # short
            elif rt <= lo:
                day_pnls.append(+nxt)  # long
        if day_pnls:
            pnls.append({"t": t, "pnl": float(np.mean(day_pnls)), "n_signals": len(day_pnls)})
    bt = pd.DataFrame(pnls)
    if len(bt):
        bt["cum"] = (1 + bt["pnl"]).cumprod()
        bt["sharpe"] = bt["pnl"].mean() / bt["pnl"].std() * np.sqrt(252) if bt["pnl"].std() > 0 else np.nan
    return bt


def plot_backtest(bt: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    ax.plot(bt["t"], bt["cum"], color="#4C72B0", lw=1.3)
    ax.set_ylabel("Cumulative wealth")
    ax.set_title(
        f"Causal spike-fade backtest (p97): short after up, long after down — Sharpe≈{bt['pnl'].mean()/bt['pnl'].std()*np.sqrt(252):.2f}"
    )
    ax.axhline(1.0, color="k", lw=0.5)

    ax = axes[1]
    ax.bar(bt["t"], bt["pnl"] * 100, width=1.0, color="#8172B3", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily strategy return (%)")
    ax.set_title("Daily PnL of equal-weight fade signals")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    returns = prices.pct_change().iloc[1:]
    print(f"Loaded {returns.shape[0]} days x {returns.shape[1]} assets")

    events, per_asset, pooled = analyze(returns)
    events.to_csv(OUTPUT_DIR / "spike_events.csv", index=False)
    per_asset.to_csv(OUTPUT_DIR / "per_asset_spike_fade.csv", index=False)
    pooled.to_csv(OUTPUT_DIR / "pooled_spike_fade.csv", index=False)

    print("\n=== Pooled mean forward returns after spikes ===")
    print(pooled.to_string(index=False))

    plot_pooled(pooled, OUTPUT_DIR / "pooled_forward_returns.png")
    plot_hit_rates(pooled, OUTPUT_DIR / "fade_hit_rates.png")
    plot_scatter_spike_vs_next(events, OUTPUT_DIR / "spike_vs_nextday_scatter.png")
    plot_per_asset(per_asset, OUTPUT_DIR / "per_asset_fade.png")
    plot_climbing_vs_spike_concept(prices, returns, OUTPUT_DIR / "climbing_vs_spikes_examples.png")

    print("\nRunning causal backtest...")
    bt = simple_backtest(returns, p=97)
    bt.to_csv(OUTPUT_DIR / "backtest_p97.csv", index=False)
    if len(bt):
        sharpe = bt["pnl"].mean() / bt["pnl"].std() * np.sqrt(252)
        print(f"  days={len(bt)}  mean daily={bt['pnl'].mean():.5f}  Sharpe={sharpe:.2f}  final wealth={bt['cum'].iloc[-1]:.3f}")
        print(f"  avg signals/day={bt['n_signals'].mean():.1f}")
        plot_backtest(bt, OUTPUT_DIR / "backtest_cum.png")

    # fraction of assets where fade works
    n_up_works = int((per_asset["after_up_fwd_1d"] < 0).sum())
    n_dn_works = int((per_asset["after_down_fwd_1d"] > 0).sum())
    print(f"\nAssets where UP-spike fades next day:   {n_up_works}/{len(per_asset)}")
    print(f"Assets where DOWN-spike bounces next day: {n_dn_works}/{len(per_asset)}")
    print(f"\nWrote PNGs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
