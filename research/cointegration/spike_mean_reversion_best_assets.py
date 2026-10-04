#!/usr/bin/env python
"""
Spike mean-reversion restricted to the 'best assets'
(top 12 by max |return ACF|).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("cointegration_output/lagged_returns/spike_mean_reversion/best_assets")
BEST_CSV = Path("cointegration_output/lagged_returns/best_assets/best_assets_stats.csv")
HORIZONS = (1, 2, 3, 5)
PERCENTILES = (95, 97, 99)


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def forward_return(r: np.ndarray, t: int, h: int) -> float:
    if t + h >= len(r):
        return np.nan
    chunk = r[t + 1 : t + 1 + h]
    if not np.all(np.isfinite(chunk)):
        return np.nan
    return float(np.prod(1.0 + chunk) - 1.0)


def analyze(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows, summary_rows = [], []
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
                }
                for h in HORIZONS:
                    row[f"fwd_{h}d"] = forward_return(r, t, h)
                event_rows.append(row)

        p = 97
        hi = np.nanpercentile(valid, p)
        lo = np.nanpercentile(valid, 100 - p)
        up_idx = np.where(r[: -max(HORIZONS) - 1] >= hi)[0]
        dn_idx = np.where(r[: -max(HORIZONS) - 1] <= lo)[0]

        def avg_fwd(idxs, h):
            vals = [forward_return(r, int(i), h) for i in idxs]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan, vals

        srow = {"ticker": col, "n_up": len(up_idx), "n_down": len(dn_idx)}
        for h in HORIZONS:
            mu_u, ups = avg_fwd(up_idx, h)
            mu_d, dns = avg_fwd(dn_idx, h)
            srow[f"after_up_fwd_{h}d"] = mu_u
            srow[f"after_down_fwd_{h}d"] = mu_d
            if h == 1:
                srow["after_up_neg_rate"] = float(np.mean(np.array(ups) < 0)) if ups else np.nan
                srow["after_down_pos_rate"] = float(np.mean(np.array(dns) > 0)) if dns else np.nan
                srow["after_up_std_1d"] = float(np.std(ups, ddof=1)) if len(ups) > 2 else np.nan
                srow["after_down_std_1d"] = float(np.std(dns, ddof=1)) if len(dns) > 2 else np.nan
        summary_rows.append(srow)

    events = pd.DataFrame(event_rows)
    per_asset = pd.DataFrame(summary_rows)

    pooled = []
    for p in PERCENTILES:
        sub = events[events["percentile"] == p]
        for side in ("up_spike", "down_spike"):
            s = sub[sub["side"] == side]
            for h in HORIZONS:
                x = s[f"fwd_{h}d"].dropna().values
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
                        "hit_fade": float(np.mean(x < 0)) if side == "up_spike" else float(np.mean(x > 0)),
                    }
                )
    return events, per_asset, pd.DataFrame(pooled)


def causal_backtest(returns: pd.DataFrame, p: int = 97) -> pd.DataFrame:
    n = len(returns)
    R = returns.values.astype(float)
    cols = returns.columns.tolist()
    rows = []
    for t in range(60, n - 1):
        day = []
        for j in range(len(cols)):
            hist = R[: t + 1, j]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 40:
                continue
            hi = np.percentile(hist, p)
            lo = np.percentile(hist, 100 - p)
            rt, nxt = R[t, j], R[t + 1, j]
            if not np.isfinite(rt) or not np.isfinite(nxt):
                continue
            if rt >= hi:
                day.append(-nxt)
            elif rt <= lo:
                day.append(+nxt)
        if day:
            rows.append({"t": t, "pnl": float(np.mean(day)), "n_signals": len(day)})
    bt = pd.DataFrame(rows)
    if len(bt):
        bt["cum"] = (1 + bt["pnl"]).cumprod()
    return bt


def plot_pooled(pooled: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, side, title in (
        (axes[0], "up_spike", "Best assets: after UP spike → expect fade"),
        (axes[1], "down_spike", "Best assets: after DOWN spike → expect bounce"),
    ):
        sub = pooled[pooled["side"] == side]
        for p in PERCENTILES:
            s = sub[sub["percentile"] == p].sort_values("horizon")
            ax.plot(s["horizon"], s["mean"] * 100, "o-", label=f"p{p}", lw=1.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Mean forward return (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(list(HORIZONS))
    fig.suptitle("Spike fade on best ACF assets only", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_asset(per_asset: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    o = per_asset.sort_values("after_up_fwd_1d")
    colors = ["#55A868" if v < 0 else "#C44E52" for v in o["after_up_fwd_1d"]]
    axes[0].barh(o["ticker"], o["after_up_fwd_1d"] * 100, color=colors)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("Mean next-day % after UP spike")
    axes[0].set_title("After UP (p97): green = fade works")

    o = per_asset.sort_values("after_down_fwd_1d")
    colors = ["#55A868" if v > 0 else "#C44E52" for v in o["after_down_fwd_1d"]]
    axes[1].barh(o["ticker"], o["after_down_fwd_1d"] * 100, color=colors)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Mean next-day % after DOWN spike")
    axes[1].set_title("After DOWN (p3): green = bounce works")

    fig.suptitle("Best assets — per-name spike fade (next day)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hit_and_table(per_asset: pd.DataFrame, pooled: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    o = per_asset.sort_values("after_up_neg_rate")
    ax.barh(o["ticker"], o["after_up_neg_rate"], color="#4C72B0")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_title("P(next < 0) after UP spike")
    ax.set_xlabel("Hit rate")

    ax = fig.add_subplot(gs[0, 1])
    o = per_asset.sort_values("after_down_pos_rate")
    ax.barh(o["ticker"], o["after_down_pos_rate"], color="#55A868")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_title("P(next > 0) after DOWN spike")
    ax.set_xlabel("Hit rate")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    sub = pooled[pooled["horizon"] == 1].copy()
    headers = ["pct", "side", "n", "mean%", "t-stat", "fade hit"]
    cells = [
        [
            str(int(r.percentile)),
            r.side.replace("_", " "),
            str(r.n),
            f"{r.mean * 100:+.3f}",
            f"{r.tstat:.2f}",
            f"{r.hit_fade:.1%}",
        ]
        for r in sub.itertuples()
    ]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Pooled next-day stats on best assets", pad=20)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(bt: pd.DataFrame, path: Path, label: str) -> None:
    sharpe = bt["pnl"].mean() / bt["pnl"].std() * np.sqrt(252) if bt["pnl"].std() > 0 else np.nan
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(bt["t"], bt["cum"], color="#4C72B0", lw=1.3)
    axes[0].axhline(1.0, color="k", lw=0.5)
    axes[0].set_ylabel("Wealth")
    axes[0].set_title(f"{label} — causal spike fade p97  |  Sharpe={sharpe:.2f}  final={bt['cum'].iloc[-1]:.3f}")
    axes[1].bar(bt["t"], bt["pnl"] * 100, width=1.0, color="#8172B3", alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Daily pnl %")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_examples(prices: pd.DataFrame, returns: pd.DataFrame, per_asset: pd.DataFrame, path: Path) -> None:
    """Show 4 names: 2 where up-fade works best, 2 where it fails."""
    good = per_asset.nsmallest(2, "after_up_fwd_1d")["ticker"].tolist()
    bad = per_asset.nlargest(2, "after_up_fwd_1d")["ticker"].tolist()
    picks = good + bad
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, picks):
        p = prices[col].values.astype(float)
        r = returns[col].values.astype(float)
        t = np.arange(1, len(p) + 1)
        ax.plot(t, p / p[0], color="#4C72B0", lw=1.0)
        hi, lo = np.nanpercentile(r, 97), np.nanpercentile(r, 3)
        up = np.where(r >= hi)[0]
        dn = np.where(r <= lo)[0]
        ax.scatter(up + 2, (p / p[0])[up + 1], s=22, c="#C44E52", zorder=3, label="up spike")
        ax.scatter(dn + 2, (p / p[0])[dn + 1], s=22, c="#55A868", zorder=3, label="down spike")
        row = per_asset.loc[per_asset["ticker"] == col].iloc[0]
        ax.set_title(
            f"{col}: after-up next={row['after_up_fwd_1d']*100:+.2f}%  "
            f"fade hit={row['after_up_neg_rate']:.0%}"
        )
        ax.legend(fontsize=7)
    fig.suptitle("Best assets: spike markers (top row ≈ fade works, bottom ≈ fails)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(BEST_CSV)["ticker"].tolist()
    prices = load_prices()[best]
    returns = prices.pct_change().iloc[1:]
    print(f"Best assets ({len(best)}): {', '.join(best)}")

    events, per_asset, pooled = analyze(returns)
    events.to_csv(OUTPUT_DIR / "spike_events.csv", index=False)
    per_asset.to_csv(OUTPUT_DIR / "per_asset.csv", index=False)
    pooled.to_csv(OUTPUT_DIR / "pooled.csv", index=False)

    print("\n=== Pooled (best assets only) ===")
    print(pooled[pooled["horizon"] == 1].to_string(index=False))

    print("\n=== Per asset (p97, next day) ===")
    cols = [
        "ticker",
        "n_up",
        "after_up_fwd_1d",
        "after_up_neg_rate",
        "n_down",
        "after_down_fwd_1d",
        "after_down_pos_rate",
    ]
    print(per_asset[cols].to_string(index=False))

    plot_pooled(pooled, OUTPUT_DIR / "pooled_forward_returns.png")
    plot_per_asset(per_asset, OUTPUT_DIR / "per_asset_fade.png")
    plot_hit_and_table(per_asset, pooled, OUTPUT_DIR / "hit_rates_and_table.png")
    plot_examples(prices, returns, per_asset, OUTPUT_DIR / "examples_spikes_on_price.png")

    bt = causal_backtest(returns, p=97)
    bt.to_csv(OUTPUT_DIR / "backtest_p97.csv", index=False)
    if len(bt):
        sharpe = bt["pnl"].mean() / bt["pnl"].std() * np.sqrt(252)
        print(
            f"\nCausal backtest (best assets only): "
            f"Sharpe={sharpe:.2f}  final={bt['cum'].iloc[-1]:.3f}  "
            f"mean daily={bt['pnl'].mean():.5f}  signals/day={bt['n_signals'].mean():.2f}"
        )
        plot_backtest(bt, OUTPUT_DIR / "backtest_cum.png", "Best-asset subset")

    n_up = int((per_asset["after_up_fwd_1d"] < 0).sum())
    n_dn = int((per_asset["after_down_fwd_1d"] > 0).sum())
    print(f"\nUP-spike fades: {n_up}/{len(per_asset)} assets")
    print(f"DOWN-spike bounces: {n_dn}/{len(per_asset)} assets")
    print(f"Wrote {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
