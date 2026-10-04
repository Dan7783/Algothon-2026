#!/usr/bin/env python
"""
Lagged-return structure across the universe (full lag spectrum, PNG outputs):
  1) Autocorrelation of each asset's returns at all lags
  2) Autocorrelation of OLS pair-spread residual returns at all lags
  3) Lead-lag (cross-correlation) of returns for every pair at all lags
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("cointegration_output/lagged_returns")
# Box–Jenkins rule of thumb: lags 1 .. n/4 (all usable lags for ACF)
SIGNIFICANCE = 0.05


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def lags_for(n: int) -> np.ndarray:
    return np.arange(1, n // 4 + 1)


def acf_all(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Pearson ACF at each lag. Returns array same length as lags."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = np.full(len(lags), np.nan)
    if len(x) < 10 or np.std(x) < 1e-15:
        return out
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-18:
        return out
    for i, lag in enumerate(lags):
        lag = int(lag)
        if lag >= len(x) - 2:
            continue
        out[i] = np.dot(x[:-lag], x[lag:]) / var
    return out


def acf_pvalue(rho: float, n: int, lag: int) -> float:
    if not np.isfinite(rho) or n <= lag + 2:
        return np.nan
    se = 1.0 / np.sqrt(n - lag)
    return float(2 * (1 - stats.norm.cdf(abs(rho / se))))


def ols_spread(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return y - mu - beta * x


def best_unordered_spread(spread_long: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered pair: keep orientation with larger |acf_1|."""
    tmp = spread_long.copy()
    tmp["key"] = tmp.apply(lambda r: "-".join(sorted([r["dep"], r["ind"]])), axis=1)
    return (
        tmp.sort_values("acf_1_abs", ascending=False)
        .groupby("key", as_index=False)
        .first()
        .sort_values("acf_1_abs", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------


def asset_return_acf(returns: pd.DataFrame, lags: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-asset summary + (n_assets x n_lags) ACF matrix."""
    tickers = returns.columns.tolist()
    n = len(returns)
    mat = np.zeros((len(tickers), len(lags)))
    rows = []
    for i, col in enumerate(tickers):
        r = returns[col].values
        acf = acf_all(r, lags)
        mat[i] = acf
        # strongest |acf| anywhere in spectrum
        abs_acf = np.abs(acf)
        k = int(np.nanargmax(abs_acf))
        row = {
            "ticker": col,
            "n": n,
            "acf_1": float(acf[0]),
            "pval_1": acf_pvalue(float(acf[0]), n, 1),
            "best_lag": int(lags[k]),
            "best_acf": float(acf[k]),
            "best_abs": float(abs_acf[k]),
            "best_pval": acf_pvalue(float(acf[k]), n, int(lags[k])),
            "n_sig_lags": int(np.sum([acf_pvalue(float(acf[j]), n, int(lags[j])) < SIGNIFICANCE for j in range(len(lags))])),
        }
        for lag in (1, 2, 3, 5, 10, 20, 40):
            if lag <= lags[-1]:
                j = int(np.where(lags == lag)[0][0])
                row[f"acf_{lag}"] = float(acf[j])
                row[f"pval_{lag}"] = acf_pvalue(float(acf[j]), n, lag)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("best_abs", ascending=False).reset_index(drop=True)
    return df, mat


def pair_spread_acf(prices: pd.DataFrame, lags: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Long table of orientations + matrix of best-orientation ACF per unordered pair."""
    tickers = prices.columns.tolist()
    pairs = list(combinations(tickers, 2))
    rows = []
    acf_mats = []  # one per orientation row, then we pick best later
    labels = []

    total = len(pairs)
    for done, (a, b) in enumerate(pairs, 1):
        if done % 300 == 0 or done == total:
            print(f"  spread ACF: {done}/{total}")

        y = prices[a].values.astype(float)
        x = prices[b].values.astype(float)
        if np.std(y) < 1e-12 or np.std(x) < 1e-12:
            continue

        for dep, ind, la, lb in ((y, x, a, b), (x, y, b, a)):
            spread = ols_spread(dep, ind)
            sret = np.diff(spread)
            acf = acf_all(sret, lags)
            n = len(sret)
            abs_acf = np.abs(acf)
            k = int(np.nanargmax(abs_acf))
            level_acf1 = float(acf_all(spread, np.array([1]))[0])
            row = {
                "pair": f"{la}-{lb}",
                "dep": la,
                "ind": lb,
                "n": n,
                "acf_1": float(acf[0]),
                "pval_1": acf_pvalue(float(acf[0]), n, 1),
                "acf_1_abs": abs(float(acf[0])),
                "best_lag": int(lags[k]),
                "best_acf": float(acf[k]),
                "best_abs": float(abs_acf[k]),
                "best_pval": acf_pvalue(float(acf[k]), n, int(lags[k])),
                "spread_level_acf_1": level_acf1,
            }
            for lag in (1, 2, 3, 5, 10, 20, 40):
                if lag <= lags[-1]:
                    j = int(np.where(lags == lag)[0][0])
                    row[f"acf_{lag}"] = float(acf[j])
            rows.append(row)
            acf_mats.append(acf)
            labels.append(f"{la}-{lb}")

    long_df = pd.DataFrame(rows)
    # best orientation matrix for unordered pairs
    best = best_unordered_spread(long_df)
    # rebuild matrix aligned to best pairs
    label_to_acf = {lab: acf for lab, acf in zip(labels, acf_mats)}
    mat = np.vstack([label_to_acf[p] for p in best["pair"]])
    return long_df, mat, best["pair"].tolist()


def lead_lag_all_pairs(returns: pd.DataFrame, lags: np.ndarray) -> pd.DataFrame:
    tickers = returns.columns.tolist()
    R = returns.values.astype(float)
    # demean once for faster rolling cross-corr
    R0 = R - R.mean(axis=0, keepdims=True)
    n = len(returns)
    rows = []
    pairs = list(combinations(range(len(tickers)), 2))
    total = len(pairs)
    key_lags = {1, 2, 3, 5, 10, 20, 40, 60, 90, 120}

    def xcorr(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / denom) if denom > 1e-18 else np.nan

    for done, (i, j) in enumerate(pairs, 1):
        if done % 200 == 0 or done == total:
            print(f"  lead-lag: {done}/{total}")

        a, b = tickers[i], tickers[j]
        ra, rb = R0[:, i], R0[:, j]
        sync = xcorr(ra, rb)

        best_abs = 0.0
        best_lag = 0
        best_dir = "sync"
        best_rho = sync

        row = {
            "pair": f"{a}-{b}",
            "ticker_a": a,
            "ticker_b": b,
            "corr_0": sync,
            "n": n,
        }

        for lag in lags:
            lag = int(lag)
            rho_a = xcorr(ra[:-lag], rb[lag:])
            rho_b = xcorr(rb[:-lag], ra[lag:])
            if lag in key_lags:
                row[f"a_leads_b_lag{lag}"] = rho_a
                row[f"b_leads_a_lag{lag}"] = rho_b
            for rho, direction in ((rho_a, f"{a}_leads_{b}"), (rho_b, f"{b}_leads_{a}")):
                if abs(rho) > best_abs:
                    best_abs = abs(rho)
                    best_lag = lag
                    best_dir = direction
                    best_rho = rho

        row["best_lag"] = best_lag
        row["best_direction"] = best_dir
        row["best_rho"] = best_rho
        row["best_abs"] = best_abs
        row["best_excess_over_sync"] = best_abs - abs(sync)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("best_abs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ci_band(n: int) -> float:
    return 1.96 / np.sqrt(n)


def plot_asset_multilag(
    asset_df: pd.DataFrame, acf_mat: np.ndarray, lags: np.ndarray, tickers_order: list[str], path: Path
) -> None:
    """Heatmap + mean ACF + top assets full ACF curves."""
    n = int(asset_df["n"].iloc[0])
    ci = _ci_band(n)
    ticker_to_i = {t: i for i, t in enumerate(tickers_order)}
    order = [ticker_to_i[t] for t in asset_df["ticker"]]
    mat = acf_mat[order]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2, interpolation="nearest")
    ax.set_yticks(range(len(asset_df)))
    ax.set_yticklabels(asset_df["ticker"], fontsize=6)
    xt = np.linspace(0, len(lags) - 1, min(16, len(lags))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels(lags[xt])
    ax.set_xlabel("Lag (days)")
    ax.set_title(f"Asset return ACF heatmap — lags 1…{int(lags[-1])} (RdBu, clipped ±0.2)")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="ACF")

    ax = fig.add_subplot(gs[1, 0])
    means = np.nanmean(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)
    ax.fill_between(lags, q25, q75, alpha=0.25, color="#4C72B0", label="IQR")
    ax.plot(lags, means, color="#4C72B0", lw=1.5, label="mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="gray", ls="--", lw=0.8)
    ax.axhline(-ci, color="gray", ls="--", lw=0.8, label=f"±1.96/√n ({ci:.3f})")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("Cross-sectional ACF by lag")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    tickers_listed = list(asset_df["ticker"])
    for _, r in asset_df.head(8).iterrows():
        row_i = tickers_listed.index(r["ticker"])
        ax.plot(
            lags,
            mat[row_i],
            lw=1.1,
            label=f"{r['ticker']} (best L{int(r['best_lag'])}={r['best_acf']:.3f})",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="gray", ls="--", lw=0.8)
    ax.axhline(-ci, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("Top 8 assets by max |ACF| across all lags")
    ax.legend(fontsize=7, loc="upper right")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_asset_best_lags(asset_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ["#C44E52" if p < SIGNIFICANCE else "#4C72B0" for p in asset_df["best_pval"]]
    ax.barh(asset_df["ticker"][::-1], asset_df["best_acf"][::-1], color=colors[::-1], height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ACF at best lag")
    ax.set_title("Each asset: strongest ACF anywhere in lags 1…60 (red=p<0.05)")

    ax = axes[1]
    max_lag = int(asset_df["best_lag"].max())
    ax.hist(asset_df["best_lag"], bins=np.arange(0.5, max_lag + 1.5, 1), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Lag where |ACF| is maximized")
    ax.set_ylabel("# assets")
    ax.set_title("Where does each asset's strongest ACF sit?")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spread_multilag(best: pd.DataFrame, acf_mat: np.ndarray, lags: np.ndarray, path: Path) -> None:
    n = int(best["n"].iloc[0])
    ci = _ci_band(n)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.32, wspace=0.28)

    # Heatmap of top 40 pairs
    ax = fig.add_subplot(gs[0, :])
    top_n = min(40, len(best))
    mat = acf_mat[:top_n]
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25, interpolation="nearest")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(best["pair"].iloc[:top_n], fontsize=7)
    xt = np.linspace(0, len(lags) - 1, 13).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels(lags[xt])
    ax.set_xlabel("Lag (days)")
    ax.set_title(f"Top {top_n} pairs by |ACF(1)| — residual-return ACF across all lags")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="ACF")

    # Distribution of ACF by selected lags
    ax = fig.add_subplot(gs[1, 0])
    show_lags = [1, 2, 3, 5, 10, 20, 40]
    show_lags = [L for L in show_lags if L <= lags[-1]]
    data = []
    for L in show_lags:
        j = int(np.where(lags == L)[0][0])
        data.append(acf_mat[:, j])
    bp = ax.boxplot(data, tick_labels=show_lags, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="gray", ls="--", lw=0.8)
    ax.axhline(-ci, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("Spread residual-return ACF distribution by lag (all pairs)")

    # Top pairs full ACF curves
    ax = fig.add_subplot(gs[1, 1])
    for i in range(min(8, len(best))):
        ax.plot(
            lags,
            acf_mat[i],
            lw=1.1,
            label=f"{best['pair'].iloc[i]} (L{best['best_lag'].iloc[i]}={best['best_acf'].iloc[i]:.3f})",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="gray", ls="--", lw=0.8)
    ax.axhline(-ci, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("Top 8 spreads — full ACF curves")
    ax.legend(fontsize=7)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spread_summary(best: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(best["acf_1"].dropna(), bins=40, color="#4C72B0", edgecolor="white", alpha=0.85, label="ACF(1)")
    ax.hist(best["best_acf"].dropna(), bins=40, color="#C44E52", edgecolor="white", alpha=0.45, label="best ACF (any lag)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ACF")
    ax.set_ylabel("# pairs")
    ax.set_title("Spread residual-return ACF: lag-1 vs strongest lag")
    ax.legend()

    ax = axes[1]
    top = best.nlargest(25, "best_abs")
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top["best_acf"]]
    labels = [f"{p} L{L}" for p, L in zip(top["pair"], top["best_lag"])]
    ax.barh(labels[::-1], top["best_acf"][::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ACF at best lag")
    ax.set_title("Top 25 pairs by max |ACF| across all lags")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leadlag(ll_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(ll_df["corr_0"].abs(), ll_df["best_abs"], s=10, alpha=0.4, color="#4C72B0")
    lim = max(ll_df["corr_0"].abs().max(), ll_df["best_abs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="lagged = sync")
    ax.set_xlabel("|contemporaneous corr|")
    ax.set_ylabel("best |lagged corr| (any lag)")
    ax.set_title("Lead-lag vs sync (search over all lags)")
    ax.legend()

    ax = axes[0, 1]
    max_lag = int(ll_df["best_lag"].max())
    ax.hist(ll_df["best_lag"], bins=np.arange(0.5, max_lag + 1.5, 1), color="#55A868", edgecolor="white")
    ax.set_xlabel("Lag of strongest cross-corr")
    ax.set_ylabel("# pairs")
    ax.set_title("Where lead-lag peaks across pairs")

    ax = axes[1, 0]
    top = ll_df.nlargest(25, "best_excess_over_sync")
    labels = [f"{r.best_direction} L{r.best_lag}" for r in top.itertuples()]
    ax.barh(labels[::-1], top["best_rho"][::-1], color="#55A868")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Best lagged correlation")
    ax.set_title("Top 25 by excess over sync (|lag|−|sync|)")

    ax = axes[1, 1]
    # mean |cross-corr| by lag (average of max(|a_leads|,|b_leads|) at key lags)
    key_lags = [L for L in (1, 2, 3, 5, 10, 20, 40, 60, 90, 120) if f"a_leads_b_lag{L}" in ll_df.columns]
    means = []
    for L in key_lags:
        m = np.maximum(ll_df[f"a_leads_b_lag{L}"].abs(), ll_df[f"b_leads_a_lag{L}"].abs()).mean()
        means.append(m)
    ax.plot(key_lags, means, "o-", color="#4C72B0")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Mean max(|cross-corr|)")
    ax.set_title("Average lead-lag strength by lag")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sig_counts_by_lag(
    asset_mat: np.ndarray,
    spread_mat: np.ndarray,
    lags: np.ndarray,
    n_ret: int,
    n_spread: int,
    path: Path,
) -> None:
    """How many series have significant ACF at each lag — shows lag-1 is not special a priori."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def count_sig(mat, n):
        counts = []
        for j, lag in enumerate(lags):
            rhos = mat[:, j]
            pvals = np.array([acf_pvalue(float(r), n, int(lag)) for r in rhos])
            counts.append(np.sum(pvals < SIGNIFICANCE))
        return np.array(counts)

    ax = axes[0]
    c = count_sig(asset_mat, n_ret)
    ax.bar(lags, c, color="#4C72B0", width=0.8)
    ax.axhline(SIGNIFICANCE * asset_mat.shape[0], color="gray", ls="--", lw=0.8, label="expected under H0 (5%)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("# assets with p<0.05")
    ax.set_title("Asset return: significant ACF count by lag")
    ax.legend(fontsize=8)

    ax = axes[1]
    c = count_sig(spread_mat, n_spread)
    ax.bar(lags, c, color="#C44E52", width=0.8)
    ax.axhline(SIGNIFICANCE * spread_mat.shape[0], color="gray", ls="--", lw=0.8, label="expected under H0 (5%)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("# pairs with p<0.05")
    ax.set_title("Spread residual-return: significant ACF count by lag")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    returns = prices.pct_change().iloc[1:]
    lags = lags_for(len(returns))
    print(f"Loaded {len(prices)} days x {len(prices.columns)} assets | {len(returns)} return days")
    print(f"Computing ACF / lead-lag for lags 1…{lags[-1]} ({len(lags)} lags)")

    print("\n1) Asset return ACF (all lags)...")
    asset_df, asset_mat = asset_return_acf(returns, lags)
    asset_df.to_csv(OUTPUT_DIR / "asset_return_acf.csv", index=False)
    np.save(OUTPUT_DIR / "asset_acf_matrix.npy", asset_mat)

    print("2) Pair-spread residual-return ACF (all lags)...")
    spread_long, spread_mat, _ = pair_spread_acf(prices, lags)
    spread_long.to_csv(OUTPUT_DIR / "pair_spread_return_acf.csv", index=False)
    best_spread = best_unordered_spread(spread_long)
    # realign matrix: pair_spread_acf already returns best-orientation mat
    best_spread.to_csv(OUTPUT_DIR / "pair_spread_return_acf_best.csv", index=False)
    np.save(OUTPUT_DIR / "spread_acf_matrix.npy", spread_mat)

    print("3) Lead-lag (all lags)...")
    ll_df = lead_lag_all_pairs(returns, lags)
    ll_df.to_csv(OUTPUT_DIR / "pair_leadlag.csv", index=False)

    # Plots
    print("\nPlotting PNGs...")
    tickers = returns.columns.tolist()
    plot_asset_multilag(asset_df, asset_mat, lags, tickers, OUTPUT_DIR / "asset_return_acf_all_lags.png")
    plot_asset_best_lags(asset_df, OUTPUT_DIR / "asset_return_best_lag.png")
    plot_spread_multilag(best_spread, spread_mat, lags, OUTPUT_DIR / "pair_spread_acf_all_lags.png")
    plot_spread_summary(best_spread, OUTPUT_DIR / "pair_spread_acf_summary.png")
    plot_leadlag(ll_df, OUTPUT_DIR / "pair_leadlag_all_lags.png")
    plot_sig_counts_by_lag(
        asset_mat,
        spread_mat,
        lags,
        len(returns),
        int(best_spread["n"].iloc[0]),
        OUTPUT_DIR / "acf_significance_by_lag.png",
    )

    summary = {
        "n_assets": len(asset_df),
        "n_pairs": len(ll_df),
        "max_lag": int(lags[-1]),
        "asset_best_lag_median": float(asset_df["best_lag"].median()),
        "asset_sig_at_lag1": int((asset_df["pval_1"] < SIGNIFICANCE).sum()),
        "asset_with_any_sig_lag": int((asset_df["n_sig_lags"] > 0).sum()),
        "spread_acf1_median": float(best_spread["acf_1"].median()),
        "spread_best_lag_median": float(best_spread["best_lag"].median()),
        "leadlag_best_lag_median": float(ll_df["best_lag"].median()),
        "leadlag_excess_over_sync": int((ll_df["best_excess_over_sync"] > 0).sum()),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nTop 10 assets by max |ACF| (any lag):")
    print(asset_df[["ticker", "best_lag", "best_acf", "best_pval", "acf_1", "n_sig_lags"]].head(10).to_string(index=False))

    print("\nTop 10 spreads by max |ACF| (any lag):")
    print(best_spread[["pair", "best_lag", "best_acf", "acf_1", "spread_level_acf_1"]].head(10).to_string(index=False))

    print("\nTop 10 lead-lag by excess over sync:")
    cols = ["pair", "best_direction", "best_lag", "best_rho", "corr_0", "best_excess_over_sync"]
    print(ll_df.nlargest(10, "best_excess_over_sync")[cols].to_string(index=False))

    print(f"\nPNG outputs in {OUTPUT_DIR}/:")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
