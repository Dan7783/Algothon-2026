#!/usr/bin/env python
"""Visualise and test short-lived cointegration and row-hub regimes.

The existing 50-day matrices plot ``-log10(p)``.  A large plotted value is
therefore a *small* raw p-value.  This script keeps those historical matrices
as the visual source, then adds causal forward tests and the stricter runtime
detector implemented in teamName.py.

Outputs are written to ``cointegration_output/short_regime_analysis``.
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller

import four_sigmas as strategy


BLOCK_SIZE = 50
SIGNIFICANCE = 0.05
SPIKE_P = 0.001
TRANSIENT_MAX_SIG_BLOCKS = 2
TOP_TRANSIENT = 40
TOP_CASES = 12
HUB_DEGREE = 10
HUB_LINKS_PER_ASSET = 5
FORWARD_DAYS = 50
MAX_HOLD = 20
ROUND_TRIP_COST = 0.0002

# Research-only candidate detector.  It mirrors the previously tested runtime
# idea without changing the current submitted strategy in teamName.py.
EPISODE_LOOKBACK = 75
EPISODE_CENTER = 30
EPISODE_CONFIRM_GAP = 10
EPISODE_ADF_T = -4.25
EPISODE_CONFIRM_T = -3.75
EPISODE_ASSET_ADF_T = -3.25
EPISODE_BETA_RATIO = 1.75
EPISODE_HL_MIN = 1.5
EPISODE_HL_MAX = 20.0
EPISODE_Z_ENTRY = 1.5
EPISODE_Z_MAX = 4.0
EPISODE_MAX = 3
_TI, _TJ = np.triu_indices(50, k=1)
_ALL_PI, _ALL_PJ = _TI + 1, _TJ + 1

ROOT = Path("cointegration_output")
BLOCK_DIR = ROOT / "block_50"
OUTPUT_DIR = ROOT / "short_regime_analysis"


def load_data():
    prices = pd.read_csv("prices.txt", sep=r"\s+")
    pvalue_files = sorted(BLOCK_DIR.glob("pvalues_block_*.csv"))
    if not pvalue_files:
        raise FileNotFoundError(f"No p-value matrices found in {BLOCK_DIR}")
    blocks = [pd.read_csv(path, index_col=0) for path in pvalue_files]
    names = prices.columns.tolist()
    for block in blocks:
        block.index = block.index.astype(str)
        block.columns = block.columns.astype(str)
        block = block.loc[names, names]
    return prices, blocks


def safe_adf_p(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 12 or np.std(values) < 1e-12:
        return np.nan
    try:
        return float(adfuller(values, maxlag=1, autolag=None)[1])
    except Exception:
        return np.nan


def ols_beta(y, x):
    yc = y - np.mean(y)
    xc = x - np.mean(x)
    return float(np.dot(yc, xc) / max(np.dot(xc, xc), 1e-18))


def spread_half_life(spread):
    lag = spread[:-1]
    delta = np.diff(spread)
    if len(lag) < 10 or np.std(lag) < 1e-12:
        return np.inf
    lag = lag - lag.mean()
    delta = delta - delta.mean()
    kappa = float(np.dot(lag, delta) / max(np.dot(lag, lag), 1e-18))
    return -np.log(2.0) / kappa if kappa < 0.0 else np.inf


def endpoint_z(log_prices, i, j, end, lookback=50, center=30):
    start = max(0, end - lookback)
    y = log_prices[start:end, i]
    x = log_prices[start:end, j]
    if len(y) < max(center + 1, 20):
        return np.nan, np.nan
    beta = ols_beta(y, x)
    spread = y - beta * x
    hist = spread[-(center + 1):-1]
    z = (spread[-1] - hist.mean()) / (hist.std() + 1e-12)
    return float(z), beta


def causal_forward_trade(
    log_prices,
    i,
    j,
    end,
    beta,
    center_window=30,
    entry_z=1.5,
    exit_z=0.25,
    max_z=4.0,
    horizon=FORWARD_DAYS,
):
    """Trade a fixed-beta spread after ``end`` using only trailing centres."""
    n_days = len(log_prices)
    last_move_start = min(end + horizon - 2, n_days - 2)
    if end < center_window + 1 or end >= n_days or last_move_start < end - 1:
        return {
            "forward_bps": np.nan,
            "n_trades": 0,
            "trade_hit_rate": np.nan,
            "worst_trade_bps": np.nan,
        }

    spread = log_prices[:, i] - beta * log_prices[:, j]
    position = 0.0
    held = 0
    trade_pnl = 0.0
    completed = []

    # q is the known close; a position chosen at q earns the move q -> q+1.
    for q in range(end - 1, last_move_start + 1):
        hist = spread[max(0, q - center_window):q]
        if len(hist) < center_window:
            continue
        mu = hist.mean()
        sigma = hist.std() + 1e-12
        z = (spread[q] - mu) / sigma

        if position == 0.0 and entry_z <= abs(z) < max_z:
            position = -np.sign(z)
            held = 0
            trade_pnl = -ROUND_TRIP_COST

        if position == 0.0:
            continue

        trade_pnl += position * (spread[q + 1] - spread[q]) / (1.0 + abs(beta))
        held += 1

        next_hist = spread[max(0, q + 1 - center_window):q + 1]
        next_mu = next_hist.mean()
        next_sigma = next_hist.std() + 1e-12
        next_z = (spread[q + 1] - next_mu) / next_sigma
        crossed = (position > 0.0 and next_z >= -exit_z) or (
            position < 0.0 and next_z <= exit_z
        )
        final_move = q == last_move_start
        if crossed or held >= MAX_HOLD or final_move:
            completed.append(trade_pnl)
            position = 0.0

    if not completed:
        return {
            "forward_bps": 0.0,
            "n_trades": 0,
            "trade_hit_rate": np.nan,
            "worst_trade_bps": np.nan,
        }
    pnl = np.asarray(completed)
    return {
        "forward_bps": float(1e4 * pnl.sum()),
        "n_trades": int(len(pnl)),
        "trade_hit_rate": float(np.mean(pnl > 0.0)),
        "worst_trade_bps": float(1e4 * pnl.min()),
    }


def pair_strength_table(blocks, names):
    rows = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            pvalues = np.array([block.iloc[i, j] for block in blocks], dtype=float)
            strength = -np.log10(np.clip(pvalues, 1e-16, 1.0))
            peak = int(np.argmax(strength))
            rows.append(
                {
                    "pair": f"{a}-{b}",
                    "ticker_a": a,
                    "ticker_b": b,
                    "i": i,
                    "j": j,
                    "n_sig_blocks": int(np.sum(pvalues < SIGNIFICANCE)),
                    "n_spike_blocks": int(np.sum(pvalues < SPIKE_P)),
                    "max_strength": float(strength[peak]),
                    "peak_block": peak + 1,
                    "peak_day_start": peak * BLOCK_SIZE + 1,
                    "peak_day_end": min((peak + 1) * BLOCK_SIZE, 750),
                    "peak_p": float(pvalues[peak]),
                    **{f"strength_b{k + 1:02d}": float(v) for k, v in enumerate(strength)},
                }
            )
    return pd.DataFrame(rows)


def add_pair_forward_metrics(pair_table, log_prices, blocks):
    records = []
    for row in pair_table.itertuples(index=False):
        end = row.peak_day_end
        z, beta = endpoint_z(log_prices, row.i, row.j, end)
        if np.isfinite(beta) and end < len(log_prices):
            forward = causal_forward_trade(log_prices, row.i, row.j, end, beta)
            next_p = (
                float(blocks[row.peak_block].iloc[row.i, row.j])
                if row.peak_block < len(blocks)
                else np.nan
            )
        else:
            forward = {
                "forward_bps": np.nan,
                "n_trades": 0,
                "trade_hit_rate": np.nan,
                "worst_trade_bps": np.nan,
            }
            next_p = np.nan
        records.append(
            {
                "peak_z": z,
                "peak_beta": beta,
                "next_block_p": next_p,
                "survives_next_block": bool(next_p < SIGNIFICANCE) if np.isfinite(next_p) else False,
                **forward,
            }
        )
    return pd.concat([pair_table.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def hub_table(prices, blocks):
    names = prices.columns.tolist()
    log_prices = np.log(prices.to_numpy(float))
    rows = []
    for block_idx, block in enumerate(blocks):
        start = block_idx * BLOCK_SIZE
        end = min((block_idx + 1) * BLOCK_SIZE, len(prices))
        p = block.to_numpy(float)
        strength = -np.log10(np.clip(p, 1e-16, 1.0))
        sig = p < SIGNIFICANCE
        ultra = p < SPIKE_P
        np.fill_diagonal(sig, False)
        np.fill_diagonal(ultra, False)

        for i, asset in enumerate(names):
            links = np.where(sig[i])[0]
            ranked = links[np.argsort(p[i, links])][:HUB_LINKS_PER_ASSET]
            forward_results = []
            for j in ranked:
                _, beta = endpoint_z(log_prices, i, j, end)
                if np.isfinite(beta) and end < len(prices):
                    forward_results.append(causal_forward_trade(log_prices, i, j, end, beta))

            forward_bps = [r["forward_bps"] for r in forward_results]
            n_trades = [r["n_trades"] for r in forward_results]
            next_survival = np.nan
            if links.size and block_idx + 1 < len(blocks):
                next_p = blocks[block_idx + 1].to_numpy(float)[i, links]
                next_survival = float(np.mean(next_p < SIGNIFICANCE))

            asset_window = log_prices[start:end, i]
            rows.append(
                {
                    "block": block_idx + 1,
                    "day_start": start + 1,
                    "day_end": end,
                    "asset": asset,
                    "asset_index": i,
                    "degree": int(sig[i].sum()),
                    "spike_degree": int(ultra[i].sum()),
                    "mean_link_strength": float(strength[i, links].mean()) if links.size else 0.0,
                    "max_link_strength": float(strength[i, links].max()) if links.size else 0.0,
                    "asset_adf_p": safe_adf_p(asset_window),
                    "next_link_survival": next_survival,
                    "forward_pairs_tested": int(len(forward_results)),
                    "forward_pairs_traded": int(np.sum(np.asarray(n_trades) > 0)) if n_trades else 0,
                    "forward_mean_bps": float(np.mean(forward_bps)) if forward_bps else np.nan,
                    "forward_hit_rate": float(np.mean(np.asarray(forward_bps) > 0.0)) if forward_bps else np.nan,
                }
            )
    return pd.DataFrame(rows)


def append_runtime_event(rows, detector, end, i, j, beta, z, log_prices, blocks):
    result = causal_forward_trade(
        log_prices,
        i,
        j,
        end,
        beta,
        center_window=60 if detector == "stable_core" else 30,
    )
    block_idx = end // BLOCK_SIZE - 1
    next_p = (
        float(blocks[block_idx + 1].iloc[i, j])
        if block_idx + 1 < len(blocks)
        else np.nan
    )
    rows.append(
        {
            "detector": detector,
            "day": end,
            "period": "holdout" if end >= 500 else "pre500",
            "pair": f"{blocks[0].index[i]}-{blocks[0].index[j]}",
            "ticker_a": blocks[0].index[i],
            "ticker_b": blocks[0].index[j],
            "z_at_detection": z,
            "beta": beta,
            "next_block_p": next_p,
            "survives_next_block": bool(next_p < SIGNIFICANCE) if np.isfinite(next_p) else False,
            **result,
        }
    )


def confirmed_episode_candidates(LP, idle, used, level_t):
    """Strict, causal all-pair short-window detector used for comparison."""
    fast = LP[:, -EPISODE_LOOKBACK:]
    prior = LP[:, -(EPISODE_LOOKBACK + EPISODE_CONFIRM_GAP):-EPISODE_CONFIRM_GAP]

    beta, spread = strategy._pair_regression(fast, _ALL_PI, _ALL_PJ)
    _, reverse = strategy._pair_regression(fast, _ALL_PJ, _ALL_PI)
    pair_t = np.maximum(strategy._adf_tstat(spread.T), strategy._adf_tstat(reverse.T))

    _, spread_prior = strategy._pair_regression(prior, _ALL_PI, _ALL_PJ)
    _, reverse_prior = strategy._pair_regression(prior, _ALL_PJ, _ALL_PI)
    prior_t = np.maximum(
        strategy._adf_tstat(spread_prior.T),
        strategy._adf_tstat(reverse_prior.T),
    )

    ratio = strategy._beta_ratio(fast, _ALL_PI, _ALL_PJ)
    half_life = strategy._spread_half_life(spread.T)
    z = strategy._local_z(spread, EPISODE_CENTER)
    ok = (
        idle[_ALL_PI] & idle[_ALL_PJ] & ~used[_ALL_PI] & ~used[_ALL_PJ]
        & (level_t[_ALL_PI] > EPISODE_ASSET_ADF_T)
        & (level_t[_ALL_PJ] > EPISODE_ASSET_ADF_T)
        & (beta > 0.0) & (ratio <= EPISODE_BETA_RATIO)
        & (pair_t < EPISODE_ADF_T) & (prior_t < EPISODE_CONFIRM_T)
        & (half_life >= EPISODE_HL_MIN) & (half_life <= EPISODE_HL_MAX)
        & (np.abs(z) >= EPISODE_Z_ENTRY) & (np.abs(z) < EPISODE_Z_MAX)
    )
    selected = np.where(ok)[0]
    rank = pair_t[selected] + 0.5 * prior_t[selected] - 0.05 * np.abs(z[selected])
    return selected[np.argsort(rank)], beta, z


def runtime_event_table(prices, blocks, hub_metrics):
    names = prices.columns.tolist()
    log_prices = np.log(prices.to_numpy(float))
    prices_t = prices.to_numpy(float).T
    rows = []

    for end in range(BLOCK_SIZE, len(prices), BLOCK_SIZE):
        block_idx = end // BLOCK_SIZE - 1
        p = blocks[block_idx].to_numpy(float)

        # Naive interpretation of a bright matrix cell: raw p < 0.001.
        for i in range(1, len(names)):
            for j in range(i + 1, len(names)):
                if p[i, j] >= SPIKE_P:
                    continue
                z, beta = endpoint_z(log_prices, i, j, end)
                append_runtime_event(rows, "naive_spike", end, i, j, beta, z, log_prices, blocks)

        # Row-hub interpretation: activate the five strongest links per hub.
        hubs = hub_metrics[
            (hub_metrics["block"] == block_idx + 1)
            & (hub_metrics["degree"] >= HUB_DEGREE)
            & (hub_metrics["asset"] != "ALGO")
        ]
        hub_pairs = set()
        for hub in hubs.itertuples(index=False):
            i = int(hub.asset_index)
            links = np.where(p[i] < SIGNIFICANCE)[0]
            links = links[np.argsort(p[i, links])][:HUB_LINKS_PER_ASSET]
            for j in links:
                if i == j or j == 0:
                    continue
                a, b = sorted((i, int(j)))
                hub_pairs.add((a, b))
        for i, j in sorted(hub_pairs):
            z, beta = endpoint_z(log_prices, i, j, end)
            append_runtime_event(rows, "row_hub", end, i, j, beta, z, log_prices, blocks)

        if end < strategy.PAIR_LOOKBACK + 2:
            continue
        history = prices_t[:, :end]
        LP = np.log(history[:, -strategy.PAIR_LOOKBACK:])
        level_t = strategy._adf_tstat(LP.T)
        idle = np.ones(len(names), dtype=bool)
        idle[0] = False
        used = np.zeros(len(names), dtype=bool)

        core, core_beta, core_z = strategy._core_pair_candidates(LP, idle, level_t)
        for k in core:
            i = int(strategy._CORE_PI[k])
            j = int(strategy._CORE_PJ[k])
            append_runtime_event(
                rows, "stable_core", end, i, j,
                float(core_beta[k]), float(core_z[k]), log_prices, blocks,
            )
            used[i] = used[j] = True

    # The strict episodic detector is intended to run more frequently than the
    # reporting blocks.  Check every five days and collapse repeated signals for
    # the same pair inside one MAX_HOLD interval into a single episode.
    last_episode = {}
    for end in range(strategy.PAIR_LOOKBACK + 2, len(prices), 5):
        history = prices_t[:, :end]
        LP = np.log(history[:, -strategy.PAIR_LOOKBACK:])
        level_t = strategy._adf_tstat(LP.T)
        idle = np.ones(len(names), dtype=bool)
        idle[0] = False
        used = np.zeros(len(names), dtype=bool)

        core, _, _ = strategy._core_pair_candidates(LP, idle, level_t)
        for k in core:
            used[int(strategy._CORE_PI[k])] = True
            used[int(strategy._CORE_PJ[k])] = True

        candidates, beta, z = confirmed_episode_candidates(LP, idle, used, level_t)
        accepted = 0
        for k in candidates:
            i, j = int(_ALL_PI[k]), int(_ALL_PJ[k])
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            key = (i, j)
            if end - last_episode.get(key, -10_000) >= MAX_HOLD:
                append_runtime_event(
                    rows, "confirmed_episode", end, i, j,
                    float(beta[k]), float(z[k]), log_prices, blocks,
                )
                last_episode[key] = end
            accepted += 1
            if accepted >= EPISODE_MAX:
                break

    return pd.DataFrame(rows)


def detector_summary(events):
    rows = []
    for detector, group in events.groupby("detector"):
        traded = group[group["n_trades"] > 0]
        rows.append(
            {
                "detector": detector,
                "detections": len(group),
                "unique_pairs": group["pair"].nunique(),
                "tradeable_detections": len(traded),
                "opportunity_rate": len(traded) / len(group) if len(group) else np.nan,
                "mean_forward_bps": traded["forward_bps"].mean() if len(traded) else np.nan,
                "median_forward_bps": traded["forward_bps"].median() if len(traded) else np.nan,
                "worst_forward_bps": traded["forward_bps"].min() if len(traded) else np.nan,
                "profitable_rate": np.mean(traded["forward_bps"] > 0.0) if len(traded) else np.nan,
                "next_block_survival": group["survives_next_block"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("detector").reset_index(drop=True)


def plot_transient_heatmap(transient, n_blocks, path):
    plot_df = transient.sort_values("max_strength", ascending=False).head(TOP_TRANSIENT)
    cols = [f"strength_b{k:02d}" for k in range(1, n_blocks + 1)]
    data = plot_df[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(15, max(9, 0.28 * len(plot_df))))
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=max(4.0, np.nanpercentile(data, 99)))
    ax.set_xticks(np.arange(n_blocks), [f"B{k}\n{(k-1)*50+1}-{min(k*50,750)}" for k in range(1, n_blocks + 1)], fontsize=8)
    ax.set_yticks(np.arange(len(plot_df)), plot_df["pair"], fontsize=8)
    ax.set_xlabel("50-day block")
    ax.set_title("Transient pair strength: large values are small raw p-values\nPairs significant in at most two blocks", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cointegration strength (-log10 raw p)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hub_heatmap(hubs, names, n_blocks, path):
    degree = hubs.pivot(index="asset", columns="block", values="degree").reindex(names)
    order = degree.max(axis=1).sort_values(ascending=False).index
    degree = degree.loc[order]
    fig, ax = plt.subplots(figsize=(15, 14))
    im = ax.imshow(degree.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(n_blocks), [f"B{k}\n{(k-1)*50+1}-{min(k*50,750)}" for k in range(1, n_blocks + 1)], fontsize=8)
    ax.set_yticks(np.arange(len(degree)), degree.index, fontsize=8)
    ax.set_title("Row-hub degree by block\nNumber of other assets with raw p < 0.05", fontweight="bold")
    ax.set_xlabel("50-day block")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Significant links in row")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_case_studies(transient, n_blocks, path):
    cases = transient[
        (transient["peak_day_end"] < 750) & transient["forward_bps"].notna()
    ].sort_values("max_strength", ascending=False).head(TOP_CASES)
    rows = int(np.ceil(len(cases) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(17, 3.5 * rows), squeeze=False)
    x = np.arange(1, n_blocks + 1)
    for ax, row in zip(axes.flat, cases.itertuples(index=False)):
        strength = np.array([getattr(row, f"strength_b{k:02d}") for k in x])
        ax.plot(x, strength, marker="o", ms=3, color="#6a3d9a")
        ax.axhline(-np.log10(SIGNIFICANCE), color="#777777", ls="--", lw=1)
        ax.axhline(-np.log10(SPIKE_P), color="#d62728", ls=":", lw=1)
        ax.axvspan(row.peak_block - 0.45, row.peak_block + 0.45, color="#ffcc66", alpha=0.35)
        ax.set_xticks(x)
        ax.set_xlim(0.5, n_blocks + 0.5)
        ax.set_ylabel("-log10(p)")
        ax.set_title(
            f"{row.pair} | peak B{row.peak_block} | next PnL {row.forward_bps:+.0f} bp",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.suptitle("Short-lived high-strength pair case studies", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hub_diagnostics(hubs, path):
    valid = hubs[(hubs["degree"] > 0) & hubs["next_link_survival"].notna()].copy()
    high = valid[valid["degree"] >= HUB_DEGREE].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    color = -np.log10(valid["asset_adf_p"].clip(1e-16))
    sc = axes[0, 0].scatter(valid["degree"], valid["next_link_survival"] * 100, c=color, cmap="plasma", alpha=0.7)
    axes[0, 0].set_xlabel("Current row degree")
    axes[0, 0].set_ylabel("Links still p<0.05 next block (%)")
    axes[0, 0].set_title("Hub size versus next-block persistence")
    fig.colorbar(sc, ax=axes[0, 0], label="Asset stationarity strength (-log10 ADF p)")

    size = 30 + 35 * high["forward_pairs_traded"].fillna(0)
    axes[0, 1].scatter(high["degree"], high["forward_mean_bps"], s=size, c=high["spike_degree"], cmap="viridis", alpha=0.75)
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_xlabel(f"Hub degree (degree >= {HUB_DEGREE})")
    axes[0, 1].set_ylabel("Forward mean PnL of five strongest links (bp)")
    axes[0, 1].set_title("A bright row is not automatically profitable")

    top = high.sort_values("degree", ascending=False).head(20).copy()
    top["label"] = top["asset"] + " B" + top["block"].astype(str)
    top = top.iloc[::-1]
    norm = plt.Normalize(vmin=np.nanmin(top["forward_mean_bps"]), vmax=np.nanmax(top["forward_mean_bps"]))
    axes[1, 0].barh(top["label"], top["degree"], color=plt.cm.RdYlGn(norm(top["forward_mean_bps"])))
    axes[1, 0].set_xlabel("Significant links")
    axes[1, 0].set_title("Largest row-hub episodes (colour = forward PnL)")

    valid["degree_band"] = pd.cut(valid["degree"], [0, 4, 9, 19, np.inf], labels=["1-4", "5-9", "10-19", "20+"])
    groups = [valid.loc[valid["degree_band"] == label, "forward_mean_bps"].dropna() for label in ["1-4", "5-9", "10-19", "20+"]]
    axes[1, 1].boxplot(groups, tick_labels=["1-4", "5-9", "10-19", "20+"], showfliers=False)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_xlabel("Row-degree band")
    axes[1, 1].set_ylabel("Forward mean PnL of strongest links (bp)")
    axes[1, 1].set_title("Hub degree versus forward opportunity distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_detector_comparison(summary, path):
    order = ["naive_spike", "row_hub", "confirmed_episode", "stable_core"]
    plot_df = summary.set_index("detector").reindex(order).dropna(how="all").reset_index()
    labels = plot_df["detector"].str.replace("_", " ")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(labels, plot_df["detections"], color="#4c78a8")
    axes[0, 0].set_title("Runtime detections")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 1].bar(labels, 100 * plot_df["opportunity_rate"], color="#72b7b2")
    axes[0, 1].set_title("Detections that produced a trade (%)")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[1, 0].bar(labels, plot_df["mean_forward_bps"], color="#54a24b")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_title("Mean forward PnL per traded detection (bp)")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].bar(labels, 100 * plot_df["profitable_rate"], color="#f58518")
    axes[1, 1].set_title("Profitable traded detections (%)")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.suptitle("Causal runtime detector comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_detector_period_comparison(summary, path):
    order = ["naive_spike", "row_hub", "confirmed_episode", "stable_core"]
    labels = [name.replace("_", " ") for name in order]
    x = np.arange(len(order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    colours = {"pre500": "#4c78a8", "holdout": "#f58518"}
    for offset, period in [(-width / 2, "pre500"), (width / 2, "holdout")]:
        p = summary[summary["period"] == period].set_index("detector").reindex(order)
        axes[0].bar(x + offset, p["mean_forward_bps"], width, label=period, color=colours[period])
        axes[1].bar(x + offset, 100 * p["profitable_rate"], width, label=period, color=colours[period])
        for xi, value, count in zip(x + offset, p["mean_forward_bps"], p["detections"]):
            if np.isfinite(value):
                axes[0].text(xi, value, f"n={int(count)}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Mean forward PnL per traded detection (bp)")
    axes[0].set_title("Forward PnL: formation period versus holdout")
    axes[1].set_ylabel("Profitable detections (%)")
    axes[1].set_title("Hit rate: formation period versus holdout")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices, blocks = load_data()
    names = prices.columns.tolist()
    log_prices = np.log(prices.to_numpy(float))

    pairs = pair_strength_table(blocks, names)
    pairs = add_pair_forward_metrics(pairs, log_prices, blocks)
    transient = pairs[
        (pairs["n_sig_blocks"] <= TRANSIENT_MAX_SIG_BLOCKS)
        & (pairs["max_strength"] >= -np.log10(SPIKE_P))
    ].sort_values("max_strength", ascending=False)

    hubs = hub_table(prices, blocks)
    events = runtime_event_table(prices, blocks, hubs)
    summary = detector_summary(events)
    period_summary = []
    for period, group in events.groupby("period"):
        part = detector_summary(group)
        part.insert(1, "period", period)
        period_summary.append(part)
    period_summary = pd.concat(period_summary, ignore_index=True)

    pairs.to_csv(OUTPUT_DIR / "all_pair_block_metrics.csv", index=False)
    transient.to_csv(OUTPUT_DIR / "transient_pair_metrics.csv", index=False)
    hubs.to_csv(OUTPUT_DIR / "hub_block_metrics.csv", index=False)
    events.to_csv(OUTPUT_DIR / "runtime_detector_events.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "runtime_detector_summary.csv", index=False)
    period_summary.to_csv(OUTPUT_DIR / "runtime_detector_summary_by_period.csv", index=False)

    plot_transient_heatmap(transient, len(blocks), OUTPUT_DIR / "transient_pair_strength_heatmap.png")
    plot_hub_heatmap(hubs, names, len(blocks), OUTPUT_DIR / "hub_degree_heatmap.png")
    plot_case_studies(transient, len(blocks), OUTPUT_DIR / "transient_pair_case_studies.png")
    plot_hub_diagnostics(hubs, OUTPUT_DIR / "hub_episode_diagnostics.png")
    plot_detector_comparison(summary, OUTPUT_DIR / "runtime_detector_comparison.png")
    plot_detector_period_comparison(
        period_summary,
        OUTPUT_DIR / "runtime_detector_period_comparison.png",
    )

    rho_persistence = spearmanr(
        hubs.loc[hubs["next_link_survival"].notna(), "degree"],
        hubs.loc[hubs["next_link_survival"].notna(), "next_link_survival"],
    )
    hub_forward = hubs[(hubs["degree"] >= HUB_DEGREE) & hubs["forward_mean_bps"].notna()]
    rho_pnl = spearmanr(hub_forward["degree"], hub_forward["forward_mean_bps"])

    print(f"Saved analysis to {OUTPUT_DIR}")
    print(f"Transient high-strength pairs: {len(transient)}")
    print(f"Hub episodes (degree >= {HUB_DEGREE}): {(hubs['degree'] >= HUB_DEGREE).sum()}")
    print(
        "Spearman hub degree vs next-block link survival: "
        f"rho={rho_persistence.statistic:.3f}, p={rho_persistence.pvalue:.4g}"
    )
    print(
        "Spearman high-hub degree vs forward PnL: "
        f"rho={rho_pnl.statistic:.3f}, p={rho_pnl.pvalue:.4g}"
    )
    print("\nRuntime detector comparison:")
    print(summary.round(3).to_string(index=False))
    print("\nRuntime detector comparison by period:")
    print(period_summary.round(3).to_string(index=False))
    print("\nTop transient pairs:")
    print(
        transient[
            ["pair", "peak_block", "peak_p", "max_strength", "peak_z", "forward_bps", "next_block_p"]
        ].head(20).round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
