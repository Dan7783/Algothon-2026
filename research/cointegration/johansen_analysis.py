#!/usr/bin/env python
"""
Johansen multivariate cointegration on 100-day blocks.

Full-system Johansen on 51 assets with T=100 is not statistically valid,
so we form multi-asset groups (size 3-8) from pairwise Engle-Granger
connected components / neighborhoods, then run Johansen on each group.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

BLOCK_SIZE = 100
SIGNIFICANCE = 0.05
MIN_GROUP = 3
MAX_GROUP = 8
DET_ORDER = 0  # constant term
K_AR_DIFF = 1
OUTPUT_DIR = Path("cointegration_output/johansen_100")


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def split_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE):
    blocks = []
    n_days = len(df)
    for start in range(0, n_days, block_size):
        end = min(start + block_size, n_days)
        blocks.append((start, end, df.iloc[start:end].copy()))
    return blocks


def pairwise_significance(block: pd.DataFrame) -> pd.DataFrame:
    tickers = block.columns.tolist()
    n = len(tickers)
    sig = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            y, x = block.iloc[:, i].values, block.iloc[:, j].values
            if np.std(y) < 1e-12 or np.std(x) < 1e-12:
                continue
            _, pvalue, _ = coint(y, x)
            if pvalue < SIGNIFICANCE:
                sig[i, j] = sig[j, i] = True
    return pd.DataFrame(sig, index=tickers, columns=tickers)


def johansen_rank(data: np.ndarray) -> dict:
    """Return cointegration rank from trace test at 5%."""
    nobs, k = data.shape
    if nobs <= k + K_AR_DIFF + 2:
        return {"rank": 0, "trace_stats": [], "crit_5pct": [], "ok": False, "reason": "too few obs"}

    try:
        res = coint_johansen(data, det_order=DET_ORDER, k_ar_diff=K_AR_DIFF)
    except Exception as e:
        return {"rank": 0, "trace_stats": [], "crit_5pct": [], "ok": False, "reason": str(e)}

    # Trace test: H0: r <= i. Reject if trace_stat > 5% critical value (column index 1)
    # crit vals columns: 90%, 95%, 99%
    rank = 0
    for i in range(k):
        if res.trace_stat[i] > res.trace_stat_crit_vals[i, 1]:
            rank = i + 1
        else:
            break

    return {
        "rank": rank,
        "trace_stats": res.trace_stat.tolist(),
        "crit_5pct": res.trace_stat_crit_vals[:, 1].tolist(),
        "max_eig_stats": res.max_eig_stat.tolist(),
        "eigenvectors": res.evec,
        "ok": True,
        "reason": "",
    }


def build_groups(sig_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """
    Build multi-asset groups from the pairwise cointegration graph:
    1) connected components of size 3-8
    2) ego neighborhoods of high-degree hubs (size clipped to MAX_GROUP)
    """
    G = nx.Graph()
    tickers = sig_df.index.tolist()
    G.add_nodes_from(tickers)
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            if sig_df.loc[a, b]:
                G.add_edge(a, b)

    groups: dict[tuple[str, ...], str] = {}

    # Connected components
    for comp in nx.connected_components(G):
        members = sorted(comp)
        if MIN_GROUP <= len(members) <= MAX_GROUP:
            groups[tuple(members)] = "component"
        elif len(members) > MAX_GROUP:
            # Split oversized components via hub neighborhoods
            sub = G.subgraph(members)
            degrees = sorted(sub.degree, key=lambda x: x[1], reverse=True)
            for hub, deg in degrees[:5]:
                if deg + 1 < MIN_GROUP:
                    continue
                nbrs = sorted(sub.neighbors(hub), key=lambda n: sub.degree(n), reverse=True)
                members_hub = sorted([hub] + nbrs[: MAX_GROUP - 1])
                if len(members_hub) >= MIN_GROUP:
                    key = tuple(members_hub)
                    groups.setdefault(key, "hub_neighborhood")

    # Also add ego neighborhoods for top hubs in the full graph
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    for hub, deg in degrees[:10]:
        if deg + 1 < MIN_GROUP:
            continue
        nbrs = sorted(G.neighbors(hub), key=lambda n: G.degree(n), reverse=True)
        members_hub = sorted([hub] + nbrs[: MAX_GROUP - 1])
        if len(members_hub) >= MIN_GROUP:
            key = tuple(members_hub)
            groups.setdefault(key, "hub_neighborhood")

    return [(label, list(members)) for members, label in groups.items()]


def run_block(block_idx: int, day_start: int, day_end: int, block: pd.DataFrame) -> pd.DataFrame:
    print(f"Block {block_idx + 1}: days {day_start + 1}-{day_end}")
    sig_df = pairwise_significance(block)
    groups = build_groups(sig_df)
    print(f"  Found {len(groups)} multi-asset groups (size {MIN_GROUP}-{MAX_GROUP})")

    rows = []
    for g_idx, (gtype, members) in enumerate(groups):
        data = block[members].values.astype(float)
        result = johansen_rank(data)
        rows.append(
            {
                "block": block_idx + 1,
                "day_start": day_start + 1,
                "day_end": day_end,
                "group_id": g_idx + 1,
                "group_type": gtype,
                "n_assets": len(members),
                "assets": ",".join(members),
                "rank": result["rank"],
                "has_coint": result["rank"] >= 1,
                "ok": result["ok"],
                "reason": result["reason"],
                "trace_r0": result["trace_stats"][0] if result["trace_stats"] else np.nan,
                "crit_r0": result["crit_5pct"][0] if result["crit_5pct"] else np.nan,
            }
        )
        status = f"rank={result['rank']}" if result["ok"] else f"FAIL:{result['reason']}"
        print(f"    G{g_idx + 1} [{gtype}] n={len(members)} {members}: {status}")

    return pd.DataFrame(rows)


def plot_rank_heatmap(all_results: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: blocks x groups showing cointegration rank."""
    # Build a long-form list of groups per block for a grid of bar charts
    blocks = sorted(all_results["block"].unique())
    n_blocks = len(blocks)
    cols = min(4, n_blocks)
    rows = (n_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, b in zip(axes.flat, blocks):
        sub = all_results[all_results["block"] == b].sort_values("rank", ascending=False)
        if sub.empty:
            ax.set_title(f"B{b}: no groups")
            ax.axis("off")
            continue
        colors = ["#2ca02c" if r >= 1 else "#d62728" for r in sub["rank"]]
        labels = [f"G{gid}\nn={n}" for gid, n in zip(sub["group_id"], sub["n_assets"])]
        ax.bar(range(len(sub)), sub["rank"], color=colors, edgecolor="white")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Johansen rank r")
        ax.set_ylim(0, max(sub["n_assets"].max(), 1) + 0.5)
        day_s, day_e = sub["day_start"].iloc[0], sub["day_end"].iloc[0]
        n_coint = int(sub["has_coint"].sum())
        ax.set_title(f"B{b}: d{day_s}-{day_e}\n{n_coint}/{len(sub)} groups r≥1", fontsize=10)
        ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.6)

    for ax in axes.flat[n_blocks:]:
        ax.axis("off")

    fig.suptitle(
        "Johansen cointegration rank by multi-asset group (100-day blocks)\n"
        f"Groups size {MIN_GROUP}-{MAX_GROUP} from pairwise EG graph | det_order={DET_ORDER}, k_ar_diff={K_AR_DIFF}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "johansen_rank_by_block.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_asset_participation(all_results: pd.DataFrame, tickers: list[str], out_dir: Path) -> None:
    """For each block, heatmap of which assets appear in cointegrating groups (r>=1)."""
    blocks = sorted(all_results["block"].unique())
    mat = np.zeros((len(blocks), len(tickers)))
    count_mat = np.zeros_like(mat)

    for bi, b in enumerate(blocks):
        sub = all_results[(all_results["block"] == b) & (all_results["has_coint"])]
        for _, row in sub.iterrows():
            for asset in row["assets"].split(","):
                if asset in tickers:
                    j = tickers.index(asset)
                    mat[bi, j] += row["rank"]
                    count_mat[bi, j] += 1

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})

    im0 = axes[0].imshow(mat, aspect="auto", cmap="YlOrRd")
    axes[0].set_yticks(range(len(blocks)))
    axes[0].set_yticklabels([f"B{b}" for b in blocks])
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=90, fontsize=7)
    axes[0].set_title("Sum of Johansen ranks for groups containing each asset (r≥1 groups only)")
    plt.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01, label="Σ rank")

    # Total appearances in cointegrating groups across blocks
    total_count = count_mat.sum(axis=0)
    axes[1].bar(range(len(tickers)), total_count, color="#4C72B0", edgecolor="white")
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=90, fontsize=7)
    axes[1].set_ylabel("# group appearances")
    axes[1].set_title("How often each asset sits in a multi-asset cointegrating group (across all blocks)")

    plt.tight_layout()
    path = out_dir / "johansen_asset_participation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_summary_overview(all_results: pd.DataFrame, out_dir: Path) -> None:
    """Summary bars: groups tested, groups with r>=1, mean rank per block."""
    blocks = sorted(all_results["block"].unique())
    n_groups = []
    n_coint = []
    mean_rank = []
    max_rank = []
    labels = []

    for b in blocks:
        sub = all_results[all_results["block"] == b]
        n_groups.append(len(sub))
        n_coint.append(int(sub["has_coint"].sum()))
        mean_rank.append(sub["rank"].mean() if len(sub) else 0)
        max_rank.append(sub["rank"].max() if len(sub) else 0)
        labels.append(f"B{b}\nd{sub['day_start'].iloc[0]}-{sub['day_end'].iloc[0]}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(blocks))
    axes[0].bar(x - 0.2, n_groups, width=0.4, label="# groups tested", color="#4C72B0")
    axes[0].bar(x + 0.2, n_coint, width=0.4, label="# with r≥1", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title("Multi-asset groups per block")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, mean_rank, color="#8172B2", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Mean Johansen rank")
    axes[1].set_title("Average cointegration rank")

    axes[2].bar(x, max_rank, color="#C44E52", edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Max Johansen rank")
    axes[2].set_title("Strongest multi-asset system (max r)")

    fig.suptitle("Johansen multivariate cointegration — 100-day block overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "johansen_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_example_trace(all_results: pd.DataFrame, prices: pd.DataFrame, out_dir: Path) -> None:
    """Pick the highest-rank group overall and show its trace test detail + normalized prices."""
    ok = all_results[all_results["ok"] & (all_results["rank"] >= 1)]
    if ok.empty:
        print("No cointegrating groups to plot as example.")
        return

    best = ok.sort_values(["rank", "n_assets"], ascending=[False, False]).iloc[0]
    members = best["assets"].split(",")
    block = prices.iloc[int(best["day_start"]) - 1 : int(best["day_end"])]
    data = block[members].values.astype(float)
    res = coint_johansen(data, det_order=DET_ORDER, k_ar_diff=K_AR_DIFF)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normalized prices
    normed = block[members] / block[members].iloc[0]
    for col in members:
        axes[0].plot(normed.index, normed[col], label=col, lw=1.2)
    axes[0].set_title(
        f"Best multi-asset system — Block {int(best['block'])} "
        f"(days {int(best['day_start'])}-{int(best['day_end'])})\n"
        f"rank r={int(best['rank'])}, n={int(best['n_assets'])}"
    )
    axes[0].set_ylabel("Price / first day")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_xlabel("Day index")

    # Trace statistics vs critical values
    k = len(members)
    x = np.arange(k)
    axes[1].bar(x - 0.15, res.trace_stat, width=0.3, label="Trace statistic", color="#4C72B0")
    axes[1].bar(x + 0.15, res.trace_stat_crit_vals[:, 1], width=0.3, label="5% critical value", color="#C44E52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"r≤{i}" for i in range(k)])
    axes[1].set_xlabel("Null hypothesis")
    axes[1].set_ylabel("Statistic")
    axes[1].set_title(f"Johansen trace test — reject H0 until r={int(best['rank'])}")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "johansen_best_group_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    # networkx may not be in requirements — fall back if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments")

    blocks = split_blocks(prices, BLOCK_SIZE)
    all_rows = []
    for block_idx, (day_start, day_end, block_df) in enumerate(blocks):
        # Skip very short final blocks for Johansen stability if needed
        if day_end - day_start < 40:
            print(f"Skipping block {block_idx + 1}: only {day_end - day_start} days")
            continue
        result = run_block(block_idx, day_start, day_end, block_df)
        all_rows.append(result)

    if not all_rows:
        print("No results.")
        return

    all_results = pd.concat(all_rows, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "johansen_groups.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'johansen_groups.csv'} ({len(all_results)} groups)")

    summary = (
        all_results.groupby("block")
        .agg(
            n_groups=("group_id", "count"),
            n_coint=("has_coint", "sum"),
            mean_rank=("rank", "mean"),
            max_rank=("rank", "max"),
            mean_n_assets=("n_assets", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "johansen_summary.csv", index=False)
    print("\n=== Block summary ===")
    print(summary.to_string(index=False))

    plot_summary_overview(all_results, OUTPUT_DIR)
    plot_rank_heatmap(all_results, OUTPUT_DIR)
    plot_asset_participation(all_results, prices.columns.tolist(), OUTPUT_DIR)
    plot_example_trace(all_results, prices, OUTPUT_DIR)

    top = all_results[all_results["has_coint"]].sort_values("rank", ascending=False).head(15)
    print("\nTop cointegrating multi-asset groups:")
    print(top[["block", "day_start", "day_end", "n_assets", "rank", "group_type", "assets"]].to_string(index=False))


if __name__ == "__main__":
    main()
