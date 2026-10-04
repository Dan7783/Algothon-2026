#!/usr/bin/env python
"""Combinatorial Purged Cross-Validation (Lopez de Prado) for a fixed book.

Implements the CPCV path construction from *Advances in Financial Machine
Learning* (Ch. 12): partition the scored day range into N contiguous groups,
form every combination of k test groups, and score each combinatorial path
after purging / embargoing observations adjacent to the test set.

Because this strategy's parameters are frozen (already selected), each path
scores the causal daily PnL series on its test groups only. Purge and embargo
still remove boundary days from path scoring so serial dependence between
adjacent groups does not inflate the path distribution.
"""

from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import importlib
import json
import math
import sys
import types
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_strategy(module_name: str):
    path = ROOT / f"{module_name}.py"
    if not path.exists():
        return importlib.import_module(module_name)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def score(mu: float, sigma: float, param: float = 1.0) -> float:
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = math.sqrt(250.0) * mu / sigma
    frac = (sr * sr) / (sr * sr + param * param)
    return float(mu * frac)


def charge_fees(dvolumes: np.ndarray, comm_rate: np.ndarray) -> float:
    return float(np.sum(dvolumes * comm_rate))


def load_prices(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def calc_daily_pnl(
    prices: np.ndarray,
    get_position,
    reset,
    start_day: int,
    end_day: int,
    comm_rate: np.ndarray,
    dlr_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Official-eval daily PnL for days [start_day, end_day]."""
    n_inst = prices.shape[0]
    if reset is not None:
        reset()

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    days: list[int] = []
    pll: list[float] = []

    for t in range(start_day - 1, end_day + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]
        if t < end_day:
            raw = get_position(hist)
            limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(raw, -limits, limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= float(cur_prices.dot(delta) + comm)
        dvolumes = cur_prices * np.abs(delta)
        tot_dvolume += float(np.sum(dvolumes))
        comm = charge_fees(dvolumes, comm_rate)
        cur_pos = new_pos
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t >= start_day:
            days.append(t)
            pll.append(today_pl)

    return np.asarray(days, dtype=int), np.asarray(pll, dtype=float)


def split_into_groups(days: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Partition chronological days into N contiguous groups (LdP CPCV)."""
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if len(days) < n_groups:
        raise ValueError("Not enough days to form the requested groups")
    edges = np.linspace(0, len(days), n_groups + 1, dtype=int)
    groups = []
    for i in range(n_groups):
        chunk = days[edges[i] : edges[i + 1]]
        if len(chunk) == 0:
            raise ValueError(f"Group {i} is empty; reduce n_groups")
        groups.append(chunk)
    return groups


def embargo_width(n_obs: int, pct_embargo: float) -> int:
    return int(math.ceil(n_obs * max(0.0, pct_embargo)))


def purged_test_mask(
    days: np.ndarray,
    groups: list[np.ndarray],
    test_group_ids: tuple[int, ...],
    purge: int,
    pct_embargo: float,
) -> np.ndarray:
    """Boolean mask of days that remain in the CPCV test path.

    For each contiguous test block, drop `purge` days at each interior boundary
    with non-test neighbours, and drop an embargo of `ceil(T * pct)` days
    immediately after the block (Lopez de Prado Ch. 7 / 12).
    """
    day_to_pos = {int(d): i for i, d in enumerate(days)}
    keep = np.zeros(len(days), dtype=bool)
    for gid in test_group_ids:
        for d in groups[gid]:
            keep[day_to_pos[int(d)]] = True

    # Contiguous runs of selected test groups in chronological group order.
    selected = sorted(test_group_ids)
    runs: list[list[int]] = []
    for gid in selected:
        if not runs or gid != runs[-1][-1] + 1:
            runs.append([gid])
        else:
            runs[-1].append(gid)

    n_obs = len(days)
    emb = embargo_width(n_obs, pct_embargo)

    for run in runs:
        first_day = int(groups[run[0]][0])
        last_day = int(groups[run[-1]][-1])
        first_pos = day_to_pos[first_day]
        last_pos = day_to_pos[last_day]

        # Purge at the left edge when a train group precedes this run.
        if run[0] > 0 and purge > 0:
            keep[first_pos : first_pos + purge] = False
        # Purge at the right edge when a train group follows this run.
        if run[-1] < len(groups) - 1 and purge > 0:
            lo = max(first_pos, last_pos - purge + 1)
            keep[lo : last_pos + 1] = False
        # Embargo immediately after the test block (also zeroed from scoring).
        if emb > 0:
            keep[last_pos + 1 : last_pos + 1 + emb] = False

    return keep


def path_metrics(pll: np.ndarray) -> dict[str, float]:
    if pll.size == 0:
        return {
            "n_days": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sharpe": float("nan"),
            "score": float("nan"),
            "total_pnl": float("nan"),
        }
    mu = float(np.mean(pll))
    sigma = float(np.std(pll, ddof=0))
    sharpe = float(math.sqrt(250.0) * mu / sigma) if sigma > 1e-12 else 0.0
    return {
        "n_days": int(pll.size),
        "mean": mu,
        "std": sigma,
        "sharpe": sharpe,
        "score": score(mu, sigma),
        "total_pnl": float(np.sum(pll)),
    }


def run_cpcv(
    days: np.ndarray,
    pll: np.ndarray,
    n_groups: int,
    n_test_groups: int,
    purge: int,
    pct_embargo: float,
) -> dict[str, object]:
    groups = split_into_groups(days, n_groups)
    group_spans = [
        {"group": i, "start": int(g[0]), "end": int(g[-1]), "n": int(len(g))}
        for i, g in enumerate(groups)
    ]
    paths = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        mask = purged_test_mask(days, groups, test_ids, purge, pct_embargo)
        metrics = path_metrics(pll[mask])
        paths.append(
            {
                "test_groups": list(test_ids),
                "test_days": [int(d) for d in days[mask]],
                "n_scored_days": int(mask.sum()),
                **metrics,
            }
        )

    scores = np.asarray([p["score"] for p in paths], dtype=float)
    sharpes = np.asarray([p["sharpe"] for p in paths], dtype=float)
    means = np.asarray([p["mean"] for p in paths], dtype=float)

    def summarize(arr: np.ndarray) -> dict[str, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "p05": float("nan"),
                "p25": float("nan"),
                "p75": float("nan"),
                "p95": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "frac_positive": float("nan"),
            }
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "p05": float(np.percentile(finite, 5)),
            "p25": float(np.percentile(finite, 25)),
            "p75": float(np.percentile(finite, 75)),
            "p95": float(np.percentile(finite, 95)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "frac_positive": float(np.mean(finite > 0.0)),
        }

    # Per-group OOS coverage: how often each group appears in a test path.
    group_path_scores = {i: [] for i in range(n_groups)}
    for path in paths:
        for gid in path["test_groups"]:
            group_path_scores[gid].append(path["score"])

    return {
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "n_paths": len(paths),
        "purge": purge,
        "pct_embargo": pct_embargo,
        "embargo_days": embargo_width(len(days), pct_embargo),
        "group_spans": group_spans,
        "paths": paths,
        "score_summary": summarize(scores),
        "sharpe_summary": summarize(sharpes),
        "mean_pnl_summary": summarize(means),
        "group_mean_path_score": {
            str(i): float(np.mean(v)) if v else float("nan")
            for i, v in group_path_scores.items()
        },
    }


def plot_results(
    full: dict[str, float],
    cpcv: dict[str, object],
    days: np.ndarray,
    pll: np.ndarray,
    out_dir: Path,
    module: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = cpcv["paths"]
    scores = np.asarray([p["score"] for p in paths], dtype=float)
    sharpes = np.asarray([p["sharpe"] for p in paths], dtype=float)

    hist_path = out_dir / f"cpcv_score_hist_{module}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(scores, bins=min(20, max(8, len(scores) // 2)), color="#4C72B0", edgecolor="white")
    axes[0].axvline(full["score"], color="#C44E52", linewidth=2, label=f"full-window score {full['score']:.1f}")
    axes[0].axvline(cpcv["score_summary"]["median"], color="#55A868", linewidth=2, linestyle="--", label="CPCV median")
    axes[0].set_title("CPCV path scores")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Paths")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(sharpes, bins=min(20, max(8, len(sharpes) // 2)), color="#8172B2", edgecolor="white")
    axes[1].axvline(full["sharpe"], color="#C44E52", linewidth=2, label=f"full Sharpe {full['sharpe']:.2f}")
    axes[1].axvline(cpcv["sharpe_summary"]["median"], color="#55A868", linewidth=2, linestyle="--", label="CPCV median")
    axes[1].set_title("CPCV path Sharpes")
    axes[1].set_xlabel("Annualised Sharpe")
    axes[1].set_ylabel("Paths")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)

    heat_path = out_dir / f"cpcv_group_heatmap_{module}.png"
    n_groups = int(cpcv["n_groups"])
    mat = np.full((len(paths), n_groups), np.nan)
    for i, path in enumerate(paths):
        for gid in path["test_groups"]:
            mat[i, gid] = path["score"]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(paths) + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Group id (test cells coloured by path score)")
    ax.set_ylabel("CPCV path")
    ax.set_xticks(range(n_groups))
    labels = [
        f"G{g['group']}\n{g['start']}-{g['end']}" for g in cpcv["group_spans"]
    ]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Which groups are OOS on each combinatorial path")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Path score")
    fig.tight_layout()
    fig.savefig(heat_path, dpi=140)
    plt.close(fig)

    cum_path = out_dir / f"cpcv_cum_pnl_{module}.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cum = np.cumsum(pll)
    ax.plot(days, cum, color="#4C72B0", linewidth=1.8)
    y_max = float(np.max(cum)) if cum.size else 0.0
    for g in cpcv["group_spans"]:
        ax.axvline(g["start"], color="grey", linewidth=0.6, alpha=0.5)
        ax.text(
            g["start"] + 2,
            y_max,
            f"G{g['group']}",
            fontsize=7,
            color="grey",
            va="top",
        )
    ax.set_title("Full-window cumulative PnL with CPCV group boundaries")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cum_path, dpi=140)
    plt.close(fig)

    return {
        "score_hist": str(hist_path),
        "group_heatmap": str(heat_path),
        "cum_pnl": str(cum_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", default="friend_book_latest")
    p.add_argument("--prices", default="prices.txt")
    p.add_argument("--start", type=int, default=251)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--n_groups", type=int, default=6)
    p.add_argument("--n_test_groups", type=int, default=2)
    p.add_argument("--purge", type=int, default=10)
    p.add_argument("--pct_embargo", type=float, default=0.01)
    p.add_argument("--out", default="auto_tuning_results/cpcv_friend_book_latest.json")
    p.add_argument("--plot_dir", default="auto_tuning_results")
    args = p.parse_args()

    if not (1 <= args.n_test_groups < args.n_groups):
        raise SystemExit("Need 1 <= n_test_groups < n_groups")

    prices = load_prices(ROOT / args.prices)
    n_inst = prices.shape[0]
    strat = load_strategy(args.module)
    comm = np.full(n_inst, 0.0001)
    comm[0] = 0.00002
    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0

    print(
        f"Running causal backtest {args.module} days {args.start}-{args.end} "
        f"({prices.shape[1]} price days, {n_inst} instruments)..."
    )
    days, pll = calc_daily_pnl(
        prices,
        strat.getMyPosition,
        getattr(strat, "reset", None),
        args.start,
        args.end,
        comm,
        limits,
    )
    full = path_metrics(pll)
    print(
        f"Full window: n={full['n_days']} mean=${full['mean']:.2f} "
        f"std=${full['std']:.2f} Sharpe={full['sharpe']:.3f} score={full['score']:.2f}"
    )

    cpcv = run_cpcv(
        days,
        pll,
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        purge=args.purge,
        pct_embargo=args.pct_embargo,
    )
    plots = plot_results(full, cpcv, days, pll, ROOT / args.plot_dir, args.module)

    # Also report the conventional development/holdout slices for context.
    slice_defs = {
        "dev_251_750": (days >= 251) & (days <= 750),
        "val_501_750": (days >= 501) & (days <= 750),
        "oos_751_1000": (days >= 751) & (days <= 1000),
        "all_251_1000": np.ones(len(days), dtype=bool),
    }
    slices = {name: path_metrics(pll[mask]) for name, mask in slice_defs.items()}

    payload = {
        "module": args.module,
        "start": args.start,
        "end": args.end,
        "method": "combinatorial_purged_cv",
        "reference": "Lopez de Prado, Advances in Financial Machine Learning, Ch.12",
        "note": (
            "Parameters are frozen. CPCV scores combinatorial OOS paths on the "
            "causal daily PnL series; purge/embargo drop boundary days from each "
            "path so adjacent-group serial dependence does not inflate the "
            "path-score distribution."
        ),
        "full_window": full,
        "conventional_slices": slices,
        "cpcv": {
            k: v
            for k, v in cpcv.items()
            if k != "paths"
        },
        "paths": [
            {
                "test_groups": path["test_groups"],
                "n_scored_days": path["n_scored_days"],
                "mean": path["mean"],
                "std": path["std"],
                "sharpe": path["sharpe"],
                "score": path["score"],
                "total_pnl": path["total_pnl"],
                "first_day": path["test_days"][0] if path["test_days"] else None,
                "last_day": path["test_days"][-1] if path["test_days"] else None,
            }
            for path in cpcv["paths"]
        ],
        "plots": plots,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ss = cpcv["score_summary"]
    print("\n=== CPCV summary ===")
    print(
        f"N={args.n_groups} groups, k={args.n_test_groups} test/path, "
        f"paths={cpcv['n_paths']}, purge={args.purge}, "
        f"embargo={cpcv['embargo_days']}d ({100 * args.pct_embargo:.1f}%)"
    )
    for g in cpcv["group_spans"]:
        print(f"  G{g['group']}: days {g['start']}-{g['end']} ({g['n']}d)")
    print(
        f"Path scores: mean={ss['mean']:.2f} median={ss['median']:.2f} "
        f"std={ss['std']:.2f} p05={ss['p05']:.2f} p95={ss['p95']:.2f} "
        f"frac>0={ss['frac_positive']:.1%}"
    )
    print(
        f"Path Sharpes: mean={cpcv['sharpe_summary']['mean']:.3f} "
        f"median={cpcv['sharpe_summary']['median']:.3f} "
        f"frac>0={cpcv['sharpe_summary']['frac_positive']:.1%}"
    )
    print("Conventional slices:")
    for name, m in slices.items():
        print(
            f"  {name}: score={m['score']:.2f} Sharpe={m['sharpe']:.3f} "
            f"mean=${m['mean']:.2f}"
        )
    print(f"Wrote {out_path}")
    for label, path in plots.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
