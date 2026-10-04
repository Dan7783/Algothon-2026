#!/usr/bin/env python
"""Combinatorial Purged CV for the ALGO combo IC sleeve.

Two evaluations:
  1) Causal online sleeve (`algo_combo_ic_sleeve.getMyPosition`) scored with
     the same CPCV path machinery as `_run_cpcv.py` (purge/embargo on the
     causal daily PnL series).
  2) Path-wise refit CPCV (Lopez de Prado Ch. 12 style for trainable models):
     for each combinatorial test-group path, estimate IC weights only on the
     purged training groups, freeze them, then trade ALGO on the test days.

Outputs JSON + plots under auto_tuning_results/.
"""

from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import algo_combo_ic_sleeve as sleeve
from _run_cpcv import (  # noqa: E402
    calc_daily_pnl,
    embargo_width,
    load_strategy,
    path_metrics,
    plot_results,
    purged_test_mask,
    run_cpcv,
    score,
    split_into_groups,
)


def load_prices(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T


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


def train_mask_for_path(
    days: np.ndarray,
    groups: list[np.ndarray],
    test_group_ids: tuple[int, ...],
    purge: int,
    pct_embargo: float,
) -> np.ndarray:
    """Train days = all days not in purged test path and not in embargo."""
    test_keep = purged_test_mask(days, groups, test_group_ids, purge, pct_embargo)
    # Start from all non-test-group days, then also drop purge/embargo borders
    # around test blocks (same borders zeroed in test_keep's complement logic).
    in_test_group = np.zeros(len(days), dtype=bool)
    day_to_pos = {int(d): i for i, d in enumerate(days)}
    for gid in test_group_ids:
        for d in groups[gid]:
            in_test_group[day_to_pos[int(d)]] = True

    # Rebuild full excluded set: original test groups + purge + embargo
    excluded = in_test_group.copy()
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
        if run[0] > 0 and purge > 0:
            excluded[first_pos : first_pos + purge] = True
        if run[-1] < len(groups) - 1 and purge > 0:
            lo = max(first_pos, last_pos - purge + 1)
            excluded[lo : last_pos + 1] = True
        if emb > 0:
            excluded[last_pos + 1 : last_pos + 1 + emb] = True

    train = ~excluded
    # Also ensure we never train on the scored test days.
    train &= ~test_keep
    return train


def fit_ics_on_day_subset(
    prices: np.ndarray,
    train_days: np.ndarray,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """Estimate ICs using only label-start indices whose calendar day is in train.

    Return index i corresponds to the move into price column i+1; we map label
    start i to calendar decision day i+1 (price column count when that return
    became known). For IC labels, feature at i and forward algo[i:i+horizon]
    must both lie in the information set of train days.
    """
    returns = sleeve._log_returns(prices)
    features = sleeve._feature_matrix(returns, lookback)
    forward = sleeve._forward_algo(returns, horizon)
    n_rets = returns.shape[1]
    train_set = set(int(d) for d in train_days)

    # Label starting at return index i uses algo returns i..i+horizon-1 which
    # complete at price column i+horizon. Require completion day in train and
    # feature day (i) associated calendar day i+1 in train.
    idxs = []
    for i in range(0, n_rets - horizon):
        complete_day = i + horizon  # price column index / day number if 1-indexed?
        # In eval, day t means prices[:, :t] with t as column count.
        # days array from calc_daily_pnl uses t as the day index equal to column.
        # Return index i is the return from price[i] -> price[i+1], observed once
        # we have column i+1. Map return index i -> day = i+1.
        feat_day = i + 1
        label_end_day = i + horizon  # last return index in forward is i+h-1
        # completion when we have price column (i+horizon)
        complete_day = i + horizon
        if feat_day in train_set and complete_day in train_set:
            if np.isfinite(forward[i]) and np.any(np.isfinite(features[i])):
                idxs.append(i)
    if len(idxs) < sleeve.MIN_IC_SAMPLES:
        # fall back to any finite completed labels whose end day is in train
        idxs = []
        for i in range(0, n_rets - horizon):
            complete_day = i + horizon
            if complete_day in train_set and np.isfinite(forward[i]):
                idxs.append(i)
    train_idx = np.asarray(idxs, dtype=int)
    if train_idx.size < 20:
        return np.zeros(features.shape[1])
    return sleeve.estimate_ics(features, forward, train_idx)


def simulate_path_pnl(
    prices: np.ndarray,
    ics: np.ndarray,
    test_days: np.ndarray,
    dollar_cap: float,
    lookback: int,
    mode: str,
    comm_rate: np.ndarray,
) -> np.ndarray:
    """Mark-to-market daily PnL on test_days with frozen ICs (ALGO only)."""
    returns = sleeve._log_returns(prices)
    features = sleeve._feature_matrix(returns, lookback)
    algo_px = prices[0]
    n_inst = prices.shape[0]

    # Precompute desired dollar sign per return-index / day.
    # On calendar day d (price column d available as last), return index = d-1.
    day_to_dir: dict[int, float] = {}
    for d in test_days:
        d = int(d)
        ret_i = d - 1
        if ret_i < 0 or ret_i >= features.shape[0]:
            day_to_dir[d] = 0.0
            continue
        score_val = sleeve.build_score(features[ret_i], ics, mode)
        day_to_dir[d] = float(np.sign(score_val)) if abs(score_val) > 1e-15 else 0.0

    # Simulate continuously from first test day - 1 through last test day so
    # positions carry correctly; only record PnL on test days.
    if test_days.size == 0:
        return np.array([])
    start = int(test_days.min())
    end = int(test_days.max())
    test_set = set(int(d) for d in test_days)

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    value = 0.0
    comm = 0.0
    pll = []

    for t in range(max(1, start - 1), end + 1):
        cur_prices = prices[:, t - 1] if t >= 1 else prices[:, 0]
        # Position decision on day t uses history through column t (prices[:, :t])
        # matching calc_daily_pnl. Feature row at ret_i = t-1.
        if t in day_to_dir:
            direction = day_to_dir[t]
        elif t <= end:
            ret_i = t - 1
            if 0 <= ret_i < features.shape[0]:
                score_val = sleeve.build_score(features[ret_i], ics, mode)
                direction = (
                    float(np.sign(score_val)) if abs(score_val) > 1e-15 else 0.0
                )
            else:
                direction = 0.0
        else:
            direction = 0.0

        px = float(cur_prices[0])
        new_pos = np.zeros(n_inst)
        if px > 0 and direction != 0.0 and t <= end:
            dollars = direction * min(dollar_cap, float(sleeve.LIMIT[0]))
            new_pos[0] = float(int(np.round(dollars / px)))

        # On final mark day beyond end we keep position; here end is last test.
        delta = new_pos - cur_pos
        dvolumes = cur_prices * np.abs(delta)
        comm = float(np.sum(dvolumes * comm_rate))
        cash -= float(cur_prices.dot(delta) + comm)
        cur_pos = new_pos
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t in test_set:
            pll.append(today_pl)

    return np.asarray(pll, dtype=float)


def run_pathwise_cpcv(
    prices: np.ndarray,
    days: np.ndarray,
    n_groups: int,
    n_test_groups: int,
    purge: int,
    pct_embargo: float,
    lookback: int,
    horizon: int,
    mode: str,
    dollar_cap: float,
) -> dict:
    groups = split_into_groups(days, n_groups)
    group_spans = [
        {"group": i, "start": int(g[0]), "end": int(g[-1]), "n": int(len(g))}
        for i, g in enumerate(groups)
    ]
    comm = np.full(prices.shape[0], 1.0e-4)
    comm[0] = 2.0e-5

    paths = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_mask = purged_test_mask(days, groups, test_ids, purge, pct_embargo)
        train_mask = train_mask_for_path(
            days, groups, test_ids, purge, pct_embargo
        )
        train_days = days[train_mask]
        test_days = days[test_mask]
        ics = fit_ics_on_day_subset(prices, train_days, lookback, horizon)
        n_active = int(np.sum(np.abs(ics) >= sleeve.MIN_ABS_IC))
        pll = simulate_path_pnl(
            prices,
            ics,
            test_days,
            dollar_cap=dollar_cap,
            lookback=lookback,
            mode=mode,
            comm_rate=comm,
        )
        metrics = path_metrics(pll)
        paths.append(
            {
                "test_groups": list(test_ids),
                "n_train_days": int(train_days.size),
                "n_scored_days": int(test_days.size),
                "n_active_ics": n_active,
                "ics_l1": float(np.abs(ics).sum()),
                "first_day": int(test_days[0]) if test_days.size else None,
                "last_day": int(test_days[-1]) if test_days.size else None,
                **metrics,
            }
        )

    scores = np.asarray([p["score"] for p in paths], dtype=float)
    sharpes = np.asarray([p["sharpe"] for p in paths], dtype=float)
    means = np.asarray([p["mean"] for p in paths], dtype=float)
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


def plot_pathwise(
    cpcv: dict,
    out_dir: Path,
    tag: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = cpcv["paths"]
    scores = np.asarray([p["score"] for p in paths], dtype=float)
    sharpes = np.asarray([p["sharpe"] for p in paths], dtype=float)

    hist_path = out_dir / f"cpcv_score_hist_{tag}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(
        scores[np.isfinite(scores)],
        bins=min(20, max(8, len(scores) // 2)),
        color="#4C72B0",
        edgecolor="white",
    )
    axes[0].axvline(
        cpcv["score_summary"]["median"],
        color="#55A868",
        linewidth=2,
        linestyle="--",
        label="median",
    )
    axes[0].axvline(0.0, color="#C44E52", linewidth=1, alpha=0.7)
    axes[0].set_title("Path-wise CPCV scores")
    axes[0].set_xlabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(
        sharpes[np.isfinite(sharpes)],
        bins=min(20, max(8, len(sharpes) // 2)),
        color="#8172B2",
        edgecolor="white",
    )
    axes[1].axvline(
        cpcv["sharpe_summary"]["median"],
        color="#55A868",
        linewidth=2,
        linestyle="--",
        label="median",
    )
    axes[1].axvline(0.0, color="#C44E52", linewidth=1, alpha=0.7)
    axes[1].set_title("Path-wise CPCV Sharpes")
    axes[1].set_xlabel("Annualised Sharpe")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)

    heat_path = out_dir / f"cpcv_group_heatmap_{tag}.png"
    n_groups = int(cpcv["n_groups"])
    mat = np.full((len(paths), n_groups), np.nan)
    for i, path in enumerate(paths):
        for gid in path["test_groups"]:
            mat[i, gid] = path["score"]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(paths) + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Group id")
    ax.set_ylabel("CPCV path")
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(
        [f"G{g['group']}\n{g['start']}-{g['end']}" for g in cpcv["group_spans"]],
        fontsize=8,
    )
    ax.set_title("Path-wise refit CPCV: test-group scores")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Path score")
    fig.tight_layout()
    fig.savefig(heat_path, dpi=140)
    plt.close(fig)

    return {"score_hist": str(hist_path), "group_heatmap": str(heat_path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", default="prices.txt")
    p.add_argument("--start", type=int, default=251)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--n_groups", type=int, default=6)
    p.add_argument("--n_test_groups", type=int, default=2)
    p.add_argument("--purge", type=int, default=30, help=">= HORIZON recommended")
    p.add_argument("--pct_embargo", type=float, default=0.01)
    p.add_argument("--mode", choices=["ic_weighted", "top_k_absic"], default="ic_weighted")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--dollar_cap", type=float, default=50_000.0)
    p.add_argument("--skip_causal", action="store_true")
    p.add_argument(
        "--out",
        default="auto_tuning_results/cpcv_algo_combo_ic_sleeve.json",
    )
    p.add_argument("--plot_dir", default="auto_tuning_results/cpcv_algo_combo")
    args = p.parse_args()

    if not (1 <= args.n_test_groups < args.n_groups):
        raise SystemExit("Need 1 <= n_test_groups < n_groups")

    # Configure sleeve module globals for causal run.
    sleeve.MODE = args.mode
    sleeve.LOOKBACK = args.lookback
    sleeve.HORIZON = args.horizon
    sleeve.DOLLAR_CAP = args.dollar_cap
    sleeve.reset()

    prices = load_prices(ROOT / args.prices)
    n_inst = prices.shape[0]
    comm = np.full(n_inst, 1.0e-4)
    comm[0] = 2.0e-5
    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0

    payload: dict = {
        "strategy": sleeve.describe(),
        "start": args.start,
        "end": args.end,
        "method": "combinatorial_purged_cv",
        "reference": "Lopez de Prado, Advances in Financial Machine Learning, Ch.12",
        "config": {
            "n_groups": args.n_groups,
            "n_test_groups": args.n_test_groups,
            "purge": args.purge,
            "pct_embargo": args.pct_embargo,
            "mode": args.mode,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "dollar_cap": args.dollar_cap,
        },
    }

    # ---- 1) Causal online sleeve CPCV ----
    if not args.skip_causal:
        print(
            f"Causal backtest algo_combo_ic_sleeve days {args.start}-{args.end} "
            f"mode={args.mode} lb={args.lookback} H={args.horizon} cap=${args.dollar_cap:,.0f}"
        )
        sleeve.reset()
        days, pll = calc_daily_pnl(
            prices,
            sleeve.getMyPosition,
            sleeve.reset,
            args.start,
            args.end,
            comm,
            limits,
        )
        full = path_metrics(pll)
        print(
            f"  Full window: n={full['n_days']} mean=${full['mean']:.2f} "
            f"Sharpe={full['sharpe']:.3f} score={full['score']:.2f} "
            f"total=${full['total_pnl']:.0f}"
        )
        causal_cpcv = run_cpcv(
            days,
            pll,
            n_groups=args.n_groups,
            n_test_groups=args.n_test_groups,
            purge=args.purge,
            pct_embargo=args.pct_embargo,
        )
        plot_dir = ROOT / args.plot_dir
        causal_plots = plot_results(
            full, causal_cpcv, days, pll, plot_dir, "algo_combo_causal"
        )
        slices = {
            "dev_251_750": path_metrics(pll[(days >= 251) & (days <= 750)]),
            "oos_751_1000": path_metrics(pll[(days >= 751) & (days <= 1000)]),
            "all": full,
        }
        payload["causal_online"] = {
            "note": (
                "Expanding IC fit inside getMyPosition (causal). CPCV scores "
                "combinatorial OOS segments of that single PnL path."
            ),
            "full_window": full,
            "conventional_slices": slices,
            "cpcv": {k: v for k, v in causal_cpcv.items() if k != "paths"},
            "paths": [
                {
                    "test_groups": path["test_groups"],
                    "n_scored_days": path["n_scored_days"],
                    "mean": path["mean"],
                    "std": path["std"],
                    "sharpe": path["sharpe"],
                    "score": path["score"],
                    "total_pnl": path["total_pnl"],
                }
                for path in causal_cpcv["paths"]
            ],
            "plots": causal_plots,
        }
        ss = causal_cpcv["score_summary"]
        print(
            f"  Causal CPCV paths={causal_cpcv['n_paths']}: "
            f"score median={ss['median']:.2f} mean={ss['mean']:.2f} "
            f"frac>0={ss['frac_positive']:.1%} | "
            f"Sharpe median={causal_cpcv['sharpe_summary']['median']:.3f}"
        )

    # ---- 2) Path-wise IC refit CPCV ----
    print(
        f"\nPath-wise refit CPCV N={args.n_groups} k={args.n_test_groups} "
        f"purge={args.purge} embargo={args.pct_embargo:.2%}"
    )
    # Use the same day index grid as the causal backtest.
    day_grid = np.arange(args.start, args.end + 1, dtype=int)
    pathwise = run_pathwise_cpcv(
        prices,
        day_grid,
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        purge=args.purge,
        pct_embargo=args.pct_embargo,
        lookback=args.lookback,
        horizon=args.horizon,
        mode=args.mode,
        dollar_cap=args.dollar_cap,
    )
    pw_plots = plot_pathwise(pathwise, ROOT / args.plot_dir, "algo_combo_pathwise")
    payload["pathwise_refit"] = {
        "note": (
            "For each combinatorial path, IC weights are estimated only on "
            "purged training groups, frozen, then applied to test-group days."
        ),
        "cpcv": {k: v for k, v in pathwise.items() if k != "paths"},
        "paths": pathwise["paths"],
        "plots": pw_plots,
    }
    pss = pathwise["score_summary"]
    print("  Group spans:")
    for g in pathwise["group_spans"]:
        print(f"    G{g['group']}: days {g['start']}-{g['end']} ({g['n']}d)")
    print(
        f"  Path-wise CPCV paths={pathwise['n_paths']}: "
        f"score median={pss['median']:.2f} mean={pss['mean']:.2f} "
        f"p05={pss['p05']:.2f} p95={pss['p95']:.2f} frac>0={pss['frac_positive']:.1%}"
    )
    print(
        f"  Sharpe median={pathwise['sharpe_summary']['median']:.3f} "
        f"mean={pathwise['sharpe_summary']['mean']:.3f} "
        f"frac>0={pathwise['sharpe_summary']['frac_positive']:.1%}"
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    for section in ("causal_online", "pathwise_refit"):
        if section in payload:
            for label, path in payload[section].get("plots", {}).items():
                print(f"  {section}/{label}: {path}")


if __name__ == "__main__":
    main()
