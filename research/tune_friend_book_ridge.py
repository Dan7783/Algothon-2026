#!/usr/bin/env python
"""Ratio-normalized purged K-window Optuna study of friend_book ridge knobs.

Tunes the three ridge forecast constants in friend_book.py **in isolation**
from the pair book / conflict voter / ALGO-residual overlay:

    RIDGE_LAM     (default 1.0)
    RIDGE_THETA   (default 0.75)
    RIDGE_GATE_N  (default 35)

Each trial scores only the raw `_ridge_forecast_details` dollar sleeve
converted to shares (same crash clip / position limits as getMyPosition).
Pair votes, RIDGE_CONFLICT_WEIGHT, deadlock overlay, and algo residual are
never applied, so conflict-policy changes cannot leak into the ridge fit.

Pipeline
--------
  Phase A: K purged windows with embargo gaps; Sobol -> TPE per window to
           estimate that window's achievable ridge-only score (denominator).
  Phase C: global study maximizing
               mean(score_w / denom_w) - std_lambda * std(score_w / denom_w)
  Phase D: re-rank top-M by purged walk-forward (mean - std across folds).

Outputs land under research/ridge_param_study/ and never touch
auto_tuning_results/.

Examples
--------
  python research/tune_friend_book_ridge.py
  python research/tune_friend_book_ridge.py --window_trials 150 --global_trials 300
  python research/tune_friend_book_ridge.py --params lam,theta --n_windows 4
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "research" / "ridge_param_study"
PRICES_FILE = ROOT / "prices.txt"
DENOM_EPS = 1.0e-6

DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000

BASELINE = {"lam": 1.0, "theta": 0.75, "gate_n": 35}

PARAM_SPECS = {
    "lam": {
        "attr": "RIDGE_LAM",
        "suggest": lambda t: t.suggest_float("lam", 0.05, 8.0, log=True),
    },
    "theta": {
        "attr": "RIDGE_THETA",
        "suggest": lambda t: t.suggest_float("theta", 0.25, 2.5),
    },
    "gate_n": {
        "attr": "RIDGE_GATE_N",
        "suggest": lambda t: t.suggest_int("gate_n", 15, 60),
    },
}


def load_strategy(module_name: str = "friend_book"):
    path = ROOT / f"{module_name}.py"
    module = types.ModuleType(module_name)
    module.__file__ = str(path.resolve())
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def score(mu: float, sigma: float, param: float = 1.0) -> float:
    if mu <= 0.0 or sigma < 1.0e-10:
        return float(mu)
    sr = np.sqrt(250.0) * mu / sigma
    return float(mu * (sr * sr) / (sr * sr + param * param))


def load_prices(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def ridge_only_shares(strategy, prices: np.ndarray) -> np.ndarray:
    """Shares from the ridge sleeve alone — no pairs, conflict votes, or overlay."""
    nins, nt = prices.shape
    zeros = np.zeros(nins, dtype=int)
    if nins != strategy.N_INST or nt < strategy.WARMUP + 10:
        return zeros
    if (
        not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return zeros

    returns = np.diff(np.log(prices), axis=1)
    dollars, _, _, _ = strategy._ridge_forecast_details(returns)

    safe_prices = np.maximum(prices[:, -1], strategy._EPS_DENOM)
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], strategy._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    dollars = dollars.copy()
    dollars[crash] = 0.0
    if len(dollars) > 1:
        dollars[1:] = np.clip(dollars[1:], -strategy.LEG, strategy.LEG)
    dollars[0] = 0.0  # ridge never holds ALGO

    raw_shares = dollars / safe_prices
    share_limits = np.floor(strategy.BANG_BANG_LIMITS / safe_prices).astype(int)
    share_limits[0] = 0
    return np.nan_to_num(
        np.clip(np.round(raw_shares), -share_limits, share_limits),
        nan=0.0,
    ).astype(int)


def calc_pl(strategy, prices: np.ndarray, start: int, end: int):
    """Official-style PnL over [start, end] using the ridge-only sleeve."""
    n_inst = prices.shape[0]
    comm_rate = np.full(n_inst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_limit = np.full(n_inst, float(DEFAULT_DLR_LIMIT))
    dlr_limit[0] = float(INST0_DLR_LIMIT)

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvolume = 0.0
    value = 0.0
    today_pll: list[float] = []

    for t in range(start - 1, end + 1):
        prc_so_far = prices[:, :t]
        cur_prices = prc_so_far[:, -1]
        if t < end:
            new_pos = ridge_only_shares(strategy, prc_so_far)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= float(cur_prices.dot(delta))
        dvolumes = cur_prices * np.abs(delta)
        cash -= float(np.sum(dvolumes * comm_rate))
        tot_dvolume += float(np.sum(dvolumes))
        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t >= start:
            today_pll.append(today_pl)

    pll = np.asarray(today_pll, dtype=float)
    mu = float(pll.mean()) if len(pll) else 0.0
    sigma = float(pll.std()) if len(pll) else 0.0
    sharpe = float(np.sqrt(250.0) * mu / sigma) if sigma > 0 else 0.0
    return mu, sigma, sharpe, score(mu, sigma), tot_dvolume, pll


def apply_params(strategy, params: dict) -> dict:
    full = dict(BASELINE)
    full.update(params)
    for name, value in full.items():
        setattr(strategy, PARAM_SPECS[name]["attr"], value)
    return full


def suggest_params(trial: optuna.Trial, param_names: list[str]) -> dict:
    return {name: PARAM_SPECS[name]["suggest"](trial) for name in param_names}


def build_windows(start: int, end: int, n_windows: int, embargo: int):
    """K contiguous equal-length windows with embargo-day gaps between them."""
    span = end - start + 1
    n_windows = max(1, min(n_windows, span))
    total = span - embargo * (n_windows - 1)
    if total < n_windows:
        embargo = max(0, (span - n_windows) // max(1, n_windows - 1))
        total = span - embargo * (n_windows - 1)
    width = max(1, total // n_windows)
    windows = []
    s = start
    for i in range(n_windows):
        e = end if i == n_windows - 1 else s + width - 1
        windows.append((s, e))
        s = e + 1 + embargo
    return windows


def eval_score(strategy, prices, params, start, end) -> float:
    apply_params(strategy, params)
    *_, sc, dvol, _ = calc_pl(strategy, prices, start, end)
    if dvol == 0.0:
        sc -= 5.0
    return float(sc)


def purged_walk_forward_score(
    strategy, prices, params, start, end, n_folds=4, embargo=10
) -> float:
    """mean(fold scores) - std(fold scores) with embargo gaps between folds."""
    total_days = end - start + 1
    if total_days < n_folds * 30:
        n_folds = max(1, total_days // 30)
    fold_len = total_days // n_folds
    fold_scores = []
    current = start
    for i in range(n_folds):
        fold_end = end if i == n_folds - 1 else current + fold_len - 1
        if fold_end >= current:
            fold_scores.append(
                eval_score(strategy, prices, params, current, fold_end)
            )
        current = fold_end + 1 + embargo
    if not fold_scores:
        return -1.0e9
    arr = np.asarray(fold_scores, dtype=float)
    return float(arr.mean() - arr.std())


def make_storage(db_path: Path) -> str:
    return f"sqlite:///{db_path.as_posix()}"


def run_window_study(
    strategy,
    prices,
    param_names,
    w_index,
    w_start,
    w_end,
    storage,
    n_trials,
    sobol_trials,
    seed,
    embargo,
):
    study_name = f"ridgeonly-w{w_index}-{w_start}-{w_end}"
    if embargo:
        study_name += f"-emb{embargo}"
    print(
        f"\n[Phase A] {study_name}: {n_trials} ridge-only trials on days {w_start}-{w_end}",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, param_names)
        sc = eval_score(strategy, prices, params, w_start, w_end)
        trial.set_user_attr("window_score", sc)
        return sc

    # Sobol warm-start then TPE (mixed continuous/int space; CMA-ES unfit for gate_n).
    sobol_n = min(sobol_trials, n_trials)
    tpe_n = max(0, n_trials - sobol_n)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.QMCSampler(seed=seed + w_index, scramble=True),
        storage=storage,
        load_if_exists=True,
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.enqueue_trial({k: BASELINE[k] for k in param_names})
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remain_sobol = max(0, sobol_n - done)
    if remain_sobol:
        study.optimize(objective, n_trials=remain_sobol, show_progress_bar=True)

    if tpe_n:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed + 100 + w_index),
            storage=storage,
            load_if_exists=True,
        )
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        remain = max(0, n_trials - done)
        if remain:
            study.optimize(objective, n_trials=remain, show_progress_bar=True)

    best_value = float(study.best_value)
    denom_raw = best_value
    denom = max(denom_raw, DENOM_EPS)
    clamped = denom_raw < DENOM_EPS
    print(
        f"  best={best_value:.4f} denom={denom:.4f}"
        f"{' (CLAMPED)' if clamped else ''} params={study.best_params}",
        flush=True,
    )
    return {
        "index": w_index,
        "start": w_start,
        "end": w_end,
        "study": study_name,
        "best_params": study.best_params,
        "best_value": best_value,
        "denominator": denom,
        "denominator_raw": denom_raw,
        "denominator_clamped": clamped,
        "n_trials": len(study.trials),
    }


def run_global_ratio_study(
    strategy,
    prices,
    param_names,
    window_results,
    storage,
    n_trials,
    sobol_trials,
    seed,
    std_lambda,
):
    study_name = f"ridgeonly-globalnorm-ratio-std{std_lambda:g}"
    windows = [(w["start"], w["end"]) for w in window_results]
    denoms = [w["denominator"] for w in window_results]

    print(
        f"\n[Phase C] {study_name}: mean - {std_lambda:g}*std of "
        f"{len(windows)} ratio-normalized ridge-only window scores "
        f"({n_trials} trials)",
        flush=True,
    )
    for w in window_results:
        flag = " (CLAMPED)" if w["denominator_clamped"] else ""
        print(
            f"  window {w['start']}-{w['end']}: denom {w['denominator']:.4f}{flag}",
            flush=True,
        )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, param_names)
        raw = [
            eval_score(strategy, prices, params, ws, we) for ws, we in windows
        ]
        ratios = [s / d for s, d in zip(raw, denoms)]
        ratio_mean = float(np.mean(ratios))
        ratio_std = float(np.std(ratios))
        obj = float(ratio_mean - std_lambda * ratio_std)
        trial.set_user_attr("window_scores", [float(s) for s in raw])
        trial.set_user_attr("ratios", [float(r) for r in ratios])
        trial.set_user_attr("ratio_mean", ratio_mean)
        trial.set_user_attr("ratio_std", ratio_std)
        trial.set_user_attr("penalized_objective", obj)
        return obj

    sobol_n = min(sobol_trials, n_trials)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.QMCSampler(seed=seed + 7, scramble=True),
        storage=storage,
        load_if_exists=True,
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.enqueue_trial({k: BASELINE[k] for k in param_names})
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remain_sobol = max(0, sobol_n - done)
    if remain_sobol:
        study.optimize(objective, n_trials=remain_sobol, show_progress_bar=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed + 17),
        storage=storage,
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remain = max(0, n_trials - done)
    if remain:
        study.optimize(objective, n_trials=remain, show_progress_bar=True)

    print(
        f"  [Phase C] best objective {study.best_value:.4f} "
        f"(trial {study.best_trial.number}) params={study.best_params}",
        flush=True,
    )
    return study


def tiebreak_purged_wf(
    strategy,
    prices,
    study,
    start,
    end,
    top_m,
    n_folds,
    embargo,
):
    done = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:top_m]
    ranked = []
    print(
        f"\n[Phase D] purged-WF tiebreak over top {len(top)} "
        f"({n_folds} folds, embargo={embargo})",
        flush=True,
    )
    for t in top:
        wf = purged_walk_forward_score(
            strategy, prices, t.params, start, end, n_folds=n_folds, embargo=embargo
        )
        ranked.append(
            {
                "trial": t.number,
                "objective": float(t.value),
                "wf_score": wf,
                "params": dict(t.params),
                "ratio_mean": t.user_attrs.get("ratio_mean"),
                "window_scores": t.user_attrs.get("window_scores"),
            }
        )
        print(
            f"  trial {t.number:>4} obj={t.value:.4f} wf={wf:.4f} params={t.params}",
            flush=True,
        )
    ranked.sort(key=lambda r: r["wf_score"], reverse=True)
    winner = ranked[0]
    obj_rank = next(i for i, t in enumerate(top) if t.number == winner["trial"]) + 1
    print(
        f"  winner trial {winner['trial']} (obj rank {obj_rank}/{len(top)}, "
        f"wf {winner['wf_score']:.4f})",
        flush=True,
    )
    return winner, ranked


def save_trial_table(study: optuna.Study, path: Path) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "objective": t.value, **t.params}
        for k, v in t.user_attrs.items():
            row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    df.to_csv(path, index=False)
    return df


def plot_optuna_views(study: optuna.Study, param_names: list[str], out_dir: Path) -> None:
    for name, plotter in (
        ("history", plot_optimization_history),
        ("slice", lambda s: plot_slice(s, params=param_names)),
        ("parallel", lambda s: plot_parallel_coordinate(s, params=param_names)),
    ):
        try:
            axes = plotter(study)
            fig = axes.figure if hasattr(axes, "figure") else plt.gcf()
            fig.tight_layout()
            fig.savefig(out_dir / f"optuna_{name}.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] could not render {name}: {exc}", flush=True)

    if len(param_names) >= 2:
        try:
            axes = plot_contour(study, params=param_names[:2])
            fig = axes.figure if hasattr(axes, "figure") else plt.gcf()
            fig.tight_layout()
            fig.savefig(out_dir / "optuna_contour.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] could not render contour: {exc}", flush=True)


def plot_window_heatmap(window_results, ranked, out_dir: Path) -> None:
    """Heatmap of per-window raw scores for top tiebreak candidates."""
    if not ranked:
        return
    windows = [f"{w['start']}-{w['end']}" for w in window_results]
    labels = [f"t{r['trial']}" for r in ranked[:15]]
    mat = np.full((len(labels), len(windows)), np.nan)
    for i, r in enumerate(ranked[:15]):
        scores = r.get("window_scores") or []
        for j, s in enumerate(scores):
            mat[i, j] = s
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(labels) + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top trials — raw score per purged window")
    fig.colorbar(im, ax=ax, label="official score")
    fig.tight_layout()
    fig.savefig(out_dir / "window_heatmap.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", default="lam,theta,gate_n")
    p.add_argument("--start", type=int, default=201)
    p.add_argument("--end", type=int, default=800)
    p.add_argument("--holdout-start", type=int, default=801)
    p.add_argument("--holdout-end", type=int, default=1000)
    p.add_argument("--n_windows", type=int, default=4)
    p.add_argument("--embargo", type=int, default=10)
    p.add_argument("--window_trials", type=int, default=150)
    p.add_argument("--global_trials", type=int, default=300)
    p.add_argument("--sobol_trials", type=int, default=32)
    p.add_argument("--std_lambda", type=float, default=1.0)
    p.add_argument("--tiebreak_top_m", type=int, default=25)
    p.add_argument("--tiebreak_folds", type=int, default=4)
    p.add_argument("--tiebreak_fold_embargo", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=OUT_DIR / "ratio_purged_ridge_only",
    )
    p.add_argument(
        "--skip_phase_a",
        action="store_true",
        help="Reuse existing per-window studies in the DB (denoms from best trials).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    param_names = [p.strip() for p in args.params.split(",") if p.strip()]
    unknown = [p for p in param_names if p not in PARAM_SPECS]
    if unknown:
        raise SystemExit(f"Unknown params {unknown}; choose from {list(PARAM_SPECS)}")
    if not 2 <= len(param_names) <= 3:
        raise SystemExit("Select 2 or 3 params via --params")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.out_dir / "optuna_ridge_ratio.db"
    storage = make_storage(db_path)

    print(f"Loading prices from {PRICES_FILE}", flush=True)
    prices = load_prices(PRICES_FILE)
    strategy = load_strategy("friend_book")
    windows = build_windows(args.start, args.end, args.n_windows, args.embargo)
    print(
        f"Purged windows (embargo={args.embargo}): {windows}",
        flush=True,
    )
    print(
        f"Ridge-ONLY tuning {param_names} | window_trials={args.window_trials} "
        f"global_trials={args.global_trials} std_lambda={args.std_lambda} "
        f"-> {args.out_dir}",
        flush=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Phase A: per-window optima = ratio denominators ---
    window_results = []
    for i, (ws, we) in enumerate(windows):
        if args.skip_phase_a:
            study_name = f"ridgeonly-w{i}-{ws}-{we}"
            if args.embargo:
                study_name += f"-emb{args.embargo}"
            study = optuna.load_study(study_name=study_name, storage=storage)
            denom_raw = float(study.best_value)
            window_results.append(
                {
                    "index": i,
                    "start": ws,
                    "end": we,
                    "study": study_name,
                    "best_params": study.best_params,
                    "best_value": denom_raw,
                    "denominator": max(denom_raw, DENOM_EPS),
                    "denominator_raw": denom_raw,
                    "denominator_clamped": denom_raw < DENOM_EPS,
                    "n_trials": len(study.trials),
                }
            )
        else:
            window_results.append(
                run_window_study(
                    strategy,
                    prices,
                    param_names,
                    i,
                    ws,
                    we,
                    storage,
                    args.window_trials,
                    args.sobol_trials,
                    args.seed,
                    args.embargo,
                )
            )

    (args.out_dir / "window_results.json").write_text(
        json.dumps(window_results, indent=2), encoding="utf-8"
    )

    # --- Phase C: ratio-normalized global study ---
    global_study = run_global_ratio_study(
        strategy,
        prices,
        param_names,
        window_results,
        storage,
        args.global_trials,
        args.sobol_trials,
        args.seed,
        args.std_lambda,
    )

    # --- Phase D: purged-WF tiebreak ---
    if args.tiebreak_top_m > 0:
        winner, ranked = tiebreak_purged_wf(
            strategy,
            prices,
            global_study,
            args.start,
            args.end,
            args.tiebreak_top_m,
            args.tiebreak_folds,
            args.tiebreak_fold_embargo,
        )
        best_params = winner["params"]
        selection = "tiebreak_purged_wf"
    else:
        best_params = dict(global_study.best_params)
        winner = {
            "trial": global_study.best_trial.number,
            "objective": float(global_study.best_value),
            "wf_score": None,
            "params": best_params,
        }
        ranked = [winner]
        selection = "best_global_ratio"

    # --- Reporting: full train + holdout vs baseline ---
    print("\n=== Final selection ===", flush=True)
    print(f"  source : {selection}", flush=True)
    print(f"  params : {best_params}", flush=True)
    print(f"  global obj / wf : {winner.get('objective')} / {winner.get('wf_score')}", flush=True)

    report = {
        "method": "ratio_normalized_purged_k_window_ridge_only",
        "isolation": {
            "pairs": False,
            "ridge_conflict_votes": False,
            "deadlock_overlay": False,
            "algo_residual": False,
            "sleeve": "_ridge_forecast_details",
        },
        "train_window": [args.start, args.end],
        "windows": [[w["start"], w["end"]] for w in window_results],
        "embargo": args.embargo,
        "std_lambda": args.std_lambda,
        "params_tuned": param_names,
        "baseline": {k: BASELINE[k] for k in param_names},
        "window_results": window_results,
        "selection": selection,
        "best_params": best_params,
        "winner": {
            k: winner[k]
            for k in ("trial", "objective", "wf_score", "params")
            if k in winner
        },
        "tiebreak_ranked": [
            {k: r[k] for k in ("trial", "objective", "wf_score", "params")}
            for r in ranked
        ],
        "n_global_trials": len(global_study.trials),
        "window_trials": args.window_trials,
        "global_trials_budget": args.global_trials,
    }

    apply_params(strategy, best_params)
    mu, sigma, sharpe, sc, _, _ = calc_pl(strategy, prices, args.start, args.end)
    report["best_train"] = {
        "mean_pl": mu,
        "std_pl": sigma,
        "sharpe": sharpe,
        "score": sc,
    }
    wf_final = purged_walk_forward_score(
        strategy,
        prices,
        best_params,
        args.start,
        args.end,
        n_folds=args.tiebreak_folds,
        embargo=args.tiebreak_fold_embargo,
    )
    report["best_purged_wf"] = wf_final

    apply_params(strategy, BASELINE)
    mu_b, sigma_b, sharpe_b, sc_b, _, _ = calc_pl(
        strategy, prices, args.start, args.end
    )
    report["baseline_train"] = {
        "mean_pl": mu_b,
        "std_pl": sigma_b,
        "sharpe": sharpe_b,
        "score": sc_b,
    }
    wf_base = purged_walk_forward_score(
        strategy,
        prices,
        BASELINE,
        args.start,
        args.end,
        n_folds=args.tiebreak_folds,
        embargo=args.tiebreak_fold_embargo,
    )
    report["baseline_purged_wf"] = wf_base

    print(
        f"\nTrain {args.start}-{args.end}: best score={sc:.4f} "
        f"(baseline {sc_b:.4f}) | purged-WF {wf_final:.4f} "
        f"(baseline {wf_base:.4f})",
        flush=True,
    )

    if args.holdout_start > 0:
        apply_params(strategy, best_params)
        mu_h, sigma_h, sharpe_h, sc_h, _, _ = calc_pl(
            strategy, prices, args.holdout_start, args.holdout_end
        )
        report["best_holdout"] = {
            "window": [args.holdout_start, args.holdout_end],
            "mean_pl": mu_h,
            "std_pl": sigma_h,
            "sharpe": sharpe_h,
            "score": sc_h,
        }
        apply_params(strategy, BASELINE)
        mu_hb, sigma_hb, sharpe_hb, sc_hb, _, _ = calc_pl(
            strategy, prices, args.holdout_start, args.holdout_end
        )
        report["baseline_holdout"] = {
            "window": [args.holdout_start, args.holdout_end],
            "mean_pl": mu_hb,
            "std_pl": sigma_hb,
            "sharpe": sharpe_hb,
            "score": sc_hb,
        }
        print(
            f"Holdout {args.holdout_start}-{args.holdout_end}: "
            f"best={sc_h:.4f} baseline={sc_hb:.4f}",
            flush=True,
        )

    df = save_trial_table(global_study, args.out_dir / "global_trials.csv")
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out_dir / 'global_trials.csv'} ({len(df)} trials)", flush=True)
    print(f"Wrote {args.out_dir / 'report.json'}", flush=True)

    plot_optuna_views(global_study, param_names, args.out_dir)
    plot_window_heatmap(window_results, ranked, args.out_dir)
    print(f"Wrote Optuna + window plots under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
