#!/usr/bin/env python
"""CPCV Optuna tuner accelerated by a functional JAX pair kernel.

Pipeline per pair:
  1. Partition the train day range into N groups and form every combination of
     k test groups (Lopez de Prado combinatorial purged CV), with purge/embargo.
  2. Optuna Sobol -> BIPOP CMA-ES maximizing
        mean(path_scores) - std_lambda * std(path_scores)
     where each trial is one jitted backtest + vmapped path scoring.
  3. Sobol init evaluates candidates in jax.vmap batches for throughput.
  4. Report train CPCV objective, full-window metrics, and strict OOS.

The JAX kernel (`jax_pair_kernel.py`) is a functional, static-shape rewrite of
the single-pair testbedv3 path (scan + fixed OU buffers + fixed-iter a*).
It is highly rank-correlated with run_official_eval but not bit-identical;
final OOS reporting still uses the official numpy engine.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import optunahub

import jax_pair_kernel as jpk
import latest_tuner
from latest_tuner import (
    RESULTS_DIR, SEARCH_SPACES, clean_sig_nt, get_strategy_classes,
    loadPrices, load_pairs_from_json, get_symbol_map, plot_trade_summary, pricesFile,
    round_sig, run_official_eval,
)

OPTUNA_DB_FILE = "optuna_studies.db"
NON_TRADING_PENALTY = 5.0

WINDOWED_CFG = {
    "module": "testbedv3",
    "start": 201,
    "end": 800,
    "oos_start": 801,
    "oos_end": 1000,
    "pairs_file": "pairs.json",
    # CPCV
    "n_groups": 6,
    "n_test_groups": 2,
    "purge": 10,
    "pct_embargo": 0.01,
    "std_lambda": 1.0,
    # Optuna budgets
    "trials": 700,
    "sobol_trials": 128,  # power of 2; evaluated in jax batches
    "batch_size": 32,
    "z_entry_rule": "new",
    "n_jobs": 1,
    "workers": 1,  # JAX already vectorises; multi-pair workers optional
    "show_progress": True,
    "db_file": os.path.join(RESULTS_DIR, OPTUNA_DB_FILE),
    "out_file": os.path.join(RESULTS_DIR, "windowed_tuning_results.json"),
    # Legacy aliases kept so old CLIs / callers don't crash.
    "n_windows": 6,
    "embargo": 10,
    "window_trials": 0,
    "global_trials": 700,
    "norm_mode": "cpcv",
    "tiebreak_top_m": 0,
    "tiebreak_folds": 4,
    "tiebreak_fold_embargo": 10,
}

_RESTART_MODULE = None


def get_storage(db_file):
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_file}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def get_restart_sampler():
    global _RESTART_MODULE
    if _RESTART_MODULE is None:
        _RESTART_MODULE = optunahub.load_module("samplers/restart_cmaes")
    return _RESTART_MODULE.RestartCmaEsSampler(restart_strategy="bipop", seed=42)


def suggest_params(trial, pair_y, pair_x):
    return {
        "pair_y": pair_y, "pair_x": pair_x,
        "q_beta": round_sig(trial.suggest_float("q_beta", **SEARCH_SPACES["q_beta"]), 3),
        "h": round_sig(trial.suggest_float("h", **SEARCH_SPACES["h"]), 3),
        "cusum_k": round_sig(trial.suggest_float("cusum_k", **SEARCH_SPACES["cusum_k"]), 3),
        "max_cov_inflate": round_sig(trial.suggest_float("max_cov_inflate", **SEARCH_SPACES["max_cov_inflate"]), 3),
        "cov_decay": round_sig(trial.suggest_float("cov_decay", **SEARCH_SPACES["cov_decay"]), 5),
        "kalman_var_ewma_alpha": round_sig(trial.suggest_float("kalman_var_ewma_alpha", **SEARCH_SPACES["kalman_var_ewma_alpha"]), 5),
        "ou_window": int(trial.suggest_int("ou_window", **SEARCH_SPACES["ou_window"])),
        "kappa_ewma_alpha": round_sig(trial.suggest_float("kappa_ewma_alpha", **SEARCH_SPACES["kappa_ewma_alpha"]), 5),
        "sigma_ewma_alpha": round_sig(trial.suggest_float("sigma_ewma_alpha", **SEARCH_SPACES["sigma_ewma_alpha"]), 5),
    }


def clean_trial_params(trial_params, pair_y, pair_x):
    params = {k: round_sig(v, 3) if isinstance(v, float) else v for k, v in trial_params.items()}
    params["ou_window"] = int(params["ou_window"])
    params["pair_y"] = pair_y
    params["pair_x"] = pair_x
    return params


def params_from_distributions(trial_number_seed, distributions, pair_y, pair_x, n):
    """Draw n Sobol/QMC points in the search box, returned as param dicts."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.QMCSampler(seed=trial_number_seed, scramble=True),
    )
    out = []

    def _obj(trial):
        out.append(suggest_params(trial, pair_y, pair_x))
        return 0.0

    study.optimize(_obj, n_trials=n, show_progress_bar=False)
    return out


def build_pair_engine(prices, pair_y, pair_x, cfg):
    symbol_map = get_symbol_map(pricesFile)
    y_idx, x_idx = symbol_map[pair_y], symbol_map[pair_x]
    y_prices = np.asarray(prices[y_idx], dtype=np.float64)
    x_prices = np.asarray(prices[x_idx], dtype=np.float64)
    n_score = cfg["end"] - cfg["start"] + 1
    masks = jpk.build_cpcv_masks(
        n_score, cfg["n_groups"], cfg["n_test_groups"], cfg["purge"], cfg["pct_embargo"],
    )
    rule_new = cfg["z_entry_rule"] == "new"
    obj_one, obj_batch = jpk.make_cpcv_fns(
        y_prices, x_prices, cfg["start"], cfg["end"], masks,
        rule_new=rule_new, std_lambda=cfg["std_lambda"],
    )
    score_one, score_batch, full_one = jpk.make_backtest_fns(
        y_prices, x_prices, cfg["start"], cfg["end"], rule_new=rule_new,
    )
    return {
        "y_idx": y_idx, "x_idx": x_idx,
        "y_prices": y_prices, "x_prices": x_prices,
        "masks": masks,
        "obj_one": obj_one, "obj_batch": obj_batch,
        "score_one": score_one, "score_batch": score_batch, "full_one": full_one,
        "n_paths": int(masks.shape[0]),
    }


def evaluate_params_cpcv(engine, params):
    return float(engine["obj_one"](jpk.params_to_vec(params)).block_until_ready())


def optimize_pair_jax(pair_y, pair_x, engine, cfg):
    """Sobol (vmapped batches) then BIPOP CMA-ES (jitted single trials)."""
    study_name = (
        f"{pair_y}-{pair_x}-cpcv-g{cfg['n_groups']}k{cfg['n_test_groups']}"
        f"-p{cfg['purge']}-e{cfg['pct_embargo']:g}-std{cfg['std_lambda']:g}"
    )
    storage = get_storage(cfg["db_file"])
    # Persistent study for resume / downstream tooling.
    persistent = optuna.create_study(
        direction="maximize", storage=storage, study_name=study_name, load_if_exists=True,
    )
    done = len(persistent.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    total = cfg["trials"]
    sobol_n = min(cfg["sobol_trials"], total)
    print(f"\n[CPCV] {study_name}: {engine['n_paths']} paths | "
          f"done={done}/{total} (sobol target {sobol_n})", flush=True)

    # Warmup compile.
    warm = {
        "pair_y": pair_y, "pair_x": pair_x,
        "q_beta": 1e-6, "h": 10.0, "cusum_k": 0.5, "max_cov_inflate": 100.0,
        "cov_decay": 0.05, "kalman_var_ewma_alpha": 0.05, "ou_window": 100,
        "kappa_ewma_alpha": 0.05, "sigma_ewma_alpha": 0.05,
    }
    t0 = time.perf_counter()
    _ = evaluate_params_cpcv(engine, warm)
    print(f"  [JIT] warmup {time.perf_counter() - t0:.2f}s", flush=True)

    # --- Phase A: batched Sobol ---
    sobol_remaining = max(0, sobol_n - done)
    if sobol_remaining > 0:
        batch = max(1, cfg["batch_size"])
        print(f"  [Sobol] {sobol_remaining} trials in batches of {batch}", flush=True)
        # Draw all remaining Sobol points up front, score in vmap chunks.
        drawn = params_from_distributions(42 + done, None, pair_y, pair_x, sobol_remaining)
        dists = {
            "q_beta": optuna.distributions.FloatDistribution(**SEARCH_SPACES["q_beta"]),
            "h": optuna.distributions.FloatDistribution(**SEARCH_SPACES["h"]),
            "cusum_k": optuna.distributions.FloatDistribution(**SEARCH_SPACES["cusum_k"]),
            "max_cov_inflate": optuna.distributions.FloatDistribution(**SEARCH_SPACES["max_cov_inflate"]),
            "cov_decay": optuna.distributions.FloatDistribution(**SEARCH_SPACES["cov_decay"]),
            "kalman_var_ewma_alpha": optuna.distributions.FloatDistribution(**SEARCH_SPACES["kalman_var_ewma_alpha"]),
            "ou_window": optuna.distributions.IntDistribution(**SEARCH_SPACES["ou_window"]),
            "kappa_ewma_alpha": optuna.distributions.FloatDistribution(**SEARCH_SPACES["kappa_ewma_alpha"]),
            "sigma_ewma_alpha": optuna.distributions.FloatDistribution(**SEARCH_SPACES["sigma_ewma_alpha"]),
        }
        for i in range(0, len(drawn), batch):
            chunk = drawn[i:i + batch]
            pvecs = np.stack([np.asarray(jpk.params_to_vec(p)) for p in chunk])
            scores = np.asarray(engine["obj_batch"](pvecs).block_until_ready())
            trials = []
            for p, s in zip(chunk, scores):
                params_only = {k: p[k] for k in jpk.PARAM_KEYS}
                trials.append(optuna.trial.create_trial(
                    params=params_only, distributions=dists, value=float(s),
                    user_attrs={"method": "jax_cpcv_sobol"},
                ))
            persistent.add_trials(trials)
            done += len(trials)
            print(f"    sobol {done}/{sobol_n}  last_batch_best={float(np.max(scores)):.4f}", flush=True)

    # --- Phase B: BIPOP CMA-ES single-trial jax objective ---
    cmaes_remaining = max(0, total - done)
    if cmaes_remaining > 0:
        print(f"  [CMA-ES] {cmaes_remaining} trials (jitted CPCV objective)", flush=True)
        ram = latest_tuner.RamFlushedStudy(
            study_name, get_restart_sampler(), cfg["db_file"], flush_every=50)

        def objective(trial):
            params = suggest_params(trial, pair_y, pair_x)
            val = evaluate_params_cpcv(engine, params)
            trial.set_user_attr("method", "jax_cpcv_cmaes")
            return val

        ram.optimize(objective, n_trials=cmaes_remaining, n_jobs=1,
                     show_progress=cfg["show_progress"])
        persistent = ram.persistent

    best = persistent.best_trial
    best_params = clean_trial_params(best.params, pair_y, pair_x)
    print(f"  [Best] trial {best.number} CPCV obj={best.value:.4f}", flush=True)
    return persistent, best_params, float(best.value), study_name


def run_pair(pair_dict, cfg):
    y_sym, x_sym = pair_dict["y"], pair_dict["x"]
    pair_name = f"{y_sym}-{x_sym}"
    prices, nt, nInst = loadPrices(pricesFile)
    symbol_map = get_symbol_map(pricesFile)
    if y_sym not in symbol_map or x_sym not in symbol_map:
        return None

    engine = build_pair_engine(prices, y_sym, x_sym, cfg)
    print(f"\n{'=' * 72}\n Pair {pair_name} | CPCV g={cfg['n_groups']} k={cfg['n_test_groups']} "
          f"paths={engine['n_paths']} | train {cfg['start']}-{cfg['end']}\n{'=' * 72}", flush=True)

    study, best_params, best_value, study_name = optimize_pair_jax(y_sym, x_sym, engine, cfg)

    # Official metrics for reporting (numpy engine).
    train = run_official_eval(best_params, start_day=cfg["start"], end_day=cfg["end"])
    oos = run_official_eval(best_params, start_day=cfg["oos_start"], end_day=cfg["oos_end"])
    t_meanpl, t_ret, t_plstd, t_sharpe, t_dvol, t_days, t_pll, t_score = train
    o_meanpl, o_ret, o_plstd, o_sharpe, o_dvol, o_days, o_pll, o_score = oos

    # JAX diagnostics on the winner.
    jax_train = engine["full_one"](jpk.params_to_vec(best_params))
    jax_score = float(jax_train["score"])
    pll = np.asarray(jax_train["pll"][cfg["start"] - 1: cfg["end"]])
    path_scores = np.asarray(jpk.cpcv_path_scores(pll, engine["masks"]))

    print(f"\n[Final] {pair_name}")
    for k, v in best_params.items():
        if k not in ("pair_y", "pair_x"):
            print(f"  {k:<24}: {v}")
    print(f"  CPCV objective: {best_value:.4f} | path mean={path_scores.mean():.3f} "
          f"std={path_scores.std():.3f} frac>0={(path_scores > 0).mean():.1%}")
    print(f"  Train official {t_score:.3f} (jax {jax_score:.3f}) | "
          f"OOS official {o_score:.3f}", flush=True)

    y_idx, x_idx = engine["y_idx"], engine["x_idx"]
    strat = get_strategy_classes()
    filt_cfg = clean_sig_nt(strat.FiltCfg(
        q_beta=best_params["q_beta"], h=best_params["h"], cusum_k=best_params["cusum_k"],
        max_cov_inflate=best_params["max_cov_inflate"], cov_decay=best_params["cov_decay"],
        kalman_var_ewma_alpha=best_params["kalman_var_ewma_alpha"],
    ))
    ou_cfg = strat.OUCfg(
        ou_window=int(best_params["ou_window"]),
        kappa_ewma_alpha=float(best_params["kappa_ewma_alpha"]),
        sigma_ewma_alpha=float(best_params["sigma_ewma_alpha"]),
    )
    try:
        plot_trade_summary(prices, y_idx, x_idx, filt_cfg, ou_cfg, y_sym, x_sym, t_score,
                           title_prefix="CPCV_JAX", start_day=cfg["start"])
    except Exception as e:
        print(f"  [plot] skipped: {e}")

    return {
        "pair_name": pair_name, "y_symbol": y_sym, "x_symbol": x_sym,
        "config": {k: cfg[k] for k in (
            "start", "end", "oos_start", "oos_end", "n_groups", "n_test_groups",
            "purge", "pct_embargo", "std_lambda", "trials", "sobol_trials",
            "batch_size", "z_entry_rule",
        )},
        "study": study_name,
        "best_value": best_value,
        "final_params": best_params,
        "selection_source": "jax_cpcv",
        "cpcv": {
            "n_paths": engine["n_paths"],
            "path_score_mean": float(path_scores.mean()),
            "path_score_std": float(path_scores.std()),
            "path_score_frac_pos": float((path_scores > 0).mean()),
            "path_scores": [float(s) for s in path_scores],
        },
        "jax_train_score": jax_score,
        "train_metrics": {
            "meanpl": float(t_meanpl), "sharpe": float(t_sharpe), "score": float(t_score),
            "ret": float(t_ret), "total_return": float(np.sum(t_pll)),
        },
        "oos_metrics": {
            "meanpl": float(o_meanpl), "sharpe": float(o_sharpe), "score": float(o_score),
            "ret": float(o_ret), "total_return": float(np.sum(o_pll)),
        },
        # Shape compatibility with old windowed report consumers.
        "windows": [],
        "global": {
            "study": study_name,
            "norm_mode": "cpcv",
            "best_value": best_value,
            "best_params": best_params,
            "final_params": best_params,
            "selection_source": "jax_cpcv",
            "train_metrics": {
                "meanpl": float(t_meanpl), "sharpe": float(t_sharpe), "score": float(t_score),
                "ret": float(t_ret), "total_return": float(np.sum(t_pll)),
            },
            "oos_metrics": {
                "meanpl": float(o_meanpl), "sharpe": float(o_sharpe), "score": float(o_score),
                "ret": float(o_ret), "total_return": float(np.sum(o_pll)),
            },
            "fragile": False,
            "sensitivities": [],
            "cluster": None,
            "tiebreak": None,
            "objective_comparison": {},
        },
    }


def init_worker(cfg):
    global WINDOWED_CFG
    WINDOWED_CFG = cfg
    latest_tuner.GLOBAL_CONFIG["module"] = cfg["module"]
    os.environ["TUNER_Z_ENTRY_RULE"] = cfg["z_entry_rule"]
    matplotlib.use("Agg")
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def process_pair_worker(pair_dict):
    try:
        return run_pair(pair_dict, WINDOWED_CFG)
    except Exception as e:
        print(f"[Worker] Error {pair_dict}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def print_summary(all_results):
    print(f"\n{'=' * 88}")
    print(" CPCV JAX TUNER SUMMARY")
    print(f"{'=' * 88}")
    for res in all_results:
        g = res["global"]
        c = res["cpcv"]
        print(f"\n{res['pair_name']}")
        print(f"  CPCV obj {res['best_value']:.3f} | paths {c['n_paths']} "
              f"mean {c['path_score_mean']:.2f} frac>0 {c['path_score_frac_pos']:.0%}")
        print(f"  train {g['train_metrics']['score']:.3f} / OOS {g['oos_metrics']['score']:.3f}")


def run_pipeline():
    cfg = WINDOWED_CFG
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    matplotlib.use("Agg")
    latest_tuner.GLOBAL_CONFIG["module"] = cfg["module"]
    os.environ["TUNER_Z_ENTRY_RULE"] = cfg["z_entry_rule"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Sync legacy aliases.
    cfg["n_windows"] = cfg["n_groups"]
    cfg["embargo"] = cfg["purge"]
    cfg["global_trials"] = cfg["trials"]

    prcAll, nt, nInst = loadPrices(pricesFile)
    pairs_data = load_pairs_from_json(cfg["pairs_file"])

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {cfg['module']} | z-entry: {cfg['z_entry_rule']}")
    print(f"CPCV: N={cfg['n_groups']} k={cfg['n_test_groups']} purge={cfg['purge']} "
          f"embargo={cfg['pct_embargo']:.2%} | objective mean - {cfg['std_lambda']}*std")
    print(f"Budget: {cfg['trials']} trials (sobol {cfg['sobol_trials']}, batch {cfg['batch_size']})")
    print(f"Train {cfg['start']}-{cfg['end']} | OOS {cfg['oos_start']}-{cfg['oos_end']}")
    print(f"Optuna DB: {cfg['db_file']}")

    get_storage(cfg["db_file"])

    all_results = []
    try:
        max_workers = max(1, min(cfg["workers"], len(pairs_data)))
        cfg["show_progress"] = max_workers == 1
        if max_workers == 1:
            init_worker(cfg)
            for pair in pairs_data:
                result = process_pair_worker(pair)
                if result:
                    all_results.append(result)
        else:
            print(f"\n>>> {len(pairs_data)} pair(s) across {max_workers} worker(s)")
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers, initializer=init_worker, initargs=(cfg,)) as ex:
                futures = {ex.submit(process_pair_worker, p): p for p in pairs_data}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        result = fut.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        print(f"Error {futures[fut]}: {e}")
    except KeyboardInterrupt:
        print("\n[!] Interrupt - saving completed results...")

    order = {f"{p['y']}-{p['x']}": i for i, p in enumerate(pairs_data)}
    all_results.sort(key=lambda r: order.get(r["pair_name"], 999))

    if all_results:
        print_summary(all_results)
        with open(cfg["out_file"], "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nSaved results to {cfg['out_file']}")
    else:
        print("No pairs completed.")
    print("\nPipeline execution fully complete.")


def run_benchmark(pair_y="RTTH", pair_x="NAYO", n_official=5, n_jax=20, batch_size=32):
    """Compare official eval vs jitted / vmapped JAX CPCV objective."""
    cfg = dict(WINDOWED_CFG)
    prices, _, _ = loadPrices(pricesFile)
    engine = build_pair_engine(prices, pair_y, pair_x, cfg)
    params = {
        "pair_y": pair_y, "pair_x": pair_x,
        "q_beta": 3.92e-09, "h": 22.07, "cusum_k": 1.02, "max_cov_inflate": 3540.0,
        "cov_decay": 0.95, "kalman_var_ewma_alpha": 0.00714, "ou_window": 171,
        "kappa_ewma_alpha": 0.01, "sigma_ewma_alpha": 0.01,
    }
    pvec = jpk.params_to_vec(params)

    # Warmup
    evaluate_params_cpcv(engine, params)
    engine["score_one"](pvec).block_until_ready()

    off_t = []
    for _ in range(n_official):
        t0 = time.perf_counter()
        run_official_eval(params, cfg["start"], cfg["end"])
        off_t.append(time.perf_counter() - t0)

    # Old windowed cost proxy: n_windows independent official evals per trial.
    n_windows = 4
    windowed_trial_s = float(np.mean(off_t)) * n_windows

    jax_score_t = []
    for _ in range(n_jax):
        t0 = time.perf_counter()
        engine["score_one"](pvec).block_until_ready()
        jax_score_t.append(time.perf_counter() - t0)

    jax_cpcv_t = []
    for _ in range(n_jax):
        t0 = time.perf_counter()
        engine["obj_one"](pvec).block_until_ready()
        jax_cpcv_t.append(time.perf_counter() - t0)

    rng = np.random.default_rng(0)
    base = np.asarray(pvec)
    pvecs = np.stack([base * (1 + 0.03 * rng.standard_normal(len(base))) for _ in range(batch_size)])
    pvecs[:, 6] = np.clip(np.round(pvecs[:, 6]), 50, 200)
    engine["obj_batch"](pvecs).block_until_ready()  # compile batch
    batch_t = []
    for _ in range(max(3, n_jax // 4)):
        t0 = time.perf_counter()
        engine["obj_batch"](pvecs).block_until_ready()
        batch_t.append(time.perf_counter() - t0)

    # Rank correlation sample
    samples = params_from_distributions(7, None, pair_y, pair_x, 16)
    off_s, jax_s = [], []
    for p in samples:
        off_s.append(run_official_eval(p, cfg["start"], cfg["end"])[-1])
        jax_s.append(float(engine["score_one"](jpk.params_to_vec(p)).block_until_ready()))
    corr = float(np.corrcoef(off_s, jax_s)[0, 1]) if np.std(off_s) > 0 else float("nan")
    spearman = float(np.corrcoef(
        np.argsort(np.argsort(off_s)), np.argsort(np.argsort(jax_s)))[0, 1])

    metrics = {
        "pair": f"{pair_y}-{pair_x}",
        "official_eval_s": float(np.mean(off_t)),
        "legacy_windowed_trial_s_est": windowed_trial_s,
        "jax_score_s": float(np.mean(jax_score_t)),
        "jax_cpcv_s": float(np.mean(jax_cpcv_t)),
        "jax_cpcv_batch_s": float(np.mean(batch_t)),
        "jax_cpcv_per_trial_in_batch_s": float(np.mean(batch_t) / batch_size),
        "batch_size": batch_size,
        "n_cpcv_paths": engine["n_paths"],
        "speedup_vs_official_score": float(np.mean(off_t) / np.mean(jax_score_t)),
        "speedup_vs_official_cpcv": float(np.mean(off_t) / np.mean(jax_cpcv_t)),
        "speedup_vs_legacy_windowed_trial": float(windowed_trial_s / (np.mean(batch_t) / batch_size)),
        "rank_corr_pearson": corr,
        "rank_corr_spearman": spearman,
        "official_score": float(run_official_eval(params, cfg["start"], cfg["end"])[-1]),
        "jax_score": float(engine["score_one"](pvec).block_until_ready()),
        "jax_cpcv_obj": float(engine["obj_one"](pvec).block_until_ready()),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPCV JAX tuner (windowed_tuner rewrite)")
    parser.add_argument("--module", default=WINDOWED_CFG["module"])
    parser.add_argument("--start", type=int, default=WINDOWED_CFG["start"])
    parser.add_argument("--end", type=int, default=WINDOWED_CFG["end"])
    parser.add_argument("--oos_start", type=int, default=WINDOWED_CFG["oos_start"])
    parser.add_argument("--oos_end", type=int, default=WINDOWED_CFG["oos_end"])
    parser.add_argument("--pairs_file", default=WINDOWED_CFG["pairs_file"])
    parser.add_argument("--n_groups", type=int, default=WINDOWED_CFG["n_groups"])
    parser.add_argument("--n_test_groups", type=int, default=WINDOWED_CFG["n_test_groups"])
    parser.add_argument("--purge", type=int, default=WINDOWED_CFG["purge"])
    parser.add_argument("--pct_embargo", type=float, default=WINDOWED_CFG["pct_embargo"])
    parser.add_argument("--std_lambda", type=float, default=WINDOWED_CFG["std_lambda"])
    parser.add_argument("--trials", type=int, default=WINDOWED_CFG["trials"])
    parser.add_argument("--sobol_trials", type=int, default=WINDOWED_CFG["sobol_trials"])
    parser.add_argument("--batch_size", type=int, default=WINDOWED_CFG["batch_size"])
    parser.add_argument("--z_entry_rule", choices=["new", "conventional"],
                        default=WINDOWED_CFG["z_entry_rule"])
    parser.add_argument("--workers", type=int, default=WINDOWED_CFG["workers"])
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--db_file", default=WINDOWED_CFG["db_file"])
    parser.add_argument("--out_file", default=WINDOWED_CFG["out_file"])
    # Legacy flags accepted but remapped.
    parser.add_argument("--n_windows", type=int, default=None,
                        help="Legacy alias for --n_groups")
    parser.add_argument("--embargo", type=int, default=None,
                        help="Legacy alias for --purge")
    parser.add_argument("--window_trials", type=int, default=None)
    parser.add_argument("--global_trials", type=int, default=None,
                        help="Legacy alias for --trials")
    parser.add_argument("--norm_mode", default=None)
    parser.add_argument("--tiebreak_top_m", type=int, default=None)
    parser.add_argument("--tiebreak_folds", type=int, default=None)
    parser.add_argument("--tiebreak_fold_embargo", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true",
                        help="Print speedup metrics and exit")
    args = parser.parse_args()

    cfg_update = {k: v for k, v in vars(args).items()
                  if v is not None and k not in (
                      "benchmark", "n_windows", "embargo", "window_trials",
                      "global_trials", "norm_mode", "tiebreak_top_m",
                      "tiebreak_folds", "tiebreak_fold_embargo",
                  )}
    WINDOWED_CFG.update(cfg_update)
    if args.n_windows is not None:
        WINDOWED_CFG["n_groups"] = args.n_windows
    if args.embargo is not None:
        WINDOWED_CFG["purge"] = args.embargo
    if args.global_trials is not None:
        WINDOWED_CFG["trials"] = args.global_trials

    if args.benchmark:
        m = run_benchmark()
        print(json.dumps(m, indent=2))
    else:
        run_pipeline()
