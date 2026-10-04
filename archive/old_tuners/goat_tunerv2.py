#!/usr/bin/env python
"""Interactive Eval Tuner - Unified Strategy Tuning & Evaluation.

Merges the interactive graphical tuning methodology with the official Algothon
evaluation engine. Includes exact MLE for OU fitting, multi-level perturbation
stability verification (5%, 10%, 20%) for both Manual and Auto tuning, multi-pair
JSON loading, a fully automated NSGA-III 3-Regime comparison, automatic JSON 
parameter exporting, an out-of-sample test period evaluation, and interactive
Optuna HTML dashboards.
"""

import argparse
import importlib
import os
import sys
import types
import json
import multiprocessing
from math import sqrt, floor, log10, log, exp
from pathlib import Path
from typing import NamedTuple
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIIISampler

# ============================================================================
# GLOBAL CONFIGURATION & SEARCH SPACES
# ============================================================================
GLOBAL_CONFIG = {
    "module": "testbedv2",
    "start": 31,  # Start eval AFTER the 30-day Kalman warmup
    "end": 750,
    "pairs_file": None,
    "skip_manual": True
}

pricesFile = "./prices.txt"
scoreDefaultParam = 1
SKIP_PLOTS = True  # Set dynamically in main
RESULTS_DIR = "auto_tuning_results"

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- OPTUNA TRIAL CONFIGURATIONS ---
N_RESTARTS = 1               
TRIALS_KALMAN = 300        
TRIALS_THRESHOLDS = 150     
TRIALS_NSGA3 = 1200          # Trials for the 3-Regime NSGA-III optimization

# --- UNIFIED SEARCH SPACES ---
SEARCH_SPACES = {
    "q_beta":          {"low": 1e-8, "high": 1e-3, "log": True},
    "h":               {"low": 2.0,  "high": 20.0},
    "cusum_k":         {"low": 0.05, "high": 2.0},
    "max_cov_inflate": {"low": 1, "high": 1e8, "log": True},
    "cov_decay":       {"low": 1e-5, "high": 0.1, "log": True},
    "z_entry":         {"low": 1.0,  "high": 3.0},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def round_sig(x, sig=3):
    if x == 0 or not np.isfinite(x):
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def run_optuna_ensemble(objective_func, direction, trials_per_restart, n_restarts=N_RESTARTS, use_cmaes=True, n_startup_trials=200):
    global_best_score = -1e6 if direction == "maximize" else 1e6
    global_best_params = {}

    for i in range(n_restarts):
        if use_cmaes:
            sampler = optuna.samplers.CmaEsSampler(
                seed=i * 100 + np.random.randint(0, 9999),
                n_startup_trials=n_startup_trials
            )
        else:
            sampler = optuna.samplers.TPESampler(
                seed=i * 100 + np.random.randint(0, 9999),
                n_startup_trials=n_startup_trials,
                multivariate=True
            )
            
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective_func, n_trials=trials_per_restart, show_progress_bar=False)

        try:
            if study.best_value is not None:
                if direction == "maximize" and study.best_value > global_best_score:
                    global_best_score = study.best_value
                    global_best_params = study.best_params
                elif direction == "minimize" and study.best_value < global_best_score:
                    global_best_score = study.best_value
                    global_best_params = study.best_params
        except ValueError:
            pass

    return global_best_params

# ============================================================================
# PAIR LOADING FROM JSON
# ============================================================================
def get_symbol_map(prices_file):
    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None, nrows=0)
    return {sym: i for i, sym in enumerate(df.columns)}

def load_pairs_from_json(file_path):
    if not file_path or not Path(file_path).exists():
        return [{"y": "HUXZ", "x": "ACAC"}]
    with open(file_path, 'r') as f:
        pairs_data = json.load(f)
    if not isinstance(pairs_data, list): return [{"y": "HUXZ", "x": "ACAC"}]
    return pairs_data

# ============================================================================
# OFFICIAL EVALUATION ENGINE
# ============================================================================
def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return (df.values).T, df.shape[0], df.shape[1]

def chargeFees(dvolumes, commRate): return np.sum(dvolumes * commRate)

def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10: return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

def loadStrategy(moduleName):
    path = Path(*moduleName.split(".")).with_suffix(".py")
    if not path.exists(): return importlib.import_module(moduleName)
    module = types.ModuleType(moduleName)
    module.__file__ = str(path.resolve())
    module.__package__ = moduleName.rpartition(".")[0]
    sys.modules[moduleName] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module

def reload_strategy():
    if GLOBAL_CONFIG["module"] in sys.modules:
        del sys.modules[GLOBAL_CONFIG["module"]]
    strat = loadStrategy(GLOBAL_CONFIG["module"])
    if hasattr(strat, "reset"): strat.reset()
    return strat

def calcPL(prcHist, testStartDay, testEndDay, getPosition_func, nInst, commRate, dlrPosLimit):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    todayPLL = []
    days = []
    startDay = testStartDay - 1

    for t in range(startDay, testEndDay + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < testEndDay:
            newPosOrig = getPosition_func(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        totDVolume += np.sum(dvolumes)
        comm = chargeFees(dvolumes, commRate)

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue

        ret = 0.0
        if totDVolume > 0: ret = value / totDVolume

        if t >= testStartDay:
            todayPLL.append(todayPL)
            days.append(t)

    pll = np.array(todayPLL)
    plmu, plstd = np.mean(pll), np.std(pll)
    annSharpe = np.sqrt(250) * plmu / plstd if plstd > 0 else 0.0
    return plmu, ret, plstd, annSharpe, totDVolume, days, pll

# ============================================================================
# PARAMETER INJECTION
# ============================================================================
ENV_MAP = {
    "pair_y":          "TUNER_PAIR_Y",
    "pair_x":          "TUNER_PAIR_X",
    "q_beta":          "TUNER_Q_BETA",
    "h":               "TUNER_H",
    "cusum_k":         "TUNER_CUSUM_K",
    "max_cov_inflate": "TUNER_MAX_COV_INFLATE",
    "cov_decay":       "TUNER_COV_DECAY",
    "z_entry":         "TUNER_Z_ENTRY",
    "z_exit":          "TUNER_Z_EXIT",
}

def inject_parameters(params: dict):
    for k, env_key in ENV_MAP.items():
        if k in params and params[k] is not None:
            os.environ[env_key] = str(params[k])
        else:
            os.environ.pop(env_key, None)

def run_official_eval(params, start_day=None, end_day=None):
    if start_day is None: start_day = GLOBAL_CONFIG["start"]
    if end_day is None: end_day = GLOBAL_CONFIG["end"]

    prcAll, nt, nInst = loadPrices(pricesFile)
    commRateArr = np.full(nInst, defaultCommRate); commRateArr[0] = inst0CommRate
    dlrPosLimitArr = np.full(nInst, defaultDlrPosLimit); dlrPosLimitArr[0] = inst0DlrPosLimit

    if "pair_y" not in params or "pair_x" not in params:
        raise ValueError("run_official_eval requires 'pair_y' and 'pair_x' in params")

    inject_parameters(params)
    strat = reload_strategy()
    meanpl, ret, plstd, sharpe, dvol, days, pll = calcPL(
        prcAll, start_day, end_day, strat.getMyPosition, nInst, commRateArr, dlrPosLimitArr
    )
    return meanpl, ret, plstd, sharpe, dvol, days, pll, score(meanpl, plstd)

# ============================================================================
# STRATEGY LOGIC — IMPORTED FROM testbedv2, NEVER DUPLICATED HERE
# ============================================================================
_STRAT_MODULE_CACHE = None

def get_strategy_classes():
    """Load the strategy module once and cache it for this process."""
    global _STRAT_MODULE_CACHE
    if _STRAT_MODULE_CACHE is None:
        _STRAT_MODULE_CACHE = loadStrategy(GLOBAL_CONFIG["module"])
    return _STRAT_MODULE_CACHE

def clean_sig_nt(nt_obj, sig=3):
    """Round every float field of a NamedTuple instance to `sig` sig figs."""
    updates = {f: round_sig(v, sig) for f, v in nt_obj._asdict().items() if isinstance(v, float)}
    return nt_obj._replace(**updates)

# ============================================================================
# MANUAL TUNING STEPS (WITH GRAPHS)
# ============================================================================
def interactive_tune_kalman(prices, y_idx, x_idx, pair_name):
    strat = get_strategy_classes()

    raw_y = prices[y_idx]
    raw_x = prices[x_idx]
    y_series = np.log(np.maximum(raw_y, 1e-12))
    x_series = np.log(np.maximum(raw_x, 1e-12))

    raw_ols_beta = float((raw_x @ raw_y) / max(float(raw_x @ raw_x), 1e-12))
    raw_spreads = raw_y - raw_ols_beta * raw_x

    print(f"\n{'=' * 60}")
    print(f" STEP 1: Unified Kalman & CUSUM Tuning ({pair_name})")
    print(f"{'=' * 60}")

    # --- Calculate Raw Beta (Super Fast Kalman) ---
    cfg_fast = strat.FiltCfg(q_beta=0.1, h=1e9, cusum_k=1.0, max_cov_inflate=1.0, cov_decay=0.95)
    filt_fast = strat.PairSegFilter(cfg_fast)
    raw_betas = []
    for t in range(len(y_series)):
        filt_fast.step(y_series[t], x_series[t])
        raw_betas.append(filt_fast.beta)
    raw_betas = np.array(raw_betas)

    def scad_penalty(x, a=3.7, lam=1.0):
        abs_x = np.abs(x)
        p = np.zeros_like(abs_x)
        mask1 = abs_x <= lam
        p[mask1] = lam * abs_x[mask1]
        mask2 = (abs_x > lam) & (abs_x < a * lam)
        p[mask2] = (2 * a * lam * abs_x[mask2] - abs_x[mask2]**2 - lam**2) / (2 * (a - 1))
        mask3 = abs_x >= a * lam
        p[mask3] = (a + 1) * lam**2 / 2.0
        return np.sum(p)

    def objective_unified(trial):
        q_beta = round_sig(trial.suggest_float("q_beta", **SEARCH_SPACES["q_beta"]), 3)
        h = round_sig(trial.suggest_float("h", **SEARCH_SPACES["h"]), 3)
        cusum_k = round_sig(trial.suggest_float("cusum_k", **SEARCH_SPACES["cusum_k"]), 3)
        max_cov = round_sig(trial.suggest_float("max_cov_inflate", **SEARCH_SPACES["max_cov_inflate"]), 3)
        cov_decay = round_sig(trial.suggest_float("cov_decay", **SEARCH_SPACES["cov_decay"]), 5)
        
        cfg = clean_sig_nt(strat.FiltCfg(q_beta=q_beta, h=h, cusum_k=cusum_k, max_cov_inflate=max_cov, cov_decay=cov_decay))
        filt = strat.PairSegFilter(cfg)
        betas, spreads = [], []
        for t in range(len(y_series)):
            filt.step(y_series[t], x_series[t])
            betas.append(filt.beta)
            spreads.append(y_series[t] - filt.beta * x_series[t])
            
        b = np.array(betas[30:])
        s = np.array(spreads[30:])
        if len(s) < 10: return 1e6
        
        spread_std = float(np.std(s))
        if spread_std < 0.01: return 1e6
        
        s_lag = s[:-1]; s_diff = np.diff(s); denom = float(s_lag @ s_lag)
        if denom < 1e-12: return 1e6
        slope = float(s_lag @ s_diff) / denom
        kappa = -slope
        if kappa <= 0 or kappa > 1.0: return 1e6
        
        if len(b) < 3: return 1e6
        b_2nd_deriv = np.diff(b, 2)
        lam = np.std(b_2nd_deriv) + 1e-9
        ramp_corner_penalty = scad_penalty(b_2nd_deriv, a=3.7, lam=lam)
        
        return (100.0 * ramp_corner_penalty) - (spread_std * kappa)

    best_params = run_optuna_ensemble(objective_unified, "minimize", trials_per_restart=max(150, TRIALS_KALMAN), use_cmaes=True)
    
    q_beta = round_sig(best_params["q_beta"], 3)
    h = round_sig(best_params["h"], 3)
    cusum_k = round_sig(best_params["cusum_k"], 3)
    max_cov = round_sig(best_params["max_cov_inflate"], 3)
    cov_decay = round_sig(best_params["cov_decay"], 5)

    current_cfg = clean_sig_nt(strat.FiltCfg(q_beta=q_beta, h=h, cusum_k=cusum_k, max_cov_inflate=max_cov, cov_decay=cov_decay))
    
    if SKIP_PLOTS:
        print(f"  [Headless] Kalman Accepted: q={current_cfg.q_beta}, h={current_cfg.h}, k={current_cfg.cusum_k}, cov={current_cfg.max_cov_inflate}, dec={current_cfg.cov_decay}")
        return current_cfg

    while True:
        filt = strat.PairSegFilter(current_cfg)
        betas, spreads, cm_ups, cm_dns, cv_ups, cv_dns, trigger_times = [], [], [], [], [], [], []
        for t in range(len(y_series)):
            out = filt.step(y_series[t], x_series[t])
            betas.append(filt.beta)
            spreads.append(out.spread)
            cm_ups.append(filt.cm_up)
            cm_dns.append(filt.cm_dn)
            cv_ups.append(filt.cv_up)
            cv_dns.append(filt.cv_dn)
            if out.cut: trigger_times.append(t)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
        ax1.plot(raw_y, label=f"Price Y ({pair_name.split('-')[0]})", color="royalblue")
        ax1.plot(raw_x, label=f"Price X ({pair_name.split('-')[1]})", color="darkorange", alpha=0.8)
        ax1.set_title(f"Step 1: Kalman & CUSUM ({pair_name}) | Triggers: {len(trigger_times)}")
        ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(raw_betas, label="Raw Beta (Fast)", color="gray", alpha=0.5, linestyle="--")
        ax2.plot(betas, label="Adaptive Beta (Unified Optuna)", color="purple")
        ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

        ax3.plot(spreads, label="Log-Spread Residuals", color="teal")
        for trig in trigger_times: ax3.axvline(trig, color="red", alpha=0.2, linewidth=1)
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left", fontsize=8); ax3.grid(alpha=0.3)

        ax4.plot(raw_spreads, label="Raw Price Spread", color="sienna")
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax4.legend(loc="upper left", fontsize=8); ax4.grid(alpha=0.3)

        ax5.plot(cm_ups, label="Mean CUSUM Up", color="crimson")
        ax5.plot(cm_dns, label="Mean CUSUM Down", color="darkorange")
        ax5.plot(cv_ups, label="Var CUSUM Up", color="blue", linestyle="--")
        ax5.plot(cv_dns, label="Var CUSUM Down", color="green", linestyle="--")
        ax5.axhline(current_cfg.h, color="black", linestyle="--", label=f"h ({current_cfg.h:.3g})")
        ax5.legend(loc="upper left", fontsize=8); ax5.grid(alpha=0.3)

        fig.subplots_adjust(top=0.96, bottom=0.04, hspace=0.35)
        print("\n>>> Close the plot window to continue...")
        plt.show()

        print(f"\n[Current Config]: q_beta={current_cfg.q_beta}, h={current_cfg.h}, cusum_k={current_cfg.cusum_k}, max_cov={current_cfg.max_cov_inflate}, cov_decay={current_cfg.cov_decay}")
        choice = input("Options:\n  [Enter] Accept\n  'edit' 'retry' (Unified)\nChoice: ").strip().lower()

        if choice == "" or choice == "accept":
            break
        elif choice == "edit":
            qb = input(f"new q_beta ({current_cfg.q_beta}): ").strip()
            hh = input(f"new h ({current_cfg.h}): ").strip()
            kk = input(f"new cusum_k ({current_cfg.cusum_k}): ").strip()
            ci = input(f"new max_cov_inflate ({current_cfg.max_cov_inflate}): ").strip()
            cd = input(f"new cov_decay ({current_cfg.cov_decay}): ").strip()
            current_cfg = strat.FiltCfg(
                q_beta=round_sig(float(qb), 3) if qb else current_cfg.q_beta,
                h=round_sig(float(hh), 3) if hh else current_cfg.h,
                cusum_k=round_sig(float(kk), 3) if kk else current_cfg.cusum_k,
                max_cov_inflate=round_sig(float(ci), 3) if ci else current_cfg.max_cov_inflate,
                cov_decay=round_sig(float(cd), 5) if cd else current_cfg.cov_decay
            )
        elif choice == "retry":
            print("  Re-running Unified optimization...")
            best_params = run_optuna_ensemble(objective_unified, "minimize", trials_per_restart=max(100, TRIALS_KALMAN // 2), use_cmaes=True)
            current_cfg = clean_sig_nt(strat.FiltCfg(
                q_beta=round_sig(best_params["q_beta"], 3),
                h=round_sig(best_params["h"], 3),
                cusum_k=round_sig(best_params["cusum_k"], 3),
                max_cov_inflate=round_sig(best_params["max_cov_inflate"], 3),
                cov_decay=round_sig(best_params["cov_decay"], 5)
            ))

    plt.close('all')
    return current_cfg


def generate_step2_data(prices, y_idx, x_idx, filt_cfg, trade_cfg):
    strat = get_strategy_classes()
    raw_y = prices[y_idx]
    raw_x = prices[x_idx]
    nt_total = len(raw_y)

    spec = strat.PairCfg(idx=0, y=0, x=1, filt=filt_cfg, trade=trade_cfg)
    agent = strat.PairAgent(spec)
    pair_prices = np.vstack([raw_y, raw_x])

    z_hist, kappa_hist, sigma_hist = [], [], []
    trades = []
    daily_pnl = []
    
    prev_pos_y, prev_pos_x = 0.0, 0.0
    prev_side = 0
    comm_rate = 0.0001

    for t in range(1, nt_total + 1):
        agent.catch_up(pair_prices[:, :t])
        z_hist.append(agent.current_z)
        kappa_hist.append(agent.current_kappa)
        sigma_hist.append(agent.current_sigma)

        if agent.side != prev_side:
            event = "ENTRY" if prev_side == 0 else "EXIT"
            trades.append((t - 1, event, agent.current_z, agent.side))

        py, px, _ = agent.dollar_positions()

        if t > 1 and raw_y[t-2] > 0 and raw_x[t-2] > 0:
            ret_y = (raw_y[t-1] / raw_y[t-2]) - 1.0
            ret_x = (raw_x[t-1] / raw_x[t-2]) - 1.0
            pnl = prev_pos_y * ret_y + prev_pos_x * ret_x
            comm = comm_rate * (abs(py - prev_pos_y) + abs(px - prev_pos_x))
            daily_pnl.append(pnl - comm)
        else:
            daily_pnl.append(0.0)

        prev_side = agent.side
        prev_pos_y, prev_pos_x = py, px

    pnl_hist = np.cumsum(daily_pnl).tolist()
    return z_hist, kappa_hist, sigma_hist, trades, pnl_hist


def plot_trade_summary(prices, y_idx, x_idx, filt_cfg, trade_cfg, pair_y, pair_x, stratScore, title_prefix="Step 2", save_png=False, filename_suffix=""):
    z_hist, kappa_hist, sigma_hist, trades, sim_pnl = generate_step2_data(prices, y_idx, x_idx, filt_cfg, trade_cfg)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    z_entry = trade_cfg.z_entry
    ax1.plot(z_hist, label="Dynamic Z-Score", color="dodgerblue", alpha=0.7)
    ax1.axhline(z_entry, color="red", linestyle="--", label=f"Entry K ({z_entry:.3g})")
    ax1.axhline(-z_entry, color="red", linestyle="--")
    ax1.axhline(0, color="black", linestyle="-", alpha=0.5)

    entry_long_t, entry_long_z = [], []
    entry_short_t, entry_short_z = [], []
    exit_t, exit_z = [], []

    for t_trade, typ, z_val, pos in trades:
        if typ == "ENTRY":
            if pos > 0:
                entry_long_t.append(t_trade)
                entry_long_z.append(z_val)
            else:
                entry_short_t.append(t_trade)
                entry_short_z.append(z_val)
        else:
            exit_t.append(t_trade)
            exit_z.append(z_val)

    if entry_long_t:
        ax1.scatter(entry_long_t, entry_long_z, color="lime", marker="^", s=120, zorder=10, label="Entry (Long)", edgecolors="black", linewidths=0.8)
    if entry_short_t:
        ax1.scatter(entry_short_t, entry_short_z, color="red", marker="v", s=120, zorder=10, label="Entry (Short)", edgecolors="black", linewidths=0.8)
    if exit_t:
        ax1.scatter(exit_t, exit_z, color="black", marker="x", s=100, zorder=10, label="Exit")
            
    ax1.set_title(f"{title_prefix}: Dynamic OU Regime & Trade Execution ({pair_y}-{pair_x})")
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(kappa_hist, label="Dynamic Kappa (Reversion Speed)", color="purple")
    ax2.plot(sigma_hist, label="Dynamic Sigma (Volatility)", color="darkorange")
    ax2.set_title("Rolling OU Parameters (Adapts to Market Regime)")
    ax2.legend(loc="upper left", fontsize=9); ax2.grid(alpha=0.3)

    ax3.plot(sim_pnl, label="Sim Cumulative PnL (Log Spread)", color="darkgreen", linewidth=2)
    ax3.set_title("Simulated Spread PnL (Official Eval Score: {:.4f})".format(stratScore))
    ax3.legend(loc="upper left", fontsize=9); ax3.grid(alpha=0.3)

    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.25)
    
    if SKIP_PLOTS or save_png:
        safe_pair = f"{pair_y}_{pair_x}"
        filepath = os.path.join(RESULTS_DIR, f"trade_summary_{safe_pair}_{title_prefix.replace(' ', '_')}{filename_suffix}.png")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  [Headless] Saved plot to {filepath}")
        plt.close('all')
    else:
        print("\n>>> Close the plot window to continue...")
        plt.show()
        plt.close('all')


def interactive_tune_thresholds_and_ou(prices, y_idx, x_idx, filt_cfg, pair_y, pair_x):
    strat = get_strategy_classes()

    print(f"\n{'=' * 60}")
    print(f" STEP 2: Dynamic OU & Thresholds ({pair_y}-{pair_x})")
    print(f"{'=' * 60}")

    def run_eval(z_entry, z_exit):
        params = {
            "pair_y": pair_y, "pair_x": pair_x,
            "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
            "max_cov_inflate": filt_cfg.max_cov_inflate, "cov_decay": filt_cfg.cov_decay, 
            "z_entry": z_entry, "z_exit": z_exit, "z_stop": 10.0,
        }
        return run_official_eval(params)

    def objective(trial):
        z_e = round_sig(trial.suggest_float("z_entry", **SEARCH_SPACES["z_entry"]), 3)
        z_x = 0.001
        *_, stratScore = run_eval(z_e, z_x)
        return stratScore if np.isfinite(stratScore) else -1e6

    best = run_optuna_ensemble(objective, "maximize", trials_per_restart=TRIALS_THRESHOLDS, use_cmaes=True)
    z_entry = round_sig(best["z_entry"], 3)
    z_exit = 0.01 

    if SKIP_PLOTS:
        *_, stratScore = run_eval(z_entry, z_exit)
        print(f"  [Headless] Thresholds Accepted: z_entry={z_entry} | Score={stratScore:.4f}")
        
        # Save manual step 2 plot to PNG in headless mode
        tc = strat.TradeCfg(z_entry, z_exit, 10.0)
        plot_trade_summary(prices, y_idx, x_idx, filt_cfg, tc, pair_y, pair_x, stratScore, title_prefix="Manual_Step2", save_png=True)
        return z_entry, z_exit

    while True:
        meanpl, ret, plstd, sharpe, dvol, days, pll, stratScore = run_eval(z_entry, z_exit)
        tc = strat.TradeCfg(z_entry, z_exit, 10.0)
        
        plot_trade_summary(prices, y_idx, x_idx, filt_cfg, tc, pair_y, pair_x, stratScore, title_prefix="Step 2")

        print(f"\n[Step 2 Results]: z_entry={z_entry} | Score={stratScore:.4f}")
        choice = input("Press [Enter] to ACCEPT, or 'override': ").strip().lower()

        if choice == "" or choice == "accept":
            break
        elif choice == "override":
            ze_in = input(f"custom z_entry ({z_entry}): ").strip()
            if ze_in: z_entry = round_sig(float(ze_in), 3)

    return z_entry, z_exit

# ============================================================================
# AUTO TUNING & STABILITY CHECKS
# ============================================================================
def run_perturbation_stability_check(base_params: dict, pair_y: str, pair_x: str, method_name: str = "Manual"):
    print(f"\n{'=' * 80}")
    print(f" STEP 3: Perturbation Stability Verification ({method_name} - {pair_y}-{pair_x})")
    print(f"{'=' * 80}")

    base = {**base_params, "pair_y": pair_y, "pair_x": pair_x}
    *_, base_score = run_official_eval(base)
    print(f"  Base Official Score: {base_score:.4f}")
    print("-" * 80)
    print(f"{'Parameter':<20} {'Value':<12} {'5% Drop':<10} {'10% Drop':<10} {'20% Drop':<10} {'Status'}")
    print("-" * 80)

    sensitivities = []
    levels = [0.05, 0.10, 0.20]

    for p_name, p_val in base_params.items():
        level_drops = []
        for level in levels:
            scores = []
            for sign in [-1, 1]:
                perturbed_val = round_sig(p_val * (1.0 + sign * level), 3)
                perturbed = {**base, p_name: perturbed_val}
                *_, s = run_official_eval(perturbed)
                scores.append(s)
            drop = max(0, (base_score - max(scores)) / abs(base_score)) * 100 if base_score != 0 else 0
            level_drops.append(drop)
        drop_5, drop_10, drop_20 = level_drops
        status = "OK" if drop_20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {p_val:<12.5g} {drop_5:<10.2f} {drop_10:<10.2f} {drop_20:<10.2f} {status}")
        sensitivities.append((p_name, drop_20, drop_5, drop_10))
        
    sensitivities.sort(key=lambda x: x[1], reverse=True)

    if not SKIP_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(sensitivities))
        drops_20 = [s[1] for s in sensitivities]

        ax.barh(y_pos, drops_20, color="crimson" if method_name == "Auto" else "royalblue", alpha=0.7, label=f"Max Score Drop at ±20% ({method_name})")
        ax.set_yticks(y_pos); ax.set_yticklabels([s[0] for s in sensitivities])
        ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Max Score Drop (%)")
        ax.set_title(f"Perturbation Tornado ({method_name} - {pair_y}-{pair_x}) | Base: {base_score:.4f}", fontweight="bold")
        ax.legend(); ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        print("\n>>> Close the plot window to continue...")
        plt.show()

    return sensitivities


def save_optuna_dashboard(study, pair_name):
    """Generates and saves an interactive HTML dashboard for the Optuna study."""
    try:
        import plotly
    except ImportError:
        print("  [Dashboard] Plotly not installed. Skipping dashboard generation. (pip install plotly)")
        return

    safe_pair = pair_name.replace('-', '_')
    filepath = os.path.join(RESULTS_DIR, f"optuna_dashboard_{safe_pair}.html")
    
    with open(filepath, "w") as f:
        f.write(f"<html><head><title>Optuna Dashboard: {pair_name}</title></head><body>")
        f.write(f"<h1>Optuna Dashboard: {pair_name}</h1>")
        
        f.write("<h2>1. Pareto Front (3 Regimes)</h2>")
        try:
            fig = optuna.visualization.plot_pareto_front(study, target_names=["Regime 1", "Regime 2", "Regime 3"])
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            f.write(f"<p>Could not generate Pareto front: {e}</p>")
            
        f.write("<h2>2. Parameter Importances (Target: Min Score across Regimes)</h2>")
        try:
            fig = optuna.visualization.plot_param_importances(
                study, target=lambda t: min(t.values), target_name="Maximin Score"
            )
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            f.write(f"<p>Could not generate importances: {e}</p>")
            
        f.write("<h2>3. Parallel Coordinate (Target: Min Score across Regimes)</h2>")
        try:
            fig = optuna.visualization.plot_parallel_coordinate(
                study, target=lambda t: min(t.values), target_name="Maximin Score"
            )
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            f.write(f"<p>Could not generate parallel coordinate: {e}</p>")
            
        f.write("<h2>4. Optimization History (Target: Min Score across Regimes)</h2>")
        try:
            fig = optuna.visualization.plot_optimization_history(
                study, target=lambda t: min(t.values), target_name="Maximin Score"
            )
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            f.write(f"<p>Could not generate history: {e}</p>")
            
        f.write("</body></html>")
        
    print(f"  [Dashboard] Saved interactive dashboard to {filepath}")


def run_full_auto_tuner_comparison_3_regimes(prices, y_idx, x_idx, pair_y: str, pair_x: str):
    print(f"\n{'=' * 60}")
    print(f" STEP 4: 3-REGIME NSGA-III AUTO TUNER ({pair_y}-{pair_x})")
    print(f"{'=' * 60}")

    def objective(trial):
        params = {
            "pair_y": pair_y, "pair_x": pair_x,
            "q_beta": round_sig(trial.suggest_float("q_beta", **SEARCH_SPACES["q_beta"]), 3),
            "h": round_sig(trial.suggest_float("h", **SEARCH_SPACES["h"]), 3),
            "cusum_k": round_sig(trial.suggest_float("cusum_k", **SEARCH_SPACES["cusum_k"]), 3),
            "max_cov_inflate": round_sig(trial.suggest_float("max_cov_inflate", **SEARCH_SPACES["max_cov_inflate"]), 3),
            "cov_decay": round_sig(trial.suggest_float("cov_decay", **SEARCH_SPACES["cov_decay"]), 5),
            "z_entry": round_sig(trial.suggest_float("z_entry", **SEARCH_SPACES["z_entry"]), 3),
            "z_exit": 0.01,
            "z_stop": 10.0,
        }
        
        # Evaluate on 3 distinct 250-day regimes
        *_, score_r1 = run_official_eval(params, start_day=31,  end_day=280)
        *_, score_r2 = run_official_eval(params, start_day=281, end_day=530)
        *_, score_r3 = run_official_eval(params, start_day=531, end_day=750)
        
        return score_r1, score_r2, score_r3

    # Optuna's NSGAIIISampler automatically generates reference points if none are provided
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize"],
        sampler=NSGAIIISampler(seed=42)
    )
    
    study.optimize(objective, n_trials=TRIALS_NSGA3, show_progress_bar=False)

    # Save the Optuna HTML Dashboard
    save_optuna_dashboard(study, f"{pair_y}-{pair_x}")

    # Maximin selection
    best_trial = None
    best_min_score = -1e6
    
    for trial in study.best_trials:
        min_score = min(trial.values[0], trial.values[1], trial.values[2])
        if min_score > best_min_score:
            best_min_score = min_score
            best_trial = trial
            
    best_params = best_trial.params
    best_params = {k: round_sig(v, 3) if isinstance(v, float) else v for k, v in best_params.items()}
    best_params["z_exit"] = 0.001 
    
    print(f"\n[3-Regime Best Parameters (Maximin Score: {best_min_score:.4f})]:")
    for k, v in best_params.items():
        print(f"  {k:<18}: {v}")

    eval_params = {**best_params, "pair_y": pair_y, "pair_x": pair_x, "z_stop": 10.0}
    meanpl, ret, plstd, sharpe, dvol, days, pll, auto_score = run_official_eval(eval_params)
    
    print(f"Official Full-Period Score: {auto_score:.4f}")

    strat = get_strategy_classes()
    filt_cfg = clean_sig_nt(strat.FiltCfg(
        q_beta=best_params["q_beta"], 
        h=best_params["h"], 
        cusum_k=best_params["cusum_k"], 
        max_cov_inflate=best_params["max_cov_inflate"], 
        cov_decay=best_params["cov_decay"]
    ))
    tc = strat.TradeCfg(best_params["z_entry"], best_params["z_exit"], 10.0)
    
    # Force saving the final Auto tuning graph
    plot_trade_summary(prices, y_idx, x_idx, filt_cfg, tc, pair_y, pair_x, auto_score, title_prefix="Auto_NSGA3", save_png=True)

    return {
        "meanpl": meanpl, "sharpe": sharpe, "score": auto_score, "ret": ret,
        "params": best_params, "days": days, "pll": pll.tolist()
    }

# ============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================
def init_worker(config):
    global GLOBAL_CONFIG, SKIP_PLOTS
    GLOBAL_CONFIG = config
    SKIP_PLOTS = True
    import matplotlib
    matplotlib.use("Agg")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def process_pair_worker(pair_dict):
    y_sym = pair_dict["y"]
    x_sym = pair_dict["x"]
    pair_name = f"{y_sym}-{x_sym}"
    
    try:
        prcAll, nt, nInst = loadPrices(pricesFile)
        symbol_map = get_symbol_map(pricesFile)
        if y_sym not in symbol_map or x_sym not in symbol_map: return None
            
        y_idx = symbol_map[y_sym]
        x_idx = symbol_map[x_sym]

        print(f"[Worker] Starting tuning for {pair_name}...", flush=True)
        
        manual_params = None
        sensitivities_manual = None
        manual_metrics = None
        train_days = None
        train_pll_manual = None

        if not GLOBAL_CONFIG.get("skip_manual", False):
            filt_cfg = interactive_tune_kalman(prcAll, y_idx, x_idx, pair_name)
            z_entry, z_exit = interactive_tune_thresholds_and_ou(prcAll, y_idx, x_idx, filt_cfg, y_sym, x_sym)

            manual_params = {
                "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
                "max_cov_inflate": filt_cfg.max_cov_inflate, "cov_decay": filt_cfg.cov_decay, 
                "z_entry": z_entry, "z_exit": z_exit, "z_stop": 10.0,
            }

            sensitivities_manual = run_perturbation_stability_check(manual_params, y_sym, x_sym, "Manual")
            
            eval_manual = {**manual_params, "pair_y": y_sym, "pair_x": x_sym}
            m_meanpl, m_ret, m_plstd, m_sharpe, m_dvol, days, pll, stratScore = run_official_eval(eval_manual)
            manual_metrics = {"meanpl": m_meanpl, "sharpe": m_sharpe, "score": stratScore, "ret": m_ret, "total_return": float(np.sum(pll))}
            train_days = days
            train_pll_manual = pll.tolist()

        auto_results = run_full_auto_tuner_comparison_3_regimes(prcAll, y_idx, x_idx, y_sym, x_sym)
        auto_metrics = {"meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], "score": auto_results['score'], "ret": auto_results['ret'], "total_return": float(np.sum(auto_results['pll']))}
        
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym, "Auto")
        
        test_start, test_end = 501, 750
        test_metrics = {}

        if manual_params:
            m_eval = {**manual_params, "pair_y": y_sym, "pair_x": x_sym}
            m_meanpl, m_ret, m_plstd, m_sharpe, m_dvol, m_days, m_pll, m_score = run_official_eval(m_eval, test_start, test_end)
            test_metrics["manual"] = {"meanpl": m_meanpl, "sharpe": m_sharpe, "score": m_score, "ret": m_ret, "total_return": float(np.sum(m_pll))}

        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym, "z_stop": 10.0}
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end)
        test_metrics["auto"] = {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret, "total_return": float(np.sum(a_pll))}

        score_str = f" | Auto Score: {auto_results['score']:.2f}"
        if manual_metrics:
            score_str = f" | Manual Score: {manual_metrics['score']:.2f}" + score_str

        print(f"[Worker] Completed {pair_name}{score_str}", flush=True)
        
        return {
            "pair_name": pair_name, "y_symbol": y_sym, "x_symbol": x_sym,
            "manual_params": manual_params, "manual_metrics": manual_metrics,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics,
            "test_metrics": test_metrics,
            "sensitivities_manual": sensitivities_manual,
            "sensitivities_auto": sensitivities_auto,
            "train_days": train_days if train_days else auto_results["days"],
            "train_pll_manual": train_pll_manual if train_pll_manual else [],
            "train_pll_auto": auto_results["pll"]
        }
        
    except Exception as e:
        print(f"[Worker] Error processing {pair_name}: {e}", flush=True)
        return None

# ============================================================================
# OUTPUT FORMATTING AND SAVING
# ============================================================================
def print_perturbation_summary(all_sensitivities):
    print(f"\n{'=' * 80}")
    print(f" PERTURBATION STABILITY SUMMARY (All Pairs & Methods)")
    print(f"{'=' * 80}")

    param_drops = {}
    for pair_data in all_sensitivities:
        if not pair_data: continue
        for p_name, drop_20, drop_5, drop_10 in pair_data["sensitivities"]:
            if p_name not in param_drops: param_drops[p_name] = {"5": [], "10": [], "20": []}
            param_drops[p_name]["5"].append(drop_5)
            param_drops[p_name]["10"].append(drop_10)
            param_drops[p_name]["20"].append(drop_20)

    print(f"{'Parameter':<20} {'5% Avg':<10} {'5% Max':<10} {'10% Avg':<10} {'10% Max':<10} {'20% Avg':<10} {'20% Max':<10} {'Status'}")
    print("-" * 80)
    for p_name, drops in param_drops.items():
        avg5, max5 = np.mean(drops["5"]), np.max(drops["5"])
        avg10, max10 = np.mean(drops["10"]), np.max(drops["10"])
        avg20, max20 = np.mean(drops["20"]), np.max(drops["20"])
        status = "OK" if max20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {avg5:<10.2f} {max5:<10.2f} {avg10:<10.2f} {max10:<10.2f} {avg20:<10.2f} {max20:<10.2f} {status}")
    print("-" * 80)

def save_params_to_json(all_pair_results):
    print("\nSaving all results to JSON...")
    save_list = []
    for pair_res in all_pair_results:
        if not pair_res: continue
        
        sens_manual = [
            {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
            for p, d20, d5, d10 in pair_res.get('sensitivities_manual', [])
        ] if pair_res.get('sensitivities_manual') else []

        sens_auto = [
            {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
            for p, d20, d5, d10 in pair_res.get('sensitivities_auto', [])
        ]

        save_list.append({
            "pair_name": pair_res['pair_name'],
            "y_symbol": pair_res['y_symbol'],
            "x_symbol": pair_res['x_symbol'],
            "manual_params": pair_res.get('manual_params'),
            "manual_metrics": pair_res.get('manual_metrics'),
            "auto_params": pair_res['auto_params'],
            "auto_metrics": pair_res['auto_metrics'],
            "test_metrics": pair_res.get('test_metrics', {}),
            "perturbation_summary": {
                "manual": sens_manual,
                "auto": sens_auto
            }
        })

    filepath = os.path.join(RESULTS_DIR, "tuning_results.json")
    with open(filepath, 'w') as f:
        json.dump(save_list, f, indent=4)
    print(f"Successfully saved all data to {filepath}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_interactive_pipeline():
    global SKIP_PLOTS, GLOBAL_CONFIG
    
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if GLOBAL_CONFIG.get("skip_manual", False):
        SKIP_PLOTS = True
        print(">>> Skip Manual flag detected. Forcing Headless parallel mode.")
        import matplotlib
        matplotlib.use("Agg")
    else:
        skip_input = input("Skip all interactive graphs and run fully automated parallel? (y/n): ").strip().lower()
        if skip_input == 'y':
            SKIP_PLOTS = True
            print(">>> Headless parallel mode activated.")
            import matplotlib
            matplotlib.use("Agg")
        else:
            SKIP_PLOTS = False
            print(">>> Interactive graphical mode activated.")

    prcAll, nt, nInst = loadPrices(pricesFile)
    pairs_data = load_pairs_from_json(GLOBAL_CONFIG["pairs_file"])

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {GLOBAL_CONFIG['module']}")
    print(f"Evaluation window: days {GLOBAL_CONFIG['start']}-{GLOBAL_CONFIG['end']}")
    print(f"Results directory: {RESULTS_DIR}/")

    all_pair_results = []
    all_sensitivities = []

    try:
        if SKIP_PLOTS:
            n_cores = multiprocessing.cpu_count()
            max_workers = min(n_cores, len(pairs_data))
            print(f"\n>>> Parallelizing across {max_workers} CPU cores...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GLOBAL_CONFIG,)) as executor:
                futures = {executor.submit(process_pair_worker, pair): pair for pair in pairs_data}
                for future in concurrent.futures.as_completed(futures):
                    pair = futures[future]
                    try:
                        result = future.result()
                        if result:
                            all_pair_results.append(result)
                            if result.get("sensitivities_manual"):
                                all_sensitivities.append({"pair_name": f"{result['pair_name']} (Manual)", "sensitivities": result["sensitivities_manual"]})
                            all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})
                    except Exception as e:
                        print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")
        else:
            for i, pair in enumerate(pairs_data):
                result = process_pair_worker(pair)
                if result:
                    all_pair_results.append(result)
                    if result.get("sensitivities_manual"):
                        all_sensitivities.append({"pair_name": f"{result['pair_name']} (Manual)", "sensitivities": result["sensitivities_manual"]})
                    all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})

    except KeyboardInterrupt:
        print("\n\n[!] Keyboard Interrupt detected. Shutting down gracefully...")
        print("[!] Saving whatever results were completed before the interrupt...")

    pair_order = {f"{p['y']}-{p['x']}": i for i, p in enumerate(pairs_data)}
    all_pair_results.sort(key=lambda x: pair_order.get(x['pair_name'], 999))

    if all_pair_results:
        print_perturbation_summary(all_sensitivities)
        save_params_to_json(all_pair_results)
    else:
        print("No pairs were completed. No JSON file will be saved.")
        
    print("\nPipeline execution fully complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive/Parallel eval tuner")
    parser.add_argument("--module", default="testbedv2", help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=31, help="First day included in score (31 to skip warmup)")
    parser.add_argument("--end", type=int, default=500, help="Last day included in score")
    parser.add_argument("--pairs_file", default="pairs.json", help="JSON file containing list of pairs")
    parser.add_argument("--skip_manual", action="store_true", help="Skip manual tuning steps and only run Auto NSGA-III")
    args = parser.parse_args()
    
    GLOBAL_CONFIG.update(vars(args))
    run_interactive_pipeline()