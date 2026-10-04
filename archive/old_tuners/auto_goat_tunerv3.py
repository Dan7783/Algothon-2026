#!/usr/bin/env python
"""Fully Automated NSGA-II Robust Eval Tuner.

Runs a purely automated multi-regime optimization for pairs.
Splits the training data into N regimes and optimizes two objectives:
    1. Maximize Median(Score) - Ensures typical performance is strong (robust to outliers).
    2. Minimize StdDev(Score) - Punishes inconsistency between regimes without over-penalizing single outliers.
Uses NSGA-II to map the Pareto front, then automatically selects the "Knee Point".

Data is streamed sequentially to an SQLite database (tuning_data.db) to allow proper 
data analysis on every possible pair without holding results in memory.
"""

import argparse
import importlib
import os
import sys
import types
import json
import sqlite3
import multiprocessing
from math import floor, log10
from pathlib import Path
from itertools import combinations, permutations
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler

# ============================================================================
# GLOBAL CONFIGURATION & SEARCH SPACES
# ============================================================================
GLOBAL_CONFIG = {
    "module": "testbedv2",
    "start": 31,
    "end": 750,
    "pairs_file": None,
    "n_regimes": 5,      # 5 regimes of 250 days aligns perfectly with sqrt(250) annualization
    "run_all_pairs": False,       # If True, ignore pairs_file and run on every possible pair
    "all_pairs_ordered": False    # If True with run_all_pairs, test both (y,x) and (x,y); else only combinations
}

pricesFile = "./prices.txt"
scoreDefaultParam = 1
RESULTS_DIR = "auto_tuning_results"
DB_FILE = os.path.join(RESULTS_DIR, "tuning_data.db")

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- OPTUNA TRIAL CONFIGURATIONS ---
TRIALS_NSGA2 = 1000        # 1200 trials is plenty for 2 objectives

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

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================
def init_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Table for final knee-point results and metrics
    c.execute('''
        CREATE TABLE IF NOT EXISTS pair_results (
            pair_name TEXT PRIMARY KEY,
            y_symbol TEXT,
            x_symbol TEXT,
            auto_params TEXT,
            auto_metrics TEXT,
            test_metrics TEXT,
            perturbation_summary TEXT
        )
    ''')
    
    # Table for raw Pareto front trials (for data analysis)
    c.execute('''
        CREATE TABLE IF NOT EXISTS pareto_fronts (
            pair_name TEXT,
            trial_number INTEGER,
            median_score REAL,
            neg_std_dev REAL,
            params TEXT,
            FOREIGN KEY(pair_name) REFERENCES pair_results(pair_name)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_result_to_db(db_path, result):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    pair_name = result['pair_name']
    
    try:
        # 1. Insert/Replace main pair results
        c.execute('''
            INSERT OR REPLACE INTO pair_results 
            (pair_name, y_symbol, x_symbol, auto_params, auto_metrics, test_metrics, perturbation_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pair_name,
            result['y_symbol'],
            result['x_symbol'],
            json.dumps(result['auto_params']),
            json.dumps(result['auto_metrics']),
            json.dumps(result.get('test_metrics', {})),
            json.dumps(result.get('sensitivities_auto', []))
        ))
        
        # 2. Clear old Pareto front data for this pair (if rerunning)
        c.execute('DELETE FROM pareto_fronts WHERE pair_name = ?', (pair_name,))
        
        # 3. Insert all Pareto front trials
        for trial in result.get('pareto_trials', []):
            c.execute('''
                INSERT INTO pareto_fronts 
                (pair_name, trial_number, median_score, neg_std_dev, params)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                pair_name,
                trial['number'],
                float(trial['median_score']),
                float(trial['neg_std_dev']),
                json.dumps(trial['params'])
            ))
            
        conn.commit()
        print(f"[DB] Saved results and Pareto front for {pair_name} to database.", flush=True)
    except Exception as e:
        print(f"[DB] Error saving {pair_name} to DB: {e}", flush=True)
    finally:
        conn.close()

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

def generate_all_pairs(prices_file, ordered=False):
    """Generate every possible pair from instruments in the prices file."""
    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None, nrows=0)
    symbols = list(df.columns)
    if ordered:
        pairs = [{"y": y, "x": x} for y, x in permutations(symbols, 2)]
    else:
        pairs = [{"y": y, "x": x} for y, x in combinations(symbols, 2)]
    print(f"[Pairs] Generated {len(pairs)} pairs from {len(symbols)} instruments (ordered={ordered})")
    return pairs

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

def run_official_eval(params, start_day=None, end_day=None, prcAll=None):
    if start_day is None: start_day = GLOBAL_CONFIG["start"]
    if end_day is None: end_day = GLOBAL_CONFIG["end"]

    if prcAll is None:
        prcAll, nt, nInst = loadPrices(pricesFile)
    else:
        nt, nInst = prcAll.shape[1], prcAll.shape[0]

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
# STRATEGY LOGIC
# ============================================================================
_STRAT_MODULE_CACHE = None

def get_strategy_classes():
    global _STRAT_MODULE_CACHE
    if _STRAT_MODULE_CACHE is None:
        _STRAT_MODULE_CACHE = loadStrategy(GLOBAL_CONFIG["module"])
    return _STRAT_MODULE_CACHE

def clean_sig_nt(nt_obj, sig=3):
    updates = {f: round_sig(v, sig) for f, v in nt_obj._asdict().items() if isinstance(v, float)}
    return nt_obj._replace(**updates)

# ============================================================================
# PLOTTING & DIAGNOSTICS
# ============================================================================
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

def plot_trade_summary(prices, y_idx, x_idx, filt_cfg, trade_cfg, pair_y, pair_x, stratScore, title_prefix="Auto_NSGA2", save_png=True):
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
                entry_long_t.append(t_trade); entry_long_z.append(z_val)
            else:
                entry_short_t.append(t_trade); entry_short_z.append(z_val)
        else:
            exit_t.append(t_trade); exit_z.append(z_val)

    if entry_long_t:
        ax1.scatter(entry_long_t, entry_long_z, color="lime", marker="^", s=120, zorder=10, label="Entry (Long)", edgecolors="black", linewidths=0.8)
    if entry_short_t:
        ax1.scatter(entry_short_t, entry_short_z, color="red", marker="v", s=120, zorder=10, label="Entry (Short)", edgecolors="black", linewidths=0.8)
    if exit_t:
        ax1.scatter(exit_t, exit_z, color="black", marker="x", s=100, zorder=10, label="Exit")
            
    ax1.set_title(f"{title_prefix}: Dynamic OU Regime & Trade Execution ({pair_y}-{pair_x})")
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(kappa_hist, label="Dynamic Kappa", color="purple")
    ax2.plot(sigma_hist, label="Dynamic Sigma", color="darkorange")
    ax2.set_title("Rolling OU Parameters")
    ax2.legend(loc="upper left", fontsize=9); ax2.grid(alpha=0.3)

    ax3.plot(sim_pnl, label="Sim Cumulative PnL", color="darkgreen", linewidth=2)
    ax3.set_title("Simulated Spread PnL (Official Score: {:.4f})".format(stratScore))
    ax3.legend(loc="upper left", fontsize=9); ax3.grid(alpha=0.3)

    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.25)
    
    if save_png:
        safe_pair = f"{pair_y}_{pair_x}"
        filepath = os.path.join(RESULTS_DIR, f"trade_summary_{safe_pair}_{title_prefix}.png")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  [Headless] Saved plot to {filepath}")
        plt.close('all')

def save_optuna_dashboard(study, pair_name):
    try:
        import plotly
    except ImportError:
        print("  [Dashboard] Plotly not installed. Skipping.")
        return

    safe_pair = pair_name.replace('-', '_')
    filepath = os.path.join(RESULTS_DIR, f"optuna_dashboard_{safe_pair}.html")
    
    with open(filepath, "w") as f:
        f.write(f"<html><head><title>Optuna Dashboard: {pair_name}</title></head><body>")
        f.write(f"<h1>Optuna Dashboard: {pair_name}</h1>")
        
        f.write("<h2>1. Pareto Front (Median Score vs. Neg Std Dev)</h2>")
        try:
            fig = optuna.visualization.plot_pareto_front(
                study, target_names=["Median Score", "Neg Std Dev"]
            )
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            f.write(f"<p>Could not generate Pareto front: {e}</p>")

        f.write("<h2>2. Hyperparameter Relationships (Contour - Target: Median)</h2>")
        try:
            fig = optuna.visualization.plot_contour(study, target=lambda t: t.values[0], target_name="Median Score")
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            f.write(f"<p>Could not generate contour: {e}</p>")
            
        f.write("<h2>3. Parameter Importances (Target: Median Score)</h2>")
        try:
            fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="Median Score")
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            f.write(f"<p>Could not generate importances: {e}</p>")
            
        f.write("</body></html>")
    print(f"  [Dashboard] Saved interactive dashboard to {filepath}")

# ============================================================================
# AUTO TUNING
# ============================================================================
def run_perturbation_stability_check(base_params: dict, pair_y: str, pair_x: str, prcAll):
    print(f"\n{'=' * 80}")
    print(f" Perturbation Stability Verification (Auto - {pair_y}-{pair_x})")
    print(f"{'=' * 80}")

    base = {**base_params, "pair_y": pair_y, "pair_x": pair_x}
    *_, base_score = run_official_eval(base, prcAll=prcAll)
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
                *_, s = run_official_eval(perturbed, prcAll=prcAll)
                scores.append(s)
            drop = max(0, (base_score - max(scores)) / abs(base_score)) * 100 if base_score != 0 else 0
            level_drops.append(drop)
        drop_5, drop_10, drop_20 = level_drops
        status = "OK" if drop_20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {p_val:<12.5g} {drop_5:<10.2f} {drop_10:<10.2f} {drop_20:<10.2f} {status}")
        sensitivities.append((p_name, drop_20, drop_5, drop_10))
        
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    return sensitivities

def run_nsga2_robust_tuner(prices, y_idx, x_idx, pair_y: str, pair_x: str, prcAll):
    n_regimes = GLOBAL_CONFIG["n_regimes"]
    print(f"\n{'=' * 60}")
    print(f" NSGA-II ROBUST TUNER ({n_regimes} Regimes) ({pair_y}-{pair_x})")
    print(f"{'=' * 60}")

    start = GLOBAL_CONFIG["start"]
    end = GLOBAL_CONFIG["end"]
    total_days = end - start + 1
    block_size = total_days // n_regimes

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
        
        scores = []
        for i in range(n_regimes):
            r_start = start + i * block_size
            r_end = start + (i + 1) * block_size - 1
            if i == n_regimes - 1: r_end = end
                
            *_, s = run_official_eval(params, start_day=r_start, end_day=r_end, prcAll=prcAll)
            scores.append(s)
            
        median_score = np.median(scores)
        std_dev = np.std(scores)
        neg_std = -std_dev/abs(median_score) if median_score != 0 else -std_dev
        
        return median_score, neg_std

    sampler = NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=["maximize", "maximize"], 
        sampler=sampler
    )
    study.optimize(objective, n_trials=TRIALS_NSGA2, show_progress_bar=False)

    save_optuna_dashboard(study, f"{pair_y}-{pair_x}")

    # Extract raw Pareto front trials for DB storage
    pareto_trials = []
    for trial in study.best_trials:
        pareto_trials.append({
            "number": trial.number,
            "median_score": trial.values[0],
            "neg_std_dev": trial.values[1],
            "params": trial.params
        })

    # Knee Point Selection (Median - Std Dev)
    best_trial = None
    best_combined_score = -1e6
    
    for trial in study.best_trials:
        combined_score = trial.values[0] + trial.values[1]  # Median + (-Std)
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_trial = trial

    best_params = best_trial.params
    best_params = {k: round_sig(v, 3) if isinstance(v, float) else v for k, v in best_params.items()}
    best_params["z_exit"] = 0.001 
    
    print(f"\n[NSGA-II Best Knee-Point Parameters (Median: {best_trial.values[0]:.4f}, StdDev: {-best_trial.values[1]:.4f})]:")
    for k, v in best_params.items():
        print(f"  {k:<18}: {v}")

    eval_params = {**best_params, "pair_y": pair_y, "pair_x": pair_x, "z_stop": 10.0}
    meanpl, ret, plstd, sharpe, dvol, days, pll, auto_score = run_official_eval(eval_params, prcAll=prcAll)
    print(f"Official Full-Period Score: {auto_score:.4f}")

    strat = get_strategy_classes()
    filt_cfg = clean_sig_nt(strat.FiltCfg(
        q_beta=best_params["q_beta"], h=best_params["h"], cusum_k=best_params["cusum_k"], 
        max_cov_inflate=best_params["max_cov_inflate"], cov_decay=best_params["cov_decay"]
    ))
    tc = strat.TradeCfg(best_params["z_entry"], best_params["z_exit"], 10.0)
    plot_trade_summary(prices, y_idx, x_idx, filt_cfg, tc, pair_y, pair_x, auto_score)

    return {
        "meanpl": meanpl, "sharpe": sharpe, "score": auto_score, "ret": ret,
        "params": best_params, "pareto_trials": pareto_trials
    }

# ============================================================================
# WORKER & PIPELINE
# ============================================================================
def init_worker(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config
    import matplotlib
    matplotlib.use("Agg")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def process_pair_worker(pair_dict):
    y_sym, x_sym = pair_dict["y"], pair_dict["x"]
    pair_name = f"{y_sym}-{x_sym}"
    
    try:
        prcAll, nt, nInst = loadPrices(pricesFile)
        symbol_map = get_symbol_map(pricesFile)
        if y_sym not in symbol_map or x_sym not in symbol_map: return None
            
        y_idx, x_idx = symbol_map[y_sym], symbol_map[x_sym]
        print(f"[Worker] Starting tuning for {pair_name}...", flush=True)
        
        auto_results = run_nsga2_robust_tuner(prcAll, y_idx, x_idx, y_sym, x_sym, prcAll)
        auto_metrics = {"meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], "score": auto_results['score'], "ret": auto_results['ret']}
        
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym, prcAll)
        
        test_start, test_end = 751, 1000
        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym, "z_stop": 10.0}
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end, prcAll=prcAll)
        test_metrics = {"auto": {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret}}

        print(f"[Worker] Completed {pair_name} | Auto Score: {auto_results['score']:.2f}", flush=True)
        
        return {
            "pair_name": pair_name, "y_symbol": y_sym, "x_symbol": x_sym,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics,
            "test_metrics": test_metrics, "sensitivities_auto": sensitivities_auto,
            "pareto_trials": auto_results["pareto_trials"]  # Pass Pareto data up
        }
    except Exception as e:
        print(f"[Worker] Error processing {pair_name}: {e}", flush=True)
        return None

def print_perturbation_summary(all_sensitivities):
    print(f"\n{'=' * 80}\n PERTURBATION STABILITY SUMMARY\n{'=' * 80}")
    param_drops = {}
    for pair_data in all_sensitivities:
        if not pair_data: continue
        for p_name, drop_20, drop_5, drop_10 in pair_data["sensitivities"]:
            if p_name not in param_drops: param_drops[p_name] = {"5": [], "10": [], "20": []}
            param_drops[p_name]["5"].append(drop_5); param_drops[p_name]["10"].append(drop_10); param_drops[p_name]["20"].append(drop_20)

    print(f"{'Parameter':<20} {'5% Avg':<10} {'5% Max':<10} {'10% Avg':<10} {'10% Max':<10} {'20% Avg':<10} {'20% Max':<10} {'Status'}")
    print("-" * 80)
    for p_name, drops in param_drops.items():
        avg5, max5 = np.mean(drops["5"]), np.max(drops["5"])
        avg10, max10 = np.mean(drops["10"]), np.max(drops["10"])
        avg20, max20 = np.mean(drops["20"]), np.max(drops["20"])
        status = "OK" if max20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {avg5:<10.2f} {max5:<10.2f} {avg10:<10.2f} {max10:<10.2f} {avg20:<10.2f} {max20:<10.2f} {status}")

def run_pipeline():
    global GLOBAL_CONFIG
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found."); return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize the SQLite Database
    init_database(DB_FILE)
    
    import matplotlib; matplotlib.use("Agg")
    prcAll, nt, nInst = loadPrices(pricesFile)
    
    # Decide which pairs to run
    if GLOBAL_CONFIG.get("run_all_pairs", False):
        pairs_data = generate_all_pairs(pricesFile, ordered=GLOBAL_CONFIG.get("all_pairs_ordered", False))
    else:
        pairs_data = load_pairs_from_json(GLOBAL_CONFIG["pairs_file"])

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Evaluation window: days {GLOBAL_CONFIG['start']}-{GLOBAL_CONFIG['end']}")
    print(f"Regimes: {GLOBAL_CONFIG['n_regimes']}")
    print(f"Total pairs to process: {len(pairs_data)}")
    print(f"Results database: {DB_FILE}")

    all_sensitivities = [] # Kept in memory only for lightweight final console summary
    try:
        n_cores = multiprocessing.cpu_count()
        max_workers = min(n_cores, len(pairs_data)) if len(pairs_data) > 0 else 1
        print(f"\n>>> Parallelizing across {max_workers} CPU cores...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GLOBAL_CONFIG,)) as executor:
            futures = {executor.submit(process_pair_worker, pair): pair for pair in pairs_data}
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        # 1. Stream to Database immediately
                        save_result_to_db(DB_FILE, result)
                        
                        # 2. Keep sensitivities in memory for final stdout summary
                        all_sensitivities.append({
                            "pair_name": result['pair_name'], 
                            "sensitivities": result["sensitivities_auto"]
                        })
                except Exception as e:
                    print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")
    except KeyboardInterrupt:
        print("\n\n[!] Keyboard Interrupt detected. Shutting down gracefully...")

    if all_sensitivities:
        print_perturbation_summary(all_sensitivities)
    else: print("No pairs were completed.")
    
    print(f"\nPipeline execution fully complete. All data saved sequentially to: {DB_FILE}")
    print("You can now analyze the data using standard SQL or Pandas:")
    print("  import pandas as pd, sqlite3")
    print("  conn = sqlite3.connect('auto_tuning_results/tuning_data.db')")
    print("  df = pd.read_sql('SELECT * FROM pareto_fronts', conn)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-II Robust Eval Tuner")
    parser.add_argument("--module", default="testbedv2", help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=31, help="First day included in score")
    parser.add_argument("--end", type=int, default=750, help="Last day included in score")
    parser.add_argument("--pairs_file", default="pairs.json", help="JSON file containing list of pairs")
    parser.add_argument("--n_regimes", type=int, default=5, help="Number of regimes to split the data into")
    parser.add_argument("--run_all_pairs", action="store_true", help="Run on every possible pair from the prices file (ignores pairs_file)")
    parser.add_argument("--all_pairs_ordered", action="store_true", help="When run_all_pairs is set, test both (y,x) and (x,y) directions")
    args = parser.parse_args()
    
    GLOBAL_CONFIG.update(vars(args))
    run_pipeline()