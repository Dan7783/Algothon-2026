#!/usr/bin/env python
"""Fully Automated BIPOP CMA-ES Eval Tuner with Purged Walk-Forward Validation.

Runs a purely automated CMA-ES optimization for pairs.
Uses Sobol initialization for the CMA-ES sampler to ensure better initial coverage.
Uses Purged Walk-Forward Validation on the training set (Days 30-800) to prevent overfitting.
Evaluates the final selected parameters on a strict Out-of-Sample window (Days 801-1000).
Persists every Optuna study to a shared SQLite database (openable with the Optuna Dashboard
VS Code extension or `optuna-dashboard`), plus trade summary PNGs and a JSON parameter export.
"""

import argparse
import importlib
import os
import sys
import threading
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
import optunahub
import pandas as pd

# ============================================================================
# GLOBAL CONFIGURATION & SEARCH SPACES
# ============================================================================
GLOBAL_CONFIG = {
    "module": "testbedv3",
    "start": 30,  # Start eval AFTER the 30-day Kalman warmup
    "end": 800,   # Train window ends at 800
    "pairs_file": None,
    "db_file": None,   # None -> RESULTS_DIR/OPTUNA_DB_FILE
    "out_file": None,  # None -> RESULTS_DIR/tuning_results.json
}

pricesFile = "./prices.txt"
scoreDefaultParam = 1
RESULTS_DIR = "auto_tuning_results"
OPTUNA_DB_FILE = "optuna_studies.db"  # single sqlite file, one study per pair inside it

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- OPTUNA TRIAL CONFIGURATIONS ---
TRIALS_CMAES = 500          # Total trials for CMA-ES optimization
N_JOBS_PER_PAIR = 4          # Parallel threads for evaluating population inside the study

# --- UNIFIED SEARCH SPACES ---
SEARCH_SPACES = {
    "q_beta":                       {"low": 1e-9, "high": 0.1, "log": True},
    "h":                            {"low": 2, "high": 40},
    "cusum_k":                      {"low": 0.05, "high": 2.0},
    "max_cov_inflate":              {"low": 1e2, "high": 1e8, "log": True},
    "cov_decay":                    {"low": 1e-5, "high": 0.1, "log": True},
    "kalman_var_ewma_alpha":        {"low": 1e-4, "high": 0.5, "log": True},
    "ou_window":                    {"low": 50, "high": 200},
    "kappa_ewma_alpha":             {"low": 1e-4, "high": 0.5, "log": True},
    "sigma_ewma_alpha":             {"low": 1e-4, "high": 0.5, "log": True},
}

# ============================================================================
# RAM-BUFFERED STUDY (batch-flushed to SQLite)
# ============================================================================
class RamFlushedStudy:
    """Optimize on InMemoryStorage; batch-flush completed trials to a persistent
    SQLite study every `flush_every` trials.

    Per-trial SQLite commits (each an fsync) are a dominant overhead when many
    workers write concurrently. InMemoryStorage does zero I/O during optimization;
    completed trials are appended to the persistent study in one transaction per
    batch, so the DB stays resumable and readable by downstream tooling. At most
    `flush_every` trials are lost on a crash. InMemoryStorage is RLock-guarded,
    so this is safe with n_jobs > 1 threads.
    """

    def __init__(self, study_name, sampler, db_file, flush_every=32):
        self.persistent = optuna.create_study(
            direction="maximize",
            storage=optuna.storages.RDBStorage(
                url=f"sqlite:///{db_file}",
                engine_kwargs={"connect_args": {"timeout": 120}}),
            study_name=study_name, load_if_exists=True)
        self.study = optuna.create_study(
            direction="maximize", sampler=sampler,
            storage=optuna.storages.InMemoryStorage())
        existing = self.persistent.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if existing:
            self.study.add_trials([self._copy(t) for t in existing])
        self.flush_every = flush_every
        self._pending = []
        self._lock = threading.Lock()

    @staticmethod
    def _copy(t):
        return optuna.trial.create_trial(
            state=t.state, value=t.value, params=t.params,
            distributions=t.distributions,
            user_attrs=t.user_attrs, system_attrs=t.system_attrs)

    def optimize(self, objective, n_trials, n_jobs=1, show_progress=False):
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                            callbacks=[self._on_trial], show_progress_bar=show_progress)
        self.flush()

    def _on_trial(self, study, trial):
        with self._lock:
            self._pending.append(self._copy(trial))
            if len(self._pending) >= self.flush_every:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if self._pending:
            self.persistent.add_trials(self._pending)
            self._pending.clear()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def round_sig(x, sig=3):
    if x == 0 or not np.isfinite(x):
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

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
_PRICES_CACHE = {}


def loadPrices(fn):
    # Cached per process: tuning workers call this once per trial; the matrix is
    # read-only downstream, so one parse per worker is enough.
    if fn not in _PRICES_CACHE:
        df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
        _PRICES_CACHE[fn] = ((df.values).T, df.shape[0], df.shape[1])
    return _PRICES_CACHE[fn]

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
    "pair_y":             "TUNER_PAIR_Y",
    "pair_x":             "TUNER_PAIR_X",
    "q_beta":             "TUNER_Q_BETA",
    "h":                  "TUNER_H",
    "cusum_k":            "TUNER_CUSUM_K",
    "max_cov_inflate":    "TUNER_MAX_COV_INFLATE",
    "cov_decay":          "TUNER_COV_DECAY",
    "kalman_var_ewma_alpha":     "TUNER_KALMAN_VAR_EWMA_ALPHA",
    "ou_window":          "TUNER_OU_WINDOW",
    "kappa_ewma_alpha":  "TUNER_KAPPA_EWMA_ALPHA",
    "sigma_ewma_alpha":  "TUNER_SIGMA_EWMA_ALPHA",
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
# STRATEGY LOGIC — IMPORTED FROM testbedv3, NEVER DUPLICATED HERE
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
# PURGED WALK-FORWARD VALIDATION
# ============================================================================
def evaluate_purged_walk_forward(params, start_day, end_day, n_folds=3, embargo=10):
    """
    Evaluates parameters using Purged Walk-Forward Validation.
    Slices the training window into contiguous blocks moving forward in time,
    injecting an embargo gap between them to prevent lookahead bias/leakage.
    Returns a robust score (mean - std) to penalize high variance across folds.
    """
    total_days = end_day - start_day + 1
    if total_days < n_folds * 30:  # Ensure folds are at least ~30 days
        n_folds = max(1, total_days // 30)
        
    fold_len = total_days // n_folds
    fold_scores = []
    current_start = start_day
    
    for i in range(n_folds):
        fold_end = current_start + fold_len - 1
        if i == n_folds - 1:
            fold_end = end_day
            
        if fold_end >= current_start:
            meanpl, ret, plstd, sharpe, dvol, days, pll, s = run_official_eval(params, start_day=current_start, end_day=fold_end)
            if dvol == 0.0:
                s -= 5.0  # Penalize non-trading folds heavily
            fold_scores.append(s)
                
        current_start = fold_end + 1 + embargo
        
    if not fold_scores:
        return -1e9
        
    mean_s = np.mean(fold_scores)
    std_s = np.std(fold_scores)
    
    # Robust objective: reward high mean, penalize high variance across walk-forward folds
    return mean_s - std_s

# ============================================================================
# PLOTTING & DIAGNOSTICS
# ============================================================================
def generate_step2_data(prices, y_idx, x_idx, filt_cfg, ou_cfg, start_day=None):
    """Replays the agent bar-by-bar and records everything needed to plot what it actually did.

    IMPORTANT: `agent.current_z` is the Kalman/CUSUM *innovation* z-score used only for regime-shift
    detection (covariance inflation) inside PairSegFilter — it is NOT the signal the state machine
    trades on. The actual trade trigger is the dimensionless OU spread z-score computed in
    testbedv3.PairAgent.catch_up() as:
        spread_z = (current_spread - current_theta) / (current_sigma / sqrt(2 * current_kappa))
    That value is a local variable in testbedv3 and isn't stored on the agent, so we recompute it
    here from the agent's already-public state (spread_hist, current_theta, current_sigma,
    current_kappa) using the identical formula, purely for plotting/diagnostics. We also record
    agent.a_star every bar since it is a *dynamic* threshold (re-solved each step in
    _update_dynamic_ou), not a fixed constant — a single static line understates how much it moves.
    """
    if start_day is None:
        start_day = GLOBAL_CONFIG.get("start", 30)
    start_day = max(1, int(start_day))

    strat = get_strategy_classes()
    raw_y = prices[y_idx]
    raw_x = prices[x_idx]
    nt_total = len(raw_y)

    spec = strat.PairCfg(idx=0, y=0, x=1, filt=filt_cfg, ou=ou_cfg)
    agent = strat.PairAgent(spec)
    pair_prices = np.vstack([raw_y, raw_x])

    z_hist, kappa_hist, sigma_hist = [], [], []
    spread_z_hist, a_star_hist = [], []
    trades = []
    daily_pnl = []
    
    prev_pos_y, prev_pos_x = 0.0, 0.0
    prev_side = 0
    comm_rate = 0.0001

    for t in range(1, nt_total + 1):
        agent.catch_up(pair_prices[:, :t])

        if t < start_day:
            prev_side = agent.side
            prev_pos_y, prev_pos_x = agent.dollar_positions()[:2]
            continue

        z_hist.append(agent.current_z)
        kappa_hist.append(agent.current_kappa)
        sigma_hist.append(agent.current_sigma)
        a_star_hist.append(agent.a_star)

        # Recompute the true entry-trigger z-score (see docstring above).
        if agent.current_kappa > 0 and agent.spread_hist:
            ou_std = agent.current_sigma / sqrt(2.0 * agent.current_kappa)
            spread_z = (agent.spread_hist[-1] - agent.current_theta) / max(ou_std, 1e-8)
        else:
            spread_z = 0.0
        spread_z_hist.append(spread_z)

        # The state machine in testbedv3 never returns to side==0 once entered — it only flips
        # directly between +1 and -1. So there is no true "EXIT" event to detect; every side
        # change is a new directional entry, either from flat (fresh) or from the opposite side
        # (a flip/reversal, which is a real round-trip trade, not a no-op).
        if agent.side != prev_side:
            fresh = prev_side == 0
            if agent.side == 1:
                trades.append((t, "ENTRY_LONG" if fresh else "FLIP_LONG", spread_z, agent.side))
            elif agent.side == -1:
                trades.append((t, "ENTRY_SHORT" if fresh else "FLIP_SHORT", spread_z, agent.side))

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
    days = list(range(start_day, nt_total + 1))
    return z_hist, kappa_hist, sigma_hist, spread_z_hist, a_star_hist, trades, pnl_hist, days, agent.a_star


def plot_trade_summary(prices, y_idx, x_idx, filt_cfg, ou_cfg, pair_y, pair_x, stratScore, title_prefix="Auto", save_png=True, start_day=None):
    (z_hist, kappa_hist, sigma_hist, spread_z_hist, a_star_hist,
     trades, sim_pnl, days, z_entry_final) = generate_step2_data(prices, y_idx, x_idx, filt_cfg, ou_cfg, start_day=start_day)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Faint reference: the Kalman/CUSUM innovation z used for regime-shift detection.
    # This is NOT what triggers trades — kept only for context.
    ax1.plot(days, z_hist, label="Innovation Z (regime signal, not the trigger)", color="gray", alpha=0.25, linewidth=0.8)

    # The actual signal the state machine trades on.
    ax1.plot(days, spread_z_hist, label="OU Spread Z (actual trade trigger)", color="dodgerblue", alpha=0.85, linewidth=1.2)

    # The threshold is dynamic (re-solved every bar), so plot it as a moving line, not one static level.
    a_star_arr = np.asarray(a_star_hist)
    ax1.plot(days, a_star_arr, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
             label=f"Dynamic Entry Threshold ±a* (now {z_entry_final:.3g})")
    ax1.plot(days, -a_star_arr, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.axhline(0, color="black", linestyle="-", alpha=0.4)
    ax1.set_xlim(200, 1000)

    entry_long_t, entry_long_z = [], []
    entry_short_t, entry_short_z = [], []
    flip_long_t, flip_long_z = [], []
    flip_short_t, flip_short_z = [], []

    for t_trade, typ, z_val, pos in trades:
        if typ == "ENTRY_LONG":
            entry_long_t.append(t_trade); entry_long_z.append(z_val)
        elif typ == "ENTRY_SHORT":
            entry_short_t.append(t_trade); entry_short_z.append(z_val)
        elif typ == "FLIP_LONG":
            flip_long_t.append(t_trade); flip_long_z.append(z_val)
        elif typ == "FLIP_SHORT":
            flip_short_t.append(t_trade); flip_short_z.append(z_val)

    # Fresh entries from flat: solid triangles.
    if entry_long_t:
        ax1.scatter(entry_long_t, entry_long_z, color="lime", marker="^", s=120, zorder=10, label="Entry (Long)", edgecolors="black", linewidths=0.8)
    if entry_short_t:
        ax1.scatter(entry_short_t, entry_short_z, color="red", marker="v", s=120, zorder=10, label="Entry (Short)", edgecolors="black", linewidths=0.8)
    # Reversal trades (side flips straight from short<->long): diamonds, same color coding.
    if flip_long_t:
        ax1.scatter(flip_long_t, flip_long_z, color="lime", marker="D", s=90, zorder=10, label="Flip \u2192 Long", edgecolors="black", linewidths=0.8)
    if flip_short_t:
        ax1.scatter(flip_short_t, flip_short_z, color="red", marker="D", s=90, zorder=10, label="Flip \u2192 Short", edgecolors="black", linewidths=0.8)

    ax1.set_title(f"{title_prefix}: OU Spread Z & Trade Execution ({pair_y}-{pair_x})")
    ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(days, kappa_hist, label="Dynamic Kappa (Reversion Speed)", color="purple")
    ax2.plot(days, sigma_hist, label="Dynamic Sigma (Volatility)", color="darkorange")
    ax2.set_title("Rolling OU Parameters (Adapts to Market Regime)")
    ax2.legend(loc="upper left", fontsize=9); ax2.grid(alpha=0.3)

    ax3.plot(days, sim_pnl, label="Sim Cumulative PnL (Log Spread)", color="darkgreen", linewidth=2)
    ax3.set_title("Simulated Spread PnL (Official Eval Score: {:.4f})".format(stratScore))
    ax3.legend(loc="upper left", fontsize=9); ax3.grid(alpha=0.3)

    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.25)
    
    if save_png:
        safe_pair = f"{pair_y}_{pair_x}"
        filepath = os.path.join(RESULTS_DIR, f"trade_summary_{safe_pair}_{title_prefix}.png")
        plt.savefig(filepath, bbox_inches='tight')
        print(f"  [Headless] Saved plot to {filepath}")
        plt.close('all')


def get_optuna_storage():
    """Shared RDBStorage backing a single sqlite file in RESULTS_DIR.

    Every pair's study is written into this one file (one study per pair, keyed by
    study_name), so it can be opened as-is with `optuna-dashboard sqlite:///<path>`
    or the Optuna Dashboard VS Code extension — no separate HTML export step needed.

    Multiple worker processes may open this file concurrently (one per pair). SQLite
    serializes writers, so a generous busy-timeout is set to ride out lock contention
    instead of raising "database is locked" errors.
    """
    storage_path = GLOBAL_CONFIG.get("db_file") or os.path.join(RESULTS_DIR, OPTUNA_DB_FILE)
    return storage_path, optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )

# ============================================================================
# AUTO TUNING & STABILITY CHECKS
# ============================================================================
def run_perturbation_stability_check(base_params: dict, pair_y: str, pair_x: str):
    print(f"\n{'=' * 80}")
    print(f" PERTURBATION STABILITY VERIFICATION (Auto - {pair_y}-{pair_x})")
    print(f"{'=' * 80}")

    base = {**base_params, "pair_y": pair_y, "pair_x": pair_x}
    *_, base_score = run_official_eval(base, start_day=201, end_day=800)
    print(f"  Base Official Score: {base_score:.4f}")
    print("-" * 80)
    print(f"{'Parameter':<20} {'Value':<12} {'5% Drop':<10} {'10% Drop':<10} {'20% Drop':<10} {'Status'}")
    print("-" * 80)

    sensitivities = []
    levels = [0.05, 0.10, 0.20]

    for p_name, p_val in base_params.items():
        if p_name in ["pair_y", "pair_x"]: continue
        level_drops = []
        for level in levels:
            scores = []
            for sign in [-1, 1]:
                perturbed_val = round_sig(p_val * (1.0 + sign * level), 3)
                perturbed = {**base, p_name: perturbed_val}
                *_, s = run_official_eval(perturbed, start_day=200, end_day=800)
                scores.append(s)
            drop = max(0, (base_score - max(scores)) / abs(base_score)) * 100 if base_score != 0 else 0
            level_drops.append(drop)
        drop_5, drop_10, drop_20 = level_drops
        status = "OK" if drop_20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {p_val:<12.5g} {drop_5:<10.2f} {drop_10:<10.2f} {drop_20:<10.2f} {status}")
        sensitivities.append((p_name, drop_20, drop_5, drop_10))
        
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    return sensitivities


def run_cmaes_tuner(prices, y_idx, x_idx, pair_y: str, pair_x: str):
    print(f"\n{'=' * 60}")
    print(f" PURGED WALK-FORWARD BIPOP CMA-ES AUTO TUNER ({pair_y}-{pair_x})")
    print(f"{'=' * 60}")

    # --- DATA SPLITS ---
    # Train: 201-800 (Optimized via Purged Walk-Forward)
    # OOS:   801-1000 (Strictly unseen)
    train_start, train_end = 201, 800
    oos_start, oos_end = 801, 1000

    def objective(trial):
        params = {
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
        
        # Evaluate using Purged Walk-Forward across days 201-800
        robust_score = evaluate_purged_walk_forward(params, start_day=train_start, end_day=train_end, n_folds=4, embargo=10)
        return robust_score
    
    # --- BIPOP CMA-ES SAMPLER WITH SOBOL INITIALIZATION ---
    restart_cmaes_module = optunahub.load_module("samplers/restart_cmaes")
    storage_path = GLOBAL_CONFIG.get("db_file") or os.path.join(RESULTS_DIR, OPTUNA_DB_FILE)

    # Phase 1: Pure Sobol initialization, on a RAM-buffered study that batch-flushes
    # completed trials to SQLite (per-trial commits were the dominant I/O overhead).
    ram = RamFlushedStudy(f"{pair_y}-{pair_x}",
                          optuna.samplers.QMCSampler(seed=42, scramble=True),
                          storage_path)
    ram.optimize(objective, n_trials=256, n_jobs=N_JOBS_PER_PAIR, show_progress=True)

    # Phase 2: CMA-ES takes over
    # RestartCmaEsSampler automatically reads the finished trials from the in-memory study
    ram.study.sampler = restart_cmaes_module.RestartCmaEsSampler(
        restart_strategy="bipop",
        seed=42
        # Do NOT pass QMCSampler here
    )
    ram.optimize(objective, n_trials=TRIALS_CMAES, n_jobs=N_JOBS_PER_PAIR, show_progress=True)
    study = ram.persistent

    print(f"  [Optuna] Study '{pair_y}-{pair_x}' persisted to {storage_path}")
    print(f"  [Optuna] View it with: optuna-dashboard sqlite:///{storage_path}")
    print(f"  [Optuna] Or open {storage_path} with the Optuna Dashboard VS Code extension.")

    # --- SELECT BEST TRIAL ---
    # Since the objective already used Purged Walk-Forward, the best trial is strictly the most robust.
    best_trial = study.best_trial
    best_params = {**best_trial.params, "pair_y": pair_y, "pair_x": pair_x}
    best_params = {k: round_sig(v, 3) if isinstance(v, float) else v for k, v in best_params.items()}
    
    print(f"\n[Best Parameters (Walk-Forward Objective: {best_trial.value:.4f})]:")
    for k, v in best_params.items():
        print(f"  {k:<18}: {v}")

    # Evaluate the selected params on the full 201-800 period for reporting
    eval_params = {**best_params, "pair_y": pair_y, "pair_x": pair_x}
    meanpl, ret, plstd, sharpe, dvol, days, pll, auto_score = run_official_eval(eval_params, start_day=train_start, end_day=train_end)
    
    print(f"Official Train Period Score (201-800): {auto_score:.4f}")

    strat = get_strategy_classes()
    filt_cfg = clean_sig_nt(strat.FiltCfg(
        q_beta=best_params["q_beta"],
        h=best_params["h"],
        cusum_k=best_params["cusum_k"],
        max_cov_inflate=best_params["max_cov_inflate"],
        cov_decay=best_params["cov_decay"],
        kalman_var_ewma_alpha=best_params["kalman_var_ewma_alpha"],
    ))
    ou_cfg = strat.OUCfg(
        ou_window=int(best_params["ou_window"]),
        kappa_ewma_alpha=float(best_params["kappa_ewma_alpha"]),
        sigma_ewma_alpha=float(best_params["sigma_ewma_alpha"]),
    )
    
    plot_trade_summary(prices, y_idx, x_idx, filt_cfg, ou_cfg, pair_y, pair_x, auto_score, title_prefix="Purged_WF_CMA-ES", start_day=train_start)

    return {
        "meanpl": meanpl, "sharpe": sharpe, "score": auto_score, "ret": ret,
        "params": best_params, "days": days, "pll": pll.tolist()
    }

# ============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================
def init_worker(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config
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
        
        auto_results = run_cmaes_tuner(prcAll, y_idx, x_idx, y_sym, x_sym)
        auto_metrics = {"meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], "score": auto_results['score'], "ret": auto_results['ret'], "total_return": float(np.sum(auto_results['pll']))}
        
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym)
        
        # --- OOS VALIDATION (Official Unseen Test: 801-1000) ---
        test_start, test_end = 801, 1000
        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym}
        
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end)
        test_metrics = {
            "auto": {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret, "total_return": float(np.sum(a_pll))}
        }

        print(f"[Worker] Completed {pair_name} | Train Score: {auto_results['score']:.2f} | OOS Score: {a_score:.2f}", flush=True)
        
        return {
            "pair_name": pair_name, "y_symbol": y_sym, "x_symbol": x_sym,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics,
            "test_metrics": test_metrics,
            "sensitivities_auto": sensitivities_auto,
            "train_days": auto_results["days"],
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
    print(f" PERTURBATION STABILITY SUMMARY (All Pairs)")
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

        sens_auto = [
            {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
            for p, d20, d5, d10 in pair_res.get('sensitivities_auto', [])
        ]

        save_list.append({
            "pair_name": pair_res['pair_name'],
            "y_symbol": pair_res['y_symbol'],
            "x_symbol": pair_res['x_symbol'],
            "auto_params": pair_res['auto_params'],
            "auto_metrics": pair_res['auto_metrics'],
            "test_metrics": pair_res.get('test_metrics', {}),
            "perturbation_summary": {
                "auto": sens_auto
            }
        })

    filepath = GLOBAL_CONFIG.get("out_file") or os.path.join(RESULTS_DIR, "tuning_results.json")
    with open(filepath, 'w') as f:
        json.dump(save_list, f, indent=4)
    print(f"Successfully saved all data to {filepath}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    global GLOBAL_CONFIG
    
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    import matplotlib
    matplotlib.use("Agg")

    prcAll, nt, nInst = loadPrices(pricesFile)
    pairs_data = load_pairs_from_json(GLOBAL_CONFIG["pairs_file"])

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {GLOBAL_CONFIG['module']}")
    print(f"Evaluation Train window: days {GLOBAL_CONFIG['start']}-{GLOBAL_CONFIG['end']}")
    print(f"Sampler: BIPOP CMA-ES with Sobol Initialization")
    print(f"Validation: Purged Walk-Forward (5 folds, 10-day embargo)")
    print(f"Results directory: {RESULTS_DIR}/")

    all_pair_results = []
    all_sensitivities = []

    try:
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
                        all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})
                except Exception as e:
                    print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")

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
    parser = argparse.ArgumentParser(description="Fully Automated Parallel Purged WF CMA-ES Eval Tuner")
    parser.add_argument("--module", default="testbedv3", help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=201, help="First day included in train score (100 to skip warmup)")
    parser.add_argument("--end", type=int, default=800, help="Last day included in train score")
    parser.add_argument("--pairs_file", default="pairs.json", help="JSON file containing list of pairs")
    parser.add_argument("--db_file", default=None, help="Optuna SQLite DB (default auto_tuning_results/optuna_studies.db)")
    parser.add_argument("--out_file", default=None, help="Results JSON path (default auto_tuning_results/tuning_results.json)")
    args = parser.parse_args()
    
    GLOBAL_CONFIG.update(vars(args))
    run_pipeline()