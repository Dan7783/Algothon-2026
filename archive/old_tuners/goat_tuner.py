#!/usr/bin/env python
"""Interactive Eval Tuner - Unified Strategy Tuning & Evaluation (Parallelized).

Merges the interactive graphical tuning methodology with the official Algothon
evaluation engine. Includes exact MLE for OU fitting, multi-level perturbation
stability verification (5%, 10%, 20%), multi-pair JSON loading, a fully
automated Optuna comparison with Random Restarts Ensemble, automatic JSON
parameter exporting, and an out-of-sample test period evaluation.

Supports multiprocessing to utilize multi-core CPUs for batch pair tuning.
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

# ============================================================================
# GLOBAL CONFIGURATION (Safe for Multiprocessing)
# ============================================================================
GLOBAL_CONFIG = {
    "module": "testbed",
    "start": 2,
    "end": 500,
    "pairs_file": None
}

pricesFile = "./prices.txt"
scoreDefaultParam = 1
SKIP_PLOTS = False

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

# --- OPTUNA TRIAL CONFIGURATIONS ---
N_RESTARTS = 3               # Number of independent Optuna paths
TRIALS_KALMAN = 200          # Trials per restart for Step 1
TRIALS_THRESHOLDS = 100      # Trials per restart for Step 3
TRIALS_KELLY_ALARM = 100     # Trials per restart for Step 4
TRIALS_FULL_AUTO = 800       # Trials per restart for Full Auto Tune

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def round_sig(x, sig=3):
    if x == 0 or not np.isfinite(x):
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def run_optuna_ensemble(objective_func, direction, trials_per_restart, n_restarts=N_RESTARTS):
    global_best_score = -1e6 if direction == "maximize" else 1e6
    global_best_params = {}

    for i in range(n_restarts):
        # HYBRID SAMPLER STRATEGY for Regime Changes:
        # Alternates between TPE (great for outlier PnL spikes) and 
        # CMA-ES (excellent for continuous, cliff-heavy CUSUM threshold landscapes).
        if i % 2 == 0:
            sampler = optuna.samplers.TPESampler(seed=i * 100 + np.random.randint(0, 9999))
        else:
            sampler = optuna.samplers.CmaEsSampler(
                seed=i * 100 + np.random.randint(0, 9999),
                n_startup_trials=20  # Needs a small random startup phase to build covariance
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
        print("No pairs JSON provided. Defaulting to HUXZ-ACAC.")
        return [{"y": "HUXZ", "x": "ACAC"}]

    with open(file_path, 'r') as f:
        pairs_data = json.load(f)

    if not isinstance(pairs_data, list):
        print("Error: pairs JSON must contain a list (array) of pair objects.")
        return [{"y": "HUXZ", "x": "ACAC"}]

    print(f"Loaded {len(pairs_data)} pairs from {file_path}")
    return pairs_data


# ============================================================================
# OFFICIAL EVALUATION ENGINE
# ============================================================================
def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return (df.values).T, df.shape[0], df.shape[1]

def chargeFees(dvolumes, commRate):
    return np.sum(dvolumes * commRate)

def score(mu, sigma, param=scoreDefaultParam):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

def loadStrategy(moduleName):
    path = Path(*moduleName.split(".")).with_suffix(".py")
    if not path.exists():
        return importlib.import_module(moduleName)
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
    if hasattr(strat, "reset"):
        strat.reset()
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
        if totDVolume > 0:
            ret = value / totDVolume

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
# Complete map of ALL 10 tuned parameters to their environment variable names.
ENV_MAP = {
    "pair_y":          "TUNER_PAIR_Y",
    "pair_x":          "TUNER_PAIR_X",
    "q_beta":          "TUNER_Q_BETA",
    "h":               "TUNER_H",
    "cusum_k":         "TUNER_CUSUM_K",
    "max_cov_inflate": "TUNER_MAX_COV_INFLATE",
    "kappa":           "TUNER_KAPPA",
    "sigma":           "TUNER_SIGMA",
    "z_entry":         "TUNER_Z_ENTRY",
    "z_exit":          "TUNER_Z_EXIT",
    "kelly_cap":       "TUNER_KELLY_CAP",
    "alarm_damp":      "TUNER_ALARM_DAMP",
}

def inject_parameters(params: dict):
    """Inject tuned params into environment for testbed to read on (re)load."""
    for k, env_key in ENV_MAP.items():
        if k in params and params[k] is not None:
            os.environ[env_key] = str(params[k])
        else:
            # Remove stale value so testbed falls back to its built-in default
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
# FILTER CONFIGURATION & SIMULATION
# ============================================================================
class FiltCfg(NamedTuple):
    q_beta: float
    ewma_lambda: float = 0.97
    cusum_k: float = 0.5
    h: float = 8.0
    max_cov_inflate: float = 25.0

    def clean_sig(self) -> "FiltCfg":
        return FiltCfg(
            q_beta=round_sig(self.q_beta, 3), ewma_lambda=round_sig(self.ewma_lambda, 3),
            cusum_k=round_sig(self.cusum_k, 3), h=round_sig(self.h, 3),
            max_cov_inflate=round_sig(self.max_cov_inflate, 3),
        )

class StepOut(NamedTuple):
    beta: float
    innov: float
    spread: float
    z: float
    cm_up: float
    cm_dn: float
    cv: float

class PairSegFilter:
    def __init__(self, cfg: FiltCfg) -> None:
        self.cfg = cfg
        self.beta = None
        self.p22 = 1.0
        self.innov_var = None
        self.resid_var = None
        self.cm_up = self.cm_dn = 0.0
        self.cv_up = self.cv_dn = 0.0
        self.t = -1
        self._y_hist = []
        self._x_hist = []

    def _ols_beta(self, window):
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < 1e-12:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def step(self, y, x):
        cfg = self.cfg
        self.t += 1
        yf, xf = float(y), float(x)
        self._y_hist.append(yf)
        self._x_hist.append(xf)

        if self.beta is None:
            self.beta = self._ols_beta(20)

        max_cusum = max(self.cm_up, self.cm_dn, self.cv_up, self.cv_dn)
        excess = max(0.0, max_cusum - cfg.h)
        multiplier = 1.0 + min(excess, cfg.max_cov_inflate)
        dynamic_q = cfg.q_beta * multiplier

        b = float(self.beta)
        pp22 = self.p22 + dynamic_q
        innov = yf - b * xf
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < 1e-12: s_pred = 1e-12
        g = (pp22 * xf) / s_pred
        self.beta = b + g * innov
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)

        resid = yf - float(self.beta) * xf

        lam = cfg.ewma_lambda
        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
        std_innov = sqrt(max(self.innov_var, 1e-12))
        z = innov / std_innov

        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        d_v = (z * z - 1.0) / sqrt(2.0)
        self.cv_up = max(0.0, self.cv_up + d_v - cfg.cusum_k)
        self.cv_dn = max(0.0, self.cv_dn - d_v - cfg.cusum_k)

        return StepOut(float(self.beta), float(innov), float(resid), z, float(self.cm_up), float(self.cm_dn), max(self.cv_up, self.cv_dn))

def evaluate_spread_quality(y_series, x_series, cfg):
    filt = PairSegFilter(cfg)
    betas, spreads = [], []
    for t in range(len(y_series)):
        out = filt.step(y_series[t], x_series[t])
        betas.append(out.beta)
        spreads.append(out.spread)

    s = np.array(spreads[30:])
    b = np.array(betas[30:])
    if len(s) < 10: return 1e6

    spread_std = float(np.std(s))
    if spread_std < 0.01: return 1e6

    beta_diff_std = float(np.std(np.diff(b)))
    if beta_diff_std > 0.05: return 1e6

    s_lag = s[:-1]
    s_diff = np.diff(s)
    denom = float(s_lag @ s_lag)
    if denom < 1e-12: return 1e6

    slope = float(s_lag @ s_diff) / denom
    kappa = -slope
    if kappa <= 0 or kappa > 1.0: return 1e6

    half_life = np.log(2) / kappa
    if half_life < 2 or half_life > 60: return 1e6

    return float(-(spread_std * kappa) + (10.0 * beta_diff_std))


# ============================================================================
# TUNING STEPS (Refactored for safe multiprocessing)
# ============================================================================
def tune_kalman(prices, y_idx, x_idx, pair_name):
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    def objective(trial):
        q_beta = round_sig(trial.suggest_float("q_beta", 1e-8, 1e-3, log=True), 3)
        h = round_sig(trial.suggest_float("h", 5.0, 50.0), 3)
        cusum_k = round_sig(trial.suggest_float("cusum_k", 0.2, 2.0), 3)
        max_cov = round_sig(trial.suggest_float("max_cov_inflate", 1.0, 100.0), 3)
        cfg = FiltCfg(q_beta=q_beta, h=h, cusum_k=cusum_k, max_cov_inflate=max_cov).clean_sig()
        return evaluate_spread_quality(y_series, x_series, cfg)

    best = run_optuna_ensemble(objective, "minimize", trials_per_restart=TRIALS_KALMAN)
    return FiltCfg(
        q_beta=best.get("q_beta", 1e-6), h=best.get("h", 15.0),
        cusum_k=best.get("cusum_k", 0.5), max_cov_inflate=best.get("max_cov_inflate", 25.0)
    ).clean_sig()

def tune_ou(prices, y_idx, x_idx, filt_cfg):
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    filt = PairSegFilter(filt_cfg)
    spreads = np.array([filt.step(y_series[t], x_series[t]).spread for t in range(len(y_series))])
    s = np.array(spreads[30:])

    s_lag = s[:-1]; s_t = s[1:]; n = len(s_lag)
    mean_lag = np.mean(s_lag); mean_t = np.mean(s_t)

    b_mle = np.sum((s_lag - mean_lag) * (s_t - mean_t)) / np.sum((s_lag - mean_lag)**2)
    a_mle = mean_t - b_mle * mean_lag
    residuals = s_t - (a_mle + b_mle * s_lag)
    sigma_eps = np.sqrt(np.sum(residuals**2) / (n - 2))

    kappa = -np.log(b_mle)
    sigma = sigma_eps * np.sqrt((2 * kappa) / (1 - np.exp(-2 * kappa)))

    return round_sig(kappa, 3), round_sig(sigma, 3)

def tune_thresholds(prices, y_idx, x_idx, filt_cfg, kappa, sigma):
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    filt = PairSegFilter(filt_cfg)
    spreads = np.array([filt.step(y_series[t], x_series[t]).spread for t in range(len(y_series))])

    def simulate(z_entry, z_exit):
        eq_std = sqrt((sigma**2) / max(2.0 * kappa, 1e-12))
        z_scores = spreads / max(eq_std, 1e-12)
        position = 0; cash = 0.0; pnl = []
        for t in range(1, len(spreads)):
            z = z_scores[t]; ds = spreads[t] - spreads[t - 1]
            if position == 0:
                if z > z_entry: position = -1
                elif z < -z_entry: position = 1
            elif position == 1:
                if z >= -z_exit: position = 0
            elif position == -1:
                if z <= z_exit: position = 0
            cash += position * ds; pnl.append(cash)
        return np.array(pnl)

    def objective(trial):
        z_entry = trial.suggest_float("z_entry", 1.0, 3.5)
        z_exit = 0.001  # FIXED: Hold until spread nearly crosses mean for max reversion capture
        pnl = simulate(z_entry, z_exit)
        return -pnl[-1] if len(pnl) > 0 else 0.0

    best = run_optuna_ensemble(objective, "minimize", trials_per_restart=TRIALS_THRESHOLDS)
    return round_sig(best["z_entry"], 3), 0.001  # Returning fixed z_exit

def tune_kelly_alarm(filt_cfg, kappa, sigma, z_entry, z_exit, pair_y, pair_x):
    def run_eval(kelly_cap, alarm_damp):
        params = {
            "pair_y": pair_y, "pair_x": pair_x,
            "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
            "max_cov_inflate": filt_cfg.max_cov_inflate, "kappa": kappa, "sigma": sigma,
            "z_entry": z_entry, "z_exit": z_exit, "kelly_cap": kelly_cap, "alarm_damp": alarm_damp,
        }
        return run_official_eval(params)

    def objective(trial):
        kc = round_sig(trial.suggest_float("kelly_cap", 0.1, 3.0), 3)
        ad = round_sig(trial.suggest_float("alarm_damp", 0.1, 1.0), 3)
        *_, stratScore = run_eval(kc, ad)
        return stratScore if np.isfinite(stratScore) else -1e6

    best = run_optuna_ensemble(objective, "maximize", trials_per_restart=TRIALS_KELLY_ALARM)
    return round_sig(best["kelly_cap"], 3), round_sig(best["alarm_damp"], 3)

def run_perturbation_stability_check(base_params: dict, pair_y: str, pair_x: str):
    base = {**base_params, "pair_y": pair_y, "pair_x": pair_x}
    *_, base_score = run_official_eval(base)
    sensitivities = []
    levels = [0.05, 0.10, 0.20]

    for p_name, p_val in base_params.items():
        level_drops = []
        for level in levels:
            scores = []
            for sign in [-1, 1]:
                perturbed = {**base, p_name: round_sig(p_val * (1.0 + sign * level), 3)}
                *_, s = run_official_eval(perturbed)
                scores.append(s)
            drop = max(0, (base_score - max(scores)) / abs(base_score)) * 100 if base_score != 0 else 0
            level_drops.append(drop)
        
        drop_5, drop_10, drop_20 = level_drops
        sensitivities.append((p_name, drop_20, drop_5, drop_10))
        
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    return sensitivities

def run_full_auto_tuner_comparison(pair_y: str, pair_x: str):
    def objective(trial):
        params = {
            "pair_y": pair_y, "pair_x": pair_x,
            "q_beta": round_sig(trial.suggest_float("q_beta", 1e-8, 1e-4, log=True), 3),
            "h": round_sig(trial.suggest_float("h", 5.0, 20.0), 3),
            "cusum_k": round_sig(trial.suggest_float("cusum_k", 0.2, 2.0), 3),
            "max_cov_inflate": round_sig(trial.suggest_float("max_cov_inflate", 1.0, 100.0), 3),
            "kappa": round_sig(trial.suggest_float("kappa", 0.05, 0.3), 3),
            "sigma": round_sig(trial.suggest_float("sigma", 0.01, 0.05), 3),
            "z_entry": round_sig(trial.suggest_float("z_entry", 1.2, 3.0), 3),
            "z_exit": 0.001,  # FIXED: reduces 10D space to 9D
            "kelly_cap": round_sig(trial.suggest_float("kelly_cap", 0.1, 3.0), 3),
            "alarm_damp": round_sig(trial.suggest_float("alarm_damp", 0.1, 1.0), 3),
        }
        *_, stratScore = run_official_eval(params)
        return stratScore if np.isfinite(stratScore) else -1e6

    best_params_raw = run_optuna_ensemble(objective, "maximize", trials_per_restart=TRIALS_FULL_AUTO)
    best_params = {k: round_sig(v, 3) for k, v in best_params_raw.items()}
    
    eval_params = {**best_params, "pair_y": pair_y, "pair_x": pair_x, "z_exit": 0.001}
    meanpl, ret, plstd, sharpe, dvol, days, pll, auto_score = run_official_eval(eval_params)
    
    return {
        "meanpl": meanpl, "sharpe": sharpe, "score": auto_score, "ret": ret,
        "params": best_params, "days": days, "pll": pll.tolist()
    }


# ============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================
def init_worker(config):
    """Initializer for multiprocessing pools to set global config and suppress logs."""
    global GLOBAL_CONFIG, SKIP_PLOTS
    GLOBAL_CONFIG = config
    SKIP_PLOTS = True
    import matplotlib
    matplotlib.use("Agg")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def process_pair_worker(pair_dict):
    """Top-level worker function that processes a single pair end-to-end."""
    y_sym = pair_dict["y"]
    x_sym = pair_dict["x"]
    pair_name = f"{y_sym}-{x_sym}"
    
    try:
        prcAll, nt, nInst = loadPrices(pricesFile)
        symbol_map = get_symbol_map(pricesFile)
        
        if y_sym not in symbol_map or x_sym not in symbol_map:
            return None
            
        y_idx = symbol_map[y_sym]
        x_idx = symbol_map[x_sym]

        print(f"[Worker] Starting tuning for {pair_name}...", flush=True)
        
        # --- MANUAL PIPELINE ---
        filt_cfg = tune_kalman(prcAll, y_idx, x_idx, pair_name)
        kappa, sigma = tune_ou(prcAll, y_idx, x_idx, filt_cfg)
        z_entry, z_exit = tune_thresholds(prcAll, y_idx, x_idx, filt_cfg, kappa, sigma)
        kelly_cap, alarm_damp = tune_kelly_alarm(filt_cfg, kappa, sigma, z_entry, z_exit, y_sym, x_sym)

        manual_params = {
            "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
            "max_cov_inflate": filt_cfg.max_cov_inflate, "kappa": kappa, "sigma": sigma,
            "z_entry": z_entry, "z_exit": z_exit, "kelly_cap": kelly_cap, "alarm_damp": alarm_damp,
        }

        sensitivities_manual = run_perturbation_stability_check(manual_params, y_sym, x_sym)
        
        eval_manual = {**manual_params, "pair_y": y_sym, "pair_x": x_sym}
        meanpl, ret, plstd, sharpe, dvol, days, pll, stratScore = run_official_eval(eval_manual)
        manual_metrics = {"meanpl": meanpl, "sharpe": sharpe, "score": stratScore, "ret": ret, "total_return": float(np.sum(pll))}

        # --- FULL AUTO PIPELINE ---
        auto_results = run_full_auto_tuner_comparison(y_sym, x_sym)
        auto_metrics = {"meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], "score": auto_results['score'], "ret": auto_results['ret'], "total_return": float(np.sum(auto_results['pll']))}
        
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym)
        
        # --- OUT-OF-SAMPLE TEST ---
        test_start, test_end = 501, 750
        m_eval = {**manual_params, "pair_y": y_sym, "pair_x": x_sym}
        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym, "z_exit": 0.001}
        
        m_meanpl, m_ret, m_plstd, m_sharpe, m_dvol, m_days, m_pll, m_score = run_official_eval(m_eval, test_start, test_end)
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end)
        
        test_metrics = {
            "manual": {"meanpl": m_meanpl, "sharpe": m_sharpe, "score": m_score, "ret": m_ret, "total_return": float(np.sum(m_pll))},
            "auto": {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret, "total_return": float(np.sum(a_pll))}
        }

        print(f"[Worker] Completed {pair_name} | Manual Score: {stratScore:.2f} | Auto Score: {auto_results['score']:.2f}", flush=True)
        
        return {
            "pair_name": pair_name,
            "y_symbol": y_sym, "x_symbol": x_sym,
            "manual_params": manual_params, "manual_metrics": manual_metrics,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics,
            "test_metrics": test_metrics,
            "sensitivities_manual": sensitivities_manual,
            "sensitivities_auto": sensitivities_auto
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
            if p_name not in param_drops:
                param_drops[p_name] = {"5": [], "10": [], "20": []}
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
        
        # Format tuple sensitivities into JSON-friendly dictionaries
        sens_manual = [
            {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
            for p, d20, d5, d10 in pair_res.get('sensitivities_manual', [])
        ]
        sens_auto = [
            {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
            for p, d20, d5, d10 in pair_res.get('sensitivities_auto', [])
        ]

        save_list.append({
            "pair_name": pair_res['pair_name'],
            "y_symbol": pair_res['y_symbol'],
            "x_symbol": pair_res['x_symbol'],
            "manual_params": pair_res['manual_params'],
            "manual_metrics": pair_res['manual_metrics'],
            "auto_params": pair_res['auto_params'],
            "auto_metrics": pair_res['auto_metrics'],
            "test_metrics": pair_res.get('test_metrics', {}),
            "perturbation_summary": {
                "manual": sens_manual,
                "auto": sens_auto
            }
        })

    filename = "tuning_results.json"
    with open(filename, 'w') as f:
        json.dump(save_list, f, indent=4)
    print(f"Successfully saved all data to {filename}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_interactive_pipeline():
    global SKIP_PLOTS, GLOBAL_CONFIG
    
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found.")
        return

    skip_input = input("Skip all interactive graphs and run fully automated? (y/n): ").strip().lower()
    if skip_input == 'y':
        SKIP_PLOTS = True
        import matplotlib
        matplotlib.use("Agg")
    else:
        SKIP_PLOTS = False

    prcAll, nt, nInst = loadPrices(pricesFile)
    pairs_data = load_pairs_from_json(GLOBAL_CONFIG["pairs_file"])

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {GLOBAL_CONFIG['module']}")
    print(f"Evaluation window: days {GLOBAL_CONFIG['start']}-{GLOBAL_CONFIG['end']}")

    all_pair_results = []
    all_sensitivities = []

    if SKIP_PLOTS:
        # --- PARALLEL EXECUTION ---
        n_cores = multiprocessing.cpu_count()
        max_workers = min(n_cores, len(pairs_data))
        print(f"\n>>> Headless mode activated. Parallelizing across {max_workers} CPU cores...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GLOBAL_CONFIG,)) as executor:
            futures = {executor.submit(process_pair_worker, pair): pair for pair in pairs_data}
            
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_pair_results.append(result)
                        all_sensitivities.append({"pair_name": f"{result['pair_name']} (Manual)", "sensitivities": result["sensitivities_manual"]})
                        all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})
                except Exception as e:
                    print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")
                    
    else:
        # --- SEQUENTIAL EXECUTION (WITH PLOTS) ---
        for i, pair in enumerate(pairs_data):
            result = process_pair_worker(pair)
            if result:
                all_pair_results.append(result)
                all_sensitivities.append({"pair_name": f"{result['pair_name']} (Manual)", "sensitivities": result["sensitivities_manual"]})
                all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})

    # --- FINAL CONSOLIDATED OUTPUTS ---
    # Sort results by original JSON order
    pair_order = {f"{p['y']}-{p['x']}": i for i, p in enumerate(pairs_data)}
    all_pair_results.sort(key=lambda x: pair_order.get(x['pair_name'], 999))

    print_perturbation_summary(all_sensitivities)
    save_params_to_json(all_pair_results)
    print("\nPipeline execution fully complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive eval tuner")
    parser.add_argument("--module", default="testbed", help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=2, help="First day included in score")
    parser.add_argument("--end", type=int, default=500, help="Last day included in score")
    parser.add_argument("--pairs_file", default="pairs.json", help="JSON file containing list of pairs")
    args = parser.parse_args()
    
    GLOBAL_CONFIG.update(vars(args))
    run_interactive_pipeline()