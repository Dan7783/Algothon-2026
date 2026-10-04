#!/usr/bin/env python
"""Interactive Eval Tuner - Unified Strategy Tuning & Evaluation.

Merges the interactive graphical tuning methodology with the official Algothon
evaluation engine. Includes exact MLE for OU fitting, perturbation stability
verification, multi-pair JSON loading, a fully automated Optuna comparison,
JSON parameter exporting, and an out-of-sample test period evaluation.
"""

import argparse
import importlib
import os
import sys
import types
import json
from math import sqrt, floor, log10, log, exp
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

# ============================================================================
# ARGUMENT PARSER
# ============================================================================
parser = argparse.ArgumentParser(
    description="Interactive eval tuner: graphical tuning + official evaluation"
)
parser.add_argument("--module", default="testbed", help="Strategy module name to load")
parser.add_argument("--start", type=int, default=2, help="First day included in score")
parser.add_argument("--end", type=int, default=500, help="Last day included in score")
parser.add_argument("--pairs_file", default=None, help="JSON file containing list of pairs")
tunerArgs = parser.parse_args()

pricesFile = "./prices.txt"
scoreDefaultParam = 1

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000


# ============================================================================
# HELPER: ROUND TO SIGNIFICANT FIGURES
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
        print("No pairs JSON provided. Defaulting to SMAH-NGTE.")
        return [{"y": "SMAH", "x": "NGTE"}]
    
    with open(file_path, 'r') as f:
        pairs_data = json.load(f)
    
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
    if tunerArgs.module in sys.modules:
        del sys.modules[tunerArgs.module]
    strat = loadStrategy(tunerArgs.module)
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
ENV_MAP = {
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
    for k, env_key in ENV_MAP.items():
        if k in params and params[k] is not None:
            os.environ[env_key] = str(params[k])

def run_official_eval(params, start_day=None, end_day=None):
    if start_day is None: start_day = tunerArgs.start
    if end_day is None: end_day = tunerArgs.end
    
    prcAll, nt, nInst = loadPrices(pricesFile)
    commRateArr = np.full(nInst, defaultCommRate); commRateArr[0] = inst0CommRate
    dlrPosLimitArr = np.full(nInst, defaultDlrPosLimit); dlrPosLimitArr[0] = inst0DlrPosLimit
    
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
            q_beta=round_sig(self.q_beta, 3),
            ewma_lambda=round_sig(self.ewma_lambda, 3),
            cusum_k=round_sig(self.cusum_k, 3),
            h=round_sig(self.h, 3),
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
# INTERACTIVE STEP 1: KALMAN FILTER & CUSUM TUNING
# ============================================================================
def interactive_tune_kalman(prices, y_idx, x_idx, pair_name):
    raw_y = prices[y_idx]
    raw_x = prices[x_idx]
    y_series = np.log(np.maximum(raw_y, 1e-12))
    x_series = np.log(np.maximum(raw_x, 1e-12))

    raw_ols_beta = float((raw_x @ raw_y) / max(float(raw_x @ raw_x), 1e-12))
    raw_spreads = raw_y - raw_ols_beta * raw_x

    print(f"\n{'=' * 60}")
    print(f" STEP 1: Interactive Kalman/CUSUM Tuning ({pair_name})")
    print(f"{'=' * 60}")

    locked_params = {}

    def run_optimization(current_locked):
        def objective(trial):
            q_beta = current_locked.get("q_beta") or round_sig(trial.suggest_float("q_beta", 1e-8, 1e-3, log=True), 3)
            h = current_locked.get("h") or round_sig(trial.suggest_float("h", 5.0, 50.0), 3)
            cusum_k = current_locked.get("cusum_k") or round_sig(trial.suggest_float("cusum_k", 0.2, 2.0), 3)
            max_cov = current_locked.get("max_cov_inflate") or round_sig(trial.suggest_float("max_cov_inflate", 1.0, 100.0), 3)
            cfg = FiltCfg(q_beta=q_beta, h=h, cusum_k=cusum_k, max_cov_inflate=max_cov).clean_sig()
            return evaluate_spread_quality(y_series, x_series, cfg)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=150, show_progress_bar=False)
        best = study.best_params
        return FiltCfg(
            q_beta=current_locked.get("q_beta", best.get("q_beta", 1e-6)),
            h=current_locked.get("h", best.get("h", 15.0)),
            cusum_k=current_locked.get("cusum_k", best.get("cusum_k", 0.5)),
            max_cov_inflate=current_locked.get("max_cov_inflate", best.get("max_cov_inflate", 25.0))
        ).clean_sig()

    current_cfg = run_optimization(locked_params)

    while True:
        filt = PairSegFilter(current_cfg)
        betas, spreads, cusum_ups, cusum_dns, trigger_times = [], [], [], [], []
        for t in range(len(y_series)):
            out = filt.step(y_series[t], x_series[t])
            betas.append(out.beta)
            spreads.append(out.spread)
            cusum_ups.append(out.cm_up)
            cusum_dns.append(out.cm_dn)
            if max(out.cm_up, out.cm_dn) > current_cfg.h:
                trigger_times.append(t)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
        ax1.plot(raw_y, label=f"Price Y ({pair_name.split('-')[0]})", color="royalblue")
        ax1.plot(raw_x, label=f"Price X ({pair_name.split('-')[1]})", color="darkorange", alpha=0.8)
        ax1.set_title(f"Step 1: Kalman & CUSUM ({pair_name}) | Triggers: {len(trigger_times)}")
        ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(betas, label="Adaptive Kalman Beta", color="purple")
        ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

        ax3.plot(spreads, label="Log-Spread Residuals", color="teal")
        for trig in trigger_times: ax3.axvline(trig, color="red", alpha=0.2, linewidth=1)
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left", fontsize=8); ax3.grid(alpha=0.3)

        ax4.plot(raw_spreads, label="Raw Price Spread", color="sienna")
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax4.legend(loc="upper left", fontsize=8); ax4.grid(alpha=0.3)

        ax5.plot(cusum_ups, label="CUSUM Up", color="crimson")
        ax5.plot(cusum_dns, label="CUSUM Down", color="darkorange")
        ax5.axhline(current_cfg.h, color="black", linestyle="--", label=f"h ({current_cfg.h:.3g})")
        ax5.legend(loc="upper left", fontsize=8); ax5.grid(alpha=0.3)

        fig.subplots_adjust(top=0.96, bottom=0.04, hspace=0.35)
        plt.show(block=False)

        print(f"\n[Current Config]: q_beta={current_cfg.q_beta}, h={current_cfg.h}, cusum_k={current_cfg.cusum_k}, max_cov={current_cfg.max_cov_inflate}")
        choice = input("\nOptions:\n  [Enter] Accept\n  'edit' 'lock' 'retry'\nChoice: ").strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            return current_cfg
        elif choice == "edit":
            qb = input(f"new q_beta ({current_cfg.q_beta}): ").strip()
            hh = input(f"new h ({current_cfg.h}): ").strip()
            kk = input(f"new cusum_k ({current_cfg.cusum_k}): ").strip()
            ci = input(f"new max_cov_inflate ({current_cfg.max_cov_inflate}): ").strip()
            current_cfg = FiltCfg(
                q_beta=round_sig(float(qb), 3) if qb else current_cfg.q_beta,
                h=round_sig(float(hh), 3) if hh else current_cfg.h,
                cusum_k=round_sig(float(kk), 3) if kk else current_cfg.cusum_k,
                max_cov_inflate=round_sig(float(ci), 3) if ci else current_cfg.max_cov_inflate
            )
        elif choice == "lock":
            p_name = input("param to lock/unlock: ").strip()
            if p_name in locked_params: del locked_params[p_name]
            elif p_name in current_cfg._asdict(): locked_params[p_name] = getattr(current_cfg, p_name)
        elif choice == "retry":
            current_cfg = run_optimization(locked_params)


# ============================================================================
# INTERACTIVE STEP 2: OU PROCESS FITTING VIA EXACT DISCRETE MLE
# ============================================================================
def interactive_tune_ou(prices, y_idx, x_idx, filt_cfg, pair_name):
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    print(f"\n{'=' * 60}")
    print(f" STEP 2: OU Parameter Fitting via Exact MLE ({pair_name})")
    print(f"{'=' * 60}")

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
    
    kappa = round_sig(kappa, 3)
    sigma = round_sig(sigma, 3)
    half_life = round_sig(np.log(2) / kappa, 3)

    while True:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s, label="Spread Series", color="darkblue", alpha=0.7)
        ax.set_title(f"Step 2: OU Physics via MLE ({pair_name}) | kappa: {kappa}, sigma: {sigma}, HL: {half_life}")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        plt.show(block=False)

        print(f"\n[MLE Fitted]: kappa={kappa} | sigma={sigma} | Half-Life={half_life}")
        choice = input("Press [Enter] to ACCEPT, or 'override': ").strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            return kappa, sigma
        elif choice == "override":
            k_in = input(f"custom kappa ({kappa}): ").strip()
            s_in = input(f"custom sigma ({sigma}): ").strip()
            if k_in: kappa = round_sig(float(k_in), 3)
            if s_in: sigma = round_sig(float(s_in), 3)
            half_life = round_sig(np.log(2) / kappa, 3)


# ============================================================================
# INTERACTIVE STEP 3: THRESHOLD OPTIMIZATION DISPLAY PANEL
# ============================================================================
def simulate_ou_strategy(spreads, kappa, sigma, z_entry, z_exit):
    eq_std = sqrt((sigma**2) / max(2.0 * kappa, 1e-12))
    z_scores = spreads / max(eq_std, 1e-12)
    position = 0; pnl_series = []; cash = 0.0
    long_entries, short_entries, exits = [], [], []

    for t in range(1, len(spreads)):
        z = z_scores[t]; ds = spreads[t] - spreads[t - 1]
        if position == 0:
            if z > z_entry: position = -1; short_entries.append((t, z))
            elif z < -z_entry: position = 1; long_entries.append((t, z))
        elif position == 1:
            if z >= -z_exit: position = 0; exits.append((t, z))
        elif position == -1:
            if z <= z_exit: position = 0; exits.append((t, z))
        cash += position * ds; pnl_series.append(cash)

    return np.array(pnl_series), z_scores, long_entries, short_entries, exits

def interactive_tune_thresholds(prices, y_idx, x_idx, filt_cfg, kappa, sigma, pair_name):
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    print(f"\n{'=' * 60}")
    print(f" STEP 3: Optimal Entry/Exit Thresholds ({pair_name})")
    print(f"{'=' * 60}")

    filt = PairSegFilter(filt_cfg)
    spreads = np.array([filt.step(y_series[t], x_series[t]).spread for t in range(len(y_series))])
    s_eval = spreads[30:]

    def objective(trial):
        z_entry = trial.suggest_float("z_entry", 1.0, 3.5)
        z_exit = trial.suggest_float("z_exit", 0.0, 1.0)
        if z_exit >= z_entry: return 0.0
        pnl, _, _, _, _ = simulate_ou_strategy(s_eval, kappa, sigma, z_entry, z_exit)
        return -pnl[-1] if len(pnl) > 0 else 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    z_entry = round_sig(study.best_params["z_entry"], 3)
    z_exit = round_sig(study.best_params["z_exit"], 3)

    while True:
        pnl_curve, z_scores, le, se, ex = simulate_ou_strategy(s_eval, kappa, sigma, z_entry, z_exit)
        final_pnl = pnl_curve[-1] if len(pnl_curve) > 0 else 0.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(z_scores, label="Spread Z-Score", color="dodgerblue", alpha=0.7)
        ax1.axhline(z_entry, color="red", linestyle="--", label=f"Entry (+{z_entry})")
        ax1.axhline(-z_entry, color="red", linestyle="--")
        ax1.axhline(z_exit, color="green", linestyle=":", label=f"Exit (+{z_exit})")
        ax1.axhline(-z_exit, color="green", linestyle=":")
        if le: ax1.scatter(*zip(*le), color="green", marker="^", s=60, label="Long", zorder=5)
        if se: ax1.scatter(*zip(*se), color="red", marker="v", s=60, label="Short", zorder=5)
        if ex: ax1.scatter(*zip(*ex), color="black", marker="x", s=50, label="Exit", zorder=5)
        ax1.set_title(f"Step 3: Thresholds & Trade Markers ({pair_name}) | Final PnL: {final_pnl:.4f}")
        ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(pnl_curve, label="Cumulative Strategy PnL", color="darkorange")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

        fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.3)
        plt.show(block=False)

        print(f"\n[Sim Backtest]: z_entry={z_entry} | z_exit={z_exit} | Final PnL={final_pnl:.4f}")
        choice = input("Press [Enter] to ACCEPT, or 'override': ").strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            return z_entry, z_exit
        elif choice == "override":
            ze = input(f"custom z_entry ({z_entry}): ").strip()
            zx = input(f"custom z_exit ({z_exit}): ").strip()
            if ze: z_entry = round_sig(float(ze), 3)
            if zx: z_exit = round_sig(float(zx), 3)


# ============================================================================
# STEP 4: KELLY CAP & ALARM DAMP INTERACTIVE TUNING
# ============================================================================
def interactive_tune_kelly_alarm(prices, y_idx, x_idx, filt_cfg, kappa, sigma, z_entry, z_exit, pair_name):
    print(f"\n{'=' * 60}")
    print(f" STEP 4: Kelly Cap & Alarm Damp Tuning ({pair_name})")
    print(f"{'=' * 60}")

    def run_eval(kelly_cap, alarm_damp):
        params = {
            "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
            "max_cov_inflate": filt_cfg.max_cov_inflate, "kappa": kappa, "sigma": sigma,
            "z_entry": z_entry, "z_exit": z_exit, "kelly_cap": kelly_cap, "alarm_damp": alarm_damp,
        }
        return run_official_eval(params)

    def objective(trial):
        kc = round_sig(trial.suggest_float("kelly_cap", 0.1, 1.0), 3)
        ad = round_sig(trial.suggest_float("alarm_damp", 0.1, 1.0), 3)
        *_, stratScore = run_eval(kc, ad)
        return stratScore if np.isfinite(stratScore) else -1e6

    print("\n  Running Optuna optimization (20 trials) using official eval engine...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    kelly_cap = round_sig(study.best_params["kelly_cap"], 3)
    alarm_damp = round_sig(study.best_params["alarm_damp"], 3)

    while True:
        meanpl, ret, plstd, sharpe, dvol, days, pll, stratScore = run_eval(kelly_cap, alarm_damp)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(days, pll, width=0.8, color=np.where(pll >= 0, "#2ca02c", "#d62728"), alpha=0.6, label="Daily PnL")
        ax.plot(days, np.cumsum(pll), label="Cumulative PnL", color="darkblue", linewidth=2)
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"Step 4: Official Eval ({pair_name}) | Score: {stratScore:.4f} | Sharpe: {sharpe:.2f}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.show(block=False)

        print(f"\n[Step 4 Results]: kelly_cap={kelly_cap} | alarm_damp={alarm_damp} | Score={stratScore:.4f}")
        choice = input("Press [Enter] to ACCEPT, or 'override': ").strip().lower()
        plt.close()

        if choice == "" or choice == "accept":
            return kelly_cap, alarm_damp
        elif choice == "override":
            kc = input(f"custom kelly_cap ({kelly_cap}): ").strip()
            ad = input(f"custom alarm_damp ({alarm_damp}): ").strip()
            if kc: kelly_cap = round_sig(float(kc), 3)
            if ad: alarm_damp = round_sig(float(ad), 3)


# ============================================================================
# STEP 5: PERTURBATION STABILITY VERIFICATION (TORNADO ANALYSIS)
# ============================================================================
def run_perturbation_stability_check(base_params: dict, pair_name: str):
    print(f"\n{'=' * 60}")
    print(f" STEP 5: Perturbation Stability Verification ({pair_name})")
    print(f"{'=' * 60}")
    
    *_, base_score = run_official_eval(base_params)
    print(f"  Base Official Score: {base_score:.4f}")
    print("-" * 60)
    print(f"{'Parameter':<20} {'Value':<12} {'-5% Score':<12} {'+5% Score':<12} {'Max Drop':<10} {'Status'}")
    print("-" * 60)
    
    sensitivities = []
    
    for p_name, p_val in base_params.items():
        scores = []
        for sign in [-1, 1]:
            perturbed = base_params.copy()
            perturbed[p_name] = round_sig(p_val * (1.0 + sign * 0.05), 3)
            *_, s = run_official_eval(perturbed)
            scores.append(s)
            
        drop = max(0, (base_score - max(scores)) / abs(base_score)) * 100 if base_score != 0 else 0
        status = "OK" if drop < 15.0 else "FRAGILE"
        print(f"{p_name:<20} {p_val:<12.5g} {scores[0]:<12.4f} {scores[1]:<12.4f} {drop:<10.2f} {status}")
        sensitivities.append((p_name, drop, scores[0], scores[1]))
        
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sensitivities))
    drops_neg = [base_score - s[2] for s in sensitivities]
    drops_pos = [s[3] - base_score for s in sensitivities]
    
    ax.barh(y_pos, drops_neg, color="crimson", alpha=0.7, label="Score Drop (-5%)")
    ax.barh(y_pos, drops_pos, color="royalblue", alpha=0.7, label="Score Change (+5%)")
    ax.set_yticks(y_pos); ax.set_yticklabels([s[0] for s in sensitivities])
    ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Change in Official Score")
    ax.set_title(f"Perturbation Tornado ({pair_name}) | Base: {base_score:.4f}", fontweight="bold")
    ax.legend(); ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.show(block=False)
    
    input("\nPress [Enter] to acknowledge stability report and proceed: ")
    plt.close()
    
    return sensitivities


# ============================================================================
# FULL AUTO TUNER COMPARISON
# ============================================================================
def run_full_auto_tuner_comparison(pair_name: str):
    print(f"\n{'=' * 60}")
    print(f" FULL AUTO TUNER COMPARISON ({pair_name})")
    print(f"{'=' * 60}")

    def objective(trial):
        params = {
            "q_beta": round_sig(trial.suggest_float("q_beta", 1e-8, 1e-4, log=True), 3),
            "h": round_sig(trial.suggest_float("h", 5.0, 20.0), 3),
            "cusum_k": round_sig(trial.suggest_float("cusum_k", 0.2, 2.0), 3),
            "max_cov_inflate": round_sig(trial.suggest_float("max_cov_inflate", 1.0, 100.0), 3),
            "kappa": round_sig(trial.suggest_float("kappa", 0.05, 0.3), 3),
            "sigma": round_sig(trial.suggest_float("sigma", 0.01, 0.05), 3),
            "z_entry": round_sig(trial.suggest_float("z_entry", 1.2, 3.0), 3),
            "z_exit": round_sig(trial.suggest_float("z_exit", 0.1, 0.8), 3),
            "kelly_cap": round_sig(trial.suggest_float("kelly_cap", 0.1, 1.0), 3),
            "alarm_damp": round_sig(trial.suggest_float("alarm_damp", 0.1, 1.0), 3),
        }
        *_, stratScore = run_official_eval(params)
        return stratScore if np.isfinite(stratScore) else -1e6

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    best_params = {k: round_sig(v, 3) for k, v in study.best_params.items()}
    meanpl, ret, plstd, sharpe, dvol, days, pll, auto_score = run_official_eval(best_params)

    print(f"\n[Full Auto Tuner Best Parameters]:")
    for k, v in best_params.items():
        print(f"  {k:<18}: {v}")

    return {
        "meanpl": meanpl, "sharpe": sharpe, "score": auto_score, "ret": ret,
        "params": best_params, "days": days, "pll": pll
    }


# ============================================================================
# OUT-OF-SAMPLE TEST PERIOD EVALUATION
# ============================================================================
def run_test_period_evaluation(all_pair_results):
    print(f"\n{'=' * 60}")
    print(f" OUT-OF-SAMPLE TEST PERIOD EVALUATION (Days 501-750)")
    print(f"{'=' * 60}")
    
    test_start, test_end = 501, 750
    
    for pair in all_pair_results:
        pair_name = pair["pair_name"]
        print(f"\nEvaluating {pair_name} on unseen data ({test_start}-{test_end})...")
        
        # Manual parameters on test set
        m_meanpl, m_ret, m_plstd, m_sharpe, m_dvol, m_days, m_pll, m_score = run_official_eval(pair["manual_params"], test_start, test_end)
        
        # Auto parameters on test set
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(pair["auto_params"], test_start, test_end)
        
        print(f"\n  --- Test Results for {pair_name} ---")
        print(f"  {'Metric':<20} {'Manual (Test)':<18} {'Auto (Test)':<18}")
        print(f"  {'-'*56}")
        print(f"  {'Mean Daily PnL':<20} ${m_meanpl:<17.2f} ${a_meanpl:<17.2f}")
        print(f"  {'Annualized Sharpe':<20} {m_sharpe:<18.2f} {a_sharpe:<18.2f}")
        print(f"  {'Official Score':<20} {m_score:<18.4f} {a_score:<18.4f}")
        print(f"  {'Total Return':<20} ${np.sum(m_pll):<17.2f} ${np.sum(a_pll):<17.2f}")
        
        # Plot test period PnL
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(m_days, np.cumsum(m_pll), label=f"Manual Test (Score: {m_score:.2f})", color="royalblue", linewidth=2)
        ax.plot(a_days, np.cumsum(a_pll), label=f"Auto Test (Score: {a_score:.2f})", color="crimson", linewidth=2, linestyle="--")
        ax.set_title(f"Out-of-Sample Test Period PnL ({test_start}-{test_end}) - {pair_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Day"); ax.set_ylabel("Cumulative PnL ($)")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11); ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show(block=False)
        plt.close()


# ============================================================================
# PERTURBATION SUMMARY & JSON SAVE
# ============================================================================
def print_perturbation_summary(all_sensitivities):
    print(f"\n{'=' * 60}")
    print(f" PERTURBATION STABILITY SUMMARY (All Pairs)")
    print(f"{'=' * 60}")
    
    param_drops = {}
    for pair_data in all_sensitivities:
        for p_name, drop, _, _ in pair_data["sensitivities"]:
            if p_name not in param_drops:
                param_drops[p_name] = []
            param_drops[p_name].append(drop)
    
    print(f"{'Parameter':<20} {'Avg Drop (%)':<15} {'Max Drop (%)':<15} {'Status'}")
    print("-" * 60)
    for p_name, drops in param_drops.items():
        avg_drop = np.mean(drops)
        max_drop = np.max(drops)
        status = "OK" if max_drop < 15.0 else "FRAGILE"
        print(f"{p_name:<20} {avg_drop:<15.2f} {max_drop:<15.2f} {status}")
    print("-" * 60)

def save_params_to_json(all_pair_results):
    print("\nSave parameters to JSON?")
    print("  [1] Manual parameters (Interactive)")
    print("  [2] Auto Optuna parameters")
    print("  [3] Don't save")
    choice = input("Choice: ").strip()
    
    if choice not in ['1', '2']:
        return
        
    save_list = []
    for pair_res in all_pair_results:
        if choice == '1':
            params = pair_res['manual_params']
            metrics = pair_res['manual_metrics']
        else:
            params = pair_res['auto_params']
            metrics = pair_res['auto_metrics']
            
        save_list.append({
            "pair_name": pair_res['pair_name'],
            "y_symbol": pair_res['y_symbol'],
            "x_symbol": pair_res['x_symbol'],
            "params": params,
            "metrics": metrics
        })
        
    filename = input("Enter filename (e.g., tuned_params.json): ").strip()
    if not filename: filename = "tuned_params.json"
    with open(filename, 'w') as f:
        json.dump(save_list, f, indent=4)
    print(f"Successfully saved parameters to {filename}")


# ============================================================================
# MAIN INTERACTIVE PIPELINE
# ============================================================================
def run_interactive_pipeline():
    if not Path(pricesFile).exists():
        print(f"Error: {pricesFile} not found.")
        return

    prcAll, nt, nInst = loadPrices(pricesFile)
    symbol_map = get_symbol_map(pricesFile)
    
    pairs_data = load_pairs_from_json(tunerArgs.pairs_file)
    
    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {tunerArgs.module}")
    print(f"Evaluation window: days {tunerArgs.start}-{tunerArgs.end}")

    all_pair_results = []
    all_sensitivities = []

    for pair in pairs_data:
        y_sym = pair["y"]; x_sym = pair["x"]
        if y_sym not in symbol_map or x_sym not in symbol_map:
            print(f"Skipping pair {y_sym}-{x_sym}: symbols not found in prices file.")
            continue
            
        y_idx = symbol_map[y_sym]; x_idx = symbol_map[x_sym]
        pair_name = f"{y_sym}-{x_sym}"

        # --- MANUAL INTERACTIVE PIPELINE ---
        filt_cfg = interactive_tune_kalman(prcAll, y_idx, x_idx, pair_name)
        kappa, sigma = interactive_tune_ou(prcAll, y_idx, x_idx, filt_cfg, pair_name)
        z_entry, z_exit = interactive_tune_thresholds(prcAll, y_idx, x_idx, filt_cfg, kappa, sigma, pair_name)
        kelly_cap, alarm_damp = interactive_tune_kelly_alarm(prcAll, y_idx, x_idx, filt_cfg, kappa, sigma, z_entry, z_exit, pair_name)

        manual_params = {
            "q_beta": filt_cfg.q_beta, "h": filt_cfg.h, "cusum_k": filt_cfg.cusum_k,
            "max_cov_inflate": filt_cfg.max_cov_inflate, "kappa": kappa, "sigma": sigma,
            "z_entry": z_entry, "z_exit": z_exit, "kelly_cap": kelly_cap, "alarm_damp": alarm_damp,
        }

        # --- PERTURBATION STABILITY VERIFICATION ---
        sensitivities = run_perturbation_stability_check(manual_params, pair_name)
        all_sensitivities.append({"pair_name": pair_name, "sensitivities": sensitivities})

        # --- FINAL MANUAL EVALUATION ---
        print(f"\n{'=' * 60}")
        print(f" FINAL TRAINING EVALUATION: Manual Process ({pair_name})")
        print(f"{'=' * 60}")
        meanpl, ret, plstd, sharpe, dvol, days, pll, stratScore = run_official_eval(manual_params)
        print(f"  Mean Daily PnL     : ${meanpl:.2f}")
        print(f"  Annualized Sharpe  : {sharpe:.2f}")
        print(f"  Total Return       : ${np.sum(pll):.2f}")
        print(f"  Official Score     : {stratScore:.4f}")

        manual_metrics = {"meanpl": meanpl, "sharpe": sharpe, "score": stratScore, "ret": ret, "total_return": float(np.sum(pll))}

        # --- FULL AUTO OPTUNA COMPARISON ---
        auto_results = run_full_auto_tuner_comparison(pair_name)
        auto_metrics = {"meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], "score": auto_results['score'], "ret": auto_results['ret'], "total_return": float(np.sum(auto_results['pll']))}

        print(f"\n{'=' * 60}")
        print(f" TRAINING COMPARISON ({pair_name}): MANUAL vs. AUTO")
        print(f"{'=' * 60}")
        print(f"{'Metric':<20} {'Manual Process':<20} {'Full Auto Optuna':<20}")
        print("-" * 60)
        print(f"{'Mean Daily PnL':<20} ${manual_metrics['meanpl']:<19.2f} ${auto_metrics['meanpl']:<19.2f}")
        print(f"{'Annualized Sharpe':<20} {manual_metrics['sharpe']:<20.2f} {auto_metrics['sharpe']:<20.2f}")
        print(f"{'Official Score':<20} {manual_metrics['score']:<20.4f} {auto_metrics['score']:<20.4f}")
        print("-" * 60)
        
        # Plot Training comparison graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(days, np.cumsum(pll), label=f"Manual Train (Score: {manual_metrics['score']:.2f})", color="royalblue", linewidth=2)
        ax.plot(auto_results["days"], np.cumsum(auto_results["pll"]), label=f"Auto Train (Score: {auto_metrics['score']:.2f})", color="crimson", linewidth=2, linestyle="--")
        ax.set_title(f"Training Period PnL Comparison ({pair_name})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Day"); ax.set_ylabel("Cumulative PnL ($)")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11); ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout(); plt.show(block=False)
        plt.close()

        all_pair_results.append({
            "pair_name": pair_name,
            "y_symbol": y_sym, "x_symbol": x_sym,
            "manual_params": manual_params, "manual_metrics": manual_metrics,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics
        })

    # --- FINAL CONSOLIDATED OUTPUTS ---
    print_perturbation_summary(all_sensitivities)
    save_params_to_json(all_pair_results)
    
    # --- OUT-OF-SAMPLE TEST PERIOD EVALUATION ---
    run_test_period_evaluation(all_pair_results)

if __name__ == "__main__":
    run_interactive_pipeline()