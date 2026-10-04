#!/usr/bin/env python
"""Algothon 2026 Strategy Tuner & Evaluator Integration.

Optimizes strategy parameters using Optuna (rounded to 3 sig figs), injects parameters via environment variables,
evaluates performance using the official evaluation engine, and plots the raw PnL graph.
"""

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy import stats as sps

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Tune strategy parameters using eval engine")
parser.add_argument("--module", default="testbed", help="Strategy module name to load")
parser.add_argument("--start", type=int, default=2, help="First day included in score")
parser.add_argument("--end", type=int, default=500, help="Last day included in score")
parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
tunerArgs = parser.parse_args()

pricesFile = "./prices.txt"
scoreDefaultParam = 1

# --- COMMISSION RATES & POSITION LIMITS ---
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def round_sig(x, sig=3):
    """Round a number to specified significant figures."""
    if x == 0:
        return 0.0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def loadPrices(fn):
    """Load prices from csv file (one instrument per column) and transpose."""
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return (df.values).T, df.shape[0], df.shape[1]

def chargeFees(dvolumes, commRate):
    return np.sum(dvolumes * commRate)

def score(mu, sigma, param=scoreDefaultParam):
    """Final score from daily-PnL mean & std."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

def loadStrategy(moduleName):
    """Load local strategy source directly so stale tracked pyc is ignored."""
    path = Path(*moduleName.split(".")).with_suffix(".py")
    if not path.exists():
        return importlib.import_module(moduleName)
    module = types.ModuleType(moduleName)
    module.__file__ = str(path.resolve())
    module.__package__ = moduleName.rpartition(".")[0]
    sys.modules[moduleName] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module

def calcPL(prcHist, testStartDay, testEndDay, getPosition_func, nInst, commRate, dlrPosLimit):
    """Loop over days and calculate PnLs using the strategy's getPosition function."""
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

# --- OPTUNA OBJECTIVE INTEGRATING EVAL SCRIPT ---
def run_optimization():
    prcAll, nt, nInst = loadPrices(pricesFile)
    
    commRateArr = np.full(nInst, defaultCommRate)
    commRateArr[0] = inst0CommRate
    dlrPosLimitArr = np.full(nInst, defaultDlrPosLimit)
    dlrPosLimitArr[0] = inst0DlrPosLimit

    def objective(trial):
        # Suggest parameters and round each to 3 significant figures to avoid overfitting
        q_beta = round_sig(trial.suggest_float("q_beta", 1e-8, 1e-4, log=True))
        h = round_sig(trial.suggest_float("h", 5.0, 20.0))
        z_entry = round_sig(trial.suggest_float("z_entry", 1.2, 3.0))
        z_exit = round_sig(trial.suggest_float("z_exit", 0.1, 0.8))
        kappa = round_sig(trial.suggest_float("kappa", 0.05, 0.3))
        sigma = round_sig(trial.suggest_float("sigma", 0.01, 0.05))

        # Inject rounded parameters via environment variables
        os.environ["TUNER_Q_BETA"] = str(q_beta)
        os.environ["TUNER_H"] = str(h)
        os.environ["TUNER_Z_ENTRY"] = str(z_entry)
        os.environ["TUNER_Z_EXIT"] = str(z_exit)
        os.environ["TUNER_KAPPA"] = str(kappa)
        os.environ["TUNER_SIGMA"] = str(sigma)

        # Reload strategy module to pick up new environment variables
        if tunerArgs.module in sys.modules:
            del sys.modules[tunerArgs.module]
        strategyModule = loadStrategy(tunerArgs.module)
        if hasattr(strategyModule, "reset"):
            strategyModule.reset()
        getPosition = strategyModule.getMyPosition

        # Evaluate using calcPL and score functions
        meanpl, ret, plstd, sharpe, dvol, days, pll = calcPL(
            prcAll, tunerArgs.start, tunerArgs.end, getPosition, nInst, commRateArr, dlrPosLimitArr
        )
        
        stratScore = score(meanpl, plstd, scoreDefaultParam)
        return stratScore if np.isfinite(stratScore) else -1e6

    print(f"Starting Optuna optimization for module '{tunerArgs.module}' over {tunerArgs.trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=tunerArgs.trials, show_progress_bar=True)

    print("\n====================")
    print("Optimization Complete")
    print("====================")
    print(f"Best Score: {study.best_value:.4f}")
    print("Best Parameters (Rounded to 3 Sig Figs):")
    
    best_params_rounded = {k: round_sig(v) for k, v in study.best_params.items()}
    for k, v in best_params_rounded.items():
        print(f"  {k}: {v}")

    # Run final evaluation with best rounded parameters
    for k, v in best_params_rounded.items():
        os.environ[f"TUNER_{k.upper()}"] = str(v)
    
    if tunerArgs.module in sys.modules:
        del sys.modules[tunerArgs.module]
    bestModule = loadStrategy(tunerArgs.module)
    if hasattr(bestModule, "reset"):
        bestModule.reset()
    
    meanpl, ret, plstd, sharpe, dvol, days, pll = calcPL(
        prcAll, tunerArgs.start, tunerArgs.end, bestModule.getMyPosition, nInst, commRateArr, dlrPosLimitArr
    )
    print(f"\nFinal Evaluation -> Mean PnL: {meanpl:.2f}, Sharpe: {sharpe:.2f}, Return: {ret:.5f}")

    # --- PLOT RAW PNL GRAPH ---
    plt.figure(figsize=(11, 6))
    plt.plot(days, pll, label="Daily PnL", color="royalblue", alpha=0.5, linewidth=1)
    plt.plot(days, np.cumsum(pll), label="Cumulative PnL", color="darkorange", linewidth=2)
    plt.title(f"Evaluation PnL Graph - Module: {tunerArgs.module}", fontsize=14, fontweight="bold")
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("PnL ($)", fontsize=12)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if Path(pricesFile).exists():
        run_optimization()
    else:
        print(f"Error: {pricesFile} not found.")