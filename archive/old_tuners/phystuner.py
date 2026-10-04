#!/usr/bin/env python
"""Interactive Eval Tuner - Physics-Informed Bayesian Optimization (Parallelized).

Optimizes 7 core parameters using pure BoTorch.
`kappa`, `sigma`, and `z_entry` are calculated dynamically inside testbed.py.
The Auto pipeline uses a custom GP with a Theoretical OU Sharpe Prior and CUSUM Physics Kernel.
Inputs are automatically scaled to [0,1] to satisfy BoTorch's GP assumptions.
"""

import argparse
import importlib
import os
import sys
import types
import json
import multiprocessing
from math import sqrt, floor, log10
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd

# ============================================================================
# PYTORCH & BOTORCH IMPORTS (Required)
# ============================================================================
try:
    import torch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from botorch.acquisition import ExpectedImprovement
    from gpytorch.means import Mean
    from gpytorch.kernels import Kernel
except ImportError:
    raise ImportError("This script requires PyTorch and BoTorch. Please run: pip install torch gpytorch botorch")

# ============================================================================
# GLOBAL CONFIGURATION
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

# --- BOTORCH TRIAL CONFIGURATIONS ---
INITIAL_RANDOM_SAMPLES = 10
TRIALS_FULL_AUTO = 60        # BoTorch is sample-efficient, 60 is plenty for 7D with Physics Prior

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
    "entry_sharpe_K":  "TUNER_ENTRY_SHARPE_K",
    "z_exit":          "TUNER_Z_EXIT",
    "kelly_cap":       "TUNER_KELLY_CAP",
    "alarm_damp":      "TUNER_ALARM_DAMP",
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
# PERTURBATION STABILITY CHECK
# ============================================================================
def run_perturbation_stability_check(base_params: dict, pair_y: str, pair_x: str):
    numeric_params = {k: v for k, v in base_params.items() if isinstance(v, (int, float)) and k != "sens_ratio"}
    
    base = {**numeric_params, "pair_y": pair_y, "pair_x": pair_x}
    *_, base_score = run_official_eval(base)
    sensitivities = []
    levels = [0.05, 0.10, 0.20]

    for p_name, p_val in numeric_params.items():
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


# ============================================================================
# PHYSICS-INFORMED BOTORCH OPTIMIZER (AUTO PIPELINE)
# ============================================================================
class TheoreticalSharpePrior(Mean):
    """
    Computes the analytical expected Sharpe ratio of a single OU trade.
    Un-scales the [0,1] GP inputs back to raw parameters before applying physics.
    """
    def __init__(self, bounds_raw, kappa_proxy=0.1, sigma_proxy=0.02):
        super().__init__()
        self.lb = bounds_raw[0]
        self.ub = bounds_raw[1]
        self.kappa = torch.tensor(kappa_proxy)
        self.sigma = torch.tensor(sigma_proxy)

    def forward(self, x):
        # Un-scale x from [0,1] to raw space
        x_raw = self.lb + x * (self.ub - self.lb)
        K = x_raw[..., 4]
        
        z_entry = K * torch.sqrt(2.0 / self.kappa)
        z_exit = torch.tensor(0.001)
        T = (1.0 / self.kappa) * torch.log(z_entry / z_exit)
        profit = (K * self.sigma) / self.kappa
        std_dev = torch.sqrt(self.sigma**2 * T)
        sharpe = profit / std_dev
        return sharpe


class CUSUMPhysicsKernel(Kernel):
    """
    Defines the covariance based on CUSUM physics. Un-scales inputs first.
    """
    has_lengthscale = True

    def __init__(self, bounds_raw):
        super().__init__()
        self.lb = bounds_raw[0]
        self.ub = bounds_raw[1]

    def forward(self, x1, x2, diag=False, **kwargs):
        # Un-scale x1, x2 from [0,1] to raw space
        x1_raw = self.lb + x1 * (self.ub - self.lb)
        x2_raw = self.lb + x2 * (self.ub - self.lb)
        
        R1, k1 = x1_raw[..., 1], x1_raw[..., 2]
        R2, k2 = x2_raw[..., 1], x2_raw[..., 2]
        delta = 1.5 
        
        T1 = (R1 * k1) / torch.clamp(delta - k1, min=0.1)
        T2 = (R2 * k2) / torch.clamp(delta - k2, min=0.1)
        
        if diag:
            dist = (T1 - T2)**2
            cov = torch.exp(-self.lengthscale * dist)
            return cov
        else:
            dist = (T1.unsqueeze(-1) - T2.unsqueeze(-2))**2
            cov = torch.exp(-self.lengthscale * dist)
            return cov


class PhysicsInformedGP(SingleTaskGP):
    """
    Custom BoTorch model that injects our Physics-Informed Prior and Kernel.
    """
    def __init__(self, train_X, train_Y, bounds_raw):
        super().__init__(train_X, train_Y)
        self.mean_module = TheoreticalSharpePrior(bounds_raw)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            CUSUMPhysicsKernel(bounds_raw) + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def botorch_objective_bridge(x_raw, pair_y, pair_x):
    log_q_beta = float(x_raw[0])
    sens_ratio = float(x_raw[1])
    cusum_k = float(x_raw[2])
    max_cov_inflate = float(x_raw[3])
    entry_sharpe_K = float(x_raw[4])
    kelly_cap = float(x_raw[5])
    alarm_damp = float(x_raw[6])
    
    q_beta = round_sig(np.exp(log_q_beta), 3)
    h = round_sig(sens_ratio * cusum_k, 3)
    
    params = {
        "pair_y": pair_y, "pair_x": pair_x,
        "q_beta": q_beta, "h": h, "cusum_k": round_sig(cusum_k, 3),
        "max_cov_inflate": round_sig(max_cov_inflate, 3), 
        "entry_sharpe_K": round_sig(entry_sharpe_K, 3), "z_exit": 0.001,
        "kelly_cap": round_sig(kelly_cap, 3), "alarm_damp": round_sig(alarm_damp, 3)
    }
    *_, stratScore = run_official_eval(params)
    return stratScore if np.isfinite(stratScore) else -1e6

def run_full_auto_tuner_comparison(pair_y: str, pair_x: str):
    # 1. Define Raw Bounds
    # log(1e-8) = -18.42, log(1e-4) = -9.2
    bounds_raw = torch.tensor([
        [-18.42, 10.0, 0.2, 1.0, 0.1, 0.1, 0.1], 
        [-9.20, 40.0, 2.0, 100.0, 2.0, 3.0, 1.0]
    ], dtype=torch.float64)
    
    # 2. Define [0, 1] Dummy Bounds for GP Input
    bounds_01 = torch.tensor([[0.0]*7, [1.0]*7], dtype=torch.float64)
    
    def evaluate(x_norm):
        # Un-scale from [0,1] to raw space before running the backtest
        x_raw = bounds_raw[0] + x_norm * (bounds_raw[1] - bounds_raw[0])
        return botorch_objective_bridge(x_raw, pair_y, pair_x)
        
    # 3. Initial Random Sampling in [0,1] space
    train_X = torch.rand(INITIAL_RANDOM_SAMPLES, 7, dtype=torch.float64)
    train_Y = torch.tensor([[evaluate(x)] for x in train_X], dtype=torch.float64)
    
    # 4. Physics-Informed GP Loop
    for i in range(TRIALS_FULL_AUTO):
        model = PhysicsInformedGP(train_X, train_Y, bounds_raw)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        EI = ExpectedImprovement(model, best_f=train_Y.max())
        candidate, _ = optimize_acqf(EI, bounds=bounds_01, q=1, num_restarts=10, raw_samples=20)
        
        new_Y = torch.tensor([[evaluate(candidate[0])]], dtype=torch.float64)
        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_Y])
        
    best_idx = train_Y.argmax()
    best_x_norm = train_X[best_idx]
    auto_score = train_Y[best_idx].item()

    # Un-scale the final best params for JSON saving
    best_x_raw = bounds_raw[0] + best_x_norm * (bounds_raw[1] - bounds_raw[0])
    best_params_list = best_x_raw.tolist()

    best_params = {
        "q_beta": round_sig(np.exp(best_params_list[0]), 3),
        "sens_ratio": round_sig(best_params_list[1], 3),
        "cusum_k": round_sig(best_params_list[2], 3),
        "max_cov_inflate": round_sig(best_params_list[3], 3),
        "entry_sharpe_K": round_sig(best_params_list[4], 3),
        "kelly_cap": round_sig(best_params_list[5], 3),
        "alarm_damp": round_sig(best_params_list[6], 3)
    }
    best_params["h"] = round_sig(best_params["sens_ratio"] * best_params["cusum_k"], 3)
    
    eval_params = {**best_params, "pair_y": pair_y, "pair_x": pair_x, "z_exit": 0.001}
    meanpl, ret, plstd, sharpe, dvol, days, pll, _ = run_official_eval(eval_params)
    
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
    # Suppress the scaling warnings entirely in workers
    import warnings
    warnings.filterwarnings("ignore")

def process_pair_worker(pair_dict):
    y_sym = pair_dict["y"]
    x_sym = pair_dict["x"]
    pair_name = f"{y_sym}-{x_sym}"
    
    try:
        symbol_map = get_symbol_map(pricesFile)
        if y_sym not in symbol_map or x_sym not in symbol_map: return None

        print(f"[Worker] Starting Physics-Informed optimization for {pair_name}...", flush=True)
        
        # --- FULL AUTO PIPELINE ---
        auto_results = run_full_auto_tuner_comparison(y_sym, x_sym)
        auto_metrics = {
            "meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], 
            "score": auto_results['score'], "ret": auto_results['ret'], 
            "total_return": float(np.sum(auto_results['pll']))
        }
        
        # --- PERTURBATION STABILITY CHECK ---
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym)
        
        # --- OUT-OF-SAMPLE TEST ---
        test_start, test_end = 501, 750
        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym, "z_exit": 0.001}
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end)
        
        test_metrics = {
            "auto": {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret, "total_return": float(np.sum(a_pll))}
        }

        print(f"[Worker] Completed {pair_name} | Auto Score: {auto_results['score']:.2f} | OOS Score: {a_score:.2f}", flush=True)
        
        return {
            "pair_name": pair_name, "y_symbol": y_sym, "x_symbol": x_sym,
            "auto_params": auto_results["params"], "auto_metrics": auto_metrics,
            "test_metrics": test_metrics,
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
    print("Using Pure Physics-Informed BoTorch GP Optimization.")

    all_pair_results = []
    all_sensitivities = []

    if SKIP_PLOTS:
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
                        all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})
                except Exception as e:
                    print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")
    else:
        for i, pair in enumerate(pairs_data):
            result = process_pair_worker(pair)
            if result:
                all_pair_results.append(result)
                all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})

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