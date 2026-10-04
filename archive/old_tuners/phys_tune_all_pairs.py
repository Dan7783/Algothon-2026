#!/usr/bin/env python
"""Interactive Eval Tuner - Physics-Informed Bayesian Optimization (All Pairs).

Automatically tests ALL possible pairs from prices.txt.
Uses a pure BoTorch GP with custom CUSUM Physics Kernel.
Streams results directly to disk to maintain zero RAM footprint.
Memory-optimized for high-core CPUs.
"""

import argparse
import importlib
import os
import sys
import types
import json
import multiprocessing
import itertools
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
TRIALS_FULL_AUTO = 60
CORR_THRESHOLD = 0 # Skip pairs with abs(corr) < 0.60 to save massive compute

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def round_sig(x, sig=3):
    if x == 0 or not np.isfinite(x):
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

# ============================================================================
# PAIR GENERATION & OFFICIAL EVALUATION ENGINE
# ============================================================================
def get_symbol_map(prices_file):
    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None, nrows=0)
    return {sym: i for i, sym in enumerate(df.columns)}

def generate_all_pairs(prices_file):
    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None, nrows=0)
    symbols = list(df.columns)
    all_pairs = [{"y": y, "x": x} for y, x in itertools.combinations(symbols, 2)]
    print(f"Generated {len(all_pairs)} total pairs from {len(symbols)} instruments.")
    return all_pairs

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
# PHYSICS-INFORMED BOTORCH OPTIMIZER
# ============================================================================
class TheoreticalSharpePrior(Mean):
    def __init__(self, bounds_raw, kappa_proxy=0.1, sigma_proxy=0.02):
        super().__init__()
        self.lb = bounds_raw[0]
        self.ub = bounds_raw[1]
        self.kappa = torch.tensor(kappa_proxy)
        self.sigma = torch.tensor(sigma_proxy)

    def forward(self, x):
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
    has_lengthscale = True

    def __init__(self, bounds_raw):
        super().__init__()
        self.lb = bounds_raw[0]
        self.ub = bounds_raw[1]

    def forward(self, x1, x2, diag=False, **kwargs):
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
    bounds_raw = torch.tensor([
        [-18.42, 10.0, 0.2, 1.0, 0.1, 0.1, 0.1], 
        [-9.20, 40.0, 2.0, 100.0, 2.0, 3.0, 1.0]
    ], dtype=torch.float64)
    
    bounds_01 = torch.tensor([[0.0]*7, [1.0]*7], dtype=torch.float64)
    
    def evaluate(x_norm):
        x_raw = bounds_raw[0] + x_norm * (bounds_raw[1] - bounds_raw[0])
        return botorch_objective_bridge(x_raw, pair_y, pair_x)
        
    train_X = torch.rand(INITIAL_RANDOM_SAMPLES, 7, dtype=torch.float64)
    train_Y = torch.tensor([[evaluate(x)] for x in train_X], dtype=torch.float64)
    
    for i in range(TRIALS_FULL_AUTO):
        model = PhysicsInformedGP(train_X, train_Y, bounds_raw)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        EI = ExpectedImprovement(model, best_f=train_Y.max())
        # FIX 2: Lowered num_restarts and raw_samples to prevent memory explosion
        candidate, _ = optimize_acqf(EI, bounds=bounds_01, q=1, num_restarts=3, raw_samples=10)
        
        new_Y = torch.tensor([[evaluate(candidate[0])]], dtype=torch.float64)
        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_Y])
        
    best_idx = train_Y.argmax()
    best_x_norm = train_X[best_idx]
    auto_score = train_Y[best_idx].item()

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
    import warnings
    warnings.filterwarnings("ignore")
    
    # FIX 1: Prevent PyTorch from spawning 24 threads PER worker.
    # This stops the memory explosion and page file thrashing.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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
        
        # --- CORRELATION PRE-FILTER ---
        y_returns = np.diff(np.log(prcAll[y_idx, :500]))
        x_returns = np.diff(np.log(prcAll[x_idx, :500]))
        corr = np.corrcoef(y_returns, x_returns)[0, 1]
        
        if abs(corr) < CORR_THRESHOLD:
            return None
        # ------------------------------

        print(f"[Worker] Optimizing {pair_name} (corr={corr:.2f})...", flush=True)
        
        auto_results = run_full_auto_tuner_comparison(y_sym, x_sym)
        auto_metrics = {
            "meanpl": auto_results['meanpl'], "sharpe": auto_results['sharpe'], 
            "score": auto_results['score'], "ret": auto_results['ret'], 
            "total_return": float(np.sum(auto_results['pll']))
        }
        
        sensitivities_auto = run_perturbation_stability_check(auto_results["params"], y_sym, x_sym)
        
        test_start, test_end = 501, 750
        a_eval = {**auto_results["params"], "pair_y": y_sym, "pair_x": x_sym, "z_exit": 0.001}
        a_meanpl, a_ret, a_plstd, a_sharpe, a_dvol, a_days, a_pll, a_score = run_official_eval(a_eval, test_start, test_end)
        
        test_metrics = {
            "auto": {"meanpl": a_meanpl, "sharpe": a_sharpe, "score": a_score, "ret": a_ret, "total_return": float(np.sum(a_pll))}
        }

        print(f"[Worker] Completed {pair_name} | Score: {auto_results['score']:.2f} | OOS: {a_score:.2f}", flush=True)
        
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
# OUTPUT FORMATTING AND SAVING (STREAMED TO DISK)
# ============================================================================
def format_result_for_json(result):
    sens_auto = [
        {"param": p, "drop_5": d5, "drop_10": d10, "drop_20": d20}
        for p, d20, d5, d10 in result.get('sensitivities_auto', [])
    ]
    return {
        "pair_name": result['pair_name'],
        "y_symbol": result['y_symbol'],
        "x_symbol": result['x_symbol'],
        "auto_params": result['auto_params'],
        "auto_metrics": result['auto_metrics'],
        "test_metrics": result.get('test_metrics', {}),
        "perturbation_summary": {"auto": sens_auto}
    }

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
    pairs_data = generate_all_pairs(pricesFile)

    print(f"Loaded {nInst} instruments for {nt} days")
    print(f"Strategy module: {GLOBAL_CONFIG['module']}")
    print(f"Evaluation window: days {GLOBAL_CONFIG['start']}-{GLOBAL_CONFIG['end']}")
    print(f"Correlation Threshold: {CORR_THRESHOLD}")
    print("Using Pure Physics-Informed BoTorch GP Optimization.")

    all_sensitivities = []
    filename = "tuning_results_all_pairs.json"
    
    with open(filename, 'w') as f:
        f.write("[\n") 
        first_record = True

        if SKIP_PLOTS:
            n_cores = multiprocessing.cpu_count()
            # FIX 3: Cap workers at 12 to prevent I/O and memory contention
            max_workers = min(24, len(pairs_data))
            print(f"\n>>> Headless mode activated. Parallelizing across {max_workers} CPU cores...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GLOBAL_CONFIG,)) as executor:
                futures = {executor.submit(process_pair_worker, pair): pair for pair in pairs_data}
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    pair = futures[future]
                    try:
                        result = future.result()
                        if result:
                            if not first_record:
                                f.write(",\n")
                            
                            json.dump(format_result_for_json(result), f, indent=4)
                            f.flush() 
                            first_record = False
                            
                            all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})
                            
                    except Exception as e:
                        print(f"Error processing pair {pair['y']}-{pair['x']}: {e}")
                        
                    if (i + 1) % 50 == 0:
                        print(f"--- Progress: {i + 1}/{len(pairs_data)} pairs processed ---")
                        
        else:
            for i, pair in enumerate(pairs_data):
                result = process_pair_worker(pair)
                if result:
                    if not first_record:
                        f.write(",\n")
                    json.dump(format_result_for_json(result), f, indent=4)
                    f.flush()
                    first_record = False
                    all_sensitivities.append({"pair_name": f"{result['pair_name']} (Auto)", "sensitivities": result["sensitivities_auto"]})

        f.write("\n]") 

    print(f"\nAll results streamed and saved to {filename}")
    print_perturbation_summary(all_sensitivities)
    print("\nPipeline execution fully complete.")

def print_perturbation_summary(all_sensitivities):
    print(f"\n{'=' * 80}")
    print(f" PERTURBATION STABILITY SUMMARY (All Tested Pairs)")
    print(f"{'=' * 80}")

    param_drops = {}
    for pair_data in all_sensitivities:
        if not pair_data: continue
        for p_name, drop_20, drop_5, drop_10 in pair_data["sensitivities"]:
            if p_name not in param_drops: param_drops[p_name] = {"5": [], "10": [], "20": []}
            param_drops[p_name]["5"].append(drop_5)
            param_drops[p_name]["10"].append(drop_10)
            param_drops[p_name]["20"].append(drop_20)

    if not param_drops:
        print("No pairs passed the correlation threshold.")
        return

    print(f"{'Parameter':<20} {'5% Avg':<10} {'5% Max':<10} {'10% Avg':<10} {'10% Max':<10} {'20% Avg':<10} {'20% Max':<10} {'Status'}")
    print("-" * 80)
    for p_name, drops in param_drops.items():
        avg5, max5 = np.mean(drops["5"]), np.max(drops["5"])
        avg10, max10 = np.mean(drops["10"]), np.max(drops["10"])
        avg20, max20 = np.mean(drops["20"]), np.max(drops["20"])
        status = "OK" if max20 < 25.0 else "FRAGILE"
        print(f"{p_name:<20} {avg5:<10.2f} {max5:<10.2f} {avg10:<10.2f} {max10:<10.2f} {avg20:<10.2f} {max20:<10.2f} {status}")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive eval tuner")
    parser.add_argument("--module", default="testbed", help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=2, help="First day included in score")
    parser.add_argument("--end", type=int, default=500, help="Last day included in score")
    args = parser.parse_args()
    
    GLOBAL_CONFIG.update(vars(args))
    run_interactive_pipeline()