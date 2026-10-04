from __future__ import annotations

import argparse
from math import sqrt, floor, log10
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

# --- HELPER: ROUND TO SIG FIGS TO LIMIT OVERFITTING ---
def round_sig(x: float, sig: int = 3) -> float:
    """Rounds a float to a specified number of significant figures."""
    if x == 0.0 or not np.isfinite(x):
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

# --- CONFIGURATION STRUCTURES ---
class FiltCfg(NamedTuple):
    q_beta: float
    ewma_lambda: float = 0.97
    cusum_k: float = 0.5
    h: float = 8.0
    max_cov_inflate: float = 25.0

    def clean_sig(self) -> FiltCfg:
        """Enforces 3-4 sig figs on all parameters to mitigate overfitting."""
        return FiltCfg(
            q_beta=round_sig(self.q_beta, 3),
            ewma_lambda=round_sig(self.ewma_lambda, 3),
            cusum_k=round_sig(self.cusum_k, 3),
            h=round_sig(self.h, 3),
            max_cov_inflate=round_sig(self.max_cov_inflate, 3)
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
        self.beta: float | None = None
        self.p22: float = 1.0
        self.innov_var: float | None = None
        self.resid_var: float | None = None
        self.cm_up = self.cm_dn = 0.0
        self.cv_up = self.cv_dn = 0.0
        self.t = -1
        self._y_hist: list[float] = []
        self._x_hist: list[float] = []

    def _ols_beta(self, window: int) -> float:
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < 1e-12:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def step(self, y: float, x: float) -> StepOut:
        cfg = self.cfg
        self.t += 1
        yf, xf = float(y), float(x)
        self._y_hist.append(yf)
        self._x_hist.append(xf)

        if self.beta is None:
            self.beta = self._ols_beta(20)

        innov = yf - float(self.beta) * xf
        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            self.innov_var = cfg.ewma_lambda * self.innov_var + (1.0 - cfg.ewma_lambda) * innov * innov
        
        std_innov = sqrt(max(self.innov_var, 1e-12))
        z_innov = innov / std_innov

        self.cm_up = max(0.0, self.cm_up + z_innov - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z_innov - cfg.cusum_k)

        max_cusum = max(self.cm_up, self.cm_dn)
        excess = max(0.0, max_cusum - cfg.h)
        multiplier = 1.0 + min(excess, cfg.max_cov_inflate)
        dynamic_q = cfg.q_beta * multiplier

        b = float(self.beta)
        pp22 = self.p22 + dynamic_q
        s_pred = pp22 * xf * xf + 1.0
        g = (pp22 * xf) / max(s_pred, 1e-12)
        self.beta = b + g * innov
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)
        resid = yf - float(self.beta) * xf

        return StepOut(
            beta=float(self.beta),
            innov=float(innov),
            spread=float(resid),
            z=z_innov,
            cm_up=float(self.cm_up),
            cm_dn=float(self.cm_dn),
            cv=max(self.cv_up, self.cv_dn)
        )

# --- ROBUST SPREAD OBJECTIVE ---
def evaluate_spread_quality(y_series, x_series, cfg: FiltCfg) -> float:
    filt = PairSegFilter(cfg)
    betas, spreads = [], []
    for t in range(len(y_series)):
        out = filt.step(y_series[t], x_series[t])
        betas.append(out.beta)
        spreads.append(out.spread)
        
    s = np.array(spreads[30:])
    b = np.array(betas[30:])
    
    if len(s) < 10:
        return 1e6
        
    spread_std = float(np.std(s))
    if spread_std < 0.01:
        return 1e6

    beta_diff_std = float(np.std(np.diff(b)))
    if beta_diff_std > 0.05:
        return 1e6

    s_lag = s[:-1]
    s_diff = np.diff(s)
    denom = float(s_lag @ s_lag)
    if denom < 1e-12:
        return 1e6
        
    slope = float(s_lag @ s_diff) / denom
    kappa = -slope
    
    if kappa <= 0 or kappa > 1.0:
        return 1e6
        
    half_life = np.log(2) / kappa
    if half_life < 2 or half_life > 60:
        return 1e6

    score = - (spread_std * kappa) + (10.0 * beta_diff_std)
    return float(score)

# --- PERTURBATION VERIFICATION CHECK ---
def verify_parameter_perturbations(y_series, x_series, cfg: FiltCfg, delta: float = 0.05) -> dict:
    print("\n--- Running Perturbation Stability Analysis (±5%) ---")
    base_score = evaluate_spread_quality(y_series, x_series, cfg)
    params_dict = cfg._asdict()
    stability_report = {}

    for param_name, val in params_dict.items():
        if param_name == "ewma_lambda":
            continue
        max_drop = 0.0
        for sign in [-1, 1]:
            perturbed_dict = params_dict.copy()
            perturbed_dict[param_name] = val * (1.0 + sign * delta)
            pert_cfg = FiltCfg(**perturbed_dict).clean_sig()
            
            pert_score = evaluate_spread_quality(y_series, x_series, pert_cfg)
            change = abs(pert_score - base_score) / (abs(base_score) + 1e-12)
            max_drop = max(max_drop, change)

        status = "STABLE" if max_drop < 0.25 else "SENSITIVE (Potential Overfit)"
        stability_report[param_name] = max_drop
        print(f"  * {param_name:<18} | Max Metric Shift: {max_drop:>6.2%} | Status: {status}")
    
    print("-" * 55)
    return stability_report

# --- INTERACTIVE STEP 1: KALMAN & CUSUM TUNING WITH FIXED BETA & RAW SPREAD ---
def interactive_tune_kalman(prices: np.ndarray, y_idx: int, x_idx: int, pair_name: str) -> FiltCfg:
    raw_y = prices[y_idx]
    raw_x = prices[x_idx]
    y_series = np.log(np.maximum(raw_y, 1e-12))
    x_series = np.log(np.maximum(raw_x, 1e-12))

    raw_ols_beta = float((raw_x @ raw_y) / max(float(raw_x @ raw_x), 1e-12))
    raw_spreads = raw_y - raw_ols_beta * raw_x
    raw_spread_std = float(np.std(raw_spreads[30:]))

    print(f"\n==========================================")
    print(f" STEP 1: Interactive Kalman/CUSUM Tuning ({pair_name})")
    print(f"==========================================")
    
    ols_beta = float((x_series @ y_series) / max(float(x_series @ x_series), 1e-12))
    ols_spreads = y_series - ols_beta * x_series
    ols_std = float(np.std(ols_spreads[30:]))
    ols_s_lag = ols_spreads[30:-1]
    ols_s_diff = np.diff(ols_spreads[30:])
    ols_kappa = max(-float((ols_s_lag @ ols_s_diff) / max(float(ols_s_lag @ ols_s_lag), 1e-12)), 1e-4)
    ols_halflife = np.log(2) / ols_kappa

    locked_params: dict[str, float] = {}
    
    def run_optimization(current_locked: dict[str, float]) -> FiltCfg:
        def objective(trial):
            q_beta = current_locked.get("q_beta")
            if q_beta is None:
                q_beta = round_sig(trial.suggest_float("q_beta", 1e-8, 1e-3, log=True), 3)
                
            h = current_locked.get("h")
            if h is None:
                h = round_sig(trial.suggest_float("h", 5.0, 50.0), 3)
                
            cusum_k = current_locked.get("cusum_k")
            if cusum_k is None:
                cusum_k = round_sig(trial.suggest_float("cusum_k", 0.2, 2.0), 3)
                
            max_cov = current_locked.get("max_cov_inflate")
            if max_cov is None:
                max_cov = round_sig(trial.suggest_float("max_cov_inflate", 1.0, 100.0), 3)
            
            cfg = FiltCfg(q_beta=q_beta, h=h, cusum_k=cusum_k, max_cov_inflate=max_cov).clean_sig()
            return evaluate_spread_quality(y_series, x_series, cfg)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        best = study.best_params
        return FiltCfg(
            q_beta=current_locked.get("q_beta", best.get("q_beta", 1e-6)),
            h=current_locked.get("h", best.get("h", 15.0)),
            cusum_k=current_locked.get("cusum_k", best.get("cusum_k", 0.5)),
            max_cov_inflate=current_locked.get("max_cov_inflate", best.get("max_cov_inflate", 25.0))
        ).clean_sig()

    current_cfg = run_optimization(locked_params)
    show_fixed_comparison = False

    while True:
        filt = PairSegFilter(current_cfg)
        betas, spreads = [], []
        cusum_ups, cusum_dns = [], []
        trigger_times = []

        for t in range(len(y_series)):
            out = filt.step(y_series[t], x_series[t])
            betas.append(out.beta)
            spreads.append(out.spread)
            cusum_ups.append(out.cm_up)
            cusum_dns.append(out.cm_dn)
            if max(out.cm_up, out.cm_dn) > current_cfg.h:
                trigger_times.append(t)

        verify_parameter_perturbations(y_series, x_series, current_cfg)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
        
        ax1.plot(raw_y, label=f"Price Y (idx {y_idx})", color="royalblue")
        ax1.plot(raw_x, label=f"Price X (idx {x_idx})", color="darkorange", alpha=0.8)
        ax1.set_title(f"Step 1: Pair Prices, Kalman & CUSUM Tuning ({pair_name}) | Triggers: {len(trigger_times)}")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(betas, label="Adaptive Kalman Hedge Ratio (beta)", color="purple")
        if show_fixed_comparison:
            ax2.axhline(ols_beta, color="gray", linestyle="--", label=f"Static OLS Beta ({ols_beta:.3f})")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)

        ax3.plot(spreads, label="Adaptive Log-Spread Residuals", color="teal")
        if show_fixed_comparison:
            ax3.plot(ols_spreads, label="Static OLS Log-Spread", color="gray", linestyle=":", alpha=0.7)
        for trig in trigger_times:
            ax3.axvline(trig, color="red", alpha=0.2, linewidth=1)
        if trigger_times:
            ax3.plot([], [], color="red", alpha=0.5, label="CUSUM Trigger Events")
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(alpha=0.3)

        ax4.plot(raw_spreads, label=f"Raw Price Spread (OLS beta: {raw_ols_beta:.3f})", color="sienna")
        for trig in trigger_times:
            ax4.axvline(trig, color="red", alpha=0.2, linewidth=1)
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax4.legend(loc="upper left", fontsize=8)
        ax4.grid(alpha=0.3)

        ax5.plot(cusum_ups, label="CUSUM Up (+)", color="crimson")
        ax5.plot(cusum_dns, label="CUSUM Down (-)", color="darkorange")
        ax5.axhline(current_cfg.h, color="black", linestyle="--", label=f"Threshold h ({current_cfg.h:.3g})")
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(alpha=0.3)
        
        fig.subplots_adjust(top=0.96, bottom=0.04, hspace=0.35)
        plt.show(block=False)

        print(f"\n[Current Config - Rounded to 3-4 Sig Figs]:")
        print(f"  q_beta          : {current_cfg.q_beta} {'[LOCKED]' if 'q_beta' in locked_params else ''}")
        print(f"  h threshold     : {current_cfg.h} {'[LOCKED]' if 'h' in locked_params else ''}")
        print(f"  cusum_k (allow) : {current_cfg.cusum_k} {'[LOCKED]' if 'cusum_k' in locked_params else ''}")
        print(f"  max_cov_inflate : {current_cfg.max_cov_inflate} {'[LOCKED]' if 'max_cov_inflate' in locked_params else ''}")
        print(f"  Total Triggers  : {len(trigger_times)} structural breaks detected")
        print(f"  Raw Price Spread Std: {raw_spread_std:.4f}")
        if show_fixed_comparison:
            print(f"\n[Comparison Mode Active]:")
            print(f"  Static OLS Beta     : {ols_beta:.4f} | Log-Spread Std: {ols_std:.4f} | Half-Life: {ols_halflife:.1f} bars")
            eval_s = np.array(spreads[30:])
            eval_std = float(np.std(eval_s))
            print(f"  Adaptive Kalman Beta: Variable | Log-Spread Std: {eval_std:.4f}")
        
        choice = input(
            "\nOptions:\n"
            "  [Enter]         -> Accept configuration and proceed\n"
            "  'edit'          -> Manually change a parameter value\n"
            "  'lock/unlock'   -> Toggle locking state of a parameter\n"
            "  'retry'         -> Re-run optimization with current locks\n"
            "  'compare fixed' -> Toggle static OLS baseline comparison\n"
            "Choice: "
        ).strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            print("-> Kalman & CUSUM parameters accepted and verified.")
            return current_cfg
        elif choice == "edit":
            qb = input(f"Enter new q_beta (current: {current_cfg.q_beta}): ").strip()
            hh = input(f"Enter new h (current: {current_cfg.h}): ").strip()
            kk = input(f"Enter new cusum_k (current: {current_cfg.cusum_k}): ").strip()
            ci = input(f"Enter new max_cov_inflate (current: {current_cfg.max_cov_inflate}): ").strip()
            
            current_cfg = FiltCfg(
                q_beta=round_sig(float(qb), 3) if qb else current_cfg.q_beta,
                h=round_sig(float(hh), 3) if hh else current_cfg.h,
                cusum_k=round_sig(float(kk), 3) if kk else current_cfg.cusum_k,
                max_cov_inflate=round_sig(float(ci), 3) if ci else current_cfg.max_cov_inflate
            )
            if qb: locked_params["q_beta"] = current_cfg.q_beta
            if hh: locked_params["h"] = current_cfg.h
            if kk: locked_params["cusum_k"] = current_cfg.cusum_k
            if ci: locked_params["max_cov_inflate"] = current_cfg.max_cov_inflate

        elif choice == "lock/unlock":
            p_name = input("Enter parameter name to toggle lock ('q_beta', 'h', 'cusum_k', 'max_cov_inflate'): ").strip()
            if p_name in current_cfg._asdict():
                if p_name in locked_params:
                    del locked_params[p_name]
                    print(f"-> Unlocked '{p_name}'. It will now be optimized.")
                else:
                    val = getattr(current_cfg, p_name)
                    locked_params[p_name] = val
                    print(f"-> Locked '{p_name}' at value {val}.")
            else:
                print("-> Invalid parameter name.")

        elif choice == "retry":
            print("-> Re-optimizing free parameters while respecting locks...")
            current_cfg = run_optimization(locked_params)

        elif choice == "compare fixed":
            show_fixed_comparison = not show_fixed_comparison
            print(f"-> Static OLS comparison mode toggled to: {show_fixed_comparison}")

# --- INTERACTIVE STEP 2: OU FITTING & HALF-LIFE ---
def interactive_tune_ou(prices: np.ndarray, y_idx: int, x_idx: int, filt_cfg: FiltCfg, pair_name: str) -> tuple[float, float]:
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    print(f"\n==========================================")
    print(f" STEP 2: OU Parameter Fitting ({pair_name})")
    print(f"==========================================")

    filt = PairSegFilter(filt_cfg)
    spreads = [filt.step(y_series[t], x_series[t]).spread for t in range(len(y_series))]
    s = np.array(spreads[30:])
    
    s_lag = s[:-1]
    s_diff = np.diff(s)
    slope = float(s_lag @ s_diff) / float(s_lag @ s_lag)
    kappa = max(-slope, 1e-4)
    residuals = s_diff - (-kappa * s_lag)
    sigma = max(float(np.std(residuals)), 1e-6)
    
    kappa = round_sig(kappa, 3)
    sigma = round_sig(sigma, 3)
    half_life = round_sig(np.log(2) / kappa, 3)

    while True:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s, label="Spread Series", color="darkblue", alpha=0.7)
        ax.set_title(f"Step 2: OU Physics ({pair_name}) | kappa: {kappa}, sigma: {sigma}, Half-Life: {half_life} bars")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        
        fig.subplots_adjust(top=0.85, bottom=0.15)
        plt.show(block=False)

        print(f"\n[Fitted OU Metrics - 3 Sig Figs]: kappa = {kappa} | sigma = {sigma} | Half-Life = {half_life} bars")
        choice = input("Press [Enter] to ACCEPT, or type 'override' to set custom values: ").strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            return kappa, sigma
        elif choice == "override":
            k_in = input(f"Enter custom kappa (current: {kappa}): ").strip()
            s_in = input(f"Enter custom sigma (current: {sigma}): ").strip()
            if k_in: kappa = round_sig(float(k_in), 3)
            if s_in: sigma = round_sig(float(s_in), 3)
            half_life = round_sig(np.log(2) / kappa, 3)

# --- BACKTEST SIMULATION ENGINE FOR STEP 3 ---
def simulate_ou_strategy(spreads: np.ndarray, kappa: float, sigma: float, z_entry: float, z_exit: float):
    eq_std = sqrt((sigma ** 2) / max(2.0 * kappa, 1e-12))
    z_scores = spreads / max(eq_std, 1e-12)
    
    position = 0 
    pnl_series = []
    cash = 0.0
    
    long_entries = []
    short_entries = []
    exits = []
    
    for t in range(1, len(spreads)):
        z = z_scores[t]
        ds = spreads[t] - spreads[t-1]
        
        if position == 0:
            if z > z_entry:
                position = -1 
                short_entries.append((t, z))
            elif z < -z_entry:
                position = 1 
                long_entries.append((t, z))
        elif position == 1: 
            if z >= -z_exit:
                position = 0
                exits.append((t, z))
        elif position == -1: 
            if z <= z_exit:
                position = 0
                exits.append((t, z))
                
        cash += position * ds
        pnl_series.append(cash)
        
    return np.array(pnl_series), z_scores, long_entries, short_entries, exits

# --- INTERACTIVE STEP 3: THRESHOLD OPTIMIZATION & PNL GRAPH ---
def interactive_tune_thresholds(prices: np.ndarray, y_idx: int, x_idx: int, filt_cfg: FiltCfg, kappa: float, sigma: float, pair_name: str) -> tuple[float, float]:
    y_series = np.log(np.maximum(prices[y_idx], 1e-12))
    x_series = np.log(np.maximum(prices[x_idx], 1e-12))

    print(f"\n==========================================")
    print(f" STEP 3: Optimal Entry/Exit Thresholds ({pair_name})")
    print(f"==========================================")

    filt = PairSegFilter(filt_cfg)
    spreads = np.array([filt.step(y_series[t], x_series[t]).spread for t in range(len(y_series))])
    s_eval = spreads[30:]

    def objective(trial):
        z_entry = trial.suggest_float("z_entry", 1.0, 3.5)
        z_exit = trial.suggest_float("z_exit", 0.0, 1.0)
        if z_exit >= z_entry:
            return 0.0
        pnl, _, _, _, _ = simulate_ou_strategy(s_eval, kappa, sigma, z_entry, z_exit)
        if len(pnl) == 0:
            return 0.0
        return -pnl[-1] 

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    z_entry = round_sig(study.best_params["z_entry"], 3)
    z_exit = round_sig(study.best_params["z_exit"], 3)

    while True:
        pnl_curve, z_scores, long_entries, short_entries, exits = simulate_ou_strategy(s_eval, kappa, sigma, z_entry, z_exit)
        final_pnl = pnl_curve[-1] if len(pnl_curve) > 0 else 0.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        ax1.plot(z_scores, label="Spread Z-Score", color="dodgerblue", alpha=0.7)
        ax1.axhline(z_entry, color="red", linestyle="--", label=f"Entry (+{z_entry})")
        ax1.axhline(-z_entry, color="red", linestyle="--", label=f"Entry (-{z_entry})")
        ax1.axhline(z_exit, color="green", linestyle=":", label=f"Exit (+{z_exit})")
        ax1.axhline(-z_exit, color="green", linestyle=":", label=f"Exit (-{z_exit})")

        # Plot trade markers on Z-Score chart
        if long_entries:
            le_x, le_y = zip(*long_entries)
            ax1.scatter(le_x, le_y, color="green", marker="^", s=60, label="Long Entry", zorder=5)
        if short_entries:
            se_x, se_y = zip(*short_entries)
            ax1.scatter(se_x, se_y, color="red", marker="v", s=60, label="Short Entry", zorder=5)
        if exits:
            ex_x, ex_y = zip(*exits)
            ax1.scatter(ex_x, ex_y, color="black", marker="x", s=50, label="Exit Position", zorder=5)

        ax1.set_title(f"Step 3: Thresholds, Trade Markets & PnL ({pair_name}) | Final PnL: {final_pnl:.4f}")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(pnl_curve, label="Cumulative Strategy PnL", color="darkorange")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)

        fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.3)
        plt.show(block=False)

        print(f"\n[Backtest Results - 3 Sig Figs]:")
        print(f"  Optimal z_entry : {z_entry}")
        print(f"  Optimal z_exit  : {z_exit}")
        print(f"  Final PnL       : {final_pnl:.4f}")

        choice = input("Press [Enter] to ACCEPT thresholds, or type 'override' to set custom z_entry/z_exit: ").strip().lower()
        plt.close(fig)

        if choice == "" or choice == "accept":
            return z_entry, z_exit
        elif choice == "override":
            ze = input(f"Enter custom z_entry (current: {z_entry}): ").strip()
            zx = input(f"Enter custom z_exit (current: {z_exit}): ").strip()
            if ze: z_entry = round_sig(float(ze), 3)
            if zx: z_exit = round_sig(float(zx), 3)

# --- MAIN EXECUTION PIPELINE ---
def run_interactive_pipeline(prices_file: str):
    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None)
    prcAll = df.values.T

    y_idx, x_idx, pair_name = 8, 27, "HUXZ-ACAC"
    
    best_filt_cfg = interactive_tune_kalman(prcAll, y_idx, x_idx, pair_name)
    kappa, sigma = interactive_tune_ou(prcAll, y_idx, x_idx, best_filt_cfg, pair_name)
    z_entry, z_exit = interactive_tune_thresholds(prcAll, y_idx, x_idx, best_filt_cfg, kappa, sigma, pair_name)
    
    print(f"\n[Success] Complete Pipeline Finished for {pair_name}!")
    print(f"Final Confirmed Parameters & Thresholds:")
    print(f"  FiltCfg : {best_filt_cfg}")
    print(f"  kappa   : {kappa}")
    print(f"  sigma   : {sigma}")
    print(f"  z_entry : {z_entry}")
    print(f"  z_exit  : {z_exit}")

if __name__ == "__main__":
    prices_path = "./prices.txt"
    if Path(prices_path).exists():
        run_interactive_pipeline(prices_path)
    else:
        print(f"Error: {prices_path} not found.")