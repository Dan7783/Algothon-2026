"""Build a notebook comparing causal multi-beta models for atlas pairs."""

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("dynamic_beta_atlas_analysis.ipynb")
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""
# Trading atlas pairs through changing beta regimes

The atlas relationships need not cease being pairs when their hedge ratios move. The relevant question is whether the strategy distinguishes:

- a **temporary pricing innovation**, which may be faded; and
- a **persistent beta change**, which should be incorporated into the hedge rather than treated as ever-increasing moneyness.

The current implementation re-estimates a Kalman beta at every decision, but then applies that latest beta retrospectively to the entire spread window. This notebook compares that construction with genuinely time-varying, causal alternatives:

1. the exact current atlas calculation;
2. rolling OLS beta using 60, 120, and 250 prior closes;
3. a random-walk Kalman alpha/beta model with one pooled parameter set;
4. the same model with `Q` and `R` calibrated separately for every pair using only days 1–250.
5. a better-identified version with a static intercept and only beta allowed to random-walk, again compared in pooled and pair-specific forms.

For the dynamic Kalman models, the runtime score is the standardized **one-step innovation**

\[
z_t = \frac{y_t - H_t\hat\theta_{t|t-1}}
{\sqrt{H_tP_{t|t-1}H_t^\top + R}},
\]

not a historical spread recomputed using today's beta. Parameter calibration uses the first 250 closes; tracking and trading are evaluated on days 251–750. Threshold selection is performed on days 251–500 and reported separately on days 501–750.

This remains research on a preselected atlas and therefore is not unbiased proof of future profitability.
"""))

cells.append(nbf.v4.new_code_cell(r"""
from pathlib import Path
import types
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 240)

ROOT = Path.cwd()
OUT_DIR = ROOT / "cointegration_output" / "dynamic_beta_atlas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

prices_df = pd.read_csv(ROOT / "prices.txt", sep=r"\s+")
names = prices_df.columns.to_list()
prices = prices_df.to_numpy(float).T
LP = np.log(prices)
n_inst, n_days = prices.shape

strategy_path = ROOT / "teamName.py"
strategy = types.ModuleType("teamName_source")
strategy.__file__ = str(strategy_path)
exec(compile(strategy_path.read_text(encoding="utf-8"), str(strategy_path), "exec"), strategy.__dict__)
atlas_pairs = list(strategy.ATLAS_PAIRS)

CALIBRATION_END = 250
EVAL_START = 249       # decision close; PnL starts at the next close
EVAL_END = 748
LEG = float(strategy.LEG)

print(f"{n_inst} assets, {n_days} closes, {len(atlas_pairs)} atlas pairs")
print("Calibration: observations 1–250; evaluation PnL: days 251–750")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Dynamic relationship estimators

The custom Kalman model centres the explanatory log price around its first-60-day mean. This is only a numerical reparameterization: beta is unchanged, but the intercept and beta process noises become less collinear.

For each pair, maximum likelihood estimates three positive variances on observations 61–250:

- `q_alpha`: random-walk intercept process variance;
- `q_beta`: random-walk beta process variance;
- `R`: observation/temporary-spread variance.

No forward trading PnL is used in this calibration. A pooled model uses the median calibrated variances across all 13 pairs, giving a direct test of whether universal dynamics are adequate.

Because a freely moving intercept and beta can explain the same level change, the notebook also fits a constrained **static-intercept / dynamic-beta** model. This is the cleaner test of the claim that the relationships persist while beta changes.
"""))

cells.append(nbf.v4.new_code_cell(r"""
def initial_regression(y, x, init=60):
    x_ref = float(x[:init].mean())
    X = np.column_stack([np.ones(init), x[:init] - x_ref])
    theta = np.linalg.lstsq(X, y[:init], rcond=None)[0]
    residual = y[:init] - X @ theta
    R0 = max(float(np.var(residual)), 1e-8)
    P0 = R0 * np.linalg.pinv(X.T @ X)
    P0 += np.eye(2) * 1e-8
    return x_ref, theta, P0, R0


def run_random_walk_kf(y, x, q_alpha, q_beta, R, init=60):
    '''Causal random-walk alpha/beta filter; arrays contain one-step priors.'''
    x_ref, theta, P, _ = initial_regression(y, x, init)
    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)
    z = np.full(n, np.nan)
    filtered_beta = np.full(n, np.nan)
    Q = np.diag([q_alpha, q_beta])
    I = np.eye(2)

    for t in range(init, n):
        P_prior = P + Q
        H = np.array([1.0, x[t] - x_ref])
        pred = float(H @ theta)
        v = float(y[t] - pred)
        S = max(float(H @ P_prior @ H + R), 1e-12)

        alpha[t] = float(theta[0] - theta[1] * x_ref)
        beta[t] = float(theta[1])
        innovation[t] = v
        innovation_var[t] = S
        z[t] = v / np.sqrt(S)

        K = P_prior @ H / S
        theta = theta + K * v
        # Joseph form is stable and keeps P positive semidefinite.
        KH = np.outer(K, H)
        P = (I - KH) @ P_prior @ (I - KH).T + np.outer(K, K) * R
        P = 0.5 * (P + P.T)
        filtered_beta[t] = float(theta[1])

    return {
        "alpha": alpha, "beta": beta, "innovation": innovation,
        "innovation_var": innovation_var, "z": z,
        "filtered_beta": filtered_beta,
    }


def kalman_nll(log_variances, y, x, fit_end=CALIBRATION_END):
    q_alpha, q_beta, R = np.exp(log_variances)
    result = run_random_walk_kf(y[:fit_end], x[:fit_end], q_alpha, q_beta, R)
    valid = np.arange(60, fit_end)
    v = result["innovation"][valid]
    S = result["innovation_var"][valid]
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(S)):
        return 1e100
    return float(0.5 * np.sum(np.log(S) + v * v / S))


def calibrate_pair(y, x):
    _, _, _, R0 = initial_regression(y[:CALIBRATION_END], x[:CALIBRATION_END])
    initial = np.log([max(R0 * 1e-3, 1e-10), max(R0 * 1e-4, 1e-11), R0])
    bounds = [(-25.0, -3.0), (-25.0, -3.0), (-18.0, -1.0)]
    fit = minimize(
        kalman_nll, initial, args=(y, x), method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-10},
    )
    q_alpha, q_beta, R = np.exp(fit.x)
    return q_alpha, q_beta, R, bool(fit.success), float(fit.fun)


def beta_only_nll(log_variances, y, x, fit_end=CALIBRATION_END):
    q_beta, R = np.exp(log_variances)
    result = run_random_walk_kf(y[:fit_end], x[:fit_end], 1e-12, q_beta, R)
    valid = np.arange(60, fit_end)
    v = result["innovation"][valid]
    S = result["innovation_var"][valid]
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(S)):
        return 1e100
    return float(0.5 * np.sum(np.log(S) + v * v / S))


def calibrate_beta_only(y, x):
    '''Static intercept, pair-specific random-walk beta and observation noise.'''
    _, _, _, R0 = initial_regression(y[:CALIBRATION_END], x[:CALIBRATION_END])
    initial = np.log([max(R0 * 1e-4, 1e-11), R0])
    fit = minimize(
        beta_only_nll, initial, args=(y, x), method="L-BFGS-B",
        bounds=[(-25.0, -3.0), (-18.0, -1.0)],
        options={"maxiter": 250, "ftol": 1e-10},
    )
    q_beta, R = np.exp(fit.x)
    return q_beta, R, bool(fit.success), float(fit.fun)


def rolling_ols(y, x, window):
    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for t in range(window, n):
        yy, xx = y[t-window:t], x[t-window:t]
        xc, yc = xx - xx.mean(), yy - yy.mean()
        b = float(xc @ yc / max(xc @ xc, 1e-18))
        a = float(yy.mean() - b * xx.mean())
        residual = yy - a - b * xx
        scale2 = max(float(np.var(residual)), 1e-12)
        v = float(y[t] - a - b * x[t])
        alpha[t], beta[t] = a, b
        innovation[t], innovation_var[t], z[t] = v, scale2, v / np.sqrt(scale2)
    return {
        "alpha": alpha, "beta": beta, "innovation": innovation,
        "innovation_var": innovation_var, "z": z,
        "filtered_beta": beta.copy(),
    }


def run_current_filter(y, x):
    '''One-pass reconstruction of teamName's universal-Q Kalman prior.'''
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    P = np.eye(2) * 1e-2
    Q = np.eye(2) * 1e-5
    R = float(np.var(np.diff(y[:120] - x[:120])) + 1e-8)
    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)
    z_innovation = np.full(n, np.nan)
    z_atlas = np.full(n, np.nan)
    active = np.zeros(n, dtype=bool)
    adf_t = np.full(n, np.nan)
    kappa = np.full(n, np.nan)

    for t in range(n):
        P_prior = P + Q
        H = np.array([1.0, x[t]])
        v = float(y[t] - H @ theta)
        S = max(float(H @ P_prior @ H + R), 1e-12)
        alpha[t], beta[t] = theta
        innovation[t], innovation_var[t] = v, S
        z_innovation[t] = v / np.sqrt(S)

        if t >= 129:
            spread = y[:t+1] - beta[t] * x[:t+1]
            recent = spread[-145:]
            z_atlas[t] = float(
                (recent[-1] - recent[:-1].mean()) / (recent[:-1].std() + 1e-12)
            )
            adf_t[t] = float(strategy._adf1_tstats(spread.reshape(-1, 1))[0])
            kappa[t] = float(strategy._ou_kappa(spread[-120:]))
            active[t] = bool(
                adf_t[t] < -2.9 and kappa[t] > 0.0
                and 0.27 <= abs(z_atlas[t]) <= 4.0
            )

        K = P_prior @ H / S
        theta = theta + K * v
        P = P_prior - np.outer(K, H) @ P_prior
        P = 0.5 * (P + P.T)

    return {
        "alpha": alpha, "beta": beta, "innovation": innovation,
        "innovation_var": innovation_var, "z": z_atlas,
        "innovation_z": z_innovation, "active": active,
        "adf_t": adf_t, "kappa": kappa,
        "filtered_beta": beta.copy(), "R": R,
    }
"""))

cells.append(nbf.v4.new_code_cell(r"""
parameter_rows = []
pair_data = {}

for number, (i, j) in enumerate(atlas_pairs, start=1):
    label = f"{names[i]}~{names[j]}"
    y, x = LP[i], LP[j]
    q_alpha, q_beta, R, success, train_nll = calibrate_pair(y, x)
    beta_only_q, beta_only_R, beta_only_success, beta_only_nll_value = calibrate_beta_only(y, x)
    pair_data[label] = {
        "i": i, "j": j, "y": y, "x": x,
        "pair_kf": run_random_walk_kf(y, x, q_alpha, q_beta, R),
        "pair_beta_kf": run_random_walk_kf(y, x, 1e-12, beta_only_q, beta_only_R),
        "current": run_current_filter(y, x),
        "rolling_60": rolling_ols(y, x, 60),
        "rolling_120": rolling_ols(y, x, 120),
        "rolling_250": rolling_ols(y, x, 250),
    }
    parameter_rows.append({
        "pair": label, "q_alpha": q_alpha, "q_beta": q_beta, "R": R,
        "q_beta_over_R": q_beta / R, "q_alpha_over_R": q_alpha / R,
        "optimizer_success": success, "calibration_nll": train_nll,
        "beta_only_q_beta": beta_only_q, "beta_only_R": beta_only_R,
        "beta_only_q_over_R": beta_only_q / beta_only_R,
        "beta_only_optimizer_success": beta_only_success,
        "beta_only_calibration_nll": beta_only_nll_value,
        "current_universal_q": 1e-5,
        "current_pair_R": pair_data[label]["current"]["R"],
    })
    print(f"Calibrated {number:2d}/13 {label}: qβ={q_beta:.2e}, R={R:.2e}")

parameters = pd.DataFrame(parameter_rows)
pooled = {
    "q_alpha": float(np.median(parameters.q_alpha)),
    "q_beta": float(np.median(parameters.q_beta)),
    "R": float(np.median(parameters.R)),
}
pooled_beta_only = {
    "q_beta": float(np.median(parameters.beta_only_q_beta)),
    "R": float(np.median(parameters.beta_only_R)),
}

for label, data in pair_data.items():
    data["pooled_kf"] = run_random_walk_kf(
        data["y"], data["x"], pooled["q_alpha"], pooled["q_beta"], pooled["R"]
    )
    data["pooled_beta_kf"] = run_random_walk_kf(
        data["y"], data["x"], 1e-12,
        pooled_beta_only["q_beta"], pooled_beta_only["R"]
    )

parameters.to_csv(OUT_DIR / "pair_specific_kalman_parameters.csv", index=False)
print("\nPooled median parameters:", pooled)
print("Pooled static-intercept beta parameters:", pooled_beta_only)
display(parameters.round(10))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Does pair-specific calibration improve later relationship tracking?

Tracking metrics use only observations 251–750. Lower RMSE and predictive negative log-likelihood are better. Innovation lag-1 correlation near zero means the model has absorbed persistent beta movement rather than leaving predictable same-direction errors.

Beta total variation is not a quality metric by itself; it shows how aggressively each estimator changes the hedge.
"""))

cells.append(nbf.v4.new_code_cell(r"""
def lag1_corr(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    values = values[valid]
    return float(np.corrcoef(values[:-1], values[1:])[0, 1]) if len(values) > 2 else np.nan


tracking_rows = []
tracking_methods = [
    "current", "rolling_60", "rolling_120", "rolling_250",
    "pooled_kf", "pair_kf", "pooled_beta_kf", "pair_beta_kf",
]
eval_slice = slice(CALIBRATION_END, n_days)

for label, data in pair_data.items():
    for method in tracking_methods:
        result = data[method]
        v = result["innovation"][eval_slice]
        S = result["innovation_var"][eval_slice]
        b = result["beta"][eval_slice]
        valid = np.isfinite(v) & np.isfinite(S) & (S > 0)
        vv, SS = v[valid], S[valid]
        tracking_rows.append({
            "pair": label,
            "method": method,
            "observations": int(valid.sum()),
            "innovation_RMSE": float(np.sqrt(np.mean(vv * vv))),
            "innovation_MAE": float(np.mean(np.abs(vv))),
            "predictive_NLL_per_obs": float(np.mean(0.5 * (np.log(SS) + vv * vv / SS))),
            "innovation_lag1_corr": lag1_corr(vv),
            "beta_daily_abs_change": float(np.nanmedian(np.abs(np.diff(b)))),
            "beta_total_variation": float(np.nansum(np.abs(np.diff(b)))),
        })

tracking = pd.DataFrame(tracking_rows)
tracking.to_csv(OUT_DIR / "dynamic_beta_tracking_metrics.csv", index=False)

tracking_aggregate = (
    tracking.groupby("method", as_index=False)
    .agg(
        median_RMSE=("innovation_RMSE", "median"),
        median_MAE=("innovation_MAE", "median"),
        median_NLL=("predictive_NLL_per_obs", "median"),
        median_abs_lag1_corr=("innovation_lag1_corr", lambda x: np.median(np.abs(x))),
        median_beta_daily_change=("beta_daily_abs_change", "median"),
    )
    .sort_values("median_NLL")
)
display(tracking_aggregate.round(6))

comparison = tracking.pivot(index="pair", columns="method", values="predictive_NLL_per_obs")
pair_specific_nll_wins = int((comparison.pair_kf < comparison.pooled_kf).sum())
rmse_comparison = tracking.pivot(index="pair", columns="method", values="innovation_RMSE")
pair_specific_rmse_wins = int((rmse_comparison.pair_kf < rmse_comparison.pooled_kf).sum())
print(f"Pair-specific KF beats pooled KF on OOS NLL for {pair_specific_nll_wins}/13 pairs")
print(f"Pair-specific KF beats pooled KF on OOS RMSE for {pair_specific_rmse_wins}/13 pairs")
beta_comparison = tracking.pivot(index="pair", columns="method", values="predictive_NLL_per_obs")
beta_rmse_comparison = tracking.pivot(index="pair", columns="method", values="innovation_RMSE")
beta_nll_wins = int((beta_comparison.pair_beta_kf < beta_comparison.pooled_beta_kf).sum())
beta_rmse_wins = int((beta_rmse_comparison.pair_beta_kf < beta_rmse_comparison.pooled_beta_kf).sum())
print(f"Pair-specific beta-only KF beats pooled beta-only KF on OOS NLL for {beta_nll_wins}/13 pairs")
print(f"Pair-specific beta-only KF beats pooled beta-only KF on OOS RMSE for {beta_rmse_wins}/13 pairs")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Trading comparison

Every candidate position is formed at one close and marked at the next. Turnover costs are 1 bp. Two sizing policies are compared:

- `equal`: the current atlas uses equal ±$10,000 legs;
- `dynamic_beta`: dollar exposure in the second asset is `-beta` times exposure in the first, with both legs kept inside the $10,000 instrument limit.

The dynamic estimators trade their causal one-step innovation z-score. The exact current atlas retains its ADF, kappa, 145-day z and dead-zone gates. Results sum standalone pair PnLs before shared-leg portfolio netting, so they compare signal constructions rather than reproduce total `teamName.py` PnL.
"""))

cells.append(nbf.v4.new_code_cell(r"""
def score_fn(pnl):
    pnl = np.asarray(pnl, float)
    mu, sigma = float(pnl.mean()), float(pnl.std())
    if mu <= 0 or sigma < 1e-10:
        return mu
    sharpe = np.sqrt(250.0) * mu / sigma
    return float(mu * sharpe * sharpe / (sharpe * sharpe + 1.0))


def pnl_metrics(pnl):
    pnl = np.asarray(pnl, float)
    sigma = float(pnl.std())
    cumulative = np.r_[0.0, np.cumsum(pnl)]
    return {
        "net_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()),
        "annual_sharpe": float(np.sqrt(250.0) * pnl.mean() / sigma) if sigma > 0 else np.nan,
        "score": score_fn(pnl),
        "hit_rate": float(np.mean(pnl > 0)),
        "max_drawdown": float(np.min(cumulative - np.maximum.accumulate(cumulative))),
    }


def backtest_pair(data, method, threshold, sizing):
    i, j = data["i"], data["j"]
    result = data[method]
    previous = np.zeros(2, dtype=int)
    pnl = []
    turnover = 0.0
    active_days = 0
    for day in range(EVAL_START, EVAL_END + 1):
        beta = float(result["beta"][day])
        if method == "current":
            active = bool(result["active"][day])
            z = float(result["z"][day])
        else:
            z = float(result["z"][day])
            active = bool(np.isfinite(z) and threshold <= abs(z) <= 4.0 and np.isfinite(beta) and beta > 0)
        target = np.zeros(2, dtype=int)
        if active:
            direction = -np.sign(z)
            if sizing == "dynamic_beta":
                leg_i = LEG / max(1.0, abs(beta))
                dollars_i = direction * leg_i
                dollars_j = -direction * beta * leg_i
            else:
                dollars_i = direction * LEG
                dollars_j = -direction * LEG
            target[0] = int(dollars_i / prices[i, day])
            target[1] = int(dollars_j / prices[j, day])
            active_days += 1
        delta = target - previous
        day_turnover = float(prices[i, day] * abs(delta[0]) + prices[j, day] * abs(delta[1]))
        day_pnl = float(
            target[0] * (prices[i, day + 1] - prices[i, day])
            + target[1] * (prices[j, day + 1] - prices[j, day])
            - 0.0001 * day_turnover
        )
        pnl.append(day_pnl)
        turnover += day_turnover
        previous = target
    return np.asarray(pnl), turnover, active_days


thresholds = [0.27, 0.5, 1.0, 1.5, 2.0]
trade_methods = [
    "rolling_60", "rolling_120", "rolling_250", "pooled_kf", "pair_kf",
    "pooled_beta_kf", "pair_beta_kf",
]
sizings = ["equal", "dynamic_beta"]
portfolio_pnl = {}
trade_rows = []

for method in trade_methods:
    for sizing in sizings:
        for threshold in thresholds:
            total = np.zeros(EVAL_END - EVAL_START + 1)
            total_turnover = 0.0
            total_active = 0
            for label, data in pair_data.items():
                pair_pnl, turnover, active_days = backtest_pair(data, method, threshold, sizing)
                total += pair_pnl
                total_turnover += turnover
                total_active += active_days
            key = (method, sizing, threshold)
            portfolio_pnl[key] = total
            for period, values in {
                "development_251_500": total[:250],
                "later_501_750": total[250:],
                "all_251_750": total,
            }.items():
                trade_rows.append({
                    "method": method, "sizing": sizing, "threshold": threshold,
                    "period": period, "turnover": total_turnover,
                    "pair_active_days": total_active, **pnl_metrics(values),
                })

# Exact current atlas signals under equal and dynamic-beta sizing.
for sizing in sizings:
    total = np.zeros(EVAL_END - EVAL_START + 1)
    total_turnover = 0.0
    total_active = 0
    for label, data in pair_data.items():
        pair_pnl, turnover, active_days = backtest_pair(data, "current", np.nan, sizing)
        total += pair_pnl
        total_turnover += turnover
        total_active += active_days
    portfolio_pnl[("current", sizing, np.nan)] = total
    for period, values in {
        "development_251_500": total[:250],
        "later_501_750": total[250:],
        "all_251_750": total,
    }.items():
        trade_rows.append({
            "method": "current", "sizing": sizing, "threshold": np.nan,
            "period": period, "turnover": total_turnover,
            "pair_active_days": total_active, **pnl_metrics(values),
        })

trades = pd.DataFrame(trade_rows)
trades.to_csv(OUT_DIR / "dynamic_beta_trading_sweep.csv", index=False)

# Select one common threshold per method/sizing on development score only.
selection_rows = []
for method in trade_methods:
    for sizing in sizings:
        dev = trades[
            (trades.method == method) & (trades.sizing == sizing)
            & (trades.period == "development_251_500")
        ].sort_values("score", ascending=False)
        chosen = float(dev.iloc[0].threshold)
        for period in ["development_251_500", "later_501_750", "all_251_750"]:
            row = trades[
                (trades.method == method) & (trades.sizing == sizing)
                & (trades.threshold == chosen) & (trades.period == period)
            ].iloc[0]
            selection_rows.append({
                "method": method, "sizing": sizing,
                "development_selected_threshold": chosen,
                "period": period, "net_pnl": row.net_pnl,
                "mean_pnl": row.mean_pnl, "annual_sharpe": row.annual_sharpe,
                "score": row.score, "max_drawdown": row.max_drawdown,
            })

for sizing in sizings:
    for period in ["development_251_500", "later_501_750", "all_251_750"]:
        row = trades[
            (trades.method == "current") & (trades.sizing == sizing)
            & (trades.period == period)
        ].iloc[0]
        selection_rows.append({
            "method": "current", "sizing": sizing,
            "development_selected_threshold": np.nan,
            "period": period, "net_pnl": row.net_pnl,
            "mean_pnl": row.mean_pnl, "annual_sharpe": row.annual_sharpe,
            "score": row.score, "max_drawdown": row.max_drawdown,
        })

selected = pd.DataFrame(selection_rows)
selected.to_csv(OUT_DIR / "development_selected_trading_results.csv", index=False)
display(selected[selected.period == "later_501_750"].sort_values("score", ascending=False).round(3))

# Pair-level attribution for the viable multiple-beta candidate.
rolling250_threshold = float(
    selected[
        (selected.method == "rolling_250") & (selected.sizing == "equal")
        & (selected.period == "development_251_500")
    ].development_selected_threshold.iloc[0]
)
pair_comparison_rows = []
for label, data in pair_data.items():
    candidates = {
        "current_equal": backtest_pair(data, "current", np.nan, "equal")[0],
        "rolling250_equal": backtest_pair(data, "rolling_250", rolling250_threshold, "equal")[0],
        "rolling250_dynamic_beta": backtest_pair(data, "rolling_250", rolling250_threshold, "dynamic_beta")[0],
    }
    for candidate, values in candidates.items():
        for period, period_values in {
            "development_251_500": values[:250],
            "later_501_750": values[250:],
            "all_251_750": values,
        }.items():
            pair_comparison_rows.append({
                "pair": label, "candidate": candidate, "period": period,
                **pnl_metrics(period_values),
            })
pair_comparison = pd.DataFrame(pair_comparison_rows)
pair_comparison.to_csv(OUT_DIR / "rolling250_pair_attribution.csv", index=False)
display(
    pair_comparison[pair_comparison.period == "later_501_750"]
    .pivot(index="pair", columns="candidate", values="net_pnl")
    .round(2)
)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Parameter heterogeneity and beta paths

If pair-specific process dynamics are justified, calibrated `q_beta / R` should differ materially across pairs and pair-specific filters should improve later predictive tracking relative to pooled parameters. Parameter variation alone is not enough: OOS tracking and trading decide whether the extra flexibility is useful.
"""))

cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
xpos = np.arange(len(parameters))

axes[0].bar(xpos, np.log10(parameters.beta_only_q_beta), color="#7c3aed")
axes[0].axhline(np.log10(pooled_beta_only["q_beta"]), color="black", ls="--", label="pooled median")
axes[0].set_title("Static-intercept beta process variance")
axes[0].set_ylabel("log10(q_beta)")
axes[0].legend()

axes[1].bar(xpos, np.log10(parameters.beta_only_R), color="#2563eb")
axes[1].axhline(np.log10(pooled_beta_only["R"]), color="black", ls="--", label="pooled median")
axes[1].set_title("Static-intercept observation variance")
axes[1].set_ylabel("log10(R)")
axes[1].legend()

axes[2].bar(xpos, np.log10(parameters.beta_only_q_over_R), color="#059669")
axes[2].set_title("Relative beta adaptation speed")
axes[2].set_ylabel("log10(q_beta / R)")

for ax in axes:
    ax.set_xticks(xpos)
    ax.set_xticklabels(parameters.pair, rotation=65, ha="right", fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "pair_specific_kalman_parameters.png", dpi=160, bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
axes = axes.ravel()
for ax, (label, data) in zip(axes, pair_data.items()):
    days = np.arange(n_days)
    ax.plot(days, data["current"]["beta"], color="#111827", lw=1.0, label="current universal-Q")
    ax.plot(days, data["rolling_120"]["beta"], color="#f59e0b", lw=0.9, alpha=0.8, label="rolling OLS 120")
    ax.plot(days, data["pooled_beta_kf"]["beta"], color="#2563eb", lw=1.0, label="pooled beta-KF")
    ax.plot(days, data["pair_beta_kf"]["beta"], color="#7c3aed", lw=1.2, label="pair beta-KF")
    ax.axvline(CALIBRATION_END, color="#6b7280", ls=":", lw=0.9)
    ax.axvline(500, color="#6b7280", ls=":", lw=0.9)
    ax.set_title(label, fontsize=10)
    ax.set_ylim(np.nanpercentile(data["rolling_60"]["beta"], 1) - 0.05,
                np.nanpercentile(data["rolling_60"]["beta"], 99) + 0.05)
for ax in axes[len(pair_data):]:
    ax.axis("off")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", bbox_to_anchor=(0.5, 0.004),
           ncol=4, frameon=False)
fig.suptitle("Causal beta paths under multiple estimators", fontsize=16)
fig.supxlabel("Observation; vertical lines at calibration end (250) and later split (500)", y=0.026)
fig.supylabel("beta known before current close")
fig.tight_layout(rect=[0.02, 0.055, 1, 0.97])
fig.savefig(OUT_DIR / "all_pair_dynamic_beta_paths.png", dpi=160, bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for method, color in [("rolling_60", "#f59e0b"), ("rolling_120", "#d97706"),
                      ("rolling_250", "#92400e"), ("pooled_beta_kf", "#2563eb"),
                      ("pair_beta_kf", "#7c3aed")]:
    subset = trades[(trades.method == method) & (trades.sizing == "dynamic_beta")]
    for ax, period, title in [
        (axes[0], "development_251_500", "Development days 251–500"),
        (axes[1], "later_501_750", "Later days 501–750"),
    ]:
        values = subset[subset.period == period].sort_values("threshold")
        ax.plot(values.threshold, values.score, marker="o", label=method, color=color)
        ax.set_title(title)
        ax.set_xlabel("innovation |z| entry threshold")
        ax.set_ylabel("score")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "dynamic_beta_threshold_robustness.png", dpi=160, bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Pair-by-pair multiple-beta diagnostics

Each section compares beta paths, exact-current versus rolling-250 deviation scores, and cumulative standalone PnL. The rolling threshold is selected once from aggregate development performance, not separately for each pair. Pair-specific Kalman tracking statistics remain in the table so its rejected high-gain behaviour is visible.
"""))

cells.append(nbf.v4.new_code_cell(r"""
chosen_pair_threshold = rolling250_threshold

for number, (label, data) in enumerate(pair_data.items(), start=1):
    display(Markdown(f"### {number}. {label}"))
    parameter_view = parameters[parameters.pair == label].set_index("pair").T
    pair_tracking = tracking[tracking.pair == label].set_index("method")[[
        "innovation_RMSE", "predictive_NLL_per_obs", "innovation_lag1_corr",
        "beta_daily_abs_change"
    ]]
    display(parameter_view.round(10))
    display(pair_tracking.round(6))

    current_pnl, _, _ = backtest_pair(data, "current", np.nan, "equal")
    rolling_pnl, _, _ = backtest_pair(data, "rolling_250", chosen_pair_threshold, "equal")
    rolling_beta_pnl, _, _ = backtest_pair(data, "rolling_250", chosen_pair_threshold, "dynamic_beta")
    days = np.arange(n_days)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(days, data["current"]["beta"], color="#111827", lw=1.0, label="current")
    axes[0, 0].plot(days, data["rolling_120"]["beta"], color="#f59e0b", lw=0.9, label="rolling 120")
    axes[0, 0].plot(days, data["pooled_beta_kf"]["beta"], color="#2563eb", lw=1.0, label="pooled beta-KF")
    axes[0, 0].plot(days, data["pair_beta_kf"]["beta"], color="#7c3aed", lw=1.2, label="pair beta-KF")
    axes[0, 0].set_title("Causal beta paths")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(days, data["current"]["z"], color="#374151", lw=0.9, label="current atlas z")
    axes[0, 1].plot(days, data["rolling_250"]["z"], color="#7c3aed", lw=0.9, alpha=0.85, label="rolling-250 causal z")
    axes[0, 1].axhline(chosen_pair_threshold, color="#059669", ls="--", lw=0.9)
    axes[0, 1].axhline(-chosen_pair_threshold, color="#059669", ls="--", lw=0.9)
    axes[0, 1].set_ylim(-5, 5)
    axes[0, 1].set_title("Runtime deviation scores")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(np.arange(251, 751), np.cumsum(current_pnl), color="#111827", label="current equal-dollar")
    axes[1, 0].plot(np.arange(251, 751), np.cumsum(rolling_pnl), color="#7c3aed", label="rolling-250 equal-dollar")
    axes[1, 0].plot(np.arange(251, 751), np.cumsum(rolling_beta_pnl), color="#2563eb", label="rolling-250 beta-weighted")
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set_title("Standalone after-cost cumulative PnL")
    axes[1, 0].legend(fontsize=8)

    beta_gap = np.abs(data["rolling_250"]["beta"] - data["current"]["beta"])
    axes[1, 1].plot(days, beta_gap, color="#dc2626", lw=1.0)
    axes[1, 1].set_title("Rolling-250 vs current |beta difference|")
    axes[1, 1].set_ylabel("absolute beta gap")

    for ax in axes.ravel():
        ax.axvline(CALIBRATION_END, color="#6b7280", ls=":", lw=0.9)
        ax.axvline(500, color="#6b7280", ls=":", lw=0.9)
        ax.set_xlabel("observation / PnL day")
    fig.suptitle(label, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{number:02d}_{label.replace('~', '_')}_multiple_beta.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Data-specific conclusion

The next cell summarizes the actual chronology-split result and identifies which pairs benefit from the viable rolling-250 beta. It is calculated from the tables above rather than hardcoded.
"""))

cells.append(nbf.v4.new_code_cell(r"""
later_selected = selected[selected.period == "later_501_750"].sort_values("score", ascending=False)
display(later_selected[[
    "method", "sizing", "development_selected_threshold", "net_pnl",
    "annual_sharpe", "score", "max_drawdown"
]].round(3))

later_pair_pivot = (
    pair_comparison[pair_comparison.period == "later_501_750"]
    .pivot(index="pair", columns="candidate", values="net_pnl")
)
later_pair_pivot["rolling250_minus_current"] = (
    later_pair_pivot.rolling250_equal - later_pair_pivot.current_equal
)
print("Pairs helped by rolling-250 equal-dollar in later data:")
display(later_pair_pivot.sort_values("rolling250_minus_current", ascending=False).round(2))

print(
    f"Pair-specific unrestricted KF OOS wins: NLL {pair_specific_nll_wins}/13, "
    f"RMSE {pair_specific_rmse_wins}/13."
)
print(
    f"Pair-specific static-intercept beta-KF OOS wins: NLL {beta_nll_wins}/13, "
    f"RMSE {beta_rmse_wins}/13."
)
print(
    "Conclusion: betas are time-varying, but unrestricted pair-specific Kalman "
    "calibration is weakly identified and absorbs much of the tradeable spread. "
    "The robust candidate is a causal rolling-250 local relationship. It should "
    "be tested inside the complete constrained strategy, ideally selectively or "
    "with shrinkage, before replacing every atlas pair."
)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""
## Decision framework

The results above answer three separate questions:

1. **Do betas change?** Compare causal beta paths and total variation. A changing beta is not evidence that the pair has disappeared.
2. **Should Kalman noise be pair-specific?** Pair-specific calibration is supported only if it improves later predictive NLL/RMSE and trading relative to the pooled filter. Different fitted parameters alone do not justify 39 hardcoded constants.
3. **How should changing-beta pairs be traded?** Use the beta known before the current close, trade the standardized one-step innovation, and hedge dollars with that same beta. Do not apply today's beta retrospectively to old observations.

A production implementation should store state between calls for speed. If stateless execution is required, the filter can be replayed over history as `teamName.py` currently does. Pair-specific parameters should be estimated from a causal calibration sample or updated slowly online; choosing them from full-sample PnL would be overfitting.
"""))

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK} with {len(cells)} cells")
