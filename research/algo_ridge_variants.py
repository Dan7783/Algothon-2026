"""Score four_sigmas variants that let the ridge trade ALGO.

The current book blocks ALGO in four places. This builds variants that unblock
it and scores them against the unmodified book over several windows, because
one window is a selection input, not evidence (VALIDATION_PROTOCOL: floor over
median, both train blocks).

Variants:
  baseline        unmodified four_sigmas
  pred_only       ALGO enters the ridge design matrix as a PREDICTOR, but its
                  own limit stays 0 - isolates "does the market factor help
                  forecast the other 50" from "should we hold ALGO"
  algo_L{n}       ALGO as predictor AND target, with a bang-bang limit of $n

Nothing in four_sigmas.py is modified; each variant is a freshly exec'd copy of
the module with three patches applied, so module state cannot leak between runs.

Run from repo root:  python research/algo_ridge_variants.py
"""
from __future__ import annotations

import sys as _sys
import types
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from synthetic_null import backtest, real_prices

OUT = _ROOT / "analysis_outputs" / "algo_ridge_probe"
OUT.mkdir(parents=True, exist_ok=True)

SRC = (_ROOT / "four_sigmas.py").read_text(encoding="utf-8")

# (start, end) windows. 501-750 is burned validation, 751-1000 the holdout,
# the earlier two are train blocks kept for the floor.
WINDOWS = [(251, 500), (376, 625), (501, 750), (751, 1000)]


def fresh_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(_ROOT / "four_sigmas.py")
    _sys.modules[name] = mod
    exec(compile(SRC, str(_ROOT / "four_sigmas.py"), "exec"), mod.__dict__)
    return mod


def patched_ridge(mod, include_algo: bool):
    """_ridge_forecast_details with ALGO optionally in the design matrix."""
    N_INST = mod.N_INST
    sel = slice(0, None) if include_algo else slice(1, None)
    width = N_INST if include_algo else N_INST - 1

    def _ridge_forecast_details(returns, *, ridge_lam=None, ridge_theta=None):
        lam = mod.RIDGE_LAM if ridge_lam is None else float(ridge_lam)
        theta = mod.RIDGE_THETA if ridge_theta is None else float(ridge_theta)
        nins, n_returns = returns.shape
        positions = np.zeros(nins)
        strengths = np.zeros(nins)
        latent_sign = np.zeros(nins)
        if n_returns + 1 < mod.RIDGE_MIN_T:
            return positions, strengths, latent_sign, False

        recent = returns[sel, -250:].T
        standardized = (recent - recent.mean(axis=0)) / (recent.std(axis=0) + 1.0e-12)
        lagged_links = (standardized[:-1].T @ standardized[1:]) / (len(recent) - 1)
        np.fill_diagonal(lagged_links, 0.0)
        significant_links = int(
            (np.abs(lagged_links) > 2.5 / np.sqrt(len(recent) - 1)).sum()
        )
        if significant_links < mod.RIDGE_GATE_N:
            return positions, strengths, latent_sign, False

        history = returns[sel, :].T
        if mod.RIDGE_LOOKBACK is not None:
            history = history[-mod.RIDGE_LOOKBACK:]
        scale = history.std(axis=0) + 1.0e-12
        predictors = history[:-1] / scale
        targets = history[1:] / scale
        coefficients = np.linalg.solve(
            predictors.T @ predictors + lam * len(predictors) * np.eye(width),
            predictors.T @ targets,
        )
        prediction_scale = (predictors @ coefficients).std(axis=0)
        prediction = (history[-1] / scale) @ coefficients
        latent_sign[sel] = np.sign(prediction)
        strengths[sel] = np.abs(prediction) / np.maximum(prediction_scale, 1.0e-12)

        active = strengths[sel] > theta
        active_indices = np.flatnonzero(active)
        if len(active_indices) > mod.RIDGE_MAX_ACTIVE:
            keep = active_indices[
                np.argsort(strengths[sel][active_indices])[-mod.RIDGE_MAX_ACTIVE:]
            ]
            active[:] = False
            active[keep] = True
        positions[sel] = mod.RIDGE_LEG * np.sign(prediction) * active
        return positions, strengths, latent_sign, True

    return _ridge_forecast_details


def patched_overlay(mod):
    """Keep the residual basket, but stop it wiping a deliberate ALGO position.

    The basket still never ADDS ALGO alpha and never hedges THROUGH ALGO; it
    just no longer force-zeroes the slot on the way out.
    """
    np_ = np
    ALGO = mod.I["ALGO"]

    def _algo_beta_neutral_overlay(base_dollars, requested_alpha, betas, crash):
        lower = -mod.BANG_BANG_LIMITS - base_dollars
        upper = mod.BANG_BANG_LIMITS - base_dollars
        alpha = np_.clip(requested_alpha, lower, upper)
        alpha[ALGO] = 0.0
        alpha[crash] = 0.0

        eligible = np_.ones(mod.N_INST, dtype=bool)
        eligible[ALGO] = False
        eligible[mod._ALGO_RESIDUAL_ASSETS] = False
        eligible[crash] = False

        alpha_beta = float(betas @ alpha)
        hedge, delivery = mod._bounded_minimum_norm_beta_hedge(
            -alpha_beta, betas, lower, upper, eligible
        )
        if delivery < 1.0:
            alpha *= delivery
            alpha_beta = float(betas @ alpha)
            hedge, _ = mod._bounded_minimum_norm_beta_hedge(
                -alpha_beta, betas, lower, upper, eligible
            )

        final = np_.clip(
            base_dollars + alpha + hedge,
            -mod.BANG_BANG_LIMITS,
            mod.BANG_BANG_LIMITS,
        )
        final[ALGO] = base_dollars[ALGO]        # was: 0.0
        return final, alpha, hedge, delivery

    return _algo_beta_neutral_overlay


def build(name: str, include_algo: bool, algo_limit: float):
    mod = fresh_module(f"fs_{name}")
    if include_algo:
        mod._ridge_forecast_details = patched_ridge(mod, True)
    if algo_limit > 0:
        mod.BANG_BANG_LIMITS = mod.BANG_BANG_LIMITS.copy()
        mod.BANG_BANG_LIMITS[mod.I["ALGO"]] = algo_limit
        mod._algo_beta_neutral_overlay = patched_overlay(mod)
    return mod


VARIANTS = [
    ("baseline", False, 0.0),
    ("pred_only", True, 0.0),
    ("algo_L10k", True, 10_000.0),
    ("algo_L25k", True, 25_000.0),
    ("algo_L50k", True, 50_000.0),
    ("algo_L100k", True, 100_000.0),
]

prc = real_prices()
rows = []
for name, inc, lim in VARIANTS:
    mod = build(name, inc, lim)
    rec = {"variant": name}
    for start, end in WINDOWS:
        mod.reset()
        stats = backtest(prc, mod.getMyPosition, start=start, end=end)
        rec[f"{start}-{end}"] = stats["score"]
        rec[f"sd_{start}-{end}"] = stats["std_pl"]
    scores = [rec[f"{s}-{e}"] for s, e in WINDOWS]
    rec["floor"] = min(scores)
    rec["median"] = float(np.median(scores))
    rows.append(rec)
    print(f"  {name:12s} " + "  ".join(
        f"{s}-{e}={rec[f'{s}-{e}']:8.1f}" for s, e in WINDOWS
    ) + f"   floor={rec['floor']:8.1f}")

df = pd.DataFrame(rows)
score_cols = ["variant"] + [f"{s}-{e}" for s, e in WINDOWS] + ["floor", "median"]
base = df[df["variant"] == "baseline"].iloc[0]
print("\n" + "=" * 92)
print("DELTA vs baseline")
for _, r in df.iterrows():
    if r["variant"] == "baseline":
        continue
    deltas = "  ".join(
        f"{s}-{e}={r[f'{s}-{e}'] - base[f'{s}-{e}']:+8.1f}" for s, e in WINDOWS
    )
    print(f"  {r['variant']:12s} {deltas}   floor={r['floor'] - base['floor']:+8.1f}")

df[score_cols].to_csv(OUT / "variant_scores.csv", index=False)
print(f"\nwrote {OUT / 'variant_scores.csv'}")
