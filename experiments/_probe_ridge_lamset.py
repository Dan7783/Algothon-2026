"""Quick probe: average friend_book ridge predictions over a LAMBDA set.

Mirrors v14's idea on the live ridge sleeve only (pairs / ALGO residual unchanged).
Compares single RIDGE_LAM=1.52 vs unit-normalised prediction average over LAMSET.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _run_cpcv import calc_daily_pnl, path_metrics, run_cpcv  # noqa: E402

LAMSET = (0.5, 1.0, 1.52, 2.0, 3.0)


def load_strategy(module_name: str):
    path = ROOT / f"{module_name}.py"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    # Reload clean each time
    if module_name in sys.modules:
        del sys.modules[module_name]
    sys.modules[module_name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def load_prices(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def patch_ridge_ensemble(mod, lamset: tuple[float, ...]):
    """Replace _ridge_forecast_details with unit-average over lamset."""
    orig = mod._ridge_forecast_details

    def _ridge_forecast_details(returns, *, ridge_lam=None, ridge_theta=None):
        # Confirmation path still uses explicit ridgelam/theta when passed.
        if ridge_lam is not None:
            return orig(returns, ridge_lam=ridge_lam, ridge_theta=ridge_theta)

        theta = mod.RIDGE_THETA if ridge_theta is None else float(ridge_theta)
        nins, n_returns = returns.shape
        positions = np.zeros(nins)
        strengths = np.zeros(nins)
        latent_sign = np.zeros(nins)
        if n_returns + 1 < mod.RIDGE_MIN_T:
            return positions, strengths, latent_sign, False

        # Gate uses same lagged-links test as original (independent of lam).
        recent = returns[1:, -250:].T
        standardized = (recent - recent.mean(axis=0)) / (recent.std(axis=0) + 1.0e-12)
        lagged_links = (standardized[:-1].T @ standardized[1:]) / (len(recent) - 1)
        np.fill_diagonal(lagged_links, 0.0)
        significant_links = int(
            (np.abs(lagged_links) > 2.5 / np.sqrt(len(recent) - 1)).sum()
        )
        if significant_links < mod.RIDGE_GATE_N:
            return positions, strengths, latent_sign, False

        history = returns[1:, :].T
        if mod.RIDGE_LOOKBACK is not None:
            history = history[-mod.RIDGE_LOOKBACK :]
        scale = history.std(axis=0) + 1.0e-12
        predictors = history[:-1] / scale
        targets = history[1:] / scale
        xtx = predictors.T @ predictors
        xty = predictors.T @ targets
        eye = np.eye(mod.N_INST - 1)
        tlen = len(predictors)

        acc = np.zeros(mod.N_INST - 1)
        got = 0
        for lam in lamset:
            coefficients = np.linalg.solve(xtx + float(lam) * tlen * eye, xty)
            prediction = (history[-1] / scale) @ coefficients
            n = float(np.linalg.norm(prediction))
            if np.isfinite(n) and n > 0:
                acc += prediction / n
                got += 1
        if got == 0:
            return positions, strengths, latent_sign, False
        prediction = acc / got

        # Strength scale from the primary lambda fit (keeps theta comparable).
        coefficients_primary = np.linalg.solve(
            xtx + float(mod.RIDGE_LAM) * tlen * eye, xty
        )
        prediction_scale = (predictors @ coefficients_primary).std(axis=0)
        latent_sign[1:] = np.sign(prediction)
        strengths[1:] = np.abs(prediction) / np.maximum(prediction_scale, 1.0e-12)
        active = strengths[1:] > theta
        active_indices = np.flatnonzero(active)
        if len(active_indices) > mod.RIDGE_MAX_ACTIVE:
            keep = active_indices[
                np.argsort(strengths[1:][active_indices])[-mod.RIDGE_MAX_ACTIVE :]
            ]
            active[:] = False
            active[keep] = True
        positions[1:] = mod.RIDGE_LEG * np.sign(prediction) * active
        return positions, strengths, latent_sign, True

    mod._ridge_forecast_details = _ridge_forecast_details


def score_module(label: str, mod) -> dict:
    prices = load_prices(ROOT / "prices.txt")
    n_inst = prices.shape[0]
    comm = np.full(n_inst, 0.0001)
    comm[0] = 0.00002
    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0

    days, pll = calc_daily_pnl(
        prices,
        mod.getMyPosition,
        getattr(mod, "reset", None),
        251,
        1000,
        comm,
        limits,
    )
    full = path_metrics(pll)
    cpcv = run_cpcv(days, pll, n_groups=6, n_test_groups=2, purge=10, pct_embargo=0.01)
    oos = path_metrics(pll[(days >= 751) & (days <= 1000)])
    return {
        "label": label,
        "full_score": full["score"],
        "oos_751_1000": oos["score"],
        "cpcv_median": cpcv["score_summary"]["median"],
        "cpcv_mean": cpcv["score_summary"]["mean"],
        "cpcv_min": cpcv["score_summary"]["min"],
        "cpcv_frac_pos": cpcv["score_summary"]["frac_positive"],
    }


def main() -> None:
    base = load_strategy("friend_book_latest")
    print("Scoring baseline friend_book_latest...")
    r0 = score_module("baseline_lam=1.52", base)
    print(r0)

    ens = load_strategy("friend_book_latest")
    patch_ridge_ensemble(ens, LAMSET)
    print(f"Scoring ridge LAMSET={LAMSET} ensemble...")
    r1 = score_module(f"ridge_lamset={LAMSET}", ens)
    print(r1)

    out = {"baseline": r0, "ensemble": r1, "lamset": list(LAMSET)}
    path = ROOT / "auto_tuning_results" / "ridge_lamset_ensemble_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    for k in ("full_score", "oos_751_1000", "cpcv_median", "cpcv_mean", "cpcv_min"):
        d = r1[k] - r0[k]
        print(f"  delta {k}: {d:+.1f}")


if __name__ == "__main__":
    main()
