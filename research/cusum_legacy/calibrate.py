"""MLE calibration of Kalman process/observation noise (Q, R) for one pair.

Maximizes the sum of log innovation likelihoods
``log N(y_k; 0, S_k)`` over a designated calm-period window via
``scipy.optimize.minimize`` on the negative log-likelihood.

Q is parameterized as a diagonal SPD matrix (independent random walks on
alpha and beta) — the identifiable, standard choice for this model.  The
fitted result is always returned as a full (2, 2) matrix with zeros
off-diagonal.  R is returned as a (1, 1) array.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

# Repo root so ``prices`` resolves when this module lives under research/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.cusum_legacy.kalman_pair import KalmanPairConfig, ols_init, run_kalman_pair
from prices import extract_pair, load_prices

# Re-export for legacy callers
__all__ = [
    "CalibrationResult",
    "calibrate_pair",
    "calibrate_pairs",
    "extract_pair",
    "load_prices",
]


# Tunable defaults (named; never buried as magic numbers in bodies)
DEFAULT_INIT_WINDOW: int = 20
DEFAULT_Q_ALPHA_SCALE: float = 1e-3  # relative to R0
DEFAULT_Q_BETA_SCALE: float = 1e-4  # relative to R0
DEFAULT_LOG_Q_BOUNDS: tuple[float, float] = (-25.0, 2.0)
DEFAULT_LOG_R_BOUNDS: tuple[float, float] = (-20.0, 5.0)
DEFAULT_MAXITER: int = 250
DEFAULT_FTOL: float = 1e-10
# Skip the OLS warm-start observations when scoring NLL (filter still runs
# from t=0 so the state is well initialized by the scoring window).
DEFAULT_NLL_SKIP: int | None = None  # None => use init_window


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted noise parameters and optimizer diagnostics for one pair."""

    y_ticker: str
    x_ticker: str
    Q: np.ndarray  # (2, 2)
    R: np.ndarray  # (1, 1)
    x0: np.ndarray  # (2,)
    P0: np.ndarray  # (2, 2)
    config: KalmanPairConfig
    nll: float
    success: bool
    message: str
    n_obs_scored: int
    calm_start: int
    calm_end: int
    init_window: int

    @property
    def converged(self) -> bool:
        return bool(self.success)


def _pack_params(q_alpha: float, q_beta: float, R: float) -> np.ndarray:
    return np.log(np.array([q_alpha, q_beta, R], dtype=float))


def _unpack_params(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_alpha, q_beta, R = np.exp(theta)
    Q = np.diag([float(q_alpha), float(q_beta)])
    R_arr = np.array([[float(R)]], dtype=float)
    return Q, R_arr


def negative_log_likelihood(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    *,
    skip: int = 0,
) -> float:
    """Gaussian innovation NLL for diagonal-Q parameterization ``theta``."""
    Q, R = _unpack_params(theta)
    config = KalmanPairConfig(Q=Q, R=R, x0=x0, P0=P0)
    result = run_kalman_pair(y, x, config)

    start = max(int(skip), 0)
    innov = result.innovations[start:]
    S = result.innov_var[start:]
    if innov.size == 0:
        return 1e100
    if not (np.all(np.isfinite(innov)) and np.all(np.isfinite(S))):
        return 1e100
    if np.any(S <= 0.0):
        return 1e100

    # log N(y; 0, S) = -0.5 (log 2π + log S + y^2 / S)
    nll = 0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(S) + innov * innov / S))
    if not np.isfinite(nll):
        return 1e100
    return nll


def calibrate_pair(
    y: np.ndarray,
    x: np.ndarray,
    *,
    y_ticker: str = "y",
    x_ticker: str = "x",
    calm_start: int = 0,
    calm_end: int | None = None,
    init_window: int = DEFAULT_INIT_WINDOW,
    q_alpha_scale: float = DEFAULT_Q_ALPHA_SCALE,
    q_beta_scale: float = DEFAULT_Q_BETA_SCALE,
    log_q_bounds: tuple[float, float] = DEFAULT_LOG_Q_BOUNDS,
    log_r_bounds: tuple[float, float] = DEFAULT_LOG_R_BOUNDS,
    maxiter: int = DEFAULT_MAXITER,
    ftol: float = DEFAULT_FTOL,
    nll_skip: int | None = DEFAULT_NLL_SKIP,
) -> CalibrationResult:
    """Calibrate diagonal Q and scalar R on a calm-period window via MLE.

    Parameters
    ----------
    y, x:
        Full price series (calm window is sliced internally).
    calm_start, calm_end:
        Inclusive-start / exclusive-end indices of the calm window.
        ``calm_end=None`` uses the full series length.
    init_window:
        OLS warm-start length inside the calm window (default 20).
    q_alpha_scale, q_beta_scale:
        Initial Q diagonal entries as fractions of the OLS residual
        variance R0 (defaults 1e-3 and 1e-4).
    log_q_bounds, log_r_bounds:
        Bounds on log-parameters for L-BFGS-B.
    maxiter, ftol:
        Optimizer controls (defaults 250 and 1e-10).
    nll_skip:
        Number of leading calm-window observations excluded from the NLL
        sum.  ``None`` (default) skips ``init_window`` observations.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError(f"y and x length mismatch: {y.shape} vs {x.shape}")

    T = len(y)
    if calm_end is None:
        calm_end = T
    if not (0 <= calm_start < calm_end <= T):
        raise ValueError(
            f"Invalid calm window [{calm_start}, {calm_end}) for series length {T}"
        )

    y_c = y[calm_start:calm_end]
    x_c = x[calm_start:calm_end]
    n_calm = len(y_c)
    if n_calm < init_window + 5:
        raise ValueError(
            f"Calm window too short ({n_calm}) for init_window={init_window}"
        )

    x0, P0, R0 = ols_init(y_c, x_c, init_window=init_window)
    theta0 = _pack_params(
        max(R0 * q_alpha_scale, 1e-12),
        max(R0 * q_beta_scale, 1e-12),
        R0,
    )
    skip = init_window if nll_skip is None else int(nll_skip)

    bounds = [log_q_bounds, log_q_bounds, log_r_bounds]

    def _objective(theta: np.ndarray) -> float:
        return negative_log_likelihood(theta, y_c, x_c, x0, P0, skip=skip)

    fit: OptimizeResult = minimize(
        _objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    Q, R = _unpack_params(fit.x)
    config = KalmanPairConfig(Q=Q, R=R, x0=x0, P0=P0)
    n_scored = n_calm - skip

    return CalibrationResult(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        Q=Q,
        R=R,
        x0=x0,
        P0=P0,
        config=config,
        nll=float(fit.fun),
        success=bool(fit.success),
        message=str(fit.message),
        n_obs_scored=n_scored,
        calm_start=calm_start,
        calm_end=calm_end,
        init_window=init_window,
    )


def calibrate_pairs(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    calm_start: int = 0,
    calm_end: int | None = None,
    **kwargs,
) -> list[CalibrationResult]:
    """Calibrate each (y, x) pair; ``kwargs`` forwarded to ``calibrate_pair``."""
    results: list[CalibrationResult] = []
    for y_ticker, x_ticker in pairs:
        y, x = extract_pair(prices, y_ticker, x_ticker)
        results.append(
            calibrate_pair(
                y,
                x,
                y_ticker=y_ticker,
                x_ticker=x_ticker,
                calm_start=calm_start,
                calm_end=calm_end,
                **kwargs,
            )
        )
    return results
