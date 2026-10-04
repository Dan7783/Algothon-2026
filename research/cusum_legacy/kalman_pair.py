"""Fixed 2-state Kalman filter for pairs regression (alpha, beta).

State-space model for pair (y, x):

    y_t = alpha_t + beta_t * x_t + eps_t,          eps_t ~ N(0, R)
    [alpha_t, beta_t]^T = [alpha_{t-1}, beta_{t-1}]^T + w_t,
                                                    w_t ~ N(0, Q)

Observation matrix H_t = [1, x_t] is time-varying.  Predict/update are
closed-form 2x2 operations (not a generic dim_x implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class KalmanPairConfig:
    """Frozen filter configuration for one pair.

    Shapes (enforced in ``__post_init__``):
      - Q:  (2, 2) process-noise covariance
      - R:  (1, 1) observation-noise variance (scalar wrapped as 1x1)
      - x0: (2,)   initial state [alpha, beta]
      - P0: (2, 2) initial state covariance
    """

    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray

    def __post_init__(self) -> None:
        Q = np.asarray(self.Q, dtype=float)
        R = np.asarray(self.R, dtype=float)
        x0 = np.asarray(self.x0, dtype=float)
        P0 = np.asarray(self.P0, dtype=float)

        if R.ndim == 0:
            R = R.reshape(1, 1)
        elif R.shape == (1,):
            R = R.reshape(1, 1)

        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "P0", P0)

        if self.Q.shape != (2, 2):
            raise ValueError(f"Q must be shape (2, 2), got {self.Q.shape}")
        if self.R.shape != (1, 1):
            raise ValueError(f"R must be shape (1, 1), got {self.R.shape}")
        if self.x0.shape != (2,):
            raise ValueError(f"x0 must be shape (2,), got {self.x0.shape}")
        if self.P0.shape != (2, 2):
            raise ValueError(f"P0 must be shape (2, 2), got {self.P0.shape}")
        if float(self.R[0, 0]) <= 0.0:
            raise ValueError("R must be strictly positive")

    @property
    def R_scalar(self) -> float:
        return float(self.R[0, 0])


class KalmanPairResult(NamedTuple):
    """Per-timestep filter outputs.

    ``innovations``, ``innov_var``, and ``z`` are plain 1-d numpy arrays
    (length T).  Downstream CUSUM / drift tests should consume ``z`` directly.
    """

    alpha: np.ndarray
    beta: np.ndarray
    innovations: np.ndarray  # y_k  (one-step prediction error)
    innov_var: np.ndarray  # S_k
    z: np.ndarray  # y_k / sqrt(S_k)


def _joseph_update(
    P_pred: np.ndarray,
    K0: float,
    K1: float,
    h0: float,
    h1: float,
    R: float,
) -> np.ndarray:
    """Joseph-form covariance update for 2-state, scalar observation."""
    # I - K H
    a00 = 1.0 - K0 * h0
    a01 = -K0 * h1
    a10 = -K1 * h0
    a11 = 1.0 - K1 * h1

    p00, p01 = P_pred[0, 0], P_pred[0, 1]
    p10, p11 = P_pred[1, 0], P_pred[1, 1]

    # M = (I - KH) P_pred
    m00 = a00 * p00 + a01 * p10
    m01 = a00 * p01 + a01 * p11
    m10 = a10 * p00 + a11 * p10
    m11 = a10 * p01 + a11 * p11

    # P = M (I - KH)^T + K R K^T
    out00 = m00 * a00 + m01 * a01 + K0 * R * K0
    out01 = m00 * a10 + m01 * a11 + K0 * R * K1
    out10 = m10 * a00 + m11 * a01 + K1 * R * K0
    out11 = m10 * a10 + m11 * a11 + K1 * R * K1

    # Symmetrize
    mid = 0.5 * (out01 + out10)
    return np.array([[out00, mid], [mid, out11]], dtype=float)


def run_kalman_pair(
    y: np.ndarray,
    x: np.ndarray,
    config: KalmanPairConfig,
    *,
    min_innov_var: float = 1e-12,
) -> KalmanPairResult:
    """Run the fixed 2-state pairs Kalman filter over aligned series y, x.

    Parameters
    ----------
    y, x:
        Price series of equal length T.
    config:
        Frozen Q, R, x0, P0.
    min_innov_var:
        Floor on S_k to avoid division by zero (default 1e-12).

    Returns
    -------
    KalmanPairResult with alpha/beta filtered paths and innovation series.
    Innovation at t=0 uses the prior (x0, P0) before any update.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError(f"y and x length mismatch: {y.shape} vs {x.shape}")

    T = y.shape[0]
    Q = config.Q
    R = config.R_scalar
    q00, q01 = Q[0, 0], Q[0, 1]
    q10, q11 = Q[1, 0], Q[1, 1]

    alpha_hat = np.empty(T, dtype=float)
    beta_hat = np.empty(T, dtype=float)
    innovations = np.empty(T, dtype=float)
    innov_var = np.empty(T, dtype=float)
    z = np.empty(T, dtype=float)

    # Filtered state / covariance entering the loop (prior for t=0)
    a = float(config.x0[0])
    b = float(config.x0[1])
    p00 = float(config.P0[0, 0])
    p01 = float(config.P0[0, 1])
    p10 = float(config.P0[1, 0])
    p11 = float(config.P0[1, 1])

    for t in range(T):
        # ---- predict (identity transition) ----
        pp00 = p00 + q00
        pp01 = p01 + q01
        pp10 = p10 + q10
        pp11 = p11 + q11
        # symmetrize predicted P
        mid = 0.5 * (pp01 + pp10)
        pp01 = mid
        pp10 = mid

        # H_t = [1, x_t]
        xt = float(x[t])
        h0 = 1.0
        h1 = xt

        # innovation
        yhat = a + b * xt
        yk = float(y[t]) - yhat
        # S = H P H' + R  (scalar)
        Sk = (
            h0 * (h0 * pp00 + h1 * pp10)
            + h1 * (h0 * pp01 + h1 * pp11)
            + R
        )
        if Sk < min_innov_var:
            Sk = min_innov_var

        # Kalman gain K = P H' / S  (2,)
        k0 = (pp00 * h0 + pp01 * h1) / Sk
        k1 = (pp10 * h0 + pp11 * h1) / Sk

        # update state
        a = a + k0 * yk
        b = b + k1 * yk

        # Joseph covariance update
        P_pred = np.array([[pp00, pp01], [pp10, pp11]], dtype=float)
        P_new = _joseph_update(P_pred, k0, k1, h0, h1, R)
        p00, p01 = P_new[0, 0], P_new[0, 1]
        p10, p11 = P_new[1, 0], P_new[1, 1]

        alpha_hat[t] = a
        beta_hat[t] = b
        innovations[t] = yk
        innov_var[t] = Sk
        z[t] = yk / np.sqrt(Sk)

    return KalmanPairResult(
        alpha=alpha_hat,
        beta=beta_hat,
        innovations=innovations,
        innov_var=innov_var,
        z=z,
    )


def ols_init(
    y: np.ndarray,
    x: np.ndarray,
    *,
    init_window: int = 20,
    p0_scale: float = 1.0,
    p0_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS initial state/covariance and residual variance from the first window.

    Parameters
    ----------
    init_window:
        Number of leading observations used for the OLS warm-start (default 20).
    p0_scale:
        Multiplier on the OLS parameter covariance (default 1.0).
    p0_floor:
        Diagonal ridge added to P0 for numerical stability (default 1e-8).

    Returns
    -------
    x0 : (2,) [alpha, beta]
    P0 : (2, 2)
    R0 : scalar residual variance (for seeding MLE)
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = min(int(init_window), len(y))
    if n < 2:
        raise ValueError("init_window must be at least 2")

    Z = np.column_stack([np.ones(n), x[:n]])
    coef, _, _, _ = np.linalg.lstsq(Z, y[:n], rcond=None)
    resid = y[:n] - Z @ coef
    R0 = float(np.var(resid, ddof=2)) if n > 2 else float(np.var(resid))
    R0 = max(R0, 1e-10)

    # Cov(beta_hat) ≈ sigma^2 (Z'Z)^{-1}
    try:
        P0 = R0 * np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        P0 = R0 * np.linalg.pinv(Z.T @ Z)
    P0 = p0_scale * P0 + np.eye(2) * p0_floor
    P0 = 0.5 * (P0 + P0.T)

    return coef.astype(float), P0.astype(float), R0
