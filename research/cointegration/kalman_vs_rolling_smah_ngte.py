#!/usr/bin/env python
"""
Kalman vs rolling-window hedge for SMAH-NGTE.

State: [mu_t, beta_t]'  (intercept + hedge)
Obs:   SMAH_t = mu_t + beta_t * NGTE_t + e_t

Shows Kalman adapts faster to regime shifts than a 100d rolling OLS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("cointegration_output/activity_vs_binary")
ROLL_W = 100


def ols(y, x):
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(mu), float(beta)


def rolling_hedge(y: np.ndarray, x: np.ndarray, w: int = ROLL_W):
    n = len(y)
    mu = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for i in range(w, n):
        m, b = ols(y[i - w : i], x[i - w : i])
        mu[i], beta[i] = m, b
        train = y[i - w : i] - m - b * x[i - w : i]
        resid[i] = y[i] - m - b * x[i]
        sig = train.std()
        z[i] = resid[i] / sig if sig > 1e-12 else 0.0
    return mu, beta, resid, z


def kalman_hedge(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-4,
    ve: float = None,
):
    """
    Time-varying regression via Kalman filter.
    delta: state noise scale (larger => faster adaptation, noisier beta)
    ve: observation noise variance (None => estimate from early residuals)
    """
    n = len(y)
    # state transition: identity (random walk)
    # process cov
    R = delta / (1 - delta) * np.eye(2) if delta < 1 else np.eye(2) * 1e-3
    # Actually classic pairs Kalman uses: Q = delta/(1-delta) * I for discounting
    Q = (delta / (1.0 - delta)) * np.eye(2)

    # init
    theta = np.zeros(2)  # [mu, beta]
    P = np.eye(2)
    if ve is None:
        # rough scale from first diffs
        ve = float(np.var(np.diff(y) - np.diff(x)))
        ve = max(ve, 1e-6)

    mu = np.zeros(n)
    beta = np.zeros(n)
    resid = np.zeros(n)
    z = np.zeros(n)
    innov_std = np.zeros(n)
    # EWMA of residual var for z-score
    ewma_var = ve
    alpha_ewma = 0.05

    for t in range(n):
        # predict
        P = P + Q
        # observation matrix
        H = np.array([1.0, x[t]])
        # prediction of y
        yhat = H @ theta
        # innovation
        e = y[t] - yhat
        S = H @ P @ H.T + ve
        K = (P @ H.T) / S
        theta = theta + K * e
        P = P - np.outer(K, H) @ P

        mu[t], beta[t] = theta[0], theta[1]
        resid[t] = e  # one-step-ahead residual (pre-update) is more OOS;
        # use post-fit residual for plotting level; for trading use innov
        innov_std[t] = np.sqrt(max(S, 1e-12))
        ewma_var = (1 - alpha_ewma) * ewma_var + alpha_ewma * (e**2)
        z[t] = e / np.sqrt(max(ewma_var, 1e-12))

    return mu, beta, resid, z, innov_std


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0)
    a, b = "SMAH", "NGTE"
    y = prices[a].values.astype(float)
    x = prices[b].values.astype(float)
    n = len(y)
    t = np.arange(1, n + 1)

    mu_roll, beta_roll, resid_roll, z_roll = rolling_hedge(y, x, ROLL_W)

    # Two Kalman speeds
    mu_slow, beta_slow, e_slow, z_slow, _ = kalman_hedge(y, x, delta=1e-5)
    mu_fast, beta_fast, e_fast, z_fast, _ = kalman_hedge(y, x, delta=1e-3)

    # Static early/late references
    _, beta_early = ols(y[:200], x[:200])
    _, beta_late = ols(y[550:700], x[550:700])

    # When does each method reach halfway from early beta to late beta?
    target = 0.5 * (beta_early + beta_late)
    # search after day 500
    def first_cross(series, after=500):
        for i in range(after, n):
            if np.isfinite(series[i]) and series[i] >= target:
                return i + 1  # day
        return None

    print(f"Early beta={beta_early:.3f}  Late beta={beta_late:.3f}  midway={target:.3f}")
    print(f"Rolling {ROLL_W}d reaches midway at day: {first_cross(beta_roll)}")
    print(f"Kalman slow (d=1e-5) midway day:      {first_cross(beta_slow)}")
    print(f"Kalman fast (d=1e-3) midway day:      {first_cross(beta_fast)}")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(t, y / y[0], label="SMAH", lw=1.1)
    axes[0].plot(t, x / x[0], label="NGTE", lw=1.1)
    axes[0].axvspan(550, 700, color="#2ca02c", alpha=0.1)
    axes[0].set_ylabel("Price / day1")
    axes[0].set_title("SMAH-NGTE: Kalman vs rolling window for regime adaptation")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(t, beta_roll, color="#999999", lw=1.5, label=f"Rolling OLS {ROLL_W}d")
    axes[1].plot(t, beta_slow, color="#4C72B0", lw=1.2, label="Kalman slow (delta=1e-5)")
    axes[1].plot(t, beta_fast, color="#C44E52", lw=1.2, label="Kalman fast (delta=1e-3)")
    axes[1].axhline(beta_early, color="#4C72B0", ls="--", lw=0.8, alpha=0.7)
    axes[1].axhline(beta_late, color="#2ca02c", ls="--", lw=0.8, alpha=0.7)
    axes[1].axhline(target, color="orange", ls=":", lw=1, label="midway early→late")
    axes[1].set_ylabel("Hedge beta")
    axes[1].set_title("Beta adaptation speed — Kalman can react in days; rolling needs ~window/2")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, mu_roll, color="#999999", lw=1.3, label="Rolling mu")
    axes[2].plot(t, mu_fast, color="#C44E52", lw=1.2, label="Kalman fast mu")
    axes[2].set_ylabel("Intercept mu")
    axes[2].set_title("Mean/intercept shift — this is what made the old spread look permanently offset")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, z_roll, color="#999999", lw=1, alpha=0.8, label="Rolling z")
    axes[3].plot(t, z_fast, color="#C44E52", lw=1, alpha=0.9, label="Kalman innov z")
    axes[3].axhline(2, color="red", ls="--", lw=0.8)
    axes[3].axhline(-2, color="red", ls="--", lw=0.8)
    axes[3].axhline(0, color="gray", lw=0.6)
    axes[3].set_ylabel("z-score")
    axes[3].set_xlabel("Day")
    axes[3].set_ylim(-5, 5)
    axes[3].set_title("Trade Kalman one-step innovations (z), not a stale rolling residual mean")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / "SMAH_NGTE_kalman_vs_rolling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Adaptation lag summary bar
    days = {
        f"Rolling {ROLL_W}d": first_cross(beta_roll),
        "Kalman slow": first_cross(beta_slow),
        "Kalman fast": first_cross(beta_fast),
    }
    # lag after day 550 (visual regime)
    lag_after = {k: (v - 550 if v else None) for k, v in days.items()}
    print("Days after ~550 until beta crosses midway:")
    for k, v in lag_after.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
