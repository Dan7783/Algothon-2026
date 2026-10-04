#!/usr/bin/env python
"""Visualise how fixed, rolling OLS, and Kalman filters track SMAH-NGTE beta & mean shifts."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from research.cointegration.cointegration_output.kalman_vs_rolling_smah_ngte import ROLL_W, kalman_hedge, ols, rolling_hedge

OUTPUT_DIR = Path("cointegration_output/activity_vs_binary")
EARLY = (0, 200)
LATE = (550, 700)
REGIME_SHADE = (280, 560)


def local_truth_track(y: np.ndarray, x: np.ndarray, w: int = ROLL_W):
    """Contemporaneous 100d OLS — reference track adaptive filters should follow."""
    n = len(y)
    mu = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    for i in range(w, n):
        mu[i], beta[i] = ols(y[i - w : i], x[i - w : i])
    return mu, beta


def rolling_abs_mean(x: np.ndarray, w: int = 50) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(w, len(x)):
        sl = x[i - w : i]
        sl = sl[np.isfinite(sl)]
        if len(sl):
            out[i] = np.mean(np.abs(sl))
    return out


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0)
    y = prices["SMAH"].values.astype(float)
    x = prices["NGTE"].values.astype(float)
    n = len(y)
    t = np.arange(1, n + 1)

    mu_fix, beta_fix = ols(y, x)
    mu_roll, beta_roll, resid_roll, z_roll = rolling_hedge(y, x, ROLL_W)
    mu_k_slow, beta_k_slow, resid_k_slow, z_k_slow, _ = kalman_hedge(y, x, delta=1e-5)
    mu_k_fast, beta_k_fast, resid_k_fast, z_k_fast, _ = kalman_hedge(y, x, delta=1e-3)
    mu_ref, beta_ref = local_truth_track(y, x, ROLL_W)

    mu_early, beta_early = ols(y[EARLY[0] : EARLY[1]], x[EARLY[0] : EARLY[1]])
    mu_late, beta_late = ols(y[LATE[0] : LATE[1]], x[LATE[0] : LATE[1]])

    spread_fix = y - mu_fix - beta_fix * x

    beta_err = {
        "Fixed": np.abs(beta_fix - beta_ref),
        "Rolling 100d": np.abs(beta_roll - beta_ref),
        "Kalman slow": np.abs(beta_k_slow - beta_ref),
        "Kalman fast": np.abs(beta_k_fast - beta_ref),
    }
    mu_err = {
        "Fixed": np.abs(mu_fix - mu_ref),
        "Rolling 100d": np.abs(mu_roll - mu_ref),
        "Kalman slow": np.abs(mu_k_slow - mu_ref),
        "Kalman fast": np.abs(mu_k_fast - mu_ref),
    }

    colors = {
        "Fixed": "#C44E52",
        "Rolling 100d": "#8172B2",
        "Kalman slow": "#4C72B0",
        "Kalman fast": "#2ca02c",
        "Reference": "#333333",
    }

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.2, 1.2, 1.2, 1.0], hspace=0.32, wspace=0.22)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, y / y[0], color="#4C72B0", lw=1.2, label="SMAH")
    ax0.plot(t, x / x[0], color="#DD8452", lw=1.2, label="NGTE")
    ax0.axvspan(*REGIME_SHADE, color="#888888", alpha=0.12, label="transition / divergence")
    ax0.axvspan(LATE[0], LATE[1], color="#2ca02c", alpha=0.10, label="late regime")
    ax0.set_ylabel("Price / day 1")
    ax0.set_title("SMAH–NGTE: filter fit to shifting hedge ratio (β) and intercept (μ)")
    ax0.legend(fontsize=8, loc="upper left", ncol=4)

    ax_b = fig.add_subplot(gs[1, :])
    ax_b.axhline(beta_fix, color=colors["Fixed"], ls="-", lw=2.0, label=f"Fixed β={beta_fix:.3f}")
    ax_b.plot(t, beta_roll, color=colors["Rolling 100d"], lw=1.4, label="Rolling 100d β")
    ax_b.plot(t, beta_k_slow, color=colors["Kalman slow"], lw=1.2, alpha=0.9, label="Kalman slow β")
    ax_b.plot(t, beta_k_fast, color=colors["Kalman fast"], lw=1.2, alpha=0.9, label="Kalman fast β")
    ax_b.plot(t, beta_ref, color=colors["Reference"], lw=1.0, ls=":", alpha=0.7, label="100d reference β")
    ax_b.axhline(beta_early, color="#4C72B0", ls="--", lw=0.9, alpha=0.6)
    ax_b.axhline(beta_late, color="#2ca02c", ls="--", lw=0.9, alpha=0.6)
    ax_b.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_b.set_ylabel("Hedge ratio β")
    ax_b.set_title(
        f"Beta tracking — early regime β≈{beta_early:.2f}, late regime β≈{beta_late:.2f}; "
        "fixed hedge is a single wrong compromise"
    )
    ax_b.legend(fontsize=7.5, loc="upper left", ncol=3)

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.axhline(mu_fix, color=colors["Fixed"], ls="-", lw=2.0, label=f"Fixed μ={mu_fix:.1f}")
    ax_m.plot(t, mu_roll, color=colors["Rolling 100d"], lw=1.4, label="Rolling 100d μ")
    ax_m.plot(t, mu_k_slow, color=colors["Kalman slow"], lw=1.2, alpha=0.9, label="Kalman slow μ")
    ax_m.plot(t, mu_k_fast, color=colors["Kalman fast"], lw=1.2, alpha=0.9, label="Kalman fast μ")
    ax_m.plot(t, mu_ref, color=colors["Reference"], lw=1.0, ls=":", alpha=0.7, label="100d reference μ")
    ax_m.axhline(mu_early, color="#4C72B0", ls="--", lw=0.9, alpha=0.6)
    ax_m.axhline(mu_late, color="#2ca02c", ls="--", lw=0.9, alpha=0.6)
    ax_m.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_m.set_ylabel("Intercept μ")
    ax_m.set_title(
        f"Mean/intercept tracking — early μ≈{mu_early:.1f}, late μ≈{mu_late:.1f}; "
        "rolling μ spikes during transition; Kalman fast keeps μ stable"
    )
    ax_m.legend(fontsize=7.5, loc="upper left", ncol=3)

    ax_r1 = fig.add_subplot(gs[3, 0])
    ax_r1.plot(t, spread_fix, color=colors["Fixed"], lw=1.0)
    ax_r1.axhline(0, color="gray", lw=0.6)
    ax_r1.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_r1.set_ylabel("Residual")
    ax_r1.set_title("Fixed — spread permanently offset")
    ax_r1.set_ylim(-35, 40)

    ax_r2 = fig.add_subplot(gs[3, 1])
    ax_r2.plot(t, resid_roll, color=colors["Rolling 100d"], lw=1.0, label="Rolling")
    ax_r2.plot(t, resid_k_slow, color=colors["Kalman slow"], lw=0.9, alpha=0.75, label="Kalman slow")
    ax_r2.plot(t, resid_k_fast, color=colors["Kalman fast"], lw=0.9, alpha=0.75, label="Kalman fast")
    ax_r2.axhline(0, color="gray", lw=0.6)
    ax_r2.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_r2.set_ylabel("Innovation / residual")
    ax_r2.set_title("Adaptive filters — residuals stay centred")
    ax_r2.legend(fontsize=7)

    ax_e1 = fig.add_subplot(gs[4, 0])
    for name, err in beta_err.items():
        ax_e1.plot(t, err, color=colors[name], lw=1.1, label=name)
    ax_e1.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_e1.set_ylabel("|β − reference|")
    ax_e1.set_xlabel("Day")
    ax_e1.set_title("Beta fit error vs contemporaneous 100d OLS")
    ax_e1.legend(fontsize=7)

    ax_e2 = fig.add_subplot(gs[4, 1])
    for name, err in mu_err.items():
        ax_e2.plot(t, err, color=colors[name], lw=1.1, label=name)
    ax_e2.axvspan(*REGIME_SHADE, color="#888888", alpha=0.08)
    ax_e2.set_ylabel("|μ − reference|")
    ax_e2.set_xlabel("Day")
    ax_e2.set_title("Intercept fit error vs contemporaneous 100d OLS")
    ax_e2.legend(fontsize=7)

    early_patch = mpatches.Patch(color="#4C72B0", alpha=0.4, label="early regime anchor")
    late_patch = mpatches.Patch(color="#2ca02c", alpha=0.4, label="late regime anchor")
    fig.legend(handles=[early_patch, late_patch], loc="upper right", fontsize=8, framealpha=0.9)

    path = OUTPUT_DIR / "SMAH_NGTE_filter_fit_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Summary table for transition window
    trans = slice(REGIME_SHADE[0], REGIME_SHADE[1])
    print("\nMean tracking error in transition window (days 280–560):")
    for name in beta_err:
        be = beta_err[name][trans]
        me = mu_err[name][trans]
        be, me = be[np.isfinite(be)], me[np.isfinite(me)]
        print(f"  {name:14s}  |beta err|={be.mean():.3f}  |mu err|={me.mean():.1f}")


if __name__ == "__main__":
    main()
