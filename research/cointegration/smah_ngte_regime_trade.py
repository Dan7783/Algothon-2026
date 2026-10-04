#!/usr/bin/env python
"""SMAH-NGTE: how to trade through a spread-mean regime change."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("cointegration_output/activity_vs_binary")
W = 100


def ols(y, x):
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(mu), float(beta)


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0)
    a, b = "SMAH", "NGTE"
    y = prices[a].values.astype(float)
    x = prices[b].values.astype(float)
    n = len(y)
    t = np.arange(1, n + 1)

    mu_full, beta_full = ols(y, x)
    spread_full = y - mu_full - beta_full * x

    mu_r = np.full(n, np.nan)
    beta_r = np.full(n, np.nan)
    spread_r = np.full(n, np.nan)
    z_r = np.full(n, np.nan)
    for i in range(W, n):
        mu, beta = ols(y[i - W : i], x[i - W : i])
        mu_r[i], beta_r[i] = mu, beta
        sp = y[i] - mu - beta * x[i]
        spread_r[i] = sp
        train = y[i - W : i] - mu - beta * x[i - W : i]
        sig = train.std()
        z_r[i] = sp / sig if sig > 1e-12 else 0.0

    mu1, beta1 = ols(y[:200], x[:200])
    mu2, beta2 = ols(y[550:700], x[550:700])
    print(f"Regime1 (d1-200):   mu={mu1:.2f}  beta={beta1:.3f}")
    print(f"Regime2 (d550-700): mu={mu2:.2f}  beta={beta2:.3f}")
    print(f"Full sample:        mu={mu_full:.2f}  beta={beta_full:.3f}")
    print(f"Fixed-spread mean early={spread_full[:200].mean():.2f} late={spread_full[550:700].mean():.2f}")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(t, y / y[0], label="SMAH", lw=1.2)
    axes[0].plot(t, x / x[0], label="NGTE", lw=1.2)
    axes[0].axvspan(100, 150, color="#4C72B0", alpha=0.12, label="early active patch")
    axes[0].axvspan(550, 700, color="#2ca02c", alpha=0.12, label="late active regime")
    axes[0].set_ylabel("Price / day1")
    axes[0].set_title("SMAH-NGTE: trading when the cointegrating mean/hedge shifts")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(t, spread_full, color="#C44E52", lw=1)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axhline(spread_full[:200].mean(), color="#4C72B0", ls="--", lw=1, label="early mean")
    axes[1].axhline(spread_full[550:700].mean(), color="#2ca02c", ls="--", lw=1, label="late mean")
    axes[1].set_ylabel("Spread\n(full-sample hedge)")
    axes[1].set_title("Do NOT trade this: fixed hedge makes the mean look permanently shifted after the regime change")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, spread_r, color="#8172B2", lw=1)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_ylabel("Residual\n(rolling 100d hedge)")
    axes[2].set_title("Re-estimate mu & beta every day on last 100d — residual stays centered near 0 in each regime")

    axes[3].plot(t, z_r, color="#4C72B0", lw=1)
    axes[3].axhline(0, color="gray", lw=0.8)
    axes[3].axhline(2, color="red", ls="--", lw=1)
    axes[3].axhline(-2, color="red", ls="--", lw=1, label="entry |z|=2")
    axes[3].axhline(0.5, color="orange", ls=":", lw=1)
    axes[3].axhline(-0.5, color="orange", ls=":", lw=1, label="exit |z|=0.5")
    axes[3].set_ylabel("z-score")
    axes[3].set_xlabel("Day")
    axes[3].set_title("Trade rolling z only while pair is active; flatten if strength dies or residual stops reverting")
    axes[3].legend(fontsize=8)
    axes[3].set_ylim(-4, 4)

    plt.tight_layout()
    path = OUTPUT_DIR / "SMAH_NGTE_regime_trade_guide.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, beta_r, color="#C44E52", lw=1.2)
    ax.axhline(beta1, color="#4C72B0", ls="--", label=f"early beta={beta1:.2f}")
    ax.axhline(beta2, color="#2ca02c", ls="--", label=f"late beta={beta2:.2f}")
    ax.set_ylabel("Rolling hedge beta")
    ax.set_xlabel("Day")
    ax.set_title("SMAH-NGTE rolling beta (100d) — hedge ratio changes across regimes")
    ax.legend()
    plt.tight_layout()
    path2 = OUTPUT_DIR / "SMAH_NGTE_rolling_beta.png"
    fig.savefig(path2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
