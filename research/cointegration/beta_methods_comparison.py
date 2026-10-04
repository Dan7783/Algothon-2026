#!/usr/bin/env python
"""
Compare hedge-ratio (beta) estimators on stable vs regime-shifting pairs.

Methods:
  1. Raw price-ratio beta     y_t / x_t
  2. Fixed full-sample OLS    constant beta
  3. Rolling OLS              100d window
  4. EWMA OLS                 exponentially weighted regression
  5. Kalman filter            time-varying state-space beta
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research.cointegration.cointegration_output.kalman_vs_rolling_smah_ngte import ROLL_W, kalman_hedge, ols, rolling_hedge

OUTPUT_DIR = Path("cointegration_output/beta_methods")
EWMA_HALFLIFE = 30  # days
KALMAN_DELTA = 1e-4

STABLE_PAIRS = [
    ("AENO", "NWIG"),
    ("HRET", "ULXY"),
]
REGIME_PAIRS = [
    ("SMAH", "NGTE"),
    ("EORC", "NGTE"),
]

COLORS = {
    "raw": "#BBBBBB",
    "fixed": "#C44E52",
    "rolling": "#8172B2",
    "ewma": "#DD8452",
    "kalman": "#4C72B0",
}


def raw_price_ratio_beta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Naive hedge: price ratio (no intercept adjustment)."""
    return np.where(np.abs(x) > 1e-12, y / x, np.nan)


def ewma_ols_beta(
    y: np.ndarray,
    x: np.ndarray,
    halflife: int = EWMA_HALFLIFE,
    min_obs: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal exponentially weighted OLS: y = mu + beta * x.
    At day t, weights on past obs decay with half-life.
    """
    n = len(y)
    lam = 0.5 ** (1.0 / halflife)
    mu = np.full(n, np.nan)
    beta = np.full(n, np.nan)

    for t in range(min_obs, n):
        w = np.array([lam ** (t - i) for i in range(t + 1)], dtype=float)
        w /= w.sum()
        yw, xw = y[: t + 1], x[: t + 1]
        xbar = np.sum(w * xw)
        ybar = np.sum(w * yw)
        xc = xw - xbar
        yc = yw - ybar
        var_x = np.sum(w * xc * xc)
        if var_x < 1e-12:
            continue
        b = np.sum(w * xc * yc) / var_x
        a = ybar - b * xbar
        mu[t], beta[t] = a, b

    return mu, beta


def beta_tracks(y: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """All beta series for one pair."""
    n = len(y)
    _, beta_fixed = ols(y, x)
    _, beta_roll, _, _ = rolling_hedge(y, x, ROLL_W)
    _, beta_ewma = ewma_ols_beta(y, x)
    _, beta_kalman, _, _, _ = kalman_hedge(y, x, delta=KALMAN_DELTA)

    return {
        "raw": raw_price_ratio_beta(y, x),
        "fixed": np.full(n, beta_fixed),
        "rolling": beta_roll,
        "ewma": beta_ewma,
        "kalman": beta_kalman,
    }


def rolling_beta_std(beta: np.ndarray, w: int = 100) -> float:
    """Mean rolling std of beta — lower = more stable."""
    out = []
    for i in range(w, len(beta)):
        sl = beta[i - w : i]
        sl = sl[np.isfinite(sl)]
        if len(sl) > 10:
            out.append(np.std(sl))
    return float(np.mean(out)) if out else np.nan


def plot_pair_row(
    axes_row: list,
    prices: pd.DataFrame,
    a: str,
    b: str,
    title_tag: str,
    shade_regime: tuple[int, int] | None = None,
):
    y = prices[a].values.astype(float)
    x = prices[b].values.astype(float)
    n = len(y)
    t = np.arange(1, n + 1)
    tracks = beta_tracks(y, x)

    ax_p, ax_b = axes_row
    ax_p.plot(t, y / y[0], lw=1.0, label=a, color="#4C72B0")
    ax_p.plot(t, x / x[0], lw=1.0, label=b, color="#DD8452")
    if shade_regime:
        ax_p.axvspan(*shade_regime, color="#888888", alpha=0.12)
    ax_p.set_ylabel("Price / day 1")
    ax_p.set_title(f"{a}–{b}  ({title_tag})")
    ax_p.legend(fontsize=7, loc="upper left")

    ax_b.plot(t, tracks["raw"], color=COLORS["raw"], lw=0.7, alpha=0.55, label="Raw (y/x ratio)")
    ax_b.axhline(
        tracks["fixed"][0],
        color=COLORS["fixed"],
        lw=2.0,
        ls="-",
        label=f"Fixed OLS ({tracks['fixed'][0]:.3f})",
    )
    ax_b.plot(t, tracks["rolling"], color=COLORS["rolling"], lw=1.3, label=f"Rolling OLS ({ROLL_W}d)")
    ax_b.plot(t, tracks["ewma"], color=COLORS["ewma"], lw=1.2, label=f"EWMA OLS (HL={EWMA_HALFLIFE}d)")
    ax_b.plot(t, tracks["kalman"], color=COLORS["kalman"], lw=1.2, label=f"Kalman (δ={KALMAN_DELTA})")
    if shade_regime:
        ax_b.axvspan(*shade_regime, color="#888888", alpha=0.08)

    stab = {k: rolling_beta_std(v) for k, v in tracks.items() if k != "fixed"}
    ax_b.set_ylabel("Hedge ratio β")
    ax_b.set_title(
        f"β estimators — rolling-std stability: "
        f"raw={stab['raw']:.3f}  roll={stab['rolling']:.3f}  "
        f"ewma={stab['ewma']:.3f}  kalman={stab['kalman']:.3f}"
    )
    ax_b.legend(fontsize=6.5, loc="upper left", ncol=2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0)
    n = len(prices)
    t = np.arange(1, n + 1)

    # --- Figure 1: stable pairs (top) vs regime pairs (bottom) ---
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col")
    fig.suptitle(
        "Beta estimation methods: stable pairs vs regime-shifting pairs",
        fontsize=13,
        y=0.98,
    )

    plot_pair_row(
        [axes[0, 0], axes[1, 0]],
        prices,
        *STABLE_PAIRS[0],
        "stable — low β drift",
    )
    plot_pair_row(
        [axes[0, 1], axes[1, 1]],
        prices,
        *STABLE_PAIRS[1],
        "stable — low β drift",
    )
    plot_pair_row(
        [axes[2, 0], axes[3, 0]],
        prices,
        *REGIME_PAIRS[0],
        "regime shift — β & μ drift",
        shade_regime=(280, 560),
    )
    plot_pair_row(
        [axes[2, 1], axes[3, 1]],
        prices,
        *REGIME_PAIRS[1],
        "regime shift — β & μ drift",
        shade_regime=(280, 560),
    )

    for ax in axes[-1]:
        ax.set_xlabel("Day")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = OUTPUT_DIR / "beta_methods_stable_vs_regime.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path1}")

    # --- Figure 2: focused SMAH-NGTE with all methods + early/late reference ---
    y = prices["SMAH"].values.astype(float)
    x = prices["NGTE"].values.astype(float)
    tracks = beta_tracks(y, x)
    _, beta_early = ols(y[:200], x[:200])
    _, beta_late = ols(y[550:700], x[550:700])

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes2[0].plot(t, y / y[0], label="SMAH", color="#4C72B0", lw=1.1)
    axes2[0].plot(t, x / x[0], label="NGTE", color="#DD8452", lw=1.1)
    axes2[0].axvspan(280, 560, color="#888888", alpha=0.12, label="transition")
    axes2[0].axvspan(550, 700, color="#2ca02c", alpha=0.10, label="late regime")
    axes2[0].set_ylabel("Price / day 1")
    axes2[0].set_title("SMAH–NGTE: regime change makes fixed β wrong; adaptive methods diverge")
    axes2[0].legend(fontsize=8)

    ax = axes2[1]
    ax.plot(t, tracks["raw"], color=COLORS["raw"], lw=0.7, alpha=0.5, label="Raw y/x")
    ax.axhline(tracks["fixed"][0], color=COLORS["fixed"], lw=2.2, label=f"Fixed OLS β={tracks['fixed'][0]:.3f}")
    ax.plot(t, tracks["rolling"], color=COLORS["rolling"], lw=1.5, label=f"Rolling {ROLL_W}d")
    ax.plot(t, tracks["ewma"], color=COLORS["ewma"], lw=1.3, label=f"EWMA HL={EWMA_HALFLIFE}d")
    ax.plot(t, tracks["kalman"], color=COLORS["kalman"], lw=1.3, label="Kalman")
    ax.axhline(beta_early, color="#4C72B0", ls="--", lw=0.9, alpha=0.7, label=f"early β≈{beta_early:.2f}")
    ax.axhline(beta_late, color="#2ca02c", ls="--", lw=0.9, alpha=0.7, label=f"late β≈{beta_late:.2f}")
    ax.axvspan(280, 560, color="#888888", alpha=0.08)
    ax.set_ylabel("Hedge ratio β")
    ax.set_xlabel("Day")
    ax.set_title(
        "Fixed OLS is a compromise between regimes; Kalman adapts fastest, rolling lags, raw ratio is noisy"
    )
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)

    plt.tight_layout()
    path2 = OUTPUT_DIR / "beta_methods_SMAH_NGTE_detail.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {path2}")

    # Summary table
    rows = []
    for a, b, tag in [
        *[(p[0], p[1], "stable") for p in STABLE_PAIRS],
        *[(p[0], p[1], "regime") for p in REGIME_PAIRS],
    ]:
        yy = prices[a].values.astype(float)
        xx = prices[b].values.astype(float)
        tr = beta_tracks(yy, xx)
        rows.append(
            {
                "pair": f"{a}-{b}",
                "type": tag,
                "beta_fixed": tr["fixed"][0],
                "std_raw": rolling_beta_std(tr["raw"]),
                "std_rolling": rolling_beta_std(tr["rolling"]),
                "std_ewma": rolling_beta_std(tr["ewma"]),
                "std_kalman": rolling_beta_std(tr["kalman"]),
            }
        )
    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "beta_stability_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
