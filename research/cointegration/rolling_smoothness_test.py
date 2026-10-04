#!/usr/bin/env python
"""
Is rolling cointegration strength smooth (tradable) or abrupt (not)?

For top pairs: rolling 50-day Engle-Granger strength, smoothness metrics,
and whether strength predicts forward spread mean-reversion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

LOOKBACK = 50
FORWARD = 50
STEP = 5  # overlapping windows every 5 days
SIGNIFICANCE = 0.05
OUTPUT_DIR = Path("cointegration_output/rolling_smoothness")

# Top pairs from walk-forward / persistence analysis
PAIRS = [
    ("ELLT", "ILVX"),
    ("AMRP", "HRET"),
    ("GARI", "IHOZ"),
    ("AENO", "RRES"),
    ("SRTX", "EAFC"),
    ("AENO", "NWIG"),
    ("SMAH", "CTGI"),
    ("RTTH", "NGTE"),
    ("EORC", "NGTE"),
    ("HRET", "ULXY"),
]


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    x_ = np.column_stack([np.ones(len(x)), x])
    beta, _ = np.linalg.lstsq(x_, y, rcond=None)[0]
    return float(beta)


def rolling_strength(prices: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Rolling lookback cointegration strength + forward OOS PnL proxy."""
    y_all = prices[a].values
    x_all = prices[b].values
    n = len(prices)
    rows = []

    for t in range(LOOKBACK, n - FORWARD + 1, STEP):
        y_lb = y_all[t - LOOKBACK : t]
        x_lb = x_all[t - LOOKBACK : t]
        y_fw = y_all[t : t + FORWARD]
        x_fw = x_all[t : t + FORWARD]

        if np.std(y_lb) < 1e-12 or np.std(x_lb) < 1e-12:
            continue

        _, pval, _ = coint(y_lb, x_lb)
        strength = -np.log10(max(pval, 1e-16))
        beta = hedge_ratio(y_lb, x_lb)

        lb_spread = y_lb - beta * x_lb
        mu, sigma = lb_spread.mean(), lb_spread.std()
        fw_spread = y_fw - beta * x_fw

        # Forward: does spread revert toward lookback mean?
        if sigma < 1e-12:
            fwd_reversion = 0.0
            oos_pnl = 0.0
        else:
            z0 = (fw_spread[0] - mu) / sigma
            z_end = (fw_spread[-1] - mu) / sigma
            # Positive if |z| shrinks toward 0
            fwd_reversion = abs(z0) - abs(z_end)

            # Simple threshold MR on OOS z
            z = (fw_spread - mu) / sigma
            pos, entry, pnl = 0.0, 0.0, 0.0
            for i in range(1, len(z)):
                if pos == 0:
                    if z[i - 1] > 2:
                        pos, entry = -1.0, fw_spread[i - 1]
                    elif z[i - 1] < -2:
                        pos, entry = 1.0, fw_spread[i - 1]
                elif (pos > 0 and z[i - 1] >= 0) or (pos < 0 and z[i - 1] <= 0):
                    pnl += pos * (fw_spread[i - 1] - entry)
                    pos = 0.0
            if pos != 0:
                pnl += pos * (fw_spread[-1] - entry)
            oos_pnl = pnl

        rows.append(
            {
                "day": t,
                "pval": pval,
                "strength": strength,
                "sig": pval < SIGNIFICANCE,
                "fwd_reversion": fwd_reversion,
                "oos_pnl": oos_pnl,
                "profitable": oos_pnl > 0,
            }
        )

    return pd.DataFrame(rows)


def smoothness_metrics(strength: np.ndarray) -> dict:
    """High lag-1 autocorr + low absolute first-diff = smooth decay/growth."""
    s = strength[~np.isnan(strength)]
    if len(s) < 5:
        return {"acf1": np.nan, "mean_abs_diff": np.nan, "smooth_score": np.nan}

    s_z = (s - s.mean()) / (s.std() + 1e-12)
    acf1 = float(np.corrcoef(s_z[:-1], s_z[1:])[0, 1])
    mad = float(np.mean(np.abs(np.diff(s))))
    # Smooth if high persistence and small jumps relative to range
    rng = float(s.max() - s.min()) + 1e-12
    smooth_score = acf1 * (1.0 - min(mad / rng, 1.0))
    return {"acf1": acf1, "mean_abs_diff": mad, "smooth_score": smooth_score}


def strength_predicts_forward(df: pd.DataFrame) -> dict:
    """Does higher lookback strength predict better forward outcomes?"""
    if len(df) < 10:
        return {"corr_strength_reversion": np.nan, "corr_strength_pnl": np.nan, "hit_when_strong": np.nan}

    corr_rev = float(np.corrcoef(df["strength"], df["fwd_reversion"])[0, 1])
    corr_pnl = float(np.corrcoef(df["strength"], df["oos_pnl"])[0, 1])
    strong = df[df["strength"] >= -np.log10(SIGNIFICANCE)]  # p < 0.05
    hit = float(strong["profitable"].mean()) if len(strong) else np.nan
    return {
        "corr_strength_reversion": corr_rev,
        "corr_strength_pnl": corr_pnl,
        "hit_when_strong": hit,
        "n_strong": int(len(strong)),
    }


def plot_pair_rolling(df: pd.DataFrame, a: str, b: str, metrics: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["day"], df["strength"], color="#4C72B0", lw=1.5)
    axes[0].axhline(-np.log10(SIGNIFICANCE), color="red", ls="--", lw=1, label="p=0.05")
    axes[0].fill_between(
        df["day"],
        0,
        df["strength"],
        where=df["sig"],
        color="#55A868",
        alpha=0.25,
        label="significant",
    )
    axes[0].set_ylabel("Strength\n(-log10 p)")
    axes[0].set_title(
        f"{a}-{b} | rolling {LOOKBACK}d lookback (step {STEP}) | "
        f"ACF1={metrics['acf1']:.2f}, smooth={metrics['smooth_score']:.2f}"
    )
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].bar(df["day"], df["fwd_reversion"], width=STEP * 0.8, color="#8172B2", alpha=0.8)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Forward |z|\nshrinkage")

    colors = ["#55A868" if p else "#C44E52" for p in df["profitable"]]
    axes[2].bar(df["day"], df["oos_pnl"], width=STEP * 0.8, color=colors, alpha=0.85)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_ylabel("OOS MR PnL")
    axes[2].set_xlabel(f"Day (end of {LOOKBACK}-day lookback)")

    plt.tight_layout()
    path = out_dir / f"rolling_{a}_{b}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    order = summary.sort_values("smooth_score", ascending=True)
    colors = plt.cm.RdYlGn((order["smooth_score"] - order["smooth_score"].min()) /
                           (order["smooth_score"].max() - order["smooth_score"].min() + 1e-12))
    axes[0].barh(order["pair"], order["smooth_score"], color=colors, edgecolor="white")
    axes[0].set_xlabel("Smoothness score (ACF1 × (1 - jumpiness))")
    axes[0].set_title("How smooth is rolling strength?")
    axes[0].axvline(0.5, color="gray", ls="--", lw=1)

    axes[1].scatter(summary["smooth_score"], summary["corr_strength_pnl"], s=60, c="#4C72B0")
    for _, r in summary.iterrows():
        axes[1].annotate(r["pair"], (r["smooth_score"], r["corr_strength_pnl"]), fontsize=7, alpha=0.8)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Smoothness")
    axes[1].set_ylabel("Corr(strength, OOS PnL)")
    axes[1].set_title("Smooth ≠ predictive unless corr > 0")

    axes[2].bar(summary["pair"], summary["hit_when_strong"] * 100, color="#CCB974", edgecolor="white")
    axes[2].axhline(50, color="gray", ls="--", lw=1)
    axes[2].set_xticklabels(summary["pair"], rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("OOS profitable when p<0.05 (%)")
    axes[2].set_title("Hit rate when signal is 'on'")

    fig.suptitle(
        "Tradability check: smooth decay of rolling cointegration strength",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "smoothness_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days | rolling lookback={LOOKBACK}, step={STEP}, forward={FORWARD}")

    rows = []
    for a, b in PAIRS:
        df = rolling_strength(prices, a, b)
        sm = smoothness_metrics(df["strength"].values)
        pred = strength_predicts_forward(df)
        plot_pair_rolling(df, a, b, sm, OUTPUT_DIR)
        row = {"pair": f"{a}-{b}", **sm, **pred, "n_windows": len(df), "pct_sig": 100 * df["sig"].mean()}
        rows.append(row)
        print(
            f"{a}-{b}: ACF1={sm['acf1']:.2f} smooth={sm['smooth_score']:.2f} "
            f"corr(pnl)={pred['corr_strength_pnl']:.2f} hit={pred['hit_when_strong']*100:.0f}% "
            f"sig={row['pct_sig']:.0f}%"
        )
        df.to_csv(OUTPUT_DIR / f"rolling_{a}_{b}.csv", index=False)

    summary = pd.DataFrame(rows).sort_values("smooth_score", ascending=False)
    summary.to_csv(OUTPUT_DIR / "smoothness_summary.csv", index=False)
    plot_summary(summary, OUTPUT_DIR)

    print("\n=== Verdict helpers ===")
    print(summary[["pair", "acf1", "smooth_score", "corr_strength_pnl", "hit_when_strong", "pct_sig"]].to_string(index=False))

    mean_smooth = summary["smooth_score"].mean()
    mean_corr = summary["corr_strength_pnl"].mean()
    mean_hit = summary["hit_when_strong"].mean()
    print(f"\nMean smoothness={mean_smooth:.2f} | mean corr(strength,pnl)={mean_corr:.2f} | mean hit={mean_hit*100:.0f}%")
    if mean_smooth >= 0.5 and mean_corr > 0.15 and mean_hit > 0.55:
        print("VERDICT: strength looks smooth AND weakly predictive -> candidate signal (still need costs/size).")
    elif mean_smooth >= 0.5 and mean_corr <= 0.15:
        print("VERDICT: strength can look smooth, but does NOT predict forward PnL -> not tradable as-is.")
    else:
        print("VERDICT: strength is jumpy / weakly persistent -> not a stable fading regime signal.")


if __name__ == "__main__":
    main()
