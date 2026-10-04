#!/usr/bin/env python
"""
Minimum rolling-window size for Engle-Granger cointegration detection.

Uses properties of prices.txt:
  1) Empirical half-lives of spreads when a long window finds cointegration
  2) Monte Carlo SIZE under the null (independent random walks matched to vol)
  3) Monte Carlo POWER under the alternative (cointegrated with data-calibrated HL)

Recommend smallest T where size ~5% and power is usable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

OUTPUT_DIR = Path("cointegration_output/window_size_study")
WINDOW_SIZES = [20, 30, 40, 50, 60, 75, 100, 125, 150]
N_MC = 400  # simulations per window size
SIGNIFICANCE = 0.05
LONG_WINDOW = 100  # used to harvest empirical half-lives
RNG = np.random.default_rng(42)


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    x_ = np.column_stack([np.ones(len(x)), x])
    beta, _ = np.linalg.lstsq(x_, y, rcond=None)[0]
    return float(beta)


def half_life(spread: np.ndarray) -> float:
    """AR(1) half-life: spread_t = c + rho * spread_{t-1} + e; HL = -ln2 / ln(rho)."""
    if len(spread) < 10 or np.std(spread) < 1e-12:
        return np.nan
    lag = spread[:-1]
    cur = spread[1:]
    x_ = np.column_stack([np.ones(len(lag)), lag])
    _, rho = np.linalg.lstsq(x_, cur, rcond=None)[0]
    if not (0 < rho < 1):
        return np.nan
    hl = -np.log(2) / np.log(rho)
    return float(hl) if 0.25 < hl < 200 else np.nan


def daily_vol(prices: pd.DataFrame) -> np.ndarray:
    rets = prices.pct_change().dropna()
    return rets.std().values


def harvest_empirical_half_lives(prices: pd.DataFrame) -> pd.DataFrame:
    """When LONG_WINDOW EG is significant, record spread half-life."""
    tickers = prices.columns.tolist()
    rows = []
    n = len(prices)
    for start in range(0, n - LONG_WINDOW + 1, LONG_WINDOW // 2):
        block = prices.iloc[start : start + LONG_WINDOW]
        for i, a in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                b = tickers[j]
                y, x = block[a].values, block[b].values
                if np.std(y) < 1e-12 or np.std(x) < 1e-12:
                    continue
                _, pval, _ = coint(y, x)
                if pval >= SIGNIFICANCE:
                    continue
                beta = hedge_ratio(y, x)
                hl = half_life(y - beta * x)
                if np.isfinite(hl):
                    rows.append({"pair": f"{a}-{b}", "start": start + 1, "pval": pval, "half_life": hl})
    return pd.DataFrame(rows)


def simulate_random_walks(T: int, vol_y: float, vol_x: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent GBM-like RW in levels (additive for simplicity, scaled by price~100)."""
    # Start near 100; daily moves ~ vol * 100
    dy = RNG.normal(0, vol_y * 100, size=(n, T))
    dx = RNG.normal(0, vol_x * 100, size=(n, T))
    y = 100 + np.cumsum(dy, axis=1)
    x = 100 + np.cumsum(dx, axis=1)
    return y, x


def simulate_cointegrated(
    T: int,
    vol_x: float,
    half_life_days: float,
    noise_scale: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x = RW, y = beta*x + spread, spread = AR(1) with given half-life.
    phi = exp(-ln2 / HL) on levels of spread.
    """
    beta = 1.0
    phi = float(np.exp(-np.log(2) / max(half_life_days, 1.0)))
    # innovation sd of spread so unconditional sd is noise_scale * (typical price move)
    # Var(spread) = sig_e^2 / (1-phi^2)
    target_sd = noise_scale * vol_x * 100
    sig_e = target_sd * np.sqrt(max(1 - phi**2, 1e-8))

    dx = RNG.normal(0, vol_x * 100, size=(n, T))
    x = 100 + np.cumsum(dx, axis=1)
    e = RNG.normal(0, sig_e, size=(n, T))
    spread = np.zeros((n, T))
    spread[:, 0] = e[:, 0]
    for t in range(1, T):
        spread[:, t] = phi * spread[:, t - 1] + e[:, t]
    y = beta * x + spread
    return y, x


def eg_reject_rate(y: np.ndarray, x: np.ndarray) -> float:
    """Fraction of paths where EG p < SIGNIFICANCE."""
    n = y.shape[0]
    rejects = 0
    for i in range(n):
        try:
            _, pval, _ = coint(y[i], x[i])
            if pval < SIGNIFICANCE:
                rejects += 1
        except Exception:
            pass
    return rejects / n


def run_size_power(
    vols: np.ndarray,
    hl_median: float,
    hl_p75: float,
    noise_scale: float,
) -> pd.DataFrame:
    # Match typical instrument vol
    vol_typ = float(np.median(vols))
    rows = []
    for T in WINDOW_SIZES:
        print(f"T={T} ...", flush=True)
        # SIZE: independent RWs
        y0, x0 = simulate_random_walks(T, vol_typ, vol_typ, N_MC)
        size = eg_reject_rate(y0, x0)

        # POWER at median and slow half-life
        y1, x1 = simulate_cointegrated(T, vol_typ, hl_median, noise_scale, N_MC)
        power_med = eg_reject_rate(y1, x1)
        y2, x2 = simulate_cointegrated(T, vol_typ, hl_p75, noise_scale, N_MC)
        power_slow = eg_reject_rate(y2, x2)

        # Rule-of-thumb: T >= k * half-life
        rows.append(
            {
                "T": T,
                "size": size,
                "power_hl_median": power_med,
                "power_hl_p75": power_slow,
                "T_over_hl_median": T / hl_median,
                "T_over_hl_p75": T / hl_p75,
            }
        )
        print(
            f"  size={size*100:.1f}%  power(med HL)={power_med*100:.1f}%  "
            f"power(p75 HL)={power_slow*100:.1f}%"
        )
    return pd.DataFrame(rows)


def empirical_size_on_returns_shuffle(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Extra check: destroy cross-sectional dependence by shuffling each series'
    returns independently, rebuild prices, run EG on random pairs for each T.
    """
    rets = prices.pct_change().dropna().values
    tickers = prices.columns.tolist()
    n_days, n_inst = rets.shape
    rows = []
    n_tests = 200

    for T in WINDOW_SIZES:
        if T >= n_days:
            continue
        rejects = 0
        for _ in range(n_tests):
            # shuffle returns per column
            shuffled = np.empty_like(rets)
            for j in range(n_inst):
                shuffled[:, j] = RNG.permutation(rets[:, j])
            # rebuild price paths
            p0 = prices.iloc[0].values
            path = np.vstack([p0, p0 * np.cumprod(1 + shuffled, axis=0)])
            # random pair, random window
            i, j = RNG.choice(n_inst, size=2, replace=False)
            start = int(RNG.integers(0, len(path) - T))
            y = path[start : start + T, i]
            x = path[start : start + T, j]
            try:
                _, pval, _ = coint(y, x)
                if pval < SIGNIFICANCE:
                    rejects += 1
            except Exception:
                pass
        rows.append({"T": T, "empirical_null_reject": rejects / n_tests})
        print(f"shuffle-null T={T}: reject={rejects/n_tests*100:.1f}%")
    return pd.DataFrame(rows)


def recommend(results: pd.DataFrame, hl_median: float) -> dict:
    """
    Prefer smallest T with size in a reasonable band and usable power.
    Prefer power_hl_fast if present (more relevant for quick mean-reversion).
    """
    power_col = "power_hl_fast" if "power_hl_fast" in results.columns else "power_hl_median"
    sized = results[results["size"] <= 0.12]
    ok50 = sized[sized[power_col] >= 0.50]
    ok80 = sized[sized[power_col] >= 0.80]
    # Also: size closest to 5% among T with power>=50%
    rec = {
        "hl_median": hl_median,
        "power_col_used": power_col,
        "min_T_power50": int(ok50["T"].min()) if len(ok50) else None,
        "min_T_power80": int(ok80["T"].min()) if len(ok80) else None,
        "rule_of_thumb_2x_hl": int(np.ceil(2 * hl_median)),
        "rule_of_thumb_4x_hl": int(np.ceil(4 * hl_median)),
        "best_T_by_power": int(sized.loc[sized[power_col].idxmax(), "T"]) if len(sized) else None,
        "best_power": float(sized[power_col].max()) if len(sized) else None,
    }
    return rec


def plot_results(results: pd.DataFrame, shuffle: pd.DataFrame, hl: pd.DataFrame, rec: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Half-life distribution
    axes[0, 0].hist(hl["half_life"].clip(upper=100), bins=30, color="#4C72B0", edgecolor="white")
    axes[0, 0].axvline(hl["half_life"].median(), color="red", ls="--", label=f"median={hl['half_life'].median():.1f}d")
    axes[0, 0].axvline(hl["half_life"].quantile(0.75), color="orange", ls="--", label=f"p75={hl['half_life'].quantile(0.75):.1f}d")
    axes[0, 0].set_xlabel("Spread half-life (days)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Empirical half-lives when {LONG_WINDOW}d EG is significant")
    axes[0, 0].legend(fontsize=8)

    # Size
    axes[0, 1].plot(results["T"], results["size"] * 100, "o-", color="#C44E52", label="MC null (indep RW)")
    if len(shuffle):
        axes[0, 1].plot(shuffle["T"], shuffle["empirical_null_reject"] * 100, "s--", color="#8172B2", label="shuffle-null (data vols)")
    axes[0, 1].axhline(5, color="gray", ls="--", lw=1, label="nominal 5%")
    axes[0, 1].set_xlabel("Window size T (days)")
    axes[0, 1].set_ylabel("False positive rate (%)")
    axes[0, 1].set_title("SIZE: how often EG falsely rejects no-cointegration")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_ylim(0, max(20, results["size"].max() * 100 * 1.2))

    # Power
    axes[1, 0].plot(results["T"], results["power_hl_median"] * 100, "o-", color="#55A868", label=f"HL=median ({rec['hl_median']:.1f}d)")
    axes[1, 0].plot(results["T"], results["power_hl_p75"] * 100, "s--", color="#CCB974", label="HL=p75 (slower)")
    if "power_hl_fast" in results.columns:
        axes[1, 0].plot(results["T"], results["power_hl_fast"] * 100, "^-", color="#4C72B0", label="HL=p25 (faster)")
    axes[1, 0].axhline(50, color="gray", ls=":", lw=1)
    axes[1, 0].axhline(80, color="gray", ls="--", lw=1)
    if rec["min_T_power50"]:
        axes[1, 0].axvline(rec["min_T_power50"], color="#55A868", ls="--", alpha=0.7, label=f"min T@50%={rec['min_T_power50']}")
    if rec["min_T_power80"]:
        axes[1, 0].axvline(rec["min_T_power80"], color="#2a7a3a", ls="--", alpha=0.7, label=f"min T@80%={rec['min_T_power80']}")
    axes[1, 0].set_xlabel("Window size T (days)")
    axes[1, 0].set_ylabel("True positive rate (%)")
    axes[1, 0].set_title("POWER: detect true cointegration (data-calibrated)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(0, 105)

    # Combined decision chart
    power_col = rec.get("power_col_used", "power_hl_median")
    axes[1, 1].plot(results["T"], results["size"] * 100, "o-", color="#C44E52", label="Size %")
    axes[1, 1].plot(results["T"], results[power_col] * 100, "o-", color="#55A868", label=f"Power % ({power_col})")
    axes[1, 1].axhline(5, color="#C44E52", ls=":", alpha=0.5)
    axes[1, 1].axhline(50, color="#55A868", ls=":", alpha=0.5)
    axes[1, 1].set_xlabel("Window size T (days)")
    axes[1, 1].set_ylabel("%")
    title = "Decision: size near 5%, power >= 50%"
    if rec["min_T_power50"]:
        title += f"  ->  recommend T >= {rec['min_T_power50']}"
    axes[1, 1].set_title(title)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "Minimum window for statistically meaningful EG cointegration detection",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "window_size_size_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    vols = daily_vol(prices)
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments")
    print(f"Median daily vol: {np.median(vols)*100:.2f}%")

    print("\n1) Harvest empirical half-lives from significant LONG_WINDOW blocks...")
    hl = harvest_empirical_half_lives(prices)
    hl = hl.dropna(subset=["half_life"])
    hl.to_csv(OUTPUT_DIR / "empirical_half_lives.csv", index=False)
    if len(hl) == 0:
        hl_median, hl_p75 = 8.0, 15.0
        print("   No finite half-lives; using defaults 8/15d")
    else:
        hl_median = float(hl["half_life"].median())
        hl_p75 = float(hl["half_life"].quantile(0.75))
        print(f"   n={len(hl)} | median HL={hl_median:.1f}d | p75 HL={hl_p75:.1f}d")

    noise_scale = 0.9  # calibrated: median residual SD / median daily $ move ~ 0.87

    print("\n2) Monte Carlo size & power vs window length...")
    # Also report power at faster empirical HL (~p25 of tight cointegrated spreads)
    hl_fast = float(hl["half_life"].quantile(0.25)) if len(hl) else 3.0
    print(f"   Using noise_scale={noise_scale}, HL fast/med/slow={hl_fast:.1f}/{hl_median:.1f}/{hl_p75:.1f}")
    results = run_size_power(vols, max(hl_median, 1.0), max(hl_p75, 2.0), noise_scale)

    # Extra power column for fast half-life (common in this dataset)
    print("   Extra power curve for fast HL...")
    extra = []
    vol_typ = float(np.median(vols))
    for T in WINDOW_SIZES:
        y, x = simulate_cointegrated(T, vol_typ, max(hl_fast, 1.0), noise_scale, N_MC)
        extra.append(eg_reject_rate(y, x))
    results["power_hl_fast"] = extra
    results["hl_fast"] = hl_fast
    print(results[["T", "size", "power_hl_fast", "power_hl_median", "power_hl_p75"]].to_string(index=False))
    results.to_csv(OUTPUT_DIR / "size_power_by_T.csv", index=False)

    print("\n3) Shuffle-null empirical size (returns reshuffled)...")
    shuffle = empirical_size_on_returns_shuffle(prices)
    shuffle.to_csv(OUTPUT_DIR / "shuffle_null_size.csv", index=False)

    rec = recommend(results, hl_median)
    pd.Series(rec).to_csv(OUTPUT_DIR / "recommendation.csv")
    plot_results(results, shuffle, hl if len(hl) else pd.DataFrame({"half_life": [hl_median]}), rec, OUTPUT_DIR)

    print("\n=== How to choose T ===")
    print("1. Estimate spread half-life when cointegration appears in a long window.")
    print("2. Check SIZE under null (should be ~5% at p<0.05).")
    print("3. Check POWER under alternative calibrated to that half-life.")
    print("4. Pick smallest T with size OK and power >= 50% (or 80% if conservative).")
    print("\n=== Recommendation for this dataset ===")
    print(f"Empirical median half-life: {rec['hl_median']:.1f} days")
    print(f"Rule of thumb 2x HL: T >= {rec['rule_of_thumb_2x_hl']}")
    print(f"Rule of thumb 4x HL: T >= {rec['rule_of_thumb_4x_hl']}")
    print(f"Min T with power>=50% & size<=10%: {rec['min_T_power50']}")
    print(f"Min T with power>=80% & size<=10%: {rec['min_T_power80']}")


if __name__ == "__main__":
    main()
