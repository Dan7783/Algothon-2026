"""Pooled-null ARL calibration for two-sided CUSUM thresholds.

Calibrates ``h`` from calm-period standardized innovations across many pairs
(not from per-pair historical breaks).  Empirical ARL0 is estimated by
simulation; the Siegmund closed-form approximation is reported only as a
sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.cusum_legacy.calibrate import calibrate_pair, extract_pair, load_prices
from research.cusum_legacy.cusum import DEFAULT_KAPPA as CUSUM_DEFAULT_KAPPA, run_length_to_alarm
from research.cusum_legacy.kalman_pair import run_kalman_pair
from statsmodels.stats.diagnostic import acorr_ljungbox


# ---- named tunable defaults ----
# Target average run length under the null (false-alarm spacing).
# This is a risk-tolerance knob, not a fit to SMAH/NGTE:
#   - ~550 keeps sensitivity to slow beta transitions the base filter
#     partially absorbs (SMAH/NGTE first alarm ~day 200).
#   - Raise toward 1500 for fewer false alarms (larger h); on SMAH/NGTE that
#     delays the first alarm from ~day 200 to ~day 357.
DEFAULT_TARGET_ARL0: float = 550.0
DEFAULT_KAPPA: float = CUSUM_DEFAULT_KAPPA
DEFAULT_H_GRID: tuple[float, ...] = (
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0
)
DEFAULT_N_SIMULATIONS: int = 300
DEFAULT_MAX_RUN_LENGTH: int = 20_000
DEFAULT_BLOCK_LENGTH: int = 80  # bootstrap block size for resampling null z
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_INIT_WINDOW: int = 20
DEFAULT_LJUNG_BOX_LAGS: int = 10
DEFAULT_SIGNIFICANCE: float = 0.05

# Pairs used for the pooled calm null.  Each entry: (y, x, calm_start, calm_end).
# SMAH/NGTE uses the validated pre-break window; others use an early calm block
# that must still pass the Ljung-Box gate before inclusion.
DEFAULT_POOL_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("SMAH", "NGTE", 0, 170),
    ("SMAH", "ILVX", 0, 200),
    ("AENO", "NWIG", 0, 200),
    ("EORC", "NGTE", 0, 200),
    ("HUXZ", "ACAC", 0, 200),
)


@dataclass(frozen=True)
class ArlRow:
    h: float
    arl0_empirical: float
    arl0_siegmund: float
    n_simulations: int
    n_censored: int  # sims that hit max_run_length without alarm


@dataclass(frozen=True)
class ArlCalibrationResult:
    kappa: float
    target_arl0: float
    chosen_h: float
    chosen_arl0: float
    table: list[ArlRow]
    pool_pair_labels: list[str]
    pool_n_obs: int
    output_dir: Path


def siegmund_arl0_two_sided(h: float, kappa: float) -> float:
    """Siegmund Brownian approximation for two-sided CUSUM ARL0 (sanity check).

    One-sided ARL0 ≈ (exp(2 κ h) − 2 κ h − 1) / (2 κ²); two-sided ≈ half.
    """
    if kappa <= 0.0:
        return float("inf")
    one_sided = (np.exp(2.0 * kappa * h) - 2.0 * kappa * h - 1.0) / (2.0 * kappa * kappa)
    return float(one_sided / 2.0)


def collect_calm_z_pool(
    prices: pd.DataFrame,
    pool_specs: Sequence[tuple[str, str, int, int]] = DEFAULT_POOL_SPECS,
    *,
    init_window: int = DEFAULT_INIT_WINDOW,
    require_ljung_box_pass: bool = True,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> tuple[list[np.ndarray], list[str]]:
    """Build calm-period z series from pairs that pass the whiteness gate."""
    series: list[np.ndarray] = []
    labels: list[str] = []

    for y_ticker, x_ticker, calm_start, calm_end in pool_specs:
        y, x = extract_pair(prices, y_ticker, x_ticker)
        cal = calibrate_pair(
            y,
            x,
            y_ticker=y_ticker,
            x_ticker=x_ticker,
            calm_start=calm_start,
            calm_end=calm_end,
            init_window=init_window,
        )
        filt = run_kalman_pair(
            y[calm_start:calm_end], x[calm_start:calm_end], cal.config
        )
        z = filt.z[init_window:]
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        if z.size < 50:
            print(f"  skip {y_ticker}/{x_ticker}: only {z.size} finite z")
            continue

        if require_ljung_box_pass:
            lb = acorr_ljungbox(z, lags=[ljung_box_lags], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[-1])
            if pval < significance:
                print(
                    f"  skip {y_ticker}/{x_ticker} [{calm_start},{calm_end}): "
                    f"Ljung-Box FAIL (p={pval:.4f})"
                )
                continue
            print(
                f"  pool + {y_ticker}/{x_ticker}[{calm_start},{calm_end})  "
                f"n={z.size}  LB p={pval:.4f}"
            )
        else:
            print(f"  pool + {y_ticker}/{x_ticker}[{calm_start},{calm_end})  n={z.size}")

        series.append(z)
        labels.append(f"{y_ticker}/{x_ticker}[{calm_start},{calm_end})")

    if not series:
        raise RuntimeError("No calm z series passed the pool gate")
    return series, labels


def _resample_null_path(
    pool: list[np.ndarray],
    length: int,
    rng: np.random.Generator,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
) -> np.ndarray:
    """Circular block-bootstrap a null z path of the requested length."""
    # Draw blocks from a randomly chosen series in the pool each time.
    out = np.empty(length, dtype=float)
    filled = 0
    while filled < length:
        src = pool[int(rng.integers(0, len(pool)))]
        n = src.size
        if n < 2:
            continue
        bl = min(block_length, n)
        start = int(rng.integers(0, n))
        take = min(bl, length - filled)
        idx = (np.arange(take) + start) % n
        out[filled : filled + take] = src[idx]
        filled += take
    return out


def estimate_arl0(
    pool: list[np.ndarray],
    h: float,
    *,
    kappa: float = DEFAULT_KAPPA,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    max_run_length: int = DEFAULT_MAX_RUN_LENGTH,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    rng: np.random.Generator | None = None,
    method: str = "parametric",
) -> tuple[float, int]:
    """Empirical ARL0 for one (kappa, h).

    method:
      - ``parametric`` (default): draw z ~ N(mu, sigma^2) from pooled calm
        moments.  Stable for large target ARL0; matches the standardized
        innovation null after the whiteness gate.
      - ``block``: circular block-bootstrap of pooled paths (dependence-preserving
        sensitivity check; noisier at large h).
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    concat = np.concatenate(pool)
    mu = float(np.mean(concat))
    sigma = float(np.std(concat, ddof=1))
    if sigma < 1e-8:
        sigma = 1.0

    run_lengths = np.empty(n_simulations, dtype=float)
    censored = 0
    for i in range(n_simulations):
        if method == "block":
            path = _resample_null_path(
                pool, max_run_length, rng, block_length=block_length
            )
        else:
            path = rng.normal(mu, sigma, size=max_run_length)
        rl = run_length_to_alarm(path, kappa=kappa, h=h)
        if rl >= max_run_length:
            censored += 1
        run_lengths[i] = rl
    return float(np.mean(run_lengths)), censored


def calibrate_h_from_arl(
    pool: list[np.ndarray],
    *,
    kappa: float = DEFAULT_KAPPA,
    target_arl0: float = DEFAULT_TARGET_ARL0,
    h_grid: Sequence[float] = DEFAULT_H_GRID,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    max_run_length: int = DEFAULT_MAX_RUN_LENGTH,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float, list[ArlRow]]:
    """Sweep ``h_grid``, pick the smallest h with empirical ARL0 >= target.

    To change risk tolerance, pass a different ``target_arl0`` (default 1500).
    """
    rng = np.random.default_rng(seed)
    table: list[ArlRow] = []
    chosen_h = float(h_grid[-1])
    chosen_arl = 0.0

    for h in h_grid:
        arl_emp, n_cens = estimate_arl0(
            pool,
            float(h),
            kappa=kappa,
            n_simulations=n_simulations,
            max_run_length=max_run_length,
            block_length=block_length,
            rng=rng,
        )
        arl_approx = siegmund_arl0_two_sided(float(h), kappa)
        table.append(
            ArlRow(
                h=float(h),
                arl0_empirical=arl_emp,
                arl0_siegmund=arl_approx,
                n_simulations=n_simulations,
                n_censored=n_cens,
            )
        )
        print(
            f"  h={h:.2f}  ARL0_emp={arl_emp:.1f}  "
            f"Siegmund~{arl_approx:.1f}  censored={n_cens}/{n_simulations}"
        )

    # Prefer the smallest h whose empirical ARL0 meets the target.
    for row in table:
        if row.arl0_empirical >= target_arl0:
            chosen_h = row.h
            chosen_arl = row.arl0_empirical
            break
    else:
        # No h met the target; take the largest (most conservative).
        chosen_h = table[-1].h
        chosen_arl = table[-1].arl0_empirical

    return chosen_h, chosen_arl, table


def _write_arl_outputs(
    result: ArlCalibrationResult,
) -> None:
    out = result.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "h": r.h,
                "arl0_empirical": r.arl0_empirical,
                "arl0_siegmund": r.arl0_siegmund,
                "n_simulations": r.n_simulations,
                "n_censored": r.n_censored,
            }
            for r in result.table
        ]
    )
    csv_path = out / "h_vs_arl0.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["h"], df["arl0_empirical"], "o-", label="Empirical ARL0", color="#1f4e79")
    ax.plot(
        df["h"],
        df["arl0_siegmund"],
        "s--",
        label="Siegmund approx (sanity)",
        color="#c44e52",
        alpha=0.8,
    )
    ax.axhline(
        result.target_arl0,
        color="gray",
        ls=":",
        label=f"target ARL0={result.target_arl0:g}",
    )
    ax.axvline(result.chosen_h, color="#2ca02c", ls="--", label=f"chosen h={result.chosen_h:g}")
    ax.set_xlabel("CUSUM threshold h")
    ax.set_ylabel("ARL0 (observations to false alarm)")
    ax.set_title("Pooled-null CUSUM ARL calibration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "h_vs_arl0.png", dpi=140)
    plt.close(fig)

    report = out / "arl_calibration_report.md"
    lines = [
        "# CUSUM ARL calibration (pooled null)",
        "",
        f"- kappa: `{result.kappa}`",
        f"- target ARL0: `{result.target_arl0}` "
        "(named parameter — raise to reduce false alarms, lower to detect sooner)",
        f"- chosen h: `{result.chosen_h}`",
        f"- achieved empirical ARL0: `{result.chosen_arl0:.1f}`",
        f"- pooled pairs: {', '.join(result.pool_pair_labels)}",
        f"- pooled calm observations (concat length): `{result.pool_n_obs}`",
        "",
        "## h vs ARL0",
        "",
        "| h | ARL0 empirical | ARL0 Siegmund | censored |",
        "|---|---|---|---|",
    ]
    for r in result.table:
        lines.append(
            f"| {r.h:.2f} | {r.arl0_empirical:.1f} | {r.arl0_siegmund:.1f} | "
            f"{r.n_censored}/{r.n_simulations} |"
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def run_arl_calibration(
    prices_path: str | Path = "prices.txt",
    *,
    pool_specs: Sequence[tuple[str, str, int, int]] = DEFAULT_POOL_SPECS,
    kappa: float = DEFAULT_KAPPA,
    target_arl0: float = DEFAULT_TARGET_ARL0,
    h_grid: Sequence[float] = DEFAULT_H_GRID,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    output_dir: str | Path = "cusum_output/arl_calibration",
) -> ArlCalibrationResult:
    """End-to-end: collect calm z pool, calibrate h, write table/plot/report."""
    prices = load_prices(prices_path)
    print("Collecting calm z pool (Ljung-Box PASS required):")
    pool, labels = collect_calm_z_pool(prices, pool_specs)
    pool_n = int(sum(s.size for s in pool))

    print(f"\nCalibrating h for target ARL0={target_arl0:g}, kappa={kappa}:")
    chosen_h, chosen_arl, table = calibrate_h_from_arl(
        pool,
        kappa=kappa,
        target_arl0=target_arl0,
        h_grid=h_grid,
        n_simulations=n_simulations,
    )
    print(f"\nChosen h={chosen_h:g} with empirical ARL0={chosen_arl:.1f}")

    result = ArlCalibrationResult(
        kappa=kappa,
        target_arl0=target_arl0,
        chosen_h=chosen_h,
        chosen_arl0=chosen_arl,
        table=table,
        pool_pair_labels=labels,
        pool_n_obs=pool_n,
        output_dir=Path(output_dir),
    )
    _write_arl_outputs(result)
    return result


if __name__ == "__main__":
    run_arl_calibration()
