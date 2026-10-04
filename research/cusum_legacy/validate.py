"""Whiteness + drift-in-Q validation for a calibrated pairs Kalman filter.

Pipeline (strict order):
  1. Calibrate Q/R on the calm window (via ``calibrate.calibrate_pair``).
  2. Ljung-Box whiteness gate on standardized innovations z_k.
  3. Only if step 2 PASSes: rolling-variance summary + Engle ARCH LM test
     to decide whether fixed Q is adequate or adaptive-Q may be warranted.

Produces a markdown validation report and a standardized-innovations plot.
Does **not** implement CUSUM, regime switching, or adaptive-Q estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from research.cusum_legacy.calibrate import (
    CalibrationResult,
    calibrate_pair,
    extract_pair,
    load_prices,
)
from research.cusum_legacy.kalman_pair import run_kalman_pair


# ---- named tunable defaults ----
DEFAULT_SIGNIFICANCE: float = 0.05
DEFAULT_LJUNG_BOX_LAGS: int = 10
DEFAULT_ROLLING_VAR_WINDOW: int = 25
DEFAULT_ARCH_LAGS: int = 5
DEFAULT_MIN_ARCH_OBS: int = 50  # below this => ARCH verdict INCONCLUSIVE
DEFAULT_INIT_WINDOW: int = 20
# SMAH/NGTE worked-example calm window (pre-break, per earlier break analysis)
SMAH_NGTE_CALM_START: int = 0
SMAH_NGTE_CALM_END: int = 170


@dataclass(frozen=True)
class LjungBoxResult:
    statistic: float
    pvalue: float
    lags: int
    significance: float
    verdict: str  # PASS | FAIL | INCONCLUSIVE
    note: str


@dataclass(frozen=True)
class ArchResult:
    statistic: float | None
    pvalue: float | None
    lags: int
    rolling_var_min: float | None
    rolling_var_max: float | None
    rolling_var_range: float | None
    rolling_var_window: int
    verdict: str
    note: str


@dataclass(frozen=True)
class ValidationReport:
    calibration: CalibrationResult
    z: np.ndarray
    ljung_box: LjungBoxResult
    arch: ArchResult | None
    report_path: Path
    plot_path: Path


def _ljung_box_gate(
    z: np.ndarray,
    *,
    lags: int = DEFAULT_LJUNG_BOX_LAGS,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> LjungBoxResult:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < lags + 5:
        return LjungBoxResult(
            statistic=float("nan"),
            pvalue=float("nan"),
            lags=lags,
            significance=significance,
            verdict="INCONCLUSIVE",
            note=(
                f"Only {z.size} finite innovations; need > {lags + 5} for a "
                f"reliable Ljung-Box test at lags={lags}."
            ),
        )

    lb = acorr_ljungbox(z, lags=[lags], return_df=True)
    stat = float(lb["lb_stat"].iloc[-1])
    pval = float(lb["lb_pvalue"].iloc[-1])

    if pval >= significance:
        verdict = "PASS"
        note = (
            f"Innovations consistent with white noise at alpha={significance} "
            f"(p={pval:.4f} >= {significance})."
        )
    else:
        verdict = "FAIL"
        # Point at the strongest short-lag ACF for a concrete diagnosis.
        max_lag = min(10, z.size // 5)
        acf = np.array(
            [np.corrcoef(z[k:], z[:-k])[0, 1] for k in range(1, max_lag + 1)]
        )
        worst_lag = int(np.argmax(np.abs(acf))) + 1
        worst_acf = float(acf[worst_lag - 1])
        note = (
            "Innovations show residual autocorrelation "
            f"(strongest sample ACF at lag {worst_lag}: {worst_acf:+.3f}). "
            "Plausible causes: calm-period window still contains mild "
            "regime structure; observation equation may need a lagged "
            "spread/error term; or Q/R alone cannot absorb short-horizon "
            "dependence under a pure random-walk alpha/beta. Filter is "
            "misspecified until this gate passes; subsequent drift-in-Q "
            "diagnostics were not run."
        )

    return LjungBoxResult(
        statistic=stat,
        pvalue=pval,
        lags=lags,
        significance=significance,
        verdict=verdict,
        note=note,
    )


def _rolling_variance(z: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan, dtype=float)
    if window < 2 or z.size < window:
        return out
    # population variance over trailing window (ddof=0)
    csum = np.cumsum(np.insert(z, 0, 0.0))
    csum2 = np.cumsum(np.insert(z * z, 0, 0.0))
    total = csum[window:] - csum[:-window]
    total2 = csum2[window:] - csum2[:-window]
    mean = total / window
    out[window - 1 :] = total2 / window - mean * mean
    return out


def _arch_diagnostic(
    z: np.ndarray,
    *,
    rolling_var_window: int = DEFAULT_ROLLING_VAR_WINDOW,
    arch_lags: int = DEFAULT_ARCH_LAGS,
    significance: float = DEFAULT_SIGNIFICANCE,
    min_obs: int = DEFAULT_MIN_ARCH_OBS,
) -> ArchResult:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    roll = _rolling_variance(z, rolling_var_window)
    valid_roll = roll[np.isfinite(roll)]

    if z.size < min_obs:
        return ArchResult(
            statistic=None,
            pvalue=None,
            lags=arch_lags,
            rolling_var_min=float(valid_roll.min()) if valid_roll.size else None,
            rolling_var_max=float(valid_roll.max()) if valid_roll.size else None,
            rolling_var_range=(
                float(valid_roll.max() - valid_roll.min()) if valid_roll.size else None
            ),
            rolling_var_window=rolling_var_window,
            verdict="inconclusive - insufficient calm-period data",
            note=(
                f"Only {z.size} finite innovations; need at least {min_obs} "
                "for a reliable ARCH LM test."
            ),
        )

    lm_stat, lm_pval, _, _ = het_arch(z, nlags=arch_lags)
    lm_stat_f = float(lm_stat)
    lm_pval_f = float(lm_pval)

    rmin = float(valid_roll.min()) if valid_roll.size else float("nan")
    rmax = float(valid_roll.max()) if valid_roll.size else float("nan")
    rrange = float(rmax - rmin) if valid_roll.size else float("nan")

    if lm_pval_f >= significance:
        verdict = "no evidence of time-varying noise - fixed Q is adequate"
        note = (
            f"Engle ARCH LM p={lm_pval_f:.4f} >= {significance}; "
            "rolling innovation variance stays consistent with constant noise."
        )
    else:
        verdict = (
            "evidence of heteroskedasticity - adaptive-Q extension may be warranted"
        )
        note = (
            f"Engle ARCH LM p={lm_pval_f:.4f} < {significance}; "
            "standardized innovations show conditional heteroskedasticity "
            "in the calm regime."
        )

    return ArchResult(
        statistic=lm_stat_f,
        pvalue=lm_pval_f,
        lags=arch_lags,
        rolling_var_min=rmin,
        rolling_var_max=rmax,
        rolling_var_range=rrange,
        rolling_var_window=rolling_var_window,
        verdict=verdict,
        note=note,
    )


def _format_matrix(M: np.ndarray, precision: int = 6) -> str:
    rows = []
    for row in np.atleast_2d(M):
        cells = ", ".join(f"{float(v):.{precision}g}" for v in row)
        rows.append(f"[{cells}]")
    return "[" + ", ".join(rows) + "]"


def _write_report(
    path: Path,
    cal: CalibrationResult,
    lb: LjungBoxResult,
    arch: ArchResult | None,
    *,
    n_innov: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Kalman filter validation — {cal.y_ticker}/{cal.x_ticker}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Base filter + calm-regime Q/R calibration + whiteness gate + "
        "drift-in-Q diagnostic. No CUSUM, switching, or adaptive-Q logic."
    )
    lines.append("")
    lines.append("## Calm window")
    lines.append("")
    lines.append(f"- Indices: `[{cal.calm_start}, {cal.calm_end})` (exclusive end)")
    lines.append(f"- OLS init window: `{cal.init_window}`")
    lines.append(f"- Innovations scored / tested: `{n_innov}`")
    lines.append("")
    lines.append("## Fitted Q / R")
    lines.append("")
    lines.append(f"- Q (2×2): `{_format_matrix(cal.Q)}`")
    lines.append(f"- R (1×1): `{_format_matrix(cal.R)}`")
    lines.append(f"- Negative log-likelihood: `{cal.nll:.6f}`")
    lines.append(f"- Optimizer success: `{cal.success}`")
    lines.append(f"- Optimizer message: `{cal.message}`")
    if not cal.success:
        lines.append("")
        lines.append(
            "> **Warning:** optimizer did not converge cleanly — treat fitted "
            "Q/R with caution."
        )
    lines.append("")
    lines.append("## Whiteness gate (Ljung-Box)")
    lines.append("")
    lines.append(f"- Significance threshold: `{lb.significance}` (stated)")
    lines.append(f"- Lags: `{lb.lags}`")
    lines.append(f"- Statistic: `{lb.statistic:.6f}`")
    lines.append(f"- p-value: `{lb.pvalue:.6f}`")
    lines.append(f"- **Verdict: {lb.verdict}**")
    lines.append(f"- Note: {lb.note}")
    lines.append("")
    lines.append("## Drift-in-Q diagnostic (ARCH / rolling variance)")
    lines.append("")
    if arch is None:
        lines.append(
            "Skipped — Ljung-Box gate did not PASS. Drift-in-Q results would "
            "be unreliable under a misspecified filter."
        )
    else:
        lines.append(
            f"- Rolling variance window: `{arch.rolling_var_window}` "
            "(default 25)"
        )
        if arch.rolling_var_min is not None:
            lines.append(
                f"- Rolling var of z_k: min=`{arch.rolling_var_min:.6f}`, "
                f"max=`{arch.rolling_var_max:.6f}`, "
                f"range=`{arch.rolling_var_range:.6f}`"
            )
        else:
            lines.append("- Rolling var of z_k: insufficient points")
        lines.append(f"- ARCH LM lags: `{arch.lags}`")
        if arch.statistic is not None:
            lines.append(f"- ARCH LM statistic: `{arch.statistic:.6f}`")
            lines.append(f"- ARCH LM p-value: `{arch.pvalue:.6f}`")
        else:
            lines.append("- ARCH LM statistic / p-value: n/a")
        lines.append(f"- **Verdict: {arch.verdict}**")
        lines.append(f"- Note: {arch.note}")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    if lb.verdict != "PASS":
        lines.append(
            f"Filter specification gate: **{lb.verdict}**. Do not interpret "
            "fixed-Q adequacy until whiteness passes."
        )
    elif arch is not None:
        lines.append(arch.verdict)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_innovations(
    path: Path,
    z: np.ndarray,
    *,
    y_ticker: str,
    x_ticker: str,
    calm_start: int,
    calm_end: int,
    skip: int,
) -> None:
    days = np.arange(calm_start + skip, calm_start + skip + len(z))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(days, z, lw=1.0, color="#1f4e79")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.axhline(2.0, color="gray", ls="--", lw=0.6)
    ax.axhline(-2.0, color="gray", ls="--", lw=0.6)
    ax.set_title(
        f"Standardized innovations z_k — {y_ticker}/{x_ticker} "
        f"calm [{calm_start}, {calm_end})"
    )
    ax.set_xlabel("Day index")
    ax.set_ylabel("z_k = y_k / sqrt(S_k)")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def validate_pair(
    y: np.ndarray,
    x: np.ndarray,
    *,
    y_ticker: str,
    x_ticker: str,
    calm_start: int = 0,
    calm_end: int | None = None,
    init_window: int = DEFAULT_INIT_WINDOW,
    significance: float = DEFAULT_SIGNIFICANCE,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
    rolling_var_window: int = DEFAULT_ROLLING_VAR_WINDOW,
    arch_lags: int = DEFAULT_ARCH_LAGS,
    min_arch_obs: int = DEFAULT_MIN_ARCH_OBS,
    output_dir: str | Path = "kalman_validation_output",
    calibration: CalibrationResult | None = None,
) -> ValidationReport:
    """Calibrate (unless provided), gate on Ljung-Box, then ARCH if PASS."""
    output_dir = Path(output_dir)
    pair_tag = f"{y_ticker}_{x_ticker}"

    if calibration is None:
        calibration = calibrate_pair(
            y,
            x,
            y_ticker=y_ticker,
            x_ticker=x_ticker,
            calm_start=calm_start,
            calm_end=calm_end,
            init_window=init_window,
        )
    else:
        calm_start = calibration.calm_start
        calm_end = calibration.calm_end
        init_window = calibration.init_window

    if calm_end is None:
        calm_end = len(y)

    y_c = y[calm_start:calm_end]
    x_c = x[calm_start:calm_end]
    filt = run_kalman_pair(y_c, x_c, calibration.config)

    # Score / test after OLS warm-start, matching calibration NLL window
    z = filt.z[init_window:]
    lb = _ljung_box_gate(z, lags=ljung_box_lags, significance=significance)

    arch: ArchResult | None = None
    if lb.verdict == "PASS":
        arch = _arch_diagnostic(
            z,
            rolling_var_window=rolling_var_window,
            arch_lags=arch_lags,
            significance=significance,
            min_obs=min_arch_obs,
        )

    report_path = output_dir / f"{pair_tag}_validation_report.md"
    plot_path = output_dir / f"{pair_tag}_standardized_innovations.png"

    _write_report(report_path, calibration, lb, arch, n_innov=len(z))
    _plot_innovations(
        plot_path,
        z,
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        calm_start=calm_start,
        calm_end=calm_end,
        skip=init_window,
    )

    return ValidationReport(
        calibration=calibration,
        z=z,
        ljung_box=lb,
        arch=arch,
        report_path=report_path,
        plot_path=plot_path,
    )


def run_smah_ngte_example(
    prices_path: str | Path = "prices.txt",
    *,
    calm_start: int = SMAH_NGTE_CALM_START,
    calm_end: int = SMAH_NGTE_CALM_END,
    output_dir: str | Path = "kalman_validation_output",
) -> ValidationReport:
    """Worked example: SMAH/NGTE on the pre-break calm window [0, 170)."""
    prices = load_prices(prices_path)
    y, x = extract_pair(prices, "SMAH", "NGTE")
    return validate_pair(
        y,
        x,
        y_ticker="SMAH",
        x_ticker="NGTE",
        calm_start=calm_start,
        calm_end=calm_end,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    report = run_smah_ngte_example()
    cal = report.calibration
    print(f"Pair: {cal.y_ticker}/{cal.x_ticker}")
    print(f"Calm window: [{cal.calm_start}, {cal.calm_end})")
    print(f"Q:\n{cal.Q}")
    print(f"R:\n{cal.R}")
    print(f"Optimizer success: {cal.success} ({cal.message})")
    print(f"NLL: {cal.nll:.6f}")
    print(
        f"Ljung-Box: stat={report.ljung_box.statistic:.4f} "
        f"p={report.ljung_box.pvalue:.4f} => {report.ljung_box.verdict}"
    )
    if report.arch is not None:
        print(f"ARCH verdict: {report.arch.verdict}")
        if report.arch.pvalue is not None:
            print(
                f"  ARCH LM stat={report.arch.statistic:.4f} "
                f"p={report.arch.pvalue:.4f}"
            )
    else:
        print("ARCH: skipped (Ljung-Box did not PASS)")
    print(f"Report: {report.report_path}")
    print(f"Plot:   {report.plot_path}")
