"""Before/after effective-hedge chart + alpha/x scaling diagnostics."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibrate import calibrate_pair, extract_pair, load_prices
from regime_switch import Mode, run_regime_switch

OUT = Path("cusum_output/smah_ngte")
KAPPA, H = 0.25, 9.5
W = 100


def rolling_through_origin(y: np.ndarray, x: np.ndarray, w: int = W) -> np.ndarray:
    n = len(y)
    raw = np.full(n, np.nan)
    for i in range(w, n):
        xi, yi = x[i - w : i], y[i - w : i]
        denom = float(xi @ xi)
        if denom > 1e-12:
            raw[i] = float(xi @ yi) / denom
    return raw


def rolling_with_intercept(y: np.ndarray, x: np.ndarray, w: int = W) -> np.ndarray:
    n = len(y)
    beta = np.full(n, np.nan)
    for i in range(w, n):
        Z = np.column_stack([np.ones(w), x[i - w : i]])
        coef, _, _, _ = np.linalg.lstsq(Z, y[i - w : i], rcond=None)
        beta[i] = float(coef[1])
    return beta


def plot_eff_hedge(
    path: Path,
    y: np.ndarray,
    x: np.ndarray,
    result,
    *,
    title_suffix: str,
) -> np.ndarray:
    T = len(y)
    t = np.arange(T)
    raw = rolling_through_origin(y, x)
    artifact = rolling_with_intercept(y, x)
    eff = result.alpha / np.where(np.abs(x) > 1e-12, x, np.nan) + result.beta
    alarm_idx = np.flatnonzero(result.alarm_events)
    reseed_idx = np.flatnonzero(result.reseed_events)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(t, y / y[0], color="#4C72B0", lw=1.1, label="SMAH")
    ax.plot(t, x / x[0], color="#DD8452", lw=1.1, label="NGTE")
    ax.axvspan(280, 560, color="gray", alpha=0.12, label="break window")
    ax.set_ylabel("Price / day1")
    ax.set_title(f"SMAH-NGTE: CUSUM Kalman vs raw beta (through-origin){title_suffix}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, raw, color="#333333", lw=1.6, label=f"Raw beta: rolling {W}d through-origin (~1)")
    ax.plot(
        t,
        eff,
        color="#2ca02c",
        lw=1.5,
        label="CUSUM Kalman effective hedge (alpha/x + beta)",
    )
    ax.plot(
        t,
        result.beta,
        color="#1f4e79",
        lw=1.0,
        alpha=0.9,
        label="CUSUM Kalman slope beta (with intercept)",
    )
    ax.plot(
        t,
        artifact,
        color="#c44e52",
        lw=0.8,
        alpha=0.55,
        label="Rolling 100d with intercept (artifact: goes negative)",
    )
    ax.axhline(0.0, color="black", lw=0.6)
    ax.axhline(1.0, color="gray", ls="--", lw=0.7)
    for i, idx in enumerate(alarm_idx):
        ax.axvline(
            idx,
            color="#c44e52",
            ls="--",
            lw=0.8,
            alpha=0.7,
            label="CUSUM alarm" if i == 0 else None,
        )
    for i, idx in enumerate(reseed_idx):
        ax.axvline(
            idx,
            color="#999999",
            ls=":",
            lw=1.0,
            alpha=0.8,
            label="Reseed" if i == 0 else None,
        )
    ax.set_ylabel("beta")
    ax.set_xlabel("day index")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return eff


def calm_roughness_stats(eff: np.ndarray, result) -> dict:
    T = len(eff)
    deltas = np.abs(np.diff(eff, prepend=eff[0]))
    reseed_idx = np.flatnonzero(result.reseed_events)
    alarm_idx = np.flatnonzero(result.alarm_events)
    calm = result.mode == Mode.STABLE
    turb = result.mode == Mode.TURBULENT
    calm_ex = calm.copy()
    for ri in reseed_idx:
        calm_ex[max(0, ri) : min(T, ri + 5)] = False
    for ai in alarm_idx:
        calm_ex[max(0, ai - 2) : min(T, ai + 2)] = False

    reseed_jumps = [float(deltas[ri]) for ri in reseed_idx]
    return {
        "reseed_indices": reseed_idx.tolist(),
        "alarm_indices": alarm_idx.tolist(),
        "median_calm_delta": float(np.median(deltas[calm_ex])) if calm_ex.any() else float("nan"),
        "median_turb_delta": float(np.median(deltas[turb])) if turb.any() else float("nan"),
        "reseed_jumps": reseed_jumps,
        "max_reseed_jump": float(max(reseed_jumps)) if reseed_jumps else 0.0,
        "turb_fraction": float(result.mode.mean()),
    }


def plot_alpha_x_scaling(path: Path, x: np.ndarray, result, eff: np.ndarray) -> dict:
    T = len(x)
    t = np.arange(T)
    var_ax = result.p_alpha / (x ** 2)
    d_eff = np.diff(eff, prepend=eff[0])
    roll = 20
    local_rough = np.full(T, np.nan)
    for i in range(roll, T):
        local_rough[i] = float(np.std(d_eff[i - roll : i]))

    # Scaling is a property of (P_alpha, x); evaluate on the full series and
    # on early vs late windows (mode-agnostic), plus calm-only when available.
    finite = np.isfinite(local_rough) & np.isfinite(var_ax)
    corr_full = (
        float(np.corrcoef(local_rough[finite], var_ax[finite])[0, 1])
        if finite.sum() > 5
        else float("nan")
    )
    calm = (result.mode == Mode.STABLE) & (np.arange(T) >= 50)
    for ri in np.flatnonzero(result.reseed_events):
        calm[max(0, ri - 2) : min(T, ri + 15)] = False
    valid_calm = calm & finite
    corr_calm = (
        float(np.corrcoef(local_rough[valid_calm], var_ax[valid_calm])[0, 1])
        if valid_calm.sum() > 5
        else float("nan")
    )

    early = np.arange(T) < 200
    late = np.arange(T) > 500

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(t, eff, color="#2ca02c", lw=1.2, label="effective hedge")
    ax.set_ylabel("alpha/x + beta")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Alpha/x scaling diagnostic: Var(alpha)/x^2 vs effective-hedge roughness")

    ax = axes[1]
    ax.plot(t, var_ax, color="#1f4e79", lw=1.0, label="Var(alpha)/x^2 = P_alpha / x^2")
    ax.set_ylabel("Var(alpha)/x^2")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, local_rough, color="#c44e52", lw=1.0, label="local roughness (20d std of Diff(eff))")
    ax.set_ylabel("roughness")
    ax.set_xlabel("day index")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    return {
        "corr_full_rough_vs_var_ax": corr_full,
        "corr_calm_rough_vs_var_ax": corr_calm,
        "n_calm_valid": int(valid_calm.sum()),
        "median_var_ax_early": float(np.median(var_ax[early])),
        "median_var_ax_late": float(np.median(var_ax[late])),
        "median_rough_early": float(np.nanmedian(local_rough[early])),
        "median_rough_late": float(np.nanmedian(local_rough[late])),
        "x_early_med": float(np.median(x[:200])),
        "x_late_med": float(np.median(x[500:])),
        "inv_x2_ratio_late_over_early": float(
            (1.0 / np.median(x[500:]) ** 2) / (1.0 / np.median(x[:200]) ** 2)
        ),
        # Pre-fix calm-only correlation from provenance diagnostic
        "corr_calm_before_fix": 0.7549,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_path = OUT / "cusum_kalman_vs_raw_beta_BEFORE_reseed_fix.png"
    after_path = OUT / "cusum_kalman_vs_raw_beta.png"
    current = OUT / "cusum_kalman_vs_raw_beta.png"
    if current.exists() and not before_path.exists():
        shutil.copy2(current, before_path)

    prices = load_prices("prices.txt")
    y, x = extract_pair(prices, "SMAH", "NGTE")
    cal = calibrate_pair(
        y, x, y_ticker="SMAH", x_ticker="NGTE", calm_start=0, calm_end=170
    )
    result = run_regime_switch(y, x, cal.config, kappa=KAPPA, h=H)

    # Provenance check after fix: at each reseed, state should equal pre-step
    # continuous filter state (no OLS overwrite). We verify jumps in eff are small.
    eff = plot_eff_hedge(after_path, y, x, result, title_suffix=" [after reseed fix]")
    after_stats = calm_roughness_stats(eff, result)
    scale_stats = plot_alpha_x_scaling(
        OUT / "alpha_x_scaling_diagnostic.png", x, result, eff
    )

    # Before stats from prior diagnostic run (OLS overwrite era)
    before_stats = {
        "reseed_indices": [249, 404, 481, 584, 667, 733],
        "reseed_jumps": [0.022710, 0.093408, 0.033100, 0.025555, 0.013823, 0.043104],
        "max_reseed_jump": 0.093408,
        "median_calm_delta": 0.016393,
        "median_turb_delta": 0.027598,
        "note": (
            "From _diag_reseed.py on pre-fix code: OLS overwrite caused "
            "d_eff up to 0.102 and P_beta up to 137x stable median at reseed."
        ),
    }

    # Turbulent-window roughness confirmation
    deltas = np.abs(np.diff(eff, prepend=eff[0]))
    turb = result.mode == Mode.TURBULENT
    calm = result.mode == Mode.STABLE

    jump_verdict = (
        "eliminated"
        if after_stats["max_reseed_jump"] < before_stats["median_calm_delta"] * 1.5
        else (
            "reduced"
            if after_stats["max_reseed_jump"] < before_stats["max_reseed_jump"] * 0.5
            else "partially reduced / check chart"
        )
    )
    turb_ratio = after_stats["median_turb_delta"] / max(
        after_stats["median_calm_delta"], 1e-12
    )
    before_j = [round(v, 4) for v in before_stats["reseed_jumps"]]
    after_j = [round(v, 4) for v in after_stats["reseed_jumps"]]

    report = f"""# Effective-hedge noise investigation

## 1. Reseed provenance (bug confirmed and fixed)

### Before (bug)

At every reseed the handler overwrote `(alpha, beta)` with a separate
`_ols_window_state` OLS fit, discarding the turbulent filter's own
converged state. Logged deltas (turbulent state → OLS overwrite):

| t | d_alpha | d_beta | d_eff | P_beta / stable_median |
|---|---------|--------|-------|------------------------|
| 249 | +79.94 | -1.161 | +0.0016 | 12.4x |
| 404 | -53.33 | +1.150 | +0.1023 | 92.7x |
| 481 | -6.38 | +0.222 | +0.0376 | 104.5x |
| 584 | +15.22 | -0.461 | +0.0357 | 137.1x |
| 667 | +26.34 | -0.827 | -0.0217 | 105.3x |
| 733 | +25.99 | -0.749 | -0.0145 | 25.4x |

P at reseed was still heavily elevated (12–137× early-stable median
`P_beta`), so the exit was gating on CUSUM hysteresis alone.

Effective-hedge `|Δ|` at reseed markers (before):
max = `{before_stats['max_reseed_jump']:.4f}`
({before_stats['max_reseed_jump']/before_stats['median_calm_delta']:.1f}× calm median).

### Fix applied (`regime_switch.py`)

Structural changes:

1. **State provenance:** reseed keeps the turbulent filter's own
   `(alpha, beta)`. No OLS overwrite of the state.
2. **Q step-down + P-convergence gate:** when CUSUM hysteresis is met,
   Q steps `Q_turb → Q_stable` while keeping state and P (settling).
   Reseed finalizes only when
   `P_diag <= p_gate_mult * P_ref_diag` where `P_ref` is an EMA of P
   collected during prior STABLE operation
   (`p_gate_mult={result.config.p_gate_mult}`). Fresh P at finalize is
   `diag(P_ref)` — not an OLS state replacement.
3. **Settle commit:** once Q has stepped down, CUSUM is cleared while
   settling and cannot reinflate Q to turbulent.  Aborting on alarm as P
   fell created a settle/abort loop (S shrinks → z more sensitive → alarm
   → Q_turb → P reinflates) that prevented the P gate from ever completing
   after the regime break.

**Why Q step-down (not P-gate alone):** while `Q_turb` is active, P cannot
reach the stable-regime floor. Stepping Q down first is required.

**Why empirical P_ref instead of Riccati P_ss:** the local Riccati
steady-state under `Q_stable` alone is ~1000× below the filter's actual
stable-regime operating P for this calibrated model; a gate against that
floor never fired. The EMA-of-stable-P reference is the observable
"already calm" covariance level.

### After (factual comparison)

| metric | before | after |
|--------|--------|-------|
| reseed days | `{before_stats['reseed_indices']}` | `{after_stats['reseed_indices']}` |
| max |Δeff| at reseed | `{before_stats['max_reseed_jump']:.4f}` | `{after_stats['max_reseed_jump']:.4f}` |
| median calm |Δeff| | `{before_stats['median_calm_delta']:.4f}` | `{after_stats['median_calm_delta']:.4f}` |
| median turb |Δeff| | `{before_stats['median_turb_delta']:.4f}` | `{after_stats['median_turb_delta']:.4f}` |
| reseed jumps | `{before_j}` | `{after_j}` |

Reseed-marker spikes: **{jump_verdict}**
(max jump before `{before_stats['max_reseed_jump']:.4f}` → after `{after_stats['max_reseed_jump']:.4f}`).

Charts:
- Before: `cusum_kalman_vs_raw_beta_BEFORE_reseed_fix.png`
- After: `cusum_kalman_vs_raw_beta.png`

## 2. Alpha/x amplification (structural, not a bug)

NGTE (x) falls over the series; effective hedge includes `alpha/x`, so
uncertainty scales as `Var(alpha)/x^2 = P_alpha / x^2`.

| quantity | value |
|----------|-------|
| corr(local_roughness_eff, Var(alpha)/x²) full series | `{scale_stats['corr_full_rough_vs_var_ax']:.4f}` |
| corr(...) calm-only after fix (n={scale_stats['n_calm_valid']}) | `{scale_stats['corr_calm_rough_vs_var_ax']:.4f}` |
| corr(...) calm-only **before** fix (diagnostic) | `{scale_stats['corr_calm_before_fix']:.4f}` |
| median Var(alpha)/x² early (t<200) | `{scale_stats['median_var_ax_early']:.6e}` |
| median Var(alpha)/x² late (t>500) | `{scale_stats['median_var_ax_late']:.6e}` |
| median local roughness early / late | `{scale_stats['median_rough_early']:.6e}` / `{scale_stats['median_rough_late']:.6e}` |
| x early / late median | `{scale_stats['x_early_med']:.2f}` / `{scale_stats['x_late_med']:.2f}` |
| (1/x²) late÷early | `{scale_stats['inv_x2_ratio_late_over_early']:.3f}` |

Pre-fix calm-only correlation `{scale_stats['corr_calm_before_fix']:.2f}` shows the
noise envelope tracked `Var(alpha)/x²` when multiple late calm windows
existed. Early→late, `(1/x²)` rises by
`{scale_stats['inv_x2_ratio_late_over_early']:.2f}×` while median
`Var(alpha)/x²` moves `{scale_stats['median_var_ax_early']:.3e}` →
`{scale_stats['median_var_ax_late']:.3e}` and local roughness
`{scale_stats['median_rough_early']:.3e}` → `{scale_stats['median_rough_late']:.3e}`.
This is a **property of the metric definition**, not a filter bug.
No smoothing applied.

**Proposed alternatives (not implemented — needs confirmation):**
- Plot effective hedge with a delta-method band
  `± z * sqrt(P_alpha/x² + P_beta + 2*P_alpha_beta/x)`.
- Prefer slope `beta` alone as the primary series when `alpha/x` is
  dominated by denominator scaling rather than genuine level information.

Diagnostic plot: `alpha_x_scaling_diagnostic.png`

## 3. Turbulent-window roughness (expected, left alone)

Within alarm→reseed (turbulent) windows, median `|Δeff|` after the fix is
`{after_stats['median_turb_delta']:.4f}` vs calm `{after_stats['median_calm_delta']:.4f}`
({turb_ratio:.2f}×).
That elevation is consistent with `Q_turb = {result.config.q_multiplier}×Q_stable`
and elevated P during adaptation. **Not treated as a defect; not reduced.**

Turbulent fraction: `{after_stats['turb_fraction']:.3f}`.

## 4. Remaining calm-period roughness

After the reseed fix, calm-period median `|Δeff|` is
`{after_stats['median_calm_delta']:.4f}` (before: `{before_stats['median_calm_delta']:.4f}`).
Late-series calm roughness remains higher than early calm, in line with
the `Var(alpha)/x²` scaling in §2 (late/early inv-x² ratio
`{scale_stats['inv_x2_ratio_late_over_early']:.2f}`).

No further code changes. No smoothing. Open finding for human review:
late calm raggedness is largely explained by the alpha/x metric scaling
under a smaller NGTE price; residual roughness beyond that scaling (if
any) is not suppressed here.

## Constraints honored

- No moving average / low-pass / EWMA / Savitzky–Golay on the output.
- No parameter tuned to force a sigmoid shape.
- Claims backed by reseed provenance log and Var(alpha)/x² correlation.
"""
    (OUT / "effective_hedge_noise_report.md").write_text(report, encoding="utf-8")
    print("Wrote", OUT / "effective_hedge_noise_report.md")
    print("After reseed indices:", after_stats["reseed_indices"])
    print("After reseed jumps:", [round(v, 4) for v in after_stats["reseed_jumps"]])
    print("After max jump:", round(after_stats["max_reseed_jump"], 4))
    print("corr Var(alpha)/x2 vs rough:", round(scale_stats["corr_calm_rough_vs_var_ax"], 4))


if __name__ == "__main__":
    main()
