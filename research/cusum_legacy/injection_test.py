"""Synthetic break-injection harness for CUSUM + regime-switch validation.

Injects step/ramp shifts into a calm pair series, measures detection delay,
reseed success, and no-break false-alarm rate vs the calibrated ARL0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.cusum_legacy.calibrate import calibrate_pair, extract_pair, load_prices
from research.cusum_legacy.cusum import DEFAULT_KAPPA, run_length_to_alarm
from research.cusum_legacy.kalman_pair import run_kalman_pair
from research.cusum_legacy.regime_switch import (
    DEFAULT_HYSTERESIS_N,
    DEFAULT_Q_MULTIPLIER,
    run_regime_switch,
)


# ---- named tunable defaults ----
DEFAULT_N_TRIALS_PER_SCENARIO: int = 30
DEFAULT_N_NO_BREAK_TRIALS: int = 60
DEFAULT_MAX_FA_RUN: int = 8_000
DEFAULT_CALM_END: int = 170
DEFAULT_INJECT_MIN_FRAC: float = 0.35  # onset not too early
DEFAULT_INJECT_MAX_FRAC: float = 0.70  # leave room after break
DEFAULT_RESEED_OK_HORIZON: int = 80  # must reseed within this many obs of alarm
DEFAULT_RANDOM_SEED: int = 7


InjectKind = Literal["step", "ramp", "subtle_step"]


@dataclass(frozen=True)
class InjectScenario:
    name: str
    kind: InjectKind
    magnitude: float  # beta-shift size (relationship break), not raw price units
    duration: int  # ramp length in observations; 1 => instantaneous step


# Minimum scenarios: fast step, slow ramp (SMAH/NGTE-like), subtler shift
DEFAULT_SCENARIOS: tuple[InjectScenario, ...] = (
    InjectScenario("fast_step", "step", magnitude=0.20, duration=1),
    InjectScenario("slow_ramp", "ramp", magnitude=0.35, duration=40),
    InjectScenario("subtle_step", "subtle_step", magnitude=0.15, duration=1),
)


@dataclass
class ScenarioSummary:
    name: str
    kind: str
    magnitude: float
    duration: int
    n_trials: int
    mean_detection_delay: float
    median_detection_delay: float
    detection_rate: float
    reseed_success_rate: float


@dataclass
class InjectionReport:
    scenarios: list[ScenarioSummary]
    no_break_false_alarm_rate: float
    no_break_mean_arl: float
    calibrated_arl0: float
    calibrated_h: float
    kappa: float
    n_no_break_trials: int
    output_dir: Path


def inject_break(
    y: np.ndarray,
    x: np.ndarray,
    *,
    onset: int,
    kind: InjectKind,
    magnitude: float,
    duration: int,
) -> np.ndarray:
    """Inject a persistent hedge-ratio (beta) shift into y.

    A raw additive blip is absorbed in one Kalman update and does not stress
    CUSUM; a beta shift ``y <- y + delta(t) * x`` creates sustained innovations
    until the filter's beta catches up — the relevant structural-break case.
    """
    y2 = np.asarray(y, dtype=float).copy()
    x = np.asarray(x, dtype=float).ravel()
    T = y2.size
    if onset >= T:
        return y2

    if kind in ("step", "subtle_step"):
        y2[onset:] = y2[onset:] + magnitude * x[onset:]
    elif kind == "ramp":
        dur = max(int(duration), 1)
        end = min(onset + dur, T)
        # delta grows linearly from 0 to magnitude over the ramp window
        fracs = np.linspace(0.0, 1.0, end - onset, endpoint=True)
        y2[onset:end] = y2[onset:end] + (fracs * magnitude) * x[onset:end]
        if end < T:
            y2[end:] = y2[end:] + magnitude * x[end:]
    else:
        raise ValueError(f"Unknown inject kind: {kind}")
    return y2


def _detection_delay(alarm_events: np.ndarray, onset: int) -> int | None:
    idx = np.flatnonzero(alarm_events)
    after = idx[idx >= onset]
    if after.size == 0:
        return None
    return int(after[0] - onset)


def _reseed_ok(
    alarm_events: np.ndarray,
    reseed_events: np.ndarray,
    onset: int,
    *,
    horizon: int = DEFAULT_RESEED_OK_HORIZON,
) -> bool:
    idx_a = np.flatnonzero(alarm_events)
    after_a = idx_a[idx_a >= onset]
    if after_a.size == 0:
        return False
    alarm_t = int(after_a[0])
    idx_r = np.flatnonzero(reseed_events)
    after_r = idx_r[(idx_r >= alarm_t) & (idx_r <= alarm_t + horizon)]
    return after_r.size > 0


def run_injection_battery(
    y_calm: np.ndarray,
    x_calm: np.ndarray,
    stable_config,
    *,
    kappa: float,
    h: float,
    calibrated_arl0: float,
    scenarios: tuple[InjectScenario, ...] = DEFAULT_SCENARIOS,
    n_trials: int = DEFAULT_N_TRIALS_PER_SCENARIO,
    n_no_break: int = DEFAULT_N_NO_BREAK_TRIALS,
    max_fa_run: int = DEFAULT_MAX_FA_RUN,
    q_multiplier: float = DEFAULT_Q_MULTIPLIER,
    hysteresis_n: int = DEFAULT_HYSTERESIS_N,
    seed: int = DEFAULT_RANDOM_SEED,
    output_dir: str | Path = "cusum_output/injection_test",
) -> InjectionReport:
    """Run step/ramp/subtle injections + no-break FA estimation."""
    rng = np.random.default_rng(seed)
    y_calm = np.asarray(y_calm, dtype=float).ravel()
    x_calm = np.asarray(x_calm, dtype=float).ravel()
    T = y_calm.size
    lo = int(DEFAULT_INJECT_MIN_FRAC * T)
    hi = int(DEFAULT_INJECT_MAX_FRAC * T)

    summaries: list[ScenarioSummary] = []
    delay_rows: list[dict] = []

    for sc in scenarios:
        delays: list[int] = []
        detected = 0
        reseed_ok = 0
        for _ in range(n_trials):
            onset = int(rng.integers(lo, max(lo + 1, hi)))
            y_inj = inject_break(
                y_calm,
                x_calm,
                onset=onset,
                kind=sc.kind,
                magnitude=sc.magnitude,
                duration=sc.duration,
            )
            result = run_regime_switch(
                y_inj,
                x_calm,
                stable_config,
                kappa=kappa,
                h=h,
                q_multiplier=q_multiplier,
                hysteresis_n=hysteresis_n,
            )
            delay = _detection_delay(result.alarm_events, onset)
            if delay is not None:
                detected += 1
                delays.append(delay)
                if _reseed_ok(result.alarm_events, result.reseed_events, onset):
                    reseed_ok += 1
            delay_rows.append(
                {
                    "scenario": sc.name,
                    "onset": onset,
                    "delay": delay if delay is not None else np.nan,
                    "detected": delay is not None,
                }
            )

        summaries.append(
            ScenarioSummary(
                name=sc.name,
                kind=sc.kind,
                magnitude=sc.magnitude,
                duration=sc.duration,
                n_trials=n_trials,
                mean_detection_delay=float(np.mean(delays)) if delays else float("nan"),
                median_detection_delay=float(np.median(delays)) if delays else float("nan"),
                detection_rate=detected / n_trials,
                reseed_success_rate=reseed_ok / n_trials,
            )
        )
        print(
            f"  {sc.name}: detect={detected}/{n_trials}  "
            f"mean_delay={summaries[-1].mean_detection_delay:.1f}  "
            f"reseed_ok={reseed_ok}/{n_trials}"
        )

    # No-break false-alarm rate via CUSUM on stable-filter z (matches ARL def)
    fa_run_lengths: list[int] = []
    fa_count = 0
    for _ in range(n_no_break):
        # Bootstrap a long null path from calm z under the stable filter
        filt = run_kalman_pair(y_calm, x_calm, stable_config)
        z = filt.z[np.isfinite(filt.z)]
        # tile / resample to max_fa_run
        idx = rng.integers(0, z.size, size=max_fa_run)
        path = z[idx]
        rl = run_length_to_alarm(path, kappa=kappa, h=h)
        fa_run_lengths.append(rl)
        if rl < max_fa_run:
            fa_count += 1

    mean_arl = float(np.mean(fa_run_lengths))
    # Empirical false-alarm rate per observation ≈ 1 / mean ARL
    fa_rate = 1.0 / mean_arl if mean_arl > 0 else float("nan")
    target_fa_rate = 1.0 / calibrated_arl0 if calibrated_arl0 > 0 else float("nan")

    print(
        f"  no-break: mean ARL={mean_arl:.1f}  "
        f"FA/obs~{fa_rate:.6f}  (calibrated ARL0={calibrated_arl0:.1f}, "
        f"target FA/obs~{target_fa_rate:.6f})"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = InjectionReport(
        scenarios=summaries,
        no_break_false_alarm_rate=fa_rate,
        no_break_mean_arl=mean_arl,
        calibrated_arl0=calibrated_arl0,
        calibrated_h=h,
        kappa=kappa,
        n_no_break_trials=n_no_break,
        output_dir=out,
    )
    _write_injection_outputs(report, delay_rows)
    return report


def _write_injection_outputs(report: InjectionReport, delay_rows: list[dict]) -> None:
    out = report.output_dir
    df_sc = pd.DataFrame(
        [
            {
                "scenario": s.name,
                "kind": s.kind,
                "magnitude": s.magnitude,
                "duration": s.duration,
                "n_trials": s.n_trials,
                "mean_detection_delay": s.mean_detection_delay,
                "median_detection_delay": s.median_detection_delay,
                "detection_rate": s.detection_rate,
                "reseed_success_rate": s.reseed_success_rate,
            }
            for s in report.scenarios
        ]
    )
    df_sc.to_csv(out / "detection_delay_by_scenario.csv", index=False)
    pd.DataFrame(delay_rows).to_csv(out / "detection_delay_trials.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        df_sc["scenario"],
        df_sc["mean_detection_delay"],
        color="#1f4e79",
        alpha=0.85,
    )
    for i, row in df_sc.iterrows():
        ax.text(
            i,
            row["mean_detection_delay"] + 0.3,
            f"det={row['detection_rate']:.0%}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("Mean detection delay (observations)")
    ax.set_title("Injection test: detection delay vs scenario")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "detection_delay_vs_magnitude.png", dpi=140)
    plt.close(fig)

    lines = [
        "# Synthetic injection test report",
        "",
        f"- kappa: `{report.kappa}`",
        f"- h (ARL-calibrated): `{report.calibrated_h}`",
        f"- calibrated ARL0 target/achieved: `{report.calibrated_arl0:.1f}`",
        f"- no-break trials: `{report.n_no_break_trials}`",
        f"- no-break mean ARL (measured): `{report.no_break_mean_arl:.1f}`",
        f"- no-break false-alarm rate per obs: `{report.no_break_false_alarm_rate:.6g}`",
        f"- implied FA rate from calibrated ARL0: `{1.0 / report.calibrated_arl0:.6g}`",
        "",
        "Mismatch between measured no-break ARL and calibrated ARL0 is reported "
        "explicitly (bootstrap / finite-sample effects), not assumed away.",
        "",
        "## Detection delay by scenario",
        "",
        "| scenario | kind | mag | dur | mean delay | median delay | detect rate | reseed ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report.scenarios:
        lines.append(
            f"| {s.name} | {s.kind} | {s.magnitude:g} | {s.duration} | "
            f"{s.mean_detection_delay:.1f} | {s.median_detection_delay:.1f} | "
            f"{s.detection_rate:.2f} | {s.reseed_success_rate:.2f} |"
        )
    lines.append("")
    (out / "injection_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_smah_ngte_injection(
    prices_path: str | Path = "prices.txt",
    *,
    calm_end: int = DEFAULT_CALM_END,
    h: float,
    kappa: float = DEFAULT_KAPPA,
    calibrated_arl0: float,
    output_dir: str | Path = "cusum_output/injection_test",
) -> InjectionReport:
    """Injection battery on SMAH/NGTE calm window [0, calm_end)."""
    prices = load_prices(prices_path)
    y, x = extract_pair(prices, "SMAH", "NGTE")
    y_c, x_c = y[:calm_end], x[:calm_end]
    cal = calibrate_pair(
        y,
        x,
        y_ticker="SMAH",
        x_ticker="NGTE",
        calm_start=0,
        calm_end=calm_end,
    )
    print("Running injection battery on SMAH/NGTE calm series:")
    return run_injection_battery(
        y_c,
        x_c,
        cal.config,
        kappa=kappa,
        h=h,
        calibrated_arl0=calibrated_arl0,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    from research.cusum_legacy.arl_calibration import run_arl_calibration

    arl = run_arl_calibration()
    run_smah_ngte_injection(
        h=arl.chosen_h,
        kappa=arl.kappa,
        calibrated_arl0=arl.chosen_arl0,
    )
