"""Run the full CUSUM + regime-switch pipeline (ARL -> injection -> SMAH/NGTE)."""

from __future__ import annotations

from research.cusum_legacy.arl_calibration import run_arl_calibration
from research.cusum_legacy.injection_test import run_smah_ngte_injection
from research.cusum_legacy.regime_switch import run_smah_ngte_worked_example


def main() -> None:
    print("=== 1. Pooled ARL calibration ===")
    arl = run_arl_calibration()

    print("\n=== 2. Synthetic injection tests ===")
    inj = run_smah_ngte_injection(
        h=arl.chosen_h,
        kappa=arl.kappa,
        calibrated_arl0=arl.chosen_arl0,
    )

    print("\n=== 3. SMAH/NGTE full-series worked example ===")
    result, cal, meta = run_smah_ngte_worked_example(
        h=arl.chosen_h,
        kappa=arl.kappa,
        arl0=arl.chosen_arl0,
    )
    # attach ARL0 into meta report already written; print summary
    print(f"h={meta['h']}, ARL0={arl.chosen_arl0:.1f}")
    print(f"first CUSUM alarm: {meta['first_alarm']}")
    print(f"first reseed:      {meta['first_reseed']}")
    print(f"legacy |z| gate:   {meta['first_legacy']}")
    print(f"injection report:  {inj.output_dir / 'injection_test_report.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
