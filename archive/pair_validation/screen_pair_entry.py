"""Full-book screen for a pair-local 0.5-sigma minimum entry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import audit_adaptive_controls as controls
import audit_robust_hybrid as accounting
import archive.old_tuners.daniel_filter as daniel
import robust_hybrid_v2 as candidate


OUT = Path("cusum_output/robust_hybrid/pair_entry_screen.json")


def run(prices, overrides):
    config = controls.configure(overrides)
    train = accounting.replay(prices, 251, 500)
    test = accounting.replay(prices, 501, 750)
    serial = {
        key: (
            dict(value) if isinstance(value, dict)
            else sorted(value) if isinstance(value, (set, frozenset))
            else value
        )
        for key, value in config.items()
    }
    return {"config": serial, "train": train, "test": test}


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    accounting.strategy = candidate
    baseline = run(prices, {})
    report: dict[str, object] = {"control": baseline, "floor_050": {}}

    for spec in daniel.PAIR_CFGS:
        if spec.trade.z_entry >= 0.50:
            continue
        result = run(prices, {"ENTRY_FLOOR_BY_PAIR": {spec.name: 0.50}})
        dt = result["train"]["score"] - baseline["train"]["score"]
        dv = result["test"]["score"] - baseline["test"]["score"]
        result.update(
            original_entry=spec.trade.z_entry,
            delta_train_score=dt,
            delta_test_score=dv,
            half_deltas={
                "train": [
                    result["train"]["half_scores"][i]
                    - baseline["train"]["half_scores"][i]
                    for i in range(2)
                ],
                "test": [
                    result["test"]["half_scores"][i]
                    - baseline["test"]["half_scores"][i]
                    for i in range(2)
                ],
            },
        )
        report["floor_050"][spec.name] = result
        print(
            f"{spec.name:12s} {spec.trade.z_entry:.3f}->.500  "
            f"{dt:+7.2f} / {dv:+7.2f}  halves "
            f"{min(result['half_deltas']['train']):+6.2f}/"
            f"{min(result['half_deltas']['test']):+6.2f}",
            flush=True,
        )

    controls.configure({})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

