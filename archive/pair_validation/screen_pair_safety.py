"""Full-book screen for pair-local beta-freeze safety controls."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import audit_adaptive_controls as controls
import audit_robust_hybrid as accounting
import archive.old_tuners.daniel_filter as daniel
import robust_hybrid_v2 as candidate


OUT = Path("cusum_output/robust_hybrid/pair_safety_screen.json")


def run(prices, overrides):
    config = controls.configure(overrides)
    train = accounting.replay(prices, 251, 500)
    test = accounting.replay(prices, 501, 750)
    serial_config = {
        key: sorted(value) if isinstance(value, (set, frozenset)) else value
        for key, value in config.items()
    }
    return {"config": serial_config, "train": train, "test": test}


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    accounting.strategy = candidate
    baseline = run(prices, {})
    report: dict[str, object] = {"control": baseline, "freeze": {}}

    # A full-sample path count is a diagnostic only; performance remains split
    # into independently reset train/test windows below.
    log_prices = __import__("numpy").log(prices)
    cut_names = []
    for spec in daniel.PAIR_CFGS:
        filt = daniel.PairSegFilter(spec.filt)
        cuts = 0
        for t in range(log_prices.shape[1]):
            cuts += int(filt.step(log_prices[spec.y, t], log_prices[spec.x, t]).cut)
        if cuts:
            cut_names.append((spec.name, cuts))

    for name, cuts in cut_names:
        result = run(prices, {"FREEZE_PAIR_NAMES": frozenset({name})})
        result["cuts"] = cuts
        result["delta_train_score"] = (
            result["train"]["score"] - baseline["train"]["score"]
        )
        result["delta_test_score"] = (
            result["test"]["score"] - baseline["test"]["score"]
        )
        report["freeze"][name] = result
        print(
            f"{name:12s} cuts {cuts:2d}  score delta "
            f"{result['delta_train_score']:+7.2f} / {result['delta_test_score']:+7.2f}",
            flush=True,
        )

    controls.configure({})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
