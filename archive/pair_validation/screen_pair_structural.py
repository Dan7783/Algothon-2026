"""Full-book pair-local screen for cut guards and damped beta snaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_adaptive_controls as controls
import audit_robust_hybrid as accounting
import archive.old_tuners.daniel_filter as daniel
import robust_hybrid_v2 as candidate


OUT = Path("cusum_output/robust_hybrid/pair_structural_screen.json")


def serial(values):
    return {
        key: sorted(value) if isinstance(value, (set, frozenset)) else value
        for key, value in values.items()
    }


def run(prices, overrides):
    config = controls.configure(overrides)
    return {
        "config": serial(config),
        "train": accounting.replay(prices, 251, 500),
        "test": accounting.replay(prices, 501, 750),
    }


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    accounting.strategy = candidate
    baseline = run(prices, {})

    log_prices = np.log(prices)
    cut_names: list[tuple[str, int]] = []
    for spec in daniel.PAIR_CFGS:
        filt = daniel.PairSegFilter(spec.filt)
        cuts = sum(
            int(filt.step(log_prices[spec.y, t], log_prices[spec.x, t]).cut)
            for t in range(log_prices.shape[1])
        )
        if cuts:
            cut_names.append((spec.name, cuts))

    modes = {
        "cut_guard": lambda name: {
            "EXIT_ON_CUT_PAIR_NAMES": frozenset({name}),
            "STAND_DOWN_PAIR_NAMES": frozenset({name}),
            "CUT_COOLDOWN_PAIR_NAMES": frozenset({name}),
            "CUT_COOLDOWN_BARS": 5,
        },
        "filter_damping": lambda name: {
            **controls.FILTER_DAMPING,
            "FILTER_CONTROL_PAIR_NAMES": frozenset({name}),
        },
    }
    report: dict[str, object] = {"control": baseline}
    for mode, build in modes.items():
        report[mode] = {}
        print(f"[{mode}]", flush=True)
        for name, cuts in cut_names:
            result = run(prices, build(name))
            dt = result["train"]["score"] - baseline["train"]["score"]
            dv = result["test"]["score"] - baseline["test"]["score"]
            result.update(cuts=cuts, delta_train_score=dt, delta_test_score=dv)
            report[mode][name] = result
            print(f"{name:12s} cuts {cuts:2d}  {dt:+7.2f} / {dv:+7.2f}", flush=True)

    controls.configure({})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

