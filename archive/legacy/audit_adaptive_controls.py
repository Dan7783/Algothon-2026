"""Ablate conservative adaptive-agent controls against the exact hybrid."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import audit_robust_hybrid as accounting
import robust_hybrid_v2 as candidate


OUT = Path("cusum_output/robust_hybrid/adaptive_controls.json")

CONTROL = {
    "OVERRIDE_FILTER_CONTROLS": False,
    "CUT_CONFIRM_BARS": 1,
    "CUT_COOLDOWN_BARS": 0,
    "STAND_DOWN_WHILE_ADAPTING": False,
    "FREEZE_BETA_WHILE_OPEN": False,
    "EXIT_ON_CUT": False,
    "ENTRY_FLOOR": 0.0,
    "ENTRY_CONFIRM_BARS": 1,
    "REQUIRE_TURN_CONFIRMATION": False,
    "STAGED_ENTRY": False,
    "ENTRY_SIZE": 1.0,
    "FULL_SIZE_Z": 1.25,
    "ALLOW_DIRECT_FLIP": True,
    "FILTER_CONTROL_PAIR_NAMES": None,
    "FREEZE_PAIR_NAMES": None,
    "EXIT_ON_CUT_PAIR_NAMES": None,
    "STAND_DOWN_PAIR_NAMES": None,
    "CUT_COOLDOWN_PAIR_NAMES": None,
    "ENTRY_FLOOR_BY_PAIR": None,
    "ENTRY_CONFIRM_PAIR_NAMES": None,
    "STAGED_ENTRY_PAIR_NAMES": None,
    "TURN_CONFIRM_PAIR_NAMES": None,
}

FILTER_DAMPING = {
    "OVERRIDE_FILTER_CONTROLS": True,
    "CUT_CONFIRM_BARS": 3,
    "SNAP_WINDOW": 40,
    "SNAP_OVERSHOOT": 1.0,
    "SNAP_MAX_REL": 0.03,
    "ADAPT_CHASE": 0.15,
    "ADAPT_Q_MULT": 50.0,
    "ADAPT_Q_FLOOR": 1.0e-6,
    "ADAPT_Q_CAP": 1.0e-4,
    "ADAPT_P_FLOOR": 1.0e-3,
}


def configure(overrides: dict[str, object]) -> dict[str, object]:
    values = dict(CONTROL)
    values.update(overrides)
    for key, value in values.items():
        setattr(candidate, key, value)
    return values


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    accounting.strategy = candidate
    variants = {
        "control": {},
        "freeze_beta": {"FREEZE_BETA_WHILE_OPEN": True},
        "cut_exit_cooldown": {
            "EXIT_ON_CUT": True,
            "CUT_COOLDOWN_BARS": 5,
        },
        "stand_down_adapt": {"STAND_DOWN_WHILE_ADAPTING": True},
        "freeze_cut_exit": {
            "FREEZE_BETA_WHILE_OPEN": True,
            "EXIT_ON_CUT": True,
            "CUT_COOLDOWN_BARS": 5,
        },
        "filter_damping": FILTER_DAMPING,
        "filter_damping_freeze": {
            **FILTER_DAMPING,
            "FREEZE_BETA_WHILE_OPEN": True,
        },
        "entry_floor_050": {"ENTRY_FLOOR": 0.50},
        "entry_confirm_2": {"ENTRY_CONFIRM_BARS": 2},
        "staged_entry": {
            "STAGED_ENTRY": True,
            "ENTRY_SIZE": 0.50,
            "FULL_SIZE_Z": 1.25,
        },
        "floor_050_confirm_2": {
            "ENTRY_FLOOR": 0.50,
            "ENTRY_CONFIRM_BARS": 2,
        },
        "floor_050_staged": {
            "ENTRY_FLOOR": 0.50,
            "STAGED_ENTRY": True,
            "ENTRY_SIZE": 0.50,
            "FULL_SIZE_Z": 1.25,
        },
        "turn_confirmation": {"REQUIRE_TURN_CONFIRMATION": True},
    }

    report: dict[str, object] = {}
    for name, overrides in variants.items():
        config = configure(overrides)
        train = accounting.replay(prices, 251, 500)
        test = accounting.replay(prices, 501, 750)
        report[name] = {"config": config, "train": train, "test": test}
        print(
            f"{name:25s} train {train['score']:8.2f}/{train['mean']:8.2f} "
            f"test {test['score']:8.2f}/{test['mean']:8.2f} "
            f"turn {train['turnover']/1e6:5.1f}/{test['turnover']/1e6:5.1f}m",
            flush=True,
        )

    configure({})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
