"""Full-portfolio validation for NAYO/SRTX allocation and timing candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (REPO_ROOT, REPO_ROOT / "research", REPO_ROOT / "archive"):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import archive.old_tuners.daniel_filter as dfil
import robust_hybrid_v2 as strategy
from validate_pair_seed_candidate import replay


PAIR_NAME = "NAYO/SRTX"
OUT_PATH = Path("cusum_output/pair_seed_explorer/nayo_srtx/portfolio_validation.json")


def replace_pair(
    configs,
    *,
    priority: float | None = None,
    filt_updates=None,
    trade_updates=None,
):
    filt_updates = {} if filt_updates is None else dict(filt_updates)
    trade_updates = {} if trade_updates is None else dict(trade_updates)
    replaced = []
    for spec in configs:
        if spec.name != PAIR_NAME:
            replaced.append(spec)
            continue
        filt = spec.filt._replace(**filt_updates)
        trade = spec.trade._replace(**trade_updates)
        replaced.append(
            spec._replace(
                priority=spec.priority if priority is None else float(priority),
                filt=filt,
                trade=trade,
            )
        )
    return tuple(replaced)


def remove_pair(configs):
    return tuple(spec for spec in configs if spec.name != PAIR_NAME)


def refresh(configs) -> None:
    dfil.apply_pair_cfgs(configs)
    strategy.DISCOVERY_I, strategy.DISCOVERY_J = strategy._discovery_universe()


def main(only: set[str] | None = None) -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    original = tuple(dfil.PAIR_CFGS)
    variants = {
        "live": original,
        "seed016_no_floor": original,
        "seed025_no_floor": replace_pair(original, trade_updates={"z_entry": 0.25}),
        "seed035_no_floor": replace_pair(original, trade_updates={"z_entry": 0.35}),
        "seed040_no_floor": replace_pair(original, trade_updates={"z_entry": 0.40}),
        "seed035_adf_off_no_floor": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "require_adf": False},
        ),
        "seed035_p022_no_floor": replace_pair(
            original, priority=22.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p025_no_floor": replace_pair(
            original, priority=25.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p030_no_floor": replace_pair(
            original, priority=30.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p032_no_floor": replace_pair(
            original, priority=32.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p033_no_floor": replace_pair(
            original, priority=33.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p034_no_floor": replace_pair(
            original, priority=34.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p035_no_floor": replace_pair(
            original, priority=35.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p036_no_floor": replace_pair(
            original, priority=36.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p037_no_floor": replace_pair(
            original, priority=37.0, trade_updates={"z_entry": 0.35},
        ),
        "seed035_p038_no_floor": replace_pair(
            original, priority=38.0, trade_updates={"z_entry": 0.35},
        ),
        "seed040_p022_no_floor": replace_pair(
            original, priority=22.0, trade_updates={"z_entry": 0.40},
        ),
        "seed040_p025_no_floor": replace_pair(
            original, priority=25.0, trade_updates={"z_entry": 0.40},
        ),
        "removed": remove_pair(original),
        "scale010": replace_pair(original, trade_updates={"leg_scale": 0.10}),
        "scale025": replace_pair(original, trade_updates={"leg_scale": 0.25}),
        "scale040": replace_pair(original, trade_updates={"leg_scale": 0.40}),
        "scale050": replace_pair(original, trade_updates={"leg_scale": 0.50}),
        "scale060": replace_pair(original, trade_updates={"leg_scale": 0.60}),
        "scale075": replace_pair(original, trade_updates={"leg_scale": 0.75}),
        "scale090": replace_pair(original, trade_updates={"leg_scale": 0.90}),
        "priority005": replace_pair(original, priority=5.0),
        "priority010": replace_pair(original, priority=10.0),
        "priority015": replace_pair(original, priority=15.0),
        "priority018": replace_pair(original, priority=18.0),
        "priority019": replace_pair(original, priority=19.0),
        "priority020": replace_pair(original, priority=20.0),
        "priority021": replace_pair(original, priority=21.0),
        "priority022": replace_pair(original, priority=22.0),
        "priority025": replace_pair(original, priority=25.0),
        "priority030": replace_pair(original, priority=30.0),
        "priority035": replace_pair(original, priority=35.0),
        "priority040": replace_pair(original, priority=40.0),
        "entry060": replace_pair(original, trade_updates={"z_entry": 0.60}),
        "entry075": replace_pair(original, trade_updates={"z_entry": 0.75}),
        "entry090": replace_pair(original, trade_updates={"z_entry": 0.90}),
        "entry110": replace_pair(original, trade_updates={"z_entry": 1.10}),
        "entry130": replace_pair(original, trade_updates={"z_entry": 1.30}),
        "adf_off": replace_pair(original, trade_updates={"require_adf": False}),
        "no_gates": replace_pair(
            original,
            trade_updates={"require_adf": False, "require_kappa": False},
        ),
    }
    results: dict[str, dict[str, object]] = {}
    if only is not None and OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    selected_policy = dict(strategy.SELECTED_POLICY)
    no_floor_variants = {
        "seed016_no_floor",
        "seed025_no_floor",
        "seed035_no_floor",
        "seed040_no_floor",
        "seed035_adf_off_no_floor",
        "seed035_p022_no_floor",
        "seed035_p025_no_floor",
        "seed035_p030_no_floor",
        "seed035_p032_no_floor",
        "seed035_p033_no_floor",
        "seed035_p034_no_floor",
        "seed035_p035_no_floor",
        "seed035_p036_no_floor",
        "seed035_p037_no_floor",
        "seed035_p038_no_floor",
        "seed040_p022_no_floor",
        "seed040_p025_no_floor",
    }
    try:
        for name, configs in variants.items():
            if only is not None and name not in only:
                continue
            policy = dict(selected_policy)
            if name in no_floor_variants:
                policy["ENTRY_FLOOR_BY_PAIR"] = {}
            strategy.SELECTED_POLICY = policy
            refresh(configs)
            train = replay(prices, 251, 500)
            test = replay(prices, 501, 750)
            results[name] = {"train": train, "test": test}
            print(
                f"{name:14s} "
                f"train score/mean/sharpe/sortino/dd/cvar="
                f"{train['score']:8.3f}/{train['mean']:8.3f}/"
                f"{train['sharpe']:6.3f}/{train['sortino']:6.3f}/"
                f"{train['max_drawdown']:8.1f}/{train['cvar_5']:8.1f} "
                f"turn={train['turnover']:10.0f} halves={train['half_scores']}  "
                f"test={test['score']:8.3f}/{test['mean']:8.3f}/"
                f"{test['sharpe']:6.3f}/{test['sortino']:6.3f}/"
                f"{test['max_drawdown']:8.1f}/{test['cvar_5']:8.1f} "
                f"turn={test['turnover']:10.0f} halves={test['half_scores']}",
                flush=True,
            )
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
        print(f"wrote {OUT_PATH}")
    finally:
        strategy.SELECTED_POLICY = selected_policy
        refresh(original)
        strategy.apply_policy(dict(strategy.SELECTED_POLICY))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    main(set(args.only) if args.only else None)
