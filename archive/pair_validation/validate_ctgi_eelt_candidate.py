"""Full-portfolio validation for CTGI/EELT beta/threshold candidates."""

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


PAIR_NAME = "CTGI/EELT"
OUT_PATH = Path("cusum_output/pair_seed_explorer/ctgi_eelt/portfolio_validation.json")


def replace_pair(configs, *, q_multiplier=1.0, filt_updates=None, trade_updates=None):
    filt_updates = {} if filt_updates is None else dict(filt_updates)
    trade_updates = {} if trade_updates is None else dict(trade_updates)
    replaced = []
    for spec in configs:
        if spec.name != PAIR_NAME:
            replaced.append(spec)
            continue
        filt = spec.filt._replace(
            q_beta=float(spec.filt.q_beta * q_multiplier), **filt_updates
        )
        trade = spec.trade._replace(**trade_updates)
        replaced.append(spec._replace(filt=filt, trade=trade))
    return tuple(replaced)


def remove_pair(configs):
    return tuple(spec for spec in configs if spec.name != PAIR_NAME)


def refresh(configs) -> None:
    dfil.apply_pair_cfgs(configs)
    strategy.DISCOVERY_I, strategy.DISCOVERY_J = strategy._discovery_universe()


def main(only: set[str] | None = None) -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    original = tuple(dfil.PAIR_CFGS)
    threshold = {"z_entry": 0.325, "z_exit": 0.125, "z_stop": 6.0}
    balanced = {"z_entry": 0.35, "z_exit": 0.05, "z_stop": 5.0}
    variants = {
        "live": original,
        "legacy_entry030": replace_pair(
            original, trade_updates={"z_entry": 0.30},
        ),
        "balanced_live_q": replace_pair(original, trade_updates=balanced),
        "entry_only": replace_pair(
            original, trade_updates={"z_entry": 0.35},
        ),
        "balanced_entry0325": replace_pair(
            original,
            trade_updates={"z_entry": 0.325, "z_exit": 0.05, "z_stop": 5.0},
        ),
        "balanced_entry0375": replace_pair(
            original,
            trade_updates={"z_entry": 0.375, "z_exit": 0.05, "z_stop": 5.0},
        ),
        "balanced_stop400": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.05, "z_stop": 4.0},
        ),
        "balanced_stop450": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.05, "z_stop": 4.5},
        ),
        "balanced_stop550": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.05, "z_stop": 5.5},
        ),
        "balanced_stop600": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.05, "z_stop": 6.0},
        ),
        "balanced_exit0075": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.075, "z_stop": 5.0},
        ),
        "balanced_exit0100": replace_pair(
            original,
            trade_updates={"z_entry": 0.35, "z_exit": 0.10, "z_stop": 5.0},
        ),
        "slow025_balanced": replace_pair(
            original, q_multiplier=0.25,
            trade_updates={"z_entry": 0.35, "z_exit": 0.05, "z_stop": 6.0},
        ),
        "threshold_live_q": replace_pair(original, trade_updates=threshold),
        "q16_threshold": replace_pair(original, q_multiplier=16.0, trade_updates=threshold),
        "q20_threshold": replace_pair(original, q_multiplier=20.0, trade_updates=threshold),
        "q32_threshold": replace_pair(original, q_multiplier=32.0, trade_updates=threshold),
        "q64_threshold": replace_pair(original, q_multiplier=64.0, trade_updates=threshold),
        "q128_threshold": replace_pair(original, q_multiplier=128.0, trade_updates=threshold),
        "q256_threshold": replace_pair(original, q_multiplier=256.0, trade_updates=threshold),
        "q32_live_threshold": replace_pair(original, q_multiplier=32.0),
        "q64_live_threshold": replace_pair(original, q_multiplier=64.0),
        "q32_entry030": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.30, "z_exit": 0.125, "z_stop": 6.0},
        ),
        "q32_entry035": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.35, "z_exit": 0.125, "z_stop": 6.0},
        ),
        "q32_exit010": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.325, "z_exit": 0.10, "z_stop": 6.0},
        ),
        "q32_exit015": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.325, "z_exit": 0.15, "z_stop": 6.0},
        ),
        "q32_stop450": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.325, "z_exit": 0.125, "z_stop": 4.5},
        ),
        "q32_stop500": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.325, "z_exit": 0.125, "z_stop": 5.0},
        ),
        "q32_stop550": replace_pair(
            original, q_multiplier=32.0,
            trade_updates={"z_entry": 0.325, "z_exit": 0.125, "z_stop": 5.5},
        ),
        "q32_scale075": replace_pair(
            original, q_multiplier=32.0, trade_updates={**threshold, "leg_scale": 0.75},
        ),
        "q32_scale085": replace_pair(
            original, q_multiplier=32.0, trade_updates={**threshold, "leg_scale": 0.85},
        ),
        "q32_scale090": replace_pair(
            original, q_multiplier=32.0, trade_updates={**threshold, "leg_scale": 0.90},
        ),
        "q32_scale095": replace_pair(
            original, q_multiplier=32.0, trade_updates={**threshold, "leg_scale": 0.95},
        ),
        "removed": remove_pair(original),
    }
    results: dict[str, dict[str, object]] = {}
    if only is not None and OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    try:
        for name, configs in variants.items():
            if only is not None and name not in only:
                continue
            refresh(configs)
            train = replay(prices, 251, 500)
            test = replay(prices, 501, 750)
            results[name] = {"train": train, "test": test}
            print(
                f"{name:22s} "
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
        refresh(original)
        strategy.apply_policy(dict(strategy.SELECTED_POLICY))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    main(set(args.only) if args.only else None)
