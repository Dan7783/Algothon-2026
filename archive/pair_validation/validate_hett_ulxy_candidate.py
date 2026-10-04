"""Full-portfolio validation of threshold-only HETT/ULXY candidates."""

from __future__ import annotations

import argparse
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


PAIR_NAME = "HETT/ULXY"


def replace_pair(configs, *, entry: float, exit_: float):
    return tuple(
        spec._replace(trade=spec.trade._replace(z_entry=entry, z_exit=exit_))
        if spec.name == PAIR_NAME
        else spec
        for spec in configs
    )


def refresh(configs) -> None:
    dfil.apply_pair_cfgs(configs)
    strategy.DISCOVERY_I, strategy.DISCOVERY_J = strategy._discovery_universe()


def main(only: set[str] | None = None) -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    original = tuple(dfil.PAIR_CFGS)
    legacy = replace_pair(original, entry=0.27, exit_=0.02)
    variants = {
        "selected_live": original,
        "legacy": legacy,
        "entry070_exit0075": replace_pair(original, entry=0.70, exit_=0.075),
        "entry0725_exit0075": replace_pair(original, entry=0.725, exit_=0.075),
        "entry0735_exit002": replace_pair(original, entry=0.735, exit_=0.02),
        "entry075_exit002": replace_pair(original, entry=0.75, exit_=0.02),
        "entry075_exit0025": replace_pair(original, entry=0.75, exit_=0.025),
        "entry075_exit003": replace_pair(original, entry=0.75, exit_=0.03),
        "entry075_exit0035": replace_pair(original, entry=0.75, exit_=0.035),
        "entry075_exit005": replace_pair(original, entry=0.75, exit_=0.05),
        "entry075_exit0075": replace_pair(original, entry=0.75, exit_=0.075),
        "entry075_exit010": replace_pair(original, entry=0.75, exit_=0.10),
        "entry075_exit015": replace_pair(original, entry=0.75, exit_=0.15),
        "entry0765_exit002": replace_pair(original, entry=0.765, exit_=0.02),
        "entry0775_exit002": replace_pair(original, entry=0.775, exit_=0.02),
    }
    try:
        for name, configs in variants.items():
            if only is not None and name not in only:
                continue
            refresh(configs)
            train = replay(prices, 251, 500)
            test = replay(prices, 501, 750)
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
                f"turn={test['turnover']:10.0f} halves={test['half_scores']}"
            )
    finally:
        refresh(original)
        strategy.apply_policy(dict(strategy.SELECTED_POLICY))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    main(set(args.only) if args.only else None)
