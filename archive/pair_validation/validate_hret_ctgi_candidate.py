"""Full-portfolio validation of HRET/CTGI gate, threshold, and scale candidates."""

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


PAIR_NAME = "HRET/CTGI"


def replace_pair(configs, **trade_updates):
    return tuple(
        spec._replace(trade=spec.trade._replace(**trade_updates))
        if spec.name == PAIR_NAME
        else spec
        for spec in configs
    )


def remove_pair(configs):
    return tuple(spec for spec in configs if spec.name != PAIR_NAME)


def refresh(configs) -> None:
    dfil.apply_pair_cfgs(configs)
    strategy.DISCOVERY_I, strategy.DISCOVERY_J = strategy._discovery_universe()


def main(only: set[str] | None = None) -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    original = tuple(dfil.PAIR_CFGS)
    legacy = replace_pair(original, leg_scale=1.00)
    variants = {
        "selected_live": original,
        "legacy": legacy,
        "adf_off": replace_pair(original, require_adf=False),
        "stop300": replace_pair(original, z_stop=3.00),
        "adf_off_stop300": replace_pair(original, require_adf=False, z_stop=3.00),
        "entry075_exit035_on": replace_pair(
            original, z_entry=0.75, z_exit=0.35,
        ),
        "entry075_exit035_off": replace_pair(
            original, z_entry=0.75, z_exit=0.35, require_adf=False,
        ),
        "entry075_exit045_off": replace_pair(
            original, z_entry=0.75, z_exit=0.45, require_adf=False,
        ),
        "scale025": replace_pair(original, leg_scale=0.25),
        "scale040": replace_pair(original, leg_scale=0.40),
        "scale045": replace_pair(original, leg_scale=0.45),
        "scale050": replace_pair(original, leg_scale=0.50),
        "scale055": replace_pair(original, leg_scale=0.55),
        "scale060": replace_pair(original, leg_scale=0.60),
        "scale0625": replace_pair(original, leg_scale=0.625),
        "scale064": replace_pair(original, leg_scale=0.64),
        "scale065": replace_pair(original, leg_scale=0.65),
        "scale066": replace_pair(original, leg_scale=0.66),
        "scale0675": replace_pair(original, leg_scale=0.675),
        "scale070": replace_pair(original, leg_scale=0.70),
        "scale075": replace_pair(original, leg_scale=0.75),
        "scale080": replace_pair(original, leg_scale=0.80),
        "scale085": replace_pair(original, leg_scale=0.85),
        "scale090": replace_pair(original, leg_scale=0.90),
        "scale095": replace_pair(original, leg_scale=0.95),
        "adf_off_scale050": replace_pair(
            original, require_adf=False, leg_scale=0.50,
        ),
        "entry075_exit035_scale050": replace_pair(
            original, z_entry=0.75, z_exit=0.35, leg_scale=0.50,
        ),
        "entry075_exit035_scale065": replace_pair(
            original, z_entry=0.75, z_exit=0.35, leg_scale=0.65,
        ),
        "entry075_exit035_scale075": replace_pair(
            original, z_entry=0.75, z_exit=0.35, leg_scale=0.75,
        ),
        "removed": remove_pair(original),
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
