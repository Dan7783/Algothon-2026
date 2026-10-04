"""Full-portfolio validation of simple HUXZ/ACAC candidates."""

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


PAIR_NAME = "HUXZ/ACAC"


def replace_pair(configs, *, filt=None, trade=None):
    return tuple(
        spec._replace(
            filt=spec.filt if filt is None else filt(spec.filt),
            trade=spec.trade if trade is None else trade(spec.trade),
        )
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
    legacy = replace_pair(
        original,
        trade=lambda trade: trade._replace(z_entry=0.50, z_exit=0.18),
    )
    variants = {
        "selected_live": original,
        "legacy": legacy,
        "entry060_exit018": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.18),
        ),
        "entry060_exit020": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.20),
        ),
        "entry060_exit0215": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.215),
        ),
        "entry060_exit022": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.22),
        ),
        "entry060_exit0225": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.225),
        ),
        "entry060_exit023": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.23),
        ),
        "entry060_exit0235": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.235),
        ),
        "entry060_exit025": replace_pair(
            original,
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.25),
        ),
        "historical_fast_beta": replace_pair(
            original,
            filt=lambda filt: filt._replace(
                q_beta=9.504e-05,
                h=4.5,
                cusum_k=0.30,
            ),
            trade=lambda trade: trade._replace(z_entry=0.625, z_exit=0.18),
        ),
        "q2_entry060": replace_pair(
            original,
            filt=lambda filt: filt._replace(q_beta=6.336e-05, h=8.5),
            trade=lambda trade: trade._replace(z_entry=0.60, z_exit=0.18),
        ),
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
