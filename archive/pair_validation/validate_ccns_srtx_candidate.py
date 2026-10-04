"""Full-portfolio validation of defensible CCNS/SRTX decisions."""

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


PAIR_NAME = "CCNS/SRTX"


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
        trade=lambda trade: trade._replace(
            z_entry=0.44,
            z_exit=0.24,
            z_stop=4.75,
            leg_scale=1.0,
            require_adf=True,
            require_kappa=True,
        ),
    )
    variants = {
        "legacy_dormant": legacy,
        "selected_live": original,
        "removed": tuple(spec for spec in legacy if spec.name != PAIR_NAME),
        "adf_off_current": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(require_adf=False),
        ),
        "current_scale_025": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                leg_scale=0.25,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "current_scale_050": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                leg_scale=0.50,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_adf_off": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.125,
                z_stop=4.75,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_scale_025": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.125,
                z_stop=4.75,
                leg_scale=0.25,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_exit_015": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.15,
                z_stop=4.75,
                leg_scale=0.25,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "entry075_exit024_scale030": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.24,
                z_stop=4.75,
                leg_scale=0.30,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_scale_050": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.125,
                z_stop=4.75,
                leg_scale=0.50,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_scale_075": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.125,
                z_stop=4.75,
                leg_scale=0.75,
                require_adf=False,
                require_kappa=True,
            ),
        ),
        "minimal_gates_off": replace_pair(
            legacy,
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.125,
                z_stop=4.75,
                require_adf=False,
                require_kappa=False,
            ),
        ),
        "multi_parameter": replace_pair(
            legacy,
            filt=lambda filt: filt._replace(h=8.0, cusum_k=0.5),
            trade=lambda trade: trade._replace(
                z_entry=0.75,
                z_exit=0.10,
                z_stop=4.0,
                require_adf=False,
                require_kappa=False,
            ),
        ),
    }
    for scale in (0.10, 0.15, 0.20, 0.30, 0.35):
        variants[f"exit015_scale_{int(100 * scale):03d}"] = replace_pair(
            legacy,
            trade=lambda trade, scale=scale: trade._replace(
                z_entry=0.75,
                z_exit=0.15,
                z_stop=4.75,
                leg_scale=scale,
                require_adf=False,
                require_kappa=True,
            ),
        )
    try:
        for name, configs in variants.items():
            if only is not None and name not in only:
                continue
            refresh(configs)
            train = replay(prices, 251, 500)
            test = replay(prices, 501, 750)
            print(
                f"{name:20s} "
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
