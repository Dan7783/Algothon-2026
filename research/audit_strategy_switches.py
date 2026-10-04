"""Audit binary and categorical execution switches in ``four_sigmas``.

Development replays are limited to days 1-750.  Every variant changes one
switch or one coherent instance of the per-pair ``vol_cut`` setting while
leaving all numerical strategy parameters unchanged.  Held-out evaluation is
an explicit second phase and accepts only a named, already-evaluated variant.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import algo12_equal_sleeve_research as evaluator
from research import benchmark_teamname


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "strategy_switch_audit"
DEVELOPMENT_END = 750
DEVELOPMENT_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
HELDOUT_WINDOWS = (
    ("TEST_A", 751, 875),
    ("TEST_B", 876, 1000),
    ("TEST", 751, 1000),
)
HALVES = (
    "TRAIN_A",
    "TRAIN_B",
    "VALIDATION_A",
    "VALIDATION_B",
)
HALF_TOLERANCE = 5.0


Mutation = Callable[[types.ModuleType], None]


def set_attr(name: str, value: object) -> Mutation:
    def mutate(strategy: types.ModuleType) -> None:
        setattr(strategy, name, value)

    return mutate


def set_vol_cut(
    *,
    main: bool | None = None,
    residual: bool | None = None,
) -> Mutation:
    def mutate(strategy: types.ModuleType) -> None:
        if main is not None:
            strategy.PAIR_CFGS = tuple(
                spec._replace(
                    filt=spec.filt._replace(vol_cut=main)
                )
                for spec in strategy.PAIR_CFGS
            )
        if residual is not None:
            strategy.ALGO_RESIDUAL_CFGS = tuple(
                spec._replace(
                    filt=spec.filt._replace(vol_cut=residual)
                )
                for spec in strategy.ALGO_RESIDUAL_CFGS
            )

    return mutate


def compose(*mutations: Mutation) -> Mutation:
    def mutate(strategy: types.ModuleType) -> None:
        for item in mutations:
            item(strategy)

    return mutate


ALGO_OFF = set_attr("USE_ALGO_RESIDUAL_BASKET", False)
BOOST_OFF = set_attr("PROTECTED_VOTE_BOOST", 1.0)
DEADLOCK_OFF = set_attr("DEADLOCK_RIDGE_OVERLAY_FRACTION", 0.0)
RESIDUAL_VOL_CUT_OFF = set_vol_cut(residual=False)


VARIANTS: dict[str, Mutation] = {
    "baseline": lambda strategy: None,
    "pair_votes_false": set_attr(
        "USE_PAIR_CONFLICT_VOTES",
        False,
    ),
    "ridge_votes_false": set_attr(
        "USE_RIDGE_CONFLICT_VOTES",
        False,
    ),
    "algo_residual_false": ALGO_OFF,
    "pair_conflict_priority": set_attr(
        "PAIR_CONFLICT_MODE",
        "priority",
    ),
    "protected_boost_off": BOOST_OFF,
    "deadlock_fill_off": DEADLOCK_OFF,
    "main_pair_vol_cut_false": set_vol_cut(main=False),
    "residual_vol_cut_false": RESIDUAL_VOL_CUT_OFF,
    "all_vol_cut_false": set_vol_cut(main=False, residual=False),
}

INTERACTIONS: dict[str, Mutation] = {
    "algo_off__boost_off": compose(ALGO_OFF, BOOST_OFF),
    "algo_off__deadlock_off": compose(ALGO_OFF, DEADLOCK_OFF),
    "algo_off__boost_off__deadlock_off": compose(
        ALGO_OFF,
        BOOST_OFF,
        DEADLOCK_OFF,
    ),
    "residual_vol_cut_off__boost_off": compose(
        RESIDUAL_VOL_CUT_OFF,
        BOOST_OFF,
    ),
    "residual_vol_cut_off__deadlock_off": compose(
        RESIDUAL_VOL_CUT_OFF,
        DEADLOCK_OFF,
    ),
    "residual_vol_cut_off__boost_off__deadlock_off": compose(
        RESIDUAL_VOL_CUT_OFF,
        BOOST_OFF,
        DEADLOCK_OFF,
    ),
    "boost_off__deadlock_off": compose(BOOST_OFF, DEADLOCK_OFF),
}


def load_prices(last_day: int | None) -> np.ndarray:
    prices = pd.read_csv(
        PRICE_PATH,
        sep=r"\s+",
    ).to_numpy(dtype=float).T
    if last_day is not None:
        prices = prices[:, :last_day].copy()
    return prices


def load_variant(name: str) -> types.ModuleType:
    mutations = {**VARIANTS, **INTERACTIONS}
    if name not in mutations:
        raise KeyError(f"Unknown variant: {name}")
    strategy = benchmark_teamname.load_strategy("four_sigmas")
    mutations[name](strategy)
    strategy.reset()
    return strategy


def collect_targets(
    strategy: types.ModuleType,
    prices: np.ndarray,
) -> np.ndarray:
    targets = np.zeros(
        (prices.shape[1] + 1, strategy.N_INST),
        dtype=int,
    )
    for history_size in range(1, prices.shape[1]):
        targets[history_size] = strategy.getMyPosition(
            prices[:, :history_size]
        )
    strategy.reset()
    return targets


def flatten(
    variant: str,
    result: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"variant": variant}
    for window, metrics in result.items():
        for metric, value in metrics.items():
            row[f"{window}_{metric}"] = float(value)
    return row


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    baseline = result[result["variant"] == "baseline"].iloc[0]
    for window in (
        "TRAIN_A",
        "TRAIN_B",
        "TRAIN",
        "VALIDATION_A",
        "VALIDATION_B",
        "VALIDATION",
    ):
        result[f"{window}_delta"] = (
            result[f"{window}_score"]
            - float(baseline[f"{window}_score"])
        )
    result["worst_half_delta"] = result[
        [f"{window}_delta" for window in HALVES]
    ].min(axis=1)
    result["positive_halves"] = (
        result[[f"{window}_delta" for window in HALVES]] > 0.0
    ).sum(axis=1)
    result["passes"] = (
        (result["TRAIN_delta"] > 0.0)
        & (result["VALIDATION_delta"] > 0.0)
        & (result["worst_half_delta"] >= -HALF_TOLERANCE)
    )
    result.loc[result["variant"] == "baseline", "passes"] = True
    return result


def inventory() -> dict[str, object]:
    strategy = load_variant("baseline")
    main_vol_cuts = [
        bool(spec.filt.vol_cut) for spec in strategy.PAIR_CFGS
    ]
    residual_vol_cuts = [
        bool(spec.filt.vol_cut)
        for spec in strategy.ALGO_RESIDUAL_CFGS
    ]
    return {
        "top_level_feature_switches": {
            "USE_PAIR_CONFLICT_VOTES": bool(
                strategy.USE_PAIR_CONFLICT_VOTES
            ),
            "USE_RIDGE_CONFLICT_VOTES": bool(
                strategy.USE_RIDGE_CONFLICT_VOTES
            ),
            "USE_ALGO_RESIDUAL_BASKET": bool(
                strategy.USE_ALGO_RESIDUAL_BASKET
            ),
        },
        "categorical_switches": {
            "PAIR_CONFLICT_MODE": strategy.PAIR_CONFLICT_MODE,
        },
        "binary_numeric_switches": {
            "DEADLOCK_RIDGE_OVERLAY_FRACTION": float(
                strategy.DEADLOCK_RIDGE_OVERLAY_FRACTION
            ),
            "PROTECTED_VOTE_BOOST": float(
                strategy.PROTECTED_VOTE_BOOST
            ),
        },
        "per_pair_boolean": {
            "field": "FiltCfg.vol_cut",
            "main_pair_count": len(main_vol_cuts),
            "main_true_count": int(sum(main_vol_cuts)),
            "residual_pair_count": len(residual_vol_cuts),
            "residual_true_count": int(sum(residual_vol_cuts)),
        },
        "not_configuration_switches": [
            "StepOut.cut",
            "StepOut.adapting",
            "StepOut.vol_shift",
            "ridge_gate_pass",
            "algo_beta_ready",
            "local exit_trade",
        ],
    }


def development() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "inventory.json").write_text(
        json.dumps(inventory(), indent=2),
        encoding="utf-8",
    )
    prices = load_prices(DEVELOPMENT_END)
    rows = []
    for name in VARIANTS:
        strategy = load_variant(name)
        targets = collect_targets(strategy, prices)
        metrics = evaluator.evaluate(
            prices,
            targets,
            DEVELOPMENT_WINDOWS,
        )
        rows.append(flatten(name, metrics))
        checkpoint = add_deltas(pd.DataFrame(rows))
        checkpoint.to_csv(
            OUTPUT_DIR / "development.csv",
            index=False,
        )
        print(
            name,
            f"train={metrics['TRAIN']['score']:.6f}",
            f"validation={metrics['VALIDATION']['score']:.6f}",
            flush=True,
        )


def interactions() -> None:
    """Test only plausible weak-switch interactions on development data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices(DEVELOPMENT_END)
    rows = []
    for name in ("baseline", *INTERACTIONS):
        strategy = load_variant(name)
        targets = collect_targets(strategy, prices)
        metrics = evaluator.evaluate(
            prices,
            targets,
            DEVELOPMENT_WINDOWS,
        )
        rows.append(flatten(name, metrics))
        checkpoint = add_deltas(pd.DataFrame(rows))
        checkpoint.to_csv(
            OUTPUT_DIR / "interactions.csv",
            index=False,
        )
        print(
            name,
            f"train={metrics['TRAIN']['score']:.6f}",
            f"validation={metrics['VALIDATION']['score']:.6f}",
            flush=True,
        )


def heldout(candidate: str) -> None:
    development_paths = (
        OUTPUT_DIR / "development.csv",
        OUTPUT_DIR / "interactions.csv",
    )
    existing = [path for path in development_paths if path.exists()]
    if not existing:
        raise FileNotFoundError("Run development phase first")
    development_frame = pd.concat(
        (pd.read_csv(path) for path in existing),
        ignore_index=True,
    )
    row = development_frame[
        development_frame["variant"] == candidate
    ]
    if row.empty:
        raise ValueError(
            f"{candidate!r} was not development-evaluated"
        )
    if not bool(row.iloc[0]["passes"]):
        raise ValueError(
            f"{candidate!r} did not pass the development rule"
        )
    prices = load_prices(None)
    rows = []
    for name in ("baseline", candidate):
        strategy = load_variant(name)
        targets = collect_targets(strategy, prices)
        metrics = evaluator.evaluate(
            prices,
            targets,
            HELDOUT_WINDOWS,
        )
        rows.append(flatten(name, metrics))
    heldout_frame = pd.DataFrame(rows)
    baseline = heldout_frame[
        heldout_frame["variant"] == "baseline"
    ].iloc[0]
    for window in ("TEST_A", "TEST_B", "TEST"):
        heldout_frame[f"{window}_delta"] = (
            heldout_frame[f"{window}_score"]
            - float(baseline[f"{window}_score"])
        )
    heldout_frame.to_csv(
        OUTPUT_DIR / f"heldout_{candidate}.csv",
        index=False,
    )
    print(
        heldout_frame[
            [
                "variant",
                "TEST_A_score",
                "TEST_B_score",
                "TEST_score",
                "TEST_delta",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("development", "interactions", "heldout"),
        required=True,
    )
    parser.add_argument(
        "--candidate",
        choices=tuple({**VARIANTS, **INTERACTIONS}),
    )
    args = parser.parse_args()
    if args.phase == "development":
        development()
        return
    if args.phase == "interactions":
        interactions()
        return
    if args.candidate is None:
        parser.error("--candidate is required for heldout")
    heldout(args.candidate)


if __name__ == "__main__":
    main()
