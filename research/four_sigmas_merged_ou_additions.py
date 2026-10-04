"""Development-only portability test for merged_friend_ou additions.

The script loads only days 1-750.  It keeps the current four_sigmas pair
catalogue and portfolio layer, then tests:

* conflict-margin, magnitude-vote, and flip-to-flat execution variants;
* the OU engine with the current pair filter parameters;
* all merged OU profiles together;
* each merged OU profile as a one-at-a-time replacement.

No candidate is written into either production strategy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as current
import merged_friend_ou as merged
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "four_sigmas_ou_additions"
WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
_PRICES: np.ndarray | None = None
_ORIGINAL_TUNED_OU = tuple(
    spec for spec in merged.PAIR_CFGS if spec.engine == "ou"
)


def _name(spec: object) -> str:
    symbols = {index: symbol for symbol, index in current.I.items()}
    return f"{symbols[int(spec.y)]}-{symbols[int(spec.x)]}"


def _base_configs() -> tuple[merged.PairCfg, ...]:
    configs: list[merged.PairCfg] = []
    for idx, spec in enumerate(current.PAIR_CFGS):
        configs.append(
            merged.PairCfg(
                idx=idx,
                y=int(spec.y),
                x=int(spec.x),
                filt=merged.FiltCfg(
                    q_beta=float(spec.filt.q_beta),
                    ewma_lambda=float(spec.filt.ewma_lambda),
                    kalman_var_ewma_alpha=None,
                    cusum_k=float(spec.filt.cusum_k),
                    h=float(spec.filt.h),
                    vol_cut=bool(spec.filt.vol_cut),
                    max_cov_inflate=float(spec.filt.max_cov_inflate),
                    cov_decay=float(spec.filt.cov_decay),
                ),
                trade=merged.TradeCfg(
                    z_entry=float(spec.trade.z_entry),
                    z_exit=float(spec.trade.z_exit),
                    z_stop=float(spec.trade.z_stop),
                ),
                ou=merged.OUCfg(),
                engine="zband",
            )
        )
    return tuple(configs)


def _as_default_ou(spec: merged.PairCfg) -> merged.PairCfg:
    return merged.PairCfg(
        idx=int(spec.idx),
        y=int(spec.y),
        x=int(spec.x),
        filt=spec.filt,
        trade=None,
        ou=merged.OUCfg(),
        engine="ou",
    )


def _tuned_ou_map() -> dict[str, merged.PairCfg]:
    return {
        _name(spec): spec
        for spec in _ORIGINAL_TUNED_OU
    }


def _configure(variant: str) -> None:
    merged.CONSENSUS_MARGIN = 0.0
    merged.MAG_WEIGHTED_VOTES = False
    merged.FLIP_TO_FLAT = False
    merged.OU_CONFLICT_MODE = "shared"
    merged.OU_EXIT_MODE = "optimal"

    configs = list(_base_configs())
    tuned = _tuned_ou_map()

    if variant.startswith("margin_"):
        merged.CONSENSUS_MARGIN = float(variant.split("_", 1)[1])
    elif variant == "magnitude_weighted":
        merged.MAG_WEIGHTED_VOTES = True
    elif variant == "flip_to_flat":
        merged.FLIP_TO_FLAT = True
    elif variant == "winner_default_ou":
        for idx, spec in enumerate(configs):
            if _name(spec) in tuned:
                configs[idx] = _as_default_ou(spec)
    elif variant == "all_default_ou":
        configs = [_as_default_ou(spec) for spec in configs]
    elif variant == "all_tuned_ou":
        for idx, spec in enumerate(configs):
            replacement = tuned.get(_name(spec))
            if replacement is not None:
                configs[idx] = replacement._replace(idx=idx)
    elif variant.startswith("tuned_ou:"):
        pair = variant.split(":", 1)[1]
        for idx, spec in enumerate(configs):
            if _name(spec) == pair:
                configs[idx] = tuned[pair]._replace(idx=idx)
                break
    elif variant != "baseline":
        raise ValueError(f"Unknown variant: {variant}")

    merged.PAIR_CFGS = tuple(configs)
    merged.reset()


def _worker_init() -> None:
    global _PRICES
    _PRICES = (
        pd.read_csv(PRICE_PATH, sep=r"\s+")
        .to_numpy(dtype=float)
        .T[:, :750]
    )


def _evaluate_variant(variant: str) -> list[dict[str, float | str]]:
    if _PRICES is None:
        _worker_init()
    assert _PRICES is not None
    _configure(variant)
    targets = np.zeros(
        (_PRICES.shape[1] + 1, current.N_INST),
        dtype=int,
    )
    for history_size in range(1, _PRICES.shape[1]):
        targets[history_size] = merged.getMyPosition(
            _PRICES[:, :history_size]
        )
    metrics = evaluator.evaluate(_PRICES, targets, WINDOWS)
    return [
        {
            "variant": variant,
            "window": window,
            **values,
        }
        for window, values in metrics.items()
    ]


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = [
            f"{float(value):.{digits}f}"
            if isinstance(value, (float, np.floating))
            else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margins-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tuned_names = sorted(_tuned_ou_map())
    if args.margins_only:
        variants = [
            "baseline",
            "margin_0.10",
            "margin_0.14",
            "margin_0.145",
            "margin_0.15",
            "margin_0.16",
            "margin_0.18",
            "margin_0.199",
            "margin_0.201",
            "margin_0.25",
        ]
    else:
        variants = [
            "baseline",
            "margin_0.05",
            "margin_0.10",
            "margin_0.15",
            "magnitude_weighted",
            "flip_to_flat",
            "winner_default_ou",
            "all_default_ou",
            "all_tuned_ou",
            *(f"tuned_ou:{name}" for name in tuned_names),
        ]

    rows: list[dict[str, float | str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4,
        initializer=_worker_init,
    ) as executor:
        futures = {
            executor.submit(_evaluate_variant, variant): variant
            for variant in variants
        }
        for future in concurrent.futures.as_completed(futures):
            variant = futures[future]
            result = future.result()
            rows.extend(result)
            scores = {
                str(row["window"]): float(row["score"])
                for row in result
            }
            print(
                f"{variant}: train={scores['TRAIN']:.2f}, "
                f"validation={scores['VALIDATION']:.2f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    baseline = (
        results[results["variant"] == "baseline"]
        .set_index("window")["score"]
    )
    results["score_delta"] = results.apply(
        lambda row: float(
            row["score"] - baseline.loc[str(row["window"])]
        ),
        axis=1,
    )
    suffix = "_margin_grid" if args.margins_only else ""
    results.to_csv(
        OUTPUT_DIR / f"results{suffix}.csv",
        index=False,
    )

    pivot = results.pivot(
        index="variant",
        columns="window",
        values="score_delta",
    ).reset_index()
    ordered = [
        "variant",
        "TRAIN_A",
        "TRAIN_B",
        "TRAIN",
        "VALIDATION_A",
        "VALIDATION_B",
        "VALIDATION",
    ]
    pivot = pivot[ordered]
    pivot["all_four_halves_positive"] = (
        (pivot["TRAIN_A"] > 0.0)
        & (pivot["TRAIN_B"] > 0.0)
        & (pivot["VALIDATION_A"] > 0.0)
        & (pivot["VALIDATION_B"] > 0.0)
    )
    pivot = pivot.sort_values(
        ["all_four_halves_positive", "VALIDATION", "TRAIN"],
        ascending=False,
    )
    pivot.to_csv(
        OUTPUT_DIR / f"score_deltas{suffix}.csv",
        index=False,
    )

    concise = pivot[
        (pivot["variant"] == "baseline")
        | (~pivot["variant"].str.startswith("tuned_ou:"))
        | pivot["all_four_halves_positive"]
    ]
    report = [
        "# Portable additions from merged_friend_ou",
        "",
        "Only days 1-750 are loaded. No exposed result is available here.",
        "",
        "All values below are score deltas from the unchanged current book.",
        "",
        markdown_table(concise, digits=2),
        "",
        "A one-at-a-time OU replacement is considered structurally consistent",
        "only if both train halves and both validation halves improve.",
    ]
    text = "\n".join(report) + "\n"
    (OUTPUT_DIR / f"report{suffix}.md").write_text(
        text,
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
