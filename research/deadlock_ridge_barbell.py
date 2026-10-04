#!/usr/bin/env python
"""Audit whether latent ridge forecasts are useful on pair-vote deadlocks.

The live strategy can leave an asset flat when:

1. two or more active pair legs vote in opposite directions; and
2. ridge is inactive, so it cannot break the exact pair-vote tie.

Ridge inactivity has two materially different causes:

* ``sub_threshold``: the global lag-link gate passed, but the asset forecast
  strength did not clear ``RIDGE_THETA``;
* ``gate_closed``: the global gate failed.  The live implementation returns
  zero strength for every asset in this state, even though the underlying
  VAR(1) ridge model can still be evaluated diagnostically.

This script reconstructs the ungated ridge prediction and its normalized
strength, measures its deadlock-only direction skill, and evaluates a small,
pre-declared family of overlays.  Candidate selection uses only days 251-500
and requires a positive score delta in both 125-day training halves.  Days
501-750 are validation.  Days 751-1000 are reported as an already-exposed,
contaminated final window and are never used for selection.

No strategy file is modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research.deadlock_ceiling import LIMIT, run_accounting, score


FIRST = 250
LAST = 999
OUTPUT_DIR = ROOT / "analysis_outputs" / "deadlock_ridge_barbell"
CACHE = OUTPUT_DIR / "ridge_deadlock_cache.npz"

WINDOWS = {
    "train_a_251_375": (251, 375),
    "train_b_376_500": (376, 500),
    "train_251_500": (251, 500),
    "validation_a_501_625": (501, 625),
    "validation_b_626_750": (626, 750),
    "validation_501_750": (501, 750),
    "contaminated_a_751_875": (751, 875),
    "contaminated_b_876_1000": (876, 1000),
    "contaminated_751_1000": (751, 1000),
}

STRENGTH_EDGES = np.asarray(
    [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, np.inf],
    dtype=float,
)
STRENGTH_LABELS = (
    "[0,.25)",
    "[.25,.50)",
    "[.50,.75)",
    "[.75,1)",
    "[1,1.5)",
    "[1.5,2)",
    "[2,3)",
    "[3,inf)",
)


def raw_ridge_diagnostics(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the live ridge model's ungated prediction and strength.

    The fit, scaling, regularization, and prediction scale match
    ``four_sigmas._ridge_forecast``.  Only the early return caused by the
    global lag-link gate and the per-asset entry threshold are omitted.
    """
    n_instruments, n_returns = returns.shape
    prediction = np.zeros(n_instruments)
    strength = np.zeros(n_instruments)
    if n_returns + 1 < strategy.RIDGE_MIN_T:
        return prediction, strength, 0

    recent = returns[1:, -250:].T
    standardized = (
        recent - recent.mean(axis=0)
    ) / (recent.std(axis=0) + 1.0e-12)
    lagged_links = (
        standardized[:-1].T @ standardized[1:]
    ) / (len(recent) - 1)
    np.fill_diagonal(lagged_links, 0.0)
    link_count = int(
        (
            np.abs(lagged_links)
            > 2.5 / np.sqrt(len(recent) - 1)
        ).sum()
    )

    history = returns[1:, :].T
    if strategy.RIDGE_LOOKBACK is not None:
        history = history[-strategy.RIDGE_LOOKBACK:]
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors
        + strategy.RIDGE_LAM
        * len(predictors)
        * np.eye(strategy.N_INST - 1),
        predictors.T @ targets,
    )
    prediction_scale = (predictors @ coefficients).std(axis=0)
    prediction[1:] = (history[-1] / scale) @ coefficients
    strength[1:] = np.abs(prediction[1:]) / np.maximum(
        prediction_scale,
        1.0e-12,
    )
    return prediction, strength, link_count


def build_cache(prices: np.ndarray) -> dict[str, np.ndarray]:
    """Replay the current strategy and capture pre-ridge pair cancellations."""
    strategy.reset()
    days = np.arange(FIRST, LAST + 1)
    shape = (len(days), prices.shape[0])
    base = np.zeros(shape)
    pair_cancel = np.zeros(shape, dtype=bool)
    unresolved = np.zeros(shape, dtype=bool)
    current_resolved = np.zeros(shape, dtype=bool)
    prediction = np.zeros(shape)
    strength = np.zeros(shape)
    gate_pass = np.zeros(len(days), dtype=bool)
    link_count = np.zeros(len(days), dtype=int)
    returns = np.zeros(shape)

    for row, day in enumerate(days):
        history = prices[:, :day]
        sleeves = strategy.sleeve_breakdown(history)
        if sleeves is None:
            continue

        base[row] = np.asarray(sleeves["dollars"], dtype=float)
        tied = (
            np.asarray(sleeves["pair_active_mask"], dtype=bool)
            & (np.abs(np.asarray(sleeves["pair_vote"], dtype=float)) < 1.0e-12)
            & ~np.asarray(sleeves["crash_mask"], dtype=bool)
        )
        tied[strategy.I["ALGO"]] = False
        ridge_active = np.asarray(
            sleeves["ridge_active_mask"],
            dtype=bool,
        )
        pair_cancel[row] = tied
        current_resolved[row] = tied & ridge_active
        unresolved[row] = tied & ~ridge_active

        raw_prediction, raw_strength, links = raw_ridge_diagnostics(
            np.diff(np.log(history), axis=1)
        )
        prediction[row] = raw_prediction
        strength[row] = raw_strength
        link_count[row] = links
        gate_pass[row] = links >= strategy.RIDGE_GATE_N
        returns[row] = prices[:, day] / prices[:, day - 1] - 1.0

        if gate_pass[row]:
            live_strength = np.asarray(
                sleeves["ridge_strength"],
                dtype=float,
            )
            if not np.allclose(live_strength, raw_strength, atol=1.0e-10):
                raise RuntimeError(
                    f"Ungated ridge reconstruction diverged on day {day}"
                )

    return {
        "days": days,
        "base": base,
        "pair_cancel": pair_cancel,
        "unresolved": unresolved,
        "current_resolved": current_resolved,
        "prediction": prediction,
        "strength": strength,
        "gate_pass": gate_pass,
        "link_count": link_count,
        "returns": returns,
    }


def event_frame(cache: dict[str, np.ndarray]) -> pd.DataFrame:
    """Flatten pair-cancellation asset-days into an auditable event table."""
    rows: list[dict[str, object]] = []
    symbols = {index: symbol for symbol, index in strategy.I.items()}
    for row, instrument in np.argwhere(cache["pair_cancel"]):
        prediction = float(cache["prediction"][row, instrument])
        realised_return = float(cache["returns"][row, instrument])
        gate_pass = bool(cache["gate_pass"][row])
        resolved = bool(cache["current_resolved"][row, instrument])
        rows.append(
            {
                "day": int(cache["days"][row]),
                "instrument": int(instrument),
                "symbol": symbols[int(instrument)],
                "state": (
                    "current_ridge_resolved"
                    if resolved
                    else "sub_threshold"
                    if gate_pass
                    else "gate_closed"
                ),
                "gate_pass": gate_pass,
                "link_count": int(cache["link_count"][row]),
                "current_ridge_resolved": resolved,
                "unresolved": bool(cache["unresolved"][row, instrument]),
                "raw_prediction": prediction,
                "raw_strength": float(
                    cache["strength"][row, instrument]
                ),
                "realised_return": realised_return,
                "hit": bool(
                    prediction != 0.0
                    and realised_return != 0.0
                    and np.sign(prediction) == np.sign(realised_return)
                ),
                "gross_full_leg_pnl": float(
                    LIMIT[instrument]
                    * np.sign(prediction)
                    * realised_return
                ),
            }
        )
    return pd.DataFrame(rows)


def add_window(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["window"] = "outside"
    for name, (start, end) in WINDOWS.items():
        if name in {
            "train_251_500",
            "validation_501_750",
            "contaminated_751_1000",
        }:
            mask = result["day"].between(start, end)
            result.loc[mask, "window"] = name
    return result


def bucket_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize ridge direction skill by fixed normalized-strength buckets."""
    unresolved = events[events["unresolved"]].copy()
    unresolved["strength_bucket"] = pd.cut(
        unresolved["raw_strength"],
        bins=STRENGTH_EDGES,
        labels=STRENGTH_LABELS,
        right=False,
        include_lowest=True,
    )
    unresolved = add_window(unresolved)
    grouped = (
        unresolved[unresolved["window"] != "outside"]
        .groupby(
            ["window", "state", "strength_bucket"],
            observed=True,
        )
        .agg(
            n=("hit", "size"),
            accuracy=("hit", "mean"),
            mean_gross_pnl=("gross_full_leg_pnl", "mean"),
            total_gross_pnl=("gross_full_leg_pnl", "sum"),
            mean_abs_move=("realised_return", lambda x: float(np.abs(x).mean())),
            mean_links=("link_count", "mean"),
        )
        .reset_index()
    )
    return grouped


def overlay(
    base: np.ndarray,
    unresolved: np.ndarray,
    prediction: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Insert a fractional ridge leg only where the current book is flat."""
    result = base.copy()
    direction = np.sign(prediction)
    target = (
        LIMIT[None, :]
        * np.clip(fraction, 0.0, 1.0)
        * direction
    )
    active = unresolved & (direction != 0.0) & (fraction > 0.0)
    result[active] = target[active]
    return result


def candidate_fractions(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a small, pre-declared deadlock overlay family."""
    unresolved = cache["unresolved"]
    strength = cache["strength"]
    gate_pass = np.broadcast_to(
        cache["gate_pass"][:, None],
        unresolved.shape,
    )
    gate_closed = ~gate_pass
    shape = unresolved.shape

    candidates: dict[str, np.ndarray] = {
        "stay_flat": np.zeros(shape),
        "all_small_10pct": np.full(shape, 0.10),
        "all_small_25pct": np.full(shape, 0.25),
        "all_half": np.full(shape, 0.50),
        "all_full": np.ones(shape),
        "confidence_small_25pct": (
            0.25 * np.clip(strength / strategy.RIDGE_THETA, 0.0, 1.0)
        ),
        "confidence_full": np.clip(
            strength / strategy.RIDGE_THETA,
            0.0,
            1.0,
        ),
        "sub_threshold_confidence_25pct": (
            0.25
            * gate_pass
            * np.clip(strength / strategy.RIDGE_THETA, 0.0, 1.0)
        ),
        "sub_threshold_confidence_full": (
            gate_pass
            * np.clip(strength / strategy.RIDGE_THETA, 0.0, 1.0)
        ),
        "sub_threshold_all_25pct": (
            0.25 * gate_pass
        ),
        "sub_threshold_all_full": gate_pass.astype(float),
        "sub_threshold_small_25pct": (
            0.25 * gate_pass * (strength >= 0.25)
        ),
        "sub_threshold_full_50plus": (
            gate_pass * (strength >= 0.50)
        ),
    }
    for threshold in (0.75, 1.00, 1.50, 2.00, 3.00):
        tail = gate_closed & (strength >= threshold)
        candidates[f"gate_closed_tail_{threshold:g}_full"] = tail.astype(float)
        candidates[f"barbell_10pct_tail_{threshold:g}"] = np.where(
            tail,
            1.0,
            0.10,
        )
        candidates[f"barbell_conf25_tail_{threshold:g}"] = np.where(
            tail,
            1.0,
            candidates["confidence_small_25pct"],
        )

    return {
        name: fraction * unresolved
        for name, fraction in candidates.items()
    }


def evaluate_candidates(
    prices: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate exact accounting score for each overlay and time block."""
    days = cache["days"]
    schedules: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for name, fraction in candidate_fractions(cache).items():
        schedule = overlay(
            cache["base"],
            cache["unresolved"],
            cache["prediction"],
            fraction,
        )
        schedules[name] = schedule
        for window, (start, end) in WINDOWS.items():
            pnl = run_accounting(
                prices,
                days,
                schedule,
                start,
                end,
            )
            rows.append(
                {
                    "candidate": name,
                    "window": window,
                    "mean_pnl": float(pnl.mean()),
                    "std_pnl": float(pnl.std()),
                    "score": score(float(pnl.mean()), float(pnl.std())),
                    "active_asset_days": int(
                        (
                            cache["unresolved"]
                            & (fraction > 0.0)
                            & np.broadcast_to(
                                (
                                    (days >= start - 1)
                                    & (days < end)
                                )[:, None],
                                fraction.shape,
                            )
                        ).sum()
                    ),
                }
            )

    results = pd.DataFrame(rows)
    baseline = (
        results[results["candidate"] == "stay_flat"]
        .set_index("window")["score"]
    )
    results["score_delta"] = results.apply(
        lambda row: float(row["score"] - baseline[row["window"]]),
        axis=1,
    )
    return results, schedules


def select_train_only(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Lock one rule from the two training halves only."""
    pivot = results.pivot(
        index="candidate",
        columns="window",
        values="score_delta",
    )
    pivot["train_floor"] = pivot[
        ["train_a_251_375", "train_b_376_500"]
    ].min(axis=1)
    pivot["train_average"] = pivot[
        ["train_a_251_375", "train_b_376_500"]
    ].mean(axis=1)
    eligible = pivot[
        (pivot.index != "stay_flat")
        & (pivot["train_floor"] > 0.0)
    ]
    if eligible.empty:
        return "stay_flat", pivot.sort_values(
            ["train_floor", "train_average"],
            ascending=False,
        )
    selected = str(
        eligible.sort_values(
            ["train_floor", "train_average"],
            ascending=False,
        ).index[0]
    )
    return selected, pivot.sort_values(
        ["train_floor", "train_average"],
        ascending=False,
    )


def select_after_validation(ranking: pd.DataFrame) -> str:
    """Choose among train-stable rules using both validation halves."""
    eligible = ranking[
        (ranking.index != "stay_flat")
        & (ranking["train_floor"] > 0.0)
        & (ranking["validation_a_501_625"] > 0.0)
        & (ranking["validation_b_626_750"] > 0.0)
    ].copy()
    if eligible.empty:
        return "stay_flat"
    eligible["validation_floor"] = eligible[
        ["validation_a_501_625", "validation_b_626_750"]
    ].min(axis=1)
    eligible["validation_average"] = eligible[
        ["validation_a_501_625", "validation_b_626_750"]
    ].mean(axis=1)
    return str(
        eligible.sort_values(
            [
                "validation_floor",
                "validation_average",
                "train_floor",
            ],
            ascending=False,
        ).index[0]
    )


def direction_controls(
    prices: np.ndarray,
    cache: dict[str, np.ndarray],
    fraction: np.ndarray,
    candidate: str,
    seeds: int = 64,
) -> pd.DataFrame:
    """Compare the selected ridge direction with inverse/random directions."""
    rows: list[dict[str, object]] = []
    for window in (
        "train_251_500",
        "validation_501_750",
        "contaminated_751_1000",
    ):
        start, end = WINDOWS[window]
        baseline_pnl = run_accounting(
            prices,
            cache["days"],
            cache["base"],
            start,
            end,
        )
        baseline_score = score(
            float(baseline_pnl.mean()),
            float(baseline_pnl.std()),
        )

        observed_schedule = overlay(
            cache["base"],
            cache["unresolved"],
            cache["prediction"],
            fraction,
        )
        observed_pnl = run_accounting(
            prices,
            cache["days"],
            observed_schedule,
            start,
            end,
        )
        observed_delta = score(
            float(observed_pnl.mean()),
            float(observed_pnl.std()),
        ) - baseline_score

        inverse_schedule = overlay(
            cache["base"],
            cache["unresolved"],
            -cache["prediction"],
            fraction,
        )
        inverse_pnl = run_accounting(
            prices,
            cache["days"],
            inverse_schedule,
            start,
            end,
        )
        inverse_delta = score(
            float(inverse_pnl.mean()),
            float(inverse_pnl.std()),
        ) - baseline_score

        random_deltas = []
        for seed in range(seeds):
            rng = np.random.default_rng(10_000 + seed)
            random_direction = np.where(
                rng.random(cache["prediction"].shape) < 0.5,
                -1.0,
                1.0,
            )
            random_schedule = overlay(
                cache["base"],
                cache["unresolved"],
                random_direction,
                fraction,
            )
            random_pnl = run_accounting(
                prices,
                cache["days"],
                random_schedule,
                start,
                end,
            )
            random_deltas.append(
                score(
                    float(random_pnl.mean()),
                    float(random_pnl.std()),
                )
                - baseline_score
            )
        random_values = np.asarray(random_deltas)
        rows.append(
            {
                "candidate": candidate,
                "window": window,
                "observed_delta": float(observed_delta),
                "inverse_delta": float(inverse_delta),
                "random_mean_delta": float(random_values.mean()),
                "random_std_delta": float(random_values.std()),
                "random_95pct": float(np.quantile(random_values, 0.95)),
                "randomization_p": float(
                    (1 + np.sum(random_values >= observed_delta))
                    / (seeds + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def state_summary(events: pd.DataFrame) -> pd.DataFrame:
    framed = add_window(events)
    return (
        framed[framed["window"] != "outside"]
        .groupby(["window", "state"])
        .agg(
            n=("hit", "size"),
            accuracy=("hit", "mean"),
            mean_strength=("raw_strength", "mean"),
            mean_gross_pnl=("gross_full_leg_pnl", "mean"),
            total_gross_pnl=("gross_full_leg_pnl", "sum"),
        )
        .reset_index()
    )


def concentration_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Measure whether sub-threshold ridge PnL is broadly distributed."""
    framed = add_window(events)
    framed = framed[framed["state"] == "sub_threshold"].copy()
    rows: list[dict[str, object]] = []
    for window in (
        "train_251_500",
        "validation_501_750",
        "contaminated_751_1000",
    ):
        sample = framed[framed["window"] == window]
        by_symbol = sample.groupby("symbol")["gross_full_leg_pnl"].sum()
        by_day = sample.groupby("day")["gross_full_leg_pnl"].sum()
        absolute = by_symbol.abs().sort_values(ascending=False)
        gross_abs_sum = float(absolute.sum())
        rows.append(
            {
                "window": window,
                "events": int(len(sample)),
                "symbols": int(len(by_symbol)),
                "profitable_symbols": int((by_symbol > 0.0).sum()),
                "positive_event_days": int((by_day > 0.0).sum()),
                "event_days": int(len(by_day)),
                "top_symbol_abs_share": float(
                    absolute.iloc[:1].sum() / gross_abs_sum
                    if gross_abs_sum > 0.0
                    else 0.0
                ),
                "top5_symbol_abs_share": float(
                    absolute.iloc[:5].sum() / gross_abs_sum
                    if gross_abs_sum > 0.0
                    else 0.0
                ),
                "best_symbol": (
                    str(by_symbol.idxmax()) if len(by_symbol) else ""
                ),
                "best_symbol_gross_pnl": float(
                    by_symbol.max() if len(by_symbol) else 0.0
                ),
                "worst_symbol": (
                    str(by_symbol.idxmin()) if len(by_symbol) else ""
                ),
                "worst_symbol_gross_pnl": float(
                    by_symbol.min() if len(by_symbol) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return str(value)
            return f"{float(value):.4f}"
        return str(value).replace("|", r"\|")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| "
        + " | ".join(render(value) for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_report(
    events: pd.DataFrame,
    states: pd.DataFrame,
    buckets: pd.DataFrame,
    results: pd.DataFrame,
    ranking: pd.DataFrame,
    train_selected: str,
    validated_selected: str,
    controls: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    selected_rows = results[
        results[
            "candidate"
        ].isin(["stay_flat", train_selected, validated_selected])
    ].copy()
    selected_rows = selected_rows[
        [
            "candidate",
            "window",
            "score",
            "score_delta",
            "mean_pnl",
            "active_asset_days",
        ]
    ]
    pair_ties = int(len(events))
    unresolved = int(events["unresolved"].sum())
    resolved = int(events["current_ridge_resolved"].sum())
    report = [
        "# Ridge barbell usefulness on pair-vote deadlocks",
        "",
        "## What the ridge score means",
        "",
        "`raw_strength = |forecast| / std(in-sample fitted forecasts)`. "
        "It is a normalized forecast magnitude, not a calibrated probability. "
        f"The live entry threshold is {strategy.RIDGE_THETA:.2f}. The global "
        f"gate also requires at least {strategy.RIDGE_GATE_N} significant "
        "lagged cross-asset links in the most recent 250 returns.",
        "",
        "A zero live ridge strength is therefore ambiguous when the global "
        "gate is closed. This audit reconstructs the latent score so those "
        "days can be tested rather than silently grouped with no-forecast days.",
        "",
        "## Population",
        "",
        f"- Pre-ridge exact pair cancellations: {pair_ties}",
        f"- Already resolved by the live ridge: {resolved}",
        f"- Still flat (true deadlocks): {unresolved}",
        "",
        "## Direction skill by deadlock state",
        "",
        markdown_table(states),
        "",
        "## Direction skill by fixed strength bucket",
        "",
        markdown_table(buckets),
        "",
        "## Train-only rule selection",
        "",
        "Candidates were ranked by their worst score delta across days "
        "251-375 and 376-500, and had to be positive in both. Days 501-750 "
        "and 751-1000 were not used for selection.",
        "",
        f"Train-locked candidate: `{train_selected}`",
        "",
        markdown_table(
            ranking.reset_index().rename(columns={"index": "candidate"})
        ),
        "",
        "## Validation selection",
        "",
        "Among rules positive in both training halves, the validated rule must "
        "also be positive in both 125-day validation halves. It is selected "
        "by validation floor; the final window is not consulted.",
        "",
        f"Validated candidate: `{validated_selected}`",
        "",
        "## Selected candidates versus staying flat",
        "",
        markdown_table(selected_rows),
        "",
        "## Direction controls for the validated candidate",
        "",
        "The same entry days and sizes are replayed with the ridge sign "
        "inverted and with 64 random directions. A useful ridge forecast "
        "should beat both controls.",
        "",
        markdown_table(controls),
        "",
        "## Concentration of the gate-respecting full fill",
        "",
        "These are gross ridge-leg contributions on sub-threshold deadlocks. "
        "They show whether the effect is spread across symbols and event days "
        "rather than generated by one relationship.",
        "",
        markdown_table(concentration),
        "",
        "Days 751-1000 are labelled contaminated because this repository has "
        "already used them repeatedly. Their result is descriptive, not a "
        "fresh out-of-sample claim.",
    ]
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        ROOT / "prices.txt",
        sep=r"\s+",
        header=0,
    ).to_numpy(float).T

    if CACHE.exists():
        with np.load(CACHE) as loaded:
            cache = {key: loaded[key] for key in loaded.files}
    else:
        print("Replaying current strategy and reconstructing latent ridge...")
        cache = build_cache(prices)
        np.savez_compressed(CACHE, **cache)

    events = event_frame(cache)
    states = state_summary(events)
    buckets = bucket_summary(events)
    results, _ = evaluate_candidates(prices, cache)
    train_selected, ranking = select_train_only(results)
    validated_selected = select_after_validation(ranking)
    fractions = candidate_fractions(cache)
    controls = direction_controls(
        prices,
        cache,
        fractions[validated_selected],
        validated_selected,
    )
    concentration = concentration_summary(events)

    events.to_csv(OUTPUT_DIR / "events.csv", index=False)
    states.to_csv(OUTPUT_DIR / "state_summary.csv", index=False)
    buckets.to_csv(OUTPUT_DIR / "strength_buckets.csv", index=False)
    results.to_csv(OUTPUT_DIR / "candidate_scores.csv", index=False)
    ranking.reset_index().to_csv(
        OUTPUT_DIR / "train_ranking.csv",
        index=False,
    )
    controls.to_csv(
        OUTPUT_DIR / "direction_controls.csv",
        index=False,
    )
    concentration.to_csv(
        OUTPUT_DIR / "concentration_summary.csv",
        index=False,
    )
    write_report(
        events,
        states,
        buckets,
        results,
        ranking,
        train_selected,
        validated_selected,
        controls,
        concentration,
    )

    train_selected_results = results[
        results["candidate"] == train_selected
    ].set_index("window")
    validated_selected_results = results[
        results["candidate"] == validated_selected
    ].set_index("window")
    summary = {
        "pair_cancellations": int(len(events)),
        "current_ridge_resolved": int(
            events["current_ridge_resolved"].sum()
        ),
        "unresolved_deadlocks": int(events["unresolved"].sum()),
        "selected_train_only_candidate": train_selected,
        "train_selected_score_deltas": {
            window: float(
                train_selected_results.loc[window, "score_delta"]
            )
            for window in WINDOWS
        },
        "selected_after_validation_candidate": validated_selected,
        "validated_selected_score_deltas": {
            window: float(
                validated_selected_results.loc[window, "score_delta"]
            )
            for window in WINDOWS
        },
        "production_modified": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
