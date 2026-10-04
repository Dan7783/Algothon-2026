"""Leakage-safe black-box routing for current four_sigmas signal overlaps.

The experiment deliberately excludes pair identity and the current protected-pair
flag from its features.  It learns only from information observable at decision
time.  Development loads days 1-750; the locked test phase loads days 1-1000.

Usage:
    py research/current_overlap_blackbox.py develop
    py research/current_overlap_blackbox.py test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "current_overlap_blackbox"
LOCK_PATH = OUTPUT_DIR / "locked_model.json"
REPORT_PATH = OUTPUT_DIR / "report.md"

TRAIN_START = 251
TRAIN_END = 495  # Five-day embargo before validation.
VAL1 = (501, 625)
VAL2 = (626, 750)
OOS1 = (751, 875)
OOS2 = (876, 1000)

CONFIDENCE_MARGINS = (0.00, 0.05, 0.10, 0.15)
ROUTER_MODES = ("pooled", "segmented")
MIN_VALIDATION_ROWS = 100
WEIGHT_CAP_QUANTILE = 0.95

FEATURES = (
    "pair_support_for_baseline",
    "pair_abs_consensus",
    "pair_priority_sum",
    "pair_priority_max",
    "pair_priority_mean",
    "pair_priority_std",
    "pair_contributors",
    "pair_support_fraction_for_baseline",
    "pair_conflict",
    "pair_gross_notional",
    "ridge_strength",
    "ridge_active",
    "ridge_support_for_baseline",
    "pair_ridge_agree",
    "pair_ridge_disagree",
    "pair_ridge_strength_ratio",
    "unboosted_vote_margin",
    "return_1d_aligned",
    "return_5d_aligned",
    "return_20d_aligned",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "market_return_1d_aligned",
    "market_return_5d_aligned",
    "market_volatility_20d",
    "market_beta_60d",
    "return_rank_1d_aligned",
)


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_l2": {
        "kind": "logistic",
        "C": 0.10,
    },
    "gbm_stumps": {
        "kind": "gradient_boosting",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_samples_leaf": 80,
        "subsample": 0.80,
    },
    "hist_gbm_shallow": {
        "kind": "hist_gradient_boosting",
        "max_iter": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 80,
        "l2_regularization": 10.0,
    },
    "random_forest_shallow": {
        "kind": "random_forest",
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 80,
        "max_features": 0.70,
    },
    "extra_trees_shallow": {
        "kind": "extra_trees",
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 80,
        "max_features": 0.70,
    },
}


@dataclass
class OverlapCache:
    prices: np.ndarray
    rows: pd.DataFrame
    baseline_dollars: np.ndarray


def _rolling_sum(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    return np.sum(values[:, -min(count, values.shape[1]) :], axis=1)


def _rolling_std(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0])
    window = values[:, -min(count, values.shape[1]) :]
    return np.std(window, axis=1)


def _market_beta(
    returns: np.ndarray,
    market_returns: np.ndarray,
    count: int = 60,
) -> np.ndarray:
    n = min(count, returns.shape[1])
    if n < 10:
        return np.zeros(returns.shape[0])
    asset = returns[:, -n:]
    market = market_returns[-n:]
    market_centered = market - np.mean(market)
    denominator = float(np.sum(market_centered**2))
    if denominator <= 1.0e-12:
        return np.zeros(returns.shape[0])
    asset_centered = asset - np.mean(asset, axis=1, keepdims=True)
    return (asset_centered @ market_centered) / denominator


def _cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    if len(values) <= 1:
        return np.zeros_like(values)
    return 2.0 * order / float(len(values) - 1) - 1.0


def _unboosted_pair_features(
    candidates: list[tuple[float, strategy.PairAgent, float, float]],
) -> dict[str, np.ndarray]:
    n = strategy.N_INST
    signed = np.zeros(n)
    total = np.zeros(n)
    maximum = np.zeros(n)
    sum_squared = np.zeros(n)
    contributors = np.zeros(n, dtype=int)
    positive = np.zeros(n, dtype=int)
    negative = np.zeros(n, dtype=int)
    gross = np.zeros(n)

    for priority, agent, pos_y, pos_x in candidates:
        weight = max(float(priority), 1.0e-12)
        for instrument, position in (
            (int(agent.spec.y), float(pos_y)),
            (int(agent.spec.x), float(pos_x)),
        ):
            if abs(position) <= 1.0e-9:
                continue
            direction = float(np.sign(position))
            signed[instrument] += weight * direction
            total[instrument] += weight
            maximum[instrument] = max(maximum[instrument], weight)
            sum_squared[instrument] += weight * weight
            contributors[instrument] += 1
            positive[instrument] += int(direction > 0.0)
            negative[instrument] += int(direction < 0.0)
            gross[instrument] += abs(position)

    consensus = np.divide(
        signed,
        total,
        out=np.zeros_like(signed),
        where=total > 0.0,
    )
    mean = np.divide(
        total,
        contributors,
        out=np.zeros_like(total),
        where=contributors > 0,
    )
    second_moment = np.divide(
        sum_squared,
        contributors,
        out=np.zeros_like(total),
        where=contributors > 0,
    )
    std = np.sqrt(np.maximum(second_moment - mean * mean, 0.0))
    positive_fraction = np.divide(
        positive,
        contributors,
        out=np.zeros_like(total),
        where=contributors > 0,
    )
    return {
        "consensus": consensus,
        "priority_sum": total,
        "priority_max": maximum,
        "priority_mean": mean,
        "priority_std": std,
        "contributors": contributors,
        "positive_fraction": positive_fraction,
        "conflict": ((positive > 0) & (negative > 0)).astype(float),
        "gross": gross,
    }


def _baseline_allocation(
    prices: np.ndarray,
    candidates: list[tuple[float, strategy.PairAgent, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_vote, pair_active, _ = strategy._pair_conflict_consensus(candidates)
    returns = np.diff(np.log(np.maximum(prices, strategy._EPS_DENOM)), axis=1)
    ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
    ridge_sign = np.sign(ridge_raw)
    ridge_active = ridge_sign != 0.0
    ridge_vote = (
        strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
    )
    combined = pair_vote + ridge_vote
    active = pair_active | ridge_active
    direction = np.sign(combined)
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]

    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], strategy._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    direction[crash] = 0.0
    dollars = strategy.BANG_BANG_LIMITS * direction
    return dollars, ridge_strength, ridge_sign, ridge_active, crash


def collect_overlap_cache(prices: np.ndarray) -> OverlapCache:
    """Replay current production logic and construct identity-free rows."""
    n_history = prices.shape[1]
    baseline = np.zeros((n_history + 1, strategy.N_INST))
    records: list[dict[str, float | int]] = []
    atlas = strategy.AtlasBook()
    log_prices = np.log(np.maximum(prices, strategy._EPS_DENOM))
    all_returns = np.diff(log_prices, axis=1)

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        candidates = atlas.candidates(history)
        if history_size < strategy.WARMUP + 10:
            continue

        (
            baseline[history_size],
            ridge_strength,
            ridge_sign,
            ridge_active,
            crash,
        ) = _baseline_allocation(history, candidates)
        pair = _unboosted_pair_features(candidates)
        pair_active = pair["contributors"] > 0
        pair_sign = np.sign(pair["consensus"])
        agree = (
            pair_active
            & ridge_active
            & (pair_sign != 0.0)
            & (pair_sign == ridge_sign)
        )
        disagree = (
            pair_active
            & ridge_active
            & (pair_sign != 0.0)
            & (pair_sign != ridge_sign)
        )
        # Routing is needed only when sources oppose one another. Agreement
        # cases are left untouched regardless of model confidence.
        overlap = (pair["conflict"] > 0.0) | disagree
        overlap &= ~crash
        baseline_direction = np.sign(baseline[history_size])
        overlap &= baseline_direction != 0.0
        overlap[0] = False
        if not np.any(overlap):
            continue

        past_returns = all_returns[:, : max(history_size - 1, 0)]
        market_returns = (
            np.mean(past_returns[1:], axis=0)
            if past_returns.shape[1]
            else np.zeros(0)
        )
        return_1d = _rolling_sum(past_returns, 1)
        return_5d = _rolling_sum(past_returns, 5)
        return_20d = _rolling_sum(past_returns, 20)
        vol_5d = _rolling_std(past_returns, 5)
        vol_20d = _rolling_std(past_returns, 20)
        vol_60d = _rolling_std(past_returns, 60)
        market_return_1d = (
            float(_rolling_sum(market_returns[None, :], 1)[0])
            if len(market_returns)
            else 0.0
        )
        market_return_5d = (
            float(_rolling_sum(market_returns[None, :], 5)[0])
            if len(market_returns)
            else 0.0
        )
        market_vol_20d = (
            float(_rolling_std(market_returns[None, :], 20)[0])
            if len(market_returns)
            else 0.0
        )
        beta_60d = _market_beta(past_returns, market_returns)
        rank_1d = _cross_sectional_rank(return_1d)

        strength_ratio = np.divide(
            ridge_strength,
            pair["priority_mean"] + 1.0e-6,
        )
        strength_ratio = np.log1p(np.minimum(strength_ratio, 1.0e6))
        unboosted_margin = np.abs(
            pair["consensus"]
            + strategy.RIDGE_CONFLICT_WEIGHT
            * ridge_sign
            * ridge_strength
        )
        forward_return = (
            prices[:, history_size]
            / np.maximum(prices[:, history_size - 1], strategy._EPS_DENOM)
            - 1.0
        )

        for instrument in np.flatnonzero(overlap):
            proposed_direction = float(baseline_direction[instrument])
            records.append(
                {
                    "history_size": int(history_size),
                    "score_day": int(history_size + 1),
                    "instrument": int(instrument),
                    "target_return": float(forward_return[instrument]),
                    "target_up": int(forward_return[instrument] > 0.0),
                    "target_keep": int(
                        proposed_direction * forward_return[instrument] > 0.0
                    ),
                    "baseline_direction": int(
                        proposed_direction
                    ),
                    "pair_support_for_baseline": float(
                        proposed_direction * pair["consensus"][instrument]
                    ),
                    "pair_abs_consensus": float(
                        abs(pair["consensus"][instrument])
                    ),
                    "pair_priority_sum": float(
                        pair["priority_sum"][instrument]
                    ),
                    "pair_priority_max": float(
                        pair["priority_max"][instrument]
                    ),
                    "pair_priority_mean": float(
                        pair["priority_mean"][instrument]
                    ),
                    "pair_priority_std": float(
                        pair["priority_std"][instrument]
                    ),
                    "pair_contributors": int(
                        pair["contributors"][instrument]
                    ),
                    "pair_support_fraction_for_baseline": float(
                        pair["positive_fraction"][instrument]
                        if proposed_direction > 0.0
                        else 1.0 - pair["positive_fraction"][instrument]
                    ),
                    "pair_conflict": float(pair["conflict"][instrument]),
                    "pair_gross_notional": float(
                        np.log1p(pair["gross"][instrument])
                    ),
                    "ridge_strength": float(ridge_strength[instrument]),
                    "ridge_active": int(ridge_active[instrument]),
                    "ridge_support_for_baseline": float(
                        proposed_direction * ridge_sign[instrument]
                    ),
                    "pair_ridge_agree": int(agree[instrument]),
                    "pair_ridge_disagree": int(disagree[instrument]),
                    "conflict_family": int(disagree[instrument]),
                    "pair_ridge_strength_ratio": float(
                        strength_ratio[instrument]
                    ),
                    "unboosted_vote_margin": float(
                        unboosted_margin[instrument]
                    ),
                    "return_1d_aligned": float(
                        proposed_direction * return_1d[instrument]
                    ),
                    "return_5d_aligned": float(
                        proposed_direction * return_5d[instrument]
                    ),
                    "return_20d_aligned": float(
                        proposed_direction * return_20d[instrument]
                    ),
                    "volatility_5d": float(vol_5d[instrument]),
                    "volatility_20d": float(vol_20d[instrument]),
                    "volatility_60d": float(vol_60d[instrument]),
                    "market_return_1d_aligned": float(
                        proposed_direction * market_return_1d
                    ),
                    "market_return_5d_aligned": float(
                        proposed_direction * market_return_5d
                    ),
                    "market_volatility_20d": market_vol_20d,
                    "market_beta_60d": float(beta_60d[instrument]),
                    "return_rank_1d_aligned": float(
                        proposed_direction * rank_1d[instrument]
                    ),
                }
            )

    rows = pd.DataFrame.from_records(records)
    if rows.empty:
        raise RuntimeError("No current-strategy overlap observations found")
    rows.loc[:, FEATURES] = (
        rows.loc[:, FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return OverlapCache(prices=prices, rows=rows, baseline_dollars=baseline)


def make_model(name: str) -> Any:
    spec = MODEL_SPECS[name]
    kind = spec["kind"]
    args = {key: value for key, value in spec.items() if key != "kind"}
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        **args,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=0,
                    ),
                ),
            ]
        )
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(random_state=0, **args)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=0, **args)
    if kind == "random_forest":
        return RandomForestClassifier(
            random_state=0,
            n_jobs=-1,
            class_weight="balanced",
            **args,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            random_state=0,
            n_jobs=-1,
            class_weight="balanced",
            **args,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def fit_model(model: Any, frame: pd.DataFrame) -> Any:
    absolute_return = np.abs(frame["target_return"].to_numpy())
    cap = float(np.quantile(absolute_return, WEIGHT_CAP_QUANTILE))
    weights = np.minimum(absolute_return, max(cap, 1.0e-8))
    weights /= max(float(np.mean(weights)), 1.0e-12)
    x = frame.loc[:, FEATURES].to_numpy(dtype=float)
    y = frame["target_keep"].to_numpy(dtype=int)
    if isinstance(model, Pipeline):
        model.fit(x, y, model__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def fit_router(
    model_name: str,
    frame: pd.DataFrame,
    router_mode: str,
) -> dict[str, Any]:
    router = {"pooled": fit_model(make_model(model_name), frame)}
    if router_mode == "segmented":
        for family in (0, 1):
            subset = frame[frame["conflict_family"] == family]
            if len(subset) >= 200 and subset["target_keep"].nunique() == 2:
                router[str(family)] = fit_model(
                    make_model(model_name),
                    subset,
                )
    return router


def predict_router(
    router: dict[str, Any],
    frame: pd.DataFrame,
    router_mode: str,
) -> np.ndarray:
    x = frame.loc[:, FEATURES].to_numpy(dtype=float)
    if router_mode == "pooled":
        return router["pooled"].predict_proba(x)[:, 1]
    probability = np.empty(len(frame))
    family_values = frame["conflict_family"].to_numpy(dtype=int)
    for family in (0, 1):
        mask = family_values == family
        if not np.any(mask):
            continue
        model = router.get(str(family), router["pooled"])
        probability[mask] = model.predict_proba(x[mask])[:, 1]
    return probability


def safe_auc(frame: pd.DataFrame, probability: np.ndarray) -> float:
    y = frame["target_keep"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probability))


def routed_dollars(
    cache: OverlapCache,
    frame: pd.DataFrame,
    probability: np.ndarray,
    confidence_margin: float,
) -> tuple[np.ndarray, dict[str, float]]:
    dollars = cache.baseline_dollars.copy()
    flip = probability <= 0.5 - confidence_margin
    changed = 0
    for row, accepted, direction in zip(
        frame.itertuples(index=False),
        flip,
        -frame["baseline_direction"].to_numpy(dtype=float),
    ):
        if not accepted:
            continue
        history_size = int(row.history_size)
        instrument = int(row.instrument)
        original = float(np.sign(dollars[history_size, instrument]))
        changed += int(original != direction)
        dollars[history_size, instrument] = (
            strategy.BANG_BANG_LIMITS[instrument] * direction
        )
    return dollars, {
        "observations": float(len(frame)),
        "confidence_rate": float(np.mean(flip)) if len(frame) else 0.0,
        "direction_changes": float(changed),
        "change_rate": float(changed / max(len(frame), 1)),
    }


def _window(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return frame[
        (frame["score_day"] >= start)
        & (frame["score_day"] <= end)
    ].copy()


def _metrics_for_windows(
    prices: np.ndarray,
    dollars: np.ndarray,
    windows: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(prices, dollars, start, end)
        for name, start, end in windows
    }


def development_phase() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(750)
    cache = collect_overlap_cache(prices)
    train = _window(cache.rows, TRAIN_START, TRAIN_END)
    val = _window(cache.rows, VAL1[0], VAL2[1])
    if len(train) < MIN_VALIDATION_ROWS or len(val) < MIN_VALIDATION_ROWS:
        raise RuntimeError(
            f"Insufficient overlap rows: train={len(train)}, validation={len(val)}"
        )

    windows = (
        ("VAL1", *VAL1),
        ("VAL2", *VAL2),
        ("VALIDATION", VAL1[0], VAL2[1]),
    )
    baseline = _metrics_for_windows(
        prices,
        cache.baseline_dollars,
        windows,
    )
    rows: list[dict[str, float | str]] = []

    for model_name in MODEL_SPECS:
        for router_mode in ROUTER_MODES:
            router = fit_router(model_name, train, router_mode)
            train_probability = predict_router(router, train, router_mode)
            val_probability = predict_router(router, val, router_mode)
            val1 = _window(val, *VAL1)
            val2 = _window(val, *VAL2)
            val1_probability = val_probability[
                (val["score_day"].to_numpy() >= VAL1[0])
                & (val["score_day"].to_numpy() <= VAL1[1])
            ]
            val2_probability = val_probability[
                (val["score_day"].to_numpy() >= VAL2[0])
                & (val["score_day"].to_numpy() <= VAL2[1])
            ]

            for margin in CONFIDENCE_MARGINS:
                dollars, routing = routed_dollars(
                    cache,
                    val,
                    val_probability,
                    margin,
                )
                results = _metrics_for_windows(prices, dollars, windows)
                record: dict[str, float | str] = {
                    "model": model_name,
                    "router_mode": router_mode,
                    "confidence_margin": margin,
                    "train_rows": float(len(train)),
                    "validation_rows": float(len(val)),
                    "train_auc": safe_auc(train, train_probability),
                    "val1_auc": safe_auc(val1, val1_probability),
                    "val2_auc": safe_auc(val2, val2_probability),
                    "validation_auc": safe_auc(val, val_probability),
                    **routing,
                }
                for name, _, _ in windows:
                    lower = name.lower()
                    record[f"{lower}_score"] = results[name]["score"]
                    record[f"{lower}_delta"] = (
                        results[name]["score"] - baseline[name]["score"]
                    )
                record["robust_delta"] = min(
                    float(record["val1_delta"]),
                    float(record["val2_delta"]),
                )
                rows.append(record)

    grid = pd.DataFrame(rows).sort_values(
        ["robust_delta", "validation_delta"],
        ascending=False,
    )
    grid.to_csv(OUTPUT_DIR / "development_grid.csv", index=False)
    eligible = grid[
        (grid["val1_delta"] > 0.0)
        & (grid["val2_delta"] > 0.0)
        & (grid["val1_auc"] > 0.50)
        & (grid["val2_auc"] > 0.50)
    ]
    accepted = not eligible.empty
    winner = eligible.iloc[0] if accepted else grid.iloc[0]
    lock = {
        "accepted": bool(accepted),
        "model": str(winner["model"]),
        "router_mode": str(winner["router_mode"]),
        "model_spec": MODEL_SPECS[str(winner["model"])],
        "confidence_margin": float(winner["confidence_margin"]),
        "features": list(FEATURES),
        "training_score_days": [TRAIN_START, TRAIN_END],
        "validation_score_days": [VAL1[0], VAL2[1]],
        "selection_rule": (
            "positive score delta and AUC > 0.50 in both validation halves; "
            "maximize worst-half score delta"
        ),
        "baseline": baseline,
        "winner": {
            key: (
                str(value)
                if isinstance(value, str)
                else float(value)
            )
            for key, value in winner.to_dict().items()
        },
        "oos_loaded": False,
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    _write_report(lock, grid, phase="development")
    print(json.dumps(lock, indent=2))


def test_phase() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run the development phase before the locked test")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not lock["accepted"]:
        raise RuntimeError(
            "Development rejected every black-box router; OOS remains sealed"
        )
    if lock.get("oos_loaded"):
        raise RuntimeError("Locked OOS confirmation has already been run")

    prices = study.load_prices(1000)
    cache = collect_overlap_cache(prices)
    train = _window(cache.rows, TRAIN_START, TRAIN_END)
    oos = _window(cache.rows, OOS1[0], OOS2[1])
    router = fit_router(lock["model"], train, lock["router_mode"])
    probability = predict_router(router, oos, lock["router_mode"])
    dollars, routing = routed_dollars(
        cache,
        oos,
        probability,
        float(lock["confidence_margin"]),
    )
    windows = (
        ("OOS1", *OOS1),
        ("OOS2", *OOS2),
        ("OOS", OOS1[0], OOS2[1]),
    )
    baseline = _metrics_for_windows(
        prices,
        cache.baseline_dollars,
        windows,
    )
    results = _metrics_for_windows(prices, dollars, windows)
    lock["oos_loaded"] = True
    lock["oos"] = {
        "auc": safe_auc(oos, probability),
        **routing,
        "baseline": baseline,
        "blackbox": results,
        "deltas": {
            name: results[name]["score"] - baseline[name]["score"]
            for name, _, _ in windows
        },
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    grid = pd.read_csv(OUTPUT_DIR / "development_grid.csv")
    _write_report(lock, grid, phase="test")
    print(json.dumps(lock["oos"], indent=2))


def _write_report(
    lock: dict[str, Any],
    grid: pd.DataFrame,
    phase: str,
) -> None:
    winner = lock["winner"]
    lines = [
        "# Current signal-overlap black-box",
        "",
        "This experiment contains no pair name, instrument identity, or protected-pair",
        "indicator. Days 751-1000 are unavailable during model and threshold",
        "selection.",
        "",
        "## Development result",
        "",
        f"- Accepted: **{lock['accepted']}**",
        f"- Model: `{lock['model']}`",
        f"- Router mode: `{lock['router_mode']}`",
        f"- Confidence margin: `{lock['confidence_margin']:.2f}`",
        f"- Train AUC: `{winner['train_auc']:.4f}`",
        f"- Validation AUC: `{winner['validation_auc']:.4f}`",
        f"- VAL1 score delta: `{winner['val1_delta']:.4f}`",
        f"- VAL2 score delta: `{winner['val2_delta']:.4f}`",
        f"- Full validation score delta: `{winner['validation_delta']:.4f}`",
        "",
        "The production strategy is not changed unless both validation halves",
        "improve and the locked OOS result also supports integration.",
        "",
        "## Leading development candidates",
        "",
        "```text",
        grid.head(10)[
            [
                "model",
                "router_mode",
                "confidence_margin",
                "train_auc",
                "val1_auc",
                "val2_auc",
                "val1_delta",
                "val2_delta",
                "validation_delta",
                "change_rate",
            ]
        ].to_string(index=False),
        "```",
    ]
    if phase == "test" and "oos" in lock:
        oos = lock["oos"]
        lines.extend(
            [
                "",
                "## Locked OOS confirmation",
                "",
                f"- OOS AUC: `{oos['auc']:.4f}`",
                f"- OOS1 score delta: `{oos['deltas']['OOS1']:.4f}`",
                f"- OOS2 score delta: `{oos['deltas']['OOS2']:.4f}`",
                f"- Full OOS score delta: `{oos['deltas']['OOS']:.4f}`",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("develop", "test"))
    args = parser.parse_args()
    if args.phase == "develop":
        development_phase()
    else:
        test_phase()


if __name__ == "__main__":
    main()
