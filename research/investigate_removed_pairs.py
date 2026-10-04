"""Re-audit every pair excluded by the current ``four_sigmas`` catalog.

The audit separates two distinct provenance classes:

* ``DEVELOPMENT_REMOVED_PAIR_NAMES`` was selected using days 1-750.
* ``PREVIOUS_OOS_REMOVED_PAIR_NAMES`` was selected after observing 751-1000.

Each non-ALGO relationship is added one at a time to today's complete
portfolio.  ALGO-HRET is treated specially because it is excluded from the
ordinary pair book but remains live inside the ALGO residual basket.  For it,
the audit measures the marginal effect of removing that residual agent.

No production source is changed.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import algo12_equal_sleeve_research as evaluator
from research import benchmark_teamname


PRICE_PATH = ROOT / "prices.txt"
DATABASE_PATH = ROOT / "tuning_data.db"
OUTPUT_DIR = ROOT / "analysis_outputs" / "removed_pair_investigation"

WINDOWS = (
    ("YEAR1", 31, 250),
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
    ("TEST_A", 751, 875),
    ("TEST_B", 876, 1000),
    ("TEST", 751, 1000),
)
PRIMARY_WINDOWS = ("TRAIN", "VALIDATION", "TEST")
DEVELOPMENT_HALVES = (
    "TRAIN_A",
    "TRAIN_B",
    "VALIDATION_A",
    "VALIDATION_B",
)
BLOCKS = tuple(
    (
        f"BLOCK_{start}_{start + 49}",
        start,
        start + 49,
    )
    for start in range(251, 1000, 50)
)


@dataclass(frozen=True)
class Profile:
    pair_name: str
    y_symbol: str
    x_symbol: str
    q_beta: float
    h: float
    cusum_k: float
    max_cov_inflate: float
    cov_decay: float
    z_entry: float
    z_exit: float
    z_stop: float
    removal_class: str
    profile_source: str
    stored_auto_metrics: dict[str, Any] | None
    stored_test_metrics: dict[str, Any] | None
    perturbation_summary: list[Any] | None


def load_prices() -> tuple[np.ndarray, tuple[str, ...]]:
    frame = pd.read_csv(PRICE_PATH, sep=r"\s+")
    return (
        frame.to_numpy(dtype=float).T,
        tuple(str(column) for column in frame.columns),
    )


def fresh_strategy() -> types.ModuleType:
    return benchmark_teamname.load_strategy("four_sigmas")


def db_rows(names: set[str]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _ in names)
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT pair_name, y_symbol, x_symbol, auto_params, "
            "auto_metrics, test_metrics, perturbation_summary "
            f"FROM pair_results WHERE pair_name IN ({placeholders})",
            tuple(sorted(names)),
        ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row[0])] = {
            "y_symbol": str(row[1]),
            "x_symbol": str(row[2]),
            "auto_params": json.loads(row[3]),
            "auto_metrics": json.loads(row[4]) if row[4] else None,
            "test_metrics": json.loads(row[5]) if row[5] else None,
            "perturbation_summary": (
                json.loads(row[6]) if row[6] else None
            ),
        }
    return result


def profiles() -> tuple[Profile, ...]:
    strategy = fresh_strategy()
    development_names = set(strategy.DEVELOPMENT_REMOVED_PAIR_NAMES)
    oos_names = set(strategy.PREVIOUS_OOS_REMOVED_PAIR_NAMES)
    all_names = development_names | oos_names
    stored = db_rows(all_names)
    current_rows = {
        f"{row[0]}-{row[1]}": row
        for row in strategy._UNFILTERED_CURRENT_PAIR_PARAMS
    }
    missing_db = all_names - set(stored)
    if missing_db:
        raise RuntimeError(f"Missing database profiles: {missing_db}")

    result = []
    for pair_name in sorted(all_names):
        database = stored[pair_name]
        if pair_name in development_names:
            row = current_rows.get(pair_name)
            if row is None:
                raise RuntimeError(
                    f"Development-removed row is absent: {pair_name}"
                )
            (
                y_symbol,
                x_symbol,
                q_beta,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                z_entry,
                z_exit,
                z_stop,
            ) = row
            profile_source = "current_excluded_catalog_row"
            removal_class = "development_1_750"
        else:
            params = database["auto_params"]
            y_symbol = database["y_symbol"]
            x_symbol = database["x_symbol"]
            q_beta = params["q_beta"]
            h = params["h"]
            cusum_k = params["cusum_k"]
            max_cov_inflate = params["max_cov_inflate"]
            cov_decay = params["cov_decay"]
            z_entry = params["z_entry"]
            z_exit = params["z_exit"]
            z_stop = 10.0
            profile_source = "tuning_data_db_auto_profile"
            removal_class = "previous_oos_751_1000"
        result.append(
            Profile(
                pair_name=pair_name,
                y_symbol=str(y_symbol),
                x_symbol=str(x_symbol),
                q_beta=float(q_beta),
                h=float(h),
                cusum_k=float(cusum_k),
                max_cov_inflate=float(max_cov_inflate),
                cov_decay=float(cov_decay),
                z_entry=float(z_entry),
                z_exit=float(z_exit),
                z_stop=float(z_stop),
                removal_class=removal_class,
                profile_source=profile_source,
                stored_auto_metrics=database["auto_metrics"],
                stored_test_metrics=database["test_metrics"],
                perturbation_summary=database["perturbation_summary"],
            )
        )
    return tuple(result)


def make_cfg(
    strategy: types.ModuleType,
    profile: Profile,
) -> Any:
    return strategy.PairCfg(
        idx=len(strategy.PAIR_CFGS),
        y=strategy.I[profile.y_symbol],
        x=strategy.I[profile.x_symbol],
        filt=strategy.FiltCfg(
            q_beta=profile.q_beta,
            cusum_k=profile.cusum_k,
            h=profile.h,
            max_cov_inflate=profile.max_cov_inflate,
            cov_decay=profile.cov_decay,
        ),
        trade=strategy.TradeCfg(
            z_entry=profile.z_entry,
            z_exit=profile.z_exit,
            z_stop=profile.z_stop,
        ),
    )


def dollars_to_shares(
    strategy: types.ModuleType,
    dollars: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(prices, strategy._EPS_DENOM)
    limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    return np.clip(
        np.round(dollars / safe),
        -limits,
        limits,
    ).astype(int)


def configure_experiment(
    profile: Profile | None,
    *,
    remove_algo_hret_residual: bool = False,
) -> types.ModuleType:
    strategy = fresh_strategy()
    if profile is not None:
        strategy.PAIR_CFGS = (
            *strategy.PAIR_CFGS,
            make_cfg(strategy, profile),
        )
    if remove_algo_hret_residual:
        strategy.ALGO_RESIDUAL_CFGS = tuple(
            spec
            for spec in strategy.ALGO_RESIDUAL_CFGS
            if strategy._pair_name(spec) != "ALGO-HRET"
        )
    strategy.reset()
    return strategy


def collect_portfolio(
    strategy: types.ModuleType,
    prices: np.ndarray,
    focus_pair: str | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    targets = np.zeros(
        (prices.shape[1] + 1, strategy.N_INST),
        dtype=int,
    )
    rows: list[dict[str, Any]] = []
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        targets[history_size] = dollars_to_shares(
            strategy,
            np.asarray(sleeves["dollars"], dtype=float),
            known[:, -1],
        )
        if focus_pair is None:
            continue
        details = [
            item
            for item in sleeves["pair_contributor_details"]
            if item["pair"] == focus_pair
        ]
        if not details:
            continue
        item = details[0]
        y = int(item["y"])
        x = int(item["x"])
        requested = {
            y: float(item["pos_y"]),
            x: float(item["pos_x"]),
        }
        final_dollars = np.asarray(sleeves["dollars"], dtype=float)
        core_dollars = np.asarray(
            sleeves["core_dollars"],
            dtype=float,
        )
        ridge_conflict = np.asarray(
            sleeves["conflict_mask"],
            dtype=bool,
        )
        for instrument in (y, x):
            requested_sign = np.sign(requested[instrument])
            final_sign = np.sign(final_dollars[instrument])
            core_sign = np.sign(core_dollars[instrument])
            rows.append(
                {
                    "pair_name": focus_pair,
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    "instrument": instrument,
                    "requested_dollars": requested[instrument],
                    "core_dollars": core_dollars[instrument],
                    "final_dollars": final_dollars[instrument],
                    "core_delivered": bool(
                        core_sign == requested_sign
                    ),
                    "final_delivered": bool(
                        final_sign == requested_sign
                    ),
                    "final_opposed": bool(
                        final_sign != 0.0
                        and final_sign != requested_sign
                    ),
                    "final_flat": bool(final_sign == 0.0),
                    "ridge_conflict": bool(
                        ridge_conflict[instrument]
                    ),
                    "pair_contributors": int(
                        sleeves["pair_contributors"][instrument]
                    ),
                }
            )
    strategy.reset()
    return targets, pd.DataFrame(rows)


def standalone_replay(
    profile: Profile,
    prices: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    strategy = fresh_strategy()
    cfg = make_cfg(strategy, profile)
    agent = strategy.PairAgent(cfg)
    targets = np.zeros(
        (prices.shape[1] + 1, strategy.N_INST),
        dtype=int,
    )
    rows: list[dict[str, Any]] = []
    prior_side = 0
    for day in range(1, prices.shape[1]):
        agent.catch_up(prices[:, :day])
        active = day >= strategy.WARMUP + 10 and agent.side != 0
        entry = bool(active and prior_side == 0)
        exit_event = bool(not active and prior_side != 0)
        prior_side = int(agent.side) if active else 0
        dollars = np.zeros(strategy.N_INST)
        if active:
            pos_y, pos_x, _ = agent.dollar_positions()
            dollars[int(cfg.y)] = pos_y
            dollars[int(cfg.x)] = pos_x
        if day >= 2:
            last_return = np.diff(
                np.log(
                    np.maximum(
                        prices[:, day - 2 : day],
                        strategy._EPS_DENOM,
                    )
                ),
                axis=1,
            )[:, 0]
            crash = np.abs(last_return) > 0.40
            crash[0] = False
            dollars[crash] = 0.0
        limits = np.full(strategy.N_INST, strategy.LEG)
        limits[strategy.I["ALGO"]] = 100_000.0
        safe = np.maximum(
            prices[:, day - 1],
            strategy._EPS_DENOM,
        )
        targets[day] = np.clip(
            np.round(dollars / safe),
            -np.floor(limits / safe),
            np.floor(limits / safe),
        ).astype(int)
        rows.append(
            {
                "pair_name": profile.pair_name,
                "history_size": day,
                "pnl_day": day + 1,
                "active": active,
                "entry": entry,
                "exit": exit_event,
                "cut": bool(agent.current_cut),
                "vol_shift": bool(agent.current_vol_shift),
                "adapting": bool(agent.current_adapting),
                "z": float(agent.current_z),
                "kappa": float(agent.current_kappa),
            }
        )
    return targets, pd.DataFrame(rows)


def evaluate_targets(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows = []
    for variant, target in targets.items():
        result = evaluator.evaluate(prices, target, windows)
        for window, metrics in result.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def activity_summary(
    daily: pd.DataFrame,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    rows = []
    for pair_name in daily["pair_name"].unique():
        pair = daily[daily["pair_name"] == pair_name]
        for window, start, end in windows:
            use = pair[pair["pnl_day"].between(start, end)]
            rows.append(
                {
                    "pair_name": pair_name,
                    "window": window,
                    "active_days": int(use["active"].sum()),
                    "entries": int(use["entry"].sum()),
                    "cuts": int(use["cut"].sum()),
                    "vol_shifts": int(use["vol_shift"].sum()),
                    "adapting_days": int(use["adapting"].sum()),
                    "mean_abs_z": float(use["z"].abs().mean()),
                    "mean_kappa": float(use["kappa"].mean()),
                }
            )
    return pd.DataFrame(rows)


def delivery_summary(
    delivery: pd.DataFrame,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    if delivery.empty:
        return pd.DataFrame()
    rows = []
    for pair_name in delivery["pair_name"].unique():
        pair = delivery[delivery["pair_name"] == pair_name]
        for window, start, end in windows:
            use = pair[pair["pnl_day"].between(start, end)]
            rows.append(
                {
                    "pair_name": pair_name,
                    "window": window,
                    "signalled_pair_days": int(
                        use["history_size"].nunique()
                    ),
                    "signalled_legs": len(use),
                    "core_delivery_rate": float(
                        use["core_delivered"].mean()
                    ),
                    "final_delivery_rate": float(
                        use["final_delivered"].mean()
                    ),
                    "opposed_rate": float(
                        use["final_opposed"].mean()
                    ),
                    "flat_rate": float(use["final_flat"].mean()),
                    "ridge_conflict_rate": float(
                        use["ridge_conflict"].mean()
                    ),
                    "overlap_rate": float(
                        (use["pair_contributors"] > 1).mean()
                    ),
                    "mean_contributors": float(
                        use["pair_contributors"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def correlation_summary(
    profiles_: tuple[Profile, ...],
    prices: np.ndarray,
    strategy: types.ModuleType,
) -> pd.DataFrame:
    rows = []
    log_prices = np.log(prices)
    for profile in profiles_:
        y = strategy.I[profile.y_symbol]
        x = strategy.I[profile.x_symbol]
        for window, start, end in WINDOWS:
            returns = np.diff(
                log_prices[[y, x], start - 1 : end],
                axis=1,
            )
            correlation = float(
                np.corrcoef(returns[0], returns[1])[0, 1]
            )
            rows.append(
                {
                    "pair_name": profile.pair_name,
                    "window": window,
                    "return_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def profile_frame(profiles_: tuple[Profile, ...]) -> pd.DataFrame:
    rows = []
    for profile in profiles_:
        sensitivity = profile.perturbation_summary or []
        max_perturbation = max(
            (
                max(float(value) for value in item[1:])
                for item in sensitivity
            ),
            default=float("nan"),
        )
        rows.append(
            {
                "pair_name": profile.pair_name,
                "y_symbol": profile.y_symbol,
                "x_symbol": profile.x_symbol,
                "removal_class": profile.removal_class,
                "profile_source": profile.profile_source,
                "q_beta": profile.q_beta,
                "h": profile.h,
                "cusum_k": profile.cusum_k,
                "max_cov_inflate": profile.max_cov_inflate,
                "cov_decay": profile.cov_decay,
                "z_entry": profile.z_entry,
                "z_exit": profile.z_exit,
                "z_stop": profile.z_stop,
                "stored_auto_score": (
                    profile.stored_auto_metrics.get("score")
                    if profile.stored_auto_metrics
                    else np.nan
                ),
                "stored_test_score": (
                    (
                        profile.stored_test_metrics.get("auto")
                        or {}
                    ).get("score")
                    if profile.stored_test_metrics
                    else np.nan
                ),
                "max_parameter_perturbation_pct": max_perturbation,
            }
        )
    return pd.DataFrame(rows)


def marginal_summary(
    profiles_: tuple[Profile, ...],
    portfolio: pd.DataFrame,
    baseline_targets: np.ndarray,
    experiment_targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    pivot_score = portfolio.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    pivot_pnl = portfolio.pivot(
        index="variant",
        columns="window",
        values="total_pnl",
    )
    rows = []
    for profile in profiles_:
        if profile.pair_name == "ALGO-HRET":
            variant = "without_residual__ALGO-HRET"
            # Presence contribution is baseline minus the removal variant.
            multiplier = -1.0
            execution_status = "still_live_as_algo_residual"
        else:
            variant = f"add__{profile.pair_name}"
            multiplier = 1.0
            execution_status = "physically_absent_from_pair_book"
        score_deltas = {
            window: multiplier
            * (
                float(pivot_score.loc[variant, window])
                - float(pivot_score.loc["baseline", window])
            )
            for window, _, _ in WINDOWS
        }
        pnl_deltas = {
            window: multiplier
            * (
                float(pivot_pnl.loc[variant, window])
                - float(pivot_pnl.loc["baseline", window])
            )
            for window in PRIMARY_WINDOWS
        }
        if profile.pair_name == "ALGO-HRET":
            changed = np.count_nonzero(
                experiment_targets[variant] - baseline_targets
            )
        else:
            changed = np.count_nonzero(
                experiment_targets[variant] - baseline_targets
            )
        worst_half = min(
            score_deltas[window]
            for window in DEVELOPMENT_HALVES
        )
        if (
            score_deltas["TRAIN"] > 0.0
            and score_deltas["VALIDATION"] > 0.0
            and worst_half >= -5.0
        ):
            verdict = "development_reconsideration_candidate"
        elif (
            score_deltas["TRAIN"] < 0.0
            and score_deltas["VALIDATION"] < 0.0
        ):
            verdict = "currently_harmful_on_both_development_windows"
        else:
            verdict = "development_unstable_or_mixed"
        rows.append(
            {
                "pair_name": profile.pair_name,
                "removal_class": profile.removal_class,
                "execution_status": execution_status,
                "verdict": verdict,
                "changed_target_cells": int(changed),
                "worst_development_half_score_delta": worst_half,
                **{
                    f"{window}_score_delta": score_deltas[window]
                    for window, _, _ in WINDOWS
                },
                **{
                    f"{window}_total_pnl_delta": pnl_deltas[window]
                    for window in PRIMARY_WINDOWS
                },
            }
        )
    return pd.DataFrame(rows)


def write_report(
    marginal: pd.DataFrame,
    standalone: pd.DataFrame,
    activity: pd.DataFrame,
    delivery: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    standalone_score = standalone.pivot(
        index="variant",
        columns="window",
        values="score",
    )
    activity_pivot = activity.pivot(
        index="pair_name",
        columns="window",
        values="active_days",
    )
    correlation_pivot = correlations.pivot(
        index="pair_name",
        columns="window",
        values="return_correlation",
    )
    lines = [
        "# Removed-pair investigation",
        "",
        "Score deltas below are the contribution of the relationship's "
        "presence to today's complete portfolio. Test is descriptive only.",
        "",
        "| Pair | Provenance | Train delta | Validation delta | Test delta | "
        "Worst dev half | Standalone T/V/Test | Corr T/V/Test | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in marginal.sort_values(
        "VALIDATION_score_delta",
        ascending=False,
    ).iterrows():
        name = str(row["pair_name"])
        stand = standalone_score.loc[name]
        corr = correlation_pivot.loc[name]
        lines.append(
            f"| {name} | {row.removal_class} | "
            f"{row.TRAIN_score_delta:+.2f} | "
            f"{row.VALIDATION_score_delta:+.2f} | "
            f"{row.TEST_score_delta:+.2f} | "
            f"{row.worst_development_half_score_delta:+.2f} | "
            f"{stand.TRAIN:+.2f}/{stand.VALIDATION:+.2f}/"
            f"{stand.TEST:+.2f} | "
            f"{corr.TRAIN:+.2f}/{corr.VALIDATION:+.2f}/"
            f"{corr.TEST:+.2f} | {row.verdict} |"
        )
    lines.extend(
        [
            "",
            "## Activity",
            "",
            "| Pair | Active train | Active validation | Active test |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in activity_pivot.iterrows():
        lines.append(
            f"| {name} | {int(row.TRAIN)} | "
            f"{int(row.VALIDATION)} | {int(row.TEST)} |"
        )
    if not delivery.empty:
        delivery_pivot = delivery[
            delivery["window"].isin(PRIMARY_WINDOWS)
        ].pivot(
            index="pair_name",
            columns="window",
            values="final_delivery_rate",
        )
        lines.extend(
            [
                "",
                "## Portfolio delivery rate",
                "",
                "| Pair | Train | Validation | Test |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, row in delivery_pivot.iterrows():
            lines.append(
                f"| {name} | {row.TRAIN:.1%} | "
                f"{row.VALIDATION:.1%} | {row.TEST:.1%} |"
            )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices, symbols = load_prices()
    profiles_ = profiles()
    baseline_strategy = configure_experiment(None)
    baseline_targets, _ = collect_portfolio(
        baseline_strategy,
        prices,
        None,
    )
    portfolio_targets = {"baseline": baseline_targets}
    all_delivery = []

    for profile in profiles_:
        if profile.pair_name == "ALGO-HRET":
            # Prove ordinary reintroduction is suppressed by AtlasBook.
            main_strategy = configure_experiment(profile)
            main_targets, _ = collect_portfolio(
                main_strategy,
                prices,
                profile.pair_name,
            )
            portfolio_targets[
                "add_main__ALGO-HRET"
            ] = main_targets
            strategy = configure_experiment(
                None,
                remove_algo_hret_residual=True,
            )
            targets, delivery = collect_portfolio(
                strategy,
                prices,
                None,
            )
            portfolio_targets[
                "without_residual__ALGO-HRET"
            ] = targets
        else:
            strategy = configure_experiment(profile)
            targets, delivery = collect_portfolio(
                strategy,
                prices,
                profile.pair_name,
            )
            portfolio_targets[
                f"add__{profile.pair_name}"
            ] = targets
            if not delivery.empty:
                all_delivery.append(delivery)
        print(f"portfolio replay complete: {profile.pair_name}", flush=True)

    portfolio = evaluate_targets(
        prices,
        portfolio_targets,
        WINDOWS + BLOCKS,
    )
    portfolio.to_csv(OUTPUT_DIR / "portfolio.csv", index=False)

    standalone_targets: dict[str, np.ndarray] = {}
    all_activity = []
    for profile in profiles_:
        targets, daily = standalone_replay(profile, prices)
        standalone_targets[profile.pair_name] = targets
        all_activity.append(daily)
    standalone = evaluate_targets(
        prices,
        standalone_targets,
        WINDOWS + BLOCKS,
    )
    standalone.to_csv(OUTPUT_DIR / "standalone.csv", index=False)

    daily_activity = pd.concat(all_activity, ignore_index=True)
    daily_activity.to_csv(
        OUTPUT_DIR / "daily_activity.csv",
        index=False,
    )
    activity = activity_summary(
        daily_activity,
        WINDOWS + BLOCKS,
    )
    activity.to_csv(OUTPUT_DIR / "activity.csv", index=False)

    delivery_events = (
        pd.concat(all_delivery, ignore_index=True)
        if all_delivery
        else pd.DataFrame()
    )
    delivery_events.to_csv(
        OUTPUT_DIR / "delivery_events.csv",
        index=False,
    )
    delivery = delivery_summary(
        delivery_events,
        WINDOWS + BLOCKS,
    )
    delivery.to_csv(OUTPUT_DIR / "delivery.csv", index=False)

    strategy = fresh_strategy()
    correlations = correlation_summary(
        profiles_,
        prices,
        strategy,
    )
    correlations.to_csv(
        OUTPUT_DIR / "correlations.csv",
        index=False,
    )
    profile_table = profile_frame(profiles_)
    profile_table.to_csv(OUTPUT_DIR / "profiles.csv", index=False)

    marginal = marginal_summary(
        profiles_,
        portfolio,
        baseline_targets,
        portfolio_targets,
    )
    marginal.to_csv(OUTPUT_DIR / "marginal_summary.csv", index=False)
    write_report(
        marginal,
        standalone,
        activity,
        delivery,
        correlations,
    )

    algo_main_difference = int(
        np.count_nonzero(
            portfolio_targets["add_main__ALGO-HRET"]
            - baseline_targets
        )
    )
    metadata = {
        "symbols": symbols,
        "development_removed_names": sorted(
            profile.pair_name
            for profile in profiles_
            if profile.removal_class == "development_1_750"
        ),
        "previous_oos_removed_names": sorted(
            profile.pair_name
            for profile in profiles_
            if profile.removal_class == "previous_oos_751_1000"
        ),
        "algo_hret_main_reintroduction_different_cells": (
            algo_main_difference
        ),
        "test_usage": (
            "descriptive investigation only; no parameter or pair selection"
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        marginal[
            [
                "pair_name",
                "removal_class",
                "TRAIN_score_delta",
                "VALIDATION_score_delta",
                "TEST_score_delta",
                "worst_development_half_score_delta",
                "verdict",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
