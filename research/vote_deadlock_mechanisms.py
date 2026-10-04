"""Mechanism audit for pair-vote deadlocks in current four_sigmas.

Tasks covered:
1. exact ties, near-tie chatter, and persistent low-margin/flat runs;
2. adaptation-state desynchronisation and causal suppression controls;
3. pair-leg completion and realised-vs-intended neutrality.

The final 751-1000 window is labelled FINAL_IN_SAMPLE because the current
catalog already contains information influenced by that window.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "vote_deadlock_deepdive"
NEAR_TIE_MARGIN = 0.10
PERSISTENT_DAYS = 3
CUT_LOOKBACK = 5
PERIODS = (
    ("YEAR1", 31, 250),
    ("YEAR2", 251, 500),
    ("VALIDATION", 501, 750),
    ("FINAL_IN_SAMPLE", 751, 1000),
)


def _pair_config_key(spec: strategy.PairCfg) -> tuple[float | bool, ...]:
    return tuple(spec.filt) + tuple(spec.trade)


def duplicate_profile_audit() -> dict[str, object]:
    groups: dict[tuple[float | bool, ...], list[strategy.PairCfg]] = defaultdict(
        list
    )
    for spec in strategy.PAIR_CFGS:
        groups[_pair_config_key(spec)].append(spec)
    duplicates = [members for members in groups.values() if len(members) > 1]
    rows = []
    sharing_groups = 0
    for members in duplicates:
        names = [strategy._pair_name(spec) for spec in members]
        assets = [
            {int(spec.y), int(spec.x)}
            for spec in members
        ]
        leg_sharing = any(
            assets[left] & assets[right]
            for left in range(len(assets))
            for right in range(left + 1, len(assets))
        )
        sharing_groups += int(leg_sharing)
        rows.append(
            {
                "count": len(members),
                "pairs": names,
                "leg_sharing": bool(leg_sharing),
                "profile": [float(value) for value in _pair_config_key(members[0])],
            }
        )
    return {
        "duplicate_profile_groups": len(duplicates),
        "pairs_in_duplicate_groups": int(sum(len(group) for group in duplicates)),
        "leg_sharing_duplicate_groups": sharing_groups,
        "groups": sorted(rows, key=lambda row: (-row["count"], row["pairs"])),
    }


def custom_consensus(
    candidates: list[tuple[float, strategy.PairAgent, float, float]],
    include_pair: Callable[[strategy.PairAgent], bool],
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros(strategy.N_INST)
    denominator = np.zeros(strategy.N_INST)
    for priority, agent, pos_y, pos_x in candidates:
        if not include_pair(agent):
            continue
        boost = (
            strategy.PROTECTED_VOTE_BOOST
            if strategy._pair_name(agent.spec) in strategy.PROTECTED_PAIR_NAMES
            else 1.0
        )
        weight = (
            boost * max(float(priority), 1.0e-12)
            if strategy.PAIR_CONFLICT_MODE == "priority"
            else boost
        )
        for instrument, position in (
            (int(agent.spec.y), pos_y),
            (int(agent.spec.x), pos_x),
        ):
            if abs(position) <= 1.0e-9:
                continue
            numerator[instrument] += weight * np.sign(position)
            denominator[instrument] += weight
    vote = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return vote, denominator > 0.0


def resolve_direction(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
    crash: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ridge_active = ridge_sign != 0.0
    combined = (
        pair_vote
        + strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
    )
    active = pair_active | ridge_active
    direction = np.sign(combined)
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]
    direction[crash] = 0.0
    return direction, combined


def market_beta(past_returns: np.ndarray) -> np.ndarray:
    if past_returns.shape[1] < 20:
        return np.zeros(strategy.N_INST)
    window = past_returns[:, -min(60, past_returns.shape[1]) :]
    market = np.mean(window[1:], axis=0)
    centered_market = market - np.mean(market)
    denominator = float(centered_market @ centered_market)
    if denominator <= 1.0e-12:
        return np.zeros(strategy.N_INST)
    centered_assets = window - np.mean(window, axis=1, keepdims=True)
    return centered_assets @ centered_market / denominator


def exposure_stats(dollars: np.ndarray, beta: np.ndarray) -> tuple[float, float]:
    gross = float(np.sum(np.abs(dollars)))
    if gross <= 1.0e-12:
        return 0.0, 0.0
    dollar_net = float(np.sum(dollars) / gross)
    beta_net = float(dollars @ beta / gross)
    return dollar_net, beta_net


def _crash_mask(history: np.ndarray) -> np.ndarray:
    last_return = np.diff(
        np.log(np.maximum(history[:, -2:], strategy._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    return crash


def replay(prices: np.ndarray) -> dict[str, object]:
    n_history = prices.shape[1]
    baseline = np.zeros((n_history + 1, strategy.N_INST))
    suppress_adapting = np.zeros_like(baseline)
    suppress_mixed_pairs = np.zeros_like(baseline)
    asset_rows: list[dict[str, float | int | bool | str]] = []
    pair_rows: list[dict[str, float | int | bool | str]] = []
    exposure_rows: list[dict[str, float | int]] = []
    atlas = strategy.AtlasBook()
    last_cut: dict[int, int] = {
        int(spec.idx): -10_000 for spec in strategy.PAIR_CFGS
    }

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        candidates = atlas.candidates(history)
        if history_size < strategy.WARMUP + 10:
            continue
        for agent in atlas.agents:
            if agent.current_cut:
                last_cut[int(agent.spec.idx)] = history_size

        pair_vote, pair_active, contributors = (
            strategy._pair_conflict_consensus(candidates)
        )
        returns = np.diff(
            np.log(np.maximum(history, strategy._EPS_DENOM)),
            axis=1,
        )
        ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
        ridge_sign = np.sign(ridge_raw)
        crash = _crash_mask(history)
        direction, combined = resolve_direction(
            pair_vote,
            pair_active,
            ridge_sign,
            ridge_strength,
            crash,
        )
        baseline[history_size] = strategy.BANG_BANG_LIMITS * direction

        contributions: dict[int, list[dict[str, object]]] = defaultdict(list)
        intended = np.zeros(strategy.N_INST)
        for priority, agent, pos_y, pos_x in candidates:
            intended[int(agent.spec.y)] += pos_y
            intended[int(agent.spec.x)] += pos_x
            detail = {
                "agent": agent,
                "pair": strategy._pair_name(agent.spec),
                "priority": float(priority),
                "kappa": float(agent.current_kappa),
                "abs_z": float(abs(agent.current_z)),
                "side": int(agent.side),
                "cut": bool(agent.current_cut),
                "cut_recent": bool(
                    history_size - last_cut[int(agent.spec.idx)]
                    <= CUT_LOOKBACK
                ),
                "adapting": bool(agent.current_adapting),
            }
            contributions[int(agent.spec.y)].append(
                {**detail, "position": float(pos_y)}
            )
            contributions[int(agent.spec.x)].append(
                {**detail, "position": float(pos_x)}
            )

            y_match = direction[int(agent.spec.y)] == np.sign(pos_y)
            x_match = direction[int(agent.spec.x)] == np.sign(pos_x)
            pair_rows.append(
                {
                    "score_day": history_size + 1,
                    "pair": strategy._pair_name(agent.spec),
                    "both_legs_win": bool(y_match and x_match),
                    "exactly_one_leg_wins": bool(y_match != x_match),
                    "neither_leg_wins": bool(not y_match and not x_match),
                    "y_flat": bool(direction[int(agent.spec.y)] == 0.0),
                    "x_flat": bool(direction[int(agent.spec.x)] == 0.0),
                    "adapting": bool(agent.current_adapting),
                    "cut": bool(agent.current_cut),
                }
            )

        mixed_assets: set[int] = set()
        for instrument, details in contributions.items():
            states = [bool(detail["adapting"]) for detail in details]
            if any(states) and not all(states):
                mixed_assets.add(instrument)

        adapting_vote, adapting_active = custom_consensus(
            candidates,
            include_pair=lambda agent: not agent.current_adapting,
        )
        adapting_direction, _ = resolve_direction(
            adapting_vote,
            adapting_active,
            ridge_sign,
            ridge_strength,
            crash,
        )
        suppress_adapting[history_size] = (
            strategy.BANG_BANG_LIMITS * adapting_direction
        )

        mixed_pair_indices = {
            int(agent.spec.idx)
            for _, agent, _, _ in candidates
            if (
                int(agent.spec.y) in mixed_assets
                or int(agent.spec.x) in mixed_assets
            )
        }
        sync_vote, sync_active = custom_consensus(
            candidates,
            include_pair=lambda agent: (
                int(agent.spec.idx) not in mixed_pair_indices
            ),
        )
        sync_direction, _ = resolve_direction(
            sync_vote,
            sync_active,
            ridge_sign,
            ridge_strength,
            crash,
        )
        suppress_mixed_pairs[history_size] = (
            strategy.BANG_BANG_LIMITS * sync_direction
        )

        for instrument, details in contributions.items():
            positions = np.asarray(
                [float(detail["position"]) for detail in details]
            )
            adapting_states = np.asarray(
                [bool(detail["adapting"]) for detail in details]
            )
            cut_states = np.asarray(
                [bool(detail["cut"]) for detail in details]
            )
            recent_cut_states = np.asarray(
                [bool(detail["cut_recent"]) for detail in details]
            )
            conflict = bool(
                np.any(positions > 0.0) and np.any(positions < 0.0)
            )
            exact_tie = bool(
                len(details) >= 2
                and pair_active[instrument]
                and abs(pair_vote[instrument]) <= 1.0e-12
                and ridge_sign[instrument] == 0.0
                and not crash[instrument]
            )
            asset_rows.append(
                {
                    "score_day": history_size + 1,
                    "instrument": instrument,
                    "symbol": strategy._SYMBOL_BY_INDEX[instrument],
                    "pair_vote": float(pair_vote[instrument]),
                    "combined_vote": float(combined[instrument]),
                    "pair_contributors": len(details),
                    "direction": int(direction[instrument]),
                    "conflict": conflict,
                    "exact_tie": exact_tie,
                    "adapting_fraction": float(np.mean(adapting_states)),
                    "mixed_adaptation": bool(
                        np.any(adapting_states)
                        and not np.all(adapting_states)
                    ),
                    "any_cut": bool(np.any(cut_states)),
                    "recent_cut_fraction": float(
                        np.mean(recent_cut_states)
                    ),
                    "mixed_recent_cut": bool(
                        np.any(recent_cut_states)
                        and not np.all(recent_cut_states)
                    ),
                    "crash": bool(crash[instrument]),
                }
            )

        beta = market_beta(returns)
        intended_dollar_net, intended_beta_net = exposure_stats(
            intended,
            beta,
        )
        realised_dollar_net, realised_beta_net = exposure_stats(
            baseline[history_size],
            beta,
        )
        exposure_rows.append(
            {
                "score_day": history_size + 1,
                "intended_dollar_net": intended_dollar_net,
                "realised_dollar_net": realised_dollar_net,
                "intended_beta_net": intended_beta_net,
                "realised_beta_net": realised_beta_net,
                "intended_gross": float(np.sum(np.abs(intended))),
                "realised_gross": float(
                    np.sum(np.abs(baseline[history_size]))
                ),
            }
        )

    return {
        "baseline": baseline,
        "suppress_adapting": suppress_adapting,
        "suppress_mixed_pairs": suppress_mixed_pairs,
        "asset_rows": pd.DataFrame(asset_rows),
        "pair_rows": pd.DataFrame(pair_rows),
        "exposure_rows": pd.DataFrame(exposure_rows),
    }


def annotate_failure_modes(asset_rows: pd.DataFrame) -> pd.DataFrame:
    rows = asset_rows.sort_values(["instrument", "score_day"]).copy()
    rows["near_tie"] = (
        (np.abs(rows["combined_vote"]) <= NEAR_TIE_MARGIN)
        & ~rows["exact_tie"]
        & (rows["direction"] != 0)
    )
    rows["near_tie_chatter"] = False
    rows["persistent_low_margin"] = False
    rows["persistent_flat"] = False
    rows["low_margin_run_length"] = 0
    rows["flat_run_length"] = 0

    for _, indices in rows.groupby("instrument").groups.items():
        group = rows.loc[indices].sort_values("score_day")
        days = group["score_day"].to_numpy(dtype=int)
        directions = group["direction"].to_numpy(dtype=int)
        near = group["near_tie"].to_numpy(dtype=bool)
        chatter = np.zeros(len(group), dtype=bool)
        if len(group) > 1:
            chatter[1:] = (
                (days[1:] == days[:-1] + 1)
                & near[1:]
                & near[:-1]
                & (directions[1:] == -directions[:-1])
            )
        rows.loc[group.index, "near_tie_chatter"] = chatter

        for column, output, persistent_output in (
            (
                (
                    np.abs(group["combined_vote"].to_numpy(dtype=float))
                    <= NEAR_TIE_MARGIN
                ),
                "low_margin_run_length",
                "persistent_low_margin",
            ),
            (
                group["exact_tie"].to_numpy(dtype=bool),
                "flat_run_length",
                "persistent_flat",
            ),
        ):
            run_lengths = np.zeros(len(group), dtype=int)
            start = 0
            while start < len(group):
                if not column[start]:
                    start += 1
                    continue
                end = start + 1
                while (
                    end < len(group)
                    and column[end]
                    and days[end] == days[end - 1] + 1
                ):
                    end += 1
                run_lengths[start:end] = end - start
                start = end
            rows.loc[group.index, output] = run_lengths
            rows.loc[group.index, persistent_output] = (
                run_lengths >= PERSISTENT_DAYS
            )
    return rows


def conditional_rate(
    frame: pd.DataFrame,
    outcome: str,
    condition: str,
) -> dict[str, float]:
    positive = frame[condition].astype(bool)
    target = frame[outcome].astype(bool)
    a = float(np.sum(positive & target))
    b = float(np.sum(positive & ~target))
    c = float(np.sum(~positive & target))
    d = float(np.sum(~positive & ~target))
    return {
        "condition_true_rate": float(target[positive].mean())
        if np.any(positive)
        else 0.0,
        "condition_false_rate": float(target[~positive].mean())
        if np.any(~positive)
        else 0.0,
        "risk_ratio": float(
            (a / max(a + b, 1.0)) / max(c / max(c + d, 1.0), 1.0e-12)
        ),
        "odds_ratio": float(
            ((a + 0.5) * (d + 0.5))
            / ((b + 0.5) * (c + 0.5))
        ),
        "correlation": float(
            np.corrcoef(
                positive.to_numpy(dtype=float),
                target.to_numpy(dtype=float),
            )[0, 1]
        ),
        "condition_observations": int(np.sum(positive)),
    }


def metrics_by_period(
    prices: np.ndarray,
    dollars: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(prices, dollars, start, end)
        for name, start, end in PERIODS
    }


def period_frame(
    frame: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    return frame[
        (frame["score_day"] >= start)
        & (frame["score_day"] <= end)
    ]


def count_flag_runs(frame: pd.DataFrame, flag: str) -> int:
    count = 0
    for _, group in frame.groupby("instrument"):
        ordered = group.sort_values("score_day")
        active = ordered[flag].to_numpy(dtype=bool)
        days = ordered["score_day"].to_numpy(dtype=int)
        if len(active) == 0:
            continue
        count += int(active[0])
        if len(active) > 1:
            count += int(
                np.sum(
                    active[1:]
                    & (
                        ~active[:-1]
                        | (days[1:] != days[:-1] + 1)
                    )
                )
            )
    return count


def oracle_movement(
    prices: np.ndarray,
    frame: pd.DataFrame,
) -> float:
    if len(frame) == 0:
        return 0.0
    instruments = frame["instrument"].to_numpy(dtype=int)
    days = frame["score_day"].to_numpy(dtype=int)
    return float(
        np.sum(
            strategy.LEG
            * np.abs(
                prices[instruments, days - 1]
                / prices[instruments, days - 2]
                - 1.0
            )
        )
    )


def summarize(
    prices: np.ndarray,
    replayed: dict[str, object],
    duplicates: dict[str, object],
) -> dict[str, object]:
    asset_rows = annotate_failure_modes(replayed["asset_rows"])
    pair_rows = replayed["pair_rows"]
    exposure_rows = replayed["exposure_rows"]
    asset_rows.to_csv(OUTPUT_DIR / "asset_day_diagnostics.csv", index=False)
    pair_rows.to_csv(OUTPUT_DIR / "pair_leg_completion.csv", index=False)
    exposure_rows.to_csv(OUTPUT_DIR / "book_neutrality.csv", index=False)

    baseline_metrics = metrics_by_period(prices, replayed["baseline"])
    adapt_metrics = metrics_by_period(prices, replayed["suppress_adapting"])
    sync_metrics = metrics_by_period(prices, replayed["suppress_mixed_pairs"])
    summary: dict[str, object] = {
        "protocol": {
            "mode_value": strategy.PAIR_CONFLICT_MODE,
            "effective_pair_vote_mode": (
                "priority"
                if strategy.PAIR_CONFLICT_MODE == "priority"
                else "equal"
            ),
            "near_tie_margin": NEAR_TIE_MARGIN,
            "persistent_days": PERSISTENT_DAYS,
            "final_window_status": (
                "in-sample/contaminated by catalog construction"
            ),
        },
        "duplicates": duplicates,
        "failure_modes": {},
        "desynchronisation": {},
        "leg_completion": {},
        "neutrality": {},
        "strategy_controls": {
            "baseline": baseline_metrics,
            "suppress_adapting": adapt_metrics,
            "suppress_mixed_pairs": sync_metrics,
            "suppress_adapting_delta": {},
            "suppress_mixed_pairs_delta": {},
        },
    }

    for name, start, end in PERIODS:
        assets = period_frame(asset_rows, start, end)
        overlapping = assets[assets["pair_contributors"] >= 2]
        pairs = period_frame(pair_rows, start, end)
        exposures = period_frame(exposure_rows, start, end)
        chatter_commission = float(
            np.sum(assets["near_tie_chatter"])
            * 2.0
            * strategy.LEG
            * 1.0e-4
        )
        exact = assets[assets["exact_tie"]]
        persistent_flat = assets[assets["persistent_flat"]]
        summary["failure_modes"][name] = {
            "overlap_asset_days": int(len(overlapping)),
            "exact_tie_asset_days": int(len(exact)),
            "exact_tie_share": float(
                len(exact) / max(len(overlapping), 1)
            ),
            "exact_tie_oracle_movement": oracle_movement(prices, exact),
            "near_tie_asset_days": int(assets["near_tie"].sum()),
            "near_tie_chatter_flips": int(
                assets["near_tie_chatter"].sum()
            ),
            "near_tie_chatter_commission": chatter_commission,
            "persistent_low_margin_asset_days": int(
                assets["persistent_low_margin"].sum()
            ),
            "persistent_low_margin_runs": int(
                count_flag_runs(assets, "persistent_low_margin")
            ),
            "persistent_flat_asset_days": int(
                assets["persistent_flat"].sum()
            ),
            "persistent_flat_runs": int(
                count_flag_runs(assets, "persistent_flat")
            ),
            "persistent_flat_oracle_movement": oracle_movement(
                prices, persistent_flat
            ),
        }
        summary["desynchronisation"][name] = {
            "exact_tie_given_mixed_adaptation": conditional_rate(
                overlapping,
                "exact_tie",
                "mixed_adaptation",
            ),
            "persistent_low_given_mixed_adaptation": conditional_rate(
                overlapping,
                "persistent_low_margin",
                "mixed_adaptation",
            ),
            "exact_tie_given_recent_cut_mix": conditional_rate(
                overlapping,
                "exact_tie",
                "mixed_recent_cut",
            ),
        }
        summary["leg_completion"][name] = {
            "active_pair_days": int(len(pairs)),
            "both_legs_win": float(pairs["both_legs_win"].mean()),
            "exactly_one_leg_wins": float(
                pairs["exactly_one_leg_wins"].mean()
            ),
            "neither_leg_wins": float(pairs["neither_leg_wins"].mean()),
            "at_least_one_leg_flat": float(
                (pairs["y_flat"] | pairs["x_flat"]).mean()
            ),
        }
        summary["neutrality"][name] = {
            "median_abs_intended_dollar_net": float(
                np.median(np.abs(exposures["intended_dollar_net"]))
            ),
            "median_abs_realised_dollar_net": float(
                np.median(np.abs(exposures["realised_dollar_net"]))
            ),
            "median_abs_intended_market_beta": float(
                np.median(np.abs(exposures["intended_beta_net"]))
            ),
            "median_abs_realised_market_beta": float(
                np.median(np.abs(exposures["realised_beta_net"]))
            ),
            "mean_intended_gross": float(exposures["intended_gross"].mean()),
            "mean_realised_gross": float(exposures["realised_gross"].mean()),
        }
        for control, metrics, output in (
            (
                "suppress_adapting",
                adapt_metrics,
                "suppress_adapting_delta",
            ),
            (
                "suppress_mixed_pairs",
                sync_metrics,
                "suppress_mixed_pairs_delta",
            ),
        ):
            summary["strategy_controls"][output][name] = {
                "score": (
                    metrics[name]["score"] - baseline_metrics[name]["score"]
                ),
                "total_pnl": (
                    metrics[name]["total_pnl"]
                    - baseline_metrics[name]["total_pnl"]
                ),
            }
    return summary


def write_report(summary: dict[str, object]) -> None:
    lines = [
        "# Vote deadlock mechanism audit",
        "",
        f"- Configured mode value: `{summary['protocol']['mode_value']}`",
        f"- Effective mode: **{summary['protocol']['effective_pair_vote_mode']}**",
        f"- Duplicate post-merge profiles: "
        f"**{summary['duplicates']['duplicate_profile_groups']} groups**",
        "",
        "## Failure modes",
        "",
        "| Period | Exact ties | Near-tie flips | Flip commission | "
        "Persistent-flat days | Persistent oracle movement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in PERIODS:
        row = summary["failure_modes"][name]
        lines.append(
            f"| {name} | {row['exact_tie_asset_days']} | "
            f"{row['near_tie_chatter_flips']} | "
            f"${row['near_tie_chatter_commission']:.2f} | "
            f"{row['persistent_flat_asset_days']} | "
            f"${row['persistent_flat_oracle_movement']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pair-leg completion",
            "",
            "| Period | Both legs | Exactly one | Neither | Any flat leg |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, _, _ in PERIODS:
        row = summary["leg_completion"][name]
        lines.append(
            f"| {name} | {row['both_legs_win']:.1%} | "
            f"{row['exactly_one_leg_wins']:.1%} | "
            f"{row['neither_leg_wins']:.1%} | "
            f"{row['at_least_one_leg_flat']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The final window is deliberately not described as out-of-sample:",
            "`PREVIOUS_OOS_REMOVED_PAIR_NAMES` and the current catalog encode",
            "information influenced by days 751-1000.",
        ]
    )
    (OUTPUT_DIR / "mechanism_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(1000)
    duplicates = duplicate_profile_audit()
    replayed = replay(prices)
    summary = summarize(prices, replayed, duplicates)
    (OUTPUT_DIR / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
