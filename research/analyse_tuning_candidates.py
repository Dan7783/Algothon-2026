"""Re-evaluate manual/auto tuning candidates on their labelled pairs.

The batch tuner historically evaluated every candidate through a testbed that
was hard-coded to HUXZ-ACAC.  This script bypasses that problem by constructing
each PairAgent directly, and checks:

* year 1 (days 2-250), year 2 (251-500), and OOS year 3 (501-750);
* OOS decay relative to the two-year training result;
* fresh entry counts, so warmup carry-over is not mistaken for repeatability;
* one-at-a-time +/-10% parameter perturbations.

The auto results in the affected export omit z_exit.  In that run, the
environment retained the immediately preceding manual z_exit, so that value is
filled explicitly here.

Usage:
    py research/analyse_tuning_candidates.py INPUT_JSON [OUTPUT_JSON]
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clean_strat as strategy


COMM = np.full(strategy.N_INST, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(strategy.N_INST, 10_000.0)
LIMIT[0] = 100_000.0
PARAMS = (
    "q_beta",
    "h",
    "cusum_k",
    "max_cov_inflate",
    "kappa",
    "sigma",
    "z_entry",
    "z_exit",
    "kelly_cap",
    "alarm_damp",
)
WINDOWS = {
    "year1": (2, 250),
    "year2": (251, 500),
    "train": (2, 500),
    "test": (501, 750),
}


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def make_cfg(row: dict[str, Any], params: dict[str, float]) -> strategy.PairCfg:
    missing = [name for name in PARAMS if name not in params]
    if missing:
        raise ValueError(f"{row['pair_name']} is missing {missing}")
    return strategy.PairCfg(
        idx=0,
        y=strategy.I[row["y_symbol"]],
        x=strategy.I[row["x_symbol"]],
        filt=strategy.FiltCfg(
            q_beta=float(params["q_beta"]),
            cusum_k=float(params["cusum_k"]),
            h=float(params["h"]),
            max_cov_inflate=float(params["max_cov_inflate"]),
        ),
        trade=strategy.TradeCfg(
            z_entry=float(params["z_entry"]),
            z_exit=float(params["z_exit"]),
            z_stop=10.0,
            kappa=float(params["kappa"]),
            sigma=float(params["sigma"]),
            kelly_cap=float(params["kelly_cap"]),
            alarm_damp=float(params["alarm_damp"]),
        ),
    )


def replay_pair(
    prices: np.ndarray, cfg: strategy.PairCfg
) -> tuple[dict[int, np.ndarray], list[int], int]:
    agent = strategy.PairAgent(cfg)
    shares: dict[int, np.ndarray] = {}
    entries: list[int] = []
    active_days = 0
    previous_side = 0

    for t in range(1, prices.shape[1] + 1):
        position = np.zeros(strategy.N_INST)
        if t >= strategy.WARMUP + 10:
            agent.catch_up(prices[:, :t])
            if t > strategy.WARMUP + 10 and previous_side == 0 and agent.side != 0:
                entries.append(t)
            previous_side = agent.side
            if agent.side != 0:
                active_days += 1

            py, px, _ = agent.dollar_positions()
            position[cfg.y] = py
            position[cfg.x] = px

            last_return = np.diff(np.log(prices[:, t - 2 : t]), axis=1)[:, 0]
            crash = np.abs(last_return) > 0.40
            crash[0] = False
            position[crash] = 0.0
            position[1:] = np.clip(position[1:], -strategy.LEG, strategy.LEG)

        shares[t] = (position / prices[:, t - 1]).astype(int)

    return shares, entries, active_days


def account(
    prices: np.ndarray, shares: dict[int, np.ndarray], start: int, end: int
) -> dict[str, float]:
    cash = value = commission = volume = 0.0
    current = np.zeros(strategy.N_INST)
    pnl: list[float] = []

    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        if t < end:
            requested = shares[t]
            limits = (LIMIT / current_prices).astype(int)
            new_position = np.clip(requested, -limits, limits).astype(float)
        else:
            new_position = current.copy()

        delta = new_position - current
        cash -= float(current_prices.dot(delta)) + commission
        daily_volume = current_prices * np.abs(delta)
        volume += float(daily_volume.sum())
        commission = float(np.sum(daily_volume * COMM))
        current = new_position
        new_value = cash + float(current.dot(current_prices))
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)

    values = np.asarray(pnl)
    mean = float(values.mean())
    std = float(values.std())
    return {
        "meanpl": mean,
        "std": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0 else 0.0,
        "score": score(mean, std),
        "total_return": float(values.sum()),
        "volume": volume,
    }


def evaluate(
    prices: np.ndarray, row: dict[str, Any], params: dict[str, float]
) -> dict[str, Any]:
    cfg = make_cfg(row, params)
    shares, entries, active_days = replay_pair(prices, cfg)
    result = {name: account(prices, shares, *window) for name, window in WINDOWS.items()}
    result["entries"] = {
        "year1": sum(2 <= day <= 250 for day in entries),
        "year2": sum(251 <= day <= 500 for day in entries),
        "test": sum(501 <= day <= 750 for day in entries),
    }
    result["active_days"] = active_days
    return result


def replay_portfolio(
    prices: np.ndarray, configs: tuple[strategy.PairCfg, ...]
) -> dict[str, dict[str, float]]:
    original_configs = strategy.PAIR_CFGS
    shares: dict[int, np.ndarray] = {}
    try:
        strategy.PAIR_CFGS = configs
        strategy.reset()
        for t in range(1, prices.shape[1] + 1):
            shares[t] = np.asarray(strategy.getMyPosition(prices[:, :t]), dtype=int)
    finally:
        strategy.PAIR_CFGS = original_configs
        strategy.reset()
    return {
        name: account(prices, shares, *window)
        for name, window in WINDOWS.items()
    }


def candidate_summary(
    prices: np.ndarray,
    row: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    params = {key: float(value) for key, value in row[f"{method}_params"].items()}
    inferred_exit = False
    if "z_exit" not in params:
        params["z_exit"] = float(row["manual_params"]["z_exit"])
        inferred_exit = True

    baseline = evaluate(prices, row, params)
    train_mean = baseline["train"]["meanpl"]
    test_mean = baseline["test"]["meanpl"]
    lower_mean = min(train_mean, test_mean)
    decay = (test_mean - train_mean) / max(abs(train_mean), 1.0e-9)

    neighbours: list[dict[str, Any]] = []
    fragile_parameters: list[str] = []
    for name in PARAMS:
        parameter_floors: list[float] = []
        for factor in (0.90, 1.10):
            perturbed = dict(params)
            perturbed[name] = params[name] * factor
            metrics = evaluate(prices, row, perturbed)
            floor = min(metrics["train"]["meanpl"], metrics["test"]["meanpl"])
            parameter_floors.append(floor)
            neighbours.append(
                {
                    "param": name,
                    "factor": factor,
                    "lower_meanpl": floor,
                    "train_meanpl": metrics["train"]["meanpl"],
                    "test_meanpl": metrics["test"]["meanpl"],
                }
            )
        if lower_mean > 0.0 and min(parameter_floors) < 0.5 * lower_mean:
            fragile_parameters.append(name)

    neighbour_floors = np.asarray([item["lower_meanpl"] for item in neighbours])
    p10_floor = float(np.percentile(neighbour_floors, 10))
    median_floor = float(np.median(neighbour_floors))
    eligible = (
        lower_mean > 0.0
        and p10_floor > 0.0
        and baseline["entries"]["year2"] > 0
        and baseline["entries"]["test"] > 0
    )
    return {
        "method": method,
        "params": params,
        "inferred_z_exit": inferred_exit,
        "baseline": baseline,
        "oos_decay": decay,
        "lower_meanpl": lower_mean,
        "permutation_p10_lower_meanpl": p10_floor,
        "permutation_median_lower_meanpl": median_floor,
        "fragile_parameters": fragile_parameters,
        "eligible": eligible,
        "neighbours": neighbours,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyse_tuning_candidates.py INPUT_JSON [OUTPUT_JSON]")
    input_path = Path(sys.argv[1])
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "cusum_output" / "clean_strat_candidate_analysis.json"
    )

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    report: list[dict[str, Any]] = []

    print(
        f"{'pair':>11} | {'choice':>6} | {'train':>8} {'test':>8} "
        f"{'decay':>7} {'p10':>8} | {'entries y2/y3':>13} | status",
        flush=True,
    )
    print("-" * 100, flush=True)
    for index, row in enumerate(rows, 1):
        manual = candidate_summary(prices, row, "manual")
        auto = candidate_summary(prices, row, "auto")
        eligible = [item for item in (manual, auto) if item["eligible"]]
        selected = (
            max(eligible, key=lambda item: item["permutation_p10_lower_meanpl"])
            if eligible
            else None
        )
        item = {
            "pair_name": row["pair_name"],
            "y_symbol": row["y_symbol"],
            "x_symbol": row["x_symbol"],
            "manual": manual,
            "auto": auto,
            "selected_method": selected["method"] if selected else None,
            "selected_params": selected["params"] if selected else None,
        }
        report.append(item)

        shown = selected or max((manual, auto), key=lambda x: x["lower_meanpl"])
        base = shown["baseline"]
        entries = base["entries"]
        status = "SELECT" if selected else "REJECT"
        if shown["fragile_parameters"]:
            status += f" fragile={','.join(shown['fragile_parameters'])}"
        print(
            f"{row['pair_name']:>11} | {shown['method']:>6} | "
            f"{base['train']['meanpl']:8.2f} {base['test']['meanpl']:8.2f} "
            f"{shown['oos_decay']:7.0%} {shown['permutation_p10_lower_meanpl']:8.2f} | "
            f"{entries['year2']:5d}/{entries['test']:<5d} | {status}",
            flush=True,
        )
        if index % 5 == 0:
            print(f"completed {index}/{len(rows)}", flush=True)

    inverse_symbols = {index: symbol for symbol, index in strategy.I.items()}
    current_configs = tuple(strategy.PAIR_CFGS)
    current_by_name = {
        f"{inverse_symbols[cfg.y]}-{inverse_symbols[cfg.x]}": cfg
        for cfg in current_configs
    }
    replacement_by_name = dict(current_by_name)
    replacements: list[str] = []
    row_by_name = {row["pair_name"]: row for row in rows}
    for item in report:
        if item["selected_method"] is None or item["pair_name"] not in current_by_name:
            continue
        replacement_by_name[item["pair_name"]] = make_cfg(
            row_by_name[item["pair_name"]], item["selected_params"]
        )
        replacements.append(item["pair_name"])
    replacement_configs = tuple(
        replacement_by_name[
            f"{inverse_symbols[cfg.y]}-{inverse_symbols[cfg.x]}"
        ]
        for cfg in current_configs
    )
    portfolio = {
        "current": replay_portfolio(prices, current_configs),
        "eligible_replacements": replay_portfolio(prices, replacement_configs),
        "replaced_pairs": replacements,
    }

    payload = {
        "methodology": {
            "training_days": [2, 500],
            "oos_days": [501, 750],
            "perturbation": "one parameter at a time, -10% and +10%",
            "eligibility": (
                "positive train/test lower-bound and perturbation p10, with at "
                "least one fresh entry in year 2 and year 3"
            ),
            "auto_z_exit": (
                "filled from manual z_exit because the affected auto export "
                "omitted it and inherited that environment value"
            ),
        },
        "pairs": report,
        "portfolio": portfolio,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selected_count = sum(item["selected_method"] is not None for item in report)
    manual_count = sum(item["selected_method"] == "manual" for item in report)
    auto_count = sum(item["selected_method"] == "auto" for item in report)
    print(
        f"\nselected {selected_count}/{len(report)} "
        f"(manual={manual_count}, auto={auto_count}); wrote {output_path}",
        flush=True,
    )
    for label, metrics in portfolio.items():
        if label == "replaced_pairs":
            continue
        print(
            f"{label}: train mean={metrics['train']['meanpl']:.2f} "
            f"score={metrics['train']['score']:.2f}; "
            f"test mean={metrics['test']['meanpl']:.2f} "
            f"score={metrics['test']['score']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
