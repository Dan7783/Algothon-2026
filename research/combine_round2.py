#!/usr/bin/env python
"""Round 2: the two levers the round-1 sweep actually exposed.

Round 1 established (see combine_schemes.py) that under this scoring function
-- score = mu * SR^2/(SR^2+1) -- the dominant term is how much capital is
deployed, because scaling every position by k scales mu and sigma together,
leaving SR unchanged while raising mu. Every scheme that reduced exposure
(conflict-flat, proportional sizing, agreement weighting) lost, and the
schemes ranked almost monotonically by gross dollars deployed.

Among the three schemes with IDENTICAL deployment (baseline, pairs-win,
ridge-win: all $435,400/day on 43.5 instruments) the ordering was perfectly
monotone in turnover -- $181k -> 865, $286k -> 790, $386k -> 623 on train.
That isolates turnover as the second lever, independent of capital.

So round 2 tests exactly two things, both structural rather than predictive:

  LEVER A -- unused capacity. Instrument 0 (ALGO) carries a 100k limit
  (10x every other instrument) and a 0.2bp commission (5x cheaper), but the
  working copy sets BANG_BANG_LIMITS[ALGO] = 0, so it never trades: 100k of
  a possible 590k book capacity, permanently idle. The ALGO-HRET pair is
  live on ~18% of days and would drive it. The counter-risk is real and must
  be measured, not assumed: ALGO's hedge leg (HRET) is capped at 10k, so a
  100k ALGO leg is 10:1 unhedged and could inject naked market beta into a
  book whose edge has always been market-neutrality.

  LEVER B -- turnover. A one-parameter stickiness family: hold the existing
  direction unless the combined vote exceeds theta. theta=0 is exactly the
  shipped baseline, so this is a strict generalisation and a plateau (not a
  spike) is required before it counts.

Selection remains train-only (both blocks independently), per
cusum_output/VALIDATION_PROTOCOL.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.combine_lab import (
    DLR_LIMIT,
    N_INST,
    TEST,
    TEST_BLOCKS,
    TRAIN,
    TRAIN_BLOCKS,
    VAL_BLOCKS,
    VALIDATION,
    crash_mask,
    region_report,
    score,
    simulate,
)
from research.combine_schemes import _resolve, _sleeve_state
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"


def make_limits(algo_limit: float) -> np.ndarray:
    limits = np.full(N_INST, strategy.LEG)
    limits[0] = algo_limit
    return limits


def evaluate_sticky(cache, prices, start, end, algo_limit: float, theta: float):
    """Baseline arbitration + optional ALGO capacity + optional stickiness."""
    lookup = {int(day): row for row, day in enumerate(cache["days"])}
    limits = make_limits(algo_limit)
    prev_direction = np.zeros(N_INST)
    dollars_by_day = {}
    for t in range(start - 1, end):
        state = _sleeve_state(cache, lookup[t], prices, t)
        combined = state["consensus"] + state["ridge_vote"]
        target = _resolve(state)
        if theta > 0.0:
            hold = np.abs(combined) < theta
            direction = np.where(hold, prev_direction, target)
        else:
            direction = target
        direction = direction.copy()
        direction[state["crash"]] = 0.0
        prev_direction = direction
        dollars_by_day[t] = np.clip(limits * direction, -DLR_LIMIT, DLR_LIMIT)
    return simulate(dollars_by_day, prices, start, end)


def market_beta(pnl: np.ndarray, prices: np.ndarray, start: int, end: int) -> float:
    """Regress daily PnL on the equal-weight market return; report R^2."""
    market = np.diff(np.log(prices[1:, start - 1:end + 1]), axis=1).mean(axis=0)
    market = market[: len(pnl)]
    if market.std() < 1e-12:
        return float("nan")
    corr = np.corrcoef(pnl, market)[0, 1]
    return float(corr * corr)


def run_grid(cache, prices, algo_limits, thetas):
    rows = []
    for algo_limit in algo_limits:
        for theta in thetas:
            record = {"algo_limit": algo_limit, "theta": theta}
            for label, (_, start, end), blocks in (
                ("train", TRAIN, TRAIN_BLOCKS),
                ("val", VALIDATION, VAL_BLOCKS),
                ("test", TEST, TEST_BLOCKS),
            ):
                pnl = evaluate_sticky(cache, prices, start, end, algo_limit, theta)
                report = region_report(pnl, start, end, blocks)
                record[f"{label}_score"] = report["score"]
                record[f"{label}_floor"] = report["floor"]
                record[f"{label}_b1"] = report["blocks"][0]
                record[f"{label}_b2"] = report["blocks"][1]
                if label == "train":
                    record["train_r2_market"] = market_beta(pnl, prices, start, end)
            rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()

    print("LEVER A: ALGO capacity (theta=0, i.e. shipped arbitration)")
    print("=" * 96)
    algo_frame = run_grid(cache, prices, [0.0, 10_000.0, 25_000.0, 50_000.0, 100_000.0], [0.0])
    print(algo_frame[["algo_limit", "train_b1", "train_b2", "train_score", "train_floor",
                      "train_r2_market", "val_score", "val_floor", "test_score", "test_floor"]]
          .round(3).to_string(index=False))

    print("\nLEVER B: turnover stickiness (ALGO kept at its current 0)")
    print("=" * 96)
    theta_frame = run_grid(cache, prices, [0.0],
                           [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    print(theta_frame[["theta", "train_b1", "train_b2", "train_score", "train_floor",
                       "val_score", "val_floor", "test_score", "test_floor"]]
          .round(2).to_string(index=False))

    frame = pd.concat([algo_frame, theta_frame], ignore_index=True)
    frame.to_csv(OUTPUT_DIR / "round2_levers.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR / 'round2_levers.csv'}")


if __name__ == "__main__":
    main()
