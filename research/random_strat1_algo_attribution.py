"""Attribute the directly traded ALGO leg inside ``random_strat1``.

For each evaluation window the strategy state starts flat, matching ``eval.py``.
The strategy is then called sequentially from the close immediately before the
window through the penultimate close.  The resulting position path is evaluated
three ways without refitting or changing its forecasts:

* the complete 51-instrument portfolio;
* ALGO alone; and
* all other instruments with ALGO forced to zero.

Daily PnL and fees are additive, while the competition score is not.  Therefore
the full-minus-rest score difference measures ALGO's realised portfolio
interaction, not merely ALGO's standalone score.
"""

from __future__ import annotations

import importlib
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICE_PATH = ROOT / "prices.txt"
OUT_DIR = ROOT / "analysis_outputs" / "random_strat1_algo"
SUMMARY_PATH = OUT_DIR / "summary.csv"
DAILY_PATH = OUT_DIR / "daily_attribution.csv"
REPORT_PATH = OUT_DIR / "report.md"

N_INST = 51
ALGO = 0
COMMISSIONS = np.full(N_INST, 1.0e-4)
COMMISSIONS[ALGO] = 2.0e-5
LIMITS = np.full(N_INST, 10_000.0)
LIMITS[ALGO] = 100_000.0

WINDOWS = (
    ("active_year1_151_250", 151, 250),
    ("year2_251_500", 251, 500),
    ("year3_validation_501_750", 501, 750),
    ("year4_exposed_751_1000", 751, 1000),
    ("active_full_151_1000", 151, 1000),
    ("long_history_701_1000", 701, 1000),
    ("long_history_801_1000", 801, 1000),
)


def load_prices() -> np.ndarray:
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy(float).T
    if prices.shape[0] != N_INST:
        raise ValueError(f"Expected {N_INST} instruments, got {prices.shape[0]}")
    if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise ValueError("Prices must be finite and positive")
    return prices


def competition_score(pnl: np.ndarray) -> float:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    if mean <= 0.0 or std < 1.0e-10:
        return mean
    sharpe = sqrt(250.0) * mean / std
    return mean * sharpe * sharpe / (sharpe * sharpe + 1.0)


def build_targets(
    prices: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    strategy = importlib.import_module("random_strat1")
    strategy.currentPos = np.zeros(N_INST, dtype=int)
    targets = np.zeros((prices.shape[1] + 1, N_INST), dtype=int)
    for history_size in range(start - 1, end):
        requested = strategy.getMyPosition(prices[:, :history_size])
        current_prices = prices[:, history_size - 1]
        share_limits = np.floor(LIMITS / current_prices).astype(int)
        targets[history_size] = np.clip(
            requested, -share_limits, share_limits
        ).astype(int)
    return targets


def account(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
    commissions: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Match the official evaluator's delayed commission timing."""
    cash = 0.0
    value = 0.0
    prior_fee = 0.0
    current = np.zeros(N_INST)
    pnl = []
    turnover = 0.0
    fees = 0.0
    for t in range(start - 1, end + 1):
        current_prices = prices[:, t - 1]
        new = targets[t].astype(float) if t < end else current.copy()
        delta = new - current
        cash -= float(current_prices @ delta) + prior_fee
        volumes = current_prices * np.abs(delta)
        prior_fee = float(volumes @ commissions)
        turnover += float(volumes.sum())
        fees += prior_fee
        current = new
        new_value = cash + float(current @ current_prices)
        day_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(day_pnl)
    return np.asarray(pnl), turnover, fees


def pnl_metrics(
    pnl: np.ndarray,
    turnover: float,
    fees: float,
) -> dict[str, float]:
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    cumulative = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(
        np.concatenate(([0.0], cumulative))
    )[1:] - cumulative
    return {
        "score": competition_score(pnl),
        "mean_pnl": mean,
        "total_pnl": float(np.sum(pnl)),
        "std_pnl": std,
        "sharpe": sqrt(250.0) * mean / std if std > 0.0 else 0.0,
        "hit_rate": float(np.mean(pnl > 0.0)),
        "max_drawdown": float(np.max(drawdown)),
        "turnover": float(turnover),
        "fees": float(fees),
    }


def algo_position_metrics(
    prices: np.ndarray,
    targets: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    anchors = np.arange(start - 1, end)
    dollars = (
        targets[anchors, ALGO]
        * prices[ALGO, anchors - 1]
    )
    next_returns = (
        prices[ALGO, anchors] / prices[ALGO, anchors - 1] - 1.0
    )
    active = dollars != 0.0
    signed_returns = np.sign(dollars[active]) * next_returns[active]
    return {
        "mean_abs_dollars": float(np.mean(np.abs(dollars))),
        "cap_utilization": float(
            np.mean(np.abs(dollars)) / LIMITS[ALGO]
        ),
        "long_fraction": float(np.mean(dollars > 0.0)),
        "short_fraction": float(np.mean(dollars < 0.0)),
        "flat_fraction": float(np.mean(dollars == 0.0)),
        "direction_changes": float(
            np.count_nonzero(np.diff(np.sign(dollars)))
        ),
        "active_direction_accuracy": float(
            np.mean(signed_returns > 0.0)
        ) if len(signed_returns) else 0.0,
        "mean_signed_next_return_bps": float(
            10_000.0 * np.mean(signed_returns)
        ) if len(signed_returns) else 0.0,
        "algo_buy_hold_return": float(
            prices[ALGO, end - 1] / prices[ALGO, start - 2] - 1.0
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    summary_rows = []
    daily_rows = []

    for window, start, end in WINDOWS:
        targets = build_targets(prices, start, end)
        algo_targets = np.zeros_like(targets)
        algo_targets[:, ALGO] = targets[:, ALGO]
        rest_targets = targets.copy()
        rest_targets[:, ALGO] = 0
        long_targets = np.zeros_like(targets)
        anchors = np.arange(start - 1, end)
        long_targets[anchors, ALGO] = np.floor(
            LIMITS[ALGO] / prices[ALGO, anchors - 1]
        ).astype(int)

        full_pnl, full_turnover, full_fees = account(
            prices, targets, start, end, COMMISSIONS
        )
        algo_pnl, algo_turnover, algo_fees = account(
            prices, algo_targets, start, end, COMMISSIONS
        )
        rest_pnl, rest_turnover, rest_fees = account(
            prices, rest_targets, start, end, COMMISSIONS
        )
        long_pnl, long_turnover, long_fees = account(
            prices, long_targets, start, end, COMMISSIONS
        )
        if not np.allclose(
            full_pnl,
            algo_pnl + rest_pnl,
            atol=1.0e-7,
            rtol=1.0e-10,
        ):
            raise AssertionError("Instrument attribution is not additive")

        full = pnl_metrics(full_pnl, full_turnover, full_fees)
        algo = pnl_metrics(algo_pnl, algo_turnover, algo_fees)
        rest = pnl_metrics(rest_pnl, rest_turnover, rest_fees)
        always_long = pnl_metrics(long_pnl, long_turnover, long_fees)
        position = algo_position_metrics(prices, targets, start, end)
        correlation = float(np.corrcoef(algo_pnl, rest_pnl)[0, 1])

        for sleeve, values in (
            ("full", full),
            ("algo", algo),
            ("rest_without_algo", rest),
            ("algo_always_long", always_long),
        ):
            summary_rows.append(
                {
                    "window": window,
                    "sleeve": sleeve,
                    **values,
                    **(
                        position
                        if sleeve == "algo"
                        else {
                            key: np.nan for key in position
                        }
                    ),
                    "algo_rest_pnl_correlation": correlation,
                    "full_minus_rest_score": (
                        full["score"] - rest["score"]
                    ),
                    "algo_minus_always_long_pnl": (
                        algo["total_pnl"] - always_long["total_pnl"]
                    ),
                }
            )

        for offset, (full_day, algo_day, rest_day) in enumerate(
            zip(full_pnl, algo_pnl, rest_pnl)
        ):
            day = start + offset
            anchor = day - 1
            daily_rows.append(
                {
                    "window": window,
                    "day": day,
                    "full_pnl": full_day,
                    "algo_pnl": algo_day,
                    "rest_pnl": rest_day,
                    "algo_shares": targets[anchor, ALGO],
                    "algo_dollars": (
                        targets[anchor, ALGO]
                        * prices[ALGO, anchor - 1]
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(daily_rows).to_csv(DAILY_PATH, index=False)

    lines = [
        "# ALGO attribution inside random_strat1",
        "",
        "The position path is produced by the unchanged 51-instrument model. "
        "ALGO is then isolated at execution; the model is not refitted.",
        "",
        "| Window | ALGO PnL | Always-long PnL | Timing PnL | "
        "ALGO score | ALGO Sharpe | "
        "Full score | Score without ALGO | Full - rest | "
        "ALGO/rest corr | Turnover | Fees |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window, _, _ in WINDOWS:
        rows = summary.loc[summary["window"] == window].set_index("sleeve")
        algo = rows.loc["algo"]
        full = rows.loc["full"]
        rest = rows.loc["rest_without_algo"]
        always_long = rows.loc["algo_always_long"]
        lines.append(
            f"| {window} | ${algo['total_pnl']:,.2f} | "
            f"${always_long['total_pnl']:,.2f} | "
            f"${algo['total_pnl'] - always_long['total_pnl']:+,.2f} | "
            f"{algo['score']:.3f} | {algo['sharpe']:.3f} | "
            f"{full['score']:.3f} | {rest['score']:.3f} | "
            f"{full['score'] - rest['score']:+.3f} | "
            f"{algo['algo_rest_pnl_correlation']:+.3f} | "
            f"${algo['turnover']:,.0f} | ${algo['fees']:,.2f} |"
        )

    lines.extend(
        [
            "",
            "## ALGO execution diagnostics",
            "",
            "| Window | Mean abs exposure | Long | Short | Flat | "
            "Direction changes | Next-day accuracy | "
            "Mean signed return | ALGO buy/hold |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for window, _, _ in WINDOWS:
        algo = summary.loc[
            (summary["window"] == window)
            & (summary["sleeve"] == "algo")
        ].iloc[0]
        lines.append(
            f"| {window} | ${algo['mean_abs_dollars']:,.0f} | "
            f"{algo['long_fraction']:.1%} | "
            f"{algo['short_fraction']:.1%} | "
            f"{algo['flat_fraction']:.1%} | "
            f"{int(algo['direction_changes'])} | "
            f"{algo['active_direction_accuracy']:.1%} | "
            f"{algo['mean_signed_next_return_bps']:+.3f} bp | "
            f"{algo['algo_buy_hold_return']:+.1%} |"
        )

    lines.extend(
        [
            "",
            "Because the competition score depends on the mean and volatility "
            "of the combined daily PnL, ALGO's standalone score does not add "
            "arithmetically to the rest-of-book score. `Full - rest` is the "
            "relevant realised portfolio score contribution.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
