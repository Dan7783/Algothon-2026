"""Runtime-safety audit for the current ``four_sigmas`` submission.

This is diagnostic only. It replays every historical prefix, checks output
contracts and determinism, and probes malformed histories that are outside the
official positive-finite price contract.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "runtime_safety"


def call_with_warnings(
    prices: np.ndarray,
) -> tuple[np.ndarray | None, list[str], str | None]:
    caught: list[str] = []
    output: np.ndarray | None = None
    error: str | None = None
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            output = strategy.getMyPosition(prices)
        except Exception as exc:  # Audit must record, rather than hide, errors.
            error = f"{type(exc).__name__}: {exc}"
        caught = [
            f"{record.category.__name__}: {record.message}"
            for record in records
        ]
    return output, caught, error


def output_contract(
    output: np.ndarray | None,
    latest_prices: np.ndarray,
) -> list[str]:
    failures: list[str] = []
    if output is None:
        return ["no_output"]
    if output.shape != (strategy.N_INST,):
        failures.append(f"shape={output.shape}")
        return failures
    if not np.issubdtype(output.dtype, np.integer):
        failures.append(f"dtype={output.dtype}")
    if not np.all(np.isfinite(output)):
        failures.append("nonfinite_output")
    if (
        not np.all(np.isfinite(latest_prices))
        or np.any(latest_prices <= 0.0)
    ):
        # Production intentionally returns a zero vector when either of the
        # latest two columns is invalid; a share-limit check is undefined for
        # that malformed mark.
        if np.any(output != 0):
            failures.append("nonzero_output_on_invalid_latest_prices")
        return failures
    safe = np.maximum(latest_prices, strategy._EPS_DENOM)
    share_limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    # Convert before abs: abs(int64.min) overflows in integer arithmetic and
    # would conceal the sentinel produced by casting NaN to int.
    output_magnitude = np.abs(output.astype(float))
    if np.any(output_magnitude > share_limits):
        failures.append("position_limit_breach")
    if output[strategy.I["ALGO"]] != 0:
        failures.append("direct_algo_position")
    return failures


def sequential_replay(prices: np.ndarray) -> dict[str, Any]:
    strategy.reset()
    outputs = np.zeros(
        (prices.shape[1] + 1, strategy.N_INST),
        dtype=np.int64,
    )
    exceptions: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    repeat_mismatches: list[int] = []

    for history_size in range(1, prices.shape[1] + 1):
        known = prices[:, :history_size]
        output, caught, error = call_with_warnings(known)
        if error is not None:
            exceptions.append({"history_size": history_size, "error": error})
            continue
        assert output is not None
        outputs[history_size] = output
        if caught:
            warning_rows.append(
                {"history_size": history_size, "warnings": caught}
            )
        failures = output_contract(output, known[:, -1])
        if failures:
            contract_rows.append(
                {"history_size": history_size, "failures": failures}
            )
        if history_size in {40, 100, 251, 500, 750, 1000}:
            repeated, repeat_warnings, repeat_error = call_with_warnings(known)
            if (
                repeat_error is not None
                or repeat_warnings
                or repeated is None
                or not np.array_equal(repeated, output)
            ):
                repeat_mismatches.append(history_size)
    return {
        "outputs": outputs,
        "exceptions": exceptions,
        "warnings": warning_rows,
        "contract_failures": contract_rows,
        "repeat_mismatches": repeat_mismatches,
    }


def isolated_output(prices: np.ndarray) -> tuple[np.ndarray | None, list[str], str | None]:
    strategy.reset()
    return call_with_warnings(prices)


def stress_cases(prices: np.ndarray) -> list[dict[str, Any]]:
    cases: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("constant_positive", lambda: np.full_like(prices[:, :300], 100.0)),
        ("tiny_positive", lambda: np.full_like(prices[:, :300], 1.0e-300)),
        ("huge_positive", lambda: np.full_like(prices[:, :300], 1.0e300)),
        (
            "last_nan",
            lambda: replace(prices[:, :300], (1, -1), np.nan),
        ),
        (
            "last_zero",
            lambda: replace(prices[:, :300], (1, -1), 0.0),
        ),
        (
            "last_negative",
            lambda: replace(prices[:, :300], (1, -1), -1.0),
        ),
        (
            "interior_nan",
            lambda: replace(prices[:, :300], (1, 100), np.nan),
        ),
        (
            "interior_zero",
            lambda: replace(prices[:, :300], (1, 100), 0.0),
        ),
        (
            "interior_negative",
            lambda: replace(prices[:, :300], (1, 100), -1.0),
        ),
        (
            "interior_positive_inf",
            lambda: replace(prices[:, :300], (1, 100), np.inf),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, builder in cases:
        sample = builder()
        output, caught, error = isolated_output(sample)
        failures = (
            output_contract(output, sample[:, -1])
            if output is not None
            else ["no_output"]
        )
        rows.append(
            {
                "case": name,
                "error": error,
                "warning_count": len(caught),
                "warnings": caught,
                "contract_failures": failures,
                "max_abs_position": (
                    float(np.max(np.abs(output.astype(float))))
                    if output is not None
                    else None
                ),
                "all_zero": (
                    bool(np.all(output == 0))
                    if output is not None
                    else None
                ),
            }
        )
    return rows


def replace(
    source: np.ndarray,
    index: tuple[int, int],
    value: float,
) -> np.ndarray:
    result = source.copy()
    result[index] = value
    return result


def state_tests(
    prices: np.ndarray,
    sequential: np.ndarray,
) -> dict[str, Any]:
    jump, jump_warnings, jump_error = isolated_output(prices)
    fresh_500, fresh_500_warnings, fresh_500_error = isolated_output(
        prices[:, :500]
    )

    strategy.reset()
    full_before, _, full_before_error = call_with_warnings(prices)
    rewind, rewind_warnings, rewind_error = call_with_warnings(
        prices[:, :500]
    )
    regrown, regrown_warnings, regrown_error = call_with_warnings(prices)

    return {
        "jump_to_1000_error": jump_error,
        "jump_to_1000_warning_count": len(jump_warnings),
        "jump_matches_sequential": (
            bool(np.array_equal(jump, sequential[1000]))
            if jump is not None
            else False
        ),
        "fresh_500_error": fresh_500_error,
        "fresh_500_warning_count": len(fresh_500_warnings),
        "rewind_setup_error": full_before_error,
        "rewind_error": rewind_error,
        "rewind_warning_count": len(rewind_warnings),
        "rewind_matches_fresh_500": (
            bool(np.array_equal(rewind, fresh_500))
            if rewind is not None and fresh_500 is not None
            else False
        ),
        "regrow_error": regrown_error,
        "regrow_warning_count": len(regrown_warnings),
        "regrow_matches_sequential": (
            bool(np.array_equal(regrown, sequential[1000]))
            if regrown is not None
            else False
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    replay = sequential_replay(prices)
    state = state_tests(prices, replay["outputs"])
    stress = stress_cases(prices)
    result = {
        "historical_days": int(prices.shape[1]),
        "sequential_exception_count": len(replay["exceptions"]),
        "sequential_warning_day_count": len(replay["warnings"]),
        "sequential_contract_failure_count": len(
            replay["contract_failures"]
        ),
        "repeat_mismatches": replay["repeat_mismatches"],
        "state_tests": state,
        "stress_cases": stress,
        "sequential_exceptions": replay["exceptions"],
        "sequential_warnings": replay["warnings"],
        "sequential_contract_failures": replay["contract_failures"],
    }
    path = OUTPUT_DIR / "runtime_safety_report.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
