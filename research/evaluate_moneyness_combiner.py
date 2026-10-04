#!/usr/bin/env python
"""One-shot evaluation of the fixed, no-tuning moneyness combiner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas_moneyness as moneyness
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "moneyness_combiner"
WINDOWS = (
    ("TRAIN1", 251, 375),
    ("TRAIN2", 376, 500),
    ("VALIDATION1", 501, 625),
    ("VALIDATION2", 626, 750),
    ("YEAR2", 251, 500),
    ("YEAR3", 501, 750),
    ("YEAR4_OOS", 751, 1000),
)


def build_targets(
    prices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    book = moneyness._WholeVectorMoneynessBook()
    dollars = np.zeros((prices.shape[1] + 1, moneyness.core.N_INST))
    pair_money = []
    ridge_money = []
    for history_size in range(1, prices.shape[1]):
        dollars[history_size] = book.dollars(
            prices[:, :history_size]
        )
        pair_money.append(book.last_pair_opportunity)
        ridge_money.append(book.last_ridge_opportunity)
    diagnostics = {
        "mean_pair_money_available": float(np.mean(pair_money)),
        "mean_ridge_money_available": float(np.mean(ridge_money)),
        "mean_active_assets": float(
            np.mean(np.sum(dollars != 0.0, axis=1))
        ),
    }
    return dollars, diagnostics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(study.TEST_END)
    baseline_cache = study.collect_signals(prices)
    moneyness_dollars, diagnostics = build_targets(prices)

    results = {}
    for name, start, end in WINDOWS:
        baseline = study.region_metrics(
            prices,
            baseline_cache.baseline_dollars,
            start,
            end,
        )
        money = study.region_metrics(
            prices,
            moneyness_dollars,
            start,
            end,
        )
        results[name] = {
            "baseline": baseline,
            "moneyness": money,
            "score_delta": money["score"] - baseline["score"],
        }

    payload = {
        "selection": (
            "None. Formula fixed before any train, validation, or OOS "
            "result was calculated."
        ),
        "new_tunable_parameters": 0,
        "diagnostics": diagnostics,
        "results": results,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        f"{'window':<14} {'baseline':>11} "
        f"{'moneyness':>11} {'delta':>11}"
    )
    for name, _, _ in WINDOWS:
        row = results[name]
        print(
            f"{name:<14} "
            f"{row['baseline']['score']:11.3f} "
            f"{row['moneyness']['score']:11.3f} "
            f"{row['score_delta']:+11.3f}"
        )
    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.3f}")
    print(f"\nWrote {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
