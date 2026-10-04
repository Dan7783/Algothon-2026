#!/usr/bin/env python
"""One-shot evaluation of equal-family per-asset moneyness forecasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas_asset_moneyness as asset_money
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "asset_moneyness"
WINDOWS = (
    ("TRAIN1", 251, 375),
    ("TRAIN2", 376, 500),
    ("VALIDATION1", 501, 625),
    ("VALIDATION2", 626, 750),
    ("OOS1", 751, 875),
    ("OOS2", 876, 1000),
    ("YEAR2", 251, 500),
    ("YEAR3", 501, 750),
    ("YEAR4_OOS", 751, 1000),
)


def build_targets(
    prices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    book = asset_money._AssetMoneynessBook()
    dollars = np.zeros(
        (prices.shape[1] + 1, asset_money.base.core.N_INST)
    )
    pair_magnitude = []
    ridge_magnitude = []
    contributors = []
    active_assets = []
    for history_size in range(1, prices.shape[1]):
        dollars[history_size] = book.dollars(
            prices[:, :history_size]
        )
        pair_magnitude.append(book.last_pair_abs_forecast)
        ridge_magnitude.append(book.last_ridge_abs_forecast)
        contributors.append(book.last_mean_pair_contributors)
        active_assets.append(
            int(np.sum(dollars[history_size] != 0.0))
        )
    return dollars, {
        "mean_pair_abs_return_forecast": float(
            np.mean(pair_magnitude)
        ),
        "mean_ridge_abs_return_forecast": float(
            np.mean(ridge_magnitude)
        ),
        "mean_pair_contributors_per_asset": float(
            np.mean(contributors)
        ),
        "mean_active_assets": float(np.mean(active_assets)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = study.load_prices(study.TEST_END)
    baseline_cache = study.collect_signals(prices)
    dollars, diagnostics = build_targets(prices)
    results = {}
    for name, start, end in WINDOWS:
        baseline = study.region_metrics(
            prices,
            baseline_cache.baseline_dollars,
            start,
            end,
        )
        candidate = study.region_metrics(
            prices,
            dollars,
            start,
            end,
        )
        results[name] = {
            "baseline": baseline,
            "asset_moneyness": candidate,
            "score_delta": candidate["score"] - baseline["score"],
        }

    payload = {
        "selection": "None; formula fixed before evaluation.",
        "new_tunable_parameters": 0,
        "oos_status": "Retrospective; 751-1000 was already exposed.",
        "diagnostics": diagnostics,
        "results": results,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        f"{'window':<14} {'baseline':>11} "
        f"{'asset money':>11} {'delta':>11}"
    )
    for name, _, _ in WINDOWS:
        row = results[name]
        print(
            f"{name:<14} "
            f"{row['baseline']['score']:11.3f} "
            f"{row['asset_moneyness']['score']:11.3f} "
            f"{row['score_delta']:+11.3f}"
        )
    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nWrote {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
