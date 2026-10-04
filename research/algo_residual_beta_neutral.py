"""Causal ALGO-residual signals with a non-ALGO beta-neutral hedge basket.

This research harness compares the previously studied ALGO-X asset legs with
the same legs after replacing their ALGO position by a distributed hedge made
from the other tradeable assets.  Instrument 0 is always zero.

The final 751-1000 window is reported only as contaminated diagnostic data.  It
is never used to select the signal universe or any hedge setting.
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
from research import algo12_equal_sleeve_research as sleeves


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_residual_beta_neutral"
ALGO = strategy.I["ALGO"]
BETA_WINDOW = 250
BETA_MIN_OBS = 60
BETA_FLOOR = 0.05
SELECTED = ("ALGO-AETS", "ALGO-ACIX")


def rolling_return_betas(
    prices: np.ndarray,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate causal return betas/correlations through ``history_size``."""
    betas = np.zeros(strategy.N_INST)
    correlations = np.zeros(strategy.N_INST)
    if history_size < BETA_MIN_OBS + 1:
        return betas, correlations

    known = np.diff(
        np.log(np.maximum(prices[:, :history_size], strategy._EPS_DENOM)),
        axis=1,
    )
    known = known[:, -BETA_WINDOW:]
    centered = known - known.mean(axis=1, keepdims=True)
    algo = centered[ALGO]
    algo_ss = float(algo @ algo)
    if algo_ss <= strategy._EPS_VAR:
        return betas, correlations

    betas = centered @ algo / algo_ss
    norms = np.sqrt(np.sum(centered * centered, axis=1) * algo_ss)
    correlations = np.divide(
        centered @ algo,
        norms,
        out=np.zeros(strategy.N_INST),
        where=norms > strategy._EPS_VAR,
    )
    betas[ALGO] = 1.0
    correlations[ALGO] = 1.0
    return betas, correlations


def bounded_minimum_norm_beta_hedge(
    target_beta_dollars: float,
    betas: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find the smallest bounded hedge meeting one beta-dollar constraint.

    The unconstrained minimum-L2 solution has ``h_j = -lambda * beta_j``.
    Clipping that expression to each leg's remaining capacity is monotonic in
    ``lambda``, so bisection gives the exact bounded solution whenever the
    requested exposure is feasible.
    """
    hedge = np.zeros(strategy.N_INST)
    use = (
        np.asarray(eligible, dtype=bool)
        & np.isfinite(betas)
        & (np.abs(betas) >= BETA_FLOOR)
    )
    use[ALGO] = False
    if not np.any(use) or abs(target_beta_dollars) <= 1.0e-9:
        return hedge, 1.0 if abs(target_beta_dollars) <= 1.0e-9 else 0.0

    b = betas[use]
    lo = lower[use]
    hi = upper[use]
    minimum = float(
        b
        @ np.where(
            b >= 0.0,
            lo,
            hi,
        )
    )
    maximum = float(
        b
        @ np.where(
            b >= 0.0,
            hi,
            lo,
        )
    )
    delivered_target = float(
        np.clip(target_beta_dollars, minimum, maximum)
    )
    delivery = (
        delivered_target / target_beta_dollars
        if abs(target_beta_dollars) > 1.0e-9
        else 1.0
    )

    def exposure(multiplier: float) -> float:
        values = np.clip(-multiplier * b, lo, hi)
        return float(b @ values)

    left = -1.0
    right = 1.0
    while exposure(left) < delivered_target:
        left *= 2.0
    while exposure(right) > delivered_target:
        right *= 2.0
    for _ in range(80):
        middle = 0.5 * (left + right)
        if exposure(middle) > delivered_target:
            left = middle
        else:
            right = middle
    multiplier = 0.5 * (left + right)
    hedge[use] = np.clip(-multiplier * b, lo, hi)
    return hedge, float(np.clip(delivery, 0.0, 1.0))


def beta_neutral_overlay(
    base_dollars: np.ndarray,
    requested_alpha: np.ndarray,
    betas: np.ndarray,
    crash: np.ndarray,
    alpha_universe: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Deliver capacity-aware alpha plus a zero-ALGO synthetic beta hedge."""
    limits = strategy.BANG_BANG_LIMITS
    lower = -limits - base_dollars
    upper = limits - base_dollars

    alpha = np.clip(requested_alpha, lower, upper)
    alpha[ALGO] = 0.0
    alpha[crash] = 0.0
    alpha_beta = float(betas @ alpha)

    eligible = np.ones(strategy.N_INST, dtype=bool)
    eligible[ALGO] = False
    eligible[np.asarray(alpha_universe, dtype=int)] = False
    eligible[crash] = False
    hedge, delivery = bounded_minimum_norm_beta_hedge(
        -alpha_beta,
        betas,
        lower,
        upper,
        eligible,
    )
    if delivery < 1.0:
        alpha *= delivery
        alpha_beta = float(betas @ alpha)
        hedge, _ = bounded_minimum_norm_beta_hedge(
            -alpha_beta,
            betas,
            lower,
            upper,
            eligible,
        )

    overlay = alpha + hedge
    overlay[ALGO] = 0.0
    final = np.clip(base_dollars + overlay, -limits, limits)
    final[ALGO] = 0.0
    effective = final - base_dollars
    diagnostics = {
        "requested_alpha_gross": float(np.sum(np.abs(requested_alpha))),
        "delivered_alpha_gross": float(np.sum(np.abs(alpha))),
        "hedge_gross": float(np.sum(np.abs(hedge))),
        "overlay_net_dollars": float(np.sum(effective)),
        "overlay_beta_dollars": float(betas @ effective),
        "alpha_scale": float(delivery),
    }
    return final, diagnostics


def build_candidate_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    individual: dict[str, np.ndarray],
    selected: tuple[str, ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create beta-neutral targets for one fixed residual-signal universe."""
    targets = np.zeros_like(base_targets)
    records: list[dict[str, float]] = []
    alpha_indices = np.asarray(
        [strategy.I[name.split("-", 1)[1]] for name in selected],
        dtype=int,
    )

    for history_size in range(1, prices.shape[1]):
        current = prices[:, history_size - 1]
        if history_size < strategy.WARMUP + 10:
            continue

        base_dollars = base_targets[history_size] * current
        requested_alpha = np.zeros(strategy.N_INST)
        for name in selected:
            pair_dollars = individual[name][history_size] * current
            pair_dollars[ALGO] = 0.0
            requested_alpha += pair_dollars

        last_return = np.diff(
            np.log(
                np.maximum(
                    prices[:, max(0, history_size - 2):history_size],
                    strategy._EPS_DENOM,
                )
            ),
            axis=1,
        )
        crash = np.zeros(strategy.N_INST, dtype=bool)
        if last_return.shape[1]:
            crash = np.abs(last_return[:, -1]) > 0.40
        crash[ALGO] = False

        betas, correlations = rolling_return_betas(prices, history_size)
        final_dollars, diagnostic = beta_neutral_overlay(
            base_dollars,
            requested_alpha,
            betas,
            crash,
            alpha_indices,
        )
        targets[history_size] = sleeves.dollars_to_shares(
            final_dollars,
            current,
        )
        effective = targets[history_size] * current - base_dollars
        diagnostic.update(
            {
                "history_size": float(history_size),
                "rounded_beta_dollars": float(betas @ effective),
                "mean_hedge_correlation": float(
                    np.mean(np.abs(correlations[1:]))
                ),
            }
        )
        records.append(diagnostic)
    return targets, pd.DataFrame(records)


def build_base_without_algo_pairs(prices: np.ndarray) -> np.ndarray:
    """Replay the production book after moving ALGO-X pairs to this overlay."""
    original_pairs = strategy.PAIR_CFGS
    original_use = strategy.USE_ALGO_RESIDUAL_BASKET
    try:
        strategy.USE_ALGO_RESIDUAL_BASKET = False
        strategy.PAIR_CFGS = tuple(
            spec
            for spec in original_pairs
            if ALGO not in (int(spec.y), int(spec.x))
        )
        return sleeves.build_base_targets(prices)
    finally:
        strategy.PAIR_CFGS = original_pairs
        strategy.USE_ALGO_RESIDUAL_BASKET = original_use
        strategy.reset()


def build_original_base(prices: np.ndarray) -> np.ndarray:
    """Replay Four Sigmas with the new residual basket disabled."""
    original_use = strategy.USE_ALGO_RESIDUAL_BASKET
    try:
        strategy.USE_ALGO_RESIDUAL_BASKET = False
        return sleeves.build_base_targets(prices)
    finally:
        strategy.USE_ALGO_RESIDUAL_BASKET = original_use
        strategy.reset()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        sleeves.PRICE_PATH,
        sep=r"\s+",
        header=0,
    ).to_numpy(dtype=float).T
    specs = sleeves.load_specs()
    current_base_targets = build_original_base(prices)
    base_targets = build_base_without_algo_pairs(prices)
    _, _, individual = sleeves.build_sleeve_targets(prices, specs)

    universes = {
        "selected_2": SELECTED,
        "all_12": sleeves.PAIR_NAMES,
    }
    results: dict[str, dict[str, dict[str, float]]] = {
        "current_baseline": sleeves.evaluate(
            prices,
            current_base_targets,
            sleeves.ALL_WINDOWS,
        ),
        "base_without_algo_pairs": sleeves.evaluate(
            prices,
            base_targets,
            sleeves.ALL_WINDOWS,
        ),
    }
    diagnostics: list[pd.DataFrame] = []
    for name, selected in universes.items():
        targets, diagnostic = build_candidate_targets(
            prices,
            base_targets,
            individual,
            selected,
        )
        results[name] = sleeves.evaluate(
            prices,
            targets,
            sleeves.ALL_WINDOWS,
        )
        standalone_targets = targets - base_targets
        results[f"{name}_standalone"] = sleeves.evaluate(
            prices,
            standalone_targets,
            sleeves.ALL_WINDOWS,
        )
        for window, start, end in sleeves.ALL_WINDOWS:
            pnl, _, _ = sleeves.account(
                prices,
                standalone_targets,
                start,
                end,
            )
            results[f"{name}_standalone"][window]["win_rate"] = float(
                np.mean(pnl > 0.0)
            )
        diagnostic.insert(0, "candidate", name)
        diagnostics.append(diagnostic)

    frame = pd.concat(diagnostics, ignore_index=True)
    frame.to_csv(OUTPUT_DIR / "diagnostics.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(
        "candidate window score delta alpha_scale beta_abs hedge_gross net_dollars"
    )
    baseline = results["current_baseline"]
    for name in universes:
        candidate_rows = frame[frame["candidate"] == name]
        for window, start, end in sleeves.ALL_WINDOWS:
            rows = candidate_rows[
                candidate_rows["history_size"].between(start - 1, end - 1)
            ]
            score = results[name][window]["score"]
            standalone = results[f"{name}_standalone"][window]
            print(
                name,
                window,
                f"{score:.4f}",
                f"{score - baseline[window]['score']:+.4f}",
                f"{rows['alpha_scale'].mean():.4f}",
                f"{rows['rounded_beta_dollars'].abs().mean():.4f}",
                f"{rows['hedge_gross'].mean():.2f}",
                f"{rows['overlay_net_dollars'].mean():.2f}",
                f"standalone={standalone['score']:.4f}",
                f"win={standalone['win_rate']:.4f}",
            )


if __name__ == "__main__":
    main()
