"""Joint constrained allocation of Four Sigmas and ALGO residual signals.

The allocator solves the signal-implied mean/variance problem

    min_w (w - d)' Q (w - d)

under per-asset bounds and one beta constraint. This is equivalent to

    max_w mu'w - lambda w'Qw

when ``mu = 2 lambda Q d``; lambda therefore cancels and is not tuned.

Only days 1-750 are loaded. Days 751-1000 are never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as sleeves
from research import algo_residual_projection_research as residual


OUTPUT_DIR = ROOT / "analysis_outputs" / "algo_joint_allocator"
LAST_DEVELOPMENT_DAY = 750
ALGO = strategy.I["ALGO"]
WINDOWS = residual.WINDOWS
VARIANTS = (
    "identity_full_neutral",
    "diag_full_neutral",
    "fullcov_full_neutral",
    "identity_overlay_neutral",
    "diag_overlay_neutral",
    "fullcov_overlay_neutral",
)


def diagonal_projection(
    desired: np.ndarray,
    metric_diag: np.ndarray,
    beta: np.ndarray,
    target: float,
) -> np.ndarray:
    """Exact box-constrained diagonal-metric projection."""
    q = np.maximum(metric_diag, 1.0e-12)

    def candidate(multiplier: float) -> np.ndarray:
        return np.clip(
            desired - multiplier * beta / q,
            -1.0,
            1.0,
        )

    minimum = float(beta @ np.where(beta >= 0.0, -1.0, 1.0))
    maximum = float(beta @ np.where(beta >= 0.0, 1.0, -1.0))
    target = float(np.clip(target, minimum, maximum))
    left = -1.0
    right = 1.0
    while float(beta @ candidate(left)) < target:
        left *= 2.0
    while float(beta @ candidate(right)) > target:
        right *= 2.0
    for _ in range(80):
        middle = 0.5 * (left + right)
        if float(beta @ candidate(middle)) > target:
            left = middle
        else:
            right = middle
    return candidate(0.5 * (left + right))


def covariance_projection(
    desired: np.ndarray,
    covariance: np.ndarray,
    beta: np.ndarray,
    target: float,
) -> np.ndarray:
    """Active-set projection for a full quadratic metric and one equality."""
    n = len(desired)
    result = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    fixed_values = np.zeros(n)
    covariance = 0.5 * (covariance + covariance.T)
    positive_diag = np.diag(covariance)
    ridge = max(float(np.median(positive_diag)) * 1.0e-5, 1.0e-12)
    covariance = covariance + ridge * np.eye(n)
    minimum = float(beta @ np.where(beta >= 0.0, -1.0, 1.0))
    maximum = float(beta @ np.where(beta >= 0.0, 1.0, -1.0))
    target = float(np.clip(target, minimum, maximum))

    for _ in range(n + 1):
        free = ~fixed
        if not np.any(free):
            result[:] = fixed_values
            return result
        free_indices = np.flatnonzero(free)
        fixed_indices = np.flatnonzero(fixed)
        q_ff = covariance[np.ix_(free_indices, free_indices)]
        b_free = beta[free]
        base_free = desired[free].copy()
        if len(fixed_indices):
            cross = covariance[np.ix_(free_indices, fixed_indices)]
            displacement = (
                fixed_values[fixed] - desired[fixed]
            )
            base_free -= np.linalg.solve(q_ff, cross @ displacement)
        solve_beta = np.linalg.solve(q_ff, b_free)
        denominator = float(b_free @ solve_beta)
        remaining_target = target - float(
            beta[fixed] @ fixed_values[fixed]
        )
        if denominator > 1.0e-14:
            multiplier = (
                float(b_free @ base_free) - remaining_target
            ) / denominator
            free_solution = base_free - multiplier * solve_beta
        else:
            free_solution = np.clip(base_free, -1.0, 1.0)

        violation = np.maximum(
            free_solution - 1.0,
            -1.0 - free_solution,
        )
        if float(np.max(violation)) <= 1.0e-9:
            result[fixed] = fixed_values[fixed]
            result[free] = free_solution
            return np.clip(result, -1.0, 1.0)
        local = int(np.argmax(violation))
        global_index = int(free_indices[local])
        fixed[global_index] = True
        fixed_values[global_index] = (
            1.0 if free_solution[local] > 1.0 else -1.0
        )

    return np.clip(result, -1.0, 1.0)


def trailing_covariance(
    prices: np.ndarray,
    history_size: int,
) -> np.ndarray:
    start = max(
        0,
        history_size - strategy.ALGO_BETA_WINDOW - 1,
    )
    returns = np.diff(
        np.log(
            np.maximum(
                prices[:, start:history_size],
                strategy._EPS_DENOM,
            )
        ),
        axis=1,
    )[1:]
    centered = returns - returns.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / max(centered.shape[1], 1)
    return covariance


def build_joint_targets(
    prices: np.ndarray,
    base_targets: np.ndarray,
    raw_alpha: np.ndarray,
    betas: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    targets = {
        name: np.zeros_like(base_targets) for name in VARIANTS
    }
    records: list[dict[str, float | int | str]] = []
    for history_size in range(
        strategy.ALGO_BETA_MIN_OBS + 1,
        prices.shape[1],
    ):
        current = prices[:, history_size - 1]
        base_dollars = base_targets[history_size] * current
        desired_dollars = base_dollars + raw_alpha[history_size]
        desired = desired_dollars[1:] / strategy.LEG
        base = base_dollars[1:] / strategy.LEG
        beta = betas[history_size, 1:]
        covariance = trailing_covariance(prices, history_size)
        variance = np.maximum(np.diag(covariance), 1.0e-12)
        metrics = {
            "identity": np.ones_like(variance),
            "diag": variance / float(np.median(variance)),
        }

        for metric_name in ("identity", "diag", "fullcov"):
            for constraint_name in ("full_neutral", "overlay_neutral"):
                name = f"{metric_name}_{constraint_name}"
                target_beta = (
                    0.0
                    if constraint_name == "full_neutral"
                    else float(beta @ base)
                )
                if metric_name == "fullcov":
                    solution = covariance_projection(
                        desired,
                        covariance,
                        beta,
                        target_beta,
                    )
                else:
                    solution = diagonal_projection(
                        desired,
                        metrics[metric_name],
                        beta,
                        target_beta,
                    )
                dollars = np.zeros(strategy.N_INST)
                dollars[1:] = strategy.LEG * solution
                targets[name][history_size] = sleeves.dollars_to_shares(
                    dollars,
                    current,
                )
                delivered = (
                    targets[name][history_size] * current
                )[1:] / strategy.LEG
                records.append(
                    {
                        "history_size": history_size,
                        "pnl_day": history_size + 1,
                        "window": residual.window_name(history_size + 1),
                        "variant": name,
                        "target_beta": target_beta,
                        "delivered_beta": float(beta @ delivered),
                        "beta_error": float(
                            beta @ delivered - target_beta
                        ),
                        "gross": float(
                            np.sum(np.abs(delivered)) * strategy.LEG
                        ),
                        "distance_to_desired": float(
                            np.linalg.norm(delivered - desired)
                            * strategy.LEG
                        ),
                        "changed_from_base": float(
                            np.sum(np.abs(delivered - base))
                            * strategy.LEG
                        ),
                    }
                )
    return targets, pd.DataFrame(records)


def evaluate(
    prices: np.ndarray,
    base_targets: np.ndarray,
    targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    baseline = sleeves.evaluate(prices, base_targets, WINDOWS)
    for window, start, end in WINDOWS:
        rows.append(
            {
                "variant": "base",
                "window": window,
                **baseline[window],
                "score_delta": 0.0,
            }
        )
        for name in VARIANTS:
            pnl, turnover, fees = sleeves.account(
                prices,
                targets[name],
                start,
                end,
            )
            metrics = sleeves.metrics(pnl, turnover, fees)
            rows.append(
                {
                    "variant": name,
                    "window": window,
                    **metrics,
                    "score_delta": (
                        metrics["score"] - baseline[window]["score"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# Joint constrained ALGO residual allocator",
        "",
        "Only days 1-750 are loaded. Days 751-1000 are excluded.",
        "",
        "The objective is the signal-implied mean/variance problem "
        "`min (w-d)'Q(w-d)` with exact box constraints.",
        "",
        "| Variant | Train delta | Validation delta | Train A | Train B | "
        "Validation A | Validation B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        values = metrics[metrics["variant"] == name].set_index("window")
        lines.append(
            f"| {name} | {values.loc['TRAIN', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION', 'score_delta']:+.2f} | "
            f"{values.loc['TRAIN_A', 'score_delta']:+.2f} | "
            f"{values.loc['TRAIN_B', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION_A', 'score_delta']:+.2f} | "
            f"{values.loc['VALIDATION_B', 'score_delta']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
        ]
    )
    active = diagnostics[
        diagnostics["window"].isin(
            residual.PRIMARY_WINDOWS
        )
    ]
    for name in VARIANTS:
        group = active[active["variant"] == name]
        lines.append(
            f"- `{name}`: mean absolute post-round beta error "
            f"{group['beta_error'].abs().mean():.4f}; mean position change "
            f"${group['changed_from_base'].mean():,.0f}."
        )

    lines.extend(
        [
            "",
            "## Acceptance rule",
            "",
            "A production candidate must improve aggregate train and validation "
            "and must not have a materially negative chronological half. "
            "No result from days after 750 is available to this script.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_plot(metrics: pd.DataFrame) -> None:
    aggregate = metrics[
        (metrics["variant"] != "base")
        & metrics["window"].isin(("TRAIN", "VALIDATION"))
    ]
    pivot = aggregate.pivot(
        index="variant",
        columns="window",
        values="score_delta",
    ).reindex(VARIANTS)
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, pivot["TRAIN"], width=0.4, label="train")
    ax.bar(
        x + 0.2,
        pivot["VALIDATION"],
        width=0.4,
        label="validation",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=20, ha="right")
    ax.set_ylabel("score change versus base")
    ax.set_title("Joint constrained allocator: days 1-750 only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "joint_allocator.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(
        sleeves.PRICE_PATH,
        sep=r"\s+",
        header=0,
    ).to_numpy(dtype=float).T[:, :LAST_DEVELOPMENT_DAY]
    raw_alpha, betas, _, _, _ = residual.replay_signals(prices)
    base_targets = sleeves.build_base_targets(prices)
    targets, diagnostics = build_joint_targets(
        prices,
        base_targets,
        raw_alpha,
        betas,
    )
    metrics = evaluate(prices, base_targets, targets)
    diagnostics.to_csv(
        OUTPUT_DIR / "allocator_diagnostics.csv",
        index=False,
    )
    metrics.to_csv(
        OUTPUT_DIR / "allocator_metrics.csv",
        index=False,
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "last_loaded_day": LAST_DEVELOPMENT_DAY,
                "variants": VARIANTS,
                "beta_window": strategy.ALGO_BETA_WINDOW,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(metrics, diagnostics)
    make_plot(metrics)
    print((OUTPUT_DIR / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
