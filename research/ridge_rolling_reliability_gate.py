"""Research a causal rolling/expanding ridge reliability gate.

Candidate selection uses only days 251-750:

* development: 251-500, split into two 125-day halves;
* validation: 501-750, split into two 125-day halves.

The already-exposed days 751-1000 are evaluated only after a policy is frozen.
No production source is modified by this script.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import algo12_equal_sleeve_research as evaluator


PRICE_PATH = ROOT / "prices.txt"
OUTPUT_DIR = ROOT / "analysis_outputs" / "ridge_rolling_reliability"
DEVELOPMENT_WINDOWS = (
    ("TRAIN_A", 251, 375),
    ("TRAIN_B", 376, 500),
    ("TRAIN", 251, 500),
    ("VALIDATION_A", 501, 625),
    ("VALIDATION_B", 626, 750),
    ("VALIDATION", 501, 750),
)
EXPOSED_WINDOWS = (
    ("EXPOSED_A", 751, 875),
    ("EXPOSED_B", 876, 1000),
    ("EXPOSED", 751, 1000),
)
BLOCK_WINDOWS = tuple(
    (
        f"BLOCK_{start}_{start + 49}",
        start,
        start + 49,
    )
    for start in range(251, 1000, 50)
)


@dataclass(frozen=True)
class GateSpec:
    name: str
    lookbacks: tuple[int, ...]
    require_active: bool
    scope: str


CANDIDATES = (
    GateSpec("roll125_sign_risky", (125,), False, "risky"),
    GateSpec("roll250_sign_risky", (250,), False, "risky"),
    GateSpec("roll125_250_sign_risky", (125, 250), False, "risky"),
    GateSpec("roll125_active_risky", (125,), True, "risky"),
    GateSpec("roll250_active_risky", (250,), True, "risky"),
    GateSpec("roll125_250_active_risky", (125, 250), True, "risky"),
    GateSpec("roll125_sign_all", (125,), False, "all"),
    GateSpec("roll250_sign_all", (250,), False, "all"),
    GateSpec("roll125_250_sign_all", (125, 250), False, "all"),
)
BASELINE = "current_production"


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def rolling_details(
    returns: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    original = strategy.RIDGE_LOOKBACK
    strategy.RIDGE_LOOKBACK = lookback
    try:
        return strategy._ridge_forecast_details(returns)
    finally:
        strategy.RIDGE_LOOKBACK = original


def dollars_to_shares(
    dollars: np.ndarray,
    current_prices: np.ndarray,
) -> np.ndarray:
    safe = np.maximum(current_prices, strategy._EPS_DENOM)
    limits = np.floor(strategy.BANG_BANG_LIMITS / safe).astype(int)
    return np.clip(
        np.round(dollars / safe),
        -limits,
        limits,
    ).astype(int)


def reliability_masks(
    main_sign: np.ndarray,
    main_latent_sign: np.ndarray,
    details: tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, bool],
        ...,
    ],
    require_active: bool,
) -> tuple[np.ndarray, np.ndarray]:
    active_ok = np.ones(strategy.N_INST, dtype=bool)
    latent_ok = np.ones(strategy.N_INST, dtype=bool)
    for raw, _, latent, gate_pass in details:
        raw_sign = np.sign(raw)
        if require_active:
            active_ok &= (
                gate_pass
                & (raw_sign != 0.0)
                & (raw_sign == main_sign)
            )
            latent_ok &= (
                gate_pass
                & (raw_sign != 0.0)
                & (raw_sign == main_latent_sign)
            )
        else:
            active_ok &= (
                gate_pass
                & (latent != 0.0)
                & (latent == main_sign)
            )
            latent_ok &= (
                gate_pass
                & (latent != 0.0)
                & (latent == main_latent_sign)
            )
    return active_ok, latent_ok


def gated_core_dollars(
    sleeves: dict[str, object],
    rolling: tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, bool],
        ...,
    ],
    spec: GateSpec,
) -> tuple[np.ndarray, dict[str, int]]:
    pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
    pair_active = np.asarray(
        sleeves["pair_active_mask"],
        dtype=bool,
    )
    pair_sign = np.sign(pair_vote)
    pair_nonzero = pair_active & (pair_sign != 0.0)
    deadlock = pair_active & ~pair_nonzero

    main_raw = np.asarray(sleeves["ridge_raw"], dtype=float)
    main_sign = np.sign(main_raw)
    main_active = np.asarray(
        sleeves["ridge_active_mask"],
        dtype=bool,
    )
    main_latent = np.asarray(
        sleeves["ridge_latent_sign"],
        dtype=float,
    )
    main_strength = np.asarray(
        sleeves["ridge_strength"],
        dtype=float,
    )
    main_gate = bool(sleeves["ridge_gate_pass"])
    confirmed = np.asarray(
        sleeves["ridge_conflict_confirmed_mask"],
        dtype=bool,
    )
    crash = np.asarray(sleeves["crash_mask"], dtype=bool)
    active_ok, latent_ok = reliability_masks(
        main_sign,
        main_latent,
        rolling,
        spec.require_active,
    )

    result = np.zeros(strategy.N_INST)
    result[pair_nonzero] = (
        strategy.BANG_BANG_LIMITS[pair_nonzero]
        * pair_sign[pair_nonzero]
    )

    free_active = ~pair_active & main_active
    deadlock_active = deadlock & main_active
    allowed_free = free_active & active_ok
    result[allowed_free] = (
        strategy.BANG_BANG_LIMITS[allowed_free]
        * main_sign[allowed_free]
    )
    if spec.scope == "all":
        allowed_deadlock = deadlock_active & active_ok
    else:
        allowed_deadlock = deadlock_active
    result[allowed_deadlock] = (
        strategy.BANG_BANG_LIMITS[allowed_deadlock]
        * main_sign[allowed_deadlock]
    )

    # Pair/ridge agreement is already represented by the pair position.
    # A conflict override is the second state where stale ridge can displace a
    # valid pair direction, so it always respects the reliability gate.
    allowed_override = confirmed & active_ok
    result[allowed_override] = (
        strategy.BANG_BANG_LIMITS[allowed_override]
        * main_sign[allowed_override]
    )

    subthreshold = (
        main_gate
        & deadlock
        & ~main_active
        & (main_strength > 0.0)
        & (main_strength <= strategy.RIDGE_THETA)
        & (main_latent != 0.0)
    )
    if spec.scope == "all":
        subthreshold &= latent_ok
    result[subthreshold] = (
        strategy.BANG_BANG_LIMITS[subthreshold]
        * main_latent[subthreshold]
    )
    result[crash] = 0.0
    result[strategy.I["ALGO"]] = 0.0
    diagnostics = {
        "potential_free": int(free_active.sum()),
        "allowed_free": int(allowed_free.sum()),
        "potential_overrides": int(confirmed.sum()),
        "allowed_overrides": int(allowed_override.sum()),
        "potential_active_deadlocks": int(deadlock_active.sum()),
        "allowed_active_deadlocks": int(allowed_deadlock.sum()),
        "potential_subthreshold_deadlocks": int(subthreshold.sum()),
    }
    return result, diagnostics


def apply_algo_overlay(
    core: np.ndarray,
    sleeves: dict[str, object],
) -> np.ndarray:
    if (
        not strategy.USE_ALGO_RESIDUAL_BASKET
        or not bool(sleeves["algo_beta_ready"])
    ):
        return core
    crash = np.asarray(sleeves["crash_mask"], dtype=bool)
    result, _, _, _ = strategy._algo_beta_neutral_overlay(
        core,
        np.asarray(
            sleeves["algo_residual_requested"],
            dtype=float,
        ),
        np.asarray(sleeves["algo_return_betas"], dtype=float),
        crash,
    )
    return result


def build_targets(
    prices: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    names = (BASELINE,) + tuple(spec.name for spec in CANDIDATES)
    targets = {
        name: np.zeros(
            (prices.shape[1] + 1, strategy.N_INST),
            dtype=int,
        )
        for name in names
    }
    rows: list[dict[str, float | int | str]] = []
    lookbacks = sorted(
        {
            lookback
            for spec in CANDIDATES
            for lookback in spec.lookbacks
        }
    )

    strategy.reset()
    for history_size in range(1, prices.shape[1]):
        known = prices[:, :history_size]
        if history_size < strategy.WARMUP + 10:
            continue
        sleeves = strategy.sleeve_breakdown(known)
        if sleeves is None:
            continue
        targets[BASELINE][history_size] = strategy.getMyPosition(known)
        current_prices = known[:, -1]
        returns = np.diff(np.log(known), axis=1)
        rolling_by_lookback = {
            lookback: rolling_details(returns, lookback)
            for lookback in lookbacks
        }
        for spec in CANDIDATES:
            rolling = tuple(
                rolling_by_lookback[lookback]
                for lookback in spec.lookbacks
            )
            core, diagnostics = gated_core_dollars(
                sleeves,
                rolling,
                spec,
            )
            dollars = apply_algo_overlay(core, sleeves)
            targets[spec.name][history_size] = dollars_to_shares(
                dollars,
                current_prices,
            )
            rows.append(
                {
                    "variant": spec.name,
                    "history_size": history_size,
                    "pnl_day": history_size + 1,
                    **diagnostics,
                }
            )
    strategy.reset()
    return targets, pd.DataFrame(rows)


def evaluate(
    prices: np.ndarray,
    targets: dict[str, np.ndarray],
    windows: tuple[tuple[str, int, int], ...],
    variants: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    names = variants or tuple(targets)
    rows: list[dict[str, float | str]] = []
    for variant in names:
        results = evaluator.evaluate(
            prices,
            targets[variant],
            windows,
        )
        for window, metrics in results.items():
            rows.append(
                {
                    "variant": variant,
                    "window": window,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_policy(
    development: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    baseline = (
        development[development["variant"] == BASELINE]
        .set_index("window")["score"]
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for spec in CANDIDATES:
        values = (
            development[development["variant"] == spec.name]
            .set_index("window")["score"]
        )
        deltas = values - baseline
        halves = deltas[
            ["TRAIN_A", "TRAIN_B", "VALIDATION_A", "VALIDATION_B"]
        ]
        rows.append(
            {
                "variant": spec.name,
                "train_delta": float(deltas["TRAIN"]),
                "validation_delta": float(deltas["VALIDATION"]),
                "train_a_delta": float(deltas["TRAIN_A"]),
                "train_b_delta": float(deltas["TRAIN_B"]),
                "validation_a_delta": float(
                    deltas["VALIDATION_A"]
                ),
                "validation_b_delta": float(
                    deltas["VALIDATION_B"]
                ),
                "positive_halves": int((halves > 0.0).sum()),
                "worst_half_delta": float(halves.min()),
                "passes": bool(
                    deltas["TRAIN"] > 0.0
                    and deltas["VALIDATION"] > 0.0
                    and (halves >= 0.0).all()
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking[ranking["passes"]]
    if eligible.empty:
        return BASELINE, ranking
    selected = (
        eligible.sort_values(
            [
                "worst_half_delta",
                "validation_delta",
                "train_delta",
            ],
            ascending=False,
        )
        .iloc[0]["variant"]
    )
    return str(selected), ranking


def gate_summary(
    diagnostics: pd.DataFrame,
    variant: str,
    windows: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    frame = diagnostics[diagnostics["variant"] == variant]
    rows = []
    for window, start, end in windows:
        use = frame[frame["pnl_day"].between(start, end)]
        row: dict[str, float | str] = {"window": window}
        for state in ("free", "overrides", "active_deadlocks"):
            potential = float(use[f"potential_{state}"].sum())
            allowed = float(use[f"allowed_{state}"].sum())
            row[f"{state}_allowed_fraction"] = (
                allowed / potential if potential > 0.0 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    development: pd.DataFrame,
    ranking: pd.DataFrame,
    exposed: pd.DataFrame,
    blocks: pd.DataFrame,
    gates: pd.DataFrame,
    selected: str,
) -> None:
    comparison = pd.concat(
        (
            development[
                development["variant"].isin((BASELINE, selected))
                & development["window"].isin(("TRAIN", "VALIDATION"))
            ],
            exposed[
                exposed["variant"].isin((BASELINE, selected))
                & (exposed["window"] == "EXPOSED")
            ],
        ),
        ignore_index=True,
    )[
        [
            "variant",
            "window",
            "score",
            "mean_pnl",
            "std_pnl",
            "total_pnl",
            "turnover",
        ]
    ]
    gate_section = (
        markdown_table(gates, 3)
        if not gates.empty
        else (
            "No gate passed the predeclared development rule, so no "
            "candidate gate coverage is promoted."
        )
    )
    lines = [
        "# Causal rolling ridge reliability gate",
        "",
        "Selection uses days 251-750 only. Days 751-1000 are reported",
        "after the policy is frozen.",
        "",
        f"Selected policy: **{selected}**.",
        "",
        "## Development ranking",
        "",
        markdown_table(ranking, 3),
        "",
        "## Frozen comparison",
        "",
        markdown_table(comparison, 2),
        "",
        "## Gate coverage",
        "",
        gate_section,
        "",
        "## Selected 50-day blocks",
        "",
        markdown_table(
            blocks[
                blocks["variant"] == selected
            ][
                [
                    "window",
                    "score",
                    "mean_pnl",
                    "std_pnl",
                    "total_pnl",
                ]
            ],
            2,
        ),
        "",
        (
            "Production should remain unchanged when current_production is "
            "selected."
        ),
    ]
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy().T
    targets, diagnostics = build_targets(prices)
    development = evaluate(
        prices,
        targets,
        DEVELOPMENT_WINDOWS,
    )
    selected, ranking = select_policy(development)
    exposed = evaluate(
        prices,
        targets,
        EXPOSED_WINDOWS,
        tuple(dict.fromkeys((BASELINE, selected))),
    )
    blocks = evaluate(
        prices,
        targets,
        BLOCK_WINDOWS,
        (selected,),
    )
    gates = (
        gate_summary(
            diagnostics,
            selected,
            DEVELOPMENT_WINDOWS + EXPOSED_WINDOWS,
        )
        if selected != BASELINE
        else pd.DataFrame()
    )

    development.to_csv(
        OUTPUT_DIR / "development_performance.csv",
        index=False,
    )
    ranking.to_csv(OUTPUT_DIR / "policy_ranking.csv", index=False)
    exposed.to_csv(
        OUTPUT_DIR / "exposed_performance.csv",
        index=False,
    )
    blocks.to_csv(OUTPUT_DIR / "block_performance.csv", index=False)
    diagnostics.to_csv(
        OUTPUT_DIR / "daily_gate_diagnostics.csv",
        index=False,
    )
    gates.to_csv(OUTPUT_DIR / "gate_coverage.csv", index=False)
    (OUTPUT_DIR / "selected_policy.txt").write_text(
        selected + "\n",
        encoding="utf-8",
    )
    write_report(
        development,
        ranking,
        exposed,
        blocks,
        gates,
        selected,
    )
    print(ranking.to_string(index=False))
    print(f"\nSelected: {selected}")
    print("\nExposed")
    print(exposed.to_string(index=False))
    print(f"\nWrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
