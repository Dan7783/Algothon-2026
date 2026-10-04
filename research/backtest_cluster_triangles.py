"""Validation-only prototype for replacing overlapping pair triangles.

The triangle set and each basket relation are selected on days 1-500 only.
No days after 750 are loaded.  The prototype compares:

1. the nine original internal pair edges in isolation;
2. three disjoint three-asset basket residuals in isolation; and
3. the full strategy after every pair/ridge signal touching those cluster
   assets is replaced by the three cluster vectors.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy
from research import analyse_pair_clusters as cluster_analysis
from research import whole_vector_ev_conflicts as study


OUTPUT_DIR = ROOT / "analysis_outputs" / "pair_clusters"
SELECTED_PATH = OUTPUT_DIR / "train_selected_disjoint_triangles.csv"
VAL_WINDOWS = (
    ("VAL1", 501, 625),
    ("VAL2", 626, 750),
    ("VALIDATION", 501, 750),
)


@dataclass
class ClusterAgent:
    name: str
    indices: tuple[int, int, int]
    vector: np.ndarray
    intercept: float
    mean: float
    std: float
    z_entry: float
    z_exit: float
    z_stop: float
    side: int = 0

    def residual(self, log_prices: np.ndarray) -> float:
        values = log_prices[np.asarray(self.indices)]
        return float(values @ self.vector - self.intercept)

    def update(self, log_prices: np.ndarray) -> tuple[np.ndarray, float]:
        residual = self.residual(log_prices)
        z = (residual - self.mean) / max(self.std, 1.0e-12)
        abs_z = abs(z)
        if self.side == 0:
            if self.z_entry <= abs_z < self.z_stop:
                self.side = int(-np.sign(z)) if z != 0.0 else 0
        else:
            exit_trade = False
            if self.side == 1 and z >= -self.z_exit:
                exit_trade = True
            elif self.side == -1 and z <= self.z_exit:
                exit_trade = True
            if exit_trade or abs_z >= self.z_stop:
                self.side = 0

        dollars = np.zeros(strategy.N_INST)
        if self.side:
            scale = strategy.LEG / max(1.0, float(np.max(np.abs(self.vector))))
            dollars[np.asarray(self.indices)] = (
                scale * float(self.side) * self.vector
            )
        return dollars, float(z)


def _internal_configs(members: set[int]) -> list[strategy.PairCfg]:
    return [
        spec
        for spec in strategy.PAIR_CFGS
        if int(spec.y) in members and int(spec.x) in members
    ]


def build_cluster_agents(
    log_prices: np.ndarray,
) -> tuple[list[ClusterAgent], set[int], set[int]]:
    selected = pd.read_csv(SELECTED_PATH)
    if selected.empty:
        raise RuntimeError("No train-selected triangle set was produced")
    agents: list[ClusterAgent] = []
    cluster_assets: set[int] = set()
    internal_pair_indices: set[int] = set()

    for row in selected.itertuples(index=False):
        anchor = int(strategy.I[str(row.anchor)])
        peer_1 = int(strategy.I[str(row.peer_1)])
        peer_2 = int(strategy.I[str(row.peer_2)])
        indices = (anchor, peer_1, peer_2)
        members = set(indices)
        cluster_assets.update(members)
        internal = _internal_configs(members)
        if len(internal) != 3:
            raise RuntimeError(
                f"{row.cluster} is not backed by exactly three configured edges"
            )
        internal_pair_indices.update(int(spec.idx) for spec in internal)
        entries = np.asarray([spec.trade.z_entry for spec in internal])
        exits = np.asarray([spec.trade.z_exit for spec in internal])
        stops = np.asarray([spec.trade.z_stop for spec in internal])
        vector = np.asarray(
            [1.0, -float(row.beta_1), -float(row.beta_2)],
            dtype=float,
        )
        train_residual = (
            log_prices[np.asarray(indices), :500].T @ vector
            - float(row.intercept)
        )
        agents.append(
            ClusterAgent(
                name=str(row.cluster),
                indices=indices,
                vector=vector,
                intercept=float(row.intercept),
                mean=float(np.mean(train_residual)),
                std=float(np.std(train_residual)),
                z_entry=float(np.median(entries)),
                z_exit=float(np.median(exits)),
                z_stop=float(np.median(stops)),
            )
        )
    return agents, cluster_assets, internal_pair_indices


def _current_direction(
    history: np.ndarray,
    candidates: list[tuple[float, strategy.PairAgent, float, float]],
    ridge_blocked_assets: set[int] | None = None,
) -> np.ndarray:
    pair_vote, pair_active, _ = strategy._pair_conflict_consensus(candidates)
    returns = np.diff(
        np.log(np.maximum(history, strategy._EPS_DENOM)),
        axis=1,
    )
    ridge_raw, ridge_strength = strategy._ridge_forecast(returns)
    ridge_sign = np.sign(ridge_raw)
    ridge_active = ridge_sign != 0.0
    if ridge_blocked_assets:
        blocked = np.asarray(sorted(ridge_blocked_assets), dtype=int)
        ridge_sign[blocked] = 0.0
        ridge_strength[blocked] = 0.0
        ridge_active[blocked] = False
    combined = (
        pair_vote
        + strategy.RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
    )
    active = pair_active | ridge_active
    direction = np.sign(combined)
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]
    last_return = np.diff(
        np.log(np.maximum(history[:, -2:], strategy._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    direction[crash] = 0.0
    return direction


def replay(prices: np.ndarray) -> dict[str, np.ndarray | int | list[dict]]:
    n_history = prices.shape[1]
    log_prices = np.log(np.maximum(prices, strategy._EPS_DENOM))
    (
        cluster_agents,
        cluster_assets,
        internal_pair_indices,
    ) = build_cluster_agents(log_prices)

    cluster_only = np.zeros((n_history + 1, strategy.N_INST))
    pair_only = np.zeros_like(cluster_only)
    full_replacement = np.zeros_like(cluster_only)
    per_cluster = {
        agent.name: np.zeros_like(cluster_only)
        for agent in cluster_agents
    }
    full_atlas = strategy.AtlasBook()
    internal_agents = [
        strategy.PairAgent(spec)
        for spec in strategy.PAIR_CFGS
        if int(spec.idx) in internal_pair_indices
    ]
    removed_pair_names = [
        strategy._pair_name(spec)
        for spec in strategy.PAIR_CFGS
        if (
            int(spec.y) in cluster_assets
            or int(spec.x) in cluster_assets
        )
    ]

    for history_size in range(1, n_history):
        history = prices[:, :history_size]
        all_candidates = full_atlas.candidates(history)
        if history_size < strategy.WARMUP + 10:
            continue

        for agent in internal_agents:
            agent.catch_up(history)
        internal_candidates = []
        for agent in internal_agents:
            pos_y, pos_x, abs_z = agent.dollar_positions()
            if abs(pos_y) <= 1.0e-9:
                continue
            priority = agent.current_kappa * abs_z
            internal_candidates.append((priority, agent, pos_y, pos_x))
        internal_vote, internal_active, _ = (
            strategy._pair_conflict_consensus(internal_candidates)
        )
        pair_only[history_size] = (
            strategy.BANG_BANG_LIMITS
            * np.sign(internal_vote)
            * internal_active
        )

        cluster_today = np.zeros(strategy.N_INST)
        if history_size >= 500:
            current_log_prices = log_prices[:, history_size - 1]
            for agent in cluster_agents:
                positions, _ = agent.update(current_log_prices)
                per_cluster[agent.name][history_size] = positions
                cluster_today += positions
        cluster_only[history_size] = cluster_today

        allowed_candidates = [
            candidate
            for candidate in all_candidates
            if (
                int(candidate[1].spec.y) not in cluster_assets
                and int(candidate[1].spec.x) not in cluster_assets
            )
        ]
        direction = _current_direction(
            history,
            allowed_candidates,
            ridge_blocked_assets=cluster_assets,
        )
        full_replacement[history_size] = (
            strategy.BANG_BANG_LIMITS * direction
        )
        full_replacement[
            history_size, np.asarray(sorted(cluster_assets), dtype=int)
        ] = cluster_today[np.asarray(sorted(cluster_assets), dtype=int)]

    return {
        "cluster_only": cluster_only,
        "pair_only": pair_only,
        "full_replacement": full_replacement,
        "per_cluster": per_cluster,
        "removed_pair_count": len(removed_pair_names),
        "removed_pair_names": removed_pair_names,
        "cluster_assets": sorted(
            strategy._SYMBOL_BY_INDEX[index] for index in cluster_assets
        ),
    }


def metrics(
    prices: np.ndarray,
    dollars: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: study.region_metrics(prices, dollars, start, end)
        for name, start, end in VAL_WINDOWS
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SELECTED_PATH.exists():
        cluster_analysis.main()
    prices = study.load_prices(750)
    baseline = study.collect_signals(prices).baseline_dollars
    replayed = replay(prices)
    cluster_priority = baseline.copy()
    cluster_confirmation = baseline.copy()
    for dollars in replayed["per_cluster"].values():
        active_by_history = np.any(np.abs(dollars) > 1.0e-9, axis=1)
        member_indices = np.flatnonzero(
            np.any(np.abs(dollars) > 1.0e-9, axis=0)
        )
        for history_size in np.flatnonzero(active_by_history):
            proposed = dollars[history_size, member_indices]
            cluster_priority[history_size, member_indices] = proposed
            existing = cluster_confirmation[
                history_size, member_indices
            ]
            disagreement = (
                (np.sign(existing) != 0.0)
                & (np.sign(existing) != np.sign(proposed))
            )
            cluster_confirmation[
                history_size, member_indices[disagreement]
            ] = 0.0

    result = {
        "protocol": {
            "relation_fit": [1, 500],
            "validation": [501, 750],
            "oos_loaded": False,
        },
        "cluster_assets": replayed["cluster_assets"],
        "removed_pair_count": replayed["removed_pair_count"],
        "removed_pair_names": replayed["removed_pair_names"],
        "full_baseline": metrics(prices, baseline),
        "full_cluster_replacement": metrics(
            prices,
            replayed["full_replacement"],
        ),
        "cluster_priority_arbitration": metrics(
            prices,
            cluster_priority,
        ),
        "cluster_disagreement_filter": metrics(
            prices,
            cluster_confirmation,
        ),
        "internal_pair_edges_only": metrics(
            prices,
            replayed["pair_only"],
        ),
        "cluster_baskets_only": metrics(
            prices,
            replayed["cluster_only"],
        ),
        "individual_clusters": {
            name: metrics(prices, dollars)
            for name, dollars in replayed["per_cluster"].items()
        },
    }
    for window, _, _ in VAL_WINDOWS:
        result.setdefault("full_replacement_score_delta", {})[window] = (
            result["full_cluster_replacement"][window]["score"]
            - result["full_baseline"][window]["score"]
        )
        result.setdefault("cluster_vs_internal_score_delta", {})[window] = (
            result["cluster_baskets_only"][window]["score"]
            - result["internal_pair_edges_only"][window]["score"]
        )
        result.setdefault("priority_arbitration_score_delta", {})[window] = (
            result["cluster_priority_arbitration"][window]["score"]
            - result["full_baseline"][window]["score"]
        )
        result.setdefault("disagreement_filter_score_delta", {})[window] = (
            result["cluster_disagreement_filter"][window]["score"]
            - result["full_baseline"][window]["score"]
        )

    (OUTPUT_DIR / "cluster_backtest.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Train-selected triangle basket backtest",
        "",
        "The triangle set, basket coefficients, thresholds, and hedge directions",
        "are fixed from days 1-500. Days 751-1000 are not loaded.",
        "",
        f"- Cluster assets: `{', '.join(result['cluster_assets'])}`",
        f"- Pair relationships removed in the clean replacement: "
        f"**{result['removed_pair_count']}**",
        "",
        "## Scores",
        "",
        "| Sleeve | VAL1 | VAL2 | Full validation |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Current full strategy", "full_baseline"),
        ("Full cluster replacement", "full_cluster_replacement"),
        ("Cluster-priority arbitration", "cluster_priority_arbitration"),
        ("Cluster disagreement filter", "cluster_disagreement_filter"),
        ("Nine internal pair edges only", "internal_pair_edges_only"),
        ("Three cluster baskets only", "cluster_baskets_only"),
    ):
        values = result[key]
        lines.append(
            f"| {label} | {values['VAL1']['score']:.4f} | "
            f"{values['VAL2']['score']:.4f} | "
            f"{values['VALIDATION']['score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Full replacement deltas",
            "",
            (
                f"- VAL1: "
                f"`{result['full_replacement_score_delta']['VAL1']:.4f}`"
            ),
            (
                f"- VAL2: "
                f"`{result['full_replacement_score_delta']['VAL2']:.4f}`"
            ),
            (
                f"- Validation: "
                f"`{result['full_replacement_score_delta']['VALIDATION']:.4f}`"
            ),
            "",
            "The clean replacement removes every configured pair touching a",
            "selected cluster asset and disables ridge on those assets, then",
            "installs one hedge-ratio-preserving vector per triangle.",
        ]
    )
    (OUTPUT_DIR / "cluster_backtest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
