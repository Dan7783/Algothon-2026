"""Analyse whether four_sigmas pair overlap represents tradable clusters.

Only days 1-750 are loaded.  Cluster relationships are estimated on days
1-500 and assessed, without refitting, on 501-625 and 626-750.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as strategy


OUTPUT_DIR = ROOT / "analysis_outputs" / "pair_clusters"
PRICE_PATH = ROOT / "prices.txt"
TRAIN = slice(0, 500)
VAL1 = slice(500, 625)
VAL2 = slice(625, 750)


@dataclass(frozen=True)
class Relation:
    anchor: int
    peers: tuple[int, int]
    intercept: float
    coefficients: tuple[float, float]

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(
            [1.0, -self.coefficients[0], -self.coefficients[1]],
            dtype=float,
        )


def load_prices() -> np.ndarray:
    prices = pd.read_csv(PRICE_PATH, sep=r"\s+").to_numpy(dtype=float).T
    if prices.shape[1] < 750:
        raise RuntimeError("Need at least 750 price days")
    return prices[:, :750]


def pair_graph() -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    edges: set[tuple[str, str]] = set()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for spec in strategy.PAIR_CFGS:
        y = strategy._SYMBOL_BY_INDEX[int(spec.y)]
        x = strategy._SYMBOL_BY_INDEX[int(spec.x)]
        edge = tuple(sorted((y, x)))
        edges.add(edge)
        adjacency[y].add(x)
        adjacency[x].add(y)
    return edges, adjacency


def connected_components(
    adjacency: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    unseen = set(adjacency)
    components: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda group: (-len(group), group))


def graph_triangles(
    adjacency: dict[str, set[str]],
) -> list[tuple[str, str, str]]:
    symbols = sorted(adjacency)
    return [
        triple
        for triple in itertools.combinations(symbols, 3)
        if (
            triple[1] in adjacency[triple[0]]
            and triple[2] in adjacency[triple[0]]
            and triple[2] in adjacency[triple[1]]
        )
    ]


def wedge_triples(
    adjacency: dict[str, set[str]],
) -> list[tuple[str, str, str]]:
    """Three-node groups with at least two configured pair edges."""
    groups: set[tuple[str, str, str]] = set()
    for center, neighbours in adjacency.items():
        for left, right in itertools.combinations(sorted(neighbours), 2):
            groups.add(tuple(sorted((center, left, right))))
    return sorted(groups)


def safe_adf(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 30 or float(np.std(values)) < 1.0e-10:
        return float("nan"), 1.0
    try:
        statistic, p_value, *_ = adfuller(
            values,
            maxlag=1,
            regression="c",
            autolag=None,
        )
        return float(statistic), float(p_value)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), 1.0


def johansen_rank(values: np.ndarray) -> int:
    try:
        result = coint_johansen(values, det_order=0, k_ar_diff=1)
    except (ValueError, np.linalg.LinAlgError):
        return 0
    return int(np.sum(result.lr1 > result.cvt[:, 1]))


def fit_best_relation(
    log_prices: np.ndarray,
    indices: tuple[int, int, int],
) -> tuple[Relation, np.ndarray]:
    best: tuple[float, Relation, np.ndarray] | None = None
    train_values = log_prices[np.asarray(indices), TRAIN].T
    for anchor_position in range(3):
        peer_positions = tuple(
            position for position in range(3) if position != anchor_position
        )
        design = np.column_stack(
            [
                np.ones(len(train_values)),
                train_values[:, peer_positions[0]],
                train_values[:, peer_positions[1]],
            ]
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            train_values[:, anchor_position],
            rcond=None,
        )
        relation = Relation(
            anchor=indices[anchor_position],
            peers=(
                indices[peer_positions[0]],
                indices[peer_positions[1]],
            ),
            intercept=float(coefficients[0]),
            coefficients=(
                float(coefficients[1]),
                float(coefficients[2]),
            ),
        )
        residual = (
            log_prices[relation.anchor]
            - relation.intercept
            - relation.coefficients[0] * log_prices[relation.peers[0]]
            - relation.coefficients[1] * log_prices[relation.peers[1]]
        )
        statistic, _ = safe_adf(residual[TRAIN])
        ranking = statistic if np.isfinite(statistic) else np.inf
        if best is None or ranking < best[0]:
            best = (ranking, relation, residual)
    if best is None:
        raise RuntimeError("Could not fit a three-asset relation")
    return best[1], best[2]


def half_life(values: np.ndarray) -> float:
    lagged = values[:-1]
    change = np.diff(values)
    design = np.column_stack([np.ones(len(lagged)), lagged])
    coefficients, *_ = np.linalg.lstsq(design, change, rcond=None)
    slope = float(coefficients[1])
    if slope >= -1.0e-8:
        return float("inf")
    return float(np.log(2.0) / -slope)


def mean_return_correlation(
    log_prices: np.ndarray,
    indices: tuple[int, int, int],
    window: slice,
) -> tuple[float, float]:
    returns = np.diff(log_prices[np.asarray(indices), window], axis=1)
    correlation = np.corrcoef(returns)
    off_diagonal = correlation[np.triu_indices(3, k=1)]
    return float(np.mean(off_diagonal)), float(np.min(off_diagonal))


def analyse_group(
    symbols: tuple[str, str, str],
    log_prices: np.ndarray,
    edge_set: set[tuple[str, str]],
) -> dict[str, float | int | str | bool]:
    indices = tuple(int(strategy.I[symbol]) for symbol in symbols)
    relation, residual = fit_best_relation(log_prices, indices)
    train_stat, train_p = safe_adf(residual[TRAIN])
    val1_stat, val1_p = safe_adf(residual[VAL1])
    val2_stat, val2_p = safe_adf(residual[VAL2])
    train_corr, train_min_corr = mean_return_correlation(
        log_prices, indices, TRAIN
    )
    val1_corr, val1_min_corr = mean_return_correlation(
        log_prices, indices, VAL1
    )
    val2_corr, val2_min_corr = mean_return_correlation(
        log_prices, indices, VAL2
    )
    train_values = log_prices[np.asarray(indices), TRAIN].T
    val1_values = log_prices[np.asarray(indices), VAL1].T
    val2_values = log_prices[np.asarray(indices), VAL2].T
    edge_count = sum(
        tuple(sorted(edge)) in edge_set
        for edge in itertools.combinations(symbols, 2)
    )
    coefficient_norm = float(
        np.sum(np.abs(np.asarray(relation.coefficients)))
    )
    robust = bool(
        train_p < 0.05
        and val1_p < 0.10
        and val2_p < 0.10
        and johansen_rank(train_values) >= 1
        and johansen_rank(val1_values) >= 1
        and johansen_rank(val2_values) >= 1
        and train_corr > 0.10
        and val1_corr > 0.10
        and val2_corr > 0.10
        and coefficient_norm < 10.0
    )
    return {
        "cluster": "-".join(symbols),
        "edge_count": edge_count,
        "is_triangle": edge_count == 3,
        "anchor": strategy._SYMBOL_BY_INDEX[relation.anchor],
        "peer_1": strategy._SYMBOL_BY_INDEX[relation.peers[0]],
        "peer_2": strategy._SYMBOL_BY_INDEX[relation.peers[1]],
        "intercept": relation.intercept,
        "beta_1": relation.coefficients[0],
        "beta_2": relation.coefficients[1],
        "coefficient_l1": coefficient_norm,
        "train_adf_stat": train_stat,
        "train_adf_p": train_p,
        "val1_adf_stat": val1_stat,
        "val1_adf_p": val1_p,
        "val2_adf_stat": val2_stat,
        "val2_adf_p": val2_p,
        "train_johansen_rank": johansen_rank(train_values),
        "val1_johansen_rank": johansen_rank(val1_values),
        "val2_johansen_rank": johansen_rank(val2_values),
        "train_mean_return_corr": train_corr,
        "val1_mean_return_corr": val1_corr,
        "val2_mean_return_corr": val2_corr,
        "train_min_return_corr": train_min_corr,
        "val1_min_return_corr": val1_min_corr,
        "val2_min_return_corr": val2_min_corr,
        "train_half_life": half_life(residual[TRAIN]),
        "val1_half_life": half_life(residual[VAL1]),
        "val2_half_life": half_life(residual[VAL2]),
        "robust_cluster": robust,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    log_prices = np.log(np.maximum(prices, strategy._EPS_DENOM))
    edges, adjacency = pair_graph()
    components = connected_components(adjacency)
    triangles = graph_triangles(adjacency)
    wedges = wedge_triples(adjacency)
    degrees = Counter({symbol: len(peers) for symbol, peers in adjacency.items()})

    rows = [
        analyse_group(group, log_prices, edges)
        for group in wedges
    ]
    frame = pd.DataFrame(rows).sort_values(
        [
            "robust_cluster",
            "is_triangle",
            "val2_adf_p",
            "val1_adf_p",
        ],
        ascending=[False, False, True, True],
    )
    frame.to_csv(OUTPUT_DIR / "triple_diagnostics.csv", index=False)

    robust = frame[frame["robust_cluster"]].copy()
    train_triangle_candidates = frame[
        frame["is_triangle"]
        & (frame["train_adf_p"] < 0.01)
        & (frame["train_johansen_rank"] >= 1)
        & (frame["train_mean_return_corr"] > 0.10)
        & (frame["coefficient_l1"] < 10.0)
        & (frame["train_half_life"] >= 1.0)
        & (frame["train_half_life"] <= 100.0)
    ].sort_values("train_adf_stat")
    selected_rows: list[pd.Series] = []
    used_symbols: set[str] = set()
    for _, row in train_triangle_candidates.iterrows():
        members = set(str(row["cluster"]).split("-"))
        if members & used_symbols:
            continue
        selected_rows.append(row)
        used_symbols.update(members)
    selected_triangles = pd.DataFrame(selected_rows)
    if len(selected_triangles):
        selected_triangles = selected_triangles.copy()
        selected_triangles["validation_adf_pass"] = (
            (selected_triangles["val1_adf_p"] < 0.10)
            & (selected_triangles["val2_adf_p"] < 0.10)
        )
    selected_triangles.to_csv(
        OUTPUT_DIR / "train_selected_disjoint_triangles.csv",
        index=False,
    )
    summary = {
        "configured_pairs": len(edges),
        "assets": len(adjacency),
        "overlap_assets": int(sum(degree > 1 for degree in degrees.values())),
        "pair_legs_on_overlap_assets": int(
            sum(degree for degree in degrees.values() if degree > 1)
        ),
        "connected_components": [list(group) for group in components],
        "component_sizes": [len(group) for group in components],
        "triangles": len(triangles),
        "two_edge_or_triangle_triples": len(wedges),
        "robust_triples": len(robust),
        "robust_triangles": int(
            robust["is_triangle"].sum() if len(robust) else 0
        ),
        "train_selected_disjoint_triangles": (
            selected_triangles["cluster"].tolist()
            if len(selected_triangles)
            else []
        ),
        "selected_triangles_passing_both_validation_halves": int(
            selected_triangles["validation_adf_pass"].sum()
            if len(selected_triangles)
            else 0
        ),
        "degree_ranking": degrees.most_common(),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Pair-cluster structure",
        "",
        "All relationship fitting uses days 1-500. The fitted relation is held",
        "fixed in validation windows 501-625 and 626-750.",
        "",
        "## Graph",
        "",
        f"- Configured edges: **{len(edges)}**",
        f"- Assets: **{len(adjacency)}**",
        f"- Assets with degree > 1: **{summary['overlap_assets']}**",
        f"- Exact triangles: **{len(triangles)}**",
        f"- Candidate triples with at least two edges: **{len(wedges)}**",
        f"- Connected-component sizes: **{summary['component_sizes']}**",
        "",
        "## Stable multivariate candidates",
        "",
        f"- Robust triples: **{len(robust)}**",
        f"- Robust exact triangles: **{summary['robust_triangles']}**",
        "",
    ]
    if len(robust):
        columns = [
            "cluster",
            "edge_count",
            "anchor",
            "beta_1",
            "beta_2",
            "train_adf_p",
            "val1_adf_p",
            "val2_adf_p",
            "train_mean_return_corr",
            "val1_mean_return_corr",
            "val2_mean_return_corr",
        ]
        lines.extend(
            [
                "```text",
                robust[columns].head(30).to_string(index=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Train-only disjoint triangle selection",
            "",
            (
                "Triangles are ranked and made disjoint using days 1-500 only; "
                "the validation columns are diagnostics, not selection inputs."
            ),
            "",
        ]
    )
    if len(selected_triangles):
        triangle_columns = [
            "cluster",
            "anchor",
            "beta_1",
            "beta_2",
            "train_adf_p",
            "val1_adf_p",
            "val2_adf_p",
            "validation_adf_pass",
        ]
        lines.extend(
            [
                "```text",
                selected_triangles[triangle_columns].to_string(index=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "A connected graph or triangle is not, by itself, evidence of a",
            "tradable cluster. `robust_cluster` additionally requires a fixed",
            "train-fitted basket residual to remain stationary in both",
            "validation halves, positive average return correlation in all",
            "three windows, and Johansen rank of at least one in every window.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if len(robust):
        print()
        print(
            robust[
                [
                    "cluster",
                    "edge_count",
                    "anchor",
                    "train_adf_p",
                    "val1_adf_p",
                    "val2_adf_p",
                ]
            ].head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()
