"""Final control-versus-selected audit for the conservative hybrid update."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_adaptive_controls as controls
import audit_robust_hybrid as accounting
import archive.old_tuners.daniel_filter as daniel
import robust_hybrid_v2 as candidate


OUT = Path("cusum_output/robust_hybrid/adaptive_v2_audit.json")


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def filter_stats(log_prices, spec, cfg):
    filt = daniel.PairSegFilter(cfg)
    beta = []
    cut = []
    for t in range(log_prices.shape[1]):
        output = filt.step(log_prices[spec.y, t], log_prices[spec.x, t])
        beta.append(output.beta)
        cut.append(output.cut)
    beta = np.asarray(beta)
    cut = np.asarray(cut, dtype=bool)
    cut_idx = np.flatnonzero(cut)
    jumps = (
        np.abs(beta[cut_idx] - beta[np.maximum(cut_idx - 1, 0)])
        if len(cut_idx)
        else np.zeros(1)
    )
    return {
        "cuts": int(cut.sum()),
        "max_cut_jump": float(jumps.max()),
        "beta_total_variation": float(np.abs(np.diff(beta)).sum()),
        "p95_daily_beta_change": float(np.percentile(np.abs(np.diff(beta)), 95)),
    }


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    log_prices = np.log(prices)
    accounting.strategy = candidate

    controls.configure({})
    control = {
        "train": accounting.replay(prices, 251, 500),
        "test": accounting.replay(prices, 501, 750),
    }

    candidate.apply_policy(candidate.SELECTED_POLICY)
    selected = {
        "train": accounting.replay(prices, 251, 500),
        "test": accounting.replay(prices, 501, 750),
    }

    mtns = next(spec for spec in daniel.PAIR_CFGS if spec.name == "MTNS/ALUT")
    report = {
        "policy": serial(candidate.SELECTED_POLICY),
        "control": control,
        "selected": selected,
        "score_delta": {
            period: selected[period]["score"] - control[period]["score"]
            for period in ("train", "test")
        },
        "mean_delta": {
            period: selected[period]["mean"] - control[period]["mean"]
            for period in ("train", "test")
        },
        "half_score_delta": {
            period: [
                selected[period]["half_scores"][i]
                - control[period]["half_scores"][i]
                for i in range(2)
            ]
            for period in ("train", "test")
        },
        "mtns_alut_filter": {
            "control": filter_stats(log_prices, mtns, mtns.filt),
            "selected": filter_stats(
                log_prices, mtns, candidate._candidate_filter(mtns)
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

