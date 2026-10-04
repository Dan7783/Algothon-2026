#!/usr/bin/env python
"""Is _pair_conflict_consensus a jury, or a dictatorship by the largest kappa?

The brief argues the asset vote suffers the Condorcet failure mode: correlated
voters make a 7-4 split look more informative than it is. That presumes there
ARE several effective voters. This measures how often the single
heaviest-weighted pair already determines the asset's consensus sign, and what
the effective number of voters per asset actually is (weight entropy).

Run from the repo root:  python research/vote_dictatorship.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as fs

FIRST_DAY, LAST_DAY = 250, 1000


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    book = fs.AtlasBook()

    n_dec = 0            # asset-days with >=2 live voters
    n_dictator = 0       # ... where the top voter's sign == consensus sign
    n_would_flip = 0     # ... where equal-weight voting would give a different sign
    voter_counts, weight_neff, top_shares = [], [], []

    for t in range(FIRST_DAY, LAST_DAY + 1):
        candidates = book.candidates(prices[:, :t])
        # rebuild the per-asset weight ledger the allocator uses
        legs: dict[int, list[tuple[float, float]]] = {}
        for priority, agent, pos_y, pos_x in candidates:
            boost = (fs.PROTECTED_VOTE_BOOST
                     if fs._pair_name(agent.spec) in fs.PROTECTED_PAIR_NAMES else 1.0)
            w = boost * max(float(priority), 1.0e-12)
            for idx, pos in ((int(agent.spec.y), pos_y), (int(agent.spec.x), pos_x)):
                if abs(pos) > 1.0e-9:
                    legs.setdefault(idx, []).append((w, float(np.sign(pos))))

        for idx, entries in legs.items():
            if len(entries) < 2:
                continue
            w = np.array([e[0] for e in entries])
            s = np.array([e[1] for e in entries])
            n_dec += 1
            voter_counts.append(len(entries))

            p = w / w.sum()
            weight_neff.append(float(np.exp(-(p * np.log(np.maximum(p, 1e-300))).sum())))
            top_shares.append(float(p.max()))

            consensus = np.sign((w * s).sum())
            if np.sign(s[np.argmax(w)]) == consensus and consensus != 0:
                n_dictator += 1
            if np.sign(s.mean()) != consensus:
                n_would_flip += 1

    vc = np.array(voter_counts)
    ne = np.array(weight_neff)
    ts = np.array(top_shares)

    print("=" * 66)
    print("ASSET-LEVEL VOTE STRUCTURE (days 250-1000, contested asset-days only)")
    print("=" * 66)
    print(f"contested asset-days (>=2 live voters): {n_dec}")
    print(f"  nominal voters per asset:  mean {vc.mean():.2f}  median {np.median(vc):.0f}"
          f"  max {vc.max()}")
    print(f"  EFFECTIVE voters (weight entropy): mean {ne.mean():.2f}"
          f"  median {np.median(ne):.2f}")
    print(f"  top voter's weight share:  mean {ts.mean()*100:.1f}%"
          f"  median {np.median(ts)*100:.1f}%")
    print(f"  share of asset-days where top voter's sign == consensus:"
          f" {n_dictator/n_dec*100:.1f}%")
    print(f"  share where ONE-PAIR-ONE-VOTE would flip the sign:"
          f" {n_would_flip/n_dec*100:.1f}%")
    print()
    print(f"  asset-days with effective voters < 1.5: {(ne < 1.5).mean()*100:.1f}%")
    print(f"  asset-days with effective voters < 2.0: {(ne < 2.0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
