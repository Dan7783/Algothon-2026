#!/usr/bin/env python
"""Measure the EFFECTIVE breadth of the four_sigmas pair book.

The book runs 88 pairs over 48 assets (mean degree 3.7, max 11). The
fundamental law says IR ~ IC*sqrt(BR) with BR = number of INDEPENDENT bets.
If the 88 pairs are really N << 88 relationships restated, expected IR is
overstated by sqrt(88/N).

This script answers the question with three independent estimators applied to
the pairs' STANDALONE daily PnL (not their signals -- PnL is what the score
sees):

  1. Meucci (2009) entropy effective number of bets on the PCA of the pair
     PnL correlation matrix.
  2. Participation ratio (sum l)^2 / sum l^2 -- the physics convention.
  3. Marchenko-Pastur: how many eigenvalues clear the RMT noise bound for
     this N/T, i.e. how many relationships are distinguishable from noise.

Also reports the vote-weight (kappa*|z|) dispersion that sets the asset-level
consensus, and the realised correlation of the votes on hub assets.

Run from the repo root:  python research/effective_breadth.py
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

FIRST_DAY = 250
LAST_DAY = 1000
OUT_DIR = ROOT / "analysis_outputs" / "effective_breadth"

# Eval windows.  501-750 is BURNED validation (see VALIDATION_PROTOCOL.md);
# 751-1000 is the organiser's scored window, which this catalog has partly
# seen via PREVIOUS_OOS_REMOVED_PAIR_NAMES.
WINDOWS = {
    "train  251-500": (251, 500),
    "val    501-750": (501, 750),
    "scored 751-1000": (751, 1000),
    "all    251-1000": (251, 1000),
}


def load_prices() -> np.ndarray:
    return pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T


def replay(prices: np.ndarray) -> dict:
    """One pass over the book, caching each pair's own requested leg dollars."""
    n_pairs = len(fs.PAIR_CFGS)
    days = np.arange(FIRST_DAY, LAST_DAY + 1)

    pos_y = np.zeros((len(days), n_pairs))
    pos_x = np.zeros((len(days), n_pairs))
    priority = np.zeros((len(days), n_pairs))
    kappa = np.zeros((len(days), n_pairs))

    book = fs.AtlasBook()
    for row, t in enumerate(days):
        history = prices[:, :t]
        for agent in book.agents:
            agent.catch_up(history)
        for j, agent in enumerate(book.agents):
            py, px, abs_z = agent.dollar_positions()
            pos_y[row, j] = py
            pos_x[row, j] = px
            priority[row, j] = agent.current_kappa * abs_z
            kappa[row, j] = agent.current_kappa

    return {
        "days": days,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "priority": priority,
        "kappa": kappa,
        "y_index": np.array([int(s.y) for s in fs.PAIR_CFGS]),
        "x_index": np.array([int(s.x) for s in fs.PAIR_CFGS]),
        "names": np.array([fs._pair_name(s) for s in fs.PAIR_CFGS]),
    }


def standalone_pnl(cache: dict, prices: np.ndarray) -> np.ndarray:
    """Per-pair daily PnL from its OWN requested legs, unconstrained by the vote.

    A position decided from prices[:, :t] is held into day t, so it earns the
    return from price[t-1] to price[t] (0-indexed: prices[:, t-1] / prices[:, t-2]).
    """
    days = cache["days"]
    # simple return realised on the day the position is held
    rets = np.zeros((len(days), prices.shape[0]))
    for row, t in enumerate(days):
        if t >= prices.shape[1]:
            continue
        rets[row] = prices[:, t] / prices[:, t - 1] - 1.0

    yi, xi = cache["y_index"], cache["x_index"]
    return cache["pos_y"] * rets[:, yi] + cache["pos_x"] * rets[:, xi]


def eff_bets(corr: np.ndarray) -> dict:
    """Meucci entropy n_eff, participation ratio, and MP noise-bound count."""
    lam = np.linalg.eigvalsh(corr)[::-1]
    lam = np.clip(lam, 0.0, None)
    total = lam.sum()
    p = lam / total if total > 0 else lam
    nz = p[p > 1e-12]
    entropy_neff = float(np.exp(-(nz * np.log(nz)).sum()))
    participation = float(total**2 / (lam**2).sum())
    return {
        "eigenvalues": lam,
        "meucci_neff": entropy_neff,
        "participation_ratio": participation,
        "top1_var_share": float(p[0]),
        "top5_var_share": float(p[:5].sum()),
    }


def mp_bound(n: int, t: int) -> float:
    """Marchenko-Pastur upper edge for an NxN correlation matrix from T obs."""
    q = n / t
    return (1.0 + np.sqrt(q)) ** 2


def analyse_window(pnl: np.ndarray, days: np.ndarray, lo: int, hi: int, label: str):
    mask = (days >= lo) & (days <= hi)
    block = pnl[mask]
    # keep pairs that actually traded in this window
    active = block.std(axis=0) > 1e-9
    block = block[:, active]
    t, n = block.shape
    corr = np.corrcoef(block, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    stats = eff_bets(corr)
    bound = mp_bound(n, t)
    above = int((stats["eigenvalues"] > bound).sum())

    off = corr[np.triu_indices(n, k=1)]

    print(f"\n--- {label}   (T={t} days, N={n} pairs trading) ---")
    print(f"  mean |pairwise PnL corr|      {np.abs(off).mean():.4f}"
          f"   mean signed {off.mean():+.4f}   max {off.max():.3f}")
    print(f"  Meucci entropy n_eff          {stats['meucci_neff']:.1f}"
          f"   ({stats['meucci_neff']/n*100:.0f}% of {n})")
    print(f"  participation ratio           {stats['participation_ratio']:.1f}")
    print(f"  PC1 variance share            {stats['top1_var_share']*100:.1f}%"
          f"   PC1-5 {stats['top5_var_share']*100:.1f}%")
    print(f"  MP noise edge (N/T={n/t:.2f})     {bound:.2f}"
          f"  ->  {above} eigenvalues above noise")
    print(f"  IR inflation if BR taken as N: sqrt({n}/{above}) ="
          f" {np.sqrt(n/max(above,1)):.2f}x   (vs Meucci:"
          f" {np.sqrt(n/max(stats['meucci_neff'],1e-9)):.2f}x)")
    return {
        "label": label, "T": t, "N": n,
        "meucci_neff": stats["meucci_neff"],
        "participation": stats["participation_ratio"],
        "mp_above": above,
        "pc1_share": stats["top1_var_share"],
        "mean_abs_corr": float(np.abs(off).mean()),
    }


def vote_weight_diagnostics(cache: dict) -> None:
    """Section 3.3: is kappa*|z| dominated by units rather than information?"""
    pri = cache["priority"]
    kap = cache["kappa"]
    names = cache["names"]
    traded = np.abs(cache["pos_y"]) > 1e-9

    print("\n=== vote-weight (priority = kappa * |z|) dispersion ===")
    med_k = np.array([np.median(kap[traded[:, j], j]) if traded[:, j].any() else np.nan
                      for j in range(kap.shape[1])])
    med_p = np.array([np.median(pri[traded[:, j], j]) if traded[:, j].any() else np.nan
                      for j in range(pri.shape[1])])
    ok = np.isfinite(med_k)
    print(f"  per-pair median kappa: min {np.nanmin(med_k):.4f}"
          f"  max {np.nanmax(med_k):.4f}  ratio {np.nanmax(med_k)/np.nanmin(med_k):.0f}x")
    print(f"  per-pair median priority: min {np.nanmin(med_p):.4f}"
          f"  max {np.nanmax(med_p):.4f}  ratio {np.nanmax(med_p)/np.nanmin(med_p):.0f}x")

    order = np.argsort(-med_p[ok])
    nm, mp_ = names[ok], med_p[ok]
    print("  heaviest voters:  " + ", ".join(f"{nm[i]}={mp_[i]:.2f}" for i in order[:5]))
    print("  lightest voters:  " + ", ".join(f"{nm[i]}={mp_[i]:.3f}" for i in order[-5:]))

    # what share of a hub asset's total vote weight does its top pair carry?
    print("\n=== vote concentration on hub assets (share of |vote weight|) ===")
    yi, xi = cache["y_index"], cache["x_index"]
    for sym in ("ULXY", "RTTH", "ANSO", "SPLZ", "FCSG"):
        a = fs.I[sym]
        legs = np.flatnonzero((yi == a) | (xi == a))
        w = pri[:, legs] * traded[:, legs]
        tot = w.sum(axis=1)
        live = tot > 0
        if not live.any():
            continue
        share = (w[live].max(axis=1) / tot[live])
        n_live = (w[live] > 0).sum(axis=1)
        print(f"  {sym}: {len(legs)} pairs | median live voters {np.median(n_live):.0f}"
              f" | top voter carries median {np.median(share)*100:.0f}% of the weight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"prices {prices.shape[0]} instruments x {prices.shape[1]} days")
    print(f"replaying {len(fs.PAIR_CFGS)} pairs over days {FIRST_DAY}-{LAST_DAY} ...")
    cache = replay(prices)
    pnl = standalone_pnl(cache, prices)

    print("\n" + "=" * 72)
    print("EFFECTIVE BREADTH of the pair book (standalone per-pair PnL)")
    print("=" * 72)
    rows = [analyse_window(pnl, cache["days"], lo, hi, label)
            for label, (lo, hi) in WINDOWS.items()]

    vote_weight_diagnostics(cache)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "effective_breadth.csv", index=False)
    np.savez_compressed(OUT_DIR / "pair_pnl.npz", pnl=pnl, days=cache["days"],
                        names=cache["names"])
    print(f"\nwrote {OUT_DIR / 'effective_breadth.csv'}")


if __name__ == "__main__":
    main()
