#!/usr/bin/env python
"""What is a deadlock actually worth, and what accuracy would it take to win it?

A "deadlock" here = an asset-day where the pair vote is live but cancels to
exactly zero AND the ridge is inactive, so `direction` stays 0 and the asset
sits flat. Equal-mode weighting created this population; priority mode had none.

Order of operations, deliberately: establish the CEILING before building any
predictor. Prior work on pair-vs-ridge conflict arbitration
(`conflict_ml_deepdive.py`) died because the accuracy needed to beat baseline
(~54-56%) exceeded the accuracy achievable (~51-52.5%). Same discipline here:

  1. Reproduce the live baseline exactly (sanity gate -- must match eval.py).
  2. ORACLE: perfect direction on deadlock days. This is the upper bound. If
     the oracle barely moves the score, nothing else can.
  3. COIN FLIP: random direction, many seeds. This is the floor, and the gap
     oracle-minus-coinflip is the entire prize being competed for.
  4. BREAK-EVEN: the accuracy at which trading the deadlock beats staying flat.
  5. Cheap structural rules, including sub-threshold ridge (the ridge HAS an
     opinion on these days -- it is just below RIDGE_THETA, which is precisely
     why the asset deadlocked).

Run from the repo root:  python research/deadlock_ceiling.py
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

COMM = np.full(51, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0

WINDOWS = {"train 251-500": (251, 500),
           "val   501-750": (501, 750),
           "org   751-1000": (751, 1000)}


def score(mu: float, sd: float) -> float:
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250.0) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def replay(prices: np.ndarray, first: int, last: int) -> dict:
    """One pass: record baseline dollars, the deadlock mask, and features."""
    fs.reset()
    days = np.arange(first, last + 1)
    n = prices.shape[0]
    dollars = np.zeros((len(days), n))
    deadlock = np.zeros((len(days), n), dtype=bool)
    ridge_pred_sign = np.zeros((len(days), n))
    ridge_str = np.zeros((len(days), n))
    pair_contrib = np.zeros((len(days), n))

    for r, t in enumerate(days):
        s = fs.sleeve_breakdown(prices[:, :t])
        if s is None:
            continue
        dollars[r] = s["dollars"]
        # live pair vote, cancelled, ridge could not rescue it, not a crash
        dl = (s["pair_active_mask"]
              & (np.abs(s["pair_vote"]) < 1e-12)
              & ~s["ridge_active_mask"]
              & ~s["crash_mask"])
        deadlock[r] = dl
        ridge_pred_sign[r] = np.sign(s["ridge_raw"])
        # ridge_strength is populated even when below RIDGE_THETA
        ridge_str[r] = s["ridge_strength"]
        pair_contrib[r] = s["pair_contributors"]

    return {"days": days, "dollars": dollars, "deadlock": deadlock,
            "ridge_sign": ridge_pred_sign, "ridge_strength": ridge_str,
            "pair_contributors": pair_contrib}


def run_accounting(prices: np.ndarray, days: np.ndarray, dollars: np.ndarray,
                   start: int, end: int) -> np.ndarray:
    """Reproduce eval.py's PnL loop exactly for a given dollar schedule."""
    cash = value = 0.0
    comm = 0.0
    pos = np.zeros(prices.shape[0])
    pnl = []
    idx = {d: i for i, d in enumerate(days)}
    for t in range(start - 1, end + 1):
        cur = prices[:, t - 1]
        if t < end:
            d = dollars[idx[t]]
            req = np.round(d / np.maximum(cur, 1e-12)).astype(int)
            lim = (LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        delta = new - pos
        cash -= cur.dot(delta) + comm
        comm = float(np.sum(cur * np.abs(delta) * COMM))
        pos = new
        v = cash + pos.dot(cur)
        pnl.append(v - value)
        value = v
    return np.asarray(pnl[1:])


def apply_override(base: np.ndarray, deadlock: np.ndarray,
                   direction: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[deadlock] = (LIMIT[None, :] * np.ones_like(base))[deadlock] * direction[deadlock]
    return out


def main() -> None:
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    first, last = 250, 1000
    print(f"replaying days {first}-{last} ...")
    c = replay(prices, first, last)
    days, base, dl = c["days"], c["dollars"], c["deadlock"]

    # realised return on the day a position is held
    rets = np.zeros_like(base)
    for r, t in enumerate(days):
        if t < prices.shape[1]:
            rets[r] = prices[:, t] / prices[:, t - 1] - 1.0
    truth = np.sign(rets)

    print(f"deadlock asset-days: {int(dl.sum())} "
          f"({dl.sum()/len(days):.2f} per day, "
          f"{dl.sum()/max((base!=0).sum(),1)*100:.1f}% of traded asset-days)")

    # ---- what a deadlock is worth, in forgone dollars ----
    print("\n" + "=" * 74)
    print("SIZE OF THE PRIZE -- forgone PnL on deadlock asset-days")
    print("=" * 74)
    print(f"{'window':>16} {'deadlocks':>10} {'/day':>6} {'oracle $/day':>13}"
          f" {'coinflip $/day':>15}")
    for wn, (lo, hi) in WINDOWS.items():
        m = (days >= lo) & (days <= hi)
        d = dl[m]
        # perfect direction, full limit, gross of costs
        gross = np.abs(LIMIT[None, :] * rets[m])[d].sum() / m.sum()
        print(f"{wn:>16} {int(d.sum()):10d} {d.sum()/m.sum():6.2f}"
              f" {gross:12.0f}  {0.0:14.0f}")

    # ---- full accounting: baseline / oracle / coin flip ----
    print("\n" + "=" * 74)
    print("SCORE CEILING (full eval accounting, incl. commission & turnover)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    print(f"{'window':>16} {'baseline':>10} {'ORACLE':>10} {'coinflip':>18}"
          f" {'sub-thr ridge':>14}")
    results = {}
    for wn, (lo, hi) in WINDOWS.items():
        pnl_b = run_accounting(prices, days, base, lo, hi)
        s_b = score(pnl_b.mean(), pnl_b.std())

        d_or = apply_override(base, dl, truth)
        pnl_o = run_accounting(prices, days, d_or, lo, hi)
        s_o = score(pnl_o.mean(), pnl_o.std())

        cf = []
        for seed in range(8):
            r = np.random.default_rng(seed)
            rd = np.where(r.random(base.shape) < 0.5, -1.0, 1.0)
            pnl_c = run_accounting(prices, days, apply_override(base, dl, rd), lo, hi)
            cf.append(score(pnl_c.mean(), pnl_c.std()))
        s_c, s_c_sd = float(np.mean(cf)), float(np.std(cf))

        rs = np.where(c["ridge_sign"] != 0, c["ridge_sign"],
                      np.where(rng.random(base.shape) < 0.5, -1.0, 1.0))
        pnl_r = run_accounting(prices, days, apply_override(base, dl, rs), lo, hi)
        s_r = score(pnl_r.mean(), pnl_r.std())

        results[wn] = (s_b, s_o, s_c, s_c_sd, s_r)
        print(f"{wn:>16} {s_b:10.1f} {s_o:10.1f} {s_c:11.1f}+-{s_c_sd:4.1f}"
              f" {s_r:14.1f}")

    print("\nprize = ORACLE - baseline (perfect foresight on deadlocks only)")
    for wn, (s_b, s_o, s_c, s_sd, s_r) in results.items():
        print(f"  {wn}: oracle +{s_o-s_b:6.1f} | coinflip {s_c-s_b:+6.1f}"
              f" | sub-threshold ridge {s_r-s_b:+6.1f}")

    # ---- break-even accuracy ----
    print("\n" + "=" * 74)
    print("BREAK-EVEN ACCURACY -- what hit rate beats staying flat?")
    print("=" * 74)
    print(f"{'window':>16} " + " ".join(f"{p:>7.0%}" for p in
                                        (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)))
    for wn, (lo, hi) in WINDOWS.items():
        row = []
        for p in (0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
            reps = []
            for seed in range(6):
                r = np.random.default_rng(1000 + seed)
                correct = r.random(base.shape) < p
                dirn = np.where(correct, truth, -truth)
                pnl_p = run_accounting(prices, days,
                                       apply_override(base, dl, dirn), lo, hi)
                reps.append(score(pnl_p.mean(), pnl_p.std()))
            row.append(float(np.mean(reps)))
        base_s = results[wn][0]
        print(f"{wn:>16} " + " ".join(f"{v-base_s:+7.1f}" for v in row))
    print("\n(cells are score DELTA vs staying flat; the accuracy where a row")
    print(" turns positive is the bar any predictor has to clear)")

    out = ROOT / "analysis_outputs" / "deadlock"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "deadlock_cache.npz", days=days, dollars=base,
                        deadlock=dl, rets=rets, ridge_sign=c["ridge_sign"],
                        ridge_strength=c["ridge_strength"],
                        pair_contributors=c["pair_contributors"])
    print(f"\nwrote {out / 'deadlock_cache.npz'}")


if __name__ == "__main__":
    main()
