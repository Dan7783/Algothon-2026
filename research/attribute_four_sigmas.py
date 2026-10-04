"""Per-sleeve PnL attribution for four_sigmas.py.

The live book scored ~600 on the organiser's held-out 751-1000. We can't see that window,
but we CAN find which SIGNAL is fragile by decomposing the book into its sleeves
(atlas pairs / extra lead-lag / ridge / discovery) and scoring each one per 125-day block on
the data we have. The last block 626-750 is the freshest = best proxy for 751-1000; a sleeve
that is strong early and dead/negative there is the drag.

Method: replay the REAL stateful book once (four_sigmas.sleeve_breakdown, called once per day
exactly as getMyPosition does), capture each sleeve's dollar targets, then run eval.py's exact
accounting on each sleeve standalone and on the full book, per block.

Run from repo root:  python research/attribute_four_sigmas.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import four_sigmas as fs

prcAll = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
N, T = prcAll.shape
LEG = fs.LEG

COMM = np.full(N, 0.0001); COMM[0] = 0.00002
DLIM = np.full(N, 10_000.0); DLIM[0] = 100_000.0


def eval_score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def dollars_to_shares(dollars, last_return, price_col):
    """Replicate getMyPosition's crash gate + clip + integerisation for one sleeve."""
    d = dollars.copy()
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    d[crash] = 0.0
    d[1:] = np.clip(d[1:], -LEG, LEG)
    d[0] = 0.0
    return (d / price_col).astype(int)


# ---- one faithful replay, capturing every sleeve's target shares per day ----
fs.reset()
SLEEVES = ["atlas", "extraLL", "ridge", "discovery", "FULL"]
shares = {k: {} for k in SLEEVES}      # sleeve -> {day t : int share vector}

for t in range(130, T + 1):
    prc = prcAll[:, :t]
    sl = fs.sleeve_breakdown(prc)      # called once/day == getMyPosition cadence
    if sl is None:
        continue
    last_ret = np.diff(np.log(prc[:, -2:]), axis=1)[:, 0]
    price_col = prc[:, -1]
    full_dollars = np.zeros(N)
    for name in ["atlas", "extraLL", "ridge", "discovery"]:
        shares[name][t] = dollars_to_shares(sl[name], last_ret, price_col)
        full_dollars += sl[name]
    shares["FULL"][t] = dollars_to_shares(full_dollars, last_ret, price_col)

print(f"replayed {len(shares['FULL'])} decision days")


def account(share_map, start, end):
    """eval.py calcPL for a captured share series over [start, end]; starts flat."""
    cash = value = 0.0
    comm = 0.0
    pos = np.zeros(N)
    pll = []
    for t in range(start - 1, end + 1):
        cur = prcAll[:, t - 1]                      # price at 'day t'
        if t < end and t in share_map:
            lim = (DLIM / cur).astype(int)
            newPos = np.clip(share_map[t], -lim, lim).astype(int)
        else:
            newPos = pos
        dpos = newPos - pos
        cash -= cur.dot(dpos) + comm
        dvol = cur * np.abs(dpos)
        comm = float(np.sum(dvol * COMM))
        pos = newPos
        posval = pos.dot(cur)
        today = cash + posval - value
        value = cash + posval
        if t >= start:
            pll.append(today)
    return np.asarray(pll)


BLOCKS = [("train1", 251, 375), ("train2", 376, 500),
          ("test1 ", 501, 625), ("test2 ", 626, 750)]

print(f"\n{'sleeve':10s} | " + " ".join(f"{b[0]:>8s}" for b in BLOCKS) +
      f" | {'floor':>8s} {'full501-750':>11s}")
print("-" * 78)
summary = {}
for name in SLEEVES:
    row = []
    for _, s, e in BLOCKS:
        pll = account(shares[name], s, e)
        row.append(eval_score(pll.mean(), pll.std()))
    full_test = account(shares[name], 501, 750)
    summary[name] = row
    print(f"{name:10s} | " + " ".join(f"{v:8.1f}" for v in row) +
          f" | {min(row):8.1f} {eval_score(full_test.mean(), full_test.std()):11.1f}")

print("\nmean daily PnL ($) per sleeve per block:")
print(f"{'sleeve':10s} | " + " ".join(f"{b[0]:>8s}" for b in BLOCKS))
for name in SLEEVES:
    row = [account(shares[name], s, e).mean() for _, s, e in BLOCKS]
    print(f"{name:10s} | " + " ".join(f"{v:8.1f}" for v in row))

# ---- decisive test: is each sleeve REGIME-CONDITIONAL? ----------------------
# classify each day by the current momentum vs reversal regime (causal: rolling
# 60d lag-1 autocorr of the market return, sign>0 = momentum).
LP = np.log(prcAll)
Rm = np.diff(LP[1:], axis=1).mean(axis=0)          # market return, index day 2..750
regime = {}                                        # day t -> +1 momentum / -1 reversal
for t in range(251, 751):
    w = Rm[t - 60 - 1:t - 1]                        # trailing 60 returns, causal (< day t)
    ac = np.corrcoef(w[:-1], w[1:])[0, 1] if w.std() > 0 else 0.0
    regime[t] = 1 if ac > 0 else -1

print("\nmean daily PnL ($) conditioned on the current market regime (days 251-750):")
print(f"{'sleeve':10s} | {'momentum':>10s} {'reversal':>10s}   {'mom - rev':>10s}  regime-fragile?")
for name in SLEEVES:
    pll = account(shares[name], 251, 750)          # daily PnL, day t=251..750
    days = np.arange(251, 751)
    rg = np.array([regime[t] for t in days])
    mom = pll[rg == 1].mean() if (rg == 1).any() else float("nan")
    rev = pll[rg == -1].mean() if (rg == -1).any() else float("nan")
    flag = "<== YES" if abs(mom - rev) > 0.5 * abs(pll.mean()) and name != "FULL" else ""
    print(f"{name:10s} | {mom:10.1f} {rev:10.1f}   {mom - rev:10.1f}  {flag}")
n_mom = sum(1 for t in regime if regime[t] == 1)
print(f"\n(days: {n_mom} momentum / {500 - n_mom} reversal in 251-750)")
