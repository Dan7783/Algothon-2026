"""Stage 2: with q_beta fixed from stage 1, tune the CUSUM to DETECT breaks.

Criterion is detection quality, not PnL:

  * "true" breaks are defined independently of the filter — a sustained shift
    in the trailing 60-day OLS hedge ratio larger than BREAK_MULT robust
    standard deviations of its ordinary day-to-day change;
  * a cut COUNTS as a detection if it lands within TOL days of such a break;
  * everything else is a false alarm.

Reported per (cusum_k, h): detections, misses, false alarms, median delay,
and the false-alarm rate per 250 trading days.  A sane operating point finds
most breaks with a false-alarm rate low enough that the book is not being
reset on noise — that is a design choice about the detector, and it can be
made without ever looking at a trading score.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json

import numpy as np
import pandas as pd

import four_sigmas as fs

TRAIN_END = 500
Q_STAGE1 = 1e-4
OLS_W = 60
BREAK_MULT = 4.0     # a "real" break: step > 4 robust sd of normal beta moves
TOL = 20             # a cut within 20 days of a break counts as detecting it


def rolling_ols(ly, lx, w=OLS_W):
    n = len(ly)
    out = np.full(n, np.nan)
    for t in range(w, n):
        a, b = lx[t - w:t], ly[t - w:t]
        d = float(a @ a)
        if d > 1e-12:
            out[t] = float(a @ b) / d
    return out


def true_breaks(beta_ref):
    """Days where the reference hedge ratio takes a sustained large step."""
    d = np.diff(beta_ref)
    ok = np.isfinite(d)
    if ok.sum() < 30:
        return []
    mad = float(np.median(np.abs(d[ok] - np.median(d[ok])))) * 1.4826
    if mad <= 0:
        return []
    idx = np.where(np.abs(d) > BREAK_MULT * mad)[0] + 1
    # collapse runs into one break each
    breaks = []
    for i in idx:
        if not breaks or i - breaks[-1] > TOL:
            breaks.append(int(i))
    return breaks


def cuts_for(ly, lx, k, h):
    f = fs.PairSegFilter(fs.FiltCfg(q_beta=Q_STAGE1, cusum_k=k, h=h))
    out = []
    for t in range(len(ly)):
        if f.step(float(ly[t]), float(lx[t])).cut:
            out.append(t)
    return out


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    lp = np.log(prices)
    pairs = [(s.name, s.y, s.x) for s in fs.PAIR_CFGS]

    refs, breaks = {}, {}
    for name, y, x in pairs:
        ly, lx = lp[y, :TRAIN_END], lp[x, :TRAIN_END]
        r = rolling_ols(ly, lx)
        refs[name] = r
        breaks[name] = true_breaks(r)
    n_breaks = sum(len(v) for v in breaks.values())
    print(f"Stage 2 — CUSUM detection tuning (q_beta fixed at {Q_STAGE1:.0e})")
    print(f"reference breaks found on TRAIN: {n_breaks} across {len(pairs)} pairs")
    print(f"(a break = trailing-{OLS_W}d OLS beta step > {BREAK_MULT} robust sd)\n")
    print(f"{'k':>5} {'h':>6} | {'detect':>7} {'miss':>5} {'false':>6} | "
          f"{'delay':>6} | {'FA/250d':>8}")
    print("-" * 56)

    rows = []
    for k in (0.25, 0.5, 0.75, 1.0):
        for h in (4.0, 6.0, 8.0, 12.0, 20.0):
            det = miss = false = 0
            delays = []
            for name, y, x in pairs:
                ly, lx = lp[y, :TRAIN_END], lp[x, :TRAIN_END]
                cuts = cuts_for(ly, lx, k, h)
                bset = breaks[name]
                used = set()
                for b in bset:
                    hit = [c for c in cuts if 0 <= c - b <= TOL and c not in used]
                    if hit:
                        det += 1
                        used.add(hit[0])
                        delays.append(hit[0] - b)
                    else:
                        miss += 1
                false += len([c for c in cuts if c not in used])
            days = len(pairs) * (TRAIN_END - OLS_W)
            fa = 250.0 * false / days
            rows.append({"k": k, "h": h, "det": det, "miss": miss,
                         "false": false, "fa_per250": fa,
                         "delay": float(np.median(delays)) if delays else None})
            dl = f"{np.median(delays):6.1f}" if delays else "     -"
            print(f"{k:5.2f} {h:6.1f} | {det:7d} {miss:5d} {false:6d} | {dl} | "
                  f"{fa:8.2f}")

    out = _ROOT / "cusum_output" / "stage2_cusum.json"
    out.write_text(json.dumps({"n_breaks": n_breaks, "grid": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nPick the operating point: highest detection at a false-alarm rate")
    print("you are willing to pay for, then stage 3 sets how fast it re-adapts.")


if __name__ == "__main__":
    main()
