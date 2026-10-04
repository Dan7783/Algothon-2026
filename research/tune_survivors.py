"""Step 3: tune the 14 surviving pairs, selecting on TRAIN BLOCKS ONLY.

The discipline here is the point.  Everything that touches selection uses
only days 251-500, split into two 125-day blocks:

  * a candidate must improve BOTH train blocks (not their average — a
    candidate that wins one and loses the other is noise);
  * it must also improve the neighbourhood-SMOOTHED train score, so an
    isolated grid spike cannot win;
  * survivors are then confirmed at PORTFOLIO level on the train blocks,
    because per-pair gains do not add through the conflict layer.

Days 501-750 are not read until the very end, and only to report what the
train-selected configuration did.  That number is a verdict on the method,
not an input to it: true OOS came in at 599 against an 878 test score
precisely because that window had been used for selection.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import itertools
import json
import time
from math import sqrt

import numpy as np
import pandas as pd

import four_sigmas as fs

TR1, TR2 = (251, 375), (376, 500)
TE1, TE2 = (501, 625), (626, 750)
DECIDE = {TR1: 249, TR2: 374, TE1: 499, TE2: 624}
COMM_RATE = 1e-4
LEG = 10_000.0
_COMM = np.full(51, 1e-4)
_COMM[0] = 2e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0
BASE = tuple(fs.PAIR_CFGS)
ACTIVE = [s for s in BASE if s.trade.leg_scale > 0]


def score(m, s):
    if m <= 0 or s < 1e-10:
        return m
    r = sqrt(250.0) * m / s
    return m * r * r / (r * r + 1.0)


# ---------- per-pair replica of the live agent ----------
def beta_path(ly, lx, filt):
    f = fs.PairSegFilter(filt)
    out = np.empty(len(ly))
    for t in range(len(ly)):
        out[t] = f.step(float(ly[t]), float(lx[t])).beta
    return out


def gates(ly, lx, beta, use_adf):
    n = len(ly)
    blocked = np.zeros(n, dtype=bool)
    z = np.zeros(n)
    for t in range(249, n):
        b = beta[t - 1]
        if use_adf:
            lo = max(0, t - 1 - fs.ADF_WINDOW)
            if fs._adf_t_spread(ly[lo:t] - b * lx[lo:t]) >= fs.PRIOR_GATE_T:
                blocked[t] = True
                continue
        if fs._ou_kappa(ly[t - 119:t + 1] - b * lx[t - 119:t + 1]) <= 0.0:
            blocked[t] = True
            continue
        s = ly[t - fs.Z_W + 1:t + 1] - b * lx[t - fs.Z_W + 1:t + 1]
        z[t] = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12))
    return blocked, z


def pair_pnl(blocked, z, py, px, trade, window):
    """Shares from a fresh state at the window's decide day, then PnL."""
    n = len(py)
    sy = np.zeros(n)
    sx = np.zeros(n)
    side = 0
    ze = float(trade.z_entry)
    zx = min(float(trade.z_exit), 0.85 * ze)
    zs = max(float(trade.z_stop), 1.25 * ze)
    for t in range(DECIDE[window], n):
        if blocked[t]:
            side = 0
        else:
            a = abs(float(z[t]))
            if side == 0:
                if ze <= a < zs:
                    side = -int(np.sign(z[t]))
            elif a <= zx or a >= zs:
                side = 0
            elif a >= ze and -int(np.sign(z[t])) != side:
                side = -int(np.sign(z[t]))
        if side:
            sy[t] = int(side * LEG / py[t])
            sx[t] = int(-side * LEG / px[t])
    k = np.arange(1, n - 1)
    gross = sy[k] * (py[k + 1] - py[k]) + sx[k] * (px[k + 1] - px[k])
    fee = COMM_RATE * (np.abs(sy[k] - sy[k - 1]) * py[k]
                       + np.abs(sx[k] - sx[k - 1]) * px[k])
    pnl = np.zeros(n + 1)
    pnl[k + 2] = gross - fee
    a, b = window
    seg = pnl[a:b + 1]
    return score(float(seg.mean()), float(seg.std()))


def axes(spec):
    f, t = spec.filt, spec.trade
    return {
        "q_beta": sorted({f.q_beta / 3, f.q_beta, f.q_beta * 3}),
        "cusum_k": sorted({0.5, f.cusum_k}),
        "h": sorted({round(f.h * 0.75, 2), f.h, round(f.h * 1.33, 2)}),
        "z_entry": sorted({round(t.z_entry * 0.75, 4), t.z_entry,
                           round(t.z_entry * 1.3, 4)}),
        "z_exit": sorted({round(t.z_exit * 0.5, 4), t.z_exit,
                          round(t.z_exit * 2.0, 4)}),
        "z_stop": sorted({round(t.z_stop * 0.85, 3), t.z_stop,
                          round(t.z_stop * 1.15, 3)}),
    }


FK = ("q_beta", "cusum_k", "h")
TK = ("z_entry", "z_exit", "z_stop")


def tune_pair(spec, prices):
    ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
    py, px = prices[spec.y], prices[spec.x]
    AX = axes(spec)
    keys = list(AX)
    res = {}
    for fv in itertools.product(*(AX[k] for k in FK)):
        cf = dict(zip(FK, fv))
        b = beta_path(ly, lx, spec.filt._replace(**cf))
        blocked, z = gates(ly, lx, b, spec.trade.require_adf)
        for tv in itertools.product(*(AX[k] for k in TK)):
            ct = dict(zip(TK, tv))
            tr = spec.trade._replace(**ct)
            res[tuple({**cf, **ct}[k] for k in keys)] = (
                pair_pnl(blocked, z, py, px, tr, TR1),
                pair_pnl(blocked, z, py, px, tr, TR2),
            )
    bkey = tuple({**{k: getattr(spec.filt, k) for k in FK},
                  **{k: getattr(spec.trade, k) for k in TK}}[k] for k in keys)
    if bkey not in res:
        return None
    b1, b2 = res[bkey]
    idx = {k: {v: i for i, v in enumerate(AX[k])} for k in keys}

    def nb(key):
        out = []
        for d, k in enumerate(keys):
            i = idx[k][key[d]]
            for st in (-1, 1):
                j = i + st
                if 0 <= j < len(AX[k]):
                    n2 = list(key)
                    n2[d] = AX[k][j]
                    out.append(tuple(n2))
        return out

    def smooth(key):
        pts = [res[key]] + [res[n] for n in nb(key) if n in res]
        return (float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])))

    s0 = smooth(bkey)
    cands = []
    for k, (a1, a2) in res.items():
        if a1 <= b1 or a2 <= b2:          # must improve BOTH train blocks
            continue
        m = smooth(k)
        if m[0] > s0[0] and m[1] > s0[1]:  # and the smoothed landscape too
            cands.append((min(m[0] - s0[0], m[1] - s0[1]), k, a1, a2))
    if not cands:
        return {"name": spec.name, "base": (b1, b2), "winner": None,
                "n": len(res)}
    cands.sort(reverse=True)
    _, k, a1, a2 = cands[0]
    c = dict(zip(keys, k))
    return {"name": spec.name, "base": (b1, b2), "cand": (a1, a2),
            "filt": {x: c[x] for x in FK}, "trade": {x: c[x] for x in TK},
            "winner": True, "n": len(res), "n_pass": len(cands)}


def book_blocks(prices, cfgs):
    fs.apply_pair_cfgs(cfgs)
    fs.reset()
    cash = val = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(250, 751):
        h = prices[:, :t]
        cur = h[:, -1]
        if t < 750:
            req = fs.getMyPosition(h)
            lim = (_LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * _COMM))
        pos = new
        v = cash + pos.dot(cur)
        pnl.append(v - val)
        val = v
    a = np.asarray(pnl[1:])
    return [score(float(a[i:i + 125].mean()), float(a[i:i + 125].std()))
            for i in (0, 125, 250, 375)]


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    t0 = time.perf_counter()
    print(f"tuning {len(ACTIVE)} surviving pairs on TRAIN BLOCKS ONLY\n")
    wins = {}
    for spec in ACTIVE:
        r = tune_pair(spec, prices)
        if r is None:
            print(f"  {spec.name:>11}: baseline off-grid, skipped", flush=True)
            continue
        if r["winner"]:
            wins[spec.name] = r
            print(f"  {spec.name:>11}: train {r['base'][0]:6.1f}/{r['base'][1]:6.1f}"
                  f" -> {r['cand'][0]:6.1f}/{r['cand'][1]:6.1f}"
                  f"  ({r['n_pass']} of {r['n']} pass)", flush=True)
        else:
            print(f"  {spec.name:>11}: train {r['base'][0]:6.1f}/{r['base'][1]:6.1f}"
                  f"  0 of {r['n']} pass -> KEEP", flush=True)
    print(f"\nper-pair sweep done in {time.perf_counter()-t0:.0f}s; "
          f"{len(wins)} pairs have a train-validated candidate\n")

    b0 = book_blocks(prices, BASE)
    print(f"baseline blocks: {[round(x) for x in b0]}  "
          f"(train {b0[0]:.1f}/{b0[1]:.1f})")
    if not wins:
        print("nothing to apply")
        return

    def with_names(names):
        return tuple(
            s._replace(filt=s.filt._replace(**wins[s.name]["filt"]),
                       trade=s.trade._replace(**wins[s.name]["trade"]))
            if s.name in names else s for s in BASE)

    print("\nportfolio confirmation on TRAIN blocks (greedy, test untouched):")
    chosen: set[str] = set()
    cur = (b0[0], b0[1])
    pool = list(wins)
    while pool:
        best = None
        for n in pool:
            bb = book_blocks(prices, with_names(chosen | {n}))
            if bb[0] >= cur[0] - 1e-9 and bb[1] >= cur[1] - 1e-9:
                g = min(bb[0] - cur[0], bb[1] - cur[1])
                if best is None or g > best[0]:
                    best = (g, n, bb)
        if best is None:
            break
        _, n, bb = best
        chosen.add(n)
        pool.remove(n)
        cur = (bb[0], bb[1])
        print(f"  + {n:>11} -> train {bb[0]:7.1f}/{bb[1]:7.1f}", flush=True)

    print(f"\nselected on train: {sorted(chosen) or 'NONE'}")
    final = book_blocks(prices, with_names(chosen)) if chosen else b0
    fs.apply_pair_cfgs(BASE)
    print("\n--- now, and only now, the held-out blocks ---")
    print(f"  baseline: train {b0[0]:7.1f}/{b0[1]:7.1f}   HELD-OUT {b0[2]:7.1f}/{b0[3]:7.1f}")
    print(f"  tuned   : train {final[0]:7.1f}/{final[1]:7.1f}   HELD-OUT {final[2]:7.1f}/{final[3]:7.1f}")
    dh = (final[2] + final[3]) / 2 - (b0[2] + b0[3]) / 2
    dt = (final[0] + final[1]) / 2 - (b0[0] + b0[1]) / 2
    print(f"\n  train gain {dt:+.1f}   held-out gain {dh:+.1f}")
    print("  -> per-pair tuning generalises" if dh > 0 else
          "  -> per-pair tuning did NOT generalise: train gain is noise")

    (_ROOT / "cusum_output" / "tune_survivors.json").write_text(
        json.dumps({"winners": wins, "chosen": sorted(chosen),
                    "baseline_blocks": b0, "tuned_blocks": final}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
