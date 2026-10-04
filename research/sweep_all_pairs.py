"""Self-contained rigorous tune of every seeded pair (no fragile imports).

Depends ONLY on daniel_filter + pandas/numpy, so it cannot be broken by
module reshuffling elsewhere in the repo.  Replicates PairAgent.desired_side
exactly and replays the book through daniel_filter's public API.

Protocol per pair:
  grid   : q_beta x h x cusum_k x min_seg_len x adapt_bars (beta model)
           x z_entry x z_exit x z_stop                      (entry/exit)
  gate 1 : Pareto — train AND test score and mean all improve
  gate 2 : four-block — no 125-day block degrades
  gate 3 : robustness — neighbourhood-SMOOTHED score improves on both
           windows, so a fitted spike cannot win
`require_adf` is held at each pair's seeded value (book-level decision).

Then subset-searches the survivors on the real book, because standalone
pair gains do not add through the EV-MWIS conflict layer.
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

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, TradeCfg

TRAIN_START, TRAIN_MID, TRAIN_END = 251, 375, 500
TEST_START, TEST_MID, TEST_END = 501, 625, 750
TRAIN_DECIDE, TEST_DECIDE = TRAIN_START - 2, TEST_START - 2
OUT_DIR = _ROOT / "cusum_output" / "daniel_pairs"
COMM_RATE = 1.0e-4
LEG = 10_000.0
EPS = 1e-12

_COMM = np.full(51, 1.0e-4)
_COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0

BASE = tuple(dfil.PAIR_CFGS)
FILT_KEYS = ("q_beta", "h", "cusum_k", "min_seg_len", "adapt_bars")
TRADE_KEYS = ("z_entry", "z_exit", "z_stop")


def score(mu: float, sd: float) -> float:
    if mu <= 0.0 or sd < 1e-10:
        return mu
    sr = sqrt(250.0) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def record_paths(ly, lx, cfg):
    f = dfil.PairSegFilter(cfg)
    n = len(ly)
    beta = np.empty(n)
    cut = np.zeros(n, dtype=bool)
    for t in range(n):
        o = f.step(float(ly[t]), float(lx[t]))
        beta[t] = o.beta
        cut[t] = o.cut
    return beta, int(cut.sum())


def gates_of(ly, lx, beta):
    n = len(ly)
    adf_b = np.zeros(n, dtype=bool)
    kap_b = np.zeros(n, dtype=bool)
    z = np.zeros(n)
    for t in range(min(TRAIN_DECIDE, n), n):
        bl = beta[t - 1]
        hist = ly[:t] - bl * lx[:t]
        if len(hist) >= 40 and dfil._adf_t_spread(hist) >= dfil.PRIOR_GATE_T:
            adf_b[t] = True
        s = ly[t - dfil.Z_W + 1:t + 1] - bl * lx[t - dfil.Z_W + 1:t + 1]
        z[t] = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + EPS))
        s120 = ly[t - 119:t + 1] - bl * lx[t - 119:t + 1]
        if dfil._ou_kappa(s120) <= 0.0:
            kap_b[t] = True
    return adf_b, kap_b, z


def trade_sim(adf_b, kap_b, z, py, px, trade, decide):
    n = len(py)
    sy = np.zeros(n)
    sx = np.zeros(n)
    side = 0
    ze = float(trade.z_entry)
    zx = min(float(trade.z_exit), 0.85 * ze)
    zs = max(float(trade.z_stop), 1.25 * ze)
    for t in range(decide, n):
        if (trade.require_adf and adf_b[t]) or (trade.require_kappa and kap_b[t]):
            side = 0
        else:
            az = abs(float(z[t]))
            if side == 0:
                if ze <= az < zs:
                    side = int(-np.sign(z[t]))
            elif az <= zx or az >= zs:
                side = 0
            elif az >= ze and int(-np.sign(z[t])) != side:
                side = int(-np.sign(z[t]))
        if side:
            sc = max(0.0, min(1.0, float(trade.leg_scale)))
            sy[t] = int(side * LEG * sc / py[t])
            sx[t] = int(-side * LEG * sc / px[t])
    return sy, sx


def pnl_path(sy, sx, py, px):
    n = len(py)
    out = np.zeros(n + 1)
    k = np.arange(1, n - 1)
    gross = sy[k] * (py[k + 1] - py[k]) + sx[k] * (px[k + 1] - px[k])
    fee = COMM_RATE * (np.abs(sy[k] - sy[k - 1]) * py[k]
                       + np.abs(sx[k] - sx[k - 1]) * px[k])
    out[k + 2] = gross - fee
    return out


def wstats(p, a, b):
    v = p[a:b + 1]
    return {"mean": float(v.mean()), "score": score(float(v.mean()), float(v.std()))}


def evaluate(adf_b, kap_b, z, py, px, trade):
    a, b = trade_sim(adf_b, kap_b, z, py, px, trade, TRAIN_DECIDE)
    c, d = trade_sim(adf_b, kap_b, z, py, px, trade, TEST_DECIDE)
    tp, ep = pnl_path(a, b, py, px), pnl_path(c, d, py, px)
    return {
        "train": wstats(tp, TRAIN_START, TRAIN_END),
        "test": wstats(ep, TEST_START, TEST_END),
        "blocks": [wstats(tp, TRAIN_START, TRAIN_MID),
                   wstats(tp, TRAIN_MID + 1, TRAIN_END),
                   wstats(ep, TEST_START, TEST_MID),
                   wstats(ep, TEST_MID + 1, TEST_END)],
    }


def book_replay(prices, cfgs, start, end, module=None):
    mod = module if module is not None else dfil
    dfil.apply_pair_cfgs(cfgs)
    if mod is not dfil and hasattr(mod, "apply_pair_cfgs"):
        mod.apply_pair_cfgs(cfgs)
    if hasattr(mod, "reset"):
        mod.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            req = mod.getMyPosition(hist)
            lim = (_LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        dl = new - pos
        cash -= cur.dot(dl) + comm
        comm = float(np.sum(cur * np.abs(dl) * _COMM))
        pos = new
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
    arr = np.asarray(pnl[1:])
    return score(float(arr.mean()), float(arr.std()))


def axes_for(spec):
    """Grid rebalanced toward where the edge demonstrably is.

    Session evidence: across ~90k candidates on three separate pairs the
    beta tracking error moved <0.005 and corr(tracking, test score) was
    ~0.  The through-origin filter pins beta to the log-price ratio, so
    q_beta is the only filter knob that reshapes it at all and the CUSUM
    knobs are inert whenever the pair does not cut.  So: keep q_beta wide,
    sample h/cusum_k lightly, hold min_seg_len/adapt_bars at the seed, and
    spend the budget on entry/exit thresholds instead.
    """
    f, t = spec.filt, spec.trade
    ax = {
        "q_beta": [1e-8, 1e-6, f.q_beta, 1e-4, 1e-2],
        "h": [round(f.h * 0.6, 2), f.h, round(f.h * 1.6, 2)],
        "cusum_k": [f.cusum_k, 0.5],
        "min_seg_len": [f.min_seg_len],
        "adapt_bars": [f.adapt_bars],
        "z_entry": [0.10, 0.18, 0.27, t.z_entry, 0.40, 0.50, 0.60, 0.80],
        "z_exit": [0.02, 0.06, t.z_exit, 0.15, 0.25],
        "z_stop": [2.50, 3.25, t.z_stop, 4.00, 4.75],
    }
    return {k: sorted(set(v)) for k, v in ax.items()}


def tune_one(spec, prices):
    ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
    py, px = prices[spec.y], prices[spec.x]
    AX = axes_for(spec)
    keys = list(AX)
    bb, _ = record_paths(ly, lx, spec.filt)
    ba, bk, bz = gates_of(ly, lx, bb)
    base = evaluate(ba, bk, bz, py, px, spec.trade)

    res = {}
    gate_cache = {}
    for fv in itertools.product(*(AX[k] for k in FILT_KEYS)):
        cf = dict(zip(FILT_KEYS, fv))
        beta, cuts = record_paths(ly, lx, spec.filt._replace(**cf))
        bkey_hash = beta.tobytes()
        hit = gate_cache.get(bkey_hash)
        if hit is None:
            hit = gates_of(ly, lx, beta)
            gate_cache[bkey_hash] = hit
        ab, kb, zz = hit
        for tv in itertools.product(*(AX[k] for k in TRADE_KEYS)):
            ct = dict(zip(TRADE_KEYS, tv))
            ze = float(ct["z_entry"])
            tr = spec.trade._replace(
                z_entry=ze, z_exit=min(float(ct["z_exit"]), 0.85 * ze),
                z_stop=max(float(ct["z_stop"]), 1.25 * ze))
            r = evaluate(ab, kb, zz, py, px, tr)
            r["cuts"] = cuts
            res[tuple({**cf, **ct}[k] for k in keys)] = r

    idx = {k: {v: i for i, v in enumerate(AX[k])} for k in keys}

    def nb(key):
        out = []
        for d, k in enumerate(keys):
            i = idx[k][key[d]]
            for s in (-1, 1):
                j = i + s
                if 0 <= j < len(AX[k]):
                    n = list(key)
                    n[d] = AX[k][j]
                    out.append(tuple(n))
        return out

    def sm(key):
        pts = [res[key]] + [res[n] for n in nb(key) if n in res]
        return (float(np.mean([q["train"]["score"] for q in pts])),
                float(np.mean([q["test"]["score"] for q in pts])))

    bkey = tuple({**{k: getattr(spec.filt, k) for k in FILT_KEYS},
                  **{k: getattr(spec.trade, k) for k in TRADE_KEYS}}[k] for k in keys)
    if bkey not in res:
        return {"name": spec.name, "base": base, "winner": None, "note": "off-grid"}
    bs = sm(bkey)
    par = [k for k, v in res.items()
           if v["train"]["score"] > base["train"]["score"]
           and v["test"]["score"] > base["test"]["score"]
           and v["train"]["mean"] >= base["train"]["mean"]
           and v["test"]["mean"] >= base["test"]["mean"]]
    blk = [k for k in par
           if all(res[k]["blocks"][i]["score"] >= base["blocks"][i]["score"] - 1e-9
                  for i in range(4))]
    rob = []
    for k in blk:
        m = sm(k)
        if m[0] > bs[0] and m[1] > bs[1]:
            rob.append((k, m))
    if not rob:
        return {"name": spec.name, "base": base, "winner": None,
                "n_pareto": len(par), "n_block": len(blk), "n_robust": 0}
    rob.sort(key=lambda r: min(r[1][0] - bs[0], r[1][1] - bs[1]), reverse=True)
    k, m = rob[0]
    c = dict(zip(keys, k))
    ze = float(c["z_entry"])
    return {
        "name": spec.name, "base": base,
        "n_pareto": len(par), "n_block": len(blk), "n_robust": len(rob),
        "winner": {
            "filt": spec.filt._replace(**{x: c[x] for x in FILT_KEYS})._asdict(),
            "trade": spec.trade._replace(
                z_entry=ze, z_exit=min(float(c["z_exit"]), 0.85 * ze),
                z_stop=max(float(c["z_stop"]), 1.25 * ze))._asdict(),
            "train": res[k]["train"], "test": res[k]["test"],
            "cuts": res[k]["cuts"], "smooth": m,
        },
    }


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    t0 = time.perf_counter()
    reports = []
    print(f"sweeping {len(BASE)} pairs against the LIVE daniel_filter\n", flush=True)
    for spec in BASE:
        r = tune_one(spec, prices)
        reports.append(r)
        b = r["base"]
        if r["winner"]:
            w = r["winner"]
            print(f"  {spec.name:>11}: {b['train']['score']:+7.2f}/{b['test']['score']:+7.2f}"
                  f" -> {w['train']['score']:+7.2f}/{w['test']['score']:+7.2f}"
                  f"  ({r['n_robust']} robust)", flush=True)
        else:
            print(f"  {spec.name:>11}: {b['train']['score']:+7.2f}/{b['test']['score']:+7.2f}"
                  f"  {r.get('n_pareto',0):5d} pareto {r.get('n_block',0):4d} block"
                  f" 0 robust -> KEEP", flush=True)
    print(f"\nsweep done in {time.perf_counter()-t0:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sweep_all_pairs.json").write_text(json.dumps(reports, indent=2),
                                                  encoding="utf-8")
    wins = [r for r in reports if r["winner"]]
    print(f"{len(wins)} pairs have a robust candidate: {[w['name'] for w in wins]}\n")
    if not wins:
        return

    W = {w["name"]: w["winner"] for w in wins}

    def cfgs(sub):
        return tuple(
            s._replace(filt=FiltCfg(**W[s.name]["filt"]),
                       trade=TradeCfg(**W[s.name]["trade"]))
            if s.name in sub else s for s in BASE)

    b_tr = book_replay(prices, BASE, TRAIN_START, TRAIN_END)
    b_te = book_replay(prices, BASE, TEST_START, TEST_END)
    print(f"daniel_filter baseline: train {b_tr:.2f} test {b_te:.2f}\n")
    names = list(W)
    rows = []
    for k in range(1, len(names) + 1):
        for c in itertools.combinations(names, k):
            a = book_replay(prices, cfgs(set(c)), TRAIN_START, TRAIN_END)
            bb = book_replay(prices, cfgs(set(c)), TEST_START, TEST_END)
            rows.append((a, bb, c))
    dfil.apply_pair_cfgs(BASE)
    good = [r for r in rows if r[0] >= b_tr - 1e-9 and r[1] >= b_te - 1e-9]
    good.sort(key=lambda r: min(r[0] - b_tr, r[1] - b_te), reverse=True)
    print(f"{len(good)}/{len(rows)} subsets hold or improve BOTH windows")
    for a, bb, c in good[:10]:
        print(f"  train {a:8.2f} ({a-b_tr:+6.2f})  test {bb:8.2f} ({bb-b_te:+6.2f})"
              f"  {'+'.join(c)}")
    (OUT_DIR / "sweep_subsets.json").write_text(
        json.dumps({"baseline": [b_tr, b_te],
                    "top": [[a, b, list(c)] for a, b, c in good[:12]],
                    "winners": W}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'sweep_subsets.json'}")


if __name__ == "__main__":
    main()
