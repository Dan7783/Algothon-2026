"""Rigorous tune of EVERY seeded pair, then portfolio-confirm the winners.

Per pair (same protocol proved out on AENO/NWIG, NAYO/SRTX, MDGI/HTRK):

  grid   : q_beta x h x cusum_k x min_seg_len x adapt_bars   (beta model)
           x z_entry x z_exit x z_stop                        (entry/exit)
  gate 1 : Pareto — train AND test score and mean all improve
  gate 2 : four-block — no 125-day block degrades
  gate 3 : robustness — the neighbourhood-SMOOTHED score must improve on
           both windows, so a fitted spike cannot win
  report : beta tracking error, so beta modelling is visible per pair

`require_adf` is held at each pair's seeded value: turning it off is a
book-level capacity decision (see adf_gate_necessity.md), not something a
single pair may optimise for itself.

Then, because standalone gains do NOT add (EV-MWIS conflict layer), the
survivors are portfolio-confirmed: each alone, then greedy forward
selection keeping only what improves BOTH windows of the daniel core, and
finally a robust_hybrid check of the chosen set.  Nothing is written to
daniel_filter.py; the selected seeds are printed and saved.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import itertools
import json
import time

import numpy as np
import pandas as pd

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, TradeCfg
from tune_daniel_pairs import (
    OUT_DIR,
    TEST_DECIDE_START,
    TEST_END,
    TEST_MID,
    TEST_START,
    TRAIN_DECIDE_START,
    TRAIN_END,
    TRAIN_MID,
    TRAIN_START,
    gate_and_z_paths,
    pnl_path,
    record_paths,
    trade_sim,
    window_stats,
)

_COMM = np.full(51, 1.0e-4); _COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0); _LIMIT[0] = 100_000.0


def _score(mu, sd):
    if mu <= 0.0 or sd < 1e-10:
        return mu
    sr = (250.0 ** 0.5) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def portfolio_window(prices, cfgs, start, end, module=None):
    """eval.py-faithful replay of the whole book with cfgs applied."""
    mod = module if module is not None else dfil
    dfil.apply_pair_cfgs(cfgs)
    if hasattr(mod, 'apply_pair_cfgs') and mod is not dfil:
        mod.apply_pair_cfgs(cfgs)
    if hasattr(mod, 'reset'):
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
        d = new - pos
        cash -= cur.dot(d) + comm
        comm = float(np.sum(cur * np.abs(d) * _COMM))
        pos = new
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
    a = np.asarray(pnl[1:])
    return {'score': _score(float(a.mean()), float(a.std()))}


BASE = tuple(dfil.PAIR_CFGS)
FILT_KEYS = ("q_beta", "h", "cusum_k", "min_seg_len", "adapt_bars")
TRADE_KEYS = ("z_entry", "z_exit", "z_stop")


def load_raw_prices(path: str = 'prices.txt') -> np.ndarray:
    return pd.read_csv(path, sep=r'\s+', header=0).to_numpy(float).T


def axes_for(spec) -> dict:
    """Grid centred on the pair's own seed, so the baseline is a member."""
    f, t = spec.filt, spec.trade
    ax = {
        "q_beta": [1e-8, 1e-7, 1e-6, f.q_beta, 1e-5, 1e-4, 1e-3],
        "h": [3.0, 5.0, f.h, 8.0, 11.0],
        "cusum_k": [0.35, 0.5, f.cusum_k, 0.65],
        "min_seg_len": [20, 40, f.min_seg_len, 60],
        "adapt_bars": [4, 8, f.adapt_bars, 16],
        "z_entry": [0.10, 0.18, 0.27, t.z_entry, 0.40, 0.60, 0.80],
        "z_exit": [0.02, 0.06, t.z_exit, 0.15],
        "z_stop": [2.50, 3.25, t.z_stop, 4.25, 4.75],
    }
    return {k: sorted(set(v)) for k, v in ax.items()}


def rolling_ols(ly, lx, w=120):
    T = len(ly)
    out = np.full(T, np.nan)
    for i in range(w, T):
        a, b = lx[i - w:i], ly[i - w:i]
        ac, bc = a - a.mean(), b - b.mean()
        d = float(ac @ ac)
        if d > 1e-12:
            out[i] = float(ac @ bc) / d
    return out


def evaluate(gates, py, px, trade) -> dict:
    a, b = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    c, d = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tp, ep = pnl_path(a, b, py, px), pnl_path(c, d, py, px)
    return {
        "train": window_stats(tp, TRAIN_START, TRAIN_END),
        "test": window_stats(ep, TEST_START, TEST_END),
        "blocks": [
            window_stats(tp, TRAIN_START, TRAIN_MID),
            window_stats(tp, TRAIN_MID + 1, TRAIN_END),
            window_stats(ep, TEST_START, TEST_MID),
            window_stats(ep, TEST_MID + 1, TEST_END),
        ],
        "d_train": int((np.abs(a[TRAIN_START - 1:TRAIN_END]) > 0).sum()),
        "d_test": int((np.abs(c[TEST_START - 1:TEST_END]) > 0).sum()),
    }


def tune_one(spec, prices) -> dict | None:
    ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])
    py, px = prices[spec.y], prices[spec.x]
    ols = rolling_ols(ly, lx)
    tsl = slice(TEST_START - 1, TEST_END)
    AX = axes_for(spec)
    keys = list(AX)

    bp = record_paths(ly, lx, spec.filt)
    bg = gate_and_z_paths(ly, lx, bp["beta"])
    base = evaluate(bg, py, px, spec.trade)
    base_track = float(np.nanmean(np.abs(bp["beta"][tsl] - ols[tsl])))

    results: dict[tuple, dict] = {}
    for fv in itertools.product(*(AX[k] for k in FILT_KEYS)):
        cf = dict(zip(FILT_KEYS, fv))
        p = record_paths(ly, lx, spec.filt._replace(**cf))
        g = gate_and_z_paths(ly, lx, p["beta"])
        track = float(np.nanmean(np.abs(p["beta"][tsl] - ols[tsl])))
        cuts = int(p["cut"].sum())
        for tv in itertools.product(*(AX[k] for k in TRADE_KEYS)):
            ct = dict(zip(TRADE_KEYS, tv))
            ze = float(ct["z_entry"])
            tr = spec.trade._replace(
                z_entry=ze,
                z_exit=min(float(ct["z_exit"]), 0.85 * ze),
                z_stop=max(float(ct["z_stop"]), 1.25 * ze),
            )
            r = evaluate(g, py, px, tr)
            r["track"] = track
            r["cuts"] = cuts
            results[tuple({**cf, **ct}[k] for k in keys)] = r

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
        pts = [results[key]] + [results[n] for n in nb(key) if n in results]
        return (float(np.mean([q["train"]["score"] for q in pts])),
                float(np.mean([q["test"]["score"] for q in pts])))

    bkey = tuple(
        {**{k: getattr(spec.filt, k) for k in FILT_KEYS},
         **{k: getattr(spec.trade, k) for k in TRADE_KEYS}}[k] for k in keys
    )
    if bkey not in results:
        return None
    bs = sm(bkey)

    par = {k: v for k, v in results.items()
           if v["train"]["score"] > base["train"]["score"]
           and v["test"]["score"] > base["test"]["score"]
           and v["train"]["mean"] >= base["train"]["mean"]
           and v["test"]["mean"] >= base["test"]["mean"]}
    blk = {k: v for k, v in par.items()
           if all(v["blocks"][i]["score"] >= base["blocks"][i]["score"] - 1e-9
                  for i in range(4))}
    rob = []
    for k, v in blk.items():
        m = sm(k)
        if m[0] > bs[0] and m[1] > bs[1]:
            rob.append((k, v, m))
    if not rob:
        return {"name": spec.name, "n_grid": len(results), "n_pareto": len(par),
                "n_block": len(blk), "n_robust": 0, "base": base,
                "base_track": base_track, "winner": None}
    rob.sort(key=lambda r: min(r[2][0] - bs[0], r[2][1] - bs[1]), reverse=True)
    k, v, m = rob[0]
    c = dict(zip(keys, k))
    ze = float(c["z_entry"])
    return {
        "name": spec.name, "pair": [spec.y, spec.x],
        "n_grid": len(results), "n_pareto": len(par), "n_block": len(blk),
        "n_robust": len(rob), "base": base, "base_track": base_track,
        "winner": {
            "filt": spec.filt._replace(**{x: c[x] for x in FILT_KEYS})._asdict(),
            "trade": spec.trade._replace(
                z_entry=ze, z_exit=min(float(c["z_exit"]), 0.85 * ze),
                z_stop=max(float(c["z_stop"]), 1.25 * ze))._asdict(),
            "train": v["train"], "test": v["test"], "track": v["track"],
            "cuts": v["cuts"], "d_train": v["d_train"], "d_test": v["d_test"],
            "smooth": m,
        },
    }


def main() -> None:
    prices = load_raw_prices()
    prices_df = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    t0 = time.perf_counter()
    reports = []
    print(f"tuning {len(BASE)} pairs\n")
    for spec in BASE:
        r = tune_one(spec, prices)
        reports.append(r)
        if r is None:
            print(f"  {spec.name:>11}: SKIPPED (baseline not on grid)", flush=True)
            continue
        w = r["winner"]
        if w is None:
            print(f"  {spec.name:>11}: base {r['base']['train']['score']:+7.2f}/"
                  f"{r['base']['test']['score']:+7.2f}  "
                  f"{r['n_pareto']:5d} pareto {r['n_block']:5d} block "
                  f"0 robust -> KEEP SEED", flush=True)
        else:
            print(f"  {spec.name:>11}: base {r['base']['train']['score']:+7.2f}/"
                  f"{r['base']['test']['score']:+7.2f} -> "
                  f"{w['train']['score']:+7.2f}/{w['test']['score']:+7.2f}  "
                  f"({r['n_robust']} robust, track {r['base_track']:.3f}->"
                  f"{w['track']:.3f})", flush=True)
    print(f"\nsweep done in {time.perf_counter()-t0:.0f}s")

    winners = [r for r in reports if r and r["winner"]]
    print(f"{len(winners)} pairs have a robust candidate\n")
    (OUT_DIR / "tune_all_pairs.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    if not winners:
        print("nothing to confirm")
        return

    def cfgs_with(names):
        out = []
        for s in BASE:
            r = next((w for w in winners if w["name"] == s.name), None)
            if r and s.name in names:
                out.append(s._replace(filt=FiltCfg(**r["winner"]["filt"]),
                                      trade=TradeCfg(**r["winner"]["trade"])))
            else:
                out.append(s)
        return tuple(out)

    b_tr = portfolio_window(prices, BASE, TRAIN_START, TRAIN_END)["score"]
    b_te = portfolio_window(prices, BASE, TEST_START, TEST_END)["score"]
    print(f"PORTFOLIO baseline core: train {b_tr:.2f} test {b_te:.2f}\n")
    print("each winner alone (core):")
    singles = {}
    for w in winners:
        c = cfgs_with({w["name"]})
        a = portfolio_window(prices, c, TRAIN_START, TRAIN_END)["score"]
        b = portfolio_window(prices, c, TEST_START, TEST_END)["score"]
        singles[w["name"]] = (a, b)
        ok = "OK" if (a >= b_tr - 1e-9 and b >= b_te - 1e-9) else "  "
        print(f"  {ok} {w['name']:>11}: train {a:8.2f} ({a-b_tr:+6.2f})  "
              f"test {b:8.2f} ({b-b_te:+6.2f})", flush=True)

    pool = [n for n, (a, b) in singles.items() if a >= b_tr - 1e-9 and b >= b_te - 1e-9]
    chosen, cur_tr, cur_te = set(), b_tr, b_te
    print(f"\ngreedy forward selection over {len(pool)} eligible:")
    while pool:
        best = None
        for n in pool:
            c = cfgs_with(chosen | {n})
            a = portfolio_window(prices, c, TRAIN_START, TRAIN_END)["score"]
            b = portfolio_window(prices, c, TEST_START, TEST_END)["score"]
            if a >= cur_tr - 1e-9 and b >= cur_te - 1e-9:
                g = min(a - cur_tr, b - cur_te)
                if best is None or g > best[0]:
                    best = (g, n, a, b)
        if best is None:
            break
        _, n, cur_tr, cur_te = best
        chosen.add(n)
        pool.remove(n)
        print(f"  + {n:>11} -> train {cur_tr:8.2f} test {cur_te:8.2f}", flush=True)

    print(f"\nFINAL core: train {cur_tr:.2f} ({cur_tr-b_tr:+.2f})  "
          f"test {cur_te:.2f} ({cur_te-b_te:+.2f})")
    print(f"selected: {sorted(chosen)}")

    if chosen:
        try:
            import robust_hybrid_v2 as rh
        except Exception:
            rh = None
        if rh is not None:
            for label, cfgs in (("baseline", BASE), ("tuned set", cfgs_with(chosen))):
                a = portfolio_window(prices, cfgs, TRAIN_START, TRAIN_END, rh)["score"]
                b = portfolio_window(prices, cfgs, TEST_START, TEST_END, rh)["score"]
                print(f"  hybrid_v2 {label:10s} train {a:8.2f} test {b:8.2f}")
        dfil.apply_pair_cfgs(BASE)

    (OUT_DIR / "tune_all_selected.json").write_text(
        json.dumps({"selected": sorted(chosen),
                    "core": [cur_tr, cur_te], "core_base": [b_tr, b_te],
                    "winners": {w["name"]: w["winner"] for w in winners}},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'tune_all_selected.json'}")


if __name__ == "__main__":
    main()
