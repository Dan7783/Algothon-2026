"""Rigorous NAYO/SRTX tune, with beta-modelling as the explicit question.

The diagnosis: the seeded filter holds beta ~0.62 while the local OLS beta
swings between -0.74 and +0.88, giving 0.65 mean tracking error in test.
So the question is not only "what scores best" but "does modelling beta
better actually help the test window?" — the sweep therefore records the
beta tracking error of every candidate alongside its scores.

Proof protocol (same as tune_single_pair): Pareto on both windows,
four-block non-degradation, and neighbourhood-smoothed robustness so a
fitted spike cannot win.  Portfolio confirmation is a separate step.
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

import archive.old_tuners.daniel_filter as dfil
from research.pair_book_lab import load_raw_prices
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

Y, X = dfil.I["NAYO"], dfil.I["SRTX"]
SPEC = next(s for s in dfil.PAIR_CFGS if {s.y, s.x} == {Y, X})

# Beta-modelling axis is deliberately wide: the seed is 9.1e-07 and the
# diagnosis says it is far too sticky for this relationship.
AXES = {
    "q_beta": [1e-8, 1e-7, SPEC.filt.q_beta, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "h": [3.0, SPEC.filt.h, 6.0, 8.0],
    "cusum_k": [0.35, SPEC.filt.cusum_k, 0.65],
    "min_seg_len": [20, SPEC.filt.min_seg_len, 60],
    "adapt_bars": [4, SPEC.filt.adapt_bars, 16],
    "z_entry": [0.10, SPEC.trade.z_entry, 0.27, 0.40, 0.60],
    "z_exit": [SPEC.trade.z_exit, 0.08, 0.15],
    "z_stop": [2.50, SPEC.trade.z_stop, 4.50],
    "require_adf": [True, False],
}
for _k in AXES:
    AXES[_k] = sorted(set(AXES[_k]), key=lambda v: (isinstance(v, bool), v))
FILT_KEYS = ("q_beta", "h", "cusum_k", "min_seg_len", "adapt_bars")
TRADE_KEYS = ("z_entry", "z_exit", "z_stop", "require_adf")


def rolling_ols(ly, lx, w=120):
    T = len(ly)
    out = np.full(T, np.nan)
    for t in range(w, T):
        a, b = lx[t - w:t], ly[t - w:t]
        ac, bc = a - a.mean(), b - b.mean()
        d = float(ac @ ac)
        if d > 1e-12:
            out[t] = float(ac @ bc) / d
    return out


def evaluate(gates, py, px, trade) -> dict:
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tp = pnl_path(sy_tr, sx_tr, py, px)
    ep = pnl_path(sy_te, sx_te, py, px)
    return {
        "train": window_stats(tp, TRAIN_START, TRAIN_END),
        "test": window_stats(ep, TEST_START, TEST_END),
        "blocks": [
            window_stats(tp, TRAIN_START, TRAIN_MID),
            window_stats(tp, TRAIN_MID + 1, TRAIN_END),
            window_stats(ep, TEST_START, TEST_MID),
            window_stats(ep, TEST_MID + 1, TEST_END),
        ],
        "d_test": int((np.abs(sy_te[TEST_START - 1:TEST_END]) > 0).sum()),
    }


def main() -> None:
    prices = load_raw_prices()
    ly, lx = np.log(prices[SPEC.y]), np.log(prices[SPEC.x])
    py, px = prices[SPEC.y], prices[SPEC.x]
    ols = rolling_ols(ly, lx)
    test_sl = slice(TEST_START - 1, TEST_END)

    base_path = record_paths(ly, lx, SPEC.filt)
    base_gates = gate_and_z_paths(ly, lx, base_path["beta"])
    baseline = evaluate(base_gates, py, px, SPEC.trade)
    base_track = float(np.nanmean(np.abs(base_path["beta"][test_sl] - ols[test_sl])))
    print(f"baseline: train {baseline['train']['score']:+7.2f} "
          f"test {baseline['test']['score']:+7.2f}  "
          f"test beta-tracking-error {base_track:.4f}\n")

    filt_combos = list(itertools.product(*(AXES[k] for k in FILT_KEYS)))
    trade_combos = list(itertools.product(*(AXES[k] for k in TRADE_KEYS)))
    print(f"grid: {len(filt_combos)} filter paths x {len(trade_combos)} trade "
          f"= {len(filt_combos) * len(trade_combos)} candidates")

    results: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    for fi, fv in enumerate(filt_combos):
        cf = dict(zip(FILT_KEYS, fv))
        filt = SPEC.filt._replace(**cf)
        p = record_paths(ly, lx, filt)
        gates = gate_and_z_paths(ly, lx, p["beta"])
        track = float(np.nanmean(np.abs(p["beta"][test_sl] - ols[test_sl])))
        cuts = int(p["cut"].sum())
        for tv in trade_combos:
            ct = dict(zip(TRADE_KEYS, tv))
            ze = float(ct["z_entry"])
            trade = SPEC.trade._replace(
                z_entry=ze,
                z_exit=min(float(ct["z_exit"]), 0.85 * ze),
                z_stop=max(float(ct["z_stop"]), 1.25 * ze),
                require_adf=bool(ct["require_adf"]),
            )
            r = evaluate(gates, py, px, trade)
            r["track"] = track
            r["cuts"] = cuts
            results[tuple({**cf, **ct}[k] for k in AXES)] = r
        if (fi + 1) % 300 == 0:
            print(f"  {fi+1}/{len(filt_combos)} paths "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"grid done in {time.perf_counter()-t0:.0f}s\n")

    keys = list(AXES)
    idx_of = {k: {v: i for i, v in enumerate(AXES[k])} for k in keys}

    def neighbours(key):
        out = []
        for d, k in enumerate(keys):
            i = idx_of[k][key[d]]
            for st in (-1, 1):
                j = i + st
                if 0 <= j < len(AXES[k]):
                    nk = list(key)
                    nk[d] = AXES[k][j]
                    out.append(tuple(nk))
        return out

    def smoothed(key):
        pts = [results[key]] + [results[n] for n in neighbours(key) if n in results]
        return (float(np.mean([p["train"]["score"] for p in pts])),
                float(np.mean([p["test"]["score"] for p in pts])))

    # --- does better beta tracking actually mean a better test window? ---
    tracks = np.array([r["track"] for r in results.values()])
    tests = np.array([r["test"]["score"] for r in results.values()])
    trains = np.array([r["train"]["score"] for r in results.values()])
    print("Does modelling beta better help the test window?")
    print(f"  corr(tracking error, test score)  = {np.corrcoef(tracks, tests)[0,1]:+.3f}")
    print(f"  corr(tracking error, train score) = {np.corrcoef(tracks, trains)[0,1]:+.3f}")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0), (1.0, 9.0)):
        m = (tracks >= lo) & (tracks < hi)
        if m.sum():
            print(f"  tracking err [{lo:.1f},{hi:.1f}): n {int(m.sum()):6d}  "
                  f"mean test {tests[m].mean():+7.2f}  best test {tests[m].max():+7.2f}"
                  f"  mean train {trains[m].mean():+7.2f}")
    print()

    pareto = {
        k: v for k, v in results.items()
        if v["train"]["score"] > baseline["train"]["score"]
        and v["test"]["score"] > baseline["test"]["score"]
        and v["train"]["mean"] >= baseline["train"]["mean"]
        and v["test"]["mean"] >= baseline["test"]["mean"]
    }
    blocks_ok = {
        k: v for k, v in pareto.items()
        if all(v["blocks"][i]["score"] >= baseline["blocks"][i]["score"] - 1e-9
               for i in range(4))
    }
    base_vals = {
        "q_beta": SPEC.filt.q_beta, "h": SPEC.filt.h,
        "cusum_k": SPEC.filt.cusum_k, "min_seg_len": SPEC.filt.min_seg_len,
        "adapt_bars": SPEC.filt.adapt_bars, "z_entry": SPEC.trade.z_entry,
        "z_exit": SPEC.trade.z_exit, "z_stop": SPEC.trade.z_stop,
        "require_adf": SPEC.trade.require_adf,
    }
    base_key = tuple(base_vals[k] for k in keys)
    if base_key not in results:
        raise SystemExit("baseline is not a grid point — axes are inconsistent")
    bs = smoothed(base_key)
    robust = []
    for k, v in blocks_ok.items():
        mtr, mte = smoothed(k)
        if mtr > bs[0] and mte > bs[1]:
            robust.append({"key": k, "v": v, "smooth": (mtr, mte)})
    print(f"{len(pareto)} pass Pareto, {len(blocks_ok)} pass four-block, "
          f"{len(robust)} pass smoothed-robustness")
    print(f"baseline smoothed: train {bs[0]:+7.2f} test {bs[1]:+7.2f}\n")

    if not robust:
        print("No robust improvement found — keep the live seed.")
        return

    robust.sort(key=lambda r: min(r["smooth"][0] - bs[0], r["smooth"][1] - bs[1]),
                reverse=True)
    print("top robust candidates:")
    for r in robust[:10]:
        c = dict(zip(keys, r["key"]))
        v = r["v"]
        print(f"  q {c['q_beta']:.0e} h {c['h']:<4} k {c['cusum_k']:<5} "
              f"ms {c['min_seg_len']:<3} ab {c['adapt_bars']:<3} | "
              f"z {c['z_entry']:.2f}/{c['z_exit']:.3f}/{c['z_stop']:.2f} "
              f"adf {str(c['require_adf'])[0]} | "
              f"train {v['train']['score']:+7.2f} test {v['test']['score']:+7.2f} | "
              f"smooth {r['smooth'][0]:+7.2f}/{r['smooth'][1]:+7.2f} | "
              f"track {v['track']:.3f} cuts {v['cuts']} dtest {v['d_test']}")

    best = robust[0]
    c = dict(zip(keys, best["key"]))
    ze = float(c["z_entry"])
    out = {
        "pair": [SPEC.y, SPEC.x], "name": SPEC.name,
        "baseline": {"train": baseline["train"], "test": baseline["test"],
                     "track": base_track},
        "chosen": {
            "filt": SPEC.filt._replace(
                **{k: c[k] for k in FILT_KEYS})._asdict(),
            "trade": SPEC.trade._replace(
                z_entry=ze, z_exit=min(float(c["z_exit"]), 0.85 * ze),
                z_stop=max(float(c["z_stop"]), 1.25 * ze),
                require_adf=bool(c["require_adf"]))._asdict(),
            "train": best["v"]["train"], "test": best["v"]["test"],
            "track": best["v"]["track"], "cuts": best["v"]["cuts"],
            "smooth_train": best["smooth"][0], "smooth_test": best["smooth"][1],
        },
        "counts": {"grid": len(results), "pareto": len(pareto),
                   "four_block": len(blocks_ok), "robust": len(robust)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "single_nayo_srtx.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'single_nayo_srtx.json'}")
    print("NEXT: python research/confirm_pair_seed.py NAYO SRTX")


if __name__ == "__main__":
    main()
