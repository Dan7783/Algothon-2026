"""Attempt RNG seed recovery on prices.txt, with honest multiple-testing control.

The competitor's proposal, stated precisely: brute-force candidate seeds, score
each against the observed data, keep the best. That is a search over ~10^5-10^7
hypotheses, so the question is not "did we find a high correlation" (we always
will) but "is the best hit better than the best hit from pure noise".

This script runs the search and calibrates it two ways:
  1. an analytic threshold for max |corr| over the number of tests performed
  2. an EXTENSION test - the only thing that actually settles it. A true seed
     match holds when you extend from 64 draws to 512. A lucky one collapses.

Run from repo root:  python research/seed_search.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = _ROOT / "analysis_outputs" / "generator_forensics"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 100_000
K_SHORT = 64        # draws compared in the wide search
K_LONG = 512        # draws compared in the extension test

raw = pd.read_csv(_ROOT / "prices.txt", sep=r"\s+", header=0)
PRC = raw.to_numpy(float).T
N, T = PRC.shape
R = np.diff(np.log(PRC), axis=1)

lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centre and scale to unit norm so a dot product IS a correlation."""
    x = x - x.mean(axis=axis, keepdims=True)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-300)


say("SEED RECOVERY ATTEMPT")
say("=" * 78)
say(f"search: {N_SEEDS:,} seeds x 2 bit generators x 2 draw orderings")

# --- targets -------------------------------------------------------------
# Each hypothesis compares a different slice length, so carry (target, k).
Z = (R - R.mean(axis=1, keepdims=True)) / R.std(axis=1, keepdims=True)
_, _, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)

TARGETS = {
    # the stream feeds one asset's consecutive days: 64 draws
    "H1 asset time-series": (unit(R[:, :K_SHORT]), K_SHORT),
    # the stream feeds one day's cross-section: one draw per asset
    "H2 daily cross-section": (unit(R[:, :K_SHORT].T), N),
    # the stream feeds a common factor directly: 64 draws
    "H3 common factors": (unit(vt[:3, :K_SHORT]), K_SHORT),
}


def draw_block(kind: str, n_seeds: int, k: int) -> np.ndarray:
    """(n_seeds, k) matrix of the first k standard normals from each seed."""
    out = np.empty((n_seeds, k))
    if kind == "PCG64":
        for s in range(n_seeds):
            out[s] = np.random.default_rng(s).standard_normal(k)
    else:
        for s in range(n_seeds):
            out[s] = np.random.RandomState(s).standard_normal(k)
    return out


K_MAX = max(K_SHORT, N)
results = []
for kind in ("PCG64", "MT19937"):
    say(f"\ngenerating {N_SEEDS:,} candidate streams from {kind} ...")
    raw_block = draw_block(kind, N_SEEDS, K_MAX)
    for hname, (tgt, k) in TARGETS.items():
        cand = unit(raw_block[:, :k])          # renormalise at this length
        corr = cand @ tgt.T                    # (n_seeds, n_targets)
        flat = np.abs(corr)
        idx = int(np.argmax(flat))
        seed, tcol = divmod(idx, flat.shape[1])
        results.append({
            "generator": kind,
            "hypothesis": hname,
            "k": k,
            "n_tests": flat.size,
            "best_abs_corr": float(flat.max()),
            "best_seed": seed,
            "best_target": tcol,
        })
        say(f"  {hname:24s} k={k:>3} tests={flat.size:>9,}  "
            f"best |r|={flat.max():.4f} at seed {seed}, target {tcol}")

res = pd.DataFrame(results)
total_tests = int(res["n_tests"].sum())

# --- what does pure noise produce over this many tests? ------------------
say("\n" + "=" * 78)
say("CALIBRATION: what would pure noise have produced?")
say(f"  total hypotheses tested: {total_tests:,}")


def max_abs_r_quantile(n_tests: int, k: int, p: float = 0.5) -> float:
    """p-quantile of max |r| over n_tests independent length-k samples."""
    df = k - 2
    u = p ** (1.0 / n_tests)                 # CDF of the maximum
    t = sps.t.ppf(0.5 + u / 2.0, df)
    return float(t / np.sqrt(df + t * t))


say(f"  {'hypothesis':24s} {'k':>4} {'tests':>10} {'observed':>9} "
    f"{'noise med':>10} {'noise p99':>10}")
over = 0
for _, r in res.iterrows():
    med = max_abs_r_quantile(int(r["n_tests"]), int(r["k"]), 0.5)
    p99 = max_abs_r_quantile(int(r["n_tests"]), int(r["k"]), 0.99)
    flag = "" if r["best_abs_corr"] <= p99 else "  <-- ABOVE"
    over += int(r["best_abs_corr"] > p99)
    say(f"  {r['hypothesis']:24s} {int(r['k']):>4} {int(r['n_tests']):>10,} "
        f"{r['best_abs_corr']:>9.4f} {med:>10.4f} {p99:>10.4f}{flag}")
say(f"  -> {over} of {len(res)} searches beat their own noise ceiling")

# --- the extension test: the only test that settles it -------------------
say("\n" + "=" * 78)
say(f"EXTENSION TEST: re-score each winner on {K_LONG} draws instead of {K_SHORT}")
say("  a real seed match is invariant to length; a lucky one collapses to ~0")
say("  (H2 has fixed width 51, so it is extended by asking whether the NEXT")
say("   block of 51 draws matches the NEXT day's cross-section)")
rows = []
for _, r in res.iterrows():
    kind, hname, seed, tcol = (
        r["generator"], r["hypothesis"], int(r["best_seed"]), int(r["best_target"])
    )
    n_draw = K_LONG if not hname.startswith("H2") else 2 * N
    if kind == "PCG64":
        stream = np.random.default_rng(seed).standard_normal(n_draw)
    else:
        stream = np.random.RandomState(seed).standard_normal(n_draw)

    if hname.startswith("H1"):
        tgt, cand = R[tcol, :K_LONG], stream
    elif hname.startswith("H3"):
        tgt, cand = vt[tcol, :K_LONG], stream
    else:
        # next 51 draws vs the next day's cross-section
        nxt = min(tcol + 1, R.shape[1] - 1)
        tgt, cand = R[:, nxt], stream[N: 2 * N]
    long_r = float(unit(cand) @ unit(tgt))
    rows.append({**r.to_dict(), "k_long": len(tgt), "long_abs_corr": abs(long_r)})
    say(f"  {kind:8s} {hname:24s} seed {seed:>6}  "
        f"|r|@{int(r['k'])} = {r['best_abs_corr']:.4f}  ->  "
        f"|r|@ext = {abs(long_r):.4f}")

ext = pd.DataFrame(rows)
survived = ext[ext["long_abs_corr"] > 0.5]
say(f"\n  candidates surviving the extension test: {len(survived)}")

# --- how much information does 2dp rounding leave? -----------------------
say("\n" + "=" * 78)
say("WHY STATE RECOVERY CANNOT WORK HERE, EVEN IN PRINCIPLE")
rel_noise = 0.005 / PRC[:, 1:]
moved = np.abs(R) > 0                     # unchanged prices carry no information
bits = np.log2(np.abs(R[moved]) / rel_noise[moved])
say(f"  prices are quoted to 2dp, so each log return is known to about "
    f"{np.median(bits):.1f} bits ({int((~moved).sum())} returns are exactly 0)")
say("  MT19937 state recovery needs 624 EXACT 32-bit outputs = 19,968 bits,")
say("  with zero tolerance for error. PCG64 needs its 128-bit state exactly.")
say(f"  Observed returns are also a MIXTURE (3 factors + idiosyncratic + OU),")
say("  so no single observation is one raw draw: you would have to invert the")
say("  mixture with estimated loadings first, adding error far larger than the")
say("  rounding itself. The map from data back to raw draws is not invertible.")

res_path = OUT / "seed_search.csv"
ext.to_csv(res_path, index=False)
(OUT / "seed_search.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {res_path}")
