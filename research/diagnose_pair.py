"""Diagnose any seeded pair, and check whether its partner is the right one.

Usage:  python research/diagnose_pair.py MDGI HTRK

Part 1 — per-window health: filter beta (through-origin, so ~y/x by
construction), residual drift, cuts, ADF t, OU kappa, |z|, participation.

Part 2 — partner screen: for each leg, rank ALL other instruments by
cointegration strength (ADF t of the through-origin residual) on the train
window, then report how those rankings hold up out of sample.  If the panel
really is built in pairs, the designed partner should sit at the top of both
windows; a partner that tops train but not test is a fitted artifact.
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

import numpy as np

import archive.old_tuners.daniel_filter as dfil
from research.pair_book_lab import load_raw_prices
from tune_daniel_pairs import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    gate_and_z_paths,
    record_paths,
)

INV = {v: k for k, v in dfil.I.items()}


def adf_t(s: np.ndarray) -> float:
    if len(s) < 40 or float(np.std(s)) <= 1e-12:
        return 0.0
    d = np.diff(s)
    t = d[1:]
    D = np.column_stack((np.ones(len(t)), s[1:-1], d[:-1]))
    inv = np.linalg.pinv(D.T @ D)
    c = inv @ D.T @ t
    e = t - D @ c
    v = float(e @ e) / max(len(t) - 3, 1)
    return float(c[1] / np.sqrt(max(v * inv[1, 1], 1e-300)))


def resid_adf(ly: np.ndarray, lx: np.ndarray, sl: slice) -> float:
    """Through-origin hedge fitted and scored inside the same window."""
    a, b = lx[sl], ly[sl]
    d = float(a @ a)
    if d <= 1e-12:
        return 0.0
    return adf_t(b - (float(a @ b) / d) * a)


def rolling_ols(ly, lx, w=120):
    T = len(ly)
    out = np.full(T, np.nan)
    for t in range(w, T):
        a, b = lx[t - w:t], ly[t - w:t]
        ac, bc = a - a.mean(), b - b.mean()
        dd = float(ac @ ac)
        if dd > 1e-12:
            out[t] = float(ac @ bc) / dd
    return out


def main() -> None:
    y_name = _sys.argv[1].upper() if len(_sys.argv) > 2 else "MDGI"
    x_name = _sys.argv[2].upper() if len(_sys.argv) > 2 else "HTRK"
    Y, X = dfil.I[y_name], dfil.I[x_name]
    spec = next((s for s in dfil.PAIR_CFGS if {s.y, s.x} == {Y, X}), None)
    prices = load_raw_prices()
    log_p = np.log(prices)

    if spec is not None:
        Y, X = spec.y, spec.x
        ly, lx = log_p[Y], log_p[X]
        print(f"{spec.name}   seed: q_beta {spec.filt.q_beta:.3e}  h {spec.filt.h}  "
              f"k {spec.filt.cusum_k}  min_seg {spec.filt.min_seg_len}  "
              f"adapt {spec.filt.adapt_bars}")
        print(f"          trade: z {spec.trade.z_entry}/{spec.trade.z_exit}/"
              f"{spec.trade.z_stop}  adf={spec.trade.require_adf} "
              f"kappa={spec.trade.require_kappa}\n")
        path = record_paths(ly, lx, spec.filt)
        gates = gate_and_z_paths(ly, lx, path["beta"])
        ols = rolling_ols(ly, lx)
        for tag, s, e in (("train", TRAIN_START, TRAIN_END),
                          ("test", TEST_START, TEST_END)):
            sl = slice(s - 1, e)
            b = path["beta"][sl]
            resid = ly[sl] - b * lx[sl]
            ratio = (ly / lx)[sl]
            print(f"=== {tag} {s}-{e} ===")
            print(f"  filter beta {b[0]:.4f} -> {b[-1]:.4f}   "
                  f"log-price ratio y/x {ratio.min():.4f}-{ratio.max():.4f} "
                  f"(through-origin beta tracks this, not OLS)")
            print(f"  OLS-120 (demeaned) {np.nanmin(ols[sl]):.3f} -> "
                  f"{np.nanmax(ols[sl]):.3f}")
            print(f"  residual sd {resid.std():.5f}   |end| {abs(resid[-1]):.5f}")
            print(f"  in-window residual ADF {resid_adf(ly, lx, sl):+.2f}")
            print(f"  cuts {int(path['cut'][sl].sum())}  adapting "
                  f"{int(path['adapting'][sl].sum())}")
            print(f"  ADF t mean {np.nanmean(gates['adf_t'][sl]):+.2f}  "
                  f"blocked {int(gates['adf_block'][sl].sum())} days")
            print(f"  kappa mean {np.nanmean(gates['kappa'][sl]):+.4f}  "
                  f"blocked {int(gates['kappa_block'][sl].sum())} days")
            print(f"  |z| mean {np.abs(gates['z'][sl]).mean():.2f}  "
                  f"p90 {np.percentile(np.abs(gates['z'][sl]), 90):.2f}\n")

    # ---- partner screen ----
    tr = slice(TRAIN_START - 1, TRAIN_END)
    te = slice(TEST_START - 1, TEST_END)
    print("PARTNER SCREEN — is this the designed partner?")
    print("(through-origin residual ADF, fitted and scored within each window)")
    for leg in (Y, X):
        scores = []
        for j in range(1, 51):
            if j == leg:
                continue
            scores.append((resid_adf(log_p[leg], log_p[j], tr),
                           resid_adf(log_p[leg], log_p[j], te), j))
        scores.sort()
        partner = X if leg == Y else Y
        rank = next(i for i, (_, _, j) in enumerate(scores) if j == partner) + 1
        print(f"\n  {INV[leg]}: top-6 partners by TRAIN ADF "
              f"(current partner {INV[partner]} ranks #{rank}/49)")
        for a, b, j in scores[:6]:
            mark = "  <-- current" if j == partner else ""
            print(f"    {INV[j]:>6}  train {a:+6.2f}  test {b:+6.2f}{mark}")
        if rank > 6:
            a, b, _ = scores[rank - 1]
            print(f"    {INV[partner]:>6}  train {a:+6.2f}  test {b:+6.2f}"
                  f"  <-- current (#{rank})")
        stable = [(a, b, j) for a, b, j in scores if a < -3.0 and b < -3.0]
        print(f"    partners with ADF < -3 in BOTH windows: "
              f"{[INV[j] for _, _, j in stable[:8]]}")


if __name__ == "__main__":
    main()
