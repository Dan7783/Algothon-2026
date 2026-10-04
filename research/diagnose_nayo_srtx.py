"""Why does NAYO/SRTX earn +44 in train and -4 in test?

Hypothesis: the hedge ratio stops tracking in the test window, so the
residual drifts, |z| sits stretched, the ADF gate blocks 102 days, and the
trades that do fire are fading a spread that is not coming back.

Prints, per window: filter beta vs trailing OLS beta, how far the residual
drifts from zero, cut/adapt activity, ADF t-stat and OU kappa levels.
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

Y, X = dfil.I["NAYO"], dfil.I["SRTX"]


def rolling_ols(ly, lx, w):
    T = len(ly)
    out = np.full(T, np.nan)
    for t in range(w, T):
        a, b = lx[t - w:t], ly[t - w:t]
        ac, bc = a - a.mean(), b - b.mean()
        d = float(ac @ ac)
        if d > 1e-12:
            out[t] = float(ac @ bc) / d
    return out


def main() -> None:
    spec = next(s for s in dfil.PAIR_CFGS if {s.y, s.x} == {Y, X})
    prices = load_raw_prices()
    ly, lx = np.log(prices[spec.y]), np.log(prices[spec.x])

    print(f"{spec.name}   seed: q_beta {spec.filt.q_beta:.3e}  h {spec.filt.h}  "
          f"cusum_k {spec.filt.cusum_k}  min_seg {spec.filt.min_seg_len}  "
          f"adapt {spec.filt.adapt_bars}")
    print(f"          trade: z {spec.trade.z_entry}/{spec.trade.z_exit}/"
          f"{spec.trade.z_stop}  adf={spec.trade.require_adf} "
          f"kappa={spec.trade.require_kappa}\n")

    path = record_paths(ly, lx, spec.filt)
    gates = gate_and_z_paths(ly, lx, path["beta"])
    beta = path["beta"]
    ols60 = rolling_ols(ly, lx, 60)
    ols120 = rolling_ols(ly, lx, 120)

    for tag, s, e in (("train", TRAIN_START, TRAIN_END),
                      ("test", TEST_START, TEST_END)):
        sl = slice(s - 1, e)
        b = beta[sl]
        resid = ly[sl] - b * lx[sl]
        lag = beta[sl][0]
        print(f"=== {tag} {s}-{e} ===")
        print(f"  filter beta   : {b.min():.4f} -> {b.max():.4f}  "
              f"(start {b[0]:.4f} end {b[-1]:.4f}, drift {b[-1]-b[0]:+.4f})")
        print(f"  OLS-60  beta  : {np.nanmin(ols60[sl]):.4f} -> "
              f"{np.nanmax(ols60[sl]):.4f}  (end {ols60[sl][-1]:.4f})")
        print(f"  OLS-120 beta  : {np.nanmin(ols120[sl]):.4f} -> "
              f"{np.nanmax(ols120[sl]):.4f}  (end {ols120[sl][-1]:.4f})")
        gap = np.abs(b - ols120[sl])
        print(f"  |filter - OLS120| : mean {np.nanmean(gap):.4f}  "
              f"max {np.nanmax(gap):.4f}   <-- tracking error")
        print(f"  residual      : mean {resid.mean():+.5f}  sd {resid.std():.5f}  "
              f"|end| {abs(resid[-1]):.5f}")
        print(f"  cuts {int(path['cut'][sl].sum())}   adapting days "
              f"{int(path['adapting'][sl].sum())}")
        adf = gates["adf_t"][sl]
        kap = gates["kappa"][sl]
        print(f"  ADF t         : mean {np.nanmean(adf):+.2f}  "
              f"(gate at {dfil.PRIOR_GATE_T}; blocked "
              f"{int(gates['adf_block'][sl].sum())} days)")
        print(f"  OU kappa      : mean {np.nanmean(kap):+.4f}  "
              f"(blocked {int(gates['kappa_block'][sl].sum())} days)")
        z = gates["z"][sl]
        print(f"  |z|           : mean {np.abs(z).mean():.2f}  "
              f"p90 {np.percentile(np.abs(z), 90):.2f}")
        print()


if __name__ == "__main__":
    main()
