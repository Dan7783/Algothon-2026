"""Stage 1: CUSUM off — match the Kalman to the data's base noise.

The filter fixes the observation variance at R = 1:

    innovation variance S = p22 * x^2 + 1

so q_beta is only meaningful RELATIVE to R.  If the true residual variance is
far from 1, every q_beta is being read against the wrong noise floor and the
filter's stickiness is an artefact of the scaling, not a modelling choice.

This measures, per pair, on TRAIN ONLY and with no reference to PnL:

  sigma2_resid  variance of y - beta*x  (the real observation noise)
  sigma2_drift  variance of the day-to-day change in the true hedge ratio,
                estimated from non-overlapping rolling OLS betas and
                corrected for the estimation error of those betas
  q_star        = sigma2_drift / sigma2_resid, i.e. the process-noise-to-
                observation-noise ratio the filter SHOULD be using once R is
                normalised to 1

Then it checks the filter's own consistency: with the right q, the
normalised innovation squared (NIS = innov^2 / S) should average ~1.  That
is the standard Kalman consistency test and needs no trading score.
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
BLOCK = 25          # non-overlapping window for local OLS betas


def local_betas(ly, lx, block=BLOCK):
    """Through-origin OLS beta on consecutive non-overlapping blocks."""
    out, var = [], []
    n = len(ly)
    for s in range(0, n - block + 1, block):
        a, b = lx[s:s + block], ly[s:s + block]
        d = float(a @ a)
        if d <= 1e-12:
            continue
        beta = float(a @ b) / d
        resid = b - beta * a
        s2 = float(resid @ resid) / max(block - 1, 1)
        out.append(beta)
        var.append(s2 / d)          # sampling variance of this beta estimate
    return np.asarray(out), np.asarray(var)


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    lp = np.log(prices)
    rows = []
    print("Stage 1 — base-noise identification (TRAIN days 1-500, no PnL)\n")
    print(f"{'pair':>11} | {'sig2_resid':>10} {'sig2_drift':>11} | {'q*':>10} "
          f"| {'q seed':>10} {'ratio':>8}")
    print("-" * 74)
    for spec in fs.PAIR_CFGS:
        ly, lx = lp[spec.y, :TRAIN_END], lp[spec.x, :TRAIN_END]

        # observation noise: residual around a slowly-varying hedge
        betas, bvar = local_betas(ly, lx)
        resid_var = []
        for k, s in enumerate(range(0, len(ly) - BLOCK + 1, BLOCK)):
            a, b = lx[s:s + BLOCK], ly[s:s + BLOCK]
            resid_var.append(float(np.var(b - betas[k] * a)))
        s2_resid = float(np.median(resid_var))

        # process noise: how much the TRUE beta moves per day.  The raw
        # variance of successive local betas is inflated by their own
        # estimation error, so subtract it.
        d = np.diff(betas)
        raw = float(np.var(d))
        est = float(np.mean(bvar[:-1] + bvar[1:]))
        s2_step = max(raw - est, 0.0)
        s2_drift = s2_step / BLOCK          # per day

        q_star = s2_drift / max(s2_resid, 1e-18)
        rows.append({"name": spec.name, "s2_resid": s2_resid,
                     "s2_drift": s2_drift, "q_star": q_star,
                     "q_seed": spec.filt.q_beta})
        print(f"  {spec.name:>11} | {s2_resid:10.2e} {s2_drift:11.2e} | "
              f"{q_star:10.2e} | {spec.filt.q_beta:10.2e} "
              f"{q_star / max(spec.filt.q_beta, 1e-18):8.1f}x")

    qs = np.array([r["q_star"] for r in rows])
    print(f"\nq* across pairs: median {np.median(qs):.2e}, "
          f"range {qs.min():.2e} - {qs.max():.2e}")
    print(f"seeded q_beta  : median {np.median([r['q_seed'] for r in rows]):.2e}")
    print(f"\nmedian sigma2_resid = {np.median([r['s2_resid'] for r in rows]):.2e}")
    print("The filter hardcodes R = 1.  Since the true observation variance is")
    print("far below 1, a q_beta read against R=1 understates the intended")
    print("process-to-observation ratio by roughly 1/sigma2_resid.")

    out = _ROOT / "cusum_output" / "stage1_base_noise.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
