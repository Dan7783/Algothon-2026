"""Fix stage 1's error: the filter hardcodes observation variance R = 1.

The measured residual variance is ~1.4e-03, so R = 1 overstates the
observation noise by ~700x.  Consequences:

  * the Kalman gain g = P*x / (P*x^2 + R) is far too small, so beta barely
    moves and q_beta appears inert across 8 orders of magnitude;
  * q_beta is uninterpretable, because only the RATIO q/R matters and the
    denominator is wrong by ~700x;
  * the CUSUM/adapt/snap machinery exists largely to force the adaptation
    that the gain should have produced on its own.

R = 1 is baked into TWO places:
    s_pred = pp22*x*x + 1.0          innovation variance
    p22    = a*a*pp22 + g*g          Joseph update, really a^2*P + g^2*R

This estimates R online as an EWMA of the post-fit squared residual, seeded
from the initial window, and uses it in both.  q_beta then means what it is
supposed to mean: process variance of beta per day, in the same units as the
observation.

Nothing here consults PnL; the check is the standard Kalman consistency
test — with R right, the normalised innovation squared (innov^2 / S) should
average about 1.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json
from math import sqrt

import numpy as np
import pandas as pd

import four_sigmas as fs

R_LAMBDA = 0.99      # slow, so R does not chase the very residuals it scales
R_FLOOR = 1e-8
R_WARMUP = 20


class RFilter:
    """PairSegFilter's Kalman core with a properly scaled observation noise.

    CUSUM is deliberately absent: this is the stage-1 object, used to
    identify q against the base noise before any break detection exists.
    """

    def __init__(self, q_beta: float):
        self.q = q_beta
        self.beta: float | None = None
        self.p22 = 1.0
        self.R: float | None = None
        self.innov_var: float | None = None
        self._y: list[float] = []
        self._x: list[float] = []

    def step(self, y: float, x: float):
        self._y.append(y)
        self._x.append(x)
        if self.beta is None:
            n = min(R_WARMUP, len(self._x))
            xs = np.asarray(self._x[-n:])
            ys = np.asarray(self._y[-n:])
            d = float(xs @ xs)
            self.beta = float(xs @ ys) / d if d > 1e-12 else 1.0
            r = ys - self.beta * xs
            self.R = max(float(np.var(r)), R_FLOOR)
            # P should start at the scale of the beta estimate's own variance
            self.p22 = self.R / max(d, 1e-12)

        pp = self.p22 + self.q
        innov = y - self.beta * x
        s = pp * x * x + self.R                    # <- R, not 1.0
        if s < 1e-18:
            s = 1e-18
        g = (pp * x) / s
        self.beta += g * innov
        a = 1.0 - g * x
        self.p22 = max(a * a * pp + g * g * self.R, 1e-18)   # <- g^2 * R

        resid = y - self.beta * x
        self.R = max(R_LAMBDA * self.R + (1.0 - R_LAMBDA) * resid * resid,
                     R_FLOOR)
        self.innov_var = (innov * innov if self.innov_var is None
                          else 0.97 * self.innov_var + 0.03 * innov * innov)
        return self.beta, innov, s


def rolling_ols(ly, lx, w=60):
    n = len(ly)
    out = np.full(n, np.nan)
    for t in range(w, n):
        a, b = lx[t - w:t], ly[t - w:t]
        d = float(a @ a)
        if d > 1e-12:
            out[t] = float(a @ b) / d
    return out


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    lp = np.log(prices)
    TR = 500
    qs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    print("Stage 1 REDONE with R estimated instead of hardcoded to 1\n")
    print("consistency check: NIS = mean(innov^2 / S) should be ~1")
    print(f"{'pair':>11} | " + " ".join(f"{q:8.0e}" for q in qs) + " | best q")
    print("-" * 76)
    track_tot = {q: [] for q in qs}
    nis_tot = {q: [] for q in qs}
    for spec in fs.PAIR_CFGS:
        ly, lx = lp[spec.y, :TR], lp[spec.x, :TR]
        ref = rolling_ols(ly, lx)
        m = np.isfinite(ref)
        errs, niss = [], []
        for q in qs:
            f = RFilter(q)
            b = np.empty(len(ly))
            nis = []
            for t in range(len(ly)):
                beta, innov, s = f.step(float(ly[t]), float(lx[t]))
                b[t] = beta
                if t > 30:
                    nis.append(innov * innov / s)
            e = float(np.sqrt(np.mean((b[m] - ref[m]) ** 2)))
            errs.append(e)
            niss.append(float(np.mean(nis)))
            track_tot[q].append(e)
            nis_tot[q].append(float(np.mean(nis)))
        best = qs[int(np.argmin(errs))]
        print(f"  {spec.name:>11} | " + " ".join(f"{e:8.4f}" for e in errs)
              + f" | {best:.0e}")

    print("\nmedian tracking error : "
          + "  ".join(f"{q:.0e}:{np.median(track_tot[q]):.4f}" for q in qs))
    print("median NIS (target ~1): "
          + "  ".join(f"{q:.0e}:{np.median(nis_tot[q]):.2f}" for q in qs))
    bq = min(qs, key=lambda q: np.median(track_tot[q]))
    print(f"\nbest q_beta under corrected R: {bq:.0e}")
    print(f"  (with R=1 the same criterion said 1e-04; the {1/1.4e-3:.0f}x "
          f"mis-scaling is why)")

    out = _ROOT / "cusum_output" / "fix_r_scaling.json"
    out.write_text(json.dumps(
        {"tracking": {str(q): np.median(track_tot[q]) for q in qs},
         "nis": {str(q): np.median(nis_tot[q]) for q in qs},
         "best_q": bq}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
