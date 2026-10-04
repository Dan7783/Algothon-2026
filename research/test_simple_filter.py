"""Can covariance inflation alone replace the whole post-cut adapt machinery?

On a CUSUM cut the filter currently does SIX things to re-adapt beta:
    snap_window   OLS window for the snap target
    snap_overshoot jump beta past that target
    cov_inflate   multiply P
    q_beta_adapt  elevated process noise while adapting
    adapt_chase   hard-track short OLS while adapting
    adapt_bars    how long to keep doing that
(plus adapt_extend / snap_max_rel / adapt_p_floor as modifiers)

All of them serve one purpose: "a break happened, let beta move fast".
Kalman theory says covariance inflation alone already does this — the gain
g = P*x/(P*x^2 + R) tends to 1/x as P grows, so beta jumps to the new level
by itself on the next observation.

SIMPLE variant: on a cut, inflate P and nothing else.  No snap, no chase,
no adapt countdown, no elevated q.

cov_inflate is chosen on the TRAIN window only; test is reported as a check,
never as the selector.
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

TRAIN, TEST = (251, 500), (501, 750)
OUT = _ROOT / "cusum_output"
_COMM = np.full(51, 1.0e-4)
_COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0
BASE = tuple(fs.PAIR_CFGS)
_ORIG_STEP = fs.PairSegFilter.step
_INFLATE = {"v": 25.0}


def _simple_step(self, y, x):
    """Filter step with the post-cut machinery reduced to covariance inflation."""
    cfg = self.cfg
    self.t += 1
    t = self.t
    cut = False
    yf, xf = float(y), float(x)
    self._y_hist.append(yf)
    self._x_hist.append(xf)
    if self.beta is None:
        self.beta = self._ols_beta(fs._INIT_WINDOW)
        self.seg_start = t
        self.h_live = cfg.h

    # one process-noise level; no adapt burst
    b = float(self.beta)
    pp22 = self.p22 + cfg.q_beta
    innov = yf - b * xf
    s_pred = pp22 * xf * xf + 1.0
    if s_pred < 1e-12:
        s_pred = 1e-12
    g = (pp22 * xf) / s_pred
    self.beta = b + g * innov
    a = 1.0 - g * xf
    self.p22 = max(a * a * pp22 + g * g, 1e-18)

    if self.innov_var is None:
        self.innov_var = innov * innov
    else:
        lam = cfg.ewma_lambda
        self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
    z = innov / sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0
    resid = yf - float(self.beta) * xf

    h_work = (self.h_live if (cfg.auto_h and self.h_live is not None) else cfg.h)

    if t >= cfg.warmup:
        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        if self.cm_up > 0.0 and prev_up == 0.0:
            self.run_m_up = t
        elif self.cm_up == 0.0:
            self.run_m_up = None
        if self.cm_dn > 0.0 and prev_dn == 0.0:
            self.run_m_dn = t
        elif self.cm_dn == 0.0:
            self.run_m_dn = None

        if self.cm_up > h_work or self.cm_dn > h_work:
            cand = [r for r in (self.run_m_up, self.run_m_dn) if r is not None]
            cut_bar = max(int(min(cand)) if cand else t, (self.seg_start or 0) + 1)
            closed = cut_bar - (self.seg_start or 0)
            self.cm_up = self.cm_dn = 0.0
            self.run_m_up = self.run_m_dn = None
            if closed >= cfg.min_seg_len:
                cut = True
                # THE WHOLE ADAPTATION: reopen the covariance.  The Kalman
                # gain does the rest on the next observation.
                self.p22 = max(self.p22 * _INFLATE["v"], 1e-18)
                if cfg.auto_h:
                    if self.seg_len_ewma is None:
                        self.seg_len_ewma = float(min(closed, 3 * cfg.target_seg))
                    else:
                        self.seg_len_ewma = 0.1 * closed + 0.9 * self.seg_len_ewma
                    err = max(-0.5, min(0.5,
                              (cfg.target_seg - self.seg_len_ewma) / cfg.target_seg))
                    h0 = self.h_live if self.h_live is not None else cfg.h
                    self.h_live = max(2.0, h0 + cfg.h_gain * err * h0)
                self.seg_start = cut_bar

    h_out = (self.h_live if (cfg.auto_h and self.h_live is not None) else cfg.h)
    return fs.StepOut(beta=float(self.beta), spread=float(resid), z=float(z),
                      h=float(h_out), cut=cut, adapting=False, vol_shift=False,
                      cm_up=float(self.cm_up), cm_dn=float(self.cm_dn),
                      cv=0.0)


def score(m, s):
    if m <= 0 or s < 1e-10:
        return m
    sr = sqrt(250.0) * m / s
    return m * sr * sr / (sr * sr + 1.0)


def replay(prices, a, b):
    fs.apply_pair_cfgs(BASE)
    fs.reset()
    cash = val = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(a - 1, b + 1):
        h = prices[:, :t]
        cur = h[:, -1]
        if t < b:
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
    arr = np.asarray(pnl[1:])
    return score(float(arr.mean()), float(arr.std()))


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    fs.PairSegFilter.step = _ORIG_STEP
    f_tr, f_te = replay(prices, *TRAIN), replay(prices, *TEST)
    print(f"full filter (6 adapt knobs): train {f_tr:8.2f}  test {f_te:8.2f}\n")

    fs.PairSegFilter.step = _simple_step
    print("simplified filter — cov_inflate chosen on TRAIN ONLY:")
    rows = []
    for v in (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0):
        _INFLATE["v"] = v
        a = replay(prices, *TRAIN)
        rows.append((a, v))
        print(f"  cov_inflate {v:7.0f}: train {a:8.2f}", flush=True)
    best_tr, best_v = max(rows)
    _INFLATE["v"] = best_v
    b = replay(prices, *TEST)
    print(f"\nchosen on train: cov_inflate={best_v:.0f}")
    print(f"  simplified: train {best_tr:8.2f} ({best_tr-f_tr:+7.2f})  "
          f"test {b:8.2f} ({b-f_te:+7.2f})   <- test is a CHECK, not the selector")

    fs.PairSegFilter.step = _ORIG_STEP
    fs.apply_pair_cfgs(BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "simple_filter.json").write_text(json.dumps(
        {"full": [f_tr, f_te], "train_sweep": [[v, a] for a, v in rows],
         "chosen": best_v, "simplified": [best_tr, b]}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'simple_filter.json'}")


if __name__ == "__main__":
    main()
