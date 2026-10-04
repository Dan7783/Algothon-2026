"""Hypothesis: the ADF prior-gate should use a TRAILING WINDOW, not expanding
history.

`PairAgent.desired_side` gates on `_adf_t_spread(LP[y,:nt-1] - beta*LP[x,:nt-1])`
— the residual over ALL history since day 1.  An early level shift poisons that
statistic permanently, so a pair that is perfectly stationary *now* can be
blocked forever.  Evidence this is real:

  * MSDP/ULXY, ACIX/ULXY, EAFC/SRTX are blocked 250/250 days in BOTH windows
    and contribute exactly 0.00 (they were pruned for it).
  * SRNA/ACIX is persistently cointegrated on a 1225-pair screen
    (-3.08 train / -3.07 test) yet is blocked 250/250 days too.

Test: replace the expanding residual with a trailing W-day residual and
measure.  Prediction if the hypothesis holds — the blocked pairs start
trading and the book improves; if it fails, the gate was doing real work and
the extra trades lose money.

NOTE ON POWER: ADF t-stats scale with sample size, so a 250-day window has
less power than 500+ days of expanding history.  A fixed -2.9 threshold is
therefore effectively STRICTER on a short window; the sweep varies the
threshold too so the two effects are not confounded.
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

TRAIN = (251, 500)
TEST = (501, 750)
OUT = _ROOT / "cusum_output" / "daniel_pairs"
_COMM = np.full(51, 1.0e-4)
_COMM[0] = 2.0e-5
_LIMIT = np.full(51, 10_000.0)
_LIMIT[0] = 100_000.0

BASE = tuple(fs.PAIR_CFGS)
PRUNED = {s.name for s in BASE if s.trade.leg_scale == 0.0}
DEAD = {"MSDP/ULXY", "ACIX/ULXY", "EAFC/SRTX"}

_ORIG = fs.PairAgent.desired_side
_CFG = {"window": 0, "gate": fs.PRIOR_GATE_T}


def _patched(self, LP):
    """desired_side with the ADF gate on a trailing window."""
    tc = self.spec.trade
    if not tc.trade_while_adapting and self.adapting:
        self.side = 0
        return 0.0
    nt = LP.shape[1]
    if nt < fs.HEDGE_W + 10:
        self.side = 0
        return 0.0
    y_i, x_i = self.spec.y, self.spec.x
    beta = self.beta_lag
    if tc.require_adf:
        lo = 0 if _CFG["window"] <= 0 else max(0, nt - 1 - _CFG["window"])
        spread = LP[y_i, lo:nt - 1] - beta * LP[x_i, lo:nt - 1]
        if len(spread) >= 40 and fs._adf_t_spread(spread) >= _CFG["gate"]:
            self.side = 0
            return 0.0
    s = LP[y_i, nt - fs.Z_W:nt] - beta * LP[x_i, nt - fs.Z_W:nt]
    z = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12))
    self.window_z = z
    if tc.require_kappa:
        s120 = LP[y_i, nt - 120:nt] - beta * LP[x_i, nt - 120:nt]
        if fs._ou_kappa(s120) <= 0.0:
            self.side = 0
            return 0.0
    az = abs(z)
    if self.side == 0:
        if tc.z_entry <= az < tc.z_stop:
            self.side = int(-np.sign(z))
    else:
        if az <= tc.z_exit or az >= tc.z_stop:
            self.side = 0
        elif az >= tc.z_entry and int(-np.sign(z)) != self.side:
            self.side = int(-np.sign(z))
    return float(self.side)


def score(mu, sd):
    if mu <= 0.0 or sd < 1e-10:
        return mu
    sr = sqrt(250.0) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def replay(prices, cfgs, start, end):
    fs.apply_pair_cfgs(cfgs)
    fs.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl = []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            req = fs.getMyPosition(hist)
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
    return score(float(a.mean()), float(a.std()))


def revived(names):
    """Restore leg_scale=1 for pairs the gate had permanently blocked."""
    return tuple(
        s._replace(trade=s.trade._replace(leg_scale=1.0)) if s.name in names else s
        for s in BASE
    )


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    rows = []

    fs.PairAgent.desired_side = _ORIG
    b_tr = replay(prices, BASE, *TRAIN)
    b_te = replay(prices, BASE, *TEST)
    print(f"baseline (expanding gate, 6 pruned): train {b_tr:.2f}  test {b_te:.2f}")
    print(f"currently pruned: {sorted(PRUNED)}\n")

    fs.PairAgent.desired_side = _patched
    print(f"{'variant':>46} | {'train':>8} {'test':>8} | vs baseline")
    print("-" * 84)
    for W in (250, 375, 500):
        for gate in (fs.PRIOR_GATE_T, -2.5):
            for revive, tag in ((set(), "as-pruned"), (DEAD, "+revive DEAD")):
                _CFG["window"], _CFG["gate"] = W, gate
                cfgs = revived(revive) if revive else BASE
                a = replay(prices, cfgs, *TRAIN)
                b = replay(prices, cfgs, *TEST)
                name = f"W={W} gate={gate} {tag}"
                ok = "OK" if (a >= b_tr and b >= b_te) else "  "
                print(f"  {ok} {name:>42} | {a:8.2f} {b:8.2f} | "
                      f"{a-b_tr:+7.2f} / {b-b_te:+7.2f}")
                rows.append({"window": W, "gate": gate, "revive": sorted(revive),
                             "train": a, "test": b})

    fs.PairAgent.desired_side = _ORIG
    fs.apply_pair_cfgs(BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "windowed_adf.json").write_text(
        json.dumps({"baseline": [b_tr, b_te], "variants": rows}, indent=2),
        encoding="utf-8")
    best = max(rows, key=lambda r: min(r["train"] - b_tr, r["test"] - b_te))
    print(f"\nbest by worst-window gain: W={best['window']} gate={best['gate']} "
          f"revive={best['revive']} -> train {best['train']:.2f} test {best['test']:.2f}")
    print(f"wrote {OUT / 'windowed_adf.json'}")


if __name__ == "__main__":
    main()
