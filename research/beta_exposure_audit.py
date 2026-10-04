"""Is the book market-neutral, or is it carrying hedgeable beta?

Three questions:

  1. Net dollar exposure — the atlas legs are EQUAL DOLLAR ($10k/-$10k), not
     beta-weighted, so a pair with hedge ratio != 1 is dollar-neutral but not
     beta-neutral.  Ridge and lead-lag are outright directional.
  2. Does daily PnL load on a market factor?  Regress book PnL on the
     equal-weighted market return and on ALGO (instrument 0).
  3. Would hedging help the SCORE?  score = mu * SR^2/(SR^2+1) rewards
     Sharpe, so removing variance at constant mean raises the score even
     with no extra PnL.  Instrument 0 is the natural hedge: 10x position
     limit ($100k) and 5x lower commission (0.2bp vs 1bp), and every
     strategy in this repo explicitly zeroes it.
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

import json
from math import sqrt

import numpy as np
import pandas as pd

import four_sigmas as rh
from tune_daniel_pairs import OUT_DIR, TEST_END, TEST_START, TRAIN_END, TRAIN_START

COMM = np.full(51, 1.0e-4)
COMM[0] = 2.0e-5
LIMIT = np.full(51, 10_000.0)
LIMIT[0] = 100_000.0


def score(mean: float, std: float) -> float:
    if mean <= 0.0 or std < 1e-10:
        return mean
    sr = sqrt(250.0) * mean / std
    return mean * sr * sr / (sr * sr + 1.0)


def replay(prices: np.ndarray, start: int, end: int) -> dict:
    """Replay and capture daily PnL plus the dollar position vector."""
    rh.reset()
    cash = value = comm = 0.0
    pos = np.zeros(51)
    pnl, net, gross, dollars_hist = [], [], [], []
    for t in range(start - 1, end + 1):
        hist = prices[:, :t]
        cur = hist[:, -1]
        if t < end:
            req = rh.getMyPosition(hist)
            lim = (LIMIT / cur).astype(int)
            new = np.clip(req, -lim, lim).astype(float)
        else:
            new = pos.copy()
        delta = new - pos
        cash -= cur.dot(delta) + comm
        comm = float(np.sum(cur * np.abs(delta) * COMM))
        pos = new
        d = pos * cur
        val = cash + pos.dot(cur)
        pnl.append(val - value)
        value = val
        net.append(float(d.sum()))
        gross.append(float(np.abs(d).sum()))
        dollars_hist.append(d.copy())
    return {
        "pnl": np.asarray(pnl[1:]),
        "net": np.asarray(net[1:]),
        "gross": np.asarray(gross[1:]),
        "dollars": np.asarray(dollars_hist[1:]),
    }


def regress(y: np.ndarray, x: np.ndarray) -> dict:
    """OLS y = a + b x with t-stat and R^2."""
    n = len(y)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(xc @ xc)
    if denom < 1e-18:
        return {"beta": 0.0, "t": 0.0, "r2": 0.0}
    b = float(xc @ yc) / denom
    resid = yc - b * xc
    s2 = float(resid @ resid) / max(n - 2, 1)
    se = sqrt(max(s2 / denom, 1e-300))
    ss_tot = float(yc @ yc)
    return {
        "beta": b,
        "t": b / se if se > 0 else 0.0,
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0,
    }


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    log_ret = np.diff(np.log(prices), axis=1)  # [inst, day-1]

    out = {}
    for tag, (s, e) in (("train", (TRAIN_START, TRAIN_END)),
                        ("test", (TEST_START, TEST_END))):
        r = replay(prices, s, e)
        pnl = r["pnl"]
        days = np.arange(s, e + 1)
        # market return on the day the PnL is earned: day t PnL comes from
        # the move from t-1 to t, i.e. log_ret column t-2 (0-indexed).
        cols = days - 2
        mkt = log_ret[1:, cols].mean(axis=0)   # equal-weight, excl. ALGO
        algo = log_ret[0, cols]

        rm = regress(pnl, mkt)
        ra = regress(pnl, algo)
        mu, sd = float(pnl.mean()), float(pnl.std())

        # Hedge simulation: remove the fitted market component entirely.
        hedged = pnl - rm["beta"] * (mkt - mkt.mean())
        hm, hs = float(hedged.mean()), float(hedged.std())

        print(f"=== {tag.upper()} {s}-{e} ===")
        print(f"  net $ exposure : mean {r['net'].mean():+10.0f}  "
              f"std {r['net'].std():9.0f}  |max| {np.abs(r['net']).max():9.0f}")
        print(f"  gross $ book   : mean {r['gross'].mean():10.0f}")
        print(f"  ALGO position  : always {np.abs(r['dollars'][:, 0]).max():.0f} "
              f"(unused hedge instrument)")
        print(f"  PnL vs market  : beta {rm['beta']:10.1f}  t {rm['t']:6.2f}  "
              f"R^2 {rm['r2']:.4f}")
        print(f"  PnL vs ALGO    : beta {ra['beta']:10.1f}  t {ra['t']:6.2f}  "
              f"R^2 {ra['r2']:.4f}")
        print(f"  score now      : {score(mu, sd):8.2f}  (mu {mu:7.2f} sd {sd:7.2f})")
        print(f"  score if market component removed: {score(hm, hs):8.2f}  "
              f"(mu {hm:7.2f} sd {hs:7.2f})")
        print()
        out[tag] = {
            "net_mean": float(r["net"].mean()), "net_std": float(r["net"].std()),
            "net_absmax": float(np.abs(r["net"]).max()),
            "gross_mean": float(r["gross"].mean()),
            "mkt_reg": rm, "algo_reg": ra,
            "score": score(mu, sd), "mu": mu, "sd": sd,
            "score_hedged": score(hm, hs), "mu_hedged": hm, "sd_hedged": hs,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "beta_exposure.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'beta_exposure.json'}")


if __name__ == "__main__":
    main()
