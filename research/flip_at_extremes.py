"""Flip-at-extremes vs flip-at-mean, on the 16 stable pairs.

User proposal: enter at the greatest stretch and switch entirely to the
opposite position at the other end of the spectrum (hold THROUGH the mean).
The current band policy instead flips at the zero crossing.

Two measurements decide it:
1. Overshoot statistics — after z crosses zero from an extreme, how often
   does it continue to the opposite extreme before returning?  OU-like
   spreads rarely do; oscillators usually do.
2. PnL — flip policies (enter |z|>=Z, reverse at opposite |z|>=Z) vs the
   shipped band, train 251-500 and test 501-750.
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

from research.fifty_pairs_lab import (
    STABLE_NAMES,
    band_sim,
    kalman_beta_path,
    stable_z_path,
    ticker_indices,
)
from research.pair_book_lab import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    account_two_leg,
    eval_score,
    load_raw_prices,
)

LEG = 10_000.0


def flip_sim(z, py, px, z_level, stop=99.0):
    """Enter at |z|>=z_level; hold through the mean; reverse at the
    opposite extreme.  Optional safety stop flattens at |z|>=stop."""
    T = len(py)
    sy = np.zeros(T)
    sx = np.zeros(T)
    side = 0
    for d in range(129, T):
        zd = z[d]
        az = abs(zd)
        if az >= stop:
            side = 0
        elif side == 0:
            if az >= z_level:
                side = -int(np.sign(zd))
        else:
            # reverse only when z is stretched on the OTHER side
            if az >= z_level and int(-np.sign(zd)) != side:
                side = int(-np.sign(zd))
        if side != 0:
            sy[d] = int(side * LEG / py[d])
            sx[d] = int(-side * LEG / px[d])
    return sy, sx


def overshoot_stats(z, hi=1.5, back=1.0):
    """After z leaves +hi and crosses 0: does it reach -hi before +back?"""
    events = 0
    overshoots = 0
    days_to_opp = []
    for sign in (1.0, -1.0):
        d = 130
        T = len(z)
        while d < T:
            if sign * z[d] >= hi:
                # find the zero crossing
                c = d
                while c < T and sign * z[c] > 0:
                    c += 1
                if c >= T:
                    break
                events += 1
                # from the crossing: opposite extreme vs bounce back
                e = c
                hit = None
                while e < T:
                    if -sign * z[e] >= hi:
                        hit = True
                        break
                    if sign * z[e] >= back:
                        hit = False
                        break
                    e += 1
                if hit:
                    overshoots += 1
                    days_to_opp.append(e - c)
                d = e + 1
            else:
                d += 1
    frac = overshoots / max(events, 1)
    med = float(np.median(days_to_opp)) if days_to_opp else float("nan")
    return events, frac, med


def main() -> None:
    prices = load_raw_prices()
    idx = ticker_indices()
    pairs = [(idx[a], idx[b]) for a, b in STABLE_NAMES]
    lp = np.log(prices)

    paths = {}
    ev_tot = 0
    ov_w = 0.0
    meds = []
    for p in pairs:
        beta = kalman_beta_path(lp[p[0]], lp[p[1]])
        z = stable_z_path(lp[p[0]], lp[p[1]], beta, 145)
        paths[p] = z
        ev, frac, med = overshoot_stats(z)
        ev_tot += ev
        ov_w += frac * ev
        if not np.isnan(med):
            meds.append(med)
    print(
        f"overshoot check (cross 0 from |z|>=1.5): {ev_tot} events, "
        f"{ov_w / max(ev_tot, 1):.0%} reach the OPPOSITE 1.5 before "
        f"bouncing back to 1.0; median days to opposite extreme "
        f"{np.median(meds):.0f}"
    )

    print("\npolicy totals (16 stable pairs, train / test):")
    rows = [("shipped band 0.27-4.0 (flip at mean)", None)]
    for zl in (1.0, 1.5, 2.0, 2.5):
        rows.append((f"flip at extremes Z={zl}", zl))
    for label, zl in rows:
        tr_sum = te_sum = None
        for p in pairs:
            py, px = prices[p[0]], prices[p[1]]
            if zl is None:
                sy, sx, _ = band_sim(paths[p], py, px, 0.27)
            else:
                sy, sx = flip_sim(paths[p], py, px, zl)
            tr = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
            te = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
            tr_sum = tr if tr_sum is None else tr_sum + tr
            te_sum = te if te_sum is None else te_sum + te
        print(
            f"  {label:40s} train {eval_score(tr_sum.mean(), tr_sum.std()):+8.1f} "
            f"test {eval_score(te_sum.mean(), te_sum.std()):+8.1f}"
        )


if __name__ == "__main__":
    main()
