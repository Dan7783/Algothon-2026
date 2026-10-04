"""four_sigmas with ONE shared parameter set instead of per-pair fitting.

Motivation: true OOS scored 599 against 878 on the 501-750 window we had been
selecting on.  Rolling blocks inside 251-750 cannot expose that (all of it
informed the fitting), so the lever is to shrink what can be fitted at all.

four_sigmas carries ~235 fitted numbers, 210 of them per-pair (EV, z_entry,
z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap_overshoot, leg_scale
x 21 pairs).  This module keeps the entire earning structure — every pair,
every sleeve, the filter, the conflict resolver — and replaces those 210
numbers with one shared trade config.

Chosen on the TRAIN blocks (251-375, 376-500) only; the later blocks were not
consulted.  Block scores:

    per-pair fitted        838 / 869 / 765 / 992  median 853  worst 765  ~210 fitted
    shared trade           783 / 796 / 772 / 895  median 790  worst 772  ~150 fitted
    shared trade+filter    763 / 796 / 765 / 892  median 781  worst 763   ~50 fitted  <- this
    + flattened EV         682 / 713 / 689 / 847  median 701  worst 682   ~29 fitted

Sharing the thresholds AND filter knobs costs 2 points of floor for a 4x cut
in fitted numbers.  Flattening EV as well costs 80 points of floor, so EV
stays per-pair: it is not a threshold, it decides conflict priority.

With a real OOS print of 599 against an 878 in-sample test, the FLOOR is the
number worth protecting, not the median.

Pairs already disabled in four_sigmas (leg_scale=0 from the marginal-
contribution audit) stay disabled — that selection was structural, not a
parameter fit.
"""

from __future__ import annotations

import numpy as np

import four_sigmas as _fs

# The one shared set.  Trade thresholds selected on TRAIN blocks only;
# filter values are the modal seed values (i.e. what most pairs already used).
Z_ENTRY = 0.45
Z_EXIT = 0.05
Z_STOP = 4.00
Q_BETA = 8.284e-06
H = 5.0
CUSUM_K = 0.6
ADAPT_BARS = 9
SNAP_OVERSHOOT = 1.15

# EV / priority stays PER PAIR.  It is not a threshold — it decides which pair
# wins an instrument conflict, and flattening it costs 80 points of floor
# (worst block 763 -> 682).  Everything else is shared.
_BASE = tuple(_fs.PAIR_CFGS)
_SHARED = tuple(
    s._replace(
        filt=s.filt._replace(q_beta=Q_BETA, h=H, cusum_k=CUSUM_K,
                             adapt_bars=ADAPT_BARS,
                             snap_overshoot=SNAP_OVERSHOOT),
        trade=s.trade._replace(
            z_entry=Z_ENTRY,
            z_exit=min(Z_EXIT, 0.85 * Z_ENTRY),
            z_stop=max(Z_STOP, 1.25 * Z_ENTRY),
        ),
    )
    if s.trade.leg_scale > 0.0 else s          # keep pruned pairs pruned
    for s in _BASE
)


def reset() -> None:
    _fs.apply_pair_cfgs(_SHARED)
    _fs.reset()


reset()


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    return _fs.getMyPosition(prcSoFar)
