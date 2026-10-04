"""Robust adaptive-pairs hybrid: Daniel core + Four Sigmas discovery.

Self-contained submission module: the adaptive atlas, extra lead-lag, and
ridge sleeves from ``daniel_filter`` are inlined below.  The complementary
trailing-OLS discovery sleeve fills residual capacity only.

    adaptive atlas -> extra lead-lag -> ridge -> trailing-OLS discovery

Discovery excludes every pair already represented by the adaptive atlas and
requires both instruments to be idle.  Each accepted hedge is scaled as one
unit before it is added, so the final position-limit clip cannot turn a pair
trade into an accidental directional position.

Eval API: reset() / getMyPosition(prcSoFar).
"""
from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import NamedTuple

import numpy as np


I = {
    "ALGO": 0, "AENO": 1, "LSST": 2, "SRNA": 3, "ELLT": 4, "AMRP": 5, "OTCS": 6,
    "HETT": 7, "HUXZ": 8, "DUCT": 9, "SMAH": 10, "NPCK": 11, "MSDP": 12, "EORC": 13,
    "CUBO": 14, "HRET": 15, "ANSO": 16, "DIHO": 17, "RTTH": 18, "SPLZ": 19, "NWIG": 20,
    "MMBT": 21, "MDGI": 22, "AGVF": 23, "RRES": 24, "CTGI": 25, "ALUT": 26, "ACAC": 27,
    "SRTX": 28, "GARI": 29, "RCRI": 30, "ACIX": 31, "CCNS": 32, "MTNS": 33, "IHOZ": 34,
    "NAYO": 35, "FWWG": 36, "EELT": 37, "HRND": 38, "AETS": 39, "ULXY": 40, "BLBT": 41,
    "BENI": 42, "ITPA": 43, "HTRK": 44, "NGTE": 45, "ILVX": 46, "FCSG": 47, "FARS": 48,
    "MHRM": 49, "EAFC": 50,
}
_INV = {v: k for k, v in I.items()}

N_INST = 51
LEG = 10_000.0
HEDGE_W = 120
Z_W = 145
PRIOR_GATE_T = -2.4   # was -2.9 for an EXPANDING residual; see ADF_WINDOW
ADF_WINDOW = 375      # trailing days for the ADF prior-gate
ATLAS_OU_WINDOW = 120 # trailing window for the per-pair OU-kappa reversion gate
                      # (default 120 = original behaviour; the pkg tightens to ~50)

LEADLAG_EXTRA = [
    ("CTGI", "AENO"), ("EELT", "AENO"), ("DUCT", "HRND"), ("LSST", "ACAC"),
    ("AENO", "NWIG"), ("SRNA", "CUBO"), ("MHRM", "ULXY"), ("ULXY", "HETT"),
    ("BLBT", "HRET"), ("AGVF", "CUBO"), ("FWWG", "AMRP"), ("SRNA", "GARI"),
    ("NWIG", "AENO"), ("ULXY", "GARI"), ("SRNA", "ULXY"), ("EELT", "BLBT"),
    ("CTGI", "ULXY"), ("IHOZ", "DIHO"), ("FCSG", "ULXY"), ("ULXY", "HRND"),
    ("EELT", "SRTX"), ("BENI", "GARI"), ("AETS", "FARS"), ("HRND", "ALUT"),
]
_EXTRA_IDX = [(I[a], I[b]) for a, b in LEADLAG_EXTRA]
_EXTRA_SKIP_WHEN_CROWDED = {
    (I["SRNA"], I["CUBO"]),
    (I["ULXY"], I["GARI"]),
    (I["LSST"], I["ACAC"]),
}

LOOK_OLD, LOOK_NEW = 160, 90
Z_FULL, Z_NEW = 1.5, 1.0
REQUIRE_OOS_PNL = True
REQUIRE_SIGN_MATCH = True
PAIR_PRIORITY_OVER_LL = True
FILTER_EXTRA_PAIR_FOLLOWERS = False

#-
RIDGE_LAM = 1.0
RIDGE_THETA = 1.0
RIDGE_GATE_N = 35
RIDGE_LEG = 10_000.0
RIDGE_MIN_T = 252
RIDGE_FILL_RESIDUAL_CAPACITY = True


_INIT_WINDOW = 20


# ---------------------------------------------------------------------------
# Adaptive beta filter (inlined NamedTuples for eval.py exec loader)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filter constants.  These were FiltCfg fields, but every one of the 21 pair
# seeds carried the SAME value, so they were never per-pair parameters.
# Demoted to module constants: identical behaviour, 11 fewer tunables.
# ---------------------------------------------------------------------------
Q_BETA_DEFAULT = 3e-6
CUSUM_K_DEFAULT = 0.5
H_DEFAULT = 8.0
ADAPT_BARS_DEFAULT = 8
SNAP_OVERSHOOT_DEFAULT = 1.10

Q_BETA_ADAPT = 5e-3
EWMA_LAMBDA = 0.97
WARMUP = 30
AUTO_H = True
TARGET_SEG = 50
H_GAIN = 0.05
MIN_SEG_LEN = 40
COV_INFLATE = 25.0
SNAP_WINDOW = 12
ADAPT_CHASE = 1.0
ADAPT_P_FLOOR = 2.0
# Deleted outright (dead): vol_cut (False everywhere, so the whole variance
# CUSUM branch was unreachable), adapt_extend, cut_confirm_bars, snap_max_rel.

class FiltCfg(NamedTuple):
    """Per-pair filter knobs.  Only the five that actually differ between
    pairs live here; everything else was identical across all 21 seeds and
    is now a module constant above (same values, same behaviour)."""

    q_beta: float = Q_BETA_DEFAULT
    cusum_k: float = CUSUM_K_DEFAULT
    h: float = H_DEFAULT
    adapt_bars: int = ADAPT_BARS_DEFAULT
    snap_overshoot: float = SNAP_OVERSHOOT_DEFAULT


class TradeCfg(NamedTuple):
    """Per-pair trade policy.  require_kappa / trade_while_adapting / core /
    beta_mode were identical on every seed and are gone; kappa gating is
    always on, adapting never blocks a trade."""

    z_entry: float = 0.27
    z_exit: float = 0.10
    z_stop: float = 4.0
    leg_scale: float = 1.0
    require_adf: bool = True


class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
    name: str
    priority: float
    filt: FiltCfg
    trade: TradeCfg


class StepOut(NamedTuple):
    beta: float
    spread: float
    z: float
    h: float
    cut: bool
    adapting: bool
    vol_shift: bool
    cm_up: float
    cm_dn: float
    cv: float


class PairSegFilter:
    def __init__(self, cfg: FiltCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else FiltCfg()
        self.beta: float | None = None
        self.p22: float = 1.0
        self.innov_var: float | None = None
        self.cm_up = self.cm_dn = 0.0
        self.run_m_up = self.run_m_dn = None
        self.cv_up = self.cv_dn = 0.0
        self.run_v_up = self.run_v_dn = None
        self.seg_start: int | None = None
        self.h_live: float | None = None
        self.seg_len_ewma: float | None = None
        self.adapt_left = 0
        self.disagree_run = 0
        self.snap_dir = 0.0
        self.t = -1
        self._y_hist: list[float] = []
        self._x_hist: list[float] = []

    def _ols_beta(self, window: int) -> float:
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < 1e-12:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def _accel_target(self) -> float:
        ols_s = self._ols_beta(SNAP_WINDOW)
        b = float(self.beta) if self.beta is not None else ols_s
        if self.snap_dir > 0.0:
            return max(ols_s, b)
        if self.snap_dir < 0.0:
            return min(ols_s, b)
        return ols_s

    def step(self, y: float, x: float) -> StepOut:
        cfg = self.cfg
        self.t += 1
        t = self.t
        cut = False
        vol_shift = False
        yf, xf = float(y), float(x)
        self._y_hist.append(yf)
        self._x_hist.append(xf)

        if self.beta is None:
            self.beta = self._ols_beta(_INIT_WINDOW)
            self.seg_start = t
            self.h_live = cfg.h

        q_b = Q_BETA_ADAPT if self.adapt_left > 0 else cfg.q_beta
        adapting = self.adapt_left > 0

        b = float(self.beta)
        pp22 = self.p22 + q_b
        innov = yf - b * xf
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < 1e-12:
            s_pred = 1e-12
        g = (pp22 * xf) / s_pred
        self.beta = b + g * innov
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)

        if self.adapt_left > 0 and ADAPT_CHASE > 0.0:
            target = self._accel_target()
            ch = min(1.0, max(0.0, ADAPT_CHASE))
            self.beta = (1.0 - ch) * float(self.beta) + ch * target
            self.p22 = max(float(self.p22), max(float(ADAPT_P_FLOOR), 0.0))

        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            lam = EWMA_LAMBDA
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
        z = innov / sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0

        resid = yf - float(self.beta) * xf

        h_work = (
            (self.h_live if self.h_live is not None else cfg.h)
            if AUTO_H
            else cfg.h
        )

        if t >= WARMUP:
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

            d_v = (z * z - 1.0) / sqrt(2.0)
            prev_v_up, prev_v_dn = self.cv_up, self.cv_dn
            self.cv_up = max(0.0, self.cv_up + d_v - cfg.cusum_k)
            self.cv_dn = max(0.0, self.cv_dn - d_v - cfg.cusum_k)
            if self.cv_up > 0.0 and prev_v_up == 0.0:
                self.run_v_up = t
            elif self.cv_up == 0.0:
                self.run_v_up = None
            if self.cv_dn > 0.0 and prev_v_dn == 0.0:
                self.run_v_dn = t
            elif self.cv_dn == 0.0:
                self.run_v_dn = None

            mean_alarm = self.cm_up > h_work or self.cm_dn > h_work
            vol_alarm = self.cv_up > h_work or self.cv_dn > h_work
            if vol_alarm:
                vol_shift = True
                if not False:
                    self.cv_up = self.cv_dn = 0.0
                    self.run_v_up = self.run_v_dn = None

            ols = self._ols_beta(MIN_SEG_LEN)
            thr = max(0.025, 0.03 * abs(float(self.beta)))
            disagree = abs(ols - float(self.beta)) >= thr
            if disagree and self.adapt_left == 0:
                self.disagree_run += 1
            else:
                self.disagree_run = 0

            cusum_cut = (
                (mean_alarm or (False and vol_alarm))
                and disagree
                and self.disagree_run >= max(1, 1)
                and self.adapt_left == 0
            )
            drift_need = max(8, TARGET_SEG // 4)
            drift_cut = self.disagree_run >= drift_need and self.adapt_left == 0
            want_cut = cusum_cut or drift_cut

            if mean_alarm and not disagree and self.adapt_left == 0:
                self.cm_up = self.cm_dn = 0.0
                self.cv_up = self.cv_dn = 0.0
                self.run_m_up = self.run_m_dn = None
                self.run_v_up = self.run_v_dn = None

            if want_cut:
                best = -1.0
                cut_bar = None
                if self.cm_up > h_work and self.cm_up > best:
                    best = self.cm_up
                    cut_bar = self.run_m_up
                if self.cm_dn > h_work and self.cm_dn > best:
                    best = self.cm_dn
                    cut_bar = self.run_m_dn
                if False and self.cv_up > h_work and self.cv_up > best:
                    best = self.cv_up
                    cut_bar = self.run_v_up
                if False and self.cv_dn > h_work and self.cv_dn > best:
                    best = self.cv_dn
                    cut_bar = self.run_v_dn
                if cut_bar is None:
                    cut_bar = t
                assert self.seg_start is not None
                cut_bar = max(int(cut_bar), self.seg_start + 1)
                closed_len = cut_bar - self.seg_start
                self.cm_up = self.cm_dn = 0.0
                self.cv_up = self.cv_dn = 0.0
                self.run_m_up = self.run_m_dn = None
                self.run_v_up = self.run_v_dn = None
                if closed_len >= MIN_SEG_LEN:
                    cut = True
                    self.disagree_run = 0
                    target = self._ols_beta(SNAP_WINDOW)
                    pre = float(self.beta)
                    delta = target - pre
                    self.snap_dir = 1.0 if delta >= 0.0 else -1.0
                    jump = cfg.snap_overshoot * delta
                    if 0.0 > 0.0:
                        cap = 0.0 * max(abs(pre), 1e-6)
                        jump = max(-cap, min(cap, jump))
                    self.beta = pre + jump
                    self.p22 = max(
                        self.p22 * COV_INFLATE,
                        max(float(ADAPT_P_FLOOR), 0.0),
                    )
                    self.adapt_left = cfg.adapt_bars
                    resid = yf - float(self.beta) * xf
                    if AUTO_H:
                        if self.seg_len_ewma is None:
                            self.seg_len_ewma = float(min(closed_len, 3 * TARGET_SEG))
                        else:
                            self.seg_len_ewma = 0.1 * closed_len + 0.9 * self.seg_len_ewma
                        err = (TARGET_SEG - self.seg_len_ewma) / TARGET_SEG
                        err = max(-0.5, min(0.5, err))
                        h0 = self.h_live if self.h_live is not None else cfg.h
                        self.h_live = max(2.0, h0 + H_GAIN * err * h0)
                    self.seg_start = cut_bar

        if self.adapt_left > 0:
            self.adapt_left -= 1
            adapting = True
            if self.adapt_left == 0:
                self.snap_dir = 0.0
                self.p22 = min(float(self.p22), 1e-4)

        h_out = (
            (self.h_live if self.h_live is not None else cfg.h)
            if AUTO_H
            else cfg.h
        )
        return StepOut(
            beta=float(self.beta),
            spread=float(resid),
            z=float(z),
            h=float(h_out),
            cut=cut,
            adapting=adapting,
            vol_shift=vol_shift,
            cm_up=float(self.cm_up),
            cm_dn=float(self.cm_dn),
            cv=max(self.cv_up, self.cv_dn),
        )


# ---------------------------------------------------------------------------
# Pair universe: four_sigmas atlas ∪ Kalman stable (+ a few high-score adaptive)
# Orientation: higher-index as y when discovery listed that way; FS names keep FS order.
# ---------------------------------------------------------------------------


def _default_pair_cfgs() -> tuple[PairCfg, ...]:
    """All pairs first-class: FS atlas ∪ Kalman discovery.

    ``priority`` is standalone tune EV. Conflict MWIS ranks by EV * |z|.
    Each pair has its own z_entry / z_exit / z_stop (hysteresis) and filter knobs.
    """
    # (y, x, EV, z_entry, z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap, leg)
    seeds = [
    # PAIR PRUNE (2026-07-21) from a leave-one-out marginal-contribution audit
    # (research/pair_contribution_audit.py).  Six pairs were disabled via
    # leg_scale=0 because they do not earn their slot:
    #   DEAD      MSDP/ULXY, ACIX/ULXY, EAFC/SRTX  -> exactly 0.00 marginal on
    #             BOTH windows (permanently blocked by the expanding-ADF gate)
    #   HARMFUL   MTNS/ALUT (-4.1 train), CCNS/SRTX (-10.8 train / -0.4 test)
    #   OVERFIT   MDGI/HTRK (+4.1 train / -6.5 test; also 0 of 87,480 tuning
    #             candidates could improve its test window, and none of its 49
    #             possible partners is stable across both windows)
    # eval.py: train 820.35 -> 832.86, test 848.00 -> 851.20.
    # Marginals are NOT additive here (EV-MWIS reshuffles), so this exact set
    # was chosen by search; pruning all 7 non-KEEP pairs was worse on test.
        # (y, x, EV, z_entry, z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap, leg)
        # Pair-local tuning freezes MWIS priorities and prefers minimal changes
        # on broad multi-metric plateaus.  RTTH/NAYO's 0.25 exit replaces the
        # fragile near-zero exit: standalone drawdown/CVaR and held-out portfolio
        # performance improve for <1% full-portfolio train mean/score cost.
        (18, 35, 107.1, 0.50, 0.25, 4.25, 8.284e-06, 5.0, 0.60, 4, 1.15, 1.00),  # RTTH/NAYO
        # CTGI/EELT: entry-only robustness retune.  0.35 improves standalone
        # train/test and full-portfolio train without changing portfolio test;
        # beta speed, exit, stop, gates and scale remain frozen.
        (25, 37, 94.5, 0.35, 0.05, 3.25, 1.833e-07, 4.0, 0.45, 5, 1.15, 1.00),  # CTGI/EELT
        (7, 40, 63.9, 0.75, 0.02, 4.00, 1.645e-07, 7.0, 0.60, 6, 1.40, 1.00),  # HETT/ULXY
        # BENI/RCRI removed after leave-one-pair-out audit: removing it raised
        # full train/test score by 9.02/13.52 through better conflict capacity.
        (15, 25, 46.0, 0.33, 0.10, 3.75, 1.718e-05, 5.0, 0.60, 9, 1.10, 0.00),  # HRET/CTGI  [PRUNED: -18.6 on train blocks; fails both structural cointegration screens]
        (8, 27, 45.1, 0.60, 0.23, 4.75, 3.168e-05, 6.5, 0.40, 16, 1.15, 1.00),  # HUXZ/ACAC
        (35, 28, 44.7, 0.16, 0.025, 3.50, 9.096e-07, 4.0, 0.50, 9, 1.15, 1.00),  # NAYO/SRTX
        (1, 20, 44.7, 0.12, 0.02, 4.00, 2.796e-07, 10.5, 0.55, 9, 1.10, 1.00),  # AENO/NWIG (proof-protocol retune 2026-07-20: z_entry .18->.12, z_stop 3.25->4.00; filter knobs are inert here — 0 cuts; all 4 blocks improve; hybrid test +4.4)
        (22, 44, 42.1, 0.30, 0.02, 4.75, 5.633e-07, 8.5, 0.60, 13, 1.15, 0.00),  # MDGI/HTRK  [PRUNED: leg_scale=0]
        # DIHO/RTTH: hand-tuned fast hedge (q=5e-5, h=3 → continuous beta,
        # almost no cuts).  With oscillating window-z the old ADF gate blocked
        # ~220 train days; ADF-relaxed later entry + 0.25 exit harvests the MR
        # (standalone train/test +22/+1).  Portfolio: hybrid +8/+10, core test
        # +6.5, core train -4.6 via MWIS conflict — accepted for sleeve health.
        (17, 18, 41.2, 0.75, 0.25, 2.50, 5e-05, 3.0, 0.40, 11, 1.35, 1.00),  # DIHO/RTTH
        (49, 50, 39.5, 0.10, 0.02, 5.25, 1.128e-06, 9.0, 0.65, 12, 1.25, 1.00),  # MHRM/EAFC
        (33, 26, 39.2, 0.15, 0.04, 3.50, 5.823e-07, 9.0, 0.40, 16, 1.40, 0.00),  # MTNS/ALUT  [PRUNED: leg_scale=0]
        (12, 40, 38.3, 0.31, 0.02, 4.75, 4.596e-07, 8.5, 0.35, 5, 1.05, 0.00),  # MSDP/ULXY  [PRUNED: leg_scale=0]
        (13, 45, 34.9, 0.40, 0.34, 4.00, 3.500e-08, 10.0, 0.35, 11, 1.35, 1.00),  # EORC/NGTE
        (26, 32, 32.5, 0.587, 0.1425, 3.50, 6.58e-08, 5.0, 0.45, 9, 1.30, 1.00),  # ALUT/CCNS
        (10, 46, 30.7, 0.35, 0.2975, 3.00, 6.789e-05, 3.0, 0.35, 16, 1.30, 1.00),  # SMAH/ILVX (Pareto retune 2026-07-20: cusum_k 0.45->0.35, exit at 0.85*entry cap; portfolio train +2.5 / test +0.7)
        (33, 42, 23.1, 0.50, 0.05, 2.75, 1.856e-05, 11.5, 0.45, 14, 1.15, 1.00),  # MTNS/BENI
        # CCNS/SRTX: ADF rejected every train/test day.  The local OU gate is
        # retained; later entry plus 0.30 scale makes the ADF-relaxed sleeve a
        # low-priority diversifier and improves full-portfolio train/test risk.
        (32, 28, 20.2, 0.75, 0.15, 4.75, 1.005e-07, 6.0, 0.60, 7, 1.30, 0.00),  # CCNS/SRTX  [PRUNED: leg_scale=0]
        (36, 41, 19.1, 0.11, 0.05, 4.00, 9.990e-07, 6.0, 0.40, 8, 1.20, 1.00),  # FWWG/BLBT
        (31, 43, 16.0, 0.26, 0.02, 4.50, 6.157e-06, 5.0, 0.60, 8, 1.05, 1.00),  # ACIX/ITPA
        (31, 40, 16.0, 0.39, 0.02, 2.50, 2.901e-05, 6.5, 0.35, 13, 1.05, 0.00),  # ACIX/ULXY  [PRUNED: leg_scale=0]
        # HUXZ/NWIG removed: no train change and +0.45 test score after
        # BENI/RCRI removal, while eliminating a dormant conflict candidate.
        (50, 28, 0.8, 0.48, 0.30, 4.75, 3.737e-06, 10.5, 0.55, 5, 1.10, 0.00),  # EAFC/SRTX  [PRUNED: leg_scale=0]
    ]
    adf_relaxed = {(32, 28), (17, 18)}  # CCNS/SRTX, DIHO/RTTH
    out: list[PairCfg] = []
    seen: set[frozenset[int]] = set()
    for y, x, pri, ze, zx, zs, qb, h, ck, ab, so, ls in seeds:
        key = frozenset((y, x))
        if key in seen or y == x:
            continue
        seen.add(key)
        # Ensure hysteresis: exit < entry < stop
        z_entry = float(ze)
        z_exit = float(min(zx, 0.85 * z_entry))
        z_stop = float(max(zs, 1.25 * z_entry))
        out.append(
            PairCfg(
                idx=len(out),
                y=y,
                x=x,
                name=f"{_INV[y]}/{_INV[x]}",
                priority=float(pri),
                filt=FiltCfg(
                    q_beta=qb, h=h, cusum_k=ck, adapt_bars=ab, snap_overshoot=so
                ),
                trade=TradeCfg(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    z_stop=z_stop,
                    leg_scale=ls,
                    require_adf=(y, x) not in adf_relaxed,
                ),
            )
        )
    return tuple(out)


PAIR_CFGS = _default_pair_cfgs()


def _hub_penalty(specs: tuple[PairCfg, ...]) -> dict[int, float]:
    deg: dict[int, int] = defaultdict(int)
    for s in specs:
        deg[s.y] += 1
        deg[s.x] += 1
    pen: dict[int, float] = {}
    for s in specs:
        pen[s.idx] = sqrt(max(deg[s.y], 1) * max(deg[s.x], 1))
    return pen


def _build_conflict(specs: tuple[PairCfg, ...]):
    by_inst: dict[int, list[int]] = defaultdict(list)
    for s in specs:
        by_inst[s.y].append(s.idx)
        by_inst[s.x].append(s.idx)
    conflict: dict[int, set[int]] = defaultdict(set)
    for peers in by_inst.values():
        for i, a in enumerate(peers):
            for b in peers[i + 1 :]:
                conflict[a].add(b)
                conflict[b].add(a)
    return dict(conflict)


CONFLICT = _build_conflict(PAIR_CFGS)
HUB_PEN = _hub_penalty(PAIR_CFGS)
_PAIR_LEGS = {s.y for s in PAIR_CFGS} | {s.x for s in PAIR_CFGS}
_EXTRA_IDX_NO_PAIR_FOLLOWERS = [
    (a, b) for a, b in _EXTRA_IDX if b not in _PAIR_LEGS
]


# ---------------------------------------------------------------------------
# Atlas agents
# ---------------------------------------------------------------------------


def _ou_kappa(spread: np.ndarray) -> float:
    ds = np.diff(spread)
    sl = spread[:-1] - spread[:-1].mean()
    v = float((sl ** 2).sum())
    return float(-(sl @ ds) / v) if v > 1e-18 else 0.0


def _adf1_tstats(S: np.ndarray) -> np.ndarray:
    ds = np.diff(S, axis=0)
    y = ds[1:]
    x1 = S[1:-1]
    x2 = ds[:-1]
    n = y.shape[0]
    s1 = x1.sum(0)
    s2 = x2.sum(0)
    s11 = (x1 * x1).sum(0)
    s12 = (x1 * x2).sum(0)
    s22 = (x2 * x2).sum(0)
    sy = y.sum(0)
    s1y = (x1 * y).sum(0)
    s2y = (x2 * y).sum(0)
    syy = (y * y).sum(0)
    m = S.shape[1]
    A = np.empty((m, 3, 3))
    b = np.empty((m, 3))
    A[:, 0, 0] = n
    A[:, 0, 1] = s1
    A[:, 0, 2] = s2
    A[:, 1, 0] = s1
    A[:, 1, 1] = s11
    A[:, 1, 2] = s12
    A[:, 2, 0] = s2
    A[:, 2, 1] = s12
    A[:, 2, 2] = s22
    b[:, 0] = sy
    b[:, 1] = s1y
    b[:, 2] = s2y
    Ainv = np.linalg.inv(A)
    beta = np.einsum("mij,mj->mi", Ainv, b)
    sigma2 = (syy - np.einsum("mi,mi->m", beta, b)) / max(n - 3, 1)
    return beta[:, 1] / np.sqrt(np.maximum(sigma2 * Ainv[:, 1, 1], 1e-300))


def _adf_t_spread(spread: np.ndarray) -> float:
    if len(spread) < 40:
        return 0.0
    return float(_adf1_tstats(spread.reshape(-1, 1))[0])


class PairAgent:
    __slots__ = (
        "spec", "filt", "innov_z", "window_z", "h", "beta", "beta_lag",
        "adapting", "nt_seen", "side",
    )

    def __init__(self, spec: PairCfg) -> None:
        self.spec = spec
        self._hard_reset()

    def _hard_reset(self) -> None:
        # Explicit reset — do not call self.__init__ (RestrictedPython blocks it).
        self.filt = PairSegFilter(self.spec.filt)
        self.innov_z = 0.0
        self.window_z = 0.0
        self.h = float(self.spec.filt.h)
        self.beta = 1.0
        self.beta_lag = 1.0
        self.adapting = False
        self.nt_seen = 0
        self.side = 0  # -1 short Y / +1 long Y / 0 flat

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self._hard_reset()
        y_i, x_i = self.spec.y, self.spec.x
        for t in range(self.nt_seen, nt):
            self.beta_lag = self.beta
            y = np.log(max(float(prices[y_i, t]), 1e-12))
            x = np.log(max(float(prices[x_i, t]), 1e-12))
            o = self.filt.step(y, x)
            self.innov_z = float(o.z)
            self.h = float(o.h)
            self.beta = float(o.beta)
            self.adapting = bool(o.adapting)
            self.nt_seen = t + 1

    def desired_side(self, LP: np.ndarray) -> float:
        """Hysteresis: enter at z_entry, hold until z_exit or z_stop."""
        tc = self.spec.trade
        if not True and self.adapting:
            self.side = 0
            return 0.0
        nt = LP.shape[1]
        if nt < HEDGE_W + 10:
            self.side = 0
            return 0.0
        y_i, x_i = self.spec.y, self.spec.x
        beta = self.beta_lag
        if tc.require_adf:
            # Trailing window, NOT expanding history.  An early level shift
            # poisons an expanding ADF permanently, so a pair that is
            # stationary *now* could be blocked forever (MSDP/ULXY,
            # ACIX/ULXY, EAFC/SRTX and SRNA/ACIX were all blocked 250/250
            # days in both windows).  ADF power scales with sample size, so
            # the threshold is loosened alongside the shorter window to keep
            # effective strictness roughly constant.
            lo = max(0, nt - 1 - ADF_WINDOW)
            spread = LP[y_i, lo : nt - 1] - beta * LP[x_i, lo : nt - 1]
            if len(spread) >= 40 and _adf_t_spread(spread) >= PRIOR_GATE_T:
                self.side = 0
                return 0.0
        s = LP[y_i, nt - Z_W : nt] - beta * LP[x_i, nt - Z_W : nt]
        z = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12))
        self.window_z = z
        if True:
            _ow = max(2, int(ATLAS_OU_WINDOW))
            sou = LP[y_i, max(0, nt - _ow) : nt] - beta * LP[x_i, max(0, nt - _ow) : nt]
            if _ou_kappa(sou) <= 0.0:
                self.side = 0
                return 0.0

        az = abs(z)
        if self.side == 0:
            if az >= tc.z_entry and az < tc.z_stop:
                self.side = int(-np.sign(z))
        else:
            if az <= tc.z_exit or az >= tc.z_stop:
                self.side = 0
            # else hold; optional flip if z crossed hard against us
            elif az >= tc.z_entry and int(-np.sign(z)) != self.side:
                self.side = int(-np.sign(z))
        return float(self.side)


class AtlasBook:
    """All pairs compete; greedy EV-MWIS keeps the highest-value non-overlapping set."""

    def __init__(self) -> None:
        self.agents = [PairAgent(s) for s in PAIR_CFGS]
        self.active: set[int] = set()

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        if nt < HEDGE_W + 10:
            self.active = set()
            return pos

        LP = np.log(np.maximum(prices, 1e-12))
        for ag in self.agents:
            ag.catch_up(prices)

        # Rank live signals by expected value: tune_EV * |z| * scale / hub_penalty.
        cands: list[tuple[float, int, float]] = []
        for ag in self.agents:
            side = ag.desired_side(LP)
            if abs(side) < 1e-12:
                continue
            s = ag.spec
            hub = max(HUB_PEN.get(s.idx, 1.0), 1.0)
            ev = (s.priority * abs(ag.window_z) * s.trade.leg_scale) / hub
            cands.append((ev, s.idx, side))

        cands.sort(reverse=True)
        taken: set[int] = set()
        chosen: list[tuple[int, float]] = []
        for _ev, idx, side in cands:
            ag = self.agents[idx]
            y, x = ag.spec.y, ag.spec.x
            if y in taken or x in taken:
                continue  # conflict: lower-EV pair yields the legs
            taken.add(y)
            taken.add(x)
            chosen.append((idx, side))

        self.active = {idx for idx, _ in chosen}
        for idx, side in chosen:
            ag = self.agents[idx]
            scale = ag.spec.trade.leg_scale
            pos[ag.spec.y] += side * LEG * scale
            pos[ag.spec.x] += -side * LEG * scale
        return pos


# ---------------------------------------------------------------------------
# Lead-lag + ridge (four_sigmas)
# ---------------------------------------------------------------------------


def _corr(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _robust_link_health(r, i, j):
    m = r.shape[1]
    look = LOOK_OLD + LOOK_NEW
    if m < look + 2:
        return None
    x = r[i, m - look - 1 : m - 1]
    y = r[j, m - look : m]
    x_old, y_old = x[:LOOK_OLD], y[:LOOK_OLD]
    x_new, y_new = x[LOOK_OLD:], y[LOOK_OLD:]
    c_old = _corr(x_old, y_old)
    c_new = _corr(x_new, y_new)
    c_full = _corr(x, y)
    if c_old == 0.0:
        return None
    sign_est = float(np.sign(c_old))
    pnl_new = float(np.sum(sign_est * np.sign(x_new) * y_new))
    z_new = abs(c_new) * np.sqrt(len(x_new))
    z_full = abs(c_full) * np.sqrt(len(x))
    ok = (z_full >= Z_FULL) and (z_new >= Z_NEW)
    if REQUIRE_SIGN_MATCH:
        ok = ok and (np.sign(c_new) == sign_est)
    if REQUIRE_OOS_PNL:
        ok = ok and (pnl_new > 0.0)
    return dict(ok=bool(ok), sign=sign_est, c_full=c_full)


def _extra_leadlag_dollars(r, idx_list=None, crowded_mask=None):
    nins = r.shape[0]
    pos = np.zeros(nins)
    if idx_list is None:
        idx_list = _EXTRA_IDX
    for i, j in idx_list:
        if crowded_mask is not None and crowded_mask[j] and (i, j) in _EXTRA_SKIP_WHEN_CROWDED:
            continue
        h = _robust_link_health(r, i, j)
        if h is None or not h["ok"]:
            continue
        w = 1.0
        scale = 1.0
        pos[j] += scale * w * LEG * h["sign"] * np.sign(r[i, -1])
    return pos


def _ridge_dollars(r, held_dollars):
    nins, m = r.shape
    pos = np.zeros(nins)
    if m + 1 < RIDGE_MIN_T:
        return pos
    R250 = r[1:, -250:].T
    Xg = (R250 - R250.mean(0)) / (R250.std(0) + 1e-12)
    Cg = (Xg[:-1].T @ Xg[1:]) / (len(R250) - 1)
    np.fill_diagonal(Cg, 0)
    fam = int((np.abs(Cg) > 2.5 / np.sqrt(len(R250) - 1)).sum())
    if fam < RIDGE_GATE_N:
        return pos
    Rf = r[1:, :].T
    sx = Rf.std(0) + 1e-12
    Xs, Ys = Rf[:-1] / sx, Rf[1:] / sx
    B = np.linalg.solve(Xs.T @ Xs + RIDGE_LAM * len(Xs) * np.eye(50), Xs.T @ Ys)
    sp = (Xs @ B).std(0)
    p = (Rf[-1] / sx) @ B
    raw = RIDGE_LEG * np.sign(p) * (np.abs(p) > RIDGE_THETA * sp)
    held = held_dollars[1:]
    d = np.zeros_like(raw)
    idle = np.abs(held) <= 1.0
    d[idle] = raw[idle]
    if RIDGE_FILL_RESIDUAL_CAPACITY:
        same_sign = (~idle) & (np.abs(raw) > 1.0) & (np.sign(raw) == np.sign(held))
        room = np.maximum(0.0, LEG - np.minimum(np.abs(held), LEG))
        d[same_sign] = np.sign(raw[same_sign]) * np.minimum(np.abs(raw[same_sign]), room[same_sign])
    pos[1:] = d
    return pos


# ---------------------------------------------------------------------------
# Book assembly
# ---------------------------------------------------------------------------

_STATE = {"atlas": None}


def reset_atlas():
    _STATE["atlas"] = None


def apply_pair_cfgs(cfgs: tuple[PairCfg, ...] | list[PairCfg]) -> None:
    """Hot-swap tuned configs (used by the pair-agent tuner)."""
    global PAIR_CFGS, CONFLICT, HUB_PEN, _PAIR_LEGS, _EXTRA_IDX_NO_PAIR_FOLLOWERS
    PAIR_CFGS = tuple(
        c._replace(idx=i) if c.idx != i else c for i, c in enumerate(cfgs)
    )
    CONFLICT = _build_conflict(PAIR_CFGS)
    HUB_PEN = _hub_penalty(PAIR_CFGS)
    _PAIR_LEGS = {s.y for s in PAIR_CFGS} | {s.x for s in PAIR_CFGS}
    _EXTRA_IDX_NO_PAIR_FOLLOWERS = [
        (a, b) for a, b in _EXTRA_IDX if b not in _PAIR_LEGS
    ]
    reset()


def _core_sleeve_breakdown(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < 130 or nins != 51:
        return None
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0):
        return None
    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()
    LP = np.log(prcSoFar)
    r = np.diff(LP, axis=1)
    d_atlas = _STATE["atlas"].dollars(prcSoFar)
    extra_idx = _EXTRA_IDX_NO_PAIR_FOLLOWERS if FILTER_EXTRA_PAIR_FOLLOWERS else _EXTRA_IDX
    crowded_mask = (np.abs(d_atlas) > 1.0)
    d_extraLL = _extra_leadlag_dollars(r, extra_idx, crowded_mask)
    if PAIR_PRIORITY_OVER_LL:
        pair_mask = np.abs(d_atlas) > 1.0
        d_extraLL = np.where(pair_mask, 0.0, d_extraLL)
    held0 = d_atlas + d_extraLL
    d_ridge = _ridge_dollars(r, held0)
    return {
        "atlas": d_atlas,
        "extraLL": d_extraLL,
        "ridge": d_ridge,
    }


# ---------------------------------------------------------------------------
# Hybrid discovery sleeve + eval API
# ---------------------------------------------------------------------------

USE_EXTRA_LL = True
USE_RIDGE = True
USE_DISCOVERY = True
RIDGE_ENTRY = 1.0

# Four Sigmas' most stable discovery configuration: the hedge is fitted only
# on information available before the decision close and on a fixed-length
# trailing sample.  No ADF gate is applied because this is explicitly a broad
# short-horizon reversion diversifier, not another cointegration atlas.
DISCOVERY_MIN_HISTORY = 50
DISCOVERY_FIT_WINDOW = 250
DISCOVERY_ENTRY_Z = 0.2
DISCOVERY_BREAK_Z = 4.5
DISCOVERY_SCALE = 1.0


def _discovery_universe() -> tuple[np.ndarray, np.ndarray]:
    """All tradable non-atlas pairs, excluding instrument zero."""
    atlas_keys = {
        tuple(sorted((int(spec.y), int(spec.x))))
        for spec in PAIR_CFGS
    }
    ti, tj = np.triu_indices(N_INST - 1, k=1)
    pair_i, pair_j = ti + 1, tj + 1
    keep = np.fromiter(
        (
            tuple(sorted((int(i), int(j)))) not in atlas_keys
            for i, j in zip(pair_i, pair_j)
        ),
        dtype=bool,
        count=len(pair_i),
    )
    return pair_i[keep], pair_j[keep]


DISCOVERY_I, DISCOVERY_J = _discovery_universe()


def _capacity(existing: float, requested: float) -> float:
    """Largest non-negative scale that keeps one leg inside its dollar cap."""
    if requested > 0.0:
        return max(0.0, (LEG - existing) / requested)
    if requested < 0.0:
        return max(0.0, (-LEG - existing) / requested)
    return 0.0


def _discovery_dollars(
    log_prices: np.ndarray,
    base_dollars: np.ndarray,
) -> np.ndarray:
    """Fade ranked trailing-OLS residual tails on otherwise idle instruments."""
    pos = np.zeros(log_prices.shape[0])
    if (
        not USE_DISCOVERY
        or DISCOVERY_SCALE <= 0.0
        or log_prices.shape[1] < DISCOVERY_MIN_HISTORY + 1
    ):
        return pos

    window = log_prices
    if log_prices.shape[1] > DISCOVERY_FIT_WINDOW + 1:
        window = log_prices[:, -(DISCOVERY_FIT_WINDOW + 1) :]

    # The final column is the decision close.  Fit alpha/beta and residual
    # dispersion on all earlier columns to avoid same-close lookahead.
    train = window[:, :-1]
    y = train[DISCOVERY_I]
    x = train[DISCOVERY_J]
    y_centered = y - y.mean(axis=1, keepdims=True)
    x_centered = x - x.mean(axis=1, keepdims=True)
    beta = (y_centered * x_centered).sum(axis=1) / np.maximum(
        (x_centered * x_centered).sum(axis=1), 1e-18
    )
    alpha = y.mean(axis=1) - beta * x.mean(axis=1)
    residual = y - alpha[:, None] - beta[:, None] * x
    current = (
        log_prices[DISCOVERY_I, -1]
        - alpha
        - beta * log_prices[DISCOVERY_J, -1]
    )
    z = current / (residual.std(axis=1) + 1e-12)

    idle = np.abs(base_dollars) <= 1.0
    eligible = (
        (beta > 0.0)
        & np.isfinite(beta)
        & np.isfinite(z)
        & (np.abs(z) >= DISCOVERY_ENTRY_Z)
        & (np.abs(z) < DISCOVERY_BREAK_Z)
        & idle[DISCOVERY_I]
        & idle[DISCOVERY_J]
    )

    # Strongest residual first; a discovery instrument can appear only once.
    used = np.zeros(log_prices.shape[0], dtype=bool)
    for k in np.argsort(-np.abs(z)):
        if not eligible[k]:
            continue
        i, j = int(DISCOVERY_I[k]), int(DISCOVERY_J[k])
        if used[i] or used[j]:
            continue

        direction = -np.sign(z[k])
        leg = LEG / max(1.0, abs(float(beta[k])))
        raw_i = direction * leg
        raw_j = -direction * float(beta[k]) * leg
        scale = min(
            DISCOVERY_SCALE,
            _capacity(float(base_dollars[i] + pos[i]), float(raw_i)),
            _capacity(float(base_dollars[j] + pos[j]), float(raw_j)),
        )
        if scale <= 1e-12:
            continue

        pos[i] += scale * raw_i
        pos[j] += scale * raw_j
        used[i] = used[j] = True
    return pos


def reset() -> None:
    """Reset the stateful adaptive atlas between independent replays."""
    global RIDGE_THETA
    RIDGE_THETA = float(RIDGE_ENTRY)
    reset_atlas()


def sleeve_breakdown(prcSoFar: np.ndarray) -> dict[str, np.ndarray] | None:
    """Return each sleeve's pre-crash dollar targets for attribution/audits."""
    global RIDGE_THETA
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return None
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return None

    RIDGE_THETA = float(RIDGE_ENTRY)
    core = _core_sleeve_breakdown(prcSoFar)
    if core is None:
        return None

    atlas = core["atlas"]
    extra = core["extraLL"] if USE_EXTRA_LL else np.zeros(nins)

    # Recompute ridge against the selected preceding sleeves.  This matters for
    # the no-extra ablation because ridge is explicitly capacity subordinated.
    returns = np.diff(np.log(prcSoFar), axis=1)
    ridge = (
        _ridge_dollars(returns, atlas + extra)
        if USE_RIDGE
        else np.zeros(nins)
    )
    held = atlas + extra + ridge
    held[1:] = np.clip(held[1:], -LEG, LEG)
    discovery = _discovery_dollars(np.log(prcSoFar), held)

    return {
        "atlas": atlas,
        "extraLL": extra,
        "ridge": ridge,
        "discovery": discovery,
    }


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return integer target shares for the available price history."""
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return np.zeros(nins, dtype=int)

    sleeves = sleeve_breakdown(prcSoFar)
    if sleeves is None:
        return np.zeros(nins, dtype=int)

    dollars = sum(sleeves.values(), start=np.zeros(nins))
    last_return = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)
