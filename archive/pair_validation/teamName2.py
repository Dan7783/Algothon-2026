"""four_sigmas book with adaptive-beta atlas + expanded known pairs.

Atlas Kalman beta is replaced by per-pair sticky PairSegFilter agents
(log prices). The atlas includes four_sigmas pairs AND Kalman-discovery
pairs with changing beta. Live conflicts are resolved by greedy EV-MWIS:
rank by (tune_EV * |z| * leg_scale / hub_penalty) and keep the highest-
value non-overlapping set. Non-conflicting pairs all trade.

Core lead-lag, extra lead-lag, and ridge sleeves are unchanged from
four_sigmas. Pair-priority still overrides LL on live atlas legs.

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
PRIOR_GATE_T = -2.9
GATE_PAIR_ZMAX = 4.0

LEADLAG = [
    ("DUCT", "HRET"), ("DUCT", "AMRP"), ("ELLT", "ULXY"),
    ("CCNS", "CUBO"), ("ELLT", "GARI"), ("DIHO", "HRND"),
]
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

LL_LOOK = 250
GATE_LL_Z = 0.5
RHO_HI, RHO_CAP, RHO_MIN_SCALE = 0.18, 0.30, 0.35
LOOK_OLD, LOOK_NEW = 160, 90
Z_FULL, Z_NEW = 1.5, 1.0
REQUIRE_OOS_PNL = True
REQUIRE_SIGN_MATCH = True
BANG_BANG_CORE_LL = True
BANG_BANG_EXTRA_LL = True
PAIR_PRIORITY_OVER_LL = True
FILTER_EXTRA_PAIR_FOLLOWERS = False

RIDGE_LAM = 1.0
RIDGE_THETA = 1.0
RIDGE_GATE_N = 35
RIDGE_LEG = 10_000.0
RIDGE_MIN_T = 252
RIDGE_FILL_RESIDUAL_CAPACITY = True

SLEEVE_ORDER = ["atlas", "coreLL", "extraLL", "ridge"]
SLEEVE_W = {"atlas": 1.0, "coreLL": 1.0, "extraLL": 1.0, "ridge": 1.0}
PROBE_W = {"atlas": 1.0, "coreLL": -1.0, "extraLL": -1.0, "ridge": 1.0}
GLOBAL_SCALAR = 1.0
MODE = "baseline"

_INIT_WINDOW = 20


# ---------------------------------------------------------------------------
# Adaptive beta filter (inlined NamedTuples for eval.py exec loader)
# ---------------------------------------------------------------------------


class FiltCfg(NamedTuple):
    q_beta: float = 3e-6
    q_beta_adapt: float = 5e-3
    ewma_lambda: float = 0.97
    warmup: int = 30
    cusum_k: float = 0.5
    auto_h: bool = True
    target_seg: int = 50
    h: float = 8.0
    h_gain: float = 0.05
    min_seg_len: int = 40
    vol_cut: bool = False
    cov_inflate: float = 25.0
    snap_window: int = 12
    adapt_chase: float = 1.0
    adapt_bars: int = 8
    adapt_extend: bool = False
    snap_overshoot: float = 1.10


class TradeCfg(NamedTuple):
    # Per-pair hysteresis on window-z (demeaned residual / rolling std).
    z_entry: float = 0.27   # enter when |z| >= z_entry
    z_exit: float = 0.10    # flatten when |z| <= z_exit (must be < z_entry)
    z_stop: float = 4.0     # flatten when |z| >= z_stop
    leg_scale: float = 1.0
    require_adf: bool = True
    require_kappa: bool = True
    trade_while_adapting: bool = True
    core: bool = True
    beta_mode: str = "pairseg"


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
        self.resid_var: float | None = None
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
        ols_s = self._ols_beta(self.cfg.snap_window)
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

        q_b = cfg.q_beta_adapt if self.adapt_left > 0 else cfg.q_beta
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

        if self.adapt_left > 0 and cfg.adapt_chase > 0.0:
            target = self._accel_target()
            ch = min(1.0, max(0.0, cfg.adapt_chase))
            self.beta = (1.0 - ch) * float(self.beta) + ch * target
            self.p22 = max(float(self.p22), 2.0)

        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            lam = cfg.ewma_lambda
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
        z = innov / sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0

        resid = yf - float(self.beta) * xf
        if self.resid_var is None:
            self.resid_var = resid * resid
        else:
            lam = cfg.ewma_lambda
            self.resid_var = lam * self.resid_var + (1.0 - lam) * resid * resid

        h_work = (
            (self.h_live if self.h_live is not None else cfg.h)
            if cfg.auto_h
            else cfg.h
        )

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
                if not cfg.vol_cut:
                    self.cv_up = self.cv_dn = 0.0
                    self.run_v_up = self.run_v_dn = None

            ols = self._ols_beta(cfg.min_seg_len)
            thr = max(0.025, 0.03 * abs(float(self.beta)))
            disagree = abs(ols - float(self.beta)) >= thr
            if disagree and self.adapt_left == 0:
                self.disagree_run += 1
            else:
                self.disagree_run = 0

            cusum_cut = (
                (mean_alarm or (cfg.vol_cut and vol_alarm))
                and disagree
                and self.adapt_left == 0
            )
            drift_need = max(8, cfg.target_seg // 4)
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
                if cfg.vol_cut and self.cv_up > h_work and self.cv_up > best:
                    best = self.cv_up
                    cut_bar = self.run_v_up
                if cfg.vol_cut and self.cv_dn > h_work and self.cv_dn > best:
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
                if closed_len >= cfg.min_seg_len:
                    cut = True
                    self.disagree_run = 0
                    target = self._ols_beta(cfg.snap_window)
                    pre = float(self.beta)
                    delta = target - pre
                    self.snap_dir = 1.0 if delta >= 0.0 else -1.0
                    self.beta = pre + cfg.snap_overshoot * delta
                    self.p22 = max(self.p22 * cfg.cov_inflate, 2.0)
                    self.adapt_left = cfg.adapt_bars
                    resid = yf - float(self.beta) * xf
                    if cfg.auto_h:
                        if self.seg_len_ewma is None:
                            self.seg_len_ewma = float(min(closed_len, 3 * cfg.target_seg))
                        else:
                            self.seg_len_ewma = 0.1 * closed_len + 0.9 * self.seg_len_ewma
                        err = (cfg.target_seg - self.seg_len_ewma) / cfg.target_seg
                        err = max(-0.5, min(0.5, err))
                        h0 = self.h_live if self.h_live is not None else cfg.h
                        self.h_live = max(2.0, h0 + cfg.h_gain * err * h0)
                    self.seg_start = cut_bar

        if self.adapt_left > 0:
            self.adapt_left -= 1
            adapting = True
            if self.adapt_left == 0:
                self.snap_dir = 0.0
                self.p22 = min(float(self.p22), 1e-4)

        h_out = (
            (self.h_live if self.h_live is not None else cfg.h)
            if cfg.auto_h
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
        # (y, x, EV, z_entry, z_exit, z_stop, q_beta, h, cusum_k, adapt_bars, snap, leg)
        # Auto-tuned by tuning/tune_pair_agents.py
        (18, 35, 107.1, 0.50, 0.02, 4.25, 8.284e-06, 5.0, 0.60, 4, 1.15, 1.00),  # RTTH/NAYO
        (25, 37, 94.5, 0.30, 0.05, 2.50, 5.498e-07, 6.0, 0.55, 5, 1.15, 1.00),  # CTGI/EELT
        (7, 40, 63.9, 0.27, 0.02, 4.00, 1.645e-07, 7.0, 0.60, 6, 1.40, 1.00),  # HETT/ULXY
        (42, 30, 49.3, 0.19, 0.04, 4.00, 7.481e-06, 6.0, 0.35, 14, 1.15, 1.00),  # BENI/RCRI
        (15, 25, 46.0, 0.33, 0.10, 3.75, 1.718e-05, 5.0, 0.60, 9, 1.10, 1.00),  # HRET/CTGI
        (8, 27, 45.1, 0.50, 0.18, 4.00, 1.056e-05, 8.5, 0.50, 16, 1.15, 1.00),  # HUXZ/ACAC
        (35, 28, 44.7, 0.16, 0.02, 4.25, 3.032e-07, 6.0, 0.60, 9, 1.15, 1.00),  # NAYO/SRTX
        (1, 20, 44.7, 0.18, 0.02, 3.25, 2.796e-07, 10.5, 0.55, 9, 1.10, 1.00),  # AENO/NWIG
        (22, 44, 42.1, 0.30, 0.02, 4.75, 5.633e-07, 8.5, 0.60, 13, 1.15, 1.00),  # MDGI/HTRK
        (17, 18, 41.2, 0.39, 0.04, 3.25, 1.219e-06, 10.0, 0.40, 11, 1.35, 1.00),  # DIHO/RTTH
        (49, 50, 39.5, 0.10, 0.02, 4.50, 1.128e-06, 7.0, 0.55, 12, 1.25, 1.00),  # MHRM/EAFC
        (33, 26, 39.2, 0.15, 0.04, 3.50, 5.823e-07, 9.0, 0.40, 16, 1.40, 1.00),  # MTNS/ALUT
        (12, 40, 38.3, 0.31, 0.02, 4.75, 4.596e-07, 8.5, 0.35, 5, 1.05, 1.00),  # MSDP/ULXY
        (13, 45, 34.9, 0.40, 0.30, 4.75, 1.050e-07, 12.0, 0.45, 11, 1.35, 1.00),  # EORC/NGTE
        (26, 32, 32.5, 0.47, 0.19, 3.50, 1.974e-07, 5.0, 0.55, 9, 1.30, 1.00),  # ALUT/CCNS
        (10, 46, 30.7, 0.35, 0.22, 3.00, 2.263e-05, 5.0, 0.55, 16, 1.30, 1.00),  # SMAH/ILVX
        (33, 42, 23.1, 0.50, 0.05, 2.75, 1.856e-05, 11.5, 0.45, 14, 1.15, 1.00),  # MTNS/BENI
        (32, 28, 20.2, 0.44, 0.24, 4.75, 1.005e-07, 6.0, 0.60, 7, 1.30, 1.00),  # CCNS/SRTX
        (36, 41, 19.1, 0.11, 0.04, 4.00, 9.990e-07, 8.0, 0.50, 8, 1.20, 1.00),  # FWWG/BLBT
        (31, 43, 16.0, 0.26, 0.02, 4.50, 6.157e-06, 5.0, 0.60, 8, 1.05, 1.00),  # ACIX/ITPA
        (31, 40, 16.0, 0.39, 0.02, 2.50, 2.901e-05, 6.5, 0.35, 13, 1.05, 1.00),  # ACIX/ULXY
        (8, 20, 14.9, 0.41, 0.24, 4.50, 3.537e-07, 11.0, 0.55, 7, 1.30, 0.80),  # HUXZ/NWIG
        (50, 28, 0.8, 0.48, 0.30, 4.75, 3.737e-06, 10.5, 0.55, 5, 1.10, 1.00),  # EAFC/SRTX
    ]
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
                    core=True,
                    beta_mode="pairseg",
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


def _beta_dollar_legs(side: float, beta: float, scale: float) -> tuple[float, float]:
    if abs(side) < 1e-12 or abs(beta) < 1e-12:
        return 0.0, 0.0
    cap = LEG * max(0.0, min(1.0, scale))
    ab = abs(beta)
    if ab >= 1.0:
        return float(side * (cap / ab)), float(-side * np.sign(beta) * cap)
    return float(side * cap), float(-side * beta * cap)


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
        self.filt = PairSegFilter(spec.filt)
        self.innov_z = 0.0
        self.window_z = 0.0
        self.h = float(spec.filt.h)
        self.beta = 1.0
        self.beta_lag = 1.0
        self.adapting = False
        self.nt_seen = 0
        self.side = 0  # -1 short Y / +1 long Y / 0 flat

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self.__init__(self.spec)
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
        if not tc.trade_while_adapting and self.adapting:
            self.side = 0
            return 0.0
        nt = LP.shape[1]
        if nt < HEDGE_W + 10:
            self.side = 0
            return 0.0
        y_i, x_i = self.spec.y, self.spec.x
        beta = self.beta_lag
        if tc.require_adf:
            spread = LP[y_i, : nt - 1] - beta * LP[x_i, : nt - 1]
            if len(spread) >= 40 and _adf_t_spread(spread) >= PRIOR_GATE_T:
                self.side = 0
                return 0.0
        s = LP[y_i, nt - Z_W : nt] - beta * LP[x_i, nt - Z_W : nt]
        z = float((s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12))
        self.window_z = z
        if tc.require_kappa:
            s120 = LP[y_i, nt - 120 : nt] - beta * LP[x_i, nt - 120 : nt]
            if _ou_kappa(s120) <= 0.0:
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


def _ll_overheat_scale(abs_rho: float) -> float:
    if abs_rho <= RHO_HI:
        return 1.0
    if abs_rho >= RHO_CAP:
        return RHO_MIN_SCALE
    w = (abs_rho - RHO_HI) / (RHO_CAP - RHO_HI)
    return 1.0 - w * (1.0 - RHO_MIN_SCALE)


def _ll_health(r, i, j, look=LL_LOOK):
    m = r.shape[1]
    look = min(look, m - 1)
    if look < 100:
        return None
    x = r[i, m - look - 1 : m - 1]
    y = r[j, m - look : m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    c = float(np.corrcoef(x, y)[0, 1])
    z = abs(c) * np.sqrt(look)
    h = look // 2
    c1 = float(np.corrcoef(x[:h], y[:h])[0, 1])
    c2 = float(np.corrcoef(x[h:], y[h:])[0, 1])
    z_recent = abs(c2) * np.sqrt(h)
    return c, z, z_recent, (np.sign(c1) == np.sign(c2) == np.sign(c))


def _leadlag_dollars(r):
    nins = r.shape[0]
    pos = np.zeros(nins)
    for lead, foll in LEADLAG:
        i, j = I[lead], I[foll]
        hh = _ll_health(r, i, j)
        if hh is None:
            continue
        c, z, _, _ = hh
        if z < GATE_LL_Z:
            continue
        w = 1.0 if BANG_BANG_CORE_LL or np.sign(r[i, -1]) == np.sign(r[0, -1]) else 0.5
        pos[j] += w * LEG * np.sign(c) * np.sign(r[i, -1])
    return pos


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
        w = 1.0 if BANG_BANG_EXTRA_LL or np.sign(r[i, -1]) == np.sign(r[0, -1]) else 0.5
        scale = 1.0 if BANG_BANG_EXTRA_LL else _ll_overheat_scale(abs(h["c_full"]))
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


def _active_weights():
    if MODE == "probe":
        return PROBE_W, GLOBAL_SCALAR
    if MODE == "fullport":
        return {k: 1.0 for k in SLEEVE_ORDER}, GLOBAL_SCALAR
    if MODE == "baseline":
        return {k: 1.0 for k in SLEEVE_ORDER}, 1.0
    return SLEEVE_W, GLOBAL_SCALAR


def reset():
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


def sleeve_breakdown(prcSoFar):
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
    d_coreLL = _leadlag_dollars(r)
    extra_idx = _EXTRA_IDX_NO_PAIR_FOLLOWERS if FILTER_EXTRA_PAIR_FOLLOWERS else _EXTRA_IDX
    crowded_mask = (np.abs(d_atlas) > 1.0) | (np.abs(d_coreLL) > 1.0)
    d_extraLL = _extra_leadlag_dollars(r, extra_idx, crowded_mask)
    if PAIR_PRIORITY_OVER_LL:
        pair_mask = np.abs(d_atlas) > 1.0
        d_coreLL = np.where(pair_mask, 0.0, d_coreLL)
        d_extraLL = np.where(pair_mask, 0.0, d_extraLL)
    held0 = d_atlas + d_coreLL + d_extraLL
    d_ridge = _ridge_dollars(r, held0)
    return {
        "atlas": d_atlas,
        "coreLL": d_coreLL,
        "extraLL": d_extraLL,
        "ridge": d_ridge,
    }


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < 130 or nins != 51:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0):
        return np.zeros(nins, dtype=int)

    sl = sleeve_breakdown(prcSoFar)
    if sl is None:
        return np.zeros(nins, dtype=int)

    weights, gscalar = _active_weights()
    dollars = np.zeros(nins)
    for name in SLEEVE_ORDER:
        dollars += float(weights.get(name, 0.0)) * sl[name]
    dollars *= float(gscalar)

    r_last = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(r_last) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)