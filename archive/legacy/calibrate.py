from __future__ import annotations

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
WARMUP = 30         # Global time threshold: CUSUM structural monitoring stays disabled until step t >= WARMUP to let the Kalman filter stabilize.
_INIT_WINDOW = 20   # Initial OLS lookback window size used to calculate the starting hedge ratio (beta) at t = 0.


class FiltCfg(NamedTuple):
    # Process noise covariance for the hedge ratio beta (Q). 
    # Controls how fast the Kalman filter adapts to drifting cointegration relationships.
    q_beta: float = 3e-6

    # Exponentially Weighted Moving Average (EWMA) decay factor (lambda). 
    # Stays fixed to slightly smooth innovation and residual variances, removing high-frequency noise.
    ewma_lambda: float = 0.97

    # CUSUM allowance/slack parameter (k). 
    # Subtracted from cumulative sums on each step to ignore minor noise fluctuations and drift.
    cusum_k: float = 0.5

    # CUSUM alarm threshold (h). 
    # If the cumulative sum exceeds this boundary, a structural mean or volatility break is flagged.
    h: float = 8.0

    # Minimum segment length guardrail (fixed heuristic). 
    # Forces a newly cut segment to span at least this many steps to prevent over-reacting to short-term noise.
    min_seg_len: int = 40

    # Volatility cut flag. 
    # When False, volatility alarms only log a shift without resetting parameters; when True, they force a segment cut.
    vol_cut: bool = False

    # Covariance inflation factor. 
    # Multiplies the error covariance matrix (p22) upon a segment reset to force the filter to temporarily re-adapt aggressively.
    cov_inflate: float = 25.0


class TradeCfg(NamedTuple):
    z_entry: float = 0.27
    z_exit: float = 0.10
    z_stop: float = 4.0
    leg_scale: float = 1.0


class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
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
        self.adapt_left = 0
        self.t = -1
        self._y_hist: list[float] = []
        self._x_hist: list[float] = []

    def _ols_beta(self, window: int) -> float:
        # Computes an initial OLS regression slope (beta) over a recent historical window 
        # to seed the Kalman filter state vector at startup: beta = (X^T X)^{-1} X^T Y
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < 1e-12:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def _update_kalman(self, yf: float, xf: float) -> tuple[float, float, float]:
        q_b = self.cfg.q_beta
        b = float(self.beta)
        
        # 1. Predict Step: Add process noise (q_beta) to the error covariance matrix element (p22)
        pp22 = self.p22 + q_b
        
        # 2. Innovation (Prediction Error): v_t = y_t - beta_{t|t-1} * x_t
        innov = yf - b * xf
        
        # 3. Innovation Variance (S_t): S_t = P_{22, t} * x_t^2 + R_t (assuming observation variance R = 1.0)
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < 1e-12:
            s_pred = 1e-12
            
        # 4. Kalman Gain Calculation: K_t = P_{22} * x_t / S_t
        g = (pp22 * xf) / s_pred
        
        # 5. State Update: beta_t = beta_{t-1} + K_t * v_t
        self.beta = b + g * innov
        
        # 6. Covariance Update: P_{22, t} = (1 - K_t * x_t) * P_{22, pred}
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, 1e-18)
        
        return innov, float(self.beta), float(self.p22)

    def _update_variances(self, innov: float, resid: float) -> float:
        lam = self.cfg.ewma_lambda
        
        # Smooth innovation variance using an EWMA filter to normalize the z-score stably
        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
            
        # Compute standardized innovation z-score: z = v_t / sqrt(EWMA variance)
        z = innov / sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0
        
        if self.resid_var is None:
            self.resid_var = resid * resid
        else:
            self.resid_var = lam * self.resid_var + (1.0 - lam) * resid * resid
        return z

    def _evaluate_cusum(self, z: float, t: int) -> tuple[bool, bool]:
        cfg = self.cfg
        vol_shift = False
        cut = False

        # --- Mean CUSUM (Tracking shifts in the mean level of the z-score) ---
        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        
        self.run_m_up = t if (self.cm_up > 0.0 and prev_up == 0.0) else (None if self.cm_up == 0.0 else self.run_m_up)
        self.run_m_dn = t if (self.cm_dn > 0.0 and prev_dn == 0.0) else (None if self.cm_dn == 0.0 else self.run_m_dn)

        # --- Variance CUSUM (Tracking shifts in volatility using standardized squared residuals: d_v) ---
        d_v = (z * z - 1.0) / sqrt(2.0)
        prev_v_up, prev_v_dn = self.cv_up, self.cv_dn
        self.cv_up = max(0.0, self.cv_up + d_v - cfg.cusum_k)
        self.cv_dn = max(0.0, self.cv_dn - d_v - cfg.cusum_k)
        
        self.run_v_up = t if (self.cv_up > 0.0 and prev_v_up == 0.0) else (None if self.cv_up == 0.0 else self.run_v_up)
        self.run_v_dn = t if (self.cv_dn > 0.0 and prev_v_dn == 0.0) else (None if self.cv_dn == 0.0 else self.run_v_dn)

        mean_alarm = self.cm_up > cfg.h or self.cm_dn > cfg.h
        vol_alarm = self.cv_up > cfg.h or self.cv_dn > cfg.h
        
        if vol_alarm:
            vol_shift = True
            if not cfg.vol_cut:
                self.cv_up = self.cv_dn = 0.0
                self.run_v_up = self.run_v_dn = None

        if mean_alarm or (cfg.vol_cut and vol_alarm):
            cut = self._trigger_segment_cut(t)

        return cut, vol_shift

    def _trigger_segment_cut(self, t: int) -> bool:
        cfg = self.cfg
        best = -1.0
        cut_bar = None
        
        if self.cm_up > cfg.h and self.cm_up > best:
            best, cut_bar = self.cm_up, self.run_m_up
        if self.cm_dn > cfg.h and self.cm_dn > best:
            best, cut_bar = self.cm_dn, self.run_m_dn
        if cfg.vol_cut and self.cv_up > cfg.h and self.cv_up > best:
            best, cut_bar = self.cv_up, self.run_v_up
        if cfg.vol_cut and self.cv_dn > cfg.h and self.cv_dn > best:
            best, cut_bar = self.cv_dn, self.run_v_dn
            
        cut_bar = max(int(cut_bar if cut_bar is not None else t), self.seg_start + 1)
        closed_len = cut_bar - self.seg_start
        
        self.cm_up = self.cm_dn = 0.0
        self.cv_up = self.cv_dn = 0.0
        self.run_m_up = self.run_m_dn = None
        self.run_v_up = self.run_v_dn = None
        
        # Enforce minimum segment length heuristic before committing a structural break reset
        if closed_len >= cfg.min_seg_len:
            # Inflate error covariance to allow rapid parameter adaptation following a structural break
            self.p22 = max(self.p22 * cfg.cov_inflate, 1e-4)
            self.adapt_left = 5
            self.seg_start = cut_bar
            return True
        return False

    def step(self, y: float, x: float) -> StepOut:
        cfg = self.cfg
        self.t += 1
        t = self.t
        yf, xf = float(y), float(x)
        
        self._y_hist.append(yf)
        self._x_hist.append(xf)

        if self.beta is None:
            self.beta = self._ols_beta(_INIT_WINDOW)
            self.seg_start = t

        innov, _, _ = self._update_kalman(yf, xf)
        resid = yf - float(self.beta) * xf
        z = self._update_variances(innov, resid)

        cut = False
        vol_shift = False
        if t >= WARMUP:
            cut, vol_shift = self._evaluate_cusum(z, t)

        adapting = self.adapt_left > 0
        if adapting:
            self.adapt_left -= 1
            if self.adapt_left == 0:
                self.p22 = min(float(self.p22), 1e-4)

        return StepOut(
            beta=float(self.beta),
            spread=float(resid),
            z=float(z),
            h=float(cfg.h),
            cut=cut,
            adapting=adapting,
            vol_shift=vol_shift,
            cm_up=float(self.cm_up),
            cm_dn=float(self.cm_dn),
            cv=max(self.cv_up, self.cv_dn),
        )


def _default_pair_cfgs() -> tuple[PairCfg, ...]:
    seeds = [
        (18, 35, 107.1, 0.50, 0.25, 4.25, 8.284e-06, 5.0, 0.60, 1.00),
        (25, 37, 94.5, 0.35, 0.05, 3.25, 1.833e-07, 4.0, 0.45, 1.00),
        (7, 40, 63.9, 0.75, 0.02, 4.00, 1.645e-07, 7.0, 0.60, 1.00),
        (15, 25, 46.0, 0.33, 0.10, 3.75, 1.718e-05, 5.0, 0.60, 0.65),
        (8, 27, 45.1, 0.60, 0.23, 4.75, 3.168e-05, 6.5, 0.40, 1.00),
        (35, 28, 44.7, 0.16, 0.025, 3.50, 9.096e-07, 4.0, 0.50, 1.00),
        (1, 20, 44.7, 0.12, 0.02, 4.00, 2.796e-07, 10.5, 0.55, 1.00),
        (22, 44, 42.1, 0.30, 0.02, 4.75, 5.633e-07, 8.5, 0.60, 1.00),
        (17, 18, 41.2, 0.75, 0.25, 2.50, 5e-05, 3.0, 0.40, 1.00),
        (49, 50, 39.5, 0.10, 0.02, 5.25, 1.128e-06, 9.0, 0.65, 1.00),
        (33, 26, 39.2, 0.15, 0.04, 3.50, 5.823e-07, 9.0, 0.40, 1.00),
        (12, 40, 38.3, 0.31, 0.02, 4.75, 4.596e-07, 8.5, 0.35, 1.00),
        (13, 45, 34.9, 0.40, 0.34, 4.00, 3.500e-08, 10.0, 0.35, 1.00),
        (26, 32, 32.5, 0.587, 0.1425, 3.50, 6.58e-08, 5.0, 0.45, 1.00),
        (10, 46, 30.7, 0.35, 0.2975, 3.00, 6.789e-05, 3.0, 0.35, 1.00),
        (33, 42, 23.1, 0.50, 0.05, 2.75, 1.856e-05, 11.5, 0.45, 1.00),
        (32, 28, 20.2, 0.75, 0.15, 4.75, 1.005e-07, 6.0, 0.60, 0.30),
        (36, 41, 19.1, 0.11, 0.05, 4.00, 9.990e-07, 6.0, 0.40, 1.00),
        (31, 43, 16.0, 0.26, 0.02, 4.50, 6.157e-06, 5.0, 0.60, 1.00),
        (31, 40, 16.0, 0.39, 0.02, 2.50, 2.901e-05, 6.5, 0.35, 1.00),
        (50, 28, 0.8, 0.48, 0.30, 4.75, 3.737e-06, 10.5, 0.55, 1.00),
    ]
    out: list[PairCfg] = []
    seen: set[frozenset[int]] = set()
    for y, x, pri, ze, zx, zs, qb, h, ck, ls in seeds:
        key = frozenset((y, x))
        if key in seen or y == x:
            continue
        seen.add(key)
        z_entry = float(ze)
        z_exit = float(min(zx, 0.85 * z_entry))
        z_stop = float(max(zs, 1.25 * z_entry))
        out.append(
            PairCfg(
                idx=len(out),
                y=y,
                x=x,
                priority=float(pri),
                filt=FiltCfg(q_beta=qb, h=h, cusum_k=ck),
                trade=TradeCfg(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    z_stop=z_stop,
                    leg_scale=ls,
                ),
            )
        )
    return tuple(out)


PAIR_CFGS = _default_pair_cfgs()


class PairAgent:
    __slots__ = ("spec", "filt", "current_z", "side", "nt_seen")

    def __init__(self, spec: PairCfg) -> None:
        self.spec = spec
        self.filt = PairSegFilter(spec.filt)
        self.current_z = 0.0
        self.side = 0
        self.nt_seen = 0

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self.__init__(self.spec)
        y_i, x_i = self.spec.y, self.spec.x
        for t in range(self.nt_seen, nt):
            y = np.log(max(float(prices[y_i, t]), 1e-12))
            x = np.log(max(float(prices[x_i, t]), 1e-12))
            o = self.filt.step(y, x)
            self.current_z = float(o.z)
            self.nt_seen = t + 1

    def desired_side(self) -> float:
        tc = self.spec.trade
        z = self.current_z
        az = abs(z)

        if self.side == 0:
            if az >= tc.z_entry and az < tc.z_stop:
                self.side = int(-np.sign(z))
        else:
            if az <= tc.z_exit or az >= tc.z_stop:
                self.side = 0
            elif az >= tc.z_entry and int(-np.sign(z)) != self.side:
                self.side = int(-np.sign(z))
        return float(self.side)


class AtlasBook:
    def __init__(self) -> None:
        self.agents = [PairAgent(s) for s in PAIR_CFGS]

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        if nt < WARMUP + 10:
            return pos

        for ag in self.agents:
            ag.catch_up(prices)

        cands: list[tuple[float, int, float]] = []
        for ag in self.agents:
            side = ag.desired_side()
            if abs(side) < 1e-12:
                continue
            s = ag.spec
            ev = s.priority * abs(ag.current_z) * s.trade.leg_scale
            cands.append((ev, s.idx, side))

        cands.sort(reverse=True)
        taken: set[int] = set()
        for _ev, idx, side in cands:
            ag = self.agents[idx]
            y, x = ag.spec.y, ag.spec.x
            if y in taken or x in taken:
                continue
            taken.add(y)
            taken.add(x)
            scale = ag.spec.trade.leg_scale
            pos[y] += side * LEG * scale
            pos[x] += -side * LEG * scale
        return pos


_STATE = {"atlas": None}


def reset() -> None:
    _STATE["atlas"] = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < WARMUP + 10:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return np.zeros(nins, dtype=int)

    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()

    dollars = _STATE["atlas"].dollars(prcSoFar)
    last_return = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)