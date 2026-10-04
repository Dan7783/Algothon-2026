from __future__ import annotations

from math import sqrt
from typing import NamedTuple, TypedDict, NotRequired

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
WARMUP = 30         # Global time threshold: CUSUM structural monitoring stays disabled until step t >= WARMUP.
_INIT_WINDOW = 20   # Initial OLS lookback window size used to calculate starting hedge ratio (beta) at t = 0.

# Numerical safety constants
_EPS_DENOM = 1e-12
_EPS_VAR = 1e-18
_EPS_COV = 1e-4


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

    # Maximum covariance inflation multiplier cap when scaling dynamically.
    max_cov_inflate: float = 25.0


class TradeCfg(NamedTuple):
    z_entry: float = 0.27
    z_exit: float = 0.10
    z_stop: float = 4.0


class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
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
        self.t = -1
        self._y_hist: list[float] = []
        self._x_hist: list[float] = []

    def _ols_beta(self, window: int) -> float:
        n = len(self._x_hist)
        w = min(window, n)
        xs = np.asarray(self._x_hist[-w:], dtype=float)
        ys = np.asarray(self._y_hist[-w:], dtype=float)
        denom = float(xs @ xs)
        if denom < _EPS_DENOM:
            return float(self.beta) if self.beta is not None else 1.0
        return float(xs @ ys) / denom

    def _update_kalman(self, yf: float, xf: float, dynamic_q: float) -> tuple[float, float, float]:
        b = float(self.beta)
        
        # 1. Predict Step: Add dynamically scaled process noise to covariance
        pp22 = self.p22 + dynamic_q
        
        # 2. Innovation (Prediction Error): v_t = y_t - beta_{t|t-1} * x_t
        innov = yf - b * xf
        
        # 3. Innovation Variance (S_t): S_t = P_{22, t} * x_t^2 + R_t (assuming observation variance R = 1.0)
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < _EPS_DENOM:
            s_pred = _EPS_DENOM
            
        # 4. Kalman Gain Calculation: K_t = P_{22} * x_t / S_t
        g = (pp22 * xf) / s_pred
        
        # 5. State Update: beta_t = beta_{t-1} + K_t * v_t
        self.beta = b + g * innov
        
        # 6. Covariance Update: P_{22, t} = (1 - K_t * x_t) * P_{22, pred}
        a = 1.0 - g * xf
        self.p22 = max(a * a * pp22 + g * g, _EPS_VAR)
        
        return innov, float(self.beta), float(self.p22)

    def _update_variances(self, innov: float, resid: float) -> float:
        lam = self.cfg.ewma_lambda
        
        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
            
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

        # --- Mean CUSUM ---
        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        
        self.run_m_up = t if (self.cm_up > 0.0 and prev_up == 0.0) else (None if self.cm_up == 0.0 else self.run_m_up)
        self.run_m_dn = t if (self.cm_dn > 0.0 and prev_dn == 0.0) else (None if self.cm_dn == 0.0 else self.run_m_dn)

        # --- Variance CUSUM ---
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
        
        if closed_len >= cfg.min_seg_len:
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

        max_cusum = max(self.cm_up, self.cm_dn, self.cv_up, self.cv_dn)
        excess = max(0.0, max_cusum - cfg.h)
        
        multiplier = 1.0 + min(excess, cfg.max_cov_inflate)
        dynamic_q = cfg.q_beta * multiplier
        adapting = excess > 0.0

        innov, _, _ = self._update_kalman(yf, xf, dynamic_q)
        resid = yf - float(self.beta) * xf
        z = self._update_variances(innov, resid)

        cut = False
        vol_shift = False
        if t >= WARMUP:
            cut, vol_shift = self._evaluate_cusum(z, t)

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


class RawPairDict(TypedDict):
    y: str
    x: str
    z_entry: float
    z_exit: float
    z_stop: float
    q_beta: float
    h: float
    cusum_k: float
    min_seg_len: NotRequired[int]
    max_cov_inflate: NotRequired[float]


def _default_pair_cfgs() -> tuple[PairCfg, ...]:
    raw_seeds: list[RawPairDict] = [
        {"y": "DIHO", "x": "NAYO", "z_entry": 0.50, "z_exit": 0.25, "z_stop": 4.25, "q_beta": 1.21e-06, "h": 5.0, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "RRES", "x": "EELT", "z_entry": 0.35, "z_exit": 0.05, "z_stop": 3.25, "q_beta": 6.57e-06, "h": 4.0, "cusum_k": 0.45, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "OTCS", "x": "ULXY", "z_entry": 0.75, "z_exit": 0.02, "z_stop": 4.00, "q_beta": 3.57e-05, "h": 7.0, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "MSDP", "x": "GARI", "z_entry": 0.33, "z_exit": 0.10, "z_stop": 3.75, "q_beta": 7.28e-05, "h": 5.0, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "HUXZ", "x": "ACAC", "z_entry": 0.60, "z_exit": 0.23, "z_stop": 4.75, "q_beta": 1.00e-09, "h": 6.5, "cusum_k": 0.40, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "SRTX", "x": "RRES", "z_entry": 0.16, "z_exit": 0.025,"z_stop": 3.50, "q_beta": 4.85e-05, "h": 4.0, "cusum_k": 0.50, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "ALGO", "x": "NWIG", "z_entry": 0.12, "z_exit": 0.02, "z_stop": 4.00, "q_beta": 8.71e-06, "h": 10.5,"cusum_k": 0.55, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "MDGI", "x": "AETS", "z_entry": 0.30, "z_exit": 0.02, "z_stop": 4.75, "q_beta": 1.09e-04, "h": 8.5, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "ANSO", "x": "ALGO", "z_entry": 0.75, "z_exit": 0.25, "z_stop": 2.50, "q_beta": 5.28e-06,   "h": 3.0, "cusum_k": 0.40, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "MHRM", "x": "EAFC", "z_entry": 0.10, "z_exit": 0.02, "z_stop": 5.25, "q_beta": 6.42e-07, "h": 9.0, "cusum_k": 0.65, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "MTNS", "x": "ALUT", "z_entry": 0.15, "z_exit": 0.04, "z_stop": 3.50, "q_beta": 3.20e-05, "h": 9.0, "cusum_k": 0.40, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "HETT", "x": "ULXY", "z_entry": 0.31, "z_exit": 0.02, "z_stop": 4.75, "q_beta": 1.00e-09, "h": 8.5, "cusum_k": 0.35, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "HETT", "x": "FARS", "z_entry": 0.40, "z_exit": 0.34, "z_stop": 4.00, "q_beta": 2.16e-05, "h": 10.0,"cusum_k": 0.35, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "ALUT", "x": "CCNS", "z_entry": 0.587,"z_exit": 0.1425,"z_stop":3.50, "q_beta": 1.00e-09,  "h": 5.0, "cusum_k": 0.45, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "SMAH", "x": "ILVX", "z_entry": 0.35, "z_exit": 0.2975,"z_stop":3.00, "q_beta": 1.00e-09, "h": 3.0, "cusum_k": 0.35, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "MTNS", "x": "BENI", "z_entry": 0.50, "z_exit": 0.05, "z_stop": 2.75, "q_beta": 1.00e-09, "h": 11.5,"cusum_k": 0.45, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "CCNS", "x": "RRES", "z_entry": 0.75, "z_exit": 0.15, "z_stop": 4.75, "q_beta": 2.21e-05, "h": 6.0, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "FWWG", "x": "BLBT", "z_entry": 0.11, "z_exit": 0.05, "z_stop": 4.00, "q_beta": 1.00e-09, "h": 6.0, "cusum_k": 0.40, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "ACIX", "x": "ITPA", "z_entry": 0.26, "z_exit": 0.02, "z_stop": 4.50, "q_beta": 1.00e-09, "h": 5.0, "cusum_k": 0.60, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "ACIX", "x": "ULXY", "z_entry": 0.39, "z_exit": 0.02, "z_stop": 2.50, "q_beta": 8.80e-07, "h": 6.5, "cusum_k": 0.35, "min_seg_len": 40, "max_cov_inflate": 25.0},
        {"y": "EAFC", "x": "RRES", "z_entry": 0.48, "z_exit": 0.30, "z_stop": 4.75, "q_beta": 3.93e-05, "h": 10.5,"cusum_k": 0.55, "min_seg_len": 40, "max_cov_inflate": 25.0},
    ]

    out: list[PairCfg] = []
    seen: set[frozenset[int]] = set()
    for row in raw_seeds:
        y_sym, x_sym = row["y"], row["x"]
        y = I[y_sym]
        x = I[x_sym]
        key = frozenset((y, x))
        if key in seen or y == x:
            continue
        seen.add(key)

        z_entry = float(row["z_entry"])
        z_exit = float(min(row["z_exit"], 0.85 * z_entry))
        z_stop = float(max(row["z_stop"], 1.25 * z_entry))

        filt_kwargs = {
            "q_beta": float(row["q_beta"]),
            "h": float(row["h"]),
            "cusum_k": float(row["cusum_k"]),
        }
        if "min_seg_len" in row:
            filt_kwargs["min_seg_len"] = int(row["min_seg_len"])
        if "max_cov_inflate" in row:
            filt_kwargs["max_cov_inflate"] = float(row["max_cov_inflate"])

        out.append(
            PairCfg(
                idx=len(out),
                y=y,
                x=x,
                filt=FiltCfg(**filt_kwargs),
                trade=TradeCfg(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    z_stop=z_stop,
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
            y = np.log(max(float(prices[y_i, t]), _EPS_DENOM))
            x = np.log(max(float(prices[x_i, t]), _EPS_DENOM))
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
            if abs(side) < _EPS_DENOM:
                continue
            s = ag.spec
            ev = abs(ag.current_z)
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
            pos[y] += side * LEG
            pos[x] += -side * LEG
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