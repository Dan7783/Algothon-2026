from __future__ import annotations

from math import sqrt
from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
import os

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

N_INST = 51
LEG = 10_000.0
WARMUP = 30         
_INIT_WINDOW = 20   

_EPS_DENOM = 1e-12
_EPS_VAR = 1e-18

class RawPairDict(TypedDict):
    y: str
    x: str
    q_beta: float
    cusum_k: float
    h: float
    max_cov_inflate: float
    z_entry: float
    z_exit: float
    z_stop: float
    kelly_cap: float
    alarm_damp: float
    kappa: float
    sigma: float

class FiltCfg(NamedTuple):
    q_beta: float
    ewma_lambda: float = 0.97
    cusum_k: float = 0.5
    h: float = 8.0
    min_seg_len: int = 40
    vol_cut: bool = True  # Enabled by default for strict segment cuts
    max_cov_inflate: float = 25.0

class TradeCfg(NamedTuple):
    z_entry: float
    z_exit: float
    z_stop: float
    kappa: float
    sigma: float
    kelly_cap: float
    alarm_damp: float

class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
    filt: FiltCfg
    trade: TradeCfg

class StepOut(NamedTuple):
    beta: float
    innov: float       # Added to feed OU-Kelly sizing
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
        self.cfg = cfg if cfg is not None else FiltCfg(q_beta=1e-6)
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
        b = float(self.beta) if self.beta is not None else 1.0
        pp22 = self.p22 + dynamic_q
        innov = yf - b * xf
        
        s_pred = pp22 * xf * xf + 1.0
        if s_pred < _EPS_DENOM:
            s_pred = _EPS_DENOM
            
        g = (pp22 * xf) / s_pred
        self.beta = b + g * innov
        
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
        vol_shift = cut = False

        # Mean CUSUM
        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        
        self.run_m_up = t if (self.cm_up > 0.0 and prev_up == 0.0) else (None if self.cm_up == 0.0 else self.run_m_up)
        self.run_m_dn = t if (self.cm_dn > 0.0 and prev_dn == 0.0) else (None if self.cm_dn == 0.0 else self.run_m_dn)

        # Variance CUSUM
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
            
        cut_bar = max(int(cut_bar if cut_bar is not None else t), (self.seg_start or 0) + 1)
        closed_len = cut_bar - (self.seg_start or 0)
        
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
        
        # FIX: max_cov_inflate now acts as a proper multiplicative scaler.
        # It uses a rapid exponential ramp: as CUSUM exceeds h, multiplier 
        # smoothly approaches max_cov_inflate.
        if excess > 0.0:
            multiplier = 1.0 + (cfg.max_cov_inflate - 1.0) * (1.0 - np.exp(-excess))
        else:
            multiplier = 1.0
            
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
            innov=float(innov),
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

# Mapped dynamically from the statistically-validated pairs and OU parameters found in Source 1.
def _build_configs() -> tuple[PairCfg, ...]:
    # Allow environment variable overrides from the tuner — ALL 10 tuned params
    q_beta = float(os.environ.get("TUNER_Q_BETA", 1.02e-08))
    cusum_k = float(os.environ.get("TUNER_CUSUM_K", 1.54))
    h = float(os.environ.get("TUNER_H", 12.0))
    max_cov_inflate = float(os.environ.get("TUNER_MAX_COV_INFLATE", 84.1))
    z_entry = float(os.environ.get("TUNER_Z_ENTRY", 1.79))
    z_exit = float(os.environ.get("TUNER_Z_EXIT", 0.223))
    kappa = float(os.environ.get("TUNER_KAPPA", 0.121))
    sigma = float(os.environ.get("TUNER_SIGMA", 0.0237))
    kelly_cap = float(os.environ.get("TUNER_KELLY_CAP", 1.0))
    alarm_damp = float(os.environ.get("TUNER_ALARM_DAMP", 1.0))

    # --- NEW: tuner-supplied pair symbols ---
    pair_y = os.environ.get("TUNER_PAIR_Y", "SMAH")
    pair_x = os.environ.get("TUNER_PAIR_X", "NGTE")

    # Validate symbols exist in the instrument table; fall back gracefully
    if pair_y not in I or pair_x not in I:
        print(f"[testbed] WARNING: unknown pair ({pair_y}, {pair_x}); defaulting to SMAH-NGTE")
        pair_y, pair_x = "SMAH", "NGTE"
    if pair_y == pair_x:
        print(f"[testbed] WARNING: pair_y == pair_x ({pair_y}); defaulting to SMAH-NGTE")
        pair_y, pair_x = "SMAH", "NGTE"

    RAW_PAIRS: list[RawPairDict] = [
        {
            "y": pair_y, "x": pair_x, 
            "q_beta": q_beta, "cusum_k": cusum_k, "h": h, 
            "max_cov_inflate": max_cov_inflate, "z_entry": z_entry, "z_exit": z_exit, 
            "z_stop": 10, "kelly_cap": kelly_cap, "alarm_damp": alarm_damp, 
            "kappa": kappa, "sigma": sigma
        },
    ]

    out: list[PairCfg] = []
    for idx, row in enumerate(RAW_PAIRS):
        out.append(PairCfg(
            idx=idx,
            y=I[row["y"]], 
            x=I[row["x"]],
            filt=FiltCfg(
                q_beta=row["q_beta"], 
                cusum_k=row["cusum_k"], 
                h=row["h"], 
                max_cov_inflate=row["max_cov_inflate"]
            ),
            trade=TradeCfg(
                z_entry=row["z_entry"], 
                z_exit=row["z_exit"], 
                z_stop=row["z_stop"],
                kappa=row["kappa"], 
                sigma=row["sigma"], 
                kelly_cap=row["kelly_cap"], 
                alarm_damp=row["alarm_damp"]
            )
        ))
    return tuple(out)

PAIR_CFGS = _build_configs()

class PairAgent:
    __slots__ = ("spec", "filt", "current_z", "current_innov", "side", "cooldown", "nt_seen")

    def __init__(self, spec: PairCfg) -> None:
        self.spec = spec
        self.filt = PairSegFilter(spec.filt)
        self.current_z = 0.0
        self.current_innov = 0.0
        self.side = 0
        self.cooldown = 0
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
            self.current_innov = float(o.innov)

            if o.cut or o.adapting:
                self.cooldown = 10
            elif self.cooldown > 0:
                self.cooldown -= 1

            tc = self.spec.trade
            az = abs(self.current_z)
            
            if self.side == 0:
                if tc.z_entry <= az < tc.z_stop:
                    self.side = int(-np.sign(self.current_z)) if self.current_z != 0 else 0
            else:
                if az <= tc.z_exit or az >= tc.z_stop:
                    self.side = 0
                    
        self.nt_seen = nt

    def dollar_positions(self) -> tuple[float, float, float]:
        if self.side == 0:
            return 0.0, 0.0, abs(self.current_z)
            
        tc = self.spec.trade
        damp = tc.alarm_damp if self.cooldown > 0 else 1.0
        raw = -(tc.kappa / (tc.sigma * tc.sigma)) * self.current_innov
        scale = tc.kappa / tc.sigma if tc.sigma > 0 else 1.0
        w = np.clip(raw / max(scale, 1e-8), -tc.kelly_cap, tc.kelly_cap) * damp
        
        pos_y = LEG * w
        pos_x = -float(self.filt.beta) * pos_y
        return pos_y, pos_x, abs(self.current_z)

class AtlasBook:
    """
    Manages the portfolio of pair trading agents, handles data catch-up,
    computes PnL-optimized priority scoring, and executes greedy non-overlapping
    capital allocation across all instruments.
    """
    def __init__(self) -> None:
        self.agents = [PairAgent(s) for s in PAIR_CFGS]

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        
        if nt < WARMUP + 10:
            return pos

        for ag in self.agents:
            ag.catch_up(prices)

        cands = []
        for ag in self.agents:
            py, px, az = ag.dollar_positions()
            
            if abs(py) < 1e-9:
                continue
            
            tc = ag.spec.trade
            
            # PnL-Optimized Priority Score: 
            # Multiplies reversion speed (kappa) by absolute Z-score (az). 
            # This targets maximum nominal drift velocity and turnover without 
            # penalizing high-volatility pairs, maximizing raw PnL.
            priority_score = tc.kappa * az
            
            cands.append((priority_score, ag, py, px))
            
        cands.sort(key=lambda c: c[0], reverse=True)
        
        taken = set()
        for priority_score, ag, py, px in cands:
            if ag.spec.y in taken or ag.spec.x in taken:
                continue
            
            taken.add(ag.spec.y)
            taken.add(ag.spec.x)
            
            pos[ag.spec.y] += py
            pos[ag.spec.x] += px
            
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
    
    if len(crash) > 0:
        crash[0] = False 
    dollars[crash] = 0.0
    
    if len(dollars) > 1:
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    
    return (dollars / prcSoFar[:, -1]).astype(int)