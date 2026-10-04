from __future__ import annotations

from math import sqrt
from typing import NamedTuple, NotRequired, TypedDict

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
    RAW_PAIRS: list[RawPairDict] = [
    {
        "y": "AENO",
        "x": "NWIG",
        "q_beta": 1.23e-06,
        "cusum_k": 0.21,
        "h": 14.3,
        "max_cov_inflate": 5.67,
        "z_entry": 1.35,
        "z_exit": 0.017,
        "z_stop": 10.0,
        "kelly_cap": 2.29,
        "alarm_damp": 0.861,
        "kappa": 0.274,
        "sigma": 0.011
    },
    {
        "y": "ILVX",
        "x": "SMAH",
        "q_beta": 1.51e-07,
        "cusum_k": 0.737,
        "h": 8.66,
        "max_cov_inflate": 85.7,
        "z_entry": 1.44,
        "z_exit": 0.00126,
        "z_stop": 10.0,
        "kelly_cap": 2.87,
        "alarm_damp": 0.959,
        "kappa": 0.0505,
        "sigma": 0.01
    },
    {
        "y": "EORC",
        "x": "NGTE",
        "q_beta": 1.21e-06,
        "cusum_k": 0.334,
        "h": 13.0,
        "max_cov_inflate": 13.6,
        "z_entry": 1.45,
        "z_exit": 0.0174,
        "z_stop": 10.0,
        "kelly_cap": 2.26,
        "alarm_damp": 1.0,
        "kappa": 0.141,
        "sigma": 0.0101
    },
    {
        "y": "ACAC",
        "x": "HUXZ",
        "q_beta": 1.04e-07,
        "cusum_k": 1.02,
        "h": 8.82,
        "max_cov_inflate": 10.1,
        "z_entry": 1.49,
        "z_exit": 0.00141,
        "z_stop": 10.0,
        "kelly_cap": 2.56,
        "alarm_damp": 0.748,
        "kappa": 0.0936,
        "sigma": 0.01
    },
    {
        "y": "HETT",
        "x": "ULXY",
        "q_beta": 1.09e-07,
        "cusum_k": 0.313,
        "h": 17.6,
        "max_cov_inflate": 64.7,
        "z_entry": 1.42,
        "z_exit": 0.00244,
        "z_stop": 10.0,
        "kelly_cap": 2.95,
        "alarm_damp": 0.442,
        "kappa": 0.0631,
        "sigma": 0.0108
    },
    {
        "y": "MHRM",
        "x": "SRTX",
        "q_beta": 1.14e-06,
        "cusum_k": 1.69,
        "h": 7.65,
        "max_cov_inflate": 57.8,
        "z_entry": 1.43,
        "z_exit": 0.0156,
        "z_stop": 10.0,
        "kelly_cap": 2.61,
        "alarm_damp": 0.351,
        "kappa": 0.111,
        "sigma": 0.0108
    },
    {
        "y": "EAFC",
        "x": "SRTX",
        "q_beta": 1.33e-07,
        "cusum_k": 0.249,
        "h": 18.0,
        "max_cov_inflate": 20.1,
        "z_entry": 1.43,
        "z_exit": 0.256,
        "z_stop": 10.0,
        "kelly_cap": 2.98,
        "alarm_damp": 0.337,
        "kappa": 0.0222,
        "sigma": 0.0234
    },
    {
        "y": "CCNS",
        "x": "SRTX",
        "q_beta": 5.16e-07,
        "cusum_k": 0.263,
        "h": 9.02,
        "max_cov_inflate": 3.93,
        "z_entry": 1.85,
        "z_exit": 0.00162,
        "z_stop": 10.0,
        "kelly_cap": 3.0,
        "alarm_damp": 0.999,
        "kappa": 0.0335,
        "sigma": 0.0233
    },
    {
        "y": "MSDP",
        "x": "ULXY",
        "q_beta": 2.74e-06,
        "cusum_k": 1.46,
        "h": 18.4,
        "max_cov_inflate": 34.1,
        "z_entry": 1.47,
        "z_exit": 0.00284,
        "z_stop": 10.0,
        "kelly_cap": 2.92,
        "alarm_damp": 0.387,
        "kappa": 0.138,
        "sigma": 0.0108
    },
    {
        "y": "FWWG",
        "x": "BLBT",
        "q_beta": 1.09e-06,
        "cusum_k": 1.01,
        "h": 18.8,
        "max_cov_inflate": 5.29,
        "z_entry": 1.41,
        "z_exit": 0.0134,
        "z_stop": 10.0,
        "kelly_cap": 1.82,
        "alarm_damp": 0.62,
        "kappa": 0.271,
        "sigma": 0.0101
    },
    {
        "y": "MTNS",
        "x": "BENI",
        "q_beta": 4.7e-08,
        "cusum_k": 0.714,
        "h": 8.12,
        "max_cov_inflate": 19.1,
        "z_entry": 1.3,
        "z_exit": 0.00227,
        "z_stop": 10.0,
        "kelly_cap": 1.41,
        "alarm_damp": 1.0,
        "kappa": 0.215,
        "sigma": 0.01
    },
    {
        "y": "HRET",
        "x": "CTGI",
        "q_beta": 1.25e-07,
        "cusum_k": 0.395,
        "h": 6.73,
        "max_cov_inflate": 20.5,
        "z_entry": 1.47,
        "z_exit": 0.00125,
        "z_stop": 10.0,
        "kelly_cap": 1.93,
        "alarm_damp": 0.56,
        "kappa": 0.0674,
        "sigma": 0.01
    },
    {
        "y": "ACIX",
        "x": "ULXY",
        "q_beta": 8.04e-07,
        "cusum_k": 1.93,
        "h": 5.98,
        "max_cov_inflate": 70.2,
        "z_entry": 1.77,
        "z_exit": 0.00267,
        "z_stop": 10.0,
        "kelly_cap": 1.73,
        "alarm_damp": 0.828,
        "kappa": 0.0515,
        "sigma": 0.01
    },
    {
        "y": "DIHO",
        "x": "RTTH",
        "q_beta": 2.83e-06,
        "cusum_k": 1.91,
        "h": 19.4,
        "max_cov_inflate": 36.5,
        "z_entry": 1.28,
        "z_exit": 0.00377,
        "z_stop": 10.0,
        "kelly_cap": 1.01,
        "alarm_damp": 0.962,
        "kappa": 0.0533,
        "sigma": 0.01
    },
    {
        "y": "MHRM",
        "x": "RCRI",
        "q_beta": 1.15e-06,
        "cusum_k": 1.91,
        "h": 20.0,
        "max_cov_inflate": 73.8,
        "z_entry": 1.31,
        "z_exit": 0.0145,
        "z_stop": 10.0,
        "kelly_cap": 2.37,
        "alarm_damp": 0.527,
        "kappa": 0.166,
        "sigma": 0.01
    },
    {
        "y": "HUXZ",
        "x": "NWIG",
        "q_beta": 1.17e-06,
        "cusum_k": 1.81,
        "h": 9.82,
        "max_cov_inflate": 4.93,
        "z_entry": 1.37,
        "z_exit": 0.0163,
        "z_stop": 10.0,
        "kelly_cap": 2.25,
        "alarm_damp": 0.836,
        "kappa": 0.17,
        "sigma": 0.01
    },
    {
        "y": "EORC",
        "x": "SMAH",
        "q_beta": 1.18e-06,
        "cusum_k": 1.76,
        "h": 9.84,
        "max_cov_inflate": 64.0,
        "z_entry": 1.48,
        "z_exit": 0.0152,
        "z_stop": 10.0,
        "kelly_cap": 1.92,
        "alarm_damp": 0.394,
        "kappa": 0.0934,
        "sigma": 0.01
    },
    {
        "y": "EAFC",
        "x": "HRET",
        "q_beta": 2.75e-06,
        "cusum_k": 1.44,
        "h": 19.7,
        "max_cov_inflate": 66.0,
        "z_entry": 1.35,
        "z_exit": 0.00254,
        "z_stop": 10.0,
        "kelly_cap": 2.72,
        "alarm_damp": 0.378,
        "kappa": 0.0509,
        "sigma": 0.0101
    },
    {
        "y": "SMAH",
        "x": "NGTE",
        "q_beta": 1.28e-06,
        "cusum_k": 1.1,
        "h": 8.84,
        "max_cov_inflate": 59.0,
        "z_entry": 1.49,
        "z_exit": 0.0181,
        "z_stop": 10.0,
        "kelly_cap": 2.69,
        "alarm_damp": 0.583,
        "kappa": 0.0696,
        "sigma": 0.0108
    },
    {
        "y": "HETT",
        "x": "SRTX",
        "q_beta": 2.22e-07,
        "cusum_k": 0.591,
        "h": 14.4,
        "max_cov_inflate": 86.0,
        "z_entry": 1.42,
        "z_exit": 0.001,
        "z_stop": 10.0,
        "kelly_cap": 2.35,
        "alarm_damp": 0.878,
        "kappa": 0.181,
        "sigma": 0.0101
    },
    {
        "y": "EAFC",
        "x": "EELT",
        "q_beta": 9.36e-08,
        "cusum_k": 1.77,
        "h": 9.6,
        "max_cov_inflate": 17.9,
        "z_entry": 1.45,
        "z_exit": 0.00188,
        "z_stop": 10.0,
        "kelly_cap": 2.73,
        "alarm_damp": 0.946,
        "kappa": 0.263,
        "sigma": 0.0111
    },
    {
        "y": "ULXY",
        "x": "ELLT",
        "q_beta": 1.29e-06,
        "cusum_k": 0.683,
        "h": 7.91,
        "max_cov_inflate": 95.0,
        "z_entry": 1.43,
        "z_exit": 0.018,
        "z_stop": 10.0,
        "kelly_cap": 1.22,
        "alarm_damp": 0.883,
        "kappa": 0.0656,
        "sigma": 0.0101
    },
    {
        "y": "FWWG",
        "x": "SMAH",
        "q_beta": 1.28e-06,
        "cusum_k": 1.46,
        "h": 14.2,
        "max_cov_inflate": 53.1,
        "z_entry": 1.32,
        "z_exit": 0.0177,
        "z_stop": 10.0,
        "kelly_cap": 2.66,
        "alarm_damp": 0.295,
        "kappa": 0.27,
        "sigma": 0.0101
    },
    {
        "y": "CUBO",
        "x": "ULXY",
        "q_beta": 3.55e-06,
        "cusum_k": 0.53,
        "h": 5.46,
        "max_cov_inflate": 64.8,
        "z_entry": 1.75,
        "z_exit": 0.00133,
        "z_stop": 10.0,
        "kelly_cap": 2.46,
        "alarm_damp": 0.399,
        "kappa": 0.283,
        "sigma": 0.0108
    },
    {
        "y": "ACAC",
        "x": "CTGI",
        "q_beta": 2.7e-07,
        "cusum_k": 0.419,
        "h": 7.28,
        "max_cov_inflate": 43.9,
        "z_entry": 1.36,
        "z_exit": 0.0011,
        "z_stop": 10.0,
        "kelly_cap": 1.38,
        "alarm_damp": 0.968,
        "kappa": 0.169,
        "sigma": 0.01
    },
    {
        "y": "HUXZ",
        "x": "CTGI",
        "q_beta": 8.26e-07,
        "cusum_k": 1.38,
        "h": 13.8,
        "max_cov_inflate": 3.73,
        "z_entry": 2.01,
        "z_exit": 0.00124,
        "z_stop": 10.0,
        "kelly_cap": 2.67,
        "alarm_damp": 0.303,
        "kappa": 0.0616,
        "sigma": 0.0109
    },
    {
        "y": "MHRM",
        "x": "ILVX",
        "q_beta": 1.18e-06,
        "cusum_k": 1.86,
        "h": 17.3,
        "max_cov_inflate": 55.4,
        "z_entry": 1.45,
        "z_exit": 0.0161,
        "z_stop": 10.0,
        "kelly_cap": 1.46,
        "alarm_damp": 0.622,
        "kappa": 0.209,
        "sigma": 0.0103
    },
    {
        "y": "AETS",
        "x": "LSST",
        "q_beta": 2.72e-06,
        "cusum_k": 1.29,
        "h": 6.56,
        "max_cov_inflate": 64.8,
        "z_entry": 1.48,
        "z_exit": 0.00294,
        "z_stop": 10.0,
        "kelly_cap": 2.84,
        "alarm_damp": 0.563,
        "kappa": 0.263,
        "sigma": 0.0113
    },
    {
        "y": "NGTE",
        "x": "ILVX",
        "q_beta": 2.77e-06,
        "cusum_k": 1.99,
        "h": 11.3,
        "max_cov_inflate": 28.2,
        "z_entry": 1.41,
        "z_exit": 0.00581,
        "z_stop": 10.0,
        "kelly_cap": 1.46,
        "alarm_damp": 0.19,
        "kappa": 0.294,
        "sigma": 0.0107
    },
    {
        "y": "HTRK",
        "x": "RTTH",
        "q_beta": 2.89e-08,
        "cusum_k": 0.235,
        "h": 43.7,
        "max_cov_inflate": 84.5,
        "z_entry": 1.22,
        "z_exit": 0.0032,
        "z_stop": 10.0,
        "kelly_cap": 2.66,
        "alarm_damp": 0.457,
        "kappa": 0.0325,
        "sigma": 0.0272
    },
    {
        "y": "NGTE",
        "x": "CTGI",
        "q_beta": 8.97e-08,
        "cusum_k": 1.59,
        "h": 8.58,
        "max_cov_inflate": 29.6,
        "z_entry": 1.37,
        "z_exit": 0.00224,
        "z_stop": 10.0,
        "kelly_cap": 2.65,
        "alarm_damp": 0.189,
        "kappa": 0.19,
        "sigma": 0.01
    },
    {
        "y": "SMAH",
        "x": "CTGI",
        "q_beta": 2.81e-06,
        "cusum_k": 1.44,
        "h": 11.2,
        "max_cov_inflate": 16.4,
        "z_entry": 1.3,
        "z_exit": 0.00376,
        "z_stop": 10.0,
        "kelly_cap": 1.15,
        "alarm_damp": 0.987,
        "kappa": 0.222,
        "sigma": 0.0101
    },
    {
        "y": "DUCT",
        "x": "FWWG",
        "q_beta": 3.18e-06,
        "cusum_k": 1.56,
        "h": 47.4,
        "max_cov_inflate": 43.5,
        "z_entry": 1.98,
        "z_exit": 0.00177,
        "z_stop": 10.0,
        "kelly_cap": 2.21,
        "alarm_damp": 0.201,
        "kappa": 0.0116,
        "sigma": 0.0211
    },
    {
        "y": "BENI",
        "x": "BLBT",
        "q_beta": 2.81e-06,
        "cusum_k": 1.0,
        "h": 12.0,
        "max_cov_inflate": 27.5,
        "z_entry": 1.41,
        "z_exit": 0.00203,
        "z_stop": 10.0,
        "kelly_cap": 2.39,
        "alarm_damp": 0.929,
        "kappa": 0.0598,
        "sigma": 0.01
    },
    {
        "y": "NPCK",
        "x": "NAYO",
        "q_beta": 1.18e-06,
        "cusum_k": 0.277,
        "h": 6.06,
        "max_cov_inflate": 62.6,
        "z_entry": 1.41,
        "z_exit": 0.0153,
        "z_stop": 10.0,
        "kelly_cap": 1.45,
        "alarm_damp": 0.957,
        "kappa": 0.226,
        "sigma": 0.0108
    },
    {
        "y": "BLBT",
        "x": "CTGI",
        "q_beta": 1.23e-06,
        "cusum_k": 0.262,
        "h": 18.4,
        "max_cov_inflate": 45.2,
        "z_entry": 1.38,
        "z_exit": 0.016,
        "z_stop": 10.0,
        "kelly_cap": 1.65,
        "alarm_damp": 0.817,
        "kappa": 0.182,
        "sigma": 0.01
    },
    {
        "y": "FARS",
        "x": "BLBT",
        "q_beta": 1.16e-06,
        "cusum_k": 1.54,
        "h": 11.4,
        "max_cov_inflate": 19.8,
        "z_entry": 1.42,
        "z_exit": 0.0154,
        "z_stop": 10.0,
        "kelly_cap": 2.88,
        "alarm_damp": 0.163,
        "kappa": 0.179,
        "sigma": 0.0105
    },
    {
        "y": "DIHO",
        "x": "FCSG",
        "q_beta": 1.25e-06,
        "cusum_k": 1.03,
        "h": 15.4,
        "max_cov_inflate": 58.4,
        "z_entry": 1.35,
        "z_exit": 0.0175,
        "z_stop": 10.0,
        "kelly_cap": 1.85,
        "alarm_damp": 0.865,
        "kappa": 0.202,
        "sigma": 0.0104
    },
    {
        "y": "LSST",
        "x": "BENI",
        "q_beta": 1.2e-06,
        "cusum_k": 1.73,
        "h": 6.7,
        "max_cov_inflate": 65.3,
        "z_entry": 1.4,
        "z_exit": 0.0159,
        "z_stop": 10.0,
        "kelly_cap": 2.77,
        "alarm_damp": 0.928,
        "kappa": 0.277,
        "sigma": 0.01
    },
    {
        "y": "FWWG",
        "x": "FARS",
        "q_beta": 1.22e-06,
        "cusum_k": 1.76,
        "h": 7.11,
        "max_cov_inflate": 38.0,
        "z_entry": 1.32,
        "z_exit": 0.0179,
        "z_stop": 10.0,
        "kelly_cap": 1.24,
        "alarm_damp": 0.627,
        "kappa": 0.292,
        "sigma": 0.0109
    },
    {
        "y": "SRTX",
        "x": "MDGI",
        "q_beta": 8.11e-07,
        "cusum_k": 0.945,
        "h": 5.81,
        "max_cov_inflate": 4.31,
        "z_entry": 1.8,
        "z_exit": 0.00145,
        "z_stop": 10.0,
        "kelly_cap": 1.56,
        "alarm_damp": 0.946,
        "kappa": 0.263,
        "sigma": 0.0101
    },
    {
        "y": "CTGI",
        "x": "MDGI",
        "q_beta": 8.56e-08,
        "cusum_k": 0.371,
        "h": 7.0,
        "max_cov_inflate": 3.48,
        "z_entry": 1.43,
        "z_exit": 0.00207,
        "z_stop": 10.0,
        "kelly_cap": 2.77,
        "alarm_damp": 0.801,
        "kappa": 0.204,
        "sigma": 0.01
    },
    {
        "y": "HRND",
        "x": "RRES",
        "q_beta": 1.15e-06,
        "cusum_k": 0.474,
        "h": 9.43,
        "max_cov_inflate": 70.4,
        "z_entry": 1.45,
        "z_exit": 0.0146,
        "z_stop": 10.0,
        "kelly_cap": 2.54,
        "alarm_damp": 0.716,
        "kappa": 0.0706,
        "sigma": 0.0108
    },
    {
        "y": "HTRK",
        "x": "MSDP",
        "q_beta": 1.23e-06,
        "cusum_k": 0.571,
        "h": 17.7,
        "max_cov_inflate": 78.4,
        "z_entry": 1.43,
        "z_exit": 0.0173,
        "z_stop": 10.0,
        "kelly_cap": 2.16,
        "alarm_damp": 0.91,
        "kappa": 0.265,
        "sigma": 0.01
    },
    {
        "y": "EORC",
        "x": "ILVX",
        "q_beta": 3.5e-06,
        "cusum_k": 0.38,
        "h": 7.59,
        "max_cov_inflate": 53.2,
        "z_entry": 1.92,
        "z_exit": 0.00201,
        "z_stop": 10.0,
        "kelly_cap": 2.49,
        "alarm_damp": 0.613,
        "kappa": 0.175,
        "sigma": 0.0108
    },
    {
        "y": "ELLT",
        "x": "IHOZ",
        "q_beta": 2.22e-07,
        "cusum_k": 0.591,
        "h": 14.4,
        "max_cov_inflate": 86.0,
        "z_entry": 1.42,
        "z_exit": 0.001,
        "z_stop": 10.0,
        "kelly_cap": 2.35,
        "alarm_damp": 0.878,
        "kappa": 0.181,
        "sigma": 0.0101
    },
    {
        "y": "MHRM",
        "x": "RRES",
        "q_beta": 1.64e-06,
        "cusum_k": 0.202,
        "h": 37.2,
        "max_cov_inflate": 10.4,
        "z_entry": 1.02,
        "z_exit": 0.00156,
        "z_stop": 10.0,
        "kelly_cap": 2.15,
        "alarm_damp": 0.518,
        "kappa": 0.0128,
        "sigma": 0.0266
    },
    {
        "y": "ITPA",
        "x": "ILVX",
        "q_beta": 1.43e-08,
        "cusum_k": 0.365,
        "h": 48.3,
        "max_cov_inflate": 77.4,
        "z_entry": 1.02,
        "z_exit": 0.0155,
        "z_stop": 10.0,
        "kelly_cap": 2.94,
        "alarm_damp": 0.234,
        "kappa": 0.0116,
        "sigma": 0.0172
    },
    {
        "y": "ILVX",
        "x": "HETT",
        "q_beta": 8.13e-07,
        "cusum_k": 0.577,
        "h": 7.99,
        "max_cov_inflate": 97.7,
        "z_entry": 2.04,
        "z_exit": 0.00167,
        "z_stop": 10.0,
        "kelly_cap": 2.31,
        "alarm_damp": 0.755,
        "kappa": 0.158,
        "sigma": 0.0108
    },
    {
        "y": "SRNA",
        "x": "IHOZ",
        "q_beta": 1.22e-06,
        "cusum_k": 1.8,
        "h": 19.9,
        "max_cov_inflate": 6.73,
        "z_entry": 1.45,
        "z_exit": 0.017,
        "z_stop": 10.0,
        "kelly_cap": 1.29,
        "alarm_damp": 0.418,
        "kappa": 0.0561,
        "sigma": 0.011
    }
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