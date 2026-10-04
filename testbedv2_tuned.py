from __future__ import annotations

from math import sqrt
from typing import NamedTuple, TypedDict

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
_OU_WINDOW = 100    # Fixed constant

_EPS_DENOM = 1e-12
_EPS_VAR = 1e-18

class RawPairDict(TypedDict):
    y: str
    x: str
    q_beta: float
    cusum_k: float
    h: float
    max_cov_inflate: float
    cov_decay: float
    z_entry: float
    z_exit: float
    z_stop: float

class FiltCfg(NamedTuple):
    q_beta: float
    ewma_lambda: float = 0.97
    cusum_k: float = 0.5
    h: float = 8.0
    vol_cut: bool = True
    max_cov_inflate: float = 25.0
    cov_decay: float = 0.95

class TradeCfg(NamedTuple):
    z_entry: float
    z_exit: float
    z_stop: float

class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
    filt: FiltCfg
    trade: TradeCfg

class StepOut(NamedTuple):
    beta: float
    innov: float
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
        self.decay_multiplier = 1.0
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
        if s_pred < _EPS_DENOM: s_pred = _EPS_DENOM
            
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

        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        self.run_m_up = t if (self.cm_up > 0.0 and prev_up == 0.0) else (None if self.cm_up == 0.0 else self.run_m_up)
        self.run_m_dn = t if (self.cm_dn > 0.0 and prev_dn == 0.0) else (None if self.cm_dn == 0.0 else self.run_m_dn)

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
                self.cv_up = max(0.0, self.cv_up - cfg.h)
                self.cv_dn = max(0.0, self.cv_dn - cfg.h)
                self.run_v_up = self.run_v_dn = None

        if mean_alarm or (cfg.vol_cut and vol_alarm):
            cut = self._trigger_segment_cut(t)
        return cut, vol_shift

    def _trigger_segment_cut(self, t: int) -> bool:
        cfg = self.cfg
        best = -1.0
        cut_bar = None
        
        if self.cm_up > cfg.h and self.cm_up > best: best, cut_bar = self.cm_up, self.run_m_up
        if self.cm_dn > cfg.h and self.cm_dn > best: best, cut_bar = self.cm_dn, self.run_m_dn
        if cfg.vol_cut and self.cv_up > cfg.h and self.cv_up > best: best, cut_bar = self.cv_up, self.run_v_up
        if cfg.vol_cut and self.cv_dn > cfg.h and self.cv_dn > best: best, cut_bar = self.cv_dn, self.run_v_dn
            
        cut_bar = max(int(cut_bar if cut_bar is not None else t), (self.seg_start or 0) + 1)
        
        self.cm_up = max(0.0, self.cm_up - cfg.h)
        self.cm_dn = max(0.0, self.cm_dn - cfg.h)
        self.cv_up = max(0.0, self.cv_up - cfg.h)
        self.cv_dn = max(0.0, self.cv_dn - cfg.h)
        
        self.run_m_up = self.run_m_dn = None
        self.run_v_up = self.run_v_dn = None
        
        self.seg_start = cut_bar
        return True

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

        b = float(self.beta)
        innov_pred = yf - b * xf
        z = self._update_variances(innov_pred, innov_pred)
        
        cut, vol_shift = False, False
        if t >= WARMUP:
            cut, vol_shift = self._evaluate_cusum(z, t)
            
        if cut:
            target_multiplier = cfg.max_cov_inflate
        else:
            max_cusum = max(self.cm_up, self.cm_dn, self.cv_up, self.cv_dn)
            excess = max(0.0, max_cusum - cfg.h)
            if excess > 0.0:
                target_multiplier = 1.0 + (cfg.max_cov_inflate - 1.0) * (1.0 - np.exp(-excess))
            else:
                target_multiplier = 1.0
                
        self.decay_multiplier = max(target_multiplier, self.decay_multiplier * (1 - cfg.cov_decay))
        dynamic_q = cfg.q_beta * self.decay_multiplier
        adapting = self.decay_multiplier > 1.01

        innov, beta_new, p22_new = self._update_kalman(yf, xf, dynamic_q)
        resid = yf - beta_new * xf

        return StepOut(
            beta=float(beta_new), innov=float(innov), spread=float(resid),
            z=float(z), h=float(cfg.h), cut=cut, adapting=adapting,
            vol_shift=vol_shift, cm_up=float(self.cm_up), cm_dn=float(self.cm_dn),
            cv=max(self.cv_up, self.cv_dn),
        )

# BEGIN GENERATED AUTO PAIR PARAMETERS
# Source: partial_tuning_data.db; performance-filtered rows: 310
# Gate: train score >= 2, validation score >= 2,
# validation/train score retention >= 25%, and positive
# mean P&L, Sharpe, and return in both stored periods.
_TUNED_PAIR_PARAMS: tuple[
    tuple[str, str, float, float, float, float, float, float, float, float],
    ...,
] = (
    ('ACAC', 'BENI', 4.85e-08, 0.654, 4.04, 4090000.0, 0.0558, 1.13, 0.001, 10.0),
    ('ACAC', 'FARS', 1.04e-06, 1.0, 17.4, 96900.0, 0.0876, 1.17, 0.001, 10.0),
    ('ACAC', 'GARI', 5.54e-06, 1.74, 7.08, 62.3, 0.00541, 1.47, 0.001, 10.0),
    ('ACAC', 'IHOZ', 7.55e-07, 0.516, 15.0, 1.94, 0.000728, 2.09, 0.001, 10.0),
    ('ACAC', 'ITPA', 6.2e-07, 0.177, 12.0, 96900.0, 0.0573, 1.36, 0.001, 10.0),
    ('ACAC', 'NAYO', 0.000318, 1.75, 11.6, 7.2, 0.000389, 1.15, 0.001, 10.0),
    ('AENO', 'ACAC', 2.71e-07, 1.51, 8.08, 11.0, 2.75e-05, 1.24, 0.001, 10.0),
    ('AENO', 'ACIX', 3.29e-06, 0.333, 17.5, 12200.0, 0.000467, 1.44, 0.001, 10.0),
    ('AENO', 'AETS', 2.66e-07, 0.0714, 2.66, 10500.0, 0.0122, 1.6, 0.001, 10.0),
    ('AENO', 'ANSO', 0.000318, 0.823, 6.31, 10400000.0, 0.000844, 1.78, 0.001, 10.0),
    ('AENO', 'DUCT', 6.2e-07, 1.62, 4.09, 96900.0, 0.0281, 1.13, 0.001, 10.0),
    ('AENO', 'FARS', 2.35e-06, 0.405, 3.32, 4810000.0, 0.00382, 1.79, 0.001, 10.0),
    ('AENO', 'FWWG', 0.000145, 0.67, 19.8, 5.37, 0.000225, 1.85, 0.001, 10.0),
    ('AENO', 'HRET', 0.000122, 0.457, 19.2, 240000.0, 0.0121, 1.02, 0.001, 10.0),
    ('AENO', 'HUXZ', 1.09e-05, 0.711, 13.4, 4.71, 0.0758, 1.6, 0.001, 10.0),
    ('AENO', 'IHOZ', 1.29e-05, 0.744, 3.32, 5.37, 0.0114, 1.31, 0.001, 10.0),
    ('AENO', 'MTNS', 0.000289, 1.62, 14.5, 5.37, 0.00706, 1.84, 0.001, 10.0),
    ('AENO', 'NGTE', 0.000323, 0.711, 10.3, 622.0, 0.000535, 1.09, 0.001, 10.0),
    ('AENO', 'NWIG', 1.44e-06, 0.711, 18.3, 5.37, 0.000189, 1.02, 0.001, 10.0),
    ('AENO', 'RTTH', 2.19e-05, 1.12, 3.75, 43.6, 0.000322, 1.59, 0.001, 10.0),
    ('AENO', 'SMAH', 2.29e-05, 1.11, 7.99, 7040.0, 0.0146, 1.98, 0.001, 10.0),
    ('AENO', 'SPLZ', 7.55e-07, 0.333, 7.85, 13100.0, 0.000273, 1.94, 0.001, 10.0),
    ('AENO', 'SRNA', 0.000348, 0.513, 6.31, 159000.0, 0.0876, 1.48, 0.001, 10.0),
    ('AENO', 'SRTX', 0.000289, 1.41, 15.3, 7840000.0, 0.000308, 1.36, 0.001, 10.0),
    ('AGVF', 'ACAC', 1.38e-05, 1.11, 4.16, 159000.0, 0.0123, 1.32, 0.001, 10.0),
    ('AGVF', 'AETS', 7.08e-06, 1.32, 8.08, 4.71, 0.00964, 1.65, 0.001, 10.0),
    ('AGVF', 'CTGI', 1.38e-05, 0.25, 8.31, 4040000.0, 0.0123, 1.45, 0.001, 10.0),
    ('AGVF', 'FCSG', 0.000318, 1.41, 7.81, 200000.0, 0.00801, 1.64, 0.001, 10.0),
    ('AGVF', 'HRND', 0.00051, 0.333, 8.36, 915.0, 1.15e-05, 1.14, 0.001, 10.0),
    ('AGVF', 'ITPA', 4.12e-06, 1.42, 8.46, 10400000.0, 0.0354, 1.78, 0.001, 10.0),
    ('AGVF', 'NGTE', 0.000348, 0.756, 19.2, 3320.0, 0.000102, 1.69, 0.001, 10.0),
    ('AGVF', 'RCRI', 5.41e-06, 0.223, 4.11, 1.94, 0.00112, 1.65, 0.001, 10.0),
    ('AGVF', 'ULXY', 9.47e-05, 0.0954, 5.63, 47300.0, 0.000147, 1.59, 0.001, 10.0),
    ('ALGO', 'ACIX', 5.54e-06, 1.53, 10.0, 607.0, 0.0363, 1.4, 0.001, 10.0),
    ('ALGO', 'AETS', 2.21e-08, 1.74, 15.7, 1970.0, 0.000111, 1.79, 0.001, 10.0),
    ('ALGO', 'ANSO', 0.000348, 1.08, 4.78, 7070.0, 0.0573, 1.48, 0.001, 10.0),
    ('ALGO', 'CUBO', 2.06e-06, 0.188, 16.4, 29.3, 0.000844, 2.13, 0.001, 10.0),
    ('ALGO', 'FCSG', 0.000318, 1.48, 15.3, 2350.0, 0.00801, 2.14, 0.001, 10.0),
    ('ALGO', 'HRET', 6.86e-06, 0.439, 13.4, 13000.0, 0.000535, 1.69, 0.001, 10.0),
    ('ALGO', 'LSST', 2.29e-07, 1.65, 6.58, 203000.0, 0.0729, 1.32, 0.001, 10.0),
    ('ALGO', 'MHRM', 2.12e-05, 1.75, 3.47, 23300000.0, 0.078, 2.04, 0.001, 10.0),
    ('ALGO', 'NGTE', 1.35e-05, 0.226, 8.39, 3320.0, 0.000108, 1.14, 0.001, 10.0),
    ('ALGO', 'NPCK', 5.79e-08, 1.32, 17.6, 1900000.0, 0.000225, 2.15, 0.001, 10.0),
    ('ALGO', 'OTCS', 8.03e-07, 1.38, 7.99, 118000.0, 1.4e-05, 1.44, 0.001, 10.0),
    ('ALGO', 'SPLZ', 3.81e-08, 0.381, 7.7, 96900.0, 0.0555, 2.05, 0.001, 10.0),
    ('ALUT', 'ACIX', 7.44e-07, 0.177, 6.58, 1240.0, 0.0231, 1.28, 0.001, 10.0),
    ('ALUT', 'BLBT', 4.81e-08, 0.185, 15.3, 4.13, 1.4e-05, 1.52, 0.001, 10.0),
    ('ALUT', 'CCNS', 1.64e-07, 0.641, 15.4, 5.37, 0.00112, 1.2, 0.001, 10.0),
    ('ALUT', 'FWWG', 1.52e-08, 1.74, 17.5, 5980.0, 0.0119, 1.77, 0.001, 10.0),
    ('ALUT', 'IHOZ', 1.39e-07, 1.24, 3.39, 13.1, 0.000296, 1.86, 0.001, 10.0),
    ('ALUT', 'RCRI', 4.32e-08, 1.92, 12.6, 103.0, 0.0031, 1.45, 0.001, 10.0),
    ('ALUT', 'SRTX', 1.95e-08, 1.22, 16.5, 2000.0, 0.00169, 1.54, 0.001, 10.0),
    ('AMRP', 'AGVF', 1.85e-06, 1.07, 11.4, 4.24, 2.95e-05, 1.88, 0.001, 10.0),
    ('AMRP', 'ALUT', 5.68e-05, 1.98, 7.85, 31.6, 0.0491, 1.86, 0.001, 10.0),
    ('AMRP', 'ANSO', 0.000365, 1.26, 5.79, 356.0, 0.0876, 2.26, 0.001, 10.0),
    ('AMRP', 'HETT', 3.14e-08, 1.3, 15.0, 13.1, 0.00984, 1.37, 0.001, 10.0),
    ('AMRP', 'RRES', 1.32e-07, 1.48, 16.6, 20.0, 0.0121, 1.64, 0.001, 10.0),
    ('AMRP', 'RTTH', 7.62e-05, 1.68, 11.7, 78000000.0, 0.0711, 1.68, 0.001, 10.0),
    ('AMRP', 'SRTX', 1.39e-07, 0.193, 14.5, 19.5, 4.78e-05, 1.82, 0.001, 10.0),
    ('AMRP', 'ULXY', 1.64e-07, 1.05, 14.7, 5.03, 0.0354, 1.86, 0.001, 10.0),
    ('ANSO', 'ACAC', 4.24e-07, 1.63, 11.6, 6220000.0, 0.000986, 1.17, 0.001, 10.0),
    ('ANSO', 'BENI', 4.54e-06, 0.195, 6.99, 203000.0, 0.000634, 1.38, 0.001, 10.0),
    ('ANSO', 'BLBT', 1.91e-06, 0.48, 6.46, 21800.0, 0.000192, 1.38, 0.001, 10.0),
    ('ANSO', 'CCNS', 2.43e-08, 1.03, 19.2, 1590000.0, 0.00637, 2.15, 0.001, 10.0),
    ('ANSO', 'CTGI', 2.94e-08, 0.0691, 7.24, 29.3, 0.00637, 1.79, 0.001, 10.0),
    ('ANSO', 'HRND', 3.02e-05, 0.672, 11.7, 639000.0, 0.0146, 1.13, 0.001, 10.0),
    ('ANSO', 'IHOZ', 2.37e-07, 0.435, 3.53, 9250000.0, 0.0491, 2.04, 0.001, 10.0),
    ('ANSO', 'ITPA', 0.000122, 1.06, 5.14, 1240.0, 0.0558, 1.28, 0.001, 10.0),
    ('ANSO', 'MTNS', 0.000244, 1.48, 19.5, 54800.0, 3.63e-05, 1.02, 0.001, 10.0),
    ('ANSO', 'NGTE', 2.35e-06, 1.82, 5.14, 3410000.0, 0.0628, 1.09, 0.001, 10.0),
    ('ANSO', 'RCRI', 9.47e-08, 1.83, 8.02, 53.3, 1.27e-05, 1.09, 0.001, 10.0),
    ('ANSO', 'RRES', 1.44e-06, 1.33, 14.4, 1.57, 0.00678, 1.47, 0.001, 10.0),
    ('ANSO', 'RTTH', 2.66e-07, 1.24, 5.33, 8230.0, 1.61e-05, 1.48, 0.001, 10.0),
    ('CTGI', 'EELT', 1.91e-06, 1.08, 12.0, 3.89, 2.95e-05, 1.17, 0.001, 10.0),
    ('CTGI', 'FARS', 1.07e-08, 1.43, 5.36, 1590000.0, 0.0729, 1.36, 0.001, 10.0),
    ('CTGI', 'FCSG', 4.32e-08, 1.55, 7.32, 73.9, 0.00774, 1.09, 0.001, 10.0),
    ('CTGI', 'ILVX', 8.9e-06, 1.59, 13.9, 462000.0, 0.0123, 1.62, 0.001, 10.0),
    ('CTGI', 'ITPA', 1.95e-08, 1.98, 9.35, 1510000.0, 0.0801, 1.35, 0.001, 10.0),
    ('CTGI', 'NGTE', 4.35e-06, 0.744, 17.7, 66900.0, 0.00116, 1.14, 0.001, 10.0),
    ('CUBO', 'ACAC', 0.000348, 0.45, 7.04, 1590000.0, 6.66e-05, 1.02, 0.001, 10.0),
    ('CUBO', 'ACIX', 2.49e-05, 1.74, 16.6, 147.0, 0.00234, 1.15, 0.001, 10.0),
    ('CUBO', 'ANSO', 0.000122, 0.0884, 8.92, 1590000.0, 0.0573, 1.21, 0.001, 10.0),
    ('CUBO', 'GARI', 5.41e-06, 1.6, 2.93, 9140.0, 0.0573, 1.56, 0.001, 10.0),
    ('CUBO', 'HRET', 4.05e-08, 0.734, 16.4, 78500000.0, 0.0123, 1.2, 0.001, 10.0),
    ('CUBO', 'IHOZ', 0.000273, 0.343, 6.31, 86000000.0, 0.0297, 1.55, 0.001, 10.0),
    ('CUBO', 'ITPA', 1.38e-05, 0.436, 13.8, 4.24, 0.00183, 1.75, 0.001, 10.0),
    ('CUBO', 'NAYO', 0.000145, 1.68, 12.5, 88300.0, 0.0558, 1.24, 0.001, 10.0),
    ('CUBO', 'RTTH', 8.75e-05, 1.4, 16.4, 9280000.0, 0.00457, 1.14, 0.001, 10.0),
    ('CUBO', 'SPLZ', 0.000111, 1.3, 11.7, 4.71, 0.000133, 2.04, 0.001, 10.0),
    ('CUBO', 'ULXY', 1.26e-05, 1.85, 19.9, 31.1, 1.61e-05, 1.99, 0.001, 10.0),
    ('DIHO', 'BLBT', 3.36e-06, 0.226, 16.5, 1.36, 0.00746, 1.54, 0.001, 10.0),
    ('DIHO', 'FCSG', 2.2e-07, 1.32, 7.81, 10500.0, 0.0523, 1.47, 0.001, 10.0),
    ('DUCT', 'ACAC', 1.91e-06, 0.247, 11.7, 13000.0, 0.0729, 1.15, 0.001, 10.0),
    ('DUCT', 'AGVF', 1.29e-05, 0.185, 2.08, 96900.0, 2.08e-05, 1.05, 0.001, 10.0),
    ('DUCT', 'ANSO', 0.000348, 0.333, 6.31, 26.3, 0.00427, 1.48, 0.001, 10.0),
    ('DUCT', 'FARS', 1.95e-08, 1.22, 7.07, 4810000.0, 0.000192, 1.24, 0.001, 10.0),
    ('DUCT', 'FCSG', 9.47e-08, 0.151, 15.3, 417000.0, 0.000274, 1.59, 0.001, 10.0),
    ('DUCT', 'HRND', 0.000704, 0.189, 3.5, 426.0, 0.0253, 1.13, 0.001, 10.0),
    ('DUCT', 'ITPA', 8.03e-07, 0.951, 7.38, 10500.0, 0.0579, 1.09, 0.001, 10.0),
    ('DUCT', 'MTNS', 1.68e-05, 0.439, 7.38, 120000.0, 0.0579, 1.09, 0.001, 10.0),
    ('DUCT', 'NGTE', 8.52e-07, 0.386, 4.09, 96900.0, 3.08e-05, 1.13, 0.001, 10.0),
    ('DUCT', 'RCRI', 1.44e-06, 1.25, 3.88, 8.92, 0.0523, 1.77, 0.001, 10.0),
    ('DUCT', 'SPLZ', 0.000595, 1.6, 11.0, 3320.0, 0.00119, 1.85, 0.001, 10.0),
    ('DUCT', 'SRTX', 1.48e-06, 1.25, 3.75, 12200.0, 0.000296, 1.09, 0.001, 10.0),
    ('ELLT', 'ACIX', 7.44e-07, 1.52, 4.09, 96900.0, 3.36e-05, 1.13, 0.001, 10.0),
    ('ELLT', 'AETS', 4.98e-07, 0.52, 6.43, 29.3, 0.000404, 1.06, 0.001, 10.0),
    ('ELLT', 'CTGI', 1.44e-06, 0.711, 12.7, 50.5, 0.00801, 2.05, 0.001, 10.0),
    ('ELLT', 'DIHO', 2.07e-08, 1.71, 16.7, 293000.0, 0.0123, 1.4, 0.001, 10.0),
    ('ELLT', 'DUCT', 4.18e-07, 0.724, 4.09, 96900.0, 0.00143, 1.13, 0.001, 10.0),
    ('ELLT', 'EORC', 1.34e-08, 1.72, 11.1, 4810000.0, 0.0709, 1.65, 0.001, 10.0),
    ('ELLT', 'FCSG', 0.000595, 1.76, 8.31, 52700.0, 0.000278, 1.09, 0.001, 10.0),
    ('ELLT', 'HRET', 5.07e-08, 1.25, 3.75, 375000.0, 0.0123, 1.35, 0.001, 10.0),
    ('ELLT', 'HRND', 7.62e-05, 0.157, 4.16, 86.1, 6.78e-05, 1.67, 0.001, 10.0),
    ('ELLT', 'HTRK', 1.85e-06, 1.32, 11.4, 43.6, 0.0528, 1.59, 0.001, 10.0),
    ('ELLT', 'HUXZ', 1.08e-05, 1.48, 2.21, 61600.0, 0.0363, 1.14, 0.001, 10.0),
    ('ELLT', 'MMBT', 1.35e-05, 1.29, 14.0, 3320.0, 1.76e-05, 1.85, 0.001, 10.0),
    ('ELLT', 'NGTE', 7.69e-05, 1.01, 9.55, 5.37, 0.0206, 1.26, 0.001, 10.0),
    ('ELLT', 'SMAH', 1.23e-06, 1.53, 2.08, 13000.0, 0.00234, 1.82, 0.001, 10.0),
    ('ELLT', 'ULXY', 1.09e-05, 1.83, 11.8, 24.3, 0.0884, 1.82, 0.001, 10.0),
    ('EORC', 'AETS', 5.41e-06, 1.4, 5.33, 2000.0, 0.0558, 1.65, 0.001, 10.0),
    ('EORC', 'ANSO', 0.000123, 0.948, 8.41, 12200.0, 1.4e-05, 1.44, 0.001, 10.0),
    ('EORC', 'CCNS', 4.12e-06, 1.74, 8.46, 10400000.0, 0.0922, 1.78, 0.001, 10.0),
    ('EORC', 'FCSG', 2.33e-07, 1.36, 9.35, 94800000.0, 0.0195, 2.14, 0.001, 10.0),
    ('EORC', 'MTNS', 9.81e-05, 1.42, 13.7, 2320000.0, 0.0363, 1.68, 0.001, 10.0),
    ('EORC', 'NAYO', 1.85e-06, 1.07, 18.1, 915.0, 1.05e-05, 1.88, 0.001, 10.0),
    ('EORC', 'NGTE', 1.63e-07, 0.151, 17.8, 426.0, 0.00678, 1.02, 0.001, 10.0),
    ('EORC', 'RTTH', 5.68e-05, 0.247, 5.7, 2.33, 0.0253, 1.33, 0.001, 10.0),
    ('EORC', 'ULXY', 6.2e-07, 0.978, 13.2, 96900.0, 1.52e-05, 1.13, 0.001, 10.0),
    ('GARI', 'ACIX', 7.08e-06, 1.82, 5.69, 147.0, 8.79e-05, 2.09, 0.001, 10.0),
    ('GARI', 'IHOZ', 1.8e-05, 0.188, 2.48, 5.7, 0.00215, 1.02, 0.001, 10.0),
    ('GARI', 'MTNS', 1.48e-06, 1.81, 6.88, 186000.0, 0.000984, 1.09, 0.001, 10.0),
    ('HETT', 'ACIX', 9.47e-08, 1.37, 6.92, 2.38, 0.0555, 1.59, 0.001, 10.0),
    ('HETT', 'AETS', 1.04e-06, 1.93, 3.53, 11.8, 0.000535, 1.77, 0.001, 10.0),
    ('HETT', 'EELT', 4.24e-07, 1.32, 5.33, 6220000.0, 0.0573, 2.14, 0.001, 10.0),
    ('HETT', 'MDGI', 1.6e-08, 0.322, 13.7, 16.2, 0.0832, 1.45, 0.001, 10.0),
    ('HETT', 'MSDP', 0.000168, 1.62, 17.7, 31.1, 0.0372, 1.09, 0.001, 10.0),
    ('HETT', 'MTNS', 1.09e-05, 1.62, 6.67, 31.1, 0.0297, 1.09, 0.001, 10.0),
    ('HETT', 'NWIG', 0.000318, 1.68, 18.3, 1.57, 0.000296, 2.67, 0.001, 10.0),
    ('HETT', 'RTTH', 3.33e-07, 1.29, 14.0, 3320.0, 1.52e-05, 1.02, 0.001, 10.0),
    ('HETT', 'SPLZ', 4.14e-05, 1.6, 6.92, 24.3, 6.88e-05, 1.99, 0.001, 10.0),
    ('HETT', 'ULXY', 1.87e-06, 1.94, 5.33, 1590000.0, 0.000108, 1.24, 0.001, 10.0),
    ('HRET', 'ACAC', 8.9e-06, 1.32, 14.5, 8570.0, 0.00706, 1.44, 0.001, 10.0),
    ('HRET', 'AETS', 1.83e-07, 1.29, 16.1, 356.0, 0.0478, 1.85, 0.001, 10.0),
    ('HRET', 'BENI', 3.48e-06, 0.189, 20.0, 1010000.0, 0.0573, 1.13, 0.001, 10.0),
    ('HRET', 'FARS', 0.000348, 1.85, 11.7, 14200.0, 1.81e-05, 1.48, 0.001, 10.0),
    ('HRET', 'FWWG', 1.91e-06, 0.555, 5.9, 13000.0, 0.0363, 1.09, 0.001, 10.0),
    ('HRET', 'HRND', 1.09e-05, 0.613, 5.07, 4810000.0, 0.0729, 1.2, 0.001, 10.0),
    ('HRET', 'HTRK', 2.27e-08, 1.94, 16.6, 126000.0, 0.0523, 1.2, 0.001, 10.0),
    ('HRET', 'MDGI', 1.07e-08, 0.457, 18.6, 200000.0, 0.00116, 1.44, 0.001, 10.0),
    ('HRET', 'MHRM', 2.77e-07, 1.18, 13.4, 53.3, 1.4e-05, 1.17, 0.001, 10.0),
    ('HRET', 'MTNS', 2.94e-08, 1.18, 13.7, 78300.0, 0.00978, 2.01, 0.001, 10.0),
    ('HRET', 'NGTE', 2.66e-07, 0.226, 19.5, 10500.0, 0.00681, 1.02, 0.001, 10.0),
    ('HRET', 'NWIG', 4.11e-06, 1.32, 13.3, 14200000.0, 0.0573, 1.99, 0.001, 10.0),
    ('HRET', 'RTTH', 0.000545, 1.3, 7.17, 10.7, 0.000151, 1.65, 0.001, 10.0),
    ('HRET', 'SPLZ', 0.00045, 0.333, 12.1, 98.7, 0.000296, 1.09, 0.001, 10.0),
    ('HRET', 'SRTX', 1.91e-06, 1.6, 3.76, 3320.0, 0.0195, 1.72, 0.001, 10.0),
    ('HRET', 'ULXY', 1.48e-06, 0.333, 12.4, 78300.0, 1.15e-05, 1.09, 0.001, 10.0),
    ('HUXZ', 'ACAC', 8.52e-07, 1.25, 12.7, 4040000.0, 0.000108, 1.46, 0.001, 10.0),
    ('HUXZ', 'AETS', 4.32e-08, 1.68, 10.6, 2000.0, 0.0876, 1.48, 0.001, 10.0),
    ('HUXZ', 'ALUT', 6.63e-08, 1.18, 19.5, 1510.0, 0.00345, 1.45, 0.001, 10.0),
    ('HUXZ', 'BLBT', 1.43e-06, 0.615, 7.24, 26.3, 0.0922, 1.86, 0.001, 10.0),
    ('HUXZ', 'CCNS', 1.34e-08, 1.32, 17.6, 8.92, 1.21e-05, 1.09, 0.001, 10.0),
    ('HUXZ', 'DUCT', 1.47e-05, 0.185, 5.33, 96900.0, 0.00157, 1.13, 0.001, 10.0),
    ('HUXZ', 'FCSG', 1.39e-07, 1.47, 15.1, 669000.0, 2.36e-05, 2.26, 0.001, 10.0),
    ('HUXZ', 'FWWG', 1.31e-07, 1.41, 12.0, 679000.0, 0.00601, 1.25, 0.001, 10.0),
    ('HUXZ', 'HRET', 5.07e-07, 0.613, 2.66, 12200.0, 0.00106, 1.13, 0.001, 10.0),
    ('HUXZ', 'IHOZ', 5.41e-06, 1.59, 16.1, 1590000.0, 0.000109, 1.61, 0.001, 10.0),
    ('HUXZ', 'MTNS', 2.63e-08, 0.615, 18.7, 14400.0, 0.00637, 1.15, 0.001, 10.0),
    ('HUXZ', 'NAYO', 1.2e-05, 1.25, 7.7, 5.37, 0.00678, 1.09, 0.001, 10.0),
    ('HUXZ', 'NGTE', 1.83e-07, 1.24, 2.66, 10500.0, 0.00116, 1.47, 0.001, 10.0),
    ('HUXZ', 'NWIG', 9.47e-08, 0.333, 11.4, 53.3, 0.0876, 1.65, 0.001, 10.0),
    ('HUXZ', 'RRES', 4.34e-07, 0.824, 15.0, 11.8, 1.56e-05, 2.19, 0.001, 10.0),
    ('HUXZ', 'ULXY', 6.49e-07, 0.518, 5.33, 657000.0, 0.0122, 1.15, 0.001, 10.0),
    ('LSST', 'ACIX', 1.68e-05, 1.49, 15.1, 10500.0, 0.0801, 1.47, 0.001, 10.0),
    ('LSST', 'AETS', 1.19e-05, 1.43, 11.6, 21800.0, 1.4e-05, 2.09, 0.001, 10.0),
    ('LSST', 'AMRP', 3.73e-07, 1.82, 3.39, 2000.0, 0.0491, 1.85, 0.001, 10.0),
    ('LSST', 'ANSO', 8.87e-05, 0.567, 3.64, 14000.0, 0.078, 1.17, 0.001, 10.0),
    ('LSST', 'BLBT', 7.46e-07, 1.62, 5.72, 607.0, 1.14e-05, 1.28, 0.001, 10.0),
    ('LSST', 'DUCT', 0.000484, 1.63, 5.36, 26900.0, 0.0523, 1.13, 0.001, 10.0),
    ('LSST', 'EAFC', 1.38e-06, 0.9, 13.2, 13000.0, 0.000133, 1.59, 0.001, 10.0),
    ('LSST', 'FCSG', 4.99e-08, 0.361, 4.54, 1590000.0, 0.0876, 1.63, 0.001, 10.0),
    ('LSST', 'FWWG', 0.00051, 1.94, 11.8, 89.5, 0.00964, 1.79, 0.001, 10.0),
    ('LSST', 'HRET', 4.34e-07, 1.8, 4.16, 5620.0, 0.0729, 1.77, 0.001, 10.0),
    ('LSST', 'IHOZ', 9.81e-05, 1.05, 15.2, 915.0, 0.0758, 1.68, 0.001, 10.0),
    ('LSST', 'MTNS', 9.76e-06, 1.67, 8.67, 3180000.0, 0.00112, 1.09, 0.001, 10.0),
    ('LSST', 'NAYO', 1.65e-05, 0.288, 7.04, 7.3, 0.0729, 2.08, 0.001, 10.0),
    ('LSST', 'RTTH', 1.95e-08, 1.22, 15.4, 10500.0, 0.000192, 1.15, 0.001, 10.0),
    ('LSST', 'SMAH', 1.91e-06, 1.32, 15.4, 13000.0, 0.00234, 1.09, 0.001, 10.0),
    ('LSST', 'SRNA', 9.81e-05, 1.42, 13.7, 2320000.0, 0.0758, 1.68, 0.001, 10.0),
    ('MDGI', 'ACAC', 5.84e-07, 1.3, 7.81, 96900.0, 0.000484, 1.13, 0.001, 10.0),
    ('MDGI', 'ACIX', 3.96e-08, 1.63, 16.6, 2220000.0, 0.0922, 1.2, 0.001, 10.0),
    ('MDGI', 'BENI', 0.000545, 0.756, 13.2, 1590000.0, 0.0372, 1.09, 0.001, 10.0),
    ('MDGI', 'FWWG', 0.000233, 1.6, 13.4, 19.5, 0.0146, 1.25, 0.001, 10.0),
    ('MDGI', 'GARI', 0.000595, 0.177, 2.66, 3.78, 1.4e-05, 1.52, 0.001, 10.0),
    ('MDGI', 'HRND', 3.81e-08, 0.724, 11.2, 4810000.0, 0.000844, 1.37, 0.001, 10.0),
    ('MDGI', 'ITPA', 2.71e-07, 0.668, 8.54, 1.25, 0.00382, 1.35, 0.001, 10.0),
    ('MDGI', 'MTNS', 1.91e-06, 0.711, 11.7, 1300.0, 0.0758, 1.68, 0.001, 10.0),
    ('MDGI', 'SRTX', 1.11e-08, 1.22, 13.9, 4810000.0, 0.0997, 1.5, 0.001, 10.0),
    ('MMBT', 'FCSG', 0.000219, 1.93, 3.76, 14400.0, 0.000133, 1.77, 0.001, 10.0),
    ('MMBT', 'GARI', 1.08e-05, 0.188, 13.4, 715.0, 1.4e-05, 2.09, 0.001, 10.0),
    ('MSDP', 'ANSO', 0.000318, 0.177, 6.99, 8230.0, 0.0573, 1.38, 0.001, 10.0),
    ('MSDP', 'EAFC', 4.11e-06, 1.6, 5.69, 7.3, 0.000192, 1.32, 0.001, 10.0),
    ('MSDP', 'FCSG', 1.95e-08, 0.674, 13.4, 7370000.0, 0.0195, 2.38, 0.001, 10.0),
    ('MSDP', 'HRET', 0.000119, 0.598, 3.47, 4810000.0, 0.000273, 1.68, 0.001, 10.0),
    ('MSDP', 'HRND', 2.35e-06, 1.32, 15.4, 13000.0, 0.00234, 1.09, 0.001, 10.0),
    ('MSDP', 'MTNS', 1.68e-05, 1.13, 5.14, 28.0, 1.4e-05, 1.09, 0.001, 10.0),
    ('MSDP', 'NGTE', 0.000318, 0.615, 13.4, 29.3, 0.00235, 2.05, 0.001, 10.0),
    ('MSDP', 'SRTX', 1.91e-06, 0.795, 13.2, 31.1, 0.0372, 1.17, 0.001, 10.0),
    ('MSDP', 'ULXY', 2.49e-05, 1.41, 2.66, 5.37, 2.36e-05, 1.06, 0.001, 10.0),
    ('NPCK', 'ACIX', 3.33e-07, 1.05, 18.4, 27200.0, 0.00526, 1.99, 0.001, 10.0),
    ('NPCK', 'ANSO', 1.24e-06, 1.41, 3.65, 121.0, 0.000634, 1.48, 0.001, 10.0),
    ('NPCK', 'BLBT', 1.32e-07, 0.744, 16.0, 8570.0, 5.13e-05, 1.6, 0.001, 10.0),
    ('NPCK', 'FARS', 1.85e-06, 1.07, 17.7, 29.3, 0.0372, 1.88, 0.001, 10.0),
    ('NPCK', 'ILVX', 2.07e-08, 0.756, 19.8, 669000.0, 0.00151, 2.26, 0.001, 10.0),
    ('NPCK', 'NGTE', 0.00051, 0.555, 7.85, 200000.0, 0.0354, 1.94, 0.001, 10.0),
    ('NPCK', 'NWIG', 1.44e-06, 0.115, 15.1, 1240.0, 0.0801, 1.28, 0.001, 10.0),
    ('NPCK', 'RCRI', 5.54e-06, 1.8, 14.5, 1.27, 2.2e-05, 1.79, 0.001, 10.0),
    ('NWIG', 'ACAC', 0.000141, 0.973, 11.6, 622.0, 0.00801, 1.09, 0.001, 10.0),
    ('NWIG', 'ACIX', 2.66e-07, 1.73, 5.14, 200000.0, 0.0354, 1.95, 0.001, 10.0),
    ('NWIG', 'AETS', 3.09e-08, 1.0, 4.51, 4.96, 0.0195, 1.72, 0.001, 10.0),
    ('NWIG', 'BENI', 0.000595, 0.889, 13.4, 12.9, 0.000322, 2.0, 0.001, 10.0),
    ('NWIG', 'CCNS', 1.91e-06, 1.83, 3.32, 13000.0, 0.000844, 1.31, 0.001, 10.0),
    ('NWIG', 'EAFC', 1.65e-05, 0.52, 17.8, 1240.0, 0.0558, 1.67, 0.001, 10.0),
    ('NWIG', 'FCSG', 0.000318, 0.188, 9.66, 29.3, 0.00151, 1.56, 0.001, 10.0),
    ('NWIG', 'HTRK', 1.85e-06, 1.07, 16.3, 98.7, 0.00541, 1.15, 0.001, 10.0),
    ('NWIG', 'MMBT', 3.33e-07, 0.734, 14.4, 5920000.0, 0.0729, 1.36, 0.001, 10.0),
    ('NWIG', 'NGTE', 1.95e-08, 1.22, 11.7, 462000.0, 0.00637, 1.15, 0.001, 10.0),
    ('NWIG', 'ULXY', 2.66e-07, 0.157, 19.2, 10500.0, 0.0372, 1.67, 0.001, 10.0),
    ('OTCS', 'ACAC', 1.72e-06, 0.405, 19.2, 200000.0, 0.00253, 2.26, 0.001, 10.0),
    ('OTCS', 'CCNS', 1.29e-05, 0.226, 14.7, 4340.0, 0.0573, 2.26, 0.001, 10.0),
    ('OTCS', 'CTGI', 1.34e-08, 0.226, 11.6, 4810000.0, 0.0628, 1.37, 0.001, 10.0),
    ('OTCS', 'DIHO', 2.35e-06, 0.838, 5.33, 88900.0, 4.97e-05, 1.2, 0.001, 10.0),
    ('OTCS', 'EAFC', 1.19e-05, 0.226, 16.1, 13000.0, 0.0528, 1.35, 0.001, 10.0),
    ('OTCS', 'EELT', 2.52e-05, 1.25, 6.46, 915.0, 0.0123, 1.09, 0.001, 10.0),
    ('OTCS', 'HRND', 9.04e-06, 1.22, 16.0, 10500.0, 2.26e-05, 2.26, 0.001, 10.0),
    ('OTCS', 'HTRK', 6.49e-07, 0.725, 19.3, 669000.0, 0.0021, 1.69, 0.001, 10.0),
    ('OTCS', 'HUXZ', 1.87e-06, 0.188, 19.7, 1060000.0, 0.00801, 1.75, 0.001, 10.0),
    ('OTCS', 'MMBT', 5.07e-08, 1.3, 6.46, 78500000.0, 0.0123, 1.63, 0.001, 10.0),
    ('OTCS', 'MSDP', 1.95e-08, 0.555, 13.9, 61600.0, 0.000242, 2.17, 0.001, 10.0),
    ('OTCS', 'SPLZ', 0.000176, 0.255, 12.4, 29.3, 0.00801, 1.92, 0.001, 10.0),
    ('RRES', 'ALUT', 5.41e-06, 1.67, 6.88, 715.0, 0.0573, 2.09, 0.001, 10.0),
    ('RRES', 'CTGI', 1.07e-08, 1.37, 9.66, 1.42, 2.67e-05, 1.53, 0.001, 10.0),
    ('RRES', 'EELT', 4.32e-08, 1.3, 15.0, 1.57, 6.78e-05, 1.49, 0.001, 10.0),
    ('RRES', 'FARS', 1.23e-05, 1.24, 16.5, 2000.0, 0.000308, 1.72, 0.001, 10.0),
    ('RRES', 'IHOZ', 0.000273, 1.18, 7.7, 3410.0, 0.0555, 2.09, 0.001, 10.0),
    ('RRES', 'NAYO', 5.34e-06, 1.32, 4.09, 31.1, 1.76e-05, 1.09, 0.001, 10.0),
    ('RTTH', 'ACAC', 6.63e-08, 1.62, 6.98, 426.0, 0.00112, 1.07, 0.001, 10.0),
    ('RTTH', 'EAFC', 3.85e-08, 1.13, 16.7, 13100.0, 0.000225, 1.73, 0.001, 10.0),
    ('RTTH', 'FARS', 0.000545, 1.88, 12.9, 13.4, 0.000308, 1.21, 0.001, 10.0),
    ('RTTH', 'FCSG', 0.000111, 0.436, 13.8, 24.3, 0.00183, 1.28, 0.001, 10.0),
    ('RTTH', 'FWWG', 8.26e-06, 1.6, 11.7, 31.1, 0.00457, 1.09, 0.001, 10.0),
    ('RTTH', 'MDGI', 0.000589, 0.711, 11.9, 19400000.0, 0.0729, 1.07, 0.001, 10.0),
    ('RTTH', 'NAYO', 1.09e-05, 0.333, 11.7, 5.83, 2.95e-05, 1.06, 0.001, 10.0),
    ('RTTH', 'RCRI', 3.09e-08, 1.0, 10.8, 10500.0, 0.0555, 1.47, 0.001, 10.0),
    ('RTTH', 'RRES', 0.000504, 0.13, 4.09, 86.1, 4.22e-05, 1.02, 0.001, 10.0),
    ('RTTH', 'SRTX', 1.95e-08, 1.22, 14.5, 426.0, 0.000147, 1.29, 0.001, 10.0),
    ('RTTH', 'ULXY', 0.000219, 0.615, 16.0, 2.33, 0.0523, 1.36, 0.001, 10.0),
    ('SMAH', 'ACAC', 1.91e-06, 0.226, 16.1, 13000.0, 0.0573, 1.09, 0.001, 10.0),
    ('SMAH', 'AETS', 1.39e-07, 1.47, 3.39, 669000.0, 0.0354, 2.26, 0.001, 10.0),
    ('SMAH', 'ALUT', 1.44e-06, 0.185, 12.9, 29.3, 0.0229, 2.05, 0.001, 10.0),
    ('SMAH', 'BLBT', 4.74e-05, 0.333, 6.46, 8230.0, 0.0876, 1.48, 0.001, 10.0),
    ('SMAH', 'FARS', 4.74e-05, 1.46, 13.1, 8730.0, 0.000535, 1.36, 0.001, 10.0),
    ('SMAH', 'FWWG', 1.91e-06, 1.22, 5.97, 13000.0, 0.00876, 1.09, 0.001, 10.0),
    ('SMAH', 'IHOZ', 6.28e-06, 0.439, 4.16, 603.0, 0.000404, 1.09, 0.001, 10.0),
    ('SMAH', 'ILVX', 3.33e-07, 1.73, 3.76, 37.0, 1.15e-05, 1.02, 0.001, 10.0),
    ('SMAH', 'NWIG', 1.91e-06, 0.333, 18.3, 8230.0, 0.0354, 1.02, 0.001, 10.0),
    ('SMAH', 'SPLZ', 0.000504, 0.435, 18.7, 7.88, 0.00169, 1.02, 0.001, 10.0),
    ('SMAH', 'SRTX', 0.000236, 0.248, 15.3, 37.0, 1.11e-05, 1.65, 0.001, 10.0),
    ('SMAH', 'ULXY', 0.000219, 1.08, 16.0, 915.0, 4.43e-05, 1.87, 0.001, 10.0),
    ('SPLZ', 'ACAC', 1.81e-06, 0.405, 5.07, 17500000.0, 0.0729, 1.64, 0.001, 10.0),
    ('SPLZ', 'AETS', 5.54e-06, 1.91, 12.7, 5730000.0, 0.00112, 1.78, 0.001, 10.0),
    ('SPLZ', 'BENI', 0.000273, 1.98, 4.09, 96900.0, 0.0573, 1.53, 0.001, 10.0),
    ('SPLZ', 'BLBT', 0.000595, 0.725, 16.0, 116000.0, 3.08e-05, 1.31, 0.001, 10.0),
    ('SPLZ', 'CCNS', 4.35e-06, 0.296, 14.5, 1.25, 2.36e-05, 1.15, 0.001, 10.0),
    ('SPLZ', 'CTGI', 0.000595, 0.502, 4.09, 1340.0, 0.0595, 1.24, 0.001, 10.0),
    ('SPLZ', 'EAFC', 1.48e-06, 1.6, 4.16, 86.1, 4.78e-05, 1.09, 0.001, 10.0),
    ('SPLZ', 'FARS', 0.000123, 0.405, 7.69, 426.0, 0.000189, 1.01, 0.001, 10.0),
    ('SPLZ', 'FCSG', 0.000145, 0.388, 3.39, 31.6, 3.04e-05, 1.35, 0.001, 10.0),
    ('SPLZ', 'GARI', 0.000504, 1.32, 17.5, 1.94, 0.00235, 2.05, 0.001, 10.0),
    ('SPLZ', 'HTRK', 0.00051, 0.195, 19.2, 2060000.0, 0.0884, 1.4, 0.001, 10.0),
    ('SPLZ', 'IHOZ', 0.000111, 0.0636, 18.6, 12200.0, 0.000467, 1.44, 0.001, 10.0),
    ('SPLZ', 'MMBT', 8.94e-06, 1.79, 4.16, 3320.0, 0.0123, 2.28, 0.001, 10.0),
    ('SPLZ', 'MTNS', 0.000109, 0.613, 18.6, 7.59, 0.000523, 1.85, 0.001, 10.0),
    ('SPLZ', 'NGTE', 8.03e-07, 0.127, 18.4, 75800000.0, 0.0253, 1.09, 0.001, 10.0),
    ('SPLZ', 'ULXY', 0.000545, 0.615, 3.39, 5.55, 0.0523, 1.09, 0.001, 10.0),
    ('SRNA', 'ACAC', 9.81e-05, 1.42, 13.7, 2320000.0, 0.0363, 1.68, 0.001, 10.0),
    ('SRNA', 'ALUT', 1.04e-06, 1.46, 13.4, 1.25, 0.00637, 1.14, 0.001, 10.0),
    ('SRNA', 'AMRP', 4.32e-08, 1.38, 11.4, 1590000.0, 0.0121, 1.99, 0.001, 10.0),
    ('SRNA', 'BLBT', 9.76e-06, 1.24, 19.5, 1.36, 0.000286, 1.73, 0.001, 10.0),
    ('SRNA', 'CCNS', 2.54e-08, 0.31, 10.7, 89.5, 0.078, 1.22, 0.001, 10.0),
    ('SRNA', 'FCSG', 1.6e-08, 1.41, 11.4, 10500.0, 1.4e-05, 1.45, 0.001, 10.0),
    ('SRNA', 'GARI', 5.07e-08, 1.22, 18.6, 78500000.0, 0.0123, 1.06, 0.001, 10.0),
    ('SRNA', 'HRET', 0.000102, 1.38, 3.48, 183000.0, 0.0995, 2.05, 0.001, 10.0),
    ('SRNA', 'HUXZ', 3.8e-05, 1.74, 2.44, 345000.0, 0.0491, 1.12, 0.001, 10.0),
    ('SRNA', 'NAYO', 1.34e-08, 1.55, 13.4, 1.99, 0.000286, 1.09, 0.001, 10.0),
    ('SRNA', 'NGTE', 3.33e-07, 1.18, 5.72, 78300.0, 0.00964, 1.22, 0.001, 10.0),
    ('SRNA', 'NPCK', 5.07e-08, 1.38, 6.92, 1010000.0, 0.00234, 1.92, 0.001, 10.0),
    ('SRNA', 'NWIG', 2.66e-07, 0.157, 18.2, 2000.0, 1.1e-05, 1.31, 0.001, 10.0),
    ('SRNA', 'RTTH', 8.52e-07, 1.41, 5.36, 222.0, 0.0132, 1.41, 0.001, 10.0),
    ('SRNA', 'SMAH', 1.95e-08, 1.1, 16.4, 679000.0, 0.000223, 1.13, 0.001, 10.0),
    ('SRNA', 'ULXY', 0.000348, 0.333, 3.76, 8230.0, 0.00116, 1.48, 0.001, 10.0),
    ('SRTX', 'ACIX', 1.7e-06, 1.98, 8.02, 5.83, 1.27e-05, 1.54, 0.001, 10.0),
    ('SRTX', 'AETS', 2.11e-08, 1.55, 17.2, 43600.0, 0.0729, 1.13, 0.001, 10.0),
    ('SRTX', 'BENI', 9.76e-06, 1.93, 3.76, 1300.0, 0.000638, 1.67, 0.001, 10.0),
    ('SRTX', 'BLBT', 6.72e-05, 0.0856, 9.66, 8.92, 0.0122, 1.09, 0.001, 10.0),
    ('SRTX', 'FCSG', 0.000109, 0.948, 2.66, 98.7, 0.000844, 1.55, 0.001, 10.0),
    ('SRTX', 'FWWG', 1.07e-08, 1.43, 14.9, 679000.0, 0.00541, 1.37, 0.001, 10.0),
    ('SRTX', 'IHOZ', 1.38e-06, 0.792, 16.7, 4.24, 0.00978, 1.15, 0.001, 10.0),
    ('SRTX', 'MTNS', 1.38e-05, 1.92, 17.7, 39000000.0, 0.00112, 1.02, 0.001, 10.0),
    ('SRTX', 'NGTE', 0.000319, 0.226, 9.59, 164.0, 0.0709, 1.37, 0.001, 10.0),
)
# END GENERATED AUTO PAIR PARAMETERS


def _build_configs() -> tuple[PairCfg, ...]:
    out: list[PairCfg] = []
    for idx, (
        y_symbol,
        x_symbol,
        q_beta,
        cusum_k,
        h,
        max_cov_inflate,
        cov_decay,
        z_entry,
        z_exit,
        z_stop,
    ) in enumerate(_TUNED_PAIR_PARAMS):
        out.append(
            PairCfg(
                idx=idx,
                y=I[y_symbol],
                x=I[x_symbol],
                filt=FiltCfg(
                    q_beta=q_beta,
                    cusum_k=cusum_k,
                    h=h,
                    max_cov_inflate=max_cov_inflate,
                    cov_decay=cov_decay,
                ),
                trade=TradeCfg(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    z_stop=z_stop,
                ),
            )
        )
    return tuple(out)


PAIR_CFGS = _build_configs()

class PairAgent:
    __slots__ = ("spec", "filt", "current_z", "current_innov", "side", "nt_seen", "spread_hist", "current_kappa", "current_sigma")

    def __init__(self, spec: PairCfg) -> None:
        self.spec = spec
        self.filt = PairSegFilter(spec.filt)
        self.current_z = 0.0
        self.current_innov = 0.0
        self.side = 0
        self.nt_seen = 0
        
        self.spread_hist = []
        self.current_kappa = 0.1
        self.current_sigma = 0.02

    def _update_dynamic_ou(self):
        if len(self.spread_hist) < 30: return
            
        s = np.array(self.spread_hist[-_OU_WINDOW:])
        s_lag = s[:-1]; s_t = s[1:]; n = len(s_lag)
        mean_lag = np.mean(s_lag); mean_t = np.mean(s_t)
        denom = np.sum((s_lag - mean_lag)**2)
        
        if denom > 1e-8:
            b_mle = np.sum((s_lag - mean_lag) * (s_t - mean_t)) / denom
            b_mle = min(max(b_mle, 1e-4), 0.9999)
            a_mle = mean_t - b_mle * mean_lag
            residuals = s_t - (a_mle + b_mle * s_lag)
            sigma_eps = np.sqrt(np.sum(residuals**2) / (n - 2))
            
            raw_kappa = -np.log(b_mle)
            raw_sigma = sigma_eps * np.sqrt((2 * raw_kappa) / (1 - np.exp(-2 * raw_kappa)))
            
            self.current_kappa = (0.9 * self.current_kappa) + (0.1 * raw_kappa)
            self.current_sigma = (0.9 * self.current_sigma) + (0.1 * raw_sigma)

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self.__init__(self.spec)
            
        y_i, x_i = self.spec.y, self.spec.x
        for t in range(self.nt_seen, nt):
            raw_y = float(np.nan_to_num(prices[y_i, t], nan=1e-12))
            raw_x = float(np.nan_to_num(prices[x_i, t], nan=1e-12))
            
            y = np.log(max(raw_y, _EPS_DENOM))
            x = np.log(max(raw_x, _EPS_DENOM))
            
            o = self.filt.step(y, x)
            
            self.current_z = float(np.nan_to_num(o.z, nan=0.0))
            self.current_innov = float(np.nan_to_num(o.innov, nan=0.0))
            self.spread_hist.append(float(np.nan_to_num(o.spread, nan=0.0)))
            self._update_dynamic_ou()

            tc = self.spec.trade
            z = self.current_z
            az = abs(z)
            
            if self.side == 0:
                if tc.z_entry <= az < tc.z_stop:
                    self.side = int(-np.sign(z)) if z != 0 else 0
            else:
                exit_trade = False
                if self.side == 1 and z >= -tc.z_exit:
                    exit_trade = True
                elif self.side == -1 and z <= tc.z_exit:
                    exit_trade = True
                
                if exit_trade or az >= tc.z_stop:
                    self.side = 0
                    
        self.nt_seen = nt

    def dollar_positions(self) -> tuple[float, float, float]:
        if self.side == 0:
            return 0.0, 0.0, abs(self.current_z)

        # side = +1  →  long the spread  → long y, short x
        # side = -1  →  short the spread → short y, long x
        beta = float(self.filt.beta)
        
        # Scale the pair size so neither leg exceeds LEG (10,000) 
        # This prevents position limits from clipping and breaking the hedge!
        scale = LEG / max(1.0, abs(beta))
        
        pos_y = scale * float(self.side)
        pos_x = -beta * pos_y          # NEGATIVE beta — the actual hedge
        return pos_y, pos_x, abs(self.current_z)

class AtlasBook:
    def __init__(self) -> None:
        self.agents = [PairAgent(s) for s in PAIR_CFGS]

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        if nt < WARMUP + 10: return pos

        for ag in self.agents:
            ag.catch_up(prices)

        cands = []
        for ag in self.agents:
            py, px, az = ag.dollar_positions()
            if abs(py) < 1e-9: continue
            # OU kappa is used here to prioritize which pairs trade
            priority_score = ag.current_kappa * az
            cands.append((priority_score, ag, py, px))
            
        cands.sort(key=lambda c: c[0], reverse=True)
        
        taken = set()
        for priority_score, ag, py, px in cands:
            if ag.spec.y in taken or ag.spec.x in taken: continue
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
        
    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()

    dollars = _STATE["atlas"].dollars(prcSoFar)
    
    safe_prices = np.maximum(prcSoFar[:, -1], _EPS_DENOM)
    last_return = np.diff(np.log(np.maximum(prcSoFar[:, -2:], _EPS_DENOM)), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    
    if len(crash) > 0: crash[0] = False 
    dollars[crash] = 0.0
    
    if len(dollars) > 1:
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    
    raw_shares = dollars / safe_prices
    
    # Safely round to integer shares without overshooting tiny values to 1 share
    final_shares = np.round(raw_shares).astype(int)
    
    return np.nan_to_num(final_shares, nan=0).astype(int)
