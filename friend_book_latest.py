"""Supplied-profile pair book with bang-bang pair/ridge voting.

Every active pair contributes directional votes to both legs. Shared-symbol
conflicts are resolved by aggregate asset-level consensus instead of dropping
complete relationships. Protected pairs receive a fixed vote boost and ridge
adds a confidence-weighted vote. Every non-zero winning direction receives
the instrument's full dollar limit.
"""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple

import numpy as np

I = {
    "ALGO": 0,
    "AENO": 1,
    "LSST": 2,
    "SRNA": 3,
    "ELLT": 4,
    "AMRP": 5,
    "OTCS": 6,
    "HETT": 7,
    "HUXZ": 8,
    "DUCT": 9,
    "SMAH": 10,
    "NPCK": 11,
    "MSDP": 12,
    "EORC": 13,
    "CUBO": 14,
    "HRET": 15,
    "ANSO": 16,
    "DIHO": 17,
    "RTTH": 18,
    "SPLZ": 19,
    "NWIG": 20,
    "MMBT": 21,
    "MDGI": 22,
    "AGVF": 23,
    "RRES": 24,
    "CTGI": 25,
    "ALUT": 26,
    "ACAC": 27,
    "SRTX": 28,
    "GARI": 29,
    "RCRI": 30,
    "ACIX": 31,
    "CCNS": 32,
    "MTNS": 33,
    "IHOZ": 34,
    "NAYO": 35,
    "FWWG": 36,
    "EELT": 37,
    "HRND": 38,
    "AETS": 39,
    "ULXY": 40,
    "BLBT": 41,
    "BENI": 42,
    "ITPA": 43,
    "HTRK": 44,
    "NGTE": 45,
    "ILVX": 46,
    "FCSG": 47,
    "FARS": 48,
    "MHRM": 49,
    "EAFC": 50,
}

N_INST = 51
LEG = 10_000.0
WARMUP = 30
_INIT_WINDOW = 20
_OU_WINDOW = 100
_EPS_DENOM = 1.0e-12
_EPS_VAR = 1.0e-18

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
    """Sticky adaptive-beta filter with CUSUM covariance inflation."""

    def __init__(self, cfg: FiltCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else FiltCfg(q_beta=1.0e-6)
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

    def _update_kalman(
        self,
        yf: float,
        xf: float,
        dynamic_q: float,
    ) -> tuple[float, float, float]:
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
            self.innov_var = (
                lam * self.innov_var
                + (1.0 - lam) * innov * innov
            )

        z = (
            innov / sqrt(self.innov_var)
            if self.innov_var > 0.0
            else 0.0
        )

        if self.resid_var is None:
            self.resid_var = resid * resid
        else:
            self.resid_var = (
                lam * self.resid_var
                + (1.0 - lam) * resid * resid
            )
        return z

    def _evaluate_cusum(self, z: float, t: int) -> tuple[bool, bool]:
        cfg = self.cfg
        vol_shift = False
        cut = False

        prev_up, prev_dn = self.cm_up, self.cm_dn
        self.cm_up = max(0.0, self.cm_up + z - cfg.cusum_k)
        self.cm_dn = max(0.0, self.cm_dn - z - cfg.cusum_k)
        self.run_m_up = (
            t
            if self.cm_up > 0.0 and prev_up == 0.0
            else None
            if self.cm_up == 0.0
            else self.run_m_up
        )
        self.run_m_dn = (
            t
            if self.cm_dn > 0.0 and prev_dn == 0.0
            else None
            if self.cm_dn == 0.0
            else self.run_m_dn
        )

        d_v = (z * z - 1.0) / sqrt(2.0)
        prev_v_up, prev_v_dn = self.cv_up, self.cv_dn
        self.cv_up = max(0.0, self.cv_up + d_v - cfg.cusum_k)
        self.cv_dn = max(0.0, self.cv_dn - d_v - cfg.cusum_k)
        self.run_v_up = (
            t
            if self.cv_up > 0.0 and prev_v_up == 0.0
            else None
            if self.cv_up == 0.0
            else self.run_v_up
        )
        self.run_v_dn = (
            t
            if self.cv_dn > 0.0 and prev_v_dn == 0.0
            else None
            if self.cv_dn == 0.0
            else self.run_v_dn
        )

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

        if self.cm_up > cfg.h and self.cm_up > best:
            best, cut_bar = self.cm_up, self.run_m_up
        if self.cm_dn > cfg.h and self.cm_dn > best:
            best, cut_bar = self.cm_dn, self.run_m_dn
        if cfg.vol_cut and self.cv_up > cfg.h and self.cv_up > best:
            best, cut_bar = self.cv_up, self.run_v_up
        if cfg.vol_cut and self.cv_dn > cfg.h and self.cv_dn > best:
            best, cut_bar = self.cv_dn, self.run_v_dn

        cut_bar = max(
            int(cut_bar if cut_bar is not None else t),
            (self.seg_start or 0) + 1,
        )

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

        cut = False
        vol_shift = False
        if t >= WARMUP:
            cut, vol_shift = self._evaluate_cusum(z, t)

        if cut:
            target_multiplier = cfg.max_cov_inflate
        else:
            max_cusum = max(
                self.cm_up,
                self.cm_dn,
                self.cv_up,
                self.cv_dn,
            )
            excess = max(0.0, max_cusum - cfg.h)
            if excess > 0.0:
                target_multiplier = 1.0 + (
                    cfg.max_cov_inflate - 1.0
                ) * (1.0 - np.exp(-excess))
            else:
                target_multiplier = 1.0

        self.decay_multiplier = max(
            target_multiplier,
            self.decay_multiplier * (1.0 - cfg.cov_decay),
        )
        dynamic_q = cfg.q_beta * self.decay_multiplier
        adapting = self.decay_multiplier > 1.01

        innov, beta_new, _ = self._update_kalman(
            yf,
            xf,
            dynamic_q,
        )
        resid = yf - beta_new * xf

        return StepOut(
            beta=float(beta_new),
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

class PairAgent:
    """Stateful execution agent for one adaptive-beta pair."""

    __slots__ = (
        "spec",
        "filt",
        "current_z",
        "current_innov",
        "side",
        "nt_seen",
        "spread_hist",
        "current_kappa",
        "current_sigma",
        "current_cut",
        "current_adapting",
        "current_vol_shift",
    )

    def __init__(self, spec: PairCfg) -> None:
        self.spec = spec
        self._reset_state()

    def _reset_state(self) -> None:
        self.filt = PairSegFilter(self.spec.filt)
        self.current_z = 0.0
        self.current_innov = 0.0
        self.side = 0
        self.nt_seen = 0
        self.spread_hist = []
        self.current_kappa = 0.1
        self.current_sigma = 0.02
        self.current_cut = False
        self.current_adapting = False
        self.current_vol_shift = False

    def _update_dynamic_ou(self) -> None:
        if len(self.spread_hist) < 30:
            return

        s = np.array(self.spread_hist[-_OU_WINDOW:])
        s_lag = s[:-1]
        s_t = s[1:]
        n = len(s_lag)
        mean_lag = np.mean(s_lag)
        mean_t = np.mean(s_t)
        denom = np.sum((s_lag - mean_lag) ** 2)

        if denom > 1.0e-8:
            b_mle = (
                np.sum(
                    (s_lag - mean_lag)
                    * (s_t - mean_t)
                )
                / denom
            )
            b_mle = min(max(b_mle, 1.0e-4), 0.9999)
            a_mle = mean_t - b_mle * mean_lag
            residuals = s_t - (a_mle + b_mle * s_lag)
            sigma_eps = np.sqrt(
                np.sum(residuals**2) / (n - 2)
            )

            raw_kappa = -np.log(b_mle)
            raw_sigma = sigma_eps * np.sqrt(
                (2.0 * raw_kappa)
                / (1.0 - np.exp(-2.0 * raw_kappa))
            )

            self.current_kappa = (
                0.9 * self.current_kappa
                + 0.1 * raw_kappa
            )
            self.current_sigma = (
                0.9 * self.current_sigma
                + 0.1 * raw_sigma
            )

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self._reset_state()

        y_i, x_i = self.spec.y, self.spec.x
        for t in range(self.nt_seen, nt):
            raw_y = float(
                np.nan_to_num(prices[y_i, t], nan=_EPS_DENOM)
            )
            raw_x = float(
                np.nan_to_num(prices[x_i, t], nan=_EPS_DENOM)
            )

            y = np.log(max(raw_y, _EPS_DENOM))
            x = np.log(max(raw_x, _EPS_DENOM))
            output = self.filt.step(y, x)

            self.current_z = float(
                np.nan_to_num(output.z, nan=0.0)
            )
            self.current_innov = float(
                np.nan_to_num(output.innov, nan=0.0)
            )
            self.current_cut = bool(output.cut)
            self.current_adapting = bool(output.adapting)
            self.current_vol_shift = bool(output.vol_shift)
            self.spread_hist.append(
                float(np.nan_to_num(output.spread, nan=0.0))
            )
            self._update_dynamic_ou()

            trade = self.spec.trade
            z = self.current_z
            abs_z = abs(z)

            if self.side == 0:
                if trade.z_entry <= abs_z < trade.z_stop:
                    self.side = (
                        int(-np.sign(z))
                        if z != 0.0
                        else 0
                    )
            else:
                exit_trade = False
                if self.side == 1 and z >= -trade.z_exit:
                    exit_trade = True
                elif self.side == -1 and z <= trade.z_exit:
                    exit_trade = True

                if exit_trade or abs_z >= trade.z_stop:
                    self.side = 0

        self.nt_seen = nt

    def dollar_positions(self) -> tuple[float, float, float]:
        if self.side == 0:
            return 0.0, 0.0, abs(self.current_z)

        beta = float(self.filt.beta)
        scale = LEG / max(1.0, abs(beta))
        pos_y = scale * float(self.side)
        pos_x = -beta * pos_y
        return pos_y, pos_x, abs(self.current_z)

# The original union contained the complete requested 39-pair accepted set and
# all 57 flagged-review relationships. A prior version reduced that universe
# to the 91 relationships that were current when this review began. This
# version keeps that scope, then removes only relationships that are
# Pareto-harmful across train, validation, both train halves, and both
# validation halves. The supplied 2026-07-26 profiles below replace 36
# retained profiles and reintroduce three relationships that had previously
# been removed. The first seven retained rows are protected relationships. Of
# the original 57
# flagged-review relationships, 48 retain their exact profiles from the
# days-1-750 train-only catalog. Seven rows that previously used a pooled
# fallback now use their completed pair-specific auto profiles from
# tuning_data.db. Stored performance metrics are not used to select or alter
# the requested set.
_SELECTED_PAIR_PARAMS = (
    ("AENO", "NWIG", 1.44e-06, 0.711, 18.3, 5.37, 0.000189, 1.02, 0.001, 10.0),
    ("HETT", "ULXY", 1.87e-06, 1.94, 5.33, 1590000.0, 0.000108, 1.24, 0.001, 10.0),
    ("HUXZ", "ACAC", 8.52e-07, 1.25, 12.7, 4040000.0, 0.000108, 1.46, 0.001, 10.0),
    ("SMAH", "ILVX", 3.33e-07, 1.73, 3.76, 37.0, 1.15e-05, 1.02, 0.001, 10.0),
    ("EORC", "NGTE", 1.63e-07, 0.151, 17.8, 426.0, 0.00678, 1.02, 0.001, 10.0),
    ("RTTH", "NAYO", 1.09e-05, 0.333, 11.7, 5.83, 2.95e-05, 1.06, 0.001, 10.0),
    ("CTGI", "EELT", 1.91e-06, 1.08, 12.0, 3.89, 2.95e-05, 1.17, 0.001, 10.0),

    # Remaining 32 relationships in the requested 39-pair accepted set.
    ("ELLT", "FCSG", 0.000595, 1.76, 8.31, 52700.0, 0.000278, 1.09, 0.001, 10.0),
    ("HUXZ", "NWIG", 9.47e-08, 0.333, 11.4, 53.3, 0.0876, 1.65, 0.001, 10.0),
    ("DUCT", "ACAC", 1.91e-06, 0.247, 11.7, 13000.0, 0.0729, 1.15, 0.001, 10.0),
    ("ANSO", "CTGI", 2.94e-08, 0.0691, 7.24, 29.3, 0.00637, 1.79, 0.001, 10.0),
    ("ANSO", "BENI", 4.54e-06, 0.195, 6.99, 203000.0, 0.000634, 1.38, 0.001, 10.0),
    ("FWWG", "BLBT", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("AETS", "ILVX", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("AETS", "FCSG", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("ULXY", "MHRM", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("ULXY", "EAFC", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("HRET", "RTTH", 0.000545, 1.3, 7.17, 10.7, 0.000151, 1.65, 0.001, 10.0),
    ("ALUT", "CCNS", 1.64e-07, 0.641, 15.4, 5.37, 0.00112, 1.2, 0.001, 10.0),
    ("SMAH", "SPLZ", 0.000504, 0.435, 18.7, 7.88, 0.00169, 1.02, 0.001, 10.0),
    ("ACIX", "ITPA", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("MTNS", "BENI", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("SRNA", "ELLT", 5.54e-06, 1.02, 18.6, 144.0, 0.0555, 1.16, 0.001, 10.0),
    ("MTNS", "BLBT", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),
    ("ALUT", "HTRK", 8.94e-06, 1.32, 15.0, 12.9, 0.00678, 1.47, 0.001, 10.0),
    ("SRNA", "NWIG", 2.66e-07, 0.157, 18.2, 2000.0, 1.1e-05, 1.31, 0.001, 10.0),
    ("ACAC", "NAYO", 0.000318, 1.75, 11.6, 7.2, 0.000389, 1.15, 0.001, 10.0),
    ("MSDP", "ULXY", 2.49e-05, 1.41, 2.66, 5.37, 2.36e-05, 1.06, 0.001, 10.0),
    ("RTTH", "ULXY", 0.000219, 0.615, 16.0, 2.33, 0.0523, 1.36, 0.001, 10.0),
    ("AENO", "HUXZ", 1.09e-05, 0.711, 13.4, 4.71, 0.0758, 1.6, 0.001, 10.0),
    ("SRNA", "FWWG", 1.95e-08, 1.22, 13.4, 426.0, 4.43e-05, 1.17, 0.001, 10.0),
    ("EORC", "ILVX", 2.49e-05, 1.68, 12.0, 24.3, 1.11e-05, 2.32, 0.001, 10.0),
    ("ANSO", "AGVF", 0.000504, 0.317, 11.0, 4340.0, 4.56e-05, 2.34, 0.001, 10.0),
    ("ANSO", "AETS", 1.47e-05, 1.22, 12.0, 10400000.0, 0.0354, 1.78, 0.001, 10.0),
    ("ANSO", "FCSG", 9.23e-06, 1.25, 4.2, 915.0, 0.000151, 1.5, 0.001, 10.0),
    ("MHRM", "EAFC", 4.54e-06, 0.711, 11.7, 53.3, 0.000189, 1.24, 0.001, 10.0),

    # 57-pair flagged review set.
    ("AGVF", "RCRI", 5.41e-06, 0.223, 4.11, 1.94, 0.00112, 1.65, 0.001, 10.0),
    ("MSDP", "SRTX", 1.91e-06, 0.795, 13.2, 31.1, 0.0372, 1.17, 0.001, 10.0),
    ("RTTH", "RRES", 0.000504, 0.13, 4.09, 86.1, 4.22e-05, 1.02, 0.001, 10.0),
    ("HRET", "HRND", 1.09e-05, 0.613, 5.07, 4810000.0, 0.0729, 1.2, 0.001, 10.0),
    ("HRET", "SRTX", 1.91e-06, 1.6, 3.76, 3320.0, 0.0195, 1.72, 0.001, 10.0),
    ("ANSO", "RCRI", 9.47e-08, 1.83, 8.02, 53.3, 1.27e-05, 1.09, 0.001, 10.0),
    ("ALGO", "HRET", 6.86e-06, 0.439, 13.4, 13000.0, 0.000535, 1.69, 0.001, 10.0),
    ("HRET", "NGTE", 2.66e-07, 0.226, 19.5, 10500.0, 0.00681, 1.02, 0.001, 10.0),
    ("CUBO", "ULXY", 1.26e-05, 1.85, 19.9, 31.1, 1.61e-05, 1.99, 0.001, 10.0),
    ("ANSO", "RRES", 1.44e-06, 1.33, 14.4, 1.57, 0.00678, 1.47, 0.001, 10.0),
    ("MTNS", "MHRM", 1.2e-07, 0.674, 14.5, 90500.0, 0.0504, 1.09, 0.001, 10.0),
    ("ANSO", "RTTH", 2.66e-07, 1.24, 5.33, 8230.0, 1.61e-05, 1.48, 0.001, 10.0),
    ("SMAH", "AETS", 1.39e-07, 1.47, 3.39, 669000.0, 0.0354, 2.26, 0.001, 10.0),
    ("HRET", "MTNS", 2.94e-08, 1.18, 13.7, 78300.0, 0.00978, 2.01, 0.001, 10.0),
    ("HUXZ", "DUCT", 1.47e-05, 0.185, 5.33, 96900.0, 0.00157, 1.13, 0.001, 10.0),
    ("HUXZ", "RRES", 4.34e-07, 0.824, 15.0, 11.8, 1.56e-05, 2.19, 0.001, 10.0),
    ("ANSO", "IHOZ", 2.37e-07, 0.435, 3.53, 9250000.0, 0.0491, 2.04, 0.001, 10.0),
    ("DUCT", "NPCK", 5.41e-06, 1.94, 5.33, 2680000.0, 0.0573, 1.65, 0.001, 10.0),
    ("LSST", "EAFC", 1.38e-06, 0.9, 13.2, 13000.0, 0.000133, 1.59, 0.001, 10.0),
    ("ELLT", "AETS", 4.98e-07, 0.52, 6.43, 29.3, 0.000404, 1.06, 0.001, 10.0),
    ("FWWG", "NGTE", 1.63e-07, 1.6, 13.9, 10500.0, 3.36e-05, 1.29, 0.001, 10.0),
    ("CUBO", "RTTH", 8.75e-05, 1.4, 16.4, 9280000.0, 0.00457, 1.14, 0.001, 10.0),
    ("HRET", "MDGI", 1.07e-08, 0.457, 18.6, 200000.0, 0.00116, 1.44, 0.001, 10.0),
    ("SMAH", "FWWG", 1.91e-06, 1.22, 5.97, 13000.0, 0.00876, 1.09, 0.001, 10.0),
    ("SMAH", "ULXY", 0.000219, 1.08, 16.0, 915.0, 4.43e-05, 1.87, 0.001, 10.0),
    ("RTTH", "FWWG", 8.26e-06, 1.6, 11.7, 31.1, 0.00457, 1.09, 0.001, 10.0),
    ("OTCS", "EELT", 2.52e-05, 1.25, 6.46, 915.0, 0.0123, 1.09, 0.001, 10.0),
    ("HRET", "SPLZ", 0.00045, 0.333, 12.1, 98.7, 0.000296, 1.09, 0.001, 10.0),
    ("NGTE", "FCSG", 1.31e-08, 1.3, 15.6, 4.71, 0.000308, 1.68, 0.001, 10.0),
    ("EORC", "FCSG", 2.33e-07, 1.36, 9.35, 94800000.0, 0.0195, 2.14, 0.001, 10.0),
    ("MTNS", "ULXY", 3.36e-06, 1.62, 16.1, 4090000.0, 0.00801, 1.09, 0.001, 10.0),
    ("AENO", "IHOZ", 1.29e-05, 0.744, 3.32, 5.37, 0.0114, 1.31, 0.001, 10.0),
    ("AGVF", "ULXY", 9.47e-05, 0.0954, 5.63, 47300.0, 0.000147, 1.59, 0.001, 10.0),
    ("SPLZ", "ULXY", 0.000545, 0.615, 3.39, 5.55, 0.0523, 1.09, 0.001, 10.0),
    ("SPLZ", "CCNS", 4.35e-06, 0.296, 14.5, 1.25, 2.36e-05, 1.15, 0.001, 10.0),
    ("MSDP", "EAFC", 4.11e-06, 1.6, 5.69, 7.3, 0.000192, 1.32, 0.001, 10.0),
    ("DUCT", "HRND", 0.000704, 0.189, 3.5, 426.0, 0.0253, 1.13, 0.001, 10.0),
    ("LSST", "RTTH", 1.95e-08, 1.22, 15.4, 10500.0, 0.000192, 1.15, 0.001, 10.0),
    ("DIHO", "FCSG", 2.2e-07, 1.32, 7.81, 10500.0, 0.0523, 1.47, 0.001, 10.0),
    ("RTTH", "RCRI", 3.09e-08, 1.0, 10.8, 10500.0, 0.0555, 1.47, 0.001, 10.0),
    ("OTCS", "CTGI", 1.34e-08, 0.226, 11.6, 4810000.0, 0.0628, 1.37, 0.001, 10.0),
    ("GARI", "FWWG", 0.000435, 0.654, 5.07, 7.2, 0.00114, 1.2, 0.001, 10.0),
    ("SPLZ", "FCSG", 0.000145, 0.388, 3.39, 31.6, 3.04e-05, 1.35, 0.001, 10.0),
    ("AMRP", "DUCT", 2.05e-06, 0.177, 9.41, 13000.0, 0.00402, 1.09, 0.001, 10.0),
    ("SPLZ", "NGTE", 8.03e-07, 0.127, 18.4, 75800000.0, 0.0253, 1.09, 0.001, 10.0),
    ("RTTH", "SPLZ", 1.04e-06, 1.18, 14.5, 11.8, 0.00345, 1.06, 0.001, 10.0),
    ("CCNS", "NAYO", 9.04e-06, 1.32, 19.4, 62.3, 0.0628, 1.48, 0.001, 10.0),
    ("SPLZ", "ALUT", 3.73e-07, 1.54, 13.3, 462000.0, 0.078, 1.79, 0.001, 10.0),
    ("ULXY", "FCSG", 4.99e-08, 0.333, 6.31, 57400000.0, 0.0729, 1.79, 0.001, 10.0),
    ("NWIG", "MHRM", 1.98e-06, 1.51, 4.75, 2700.0, 0.0558, 1.56, 0.001, 10.0),
    ("SRTX", "MHRM", 0.000341, 0.228, 11.9, 3.47, 0.0398, 1.53, 0.001, 10.0),
    ("MDGI", "ULXY", 0.000219, 1.22, 12.0, 4810000.0, 1.34e-05, 1.59, 0.001, 10.0),
)

# Supplied auto-tuned profiles (2026-07-26). Existing relationships are
# replaced in-place; newly supplied relationships are appended below. This
# keeps the rest of the 88-pair catalog unchanged while trialling all 39
# supplied profiles.
_SUPPLIED_AUTO_PAIR_PARAMS = (
    ("AENO", "NWIG", 5.51e-06, 1.6, 13.3, 892000.0, 2.5e-05, 0.816, 0.001, 10.0),
    ("ELLT", "FCSG", 1.84e-06, 1.09, 7.37, 35000.0, 2.6e-05, 1.75, 0.001, 10.0),
    ("HETT", "ULXY", 6.93e-07, 1.5, 11.1, 457.0, 0.000478, 0.881, 0.001, 10.0),
    ("HUXZ", "NWIG", 3.07e-08, 1.46, 13.2, 159.0, 0.0492, 0.843, 0.001, 10.0),
    ("HUXZ", "ACAC", 1.54e-08, 0.405, 2.7, 119.0, 0.00534, 0.862, 0.001, 10.0),
    ("DUCT", "ACAC", 7.44e-07, 0.688, 12.1, 91600.0, 0.0264, 0.865, 0.001, 10.0),
    ("SMAH", "ILVX", 5.26e-08, 1.02, 9.78, 4850.0, 0.00211, 0.926, 0.001, 10.0),
    ("EORC", "NGTE", 1.51e-07, 0.545, 17.9, 1930.0, 0.000188, 0.801, 0.001, 10.0),
    ("ANSO", "CTGI", 7.15e-08, 1.05, 6.35, 295.0, 0.000193, 1.74, 0.001, 10.0),
    ("ANSO", "BENI", 9.91e-08, 1.2, 12.1, 1130.0, 0.000641, 2.26, 0.001, 10.0),
    ("RTTH", "NAYO", 6.32e-08, 0.873, 18.3, 5250.0, 0.0268, 0.925, 0.001, 10.0),
    ("CTGI", "EELT", 5.22e-08, 1.9, 18.4, 479000.0, 0.00259, 0.805, 0.001, 10.0),
    ("FWWG", "BLBT", 2.13e-08, 1.17, 11.0, 204.0, 0.0116, 0.955, 0.001, 10.0),
    ("AETS", "ILVX", 2.65e-06, 0.671, 10.2, 85600.0, 0.00197, 2.24, 0.001, 10.0),
    ("AETS", "FCSG", 2.45e-06, 0.166, 6.02, 31200.0, 0.000147, 1.01, 0.001, 10.0),
    ("ULXY", "MHRM", 5.55e-08, 1.01, 14.5, 22100000.0, 0.00893, 2.11, 0.001, 10.0),
    ("ULXY", "EAFC", 0.000123, 1.68, 12.2, 1430000.0, 0.00028, 2.14, 0.001, 10.0),
    ("HRET", "RTTH", 0.000248, 1.57, 5.81, 100.0, 0.000199, 1.01, 0.001, 10.0),
    ("ALUT", "CCNS", 8.07e-08, 1.64, 7.93, 310.0, 2.22e-05, 1.57, 0.001, 10.0),
    ("SMAH", "SPLZ", 3.42e-07, 0.308, 8.46, 21100.0, 0.0197, 0.969, 0.001, 10.0),
    ("ACIX", "ITPA", 0.000381, 1.51, 12.4, 6720.0, 0.000123, 0.967, 0.001, 10.0),
    ("MTNS", "BENI", 1.55e-06, 1.52, 18.4, 5170.0, 0.00332, 1.49, 0.001, 10.0),
    ("SRNA", "ELLT", 3.92e-05, 0.482, 11.6, 771000.0, 0.00959, 2.07, 0.001, 10.0),
    ("MTNS", "BLBT", 8.89e-05, 0.669, 5.48, 573000.0, 0.0158, 2.29, 0.001, 10.0),
    ("ALUT", "HTRK", 2.13e-05, 1.13, 3.47, 688000.0, 0.000808, 2.18, 0.001, 10.0),
    ("SRNA", "NWIG", 3.02e-07, 0.925, 14.6, 17600.0, 0.00678, 1.54, 0.001, 10.0),
    ("ACAC", "NAYO", 1.03e-06, 1.44, 14.2, 230000.0, 0.00241, 1.57, 0.001, 10.0),
    ("MSDP", "ULXY", 9.77e-08, 0.314, 8.33, 2610.0, 0.0882, 1.09, 0.001, 10.0),
    ("RTTH", "ULXY", 4.2e-07, 0.293, 9.78, 499.0, 0.00864, 1.36, 0.001, 10.0),
    ("AENO", "HUXZ", 4.76e-08, 1.79, 5.5, 1120.0, 0.000721, 1.54, 0.001, 10.0),
    ("SRNA", "FWWG", 2.62e-08, 0.732, 9.16, 553.0, 0.000186, 0.953, 0.001, 10.0),
    ("SMAH", "NGTE", 7.83e-07, 0.975, 4.66, 1420.0, 0.01, 1.27, 0.001, 10.0),
    ("EORC", "ILVX", 5.08e-05, 0.251, 19.4, 26100000.0, 0.000261, 0.941, 0.001, 10.0),
    ("ANSO", "AGVF", 1.29e-06, 0.577, 2.86, 237000.0, 0.000511, 1.8, 0.001, 10.0),
    ("ANSO", "AETS", 8.72e-08, 0.072, 10.9, 144000.0, 1.45e-05, 2.08, 0.001, 10.0),
    ("ANSO", "FCSG", 1.48e-07, 0.843, 13.5, 87500.0, 0.0228, 2.03, 0.001, 10.0),
    ("ANSO", "MHRM", 4.79e-08, 0.995, 15.1, 47000.0, 0.000561, 2.39, 0.001, 10.0),
    ("NGTE", "ILVX", 6.28e-07, 1.25, 12.0, 97000.0, 0.0572, 0.91, 0.001, 10.0),
    ("MHRM", "EAFC", 3.83e-07, 0.133, 2.43, 693.0, 0.0105, 0.994, 0.001, 10.0),
)

def _merge_supplied_profiles() -> tuple[tuple[object, ...], ...]:
    """Overlay supplied profiles by oriented pair name, preserving book order."""
    supplied = {(row[0], row[1]): row for row in _SUPPLIED_AUTO_PAIR_PARAMS}
    if len(supplied) != len(_SUPPLIED_AUTO_PAIR_PARAMS):
        raise RuntimeError("The supplied auto-profile catalog has duplicates")

    merged = []
    seen = set()
    for row in _SELECTED_PAIR_PARAMS:
        key = (row[0], row[1])
        merged.append(supplied.get(key, row))
        seen.add(key)
    merged.extend(
        row
        for row in _SUPPLIED_AUTO_PAIR_PARAMS
        if (row[0], row[1]) not in seen
    )
    return tuple(merged)

# Selected without days 751-1000: removing each relationship improved the
# aggregate train and validation scores and every constituent development
# half-window at the point it was removed.
DEVELOPMENT_REMOVED_PAIR_NAMES = frozenset(
    {
        "ALGO-HRET",
        "SRNA-ELLT",
        "ULXY-EAFC",
    }
)

_UNFILTERED_CURRENT_PAIR_PARAMS = _merge_supplied_profiles()
_ACTIVE_PAIR_PARAMS = tuple(
    row
    for row in _UNFILTERED_CURRENT_PAIR_PARAMS
    if f"{row[0]}-{row[1]}" not in DEVELOPMENT_REMOVED_PAIR_NAMES
)

def _build_selected_pair_cfgs() -> tuple[PairCfg, ...]:
    configs = []
    for idx, row in enumerate(_ACTIVE_PAIR_PARAMS):
        (
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
        ) = row
        configs.append(
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
    return tuple(configs)

PAIR_CFGS = _build_selected_pair_cfgs()

# Standalone ridge parameters. Portfolio conflicts use the legacy
# lambda=1/theta=0.75 model only as an independent sign confirmation: the new
# ridge may overturn an opposing nonzero pair vote only when both ridge models
# point the same way. The categorical policy was selected on days 251-500 and
# confirmed on 501-750 before 751-1000 was scored.
RIDGE_LAM = 1.52
RIDGE_THETA = 0.31
RIDGE_CONFIRM_LAM = 2.0
RIDGE_CONFIRM_THETA = 1.35
RIDGE_GATE_N = 35
RIDGE_LEG = 10_000.0
RIDGE_MIN_T = 252
RIDGE_LOOKBACK: int | None = None
RIDGE_MAX_ACTIVE = N_INST - 1
# Exact pair deadlocks leave the full instrument slot unused. When the global
# ridge gate is open, use that capacity rather than the former 25% overlay.
DEADLOCK_RIDGE_OVERLAY_FRACTION = 1.0

# Asset-level conflict policy: days 31-500 train, days 501-750 validation.
# Days 751-1000 are held out from configuration selection.
PAIR_CONFLICT_MODE = "we love sussy q"
PROTECTED_VOTE_BOOST = 4.0
RIDGE_CONFLICT_WEIGHT = 0.75
USE_PAIR_CONFLICT_VOTES = True
USE_RIDGE_CONFLICT_VOTES = True
BANG_BANG_LIMITS = np.full(N_INST, LEG)
BANG_BANG_LIMITS[I["ALGO"]] = 0

# ALGO-residual basket. These are the twelve pre-existing ALGO-X auto profiles
# studied on chronological development/validation folds. ALGO is used only to
# form each residual signal; it is never held. The asset legs are combined and
# their estimated ALGO beta is removed with a distributed, capacity-aware hedge
# over the other non-ALGO assets.
_ALGO_RESIDUAL_PARAMS = (
    ("ALGO", "ACIX", 5.54e-06, 1.53, 10.0, 607.0, 0.0363, 1.40, 0.001, 10.0),
    ("ALGO", "AETS", 2.21e-08, 1.74, 15.7, 1970.0, 0.000111, 1.79, 0.001, 10.0),
    ("ALGO", "ANSO", 0.000348, 1.08, 4.78, 7070.0, 0.0573, 1.48, 0.001, 10.0),
    ("ALGO", "CUBO", 2.06e-06, 0.188, 16.4, 29.3, 0.000844, 2.13, 0.001, 10.0),
    ("ALGO", "FCSG", 0.000318, 1.48, 15.3, 2350.0, 0.00801, 2.14, 0.001, 10.0),
    ("ALGO", "HRET", 6.86e-06, 0.439, 13.4, 13000.0, 0.000535, 1.69, 0.001, 10.0),
    ("ALGO", "LSST", 2.29e-07, 1.65, 6.58, 203000.0, 0.0729, 1.32, 0.001, 10.0),
    ("ALGO", "MHRM", 2.12e-05, 1.75, 3.47, 23300000.0, 0.078, 2.04, 0.001, 10.0),
    ("ALGO", "NGTE", 1.35e-05, 0.226, 8.39, 3320.0, 0.000108, 1.14, 0.001, 10.0),
    ("ALGO", "NPCK", 5.79e-08, 1.32, 17.6, 1900000.0, 0.000225, 2.15, 0.001, 10.0),
    ("ALGO", "OTCS", 8.03e-07, 1.38, 7.99, 118000.0, 0.000014, 1.44, 0.001, 10.0),
    ("ALGO", "SPLZ", 3.81e-08, 0.381, 7.70, 96900.0, 0.0555, 2.05, 0.001, 10.0),
)
ALGO_RESIDUAL_SLOT = 100_000.0 / len(_ALGO_RESIDUAL_PARAMS)
ALGO_BETA_WINDOW = 250
ALGO_BETA_MIN_OBS = 60
ALGO_HEDGE_BETA_FLOOR = 0.05
# The beta-neutral ALGO residual basket is part of the current combined book.
# Its positions remain subject to the same per-instrument limits below.
USE_ALGO_RESIDUAL_BASKET = True

def _build_algo_residual_cfgs() -> tuple[PairCfg, ...]:
    configs = []
    for idx, row in enumerate(_ALGO_RESIDUAL_PARAMS):
        (
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
        ) = row
        configs.append(
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
    return tuple(configs)

ALGO_RESIDUAL_CFGS = _build_algo_residual_cfgs()
_ALGO_RESIDUAL_ASSETS = np.asarray(
    [int(spec.x) for spec in ALGO_RESIDUAL_CFGS],
    dtype=int,
)

PROTECTED_PAIR_NAMES = frozenset(
    {
        "AENO-NWIG",
        "CTGI-EELT",
        "EORC-NGTE",
        "HETT-ULXY",
        "HUXZ-ACAC",
        "RTTH-NAYO",
        "SMAH-ILVX",
    }
)

if PAIR_CONFLICT_MODE not in {"priority", "we love sussy q"}:
    raise RuntimeError(
        f"Unsupported pair conflict mode: {PAIR_CONFLICT_MODE!r}"
    )

PREVIOUS_OOS_REMOVED_PAIR_NAMES = frozenset(
    {
        "CCNS-FARS",
        "DIHO-RTTH",
        "EORC-SPLZ",
        "NAYO-FWWG",
        "SPLZ-FWWG",
    }
)

SUPPLIED_REINTRODUCED_PAIR_NAMES = frozenset(
    {
        "ANSO-MHRM",
        "NGTE-ILVX",
        "SMAH-NGTE",
    }
)

_SYMBOL_BY_INDEX = {index: symbol for symbol, index in I.items()}

def _pair_name(spec: PairCfg) -> str:
    return f"{_SYMBOL_BY_INDEX[spec.y]}-{_SYMBOL_BY_INDEX[spec.x]}"

_CONFIGURED_PAIR_NAMES = {_pair_name(spec) for spec in PAIR_CFGS}
if len(PAIR_CFGS) > 88 or len(_CONFIGURED_PAIR_NAMES) > 88:
    raise RuntimeError("The development-filtered catalog must contain at most 88 pairs")
if not PROTECTED_PAIR_NAMES.issubset(_CONFIGURED_PAIR_NAMES):
    raise RuntimeError("A protected pair is absent from the approved catalog")
if not SUPPLIED_REINTRODUCED_PAIR_NAMES.issubset(_CONFIGURED_PAIR_NAMES):
    raise RuntimeError("A supplied reintroduced relationship is absent")
if DEVELOPMENT_REMOVED_PAIR_NAMES & _CONFIGURED_PAIR_NAMES:
    raise RuntimeError("A development-rejected relationship remains configured")
if PREVIOUS_OOS_REMOVED_PAIR_NAMES & _CONFIGURED_PAIR_NAMES:
    raise RuntimeError("A previously excluded relationship was reintroduced")

def _ridge_forecast_details(
    returns: np.ndarray,
    *,
    ridge_lam: float | None = None,
    ridge_theta: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return active positions, strengths, latent signs, and gate state."""
    lam = RIDGE_LAM if ridge_lam is None else float(ridge_lam)
    theta = RIDGE_THETA if ridge_theta is None else float(ridge_theta)
    nins, n_returns = returns.shape
    positions = np.zeros(nins)
    strengths = np.zeros(nins)
    latent_sign = np.zeros(nins)
    if n_returns + 1 < RIDGE_MIN_T:
        return positions, strengths, latent_sign, False

    recent = returns[1:, -250:].T
    standardized = (
        recent - recent.mean(axis=0)
    ) / (recent.std(axis=0) + 1.0e-12)
    lagged_links = (
        standardized[:-1].T @ standardized[1:]
    ) / (len(recent) - 1)
    np.fill_diagonal(lagged_links, 0.0)
    significant_links = int(
        (
            np.abs(lagged_links)
            > 2.5 / np.sqrt(len(recent) - 1)
        ).sum()
    )
    if significant_links < RIDGE_GATE_N:
        return positions, strengths, latent_sign, False

    history = returns[1:, :].T
    if RIDGE_LOOKBACK is not None:
        history = history[-RIDGE_LOOKBACK:]
    scale = history.std(axis=0) + 1.0e-12
    predictors = history[:-1] / scale
    targets = history[1:] / scale
    coefficients = np.linalg.solve(
        predictors.T @ predictors
        + lam * len(predictors) * np.eye(N_INST - 1),
        predictors.T @ targets,
    )
    prediction_scale = (predictors @ coefficients).std(axis=0)
    prediction = (history[-1] / scale) @ coefficients
    latent_sign[1:] = np.sign(prediction)
    strengths[1:] = np.abs(prediction) / np.maximum(
        prediction_scale,
        1.0e-12,
    )
    active = strengths[1:] > theta
    active_indices = np.flatnonzero(active)
    if len(active_indices) > RIDGE_MAX_ACTIVE:
        keep = active_indices[
            np.argsort(strengths[1:][active_indices])[-RIDGE_MAX_ACTIVE:]
        ]
        active[:] = False
        active[keep] = True
    positions[1:] = RIDGE_LEG * np.sign(prediction) * active
    return positions, strengths, latent_sign, True

def _resolve_pair_ridge_direction(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_active: np.ndarray,
    confirm_sign: np.ndarray,
    confirm_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve asset conflicts with an independently confirmed ridge override.

    The tuned ridge remains free to trade an unused or deadlocked asset slot.
    When a nonzero pair consensus points the other way, however, ridge can
    overturn it only if the legacy ridge specification independently agrees
    and the tuned ridge vote is large enough to win the unfiltered vote.
    """
    ridge_vote = RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength
    combined_vote = pair_vote + ridge_vote
    preliminary_direction = np.sign(combined_vote)
    direction = preliminary_direction.copy()

    # A non-conflict tie with an active ridge uses ridge's direction. Pair-only
    # ties remain flat; opposing pair/ridge ties are handled as conflicts below.
    active = pair_active | ridge_active
    tied = active & (direction == 0.0) & ridge_active
    direction[tied] = ridge_sign[tied]

    pair_direction = np.sign(pair_vote)
    conflict = (
        pair_active
        & ridge_active
        & (pair_direction != 0.0)
        & (pair_direction != ridge_sign)
    )
    ridge_confirmed = (
        conflict
        & confirm_active
        & (confirm_sign == ridge_sign)
        & (preliminary_direction == ridge_sign)
    )
    direction[conflict] = pair_direction[conflict]
    direction[ridge_confirmed] = ridge_sign[ridge_confirmed]
    return combined_vote, direction, conflict, ridge_confirmed

def _ridge_forecast(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal VAR(1) ridge forecast using this strategy's tuned constants."""
    positions, strengths, _, _ = _ridge_forecast_details(returns)
    return positions, strengths

def _rolling_algo_return_betas(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Estimate causal asset return betas and correlations to ALGO."""
    betas = np.zeros(N_INST)
    correlations = np.zeros(N_INST)
    if returns.shape[1] < ALGO_BETA_MIN_OBS:
        return betas, correlations, False

    known = returns[:, -ALGO_BETA_WINDOW:]
    centered = known - known.mean(axis=1, keepdims=True)
    algo = centered[I["ALGO"]]
    algo_ss = float(algo @ algo)
    if algo_ss <= _EPS_VAR:
        return betas, correlations, False

    cross = centered @ algo
    betas = cross / algo_ss
    norms = np.sqrt(np.sum(centered * centered, axis=1) * algo_ss)
    correlations = np.divide(
        cross,
        norms,
        out=np.zeros(N_INST),
        where=norms > _EPS_VAR,
    )
    betas[I["ALGO"]] = 1.0
    correlations[I["ALGO"]] = 1.0
    return betas, correlations, True

def _bounded_minimum_norm_beta_hedge(
    target_beta_dollars: float,
    betas: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a bounded minimum-L2 hedge for one beta-dollar constraint."""
    hedge = np.zeros(N_INST)
    use = (
        np.asarray(eligible, dtype=bool)
        & np.isfinite(betas)
        & (np.abs(betas) >= ALGO_HEDGE_BETA_FLOOR)
    )
    use[I["ALGO"]] = False
    if abs(target_beta_dollars) <= 1.0e-9:
        return hedge, 1.0
    if not np.any(use):
        return hedge, 0.0

    b = betas[use]
    lo = lower[use]
    hi = upper[use]
    minimum = float(b @ np.where(b >= 0.0, lo, hi))
    maximum = float(b @ np.where(b >= 0.0, hi, lo))
    delivered_target = float(
        np.clip(target_beta_dollars, minimum, maximum)
    )
    delivery = float(
        np.clip(
            delivered_target / target_beta_dollars,
            0.0,
            1.0,
        )
    )

    def exposure(multiplier: float) -> float:
        return float(b @ np.clip(-multiplier * b, lo, hi))

    left = -1.0
    right = 1.0
    while exposure(left) < delivered_target:
        left *= 2.0
    while exposure(right) > delivered_target:
        right *= 2.0
    for _ in range(80):
        middle = 0.5 * (left + right)
        if exposure(middle) > delivered_target:
            left = middle
        else:
            right = middle
    multiplier = 0.5 * (left + right)
    hedge[use] = np.clip(-multiplier * b, lo, hi)
    return hedge, delivery

def _algo_beta_neutral_overlay(
    base_dollars: np.ndarray,
    requested_alpha: np.ndarray,
    betas: np.ndarray,
    crash: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Add ALGO-residual alpha and hedge its beta without holding ALGO."""
    lower = -BANG_BANG_LIMITS - base_dollars
    upper = BANG_BANG_LIMITS - base_dollars
    alpha = np.clip(requested_alpha, lower, upper)
    alpha[I["ALGO"]] = 0.0
    alpha[crash] = 0.0

    eligible = np.ones(N_INST, dtype=bool)
    eligible[I["ALGO"]] = False
    eligible[_ALGO_RESIDUAL_ASSETS] = False
    eligible[crash] = False

    alpha_beta = float(betas @ alpha)
    hedge, delivery = _bounded_minimum_norm_beta_hedge(
        -alpha_beta,
        betas,
        lower,
        upper,
        eligible,
    )
    if delivery < 1.0:
        alpha *= delivery
        alpha_beta = float(betas @ alpha)
        hedge, _ = _bounded_minimum_norm_beta_hedge(
            -alpha_beta,
            betas,
            lower,
            upper,
            eligible,
        )

    final = np.clip(
        base_dollars + alpha + hedge,
        -BANG_BANG_LIMITS,
        BANG_BANG_LIMITS,
    )
    final[I["ALGO"]] = 0.0
    return final, alpha, hedge, delivery

class AtlasBook:
    """Stateful pair agents; allocation is performed by portfolio voting."""

    def __init__(self) -> None:
        self.agents = [
            PairAgent(spec)
            for spec in PAIR_CFGS
            if (
                not USE_ALGO_RESIDUAL_BASKET
                or I["ALGO"] not in (int(spec.y), int(spec.x))
            )
        ]
        self.algo_residual_agents = (
            [PairAgent(spec) for spec in ALGO_RESIDUAL_CFGS]
            if USE_ALGO_RESIDUAL_BASKET
            else []
        )

    def candidates(
        self,
        prices: np.ndarray,
    ) -> list[tuple[float, PairAgent, float, float]]:
        if prices.shape[1] < WARMUP + 10:
            return []
        for agent in self.agents:
            agent.catch_up(prices)

        candidates = []
        for agent in self.agents:
            pos_y, pos_x, abs_z = agent.dollar_positions()
            if abs(pos_y) < 1.0e-9:
                continue
            priority = agent.current_kappa * abs_z
            candidates.append((priority, agent, pos_y, pos_x))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def algo_residual_alpha(self, prices: np.ndarray) -> np.ndarray:
        """Return asset-only residual proposals before portfolio hedging."""
        alpha = np.zeros(N_INST)
        scale = ALGO_RESIDUAL_SLOT / LEG
        for agent in self.algo_residual_agents:
            agent.catch_up(prices)
            _, pos_x, _ = agent.dollar_positions()
            alpha[int(agent.spec.x)] += scale * pos_x
        alpha[I["ALGO"]] = 0.0
        return alpha

def _pair_conflict_consensus(
    candidates: list[tuple[float, PairAgent, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine every pair leg into a normalized signed asset vote."""
    numerator = np.zeros(N_INST)
    denominator = np.zeros(N_INST)
    contributor_count = np.zeros(N_INST, dtype=int)
    for priority, agent, pos_y, pos_x in candidates:
        boost = (
            PROTECTED_VOTE_BOOST
            if _pair_name(agent.spec) in PROTECTED_PAIR_NAMES
            else 1.0
        )
        if PAIR_CONFLICT_MODE == "priority":
            vote_weight = boost * max(float(priority), 1.0e-12)
        else:
            vote_weight = boost
        for index, position in (
            (int(agent.spec.y), pos_y),
            (int(agent.spec.x), pos_x),
        ):
            if abs(position) <= 1.0e-9:
                continue
            numerator[index] += vote_weight * np.sign(position)
            denominator[index] += vote_weight
            contributor_count[index] += 1
    consensus = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return consensus, denominator > 0.0, contributor_count

_STATE: dict[str, AtlasBook | None] = {"atlas": None}

def reset() -> None:
    _STATE["atlas"] = None

def sleeve_breakdown(prcSoFar: np.ndarray) -> dict[str, object] | None:
    prices = np.asarray(prcSoFar, dtype=float)
    nins, nt = prices.shape
    if nins != N_INST or nt < WARMUP + 10:
        return None
    if (
        not np.all(np.isfinite(prices[:, -2:]))
        or np.any(prices[:, -2:] <= 0.0)
    ):
        return None

    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()
    candidates = _STATE["atlas"].candidates(prices)

    pair_vote, pair_active, pair_contributors = (
        _pair_conflict_consensus(candidates)
    )
    if not USE_PAIR_CONFLICT_VOTES:
        pair_vote[:] = 0.0
        pair_active[:] = False

    returns = np.diff(np.log(prices), axis=1)
    (
        ridge_raw,
        ridge_strength,
        ridge_latent_sign,
        ridge_gate_pass,
    ) = _ridge_forecast_details(returns)
    (
        ridge_confirmation_raw,
        ridge_confirmation_strength,
        ridge_confirmation_latent_sign,
        ridge_confirmation_gate_pass,
    ) = _ridge_forecast_details(
        returns,
        ridge_lam=RIDGE_CONFIRM_LAM,
        ridge_theta=RIDGE_CONFIRM_THETA,
    )
    ridge_sign = np.sign(ridge_raw)
    ridge_active = ridge_sign != 0.0
    ridge_confirmation_sign = np.sign(ridge_confirmation_raw)
    ridge_confirmation_active = ridge_confirmation_sign != 0.0
    effective_ridge_strength = ridge_strength.copy()
    if not USE_RIDGE_CONFLICT_VOTES:
        effective_ridge_strength[:] = 0.0
        ridge_active[:] = False
        ridge_confirmation_active[:] = False

    ridge_vote = (
        RIDGE_CONFLICT_WEIGHT
        * ridge_sign
        * effective_ridge_strength
    )
    (
        combined_vote,
        direction,
        conflict,
        ridge_conflict_confirmed,
    ) = _resolve_pair_ridge_direction(
        pair_vote,
        pair_active,
        ridge_sign,
        effective_ridge_strength,
        ridge_active,
        ridge_confirmation_sign,
        ridge_confirmation_active,
    )
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], _EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    direction[crash] = 0.0
    dollars = BANG_BANG_LIMITS * direction
    deadlock_ridge_overlay = (
        USE_RIDGE_CONFLICT_VOTES
        & ridge_gate_pass
        & pair_active
        & (np.abs(pair_vote) < 1.0e-12)
        & ~ridge_active
        & (ridge_strength > 0.0)
        & (ridge_strength <= RIDGE_THETA)
        & (ridge_latent_sign != 0.0)
        & ~crash
    )
    deadlock_ridge_overlay_dollars = np.zeros(N_INST)
    deadlock_ridge_overlay_dollars[deadlock_ridge_overlay] = (
        DEADLOCK_RIDGE_OVERLAY_FRACTION
        * BANG_BANG_LIMITS[deadlock_ridge_overlay]
        * ridge_latent_sign[deadlock_ridge_overlay]
    )
    dollars[deadlock_ridge_overlay] = (
        deadlock_ridge_overlay_dollars[deadlock_ridge_overlay]
    )
    core_dollars = dollars.copy()
    core_direction = np.sign(core_dollars)

    algo_residual_requested = np.zeros(N_INST)
    algo_return_betas = np.zeros(N_INST)
    algo_return_correlations = np.zeros(N_INST)
    algo_beta_ready = False
    algo_residual_alpha = np.zeros(N_INST)
    algo_residual_hedge = np.zeros(N_INST)
    algo_residual_delivery = 0.0
    if USE_ALGO_RESIDUAL_BASKET:
        algo_residual_requested = _STATE[
            "atlas"
        ].algo_residual_alpha(prices)
        (
            algo_return_betas,
            algo_return_correlations,
            algo_beta_ready,
        ) = _rolling_algo_return_betas(returns)
        if algo_beta_ready:
            (
                dollars,
                algo_residual_alpha,
                algo_residual_hedge,
                algo_residual_delivery,
            ) = _algo_beta_neutral_overlay(
                core_dollars,
                algo_residual_requested,
                algo_return_betas,
                crash,
            )
    algo_residual_overlay = dollars - core_dollars
    direction = np.sign(dollars)

    ridge_won = ridge_active & (core_direction == ridge_sign)
    pair_contributor_details = [
        {
            "pair": _pair_name(agent.spec),
            "y": int(agent.spec.y),
            "x": int(agent.spec.x),
            "side": int(agent.side),
            "kappa": float(agent.current_kappa),
            "abs_z": float(abs_z),
            "priority": float(priority),
            "cut": bool(agent.current_cut),
            "adapting": bool(agent.current_adapting),
            "vol_shift": bool(agent.current_vol_shift),
            "pos_y": float(pos_y),
            "pos_x": float(pos_x),
        }
        for priority, agent, pos_y, pos_x in candidates
        for abs_z in (abs(float(agent.current_z)),)
    ]
    return {
        "pair_vote": pair_vote,
        "pair_active_mask": pair_active,
        "pair_contributors": pair_contributors,
        "ridge_raw": ridge_raw,
        "ridge_vote": ridge_vote,
        "ridge_strength": ridge_strength,
        "ridge_latent_sign": ridge_latent_sign,
        "ridge_gate_pass": ridge_gate_pass,
        "ridge_active_mask": ridge_active,
        "ridge_confirmation_raw": ridge_confirmation_raw,
        "ridge_confirmation_strength": ridge_confirmation_strength,
        "ridge_confirmation_latent_sign": (
            ridge_confirmation_latent_sign
        ),
        "ridge_confirmation_sign": ridge_confirmation_sign,
        "ridge_confirmation_gate_pass": (
            ridge_confirmation_gate_pass
        ),
        "ridge_confirmation_active_mask": (
            ridge_confirmation_active
        ),
        "deadlock_ridge_overlay_mask": deadlock_ridge_overlay,
        "deadlock_ridge_overlay_dollars": deadlock_ridge_overlay_dollars,
        "core_dollars": core_dollars,
        "algo_beta_ready": algo_beta_ready,
        "algo_return_betas": algo_return_betas,
        "algo_return_correlations": algo_return_correlations,
        "algo_residual_requested": algo_residual_requested,
        "algo_residual_alpha": algo_residual_alpha,
        "algo_residual_hedge": algo_residual_hedge,
        "algo_residual_overlay": algo_residual_overlay,
        "algo_residual_delivery": algo_residual_delivery,
        "algo_residual_beta_dollars": float(
            algo_return_betas @ algo_residual_overlay
        ),
        "algo_residual_net_dollars": float(
            np.sum(algo_residual_overlay)
        ),
        "combined_vote": combined_vote,
        "direction": direction,
        "dollars": dollars,
        "conflict_mask": conflict,
        "ridge_conflict_confirmed_mask": ridge_conflict_confirmed,
        "ridge_won_mask": ridge_won,
        "crash_mask": crash,
        "pair_contributor_details": pair_contributor_details,
    }

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    prices = np.asarray(prcSoFar, dtype=float)
    nins, nt = prices.shape
    if nins != N_INST or nt < WARMUP + 10:
        return np.zeros(nins, dtype=int)

    sleeves = sleeve_breakdown(prices)
    if sleeves is None:
        return np.zeros(nins, dtype=int)

    dollars = sleeves["dollars"].copy()

    safe_prices = np.maximum(prices[:, -1], _EPS_DENOM)
    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], _EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    if len(crash) > 0:
        crash[0] = False
    dollars[crash] = 0.0

    if len(dollars) > 1:
        dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = np.clip(
        dollars[0],
        -BANG_BANG_LIMITS[0],
        BANG_BANG_LIMITS[0],
    )

    raw_shares = dollars / safe_prices
    share_limits = np.floor(
        BANG_BANG_LIMITS / safe_prices
    ).astype(int)
    final_shares = np.clip(
        np.round(raw_shares),
        -share_limits,
        share_limits,
    ).astype(int)
    return np.nan_to_num(final_shares, nan=0).astype(int)
