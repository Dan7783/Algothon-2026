"""Friend latest portfolio layer + OU engine on individually improved pairs.

Portfolio/conflict: friend_book_latest (ridge-confirm dual ridge, deadlock
overlay 1.0, protected boost, ALGO residual, bang-bang).
Engines: z-band for the friend's non-winner rows; OU for the catalog winners
(+ justified screened extras). OU exit mode via MERGED_OU_EXIT:
  optimal / new — hold until opposite a* (flip live; default)
  mean / conventional — exit flat when spread_z crosses 0
Ablation knobs via MERGED_* env vars (CONSENSUS_MARGIN, PROTECTED_BOOST,
FLIP_TO_FLAT, MAG_WEIGHTED, CATALOG_JSON, OU_CONFLICT, OU_SLEEVE_FRAC,
OU_SLEEVE_SPLIT, OU_SLEEVE_OPPOSE, OU_EXIT).

OU-specific conflict (MERGED_OU_CONFLICT):
  shared  — single bang-bang voter (default; margin/flip knobs apply)
  sleeve  — OU and z-band resolve inside separate sleeves, then combine
            (solo/agree → full LEG; oppose → net|flat|stronger)
  ou_flat — if shared direction would flip OU consensus, OU abstains (z-band keeps)
  twopass — z-band+ridge settle first; OU only fills unused asset slots
  flat_conflict — on pair/ridge conflict go flat (no bang-bang pick)
  stronger_vote — on conflict take larger |pair_vote| vs |ridge_vote|
  ml — sklearn model override on conflict (MERGED_ML_MODEL path)
"""

from __future__ import annotations

import json
import math
import os
from math import sqrt
from typing import NamedTuple

import numpy as np
from scipy.optimize import root_scalar

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
    kalman_var_ewma_alpha: float | None = None
    cusum_k: float = 0.5
    h: float = 8.0
    vol_cut: bool = True
    max_cov_inflate: float = 25.0
    cov_decay: float = 0.95

class TradeCfg(NamedTuple):
    z_entry: float
    z_exit: float
    z_stop: float


TRANSACTION_COST = 0.0001
# OU exit/hold policy. 'optimal'/'new' = flip at opposite a* (Leung-Li hold).
# 'mean'/'conventional' = flatten when spread_z crosses 0 (friend-style).
_OU_EXIT_RAW = os.environ.get("MERGED_OU_EXIT", "optimal").strip().lower()
if _OU_EXIT_RAW in ("mean", "conventional"):
    OU_EXIT_MODE = "mean"
elif _OU_EXIT_RAW in ("optimal", "new", ""):
    OU_EXIT_MODE = "optimal"
else:
    raise RuntimeError(
        f"Unsupported MERGED_OU_EXIT={_OU_EXIT_RAW!r}; "
        "expected optimal|new|mean|conventional"
    )
# Kept for get_optimal_z_entry call site compatibility (solver ignores rule).
Z_ENTRY_RULE = "conventional" if OU_EXIT_MODE == "mean" else "new"


class OUCfg(NamedTuple):
    ou_window: int = 100
    kappa_ewma_alpha: float = 0.1
    sigma_ewma_alpha: float = 0.1


def evaluate_ou_series(a: float, eff_c: float, N: int = 30) -> float:
    """Leung-Li V(a) residual used to solve optimal entry a*."""
    if a <= 0.0:
        return 1.0e6
    terms = []
    for n in range(N):
        log_num = math.lgamma(n + 0.5) - math.lgamma(0.5) - math.lgamma(n + 1)
        coeff = math.exp(log_num) * (math.sqrt(2.0) * a) ** (2 * n + 1) / (2 * n + 1)
        terms.append(coeff / math.factorial(n))
    series = sum(terms)
    return series - (a + eff_c)


def get_optimal_z_entry(
    kappa: float,
    sigma: float,
    roundtrip_cost: float,
    rule: str = "new",
) -> float:
    del rule  # 'new' and conventional share the same a* solver here
    if kappa <= 1.0e-12 or sigma <= 1.0e-12:
        return 1.0
    ou_std = sigma / math.sqrt(2.0 * kappa)
    eff_c = roundtrip_cost / max(ou_std, 1.0e-12)
    try:
        root = root_scalar(
            lambda a: evaluate_ou_series(a, eff_c),
            bracket=[1.0e-6, 8.0],
        )
        return float(root.root) if root.converged else 1.0
    except Exception:
        return 1.0


class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
    filt: FiltCfg
    trade: TradeCfg | None = None
    ou: "OUCfg" = None  # type: ignore[assignment]
    engine: str = "zband"

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
        if self.cfg.kalman_var_ewma_alpha is not None:
            lam = 1.0 - float(self.cfg.kalman_var_ewma_alpha)
        else:
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
        "current_theta",
        "a_star",
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
        self.current_theta = 0.0
        self.a_star = 1.0
        self.current_cut = False
        self.current_adapting = False
        self.current_vol_shift = False

    def _update_dynamic_ou(self) -> None:
        ou = self.spec.ou if self.spec.ou is not None else OUCfg()
        min_samples = max(2, int(ou.ou_window))
        if len(self.spread_hist) < min_samples:
            return

        s = np.array(self.spread_hist[-min_samples:])
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
            theta = a_mle / (1.0 - b_mle)

            kappa_lambda = 1.0 - ou.kappa_ewma_alpha
            sigma_lambda = 1.0 - ou.sigma_ewma_alpha
            self.current_kappa = (
                kappa_lambda * self.current_kappa
                + (1.0 - kappa_lambda) * raw_kappa
            )
            self.current_sigma = (
                sigma_lambda * self.current_sigma
                + (1.0 - sigma_lambda) * raw_sigma
            )
            self.current_theta = (
                kappa_lambda * self.current_theta
                + (1.0 - kappa_lambda) * theta
            )
            self.a_star = get_optimal_z_entry(
                self.current_kappa,
                self.current_sigma,
                TRANSACTION_COST * 2.0,
                rule=Z_ENTRY_RULE,
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

            if self.spec.engine == "zband":
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
                continue

            self._update_dynamic_ou()

            if self.current_kappa > 0.0:
                ou_std = self.current_sigma / sqrt(2.0 * self.current_kappa)
                spread_z = (
                    (self.spread_hist[-1] - self.current_theta)
                    / max(ou_std, 1.0e-8)
                )
            else:
                spread_z = 0.0

            # optimal: hold until opposite a* (flip live).
            # mean/conventional: flatten when spread_z crosses 0.
            if self.side == 0:
                if spread_z >= self.a_star:
                    self.side = -1
                elif spread_z <= -self.a_star:
                    self.side = 1
            elif self.side == 1:
                if OU_EXIT_MODE == "mean":
                    if spread_z >= 0.0:
                        self.side = 0
                elif spread_z >= self.a_star:
                    self.side = -1
            elif self.side == -1:
                if OU_EXIT_MODE == "mean":
                    if spread_z <= 0.0:
                        self.side = 0
                elif spread_z <= -self.a_star:
                    self.side = 1

        self.nt_seen = nt

    def dollar_positions(self) -> tuple[float, float, float]:
        if self.side == 0:
            return 0.0, 0.0, abs(self.current_z)

        beta = float(self.filt.beta)
        scale = LEG / max(1.0, abs(beta))
        pos_y = scale * float(self.side)
        pos_x = -beta * pos_y
        return pos_y, pos_x, abs(self.current_z)


# Engine-tagged hybrid catalog (friend zband base + OU winners). Provenance:
# auto_tuning_results/merged_friend_ou_catalog.json
_MERGED_PAIR_PARAMS = (
    ("zband", "AENO", "NWIG", 5.51e-06, 1.6, 13.3, 892000.0, 2.5e-05, 0.816, 0.001, 10.0),
    ("zband", "HETT", "ULXY", 6.93e-07, 1.5, 11.1, 457.0, 0.000478, 0.881, 0.001, 10.0),
    ("zband", "HUXZ", "ACAC", 1.54e-08, 0.405, 2.7, 119.0, 0.00534, 0.862, 0.001, 10.0),
    ("zband", "SMAH", "ILVX", 5.26e-08, 1.02, 9.78, 4850.0, 0.00211, 0.926, 0.001, 10.0),
    ("zband", "EORC", "NGTE", 1.51e-07, 0.545, 17.9, 1930.0, 0.000188, 0.801, 0.001, 10.0),
    ("zband", "RTTH", "NAYO", 6.32e-08, 0.873, 18.3, 5250.0, 0.0268, 0.925, 0.001, 10.0),
    ("zband", "CTGI", "EELT", 5.22e-08, 1.9, 18.4, 479000.0, 0.00259, 0.805, 0.001, 10.0),
    ("ou", "ELLT", "FCSG", 3.51e-06, 0.0511, 1.64, 28.2, 291000.0, 0.0397, 81, 0.0633, 0.35),
    ("zband", "HUXZ", "NWIG", 3.07e-08, 1.46, 13.2, 159.0, 0.0492, 0.843, 0.001, 10.0),
    ("zband", "DUCT", "ACAC", 7.44e-07, 0.688, 12.1, 91600.0, 0.0264, 0.865, 0.001, 10.0),
    ("zband", "ANSO", "CTGI", 7.15e-08, 1.05, 6.35, 295.0, 0.000193, 1.74, 0.001, 10.0),
    ("ou", "ANSO", "BENI", 6.68e-06, 0.0106, 1.05, 20.3, 244000.0, 0.00208, 103, 0.00459, 0.00111),
    ("zband", "FWWG", "BLBT", 2.13e-08, 1.17, 11.0, 204.0, 0.0116, 0.955, 0.001, 10.0),
    ("ou", "AETS", "ILVX", 3.7e-09, 0.0641, 1.32, 24.8, 5130000.0, 0.00189, 175, 0.000275, 0.00312),
    ("ou", "AETS", "FCSG", 0.00457, 0.0138, 0.848, 19.6, 9220000.0, 0.00302, 131, 0.0787, 0.00611),
    ("zband", "ULXY", "MHRM", 5.55e-08, 1.01, 14.5, 22100000.0, 0.00893, 2.11, 0.001, 10.0),
    ("zband", "HRET", "RTTH", 0.000248, 1.57, 5.81, 100.0, 0.000199, 1.01, 0.001, 10.0),
    ("ou", "ALUT", "CCNS", 2.39e-08, 0.239, 1.22, 37.3, 10300.0, 0.00437, 191, 0.0037, 0.00532),
    ("zband", "SMAH", "SPLZ", 3.42e-07, 0.308, 8.46, 21100.0, 0.0197, 0.969, 0.001, 10.0),
    ("zband", "ACIX", "ITPA", 0.000381, 1.51, 12.4, 6720.0, 0.000123, 0.967, 0.001, 10.0),
    ("zband", "MTNS", "BENI", 1.55e-06, 1.52, 18.4, 5170.0, 0.00332, 1.49, 0.001, 10.0),
    ("ou", "MTNS", "BLBT", 1.32e-08, 0.0271, 1.69, 10.6, 5930.0, 0.0107, 167, 0.0327, 0.00247),
    ("ou", "ALUT", "HTRK", 3.47e-07, 0.0628, 1.6, 24.6, 6060000.0, 0.000385, 96, 0.000234, 0.0168),
    ("ou", "SRNA", "NWIG", 7.28e-05, 0.109, 1.31, 28.7, 10400000.0, 6.28e-05, 109, 0.0348, 0.00263),
    ("zband", "ACAC", "NAYO", 1.03e-06, 1.44, 14.2, 230000.0, 0.00241, 1.57, 0.001, 10.0),
    ("ou", "MSDP", "ULXY", 0.000383, 0.0829, 1.42, 27.9, 501000.0, 0.0436, 95, 0.000207, 0.000104),
    ("zband", "RTTH", "ULXY", 4.2e-07, 0.293, 9.78, 499.0, 0.00864, 1.36, 0.001, 10.0),
    ("zband", "AENO", "HUXZ", 4.76e-08, 1.79, 5.5, 1120.0, 0.000721, 1.54, 0.001, 10.0),
    ("zband", "SRNA", "FWWG", 2.62e-08, 0.732, 9.16, 553.0, 0.000186, 0.953, 0.001, 10.0),
    ("zband", "EORC", "ILVX", 5.08e-05, 0.251, 19.4, 26100000.0, 0.000261, 0.941, 0.001, 10.0),
    ("zband", "ANSO", "AGVF", 1.29e-06, 0.577, 2.86, 237000.0, 0.000511, 1.8, 0.001, 10.0),
    ("zband", "ANSO", "AETS", 8.72e-08, 0.072, 10.9, 144000.0, 1.45e-05, 2.08, 0.001, 10.0),
    ("zband", "ANSO", "FCSG", 1.48e-07, 0.843, 13.5, 87500.0, 0.0228, 2.03, 0.001, 10.0),
    ("zband", "MHRM", "EAFC", 3.83e-07, 0.133, 2.43, 693.0, 0.0105, 0.994, 0.001, 10.0),
    ("zband", "AGVF", "RCRI", 5.41e-06, 0.223, 4.11, 1.94, 0.00112, 1.65, 0.001, 10.0),
    ("zband", "MSDP", "SRTX", 1.91e-06, 0.795, 13.2, 31.1, 0.0372, 1.17, 0.001, 10.0),
    ("ou", "RTTH", "RRES", 9.35e-08, 0.00129, 1.67, 39.6, 3310.0, 0.00454, 105, 0.00102, 0.00725),
    ("ou", "HRET", "HRND", 6.11e-07, 0.00591, 0.667, 3.36, 8590.0, 0.0356, 105, 0.0908, 0.0105),
    ("zband", "HRET", "SRTX", 1.91e-06, 1.6, 3.76, 3320.0, 0.0195, 1.72, 0.001, 10.0),
    ("ou", "ANSO", "RCRI", 3.24e-08, 0.344, 1.67, 15.9, 19400000.0, 0.000207, 179, 0.154, 0.00344),
    ("zband", "HRET", "NGTE", 2.66e-07, 0.226, 19.5, 10500.0, 0.00681, 1.02, 0.001, 10.0),
    ("ou", "CUBO", "ULXY", 4.16e-05, 0.267, 1.92, 24.7, 839.0, 0.00556, 134, 0.00176, 0.114),
    ("ou", "ANSO", "RRES", 5.72e-05, 0.00964, 0.926, 11.0, 38900.0, 1.56e-05, 125, 0.00387, 0.0395),
    ("zband", "MTNS", "MHRM", 1.2e-07, 0.674, 14.5, 90500.0, 0.0504, 1.09, 0.001, 10.0),
    ("ou", "ANSO", "RTTH", 6.09e-05, 0.019, 1.63, 33.8, 21400000.0, 0.00984, 149, 0.00449, 0.109),
    ("zband", "SMAH", "AETS", 1.39e-07, 1.47, 3.39, 669000.0, 0.0354, 2.26, 0.001, 10.0),
    ("ou", "HRET", "MTNS", 1.85e-08, 0.000107, 0.999, 25.7, 324.0, 0.0862, 54, 0.00033, 0.00144),
    ("zband", "HUXZ", "DUCT", 1.47e-05, 0.185, 5.33, 96900.0, 0.00157, 1.13, 0.001, 10.0),
    ("zband", "HUXZ", "RRES", 4.34e-07, 0.824, 15.0, 11.8, 1.56e-05, 2.19, 0.001, 10.0),
    ("ou", "ANSO", "IHOZ", 4.21e-07, 0.189, 1.86, 36.0, 4470000.0, 0.00056, 57, 0.000874, 0.0302),
    ("zband", "DUCT", "NPCK", 5.41e-06, 1.94, 5.33, 2680000.0, 0.0573, 1.65, 0.001, 10.0),
    ("ou", "LSST", "EAFC", 0.00135, 0.00223, 0.9, 22.0, 23600.0, 0.000829, 154, 0.00278, 0.026),
    ("zband", "ELLT", "AETS", 4.98e-07, 0.52, 6.43, 29.3, 0.000404, 1.06, 0.001, 10.0),
    ("ou", "FWWG", "NGTE", 9.19e-05, 0.00429, 0.689, 19.5, 7020.0, 0.0154, 71, 0.00142, 0.00175),
    ("zband", "CUBO", "RTTH", 8.75e-05, 1.4, 16.4, 9280000.0, 0.00457, 1.14, 0.001, 10.0),
    ("ou", "HRET", "MDGI", 3.6e-06, 0.00279, 1.75, 9.52, 1540000.0, 0.00108, 93, 0.00132, 0.00553),
    ("ou", "SMAH", "FWWG", 1.78e-05, 0.00025, 1.46, 12.7, 81600.0, 0.000764, 98, 0.00854, 0.0081),
    ("ou", "SMAH", "ULXY", 0.000108, 0.00262, 0.209, 26.9, 1100000.0, 0.00515, 56, 0.317, 0.0627),
    ("ou", "RTTH", "FWWG", 3.86e-08, 0.283, 1.08, 33.6, 19000000.0, 0.00133, 163, 0.0473, 0.0416),
    ("ou", "OTCS", "EELT", 9.9e-08, 0.0458, 0.328, 34.1, 9260000.0, 0.00461, 187, 0.285, 0.000126),
    ("zband", "HRET", "SPLZ", 0.00045, 0.333, 12.1, 98.7, 0.000296, 1.09, 0.001, 10.0),
    ("zband", "NGTE", "FCSG", 1.31e-08, 1.3, 15.6, 4.71, 0.000308, 1.68, 0.001, 10.0),
    ("zband", "EORC", "FCSG", 2.33e-07, 1.36, 9.35, 94800000.0, 0.0195, 2.14, 0.001, 10.0),
    ("zband", "MTNS", "ULXY", 3.36e-06, 1.62, 16.1, 4090000.0, 0.00801, 1.09, 0.001, 10.0),
    ("ou", "AENO", "IHOZ", 9.15e-06, 0.146, 0.801, 18.9, 8900.0, 0.0159, 135, 0.17, 0.0474),
    ("zband", "AGVF", "ULXY", 9.47e-05, 0.0954, 5.63, 47300.0, 0.000147, 1.59, 0.001, 10.0),
    ("zband", "SPLZ", "ULXY", 0.000545, 0.615, 3.39, 5.55, 0.0523, 1.09, 0.001, 10.0),
    ("zband", "SPLZ", "CCNS", 4.35e-06, 0.296, 14.5, 1.25, 2.36e-05, 1.15, 0.001, 10.0),
    ("ou", "MSDP", "EAFC", 6.88e-09, 0.00229, 0.306, 19.5, 2480.0, 0.0102, 161, 0.0448, 0.000608),
    ("ou", "DUCT", "HRND", 1.29e-06, 0.00429, 1.46, 13.9, 65600.0, 0.00575, 110, 0.000687, 0.0262),
    ("ou", "LSST", "RTTH", 5.43e-09, 0.00575, 1.31, 16.4, 273.0, 0.00554, 126, 0.186, 0.00551),
    ("zband", "DIHO", "FCSG", 2.2e-07, 1.32, 7.81, 10500.0, 0.0523, 1.47, 0.001, 10.0),
    ("ou", "RTTH", "RCRI", 0.000108, 0.00262, 0.209, 26.9, 1100000.0, 0.00515, 56, 0.317, 0.0627),
    ("zband", "OTCS", "CTGI", 1.34e-08, 0.226, 11.6, 4810000.0, 0.0628, 1.37, 0.001, 10.0),
    ("ou", "GARI", "FWWG", 2.52e-07, 0.0115, 1.91, 21.7, 11600.0, 0.0124, 131, 0.00024, 0.0123),
    ("zband", "SPLZ", "FCSG", 0.000145, 0.388, 3.39, 31.6, 3.04e-05, 1.35, 0.001, 10.0),
    ("ou", "AMRP", "DUCT", 4.48e-05, 0.000284, 0.257, 37.9, 28700.0, 0.00048, 141, 0.000553, 0.018),
    ("zband", "SPLZ", "NGTE", 8.03e-07, 0.127, 18.4, 75800000.0, 0.0253, 1.09, 0.001, 10.0),
    ("ou", "RTTH", "SPLZ", 6.3e-06, 0.0216, 0.196, 18.5, 425.0, 0.00392, 149, 0.00378, 0.0559),
    ("ou", "CCNS", "NAYO", 1e-07, 0.000342, 0.416, 38.9, 1350000.0, 0.0745, 75, 0.0499, 0.000218),
    ("zband", "SPLZ", "ALUT", 3.73e-07, 1.54, 13.3, 462000.0, 0.078, 1.79, 0.001, 10.0),
    ("zband", "ULXY", "FCSG", 4.99e-08, 0.333, 6.31, 57400000.0, 0.0729, 1.79, 0.001, 10.0),
    ("zband", "NWIG", "MHRM", 1.98e-06, 1.51, 4.75, 2700.0, 0.0558, 1.56, 0.001, 10.0),
    ("zband", "SRTX", "MHRM", 0.000341, 0.228, 11.9, 3.47, 0.0398, 1.53, 0.001, 10.0),
    ("ou", "MDGI", "ULXY", 5.43e-09, 0.00575, 1.31, 16.4, 273.0, 0.00554, 126, 0.186, 0.00551),
    ("zband", "SMAH", "NGTE", 7.83e-07, 0.975, 4.66, 1420.0, 0.01, 1.27, 0.001, 10.0),
    ("zband", "ANSO", "MHRM", 4.79e-08, 0.995, 15.1, 47000.0, 0.000561, 2.39, 0.001, 10.0),
    ("ou", "NGTE", "ILVX", 0.0005, 0.00868, 0.953, 20.6, 7390.0, 0.0029, 165, 0.000923, 0.00144),
)

if os.environ.get("MERGED_CATALOG_JSON"):
    with open(os.environ["MERGED_CATALOG_JSON"]) as _catalog_f:
        _MERGED_PAIR_PARAMS = tuple(
            tuple(_row) for _row in json.load(_catalog_f)
        )

_ACTIVE_PAIR_PARAMS = _MERGED_PAIR_PARAMS


def _build_selected_pair_cfgs() -> tuple[PairCfg, ...]:
    configs = []
    for idx, row in enumerate(_ACTIVE_PAIR_PARAMS):
        engine, y_symbol, x_symbol = row[0], row[1], row[2]
        if engine == "ou":
            (
                q_beta,
                kalman_var_ewma_alpha,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                ou_window,
                kappa_ewma_alpha,
                sigma_ewma_alpha,
            ) = row[3:]
            configs.append(
                PairCfg(
                    idx=idx,
                    y=I[y_symbol],
                    x=I[x_symbol],
                    filt=FiltCfg(
                        q_beta=q_beta,
                        ewma_lambda=0.97,
                        kalman_var_ewma_alpha=kalman_var_ewma_alpha,
                        cusum_k=cusum_k,
                        h=h,
                        max_cov_inflate=max_cov_inflate,
                        cov_decay=cov_decay,
                    ),
                    trade=None,
                    ou=OUCfg(
                        ou_window=int(ou_window),
                        kappa_ewma_alpha=kappa_ewma_alpha,
                        sigma_ewma_alpha=sigma_ewma_alpha,
                    ),
                    engine="ou",
                )
            )
        elif engine == "zband":
            (
                q_beta,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                z_entry,
                z_exit,
                z_stop,
            ) = row[3:]
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
                    ou=OUCfg(),
                    engine="zband",
                )
            )
        else:
            raise ValueError(f"unknown engine tag {engine!r}")
    return tuple(configs)

PAIR_CFGS = _build_selected_pair_cfgs()

# Kept for the post-catalog integrity guards (pairs already filtered out of
# friend_book_latest._ACTIVE_PAIR_PARAMS before this hybrid catalog was built).
DEVELOPMENT_REMOVED_PAIR_NAMES = frozenset(
    ('ALGO-HRET', 'SRNA-ELLT', 'ULXY-EAFC')
)


# Standalone ridge parameters. Portfolio conflicts use the legacy
# lambda=1/theta=0.75 model only as an independent sign confirmation: the new
# ridge may overturn an opposing nonzero pair vote only when both ridge models
# point the same way. The categorical policy was selected on days 251-500 and
# confirmed on 501-750 before 751-1000 was scored.
RIDGE_LAM = 1.52
RIDGE_THETA = 0.31
RIDGE_CONFIRM_LAM = 1.0
RIDGE_CONFIRM_THETA = 0.75
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
PROTECTED_VOTE_BOOST = float(os.environ.get("MERGED_PROTECTED_BOOST", "4.0"))
RIDGE_CONFLICT_WEIGHT = float(os.environ.get("MERGED_RIDGE_WEIGHT", "0.75"))
# Near-tie flat: suppress weak asset consensus instead of bang-bang flip.
CONSENSUS_MARGIN = float(os.environ.get("MERGED_CONSENSUS_MARGIN", "0.15"))
# If >0, convert flipped legs to flat rather than opposite side.
FLIP_TO_FLAT = os.environ.get("MERGED_FLIP_TO_FLAT", "0") == "1"
# Magnitude-weighted pair votes (else one vote per contributing pair).
MAG_WEIGHTED_VOTES = os.environ.get("MERGED_MAG_WEIGHTED", "0") == "1"
# Structurally different OU vs z-band conflict path (not a margin/boost knob).
OU_CONFLICT_MODE = os.environ.get("MERGED_OU_CONFLICT", "shared").strip().lower()
OU_SLEEVE_FRAC = float(os.environ.get("MERGED_OU_SLEEVE_FRAC", "0.5"))
OU_SLEEVE_SPLIT = os.environ.get("MERGED_OU_SLEEVE_SPLIT", "equal").strip().lower()
# When both sleeves want opposite signs on an asset:
#   net      — weighted partial cancel (default)
#   flat     — both stand down (mutual veto)
#   stronger — denser/higher-|vote| sleeve takes full LEG
OU_SLEEVE_OPPOSE = os.environ.get("MERGED_OU_SLEEVE_OPPOSE", "net").strip().lower()
USE_PAIR_CONFLICT_VOTES = True
USE_RIDGE_CONFLICT_VOTES = True
BANG_BANG_LIMITS = np.full(N_INST, LEG)
BANG_BANG_LIMITS[I["ALGO"]] = 0
_VALID_OU_CONFLICT = frozenset({
    "shared",
    "sleeve",
    "ou_flat",
    "twopass",
    "flat_conflict",
    "stronger_vote",
    "ml",
})
_VALID_SLEEVE_OPPOSE = frozenset({"net", "flat", "stronger"})
_ML_MODEL = None
_ML_MODEL_PATH = os.environ.get("MERGED_ML_MODEL", "").strip()
if OU_CONFLICT_MODE not in _VALID_OU_CONFLICT:
    raise RuntimeError(
        f"Unsupported MERGED_OU_CONFLICT={OU_CONFLICT_MODE!r}; "
        f"expected one of {sorted(_VALID_OU_CONFLICT)}"
    )
if OU_SLEEVE_OPPOSE not in _VALID_SLEEVE_OPPOSE:
    raise RuntimeError(
        f"Unsupported MERGED_OU_SLEEVE_OPPOSE={OU_SLEEVE_OPPOSE!r}; "
        f"expected one of {sorted(_VALID_SLEEVE_OPPOSE)}"
    )

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
                ou=OUCfg(),
                engine="zband",
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
            signed = np.sign(position)
            w = vote_weight
            if MAG_WEIGHTED_VOTES:
                w = vote_weight * min(abs(position) / LEG, 1.0)
            numerator[index] += w * signed
            denominator[index] += w
            contributor_count[index] += 1
    consensus = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return consensus, denominator > 0.0, contributor_count


def _split_engine_candidates(
    candidates: list[tuple[float, PairAgent, float, float]],
) -> tuple[
    list[tuple[float, PairAgent, float, float]],
    list[tuple[float, PairAgent, float, float]],
]:
    ou: list[tuple[float, PairAgent, float, float]] = []
    zband: list[tuple[float, PairAgent, float, float]] = []
    for item in candidates:
        if getattr(item[1].spec, "engine", "zband") == "ou":
            ou.append(item)
        else:
            zband.append(item)
    return ou, zband


def _apply_margin_and_flip(
    direction: np.ndarray,
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
) -> np.ndarray:
    out = direction.copy()
    if CONSENSUS_MARGIN > 0.0:
        weak = (
            (np.abs(pair_vote) > 0.0)
            & (np.abs(pair_vote) < CONSENSUS_MARGIN)
            & pair_active
        )
        out[weak] = 0.0
    if FLIP_TO_FLAT:
        pair_dir = np.sign(pair_vote)
        flipped = (
            pair_active
            & (pair_dir != 0.0)
            & (out != 0.0)
            & (out != pair_dir)
        )
        out[flipped] = 0.0
    return out


def _resolve_pair_direction_bundle(
    candidates: list[tuple[float, PairAgent, float, float]],
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_active: np.ndarray,
    confirm_sign: np.ndarray,
    confirm_active: np.ndarray,
    *,
    use_ridge: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Pair consensus (+ optional ridge) → asset direction."""
    pair_vote, pair_active, pair_contributors = _pair_conflict_consensus(
        candidates
    )
    if not USE_PAIR_CONFLICT_VOTES:
        pair_vote = np.zeros(N_INST)
        pair_active = np.zeros(N_INST, dtype=bool)
        pair_contributors = np.zeros(N_INST, dtype=int)

    eff_strength = ridge_strength.copy()
    eff_active = ridge_active.copy()
    eff_confirm_active = confirm_active.copy()
    if not (use_ridge and USE_RIDGE_CONFLICT_VOTES):
        eff_strength = np.zeros(N_INST)
        eff_active = np.zeros(N_INST, dtype=bool)
        eff_confirm_active = np.zeros(N_INST, dtype=bool)

    combined_vote, direction, conflict, ridge_confirmed = (
        _resolve_pair_ridge_direction(
            pair_vote,
            pair_active,
            ridge_sign,
            eff_strength,
            eff_active,
            confirm_sign,
            eff_confirm_active,
        )
    )
    direction = _apply_margin_and_flip(direction, pair_vote, pair_active)
    return (
        pair_vote,
        pair_active,
        pair_contributors,
        combined_vote,
        direction,
        conflict,
        ridge_confirmed,
    )


def _sleeve_limits(
    ou_contributors: np.ndarray,
    zb_contributors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset OU/z-band dollar limits that sum to BANG_BANG_LIMITS."""
    base = BANG_BANG_LIMITS.copy()
    frac = float(np.clip(OU_SLEEVE_FRAC, 0.0, 1.0))
    if OU_SLEEVE_SPLIT == "demand":
        ou_n = ou_contributors.astype(float)
        zb_n = zb_contributors.astype(float)
        total = ou_n + zb_n
        ou_share = np.full(N_INST, frac)
        nonzero = total > 0.0
        ou_share[nonzero] = ou_n[nonzero] / total[nonzero]
        ou_limits = base * ou_share
        zb_limits = base - ou_limits
    else:
        ou_limits = base * frac
        zb_limits = base * (1.0 - frac)
    ou_limits[I["ALGO"]] = 0.0
    zb_limits[I["ALGO"]] = 0.0
    return ou_limits, zb_limits


def _deadlock_ridge_overlay_dollars(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    ridge_gate_pass: bool,
    ridge_active: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_latent_sign: np.ndarray,
    crash: np.ndarray,
    limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay_mask = (
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
    overlay = np.zeros(N_INST)
    overlay[overlay_mask] = (
        DEADLOCK_RIDGE_OVERLAY_FRACTION
        * limits[overlay_mask]
        * ridge_latent_sign[overlay_mask]
    )
    return overlay_mask, overlay

_STATE: dict[str, AtlasBook | None] = {"atlas": None}

def reset() -> None:
    _STATE["atlas"] = None


def _load_ml_model():
    global _ML_MODEL
    if _ML_MODEL is not None:
        return _ML_MODEL
    path = _ML_MODEL_PATH or os.environ.get("MERGED_ML_MODEL", "").strip()
    if not path:
        raise RuntimeError(
            "MERGED_OU_CONFLICT=ml requires MERGED_ML_MODEL path to a joblib model"
        )
    import joblib

    _ML_MODEL = joblib.load(path)
    return _ML_MODEL


def ml_conflict_feature_matrix(
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    pair_contributors: np.ndarray,
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_active: np.ndarray,
    confirm_sign: np.ndarray,
    confirm_active: np.ndarray,
    ou_vote: np.ndarray,
    zb_vote: np.ndarray,
    combined_vote: np.ndarray,
    conflict: np.ndarray,
) -> np.ndarray:
    """Per-asset feature rows for conflict ML (shape N_INST x F)."""
    return np.column_stack(
        [
            pair_vote,
            np.sign(pair_vote),
            pair_active.astype(float),
            pair_contributors.astype(float),
            ridge_sign,
            ridge_strength,
            ridge_active.astype(float),
            confirm_sign,
            confirm_active.astype(float),
            ou_vote,
            zb_vote,
            np.sign(ou_vote),
            np.sign(zb_vote),
            combined_vote,
            conflict.astype(float),
            np.abs(pair_vote) - np.abs(RIDGE_CONFLICT_WEIGHT * ridge_sign * ridge_strength),
            BANG_BANG_LIMITS.astype(float),
        ]
    )


def _ml_conflict_override(
    direction: np.ndarray,
    conflict: np.ndarray,
    pair_vote: np.ndarray,
    pair_active: np.ndarray,
    pair_contributors: np.ndarray,
    ridge_sign: np.ndarray,
    ridge_strength: np.ndarray,
    ridge_active: np.ndarray,
    confirm_sign: np.ndarray,
    confirm_active: np.ndarray,
    ou_vote: np.ndarray,
    zb_vote: np.ndarray,
    combined_vote: np.ndarray,
) -> np.ndarray:
    """Override conflict assets with sklearn model prediction in {-1,0,1}."""
    out = direction.copy()
    idx = np.flatnonzero(conflict)
    if idx.size == 0:
        return out
    model = _load_ml_model()
    feats = ml_conflict_feature_matrix(
        pair_vote,
        pair_active,
        pair_contributors,
        ridge_sign,
        ridge_strength,
        ridge_active,
        confirm_sign,
        confirm_active,
        ou_vote,
        zb_vote,
        combined_vote,
        conflict,
    )
    pred = np.asarray(model.predict(feats[idx]), dtype=float)
    pred = np.clip(np.rint(pred), -1, 1)
    out[idx] = pred
    return out


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
    ou_cands, zb_cands = _split_engine_candidates(candidates)

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

    last_return = np.diff(
        np.log(np.maximum(prices[:, -2:], _EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False

    ou_limits = np.zeros(N_INST)
    zb_limits = BANG_BANG_LIMITS.copy()
    ou_core = np.zeros(N_INST)
    zb_core = np.zeros(N_INST)
    deadlock_ridge_overlay = np.zeros(N_INST, dtype=bool)
    deadlock_ridge_overlay_dollars = np.zeros(N_INST)

    if OU_CONFLICT_MODE == "sleeve":
        (
            ou_vote,
            ou_active,
            ou_contrib,
            _,
            ou_direction,
            _,
            _,
        ) = _resolve_pair_direction_bundle(
            ou_cands,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=False,
        )
        (
            zb_vote,
            zb_active,
            zb_contrib,
            combined_vote,
            zb_direction,
            conflict,
            ridge_conflict_confirmed,
        ) = _resolve_pair_direction_bundle(
            zb_cands,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=True,
        )
        ou_limits, zb_limits = _sleeve_limits(ou_contrib, zb_contrib)
        ou_direction = ou_direction.copy()
        zb_direction = zb_direction.copy()
        ou_direction[crash] = 0.0
        zb_direction[crash] = 0.0
        # Capacity rule: solo sleeve gets full LEG; both same-sign → full LEG;
        # opposing signs → weighted net (neither engine can bang-bang flip the other).
        ou_core = np.zeros(N_INST)
        zb_core = np.zeros(N_INST)
        for i in range(N_INST):
            lim = float(BANG_BANG_LIMITS[i])
            if lim <= 0.0:
                continue
            od = float(ou_direction[i])
            zd = float(zb_direction[i])
            if od == 0.0 and zd == 0.0:
                continue
            if od == 0.0:
                zb_core[i] = lim * zd
            elif zd == 0.0:
                ou_core[i] = lim * od
            elif od == zd:
                # Agree: fill full limit; split notionally by sleeve weights.
                ou_share = float(ou_limits[i])
                zb_share = float(zb_limits[i])
                total = ou_share + zb_share
                if total <= 1.0e-12:
                    ou_core[i] = 0.5 * lim * od
                    zb_core[i] = 0.5 * lim * zd
                else:
                    ou_core[i] = lim * od * (ou_share / total)
                    zb_core[i] = lim * zd * (zb_share / total)
            elif OU_SLEEVE_OPPOSE == "flat":
                continue
            elif OU_SLEEVE_OPPOSE == "stronger":
                ou_strength = abs(float(ou_vote[i])) * max(int(ou_contrib[i]), 1)
                zb_strength = abs(float(zb_vote[i])) * max(int(zb_contrib[i]), 1)
                if ou_strength >= zb_strength:
                    ou_core[i] = lim * od
                else:
                    zb_core[i] = lim * zd
            else:
                # Opposing net: fractional sleeves cancel; neither bang-bang flips.
                ou_core[i] = float(ou_limits[i]) * od
                zb_core[i] = float(zb_limits[i]) * zd
        deadlock_ridge_overlay, deadlock_ridge_overlay_dollars = (
            _deadlock_ridge_overlay_dollars(
                zb_vote,
                zb_active,
                ridge_gate_pass,
                ridge_active,
                ridge_strength,
                ridge_latent_sign,
                crash,
                BANG_BANG_LIMITS,
            )
        )
        # Deadlock overlay only on still-empty slots after sleeve fill.
        empty = (np.abs(ou_core) + np.abs(zb_core)) < 1.0e-9
        apply_overlay = deadlock_ridge_overlay & empty
        zb_core[apply_overlay] = deadlock_ridge_overlay_dollars[apply_overlay]
        deadlock_ridge_overlay = apply_overlay
        pair_vote = ou_vote + zb_vote
        # Prefer stronger sleeve's normalized vote for diagnostics.
        pair_active = ou_active | zb_active
        pair_contributors = ou_contrib + zb_contrib
        direction = np.sign(ou_core + zb_core)
        core_dollars = ou_core + zb_core

    elif OU_CONFLICT_MODE == "twopass":
        (
            zb_vote,
            zb_active,
            zb_contrib,
            combined_vote,
            zb_direction,
            conflict,
            ridge_conflict_confirmed,
        ) = _resolve_pair_direction_bundle(
            zb_cands,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=True,
        )
        zb_direction = zb_direction.copy()
        zb_direction[crash] = 0.0
        zb_core = BANG_BANG_LIMITS * zb_direction
        deadlock_ridge_overlay, deadlock_ridge_overlay_dollars = (
            _deadlock_ridge_overlay_dollars(
                zb_vote,
                zb_active,
                ridge_gate_pass,
                ridge_active,
                ridge_strength,
                ridge_latent_sign,
                crash,
                BANG_BANG_LIMITS,
            )
        )
        zb_core[deadlock_ridge_overlay] = deadlock_ridge_overlay_dollars[
            deadlock_ridge_overlay
        ]
        (
            ou_vote,
            ou_active,
            ou_contrib,
            _,
            ou_direction,
            _,
            _,
        ) = _resolve_pair_direction_bundle(
            ou_cands,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=False,
        )
        ou_direction = ou_direction.copy()
        # Residual = unused slots only (bang-bang z-band already took full LEG).
        unused = (np.abs(zb_core) < 1.0e-9) & ~crash
        ou_direction[~unused] = 0.0
        ou_direction[crash] = 0.0
        residual = np.maximum(BANG_BANG_LIMITS - np.abs(zb_core), 0.0)
        ou_core = residual * ou_direction
        pair_vote = zb_vote.copy()
        pair_vote[unused] = ou_vote[unused]
        pair_active = zb_active | (ou_active & unused)
        pair_contributors = zb_contrib + ou_contrib
        direction = np.sign(zb_core + ou_core)
        core_dollars = zb_core + ou_core
        ou_limits = residual
        zb_limits = BANG_BANG_LIMITS.copy()

    elif OU_CONFLICT_MODE == "ou_flat":
        (
            pair_vote,
            pair_active,
            pair_contributors,
            combined_vote,
            direction,
            conflict,
            ridge_conflict_confirmed,
        ) = _resolve_pair_direction_bundle(
            candidates,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=True,
        )
        ou_vote, ou_active, ou_contrib = _pair_conflict_consensus(ou_cands)
        (
            zb_vote,
            zb_active,
            zb_contrib,
            _,
            zb_direction,
            _,
            _,
        ) = _resolve_pair_direction_bundle(
            zb_cands,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=True,
        )
        ou_des = np.sign(ou_vote)
        would_flip_ou = (
            ou_active
            & (ou_des != 0.0)
            & (direction != 0.0)
            & (direction != ou_des)
        )
        # OU abstains on flip assets: keep z-band (+ridge) direction instead.
        direction = direction.copy()
        direction[would_flip_ou] = zb_direction[would_flip_ou]
        direction[crash] = 0.0
        core_dollars = BANG_BANG_LIMITS * direction
        deadlock_ridge_overlay, deadlock_ridge_overlay_dollars = (
            _deadlock_ridge_overlay_dollars(
                pair_vote,
                pair_active,
                ridge_gate_pass,
                ridge_active,
                ridge_strength,
                ridge_latent_sign,
                crash,
                BANG_BANG_LIMITS,
            )
        )
        # Only overlay where still flat after ou_flat adjustment.
        still_flat = np.abs(core_dollars) < 1.0e-9
        apply_overlay = deadlock_ridge_overlay & still_flat
        core_dollars[apply_overlay] = deadlock_ridge_overlay_dollars[
            apply_overlay
        ]
        deadlock_ridge_overlay = apply_overlay
        ou_limits[:] = 0.0
        zb_limits = BANG_BANG_LIMITS.copy()

    elif OU_CONFLICT_MODE in ("shared", "flat_conflict", "stronger_vote", "ml"):
        (
            pair_vote,
            pair_active,
            pair_contributors,
            combined_vote,
            direction,
            conflict,
            ridge_conflict_confirmed,
        ) = _resolve_pair_direction_bundle(
            candidates,
            ridge_sign,
            effective_ridge_strength,
            ridge_active,
            ridge_confirmation_sign,
            ridge_confirmation_active,
            use_ridge=True,
        )
        direction = direction.copy()
        if OU_CONFLICT_MODE == "flat_conflict":
            direction[conflict] = 0.0
        elif OU_CONFLICT_MODE == "stronger_vote":
            pair_mag = np.abs(pair_vote)
            ridge_mag = np.abs(
                RIDGE_CONFLICT_WEIGHT * ridge_sign * effective_ridge_strength
            )
            take_ridge = conflict & (ridge_mag > pair_mag)
            take_pair = conflict & ~take_ridge
            direction[take_pair] = np.sign(pair_vote[take_pair])
            direction[take_ridge] = ridge_sign[take_ridge]
        elif OU_CONFLICT_MODE == "ml":
            ou_vote_ml, _, _ = _pair_conflict_consensus(ou_cands)
            zb_vote_ml, _, _ = _pair_conflict_consensus(zb_cands)
            direction = _ml_conflict_override(
                direction,
                conflict,
                pair_vote,
                pair_active,
                pair_contributors,
                ridge_sign,
                effective_ridge_strength,
                ridge_active,
                ridge_confirmation_sign,
                ridge_confirmation_active,
                ou_vote_ml,
                zb_vote_ml,
                combined_vote,
            )
        direction[crash] = 0.0
        core_dollars = BANG_BANG_LIMITS * direction
        deadlock_ridge_overlay, deadlock_ridge_overlay_dollars = (
            _deadlock_ridge_overlay_dollars(
                pair_vote,
                pair_active,
                ridge_gate_pass,
                ridge_active,
                ridge_strength,
                ridge_latent_sign,
                crash,
                BANG_BANG_LIMITS,
            )
        )
        core_dollars[deadlock_ridge_overlay] = deadlock_ridge_overlay_dollars[
            deadlock_ridge_overlay
        ]
        zb_limits = BANG_BANG_LIMITS.copy()

    else:  # unreachable — validated at import
        raise RuntimeError(f"Unhandled MERGED_OU_CONFLICT={OU_CONFLICT_MODE!r}")

    # Clip summed sleeves to instrument caps before ALGO overlay.
    dollars = np.clip(core_dollars, -BANG_BANG_LIMITS, BANG_BANG_LIMITS)
    # Re-zero ALGO (pair book never holds it).
    dollars[I["ALGO"]] = 0.0
    core_dollars = dollars.copy()
    core_direction = np.sign(core_dollars)
    direction = core_direction.copy()

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
            "engine": getattr(agent.spec, "engine", "zband"),
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
        "ou_conflict_mode": OU_CONFLICT_MODE,
        "ou_limits": ou_limits,
        "zb_limits": zb_limits,
        "ou_core_dollars": ou_core,
        "zb_core_dollars": zb_core,
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
