from __future__ import annotations

import math
from typing import NamedTuple, TypedDict
from scipy.optimize import root_scalar

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
TRANSACTION_COST = 0.0001

WARMUP = 30         
_INIT_WINDOW = 20   

_EPS_DENOM = 1e-12
_EPS_VAR = 1e-18

class RawPairDict(TypedDict):
    y: str
    x: str

    q_beta: float
    kalman_var_ewma_alpha: float
    cusum_k: float
    h: float
    max_cov_inflate: float
    cov_decay: float
    ou_window: int
    kappa_ewma_alpha: float
    sigma_ewma_alpha: float

class FiltCfg(NamedTuple):
    q_beta: float
    kalman_var_ewma_alpha: float = 0.97
    cusum_k: float = 0.5
    h: float = 8.0
    vol_cut: bool = True
    max_cov_inflate: float = 25.0
    cov_decay: float = 0.95

class OUCfg(NamedTuple):
    ou_window: int = 100
    kappa_ewma_alpha: float = 0.97
    sigma_ewma_alpha: float = 0.97

class PairCfg(NamedTuple):
    idx: int
    y: int
    x: int
    filt: FiltCfg
    ou: OUCfg

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
        lam = 1.0 - self.cfg.kalman_var_ewma_alpha
        if self.innov_var is None:
            self.innov_var = innov * innov
        else:
            self.innov_var = lam * self.innov_var + (1.0 - lam) * innov * innov
            
        z = innov / math.sqrt(self.innov_var) if self.innov_var > 0.0 else 0.0
        
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

        d_v = (z * z - 1.0) / math.sqrt(2.0)
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

# Per-pair parameters selected by the 8-pair windowed tuner bake-off
# (auto_tuning_results/_bakeoff_wratio_tbrk.json): ratio-normalized global
# study + purged walk-forward tiebreak re-rank, tuned under z_entry_rule='new'.
RAW_PAIRS: list[RawPairDict] = [
    {"y": "RTTH", "x": "NAYO", "q_beta": 2.56e-05, "kalman_var_ewma_alpha": 0.124,
     "cusum_k": 0.604, "h": 31.9, "max_cov_inflate": 7480.0, "cov_decay": 0.00309,
     "ou_window": 145, "kappa_ewma_alpha": 0.0737, "sigma_ewma_alpha": 0.286},
    {"y": "AENO", "x": "NWIG", "q_beta": 2.24e-07, "kalman_var_ewma_alpha": 0.331,
     "cusum_k": 0.967, "h": 30.3, "max_cov_inflate": 264.0, "cov_decay": 0.000759,
     "ou_window": 52, "kappa_ewma_alpha": 0.164, "sigma_ewma_alpha": 0.0129},
    {"y": "HETT", "x": "ULXY", "q_beta": 1.26e-05, "kalman_var_ewma_alpha": 0.0512,
     "cusum_k": 1.72, "h": 30.2, "max_cov_inflate": 1680000.0, "cov_decay": 7.02e-05,
     "ou_window": 152, "kappa_ewma_alpha": 0.00192, "sigma_ewma_alpha": 0.00154},
    {"y": "HUXZ", "x": "ACAC", "q_beta": 3.21e-07, "kalman_var_ewma_alpha": 0.0824,
     "cusum_k": 1.33, "h": 21.6, "max_cov_inflate": 74000.0, "cov_decay": 0.000535,
     "ou_window": 186, "kappa_ewma_alpha": 0.0419, "sigma_ewma_alpha": 0.00974},
    {"y": "SMAH", "x": "ILVX", "q_beta": 1.16e-05, "kalman_var_ewma_alpha": 0.175,
     "cusum_k": 1.19, "h": 24.8, "max_cov_inflate": 1580000.0, "cov_decay": 0.00252,
     "ou_window": 130, "kappa_ewma_alpha": 0.00272, "sigma_ewma_alpha": 0.0184},
    {"y": "EORC", "x": "NGTE", "q_beta": 6.29e-06, "kalman_var_ewma_alpha": 0.232,
     "cusum_k": 0.984, "h": 24.8, "max_cov_inflate": 59000000.0, "cov_decay": 0.000568,
     "ou_window": 114, "kappa_ewma_alpha": 0.101, "sigma_ewma_alpha": 0.000565},
    {"y": "CTGI", "x": "EELT", "q_beta": 6.04e-08, "kalman_var_ewma_alpha": 0.0686,
     "cusum_k": 1.15, "h": 21.3, "max_cov_inflate": 143000.0, "cov_decay": 0.00147,
     "ou_window": 123, "kappa_ewma_alpha": 0.00462, "sigma_ewma_alpha": 0.011},
    {"y": "ALUT", "x": "CCNS", "q_beta": 2.39e-08, "kalman_var_ewma_alpha": 0.239,
     "cusum_k": 1.22, "h": 37.3, "max_cov_inflate": 10300.0, "cov_decay": 0.00437,
     "ou_window": 191, "kappa_ewma_alpha": 0.0037, "sigma_ewma_alpha": 0.00532},
]

def _build_configs() -> tuple[PairCfg, ...]:
    out: list[PairCfg] = []
    for idx, row in enumerate(RAW_PAIRS):
        out.append(PairCfg(
            idx=idx, y=I[row["y"]], x=I[row["x"]],
            filt=FiltCfg(
                q_beta=row["q_beta"],
                kalman_var_ewma_alpha=row["kalman_var_ewma_alpha"],
                cusum_k=row["cusum_k"],
                h=row["h"],
                max_cov_inflate=row["max_cov_inflate"],
                cov_decay=row["cov_decay"],
            ),
            ou=OUCfg(
                ou_window=row["ou_window"],
                kappa_ewma_alpha=row["kappa_ewma_alpha"],
                sigma_ewma_alpha=row["sigma_ewma_alpha"],
            )))
    return tuple(out)

PAIR_CFGS = _build_configs()

# Which optimal-stopping rule to use for the dynamic z-entry threshold: 'new' (hold until
# reversion, halved effective cost) or 'conventional' (exit at mean). The hardcoded
# per-pair parameters above were selected under 'new'.
Z_ENTRY_RULE = "new"

def evaluate_ou_series(a: float, eff_c: float, N: int = 30) -> float:
    """
    Evaluates the truncated infinite series F(a) for the optimal stopping equation.
    
    Parameters:
    a (float): The dimensionless z-entry threshold we are testing.
    eff_c (float): The effective dimensionless transaction cost.
    N (int): The number of terms to calculate before truncating the series.
    """
    s1 = 0.0
    s2 = 0.0
    sqrt2_a = math.sqrt(2.0) * a
    
    for n in range(N):
        # Calculate Gamma(n + 1/2)
        gamma_val = math.gamma(n + 0.5)
        
        # Calculate the first series term: (sqrt(2)a)^(2n+1) / (2n+1)! * Gamma(n + 1/2)
        term1 = (sqrt2_a ** (2 * n + 1)) / math.factorial(2 * n + 1) * gamma_val
        s1 += term1
        
        # Calculate the second series term: (sqrt(2)a)^(2n) / (2n)! * Gamma(n + 1/2)
        term2 = (sqrt2_a ** (2 * n)) / math.factorial(2 * n) * gamma_val
        s2 += term2
        
    return 0.5 * s1 - (a - eff_c) * (math.sqrt(2.0) / 2.0) * s2

def get_optimal_z_entry(kappa: float, sigma: float, cost_real: float, rule: str = 'conventional') -> float:
    """
    Solves for the optimal z-entry (a*) given real-world parameters.
    
    Parameters:
    kappa (float): Speed of mean reversion.
    sigma (float): Volatility of the OU process.
    cost_real (float): Real-world transaction cost/slippage in spread units.
    rule (str): 'conventional' (exit at mean) or 'new' (hold until reversion).
    """
    if kappa <= 0 or sigma <= 0:
        raise ValueError("Kappa and Sigma must be strictly positive.")
        
    # 1. Convert real cost to dimensionless cost (c)
    c = cost_real * math.sqrt(2.0 * kappa) / sigma
    
    # 2. Adjust c based on the chosen strategy rule
    eff_c = (c / 2.0) if rule == 'new' else c
    
    # 3. Define the objective function for the root solver
    def objective_function(a):
        return evaluate_ou_series(a, eff_c)
        
    # 4. Find the root using Brent's method
    # The optimal 'a' must be greater than the effective cost.
    # We provide a bracket [eff_c, eff_c + 10.0] where the function changes signs.
    try:
        # Avoid setting the lower bracket exactly to eff_c to prevent division/evaluation edge cases
        lower_bound = eff_c + 1e-5
        upper_bound = max(eff_c + 10.0, 5.0) 
        
        result = root_scalar(objective_function, bracket=[lower_bound, upper_bound], method='brentq')
        
        if result.converged:
            return result.root
        else:
            raise RuntimeError("Root solver failed to converge.")
            
    except ValueError as e:
        print(f"Failed to find a sign change in the bracket: {e}")
        return max(1.0, eff_c + 0.5) # Fallback safety threshold

class PairAgent:
    __slots__ = ("spec", "filt", "current_z", "current_innov", "side", "nt_seen", "spread_hist", "current_kappa", "current_sigma", "current_theta", "a_star")

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
        self.current_theta = 0.0
        self.a_star = 1.0

    def _update_dynamic_ou(self):
        min_samples = max(2, self.spec.ou.ou_window)
        if len(self.spread_hist) < min_samples:
            return

        s = np.array(self.spread_hist[-min_samples:])
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
            theta = a_mle / (1.0 - b_mle)

            kappa_lambda = 1 - self.spec.ou.kappa_ewma_alpha
            sigma_lambda = 1 - self.spec.ou.sigma_ewma_alpha

            self.current_kappa = (kappa_lambda * self.current_kappa) + ((1.0 - kappa_lambda) * raw_kappa)
            self.current_sigma = (sigma_lambda * self.current_sigma) + ((1.0 - sigma_lambda) * raw_sigma)
            self.current_theta = (kappa_lambda * self.current_theta) + ((1.0 - kappa_lambda) * theta)

            self.a_star = get_optimal_z_entry(self.current_kappa, self.current_sigma, TRANSACTION_COST * 2, rule=Z_ENTRY_RULE)

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
            
            # 1. Update Kappa, Sigma, and current_a_star
            self._update_dynamic_ou()

            if self.current_kappa > 0:
                # Calculate the unconditional standard deviation of the OU process
                ou_std = self.current_sigma / math.sqrt(2.0 * self.current_kappa)
                # Get the latest raw spread from the Kalman filter step
                current_spread = self.spread_hist[-1] 
                # Calculate the true dimensionless spread Z-score
                spread_z = (current_spread - self.current_theta) / max(ou_std, 1e-8)
            else:
                spread_z = 0.
            
            # 2. State Machine Transitions
            # 'new' rule: hold until full reversion to the opposite threshold (flip live).
            # 'conventional' rule: exit at the mean - a live position goes flat the moment
            # spread_z crosses 0, and re-entry then follows the flat-side rules.
            if self.side == 0:
                if spread_z >= self.a_star:
                    self.side = -1  # Short entry
                elif spread_z <= -self.a_star:
                    self.side = 1   # Long entry

            elif self.side == 1:    # Currently LONG
                if Z_ENTRY_RULE == "conventional":
                    if spread_z >= 0.0:
                        self.side = 0
                elif spread_z >= self.a_star:
                    self.side = -1  # Flip to SHORT live

            elif self.side == -1:   # Currently SHORT
                if Z_ENTRY_RULE == "conventional":
                    if spread_z <= 0.0:
                        self.side = 0
                elif spread_z <= -self.a_star:
                    self.side = 1   # Flip to LONG live

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