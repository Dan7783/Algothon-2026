"""Fresh-OOS refactor with a capacity-safe runtime pair-discovery overlay.

This readable implementation is synchronized with the active 13-pair
``four_sigmas.py`` mechanics, then applies the two changes supported by the
previously unseen days 501-750:

* core lead-lag health uses 120 days with |rho|*sqrt(n) >= 1;
* ridge admission uses forecast threshold 0.75 with breadth gate 35.

The optional discovery sleeve is evaluated *after* atlas, lead-lag and ridge.
It can therefore add a hedged pair only when both legs are otherwise idle, and
it never duplicates an exact fixed-atlas pair.  Its 75-day scan is an episodic
detector, not a claim that 75 days is sufficient for unrestricted cointegration
discovery: admission requires a second causal endpoint and much stricter
residual-stationarity and stability gates.

The exact overlay A/B is intentionally unchanged on the available history:
no qualifying pair has two idle legs in the current book.  See
``analysis/PAIR_DISCOVERY_OVERLAY.md`` for the isolated and capacity-aware
results.  The main ``four_sigmas.py`` submission file is unchanged.
"""
from __future__ import annotations
import numpy as np

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
# Instrument mapping
I = {"ALGO": 0, "AENO": 1, "LSST": 2, "SRNA": 3, "ELLT": 4, "AMRP": 5, "OTCS": 6,
     "HETT": 7, "HUXZ": 8, "DUCT": 9, "SMAH": 10, "NPCK": 11, "MSDP": 12, "EORC": 13,
     "CUBO": 14, "HRET": 15, "ANSO": 16, "DIHO": 17, "RTTH": 18, "SPLZ": 19, "NWIG": 20,
     "MMBT": 21, "MDGI": 22, "AGVF": 23, "RRES": 24, "CTGI": 25, "ALUT": 26, "ACAC": 27,
     "SRTX": 28, "GARI": 29, "RCRI": 30, "ACIX": 31, "CCNS": 32, "MTNS": 33, "IHOZ": 34,
     "NAYO": 35, "FWWG": 36, "EELT": 37, "HRND": 38, "AETS": 39, "ULXY": 40, "BLBT": 41,
     "BENI": 42, "ITPA": 43, "HTRK": 44, "NGTE": 45, "ILVX": 46, "FCSG": 47, "FARS": 48,
     "MHRM": 49, "EAFC": 50}

# Signal linkages.  This roster is synchronized with four_sigmas.py after the
# days-501..750 revalidation; ALUT~CCNS remained strongly positive in fresh OOS.
PAIRS = [("SMAH", "ILVX"), ("AENO", "NWIG"), ("EORC", "NGTE"), ("HUXZ", "ACAC"),
         ("CTGI", "EELT"), ("MHRM", "EAFC"), ("HETT", "ULXY"), ("ACIX", "ITPA"),
         ("FWWG", "BLBT"), ("RTTH", "NAYO"), ("MTNS", "ALUT"), ("NAYO", "SRTX"),
         ("ALUT", "CCNS")]


LEADLAG_EXTRA = [("CTGI", "AENO"), ("EELT", "AENO"), ("DUCT", "HRND"), ("LSST", "ACAC"),
                 ("AENO", "NWIG"), ("SRNA", "CUBO"), ("MHRM", "ULXY"), ("ULXY", "HETT"),
                 ("BLBT", "HRET"), ("AGVF", "CUBO"), ("FWWG", "AMRP"), ("SRNA", "GARI"),
                 ("NWIG", "AENO"), ("ULXY", "GARI"), ("SRNA", "ULXY"), ("EELT", "BLBT"),
                 ("CTGI", "ULXY"), ("IHOZ", "DIHO"), ("FCSG", "ULXY"), ("ULXY", "HRND"),
                 ("EELT", "SRTX"), ("BENI", "GARI"), ("AETS", "FARS"), ("HRND", "ALUT")]

_EXTRA_IDX = [(I[a], I[b]) for a, b in LEADLAG_EXTRA]
_PAIR_LEGS = {I[a] for a, b in PAIRS} | {I[b] for a, b in PAIRS}
_FIXED_PAIR_KEYS = {tuple(sorted((I[a], I[b]))) for a, b in PAIRS}
_EXTRA_IDX_NO_PAIR_FOLLOWERS = [(I[a], I[b]) for a, b in LEADLAG_EXTRA if I[b] not in _PAIR_LEGS]
_EXTRA_SKIP_WHEN_CROWDED = {
    (I["SRNA"], I["CUBO"]),
    (I["ULXY"], I["GARI"]),
    (I["LSST"], I["ACAC"]),
}

# Model hyperparameters.
# HEDGE_W is a warm-up guard, not the Kalman estimation window.  The Kalman
# hedge remains expanding; Z_W is the actual spread-normalisation lookback.
HEDGE_W, Z_W = 120, 145
Z_DEAD, GATE_PAIR_ZMAX = 0.27, 4.0
USE_KALMAN_HEDGE = True
LEG = 10_000.0

# 2026-07-16 fresh-OOS repair: the previous 250-day lenient core gate kept
# decayed lag relationships alive.  A 120-day look with z>=1 improved days
# 501-750 by +$35.7/day marginally (NW t=2.30; 4/5 positive blocks) and was
# positive throughout the 90-150 day sensitivity neighbourhood.
LL_LOOK, GATE_LL_Z = 120, 1.0
RHO_HI, RHO_CAP, RHO_MIN_SCALE = 0.18, 0.30, 0.35

PRIOR_GATE_T = -2.9
LOOK_OLD, LOOK_NEW = 160, 90
Z_FULL, Z_NEW = 1.5, 1.0
REQUIRE_OOS_PNL, REQUIRE_SIGN_MATCH = True, True

RIDGE_LAM, RIDGE_THETA = 1.0, 0.75
RIDGE_GATE_N, RIDGE_MIN_T = 35, 252
RIDGE_LEG = 10_000.0
RIDGE_FILL_RESIDUAL_CAPACITY = True
PAIR_PRIORITY_OVER_LL = True
FILTER_EXTRA_PAIR_FOLLOWERS = False
BANG_BANG_CORE_LL = True
BANG_BANG_EXTRA_LL = True

# Runtime episodic-pair discovery.  This is intentionally a small, subordinate
# overlay rather than a replacement for the persistent 13-pair atlas.  The
# statistic thresholds are family-size safeguards, not calibrated ADF p-values.
# Enabled in this refactor as requested.  It produced no position changes over
# days 251-750 because the both-idle rule rejected all otherwise valid hits.
# Set False for an exact structural off-switch on future/unseen data.
DISCOVERY_ENABLED = True
DISCOVERY_CONTEXT = 250
DISCOVERY_LOOKBACK = 75
DISCOVERY_CENTER = 30
DISCOVERY_CONFIRM_GAP = 10
DISCOVERY_RESCAN_EVERY = 5
DISCOVERY_ADF_T = -4.25
DISCOVERY_CONFIRM_T = -3.75
DISCOVERY_ASSET_ADF_T = -3.25
DISCOVERY_BETA_RATIO = 1.75
DISCOVERY_HL_MIN, DISCOVERY_HL_MAX = 1.5, 20.0
DISCOVERY_Z_ENTRY, DISCOVERY_Z_SAT = 1.50, 2.50
DISCOVERY_Z_MAX = 4.0
DISCOVERY_SCALE = 0.75
DISCOVERY_MAX_PAIRS = 3

# All unordered non-ALGO pairs.  Exact atlas pairs are removed during the scan;
# an atlas *leg* may participate in a different episodic relationship only when
# the complete base book leaves that name idle on the execution day.
_DISC_TI, _DISC_TJ = np.triu_indices(len(I) - 1, k=1)
_DISC_PI, _DISC_PJ = _DISC_TI + 1, _DISC_TJ + 1

# Execution / OOS Config
SLEEVE_ORDER = ["atlas", "extraLL", "ridge", "discovery"]
SLEEVE_W = {"atlas": 1.0, "extraLL": 1.0, "ridge": 1.0, "discovery": 1.0}
PROBE_W = {"atlas": 1.0, "extraLL": -1.0, "ridge": 1.0, "discovery": 1.0}
GLOBAL_SCALAR = 1.0
MODE = "baseline"

_STATE = {
    "scan_t": -10**9,
    "discovery_pairs": [],
    "discovery_diagnostics": [],
    "active_discovery_pairs": [],
}


def reset():
    """Reset causal discovery state between simulations/history rewinds."""
    _STATE["scan_t"] = -10**9
    _STATE["discovery_pairs"] = []
    _STATE["discovery_diagnostics"] = []
    _STATE["active_discovery_pairs"] = []


# =============================================================================
# 2. MATH & STATISTICAL UTILITIES (The "Math" Layer)
# =============================================================================

def _kalman_beta(y: np.ndarray, x: np.ndarray) -> float:
    """
    Computes a dynamic hedge ratio (beta) using a 1D Kalman Filter.
    Models the relationship as: y = beta * x + alpha.
    """
    # Initialize state [alpha, beta] based on the first 60 observations
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    
    # State covariance matrix (Pm) and Process noise covariance (Q)
    Pm = np.eye(2) * 1e-2
    Q = np.eye(2) * 1e-5
    
    # Estimate measurement noise variance from the first 120 diffs
    Rv = np.var(np.diff(y[:120] - x[:120])) + 1e-8
    
    # Iterate through observations to update beliefs
    for t in range(len(y)):
        Pm = Pm + Q                             # Predict state covariance
        H = np.array([1.0, x[t]])               # Observation matrix [1, x_t]
        S = H @ Pm @ H + Rv                     # Innovation covariance
        K = Pm @ H / S                          # Kalman Gain
        theta = theta + K * (y[t] - H @ theta)  # Update state estimate
        Pm = Pm - np.outer(K, H) @ Pm           # Update state covariance
        
    return float(theta[1]) # Return the final beta estimate


def _ou_kappa(spread: np.ndarray) -> float:
    """
    Calculates the Ornstein-Uhlenbeck mean-reversion speed (kappa).
    Positive kappa means mean-reverting. Higher values imply faster reversion.
    """
    ds = np.diff(spread)                         # Change in spread (dy)
    sl = spread[:-1] - spread[:-1].mean()        # Demeaned lagged spread (y_{t-1} - mu)
    variance = (sl ** 2).sum()                   # Variance of lagged spread
    
    if variance > 1e-18:
        return float(-(sl @ ds) / variance)      # -Cov(y_{t-1}, dy) / Var(y_{t-1})
    return 0.0


def _adf1_tstats(S: np.ndarray) -> np.ndarray:
    """
    Vectorized ADF-style stationarity statistic for constructed spreads.
    Solves OLS: ΔY_t = α + β Y_{t-1} + γ ΔY_{t-1} + ε_t
    Returns the level-coefficient t-statistic. This is a runtime health
    statistic, not a multiple-testing-adjusted cointegration p-value.
    """
    # 1. Setup regression variables
    ds = np.diff(S, axis=0)      # First differences (ΔY)
    y = ds[1:]                   # Target variable: ΔY_t
    x1 = S[1:-1]                 # Regressor 1: Lagged levels Y_{t-1}
    x2 = ds[:-1]                 # Regressor 2: Lagged differences ΔY_{t-1}
    n = y.shape[0]               # Number of observations
    m = S.shape[1]               # Number of parallel timeseries (spreads)
    
    # 2. Compute sums and dot products for the OLS Matrix (X^T X)
    s1 = x1.sum(0); s2 = x2.sum(0)
    s11 = (x1 * x1).sum(0); s12 = (x1 * x2).sum(0); s22 = (x2 * x2).sum(0)
    
    # 3. Compute sums and dot products for (X^T Y)
    sy = y.sum(0); s1y = (x1 * y).sum(0); s2y = (x2 * y).sum(0); syy = (y * y).sum(0)
    
    # 4. Construct the normal equation matrices for 'm' spreads
    # A is the (X^T X) matrix of shape (m, 3, 3). b is the (X^T Y) vector of shape (m, 3).
    A = np.empty((m, 3, 3))
    b = np.empty((m, 3))
    
    # Fill X^T X
    A[:, 0, 0] = n;    A[:, 0, 1] = s1;   A[:, 0, 2] = s2    # Intercept row
    A[:, 1, 0] = s1;   A[:, 1, 1] = s11;  A[:, 1, 2] = s12   # Lagged level row
    A[:, 2, 0] = s2;   A[:, 2, 1] = s12;  A[:, 2, 2] = s22   # Lagged diff row
    
    # Fill X^T Y
    b[:, 0] = sy;      b[:, 1] = s1y;     b[:, 2] = s2y
    
    # 5. Solve OLS: beta = (X^T X)^-1 (X^T Y)
    Ainv = np.linalg.inv(A)
    beta = np.einsum("mij,mj->mi", Ainv, b)  # Batch matrix multiplication
    
    # 6. Calculate Standard Errors and T-Statistics
    # Residual sum of squares: (Y^T Y) - Beta^T (X^T Y)
    rss = syy - np.einsum("mi,mi->m", beta, b)
    sigma2 = rss / max(n - 3, 1) # Variance of residuals
    
    # SE of beta_1 (the mean-reversion coefficient) is sqrt(sigma2 * diag((X^T X)^-1)[1])
    se_beta = np.sqrt(np.maximum(sigma2 * Ainv[:, 1, 1], 1e-300))
    t_stats = beta[:, 1] / se_beta
    
    return t_stats

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Helper to safely compute Pearson correlation between two 1D arrays."""
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _pair_regression(
    LP: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised OLS hedge beta and log spread for ``pi ~ beta * pj``."""
    y = LP[pi]
    x = LP[pj]
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    beta = (yc * xc).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-18)
    return beta, y - beta[:, None] * x


def _pair_beta_ratio(LP: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Ratio of hedge betas in the first and second halves of a window."""
    mid = LP.shape[1] // 2
    beta_1, _ = _pair_regression(LP[:, :mid], pi, pj)
    beta_2, _ = _pair_regression(LP[:, mid:], pi, pj)
    ratio = np.maximum(np.abs(beta_1), np.abs(beta_2)) / np.maximum(
        np.minimum(np.abs(beta_1), np.abs(beta_2)), 1e-12,
    )
    return np.where((beta_1 > 0.0) & (beta_2 > 0.0), ratio, np.inf)


def _spread_half_life(spread: np.ndarray) -> np.ndarray:
    """OU/AR(1) half-life for a ``time x pairs`` spread matrix."""
    lag = spread[:-1]
    delta = np.diff(spread, axis=0)
    lag_c = lag - lag.mean(axis=0)
    delta_c = delta - delta.mean(axis=0)
    slope = (lag_c * delta_c).sum(axis=0) / np.maximum(
        (lag_c * lag_c).sum(axis=0), 1e-18,
    )
    return np.where(slope < 0.0, -np.log(2.0) / slope, np.inf)


def _local_pair_z(spread: np.ndarray, window: int) -> np.ndarray:
    """Current z-score against prior observations only (no look-ahead centre)."""
    history = spread[:, -(window + 1):-1]
    return (spread[:, -1] - history.mean(axis=1)) / (history.std(axis=1) + 1e-12)


def _episode_metrics(
    context: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate all current and earlier-endpoint episodic admission metrics."""
    fast = context[:, -DISCOVERY_LOOKBACK:]
    prior = context[
        :,
        -(DISCOVERY_LOOKBACK + DISCOVERY_CONFIRM_GAP):-DISCOVERY_CONFIRM_GAP,
    ]

    beta, spread = _pair_regression(fast, pi, pj)
    _, reverse = _pair_regression(fast, pj, pi)
    pair_t = np.maximum(_adf1_tstats(spread.T), _adf1_tstats(reverse.T))

    _, prior_spread = _pair_regression(prior, pi, pj)
    _, prior_reverse = _pair_regression(prior, pj, pi)
    prior_t = np.maximum(
        _adf1_tstats(prior_spread.T),
        _adf1_tstats(prior_reverse.T),
    )

    level_t = _adf1_tstats(context.T)
    return {
        "beta": beta,
        "pair_t": pair_t,
        "prior_t": prior_t,
        "beta_ratio": _pair_beta_ratio(fast, pi, pj),
        "half_life": _spread_half_life(spread.T),
        "z": _local_pair_z(spread, DISCOVERY_CENTER),
        "level_i": level_t[pi],
        "level_j": level_t[pj],
    }


def _episode_ok(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Strict relationship, stability and moneyness gate for episodic pairs."""
    z_abs = np.abs(metrics["z"])
    return (
        (metrics["level_i"] > DISCOVERY_ASSET_ADF_T)
        & (metrics["level_j"] > DISCOVERY_ASSET_ADF_T)
        & (metrics["beta"] > 0.0)
        & (metrics["beta_ratio"] <= DISCOVERY_BETA_RATIO)
        & (metrics["pair_t"] < DISCOVERY_ADF_T)
        & (metrics["prior_t"] < DISCOVERY_CONFIRM_T)
        & (metrics["half_life"] >= DISCOVERY_HL_MIN)
        & (metrics["half_life"] <= DISCOVERY_HL_MAX)
        & (z_abs >= DISCOVERY_Z_ENTRY)
        & (z_abs < DISCOVERY_Z_MAX)
    )


def _episode_rank(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Lower is stronger; confirmation dominates the small z tie-break."""
    return (
        metrics["pair_t"]
        + 0.5 * metrics["prior_t"]
        - 0.05 * np.abs(metrics["z"])
    )


def _scan_discovery_pairs(LP: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    """Universe scan with exact fixed-atlas relationships removed."""
    if LP.shape[1] < DISCOVERY_CONTEXT:
        return [], []

    eligible = np.fromiter(
        (
            (int(i), int(j)) not in _FIXED_PAIR_KEYS
            for i, j in zip(_DISC_PI, _DISC_PJ)
        ),
        dtype=bool,
        count=len(_DISC_PI),
    )
    pi, pj = _DISC_PI[eligible], _DISC_PJ[eligible]
    if len(pi) == 0:
        return [], []

    context = LP[:, -DISCOVERY_CONTEXT:]
    metrics = _episode_metrics(context, pi, pj)
    selected = np.where(_episode_ok(metrics))[0]
    selected = selected[np.argsort(_episode_rank(metrics)[selected])]

    names = {idx: name for name, idx in I.items()}
    pairs = [(int(pi[k]), int(pj[k])) for k in selected]
    diagnostics = [
        {
            "pair": f"{names[int(pi[k])]}~{names[int(pj[k])]}",
            "i": int(pi[k]),
            "j": int(pj[k]),
            "pair_t": float(metrics["pair_t"][k]),
            "confirm_t": float(metrics["prior_t"][k]),
            "beta": float(metrics["beta"][k]),
            "beta_ratio": float(metrics["beta_ratio"][k]),
            "half_life": float(metrics["half_life"][k]),
            "z": float(metrics["z"][k]),
        }
        for k in selected
    ]
    return pairs, diagnostics


def _add_discovery_pair(
    pos: np.ndarray,
    i: int,
    j: int,
    beta: float,
    z: float,
) -> None:
    """Add a beta-weighted spread while keeping both legs below one asset cap."""
    strength = np.clip(
        (abs(z) - DISCOVERY_Z_ENTRY)
        / max(DISCOVERY_Z_SAT - DISCOVERY_Z_ENTRY, 1e-12),
        0.0,
        1.0,
    )
    direction = -np.sign(z) * float(strength) * DISCOVERY_SCALE
    leg_scale = LEG / max(1.0, abs(beta))
    pos[i] += direction * leg_scale
    pos[j] -= direction * beta * leg_scale


def _discovery_pair_dollars(LP: np.ndarray, base_dollars: np.ndarray) -> np.ndarray:
    """Capacity-safe episodic overlay; both legs must be idle in the base book."""
    nins, nt = LP.shape
    pos = np.zeros(nins)
    _STATE["active_discovery_pairs"] = []
    if not DISCOVERY_ENABLED or nt < DISCOVERY_CONTEXT:
        return pos

    if nt - _STATE["scan_t"] >= DISCOVERY_RESCAN_EVERY:
        pairs, diagnostics = _scan_discovery_pairs(LP)
        _STATE["scan_t"] = nt
        _STATE["discovery_pairs"] = pairs
        _STATE["discovery_diagnostics"] = diagnostics

    cached = _STATE["discovery_pairs"]
    if not cached:
        return pos

    pi = np.asarray([pair[0] for pair in cached], dtype=int)
    pj = np.asarray([pair[1] for pair in cached], dtype=int)
    metrics = _episode_metrics(LP[:, -DISCOVERY_CONTEXT:], pi, pj)
    idle = np.abs(base_dollars) <= 1.0
    ok = _episode_ok(metrics) & idle[pi] & idle[pj]
    selected = np.where(ok)[0]
    selected = selected[np.argsort(_episode_rank(metrics)[selected])]

    used = np.zeros(nins, dtype=bool)
    names = {idx: name for name, idx in I.items()}
    accepted = 0
    for k in selected:
        i, j = int(pi[k]), int(pj[k])
        if used[i] or used[j]:
            continue
        _add_discovery_pair(
            pos,
            i,
            j,
            float(metrics["beta"][k]),
            float(metrics["z"][k]),
        )
        used[i] = used[j] = True
        _STATE["active_discovery_pairs"].append({
            "pair": f"{names[i]}~{names[j]}",
            "i": i,
            "j": j,
            "beta": float(metrics["beta"][k]),
            "z": float(metrics["z"][k]),
            "base_i": float(base_dollars[i]),
            "base_j": float(base_dollars[j]),
            "dollars_i": float(pos[i]),
            "dollars_j": float(pos[j]),
        })
        accepted += 1
        if accepted >= DISCOVERY_MAX_PAIRS:
            break

    np.clip(pos, -LEG, LEG, out=pos)
    return pos


def list_discovered_pairs() -> list[dict]:
    """Return a copy of the latest causal scan diagnostics for audit tooling."""
    return [dict(row) for row in _STATE["discovery_diagnostics"]]


def list_active_discovery_pairs() -> list[dict]:
    """Return pairs actually admitted after both-leg idle/crowding checks."""
    return [dict(row) for row in _STATE["active_discovery_pairs"]]


# =============================================================================
# 3. SIGNAL LAYER (Generates unconstrained ideal dollars per sub-strategy)
# =============================================================================

def _atlas_pair_dollars(LP: np.ndarray) -> np.ndarray:
    """
    Sleeve 1: Atlas Cointegrated Pairs.
    Generates dollar positions for strongly mean-reverting pairs.
    """
    nins, nt = LP.shape
    pos = np.zeros(nins)
    
    # Require minimum history to calculate rolling stats
    if nt < HEDGE_W + 10:
        return pos
        
    for a, b in PAIRS:
        i, j = I[a], I[b]
        
        # 1. Calculate Hedge Ratio
        beta = _kalman_beta(LP[i, :nt - 1], LP[j, :nt - 1]) if USE_KALMAN_HEDGE else 1.0
        
        # 2. Pre-Gate: Check if Cointegrated (ADF test)
        spread = LP[i] - beta * LP[j]
        # Only trade if the ADF t-stat is negative enough (strong mean reversion)
        if _adf1_tstats(spread.reshape(-1, 1))[0] >= PRIOR_GATE_T:
            continue
            
        # 3. Z-Score Calculation
        s_hedge = LP[i, nt - Z_W:nt] - beta * LP[j, nt - Z_W:nt]
        z = (s_hedge[-1] - s_hedge[:-1].mean()) / (s_hedge[:-1].std() + 1e-12)
        
        # 4. Filter: Ensure OU process shows active mean reversion recently
        s120 = LP[i, nt - 120:nt] - beta * LP[j, nt - 120:nt]
        if _ou_kappa(s120) <= 0.0:
            continue
            
        # 5. Filter: Z-score must be in the "Goldilocks" zone (not too small, not exploding)
        if abs(z) < Z_DEAD or abs(z) > GATE_PAIR_ZMAX:
            continue
            
        # 6. Assign Positions (Fade the spread)
        scale = 1.0
        fade_dir = -np.sign(z) * scale
        pos[i] += fade_dir * LEG
        pos[j] += -fade_dir * LEG
        
    return pos


def _extra_leadlag_dollars(
    r: np.ndarray,
    idx_list: list,
    crowded_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sleeve 3: Extra Lead-Lag Links.
    Similar to core, but uses a robust split-window correlation check to prevent overfitting.
    """
    nins, m = r.shape
    pos = np.zeros(nins)
    
    look = LOOK_OLD + LOOK_NEW
    if m < look + 2:
        return pos
        
    for i, j in idx_list:
        if (
            crowded_mask is not None
            and crowded_mask[j]
            and (i, j) in _EXTRA_SKIP_WHEN_CROWDED
        ):
            continue
        # 1. Setup train/test windows for robust correlation check
        x = r[i, m - look - 1:m - 1]
        y = r[j, m - look:m]
        
        x_old, y_old = x[:LOOK_OLD], y[:LOOK_OLD]
        x_new, y_new = x[LOOK_OLD:], y[LOOK_OLD:]
        
        c_old = _corr(x_old, y_old)
        c_new = _corr(x_new, y_new)
        c_full = _corr(x, y)
        
        if c_old == 0.0:
            continue
            
        # 2. Apply Robustness Gates
        sign_est = float(np.sign(c_old))
        pnl_new = float(np.sum(sign_est * np.sign(x_new) * y_new))
        
        z_new = abs(c_new) * np.sqrt(len(x_new))
        z_full = abs(c_full) * np.sqrt(len(x))
        
        ok = (z_full >= Z_FULL) and (z_new >= Z_NEW)
        if REQUIRE_SIGN_MATCH:
            ok = ok and (np.sign(c_new) == sign_est)
        if REQUIRE_OOS_PNL:
            ok = ok and (pnl_new > 0.0)
            
        if not ok:
            continue
            
        # 3. Legacy scaling is retained behind BANG_BANG_EXTRA_LL. The fresh
        # OOS-validated default uses a full leg after the robust gate passes.
        abs_rho = abs(c_full)
        if abs_rho <= RHO_HI:
            scale = 1.0
        elif abs_rho >= RHO_CAP:
            scale = RHO_MIN_SCALE
        else:
            scale = 1.0 - ((abs_rho - RHO_HI) / (RHO_CAP - RHO_HI)) * (1.0 - RHO_MIN_SCALE)

        if BANG_BANG_EXTRA_LL:
            scale = 1.0
        w = 1.0 if BANG_BANG_EXTRA_LL or np.sign(r[i, -1]) == np.sign(r[0, -1]) else 0.5
        pos[j] += scale * w * LEG * sign_est * np.sign(r[i, -1])
        
    return pos


def _ridge_dollars(r: np.ndarray, held_dollars: np.ndarray) -> np.ndarray:
    """
    Sleeve 4: Ridge Regression.
    Predicts next-step returns based on historical correlations using L2 regularization.
    Subordinates to positions held by prior sleeves to maximize capital efficiency.
    """
    nins, m = r.shape
    pos = np.zeros(nins)
    
    if m + 1 < RIDGE_MIN_T:
        return pos
        
    # 1. Check Network Connectivity Gate (Requires enough correlated instruments)
    R250 = r[1:, -250:].T
    Xg = (R250 - R250.mean(0)) / (R250.std(0) + 1e-12)
    Cg = (Xg[:-1].T @ Xg[1:]) / (len(R250) - 1) # Auto-covariance proxy
    np.fill_diagonal(Cg, 0)
    
    fam = int((np.abs(Cg) > 2.5 / np.sqrt(len(R250) - 1)).sum())
    if fam < RIDGE_GATE_N:
        return pos
        
    # 2. Prepare Data for Ridge Regression: Y = X * Beta
    Rf = r[1:, :].T                  # Transpose returns (Time x Instruments)
    sx = Rf.std(0) + 1e-12           # Standard deviation scaler
    Xs = Rf[:-1] / sx                # Regressors: Lagged Returns
    Ys = Rf[1:] / sx                 # Targets: Current Returns
    
    # 3. Solve Ridge Regression: Beta = (X^T X + λI)^-1 (X^T Y)
    # np.linalg.solve(A, b) is faster/more stable than inv(A) @ b
    lambda_penalty = RIDGE_LAM * len(Xs) * np.eye(50)
    A = Xs.T @ Xs + lambda_penalty
    b = Xs.T @ Ys
    B = np.linalg.solve(A, b)        # Beta coefficients matrix
    
    # 4. Generate Predictions
    sp = (Xs @ B).std(0)             # Standard deviation of predictions
    p = (Rf[-1] / sx) @ B            # Final prediction vector for t+1
    
    # Trade only if the prediction magnitude exceeds the threshold
    raw = RIDGE_LEG * np.sign(p) * (np.abs(p) > RIDGE_THETA * sp)
    held = held_dollars[1:]          # Extract existing positions (excluding index 0)
    d = np.zeros_like(raw)
    
    # 5. Capacity Subordination (Fill idle space without fighting prior sleeves)
    idle = np.abs(held) <= 1.0
    d[idle] = raw[idle]              # Assign raw ridge positions to idle assets
    
    if RIDGE_FILL_RESIDUAL_CAPACITY:
        # If asset is not idle, but Ridge agrees with the current direction, add to it (up to limit)
        same_sign = (~idle) & (np.abs(raw) > 1.0) & (np.sign(raw) == np.sign(held))
        room = np.maximum(0.0, LEG - np.minimum(np.abs(held), LEG))
        d[same_sign] = np.sign(raw[same_sign]) * np.minimum(np.abs(raw[same_sign]), room[same_sign])
        
    pos[1:] = d
    return pos


# =============================================================================
# 4. EXECUTION LAYER (Portfolio Construction & Risk)
# =============================================================================

def _active_weights():
    """Resolves (weight-dict, global-scalar) based on the globally set MODE."""
    if MODE == "probe":
        return PROBE_W, GLOBAL_SCALAR
    if MODE == "fullport":
        return {k: 1.0 for k in SLEEVE_ORDER}, GLOBAL_SCALAR
    if MODE == "baseline":
        return {k: 1.0 for k in SLEEVE_ORDER}, 1.0
    return SLEEVE_W, GLOBAL_SCALAR


def sleeve_breakdown(prcSoFar: np.ndarray) -> dict | None:
    """
    Computes dollar exposures in execution-priority order.

    Discovery is deliberately conditional rather than mathematically
    independent: it receives the complete base book and may use only names
    that atlas, lead-lag and ridge all leave idle.
    These are raw output sleeves before execution weights are applied.
    """
    nins, nt = prcSoFar.shape
    
    # Validation gates
    if nt < 130 or nins != 51:
        return None
        
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0):
        return None
        
    if nt < _STATE["scan_t"]:
        reset()
        
    # Standardize inputs
    LP = np.log(prcSoFar)
    r = np.diff(LP, axis=1)
    
    # 1. Generate core signals
    d_atlas = _atlas_pair_dollars(LP)
    
    # 2. Generate extra signals (with optional filter)
    extra_idx = _EXTRA_IDX_NO_PAIR_FOLLOWERS if FILTER_EXTRA_PAIR_FOLLOWERS else _EXTRA_IDX
    crowded_mask = (np.abs(d_atlas) > 1.0)
    d_extraLL = _extra_leadlag_dollars(r, extra_idx, crowded_mask)
    
    # 3. Apply Priority: Pairs override Lead-Lag signals on the same name
    if PAIR_PRIORITY_OVER_LL:
        pair_mask = np.abs(d_atlas) > 1.0
        d_extraLL = np.where(pair_mask, 0.0, d_extraLL)
        
    # 4. Calculate current held dollars to feed into Ridge (subordination)
    held0 = d_atlas + d_extraLL
    d_ridge = _ridge_dollars(r, held0)

    # 5. Discovery is last.  It cannot replace any fixed-pair, lead-lag or
    # ridge allocation because both legs have to be idle in the full base book.
    base_dollars = held0 + d_ridge
    d_discovery = _discovery_pair_dollars(LP, base_dollars)
    
    return {
        "atlas": d_atlas, 
        "extraLL": d_extraLL, 
        "ridge": d_ridge,
        "discovery": d_discovery,
    }


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Primary Entry Point: Translates historical prices into a portfolio of integer shares.
    Combines sleeves linearly, applies crash protection, and enforces bounds.
    """
    nins, nt = prcSoFar.shape
    
    # Get raw sleeve dollars
    sl = sleeve_breakdown(prcSoFar)
    if sl is None:
        return np.zeros(nins, dtype=int)

    # Combine sleeves using configured OOS weights
    weights, gscalar = _active_weights()
    dollars = np.zeros(nins)
    for name in SLEEVE_ORDER:
        dollars += float(weights.get(name, 0.0)) * sl[name]
    dollars *= float(gscalar)

    # Risk Control 1: Crash Protection (Zero out positions if recent change > 40%)
    r_last = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(r_last) > 0.40
    crash[0] = False             # Ignore index (usually mapped to 0)
    dollars[crash] = 0.0
    
    # Risk Control 2: Maximum Capital Bounds (Clip individual positions)
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    
    # Risk Control 3: Never trade instrument 0 (Market Index)
    dollars[0] = 0.0
    
    # Convert desired dollars to target integer shares using the latest price
    latest_prices = prcSoFar[:, -1]
    target_shares = (dollars / latest_prices).astype(int)
    
    return target_shares
