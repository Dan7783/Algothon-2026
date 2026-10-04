"""Four Sigmas HYBRID + aggressive unused-capacity overlay (2026-07-12).

Self-contained (NumPy only). Book:
  - 11 atlas Kalman pairs + ADF prior gate + capped ADF discovery
  - 6 core lead-lag on the LENIENT gate (|ρ|·√n ≥ 1)
  - 24 EXTRA lead-lag on the ROBUST split-window OOS gate
  - subordinated ridge VAR(1) θ=1 + crash gate
  - three-signal overlay on instruments left unused by the completed core

Priority rule: the hybrid core always wins. The overlay can trade an instrument
only when the final core position for that instrument is zero shares.

Visible-data exact evaluator (days 250-500): score 807.98, mean daily PnL
$826.0, annualised Sharpe 6.70, cumulative PnL $207,322.42.

Eval: from four_sigmas import getMyPosition
"""
from __future__ import annotations
import numpy as np

I = {"ALGO": 0, "AENO": 1, "LSST": 2, "SRNA": 3, "ELLT": 4, "AMRP": 5, "OTCS": 6,
     "HETT": 7, "HUXZ": 8, "DUCT": 9, "SMAH": 10, "NPCK": 11, "MSDP": 12, "EORC": 13,
     "CUBO": 14, "HRET": 15, "ANSO": 16, "DIHO": 17, "RTTH": 18, "SPLZ": 19, "NWIG": 20,
     "MMBT": 21, "MDGI": 22, "AGVF": 23, "RRES": 24, "CTGI": 25, "ALUT": 26, "ACAC": 27,
     "SRTX": 28, "GARI": 29, "RCRI": 30, "ACIX": 31, "CCNS": 32, "MTNS": 33, "IHOZ": 34,
     "NAYO": 35, "FWWG": 36, "EELT": 37, "HRND": 38, "AETS": 39, "ULXY": 40, "BLBT": 41,
     "BENI": 42, "ITPA": 43, "HTRK": 44, "NGTE": 45, "ILVX": 46, "FCSG": 47, "FARS": 48,
     "MHRM": 49, "EAFC": 50}

PAIRS = [("SMAH", "ILVX"), ("AENO", "NWIG"), ("EORC", "NGTE"), ("HUXZ", "ACAC"),
         ("CTGI", "EELT"), ("MHRM", "EAFC"), ("HETT", "ULXY"), ("ACIX", "ITPA"),
         ("FWWG", "BLBT"), ("RTTH", "NAYO"), ("MTNS", "ALUT")]

LEADLAG = [("DUCT", "HRET"), ("DUCT", "AMRP"), ("ELLT", "ULXY"),
           ("CCNS", "CUBO"), ("ELLT", "GARI"), ("DIHO", "HRND")]

# alt's 30 minus the 6 core — robust-gated only
LEADLAG_EXTRA = [("CTGI", "AENO"), ("EELT", "AENO"), ("DUCT", "HRND"), ("LSST", "ACAC"),
                 ("AENO", "NWIG"), ("SRNA", "CUBO"), ("MHRM", "ULXY"), ("ULXY", "HETT"),
                 ("BLBT", "HRET"), ("AGVF", "CUBO"), ("FWWG", "AMRP"), ("SRNA", "GARI"),
                 ("NWIG", "AENO"), ("ULXY", "GARI"), ("SRNA", "ULXY"), ("EELT", "BLBT"),
                 ("CTGI", "ULXY"), ("IHOZ", "DIHO"), ("FCSG", "ULXY"), ("ULXY", "HRND"),
                 ("EELT", "SRTX"), ("BENI", "GARI"), ("AETS", "FARS"), ("HRND", "ALUT")]

_EXTRA_IDX = [(I[a], I[b]) for a, b in LEADLAG_EXTRA]

HEDGE_W, Z_W = 120, 145
Z_DEAD = 0.28
USE_KALMAN_HEDGE = True
LEG = 10_000.0
LL_LOOK = 250
GATE_LL_Z = 1.0
GATE_PAIR_ZMAX = 4.0
RHO_HI, RHO_CAP, RHO_MIN_SCALE = 0.18, 0.30, 0.35

PRIOR_GATE_T = -2.9
RESCAN_EVERY = 5
FULL_T = -4.3
SOFT_T = -3.9
MAX_DISC = 3
DISC_MIN_T = 120

RIDGE_LAM = 1.0
RIDGE_THETA = 1.0
RIDGE_GATE_N = 45
RIDGE_LEG = 10_000.0
RIDGE_MIN_T = 252

# Robust LL gate (inlined from leadlag_robust)
LOOK_OLD = 160
LOOK_NEW = 90
Z_FULL = 1.8
Z_NEW = 1.0
REQUIRE_OOS_PNL = True
REQUIRE_SIGN_MATCH = True

_STATE = {"scan_t": -10**9, "disc": []}


def reset():
    _STATE["scan_t"] = -10**9
    _STATE["disc"] = []


def _kalman_beta(y, x):
    theta = np.array([y[:60].mean() - x[:60].mean(), 1.0])
    Pm = np.eye(2) * 1e-2
    Q = np.eye(2) * 1e-5
    Rv = np.var(np.diff(y[:120] - x[:120])) + 1e-8
    for t in range(len(y)):
        Pm = Pm + Q
        H = np.array([1.0, x[t]])
        S = H @ Pm @ H + Rv
        K = Pm @ H / S
        theta = theta + K * (y[t] - H @ theta)
        Pm = Pm - np.outer(K, H) @ Pm
    return float(theta[1])


def _ou_kappa(spread):
    ds = np.diff(spread)
    sl = spread[:-1] - spread[:-1].mean()
    v = (sl ** 2).sum()
    return float(-(sl @ ds) / v) if v > 1e-18 else 0.0


def _ll_overheat_scale(abs_rho):
    if abs_rho <= RHO_HI:
        return 1.0
    if abs_rho >= RHO_CAP:
        return RHO_MIN_SCALE
    w = (abs_rho - RHO_HI) / (RHO_CAP - RHO_HI)
    return 1.0 - w * (1.0 - RHO_MIN_SCALE)


def _adf1_tstats(S):
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


def _adf_t_pair(LP, i, j):
    s = (LP[i] - LP[j]).reshape(-1, 1)
    if s.shape[0] < 40:
        return 0.0
    return float(_adf1_tstats(s)[0])


def _detect_extra_pairs(LP, used_legs):
    LPT = LP.T
    stocks = LPT[:, 1:]
    if stocks.shape[0] < DISC_MIN_T:
        return []
    ii, jj = np.triu_indices(stocks.shape[1], 1)
    tstats = _adf1_tstats(stocks[:, ii] - stocks[:, jj])
    out = []
    used = set(used_legs)
    for k in np.argsort(tstats):
        if tstats[k] >= SOFT_T or len(out) >= MAX_DISC:
            break
        i, j = int(ii[k]) + 1, int(jj[k]) + 1
        if i in used or j in used:
            continue
        used.update((i, j))
        w = 1.0 if tstats[k] < FULL_T else 0.5
        out.append((i, j, w))
    return out


def _fade_pair(LP, i, j, scale=1.0):
    nins, nt = LP.shape
    if nt < HEDGE_W + 10:
        return 0.0, 0.0
    beta = _kalman_beta(LP[i, :nt - 1], LP[j, :nt - 1]) if USE_KALMAN_HEDGE else 1.0
    # Avoid negative-index wrapping during the early warm-up period.
    z_start = max(0, nt - Z_W)
    s = LP[i, z_start:nt] - beta * LP[j, z_start:nt]
    z = (s[-1] - s[:-1].mean()) / (s[:-1].std() + 1e-12)
    s120 = LP[i, nt - 120:nt] - beta * LP[j, nt - 120:nt]
    if _ou_kappa(s120) <= 0.0:
        return 0.0, 0.0
    if abs(z) < Z_DEAD or abs(z) > GATE_PAIR_ZMAX:
        return 0.0, 0.0
    f = -np.sign(z) * scale
    return f * LEG, -f * LEG


def _atlas_pair_dollars(LP):
    nins, nt = LP.shape
    pos = np.zeros(nins)
    if nt < HEDGE_W + 10:
        return pos
    for a, b in PAIRS:
        i, j = I[a], I[b]
        if _adf_t_pair(LP, i, j) >= PRIOR_GATE_T:
            continue
        di, dj = _fade_pair(LP, i, j, scale=1.0)
        pos[i] += di
        pos[j] += dj
    return pos


def _disc_pair_dollars(LP):
    nins, nt = LP.shape
    pos = np.zeros(nins)
    if nt < DISC_MIN_T:
        return pos
    if nt - _STATE["scan_t"] >= RESCAN_EVERY:
        used = {I[a] for p in PAIRS for a in p}
        _STATE["disc"] = _detect_extra_pairs(LP, used)
        _STATE["scan_t"] = nt
    for i, j, w in _STATE["disc"]:
        di, dj = _fade_pair(LP, i, j, scale=w)
        pos[i] += di
        pos[j] += dj
    return pos


def _ll_health(r, i, j, look=LL_LOOK):
    m = r.shape[1]
    look = min(look, m - 1)
    if look < 100:
        return None
    x = r[i, m - look - 1:m - 1]
    y = r[j, m - look:m]
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
    """6 core LL — lenient gate."""
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
        scale = _ll_overheat_scale(abs(c))
        w = 1.0 if np.sign(r[i, -1]) == np.sign(r[0, -1]) else 0.5
        pos[j] += scale * w * LEG * np.sign(c) * np.sign(r[i, -1])
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
    x = r[i, m - look - 1:m - 1]
    y = r[j, m - look:m]
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


def _extra_leadlag_dollars(r):
    """24 EXTRA LL — robust split-window OOS gate."""
    nins = r.shape[0]
    pos = np.zeros(nins)
    for i, j in _EXTRA_IDX:
        h = _robust_link_health(r, i, j)
        if h is None or not h["ok"]:
            continue
        w = 1.0 if np.sign(r[i, -1]) == np.sign(r[0, -1]) else 0.5
        scale = _ll_overheat_scale(abs(h["c_full"]))
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
    d = RIDGE_LEG * np.sign(p) * (np.abs(p) > RIDGE_THETA * sp)
    d[np.abs(held_dollars[1:]) > 1.0] = 0.0
    pos[1:] = d
    return pos


def _core_position(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < 130 or nins != 51:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0):
        return np.zeros(nins, dtype=int)

    if nt < _STATE["scan_t"]:
        reset()

    LP = np.log(prcSoFar)
    r = np.diff(LP, axis=1)
    crash = np.abs(r[:, -1]) > 0.40
    crash[0] = False

    dollars = (_atlas_pair_dollars(LP)
               + _disc_pair_dollars(LP)
               + _leadlag_dollars(r)
               + _extra_leadlag_dollars(r))
    dollars = dollars + _ridge_dollars(r, dollars)

    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)


# ---------------------------------------------------------------------------
# Aggressive unused-capacity overlay
# ---------------------------------------------------------------------------

OVERLAY_RIDGE_ALPHA = 200.0
OVERLAY_MIN_RIDGE_RETURNS = 80


def _overlay_scale(values):
    scale = float(np.median(np.abs(values)))
    return scale if np.isfinite(scale) and scale > 1e-12 else 1.0


def _overlay_reversion_20(log_prices):
    if log_prices.shape[1] <= 20:
        return np.zeros(log_prices.shape[0])
    move = log_prices[:, -1] - log_prices[:, -21]
    return -np.tanh(move / _overlay_scale(move))


def _overlay_price_z_5(log_prices):
    if log_prices.shape[1] < 3:
        return np.zeros(log_prices.shape[0])
    recent = log_prices[:, -min(5, log_prices.shape[1]):]
    mean = np.mean(recent, axis=1)
    std = np.maximum(np.std(recent, axis=1), 1e-12)
    z_score = (log_prices[:, -1] - mean) / std
    return -np.tanh(z_score / 2.0)


def _overlay_ridge_var(log_prices):
    """Expanding ridge VAR(1) estimated only from the supplied price prefix."""
    returns = np.diff(log_prices, axis=1).T
    n_returns, n_assets = returns.shape
    if n_returns < OVERLAY_MIN_RIDGE_RETURNS:
        return np.zeros(n_assets)

    x_train = returns[:-1]
    y_train = returns[1:]
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = np.mean(y_train, axis=0)

    x_standard = (x_train - x_mean) / x_std
    y_centered = y_train - y_mean
    regularised_cov = x_standard.T @ x_standard
    regularised_cov.flat[::n_assets + 1] += OVERLAY_RIDGE_ALPHA
    try:
        coefficients = np.linalg.solve(
            regularised_cov,
            x_standard.T @ y_centered,
        )
    except np.linalg.LinAlgError:
        return np.zeros(n_assets)

    latest_standard = (returns[-1] - x_mean) / x_std
    prediction = latest_standard @ coefficients + y_mean
    return np.tanh(prediction / _overlay_scale(prediction))


def _overlay_position(prcSoFar):
    prices = np.asarray(prcSoFar, dtype=float)
    n_assets, n_days = prices.shape
    if n_days == 0 or np.any(prices[:, -1] <= 0):
        return np.zeros(n_assets, dtype=int)

    log_prices = np.log(prices)
    signal = (
        _overlay_reversion_20(log_prices)
        + _overlay_price_z_5(log_prices)
        + _overlay_ridge_var(log_prices)
    )
    signal = np.clip(
        np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0),
        -1.0,
        1.0,
    )
    limits = np.full(n_assets, 10_000.0)
    limits[0] = 100_000.0
    return np.trunc(limits * signal / prices[:, -1]).astype(int)


def getMyPosition(prcSoFar):
    """Return the hybrid core plus overlay positions on unused instruments."""
    core = _core_position(prcSoFar)
    overlay = _overlay_position(prcSoFar)
    combined = np.array(core, dtype=int, copy=True)
    unused = combined == 0
    combined[unused] = overlay[unused]
    return combined
