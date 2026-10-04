"""Can the regime-drift finding refine the PAIRS signal for OOS?

Pairs reversion has a fixed sign (spreads revert), so the yearly momentum<->reversal
sign flip does not invert it -- BUT reversion only pays while a spread is actually in a
mean-reverting regime. That regime is trackable (persistence ~0.74). So gate each pair by
its OWN recent reversion strength (rolling OU slope of the spread): trade only while the
spread is reverting; stand down when it starts trending (= the cointegration breakdown that
kills pairs out-of-sample).

Two tests:
  A) universe-wide (1225 beta=1 pairs, vectorised reversion-PnL proxy): does the gate raise
     holdout PnL and shrink the selection->holdout decay?
  B) faithful eval.py accounting on the 16 stable pairs (real commission / integer shares):
     baseline band vs regime-gated band, train 251-500 / test 501-750 + per-year blocks.

Run from repo root:  python research/regime_gated_pairs.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

df = pd.read_csv("prices.txt", sep=r"\s+", header=0)
NAMES = list(df.columns)
P = df.to_numpy(float).T
L = np.log(P)
TRAD = np.arange(1, 51)
IDX = {n: k for k, n in enumerate(NAMES)}

WIN = 145                      # z lookback (four_sigmas Z_W-ish)
GATE_W = 60                    # rolling window for the OU reversion slope
LEG, MIN_HISTORY, BAND_HI = 10_000.0, 130, 4.0
BAND_LO = 0.27
COMM_RATE, POS_LIMIT = 0.0001, 10_000.0
TRAIN_START, TRAIN_END, TEST_START, TEST_END = 251, 500, 501, 750


def eval_score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


# ===========================================================================
# A) universe-wide vectorised test over all 1225 tradeable pairs
# ===========================================================================
ii, jj = np.triu_indices(len(TRAD), 1)
Sp = L[TRAD][ii] - L[TRAD][jj]          # (1225, 750) beta=1 log-spreads
Pn, T = Sp.shape

# trailing z (history = WIN points strictly before day t)
csum = np.concatenate([np.zeros((Pn, 1)), np.cumsum(Sp, axis=1)], axis=1)
csq = np.concatenate([np.zeros((Pn, 1)), np.cumsum(Sp * Sp, axis=1)], axis=1)
t = np.arange(WIN, T)
mean = (csum[:, t] - csum[:, t - WIN]) / WIN
sd = np.sqrt(np.maximum((csq[:, t] - csq[:, t - WIN]) / WIN - mean ** 2, 0.0))
Z = np.full((Pn, T), np.nan)
Z[:, t] = np.where(sd > 1e-12, (Sp[:, t] - mean) / sd, np.nan)

# rolling OU slope b: dS_t = a + b*S_{t-1}; b<0 => reverting.  Causal, per day d.
allow = np.zeros((Pn, T), dtype=bool)
for d in range(GATE_W + 1, T):
    sw = Sp[:, d - GATE_W:d + 1]        # GATE_W+1 points ending at day d
    x = sw[:, :-1]
    y = np.diff(sw, axis=1)
    xb = x.mean(1, keepdims=True)
    yb = y.mean(1, keepdims=True)
    num = ((x - xb) * (y - yb)).sum(1)
    den = ((x - xb) ** 2).sum(1)
    b = np.where(den > 1e-18, num / den, 0.0)
    allow[:, d] = b < 0.0               # spread currently mean-reverting

pos = -np.sign(Z[:, :-1])              # reversion position at day t (applied t->t+1)
dSp = np.diff(Sp, axis=1)
pnl_base = pos * dSp
pnl_gate = pos * allow[:, :-1] * dSp   # same but only while reverting

d_idx = np.arange(1, T)
sel = (d_idx >= 150) & (d_idx <= 500)
oos = (d_idx >= 501) & (d_idx <= 749)


def mean_over(mask, pnl):
    return np.nanmean(np.where(mask, pnl, np.nan), axis=1)


for label, pnl in [("baseline (always-in)", pnl_base), ("regime-gated", pnl_gate)]:
    S_in = mean_over(sel, pnl)
    S_out = mean_over(oos, pnl)
    rho = np.corrcoef(np.nan_to_num(S_in), np.nan_to_num(S_out))[0, 1]
    top = S_in >= np.nanpercentile(S_in, 90)
    print(f"[A] {label:22s} universe holdout mean={np.nanmean(S_out):+.5f}  "
          f"median={np.nanmedian(S_out):+.5f}  corr(sel,oos)={rho:+.3f}  "
          f"top-decile holdout={np.nanmean(S_out[top]):+.5f}  "
          f"shrink={1 - np.nanmean(S_out[top]) / np.nanmean(S_in[top]):.0%}")
frac_traded = allow[:, :-1].mean()
print(f"[A] gate keeps {frac_traded:.0%} of pair-days (turnover/commission down accordingly)")


# ===========================================================================
# B) faithful eval.py accounting on the 16 stable pairs
# ===========================================================================
STABLE = [("AENO", "NWIG"), ("ILVX", "SMAH"), ("EORC", "NGTE"), ("ACAC", "HUXZ"),
          ("HETT", "ULXY"), ("MHRM", "SRTX"), ("EAFC", "SRTX"), ("CCNS", "SRTX"),
          ("MSDP", "ULXY"), ("FWWG", "BLBT"), ("MTNS", "BENI"), ("HRET", "CTGI"),
          ("ACIX", "ULXY"), ("DIHO", "RTTH"), ("MHRM", "RCRI"), ("HUXZ", "NWIG")]
PAIRS = [(IDX[a], IDX[b]) for a, b in STABLE]


def kalman_beta_path(ly, lx):
    Tn = len(ly)
    beta = np.full(Tn, np.nan)
    theta = np.array([ly[:60].mean() - lx[:60].mean(), 1.0])
    cov = np.eye(2) * 1e-2
    Q = np.eye(2) * 1e-5
    Rv = np.var(np.diff(ly[:120] - lx[:120])) + 1e-8
    for tt in range(1, Tn):
        cov += Q
        obs = np.array([1.0, lx[tt - 1]])
        iv = obs @ cov @ obs + Rv
        g = cov @ obs / iv
        theta = theta + g * (ly[tt - 1] - obs @ theta)
        cov = cov - np.outer(g, obs) @ cov
        beta[tt] = theta[1]
    return beta


def z_and_gate(ly, lx, beta, window, gate_w):
    Tn = len(ly)
    z = np.zeros(Tn)
    rev = np.zeros(Tn, dtype=bool)
    for tt in range(MIN_HISTORY - 1, Tn):
        b = beta[tt]
        lo = max(0, tt - window + 1)
        recent = ly[lo:tt + 1] - b * lx[lo:tt + 1]
        hist = recent[:-1]
        s = hist.std()
        if s > 1e-12:
            z[tt] = (recent[-1] - hist.mean()) / s
        glo = max(0, tt - gate_w)
        sw = ly[glo:tt + 1] - b * lx[glo:tt + 1]
        if len(sw) > 10:
            x = sw[:-1]
            y = np.diff(sw)
            xb, yb = x.mean(), y.mean()
            den = ((x - xb) ** 2).sum()
            bb = ((x - xb) * (y - yb)).sum() / den if den > 1e-18 else 0.0
            rev[tt] = bb < 0.0
    return z, rev


def band_sim(z, py, px, lo, gate=None):
    Tn = len(py)
    sy = np.zeros(Tn)
    sx = np.zeros(Tn)
    side = 0
    for d in range(MIN_HISTORY - 1, Tn):
        ok = (gate is None) or gate[d]
        side = -int(np.sign(z[d])) if (ok and lo <= abs(z[d]) <= BAND_HI) else 0
        if side:
            sy[d] = int(side * LEG / py[d])
            sx[d] = int(-side * LEG / px[d])
    return sy, sx


def account_two_leg(sy, sx, py, px, start, end):
    cash = cy = cx = value = comm = 0.0
    pll = []
    for tt in range(start - 1, end + 1):
        cpy, cpx = py[tt - 1], px[tt - 1]
        if tt < end:
            ly_, lx_ = int(POS_LIMIT / cpy), int(POS_LIMIT / cpx)
            ny = float(np.clip(sy[tt - 1], -ly_, ly_))
            nx = float(np.clip(sx[tt - 1], -lx_, lx_))
        else:
            ny, nx = cy, cx
        dy, dx = ny - cy, nx - cx
        cash -= cpy * dy + cpx * dx + comm
        comm = COMM_RATE * (abs(dy) * cpy + abs(dx) * cpx)
        cy, cx = ny, nx
        pv = cy * cpy + cx * cpx
        today = cash + pv - value
        value = cash + pv
        if tt >= start:
            pll.append(today)
    return np.asarray(pll)


paths = {}
for (i, j) in PAIRS:
    beta = kalman_beta_path(L[i], L[j])
    z, rev = z_and_gate(L[i], L[j], beta, WIN, GATE_W)
    paths[(i, j)] = (z, rev)


def book(gate_on):
    tr = te = h1 = h2 = None
    for (i, j) in PAIRS:
        z, rev = paths[(i, j)]
        py, px = P[i], P[j]
        sy, sx = band_sim(z, py, px, BAND_LO, gate=rev if gate_on else None)
        a = account_two_leg(sy, sx, py, px, TRAIN_START, TRAIN_END)
        b = account_two_leg(sy, sx, py, px, TEST_START, TEST_END)
        c = account_two_leg(sy, sx, py, px, TEST_START, 625)
        d = account_two_leg(sy, sx, py, px, 626, TEST_END)
        tr = a if tr is None else tr + a
        te = b if te is None else te + b
        h1 = c if h1 is None else h1 + c
        h2 = d if h2 is None else h2 + d
    return tr, te, h1, h2


print()
for name, on in [("baseline band     ", False), ("regime-gated band ", True)]:
    tr, te, h1, h2 = book(on)
    print(f"[B] {name} train {eval_score(tr.mean(), tr.std()):7.1f}  "
          f"test {eval_score(te.mean(), te.std()):7.1f}  "
          f"| test-H1 {eval_score(h1.mean(), h1.std()):7.1f}  "
          f"test-H2 {eval_score(h2.mean(), h2.std()):7.1f}  "
          f"| decay {eval_score(tr.mean(), tr.std()) - eval_score(te.mean(), te.std()):+6.1f}")
