"""Robustness check: is the pairs reversion-gate STRUCTURAL or just fitting the burned
501-750 window?  Per cusum_output/VALIDATION_PROTOCOL.md we judge by (i) both TRAIN blocks
(251-375, 376-500), (ii) the block FLOOR (worst 125d block), (iii) consistency across the
gate window -- NOT by the test score alone.  Gate rule is fixed a priori: b<0 OU slope over
GATE_W days (structural sign test, nothing tuned to any score).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

df = pd.read_csv("prices.txt", sep=r"\s+", header=0)
NAMES = list(df.columns)
P = df.to_numpy(float).T
L = np.log(P)
IDX = {n: k for k, n in enumerate(NAMES)}

WIN = 145
LEG, MIN_HISTORY, BAND_HI, BAND_LO = 10_000.0, 130, 4.0, 0.27
COMM_RATE, POS_LIMIT = 0.0001, 10_000.0

STABLE = [("AENO", "NWIG"), ("ILVX", "SMAH"), ("EORC", "NGTE"), ("ACAC", "HUXZ"),
          ("HETT", "ULXY"), ("MHRM", "SRTX"), ("EAFC", "SRTX"), ("CCNS", "SRTX"),
          ("MSDP", "ULXY"), ("FWWG", "BLBT"), ("MTNS", "BENI"), ("HRET", "CTGI"),
          ("ACIX", "ULXY"), ("DIHO", "RTTH"), ("MHRM", "RCRI"), ("HUXZ", "NWIG")]
PAIRS = [(IDX[a], IDX[b]) for a, b in STABLE]

# 125-day blocks: 2 train (251-375, 376-500) + 2 test (501-625, 626-750)
BLOCKS = [("tr1", 251, 375), ("tr2", 376, 500), ("te1", 501, 625), ("te2", 626, 750)]


def eval_score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


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
        g = cov @ obs / (obs @ cov @ obs + Rv)
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
            x, y = sw[:-1], np.diff(sw)
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


def account(sy, sx, py, px, start, end):
    cash = cy = cx = value = comm = 0.0
    pll = []
    for tt in range(start - 1, end + 1):
        cpy, cpx = py[tt - 1], px[tt - 1]
        if tt < end:
            ny = float(np.clip(sy[tt - 1], -int(POS_LIMIT / cpy), int(POS_LIMIT / cpy)))
            nx = float(np.clip(sx[tt - 1], -int(POS_LIMIT / cpx), int(POS_LIMIT / cpx)))
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


def book_blocks(gate_w):
    z_rev = {}
    for (i, j) in PAIRS:
        beta = kalman_beta_path(L[i], L[j])
        z_rev[(i, j)] = z_and_gate(L[i], L[j], beta, WIN, gate_w) if gate_w else \
            (z_and_gate(L[i], L[j], beta, WIN, 60)[0], None)
    out = {}
    for name, s, e in BLOCKS:
        tot = None
        for (i, j) in PAIRS:
            z, rev = z_rev[(i, j)]
            sy, sx = band_sim(z, P[i], P[j], BAND_LO, gate=rev)
            pll = account(sy, sx, P[i], P[j], s, e)
            tot = pll if tot is None else tot + pll
        out[name] = eval_score(tot.mean(), tot.std())
    return out


print(f"{'gate':>10s} | {'tr1':>7s} {'tr2':>7s} {'te1':>7s} {'te2':>7s} | "
      f"{'trainF':>7s} {'testF':>7s} {'floorAll':>8s}")
for gate_w in [0, 40, 60, 90, 120]:
    b = book_blocks(gate_w)
    trF = min(b["tr1"], b["tr2"])
    teF = min(b["te1"], b["te2"])
    allF = min(b.values())
    lbl = "no gate" if gate_w == 0 else f"b<0 W={gate_w}"
    print(f"{lbl:>10s} | {b['tr1']:7.1f} {b['tr2']:7.1f} {b['te1']:7.1f} {b['te2']:7.1f} | "
          f"{trF:7.1f} {teF:7.1f} {allF:8.1f}")
