"""Forensics on prices.txt: what generated this data, and is it recoverable?

Two separate questions, often conflated:

  (A) STRUCTURE   what family of process produced these prices? Gaussian or
                  fat-tailed, constant or stochastic vol, how many factors,
                  is the cointegration exact? This is cheap and it works.

  (B) STATE       the exact RNG seed, so future draws can be reproduced. This
                  needs the exact pipeline AND the exact seed AND the exact
                  draw order. Searching it blind is where multiple testing
                  eats you alive; the script quantifies that too.

Run from repo root:  python research/generator_forensics.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = _ROOT / "analysis_outputs" / "generator_forensics"

raw = pd.read_csv(_ROOT / "prices.txt", sep=r"\s+", header=0)
SYMBOLS = list(raw.columns)
PRC = raw.to_numpy(float).T          # (n_inst, n_days)
N, T = PRC.shape
R = np.diff(np.log(PRC), axis=1)     # log returns, (n_inst, T-1)
lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


say(f"prices.txt: {N} instruments x {T} days")
say("=" * 78)

# ---------------------------------------------------------------- quantisation
say("\n[1] PRICE QUANTISATION")
cents = np.round(PRC * 100.0)
exact2dp = np.allclose(PRC * 100.0, cents, atol=1e-6)
say(f"  every price exactly 2dp: {exact2dp}")
say(f"  min price {PRC.min():.2f}   max price {PRC.max():.2f}")
# how much information does rounding destroy relative to a daily move?
tick_rel = 0.005 / PRC              # half a cent as a fraction of price
typ_move = np.abs(R).mean()
say(f"  median rounding noise / typical daily move: "
    f"{np.median(tick_rel) / typ_move:.4f}")
say("  -> rounding is small vs daily moves, so it does NOT hide the dynamics,")
say("     but it does destroy exact float reproduction of any candidate stream.")

# ---------------------------------------------------------------- distribution
say("\n[2] RETURN DISTRIBUTION  (is it plain Gaussian?)")
sk = sps.skew(R, axis=1, bias=False)
ku = sps.kurtosis(R, axis=1, fisher=True, bias=False)
jb_p = np.array([sps.jarque_bera(R[i]).pvalue for i in range(N)])
sw_p = np.array([sps.shapiro(R[i]).pvalue for i in range(N)])
say(f"  skew      mean {sk.mean():+.3f}  min {sk.min():+.3f}  max {sk.max():+.3f}")
say(f"  exkurt    mean {ku.mean():+.3f}  min {ku.min():+.3f}  max {ku.max():+.3f}")
say(f"  Jarque-Bera rejected at 5%: {int((jb_p < 0.05).sum())}/{N} assets")
say(f"  Shapiro-Wilk rejected at 5%: {int((sw_p < 0.05).sum())}/{N} assets")
say(f"  (at 5% you expect ~{0.05 * N:.1f}/{N} rejections by chance)")

# ---------------------------------------------------------------- vol dynamics
say("\n[3] VOLATILITY DYNAMICS  (constant sigma, or GARCH-like?)")
z = (R - R.mean(axis=1, keepdims=True)) / R.std(axis=1, keepdims=True)


def ljung_box(x: np.ndarray, lags: int = 10) -> float:
    n = len(x)
    x = x - x.mean()
    denom = np.sum(x * x)
    acf = [np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)]
    stat = n * (n + 2) * sum(a * a / (n - k - 1) for k, a in enumerate(acf))
    return float(sps.chi2.sf(stat, lags))


arch_p = np.array([ljung_box(z[i] ** 2) for i in range(N)])
say(f"  ARCH (Ljung-Box on squared z, 10 lags) rejected at 5%: "
    f"{int((arch_p < 0.05).sum())}/{N} assets")
half = (T - 1) // 2
vol_ratio = R[:, half:].std(axis=1) / R[:, :half].std(axis=1)
say(f"  2nd-half / 1st-half vol ratio: mean {vol_ratio.mean():.3f} "
    f"[{vol_ratio.min():.3f}, {vol_ratio.max():.3f}]")

# ---------------------------------------------------------------- autocorr
say("\n[4] RETURN AUTOCORRELATION  (is each asset a clean random walk?)")
lb_p = np.array([ljung_box(z[i]) for i in range(N)])
say(f"  Ljung-Box on returns rejected at 5%: {int((lb_p < 0.05).sum())}/{N} assets")
ac1 = np.array([np.corrcoef(R[i, :-1], R[i, 1:])[0, 1] for i in range(N)])
say(f"  lag-1 autocorrelation: mean {ac1.mean():+.4f}  "
    f"range [{ac1.min():+.4f}, {ac1.max():+.4f}]")
se = 1.0 / np.sqrt(T - 1)
say(f"  |lag-1| > 2SE ({2 * se:.4f}): {int((np.abs(ac1) > 2 * se).sum())}/{N} assets")

# ---------------------------------------------------------------- factors
say("\n[5] FACTOR STRUCTURE  (Marchenko-Pastur: how many real factors?)")
Z = (R - R.mean(axis=1, keepdims=True)) / R.std(axis=1, keepdims=True)
corr = (Z @ Z.T) / (T - 1)
eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
q = N / (T - 1)
mp_edge = (1.0 + np.sqrt(q)) ** 2
say(f"  q = N/T = {q:.3f}   MP upper edge = {mp_edge:.3f}")
say(f"  eigenvalues above MP edge: {int((eig > mp_edge).sum())}")
say("  top 8: " + "  ".join(f"{v:.2f}" for v in eig[:8]))
say(f"  top eigenvalue explains {100 * eig[0] / N:.1f}% of variance")

# ---------------------------------------------------------------- lattice
say("\n[6] LATTICE / DISCRETENESS  (does a raw RNG stream show through?)")
uniq = np.unique(np.round(R, 12))
say(f"  distinct log returns: {uniq.size:,} of {R.size:,}")
diffs = np.diff(np.sort(uniq))
say(f"  smallest gap between distinct returns: {diffs.min():.3e}")
say("  -> log returns inherit price rounding, so they sit on a price-dependent")
say("     lattice; there is no single global tick to invert.")

# ---------------------------------------------------------------- cointegration
say("\n[7] IS THE COINTEGRATION EXACT?  (imposed by construction, or estimated?)")
say("  For a hand-built generator, an imposed pair has a beta that is EXACTLY")
say("  constant across subsamples. Drift in beta means it was not imposed.")
best = []
for i in range(N):
    for j in range(i + 1, N):
        y, x = np.log(PRC[i]), np.log(PRC[j])
        b_all = np.polyfit(x, y, 1)[0]
        b1 = np.polyfit(x[: T // 2], y[: T // 2], 1)[0]
        b2 = np.polyfit(x[T // 2:], y[T // 2:], 1)[0]
        resid = y - (np.polyfit(x, y, 1)[1] + b_all * x)
        # ADF-lite: AR(1) coefficient of the residual
        rho = np.polyfit(resid[:-1], resid[1:], 1)[0]
        best.append((rho, abs(b2 - b1) / max(abs(b_all), 1e-9),
                     SYMBOLS[i], SYMBOLS[j]))
best.sort(key=lambda r: r[0])
say("  most mean-reverting pairs (rho = residual AR(1); drift = |b2-b1|/|b|):")
for rho, drift, a, b in best[:8]:
    say(f"    {a}-{b:5s}  rho={rho:.4f}  half-life={np.log(2)/max(-np.log(rho),1e-9):5.1f}d"
        f"  beta drift={100 * drift:6.2f}%")
say(f"  median beta drift across all {len(best):,} pairs: "
    f"{100 * np.median([d for _, d, _, _ in best]):.2f}%")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "forensics.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT / 'forensics.txt'}")
