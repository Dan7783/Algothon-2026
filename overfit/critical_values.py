#!/usr/bin/env python
"""Novy-Marx (2015) multi-signal overfitting null: reference implementation.

Task 1 of the overfitting audit. Implements the iid null exactly as the paper
specifies. See ``novy-marx-multisignal-brief.md`` for the formulas and
equation numbers referenced throughout.

Everything here assumes the paper's model: n individual-strategy return
series that are **uncorrelated with equal volatility**, each signed so its
in-sample return is positive (hence the t-vector is |N(0,1)|^n).

--------------------------------------------------------------------------
IMPORTANT -- the iid assumption is load-bearing, and eq. (2) breaks without it
--------------------------------------------------------------------------
Eq. (2), ``t_omega = w'.t / sqrt(w'.w)``, is NOT the general case. For a
composite with weights w over individual strategies with return covariance
Sigma and correlation C:

    r_w    = sum_i w_i r_i
    t_w    = mean(r_w) sqrt(T) / std(r_w)
           = sum_i w_i sigma_i t_i / sqrt(w' Sigma w)

and only when all sigma_i are equal does this reduce to

    t_w    = w'.t / sqrt(w' C w)                                    (2')

Eq. (2) is (2') with C = I. So under correlated signals the DENOMINATOR
changes too -- it is not enough to draw the null from N(0, C) and keep the
L1/sqrt(k) and L2 norms. The correct generalisations are

    MV  :  t = (sum of top-k t_i) / sqrt(1' C_k 1)
    MVE :  t = sqrt(t_k' C_k^{-1} t_k)          (true ex-post MVE weights are
                                                 w propto C^{-1} t, not t)

both of which collapse to eqs (3) and (4) when C = I. Using sqrt(k) and
||.||_2 against a correlated null would compute the null statistic and the
realised statistic under different definitions, which is not a valid test.
Those C-aware forms live in ``overfit/correlated_null.py`` (Task 2); this
module is deliberately the iid reference only, so the validation gate tests
the paper's published numbers and nothing else.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "t_mv", "t_mve", "crit_1_of_n", "crit_n_of_n_mve", "crit_n_of_n_mv",
    "moment_block", "crit_k_of_n_approx", "crit_k_of_n_exact",
    "expected_t_mv", "n_star", "DEFAULT_SEED",
]

DEFAULT_SEED = 20260727


# --------------------------------------------------------------------------
# Composite statistics -- eqs (3) and (4)
# --------------------------------------------------------------------------

def _top_k(t_vec: np.ndarray, k: int, signed: bool) -> np.ndarray:
    t = np.abs(np.asarray(t_vec, dtype=float)) if signed else np.asarray(t_vec, float)
    if k > t.shape[-1]:
        raise ValueError(f"k={k} exceeds number of signals {t.shape[-1]}")
    # Partition is O(n); we only need the top k, unordered.
    part = np.partition(t, -k, axis=-1)[..., -k:]
    return part


def t_mv(t_vec: np.ndarray, k: int, signed: bool = True) -> float:
    """Equal-weighted composite t-stat, eq (3): ||top-k||_1 / sqrt(k).

    `signed=True` applies the paper's sign convention (each signal flipped to
    predict positive in sample), i.e. order statistics of |t|.
    """
    return float(_top_k(t_vec, k, signed).sum(axis=-1) / np.sqrt(k))


def t_mve(t_vec: np.ndarray, k: int, signed: bool = True) -> float:
    """Signal-weighted composite t-stat, eq (4): ||top-k||_2."""
    return float(np.sqrt((_top_k(t_vec, k, signed) ** 2).sum(axis=-1)))


# --------------------------------------------------------------------------
# Closed-form critical values
# --------------------------------------------------------------------------

def crit_1_of_n(n: int, p: float = 0.05) -> float:
    """Best 1-of-n, pure selection bias. Eq (5)."""
    return float(stats.norm.ppf((1.0 - p / 2.0) ** (1.0 / n)))


def crit_n_of_n_mve(n: int, p: float = 0.05) -> float:
    """Best n-of-n signal-weighted, pure overfitting. Eq (7): sqrt(chi2 ppf)."""
    return float(np.sqrt(stats.chi2.ppf(1.0 - p, df=n)))


def crit_n_of_n_mv(n: int, p: float = 0.05, skew_correction: bool = True) -> float:
    """Best n-of-n equal-weighted, pure overfitting. Eq (10) (or (9))."""
    tail = (n * p / (n + 1.0)) if skew_correction else p
    return float(
        np.sqrt(2.0 / np.pi) * np.sqrt(n)
        + np.sqrt(1.0 - 2.0 / np.pi) * stats.norm.ppf(1.0 - tail)
    )


# --------------------------------------------------------------------------
# General best k-of-n
# --------------------------------------------------------------------------

def moment_block(n: int, k: int) -> tuple[float, float, float]:
    """The (mu, lambda, Sigma) block of section 4.4.

    mu     = expected (n-k)th order statistic of |Z|, the selection threshold
    lambda = inverse Mills ratio at mu = E[chi | chi > mu]
    Sigma  = E[chi^2 | chi > mu] = 1 + mu*lambda
    """
    mu = float(stats.norm.ppf(1.0 - (k + 1.0) / (2.0 * (n + 1.0))))
    lam = float(2.0 * ((n + 1.0) / (k + 1.0)) * stats.norm.pdf(mu))
    sigma = float(1.0 + mu * lam)
    return mu, lam, sigma


def crit_k_of_n_approx(n: int, k: int, p: float = 0.05,
                       weighting: str = "mv") -> float:
    """Analytic approximation to the best k-of-n critical value. Eqs (13)/(14).

    Uses mu_MVE2 = k * Sigma_{n,k}; the paper's eq. (37) prints k^2(1+mu*lam),
    which contradicts its own derivation at (30)-(31). See the brief.
    """
    if k > n:
        raise ValueError("k cannot exceed n")
    mu, lam, sig = moment_block(n, k)
    # Small-sample right-skew correction: p -> kp/(k+1), same trick as eq (10).
    z = float(stats.norm.ppf(1.0 - k * p / (k + 1.0)))
    spread = k * (n - k) / ((k + 1.0) * (n + 2.0))

    if weighting.lower() == "mv":
        mean = np.sqrt(k) * lam
        var = sig - lam ** 2 + spread * (lam - mu) ** 2
        return float(mean + np.sqrt(max(var, 0.0)) * z)

    if weighting.lower() == "mve":
        mean2 = k * sig
        var2 = (k * (mu ** 3 * lam + 3.0 * sig - sig ** 2)
                + k * spread * (sig - mu ** 2) ** 2)
        return float(np.sqrt(max(mean2 + np.sqrt(max(var2, 0.0)) * z, 0.0)))

    raise ValueError("weighting must be 'mv' or 'mve'")


def _draw_top_k_stats(n: int, k: int, n_sims: int, weighting: str,
                      rng: np.random.Generator, chunk: int = 200_000
                      ) -> np.ndarray:
    """Vectorised draws of the best k-of-n composite statistic under the null."""
    out = np.empty(n_sims, dtype=float)
    filled = 0
    while filled < n_sims:
        size = min(chunk, n_sims - filled)
        draws = np.abs(rng.standard_normal((size, n)))
        top = np.partition(draws, -k, axis=1)[:, -k:]
        if weighting.lower() == "mv":
            stat = top.sum(axis=1) / np.sqrt(k)
        elif weighting.lower() == "mve":
            stat = np.sqrt((top ** 2).sum(axis=1))
        else:
            raise ValueError("weighting must be 'mv' or 'mve'")
        out[filled:filled + size] = stat
        filled += size
    return out


def crit_k_of_n_exact(n: int, k: int, p: float = 0.05, weighting: str = "mv",
                      n_sims: int = 200_000, seed: int = DEFAULT_SEED) -> float:
    """Monte Carlo critical value per eqs (11)/(12).

    Draws n iid standard normals, takes |.|, selects the top k, and forms the
    L1/sqrt(k) (MV) or L2 (MVE) composite. Returns the (1-p) quantile.
    """
    if k > n:
        raise ValueError("k cannot exceed n")
    rng = np.random.default_rng(seed)
    stat = _draw_top_k_stats(n, k, n_sims, weighting, rng)
    return float(np.quantile(stat, 1.0 - p))


def expected_t_mv(n: int, k: int, n_sims: int = 100_000,
                  seed: int = DEFAULT_SEED) -> float:
    """E[t^MV_{n,k}] by simulation -- used for the k ~ n/2 result, eq (17)."""
    rng = np.random.default_rng(seed)
    return float(_draw_top_k_stats(n, k, n_sims, "mv", rng).mean())


def n_star(tau: float, p: float = 0.05) -> float:
    """Equivalent number of single-signal candidates. Eq (18)."""
    denom = stats.norm.logcdf(tau)          # log N(tau), stable in the far tail
    return float(np.log1p(-p / 2.0) / denom)


if __name__ == "__main__":
    print("Novy-Marx (2015) iid null -- 5% critical values\n")
    print(f"{'n':>5} {'k':>4} {'MV exact':>10} {'MV approx':>10} "
          f"{'MVE exact':>10} {'MVE approx':>10} {'n* (MV)':>14}")
    print("-" * 70)
    for n in (10, 20, 50, 100):
        for k in (1, 2, 3, 5, 10):
            if k > n:
                continue
            mv_e = crit_k_of_n_exact(n, k, weighting="mv")
            mv_a = crit_k_of_n_approx(n, k, weighting="mv")
            mve_e = crit_k_of_n_exact(n, k, weighting="mve")
            mve_a = crit_k_of_n_approx(n, k, weighting="mve")
            print(f"{n:>5} {k:>4} {mv_e:>10.3f} {mv_a:>10.3f} "
                  f"{mve_e:>10.3f} {mve_a:>10.3f} {n_star(mv_e):>14,.0f}")
