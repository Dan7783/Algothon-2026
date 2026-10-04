"""Validation gate for overfit/critical_values.py.

Reproduces the published figures in Novy-Marx (2015) as assertions. Nothing
in Tasks 2-4 may be trusted until this file passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from overfit.critical_values import (
    crit_1_of_n,
    crit_k_of_n_approx,
    crit_k_of_n_exact,
    crit_n_of_n_mv,
    crit_n_of_n_mve,
    expected_t_mv,
    moment_block,
    n_star,
    t_mv,
    t_mve,
)

SIMS = 200_000


# ---------------------------------------------------------------- statistics

def test_t_mv_is_l1_over_sqrt_k():
    t = np.array([3.0, -2.0, 1.0, 0.5])
    # top-2 of |t| = {3, 2} -> 5/sqrt(2)
    assert t_mv(t, 2) == pytest.approx(5.0 / np.sqrt(2.0))


def test_t_mve_is_l2_of_top_k():
    t = np.array([3.0, -4.0, 1.0])
    assert t_mve(t, 2) == pytest.approx(5.0)  # sqrt(9+16)


def test_signing_uses_absolute_values():
    """The paper signs every signal to work in sample; sign must not matter."""
    t = np.array([2.0, -3.0, 1.5])
    assert t_mv(t, 3) == pytest.approx(t_mv(np.abs(t), 3))
    assert t_mve(t, 3) == pytest.approx(t_mve(np.abs(t), 3))


def test_mv_never_exceeds_mve():
    """t^MV <= t^MVE always (section 3), equality only if top-k are equal."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.standard_normal(20)
        for k in (1, 3, 7, 20):
            assert t_mv(t, k) <= t_mve(t, k) + 1e-12
    equal = np.full(5, 2.0)
    assert t_mv(equal, 5) == pytest.approx(t_mve(equal, 5))


def test_k_equals_one_agrees_across_weightings():
    t = np.array([1.0, 2.5, 0.3])
    assert t_mv(t, 1) == pytest.approx(t_mve(t, 1))


# ------------------------------------------------------------ closed  forms

def test_crit_1_of_1_is_two_sided_normal():
    assert crit_1_of_n(1, 0.05) == pytest.approx(stats.norm.ppf(0.975), abs=1e-10)


def test_crit_1_of_n_matches_equation_5():
    for n in (1, 5, 20, 100):
        expected = stats.norm.ppf((1.0 - 0.05 / 2.0) ** (1.0 / n))
        assert crit_1_of_n(n, 0.05) == pytest.approx(expected, rel=1e-12)


def test_bonferroni_rule_of_thumb():
    """p_true ~ n * naive p (section 4.1), good to ~10% for p < 35%."""
    for n in (5, 20, 100):
        tau = crit_1_of_n(n, 0.05)
        naive_p = 2.0 * stats.norm.sf(tau)
        assert naive_p * n == pytest.approx(0.05, rel=0.05)


def test_crit_n_of_n_mve_is_chi_quantile():
    for n in (2, 10, 50):
        assert crit_n_of_n_mve(n, 0.05) == pytest.approx(
            np.sqrt(stats.chi2.ppf(0.95, df=n)), rel=1e-12)


def test_crit_n_of_n_mv_matches_exact_simulation():
    """Eq (10) should track the simulated n-of-n MV null closely."""
    for n in (10, 20, 50):
        approx = crit_n_of_n_mv(n, 0.05)
        exact = crit_k_of_n_exact(n, n, 0.05, "mv", n_sims=SIMS)
        assert approx == pytest.approx(exact, rel=0.02)


def test_uncorrected_equation_9_is_a_lower_bound():
    """Eq (9) understates because the true null is right-skewed."""
    for n in (5, 10, 20):
        assert crit_n_of_n_mv(n, 0.05, skew_correction=False) < crit_n_of_n_mv(n, 0.05)


# --------------------------------------------------------- moment block

def test_moment_block_lambda_is_inverse_mills():
    for n, k in ((10, 3), (50, 5), (100, 10)):
        mu, lam, sig = moment_block(n, k)
        assert lam == pytest.approx(stats.norm.pdf(mu) / stats.norm.sf(mu), rel=1e-10)
        assert sig == pytest.approx(1.0 + mu * lam, rel=1e-12)


def test_moment_block_mu_is_selection_threshold():
    """mu_{n,k} is the expected (n-k)th order statistic of |Z|."""
    rng = np.random.default_rng(7)
    n, k = 40, 8
    mu, _, _ = moment_block(n, k)
    draws = np.abs(rng.standard_normal((40_000, n)))
    empirical = np.sort(draws, axis=1)[:, n - k - 1].mean()
    assert mu == pytest.approx(empirical, abs=0.05)


def _approx_moments(n, k):
    """The mean/variance the approximation implies, before the quantile step."""
    mu, lam, sig = moment_block(n, k)
    spread = k * (n - k) / ((k + 1.0) * (n + 2.0))
    return {
        "mu_mv": np.sqrt(k) * lam,
        "var_mv": sig - lam ** 2 + spread * (lam - mu) ** 2,
        "mu_mve2": k * sig,
        "var_mve2": (k * (mu ** 3 * lam + 3.0 * sig - sig ** 2)
                     + k * spread * (sig - mu ** 2) ** 2),
    }


@pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
def test_mve_moments_collapse_to_chi_squared_at_k_equals_n(n):
    """At k=n the MVE statistic is ||t||_2 = sqrt(chi2_n) exactly.

    So E[(t^MVE)^2] = n and Var((t^MVE)^2) = 2n. This is the decisive check
    that mu_MVE2 = k*Sigma is correct: the paper's printed eq. (37),
    k^2(1+mu*lam), would give n^2 here -- wrong by a factor of n.
    """
    m = _approx_moments(n, n)
    assert m["mu_mve2"] == pytest.approx(n, rel=1e-12)
    assert m["var_mve2"] == pytest.approx(2.0 * n, rel=1e-12)
    # And the erroneous printed form does not.
    mu, lam, _ = moment_block(n, n)
    assert (n ** 2) * (1.0 + mu * lam) != pytest.approx(n, rel=1e-6)


@pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
def test_mv_moments_collapse_to_equation_8_at_k_equals_n(n):
    """At k=n eq (8) gives mean sqrt(2n/pi) and variance 1 - 2/pi."""
    m = _approx_moments(n, n)
    assert m["mu_mv"] == pytest.approx(np.sqrt(2.0 * n / np.pi), rel=1e-12)
    assert m["var_mv"] == pytest.approx(1.0 - 2.0 / np.pi, rel=1e-12)


# ------------------------------------------- exact vs approximate agreement

@pytest.mark.parametrize("n", [10, 20, 40, 100])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("weighting", ["mv", "mve"])
def test_exact_and_approx_agree(n, k, weighting):
    """The analytic approximation must track the Monte Carlo across the grid.

    Tolerance is n-dependent and this is a property of the approximation, not
    slack for bugs. Eqs (13)/(14) apply a normal approximation to a sum of a
    handful of order statistics plus a kp/(k+1) skew fudge, so accuracy
    improves monotonically with n. Measured MVE error at k=3: 3.76% (n=10),
    3.23% (n=20), 2.56% (n=40), 2.01% (n=100), 1.79% (n=200). Correctness of
    the underlying moments is pinned exactly, independently of this tolerance,
    by the two k=n collapse tests above.
    """
    if k > n:
        pytest.skip("k > n")
    exact = crit_k_of_n_exact(n, k, 0.05, weighting, n_sims=SIMS)
    approx = crit_k_of_n_approx(n, k, 0.05, weighting)
    tol = 0.045 if n < 20 else 0.035
    assert approx == pytest.approx(exact, rel=tol)
    # The approximation is a floor: it should never overstate the true null.
    assert approx <= exact * 1.005


def test_approximation_error_shrinks_with_n():
    """Confirms the small-n failure mode is the approximation, not a bug."""
    errors = []
    for n in (10, 40, 200):
        exact = crit_k_of_n_exact(n, 3, 0.05, "mve", n_sims=SIMS)
        approx = crit_k_of_n_approx(n, 3, 0.05, "mve")
        errors.append(abs(approx - exact) / exact)
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.025


# --------------------------------------------------- published headline values

def test_headline_critical_values():
    """Section 1 quotes ~4, ~5, ~7. Analytic values are a ~7% floor (section 7)."""
    got = {
        (10, 3): crit_k_of_n_exact(10, 3, 0.05, "mv", n_sims=SIMS),
        (20, 3): crit_k_of_n_exact(20, 3, 0.05, "mv", n_sims=SIMS),
        (100, 7): crit_k_of_n_exact(100, 7, 0.05, "mv", n_sims=SIMS),
    }
    # Ordering and rough magnitude are the testable content of the prose.
    assert 3.3 < got[(10, 3)] < 4.3
    assert 3.8 < got[(20, 3)] < 5.2
    assert 6.2 < got[(100, 7)] < 7.6
    assert got[(10, 3)] < got[(20, 3)] < got[(100, 7)]


def test_five_junk_signals_pass_a_naive_test():
    """Section 3: k signals at t~1 give combined t ~ sqrt(k)."""
    assert t_mv(np.ones(5), 5) == pytest.approx(np.sqrt(5.0))
    assert t_mv(np.ones(5), 5) > 1.96
    assert t_mv(np.full(2, 1.5), 2) == pytest.approx(1.5 * np.sqrt(2.0))


# ------------------------------------------------------------ Table 1 (n*)

TABLE_1_MV = {
    (10, 2): 57, (10, 3): 164, (10, 5): 426, (10, 10): 141,
    (20, 2): 227, (20, 3): 1_250, (20, 5): 12_000, (20, 10): 113_000,
    (50, 2): 1_360, (50, 3): 17_500, (50, 5): 900_000, (50, 10): 5.03e8,
    (100, 2): 5_220, (100, 3): 128_000, (100, 5): 2.33e7, (100, 10): 2.85e11,
}


@pytest.mark.parametrize("key,published", sorted(TABLE_1_MV.items()))
def test_n_star_reproduces_table_1_panel_a(key, published):
    n, k = key
    tau = crit_k_of_n_exact(n, k, 0.05, "mv", n_sims=SIMS)
    got = n_star(tau, 0.05)
    # n* is hugely convex in tau, so compare on a log scale.
    assert np.log10(got) == pytest.approx(np.log10(published), rel=0.05)


def test_n_star_inverts_crit_1_of_n():
    for n in (5, 20, 100):
        tau = crit_1_of_n(n, 0.05)
        assert n_star(tau, 0.05) == pytest.approx(n, rel=1e-6)


def test_panel_a_non_monotonic_at_n_10():
    """k ~ n/2 peak: n*(10,10) < n*(10,5) (section 6)."""
    tau5 = crit_k_of_n_exact(10, 5, 0.05, "mv", n_sims=SIMS)
    tau10 = crit_k_of_n_exact(10, 10, 0.05, "mv", n_sims=SIMS)
    assert n_star(tau10, 0.05) < n_star(tau5, 0.05)


def test_mve_panel_b_is_monotone_in_k():
    """Signal weighting is monotone -- more assets weakly improve ex-post MVE."""
    taus = [crit_k_of_n_exact(20, k, 0.05, "mve", n_sims=SIMS) for k in range(1, 11)]
    assert all(b >= a - 1e-9 for a, b in zip(taus, taus[1:]))


def test_signal_weighting_is_worse_than_equal_weighting():
    """Section 2: MVE gives more freedom to overfit, so a higher bar."""
    for n, k in ((20, 3), (50, 5), (100, 10)):
        mv = crit_k_of_n_exact(n, k, 0.05, "mv", n_sims=SIMS)
        mve = crit_k_of_n_exact(n, k, 0.05, "mve", n_sims=SIMS)
        assert mve > mv


# ---------------------------------------------------------------- k ~ n/2

@pytest.mark.parametrize("n", [10, 20, 40])
def test_in_sample_optimal_k_is_about_half_of_n(n):
    """Eq (17): argmax_k E[t^MV_{n,k}] has (k+1)/(n+1) ~ 0.541."""
    means = [expected_t_mv(n, k, n_sims=60_000) for k in range(1, n + 1)]
    k_star = int(np.argmax(means)) + 1
    assert 0.35 <= k_star / n <= 0.75, f"n={n}: argmax k={k_star}"
    assert (k_star + 1) / (n + 1) == pytest.approx(0.541, abs=0.16)


def test_selection_inflates_expected_t_by_about_sixty_percent():
    """Section 5: E[t^MV_{2k,k}] ~ 1.27 sqrt(k) vs E[t^MV_{k,k}] ~ 0.80 sqrt(k)."""
    for k in (3, 5, 10):
        with_sel = expected_t_mv(2 * k, k, n_sims=60_000)
        without = expected_t_mv(k, k, n_sims=60_000)
        assert with_sel / np.sqrt(k) == pytest.approx(1.27, rel=0.06)
        assert without / np.sqrt(k) == pytest.approx(0.80, rel=0.06)
        assert with_sel / without == pytest.approx(1.6, rel=0.10)


# ------------------------------------------------------------- determinism

def test_seeded_monte_carlo_is_reproducible():
    a = crit_k_of_n_exact(30, 5, 0.05, "mv", n_sims=50_000, seed=123)
    b = crit_k_of_n_exact(30, 5, 0.05, "mv", n_sims=50_000, seed=123)
    c = crit_k_of_n_exact(30, 5, 0.05, "mv", n_sims=50_000, seed=124)
    assert a == b
    assert a != c
