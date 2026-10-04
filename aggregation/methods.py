#!/usr/bin/env python
"""The seven model-free aggregators, adapted for Algothon 2026.

See ``model-free-aggregation-brief.md``. Every method maps a trailing window
of per-model P/L to a weight vector over models.

THREE MANDATED ADAPTATIONS (brief PART 3)
  1. No sum-to-1 normalisation in the live path -- Algothon has no portfolio
     budget and the score is homogeneous in size. Methods return UNNORMALISED
     conviction; ``normalise()`` exists for diagnostics only.
  2. Long/short supported. ``clip_negative=True`` reproduces the article;
     ``clip_negative=False`` allows inverting a persistently losing model.
     The signed variant is the exact "signing step" Novy-Marx flags as the
     source of overfitting bias, so it is only reachable through
     ``CausalWindow``, which makes full-sample access raise (see below).
  3. Mean-variance reimplemented on numpy/scipy (PyPortfolioOpt is not in the
     sandbox), annualised with 250 not 52.

TWO FORMULA MISMATCHES FIXED, BOTH VERSIONS KEPT (brief 2.4)
  * ``softmax_mean`` vs ``softmax_sum``. The article's text says mean, its code
    uses the column sum; over a fixed window these differ by a factor of
    ``lookback``, so the sum version's effective temperature is beta*lookback
    and it becomes winner-take-all as the lookback grows.
  * ``kelly_continuous`` (mu/sigma^2, the stated Thorp formula) vs
    ``sharpe_weighted`` (mu/sigma, what the article actually computes). The
    second is Sharpe-proportional weighting and is NOT Kelly; it is named
    honestly here.

MEASURED CAVEAT (brief PART 3b): after scaling to the caps, ~94% of active
positions saturate and the book is 0.989 correlated with a pure bang-bang
sign book. So these methods differ mainly through SIGN and ZERO/NON-ZERO
selection, not through relative conviction. Expect their gross performance to
converge and turnover to be the real differentiator.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CausalWindow", "equal_weight", "weighted_average", "softmax_mean",
    "softmax_sum", "mean_variance", "binary_kelly", "kelly_continuous",
    "sharpe_weighted", "running_count", "normalise", "METHODS",
    "ANNUALISATION",
]

ANNUALISATION = 250.0          # daily data, not the article's weekly 52
_EPS = 1e-12


class LookaheadError(RuntimeError):
    """Raised when an aggregator is offered data at or beyond the decision day."""


class CausalWindow:
    """A trailing P/L window that structurally cannot expose future rows.

    The signed (non-clipping) variants invert a model because it lost money,
    which is precisely the full-sample signing operation Novy-Marx identifies
    as the origin of overfitting bias. Rather than relying on discipline, the
    aggregators only accept a CausalWindow: it is constructed with the
    decision day ``t`` and refuses to hand back anything from row ``t`` on.
    """

    def __init__(self, pnl: np.ndarray, t: int, lookback: int):
        pnl = np.asarray(pnl, dtype=float)
        if pnl.ndim != 2:
            raise ValueError("pnl must be (T, m)")
        if t > pnl.shape[0]:
            raise LookaheadError(f"decision day {t} beyond history {pnl.shape[0]}")
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._t = int(t)
        self.lookback = int(lookback)
        self.start = max(0, self._t - self.lookback)
        # Materialise ONLY the causal slice; the future is never held.
        self._data = pnl[self.start:self._t].copy()
        self.n_models = pnl.shape[1]

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def n_obs(self) -> int:
        return self._data.shape[0]

    def is_warm(self, minimum: int) -> bool:
        return self.n_obs >= minimum

    def __repr__(self) -> str:
        return (f"CausalWindow(t={self._t}, rows={self.start}:{self._t}, "
                f"n_obs={self.n_obs}, m={self.n_models})")


def _as_window(d) -> np.ndarray:
    if isinstance(d, CausalWindow):
        return d.data
    raise TypeError(
        "aggregators require a CausalWindow, not a raw array -- this is the "
        "guard that prevents full-sample signing (see module docstring)"
    )


def _finish(w: np.ndarray, clip_negative: bool) -> np.ndarray:
    w = np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_negative:
        w = np.maximum(w, 0.0)
    return w


def normalise(w: np.ndarray) -> np.ndarray:
    """L1-normalise to sum 1. DIAGNOSTICS ONLY -- never in the live path."""
    total = np.abs(w).sum()
    return w / total if total > _EPS else np.zeros_like(w)


# --------------------------------------------------------------------------
# 1. Equal weight -- the honest baseline (no lookback, so no selection bias)
# --------------------------------------------------------------------------

def equal_weight(d, clip_negative: bool = True) -> np.ndarray:
    # clip_negative is accepted for uniform dispatch across METHODS but is
    # genuinely inert here: equal weights are never negative, so there is
    # nothing to clip and no signed variant to expose. This is also why equal
    # weight carries no selection bias -- it has no free parameter at all.
    del clip_negative
    data = _as_window(d)
    return np.ones(data.shape[1], dtype=float)


# --------------------------------------------------------------------------
# 2. Weighted average
# --------------------------------------------------------------------------

def weighted_average(d, clip_negative: bool = True) -> np.ndarray:
    data = _as_window(d)
    return _finish(data.mean(axis=0), clip_negative)


# --------------------------------------------------------------------------
# 3. Softmax -- both the documented (mean) and implemented (sum) versions
# --------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)                     # shift for numerical stability
    e = np.exp(x)
    total = e.sum()
    return e / total if total > _EPS else np.full_like(e, 1.0 / len(e))


def _softmax_scores(scores: np.ndarray, beta: float, standardize: bool) -> np.ndarray:
    """Optionally make beta unit-free by z-scoring the per-model scores.

    THIRD MISMATCH (not in the brief; measured here). beta is dimensional:
    softmax(beta*x) depends on the units of x. The article's beta=30 is
    calibrated to whatever P/L scale it used. On this book's dollar P/L
    (hundreds), beta=30 saturates completely -- measured max weight 1.0000,
    i.e. pure argmax -- whereas on fractional returns the same beta gives
    max weight 0.2007 against 1/m = 0.200, i.e. essentially equal weight.
    The SAME beta therefore spans the entire range from equal-weight to
    winner-take-all depending only on units.

    With standardize=True the scores are z-scored across models first, so beta
    means "how many cross-model standard deviations of edge halve the weight"
    and is comparable across P/L scales. Default True; pass False to reproduce
    the article's raw behaviour.
    """
    if not standardize:
        return beta * scores
    centred = scores - scores.mean()
    spread = scores.std()
    return beta * centred / spread if spread > _EPS else np.zeros_like(scores)


def softmax_mean(d, beta: float = 1.0, clip_negative: bool = True,
                 standardize: bool = True) -> np.ndarray:
    """Softmax of beta * MEAN P/L -- what the article's text describes."""
    data = _as_window(d)
    return _finish(_softmax(_softmax_scores(data.mean(axis=0), beta, standardize)),
                   clip_negative)


def softmax_sum(d, beta: float = 1.0, clip_negative: bool = True,
                standardize: bool = True) -> np.ndarray:
    """Softmax of beta * SUM P/L -- what the article's code does.

    Unstandardized, the effective temperature is beta*lookback, so this
    becomes progressively more winner-take-all as the lookback grows and any
    measured lookback sensitivity is confounded with that. Standardizing
    removes the confound, because z-scoring is invariant to the sum-vs-mean
    factor -- which is itself the cleanest way to show how much of the
    article's lookback curve was this artefact.
    """
    data = _as_window(d)
    return _finish(_softmax(_softmax_scores(data.sum(axis=0), beta, standardize)),
                   clip_negative)


# --------------------------------------------------------------------------
# 4. Mean-variance -- scipy/numpy reimplementation
# --------------------------------------------------------------------------

def mean_variance(d, gamma: float = 2.0, clip_negative: bool = True,
                  annualisation: float = ANNUALISATION) -> np.ndarray:
    """L2-regularised max-Sharpe (tangency) weights.

    Maximising w'mu / sqrt(w'Sigma w) subject to an L2 penalty gamma||w||^2
    has the closed form w propto (Sigma + gamma*I)^-1 mu. Using the analytic
    solution rather than a numerical optimiser keeps this deterministic and
    fast enough for the 10-minute budget; there is no constraint set here for
    an optimiser to buy us, because sum-to-1 has been deliberately dropped.
    """
    data = _as_window(d)
    if data.shape[0] < 2:
        return np.zeros(data.shape[1])
    mu = data.mean(axis=0) * annualisation
    cov = np.cov(data, rowvar=False) * annualisation
    cov = np.atleast_2d(cov)
    reg = cov + gamma * np.eye(cov.shape[0])
    try:
        w = np.linalg.solve(reg, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(reg) @ mu
    return _finish(w, clip_negative)


# --------------------------------------------------------------------------
# 5. Binary Kelly
# --------------------------------------------------------------------------

def binary_kelly(d, clip_negative: bool = True) -> np.ndarray:
    """f* = p/a - (1-p)/b, with p = hit rate, b = mean win, a = |mean loss|."""
    data = _as_window(d)
    n_obs, m = data.shape
    out = np.zeros(m)
    for i in range(m):
        col = data[:, i]
        wins, losses = col[col > 0.0], col[col < 0.0]
        if n_obs == 0:
            continue
        p = len(wins) / n_obs
        b = wins.mean() if len(wins) else 0.0
        a = abs(losses.mean()) if len(losses) else 0.0
        if a <= _EPS:
            # Never lost in-window: unbounded Kelly. Cap rather than divide by
            # zero -- this is a small-sample artefact, not real conviction.
            out[i] = p / _EPS if b > _EPS else 0.0
            out[i] = min(out[i], 1.0e6)
        elif b <= _EPS:
            out[i] = -(1.0 - p) / _EPS
            out[i] = max(out[i], -1.0e6)
        else:
            out[i] = p / a - (1.0 - p) / b
    return _finish(out, clip_negative)


# --------------------------------------------------------------------------
# 6. Continuous Kelly, and the thing the article actually computed
# --------------------------------------------------------------------------

def kelly_continuous(d, clip_negative: bool = True) -> np.ndarray:
    """Thorp (1992) continuous Kelly: f* = mu / sigma^2 (r omitted)."""
    data = _as_window(d)
    if data.shape[0] < 2:
        return np.zeros(data.shape[1])
    mu = data.mean(axis=0)
    var = data.var(axis=0, ddof=1)
    return _finish(np.divide(mu, var, out=np.zeros_like(mu), where=var > _EPS),
                   clip_negative)


def sharpe_weighted(d, clip_negative: bool = True) -> np.ndarray:
    """mu / sigma -- what the article's 'non-binary Kelly' code computes.

    This is the Sharpe ratio, not Kelly. Named honestly. It differs from
    kelly_continuous by a factor 1/sigma, so after normalisation the two
    coincide IFF all model volatilities are equal -- confirm cross-model vol
    dispersion exists before treating them as distinct methods.

    ERRATA E4 -- prior: for diagonal Sigma the max-Sharpe tangency weights are
    proportional to mu/sigma^2, so ``kelly_continuous`` is the theoretically
    aligned one for an SR-driven objective and this is not. Keep both, but
    expect kelly_continuous to be the better-motivated of the pair.
    """
    data = _as_window(d)
    if data.shape[0] < 2:
        return np.zeros(data.shape[1])
    mu = data.mean(axis=0)
    sd = data.std(axis=0, ddof=1)
    return _finish(np.divide(mu, sd, out=np.zeros_like(mu), where=sd > _EPS),
                   clip_negative)


# --------------------------------------------------------------------------
# 7. Running count -- the article's own proposal
# --------------------------------------------------------------------------

def running_count(d, loss_penalty: float = 2.0,
                  clip_negative: bool = True) -> np.ndarray:
    """+1 per winning period, -loss_penalty per losing period.

    ``loss_penalty`` is the article's hardcoded 2 exposed as a parameter, so
    its sensitivity can be measured (brief PART 4 task 4). Discontinuous by
    construction -- an integer counter with a clip -- hence the expectation of
    the highest turnover of the seven.
    """
    data = _as_window(d)
    wins = (data > 0.0).sum(axis=0).astype(float)
    losses = (data < 0.0).sum(axis=0).astype(float)
    return _finish(wins - loss_penalty * losses, clip_negative)


METHODS = {
    "equal_weight": equal_weight,
    "weighted_average": weighted_average,
    "softmax_mean": softmax_mean,
    "softmax_sum": softmax_sum,
    "mean_variance": mean_variance,
    "binary_kelly": binary_kelly,
    "kelly_continuous": kelly_continuous,
    "sharpe_weighted": sharpe_weighted,
    "running_count": running_count,
}
