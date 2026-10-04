"""Functional JAX pair backtest with static shapes for jit / vmap.

Mirrors the single-pair path in testbedv3.py:
  Kalman + CUSUM adaptive filter -> rolling OU MLE -> z-entry state machine -> PnL.

Dynamic lengths (ou_window, history) are expressed as values + masks over fixed
buffers so every array shape is compile-time constant.
"""

from __future__ import annotations

import os
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Static compile-time sizes (must cover SEARCH_SPACES / eval windows).
MAX_OU = 200
INIT_WINDOW = 20
WARMUP = 30
SERIES_N = 30
BISECT_ITERS = 32
N_INST = 51
LEG = 10_000.0
TRANSACTION_COST = 1e-4
EPS_DENOM = 1e-12
EPS_VAR = 1e-18
SCORE_PARAM = 1.0

# Param vector layout (vmap axis).
PARAM_KEYS = (
    "q_beta",
    "h",
    "cusum_k",
    "max_cov_inflate",
    "cov_decay",
    "kalman_var_ewma_alpha",
    "ou_window",
    "kappa_ewma_alpha",
    "sigma_ewma_alpha",
)


class FilterState(NamedTuple):
    beta: jnp.ndarray
    p22: jnp.ndarray
    innov_var: jnp.ndarray
    resid_var: jnp.ndarray
    cm_up: jnp.ndarray
    cm_dn: jnp.ndarray
    cv_up: jnp.ndarray
    cv_dn: jnp.ndarray
    run_m_up: jnp.ndarray
    run_m_dn: jnp.ndarray
    run_v_up: jnp.ndarray
    run_v_dn: jnp.ndarray
    seg_start: jnp.ndarray
    decay_multiplier: jnp.ndarray
    y_buf: jnp.ndarray  # (MAX_OU,)
    x_buf: jnp.ndarray
    n_hist: jnp.ndarray
    beta_init: jnp.ndarray  # 0/1


class AgentState(NamedTuple):
    filt: FilterState
    side: jnp.ndarray
    kappa: jnp.ndarray
    sigma: jnp.ndarray
    theta: jnp.ndarray
    a_star: jnp.ndarray
    spread_buf: jnp.ndarray  # (MAX_OU,)
    n_spread: jnp.ndarray
    z_innov: jnp.ndarray


class PnLState(NamedTuple):
    cash: jnp.ndarray
    pos_y: jnp.ndarray
    pos_x: jnp.ndarray
    value: jnp.ndarray
    comm: jnp.ndarray
    tot_dvolume: jnp.ndarray


def params_to_vec(params: dict) -> jnp.ndarray:
    return jnp.asarray([float(params[k]) for k in PARAM_KEYS], dtype=jnp.float64)


def vec_to_params(vec, pair_y: str, pair_x: str) -> dict:
    out = {k: float(vec[i]) for i, k in enumerate(PARAM_KEYS)}
    out["ou_window"] = int(round(out["ou_window"]))
    out["pair_y"] = pair_y
    out["pair_x"] = pair_x
    return out


def _init_filter() -> FilterState:
    z = jnp.zeros((), dtype=jnp.float64)
    neg = jnp.asarray(-1, dtype=jnp.int32)
    return FilterState(
        beta=jnp.asarray(1.0, dtype=jnp.float64),
        p22=jnp.asarray(1.0, dtype=jnp.float64),
        innov_var=jnp.asarray(0.0, dtype=jnp.float64),
        resid_var=jnp.asarray(0.0, dtype=jnp.float64),
        cm_up=z, cm_dn=z, cv_up=z, cv_dn=z,
        run_m_up=neg, run_m_dn=neg, run_v_up=neg, run_v_dn=neg,
        seg_start=jnp.asarray(0, dtype=jnp.int32),
        decay_multiplier=jnp.asarray(1.0, dtype=jnp.float64),
        y_buf=jnp.zeros(MAX_OU, dtype=jnp.float64),
        x_buf=jnp.zeros(MAX_OU, dtype=jnp.float64),
        n_hist=jnp.asarray(0, dtype=jnp.int32),
        beta_init=jnp.asarray(0, dtype=jnp.int32),
    )


def _init_agent() -> AgentState:
    return AgentState(
        filt=_init_filter(),
        side=jnp.asarray(0, dtype=jnp.int32),
        kappa=jnp.asarray(0.1, dtype=jnp.float64),
        sigma=jnp.asarray(0.02, dtype=jnp.float64),
        theta=jnp.asarray(0.0, dtype=jnp.float64),
        a_star=jnp.asarray(1.0, dtype=jnp.float64),
        spread_buf=jnp.zeros(MAX_OU, dtype=jnp.float64),
        n_spread=jnp.asarray(0, dtype=jnp.int32),
        z_innov=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _push_buf(buf: jnp.ndarray, val: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([buf[1:], val[None]])


def _ols_beta(y_buf, x_buf, n_hist) -> jnp.ndarray:
    """OLS beta on the last INIT_WINDOW samples of the fixed buffers."""
    idxs = jnp.arange(MAX_OU)
    start = MAX_OU - INIT_WINDOW
    in_win = (idxs >= start) & (idxs >= (MAX_OU - n_hist))
    xs = jnp.where(in_win, x_buf, 0.0)
    ys = jnp.where(in_win, y_buf, 0.0)
    denom = jnp.dot(xs, xs)
    safe = denom > EPS_DENOM
    return jnp.where(safe, jnp.dot(xs, ys) / jnp.maximum(denom, EPS_DENOM), 1.0)


def _evaluate_ou_series(a, eff_c):
    n = jnp.arange(SERIES_N, dtype=jnp.float64)
    # Gamma(n+1/2) = (2n-1)!!/2^n * sqrt(pi); use lax.lgamma for stability.
    gamma_val = jnp.exp(jax.lax.lgamma(n + 0.5))
    sqrt2_a = jnp.sqrt(2.0) * a
    # Powers via exp(log) with care at a~0.
    log_base = jnp.log(jnp.maximum(sqrt2_a, 1e-30))
    # term1: (sqrt2 a)^(2n+1) / (2n+1)! * gamma
    log_t1 = (2.0 * n + 1.0) * log_base - jax.lax.lgamma(2.0 * n + 2.0) + jnp.log(gamma_val)
    # term2: (sqrt2 a)^(2n) / (2n)! * gamma
    log_t2 = (2.0 * n) * log_base - jax.lax.lgamma(2.0 * n + 1.0) + jnp.log(gamma_val)
    s1 = jnp.sum(jnp.exp(log_t1))
    s2 = jnp.sum(jnp.exp(log_t2))
    return 0.5 * s1 - (a - eff_c) * (jnp.sqrt(2.0) / 2.0) * s2


def _optimal_z_entry(kappa, sigma, cost_real, rule_new):
    """Fixed-iteration bisection stand-in for scipy Brentq."""
    safe = (kappa > 0) & (sigma > 0)
    c = cost_real * jnp.sqrt(2.0 * kappa) / jnp.maximum(sigma, 1e-12)
    eff_c = jnp.where(rule_new, c * 0.5, c)
    lo = eff_c + 1e-5
    hi = jnp.maximum(eff_c + 10.0, 5.0)
    fallback = jnp.maximum(1.0, eff_c + 0.5)

    def body(_, state):
        lo_, hi_ = state
        mid = 0.5 * (lo_ + hi_)
        fmid = _evaluate_ou_series(mid, eff_c)
        # F(a) changes - -> + around the root; keep the half where sign flips from F(lo).
        flo = _evaluate_ou_series(lo_, eff_c)
        go_right = flo * fmid > 0
        lo_n = jnp.where(go_right, mid, lo_)
        hi_n = jnp.where(go_right, hi_, mid)
        return lo_n, hi_n

    lo_f, hi_f = jax.lax.fori_loop(0, BISECT_ITERS, body, (lo, hi))
    root = 0.5 * (lo_f + hi_f)
    return jnp.where(safe, root, fallback)


def _filter_step(filt: FilterState, yf, xf, t, q_beta, h, cusum_k, max_cov_inflate, cov_decay, kalman_alpha):
    y_buf = _push_buf(filt.y_buf, yf)
    x_buf = _push_buf(filt.x_buf, xf)
    n_hist = jnp.minimum(filt.n_hist + 1, MAX_OU)

    beta0 = _ols_beta(y_buf, x_buf, n_hist)
    beta = jnp.where(filt.beta_init == 0, beta0, filt.beta)
    seg_start = jnp.where(filt.beta_init == 0, t, filt.seg_start)
    beta_init = jnp.asarray(1, dtype=jnp.int32)

    innov_pred = yf - beta * xf
    lam = 1.0 - kalman_alpha
    innov_var = jnp.where(
        filt.n_hist == 0,
        innov_pred * innov_pred,
        lam * filt.innov_var + (1.0 - lam) * innov_pred * innov_pred,
    )
    resid_var = jnp.where(
        filt.n_hist == 0,
        innov_pred * innov_pred,
        lam * filt.resid_var + (1.0 - lam) * innov_pred * innov_pred,
    )
    z = jnp.where(innov_var > 0, innov_pred / jnp.sqrt(jnp.maximum(innov_var, EPS_VAR)), 0.0)

    # CUSUM (only after warmup).
    prev_up, prev_dn = filt.cm_up, filt.cm_dn
    cm_up = jnp.maximum(0.0, filt.cm_up + z - cusum_k)
    cm_dn = jnp.maximum(0.0, filt.cm_dn - z - cusum_k)
    run_m_up = jnp.where((cm_up > 0) & (prev_up == 0), t, jnp.where(cm_up == 0, -1, filt.run_m_up))
    run_m_dn = jnp.where((cm_dn > 0) & (prev_dn == 0), t, jnp.where(cm_dn == 0, -1, filt.run_m_dn))

    d_v = (z * z - 1.0) / jnp.sqrt(2.0)
    prev_v_up, prev_v_dn = filt.cv_up, filt.cv_dn
    cv_up = jnp.maximum(0.0, filt.cv_up + d_v - cusum_k)
    cv_dn = jnp.maximum(0.0, filt.cv_dn - d_v - cusum_k)
    run_v_up = jnp.where((cv_up > 0) & (prev_v_up == 0), t, jnp.where(cv_up == 0, -1, filt.run_v_up))
    run_v_dn = jnp.where((cv_dn > 0) & (prev_v_dn == 0), t, jnp.where(cv_dn == 0, -1, filt.run_v_dn))

    mean_alarm = (cm_up > h) | (cm_dn > h)
    vol_alarm = (cv_up > h) | (cv_dn > h)
    active = t >= WARMUP
    mean_alarm = mean_alarm & active
    vol_alarm = vol_alarm & active
    cut = mean_alarm | vol_alarm

    # Trigger segment cut bookkeeping (vol_cut=True path).
    best = jnp.asarray(-1.0, dtype=jnp.float64)
    cut_bar = t
    use = (cm_up > h) & (cm_up > best)
    best = jnp.where(use, cm_up, best)
    cut_bar = jnp.where(use, run_m_up, cut_bar)
    use = (cm_dn > h) & (cm_dn > best)
    best = jnp.where(use, cm_dn, best)
    cut_bar = jnp.where(use, run_m_dn, cut_bar)
    use = (cv_up > h) & (cv_up > best)
    best = jnp.where(use, cv_up, best)
    cut_bar = jnp.where(use, run_v_up, cut_bar)
    use = (cv_dn > h) & (cv_dn > best)
    cut_bar = jnp.where(use, run_v_dn, cut_bar)
    cut_bar = jnp.maximum(cut_bar, seg_start + 1)

    cm_up_c = jnp.where(cut, jnp.maximum(0.0, cm_up - h), cm_up)
    cm_dn_c = jnp.where(cut, jnp.maximum(0.0, cm_dn - h), cm_dn)
    cv_up_c = jnp.where(cut, jnp.maximum(0.0, cv_up - h), cv_up)
    cv_dn_c = jnp.where(cut, jnp.maximum(0.0, cv_dn - h), cv_dn)
    run_m_up_c = jnp.where(cut, -1, run_m_up)
    run_m_dn_c = jnp.where(cut, -1, run_m_dn)
    run_v_up_c = jnp.where(cut, -1, run_v_up)
    run_v_dn_c = jnp.where(cut, -1, run_v_dn)
    seg_start_c = jnp.where(cut, cut_bar, seg_start)

    max_cusum = jnp.maximum(jnp.maximum(cm_up_c, cm_dn_c), jnp.maximum(cv_up_c, cv_dn_c))
    excess = jnp.maximum(0.0, max_cusum - h)
    target_mult = jnp.where(
        cut,
        max_cov_inflate,
        jnp.where(
            excess > 0,
            1.0 + (max_cov_inflate - 1.0) * (1.0 - jnp.exp(-excess)),
            1.0,
        ),
    )
    # Before warmup, force target 1 like inactive CUSUM.
    target_mult = jnp.where(active, target_mult, 1.0)
    decay_multiplier = jnp.maximum(target_mult, filt.decay_multiplier * (1.0 - cov_decay))
    dynamic_q = q_beta * decay_multiplier

    # Kalman update.
    pp22 = filt.p22 + dynamic_q
    innov = yf - beta * xf
    s_pred = jnp.maximum(pp22 * xf * xf + 1.0, EPS_DENOM)
    g = (pp22 * xf) / s_pred
    beta_new = beta + g * innov
    a = 1.0 - g * xf
    p22_new = jnp.maximum(a * a * pp22 + g * g, EPS_VAR)
    resid = yf - beta_new * xf

    new_filt = FilterState(
        beta=beta_new, p22=p22_new, innov_var=innov_var, resid_var=resid_var,
        cm_up=cm_up_c, cm_dn=cm_dn_c, cv_up=cv_up_c, cv_dn=cv_dn_c,
        run_m_up=run_m_up_c, run_m_dn=run_m_dn_c, run_v_up=run_v_up_c, run_v_dn=run_v_dn_c,
        seg_start=seg_start_c, decay_multiplier=decay_multiplier,
        y_buf=y_buf, x_buf=x_buf, n_hist=n_hist, beta_init=beta_init,
    )
    return new_filt, resid, z


def _ou_update(agent: AgentState, spread, ou_window, kappa_alpha, sigma_alpha, rule_new):
    spread_buf = _push_buf(agent.spread_buf, spread)
    n_spread = jnp.minimum(agent.n_spread + 1, MAX_OU)
    w = jnp.clip(ou_window.astype(jnp.int32), 2, MAX_OU)
    ready = n_spread >= w

    idxs = jnp.arange(MAX_OU)
    # Last w samples occupy buf[MAX_OU-w : MAX_OU].
    in_win = idxs >= (MAX_OU - w)
    # Lag pairs: (i, i+1) both in window.
    lag_ok = (idxs[:-1] >= (MAX_OU - w)) & (idxs[1:] >= (MAX_OU - w))
    s_lag = jnp.where(lag_ok, spread_buf[:-1], 0.0)
    s_t = jnp.where(lag_ok, spread_buf[1:], 0.0)
    n_pairs = jnp.maximum(jnp.sum(lag_ok.astype(jnp.float64)), 1.0)
    mean_lag = jnp.sum(s_lag) / n_pairs
    mean_t = jnp.sum(s_t) / n_pairs
    denom = jnp.sum(jnp.where(lag_ok, (spread_buf[:-1] - mean_lag) ** 2, 0.0))
    valid = denom > 1e-8
    b_mle = jnp.sum(jnp.where(lag_ok, (spread_buf[:-1] - mean_lag) * (spread_buf[1:] - mean_t), 0.0)) / jnp.maximum(denom, 1e-8)
    b_mle = jnp.clip(b_mle, 1e-4, 0.9999)
    a_mle = mean_t - b_mle * mean_lag
    residuals = jnp.where(lag_ok, spread_buf[1:] - (a_mle + b_mle * spread_buf[:-1]), 0.0)
    sigma_eps = jnp.sqrt(jnp.sum(residuals ** 2) / jnp.maximum(n_pairs - 2.0, 1.0))
    raw_kappa = -jnp.log(b_mle)
    raw_sigma = sigma_eps * jnp.sqrt((2.0 * raw_kappa) / jnp.maximum(1.0 - jnp.exp(-2.0 * raw_kappa), 1e-12))
    theta_raw = a_mle / jnp.maximum(1.0 - b_mle, 1e-12)

    kappa_lam = 1.0 - kappa_alpha
    sigma_lam = 1.0 - sigma_alpha
    kappa = jnp.where(ready & valid, kappa_lam * agent.kappa + (1.0 - kappa_lam) * raw_kappa, agent.kappa)
    sigma = jnp.where(ready & valid, sigma_lam * agent.sigma + (1.0 - sigma_lam) * raw_sigma, agent.sigma)
    theta = jnp.where(ready & valid, kappa_lam * agent.theta + (1.0 - kappa_lam) * theta_raw, agent.theta)
    a_star = jnp.where(
        ready & valid,
        _optimal_z_entry(kappa, sigma, TRANSACTION_COST * 2.0, rule_new),
        agent.a_star,
    )
    return agent._replace(
        spread_buf=spread_buf, n_spread=n_spread, kappa=kappa, sigma=sigma, theta=theta, a_star=a_star,
    )


def _side_update(side, spread_z, a_star, rule_new):
    """Position state machine; rule_new=True -> 'new', else 'conventional'."""
    flat = side == 0
    long_ = side == 1
    short = side == -1

    # Flat entries.
    enter_short = flat & (spread_z >= a_star)
    enter_long = flat & (spread_z <= -a_star)

    # Conventional: exit at mean.
    exit_long_c = long_ & (~rule_new) & (spread_z >= 0.0)
    exit_short_c = short & (~rule_new) & (spread_z <= 0.0)

    # New: flip at opposite threshold.
    flip_to_short = long_ & rule_new & (spread_z >= a_star)
    flip_to_long = short & rule_new & (spread_z <= -a_star)

    side_n = side
    side_n = jnp.where(enter_short, -1, side_n)
    side_n = jnp.where(enter_long, 1, side_n)
    side_n = jnp.where(exit_long_c | exit_short_c, 0, side_n)
    side_n = jnp.where(flip_to_short, -1, side_n)
    side_n = jnp.where(flip_to_long, 1, side_n)
    return side_n.astype(jnp.int32)


def _dollar_positions(side, beta, z_innov):
    scale = LEG / jnp.maximum(1.0, jnp.abs(beta))
    pos_y = jnp.where(side == 0, 0.0, scale * side.astype(jnp.float64))
    pos_x = jnp.where(side == 0, 0.0, -beta * pos_y)
    return pos_y, pos_x, jnp.abs(z_innov)


def _agent_step(agent: AgentState, y_raw, x_raw, t, pvec, rule_new):
    q_beta, h, cusum_k, max_cov_inflate, cov_decay, kalman_alpha, ou_window, kappa_alpha, sigma_alpha = (
        pvec[0], pvec[1], pvec[2], pvec[3], pvec[4], pvec[5], pvec[6], pvec[7], pvec[8]
    )
    yf = jnp.log(jnp.maximum(y_raw, EPS_DENOM))
    xf = jnp.log(jnp.maximum(x_raw, EPS_DENOM))
    filt, spread, z = _filter_step(
        agent.filt, yf, xf, t, q_beta, h, cusum_k, max_cov_inflate, cov_decay, kalman_alpha,
    )
    agent = agent._replace(filt=filt, z_innov=z)
    agent = _ou_update(agent, spread, ou_window, kappa_alpha, sigma_alpha, rule_new)

    ou_std = agent.sigma / jnp.sqrt(jnp.maximum(2.0 * agent.kappa, 1e-12))
    spread_z = jnp.where(
        agent.kappa > 0,
        (spread - agent.theta) / jnp.maximum(ou_std, 1e-8),
        0.0,
    )
    side = _side_update(agent.side, spread_z, agent.a_star, rule_new)
    agent = agent._replace(side=side)
    pos_y, pos_x, _ = _dollar_positions(side, filt.beta, z)
    return agent, pos_y, pos_x


def _shares_from_dollars(pos_y_d, pos_x_d, py, px, y_ret, x_ret):
    """Crash flatten + clip + round to integer shares."""
    pos_y_d = jnp.where(jnp.abs(y_ret) > 0.40, 0.0, pos_y_d)
    pos_x_d = jnp.where(jnp.abs(x_ret) > 0.40, 0.0, pos_x_d)
    pos_y_d = jnp.clip(pos_y_d, -LEG, LEG)
    pos_x_d = jnp.clip(pos_x_d, -LEG, LEG)
    sy = jnp.round(pos_y_d / jnp.maximum(py, EPS_DENOM))
    sx = jnp.round(pos_x_d / jnp.maximum(px, EPS_DENOM))
    # Position dollar limits in share space (official dlrPosLimit=10k).
    lim_y = jnp.floor(10_000.0 / jnp.maximum(py, EPS_DENOM))
    lim_x = jnp.floor(10_000.0 / jnp.maximum(px, EPS_DENOM))
    return jnp.clip(sy, -lim_y, lim_y), jnp.clip(sx, -lim_x, lim_x)


def score_mu_sigma(mu, sigma, param=SCORE_PARAM):
    sr = jnp.sqrt(250.0) * mu / jnp.maximum(sigma, 1e-12)
    frac = (sr * sr) / (sr * sr + param * param)
    return jnp.where((mu <= 0) | (sigma < 1e-10), mu, mu * frac)


def backtest_pair(
    pvec: jnp.ndarray,
    y_prices: jnp.ndarray,
    x_prices: jnp.ndarray,
    start_day: int,
    end_day: int,
    rule_new: bool = True,
    comm_rate: float = 1e-4,
):
    """Causal pair backtest over prices[0..T).

    Days are 1-indexed like the official eval: scored PLL covers start_day..end_day.
    Agent warms up on bars before the window (mirrors catch_up on prices[:, :t]).
    `start_day` / `end_day` must be Python ints (static shapes for scan).
    """
    start_day = int(start_day)
    end_day = int(end_day)
    start_j = jnp.asarray(start_day, dtype=jnp.int32)
    end_j = jnp.asarray(end_day, dtype=jnp.int32)
    rule_new_j = jnp.asarray(rule_new, dtype=jnp.bool_)

    def step(carry, t):
        agent, pnl = carry
        # Official loop variable t: history is prices[:, :t], current bar index t-1.
        t = t.astype(jnp.int32)
        py = y_prices[t - 1]
        px = x_prices[t - 1]
        py_prev = y_prices[jnp.maximum(t - 2, 0)]
        px_prev = x_prices[jnp.maximum(t - 2, 0)]
        y_ret = jnp.where(
            t >= 2,
            jnp.log(jnp.maximum(py, EPS_DENOM)) - jnp.log(jnp.maximum(py_prev, EPS_DENOM)),
            0.0,
        )
        x_ret = jnp.where(
            t >= 2,
            jnp.log(jnp.maximum(px, EPS_DENOM)) - jnp.log(jnp.maximum(px_prev, EPS_DENOM)),
            0.0,
        )

        agent, dy, dx = _agent_step(agent, py, px, t - 1, pvec, rule_new_j)
        warm = t >= (WARMUP + 10)
        dy = jnp.where(warm, dy, 0.0)
        dx = jnp.where(warm, dx, 0.0)

        sy, sx = _shares_from_dollars(dy, dx, py, px, y_ret, x_ret)
        at_end = t >= end_j
        sy = jnp.where(at_end, pnl.pos_y, sy)
        sx = jnp.where(at_end, pnl.pos_x, sx)

        # Positions stay flat until the official window; first active bar trades 0 -> target.
        active = t >= (start_j - 1)

        delta_y = jnp.where(active, sy - pnl.pos_y, 0.0)
        delta_x = jnp.where(active, sx - pnl.pos_x, 0.0)
        trade_notional = py * delta_y + px * delta_x
        cash = jnp.where(active, pnl.cash - trade_notional - pnl.comm, pnl.cash)
        dvol = jnp.abs(py * delta_y) + jnp.abs(px * delta_x)
        tot_dvolume = jnp.where(active, pnl.tot_dvolume + dvol, pnl.tot_dvolume)
        comm = jnp.where(active, dvol * comm_rate, 0.0)
        pos_y = jnp.where(active, sy, pnl.pos_y)
        pos_x = jnp.where(active, sx, pnl.pos_x)
        pos_value = pos_y * py + pos_x * px
        today_pl = jnp.where(active, cash + pos_value - pnl.value, 0.0)
        value = jnp.where(active, cash + pos_value, pnl.value)
        pnl = PnLState(
            cash=cash, pos_y=pos_y, pos_x=pos_x, value=value, comm=comm, tot_dvolume=tot_dvolume,
        )

        in_score = (t >= start_j) & (t <= end_j)
        scored_pl = jnp.where(in_score, today_pl, 0.0)
        scored_flag = in_score.astype(jnp.float64)
        return (agent, pnl), (scored_pl, scored_flag)

    ts = jnp.arange(1, end_day + 1, dtype=jnp.int32)
    init_pnl = PnLState(
        cash=jnp.asarray(0.0, dtype=jnp.float64),
        pos_y=jnp.asarray(0.0, dtype=jnp.float64),
        pos_x=jnp.asarray(0.0, dtype=jnp.float64),
        value=jnp.asarray(0.0, dtype=jnp.float64),
        comm=jnp.asarray(0.0, dtype=jnp.float64),
        tot_dvolume=jnp.asarray(0.0, dtype=jnp.float64),
    )
    (_, final_pnl), (pls, flags) = jax.lax.scan(step, (_init_agent(), init_pnl), ts)

    n = jnp.maximum(jnp.sum(flags), 1.0)
    mu = jnp.sum(pls) / n
    centered = jnp.where(flags > 0, pls - mu, 0.0)
    sigma = jnp.sqrt(jnp.sum(centered * centered) / n)
    sc = score_mu_sigma(mu, sigma)
    sc = jnp.where(final_pnl.tot_dvolume == 0.0, sc - 5.0, sc)
    sharpe = jnp.where(sigma > 0, jnp.sqrt(250.0) * mu / sigma, 0.0)
    return {
        "score": sc,
        "mean": mu,
        "std": sigma,
        "sharpe": sharpe,
        "dvolume": final_pnl.tot_dvolume,
        "pll": pls,
        "flags": flags,
    }


def backtest_score(pvec, y_prices, x_prices, start_day, end_day, rule_new=True):
    return backtest_pair(pvec, y_prices, x_prices, start_day, end_day, rule_new)["score"]


def make_backtest_fns(y_prices: np.ndarray, x_prices: np.ndarray, start_day: int, end_day: int, rule_new: bool = True):
    """Build jit/vmap closures with prices + window captured as constants."""
    y = jnp.asarray(y_prices, dtype=jnp.float64)
    x = jnp.asarray(x_prices, dtype=jnp.float64)

    @jax.jit
    def score_one(pvec):
        return backtest_score(pvec, y, x, start_day, end_day, rule_new)

    @jax.jit
    def score_batch(pvecs):
        return jax.vmap(score_one)(pvecs)

    @jax.jit
    def full_one(pvec):
        return backtest_pair(pvec, y, x, start_day, end_day, rule_new)

    return score_one, score_batch, full_one


# ---------------------------------------------------------------------------
# Combinatorial purged CV (static path masks + jax scoring)
# ---------------------------------------------------------------------------
def split_into_groups(n_days: int, n_groups: int) -> list[np.ndarray]:
    edges = np.linspace(0, n_days, n_groups + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]


def build_cpcv_masks(
    n_days: int,
    n_groups: int,
    n_test_groups: int,
    purge: int,
    pct_embargo: float,
) -> np.ndarray:
    """Return bool masks of shape (n_paths, n_days) for combinatorial test paths."""
    from itertools import combinations

    groups = split_into_groups(n_days, n_groups)
    emb = int(np.ceil(n_days * max(0.0, pct_embargo)))
    masks = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        keep = np.zeros(n_days, dtype=bool)
        for gid in test_ids:
            keep[groups[gid]] = True
        selected = sorted(test_ids)
        runs: list[list[int]] = []
        for gid in selected:
            if not runs or gid != runs[-1][-1] + 1:
                runs.append([gid])
            else:
                runs[-1].append(gid)
        for run in runs:
            first_pos = int(groups[run[0]][0])
            last_pos = int(groups[run[-1]][-1])
            if run[0] > 0 and purge > 0:
                keep[first_pos: first_pos + purge] = False
            if run[-1] < n_groups - 1 and purge > 0:
                lo = max(first_pos, last_pos - purge + 1)
                keep[lo: last_pos + 1] = False
            if emb > 0:
                keep[last_pos + 1: last_pos + 1 + emb] = False
        masks.append(keep)
    return np.stack(masks, axis=0)


@jax.jit
def path_score_from_mask(pll: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    n = jnp.maximum(jnp.sum(mask.astype(jnp.float64)), 1.0)
    mu = jnp.sum(jnp.where(mask, pll, 0.0)) / n
    centered = jnp.where(mask, pll - mu, 0.0)
    sigma = jnp.sqrt(jnp.sum(centered * centered) / n)
    return score_mu_sigma(mu, sigma)


def cpcv_path_scores(pll: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda m: path_score_from_mask(pll, m))(masks)


def cpcv_objective(pll: jnp.ndarray, masks: jnp.ndarray, std_lambda: float = 1.0) -> jnp.ndarray:
    scores = cpcv_path_scores(pll, masks)
    return jnp.mean(scores) - std_lambda * jnp.std(scores)


def make_cpcv_fns(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    start_day: int,
    end_day: int,
    masks: np.ndarray,
    rule_new: bool = True,
    std_lambda: float = 1.0,
):
    """Jitted single/batch CPCV objectives. masks: (P, n_scored_days)."""
    y = jnp.asarray(y_prices, dtype=jnp.float64)
    x = jnp.asarray(x_prices, dtype=jnp.float64)
    masks_j = jnp.asarray(masks, dtype=jnp.bool_)
    n_score = end_day - start_day + 1
    if masks.shape[1] != n_score:
        raise ValueError(f"masks second dim {masks.shape[1]} != scored days {n_score}")
    # Scan runs t=1..end_day; scored rows sit at indices [start_day-1, end_day).
    lo = start_day - 1
    hi = end_day

    @jax.jit
    def objective_one(pvec):
        out = backtest_pair(pvec, y, x, start_day, end_day, rule_new)
        pll_scored = out["pll"][lo:hi]
        return cpcv_objective(pll_scored, masks_j, std_lambda)

    @jax.jit
    def objective_batch(pvecs):
        return jax.vmap(objective_one)(pvecs)

    return objective_one, objective_batch
