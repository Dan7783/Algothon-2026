"""Synthetic-data and NULL testing for four_sigmas.py.

Question: is the live book's score on days 751-1000 distinguishable from what the
same machine would earn on data containing no exploitable structure?

We answer it by Monte Carlo. Each replication builds a full 1000-day synthetic
price matrix, replays the book over it with eval.py's exact accounting, and
records the score. The distribution of those scores is the null distribution;
the real score is compared against it.

Generators (all preserve day-1 price levels, so the dollar position limits and
integer-share rounding bite exactly as they do on real data):

  iid_perm    NULL. Independent time permutation of each asset's log returns.
              Kills autocorrelation, cross-correlation, cointegration, lead-lag.
              Preserves each asset's return marginal exactly (and hence drift).

  xs_perm     NULL. One common time permutation applied to every asset at once.
              Preserves the contemporaneous cross-sectional covariance matrix
              and every marginal exactly; destroys ALL temporal dependence.
              This is the sharpest null for a pairs/lead-lag book: everything
              static about the cross-section survives, only time structure dies.

  block_boot  NULL-ish. Stationary (geometric) block bootstrap on whole return
              vectors, mean block 20d. Keeps short-run dynamics and the cross
              section; destroys long-run cointegrating relationships.

  gauss_corr  NULL. Parametric. Multivariate Gaussian matched to the real mean
              vector and covariance matrix. Cross-check that resampling artefacts
              are not driving the result.

  phase_surr  SURROGATE, not a null. Multivariate phase randomisation with phases
              shared across assets: preserves the entire cross-spectral density,
              i.e. every auto- and cross-covariance at every lag, including the
              linear cointegration structure. Destroys nonlinearity, non-Gaussian
              higher moments, and the specific realised path. Gives a parametric
              bootstrap CI for the score under "the linear structure is real".

  coint_synth POSITIVE CONTROL. Factor model plus genuine OU-cointegrated pairs
              imposed on a disjoint subset of the book's own configured pairs.
              kappa=0 degenerates to a random-walk null; kappa>0 has real,
              known mean reversion. Confirms the harness can detect edge, and
              measures how much structure the score implies.

Run from repo root:
    python research/synthetic_null.py --mode null    --reps 200 --workers 7
    python research/synthetic_null.py --mode control --reps 30  --workers 7
"""
from __future__ import annotations

import argparse
import os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

OUT = _ROOT / "analysis_outputs" / "synthetic_null"

# eval.py constants, replicated verbatim
DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000
TEST_START = 751
TEST_END = 1000

_REAL: np.ndarray | None = None


def real_prices() -> np.ndarray:
    """(nInst, nDays) real price matrix, cached per process."""
    global _REAL
    if _REAL is None:
        _REAL = pd.read_csv(
            _ROOT / "prices.txt", sep=r"\s+", header=0
        ).to_numpy(float).T
    return _REAL


def eval_score(mu: float, sigma: float, param: float = 1.0) -> float:
    """eval.py score(): mean PnL discounted by a Sharpe-based fraction."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


# --------------------------------------------------------------------------
# accounting: a verbatim replica of eval.py calcPL, including the one-day
# commission lag (yesterday's comm is deducted from today's cash) and the
# truncating .astype(int) on the share limits.
# --------------------------------------------------------------------------
def backtest(
    prc: np.ndarray,
    get_position,
    start: int = TEST_START,
    end: int = TEST_END,
) -> dict[str, float]:
    n_inst = prc.shape[0]
    comm_rate = np.full(n_inst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_limit = np.full(n_inst, float(DEFAULT_DLR_LIMIT))
    dlr_limit[0] = float(INST0_DLR_LIMIT)

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    today_pl: list[float] = []
    gross: list[float] = []
    net: list[float] = []

    for t in range(start - 1, end + 1):
        hist = prc[:, :t]
        cur_prices = hist[:, -1]

        if t < end:
            raw = get_position(hist)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(raw, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        dvolumes = cur_prices * np.abs(delta)
        tot_dvolume += float(np.sum(dvolumes))
        comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        pl = cash + pos_value - value
        value = cash + pos_value

        if t >= start:
            today_pl.append(pl)
            gross.append(float(np.abs(cur_pos).dot(cur_prices)))
            net.append(pos_value)

    pll = np.array(today_pl)
    mu = float(np.mean(pll))
    sd = float(np.std(pll))
    sharpe = float(np.sqrt(250) * mu / sd) if sd > 0 else 0.0
    return {
        "score": float(eval_score(mu, sd)),
        "mean_pl": mu,
        "std_pl": sd,
        "sharpe": sharpe,
        "dvolume": tot_dvolume,
        "mean_gross": float(np.mean(gross)),
        "mean_net": float(np.mean(net)),
    }


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def _prices_from_logret(logret: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Rebuild a positive price matrix from log returns and day-1 prices."""
    log_p = np.log(p0)[:, None] + np.cumsum(logret, axis=1)
    return np.column_stack([p0, np.exp(log_p)])


def gen_iid_perm(rng: np.random.Generator) -> np.ndarray:
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    out = np.empty_like(r)
    for i in range(r.shape[0]):
        out[i] = rng.permutation(r[i])
    return _prices_from_logret(out, prc[:, 0])


def gen_xs_perm(rng: np.random.Generator) -> np.ndarray:
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    return _prices_from_logret(r[:, rng.permutation(r.shape[1])], prc[:, 0])


def gen_iid_boot(rng: np.random.Generator) -> np.ndarray:
    """Per-asset i.i.d. bootstrap WITH replacement.

    The with-replacement versions exist because a permutation preserves the sum
    of log returns exactly, so every permuted path lands on the real terminal
    price. That makes the path a Brownian bridge - E[r_t+1 | S_t] pulls toward
    a pinned endpoint - which is real, exploitable mean reversion that the null
    is not supposed to contain. Resampling with replacement unpins the endpoint.
    """
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    idx = rng.integers(0, r.shape[1], size=r.shape)
    out = np.take_along_axis(r, idx, axis=1)
    return _prices_from_logret(out, prc[:, 0])


def gen_xs_boot(rng: np.random.Generator) -> np.ndarray:
    """Common-index bootstrap with replacement: cross-section kept, endpoint free."""
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    idx = rng.integers(0, r.shape[1], size=r.shape[1])
    return _prices_from_logret(r[:, idx], prc[:, 0])


def gen_block_boot(rng: np.random.Generator, mean_block: int = 20) -> np.ndarray:
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    n = r.shape[1]
    idx = np.empty(n, dtype=int)
    filled = 0
    p_new = 1.0 / mean_block
    cur = rng.integers(n)
    while filled < n:
        idx[filled] = cur
        filled += 1
        if rng.random() < p_new:
            cur = int(rng.integers(n))
        else:
            cur = (cur + 1) % n
    return _prices_from_logret(r[:, idx], prc[:, 0])


def gen_gauss_corr(rng: np.random.Generator) -> np.ndarray:
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    mu = r.mean(axis=1)
    cov = np.cov(r)
    # jitter for numerical PD-ness
    cov = cov + 1e-14 * np.eye(cov.shape[0])
    sim = rng.multivariate_normal(mu, cov, size=r.shape[1], method="cholesky").T
    return _prices_from_logret(sim, prc[:, 0])


def gen_phase_surr(rng: np.random.Generator) -> np.ndarray:
    """Shared-phase FT surrogate: preserves the full cross-spectrum."""
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    mean = r.mean(axis=1, keepdims=True)
    centred = r - mean
    n = centred.shape[1]
    spec = np.fft.rfft(centred, axis=1)
    n_freq = spec.shape[1]
    phases = rng.uniform(0.0, 2.0 * np.pi, n_freq)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist must stay real
    rot = np.exp(1j * phases)[None, :]
    surro = np.fft.irfft(spec * rot, n=n, axis=1) + mean
    return _prices_from_logret(surro, prc[:, 0])


def _disjoint_pairs(pair_cfgs) -> list[tuple[int, int]]:
    """Greedy asset-disjoint subset of the book's own configured pairs."""
    used: set[int] = {0}  # never rewire ALGO
    picked: list[tuple[int, int]] = []
    for spec in pair_cfgs:
        y, x = int(spec.y), int(spec.x)
        if y in used or x in used:
            continue
        used.update((y, x))
        picked.append((y, x))
    return picked


def gen_coint_synth(
    rng: np.random.Generator,
    kappa: float = 0.08,
    spread_sd: float = 0.03,
    pair_cfgs=None,
) -> np.ndarray:
    """Factor model + genuine OU cointegration on disjoint configured pairs.

    kappa is the OU mean-reversion rate per day (half-life = ln2/kappa);
    kappa = 0 degenerates to a random walk spread, i.e. no cointegration.
    spread_sd is the stationary sd of the log-price spread.
    """
    prc = real_prices()
    r = np.diff(np.log(prc), axis=1)
    n_inst, n_ret = r.shape
    vol = r.std(axis=1)
    drift = r.mean(axis=1)

    # one common factor taken from the real cross-section
    centred = r - r.mean(axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(centred @ centred.T)
    load = u[:, 0] * np.sqrt(s[0] / n_ret)
    load_frac = np.clip(np.abs(load) / np.maximum(vol, 1e-12), 0.0, 0.95)
    idio_vol = vol * np.sqrt(np.maximum(1.0 - load_frac**2, 1e-6))

    factor = rng.standard_normal(n_ret)
    sim = (
        drift[:, None]
        + np.sign(load)[:, None] * (load_frac * vol)[:, None] * factor[None, :]
        + idio_vol[:, None] * rng.standard_normal((n_inst, n_ret))
    )

    log_p = np.log(prc[:, 0])[:, None] + np.cumsum(sim, axis=1)
    log_p = np.column_stack([np.log(prc[:, 0]), log_p])

    pairs = _disjoint_pairs(pair_cfgs)
    n_days = log_p.shape[1]
    b = float(np.exp(-kappa)) if kappa > 0 else 1.0
    if kappa > 0:
        eps_sd = spread_sd * np.sqrt(1.0 - b * b)
    else:
        # random-walk spread scaled to end up with a comparable spread size
        eps_sd = spread_sd / np.sqrt(n_days)

    for y, x in pairs:
        beta = float(rng.uniform(0.8, 1.2))
        shocks = eps_sd * rng.standard_normal(n_days)
        spread = np.empty(n_days)
        spread[0] = spread_sd * rng.standard_normal() if kappa > 0 else 0.0
        for t in range(1, n_days):
            spread[t] = b * spread[t - 1] + shocks[t]
        const = log_p[y, 0] - beta * log_p[x, 0] - spread[0]
        log_p[y] = beta * log_p[x] + const + spread

    return np.exp(log_p)


GENERATORS = {
    "iid_perm": gen_iid_perm,
    "xs_perm": gen_xs_perm,
    "iid_boot": gen_iid_boot,
    "xs_boot": gen_xs_boot,
    "gauss_corr": gen_gauss_corr,
    "phase_surr": gen_phase_surr,
}

# the endpoint-free nulls are the ones the verdict rests on; the pinned
# permutation variants are kept because the contrast between them measures the
# bridge artifact directly
NULL_GENS = ["xs_boot", "iid_boot", "gauss_corr"]
PINNED_GENS = ["xs_perm", "iid_perm"]

# Block bootstrap is written "block_boot_L<n>". Sweeping the block length turns
# it from a (bad) null into a horizon decomposition: dependence at lags shorter
# than the block survives resampling, everything longer is destroyed. The block
# length at which the score reappears IS the horizon the book actually trades.
BLOCK_LENGTHS = [2, 5, 10, 20, 60]

# generator -> replications, used by --mode full
FULL_PLAN: dict[str, int] = {
    "xs_perm": 200,      # sharpest null, needs the finest p-value
    "iid_perm": 150,
    "gauss_corr": 150,
    "phase_surr": 100,   # surrogate CI, not a null
}


def build_generator(name: str):
    if name.startswith("block_boot_L"):
        length = int(name.split("L")[1])
        return lambda rng: gen_block_boot(rng, mean_block=length)
    return GENERATORS[name]


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
class RandomSignBook:
    """Signal-free control with four_sigmas' exact sizing mechanics.

    Fixed random long/short signs, rebalanced every day to the full dollar
    limit, ALGO flat (its bang-bang limit is 0 in the live book). Any PnL this
    earns under a null is mechanical - constant-dollar rebalancing over a
    fat-tailed return distribution - and cannot be attributed to a signal.
    """

    def __init__(self, seed: int, n_inst: int) -> None:
        rng = np.random.default_rng(seed)
        self.signs = rng.choice([-1.0, 1.0], size=n_inst)
        self.signs[0] = 0.0
        self.limits = np.full(n_inst, float(DEFAULT_DLR_LIMIT))
        self.limits[0] = 0.0

    def __call__(self, prc: np.ndarray) -> np.ndarray:
        return (self.signs * self.limits / prc[:, -1]).astype(int)


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def run_one(task: tuple[str, int, float]) -> dict:
    """One replication: build a synthetic path, replay the book, score it."""
    name, seed, kappa = task
    import four_sigmas as fs

    stub = name.startswith("stub:")
    if stub:
        name = name.split(":", 1)[1]

    rng = np.random.default_rng(seed)
    if name == "coint_synth":
        prc = gen_coint_synth(rng, kappa=kappa, pair_cfgs=fs.PAIR_CFGS)
    elif name == "real":
        prc = real_prices()
    else:
        prc = build_generator(name)(rng)

    label = f"stub:{name}" if stub else name
    if not np.all(np.isfinite(prc)) or np.any(prc <= 0):
        return {"generator": label, "seed": seed, "kappa": kappa, "failed": 1}

    fs.reset()
    book = RandomSignBook(seed, prc.shape[0]) if stub else fs.getMyPosition
    try:
        stats = backtest(prc, book)
    except Exception as exc:  # a path the book cannot handle is a real result
        return {
            "generator": label, "seed": seed, "kappa": kappa,
            "failed": 1, "error": repr(exc)[:200],
        }
    stats.update({"generator": label, "seed": seed, "kappa": kappa, "failed": 0})
    return stats


def run_pool(tasks: list[tuple[str, int, float]], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for row in ex.map(run_one, tasks, chunksize=1):
            rows.append(row)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} paths", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def summarise(df: pd.DataFrame, real_score: float, real_mean: float) -> pd.DataFrame:
    rows = []
    for (gen, kappa), grp in df.groupby(["generator", "kappa"], dropna=False):
        ok = grp[grp["failed"] == 0]
        if ok.empty:
            continue
        scores = ok["score"].to_numpy()
        means = ok["mean_pl"].to_numpy()
        # +1 correction: real score is treated as one draw under the null
        p_score = (np.sum(scores >= real_score) + 1) / (len(scores) + 1)
        p_mean = (np.sum(means >= real_mean) + 1) / (len(means) + 1)
        rows.append({
            "generator": gen,
            "kappa": kappa,
            "n": len(ok),
            "failed": int((grp["failed"] == 1).sum()),
            "score_mean": scores.mean(),
            "score_sd": scores.std(ddof=1) if len(scores) > 1 else 0.0,
            "score_p05": np.percentile(scores, 5),
            "score_med": np.median(scores),
            "score_p95": np.percentile(scores, 95),
            "score_max": scores.max(),
            "mean_pl_mean": means.mean(),
            "mean_pl_sd": means.std(ddof=1) if len(means) > 1 else 0.0,
            "sharpe_mean": ok["sharpe"].mean(),
            "dvolume_mean": ok["dvolume"].mean(),
            "gross_mean": ok["mean_gross"].mean(),
            "net_mean": ok["mean_net"].mean(),
            "p_score_ge_real": p_score,
            "p_meanpl_ge_real": p_mean,
        })
    return pd.DataFrame(rows).sort_values(["generator", "kappa"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="null",
        choices=["null", "control", "blocks", "full", "mechanism", "all"],
    )
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--block-reps", type=int, default=60)
    ap.add_argument("--control-reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument(
        "--kappas", default="0,0.02,0.05,0.08,0.15,0.30",
        help="OU mean-reversion rates for the positive control",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import four_sigmas as fs

    fs.reset()
    real = backtest(real_prices(), fs.getMyPosition)
    print(f"REAL days {TEST_START}-{TEST_END}: score={real['score']:.2f} "
          f"mean=${real['mean_pl']:.1f} sd=${real['std_pl']:.1f} "
          f"sharpe={real['sharpe']:.2f}", flush=True)
    pd.DataFrame([real]).to_csv(OUT / "real_baseline.csv", index=False)

    block_gens = [f"block_boot_L{n}" for n in BLOCK_LENGTHS]
    order = list(FULL_PLAN) + block_gens

    def seed_for(gen: str, k: int) -> int:
        if gen not in order:
            order.append(gen)
        return args.seed + 1000 * (order.index(gen) + 1) + k

    plan: dict[str, int] = {}
    if args.mode in ("null", "all"):
        plan.update({g: args.reps for g in NULL_GENS + ["phase_surr"]})
    if args.mode in ("blocks", "all"):
        plan.update({g: args.block_reps for g in block_gens})
    if args.mode == "full":
        plan.update(FULL_PLAN)
        plan.update({g: args.block_reps for g in block_gens})
    if args.mode == "mechanism":
        # signal-free control on the same nulls, plus on the real path
        plan.update({f"stub:{g}": args.reps for g in NULL_GENS + ["real"]})

    tasks: list[tuple[str, int, float]] = [
        (gen, seed_for(gen, k), float("nan"))
        for gen, reps in plan.items()
        for k in range(reps)
    ]

    if args.mode in ("control", "full", "all"):
        kappas = [float(v) for v in args.kappas.split(",")]
        for ki, kappa in enumerate(kappas):
            for k in range(args.control_reps):
                tasks.append(("coint_synth", args.seed + 90_000 + 500 * ki + k, kappa))

    print(f"running {len(tasks)} synthetic paths on {args.workers} workers",
          flush=True)
    df = run_pool(tasks, args.workers)
    suffix = args.mode
    df.to_csv(OUT / f"paths_{suffix}.csv", index=False)

    summary = summarise(df, real["score"], real["mean_pl"])
    summary.to_csv(OUT / f"summary_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
