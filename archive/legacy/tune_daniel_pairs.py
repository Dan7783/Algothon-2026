"""Robust pair-specific tuning for the live ``daniel_filter`` atlas.

The search is intentionally local to each pair's current configuration.  It
records each Kalman/CUSUM path once, then sweeps the entry/exit/stop policy on
that path.  Train (days 251--500) and test (days 501--750) are simulated with
independent hysteresis state, matching separate official-evaluator runs.

Selection is Pareto guarded: a replacement must improve both official score
and mean PnL in train and test.  If possible, it must also avoid degrading any
of the four 125-day half-windows.  Otherwise the existing pair configuration
is retained.  Test therefore participates in selection at the user's explicit
request and must no longer be described as an untouched holdout; the half-
window guard and local grid only reduce, rather than remove, that overfit risk.

Outputs are written below ``cusum_output/daniel_pairs``.  The script does not
rewrite ``daniel_filter.py``; selected seeds are applied there only after the
pair and full-portfolio audits pass.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import json
import time
from pathlib import Path

import numpy as np

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, PairCfg, PairSegFilter, TradeCfg
from pair_book_lab import eval_score, load_raw_prices


LEG = 10_000.0
COMM_RATE = 1.0e-4
TRAIN_START, TRAIN_MID, TRAIN_END = 251, 375, 500
TEST_START, TEST_MID, TEST_END = 501, 625, 750
TRAIN_DECIDE_START = TRAIN_START - 2
TEST_DECIDE_START = TEST_START - 2
OUT_DIR = Path("cusum_output/daniel_pairs")
EPSILON = 1.0e-12


def record_paths(ly: np.ndarray, lx: np.ndarray, cfg: FiltCfg) -> dict:
    """Run one complete filter path; trade thresholds are not involved."""
    filt = PairSegFilter(cfg)
    beta = np.empty(len(ly))
    adapting = np.zeros(len(ly), dtype=bool)
    cut = np.zeros(len(ly), dtype=bool)
    for t in range(len(ly)):
        output = filt.step(float(ly[t]), float(lx[t]))
        beta[t] = output.beta
        adapting[t] = output.adapting
        cut[t] = output.cut
    return {"beta": beta, "adapting": adapting, "cut": cut}


def gate_and_z_paths(ly: np.ndarray, lx: np.ndarray, beta: np.ndarray) -> dict:
    """Reproduce ``PairAgent.desired_side`` gates and lagged-beta z-score."""
    total = len(ly)
    adf_block = np.zeros(total, dtype=bool)
    kappa_block = np.zeros(total, dtype=bool)
    adf_t = np.full(total, np.nan)
    kappa = np.full(total, np.nan)
    z_path = np.zeros(total)
    first = min(TRAIN_DECIDE_START, total)
    for t in range(first, total):
        beta_lag = beta[t - 1]
        spread_history = ly[:t] - beta_lag * lx[:t]
        adf_t[t] = dfil._adf_t_spread(spread_history)
        if len(spread_history) >= 40 and adf_t[t] >= dfil.PRIOR_GATE_T:
            adf_block[t] = True
        spread = (
            ly[t - dfil.Z_W + 1 : t + 1]
            - beta_lag * lx[t - dfil.Z_W + 1 : t + 1]
        )
        z_path[t] = float(
            (spread[-1] - spread[:-1].mean())
            / (spread[:-1].std() + EPSILON)
        )
        spread_120 = ly[t - 119 : t + 1] - beta_lag * lx[t - 119 : t + 1]
        kappa[t] = dfil._ou_kappa(spread_120)
        if kappa[t] <= 0.0:
            kappa_block[t] = True
    return {
        "adf_block": adf_block,
        "kappa_block": kappa_block,
        "adf_t": adf_t,
        "kappa": kappa,
        "z": z_path,
    }


def trade_sim(
    gates: dict,
    price_y: np.ndarray,
    price_x: np.ndarray,
    trade: TradeCfg,
    decide_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pair hysteresis from a fresh evaluator state."""
    total = len(price_y)
    shares_y = np.zeros(total)
    shares_x = np.zeros(total)
    side = 0
    z_entry = float(trade.z_entry)
    z_exit = min(float(trade.z_exit), 0.85 * z_entry)
    z_stop = max(float(trade.z_stop), 1.25 * z_entry)
    for t in range(decide_start, total):
        if (
            (trade.require_adf and gates["adf_block"][t])
            or (trade.require_kappa and gates["kappa_block"][t])
        ):
            side = 0
        else:
            z = float(gates["z"][t])
            absolute_z = abs(z)
            if side == 0:
                if z_entry <= absolute_z < z_stop:
                    side = int(-np.sign(z))
            elif absolute_z <= z_exit or absolute_z >= z_stop:
                side = 0
            elif absolute_z >= z_entry and int(-np.sign(z)) != side:
                side = int(-np.sign(z))
        if side:
            scale = max(0.0, min(1.0, float(trade.leg_scale)))
            shares_y[t] = int(side * LEG * scale / price_y[t])
            shares_x[t] = int(-side * LEG * scale / price_x[t])
    return shares_y, shares_x


def pnl_path(
    shares_y: np.ndarray,
    shares_x: np.ndarray,
    price_y: np.ndarray,
    price_x: np.ndarray,
) -> np.ndarray:
    """Daily PnL indexed by official day number, including one-day-late fees."""
    total = len(price_y)
    pnl = np.zeros(total + 1)
    k = np.arange(1, total - 1)
    gross = (
        shares_y[k] * (price_y[k + 1] - price_y[k])
        + shares_x[k] * (price_x[k + 1] - price_x[k])
    )
    fee = COMM_RATE * (
        np.abs(shares_y[k] - shares_y[k - 1]) * price_y[k]
        + np.abs(shares_x[k] - shares_x[k - 1]) * price_x[k]
    )
    pnl[k + 2] = gross - fee
    return pnl


def window_stats(pnl: np.ndarray, start: int, end: int) -> dict[str, float]:
    values = pnl[start : end + 1]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "std": std,
        "score": float(eval_score(mean, std)),
        "net": float(np.sum(values)),
    }


def _local_values(value: float, lower: float, upper: float, mode: str) -> list[float]:
    if mode == "ratio":
        candidates = (value / 3.0, value, value * 3.0)
    elif mode == "entry":
        candidates = (value * 0.75, value, value * 1.25)
    else:
        candidates = (value - float(mode), value, value + float(mode))
    return sorted({float(np.clip(item, lower, upper)) for item in candidates})


def filter_grid(spec: PairCfg) -> list[FiltCfg]:
    base = spec.filt
    q_values = _local_values(base.q_beta, 1.0e-8, 1.0e-4, "ratio")
    h_values = _local_values(base.h, 3.0, 15.0, "2.0")
    k_values = _local_values(base.cusum_k, 0.2, 0.8, "0.1")
    return [
        base._replace(q_beta=q_beta, h=h, cusum_k=k)
        for q_beta in q_values
        for h in h_values
        for k in k_values
    ]


def trade_grid(spec: PairCfg) -> list[TradeCfg]:
    base = spec.trade
    entries = _local_values(base.z_entry, 0.08, 0.80, "entry")
    exits = _local_values(base.z_exit, 0.01, 0.50, "entry")
    stops = _local_values(base.z_stop, 2.0, 6.0, "0.75")
    configs = []
    for entry in entries:
        for exit_ in exits:
            for stop in stops:
                configs.append(
                    base._replace(
                        z_entry=entry,
                        z_exit=min(exit_, 0.85 * entry),
                        z_stop=max(stop, 1.25 * entry),
                    )
                )
    return list(dict.fromkeys(configs))


def evaluate_trade(
    gates: dict,
    price_y: np.ndarray,
    price_x: np.ndarray,
    trade: TradeCfg,
) -> dict:
    train_y, train_x = trade_sim(
        gates, price_y, price_x, trade, TRAIN_DECIDE_START
    )
    test_y, test_x = trade_sim(
        gates, price_y, price_x, trade, TEST_DECIDE_START
    )
    train_pnl = pnl_path(train_y, train_x, price_y, price_x)
    test_pnl = pnl_path(test_y, test_x, price_y, price_x)
    train = window_stats(train_pnl, TRAIN_START, TRAIN_END)
    test = window_stats(test_pnl, TEST_START, TEST_END)
    blocks = [
        window_stats(train_pnl, TRAIN_START, TRAIN_MID),
        window_stats(train_pnl, TRAIN_MID + 1, TRAIN_END),
        window_stats(test_pnl, TEST_START, TEST_MID),
        window_stats(test_pnl, TEST_MID + 1, TEST_END),
    ]
    return {
        "train": train,
        "test": test,
        "blocks": blocks,
        "worst_block_score": min(block["score"] for block in blocks),
        "turnover_dollars": float(
            np.sum(np.abs(np.diff(train_y)) * price_y[1:])
            + np.sum(np.abs(np.diff(train_x)) * price_x[1:])
            + np.sum(np.abs(np.diff(test_y)) * price_y[1:])
            + np.sum(np.abs(np.diff(test_x)) * price_x[1:])
        ),
    }


def _serialize_cfg(cfg: tuple) -> dict:
    return cfg._asdict()


def tune_pair(spec: PairCfg, prices: np.ndarray) -> dict:
    y, x = spec.y, spec.x
    log_y = np.log(prices[y])
    log_x = np.log(prices[x])
    price_y, price_x = prices[y], prices[x]

    path_cache: dict[FiltCfg, tuple[dict, int]] = {}
    for filt in filter_grid(spec):
        path = record_paths(log_y, log_x, filt)
        path_cache[filt] = (
            gate_and_z_paths(log_y, log_x, path["beta"]),
            int(np.sum(path["cut"])),
        )

    rows = []
    for filt, (gates, cuts) in path_cache.items():
        for trade in trade_grid(spec):
            stats = evaluate_trade(gates, price_y, price_x, trade)
            rows.append(
                {
                    "filt": filt,
                    "trade": trade,
                    "cuts": cuts,
                    **stats,
                }
            )

    baseline = next(
        row
        for row in rows
        if row["filt"] == spec.filt and row["trade"] == spec.trade
    )

    def pareto_improves(row: dict) -> bool:
        return (
            row["train"]["score"] > baseline["train"]["score"] + 1.0e-9
            and row["test"]["score"] > baseline["test"]["score"] + 1.0e-9
            and row["train"]["mean"] >= baseline["train"]["mean"] - 1.0e-9
            and row["test"]["mean"] >= baseline["test"]["mean"] - 1.0e-9
        )

    pareto = [row for row in rows if pareto_improves(row)]
    block_safe = [
        row
        for row in pareto
        if all(
            row["blocks"][index]["score"]
            >= baseline["blocks"][index]["score"] - 1.0e-9
            for index in range(4)
        )
    ]
    pool = block_safe or pareto
    if pool:
        train_scale = max(abs(baseline["train"]["score"]), 20.0)
        test_scale = max(abs(baseline["test"]["score"]), 20.0)

        def selection_key(row: dict) -> tuple:
            train_gain = (
                row["train"]["score"] - baseline["train"]["score"]
            ) / train_scale
            test_gain = (
                row["test"]["score"] - baseline["test"]["score"]
            ) / test_scale
            return (
                min(train_gain, test_gain),
                row["worst_block_score"],
                train_gain + test_gain,
                -row["cuts"],
                -row["turnover_dollars"],
            )

        chosen = max(pool, key=selection_key)
        reason = "four-block Pareto improvement" if block_safe else "train/test Pareto improvement"
    else:
        chosen = baseline
        reason = "kept baseline: no local train/test Pareto improvement"

    # A fully disabled candidate is a valid conservative adjustment only when
    # the current pair loses in both windows.  It cannot be used to erase a
    # positive window merely to improve the other one.
    disabled = False
    if baseline["train"]["score"] < 0.0 and baseline["test"]["score"] < 0.0:
        disabled = True
        chosen = dict(chosen)
        chosen["trade"] = chosen["trade"]._replace(leg_scale=0.0)
        chosen["train"] = {"mean": 0.0, "std": 0.0, "score": 0.0, "net": 0.0}
        chosen["test"] = {"mean": 0.0, "std": 0.0, "score": 0.0, "net": 0.0}
        reason = "disabled: baseline loses in both train and test"

    return {
        "pair": [y, x],
        "name": spec.name,
        "priority_before": float(spec.priority),
        "baseline_filt": _serialize_cfg(spec.filt),
        "baseline_trade": _serialize_cfg(spec.trade),
        "chosen_filt": _serialize_cfg(chosen["filt"]),
        "chosen_trade": _serialize_cfg(chosen["trade"]),
        "baseline_train": baseline["train"],
        "baseline_test": baseline["test"],
        "chosen_train": chosen["train"],
        "chosen_test": chosen["test"],
        "baseline_blocks": baseline["blocks"],
        "chosen_blocks": chosen.get("blocks", baseline["blocks"]),
        "baseline_cuts": int(baseline["cuts"]),
        "chosen_cuts": int(chosen["cuts"]),
        "improved": chosen is not baseline or disabled,
        "disabled": disabled,
        "reason": reason,
        "candidate_count": len(rows),
    }


def configs_from_results(
    baseline: tuple[PairCfg, ...],
    results: list[dict],
    *,
    rerank: bool,
) -> tuple[PairCfg, ...]:
    by_pair = {tuple(result["pair"]): result for result in results}
    configs = []
    for index, spec in enumerate(baseline):
        result = by_pair[(spec.y, spec.x)]
        filt = FiltCfg(**result["chosen_filt"])
        trade = TradeCfg(**result["chosen_trade"])
        priority = (
            max(float(result["chosen_train"]["score"]), 0.01)
            if rerank and trade.leg_scale > 0.0
            else (0.0 if trade.leg_scale <= 0.0 else spec.priority)
        )
        configs.append(
            spec._replace(
                idx=index,
                priority=priority,
                filt=filt,
                trade=trade,
            )
        )
    return tuple(configs)


def portfolio_window(
    prices: np.ndarray,
    configs: tuple[PairCfg, ...],
    start: int,
    end: int,
) -> dict[str, float]:
    """Direct replay of Daniel's public API for one independently reset window."""
    dfil.apply_pair_cfgs(configs)
    dfil.reset()
    rates = np.full(dfil.N_INST, COMM_RATE)
    rates[0] = 2.0e-5
    previous = np.zeros(dfil.N_INST, dtype=int)
    pnl = []
    turnover = 0.0
    for t in range(start - 1, end):
        history = prices[:, :t]
        current = history[:, -1]
        target = dfil.getMyPosition(history)
        change = target - previous
        fee = float(np.sum(current * np.abs(change) * rates))
        gross = float(target @ (prices[:, t] - current))
        pnl.append(gross - fee)
        turnover += float(np.sum(current * np.abs(change)))
        previous = target
    values = np.asarray(pnl)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "net": float(np.sum(values)),
        "mean": mean,
        "std": std,
        "score": float(eval_score(mean, std)),
        "sharpe": float(np.sqrt(250.0) * mean / std) if std > EPSILON else 0.0,
        "turnover": turnover,
    }


def portfolio_audit(
    prices: np.ndarray,
    baseline: tuple[PairCfg, ...],
    results: list[dict],
) -> dict:
    candidates = {
        "baseline": baseline,
        "tuned_keep_priority": configs_from_results(baseline, results, rerank=False),
        "tuned_rerank_train": configs_from_results(baseline, results, rerank=True),
    }
    audit = {}
    for name, configs in candidates.items():
        audit[name] = {
            "train": portfolio_window(prices, configs, TRAIN_START, TRAIN_END),
            "test": portfolio_window(prices, configs, TEST_START, TEST_END),
        }

    base = audit["baseline"]
    eligible = [
        name
        for name in ("tuned_keep_priority", "tuned_rerank_train")
        if audit[name]["train"]["score"] >= base["train"]["score"]
        and audit[name]["test"]["score"] >= base["test"]["score"]
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda name: min(
                audit[name]["train"]["score"] - base["train"]["score"],
                audit[name]["test"]["score"] - base["test"]["score"],
            ),
        )
    else:
        selected = "baseline"
    audit["selected"] = selected
    dfil.apply_pair_cfgs(candidates[selected])
    return {"audit": audit, "configs": candidates}


def _report(results: list[dict], portfolio: dict) -> str:
    lines = [
        "# Daniel pair-specific Pareto tuning",
        "",
        "Test participates in selection by explicit request; it is not an untouched holdout.",
        "Each replacement improves train and test score/mean, with four-block non-degradation preferred.",
        "",
        "| Pair | Train before -> after | Test before -> after | Decision |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['baseline_train']['score']:.2f} -> "
            f"{result['chosen_train']['score']:.2f} | "
            f"{result['baseline_test']['score']:.2f} -> "
            f"{result['chosen_test']['score']:.2f} | {result['reason']} |"
        )
    lines.extend(["", "## Portfolio audit", "", "```json"])
    lines.append(json.dumps(portfolio["audit"], indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_raw_prices()
    baseline = tuple(dfil.PAIR_CFGS)
    started = time.perf_counter()
    results = []
    for spec in baseline:
        result = tune_pair(spec, prices)
        results.append(result)
        print(
            f"{spec.name:>11}: train {result['baseline_train']['score']:+8.2f} -> "
            f"{result['chosen_train']['score']:+8.2f}; test "
            f"{result['baseline_test']['score']:+8.2f} -> "
            f"{result['chosen_test']['score']:+8.2f}; {result['reason']}",
            flush=True,
        )

    portfolio = portfolio_audit(prices, baseline, results)
    serializable_audit = portfolio["audit"]
    (OUT_DIR / "pareto_tuning.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "portfolio_audit.json").write_text(
        json.dumps(serializable_audit, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "tuning_report.md").write_text(
        _report(results, portfolio), encoding="utf-8"
    )
    print(
        f"selected portfolio: {serializable_audit['selected']}; "
        f"completed in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
