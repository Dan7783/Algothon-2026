#!/usr/bin/env python
"""Evaluate a strategy module with the same accounting as eval.py, without plots."""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import argparse
import ast
import importlib
from pathlib import Path
import time
import types

import numpy as np
import pandas as pd


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sharpe = np.sqrt(250.0) * mu / sigma
    return mu * sharpe**2 / (sharpe**2 + param**2)


def load_strategy(module_name):
    """Load local strategy source directly so stale tracked pyc is ignored."""
    path = Path(*module_name.split(".")).with_suffix(".py")
    if not path.exists():
        return importlib.import_module(module_name)
    strategy = types.ModuleType(module_name)
    strategy.__file__ = str(path.resolve())
    strategy.__package__ = module_name.rpartition(".")[0]
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
         strategy.__dict__)
    return strategy


def evaluate(module_name, start=251, end=750, overrides=()):
    strategy = load_strategy(module_name)
    applied = {}
    for assignment in overrides:
        name, raw_value = assignment.split("=", 1)
        value = ast.literal_eval(raw_value)
        if not hasattr(strategy, name):
            raise AttributeError(f"{module_name} has no setting {name!r}")
        setattr(strategy, name, value)
        applied[name] = value
    prices = pd.read_csv("prices.txt", sep=r"\s+").to_numpy(float).T
    n_inst = len(prices)
    commission = np.full(n_inst, 0.0001)
    commission[0] = 0.00002
    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0

    cash = 0.0
    current = np.zeros(n_inst)
    total_volume = 0.0
    value = 0.0
    previous_commission = 0.0
    pnl = []
    call_seconds = []

    for t in range(start - 1, end + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]
        if t < end:
            began = time.perf_counter()
            requested = strategy.getMyPosition(history)
            call_seconds.append(time.perf_counter() - began)
            position_limits = (limits / current_prices).astype(int)
            new_position = np.clip(requested, -position_limits, position_limits).astype(int)
        else:
            new_position = current.copy()

        delta = new_position - current
        cash -= current_prices.dot(delta) + previous_commission
        dollar_volume = current_prices * np.abs(delta)
        total_volume += dollar_volume.sum()
        previous_commission = np.sum(dollar_volume * commission)
        current = new_position.copy()
        new_value = cash + current.dot(current_prices)
        daily_pnl = new_value - value
        value = new_value
        if t >= start:
            pnl.append(daily_pnl)

    pnl = np.asarray(pnl)
    mu = float(pnl.mean())
    sigma = float(pnl.std())
    sharpe = float(np.sqrt(250.0) * mu / sigma) if sigma > 0 else 0.0
    result = {
        "module": module_name,
        "start": start,
        "end": end,
        "days": len(pnl),
        "score": float(score(mu, sigma)),
        "mean_pnl": mu,
        "std_pnl": sigma,
        "sharpe": sharpe,
        "final_value": float(value),
        "total_volume": float(total_volume),
        "mean_call_ms": float(1e3 * np.mean(call_seconds)),
        "p95_call_ms": float(1e3 * np.percentile(call_seconds, 95)),
        "overrides": applied,
    }
    return result, pnl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", nargs="?", default="teamName")
    parser.add_argument("--start", type=int, default=251)
    parser.add_argument("--end", type=int, default=750)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    result, pnl = evaluate(args.module, args.start, args.end, args.set)
    for key, value in result.items():
        print(f"{key}: {value}")
    block = 50
    for offset in range(0, len(pnl), block):
        values = pnl[offset : offset + block]
        mu = values.mean()
        sigma = values.std()
        print(
            f"block {offset // block + 1}: n={len(values)} "
            f"mean={mu:.3f} score={score(mu, sigma):.3f}"
        )


if __name__ == "__main__":
    main()
