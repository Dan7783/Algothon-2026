#!/usr/bin/env python
"""Lightweight train-only correlation + CUSUM screen for testbedv2_tuned pairs.

Deliberately simpler than analyse_testbedv2_pair_structure.py's three-path
cointegration audit. That script treats persistent Engle-Granger / adaptive /
segmented cointegration as the primary keep criterion. This script instead
filters "fake" (spurious) pairs off just two things: correlation and CUSUM
break behaviour. The classic fixed-beta Engle-Granger test assumes one beta
for the whole window; that assumption is known false here (see the shared,
untuned filter's per-fold beta below, which visibly moves across folds for
most pairs), so a single full-window ADF/cointegration p-value is reported
purely as a diagnostic -- expected to be a redundant or unreliable signal --
never used to reject a pair.

Both real criteria are validated walk-forward: days 1-750 are split into
three sequential, non-overlapping 250-day folds, and a pair only survives if
its correlation and its (shared, untuned) CUSUM break count are BOTH sane in
most folds, not just pooled over the whole window. A pair whose apparent
relationship is concentrated in a single sub-period (and vanishes elsewhere)
is exactly the overfitting failure mode this is meant to catch. Days 751-1000
are never read -- reserved as the true, untouched holdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.analyse_testbedv2_pair_structure import (
    half_life,
    ols_residual,
    pair_name,
    safe_adf,
    safe_correlation,
    symbols_for,
)
from research.shortlist_pair_review import load_prices, load_strategy


MODULE = "testbedv2_tuned"
OUTPUT_DIR = ROOT / "analysis_outputs" / "testbedv2_corr_cusum_screen"
PRIOR_KEPT_NAMES_PATH = (
    ROOT / "analysis_outputs" / "testbedv2_pair_structure" / "kept_pair_names.json"
)

# Three sequential, non-overlapping walk-forward folds within days 1-750.
FOLDS = (
    ("fold1", 1, 250),
    ("fold2", 251, 500),
    ("fold3", 501, 750),
)

# --- The two actual filter criteria -----------------------------------
MIN_FOLD_RETURN_CORR = 0.10
MIN_FOLD_LEVEL_CORR = 0.50
MIN_CORR_PASS_FOLDS = 2  # correlation must be sane in >= 2/3 folds
MAX_CUTS_PER_FOLD = 15  # shared/untuned CUSUM breaks per ~250-day fold
MAX_TOTAL_CUTS = 35  # across the full 750-day window
MIN_CUSUM_PASS_FOLDS = 2  # break count must stay sane in >= 2/3 folds


def analyse_pair(
    log_prices: np.ndarray,
    cfg: Any,
    strategy: Any,
    symbols: dict[int, str],
) -> dict[str, Any]:
    y_idx, x_idx = int(cfg.y), int(cfg.x)
    y_full = log_prices[y_idx, :750]
    x_full = log_prices[x_idx, :750]

    record: dict[str, Any] = {"pair_name": pair_name(cfg, symbols)}

    # Simple full-window ADF (fixed beta, whole 750-day OLS fit). Reported
    # only -- see module docstring for why this is not a filter criterion.
    _, full_beta, full_resid = ols_residual(y_full, x_full)
    _, full_adf_p = safe_adf(full_resid)
    record["full_beta"] = full_beta
    record["full_adf_p"] = full_adf_p
    record["full_half_life"] = half_life(full_resid)

    # Same shared, untuned filter used throughout this repo's structural
    # audits (FiltCfg(q_beta=1e-6), class defaults otherwise) -- not tuned
    # per pair, so break counts are a non-circular structural signal.
    struct_filt = strategy.PairSegFilter(strategy.FiltCfg(q_beta=1e-6))
    cut_flags = np.empty(750, dtype=bool)
    for t, (y_val, x_val) in enumerate(zip(y_full, x_full, strict=True)):
        cut_flags[t] = bool(struct_filt.step(float(y_val), float(x_val)).cut)

    fold_betas: list[float] = []
    for label, start_day, end_day in FOLDS:
        y_seg = log_prices[y_idx, start_day - 1 : end_day]
        x_seg = log_prices[x_idx, start_day - 1 : end_day]
        returns = np.diff(np.vstack([y_seg, x_seg]), axis=1)
        _, fold_beta, fold_resid = ols_residual(y_seg, x_seg)
        _, fold_adf_p = safe_adf(fold_resid)
        n_cuts = int(cut_flags[start_day - 1 : end_day].sum())
        fold_betas.append(fold_beta)
        record.update(
            {
                f"{label}_return_corr": safe_correlation(returns[0], returns[1]),
                f"{label}_level_corr": safe_correlation(y_seg, x_seg),
                f"{label}_beta": fold_beta,
                f"{label}_adf_p": fold_adf_p,
                f"{label}_half_life": half_life(fold_resid),
                f"{label}_cusum_cuts": n_cuts,
            }
        )

    record["total_cusum_cuts"] = int(cut_flags.sum())
    finite_pos_betas = [b for b in fold_betas if np.isfinite(b) and b > 0.0]
    record["fold_beta_ratio"] = (
        float(max(finite_pos_betas) / min(finite_pos_betas))
        if len(finite_pos_betas) == len(fold_betas)
        else float("inf")
    )

    corr_pass_folds = sum(
        1
        for label, _, _ in FOLDS
        if record[f"{label}_return_corr"] >= MIN_FOLD_RETURN_CORR
        or record[f"{label}_level_corr"] >= MIN_FOLD_LEVEL_CORR
    )
    cusum_pass_folds = sum(
        1 for label, _, _ in FOLDS if record[f"{label}_cusum_cuts"] <= MAX_CUTS_PER_FOLD
    )
    record["corr_pass_folds"] = corr_pass_folds
    record["cusum_pass_folds"] = cusum_pass_folds
    record["median_return_corr"] = float(
        np.nanmedian([record[f"{label}_return_corr"] for label, _, _ in FOLDS])
    )

    failures: list[str] = []
    if corr_pass_folds < MIN_CORR_PASS_FOLDS:
        failures.append("correlation_not_persistent")
    if cusum_pass_folds < MIN_CUSUM_PASS_FOLDS:
        failures.append("cusum_breaks_not_persistent")
    if record["total_cusum_cuts"] > MAX_TOTAL_CUTS:
        failures.append("excessive_total_cusum_breaks")

    record["keep"] = not failures
    record["rejection_reasons"] = ";".join(failures)
    return record


def make_plot(frame: pd.DataFrame, path: Path) -> None:
    colors = np.where(frame["keep"], "#55a868", "#c44e52")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(frame["median_return_corr"], frame["total_cusum_cuts"], c=colors, alpha=0.75)
    axes[0].axvline(MIN_FOLD_RETURN_CORR, color="grey", linestyle=":")
    axes[0].axhline(MAX_TOTAL_CUTS, color="grey", linestyle=":")
    axes[0].set_xlabel("Median fold return correlation")
    axes[0].set_ylabel("Total CUSUM breaks (days 1-750)")
    axes[0].set_title("Correlation vs CUSUM break count")

    axes[1].scatter(frame["corr_pass_folds"], frame["cusum_pass_folds"], c=colors, alpha=0.5, s=120)
    axes[1].set_xlabel("Folds passing correlation gate (of 3)")
    axes[1].set_ylabel("Folds passing CUSUM gate (of 3)")
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_title("Walk-forward persistence")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(frame: pd.DataFrame, prior_kept: set[str], path: Path) -> None:
    kept = frame.loc[frame["keep"]]
    rejected = frame.loc[~frame["keep"]]
    reasons = rejected["rejection_reasons"].str.split(";").explode().value_counts()
    kept_names = set(kept["pair_name"])
    new_pairs = sorted(kept_names - prior_kept)
    dropped_pairs = sorted(prior_kept - kept_names)
    overlap = sorted(kept_names & prior_kept)

    lines = [
        "# testbedv2_tuned correlation + CUSUM walk-forward screen",
        "",
        "All statistics use days 1-750 only, split into three sequential "
        "walk-forward folds (1-250, 251-500, 501-750). Days 751-1000 were "
        "not read. Unlike the cointegration audit in "
        "pair_structure_report.md, a pair is filtered off correlation and "
        "shared/untuned CUSUM break behaviour ONLY -- no Engle-Granger or "
        "ADF gate, since a single fixed-beta test is known unreliable when "
        "beta genuinely moves across folds (see the beta columns below).",
        "",
        "## Result",
        "",
        f"- Active input pairs: **{len(frame)}**",
        f"- Kept: **{len(kept)}**",
        f"- Rejected: **{len(rejected)}**",
        "",
        "## Filter rules",
        "",
        f"- Correlation: return corr >= {MIN_FOLD_RETURN_CORR:.2f} OR level "
        f"corr >= {MIN_FOLD_LEVEL_CORR:.2f}, sane in >= {MIN_CORR_PASS_FOLDS}/3 "
        "walk-forward folds.",
        f"- CUSUM breaks (shared, untuned filter): <= {MAX_CUTS_PER_FOLD} per "
        f"fold, sane in >= {MIN_CUSUM_PASS_FOLDS}/3 folds, and <= "
        f"{MAX_TOTAL_CUTS} total across days 1-750.",
        "- Fixed-beta ADF / cointegration: reported per fold and full-window "
        "for reference only. NOT used to keep or reject a pair.",
        "",
        "## Comparison against the strict cointegration-audit pair list",
        "",
        f"- Prior strict list size: **{len(prior_kept)}**",
        f"- Overlap (kept by both): **{len(overlap)}** -- {', '.join(overlap) if overlap else 'none'}",
        f"- New pairs (kept here, not in strict list): **{len(new_pairs)}**",
        f"- Dropped pairs (in strict list, rejected here): **{len(dropped_pairs)}** "
        f"-- {', '.join(dropped_pairs) if dropped_pairs else 'none'}",
        "",
        "## Kept pairs",
        "",
        "| Pair | Return corr F1/F2/F3 | Corr folds | CUSUM cuts F1/F2/F3 | "
        "CUSUM folds | Beta F1/F2/F3 | Beta ratio | Full ADF p (diagnostic) |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for row in kept.sort_values("pair_name").itertuples(index=False):
        lines.append(
            f"| {row.pair_name} | "
            f"{row.fold1_return_corr:.3f}/{row.fold2_return_corr:.3f}/{row.fold3_return_corr:.3f} | "
            f"{row.corr_pass_folds}/3 | "
            f"{row.fold1_cusum_cuts}/{row.fold2_cusum_cuts}/{row.fold3_cusum_cuts} | "
            f"{row.cusum_pass_folds}/3 | "
            f"{row.fold1_beta:.3f}/{row.fold2_beta:.3f}/{row.fold3_beta:.3f} | "
            f"{row.fold_beta_ratio:.2f} | {row.full_adf_p:.4f} |"
        )
    lines.extend(["", "## Rejection counts", ""])
    lines.extend(f"- {reason}: **{count}**" for reason, count in reasons.items())
    lines.extend(
        [
            "",
            "## Rejected pairs",
            "",
            "| Pair | Median return corr | Corr folds | Total CUSUM cuts | "
            "CUSUM folds | Reasons |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rejected.sort_values(
        ["corr_pass_folds", "cusum_pass_folds", "median_return_corr"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.pair_name} | {row.median_return_corr:.3f} | "
            f"{row.corr_pass_folds}/3 | {row.total_cusum_cuts} | "
            f"{row.cusum_pass_folds}/3 | {row.rejection_reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


FILTERED_MODULE_PATH = ROOT / "testbedv2_tuned_pairfiltered.py"

FILTERED_MODULE_TEMPLATE = '''"""Correlation + CUSUM filtered version of :mod:`testbedv2_tuned`.

The trading logic and tuned parameters are inherited unchanged.  The only
difference is the active pair universe.  Pair selection was performed using
days 1-750 only; days 751-1000 were reserved and were not used by the screen.

Unlike a classic Engle-Granger screen, a pair is NOT required to pass a
fixed-beta cointegration test -- that assumption (one beta for the whole
window) is known false here, since hedge ratios visibly move (and for many
pairs, flip sign) across sub-periods. Instead a pair is kept purely off two
walk-forward-validated criteria, checked in three sequential 250-day folds
across days 1-750: (1) return or level correlation staying sane in >= 2/3
folds, and (2) a shared, untuned CUSUM break detector staying within a sane
break count in >= 2/3 folds (and under a total cap across all 750 days) --
too many breaks means the filter never finds a stable segment, which is
treated as the signature of a spurious ("fake") pair. Fixed-beta ADF /
cointegration is still computed per fold and full-window but is reported for
reference only and never used to keep or reject a pair.

See ``analysis_outputs/testbedv2_corr_cusum_screen/corr_cusum_screen_report.md``
for the complete per-pair correlation, CUSUM-break and diagnostic-ADF audit,
including a comparison against the stricter cointegration-audit pair list.

ALGO (instrument index 0) is allowed to trade like any other leg. The base
strategy's getMyPosition forced ``dollars[0] = 0.0`` unconditionally, which
left every selected ALGO pair (e.g. ALGO-HRET) trading its other leg naked
with no hedge, since ALGO's own leg was always zeroed regardless of which
pairs were active. That override -- and the matching ``crash[0] = False``
carve-out that made ALGO immune to the single-day crash filter -- have been
removed here so ALGO gets the same leg-cap and crash protection as every
other instrument.
"""

from __future__ import annotations

import numpy as np

import testbedv2_tuned as _base


# Re-export the strategy constants and classes so this module remains as easy
# to inspect and test as the source strategy.
I = _base.I
N_INST = _base.N_INST
LEG = _base.LEG
WARMUP = _base.WARMUP
FiltCfg = _base.FiltCfg
TradeCfg = _base.TradeCfg
PairCfg = _base.PairCfg
PairSegFilter = _base.PairSegFilter
PairAgent = _base.PairAgent


SELECTED_PAIR_NAMES = frozenset(
    {{
{names_block}
    }}
)

_SYMBOL_BY_INDEX = {{index: symbol for symbol, index in I.items()}}


def _pair_name(spec: PairCfg) -> str:
    return f"{{_SYMBOL_BY_INDEX[spec.y]}}-{{_SYMBOL_BY_INDEX[spec.x]}}"


# Keep the original pair orientation and every tuned parameter.  Reindexing
# only makes the surviving PairCfg identifiers contiguous.
PAIR_CFGS = tuple(
    spec._replace(idx=new_idx)
    for new_idx, spec in enumerate(
        spec for spec in _base.PAIR_CFGS if _pair_name(spec) in SELECTED_PAIR_NAMES
    )
)

if {{_pair_name(spec) for spec in PAIR_CFGS}} != SELECTED_PAIR_NAMES:
    missing = SELECTED_PAIR_NAMES - {{_pair_name(spec) for spec in PAIR_CFGS}}
    raise RuntimeError(f"Filtered pair configuration is missing: {{sorted(missing)}}")


class AtlasBook:
    """The original portfolio allocator operating on the filtered universe."""

    def __init__(self) -> None:
        self.agents = [PairAgent(spec) for spec in PAIR_CFGS]

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        if nt < WARMUP + 10:
            return pos

        for agent in self.agents:
            agent.catch_up(prices)

        candidates = []
        for agent in self.agents:
            pos_y, pos_x, abs_z = agent.dollar_positions()
            if abs(pos_y) < 1e-9:
                continue
            priority_score = agent.current_kappa * abs_z
            candidates.append((priority_score, agent, pos_y, pos_x))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        taken: set[int] = set()
        for _, agent, pos_y, pos_x in candidates:
            if agent.spec.y in taken or agent.spec.x in taken:
                continue
            taken.add(agent.spec.y)
            taken.add(agent.spec.x)
            pos[agent.spec.y] += pos_y
            pos[agent.spec.x] += pos_x

        return pos


_STATE: dict[str, AtlasBook | None] = {{"atlas": None}}


def reset() -> None:
    _STATE["atlas"] = None


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < WARMUP + 10:
        return np.zeros(nins, dtype=int)

    if _STATE["atlas"] is None:
        _STATE["atlas"] = AtlasBook()

    dollars = _STATE["atlas"].dollars(prcSoFar)

    safe_prices = np.maximum(prcSoFar[:, -1], _base._EPS_DENOM)
    last_return = np.diff(
        np.log(np.maximum(prcSoFar[:, -2:], _base._EPS_DENOM)),
        axis=1,
    )[:, 0]
    crash = np.abs(last_return) > 0.40
    dollars[crash] = 0.0

    dollars = np.clip(dollars, -LEG, LEG)

    raw_shares = dollars / safe_prices
    final_shares = np.round(raw_shares).astype(int)
    return np.nan_to_num(final_shares, nan=0).astype(int)
'''


def write_filtered_module(kept_pair_names: list[str], path: Path) -> None:
    names_block = "\n".join(f'        "{name}",' for name in sorted(kept_pair_names))
    content = FILTERED_MODULE_TEMPLATE.format(names_block=names_block)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = load_strategy(MODULE)
    prices, price_symbols = load_prices(ROOT / "prices.txt")
    train_prices = prices[:, :750].copy()
    log_prices = np.log(train_prices)
    symbols = symbols_for(strategy)
    expected = [symbols[index] for index in range(strategy.N_INST)]
    if price_symbols != expected:
        raise ValueError("prices.txt symbols do not match strategy instrument map")

    records = [
        analyse_pair(log_prices, cfg, strategy, symbols) for cfg in strategy.PAIR_CFGS
    ]
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(OUTPUT_DIR / "all_pair_corr_cusum_metrics.csv", index=False)
    frame.loc[frame["keep"]].to_csv(OUTPUT_DIR / "kept_pairs.csv", index=False)
    frame.loc[~frame["keep"]].to_csv(OUTPUT_DIR / "rejected_pairs.csv", index=False)
    (OUTPUT_DIR / "kept_pair_names.json").write_text(
        json.dumps(frame.loc[frame["keep"], "pair_name"].tolist(), indent=2),
        encoding="utf-8",
    )

    prior_kept: set[str] = set()
    if PRIOR_KEPT_NAMES_PATH.exists():
        prior_kept = set(json.loads(PRIOR_KEPT_NAMES_PATH.read_text(encoding="utf-8")))

    write_report(frame, prior_kept, OUTPUT_DIR / "corr_cusum_screen_report.md")
    make_plot(frame, OUTPUT_DIR / "corr_cusum_screen_diagnostics.png")

    kept_names = frame.loc[frame["keep"], "pair_name"].tolist()
    write_filtered_module(kept_names, FILTERED_MODULE_PATH)

    print(f"Analysed {len(frame)} pairs: {int(frame['keep'].sum())} kept, {int((~frame['keep']).sum())} rejected")
    print(f"Prior strict list: {len(prior_kept)} pairs; overlap: {len(prior_kept & set(frame.loc[frame['keep'], 'pair_name']))}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Filtered module written: {FILTERED_MODULE_PATH}")


if __name__ == "__main__":
    main()
