"""Correlation + CUSUM filtered version of :mod:`testbedv2_tuned`.

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
    {
        "ACAC-IHOZ",
        "ACAC-NAYO",
        "AENO-ACAC",
        "AENO-AETS",
        "AENO-HUXZ",
        "AENO-IHOZ",
        "AENO-NWIG",
        "ALGO-HRET",
        "ALUT-BLBT",
        "ALUT-CCNS",
        "ALUT-FWWG",
        "ALUT-IHOZ",
        "ALUT-RCRI",
        "ALUT-SRTX",
        "AMRP-ULXY",
        "ANSO-ACAC",
        "ANSO-CTGI",
        "ANSO-IHOZ",
        "ANSO-RCRI",
        "CTGI-EELT",
        "CUBO-ACIX",
        "CUBO-ULXY",
        "DUCT-SRTX",
        "ELLT-AETS",
        "ELLT-FCSG",
        "ELLT-HTRK",
        "EORC-AETS",
        "EORC-NGTE",
        "HETT-MTNS",
        "HETT-ULXY",
        "HRET-HTRK",
        "HRET-SRTX",
        "HUXZ-ACAC",
        "HUXZ-AETS",
        "HUXZ-BLBT",
        "HUXZ-CCNS",
        "HUXZ-IHOZ",
        "HUXZ-NWIG",
        "HUXZ-RRES",
        "HUXZ-ULXY",
        "LSST-ANSO",
        "LSST-HRET",
        "LSST-SMAH",
        "MDGI-MTNS",
        "MSDP-SRTX",
        "MSDP-ULXY",
        "NPCK-NGTE",
        "NPCK-RCRI",
        "RRES-EELT",
        "RTTH-ACAC",
        "RTTH-FARS",
        "RTTH-FWWG",
        "RTTH-MDGI",
        "RTTH-NAYO",
        "RTTH-RCRI",
        "SMAH-ILVX",
        "SMAH-ULXY",
        "SPLZ-AETS",
        "SPLZ-BLBT",
        "SPLZ-CCNS",
        "SRNA-ALUT",
        "SRNA-FCSG",
        "SRNA-HUXZ",
        "SRNA-NAYO",
        "SRNA-RTTH",
        "SRTX-AETS",
        "SRTX-BENI",
        "SRTX-BLBT",
        "SRTX-FWWG",
        "SRTX-IHOZ",
        "SRTX-MTNS",
    }
)

_SYMBOL_BY_INDEX = {index: symbol for symbol, index in I.items()}


def _pair_name(spec: PairCfg) -> str:
    return f"{_SYMBOL_BY_INDEX[spec.y]}-{_SYMBOL_BY_INDEX[spec.x]}"


# Keep the original pair orientation and every tuned parameter.  Reindexing
# only makes the surviving PairCfg identifiers contiguous.
PAIR_CFGS = tuple(
    spec._replace(idx=new_idx)
    for new_idx, spec in enumerate(
        spec for spec in _base.PAIR_CFGS if _pair_name(spec) in SELECTED_PAIR_NAMES
    )
)

if {_pair_name(spec) for spec in PAIR_CFGS} != SELECTED_PAIR_NAMES:
    missing = SELECTED_PAIR_NAMES - {_pair_name(spec) for spec in PAIR_CFGS}
    raise RuntimeError(f"Filtered pair configuration is missing: {sorted(missing)}")


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


_STATE: dict[str, AtlasBook | None] = {"atlas": None}


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
