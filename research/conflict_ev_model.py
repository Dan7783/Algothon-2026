#!/usr/bin/env python
"""Learned arbitration for pair/ridge conflicts: EV first, black box second.

THE PROBLEM, REDUCED. On a conflicted instrument-day the two sleeves point
opposite ways by definition (pair_dir = -ridge_sign), so

    PnL(follow pair) = -PnL(follow ridge)

exactly. Arbitration is therefore ONE binary bet per conflicted asset-day:
"is ridge right here?" The natural EV target is the signed quantity

    ev = ridge_sign * forward_return

Positive ev means following ridge earns; negative means following the pair
earns. A model that predicts sign(E[ev]) IS the arbitrator, and predicting
E[ev] (magnitude included) lets it concentrate on conflicts that matter.

WHY EV BEFORE A BLACK BOX. The diagnostic established that the conflict
winner is regime-unstable: following ridge on conflicts LOSES $7.9k on train
and MAKES $15.3k / $30.8k on validation / test. A statically-fitted model
learns the train regime and inverts out of sample -- exactly the failure that
sank the s2 "pairs win" scheme (train 865, test 677). So the first model is
an ONLINE EV estimator that is causal by construction (every prediction uses
only strictly-prior conflicts) and adapts when the regime turns, rather than
freezing one window's answer.

MODELS
  A  online global EWMA EV     -- no features, just "who has been right lately"
  B  online EV per strength bucket -- A, conditioned on ridge confidence
  C  static linear EV regression   -- fit on train blocks only
  D  static logistic classifier    -- fit on train blocks only
  E  gradient boosting (black box) -- fallback, train blocks only

Every model is scored the way it will actually be used: substituted into the
live allocator on conflicted assets only, then run through the exact eval.py
simulator. Classification accuracy is reported too, but score is the verdict.

Selection remains train-only per cusum_output/VALIDATION_PROTOCOL.md; the
online models have no fitted window at all, so their only choices are the
halflife/bucket count, which are checked for a plateau rather than a peak.

RESULT (2026-07-27): NEGATIVE -- kept as the record of a closed question.
All eight variants lose on test against the 914.7 baseline (best: online
global EWMA hl60 at 869.9; worst: static logistic at 720.8). Gradient
boosting reaches train 1488.3 -- nearly double the baseline -- and 801.4 on
test, i.e. worse than not arbitrating at all; train AUC 0.845 vs test AUC
0.514. The cause is that the label inverts across the split: P(ridge right)
is 0.473 on train but 0.522 on test, each only ~2 sigma at n~2000. Test AUC
is chance for every model, restricting to high-confidence predictions makes
accuracy worse rather than better, and no feature holds a stable sign across
regions (largest |corr| 0.066 against SE 0.022). Conditioning on disagreement
selects exactly the instrument-days where the two sleeves' information
cancels, so there is nothing left to learn. Do not re-run this hoping for a
different answer; see analysis_outputs/signal_overlap/OVERLAP_VERDICT.md 7c.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.combine_lab import (
    DLR_LIMIT,
    N_INST,
    TEST,
    TEST_BLOCKS,
    TRAIN,
    TRAIN_BLOCKS,
    VAL_BLOCKS,
    VALIDATION,
    region_report,
    score,
    simulate,
)
from research.combine_schemes import _resolve, _sleeve_state
from research.signal_cache import load_or_build, load_prices

import testbedv2_tuned_ridge_protectedbarbell as strategy

OUTPUT_DIR = ROOT / "analysis_outputs" / "signal_overlap"
BB = strategy.BANG_BANG_LIMITS
FEATURES = [
    "ridge_strength",
    "abs_consensus",
    "contributors",
    "max_priority",
    "strength_rank",
    "vol_20d",
    "abs_z_proxy",
    "protected",
    "gate_links",
    "vote_margin",
]


def build_dataset(cache: dict, prices: np.ndarray) -> pd.DataFrame:
    """Every conflicted instrument-day with decision-time-only features."""
    lookup = {int(day): row for row, day in enumerate(cache["days"])}
    y_idx = cache["pair_y_index"]
    x_idx = cache["pair_x_index"]
    protected_pair = cache["pair_protected"]
    log_prices = np.log(np.maximum(prices, 1e-12))
    records = []

    for t in range(251, 1000):
        row = lookup[t]
        state = _sleeve_state(cache, row, prices, t)
        pair_active = state["pair_active"]
        ridge_active = state["ridge_active"]
        pair_dir = state["pair_dir"]
        ridge_sign = state["ridge_sign"]
        conflict = pair_active & ridge_active & (pair_dir != ridge_sign)
        conflict = conflict & ~state["crash"]
        if not conflict.any():
            continue

        strength = state["ridge_strength"]
        consensus = state["consensus"]
        # Contributor count and best pair priority per instrument.
        contributors = np.zeros(N_INST)
        max_priority = np.zeros(N_INST)
        pos_y = cache["pair_pos_y"][row]
        priority = cache["pair_priority"][row]
        is_protected = np.zeros(N_INST, dtype=bool)
        for pair_idx in range(len(y_idx)):
            if abs(pos_y[pair_idx]) <= 1e-9:
                continue
            for inst in (int(y_idx[pair_idx]), int(x_idx[pair_idx])):
                contributors[inst] += 1
                max_priority[inst] = max(max_priority[inst], priority[pair_idx])
                if protected_pair[pair_idx]:
                    is_protected[inst] = True

        # Cross-sectional rank of ridge conviction that day.
        active_strength = strength[ridge_active]
        rank = np.zeros(N_INST)
        if len(active_strength):
            order = np.argsort(np.argsort(active_strength))
            rank[ridge_active] = order / max(len(active_strength) - 1, 1)

        # Regime proxy: how many significant lead-lag links the ridge gate saw.
        recent = np.diff(log_prices[1:, max(0, t - 250):t], axis=1).T
        if len(recent) > 2:
            std = recent.std(axis=0) + 1e-12
            z = (recent - recent.mean(axis=0)) / std
            links = (z[:-1].T @ z[1:]) / (len(recent) - 1)
            np.fill_diagonal(links, 0.0)
            gate_links = int((np.abs(links) > 2.5 / np.sqrt(len(recent) - 1)).sum())
        else:
            gate_links = 0

        returns_20 = np.diff(log_prices[:, max(0, t - 21):t], axis=1)
        vol_20 = returns_20.std(axis=1) if returns_20.shape[1] > 1 else np.zeros(N_INST)

        forward = prices[:, t] / prices[:, t - 1] - 1.0
        combined = consensus + state["ridge_vote"]

        for inst in np.flatnonzero(conflict):
            ev = float(ridge_sign[inst] * forward[inst])
            records.append(
                {
                    "day": t,
                    "inst": int(inst),
                    "ridge_strength": float(strength[inst]),
                    "abs_consensus": float(abs(consensus[inst])),
                    "contributors": float(contributors[inst]),
                    "max_priority": float(max_priority[inst]),
                    "strength_rank": float(rank[inst]),
                    "vol_20d": float(vol_20[inst]),
                    "abs_z_proxy": float(max_priority[inst] / max(vol_20[inst], 1e-6)),
                    "protected": float(is_protected[inst]),
                    "gate_links": float(gate_links),
                    "vote_margin": float(abs(combined[inst])),
                    "ridge_sign": float(ridge_sign[inst]),
                    "ev": ev,
                    "is_ridge_right": float(ev > 0),
                    "shipped_follows_ridge": float(np.sign(combined[inst]) == ridge_sign[inst]),
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Arbitrators. Each returns, for a conflicted instrument-day, +1 to follow
# ridge or -1 to follow the pair. All must be causal.
# --------------------------------------------------------------------------

class OnlineGlobalEV:
    """EWMA of realised ev over past conflicts. No features, fully causal."""

    def __init__(self, halflife: float = 60.0, min_obs: int = 50):
        self.decay = 0.5 ** (1.0 / halflife)
        self.min_obs = min_obs
        self.value = 0.0
        self.weight = 0.0
        self.seen = 0

    def decide(self, features: dict) -> float:
        if self.seen < self.min_obs:
            return 0.0  # abstain -> fall back to shipped rule
        return 1.0 if self.value > 0.0 else -1.0

    def update(self, features: dict, ev: float) -> None:
        self.value = self.decay * self.value + ev
        self.weight = self.decay * self.weight + 1.0
        self.seen += 1


class OnlineBucketEV:
    """Online EV conditioned on ridge-confidence bucket."""

    def __init__(self, edges=(1.0, 1.5, 2.5), halflife: float = 60.0, min_obs: int = 25):
        self.edges = edges
        self.decay = 0.5 ** (1.0 / halflife)
        self.min_obs = min_obs
        n = len(edges) + 1
        self.value = np.zeros(n)
        self.seen = np.zeros(n, dtype=int)

    def _bucket(self, features: dict) -> int:
        return int(np.searchsorted(self.edges, features["ridge_strength"]))

    def decide(self, features: dict) -> float:
        b = self._bucket(features)
        if self.seen[b] < self.min_obs:
            return 0.0
        return 1.0 if self.value[b] > 0.0 else -1.0

    def update(self, features: dict, ev: float) -> None:
        b = self._bucket(features)
        self.value[b] = self.decay * self.value[b] + ev
        self.seen[b] += 1


class StaticModel:
    """Wraps a fitted sklearn estimator; predicts follow-ridge vs follow-pair."""

    def __init__(self, model, kind: str, threshold: float = 0.0):
        self.model = model
        self.kind = kind
        self.threshold = threshold

    def decide(self, features: dict) -> float:
        x = np.array([[features[name] for name in FEATURES]])
        if self.kind == "regressor":
            pred = float(self.model.predict(x)[0])
            return 1.0 if pred > self.threshold else -1.0
        prob = float(self.model.predict_proba(x)[0, 1])
        return 1.0 if prob > 0.5 else -1.0

    def update(self, features: dict, ev: float) -> None:
        pass


def evaluate_arbitrator(factory, cache, prices, start, end, warm_from=250):
    """Run the live allocator, replacing only conflict decisions.

    The arbitrator is warmed from `warm_from` so online models enter the
    scored window with history, but it only ever sees strictly-past labels.
    """
    lookup = {int(day): row for row, day in enumerate(cache["days"])}
    arbitrator = factory()
    dollars_by_day = {}
    n_used = n_total = 0

    for t in range(warm_from, end):
        row = lookup[t]
        state = _sleeve_state(cache, row, prices, t)
        direction = _resolve(state)
        conflict = (
            state["pair_active"]
            & state["ridge_active"]
            & (state["pair_dir"] != state["ridge_sign"])
            & ~state["crash"]
        )
        forward = prices[:, t] / prices[:, t - 1] - 1.0

        pending = []
        for inst in np.flatnonzero(conflict):
            feats = _features_for(cache, prices, state, row, t, int(inst))
            choice = arbitrator.decide(feats)
            if t >= start - 1:
                n_total += 1
                if choice != 0.0:
                    n_used += 1
            if choice != 0.0:
                direction[inst] = (
                    state["ridge_sign"][inst] if choice > 0 else state["pair_dir"][inst]
                )
            pending.append((feats, float(state["ridge_sign"][inst] * forward[inst])))

        if t >= start - 1:
            direction = direction.copy()
            direction[state["crash"]] = 0.0
            dollars_by_day[t] = np.clip(BB * direction, -DLR_LIMIT, DLR_LIMIT)

        for feats, ev in pending:
            arbitrator.update(feats, ev)

    pnl = simulate(dollars_by_day, prices, start, end)
    return pnl, (n_used / max(n_total, 1))


_FEATURE_CACHE: dict = {}


def _features_for(cache, prices, state, row, t, inst) -> dict:
    """Per-instrument features, reusing the per-day parts."""
    key = t
    if _FEATURE_CACHE.get("day") != key:
        log_prices = np.log(np.maximum(prices, 1e-12))
        recent = np.diff(log_prices[1:, max(0, t - 250):t], axis=1).T
        if len(recent) > 2:
            std = recent.std(axis=0) + 1e-12
            z = (recent - recent.mean(axis=0)) / std
            links = (z[:-1].T @ z[1:]) / (len(recent) - 1)
            np.fill_diagonal(links, 0.0)
            gate_links = int((np.abs(links) > 2.5 / np.sqrt(len(recent) - 1)).sum())
        else:
            gate_links = 0
        returns_20 = np.diff(log_prices[:, max(0, t - 21):t], axis=1)
        vol_20 = returns_20.std(axis=1) if returns_20.shape[1] > 1 else np.zeros(N_INST)

        contributors = np.zeros(N_INST)
        max_priority = np.zeros(N_INST)
        is_protected = np.zeros(N_INST, dtype=bool)
        pos_y = cache["pair_pos_y"][row]
        priority = cache["pair_priority"][row]
        for pair_idx in range(len(cache["pair_y_index"])):
            if abs(pos_y[pair_idx]) <= 1e-9:
                continue
            for i2 in (int(cache["pair_y_index"][pair_idx]), int(cache["pair_x_index"][pair_idx])):
                contributors[i2] += 1
                max_priority[i2] = max(max_priority[i2], priority[pair_idx])
                if cache["pair_protected"][pair_idx]:
                    is_protected[i2] = True

        strength = state["ridge_strength"]
        ridge_active = state["ridge_active"]
        rank = np.zeros(N_INST)
        act = strength[ridge_active]
        if len(act):
            rank[ridge_active] = np.argsort(np.argsort(act)) / max(len(act) - 1, 1)

        _FEATURE_CACHE.clear()
        _FEATURE_CACHE.update(
            day=key, gate_links=gate_links, vol_20=vol_20, contributors=contributors,
            max_priority=max_priority, is_protected=is_protected, rank=rank,
        )

    c = _FEATURE_CACHE
    combined = state["consensus"] + state["ridge_vote"]
    return {
        "ridge_strength": float(state["ridge_strength"][inst]),
        "abs_consensus": float(abs(state["consensus"][inst])),
        "contributors": float(c["contributors"][inst]),
        "max_priority": float(c["max_priority"][inst]),
        "strength_rank": float(c["rank"][inst]),
        "vol_20d": float(c["vol_20"][inst]),
        "abs_z_proxy": float(c["max_priority"][inst] / max(c["vol_20"][inst], 1e-6)),
        "protected": float(c["is_protected"][inst]),
        "gate_links": float(c["gate_links"]),
        "vote_margin": float(abs(combined[inst])),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    cache = load_or_build()

    print("Building conflict dataset ...")
    data = build_dataset(cache, prices)
    data.to_csv(OUTPUT_DIR / "conflict_dataset.csv", index=False)
    train_mask = (data["day"] >= 251) & (data["day"] <= 500)
    val_mask = (data["day"] >= 501) & (data["day"] <= 750)
    test_mask = (data["day"] >= 751)
    print(f"  conflicts: {len(data)} total | train {train_mask.sum()} "
          f"| val {val_mask.sum()} | test {test_mask.sum()}")
    print(f"  base rate P(ridge right): train {data.loc[train_mask,'is_ridge_right'].mean():.4f} "
          f"| val {data.loc[val_mask,'is_ridge_right'].mean():.4f} "
          f"| test {data.loc[test_mask,'is_ridge_right'].mean():.4f}")
    print(f"  mean ev (follow ridge):   train {data.loc[train_mask,'ev'].mean():+.6f} "
          f"| val {data.loc[val_mask,'ev'].mean():+.6f} "
          f"| test {data.loc[test_mask,'ev'].mean():+.6f}")

    # ---- static models, fit on TRAIN BLOCKS ONLY -------------------------
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = data.loc[train_mask, FEATURES].to_numpy()
    ev_train = data.loc[train_mask, "ev"].to_numpy()
    y_train = data.loc[train_mask, "is_ridge_right"].to_numpy()

    linear_ev = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, ev_train)
    logistic = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.1)).fit(x_train, y_train)
    boosted = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=0).fit(x_train, y_train)

    factories = {
        "A_online_global_ev_hl60": lambda: OnlineGlobalEV(halflife=60.0),
        "A_online_global_ev_hl120": lambda: OnlineGlobalEV(halflife=120.0),
        "A_online_global_ev_hl250": lambda: OnlineGlobalEV(halflife=250.0),
        "B_online_bucket_ev_hl60": lambda: OnlineBucketEV(halflife=60.0),
        "B_online_bucket_ev_hl120": lambda: OnlineBucketEV(halflife=120.0),
        "C_static_linear_ev": lambda: StaticModel(linear_ev, "regressor"),
        "D_static_logistic": lambda: StaticModel(logistic, "classifier"),
        "E_gradient_boosting": lambda: StaticModel(boosted, "classifier"),
    }

    print("\nEvaluating arbitrators through the real simulator ...")
    rows = []
    for name, factory in factories.items():
        record = {"model": name}
        for label, (_, start, end), blocks in (
            ("train", TRAIN, TRAIN_BLOCKS),
            ("val", VALIDATION, VAL_BLOCKS),
            ("test", TEST, TEST_BLOCKS),
        ):
            pnl, used = evaluate_arbitrator(factory, cache, prices, start, end)
            rep = region_report(pnl, start, end, blocks)
            record[f"{label}_score"] = rep["score"]
            record[f"{label}_floor"] = rep["floor"]
            record[f"{label}_b1"] = rep["blocks"][0]
            record[f"{label}_b2"] = rep["blocks"][1]
            record[f"{label}_coverage"] = used
        rows.append(record)
        print(f"  {name}: train {record['train_score']:.1f} "
              f"val {record['val_score']:.1f} test {record['test_score']:.1f}")

    frame = pd.DataFrame(rows).set_index("model")
    frame.to_csv(OUTPUT_DIR / "conflict_model_results.csv")

    baseline = {"train": 789.8, "val": 847.0, "test": 914.7}
    oracle = {"train": 2195.7, "val": 2124.0, "test": 2256.8}
    print("\n" + "=" * 104)
    print("RESULTS vs shipped baseline (train-only selection; both train blocks must beat it)")
    print("=" * 104)
    print(f"{'model':<28} {'tr_b1':>8} {'tr_b2':>8} {'train':>8} {'val':>8} {'test':>8} "
          f"{'d_test':>8} {'cov':>6} {'gate':>6}")
    print("-" * 104)
    print(f"{'baseline (shipped)':<28} {649.7:8.1f} {929.5:8.1f} {789.8:8.1f} "
          f"{847.0:8.1f} {914.7:8.1f} {0.0:8.1f} {'-':>6} {'-':>6}")
    for name, row in frame.iterrows():
        gate = "PASS" if (row["train_b1"] > 649.7 and row["train_b2"] > 929.5) else "no"
        print(f"{name:<28} {row['train_b1']:8.1f} {row['train_b2']:8.1f} "
              f"{row['train_score']:8.1f} {row['val_score']:8.1f} {row['test_score']:8.1f} "
              f"{row['test_score']-914.7:+8.1f} {row['test_coverage']:6.2f} {gate:>6}")
    print(f"\noracle ceiling for reference: train {oracle['train']:.1f} "
          f"val {oracle['val']:.1f} test {oracle['test']:.1f}")
    print(f"wrote {OUTPUT_DIR / 'conflict_model_results.csv'}")


if __name__ == "__main__":
    main()
