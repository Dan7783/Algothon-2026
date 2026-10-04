"""Investigate pair-pair conflict winners on friend_book_latest.

For each day/asset where active pairs disagree on sign, record:
  - per-side aggregates: n_pairs, sum/max kappa, sigma proxy, abs_z, priority
  - protected presence, current bang-bang choice
  - label = sign of next-day log return (profitable direction)

Compare rules (fit nothing on OOS; report train 31-500 / val 501-750):
  majority, protected_wins, max_kappa, max_priority (kappa*|z|), max_abs_z,
  ridge_if_active, shallow DecisionTree on side features.

Then bake-off book scores for conflict modes judged on train/val only.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "auto_tuning_results" / "_conflict_priority_study.json"
RESULTS = ROOT / "auto_tuning_results"

FIT = (31, 500)
VAL = (501, 750)
BOOK_TRAIN = (31, 750)
OOS_VERIFY = (750, 1000)
DEADZONE = 1e-4


def clear_env():
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("MERGED_") or k.startswith("FRIEND_"):
            del env[k]
    return env


def load_prices():
    import latest_tuner

    prices, _, _ = latest_tuner.loadPrices("prices.txt")
    return np.asarray(prices, dtype=float)


def load_friend():
    for name in list(sys.modules):
        if name == "friend_book_latest" or name.startswith("friend_book_latest."):
            del sys.modules[name]
    return importlib.import_module("friend_book_latest")


def collect_pair_pair_conflicts(mod, prices, start, end):
    """Asset-days where ≥2 pairs pull opposite signs (true pair conflicts)."""
    mod.reset()
    rows = []
    sym = {i: s for s, i in mod.I.items()}

    for t in range(start, end - 1):
        prc = prices[:, : t + 1]
        # drive state
        sleeves = mod.sleeve_breakdown(prc)
        if sleeves is None:
            continue
        cands = mod._STATE["atlas"].candidates(prc)
        if not cands:
            continue

        # group legs by asset
        # asset -> list of (sign, kappa, abs_z, priority, protected, pair)
        by_asset: dict[int, list] = {}
        for priority, agent, pos_y, pos_x in cands:
            prot = mod._pair_name(agent.spec) in mod.PROTECTED_PAIR_NAMES
            kappa = float(agent.current_kappa)
            sigma = float(getattr(agent, "current_sigma", 0.0))
            abs_z = abs(float(agent.current_z))
            for idx, pos in ((int(agent.spec.y), pos_y), (int(agent.spec.x), pos_x)):
                if abs(pos) <= 1e-9:
                    continue
                by_asset.setdefault(idx, []).append(
                    {
                        "sign": int(np.sign(pos)),
                        "kappa": kappa,
                        "sigma": sigma,
                        "abs_z": abs_z,
                        "priority": float(priority),
                        "protected": prot,
                        "pair": mod._pair_name(agent.spec),
                    }
                )

        # next-day returns
        if t + 1 >= prices.shape[1]:
            break
        rets = np.log(np.maximum(prices[:, t + 1], 1e-12)) - np.log(
            np.maximum(prices[:, t], 1e-12)
        )

        pair_vote = np.asarray(sleeves["pair_vote"], dtype=float)
        direction = np.asarray(sleeves["direction"], dtype=float)
        ridge_sign = np.sign(np.asarray(sleeves["ridge_raw"], dtype=float))
        ridge_active = np.asarray(sleeves["ridge_active_mask"], dtype=bool)

        for asset, legs in by_asset.items():
            if asset == 0:  # ALGO
                continue
            signs = {leg["sign"] for leg in legs}
            if len(signs) < 2 or 0 in signs and len(signs) == 2:
                # need both +1 and -1
                if not (+1 in signs and -1 in signs):
                    continue

            long_legs = [L for L in legs if L["sign"] > 0]
            short_legs = [L for L in legs if L["sign"] < 0]
            if not long_legs or not short_legs:
                continue

            def agg(group):
                return {
                    "n": len(group),
                    "sum_kappa": sum(L["kappa"] for L in group),
                    "max_kappa": max(L["kappa"] for L in group),
                    "sum_sigma": sum(L["sigma"] for L in group),
                    "max_sigma": max(L["sigma"] for L in group),
                    "sum_abs_z": sum(L["abs_z"] for L in group),
                    "max_abs_z": max(L["abs_z"] for L in group),
                    "sum_priority": sum(L["priority"] for L in group),
                    "max_priority": max(L["priority"] for L in group),
                    "any_protected": any(L["protected"] for L in group),
                }

            A = agg(long_legs)
            B = agg(short_legs)
            nxt = float(rets[asset])
            if abs(nxt) < DEADZONE:
                label = 0
            else:
                label = int(np.sign(nxt))  # +1 if long would win

            chosen = int(np.sign(direction[asset]))
            rows.append(
                {
                    "t": t,
                    "asset": int(asset),
                    "symbol": sym.get(asset, str(asset)),
                    "label": label,
                    "next_ret": nxt,
                    "chosen_bangbang": chosen,
                    "pair_vote_sign": int(np.sign(pair_vote[asset])),
                    "ridge_sign": int(ridge_sign[asset]) if ridge_active[asset] else 0,
                    "n_long": A["n"],
                    "n_short": B["n"],
                    "long": A,
                    "short": B,
                }
            )
    return rows


def rule_pick(row, rule: str) -> int:
    """Return +1 (long) / -1 (short) / 0 (flat)."""
    L, S = row["long"], row["short"]
    if rule == "majority":
        if L["n"] > S["n"]:
            return 1
        if S["n"] > L["n"]:
            return -1
        return 0
    if rule == "protected":
        if L["any_protected"] and not S["any_protected"]:
            return 1
        if S["any_protected"] and not L["any_protected"]:
            return -1
        # fall back majority
        return rule_pick(row, "majority")
    if rule == "max_kappa":
        if L["max_kappa"] > S["max_kappa"]:
            return 1
        if S["max_kappa"] > L["max_kappa"]:
            return -1
        return 0
    if rule == "sum_kappa":
        if L["sum_kappa"] > S["sum_kappa"]:
            return 1
        if S["sum_kappa"] > L["sum_kappa"]:
            return -1
        return 0
    if rule == "max_priority":
        if L["max_priority"] > S["max_priority"]:
            return 1
        if S["max_priority"] > L["max_priority"]:
            return -1
        return 0
    if rule == "sum_priority":
        if L["sum_priority"] > S["sum_priority"]:
            return 1
        if S["sum_priority"] > L["sum_priority"]:
            return -1
        return 0
    if rule == "max_abs_z":
        if L["max_abs_z"] > S["max_abs_z"]:
            return 1
        if S["max_abs_z"] > L["max_abs_z"]:
            return -1
        return 0
    if rule == "max_sigma":
        # higher sigma = noisier → prefer LOWER sigma side
        if L["max_sigma"] < S["max_sigma"]:
            return 1
        if S["max_sigma"] < L["max_sigma"]:
            return -1
        return 0
    if rule == "bangbang":
        return row["chosen_bangbang"]
    if rule == "pair_vote":
        return row["pair_vote_sign"]
    if rule == "oracle":
        return row["label"]
    return 0


def score_rule(rows, rule: str):
    """Accuracy on non-flat labels + mean signed capture of |ret| when correct."""
    y_true, y_pred = [], []
    pnl = 0.0
    n = 0
    for r in rows:
        if r["label"] == 0:
            continue
        pred = rule_pick(r, rule)
        y_true.append(r["label"])
        y_pred.append(pred if pred != 0 else 0)
        if pred != 0:
            pnl += pred * r["next_ret"]
            n += 1
    if not y_true:
        return {"acc": None, "n": 0, "mean_edge": None}
    # treat flat pred as wrong for acc
    acc = accuracy_score(y_true, [p if p != 0 else -y for y, p in zip(y_true, y_pred)])
    # simpler: fraction where pred == label
    hit = sum(1 for yt, yp in zip(y_true, y_pred) if yp == yt)
    return {
        "acc": hit / len(y_true),
        "n": len(y_true),
        "mean_edge": pnl / max(n, 1),
        "flat_rate": sum(1 for yp in y_pred if yp == 0) / len(y_pred),
    }


def side_features(row):
    """Feature vector for tree: long minus short aggregates + meta."""
    L, S = row["long"], row["short"]
    return [
        L["n"] - S["n"],
        L["sum_kappa"] - S["sum_kappa"],
        L["max_kappa"] - S["max_kappa"],
        L["sum_priority"] - S["sum_priority"],
        L["max_priority"] - S["max_priority"],
        L["sum_abs_z"] - S["sum_abs_z"],
        L["max_abs_z"] - S["max_abs_z"],
        L["max_sigma"] - S["max_sigma"],
        float(L["any_protected"]) - float(S["any_protected"]),
        float(row["ridge_sign"]),
        float(row["pair_vote_sign"]),
    ]


FEAT_NAMES = [
    "dn", "d_sum_kappa", "d_max_kappa", "d_sum_pri", "d_max_pri",
    "d_sum_z", "d_max_z", "d_max_sigma", "d_prot", "ridge_sign", "pair_vote",
]


def train_tree(fit_rows, val_rows):
    Xtr, ytr = [], []
    for r in fit_rows:
        if r["label"] == 0:
            continue
        Xtr.append(side_features(r))
        ytr.append(r["label"])
    Xva, yva = [], []
    for r in val_rows:
        if r["label"] == 0:
            continue
        Xva.append(side_features(r))
        yva.append(r["label"])
    if len(set(ytr)) < 2 or len(Xtr) < 50:
        return None, {"error": "insufficient fit data"}

    best = None
    for depth in (2, 3, 4):
        clf = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=30, random_state=0
        )
        clf.fit(Xtr, ytr)
        tr_acc = accuracy_score(ytr, clf.predict(Xtr))
        va_acc = accuracy_score(yva, clf.predict(Xva)) if Xva else None
        cand = {
            "depth": depth,
            "fit_acc": float(tr_acc),
            "val_acc": float(va_acc) if va_acc is not None else None,
            "n_fit": len(ytr),
            "n_val": len(yva),
            "tree_text": export_text(clf, feature_names=FEAT_NAMES, max_depth=4),
        }
        if best is None or (cand["val_acc"] or 0) > (best["val_acc"] or 0):
            best = cand
            best_clf = clf
    return best_clf, best


def eval_score(module: str, start: int, end: int, env_extra: dict | None = None):
    env = clear_env()
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", module, "--start", str(start), "--end", str(end)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    m = re.search(r"Score:\s*([\d.]+)", p.stdout)
    sh = re.search(r"annSharpe\(PL\):\s*([\d.]+)", p.stdout)
    if not m:
        return {"error": p.stderr[-500:] or p.stdout[-500:]}
    return {
        "score": float(m.group(1)),
        "sharpe": float(sh.group(1)) if sh else None,
    }


def main():
    prices = load_prices()
    mod = load_friend()
    print("collecting fit conflicts...", flush=True)
    fit_rows = collect_pair_pair_conflicts(mod, prices, FIT[0], FIT[1])
    print(f"  fit pair-pair conflicts: {len(fit_rows)}", flush=True)
    print("collecting val conflicts...", flush=True)
    # fresh state
    mod = load_friend()
    val_rows = collect_pair_pair_conflicts(mod, prices, VAL[0], VAL[1])
    print(f"  val pair-pair conflicts: {len(val_rows)}", flush=True)

    rules = [
        "bangbang",
        "pair_vote",
        "majority",
        "protected",
        "max_kappa",
        "sum_kappa",
        "max_priority",
        "sum_priority",
        "max_abs_z",
        "max_sigma",
        "oracle",
    ]
    rule_table = {}
    for rule in rules:
        rule_table[rule] = {
            "fit": score_rule(fit_rows, rule),
            "val": score_rule(val_rows, rule),
        }
        print(
            f"  {rule:14} fit_acc={rule_table[rule]['fit']['acc']} "
            f"val_acc={rule_table[rule]['val']['acc']}",
            flush=True,
        )

    clf, tree_info = train_tree(fit_rows, val_rows)
    tree_scores = None
    if clf is not None:
        def tree_rule_rows(rows):
            hit = n = 0
            for r in rows:
                if r["label"] == 0:
                    continue
                pred = int(clf.predict([side_features(r)])[0])
                hit += int(pred == r["label"])
                n += 1
            return {"acc": hit / n if n else None, "n": n}

        tree_scores = {
            "model": tree_info,
            "fit": tree_rule_rows(fit_rows),
            "val": tree_rule_rows(val_rows),
        }
        print(
            f"  tree depth={tree_info['depth']} "
            f"fit={tree_scores['fit']['acc']:.3f} val={tree_scores['val']['acc']:.3f}",
            flush=True,
        )
        print(tree_info.get("tree_text", "")[:800], flush=True)

    # How often does bang-bang agree with max_priority / max_kappa?
    agree = {"priority": 0, "kappa": 0, "n": 0}
    for r in fit_rows + val_rows:
        if r["label"] == 0:
            continue
        agree["n"] += 1
        bb = r["chosen_bangbang"]
        if bb == rule_pick(r, "max_priority"):
            agree["priority"] += 1
        if bb == rule_pick(r, "max_kappa"):
            agree["kappa"] += 1

    # Bake-off: current mode vs priority mode via env on a thin wrapper module.
    # friend_book_latest hardcodes mode; monkeypatch file copy with env.
    print("bake-off book scores...", flush=True)
    # Write tiny override modules
    for mode, fname in (
        ("we love sussy q", "_tmp_friend_equal.py"),
        ("priority", "_tmp_friend_priority.py"),
    ):
        src = (ROOT / "friend_book_latest.py").read_text(encoding="utf-8")
        src = src.replace(
            'PAIR_CONFLICT_MODE = "we love sussy q"',
            f'PAIR_CONFLICT_MODE = "{mode}"',
            1,
        )
        # rename module docstring lightly
        (ROOT / fname).write_text(src, encoding="utf-8")

    book = {}
    for label, modname in (
        ("equal_boost_current", "_tmp_friend_equal"),
        ("priority_kappa_z", "_tmp_friend_priority"),
    ):
        book[label] = {
            "train_31_750": eval_score(modname, *BOOK_TRAIN),
            "val_501_750": eval_score(modname, *VAL),
            "oos_750_1000_verify": eval_score(modname, *OOS_VERIFY),
        }
        print(label, book[label], flush=True)

    out = {
        "protocol": {
            "fit": FIT,
            "val": VAL,
            "oos_verify_only": OOS_VERIFY,
            "note": "Pair-pair sign conflicts only; label=sign(next day return)",
        },
        "n_conflicts": {"fit": len(fit_rows), "val": len(val_rows)},
        "rules": rule_table,
        "tree": tree_scores,
        "bangbang_agreement_with": {
            "max_priority": agree["priority"] / max(agree["n"], 1),
            "max_kappa": agree["kappa"] / max(agree["n"], 1),
            "n": agree["n"],
        },
        "book_bakeoff": book,
        "recommendation": None,
    }

    # Pick best val rule among simple rules (exclude oracle)
    best_rule, best_val = None, -1.0
    for rule, block in rule_table.items():
        if rule == "oracle":
            continue
        acc = block["val"]["acc"]
        if acc is not None and acc > best_val:
            best_val, best_rule = acc, rule
    tree_val = (tree_scores or {}).get("val", {}).get("acc")
    out["recommendation"] = {
        "best_simple_rule_on_val": best_rule,
        "best_simple_val_acc": best_val,
        "tree_val_acc": tree_val,
        "prefer_tree": bool(tree_val is not None and tree_val > best_val + 0.01),
        "book_priority_beats_equal_on_train": (
            book["priority_kappa_z"]["train_31_750"].get("score", 0)
            > book["equal_boost_current"]["train_31_750"].get("score", 0)
        ),
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)

    # cleanup temp modules
    for fname in ("_tmp_friend_equal.py", "_tmp_friend_priority.py"):
        try:
            (ROOT / fname).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
