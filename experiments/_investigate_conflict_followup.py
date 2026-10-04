"""Follow-up: conflict labels from isolated pair-leg PnL, not asset returns.

Also bake-off a 'flat when both sides strong' rule vs friend equal-boost.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "auto_tuning_results" / "_conflict_priority_followup.json"
FIT, VAL = (31, 500), (501, 750)


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


def collect(mod, prices, start, end):
    """Conflicts labeled by sum of isolated next-day pair-leg dollar PnL."""
    mod.reset()
    rows = []
    for t in range(start, end - 1):
        prc = prices[:, : t + 1]
        sleeves = mod.sleeve_breakdown(prc)
        if sleeves is None:
            continue
        cands = mod._STATE["atlas"].candidates(prc)
        if not cands:
            continue
        px0 = prices[:, t]
        px1 = prices[:, t + 1]
        dpx = px1 - px0

        by_asset = {}
        for priority, agent, pos_y, pos_x in cands:
            kappa = float(agent.current_kappa)
            abs_z = abs(float(agent.current_z))
            prot = mod._pair_name(agent.spec) in mod.PROTECTED_PAIR_NAMES
            # isolated next-day PnL of this pair's desired dollars
            iso = pos_y * dpx[int(agent.spec.y)] + pos_x * dpx[int(agent.spec.x)]
            for idx, pos in ((int(agent.spec.y), pos_y), (int(agent.spec.x), pos_x)):
                if abs(pos) <= 1e-9 or idx == 0:
                    continue
                by_asset.setdefault(idx, []).append(
                    {
                        "sign": int(np.sign(pos)),
                        "kappa": kappa,
                        "abs_z": abs_z,
                        "priority": float(priority),
                        "protected": prot,
                        "iso_pnl": float(iso),
                        "pair": mod._pair_name(agent.spec),
                    }
                )

        direction = np.asarray(sleeves["direction"], dtype=float)
        for asset, legs in by_asset.items():
            signs = {L["sign"] for L in legs}
            if not (+1 in signs and -1 in signs):
                continue
            long_legs = [L for L in legs if L["sign"] > 0]
            short_legs = [L for L in legs if L["sign"] < 0]
            long_iso = sum(L["iso_pnl"] for L in long_legs)
            short_iso = sum(L["iso_pnl"] for L in short_legs)
            # label: which side's pairs earned more in isolation that day
            if abs(long_iso - short_iso) < 1e-6:
                label = 0
            else:
                label = 1 if long_iso > short_iso else -1

            def agg(g):
                return {
                    "n": len(g),
                    "max_kappa": max(L["kappa"] for L in g),
                    "sum_kappa": sum(L["kappa"] for L in g),
                    "max_priority": max(L["priority"] for L in g),
                    "sum_priority": sum(L["priority"] for L in g),
                    "max_abs_z": max(L["abs_z"] for L in g),
                    "any_protected": any(L["protected"] for L in g),
                    "iso_pnl": sum(L["iso_pnl"] for L in g),
                }

            rows.append(
                {
                    "label": label,
                    "chosen": int(np.sign(direction[asset])),
                    "long": agg(long_legs),
                    "short": agg(short_legs),
                    "long_iso": long_iso,
                    "short_iso": short_iso,
                }
            )
    return rows


def pick(row, rule):
    L, S = row["long"], row["short"]
    if rule == "bangbang":
        return row["chosen"]
    if rule == "max_kappa":
        return 1 if L["max_kappa"] >= S["max_kappa"] else -1
    if rule == "sum_priority":
        return 1 if L["sum_priority"] >= S["sum_priority"] else -1
    if rule == "max_priority":
        return 1 if L["max_priority"] >= S["max_priority"] else -1
    if rule == "protected":
        if L["any_protected"] ^ S["any_protected"]:
            return 1 if L["any_protected"] else -1
        return 1 if L["n"] >= S["n"] else -1
    if rule == "majority":
        return 1 if L["n"] >= S["n"] else -1
    if rule == "max_abs_z":
        return 1 if L["max_abs_z"] >= S["max_abs_z"] else -1
    if rule == "iso_oracle":
        return row["label"]
    return 0


def score(rows, rule):
    hit = n = edge = 0
    for r in rows:
        if r["label"] == 0:
            continue
        pred = pick(r, rule)
        n += 1
        hit += int(pred == r["label"])
        # edge = iso pnl of chosen side
        if pred > 0:
            edge += r["long_iso"]
        elif pred < 0:
            edge += r["short_iso"]
    return {"acc": hit / n if n else None, "n": n, "captured_iso_pnl": edge}


def eval_mod(path_src: str, mode_line_old: str, mode_line_new: str, tag: str):
    """Materialize a temp module with PAIR_CONFLICT_MODE / consensus tweak."""
    src = Path(path_src).read_text(encoding="utf-8")
    if mode_line_old not in src:
        raise RuntimeError(f"mode line missing for {tag}")
    text = src.replace(mode_line_old, mode_line_new, 1)
    fname = f"_tmp_conflict_{tag}.py"
    Path(fname).write_text(text, encoding="utf-8")
    env = clear_env()
    out = {}
    for name, a, b in (("train", 31, 750), ("val", 501, 750), ("oos", 750, 1000)):
        p = subprocess.run(
            [sys.executable, "eval.py", "--module", fname[:-3], "--start", str(a), "--end", str(b)],
            capture_output=True, text=True, env=env, cwd=str(ROOT),
        )
        m = re.search(r"Score:\s*([\d.]+)", p.stdout)
        out[name] = float(m.group(1)) if m else None
    Path(fname).unlink(missing_ok=True)
    return out


def main():
    prices = load_prices()
    print("collect fit...", flush=True)
    fit = collect(load_friend(), prices, *FIT)
    print(f"  n={len(fit)}", flush=True)
    print("collect val...", flush=True)
    val = collect(load_friend(), prices, *VAL)
    print(f"  n={len(val)}", flush=True)

    rules = [
        "bangbang", "majority", "protected", "max_kappa", "max_priority",
        "sum_priority", "max_abs_z", "iso_oracle",
    ]
    table = {}
    for r in rules:
        table[r] = {"fit": score(fit, r), "val": score(val, r)}
        print(
            f"{r:14} fit_acc={table[r]['fit']['acc']:.3f} "
            f"val_acc={table[r]['val']['acc']:.3f} "
            f"val_iso={table[r]['val']['captured_iso_pnl']:.0f}",
            flush=True,
        )

    # How often does bangbang pick the iso-better side?
    # Strength gap: |sum_priority long - short|
    gaps = []
    for r in fit + val:
        if r["label"] == 0:
            continue
        gaps.append(abs(r["long"]["sum_priority"] - r["short"]["sum_priority"]))
    gaps = np.asarray(gaps)
    print(
        f"priority gap p50={np.median(gaps):.3f} p90={np.percentile(gaps,90):.3f}",
        flush=True,
    )

    # Bake-off: enable built-in priority mode (already tested) — skip.
    # Instead patch consensus to FLAT when both sides have protected or close vote.
    # We'll inject CONSENSUS_MARGIN-like flatten into _pair_conflict_consensus via a
    # post-process in a copied module — simpler: wrap sleeve_breakdown.

    # Create flat-on-weak-consensus by modifying return of consensus:
    src = (ROOT / "friend_book_latest.py").read_text(encoding="utf-8")
    # After consensus computed in sleeve_breakdown, flatten weak |vote|<margin
    needle = "    pair_vote, pair_active, pair_contributors = (\n        _pair_conflict_consensus(candidates)\n    )\n"
    inject = (
        "    pair_vote, pair_active, pair_contributors = (\n"
        "        _pair_conflict_consensus(candidates)\n"
        "    )\n"
        "    # Flatten weak pair consensus (both sides fighting → sit out)\n"
        "    _m = float(__import__('os').environ.get('FRIEND_CONSENSUS_MARGIN', '0.0'))\n"
        "    if _m > 0.0:\n"
        "        weak = pair_active & (np.abs(pair_vote) < _m)\n"
        "        pair_vote = pair_vote.copy()\n"
        "        pair_vote[weak] = 0.0\n"
    )
    if needle not in src:
        raise SystemExit("needle missing")
    flat_src = src.replace(needle, inject, 1)
    Path("_tmp_conflict_flatmargin.py").write_text(flat_src, encoding="utf-8")

    book = {"baseline": {}, "flat_m030": {}, "flat_m050": {}, "priority": {}}
    env = clear_env()
    for tag, mod, extra in (
        ("baseline", "friend_book_latest", {}),
        ("flat_m030", "_tmp_conflict_flatmargin", {"FRIEND_CONSENSUS_MARGIN": "0.30"}),
        ("flat_m050", "_tmp_conflict_flatmargin", {"FRIEND_CONSENSUS_MARGIN": "0.50"}),
    ):
        e = clear_env()
        e.update(extra)
        for name, a, b in (("train", 31, 750), ("val", 501, 750), ("oos", 750, 1000)):
            p = subprocess.run(
                [sys.executable, "eval.py", "--module", mod, "--start", str(a), "--end", str(b)],
                capture_output=True, text=True, env=e, cwd=str(ROOT),
            )
            m = re.search(r"Score:\s*([\d.]+)", p.stdout)
            book[tag][name] = float(m.group(1)) if m else None
        print(tag, book[tag], flush=True)

    # priority mode module
    pri = src.replace(
        'PAIR_CONFLICT_MODE = "we love sussy q"',
        'PAIR_CONFLICT_MODE = "priority"',
        1,
    )
    Path("_tmp_conflict_priority.py").write_text(pri, encoding="utf-8")
    e = clear_env()
    for name, a, b in (("train", 31, 750), ("val", 501, 750), ("oos", 750, 1000)):
        p = subprocess.run(
            [sys.executable, "eval.py", "--module", "_tmp_conflict_priority",
             "--start", str(a), "--end", str(b)],
            capture_output=True, text=True, env=e, cwd=str(ROOT),
        )
        m = re.search(r"Score:\s*([\d.]+)", p.stdout)
        book["priority"][name] = float(m.group(1)) if m else None
    print("priority", book["priority"], flush=True)

    out = {
        "label": "side with higher isolated pair-leg next-day PnL",
        "n": {"fit": len(fit), "val": len(val)},
        "rules": table,
        "book": book,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    for f in ("_tmp_conflict_flatmargin.py", "_tmp_conflict_priority.py"):
        Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
