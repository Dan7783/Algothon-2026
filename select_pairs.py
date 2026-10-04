#!/usr/bin/env python
"""Select the tuner cohort from saved screen results (no re-screening needed).

Shortlist: top screen-score pairs passing the tradability sanity gates
(G2 half-life, G3 structural, G4 vol-regime, G5 tradable), excluding the current 8.
Placebos: seeded random sample of pairs with negative screen score (no edge arm).

Writes the tuner-ready pairs JSON and prints a selection summary with each pair's
evidence (gates, ADF windows, rank percentile, user-candidate tag).
"""

import argparse
import json
import random

CFG = {
    "screen": "auto_tuning_results/screen_train.json",
    "pairs_out": "auto_tuning_results/pairs_screened.json",
    "meta_out": "auto_tuning_results/selection_meta.json",
    "topn": 10,
    "n_placebo": 6,
    "placebo_seed": 7,
    "user_pairs_file": "user_candidate_pairs.json",
}

CURRENT_PAIRS = [
    ("RTTH", "NAYO"), ("AENO", "NWIG"), ("HETT", "ULXY"), ("HUXZ", "ACAC"),
    ("SMAH", "ILVX"), ("EORC", "NGTE"), ("CTGI", "EELT"), ("ALUT", "CCNS"),
]
SANITY = ("G2_halflife", "G3_structural", "G4_vol_regime", "G5_tradable")


def key_of(pair):
    return tuple(sorted(pair))


def chosen(r):
    return r["A"] if r["orient"] == "A" else r["B"]


def run():
    ap = argparse.ArgumentParser()
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", default=v, type=type(v) if not isinstance(v, bool) else str)
    args = ap.parse_args()
    cfg = {k: (int(v) if isinstance(CFG[k], int) else v) for k, v in vars(args).items()}

    with open(cfg["screen"]) as f:
        screen = json.load(f)
    records = screen["records"]
    with open(cfg["user_pairs_file"]) as f:
        user_pairs = [(d["y"], d["x"]) for d in json.load(f)]
    cur = {key_of(p) for p in CURRENT_PAIRS}
    usr = {key_of(p) for p in user_pairs}

    rank_of = {key_of(r["pair"]): i + 1 for i, r in enumerate(records)}

    eligible = [r for r in records if all(r["gate_eval"]["gates"][g] for g in SANITY)]
    new_eligible = [r for r in eligible if key_of(r["pair"]) not in cur]
    shortlist = new_eligible[: cfg["topn"]]

    no_edge = [r for r in records if r["gate_eval"]["screen_score"] < 0]
    rng = random.Random(cfg["placebo_seed"])
    placebos = rng.sample(no_edge, min(cfg["n_placebo"], len(no_edge)))

    def line(r, role):
        g = r["gate_eval"]
        k = key_of(r["pair"])
        gates = g["gates"]
        gstr = "".join("Y" if gates[x] else "." for x in
                       ("G1_coint_fdr", "G2_halflife", "G3_structural",
                        "G4_vol_regime", "G5_tradable"))
        tag = "user89" if k in usr else ""
        n_adf = g["n_win_adf05"]
        hl = g["median_halflife"]
        return (f"  {r['pair'][0]}-{r['pair'][1]:<7} rank {rank_of[k]:>4}/1225 "
                f"({100*rank_of[k]/1225:>4.1f}%) score {g['screen_score']:>7.2f} "
                f"gates {gstr} adfW {n_adf}/5 medHL {hl:>5.1f} {tag} [{role}]")

    print("=" * 100)
    print(f"SELECTED TUNER COHORT (top {cfg['topn']} new + {cfg['n_placebo']} placebos)")
    print("=" * 100)
    print(f"eligible pairs (pass G2+G3+G4+G5): {len(eligible)}/1225 "
          f"({len(new_eligible)} excluding current 8)")
    print("SHORTLIST (new):")
    for r in shortlist:
        print(line(r, "new"))
    print("PLACEBOS (negative screen score, seeded random):")
    for r in placebos:
        g = r["gate_eval"]
        failed = [k2 for k2, v in g["gates"].items() if not v]
        print(line(r, "placebo") + f"  fails: {','.join(failed)}")

    tuner_pairs = []
    for r in shortlist + placebos:
        y, x = (r["pair"] if r["orient"] == "A" else r["pair"][::-1])
        tuner_pairs.append({"y": y, "x": x})
    with open(cfg["pairs_out"], "w") as f:
        json.dump(tuner_pairs, f, indent=2)

    meta = {
        "shortlist": [{"pair": r["pair"], "orient": r["orient"],
                       "screen_score": r["gate_eval"]["screen_score"]} for r in shortlist],
        "placebos": [{"pair": r["pair"], "orient": r["orient"],
                      "screen_score": r["gate_eval"]["screen_score"],
                      "failed_gates": [k2 for k2, v in r["gate_eval"]["gates"].items() if not v]}
                     for r in placebos],
        "sanity_gates": list(SANITY),
        "n_eligible": len(eligible),
    }
    with open(cfg["meta_out"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n-> tuner pairs ({len(tuner_pairs)}) -> {cfg['pairs_out']}")
    print(f"-> selection metadata -> {cfg['meta_out']}")


if __name__ == "__main__":
    run()
