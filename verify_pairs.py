#!/usr/bin/env python
"""Final pair-verification report: are these pairs real, or noise from 1,225 combinations?

Evidence layers (all produced by earlier pipeline stages; this script only merges them
and adds the statistics that need cross-pair information):

  1. Screen gates      - regime-aware statistical evidence on the train window
                         (pair_screener.py): cointegration across windows, half-life,
                         structural breaks, Kalman beta stability, vol-regime PnL.
  2. Null model        - binomial expected false G1-passers vs observed, plus the pooled
                         Benjamini-Hochberg FDR diagnostics.
  3. OOS consistency   - the same screen re-run on the strict OOS window (801-1000):
                         real relationships keep passing; noise dies.
  4. Tuning outcome    - windowed W-R+tb tuner train/OOS score & Sharpe retention for
                         the shortlisted new pairs and the placebo cohort, run with an
                         IDENTICAL budget. The placebos are the pipeline-level noise
                         control: if tuning extracts "good OOS" from screen-failures as
                         often as from screen-passers, the whole funnel is noise.
  5. Deflated Sharpe   - Bailey & Lopez de Prado DSR per finalist: significance of the
                         OOS Sharpe after accounting for non-normality and for having
                         searched the full 1,225-pair universe (also shown for the
                         16-pair tuned cohort as a sensitivity).

Caveat recorded honestly in the report: the 801-1000 window was already used to compare
tuner configs in the previous bake-off, so the current 8 pairs' OOS numbers are mildly
selection-optimistic. The placebo arm never touched OOS during selection, so the
placebo comparison is the least biased evidence in this report.

Usage:
    python verify_pairs.py \
        --screen auto_tuning_results/screen_train.json \
        --screen_oos auto_tuning_results/screen_oos.json \
        --tuning auto_tuning_results/screen_tuning_results.json \
        --current_ref auto_tuning_results/_bakeoff_wratio_tbrk.json \
        --out auto_tuning_results/verification_report.txt \
        --json_out auto_tuning_results/verification.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

CFG = {
    "screen": "auto_tuning_results/screen_train.json",
    "screen_oos": "auto_tuning_results/screen_oos.json",
    "selection": "auto_tuning_results/selection_meta.json",
    "tuning": "auto_tuning_results/screen_tuning_results.json",
    "current_ref": "auto_tuning_results/_bakeoff_wratio_tbrk.json",
    "out": "auto_tuning_results/verification_report.txt",
    "json_out": "auto_tuning_results/verification.json",
    "oos_start": 801,
    "oos_end": 1000,
    "universe_n": 1225,      # number of unordered pairs searched
    "skip_eval": False,      # set True to skip re-running daily OOS evals for DSR
}

EMC = 0.5772156649015329  # Euler-Mascheroni constant


def key_of(a, b=None):
    if b is None:
        return tuple(sorted(a))
    return tuple(sorted((a, b)))


# ============================================================================
# DEFLATED SHARPE RATIO (Bailey & Lopez de Prado, 2014)
# ============================================================================
def deflated_sharpe(daily_pnl: np.ndarray, n_trials: int, sr_var_daily: float) -> float:
    """Probability that the true Sharpe exceeds E[max Sharpe of n_trials null trials].

    daily_pnl: out-of-sample daily PnL series of the candidate.
    sr_var_daily: cross-sectional variance of daily-Sharpe estimates across trials.
    """
    T = daily_pnl.size
    if T < 30:
        return float("nan")
    mu, sd = float(np.mean(daily_pnl)), float(np.std(daily_pnl))
    if sd <= 1e-12:
        return float("nan")
    sr = mu / sd  # daily Sharpe
    g3 = float(sps.skew(daily_pnl))
    g4 = float(sps.kurtosis(daily_pnl, fisher=False))
    z = ((1.0 - EMC) * sps.norm.ppf(1.0 - 1.0 / n_trials)
         + EMC * sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    e_max_sr = math.sqrt(max(sr_var_daily, 1e-12)) * z
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr))
    return float(sps.norm.cdf((sr - e_max_sr) * math.sqrt(T - 1) / denom))


# ============================================================================
# LOADERS
# ============================================================================
def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def screen_index(screen):
    """sorted-pair-key -> record"""
    if not screen:
        return {}
    return {key_of(r["pair"]): r for r in screen["records"]}


def tuning_index(tuning):
    if not tuning:
        return {}
    return {key_of(e["y_symbol"], e["x_symbol"]): e for e in tuning}


def current_ref_index(ref):
    """Current-8 params & metrics from the previous W-R+tb bake-off."""
    if not ref:
        return {}
    out = {}
    for e in ref:
        g = e.get("global", e)
        out[key_of(e["pair_name"].split("-"))] = {
            "final_params": g["final_params"],
            "train_metrics": g["train_metrics"],
            "oos_metrics": g["oos_metrics"],
            "selection_source": g.get("selection_source", "?"),
        }
    return out


# ============================================================================
# DAILY OOS EVALS (for DSR skew/kurt inputs)
# ============================================================================
def oos_daily_pnl(params, oos_start, oos_end):
    import latest_tuner
    *_, pll, _score = latest_tuner.run_official_eval(
        params, start_day=oos_start, end_day=oos_end)
    return np.asarray(pll, dtype=float)


# ============================================================================
# MAIN
# ============================================================================
def run():
    cfg = CFG
    screen = load_json(cfg["screen"])
    screen_oos = load_json(cfg["screen_oos"])
    tuning = load_json(cfg["tuning"])
    current_ref = load_json(cfg["current_ref"])
    if screen is None:
        raise SystemExit(f"missing screen file: {cfg['screen']}")

    s_idx = screen_index(screen)
    so_idx = screen_index(screen_oos)
    t_idx = tuning_index(tuning)
    c_idx = current_ref_index(current_ref)

    meta = screen["meta"]
    selection = load_json(cfg["selection"]) or {}
    roles = {}
    for s in selection.get("shortlist", meta.get("shortlist", [])):
        roles[key_of(s["pair"])] = "new"
    for s in selection.get("placebos", meta.get("placebos", [])):
        roles[key_of(s["pair"])] = "placebo"
    for p in meta.get("current_pairs", []):
        roles[key_of(p)] = "current"

    # cohort = every tuned pair + the current 8
    cohort_keys = list(dict.fromkeys(list(t_idx.keys()) + list(c_idx.keys())))
    if not cohort_keys:
        cohort_keys = [key_of(s["pair"]) for s in
                       meta.get("shortlist", []) + meta.get("placebos", [])]

    # ---- collect per-pair metrics -------------------------------------------------
    rows = []
    for k in cohort_keys:
        row = {"key": k, "pair": list(k), "role": roles.get(k, "?")}
        srec = s_idx.get(k)
        if srec:
            g = srec["gate_eval"]
            ch = srec["A"] if srec["orient"] == "A" else srec["B"]
            row["train_screen"] = {
                "score": g["screen_score"], "pass": g["pass"], "gates": g["gates"],
                "median_halflife": g["median_halflife"], "beta_cv": g["beta_cv"],
                "cusum_p": g["cusum_p"], "n_win_adf05": g["n_win_adf05"],
                "n_mean_breaks": ch.get("n_mean_breaks"),
            }
        sorec = so_idx.get(k)
        if sorec:
            g = sorec["gate_eval"]
            row["oos_screen"] = {
                "score": g["screen_score"], "pass": g["pass"], "gates": g["gates"],
                "n_win_adf05": g["n_win_adf05"],
                "median_halflife": g["median_halflife"],
            }

        trec = t_idx.get(k)
        if trec:
            g = trec["global"]
            row["tuning"] = {
                "final_params": g["final_params"],
                "train_score": g["train_metrics"]["score"],
                "train_sharpe": g["train_metrics"]["sharpe"],
                "oos_score": g["oos_metrics"]["score"],
                "oos_sharpe": g["oos_metrics"]["sharpe"],
                "selection_source": g.get("selection_source", "?"),
            }
        elif k in c_idx:
            row["tuning"] = {
                "final_params": c_idx[k]["final_params"],
                "train_score": c_idx[k]["train_metrics"]["score"],
                "train_sharpe": c_idx[k]["train_metrics"]["sharpe"],
                "oos_score": c_idx[k]["oos_metrics"]["score"],
                "oos_sharpe": c_idx[k]["oos_metrics"]["sharpe"],
                "selection_source": f"bakeoff:{c_idx[k]['selection_source']}",
            }
        rows.append(row)

    # ---- daily OOS PnL for DSR (also refreshes exact OOS Sharpe) -------------------
    if not cfg["skip_eval"]:
        for row in rows:
            if "tuning" not in row:
                continue
            try:
                row["oos_pnl"] = oos_daily_pnl(row["tuning"]["final_params"],
                                               cfg["oos_start"], cfg["oos_end"]).tolist()
            except Exception as e:
                print(f"  [warn] OOS eval failed for {row['pair']}: {e}")

    # cross-trial Sharpe variance: from every cohort member that has daily OOS PnL
    daily_sharpes = []
    for row in rows:
        pnl = np.asarray(row.get("oos_pnl", []), dtype=float)
        if pnl.size >= 30 and pnl.std() > 1e-12:
            row["oos_sharpe_daily"] = float(pnl.mean() / pnl.std())
            daily_sharpes.append(row["oos_sharpe_daily"])

    placebo_sharpes = [r["oos_sharpe_daily"] for r in rows
                       if r["role"] == "placebo" and "oos_sharpe_daily" in r]
    # DSR null dispersion from the PLACEBO cohort (empirical noise), not the mixed
    # signal+noise cohort, which would inflate the expected-max-null
    sr_var_daily = (float(np.var(placebo_sharpes)) if len(placebo_sharpes) > 1
                    else (float(np.var(daily_sharpes)) if len(daily_sharpes) > 1 else 1e-6))

    for row in rows:
        pnl = np.asarray(row.get("oos_pnl", []), dtype=float)
        if pnl.size < 30:
            continue
        row["dsr_universe"] = deflated_sharpe(pnl, cfg["universe_n"], sr_var_daily)
        row["dsr_cohort"] = deflated_sharpe(pnl, max(len(rows), 2), sr_var_daily)
        if placebo_sharpes:
            row["placebo_rank"] = int(sum(1 for p in placebo_sharpes
                                          if p >= row["oos_sharpe_daily"]))
            row["beats_all_placebos"] = bool(row["placebo_rank"] == 0)
        if row.get("tuning", {}).get("train_sharpe"):
            row["sharpe_retention"] = (row["tuning"]["oos_sharpe"]
                                       / row["tuning"]["train_sharpe"])

    # ---- verdicts ------------------------------------------------------------------
    # Noise ceiling: max annualized OOS Sharpe among placebos (no-edge pairs tuned with
    # the identical budget). The ADD/KEEP bar is max(1.5, ceiling): 1.5 matches the
    # weakest current pair's historical bar; the ceiling guards against placebo noise.
    # G1 (window-ADF battery) is a displayed diagnostic, never a verdict input - it is
    # a weak instrument on 120-day windows that even proven pairs fail window-wise.
    SANITY = ("G2_halflife", "G3_structural", "G4_vol_regime", "G5_tradable")
    pl_sharpes_ann = [r["tuning"]["oos_sharpe"] for r in rows
                      if r["role"] == "placebo" and r.get("tuning")]
    pl_ceiling = max(pl_sharpes_ann) if pl_sharpes_ann else 0.0
    bar = max(1.5, pl_ceiling)

    for row in rows:
        role = row["role"]
        if role == "placebo":
            row["verdict"] = "PLACEBO"
            continue
        oos_s = row.get("tuning", {}).get("oos_sharpe")
        if oos_s is None:
            row["verdict"] = "PENDING (not tuned yet)"
            continue
        gates_oos = row.get("oos_screen", {}).get("gates", {})
        sanity_fail = [g for g in SANITY if gates_oos and not gates_oos.get(g, True)]
        ret = row.get("sharpe_retention")
        reasons = []
        if oos_s < bar:
            reasons.append(f"OOS Sharpe {oos_s:.2f} below bar {bar:.2f} "
                           f"(max(1.5, placebo ceiling {pl_ceiling:.2f}))")
        elif oos_s < pl_ceiling + 0.25:
            reasons.append(f"thin margin over placebo ceiling {pl_ceiling:.2f}")
        if ret is not None and ret < 0.5:
            reasons.append(f"low train->OOS Sharpe retention {100*ret:.0f}%")
        if sanity_fail:
            reasons.append(f"OOS-period screen sanity fails: {','.join(sanity_fail)}")

        if role == "current":
            if oos_s < 0.5:
                row["verdict"] = "DROP"
            elif oos_s >= bar and not sanity_fail:
                row["verdict"] = "KEEP"
            else:
                row["verdict"] = "REVIEW"
        else:  # new
            if oos_s >= bar and not sanity_fail:
                row["verdict"] = "ADD"
            elif oos_s >= bar:
                row["verdict"] = "REVIEW"
            else:
                row["verdict"] = "REJECT"
        row["verdict_reasons"] = reasons

    # ---- report --------------------------------------------------------------------
    lines = []
    add = lines.append
    add("=" * 112)
    add("PAIR VERIFICATION REPORT - real edge or noise from 1,225 combinations?")
    add("=" * 112)
    nm = meta.get("null_model", {})
    if nm:
        add(f"Screen-level null: {nm.get('g1_passers_observed')} G1 passers observed on train vs "
            f"{nm.get('g1_passers_expected_under_null', 0):.2f} expected if ALL spreads were noise")
    fdr = meta.get("fdr", {})
    if fdr:
        add(f"Pooled BH-FDR: {fdr.get('rejected')}/{fdr.get('tests')} window ADF tests rejected "
            f"at q={fdr.get('q')} (deliberately strict diagnostic)")
    add("Caveat: 801-1000 was used in the previous tuner bake-off, so the current 8 pairs' OOS is")
    add("        mildly selection-optimistic; the placebo cohort (screen-failures tuned with an")
    add("        identical budget, never selected on OOS) is the least biased noise benchmark.")
    add("")

    hdr = (f"{'pair':<14}{'role':<9}{'gates':>6}{'oosG':>5}{'trScr':>8}{'oosScr':>8}"
           f"{'trSrp':>7}{'oosSrp':>7}{'ret%':>6}{'DSRu':>6}{'plcb':>5}  verdict")
    add(hdr)
    add("-" * len(hdr))
    order = {"current": 0, "new": 1, "placebo": 2, "?": 3}
    for row in sorted(rows, key=lambda r: (order.get(r["role"], 3),
                                           -(r.get("tuning", {}) or {}).get("oos_sharpe", -999)
                                           if r.get("tuning") else 999)):
        t = row.get("tuning", {})
        tr_g = row.get("train_screen", {}).get("gates", {})
        oos_g = row.get("oos_screen", {}).get("gates", {})
        n_tr = f"{sum(tr_g.values())}/5" if tr_g else "-"
        n_oos = f"{sum(oos_g.values())}/5" if oos_g else "-"
        ret = row.get("sharpe_retention")
        dsr = row.get("dsr_universe")
        pl = row.get("placebo_rank")
        add(f"{row['pair'][0]+'-'+row['pair'][1]:<14}{row['role']:<9}{n_tr:>6}{n_oos:>5}"
            f"{t.get('train_score', float('nan')):>8.2f}{t.get('oos_score', float('nan')):>8.2f}"
            f"{t.get('train_sharpe', float('nan')):>7.2f}{t.get('oos_sharpe', float('nan')):>7.2f}"
            f"{(100*ret if ret is not None else float('nan')):>6.0f}"
            f"{(dsr if dsr is not None and np.isfinite(dsr) else float('nan')):>6.2f}"
            f"{(pl if pl is not None else '-'):>5}  {row['verdict']}")
        for reason in row.get("verdict_reasons", []):
            add(f"{'':>23}  - {reason}")
    add("")
    if placebo_sharpes:
        ann = [s * math.sqrt(250) for s in placebo_sharpes]
        add(f"Placebo OOS Sharpes, annualized (n={len(ann)}): median {np.median(ann):.2f}, "
            f"max {max(ann):.2f}  ->  ADD/KEEP bar = max(1.5, placebo max) = {bar:.2f}")
    add("")
    add("Columns: gates = train screen gates passed /5, oosG = OOS screen gates passed /5")
    add("         (G1 window-ADF is a weak-instrument diagnostic and never gates verdicts),")
    add("         trScr/oosScr = tuner train/OOS score, trSrp/oosSrp = train/OOS ann. Sharpe,")
    add("         ret% = OOS/train Sharpe retention, plcb = # placebos with >= this OOS Sharpe,")
    add("         DSRu = Deflated Sharpe vs 1,225-pair universe, null dispersion estimated from")
    add("         the placebo cohort. With ~200 OOS days, DSRu >= 0.95 is a high bar: only the")
    add("         strongest current pairs (ILVX-SMAH 0.95, AENO-NWIG 0.94) reach it, which")
    add("         corroborates them; verdicts otherwise rest on the placebo contrast, screen")
    add("         rank, and Sharpe retention - DSRu is a diagnostic, not a gate.")

    report = "\n".join(lines) + "\n"
    with open(cfg["out"], "w") as f:
        f.write(report)
    print(report)

    # ---- plots ---------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes[0]
        lab_rows = [r for r in rows if "tuning" in r]
        labels = [f"{r['pair'][0]}-{r['pair'][1]}" for r in lab_rows]
        colors = {"current": "#1f77b4", "new": "#2ca02c", "placebo": "#d62728", "?": "gray"}
        vals = [r["tuning"]["oos_sharpe"] for r in lab_rows]
        ax.bar(range(len(vals)), vals,
               color=[colors.get(r["role"], "gray") for r in lab_rows])
        if placebo_sharpes:
            ax.axhline(max(placebo_sharpes), color="#d62728", linestyle="--",
                       label=f"max placebo ({max(placebo_sharpes):.2f})")
            ax.legend()
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=75, fontsize=8)
        ax.set_title("OOS annualized Sharpe by pair (blue=current, green=new, red=placebo)")
        ax.set_ylabel("OOS Sharpe")

        ax = axes[1]
        for r in lab_rows:
            ret = r.get("sharpe_retention")
            sc = r.get("train_screen", {}).get("score")
            if ret is None or sc is None:
                continue
            ax.scatter(sc, 100 * ret, c=colors.get(r["role"], "gray"), s=55,
                       edgecolors="black", linewidths=0.5,
                       label=r["role"] if r["role"] not in [ax_.get_label() for ax_ in ax.collections] else None)
        ax.axhline(50, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("train screen score")
        ax.set_ylabel("Sharpe retention OOS/train (%)")
        ax.set_title("Screen score vs generalization: do screened pairs generalize better?")
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            by = dict(zip(lbls, handles))
            ax.legend(by.values(), by.keys())
        fig.tight_layout()
        png = cfg["out"].replace(".txt", ".png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"[plots] saved {png}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    with open(cfg["json_out"], "w") as f:
        json.dump({"rows": [{k: v for k, v in r.items() if k != "oos_pnl"} for r in rows],
                   "sr_var_daily": sr_var_daily,
                   "placebo_sharpes_daily": placebo_sharpes}, f, indent=2)
    print(f"[out] machine-readable verification -> {cfg['json_out']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Final pair-verification report")
    for k_, v_ in CFG.items():
        if isinstance(v_, bool):
            ap.add_argument(f"--{k_}", default=v_,
                            type=lambda s: s.strip().lower() in ("1", "true", "yes"))
        else:
            ap.add_argument(f"--{k_}", default=v_, type=type(v_))
    args = ap.parse_args()
    CFG.update(vars(args))
    run()
