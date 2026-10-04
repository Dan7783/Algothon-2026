#!/usr/bin/env python
"""Density-based clustering (HDBSCAN) over Optuna trial history -> medoid of the best cluster.

Reads a study from the shared tuner SQLite DB, clusters all COMPLETE trials in
normalized parameter space, ranks clusters by mean objective value, and returns the
medoid (the actual trial minimizing intra-cluster distance) of the best-performing
cluster. The medoid is a robust "typical good" parameter set, unlike the single best
trial which may be a lucky outlier.

Normalization: log10-transform for log-scale params (per latest_tuner.SEARCH_SPACES),
then min-max scale to [0,1] over the search bounds, so all dimensions contribute
comparably to Euclidean distances (max_cov_inflate spans 1e2-1e8 raw).

The DB is opened read-only so this never interferes with a running tuner.

Standalone CLI:
    python cluster_medoid.py --study RTTH-NAYO
    python cluster_medoid.py --all
    python cluster_medoid.py --study RTTH-NAYO --trajectory
    python cluster_medoid.py --study RTTH-NAYO --augment   # verify sparse peaks via perturbation
Programmatic:
    from cluster_medoid import find_best_cluster_medoid
"""

import argparse
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from latest_tuner import (
    RESULTS_DIR, SEARCH_SPACES, OPTUNA_DB_FILE,
    evaluate_purged_walk_forward, round_sig, run_official_eval,
)

PARAM_NAMES = list(SEARCH_SPACES.keys())
LOG_PARAMS = {k for k, s in SEARCH_SPACES.items() if s.get("log")}
INT_PARAMS = {"ou_window"}


def list_studies(db_path):
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    names = [r[0] for r in con.execute("SELECT study_name FROM studies ORDER BY study_name")]
    con.close()
    return names


def load_trials(db_path, study_name):
    """Return DataFrame of COMPLETE trials: trial number, objective value, one column per param."""
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    df = pd.read_sql_query(
        """
        SELECT t.number AS number, v.value AS value,
               p.param_name AS param_name, p.param_value AS param_value
        FROM trials t
        JOIN studies s ON s.study_id = t.study_id
        JOIN trial_values v ON v.trial_id = t.trial_id
        JOIN trial_params p ON p.trial_id = t.trial_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE'
        """,
        con, params=(study_name,),
    )
    con.close()
    if df.empty:
        raise ValueError(f"No COMPLETE trials found for study '{study_name}' in {db_path}")
    wide = df.pivot_table(index="number", values="param_value", columns="param_name", aggfunc="first")
    vals = df[["number", "value"]].drop_duplicates("number").set_index("number")
    out = wide.join(vals).reset_index()
    missing = [p for p in PARAM_NAMES if p not in out.columns]
    if missing:
        raise ValueError(f"Study '{study_name}' trials are missing params: {missing}")
    return out


def normalize_params(df):
    """Param matrix -> [0,1]^d array (log10 for log-scale dims, min-max over search bounds)."""
    cols = []
    for name, spec in SEARCH_SPACES.items():
        lo, hi = float(spec["low"]), float(spec["high"])
        x = df[name].astype(float).values
        if name in LOG_PARAMS:
            x = np.log10(np.clip(x, 1e-300, None))
            lo, hi = np.log10(lo), np.log10(hi)
        cols.append(np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0))
    return np.column_stack(cols)


def cluster_trials(X, min_cluster_size=None):
    """HDBSCAN labels with graceful fallbacks. Returns (labels, method_description)."""
    n = len(X)
    if n < 10:
        return np.zeros(n, dtype=int), "single-cluster-fallback(too-few-trials)"
    base = min_cluster_size or max(8, int(round(0.02 * n)))
    for mcs in dict.fromkeys([base, max(5, base // 2), 5]):
        if mcs >= n:
            continue
        labels = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(X)
        if len(set(labels) - {-1}) >= 1:
            return labels, f"hdbscan(min_cluster_size={mcs})"
    k = min(10, n - 1)
    dists = np.sort(NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1])
    eps = max(float(dists[int(0.9 * (n - 1))]), 1e-3)
    labels = DBSCAN(eps=eps, min_samples=max(3, k // 2)).fit_predict(X)
    if len(set(labels) - {-1}) >= 1:
        return labels, f"dbscan(eps={eps:.4g})"
    return np.zeros(n, dtype=int), "single-cluster-fallback(no-density-structure)"


def denormalize_params(x_vec):
    """[0,1]^d normalized vector -> raw param dict (exact inverse of normalize_params)."""
    params = {}
    for j, (name, spec) in enumerate(SEARCH_SPACES.items()):
        lo, hi = float(spec["low"]), float(spec["high"])
        v = float(np.clip(x_vec[j], 0.0, 1.0))
        if name in LOG_PARAMS:
            v = 10.0 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo)))
        else:
            v = lo + v * (hi - lo)
        params[name] = int(round(v)) if name in INT_PARAMS else v
    return params


def select_best_cluster(labels, values, sources=None):
    """Rank clusters (noise excluded) by mean objective value. Returns (best_label, table)."""
    rows = []
    for lab in sorted(set(labels)):
        mask = labels == lab
        row = {
            "cluster_id": int(lab),
            "size": int(mask.sum()),
            "mean_value": float(np.mean(values[mask])),
            "max_value": float(np.max(values[mask])),
            "is_noise": bool(lab == -1),
        }
        if sources is not None:
            row["n_augmented"] = int(np.sum(sources[mask] == "augmented"))
        rows.append(row)
    candidates = [r for r in rows if not r["is_noise"]]
    if not candidates:
        candidates = rows  # everything was noise: treat all trials as one cluster
    best = max(candidates, key=lambda r: r["mean_value"])
    rows.sort(key=lambda r: r["mean_value"], reverse=True)
    return best["cluster_id"], rows


def compute_medoid_index(X, mask):
    """Index (into the full array) of the point minimizing summed intra-cluster distance."""
    idx = np.flatnonzero(mask)
    if len(idx) == 1:
        return int(idx[0])
    d = pairwise_distances(X[mask])
    return int(idx[np.argmin(d.sum(axis=1))])


def trajectory_diagnostics(db_path, study_name):
    """Distinguish 'lucky spike' from 'under-sampled convergence target' for the best trial.

    HDBSCAN noise only means the optimizer rarely sampled near the point; it says nothing
    about parameter stability. If normalized distances to the best trial shrink over the
    trial sequence and late trials keep improving, the sampler was still converging toward
    the peak when the budget ran out, so the region is under-sampled rather than unstable
    (stability itself is what the perturbation check measures).
    """
    df = load_trials(db_path, study_name)
    X = normalize_params(df)
    order = np.argsort(df["number"].values)
    numbers = df["number"].values[order]
    values = df["value"].values.astype(float)[order]
    Xo = X[order]

    best_i = int(np.argmax(values))
    d = np.linalg.norm(Xo - Xo[best_i], axis=1).astype(float)
    d[best_i] = np.nan

    n = len(values)
    k = min(5, n - 1)
    knn_mean = float(np.nanmean(np.sort(d[~np.isnan(d)])[:k])) if k > 0 else float("nan")
    half = n // 2
    med_first = float(np.nanmedian(d[:half])) if half else float("nan")
    med_second = float(np.nanmedian(d[half:])) if half else float("nan")
    q = max(1, n // 4)
    quarter_best = [float(np.max(values[i * q:(i + 1) * q])) for i in range(4) if i * q < n]

    converging = bool(med_second < med_first) if half else False
    verdict = ("UNDER-SAMPLED CONVERGENCE TARGET - sampler was still approaching the peak; "
               "budget, not instability, explains the sparse region"
               if converging else
               "NO CONVERGENCE TREND toward the best trial - treat an isolated peak with "
               "suspicion unless a perturbation check says otherwise")

    return {
        "study": study_name,
        "best_trial_number": int(numbers[best_i]),
        "best_trial_value": float(values[best_i]),
        "knn_mean_dist_to_best": knn_mean,
        "median_dist_first_half": med_first,
        "median_dist_second_half": med_second,
        "best_value_by_quarter": quarter_best,
        "converging_toward_best": converging,
        "verdict": verdict,
        "_numbers": numbers, "_values": values, "_dist": d, "_best_i": best_i,
    }


def plot_trajectory(diag, out_dir=RESULTS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numbers, values, d = diag["_numbers"], diag["_values"], diag["_dist"]
    best_i = diag["_best_i"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.scatter(numbers, values, s=18, c="steelblue", alpha=0.6, label="trial value")
    ax1.plot(numbers, np.maximum.accumulate(values), color="darkorange", linewidth=2,
             label="running best")
    ax1.scatter([numbers[best_i]], [values[best_i]], marker="*", c="red", s=350,
                edgecolors="black", zorder=10,
                label=f"best trial {diag['best_trial_number']} ({diag['best_trial_value']:.4g})")
    ax1.set_ylabel("Objective value")
    ax1.set_title(f"Sampler trajectory - {diag['study']}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    running_min = np.minimum.accumulate(np.where(np.isnan(d), np.inf, d))
    ax2.scatter(numbers, d, s=18, c="gray", alpha=0.6, label="distance to best trial")
    ax2.plot(numbers, running_min, color="crimson", linewidth=2,
             label="running min distance")
    ax2.axhline(diag["knn_mean_dist_to_best"], color="darkgreen", linestyle="--",
                linewidth=1.2, label=f"5-NN mean ({diag['knn_mean_dist_to_best']:.3g})")
    ax2.set_xlabel("Trial number")
    ax2.set_ylabel("Normalized-space distance")
    ax2.set_title(f"Convergence check: median dist 1st half {diag['median_dist_first_half']:.3f} "
                  f"-> 2nd half {diag['median_dist_second_half']:.3f} "
                  f"({'converging' if diag['converging_toward_best'] else 'no trend'})")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    safe = diag["study"].replace("/", "_")
    path = os.path.join(out_dir, f"trajectory_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_trajectory(diag):
    print(f"\n  TRAJECTORY DIAGNOSTIC - {diag['study']}")
    print(f"    best trial            : {diag['best_trial_number']} "
          f"(value {diag['best_trial_value']:.4f})")
    print(f"    5-NN mean distance    : {diag['knn_mean_dist_to_best']:.4f}")
    print(f"    median dist to best   : 1st half {diag['median_dist_first_half']:.4f} -> "
          f"2nd half {diag['median_dist_second_half']:.4f}")
    print(f"    best value by quarter : {[round(v, 3) for v in diag['best_value_by_quarter']]}")
    print(f"    verdict               : {diag['verdict']}")


def find_best_cluster_medoid(db_path, study_name, min_cluster_size=None, extra_trials=None):
    """Full pipeline: load -> normalize -> cluster -> rank by mean value -> medoid.

    extra_trials: optional DataFrame of additional evaluated points (same param columns
    plus number/value), e.g. perturbation-augmented evals. They are tagged
    source="augmented" and participate in clustering/ranking/medoid selection like real
    trials (they ARE real objective evaluations).

    Returns a JSON-serializable dict plus internal objects under "_" keys for plotting
    (strip those before writing to JSON).
    """
    df = load_trials(db_path, study_name)
    df["source"] = "trial"
    if extra_trials is not None and len(extra_trials) > 0:
        extra = extra_trials.copy()
        extra["source"] = "augmented"
        df = pd.concat([df, extra], ignore_index=True)

    X = normalize_params(df)
    values = df["value"].values.astype(float)
    sources = df["source"].values

    labels, method = cluster_trials(X, min_cluster_size)
    best_label, table = select_best_cluster(labels, values, sources)
    mask = labels == best_label
    if not mask.any():
        mask = np.ones(len(df), dtype=bool)

    med_i = compute_medoid_index(X, mask)
    row = df.iloc[med_i]
    params = {}
    for name in PARAM_NAMES:
        v = float(row[name])
        params[name] = int(round(v)) if name in INT_PARAMS else v

    # "global best" refers to the study's own trials, not augmented points
    trial_only_values = np.where(sources == "trial", values, -np.inf)
    global_best_i = int(np.argmax(trial_only_values))
    global_best_row = df.iloc[global_best_i]

    return {
        "study": study_name,
        "method": method,
        "n_trials": int((sources == "trial").sum()),
        "n_augmented": int((sources == "augmented").sum()),
        "n_noise": int((labels == -1).sum()),
        "global_best_trial_number": int(global_best_row["number"]),
        "global_best_value": float(global_best_row["value"]),
        "global_best_is_noise": bool(labels[global_best_i] == -1),
        "best_cluster_id": int(best_label),
        "best_cluster_size": int(mask.sum()),
        "best_cluster_mean_value": float(np.mean(values[mask])),
        "medoid_trial_number": int(row["number"]) if str(row["number"]).isdigit() else None,
        "medoid_ref": str(row["number"]),
        "medoid_source": str(row["source"]),
        "medoid_value": float(row["value"]),
        "medoid_params": params,
        "cluster_table": table,
        "_df": df, "_X": X, "_labels": labels, "_medoid_index": med_i,
    }


def augment_around_noise_candidates(db_path, study_name, objective_fn, top_m=3,
                                    n_per_candidate=12, level=0.05, seed=42,
                                    min_cluster_size=None):
    """Evaluate n_per_candidate local perturbations around each of the top_m noise points.

    Perturbations are uniform within +/- level per dimension in normalized [0,1] space,
    mapped back to raw params and evaluated with objective_fn. A stable plateau yields a
    dense clump of similarly-valued points that HDBSCAN clusters on the re-run; a fragile
    peak yields a spatial clump with poor values that ranks low.

    Returns (aug_df, candidate_summaries). aug_df is NOT written to the Optuna DB - the
    study's sampler history stays clean.
    """
    df = load_trials(db_path, study_name)
    X = normalize_params(df)
    values = df["value"].values.astype(float)
    labels, _ = cluster_trials(X, min_cluster_size)
    noise_idx = np.flatnonzero(labels == -1)
    if len(noise_idx) == 0:
        print("  [augment] no noise points in this study - nothing to augment")
        return pd.DataFrame(columns=list(df.columns)), []
    cand_idx = noise_idx[np.argsort(values[noise_idx])[::-1][:top_m]]

    rng = np.random.default_rng(seed)
    rows, candidates = [], []
    total = len(cand_idx) * n_per_candidate
    done = 0
    for ci in cand_idx:
        trial_no = int(df.iloc[ci]["number"])
        x0 = X[ci]
        cand_vals = []
        for k in range(n_per_candidate):
            x = np.clip(x0 + rng.uniform(-level, level, size=X.shape[1]), 0.0, 1.0)
            params = denormalize_params(x)
            params = {name: (v if name in INT_PARAMS else round_sig(float(v), 3))
                      for name, v in params.items()}
            val = float(objective_fn(params))
            done += 1
            cand_vals.append(val)
            print(f"  [augment {done}/{total}] around trial {trial_no}: value {val:.4f}", flush=True)
            rows.append({"number": f"aug{trial_no}-{k}", "value": val, **params})
        candidates.append({
            "trial_number": trial_no,
            "trial_value": float(values[ci]),
            "plateau_mean": float(np.mean(cand_vals)),
            "plateau_min": float(np.min(cand_vals)),
            "plateau_max": float(np.max(cand_vals)),
            "plateau_std": float(np.std(cand_vals)),
        })
        print(f"  [augment] trial {trial_no} plateau: mean {np.mean(cand_vals):.4f} "
              f"(min {np.min(cand_vals):.4f}, max {np.max(cand_vals):.4f})", flush=True)
    return pd.DataFrame(rows), candidates


def public_result(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def plot_cluster_scatter(result, out_dir=RESULTS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, X, labels = result["_df"], result["_X"], result["_labels"]
    med_i, best_label = result["_medoid_index"], result["best_cluster_id"]
    values = df["value"].values.astype(float)

    coords = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X

    fig, ax = plt.subplots(figsize=(10, 7))
    noise = labels == -1
    if noise.any():
        ax.scatter(coords[noise, 0], coords[noise, 1], c="lightgray", marker="x",
                   s=30, alpha=0.7, label=f"noise ({noise.sum()})")
    clustered = ~noise
    sc = ax.scatter(coords[clustered, 0], coords[clustered, 1], c=values[clustered],
                    cmap="viridis", s=45, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Objective value")
    best_mask = labels == best_label
    ax.scatter(coords[best_mask, 0], coords[best_mask, 1], facecolors="none",
               edgecolors="red", s=140, linewidths=1.4,
               label=f"best cluster #{best_label} (mean {result['best_cluster_mean_value']:.4g})")
    ax.scatter(coords[med_i, 0], coords[med_i, 1], marker="*", c="red", s=420,
               edgecolors="black", linewidths=1.0, zorder=10,
               label=f"medoid (trial {result['medoid_trial_number']}, value {result['medoid_value']:.4g})")

    ax.set_title(f"HDBSCAN trial clusters - {result['study']}\n{result['method']}, "
                 f"{result['n_trials']} trials, {len(set(labels) - {-1})} clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe = result["study"].replace("/", "_")
    path = os.path.join(out_dir, f"cluster_medoid_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(result):
    print(f"\n{'=' * 72}")
    print(f" CLUSTER MEDOID REPORT - {result['study']}")
    print(f"{'=' * 72}")
    print(f"  method            : {result['method']}")
    aug_txt = f" + {result['n_augmented']} augmented" if result.get("n_augmented") else ""
    print(f"  trials            : {result['n_trials']}{aug_txt} (noise: {result['n_noise']})")
    print("  note              : 'noise' = sparsely sampled by the optimizer, NOT necessarily")
    print("                      unstable - stability is what the perturbation check measures.")
    if result["global_best_is_noise"]:
        print(f"  global best       : trial {result['global_best_trial_number']} "
              f"(value {result['global_best_value']:.4f}) is a noise point. If perturbation")
        print("                      says it is stable, the region is likely under-sampled, not")
        print("                      lucky - run with --trajectory / --augment to verify.")
    print(f"  best cluster      : #{result['best_cluster_id']} "
          f"(size {result['best_cluster_size']}, mean value {result['best_cluster_mean_value']:.4f})")
    print(f"  medoid            : {result['medoid_ref']} [{result['medoid_source']}] "
          f"(objective value {result['medoid_value']:.4f})")
    print("  medoid params     :")
    for k, v in result["medoid_params"].items():
        print(f"    {k:<24}: {v}")
    print("  cluster ranking (by mean value):")
    aug_hdr = f" {'aug':>5}" if result.get("n_augmented") else ""
    print(f"    {'cluster':>8} {'size':>6} {'mean':>12} {'max':>12}{aug_hdr}  notes")
    for r in result["cluster_table"]:
        notes = []
        if r["is_noise"]:
            notes.append("noise")
        if r["cluster_id"] == result["best_cluster_id"]:
            notes.append("BEST")
        aug_col = f" {r.get('n_augmented', 0):>5}" if result.get("n_augmented") else ""
        print(f"    {r['cluster_id']:>8} {r['size']:>6} {r['mean_value']:>12.4f} "
              f"{r['max_value']:>12.4f}{aug_col}  {', '.join(notes)}")


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN cluster medoid extractor for Optuna studies")
    parser.add_argument("--db", default=os.path.join(RESULTS_DIR, OPTUNA_DB_FILE),
                        help="Path to the Optuna SQLite DB")
    parser.add_argument("--study", default=None, help="Study name (e.g. RTTH-NAYO)")
    parser.add_argument("--all", action="store_true", help="Process every study in the DB")
    parser.add_argument("--min_cluster_size", type=int, default=None,
                        help="HDBSCAN min_cluster_size override (default: max(8, 2%% of trials))")
    parser.add_argument("--trajectory", action="store_true",
                        help="Also run the convergence-trajectory diagnostic for the best trial")
    parser.add_argument("--augment", action="store_true",
                        help="Perturbation-augmented clustering: evaluate local perturbations "
                        "around the top noise candidates before clustering, so stable plateaus "
                        "can form real clusters. Pair is parsed from the study name")
    parser.add_argument("--aug_top_m", type=int, default=3,
                        help="Number of top-value noise candidates to augment (default 3)")
    parser.add_argument("--aug_per_candidate", type=int, default=12,
                        help="Perturbation evals per candidate (default 12)")
    parser.add_argument("--aug_level", type=float, default=0.05,
                        help="Perturbation half-width in normalized space per dim (default 0.05)")
    parser.add_argument("--aug_objective", choices=["purged_wf", "window"], default="purged_wf",
                        help="Objective for augmented evals: purged_wf (matches the main study) "
                        "or window (plain score over --aug_start/--aug_end)")
    parser.add_argument("--aug_start", type=int, default=201)
    parser.add_argument("--aug_end", type=int, default=800)
    parser.add_argument("--aug_folds", type=int, default=4)
    parser.add_argument("--aug_embargo", type=int, default=10)
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip PCA scatter PNG")
    parser.add_argument("--out", default=None, help="JSON output path "
                        "(default: <db dir>/cluster_medoids.json)")
    args = parser.parse_args()

    if not args.study and not args.all:
        parser.error("Pass --study <name> or --all")

    studies = list_studies(args.db) if args.all else [args.study]
    out_path = args.out or os.path.join(os.path.dirname(args.db) or ".", "cluster_medoids.json")

    def build_aug_objective(pair_y, pair_x):
        if args.aug_objective == "purged_wf":
            def obj(params):
                return evaluate_purged_walk_forward(
                    {**params, "pair_y": pair_y, "pair_x": pair_x},
                    args.aug_start, args.aug_end, n_folds=args.aug_folds, embargo=args.aug_embargo)
        else:
            def obj(params):
                *_, dvol, days, pll, s = run_official_eval(
                    {**params, "pair_y": pair_y, "pair_x": pair_x},
                    start_day=args.aug_start, end_day=args.aug_end)
                return s - 5.0 if dvol == 0.0 else s
        return obj

    all_results = {}
    for study_name in studies:
        try:
            aug_df, aug_candidates = None, None
            if args.augment:
                parts = study_name.split("-")
                if len(parts) < 2:
                    print(f"  [skip] --augment needs study names like '<Y>-<X>[...]', got '{study_name}'")
                    continue
                n_evals = args.aug_top_m * args.aug_per_candidate
                print(f"\n  [augment] {study_name}: {n_evals} perturbation evals "
                      f"(objective={args.aug_objective}, days {args.aug_start}-{args.aug_end})")
                aug_df, aug_candidates = augment_around_noise_candidates(
                    args.db, study_name, build_aug_objective(parts[0], parts[1]),
                    top_m=args.aug_top_m, n_per_candidate=args.aug_per_candidate,
                    level=args.aug_level, min_cluster_size=args.min_cluster_size)
            result = find_best_cluster_medoid(
                args.db, study_name, min_cluster_size=args.min_cluster_size,
                extra_trials=aug_df if aug_df is not None and len(aug_df) else None)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue
        print_summary(result)
        if args.plot:
            path = plot_cluster_scatter(result)
            print(f"  plot saved to {path}")
        entry = public_result(result)
        if aug_candidates is not None:
            entry["augmentation"] = {
                "objective": args.aug_objective, "start": args.aug_start, "end": args.aug_end,
                "level": args.aug_level, "n_evals": int(len(aug_df)),
                "candidates": aug_candidates,
                "points": aug_df.to_dict("records") if len(aug_df) else [],
            }
        if args.trajectory:
            diag = trajectory_diagnostics(args.db, study_name)
            print_trajectory(diag)
            if args.plot:
                path = plot_trajectory(diag)
                print(f"  trajectory plot saved to {path}")
            entry["trajectory"] = {k: v for k, v in diag.items() if not k.startswith("_")}
        all_results[study_name] = entry

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved medoid results for {len(all_results)} study(ies) to {out_path}")


if __name__ == "__main__":
    main()
