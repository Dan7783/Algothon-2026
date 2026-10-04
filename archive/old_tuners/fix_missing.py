#!/usr/bin/env python
"""Resume script: read missing pairs from DB and run NSGA-II tuning on them.

This reads the existing `auto_tuning_results/tuning_data.db` to find which pairs
already have results, compares against every possible pair derivable from
`prices.txt`, and runs the NSGA-II robust tuner only on the missing ones.

Usage:
    # Just run all missing pairs (combinations, unordered)
    python resume_missing_pairs.py

    # Run missing pairs in both (y,x) and (x,y) orderings
    python resume_missing_pairs.py --all_pairs_ordered

    # Limit to a specific pairs JSON file (instead of all possible pairs)
    python resume_missing_pairs.py --pairs_file pairs.json

    # Override tuning parameters
    python resume_missing_pairs.py --n_regimes 5 --start 31 --end 750
"""

import argparse
import json
import os
import sqlite3
import sys
from itertools import combinations, permutations
from pathlib import Path

import pandas as pd

# Reuse everything from the existing tuner module
import auto_goat_tunerv3  # <-- rename this to match your actual filename (without .py)
                      # e.g., if your file is "auto_tuner.py", use `import auto_tuner as tuner_module`

# Use the same global paths as the tuner
DB_FILE = auto_goat_tunerv3.DB_FILE
PRICES_FILE = auto_goat_tunerv3.pricesFile
RESULTS_DIR = auto_goat_tunerv3.RESULTS_DIR


def get_completed_pairs(db_path):
    """Return the set of pair_name strings already in the pair_results table."""
    if not Path(db_path).exists():
        print(f"[DB] No database found at {db_path}; assuming nothing is completed.")
        return set()

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT pair_name FROM pair_results")
        rows = c.fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        print(f"[DB] Table pair_results not found; assuming nothing is completed.")
        return set()
    finally:
        conn.close()


def get_target_pairs(prices_file, pairs_file=None, ordered=False):
    """Determine the full set of pairs we want to have in the DB eventually."""
    if pairs_file:
        return auto_goat_tunerv3.load_pairs_from_json(pairs_file)

    df = pd.read_csv(prices_file, sep=r"\s+", header=0, index_col=None, nrows=0)
    symbols = list(df.columns)
    if ordered:
        return [{"y": y, "x": x} for y, x in permutations(symbols, 2)]
    else:
        return [{"y": y, "x": x} for y, x in combinations(symbols, 2)]


def compute_missing_pairs(target_pairs, completed_pairs):
    """Subtract completed pairs from the target list."""
    missing = []
    for p in target_pairs:
        pair_name = f"{p['y']}-{p['x']}"
        if pair_name not in completed_pairs:
            missing.append(p)
    return missing


def run_missing_pairs(missing_pairs):
    """Run the existing process_pair_worker over the missing pairs in parallel."""
    if not missing_pairs:
        print("\n[Done] No missing pairs to process. Database is complete.")
        return

    import concurrent.futures
    import multiprocessing

    print(f"\n>>> Running NSGA-II tuning on {len(missing_pairs)} missing pairs...")
    print(f"    Pairs: {[f'{p['y']}-{p['x']}' for p in missing_pairs]}")

    n_cores = multiprocessing.cpu_count()
    max_workers = min(n_cores, len(missing_pairs))

    all_sensitivities = []
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=auto_goat_tunerv3.init_worker,
            initargs=(auto_goat_tunerv3.GLOBAL_CONFIG,),
        ) as executor:
            futures = {executor.submit(auto_goat_tunerv3.process_pair_worker, pair): pair for pair in missing_pairs}
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result:
                        auto_goat_tunerv3.save_result_to_db(DB_FILE, result)
                        all_sensitivities.append({
                            "pair_name": result["pair_name"],
                            "sensitivities": result["sensitivities_auto"],
                        })
                        print(f"[Resume] Completed {result['pair_name']}")
                except Exception as e:
                    print(f"[Resume] Error processing pair {pair['y']}-{pair['x']}: {e}")
    except KeyboardInterrupt:
        print("\n[!] Keyboard interrupt detected. Shutting down gracefully...")

    if all_sensitivities:
        auto_goat_tunerv3.print_perturbation_summary(all_sensitivities)


def main():
    parser = argparse.ArgumentParser(
        description="Resume NSGA-II tuning on pairs missing from the DB"
    )
    parser.add_argument("--module", default="testbedv2",
                        help="Strategy module name to load")
    parser.add_argument("--start", type=int, default=31,
                        help="First day included in score")
    parser.add_argument("--end", type=int, default=750,
                        help="Last day included in score")
    parser.add_argument("--pairs_file", default=None,
                        help="Optional JSON file containing list of pairs (else use all pairs from prices file)")
    parser.add_argument("--n_regimes", type=int, default=5,
                        help="Number of regimes to split the data into")
    parser.add_argument("--all_pairs_ordered", action="store_true",
                        help="When scanning all pairs, test both (y,x) and (x,y) directions")
    args = parser.parse_args()

    # Propagate config into the tuner module's globals
    auto_goat_tunerv3.GLOBAL_CONFIG.update(vars(args))
    auto_goat_tunerv3.GLOBAL_CONFIG["run_all_pairs"] = args.pairs_file is None
    auto_goat_tunerv3.GLOBAL_CONFIG["all_pairs_ordered"] = args.all_pairs_ordered

    # Make sure the DB and results dir exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    auto_goat_tunerv3.init_database(DB_FILE)

    # Validate prices file
    if not Path(PRICES_FILE).exists():
        print(f"Error: prices file '{PRICES_FILE}' not found.")
        sys.exit(1)

    print("=" * 70)
    print(" RESUME MISSING PAIRS")
    print("=" * 70)
    print(f"Prices file     : {PRICES_FILE}")
    print(f"Database        : {DB_FILE}")
    print(f"Eval window     : days {auto_goat_tunerv3.GLOBAL_CONFIG['start']}-{auto_goat_tunerv3.GLOBAL_CONFIG['end']}")
    print(f"Regimes         : {auto_goat_tunerv3.GLOBAL_CONFIG['n_regimes']}")
    print(f"Pairs file      : {args.pairs_file or '(all possible pairs)'}")
    print(f"Ordered         : {args.all_pairs_ordered}")
    print("=" * 70)

    # 1. Determine which pairs SHOULD exist
    target_pairs = get_target_pairs(
        PRICES_FILE,
        pairs_file=args.pairs_file,
        ordered=args.all_pairs_ordered,
    )
    print(f"\n[1] Target pair count: {len(target_pairs)}")

    # 2. Read which pairs ALREADY exist in DB
    completed = get_completed_pairs(DB_FILE)
    print(f"[2] Completed pair count in DB: {len(completed)}")

    # 3. Diff
    missing = compute_missing_pairs(target_pairs, completed)
    print(f"[3] Missing pair count: {len(missing)}")

    if not missing:
        print("\n✓ Database already contains all target pairs. Nothing to do.")
        return

    # 4. Run the tuner on the missing ones
    run_missing_pairs(missing)

    print("\n✓ Resume run complete. You can now query the DB:")
    print(f"    sqlite3 {DB_FILE}")
    print("    SELECT pair_name, json_extract(auto_metrics, '$.score') AS score FROM pair_results ORDER BY score DESC;")


if __name__ == "__main__":
    main()