#!/usr/bin/env python
"""Load and display data for a specific pair from the SQLite database."""

import sqlite3
import json
import pandas as pd
import os

DB_FILE = "auto_tuning_results/tuning_data.db"
PAIR_NAME = "SMAH-NGTE"

def load_pair_data(pair_name):
    if not os.path.exists(DB_FILE):
        print(f"Error: Database not found at {DB_FILE}")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Fetch main pair results
    cursor.execute("SELECT * FROM pair_results WHERE pair_name = ?", (pair_name,))
    row = cursor.fetchone()
    
    if not row:
        print(f"No data found in database for pair: {pair_name}")
        conn.close()
        return

    # Unpack row (based on table schema)
    _, y_sym, x_sym, auto_params_str, auto_metrics_str, test_metrics_str, perturbation_str = row
    
    # Parse JSON strings
    auto_params = json.loads(auto_params_str) if auto_params_str else {}
    auto_metrics = json.loads(auto_metrics_str) if auto_metrics_str else {}
    test_metrics_raw = json.loads(test_metrics_str) if test_metrics_str else {}
    perturbation_summary = json.loads(perturbation_str) if perturbation_str else []
    
    # Test metrics are nested under 'auto'
    test_metrics = test_metrics_raw.get("auto", {}) if isinstance(test_metrics_raw, dict) else {}

    # 2. Fetch Pareto Front trials
    cursor.execute("SELECT trial_number, median_score, neg_std_dev, params FROM pareto_fronts WHERE pair_name = ?", (pair_name,))
    pareto_rows = cursor.fetchall()
    conn.close()

    # --- PRINT SUMMARY ---
    print("=" * 70)
    print(f" DATA REPORT FOR PAIR: {pair_name} ({y_sym}-{x_sym})")
    print("=" * 70)

    print("\n[1] OPTIMAL KNEE-POINT PARAMETERS")
    print("-" * 70)
    for param, value in auto_params.items():
        if isinstance(value, float):
            print(f"  {param:<20}: {value:.5g}")
        else:
            print(f"  {param:<20}: {value}")

    print("\n[2] PERFORMANCE METRICS")
    print("-" * 70)
    print("  --- Train (In-Sample: Days 31-750) ---")
    for k, v in auto_metrics.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
        
    print("\n  --- Test (Out-of-Sample: Days 751-1000) ---")
    for k, v in test_metrics.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")

    print("\n[3] PERTURBATION ROBUSTNESS (20% Drop)")
    print("-" * 70)
    if perturbation_summary:
        print(f"  {'Parameter':<20} {'5% Drop':<10} {'10% Drop':<10} {'20% Drop':<10} {'Status'}")
        for item in perturbation_summary:
            # Format: [param_name, drop_20, drop_5, drop_10]
            p_name = item[0]
            drop_20 = item[1]
            drop_5 = item[2]
            drop_10 = item[3]
            status = "FRAGILE" if drop_20 > 25.0 else "OK"
            print(f"  {p_name:<20} {drop_5:<10.2f} {drop_10:<10.2f} {drop_20:<10.2f} {status}")
    else:
        print("  No perturbation data available.")

    print("\n[4] PARETO FRONT TRIALS")
    print("-" * 70)
    if pareto_rows:
        print(f"  Found {len(pareto_rows)} optimal trials. Saving to CSV...")
        # Save Pareto front to a CSV for easy viewing
        pareto_df = pd.DataFrame(pareto_rows, columns=['Trial Number', 'Median Score', 'Neg Std Dev', 'Params'])
        
        # Unpack params JSON into separate columns
        params_list = [json.loads(p) if isinstance(p, str) else p for p in pareto_df['Params']]
        params_df = pd.DataFrame(params_list)
        
        pareto_df = pd.concat([pareto_df.drop(columns=['Params']), params_df], axis=1)
        
        csv_filename = f"pareto_front_{pair_name.replace('-', '_')}.csv"
        pareto_df.to_csv(csv_filename, index=False)
        print(f"  Saved to: {csv_filename}")
        
        print("\n  Top 5 Trials by Median Score:")
        print(pareto_df.sort_values(by='Median Score', ascending=False).head(5).to_string(index=False))
    else:
        print("  No Pareto front data found for this pair.")
        
    print("\n" + "=" * 70)

if __name__ == "__main__":
    load_pair_data(PAIR_NAME)