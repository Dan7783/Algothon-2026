# experiments/

One-off drivers behind the OU-engine, hybrid-book, CPCV and conflict-resolution studies
(mostly the work written up in [`docs/06-ou-engine-and-tuning/`](../docs/06-ou-engine-and-tuning/)).
In the working repo these lived at the root with a leading underscore; they were moved here for the
public release and given a small path bootstrap so they still import the strategy modules from the
repo root. **Run them from the repo root**, e.g.

```bash
python experiments/_run_cpcv.py --module random_strat1 --start 251 --end 1000
python experiments/_investigate_new_oos_decay.py
```

Most write their results to `auto_tuning_results/` (the JSON files already there are the outputs
quoted in the docs).

| Script | What it does |
|---|---|
| [`_analyze_oos.py`](_analyze_oos.py) | Compare per-pair train/OOS scores and retention: non-windowed latest_tuner vs windowed tuner. |
| [`_bakeoff_report.py`](_bakeoff_report.py) | Bake-off report: Sharpe-decay-primary comparison of tuner configs across pairs. |
| [`_bench_eval.py`](_bench_eval.py) | Benchmark one official-engine evaluation of a testbedv3 pair parameter set. |
| [`_build_friend_stable8_ou.py`](_build_friend_stable8_ou.py) | Build friend_stable8_ou.py: friend's latest book with stable-8 on dynamic OU. |
| [`_build_merged_catalog.py`](_build_merged_catalog.py) | Build the merged pair catalog for merged_strategy.py. |
| [`_build_merged_friend_ou.py`](_build_merged_friend_ou.py) | Build merged_friend_ou.py from friend_book_latest + OU winners. |
| [`_build_ou_candidates.py`](_build_ou_candidates.py) | Build the OU-search candidate file: top gate-passing screened pairs that are |
| [`_eval_cherry_clean.py`](_eval_cherry_clean.py) | Clean re-eval of the cherry-picked catalog (margin 0.30, all cohorts) — |
| [`_eval_ou_search_individual.py`](_eval_ou_search_individual.py) | Individual success evaluation for the 80-pair OU search. |
| [`_factor_check.py`](_factor_check.py) | Reality check: are the pairs real, or factor exposure in disguise? |
| [`_greedy_assemble.py`](_greedy_assemble.py) | Greedy portfolio assembly under the live 88-pair friend_ou ship book. |
| [`_investigate_combined_stable8_markowitz.py`](_investigate_combined_stable8_markowitz.py) | Combined sleeve book: stable-8 pairs get protected, exclusive control of |
| [`_investigate_conflict_followup.py`](_investigate_conflict_followup.py) | Follow-up: conflict labels from isolated pair-leg PnL, not asset returns. |
| [`_investigate_conflict_priority.py`](_investigate_conflict_priority.py) | Investigate pair-pair conflict winners on friend_book_latest. |
| [`_investigate_good_book_stable8.py`](_investigate_good_book_stable8.py) | Score the small "honest" 8-pair book (protected-7 union stable-8, since the |
| [`_investigate_new_oos_decay.py`](_investigate_new_oos_decay.py) | Per-pair desired-PnL attribution for four_sigmas.py on the newly extended |
| [`_pair_attribution.py`](_pair_attribution.py) | Per-pair attribution for merged_strategy book variants. |
| [`_post_tune_pipeline.py`](_post_tune_pipeline.py) | Post-tuning pipeline: archive db, regenerate merged catalog with tuned |
| [`_probe_ridge_lamset.py`](_probe_ridge_lamset.py) | Quick probe: average friend_book ridge predictions over a LAMBDA set. |
| [`_profile_window.py`](_profile_window.py) | Profile a single windowed-tuner objective evaluation. |
| [`_report_engine_winners.py`](_report_engine_winners.py) | List catalog pairs where the tuned OU engine beats the z-band engine on desired OOS PnL. |
| [`_run_bakeoff.py`](_run_bakeoff.py) | Bake-off orchestrator: runs each tuner config as one process PER PAIR, each with its |
| [`_run_cpcv.py`](_run_cpcv.py) | Combinatorial Purged Cross-Validation (Lopez de Prado) for a fixed book. |
| [`_run_cpcv_algo_combo.py`](_run_cpcv_algo_combo.py) | Combinatorial Purged CV for the ALGO combo IC sleeve. |
| [`_run_friend_ou_attribution.py`](_run_friend_ou_attribution.py) | Attribution for friend_book_latest vs merged_friend_ou (pinned env). |
| [`_run_friend_ou_bakeoff.py`](_run_friend_ou_bakeoff.py) | Pinned-env bake-off: friend_book_latest vs merged_friend_ou + conflict ablations. |
| [`_run_greedy_supervised.py`](_run_greedy_supervised.py) | Supervise _greedy_assemble.py; restart once on abnormal exit. |
| [`_run_mean_exit_close_gap.py`](_run_mean_exit_close_gap.py) | Close the friend gap under MERGED_OU_EXIT=mean. |
| [`_run_ml_conflict_prototype.py`](_run_ml_conflict_prototype.py) | Train/eval non-bang-bang conflict prototypes (in-sample first). |
| [`_run_ou_conflict_attribution.py`](_run_ou_conflict_attribution.py) | Attribution flip/suppression for OU conflict modes (pinned env). |
| [`_run_ou_conflict_bakeoff.py`](_run_ou_conflict_bakeoff.py) | Pinned-env bake-off: OU-specific conflict modes vs friend / shared voter. |
| [`_run_ou_exit_bakeoff.py`](_run_ou_exit_bakeoff.py) | Pinned-env bake-off: optimal OU exit vs conventional mean-exit. |
| [`_run_ou_search_supervised.py`](_run_ou_search_supervised.py) | Supervise the 80-pair OU-search tuner: relaunch on silent death until the |
| [`_run_standalone_exit_compare.py`](_run_standalone_exit_compare.py) | Standalone (conflict-free) desired PnL: friend z-band vs OU optimal vs OU mean. |
| [`_run_train_ou_winners_bakeoff.py`](_run_train_ou_winners_bakeoff.py) | Build/eval catalog keeping only OU pairs that beat friend z-band on TRAIN standalone. |
| [`_run_tuner_supervised.py`](_run_tuner_supervised.py) | Supervise windowed_tuner: relaunch on silent death until results JSON exists. |
| [`_summ_results.py`](_summ_results.py) | Summarise per-window Kalman-ADF screen evidence for the eight core pairs. |
| [`_tiebreak_rerank.py`](_tiebreak_rerank.py) | Tiebreak selection test: re-rank the top-M trials of each pair's global study by |
