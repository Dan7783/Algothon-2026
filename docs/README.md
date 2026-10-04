# docs/

All 102 research reports from the project, grouped by theme. The annotated catalogue, with a one-line summary
of every report, is in the [main README](../README.md#research-library-every-report).

| Folder | Theme |
|---|---|
| [`literature/`](literature/) | Briefs on the papers we built on, and assessments of external research briefs |
| [`01-data-and-market-structure/`](01-data-and-market-structure/) | What the synthetic generator is, regime drift, seasonality, market beta |
| [`02-pair-discovery/`](02-pair-discovery/) | Cointegration screens, Johansen networks, pair audits and per-pair investigations |
| [`03-kalman-cusum-filter/`](03-kalman-cusum-filter/) | The adaptive-beta pair filter: tuning, simplification, change-detection calibration |
| [`04-books-and-sleeves/`](04-books-and-sleeves/) | Portfolio books, sleeve attribution, execution and factor audits |
| [`05-validation-and-overfitting/`](05-validation-and-overfitting/) | Validation protocol, null testing, decay and overfitting studies |
| [`06-ou-engine-and-tuning/`](06-ou-engine-and-tuning/) | OU optimal-stopping engine, universe screen, verification, hybrids |
| [`07-signal-overlap-and-conflicts/`](07-signal-overlap-and-conflicts/) | How pair and ridge signals are combined: voting, deadlocks, EV and ML arbitrators |
| [`08-ridge-var-sleeve/`](08-ridge-var-sleeve/) | The cross-sectional ridge VAR lead-lag sleeve |
| [`09-algo-instrument/`](09-algo-instrument/) | Instrument 0 (ALGO): market proxy, residual basket, forecasts |
| [`figures/`](figures/) | Figures embedded in the reports and the main README |

Every report opens with a provenance line naming its original location in the working repository. Script paths
inside the reports were updated to this repo's layout.

## Where each report came from

| Original path | Now |
|---|---|
| `novy-marx-multisignal-brief.md` | [`literature/novy-marx-multisignal-brief.md`](literature/novy-marx-multisignal-brief.md) |
| `model-free-aggregation-brief.md` | [`literature/model-free-aggregation-brief.md`](literature/model-free-aggregation-brief.md) |
| `signal_overlap_brief_assessment.md` | [`literature/signal-overlap-brief-assessment.md`](literature/signal-overlap-brief-assessment.md) |
| `analysis_outputs/generator_forensics/forensics.txt` | [`01-data-and-market-structure/generator-forensics.md`](01-data-and-market-structure/generator-forensics.md) |
| `analysis_outputs/generator_forensics/seed_search.txt` | [`01-data-and-market-structure/rng-seed-recovery-attempt.md`](01-data-and-market-structure/rng-seed-recovery-attempt.md) |
| `TODO_seasonality.md` | [`01-data-and-market-structure/seasonality-and-reversal.md`](01-data-and-market-structure/seasonality-and-reversal.md) |
| `analysis_outputs/pair_correlation_by_year/pair_correlation_by_year.md` | [`01-data-and-market-structure/pair-correlation-by-year.md`](01-data-and-market-structure/pair-correlation-by-year.md) |
| `cusum_output/daniel_pairs/beta_exposure.md` | [`01-data-and-market-structure/market-beta-exposure.md`](01-data-and-market-structure/market-beta-exposure.md) |
| `research/cointegration/cointegration_output/all_pair_spread_scores/README.md` | [`02-pair-discovery/all-pair-spread-scores.md`](02-pair-discovery/all-pair-spread-scores.md) |
| `research/cointegration/cointegration_output/short_regime_analysis/README.md` | [`02-pair-discovery/short-regime-analysis.md`](02-pair-discovery/short-regime-analysis.md) |
| `cusum_output/kalman_pair_discovery/kalman_stationary_pairs.md` | [`02-pair-discovery/kalman-stationary-pairs.md`](02-pair-discovery/kalman-stationary-pairs.md) |
| `cusum_output/kalman_pair_discovery/johansen_report.md` | [`02-pair-discovery/johansen-kalman-network.md`](02-pair-discovery/johansen-kalman-network.md) |
| `cusum_output/kalman_pair_discovery/johansen_all_pairs.md` | [`02-pair-discovery/johansen-all-pairs.md`](02-pair-discovery/johansen-all-pairs.md) |
| `cusum_output/kalman_pair_discovery/johansen_all_pairs_i1.md` | [`02-pair-discovery/johansen-all-pairs-i1.md`](02-pair-discovery/johansen-all-pairs-i1.md) |
| `cusum_output/kalman_pair_discovery/johansen_all_pairs_i1_strict.md` | [`02-pair-discovery/johansen-all-pairs-i1-strict.md`](02-pair-discovery/johansen-all-pairs-i1-strict.md) |
| `analysis_outputs/testbedv2_pair_structure/pair_structure_report.md` | [`02-pair-discovery/pair-structure-audit.md`](02-pair-discovery/pair-structure-audit.md) |
| `analysis_outputs/testbedv2_corr_cusum_screen/corr_cusum_screen_report.md` | [`02-pair-discovery/correlation-cusum-screen.md`](02-pair-discovery/correlation-cusum-screen.md) |
| `analysis_outputs/pair_clusters/report.md` | [`02-pair-discovery/pair-clusters.md`](02-pair-discovery/pair-clusters.md) |
| `analysis_outputs/pair_clusters/cluster_backtest.md` | [`02-pair-discovery/pair-cluster-basket-backtest.md`](02-pair-discovery/pair-cluster-basket-backtest.md) |
| `analysis_outputs/new_tuning_pairs_walkforward/walkforward_report.md` | [`02-pair-discovery/restored-pairs-walkforward.md`](02-pair-discovery/restored-pairs-walkforward.md) |
| `analysis_outputs/new_tuning_pairs_annual_walkforward/annual_walkforward_report.md` | [`02-pair-discovery/restored-pairs-annual-walkforward.md`](02-pair-discovery/restored-pairs-annual-walkforward.md) |
| `analysis_outputs/removed_pair_investigation/report.md` | [`02-pair-discovery/removed-pair-investigation.md`](02-pair-discovery/removed-pair-investigation.md) |
| `cusum_output/daniel_pairs/pair_prune.md` | [`02-pair-discovery/pair-prune-leave-one-out.md`](02-pair-discovery/pair-prune-leave-one-out.md) |
| `cusum_output/daniel_pairs/mdgi_htrk_investigation.md` | [`02-pair-discovery/mdgi-htrk-investigation.md`](02-pair-discovery/mdgi-htrk-investigation.md) |
| `cusum_output/daniel_pairs/nayo_srtx_investigation.md` | [`02-pair-discovery/nayo-srtx-investigation.md`](02-pair-discovery/nayo-srtx-investigation.md) |
| `cusum_output/daniel_pairs/aeno_nwig_proof.md` | [`02-pair-discovery/aeno-nwig-retune-proof.md`](02-pair-discovery/aeno-nwig-retune-proof.md) |
| `cusum_output/pair_seed_explorer/report.md` | [`02-pair-discovery/pair-seed-explorer/rtth-nayo.md`](02-pair-discovery/pair-seed-explorer/rtth-nayo.md) |
| `cusum_output/pair_seed_explorer/ccns_srtx/report.md` | [`02-pair-discovery/pair-seed-explorer/ccns-srtx.md`](02-pair-discovery/pair-seed-explorer/ccns-srtx.md) |
| `cusum_output/pair_seed_explorer/ctgi_eelt/report.md` | [`02-pair-discovery/pair-seed-explorer/ctgi-eelt.md`](02-pair-discovery/pair-seed-explorer/ctgi-eelt.md) |
| `cusum_output/pair_seed_explorer/hett_ulxy/report.md` | [`02-pair-discovery/pair-seed-explorer/hett-ulxy.md`](02-pair-discovery/pair-seed-explorer/hett-ulxy.md) |
| `cusum_output/pair_seed_explorer/hret_ctgi/report.md` | [`02-pair-discovery/pair-seed-explorer/hret-ctgi.md`](02-pair-discovery/pair-seed-explorer/hret-ctgi.md) |
| `cusum_output/pair_seed_explorer/huxz_acac/report.md` | [`02-pair-discovery/pair-seed-explorer/huxz-acac.md`](02-pair-discovery/pair-seed-explorer/huxz-acac.md) |
| `cusum_output/STAGED_TUNING.md` | [`03-kalman-cusum-filter/staged-tuning.md`](03-kalman-cusum-filter/staged-tuning.md) |
| `cusum_output/FILTER_SIMPLIFICATION.md` | [`03-kalman-cusum-filter/filter-simplification.md`](03-kalman-cusum-filter/filter-simplification.md) |
| `cusum_output/SLIM_BOOK.md` | [`03-kalman-cusum-filter/slim-book.md`](03-kalman-cusum-filter/slim-book.md) |
| `cusum_output/DEAD_PAIR_ROWS.md` | [`03-kalman-cusum-filter/dead-pair-rows.md`](03-kalman-cusum-filter/dead-pair-rows.md) |
| `cusum_output/pair_seg_tuning/tuning_report.md` | [`03-kalman-cusum-filter/pairseg-kalman-tuning.md`](03-kalman-cusum-filter/pairseg-kalman-tuning.md) |
| `cusum_output/multi_pair_dashboard/multi_pair_dashboard_report.md` | [`03-kalman-cusum-filter/multi-pair-dashboard.md`](03-kalman-cusum-filter/multi-pair-dashboard.md) |
| `cusum_output/pair_strategy_wf/wf_report.md` | [`03-kalman-cusum-filter/pairseg-walkforward.md`](03-kalman-cusum-filter/pairseg-walkforward.md) |
| `research/cusum_legacy/arl_calibration/arl_calibration_report.md` | [`03-kalman-cusum-filter/cusum-arl-calibration.md`](03-kalman-cusum-filter/cusum-arl-calibration.md) |
| `research/cusum_legacy/injection_test/injection_test_report.md` | [`03-kalman-cusum-filter/cusum-injection-test.md`](03-kalman-cusum-filter/cusum-injection-test.md) |
| `research/cusum_legacy/kalman_validation_output/SMAH_NGTE_validation_report.md` | [`03-kalman-cusum-filter/smah-ngte-kalman-validation.md`](03-kalman-cusum-filter/smah-ngte-kalman-validation.md) |
| `cusum_output/daniel_pairs/ou_scale_fix.md` | [`03-kalman-cusum-filter/ou-signal-scale-fix.md`](03-kalman-cusum-filter/ou-signal-scale-fix.md) |
| `cusum_output/daniel_pairs/adf_gate_necessity.md` | [`03-kalman-cusum-filter/adf-gate-necessity.md`](03-kalman-cusum-filter/adf-gate-necessity.md) |
| `cusum_output/fifty_pairs/report.md` | [`04-books-and-sleeves/fifty-pairs.md`](04-books-and-sleeves/fifty-pairs.md) |
| `cusum_output/fifty_pairs/timing_analysis.md` | [`04-books-and-sleeves/entry-timing-analysis.md`](04-books-and-sleeves/entry-timing-analysis.md) |
| `cusum_output/pair_books/report.md` | [`04-books-and-sleeves/pair-books.md`](04-books-and-sleeves/pair-books.md) |
| `cusum_output/robust_hybrid/report.md` | [`04-books-and-sleeves/robust-hybrid.md`](04-books-and-sleeves/robust-hybrid.md) |
| `cusum_output/robust_hybrid/adaptive_v2_report.md` | [`04-books-and-sleeves/robust-hybrid-adaptive-v2.md`](04-books-and-sleeves/robust-hybrid-adaptive-v2.md) |
| `random_strat1_assessment.md` | [`04-books-and-sleeves/random-strat1-assessment.md`](04-books-and-sleeves/random-strat1-assessment.md) |
| `analysis_outputs/current_execution_audit/report.md` | [`04-books-and-sleeves/execution-audit.md`](04-books-and-sleeves/execution-audit.md) |
| `analysis_outputs/factor_exposure/report.md` | [`04-books-and-sleeves/factor-exposure.md`](04-books-and-sleeves/factor-exposure.md) |
| `analysis_outputs/block_850_950_attribution/report.md` | [`04-books-and-sleeves/weak-block-851-950-attribution.md`](04-books-and-sleeves/weak-block-851-950-attribution.md) |
| `cusum_output/VALIDATION_PROTOCOL.md` | [`05-validation-and-overfitting/validation-protocol.md`](05-validation-and-overfitting/validation-protocol.md) |
| `cusum_output/OVERFIT_REDUCTION.md` | [`05-validation-and-overfitting/overfit-reduction.md`](05-validation-and-overfitting/overfit-reduction.md) |
| `cusum_output/STEP3_PER_PAIR_VERDICT.md` | [`05-validation-and-overfitting/per-pair-tuning-verdict.md`](05-validation-and-overfitting/per-pair-tuning-verdict.md) |
| `cusum_output/SIMPLE_VS_COMPLEX.md` | [`05-validation-and-overfitting/simple-vs-complex.md`](05-validation-and-overfitting/simple-vs-complex.md) |
| `analysis_outputs/synthetic_null/report.md` | [`05-validation-and-overfitting/synthetic-null-testing.md`](05-validation-and-overfitting/synthetic-null-testing.md) |
| `analysis_outputs/top30_oos_decay_report.md` | [`05-validation-and-overfitting/top30-oos-decay.md`](05-validation-and-overfitting/top30-oos-decay.md) |
| `cusum_output/daniel_pairs/tuning_report.md` | [`05-validation-and-overfitting/pareto-pair-tuning.md`](05-validation-and-overfitting/pareto-pair-tuning.md) |
| `cusum_output/daniel_pairs/train_test_balance.md` | [`05-validation-and-overfitting/train-test-balance.md`](05-validation-and-overfitting/train-test-balance.md) |
| `DIAGNOSIS_AND_RECOMMENDATIONS.md` | [`06-ou-engine-and-tuning/ou-engine-diagnosis.md`](06-ou-engine-and-tuning/ou-engine-diagnosis.md) |
| `FRIEND_BOOK_REFERENCE.md` | [`06-ou-engine-and-tuning/baseline-book-reference.md`](06-ou-engine-and-tuning/baseline-book-reference.md) |
| `auto_tuning_results/screen_report.txt` | [`06-ou-engine-and-tuning/pair-screen-train.md`](06-ou-engine-and-tuning/pair-screen-train.md) |
| `auto_tuning_results/screen_oos_report.txt` | [`06-ou-engine-and-tuning/pair-screen-oos.md`](06-ou-engine-and-tuning/pair-screen-oos.md) |
| `auto_tuning_results/verification_report.txt` | [`06-ou-engine-and-tuning/pair-verification.md`](06-ou-engine-and-tuning/pair-verification.md) |
| `auto_tuning_results/_ou_winners_table.txt` | [`06-ou-engine-and-tuning/ou-winners-table.md`](06-ou-engine-and-tuning/ou-winners-table.md) |
| `analysis_outputs/four_sigmas_ou_additions/report.md` | [`06-ou-engine-and-tuning/ou-additions-to-four-sigmas.md`](06-ou-engine-and-tuning/ou-additions-to-four-sigmas.md) |
| `analysis_outputs/four_sigmas_ou_additions/report_margin_grid.md` | [`06-ou-engine-and-tuning/ou-additions-margin-grid.md`](06-ou-engine-and-tuning/ou-additions-margin-grid.md) |
| `analysis_outputs/signal_overlap/OVERLAP_VERDICT.md` | [`07-signal-overlap-and-conflicts/overlap-verdict.md`](07-signal-overlap-and-conflicts/overlap-verdict.md) |
| `analysis_outputs/signal_overlap/GBM_DEEPDIVE.md` | [`07-signal-overlap-and-conflicts/conflict-ml-gbm-deepdive.md`](07-signal-overlap-and-conflicts/conflict-ml-gbm-deepdive.md) |
| `analysis_outputs/vote_deadlock_deepdive/final_report.md` | [`07-signal-overlap-and-conflicts/vote-deadlock-final-report.md`](07-signal-overlap-and-conflicts/vote-deadlock-final-report.md) |
| `analysis_outputs/vote_deadlock_deepdive/mechanism_report.md` | [`07-signal-overlap-and-conflicts/vote-deadlock-mechanism.md`](07-signal-overlap-and-conflicts/vote-deadlock-mechanism.md) |
| `analysis_outputs/vote_deadlock_deepdive/control_report.md` | [`07-signal-overlap-and-conflicts/vote-deadlock-controls.md`](07-signal-overlap-and-conflicts/vote-deadlock-controls.md) |
| `analysis_outputs/current_deadlocks/report.md` | [`07-signal-overlap-and-conflicts/deadlock-accounting.md`](07-signal-overlap-and-conflicts/deadlock-accounting.md) |
| `analysis_outputs/deadlock_opportunity/report.md` | [`07-signal-overlap-and-conflicts/deadlock-opportunity.md`](07-signal-overlap-and-conflicts/deadlock-opportunity.md) |
| `analysis_outputs/deadlock_ridge_barbell/report.md` | [`07-signal-overlap-and-conflicts/deadlock-ridge-barbell.md`](07-signal-overlap-and-conflicts/deadlock-ridge-barbell.md) |
| `analysis_outputs/whole_vector_ev/README.md` | [`07-signal-overlap-and-conflicts/whole-vector-ev.md`](07-signal-overlap-and-conflicts/whole-vector-ev.md) |
| `analysis_outputs/moneyness_combiner/README.md` | [`07-signal-overlap-and-conflicts/moneyness-combiner.md`](07-signal-overlap-and-conflicts/moneyness-combiner.md) |
| `analysis_outputs/daily_moneyness_combiner/README.md` | [`07-signal-overlap-and-conflicts/daily-moneyness-combiner.md`](07-signal-overlap-and-conflicts/daily-moneyness-combiner.md) |
| `analysis_outputs/asset_moneyness/README.md` | [`07-signal-overlap-and-conflicts/asset-moneyness.md`](07-signal-overlap-and-conflicts/asset-moneyness.md) |
| `analysis_outputs/adaptive_ev_pair_filter/README.md` | [`07-signal-overlap-and-conflicts/adaptive-ev-pair-filter.md`](07-signal-overlap-and-conflicts/adaptive-ev-pair-filter.md) |
| `analysis_outputs/current_overlap_blackbox/report.md` | [`07-signal-overlap-and-conflicts/overlap-blackbox-ml.md`](07-signal-overlap-and-conflicts/overlap-blackbox-ml.md) |
| `analysis_outputs/winning_pair_hedge/development_report.md` | [`07-signal-overlap-and-conflicts/winning-pair-hedge-development.md`](07-signal-overlap-and-conflicts/winning-pair-hedge-development.md) |
| `analysis_outputs/winning_pair_hedge/heldout_report.md` | [`07-signal-overlap-and-conflicts/winning-pair-hedge-heldout.md`](07-signal-overlap-and-conflicts/winning-pair-hedge-heldout.md) |
| `analysis_outputs/pair_hedge_execution/report.md` | [`07-signal-overlap-and-conflicts/pair-hedge-execution.md`](07-signal-overlap-and-conflicts/pair-hedge-execution.md) |
| `analysis_outputs/ridge_smallbarbell_report.md` | [`08-ridge-var-sleeve/ridge-capacity-small-barbell.md`](08-ridge-var-sleeve/ridge-capacity-small-barbell.md) |
| `analysis_outputs/ridge_time_trend/report.md` | [`08-ridge-var-sleeve/ridge-time-trend.md`](08-ridge-var-sleeve/ridge-time-trend.md) |
| `analysis_outputs/ridge_parameter_conflicts/report.md` | [`08-ridge-var-sleeve/ridge-parameter-conflicts.md`](08-ridge-var-sleeve/ridge-parameter-conflicts.md) |
| `analysis_outputs/ridge_conflict_policy/report.md` | [`08-ridge-var-sleeve/ridge-conflict-policy.md`](08-ridge-var-sleeve/ridge-conflict-policy.md) |
| `analysis_outputs/ridge_rolling_reliability/report.md` | [`08-ridge-var-sleeve/ridge-rolling-reliability.md`](08-ridge-var-sleeve/ridge-rolling-reliability.md) |
| `analysis_outputs/algo_1000_deepdive/report.md` | [`09-algo-instrument/algo-1000-day-deepdive.md`](09-algo-instrument/algo-1000-day-deepdive.md) |
| `analysis_outputs/algo_ridge_probe/probe.txt` | [`09-algo-instrument/algo-ridge-probe.md`](09-algo-instrument/algo-ridge-probe.md) |
| `analysis_outputs/algo12_equal_sleeves/equal_sleeve_report.md` | [`09-algo-instrument/algo-twelve-equal-sleeves.md`](09-algo-instrument/algo-twelve-equal-sleeves.md) |
| `analysis_outputs/algo_deadlock_prediction/report.md` | [`09-algo-instrument/algo-deadlock-prediction.md`](09-algo-instrument/algo-deadlock-prediction.md) |
| `analysis_outputs/algo_dedicated_horizon/report.md` | [`09-algo-instrument/algo-dedicated-horizon.md`](09-algo-instrument/algo-dedicated-horizon.md) |
| `analysis_outputs/algo_joint_allocator/report.md` | [`09-algo-instrument/algo-joint-allocator.md`](09-algo-instrument/algo-joint-allocator.md) |
| `analysis_outputs/algo_long_tilt/algo_long_tilt_report.md` | [`09-algo-instrument/algo-long-tilt.md`](09-algo-instrument/algo-long-tilt.md) |
| `analysis_outputs/algo_pair_market_relative/algo_pair_interpretation.md` | [`09-algo-instrument/algo-pair-market-relative.md`](09-algo-instrument/algo-pair-market-relative.md) |
| `analysis_outputs/algo_residual_projection/report.md` | [`09-algo-instrument/algo-residual-projection.md`](09-algo-instrument/algo-residual-projection.md) |
| `analysis_outputs/mmbt_algo_forward_ols/report.md` | [`09-algo-instrument/mmbt-algo-forward-ols.md`](09-algo-instrument/mmbt-algo-forward-ols.md) |
| `analysis_outputs/random_strat1_algo/report.md` | [`09-algo-instrument/algo-in-final-submission.md`](09-algo-instrument/algo-in-final-submission.md) |
