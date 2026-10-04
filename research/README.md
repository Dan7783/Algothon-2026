# research/

Research scripts behind almost every report in [`docs/`](../docs/). Run them from the repo root,
`python research/<script>.py` (most carry a small path bootstrap so root modules import). Outputs land in
`analysis_outputs/` or `cusum_output/`, which are created on demand. The reports those runs produced were
moved to `docs/`.

These are research snapshots, not a maintained package. A handful of the older scripts import helpers
that now live in `archive/legacy/` (for example `tune_daniel_pairs`). Add that folder to `PYTHONPATH` to
run them.

The Discord pair-review bot used to hand-review ~500 candidate pair diagnostics as a team is
documented in [`DISCORD_PAIR_REVIEW.md`](DISCORD_PAIR_REVIEW.md) (it needs `requirements-research.txt`
and a bot token in your environment; see `.env.discord.example`).

## Sub-folders

| Folder | Contents |
|---|---|
| [`cointegration/`](cointegration/) | Cointegration studies: rolling/blocked Engle-Granger, Johansen, Gregory-Hansen + Hansen Lc stability, persistence spells, lagged-return and spike-reversion tests. |
| [`cusum_legacy/`](cusum_legacy/) | The first Kalman + CUSUM regime-switch stack (ARL calibration, synthetic break injection, SMAH/NGTE validation). |
| [`pairs_analysis/`](pairs_analysis/) | Kalman pair discovery, Johansen all-pairs screens and pair networks, the sticky through-origin `PairSegFilter`, and a TradingView Pine port of the Kalman + CUSUM beta filter. |
| [`ridge_param_study/`](ridge_param_study/) | Optuna study of the ridge sleeve's lambda / theta surface (`report.json`, `trials.csv`). |

## Scripts

| Script | What it does |
|---|---|
| [`adaptive_ev_pair_filter.py`](adaptive_ev_pair_filter.py) | Adaptive whole-vector EV using pair-filter-style regime handling. |
| [`adaptive_identity_ev_pair_filter.py`](adaptive_identity_ev_pair_filter.py) | Per-signal adaptive EV states with shared EWMA/CUSUM controls. |
| [`adf_gate_ablation.py`](adf_gate_ablation.py) | Full ablation: remove the ADF prior-gate from EVERY pair. |
| [`algo12_equal_sleeve_research.py`](algo12_equal_sleeve_research.py) | Test twelve independent ALGO pair sub-books sharing the $100k ALGO cap. |
| [`algo_1000_deepdive.py`](algo_1000_deepdive.py) | Comprehensive 1,000-day investigation of instrument 0 (ALGO). |
| [`algo_deadlock_prediction_research.py`](algo_deadlock_prediction_research.py) | Can ALGO movements predict pair-vote deadlocks or resolve their direction? |
| [`algo_dedicated_horizon_audit.py`](algo_dedicated_horizon_audit.py) | Leakage-safe research for a dedicated, short-horizon ALGO sleeve. |
| [`algo_joint_allocator_research.py`](algo_joint_allocator_research.py) | Joint constrained allocation of Four Sigmas and ALGO residual signals. |
| [`algo_long_tilt_research.py`](algo_long_tilt_research.py) | Leakage-safe ALGO long-index tilt research for ``four_sigmas``. |
| [`algo_overlay_research.py`](algo_overlay_research.py) | Leakage-safe ALGO overlay research for the protected barbell strategy. |
| [`algo_residual_beta_neutral.py`](algo_residual_beta_neutral.py) | Causal ALGO-residual signals with a non-ALGO beta-neutral hedge basket. |
| [`algo_residual_projection_research.py`](algo_residual_projection_research.py) | Train/validation research on ALGO residual forecasts and beta projection. |
| [`algo_ridge_probe.py`](algo_ridge_probe.py) | Can the ridge predict ALGO? Three questions, in order. |
| [`algo_ridge_variants.py`](algo_ridge_variants.py) | Score four_sigmas variants that let the ridge trade ALGO. |
| [`all_pair_spread_score_analysis.py`](all_pair_spread_score_analysis.py) | Causal per-day spread-score research across every asset pair. |
| [`analyse_algo_pairs_market_relative.py`](analyse_algo_pairs_market_relative.py) | Determine what the historical ALGO pair signals were actually trading. |
| [`analyse_pair_clusters.py`](analyse_pair_clusters.py) | Analyse whether four_sigmas pair overlap represents tradable clusters. |
| [`analyse_ridge_barbell_modes.py`](analyse_ridge_barbell_modes.py) | Train-only attribution of ridge/pair priority rules. |
| [`analyse_testbedv2_corr_cusum_screen.py`](analyse_testbedv2_corr_cusum_screen.py) | Lightweight train-only correlation + CUSUM screen for testbedv2_tuned pairs. |
| [`analyse_testbedv2_pair_structure.py`](analyse_testbedv2_pair_structure.py) | Train-only structural audit of every active testbedv2_tuned pair. |
| [`analyse_top30_oos_decay.py`](analyse_top30_oos_decay.py) | Explain IS-to-OOS decay for the Discord-approved top30 TestbedV2 book. |
| [`analyse_tuning_candidates.py`](analyse_tuning_candidates.py) | Re-evaluate manual/auto tuning candidates on their labelled pairs. |
| [`attribute_four_sigmas.py`](attribute_four_sigmas.py) | Per-sleeve PnL attribution for four_sigmas.py. |
| [`audit_robust_hybrid.py`](audit_robust_hybrid.py) | Reproducible train/test ablation audit for robust_hybrid.py. |
| [`audit_strategy_switches.py`](audit_strategy_switches.py) | Audit binary and categorical execution switches in ``four_sigmas``. |
| [`backtest_cluster_triangles.py`](backtest_cluster_triangles.py) | Validation-only prototype for replacing overlapping pair triangles. |
| [`backtest_package.py`](backtest_package.py) | Backtest four_sigmas vs four_sigmas_pkg (+ ablations), eval.py-faithful. |
| [`barbell_untraded.py`](barbell_untraded.py) | Barbell: add SMALL confidence-scaled positions where the book currently sits flat. |
| [`benchmark_teamname.py`](benchmark_teamname.py) | Evaluate a strategy module with the same accounting as eval.py, without plots. |
| [`beta_exposure_audit.py`](beta_exposure_audit.py) | Is the book market-neutral, or is it carrying hedgeable beta? |
| [`block_850_950_attribution.py`](block_850_950_attribution.py) | Attribute the weak days 851-950 block in the current production strategy. |
| [`block_consistency.py`](block_consistency.py) | Judge strategies by consistency across MANY windows, not one test split. |
| [`build_atlas_pair_notebook.py`](build_atlas_pair_notebook.py) | Build the executed-source notebook for all atlas pair spreads. |
| [`build_dynamic_beta_notebook.py`](build_dynamic_beta_notebook.py) | Build a notebook comparing causal multi-beta models for atlas pairs. |
| [`build_momentum_reversion_notebook.py`](build_momentum_reversion_notebook.py) | Builder for notebooks/momentum_reversion_yearly.ipynb. |
| [`build_seasonality_notebook.py`](build_seasonality_notebook.py) | Builder for notebooks/seasonality_and_reversal.ipynb. |
| [`can_tuning_replace_adf.py`](can_tuning_replace_adf.py) | Can per-pair threshold tuning replace the ADF gate? |
| [`combine_lab.py`](combine_lab.py) | Combination-scheme lab for the pair and ridge sleeves. |
| [`combine_round2.py`](combine_round2.py) | Round 2: the two levers the round-1 sweep actually exposed. |
| [`combine_schemes.py`](combine_schemes.py) | Structurally distinct ways of merging the pair and ridge sleeves. |
| [`confirm_gate_removal.py`](confirm_gate_removal.py) | Portfolio confirmation for dropping require_adf on the unblocked pairs. |
| [`confirm_pair_seed.py`](confirm_pair_seed.py) | Portfolio confirmation for a tuned single-pair seed (step 4 of the proof). |
| [`conflict_ev_model.py`](conflict_ev_model.py) | Learned arbitration for pair/ridge conflicts: EV first, black box second. |
| [`conflict_ml_deepdive.py`](conflict_ml_deepdive.py) | Deep dive: can the conflict arbitrator be regularised into working? |
| [`conflict_online_ceiling.py`](conflict_online_ceiling.py) | Does continuously refeeding outcomes fix the arbitrator? Find its ceiling. |
| [`contradiction_and_netting.py`](contradiction_and_netting.py) | Two questions the who-wins framing skipped. |
| [`current_deadlock_accounting.py`](current_deadlock_accounting.py) | Count and value deadlocks under the current production strategy. |
| [`current_execution_audit.py`](current_execution_audit.py) | Audit the current four_sigmas execution layer without changing production. |
| [`current_overlap_blackbox.py`](current_overlap_blackbox.py) | Leakage-safe black-box routing for current four_sigmas signal overlaps. |
| [`deadlock_ceiling.py`](deadlock_ceiling.py) | What is a deadlock actually worth, and what accuracy would it take to win it? |
| [`deadlock_opportunity.py`](deadlock_opportunity.py) | Quantify true zero-position deadlocks caused by overlapping pair legs. |
| [`deadlock_predict.py`](deadlock_predict.py) | Can the direction on a deadlock asset-day be predicted at >52% out of window? |
| [`deadlock_ridge_barbell.py`](deadlock_ridge_barbell.py) | Audit whether latent ridge forecasts are useful on pair-vote deadlocks. |
| [`deadlock_subthreshold_ridge.py`](deadlock_subthreshold_ridge.py) | Properly test the ridge's SUB-THRESHOLD opinion on deadlock asset-days. |
| [`diagnose_nayo_srtx.py`](diagnose_nayo_srtx.py) | Why does NAYO/SRTX earn +44 in train and -4 in test? |
| [`diagnose_pair.py`](diagnose_pair.py) | Diagnose any seeded pair, and check whether its partner is the right one. |
| [`diagnose_signal_overlap.py`](diagnose_signal_overlap.py) | Measure what the pair/ridge merge actually does to the two sleeves. |
| [`diagnose_whole_vector_ev_oos.py`](diagnose_whole_vector_ev_oos.py) | Post-test diagnosis of the frozen whole-vector EV model. |
| [`discord_pair_review_bot.py`](discord_pair_review_bot.py) | Discord approval queue for pair-shortlist diagnostic plots. |
| [`effective_breadth.py`](effective_breadth.py) | Measure the EFFECTIVE breadth of the four_sigmas pair book. |
| [`evaluate_asset_moneyness.py`](evaluate_asset_moneyness.py) | One-shot evaluation of equal-family per-asset moneyness forecasts. |
| [`evaluate_daily_moneyness_combiner.py`](evaluate_daily_moneyness_combiner.py) | One-shot evaluation of fixed one-day dollar moneyness. |
| [`evaluate_moneyness_combiner.py`](evaluate_moneyness_combiner.py) | One-shot evaluation of the fixed, no-tuning moneyness combiner. |
| [`factor_exposure_audit.py`](factor_exposure_audit.py) | Economic audit of pair, ridge, and ALGO-residual factor exposure. |
| [`fifty_pairs_gate_test.py`](fifty_pairs_gate_test.py) | Adaptive-sleeve cut-frequency gate: train-only sleeve-level selection. |
| [`fifty_pairs_lab.py`](fifty_pairs_lab.py) | Analyse, tune and plot the user's 50 labeled pairs (16 stable / 34 adaptive). |
| [`filter_protectedbarbell_oos_losers.py`](filter_protectedbarbell_oos_losers.py) | Identify current pair agents with negative standalone days-751-1000 P&L. |
| [`fix_r_scaling.py`](fix_r_scaling.py) | Fix stage 1's error: the filter hardcodes observation variance R = 1. |
| [`flip_at_extremes.py`](flip_at_extremes.py) | Flip-at-extremes vs flip-at-mean, on the 16 stable pairs. |
| [`four_sigmas_merged_ou_additions.py`](four_sigmas_merged_ou_additions.py) | Development-only portability test for merged_friend_ou additions. |
| [`gate_toggle_test.py`](gate_toggle_test.py) | Does the ADF gate cost the train window trades that were worth taking? |
| [`generator_forensics.py`](generator_forensics.py) | Forensics on prices.txt: what generated this data, and is it recoverable? |
| [`investigate_removed_pairs.py`](investigate_removed_pairs.py) | Re-audit every pair excluded by the current ``four_sigmas`` catalog. |
| [`mmbt_algo_forward_ols_audit.py`](mmbt_algo_forward_ols_audit.py) | Audit MMBT lag-1 return as a forecast of ALGO's 30-day return. |
| [`pair_audit_trainonly.py`](pair_audit_trainonly.py) | Leave-one-out pair audit judged on TRAIN BLOCKS ONLY. |
| [`pair_book_lab.py`](pair_book_lab.py) | Per-pair book lab: benchmark, screen, and tune smah-style pair books. |
| [`pair_contribution_audit.py`](pair_contribution_audit.py) | Which pairs actually earn their slot in four_sigmas? |
| [`pair_correlation_by_year.py`](pair_correlation_by_year.py) | Per-year correlation drift for every pair in four_sigmas.py. |
| [`pair_discovery_research.py`](pair_discovery_research.py) | Configurable, causal pair-discovery ablation around ``teamName``. |
| [`pair_hedge_execution_audit.py`](pair_hedge_execution_audit.py) | Audit whether Four Sigmas executes its pair signals as hedged spreads. |
| [`protected_hit_rate.py`](protected_hit_rate.py) | Hit rate of the 7 protected pairs, and whether their 96.6% veto is earned. |
| [`random_strat1_algo_attribution.py`](random_strat1_algo_attribution.py) | Attribute the directly traded ALGO leg inside ``random_strat1``. |
| [`regime_gated_pairs.py`](regime_gated_pairs.py) | Can the regime-drift finding refine the PAIRS signal for OOS? |
| [`ridge_conflict_policy_research.py`](ridge_conflict_policy_research.py) | Select a simple conflict policy for the new standalone ridge parameters. |
| [`ridge_parameter_conflict_attribution.py`](ridge_parameter_conflict_attribution.py) | Attribute ridge parameter performance and pair-conflict delivery. |
| [`ridge_rolling_reliability_gate.py`](ridge_rolling_reliability_gate.py) | Research a causal rolling/expanding ridge reliability gate. |
| [`ridge_time_trend_analysis.py`](ridge_time_trend_analysis.py) | Describe how the frozen Four Sigmas ridge sleeve changes through time. |
| [`runtime_safety_audit.py`](runtime_safety_audit.py) | Runtime-safety audit for the current ``four_sigmas`` submission. |
| [`seed_search.py`](seed_search.py) | Attempt RNG seed recovery on prices.txt, with honest multiple-testing control. |
| [`short_window_cointegration_analysis.py`](short_window_cointegration_analysis.py) | Visualise and test short-lived cointegration and row-hub regimes. |
| [`shortlist_pair_review.py`](shortlist_pair_review.py) | Shortlist robust pair configurations and generate visual trade reviews. |
| [`signal_cache.py`](signal_cache.py) | Cache the raw pair/ridge primitives once so combination schemes are cheap. |
| [`sleeve_by_window.py`](sleeve_by_window.py) | Which sleeve makes the book test-heavy? |
| [`stage1_base_noise.py`](stage1_base_noise.py) | Stage 1: CUSUM off — match the Kalman to the data's base noise. |
| [`stage2_cusum.py`](stage2_cusum.py) | Stage 2: with q_beta fixed from stage 1, tune the CUSUM to DETECT breaks. |
| [`static_ev_regime_gate.py`](static_ev_regime_gate.py) | Causal regime gate between the static EV book and shipped baseline. |
| [`subset_search_tuned.py`](subset_search_tuned.py) | Exhaustive subset search over the robust per-pair candidates. |
| [`sweep_all_pairs.py`](sweep_all_pairs.py) | Self-contained rigorous tune of every seeded pair (no fragile imports). |
| [`sweep_regime_gate.py`](sweep_regime_gate.py) | Robustness check: is the pairs reversion-gate STRUCTURAL or just fitting the burned |
| [`synthetic_null.py`](synthetic_null.py) | Synthetic-data and NULL testing for four_sigmas.py. |
| [`synthetic_null_report.py`](synthetic_null_report.py) | Build the report and figure for research/synthetic_null.py output. |
| [`test_simple_filter.py`](test_simple_filter.py) | Can covariance inflation alone replace the whole post-cut adapt machinery? |
| [`test_srna_acix.py`](test_srna_acix.py) | SRNA/ACIX is the only persistently-cointegrated pair NOT in the atlas. |
| [`test_windowed_adf.py`](test_windowed_adf.py) | Hypothesis: the ADF prior-gate should use a TRAILING WINDOW, not expanding |
| [`trade_timing_analysis.py`](trade_timing_analysis.py) | Entry/exit timing analysis for the fifty-pairs stable band books. |
| [`tune_all_pairs.py`](tune_all_pairs.py) | Rigorous tune of EVERY seeded pair, then portfolio-confirm the winners. |
| [`tune_bangbang_conflicts.py`](tune_bangbang_conflicts.py) | Train/validation/test tuning of bang-bang pair/ridge conflicts. |
| [`tune_friend_book_ridge.py`](tune_friend_book_ridge.py) | Ratio-normalized purged K-window Optuna study of friend_book ridge knobs. |
| [`tune_hybrid_discovery.py`](tune_hybrid_discovery.py) | Discovery-sleeve knob sweep for robust_hybrid (two-window Pareto rule). |
| [`tune_nayo_srtx.py`](tune_nayo_srtx.py) | Rigorous NAYO/SRTX tune, with beta-modelling as the explicit question. |
| [`tune_pair_rigorous.py`](tune_pair_rigorous.py) | Rigorous NAYO/SRTX tune, with beta-modelling as the explicit question. |
| [`tune_pair_seg.py`](tune_pair_seg.py) | Sequential tuning of the sticky through-origin Kalman pair filter. |
| [`tune_ridge_confirmation.py`](tune_ridge_confirmation.py) | Tune the independent ridge conflict-confirmation model. |
| [`tune_single_pair.py`](tune_single_pair.py) | Single-pair tuning with a proof protocol, not just a score. |
| [`tune_survivors.py`](tune_survivors.py) | Step 3: tune the 14 surviving pairs, selecting on TRAIN BLOCKS ONLY. |
| [`v4_ablation.py`](v4_ablation.py) | Attribute nontestbedv4's score to the two mechanisms ported from four_sigmas. |
| [`validate_new_tuning_pairs_annual_walkforward.py`](validate_new_tuning_pairs_annual_walkforward.py) | Annual expanding walk-forward audit for the seven restored pair profiles. |
| [`validate_new_tuning_pairs_walkforward.py`](validate_new_tuning_pairs_walkforward.py) | Correlation audit and singular-pair walk-forward replay. |
| [`variant_ridge_algo_predictor.py`](variant_ridge_algo_predictor.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`variant_tiebreak_priority.py`](variant_tiebreak_priority.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`variant_vote_absz.py`](variant_vote_absz.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`variant_vote_equal.py`](variant_vote_equal.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`variant_vote_equal_noboost.py`](variant_vote_equal_noboost.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`variant_vote_rank.py`](variant_vote_rank.py) | Supplied-profile pair book with bang-bang pair/ridge voting. |
| [`vote_deadlock_controls.py`](vote_deadlock_controls.py) | Walk-forward controls for vote abstention, hysteresis, joint resolution, |
| [`vote_deadlock_mechanisms.py`](vote_deadlock_mechanisms.py) | Mechanism audit for pair-vote deadlocks in current four_sigmas. |
| [`vote_dictatorship.py`](vote_dictatorship.py) | Is _pair_conflict_consensus a jury, or a dictatorship by the largest kappa? |
| [`vote_tie_audit.py`](vote_tie_audit.py) | How often does the asset vote cancel to exactly zero, and what does it cost? |
| [`whole_vector_ev_conflicts.py`](whole_vector_ev_conflicts.py) | Train/validate/test study for whole-vector EV signal combination. |
| [`window_participation.py`](window_participation.py) | Per-window participation audit: is the train window under-traded? |
| [`winning_pair_hedge_audit.py`](winning_pair_hedge_audit.py) | Test complete dynamic-beta pair execution on consensus-winning sides. |
