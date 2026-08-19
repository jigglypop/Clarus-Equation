% Figure 4 Analysis - Longitudinal Dendritic Coding During Rule Learning
%
% This script analyzes how dendritic representations change across 5 consecutive
% sessions during either:
% - TEST condition: Rule relearning (A→B→A paradigm)
% - CONTROL condition: Repeated Rule A (no rule change)
%
% The analysis tracks the same dendrites across days to measure representational
% stability and drift during learning.
%
% Associated publication: Maristany de las Casas et al. (2025)
% Figure 4: Longitudinal tracking of dendritic representations

%% Section 1: Remove Inactive Dendrites
% DeleteFullZero removes dendrites that show zero activity across all sessions
% This ensures we only analyze dendrites with detectable signals
% 
% Inputs: type_sum, sig, cd_choice, cd_stim (all with 5 days of data)
% Outputs: Same structures with inactive dendrites removed

[type_sum_ctr, sig_ctr, cd_choice_ctr, cd_stim_ctr] = ...
    DeleteFullZero(type_sum_ctr, sig_ctr, cd_choice_ctr, cd_stim_ctr);

[type_sum_test, sig_test, cd_choice_test, cd_stim_test] = ...
    DeleteFullZero(type_sum_test, sig_test, cd_choice_test, cd_stim_test);

%% Section 2: Sort Dendrites by Selectivity Peak 
% SortROIsByPeak organizes dendrites based on when they show peak selectivity
% during the trial, creating sorted heatmaps for visualization
%
% For CONTROL: Align all days to Day 5 (final day of Rule A)
% For TEST: Align all days to Day 3 (relearning day)
%
% Outputs:
% - Z: Sorted activity traces (type1 and type2 concatenated)
% - B: Sorted selectivity matrix
% - sigs: Significance flags for each dendrite (Choice, Stimuli, Delay)

aligned = 5;  % Align control data to Day 5
[Z_ctr, B_ctr, sigs_ctr] = SortROIsByPeak(type_sum_ctr, sig_ctr, aligned, x_df, 0.5);

aligned = 3;  % Align test data to Day 3 (relearning day)
[Z_test, B_test, sigs_test] = SortROIsByPeak(type_sum_test, sig_test, aligned, x_df, 0.5);

%% Section 3: Extract Epoch-Specific Activity
% coding_arranger extracts activity during specific task epochs:
%
% For TEST: Analyze days 1-3 
% For CONTROL: Analyze days 3-5 
%
% mindays parameter: Minimum number of sessions a dendrite must be significant
% to be included in the "coding" population 
%
% Outputs:
% - Z_choi/stim/delay: Activity during each epoch
% - coding_choice/stim/delay: Indices of significantly coding dendrites

tested = 3:5;  % Control: analyze sessions 3-5
mindays = 0;   % at least 1 day (>0) 
[Zctr_choi, Zctr_stim, Zctr_delay, coding_choiceCtr, coding_stimCtr, coding_delayCtr] = ...
    coding_arranger(Z_ctr, x_df, sigs_ctr, tested, mindays);

tested = 1:3;  % Test: analyze sessions 1-3 (relearning period)
[Ztest_choi, Ztest_stim, Ztest_delay, coding_choiceTest, coding_stimTest, coding_delayTest] = ...
    coding_arranger(Z_test, x_df, sigs_test, tested, mindays);

%% Section 4: Quantify Representational Drift
% rep_drift_selec calculates how similar dendritic representations are
% across different days using pairwise correlations
%
% The function compares:
% - A-A: Same rule across consecutive days (stability baseline)
% - A-B: Rule change (expected drift)
% - A-A2: Same rule after intervening rule change (recovery)
%
% Outputs:
% - p_AB, p_AA2, p_A2B: Pairwise comparisons (Wilcoxon signed-rank)
% - p_friedman: Friedman test across all comparisons
% - p_anova: One-way ANOVA
% - kendall: Correlation matrix across all day pairs

% Control condition - Choice period
[ctr.p_AB, ctr.p_AA2, ctr.p_A2B, ctr.p_friedman, ctr.p_anova, ctr.kendall] = ...
    rep_drift_selec(Zctr_choi, coding_choiceCtr);

% Test condition - Choice period
[test.p_AB, test.p_AA2, test.p_A2B, test.p_friedman, test.p_anova, test.kendall] = ...
    rep_drift_selec(Ztest_choi, coding_choiceTest);

% Test condition - Delay period
[delay.p_AB, delay.p_AA2, delay.p_A2B, delay.p_friedman, delay.p_anova, delay.kendall] = ...
    rep_drift_selec(Ztest_delay, coding_delayTest);

%% Section 5: Analyze Instruction Period Drift (Optional)
% Same representational drift analysis for instruction-selective dendrites
% Note: Variable names suggest this section may need Z_stim and coding_stim
% to be properly defined - ensure these are available from coding_arranger

% [stim.p_AB, stim.p_AA2, stim.p_A2B, stim.p_friedman, stim.p_anova] = ...
%     rep_drift_selec(Z_stim, coding_stim);

%% Section 6: Visualize Population Activity Across Days
% Plot_Selec_CD_Avg creates a comprehensive 4-row visualization:
% Row 1: Average activity for left and right trials (type1 vs type2)
% Row 2: Selectivity (|type1 - type2|)
% Row 3: Coding direction during instruction period (cd_stim)
% Row 4: Coding direction during choice period (cd_choice)
%
% Each column represents one day/session
%
% Y-axis limits control the scale of each row:
% - ylim_avg: Activity amplitude (typically 0.4)
% - ylim_sel: Selectivity magnitude (typically 0.2)
% - ylim_cd: Coding direction projection (typically 0.1)

ylim_avg = 0.4;  % Y-limit for average activity plots
ylim_sel = 0.2;  % Y-limit for selectivity plots
ylim_cd = 0.1;   % Y-limit for coding direction plots

% Plot control condition (5 days of Rule A)
Plot_Selec_CD_Avg(type_sum_ctr, cd_choice_ctr, cd_stim_ctr, x_df, ylim_avg, ylim_sel, ylim_cd)

% Plot test condition (Rule A → B → A relearning)
Plot_Selec_CD_Avg(type_sum_test, cd_choice_test, cd_stim_test, x_df, ylim_avg, ylim_sel, ylim_cd)
