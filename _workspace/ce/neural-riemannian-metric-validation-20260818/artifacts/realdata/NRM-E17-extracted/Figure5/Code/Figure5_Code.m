% Figure 5 Analysis - NDNF Interneuron Selectivity and Representational Drift
%
% This script analyzes NDNF interneuron soma calcium activity during rule
% learning
%
% Data: Longitudinal GCaMP imaging from NDNF interneuron somas across 5 sessions
% Paradigm: Rule A → B → A relearning
%
% Associated publication: Maristany de las Casas et al. (2025)
% Figure 5: NDNF interneuron coding during rule learning

%% Section 1: Remove Inactive Interneurons
% DeleteFullZero removes NDNF somas that show zero activity across all sessions
% This ensures analysis focuses only on active interneurons with detectable signals
%
% Inputs: type_sum_all, sig_all, cd_choice_all, cd_stim_all (5 sessions each)
% Outputs: Same structures with inactive somas removed

[type_sum_all, sig_all, cd_choice_all, cd_stim_all] = ...
    DeleteFullZero(type_sum_all, sig_all, cd_choice_all, cd_stim_all);

%% Section 2: Sort Interneurons by Selectivity Peak
% SortROIsByPeak organizes interneurons based on when they show peak selectivity
% during the trial, creating sorted heatmaps for visualization
%
% aligned = 3: Align all days to Day 3 (first relearning day, B→A transition)
% This allows comparison of selectivity patterns relative to the relearning session
%
% Outputs:
% - Z: Sorted activity traces (type1 and type2 concatenated)
% - B: Sorted selectivity matrix
% - sigs: Significance flags for Choice, Stimuli, Delay periods

aligned = 3;  % Align all sessions to Day 3 (relearning day)
[Z, B, sigs] = SortROIsByPeak(type_sum_all, sig_all, aligned, x_df, 0.5);

%% Section 3: Extract Epoch-Specific Activity
% coding_arranger extracts NDNF interneuron activity during specific task epochs:
% - Instruction period: Sensory input encoding
% - Choice period: Motor output encoding  
% - Delay period: Working memory maintenance
%
% tested = 1:3: Analyze days 1-3 (Rule A baseline sessions)
%
% Outputs:
% - Z_choi/stim/delay: Activity during each epoch
% - coding_choice/stim/delay: Indices of significantly coding interneurons

tested = 1:3;  
mindays = 0;   % Include interneurons that are selective in at least in >0 days
[Z_choi, Z_stim, Z_delay, coding_choice, coding_stim, coding_delay] = ...
    coding_arranger(Z, x_df, sigs, tested, mindays);

%% Section 4: Quantify Representational Drift Across Epochs
% rep_drift_selec calculates how similar NDNF interneuron representations are
% across different days using pairwise correlations
%
% The function compares:
% - A-A: Same rule across days (stability baseline)
% - A-B: Rule change (expected drift)
% - A-A2: Return to original rule after intervening change
%
% Analysis performed separately for three task periods:
% 1. Choice period: Motor output encoding
% 2. Instruction period: Sensory input encoding
% 3. Delay period: Working memory maintenance
%
% Outputs for each epoch:
% - p_AB, p_AA2, p_A2B: Pairwise statistical comparisons
% - p_friedman: Friedman test across all comparisons
% - p_anova: One-way ANOVA

% Choice period drift
[choi.p_AB, choi.p_AA2, choi.p_A2B, choi.p_friedman, choi.p_anova] = ...
    rep_drift_selec(Z_choi, coding_choice);

% Instruction period drift
[stim.p_AB, stim.p_AA2, stim.p_A2B, stim.p_friedman, stim.p_anova] = ...
    rep_drift_selec(Z_stim, coding_stim);

% Delay period drift
[del.p_AB, del.p_AA2, del.p_A2B, del.p_friedman, del.p_anova] = ...
    rep_drift_selec(Z_delay, coding_delay);

%% Section 5: Visualize Population Activity Across All Sessions
% Plot_Selec_CD_Avg creates comprehensive 4-row visualization across all 5 days:
% Row 1: Average activity for left vs right trials (type1 vs type2)
% Row 2: Selectivity magnitude (|type1 - type2|)
% Row 3: Coding direction during instruction period (cd_stim)
% Row 4: Coding direction during choice period (cd_choice)
%
% Each column = one session, showing how NDNF coding evolves across learning
%
% Y-axis limits:
% - ylim_avg (0.4): Scale for average activity
% - ylim_sel (0.2): Scale for selectivity
% - ylim_cd (0.1): Scale for coding direction projections

ylim_avg = 0.4;  % Y-limit for average activity
ylim_sel = 0.2;  % Y-limit for selectivity
ylim_cd = 0.1;   % Y-limit for coding direction

% Generate multi-panel plot showing all 5 sessions
Plot_Selec_CD_Avg(type_sum_all, cd_choice_all, cd_stim_all, x_df, ylim_avg, ylim_sel, ylim_cd);
