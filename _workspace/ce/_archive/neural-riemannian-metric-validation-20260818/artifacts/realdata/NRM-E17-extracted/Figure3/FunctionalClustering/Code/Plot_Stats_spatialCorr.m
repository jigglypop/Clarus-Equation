function summary = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
                                          branch_signal, branch_signal_RuleA, ...
                                          inst_noise_nearneigh, resp_noise_nearneigh)
%PLOT_STATS_SPATIALCORR Analyze spatial clustering of synaptic activity
%
% This function analyzes functional clustering by examining how synaptic
% correlations depend on spatial distance along dendritic branches.
%
% SYNTAX:
%   summary = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
%                                    branch_signal, branch_signal_RuleA)
%
%   summary = Plot_Stats_spatialCorr(branch_noise, branch_noise_RuleA, ...
%                                    branch_signal, branch_signal_RuleA, ...
%                                    inst_noise_nearneigh, resp_noise_nearneigh)
%
% INPUTS:
%   branch_noise         - [n×2] Coactivity vs distance for Rule B
%                          Column 1: Distance between synapse pairs (μm, pre-scaling)
%                          Column 2: Coactivity (noise correlation) values
%
%   branch_noise_RuleA   - [n×2] Coactivity vs distance for Rule A
%                          Same structure as branch_noise
%
%   branch_signal        - [n×2] Signal correlation vs distance for Rule B
%                          Column 1: Distance between synapse pairs (μm, pre-scaling)
%                          Column 2: Coding correlation values
%
%   branch_signal_RuleA  - [n×2] Signal correlation vs distance for Rule A
%                          Same structure as branch_signal
%
%   inst_noise_nearneigh - [n×1] Nearest neighbor distances for instruction-selective
%                          synapses (μm, pre-scaling) [OPTIONAL]
%
%   resp_noise_nearneigh - [n×1] Nearest neighbor distances for choice-selective
%                          synapses (μm, pre-scaling) [OPTIONAL]
%
% OUTPUT:
%   summary - Structure containing statistical results:
%             .RHO_signal         - Spearman correlation (distance vs signal corr, Rule B)
%             .PVAL_signal        - P-value for signal correlation (Rule B)
%             .RHO_noise          - Spearman correlation (distance vs coactivity, Rule B)
%             .PVAL_noise         - P-value for coactivity (Rule B)
%             .RHO_signal_RuleA   - Spearman correlation (distance vs signal corr, Rule A)
%             .PVAL_signal_RuleA  - P-value for signal correlation (Rule A)
%             .RHO_noise_RuleA    - Spearman correlation (distance vs coactivity, Rule A)
%             .PVAL_noise_RuleA   - P-value for coactivity (Rule A)
%             .l_noise (optional) - Test statistic comparing Rule A vs B coactivity
%             .p_noise (optional) - P-value for Rule A vs B comparison
%             .pnearest (optional)- P-value comparing nearest neighbor distances
%
% DESCRIPTION:
%   This function analyzes two types of synaptic correlations as a function
%   of spatial distance:
%   
%   1. NOISE CORRELATION (Coactivity)
%
%   2. SIGNAL CORRELATION (Coding Correlation)
%
%   The analysis tests whether correlation strength depends on spatial
%   distance, which would indicate functional clustering of synapses.
%
% ANALYSIS STEPS:
%   1. Scale distances by factor of 3 (correct ScanImage scaling error)
%   2. Remove synapse pairs separated by >20 μm
%   3. Optionally compare nearest neighbor distances (instruction vs choice)
%   4. Calculate Spearman correlations (distance vs correlation strength)
%   5. Fit linear models for visualization
%   6. Generate comparison plots
%
% EXAMPLES:
%   % Basic analysis comparing Rule A vs Rule B
%   stats = Plot_Stats_spatialCorr(noise_B, noise_A, signal_B, signal_A);
%
%   % Full analysis including nearest neighbor comparison
%   stats = Plot_Stats_spatialCorr(noise_B, noise_A, signal_B, signal_A, ...
%                                  inst_nearest, choice_nearest);
%
%
% Associated publication: Maristany de las Casas et al. (2025)

%% Section 1: Correct Distance Scaling
% Multiply distances by 3 to correct for ScanImage pixel-to-micron scaling error
% Original data was acquired with incorrect scaling factor

branch_noise(:, 1) = branch_noise(:, 1) * 3;
branch_signal(:, 1) = branch_signal(:, 1) * 3;
branch_noise_RuleA(:, 1) = branch_noise_RuleA(:, 1) * 3;
branch_signal_RuleA(:, 1) = branch_signal_RuleA(:, 1) * 3;

%% Section 2: Remove Long-Distance Pairs
% Exclude synapse pairs separated by more than 20 μm
% Analysis focuses on nearby synapses where clustering is most relevant

branch_noise(branch_noise(:, 1) > 20, :) = [];
branch_signal(branch_signal(:, 1) > 20, :) = [];
branch_noise_RuleA(branch_noise_RuleA(:, 1) > 20, :) = [];
branch_signal_RuleA(branch_signal_RuleA(:, 1) > 20, :) = [];

%% Section 3: Nearest Neighbor Analysis (Optional)
% Compare nearest neighbor distances between instruction and choice selective synapses
% This tests whether different synapse types have different clustering properties

if nargin > 4
    % Scale nearest neighbor distances
    inst_noise_nearneigh = inst_noise_nearneigh * 3;
    resp_noise_nearneigh = resp_noise_nearneigh * 3;
    
    % Statistical comparison using helper function
    % Returns p-value from Mann-Whitney U test
    [summary.pnearest, ~] = plot_bar_errors(inst_noise_nearneigh, ...
                                           resp_noise_nearneigh, 0, 0, 20);
    hold on
    ylabel('Nearest Neighbour (μm)')
end

%% Section 4: Spearman Correlation Analysis
% Test whether correlation strength depends on spatial distance
% Significant negative correlation indicates functional clustering
% (nearby synapses are more correlated than distant ones)

% Rule B - Signal correlation (coding correlation)
[summary.RHO_signal, summary.PVAL_signal] = corr(branch_signal(:, 1), ...
                                                  branch_signal(:, 2), ...
                                                  'Type', 'Spearman');

% Rule B - Noise correlation (coactivity)
[summary.RHO_noise, summary.PVAL_noise] = corr(branch_noise(:, 1), ...
                                               branch_noise(:, 2), ...
                                               'Type', 'Spearman');

% Rule A - Signal correlation
[summary.RHO_signal_RuleA, summary.PVAL_signal_RuleA] = corr(branch_signal_RuleA(:, 1), ...
                                                              branch_signal_RuleA(:, 2), ...
                                                              'Type', 'Spearman');

% Rule A - Noise correlation
[summary.RHO_noise_RuleA, summary.PVAL_noise_RuleA] = corr(branch_noise_RuleA(:, 1), ...
                                                            branch_noise_RuleA(:, 2), ...
                                                            'Type', 'Spearman');

%% Section 5: Compare Coactivity Between Rules (Optional)
% Test whether overall coactivity levels differ between Rule A and Rule B
% Only performed when nearest neighbor data is not provided

if nargin < 4
    [summary.l_noise, summary.p_noise] = ranksum(branch_noise(:, 2), ...
                                                 branch_noise_RuleA(:, 2));
end

%% Section 6: Fit Linear Models
% Create linear regression models to visualize distance-correlation relationships
% These models are used for plotting and to estimate effect sizes

% Rule A models
mdl_noise_RuA = fitlm(branch_noise_RuleA(:, 1), branch_noise_RuleA(:, 2));
[ypred_noise_RuA, yci_noise_RuA] = predict(mdl_noise_RuA, branch_noise_RuleA(:, 1));

mdl_signal_RuA = fitlm(branch_signal_RuleA(:, 1), branch_signal_RuleA(:, 2));
[ypred_signal_RuA, yci_signal_RuA] = predict(mdl_signal_RuA, branch_signal_RuleA(:, 1));

% Rule B models
mdl_noise_RuB = fitlm(branch_noise(:, 1), branch_noise(:, 2));
[ypred_noise_RuB, yci_noise_RuB] = predict(mdl_noise_RuB, branch_noise(:, 1));

mdl_signal_RuB = fitlm(branch_signal(:, 1), branch_signal(:, 2));
[ypred_signal_RuB, yci_signal_RuB] = predict(mdl_signal_RuB, branch_signal(:, 1));

%% Section 7: Visualize Distance-Correlation Relationships
% Create 2×2 subplot showing all four relationships:
% Top row: Coactivity (noise correlation)
% Bottom row: Signal correlation (coding correlation)
% Left column: Rule A
% Right column: Rule B

figure()

% Rule A - Coactivity
subplot(2, 2, 1)
plot(mdl_noise_RuA)
ylim([-0.2 0.6])
xlim([0 20])
ylabel('Coactivity')
xlabel('Distance (μm)')
title('Rule A - Coactivity')
legend off

% Rule B - Coactivity
subplot(2, 2, 2)
plot(mdl_noise_RuB)
ylim([-0.2 0.6])
xlim([0 20])
ylabel('Coactivity')
xlabel('Distance (μm)')
title('Rule B - Coactivity')
legend off

% Rule A - Signal Correlation
subplot(2, 2, 3)
plot(mdl_signal_RuA)
ylim([-0.5 1])
xlim([0 20])
ylabel('Signal Correlation')
xlabel('Distance (μm)')
title('Rule A - Signal Correlation')
legend off

% Rule B - Signal Correlation
subplot(2, 2, 4)
plot(mdl_signal_RuB)
ylim([-0.5 1])
xlim([0 20])
ylabel('Signal Correlation')
xlabel('Distance (μm)')
title('Rule B - Signal Correlation')
legend off

%% Section 8: Compare Overall Coactivity Distributions (Optional)
% Create violin plots comparing coactivity levels between Rule A and Rule B
% Only generated when nearest neighbor analysis is not performed

if nargin < 4
    figure()
    
    % Create violin plot
    vs4 = violinplot({branch_noise_RuleA(:, 2), branch_noise(:, 2)}, ...
                     {'Rule A', 'Rule B'}, ...
                     'ViolinColor', {[0.25 0.25 0.25], [0.8 0.8 0.8]}, ...
                     'ViolinAlpha', 0.3, ...
                     'ShowMean', false, ...
                     'MarkerSize', 8);
    
    ylabel('Coactivity')
    ylim([-0.1 0.5])
    yticks([0 0.5 1])
    xlim([0 2])
    
    % Add mean lines for reference
    yline(mean(branch_noise_RuleA(:, 2)), 'k--', 'LineWidth', 1.5)
    yline(mean(branch_noise(:, 2)), 'k--', 'LineWidth', 1.5)
    
    title('Coactivity Distribution by Rule')
end

end
