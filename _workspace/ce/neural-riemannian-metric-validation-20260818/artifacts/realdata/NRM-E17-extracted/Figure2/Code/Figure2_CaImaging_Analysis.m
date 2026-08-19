%% Calcium Imaging Analysis - Figure 2
% Analysis of dendritic calcium activity in L5b pyramidal neurons during
% chemogenetic manipulation of NDNF interneurons
%
% This script analyzes:
% 1. Calcium transient properties (amplitude, frequency) in dendritic branches and spines
% 2. Behavioral performance during NDNF activation
% 3. Trial-averaged calcium responses under different experimental conditions
% 4. Effect sizes of NDNF activation on calcium activity
%
% Input data structures are loaded via DataSummarizer function (see README.md)

%% Section 1: Load Calcium Transient Properties
% Extract amplitude and frequency data for branches and spines

[branch_amp, looking] = DataSummarizer('matrix', 'branch_amp', 'vert');
[spine_amp, looking] = DataSummarizer('matrix', 'spine_amp', 'vert', looking);
[branch_freq, looking] = DataSummarizer('matrix', 'branch_freq', 'vert', looking);
[spine_freq, looking] = DataSummarizer('matrix', 'spine_freq', 'vert', looking);

%% Section 2: Organize Data by Experimental Condition
% Separate saline (control) and DCZ (NDNF activation) conditions

% Branch transient properties
branch_amp_sal = vertcat(branch_amp.Sal);
branch_amp_dcz = vertcat(branch_amp.DCZ);
branch_freq_sal = vertcat(branch_freq.Sal);
branch_freq_dcz = vertcat(branch_freq.DCZ);

% Spine transient properties
spine_amp_sal = vertcat(spine_amp.Sal);
spine_amp_dcz = vertcat(spine_amp.DCZ);
spine_freq_sal = vertcat(spine_freq.Sal);
spine_freq_dcz = vertcat(spine_freq.DCZ);

%% Section 3: Calculate Geometric Means
% Combine amplitude and frequency using geometric mean as overall activity measure

spine_geomean_sal = sqrt(spine_amp_sal .* spine_freq_sal);
spine_geomean_dcz = sqrt(spine_amp_dcz .* spine_freq_dcz);

branch_geomean_sal = sqrt(branch_amp_sal .* branch_freq_sal);
branch_geomean_dcz = sqrt(branch_amp_dcz .* branch_freq_dcz);

%% Section 4: Visualize Amplitude Distributions
% Compare calcium transient amplitudes between conditions

% Spine amplitudes
figure
boxplot([spine_amp_sal, spine_amp_dcz], 'Notch', 'off', 'Whisker', 1)
title('Spine Calcium Transient Amplitudes')
xlabel('Condition')
xticklabels({'Saline', 'DCZ'})
ylabel('Amplitude (ΔF/F)')

% Branch amplitudes
figure
boxplot([branch_amp_sal, branch_amp_dcz], 'Notch', 'off', 'Whisker', 1)
title('Branch Calcium Transient Amplitudes')
xlabel('Condition')
xticklabels({'Saline', 'DCZ'})
ylabel('Amplitude (ΔF/F)')

%% Section 5: Calculate Effect Sizes
% Quantify the impact of DCZ on calcium activity
% Effect size = (Saline - DCZ) / (Saline + DCZ)
% Range: -1 (complete suppression) to 1 (complete enhancement)

effect_branch = (branch_amp_sal - branch_amp_dcz) ./ (branch_amp_sal + branch_amp_dcz);
effect_branch(isnan(effect_branch)) = 1;  % Handle division by zero cases

effect_spine = (spine_amp_sal - spine_amp_dcz) ./ (spine_amp_sal + spine_amp_dcz);

%% Section 6: Compare Effect Sizes Between ROI Types

% Combine data for visualization
x = [effect_spine; effect_branch];
g1 = repmat({'Spines'}, length(effect_spine), 1);
g2 = repmat({'Shafts'}, length(effect_branch), 1);
g = [g1; g2];

% Create grouped boxplot
figure
boxplot(x, g, 'Notch', 'off', 'Whisker', 1, 'Symbol', '.')
ylim([-0.25 1])
ylabel('Effect Size (Saline - DCZ) / (Saline + DCZ)')
title('DCZ Effect on Calcium Activity')

% Statistical comparison (helper function plot_bar_errors)
[pvalue] = plot_bar_errors(effect_spine, effect_branch, 1);

%% Section 7: Load and Analyze Behavioral Data
% Extract behavioral performance data from cont_data structure

[cont_data_struct, looking] = DataSummarizer('matrix', 'cont_data', 'horz');

%% Section 8: Calculate Behavioral Performance
% Compute correct choice percentage for each session

n_sessions = length(cont_data_struct);
DCZ_perf = zeros(n_sessions, 1);
Sal_perf = zeros(n_sessions, 1);

for ii = 1:n_sessions
    % DCZ condition: exclude omitted trials (outcome ~= 3)
    DCZ_perf(ii, 1) = length(find(cont_data_struct(ii).DCZ.DirOut == 1)) / ...
                      length(find(cont_data_struct(ii).DCZ.DirOut ~= 3));
    
    % Saline condition: exclude omitted trials
    Sal_perf(ii, 1) = length(find(cont_data_struct(ii).Sal.DirOut == 1)) / ...
                      length(find(cont_data_struct(ii).Sal.DirOut ~= 3));
end

%% Section 9: Visualize Behavioral Performance

figure()
hold on
bar(1, nanmean(DCZ_perf), 'FaceColor', [0 1 1], 'EdgeColor', 'k')
bar(2, nanmean(Sal_perf), 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k')

% Plot individual session trajectories
for jj = 1:size(DCZ_perf, 1)
    plot([1, 2], [DCZ_perf(jj, 1) Sal_perf(jj, 1)], 'k-')
end

xticks([1 2])
xticklabels({'DCZ', 'Saline'})
ylabel('Correct Choice (%)')
title('Behavioral Performance')

% Statistical test
p_perf_chemo = signrank(DCZ_perf, Sal_perf);
disp(['Performance comparison p-value: ' num2str(p_perf_chemo)])

%% Section 10: Filter for Left Instructed Trials
% Focus analysis on left trials to match airpuff stimulation experiments

cont_data_left = cont_data_struct;
field_names = fieldnames(cont_data_left(1).DCZ);

for ii = 1:length(cont_data_left)
    for jj = 1:length(field_names)
        % Extract only left instructed trials (TrialTypes == 1)
        left_trial_idx = cont_data_left(ii).DCZ.TrialTypes(:, 1) == 1;
        cont_data_left(ii).DCZ.(field_names{jj, 1}) = ...
            cont_data_left(ii).DCZ.(field_names{jj, 1})(left_trial_idx);
    end
end

%% Section 11: Extract Trial-Averaged Calcium Activity
% Use helper function to organize calcium imaging data by ROI type

% Saline condition
[Sal.local] = iam_lazy(cont_data_left, 'Sal', 'spine_local');
[Sal.all] = iam_lazy(cont_data_left, 'Sal', 'spine_all');
[Sal.branch] = iam_lazy(cont_data_left, 'Sal', 'branch');

% DCZ condition
[DCZ.local] = iam_lazy(cont_data_left, 'DCZ', 'spine_local');
[DCZ.all] = iam_lazy(cont_data_left, 'DCZ', 'spine_all');
[DCZ.branch] = iam_lazy(cont_data_left, 'DCZ', 'branch');

%% Section 12: Baseline Normalization
% Remove minimum value to set baseline to zero

Sal.local = Sal.local - min(Sal.local, [], 1);
Sal.branch = Sal.branch - min(Sal.branch, [], 1);
DCZ.local = DCZ.local - min(DCZ.local, [], 1);
DCZ.branch = DCZ.branch - min(DCZ.branch, [], 1);

%% Section 13: Visualize Trial-Averaged Activity
% Create time vector for x-axis (assuming standard trial structure)
% Note: x_df should be defined based on your imaging frame rate and trial duration
trial_duration=180; %6 seconds times 30 Hz frame rate
x_df=linspace(-3,3,trial_duration);

figure()

% Isolated spine activity
subplot(1, 3, 1)
hold on
stdshade(Sal.local', 0.2, 'k', x_df);
stdshade(DCZ.local', 0.2, 'c', x_df);
title('Isolated Spine Events')
ylabel('z-score (ΔF/F)')
plot_epochs  % Helper function to mark trial epochs
xlim([-2.5 -0.5])

% All spine events
subplot(1, 3, 2)
hold on
stdshade(Sal.all', 0.2, 'k', x_df);
stdshade(DCZ.all', 0.2, 'c', x_df);
title('All Spine Events')
xlabel('Time (seconds)')
plot_epochs
xlim([-2.5 -0.5])

% Branch events
subplot(1, 3, 3)
hold on
stdshade(Sal.branch', 0.2, 'k', x_df);
stdshade(DCZ.branch', 0.2, 'c', x_df);
title('Branch Events')
plot_epochs
xlim([-2.5 -0.5])

%% Section 14: Alternative Normalization - Per ROI
% Normalize each ROI individually to its minimum

for ii = 1:size(Sal.branch, 2)
    Sal.branch(:, ii) = Sal.branch(:, ii) - min(Sal.branch(:, ii));
    DCZ.branch(:, ii) = DCZ.branch(:, ii) - min(DCZ.branch(:, ii));
end

for ii = 1:size(Sal.local, 2)
    Sal.local(:, ii) = Sal.local(:, ii) - min(Sal.local(:, ii));
    DCZ.local(:, ii) = DCZ.local(:, ii) - min(DCZ.local(:, ii));
end

%% Section 15: Calculate Effect Size on Mean Activity
% Quantify DCZ impact on trial-averaged responses

effect_size_branch = 1 - mean(DCZ.branch) ./ mean(Sal.branch);
effect_size_local = 1 - mean(DCZ.local) ./ mean(Sal.local);

% Statistical comparison
[l, p] = plot_bar_errors(effect_size_branch, effect_size_local, 1, 0, 1);

%% Section 16: Detailed Effect Size Visualization

figure()
hold on

% Bar plot with individual data points
bar(1, mean(effect_size_branch), 'white', 'EdgeColor', [0 1 0], 'LineWidth', 2)
bar(2, mean(effect_size_local), 'white', 'EdgeColor', [1 0 1], 'LineWidth', 2)

% Overlay individual measurements with jitter
for ee = 1:length(effect_size_branch)
    r = 0.9 + (1.1 - 0.9) .* rand(1, 1);  % Add jitter
    plot(r, effect_size_branch(ee), '.', 'Color', [0 1 0], 'MarkerSize', 15);
end

for ee = 1:length(effect_size_local)
    r = 1.9 + (2.1 - 1.9) .* rand(1, 1);  % Add jitter
    plot(r, effect_size_local(ee), '.', 'Color', [1 0 1], 'MarkerSize', 15);
end

ylabel('% Effect Size DCZ')
xticks([1 2])
xticklabels({'Branch', 'Spines'})
title('DCZ Effect on Calcium Activity by ROI Type')

%% Section 17: Load and Analyze Selectivity Data
% Extract trial-type selectivity from dff_types structure

[dff_types, looking] = DataSummarizer('matrix', 'dff_types', 'horz');

% Extract selectivity for spine and branch ROIs
[Sal.local] = iam_lazyV2(dff_types, 'Sal', 'spine', 'selec');
[Sal.branch] = iam_lazyV2(dff_types, 'Sal', 'branch', 'selec');

[DCZ.local] = iam_lazyV2(dff_types, 'DCZ', 'spine', 'selec');
[DCZ.branch] = iam_lazyV2(dff_types, 'DCZ', 'branch', 'selec');

% Baseline normalization
Sal.local = Sal.local - min(Sal.local, [], 1);
Sal.branch = Sal.branch - min(Sal.branch, [], 1);
DCZ.local = DCZ.local - min(DCZ.local, [], 1);
DCZ.branch = DCZ.branch - min(DCZ.branch, [], 1);

%% Section 18: Visualize Selectivity Traces

% Spine selectivity
figure()
hold on
stdshade(Sal.local', 0.3, 'k', x_df)
stdshade(DCZ.local', 0.3, 'c', x_df)
plot_epochs
title('Spine Selectivity: Saline vs DCZ')
ylabel('Selectivity (|Left - Right|)')
xlabel('Time (seconds)')

% Branch selectivity
figure()
hold on
stdshade(Sal.branch', 0.3, 'k', x_df)
stdshade(DCZ.branch', 0.3, 'c', x_df)
plot_epochs
title('Branch Selectivity: Saline vs DCZ')
ylabel('Selectivity (|Left - Right|)')
xlabel('Time (seconds)')
