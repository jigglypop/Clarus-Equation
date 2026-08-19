% Figure 5 - Transition Error Analysis
%
% This script analyzes how NDNF interneurons encode unexpected errors during
% rule transitions. Specifically, it examines selectivity during transition
% sessions when animals make errors due to applying the wrong rule.
%
% Transition types:
% - A→B: Switching from Rule A to Rule B
% - B→A: Switching from Rule B back to Rule A
%
%
% Associated publication: Maristany de las Casas et al. (2025)
% Figure 5: NDNF interneurons and transition error coding

%% Section 1: Combine A→B and B→A Transition Data
% Merge data from both transition directions to increase statistical power
% and test general transition error coding regardless of direction

% Get field names from the structures
lel = fieldnames(ConAB_Summ);

% Initialize combined structure with B→A data
ConSum = ConBA_Summ;

% Concatenate A→B and B→A data for each field
for ii = 1:length(lel)
    ConSum.(lel{ii,1}) = [ConAB_Summ.(lel{ii,1}), ConBA_Summ.(lel{ii,1})];
end

% Combine probability data (significance flags for different coding types)
lel = fieldnames(prob_startAB);
prob_start = prob_startAB;

for ii = 1:length(lel)
    prob_start.(lel{ii,1}) = [prob_startAB.(lel{ii,1}), prob_startBA.(lel{ii,1})];
end

% Combine selectivity data (which interneurons encode errors)
lel = fieldnames(selec_startAB);
selec_start = selec_startAB;

for ii = 1:length(lel)
    selec_start.(lel{ii,1}) = [selec_startAB.(lel{ii,1}), selec_startBA.(lel{ii,1})];
end

%% Section 2: Analyze Transition Error Coding
% trans_gcamp_general performs the main analysis:
% - Identifies interneurons that encode transition errors
% - Tests if error-related selectivity is directionally biased
% - Compares selectivity magnitude during initial vs response periods
%
% Inputs:
% - selec_start: Selectivity values for error trials
% - prob_start: Statistical significance flags
% - ConSum: Combined trial-averaged data from both transitions
% - trial_duration: Number of frames (180 frames for 6 seconds at 30 Hz)
%
% Outputs:
% - p_negpos: Tests if positive vs negative error selectivity differ
% - p_abs: Tests if selectivity increases during response period

trial_duration = 180;  % 6 seconds at 30 Hz sampling
[p_negpos, p_abs] = trans_gcamp_general(selec_start, prob_start, ConSum, trial_duration);
