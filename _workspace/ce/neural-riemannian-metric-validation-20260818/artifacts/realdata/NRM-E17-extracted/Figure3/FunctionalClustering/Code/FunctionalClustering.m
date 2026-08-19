%% Run this section for Spatial Analysis comparing Rule A to Rule B. For this load SpatialCorr_RuleARuleB.mat
summar_rule=Plot_Stats_spatialCorr(branch_noise_RuleB,branch_noise_RuleA,branch_signal_RuleB,branch_signal_RuleA);

%% Run this section for Spatial Analysis comparing Instruction to Choice synapses. For this load SpatialCorr_InstructionVSChoiceStats.mat
summar_cod=Plot_Stats_spatialCorr(inst_noise.all,resp_noise.all,inst_signal.all,resp_signal.all,inst_noise.nearneigh,resp_noise.nearneigh);

