%% figure 5:REM and SWS power and memory representational transformation
clc,clear;close all
addpath(genpath([pwd filesep 'functions']))

suball_incld = 1:35;
sub_selected =  setdiff(suball_incld,[5]); %one participant with disconnected sleep EEG was further excluded
sub_selected_ID = [18 19 20 22 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 41 42 44 45 47 48 49 51 52 53 54 56 57 60 61];

cond_selected = [5 6];%rem item
cond_selected2 = [6 7];%rem cate

closeeye_time_window = 500;%ms%200
closeeye_sample_step = 10; %10?1
encode_time_window = 500;%ms
encode_sample_step = 10; %10?1
sr = 100;
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
ifreq = 6;
frex_band_name_tmp = frex_bands_name{1,ifreq};

PSD_folder = 'Sleep_PSD';
stage_name = 'REM';%N3;%REM
freq_band_name = 'theta';
if strncmp(freq_band_name,'theta',4)
    freq_sel = [4 7];
    color_rem = [0 0 255]./255.*0.8;
end

for isub = 1:length(sub_selected)
    subID = sub_selected_ID(isub);

    load([PSD_folder filesep 'psd_irasa_' stage_name '_' num2str(subID,'%02d') '_1-40Hz']);

    freq_idx = find(freq > freq_sel(1)-0.0001 & freq <= freq_sel(2));
    ichan = [1 3];

    REM_power(isub,:) = squeeze(mean(mean(sub_psd_oscillatory_clean(ichan,freq_idx),2),1))./squeeze(mean(sum(sub_psd_oscillatory_clean(ichan,:),2),1));%sub_psd_raw_clean
end

load([pwd filesep 'Sleep_staging_data\suball_neural_sleep_param.mat'])
SWS_ratio = suball_neural_sleep_rem_cluster(:,3);
REM_ratio = suball_neural_sleep_rem_cluster(:,2);

%define the RSA folder:
RSA_Input_folder = ['RSA_proc'];
ERS_Input_subfolder = [ 'EM_closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
EES_Input_subfolder = ['encode_tw' num2str(encode_time_window) '_step' num2str((1000/sr*encode_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
RRS_Input_subfolder = ['closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
load([pwd filesep RSA_Input_folder filesep ERS_Input_subfolder filesep 'EM_closeeye_corr_suball_final'],'EM_closeeye_corr_suball');
output_folder = 'Sleep_RSA_corr';
mkdir([pwd filesep output_folder]);

clear p_max_cluster clusters_idx P_cluster_idx p_max_cluster p_min_idx
load([pwd filesep RSA_Input_folder filesep RRS_Input_subfolder filesep 'closeeye_corr_conds_final'],'closeeye_corr_suball');

clear data_ttmp
for isub = 1:length(sub_selected)
    subidx = sub_selected(isub);

    for icond = 1:length(cond_selected)
        RRS_item_ttmp(isub,icond,:,:) = closeeye_corr_suball{subidx, cond_selected(icond)};
    end
end


for isub = 1:length(sub_selected)
    subidx = sub_selected(isub);

    for icond = 1:length(cond_selected2)
        RRS_cate_ttmp(isub,icond,:,:) = closeeye_corr_suball{subidx, cond_selected2(icond)};
    end
end

%get the individual difference in EES
load([pwd filesep RSA_Input_folder filesep EES_Input_subfolder filesep 'encode_corr_conds_final'],'encode_corr_suball');

clear data_ttmp
for isub = 1:length(sub_selected)
    subidx = sub_selected(isub);

    for icond = 1:length(cond_selected)
        EES_item_ttmp(isub,icond,:,:) = encode_corr_suball{subidx, cond_selected(icond)};
    end
end

clear data_ttmp
for isub = 1:length(sub_selected)
    subidx = sub_selected(isub);

    for icond = 1:length(cond_selected2)
        EES_cate_ttmp(isub,icond,:,:) = encode_corr_suball{subidx, cond_selected2(icond)};
    end
end
RRS_item_adjusted = squeeze(RRS_item_ttmp(:,1,:,:)-RRS_item_ttmp(:,2,:,:))-repmat(squeeze(mean(mean(EES_item_ttmp(:,1,:,:)-EES_item_ttmp(:,2,:,:),4),3)),[1 size(RRS_item_ttmp,3),size(RRS_item_ttmp,4)]);
RRS_cate_adjusted = squeeze(RRS_cate_ttmp(:,1,:,:)-RRS_cate_ttmp(:,2,:,:))-repmat(squeeze(mean(mean(EES_cate_ttmp(:,1,:,:)-EES_cate_ttmp(:,2,:,:),4),3)),[1 size(RRS_cate_ttmp,3),size(RRS_cate_ttmp,4)]);

load([pwd filesep output_folder filesep 'cluster_permute_test_RRS_item_rem2sws'],'p_max_cluster','p_min_cluster','clusters_idx','p_max_idx','p_min_idx','P_cluster_idx')

%line plot for the correlation
for icluster = 1:length(P_cluster_idx)
    cluster_ttmp = clusters_idx{P_cluster_idx(icluster),1};

    for il = 1:length(cluster_ttmp)
        RRS_item_cluster_tmp(:,il)=RRS_item_adjusted(:,cluster_ttmp(il,1),cluster_ttmp(il,2));
    end

    RRS_item_clusters(:,icluster) = squeeze(mean(RRS_item_cluster_tmp,2));
    clear RRS_item_cluster_tmp cluster_ttmp
end

%define the x variable in the entire analysis
x = REM_power.*REM_ratio;%REM_ratio
total_REM_theta_power = REM_power;

y = RRS_item_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

% Plot the data points
figure
set(gcf,'color','w')
set(gcf,'position',[0 0 1200 960]);
subplot(3,3,1)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Item-level strength change';'(Post minus Pre)'});
xlabel([{['Total ' freq_band_name ' power']};{'(REM sleep)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
ylim([-0.06 0.06])
clear y lm_cluster coef p x_pred y_pred y_CI


load([pwd filesep output_folder filesep 'cluster_permute_test_RRS_cate_rem2sws'],'p_max_cluster','p_min_cluster','clusters_idx','p_max_idx','p_min_idx','P_cluster_idx')

%line plot for the correlation
for icluster = 1:length(P_cluster_idx)
    cluster_ttmp = clusters_idx{P_cluster_idx(icluster),1};

    for il = 1:length(cluster_ttmp)
        RRS_cate_cluster_tmp(:,il)=RRS_cate_adjusted(:,cluster_ttmp(il,1),cluster_ttmp(il,2));
    end

    RRS_cate_clusters(:,icluster) = squeeze(mean(RRS_cate_cluster_tmp,2));
    clear RRS_cate_cluster_tmp cluster_ttmp
end

y = RRS_cate_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

subplot(3,3,2)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Category-level strength change';'(Post minus Pre)'});
xlabel([{['Total ' freq_band_name ' power']};{'(REM sleep)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
ylim([-0.03 0.03])
clear y lm_cluster coef p x_pred y_pred y_CI



%%%%%%%%%%%%%% REM sleep beta power %%%%%%%%%%%%%%%%%%%%
clear REM_power freq_sel color_rem x
freq_band_name = 'beta';
if strncmp(freq_band_name,'beta',4)
    freq_sel = [15 25];
    color_rem = [148 0 210]./255;
end

for isub = 1:length(sub_selected)
    subID = sub_selected_ID(isub);

    load([PSD_folder filesep 'psd_irasa_' stage_name '_' num2str(subID,'%02d') '_1-40Hz']);

    freq_idx = find(freq > freq_sel(1)-0.0001 & freq <= freq_sel(2));
    ichan = [1 3];

    REM_power(isub,:) = squeeze(mean(mean(sub_psd_oscillatory_clean(ichan,freq_idx),2),1))./squeeze(mean(sum(sub_psd_oscillatory_clean(ichan,:),2),1));%sub_psd_raw_clean
end

x = REM_power.*REM_ratio;
total_REM_beta_power = REM_power;

y = RRS_item_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

% Plot the data points

subplot(3,3,4)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Item-level strength change';'(Post minus Pre)'});
xlabel([{['Total ' freq_band_name ' power']};{'(REM sleep)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
ylim([-0.06 0.06])
hold off;
clear y lm_cluster coef p x_pred y_pred y_CI


y = RRS_cate_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

subplot(3,3,5)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Category-level strength change';'(Post minus Pre)'});
xlabel([{['Total ' freq_band_name ' power']};{'(REM sleep)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
ylim([-0.03 0.03])

clear y lm_cluster coef p x_pred y_pred y_CI

%SWS delta power and representational transformation
clear REM_power freq_sel color_rem x

PSD_folder = 'Sleep_PSD';
stage_name = 'N3';%N3;%REM
freq_band_name = 'SO';
if strncmp(freq_band_name,'SO',4)
    freq_sel = [1 1.25];
    color_rem = [0 204 204]./255.*0.8;
end

for isub = 1:length(sub_selected)
    subID = sub_selected_ID(isub);

    load([PSD_folder filesep 'psd_irasa_' stage_name '_' num2str(subID,'%02d') '_1-40Hz']);

    freq_idx = find(freq > freq_sel(1)-0.0001 & freq <= freq_sel(2));
    if strncmp(stage_name,'N3',2)
        ichan = [2];
    elseif strncmp(stage_name,'REM',2)
        ichan = [1 3];
    end

    SWS_power(isub,:) = squeeze(mean(mean(sub_psd_oscillatory_clean(ichan,freq_idx),2),1))./squeeze(mean(sum(sub_psd_oscillatory_clean(ichan,:),2),1));%sub_psd_raw_clean;sub_psd_oscillatory_clean
end

x = SWS_power.*SWS_ratio;
total_SWS_delta_power = SWS_power;

y = RRS_item_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

% Plot the data points

subplot(3,3,7)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Item-level strength change';'(Post minus Pre)'});
xlabel([{['Total SO-related power']};{'(SWS)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
ylim([-0.06 0.06])
hold off;
clear y lm_cluster coef p x_pred y_pred y_CI


y = RRS_cate_clusters(:,1);
lm_cluster = fitlm(x,y,'RobustOpts','on')

% Predictions and confidence intervals
x_pred = linspace(min(x), max(x), 100);
[y_pred, y_CI] = predict(lm_cluster, x_pred');

subplot(3,3,8)
scatter(x, y, 'MarkerFaceColor', color_rem,'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_rem, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_rem, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Category-level strength change';'(Post minus Pre)'});
xlabel([{['Total SO-related power']};{'(SWS)'}]);
set(gca,'linewidth',1.5,'fontsize',11.5,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
ylim([-0.03 0.03])

clear y lm_cluster coef p x_pred y_pred y_CI


