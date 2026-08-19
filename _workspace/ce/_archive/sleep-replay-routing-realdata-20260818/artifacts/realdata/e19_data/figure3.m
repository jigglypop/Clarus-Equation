%% Figure 3: cross-session representation and its comparison with within-session representation
clc,clear;close all

%%
suball_incld = 1:35;
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});

sr = 100;
frex_bands = {2 4; 4 8; 9 12; 11 19; 25 40; 2 40};
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
closeeye_time_window = 500;%ms%200
closeeye_sample_step = 10; %10?1
ifreq = 6;
frex_band_name_tmp = frex_bands_name{1,ifreq};
RSA_Output_folder = ['RSA_proc'];
RSA_Output_subfolder = [ 'EM_closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'EM_closeeye_corr_suball_final'],'EM_closeeye_corr_suball');

%item-level for remembered item
cond_selected = [5 6];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWI_remWC'])

clear data_ttmp
for isub = 1:length(suball_incld)
    subidx = suball_incld(isub);

    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = EM_closeeye_corr_suball{subidx, cond_selected(icond)};
    end
end


J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWI_remWC'])

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = EM_closeeye_corr_suball{subID, cond_selected(icond)};
    end
end

[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
figure;
set(gcf,'color','w');
set(gcf,'position',[0 0 1300 680]);
subplot(2,3,1)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:))));caxis([-0.006 0.006]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
set(gca,'yticklabel',[0 1 2 3 4 5]);
ylabel('Pre-sleep learning (s)')
hold on;
% plot([1 size(stats_map,2)],[20 20],'--','color',[128 128 128]./255,'linewidth',2.0);
% hold on;
xlabel('Post-sleep retrieval (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Item-level (WI - WC)','Remember'});
hold on;
set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold on;
axis xy;
colorbar;
clear stats_map h_map h p ci stats P_cluster_idx clusters_idx

%category-level representations for remembered items
cond_selected = [6 7];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWC_remBC'])

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = EM_closeeye_corr_suball{subID, cond_selected(icond)};
    end
end

[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
subplot(2,3,2)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:))));caxis([-0.006 0.006]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
set(gca,'yticklabel',[0 1 2 3 4 5]);
ylabel('Pre-sleep learning (s)')
hold on;
% plot([1 size(stats_map,2)],[20 20],'--','color',[128 128 128]./255,'linewidth',2.0);
% hold on;
xlabel('Post-sleep retrieval (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Category-level (WC - BC)','Remember'});

hold on;
set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold on;

axis xy;
colorbar;
hold on

%%%%%%%%%% Pre-Pre Pre-Post Post-Post cluster-based based comparison %%%%%%%%%%%%%%%%%
%category-level
clc,clear;
suball_incld = 1:35;
cond_selected = [6 7];
RSA_Input_folder = ['RSA_proc'];
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
sr = 100;
closeeye_epoch_limits_tf = [-1 6];
encode_epoch_limits_tf = [-1 6];
encode_time_window = 500;%ms
encode_sample_step = 10; %10?1
encode_time_dura = [-250 5000]; %ms; time period for RSA analysis
closeeye_time_window = 500;%ms%200
closeeye_sample_step = 10; %10?1

sr = 100;
frex_bands = {2 4; 4 8; 9 12; 11 19; 25 40; 2 40};
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
closeeye_time_window = 500;%ms%200
closeeye_sample_step = 10; %10?1
ifreq = 6;
frex_band_name_tmp = frex_bands_name{1,ifreq};
RSA_Output_folder = ['RSA_proc'];
RSA_Output_subfolder = [ 'EM_closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'EM_closeeye_corr_suball_final'],'EM_closeeye_corr_suball');

%category-level representations for remembered items
cond_selected = [6 7];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWC_remBC'])

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = EM_closeeye_corr_suball{subID, cond_selected(icond)};
    end
end

rem_encode_cate_clusters_idx = [1:11 25:48];
rem_cate_retrieval_cluster_idx = [1:30];
ERS_cate_sel = squeeze(mean(mean(data_ttmp(:,:,rem_encode_cate_clusters_idx,rem_cate_retrieval_cluster_idx),4),3));
[h p ci statas] = ttest(ERS_cate_sel(:,1),ERS_cate_sel(:,2));

cond_selected2 = [6 7];%rem cate
RSA_Input_folder = ['RSA_proc'];
EES_Input_subfolder = ['encode_tw500_step100_freqchan_allbands_percent_pre_2D_fill2'];
RRS_Input_subfolder = ['closeeye_tw500_step100_freqchan_allbands_percent_pre_2D_fill2'];
load([pwd filesep RSA_Input_folder filesep EES_Input_subfolder filesep 'encode_corr_conds_final']);
load([pwd filesep RSA_Input_folder filesep EES_Input_subfolder filesep 'cluster_diff_remWC_remBC'])
cluster_size_tmp = [clusterinfo.pos_clusters.clusterstat];
cluster_idx = find(cluster_size_tmp == max(cluster_size_tmp));
h_map = double(clusterinfo.pos_clusters(cluster_idx).inds);

clear L num cluster_value clusters_idx cluster_idx
[L,num]=bwlabeln(h_map,8);
cluster_value = unique(L);
cluster_value = setdiff(cluster_value,0);
if length(cluster_value)>0
    for ic = 1:length(cluster_value)
        [i,j] = find(L == cluster_value(ic));
        clusters_idx{ic,1} = [i,j];
        clear i j
    end
end

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected2)
        EES_data_ttmp(isub,icond,:,:) = encode_corr_suball{subID, cond_selected2(icond)};
    end
end


cluster_ttmp = clusters_idx{1,1};

for il = 1:length(cluster_ttmp)
    EES_cluster_tmp(:,:,il)=EES_data_ttmp(:,:,cluster_ttmp(il,1),cluster_ttmp(il,2));
end

EES_cate_cluster = squeeze(mean(EES_cluster_tmp,3));
clear EES_cluster_tmp cluster_ttmp



EES_cate_cluster_diff = squeeze(EES_cate_cluster(:,1,:)-EES_cate_cluster(:,2,:));
clear EES_data_ttmp
[h p ci stats] = ttest(EES_cate_cluster_diff,ERS_cate_sel(:,1)-ERS_cate_sel(:,2))

%get the RRS mean across all time windows
clear p_max_cluster clusters_idx P_cluster_idx p_max_cluster p_min_idx
load([pwd filesep RSA_Input_folder filesep RRS_Input_subfolder filesep 'closeeye_corr_conds_final']);
load([pwd filesep RSA_Input_folder filesep RRS_Input_subfolder filesep 'cluster_diff_remWC_remBC']);
cluster_size_tmp = [clusterinfo.pos_clusters.clusterstat];
cluster_idx = find(cluster_size_tmp == max(cluster_size_tmp));
h_map = double(clusterinfo.pos_clusters(cluster_idx).inds);

clear L num cluster_value clusters_idx cluster_idx
[L,num]=bwlabeln(h_map,8);
cluster_value = unique(L);
cluster_value = setdiff(cluster_value,0);
if length(cluster_value)>0
    for ic = 1:length(cluster_value)
        [i,j] = find(L == cluster_value(ic));
        clusters_idx{ic,1} = [i,j];
        clear i j
    end
end

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected2)
        RRS_data_ttmp(isub,icond,:,:) = closeeye_corr_suball{subID, cond_selected2(icond)};
    end
end

cluster_ttmp = clusters_idx{1,1};

for il = 1:length(cluster_ttmp)
    RRS_cluster_tmp(:,:,il)=RRS_data_ttmp(:,:,cluster_ttmp(il,1),cluster_ttmp(il,2));
end

RRS_cate_cluster = squeeze(mean(RRS_cluster_tmp,3));
clear RRS_cluster_tmp cluster_ttmp

RRS_cate_cluster_diff = squeeze(RRS_cate_cluster(:,1,:)-RRS_cate_cluster(:,2,:));
clear RRS_data_ttmp
[h p ci stats] = ttest(RRS_cate_cluster_diff,ERS_cate_sel(:,1)-ERS_cate_sel(:,2))

EES_ERS_RRS_cate = [EES_cate_cluster_diff ERS_cate_sel(:,1)-ERS_cate_sel(:,2) RRS_cate_cluster_diff];
% EES_ERS_RRS_color = [244 164 96; 210 105 30; 160 82 45]./255;
EES_ERS_RRS_color = [223 160 20; 50 100 0; 200 100 0]./255;

%EES, ERS, RRS plot

% Sample data
data = EES_ERS_RRS_cate;

% Create subplot
figure;
set(gcf,'color','w');
set(gcf,'position',[0 0 420 320]);

% Create a boxplot
h = boxplot(data, 'Labels', {'Pre-Pre', 'Pre-Post', 'Post-Post'}, 'Symbol', '');

% Customize the boxplot
colors = EES_ERS_RRS_color; % Define your colors here

% Set the color and width of the boxplot lines
set(h, 'LineWidth', 1.5);

% Customize each box
for i = 1:size(data, 2)
    % Set the color of the box edges
    set(h(:, i), 'Color', colors(i, :));
    
    % Fill the boxes with color
    patch(get(h(5, i), 'XData'), get(h(5, i), 'YData'), colors(i, :), 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end

% Customize the plot
ylabel({'Category-level'; 'representations'});
set(gca, 'FontSize', 13, 'FontName', 'Arial', 'ticklength', [0.02 0.01], 'linewidth', 1.5);
xtickangle(45);
ylim([-0.04 0.1]);
title('');
box off;

hold on
for isub = 1:length(data)
    x_adjusted = [1.3 1.7];
    for il = 1:2
        % Make dots more transparent
        scatter(x_adjusted(il), data(isub, il), 20, ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceColor', EES_ERS_RRS_color(il, :), ...
                'MarkerFaceAlpha', 0.5); % Set transparency
        hold on;
    end
    % Lighten the line color
    plot(x_adjusted, data(isub, 1:2), 'color', [180 180 180] / 255, 'LineWidth', 0.4,'linestyle','-'); % Lighter color
    hold on;
end

hold on;
for isub = 1:length(data)
    x_adjusted = [2.3 2.7];
    for il = 1:2
        % Make dots more transparent
        scatter(x_adjusted(il), data(isub, il+1), 20, ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceColor', EES_ERS_RRS_color(il+1, :), ...
                'MarkerFaceAlpha', 0.8); % Set transparency
        hold on;
    end
    % Lighten the line color
    plot(x_adjusted, data(isub, 2:3), 'color', [180 180 180] / 255, 'LineWidth', 0.4,'linestyle','-'); % Lighter color
    hold on;
end

hold off;
clear stats_map h_map h p ci stats P_cluster_idx clusters_idx