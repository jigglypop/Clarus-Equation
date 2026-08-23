%% figure 2: pre-sleep and post-sleep item-level category-level RSA
close all;clc,clear;

%%%%%%%%%%%%% Pre-sleep learning RSA plot %%%%%%%%%%%%%%%%%%%%%%%%
suball_incld = 1:35; %
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
J_tstats = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#FAD9D9','#FCEFEF','#ffffff','#E7E7EE','#ECEDF7','#857FDF','#6A70BC','#4B4F8A'});

encode_epoch_limits_tf = [-1 6];
sr = 100;
task_phase = 'encode';
frex_bands = {2 4; 4 8; 9 12; 11 19; 25 40; 2 40};
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
encode_time_window = 500;%ms
encode_sample_step = 10; %10?1
encode_time_dura = [-250 5000]; %ms; time period for RSA analysis

% cond_names =  {
%     ['forgWI'];['forgWC'];['forgBC'];['forgBI'];1-4
%     ['remWI'];['remWC'];['remBC'];['remBI'];5-8
%
%     ['WINpre'];['WCNpre'];['BCNpre'];['BINpre'];9-12
%     ['WIpre'];['WCpre'];['BCpre'];['BIpre'];13-16
%
%     ['WINtmr'];['WCNtmr'];['BCNtmr'];['BINtmr'];17-20
%     ['WItmr'];['WCtmr'];['BCtmr'];['BItmr'];21-24
%
%     ['forgWINpre'];['forgWCNpre'];['forgBCNpre'];['forgBINpre']; 25-28
%     ['forgWIpre'];['forgWCpre'];['forgBCpre'];['forgBIpre'];29-32
%
%     ['remWINpre'];['remWCNpre'];['remBCNpre'];['remBINpre'];33-36
%     ['remWIpre'];['remWCpre'];['remBCpre'];['remBIpre'];37-40
%
%     ['forgWINtmr'];['forgWCNtmr'];['forgBCNtmr'];['forgBINtmr'];41-44
%     ['forgWItmr'];['forgWCtmr'];['forgBCtmr'];['forgBItmr'];45-48
%
%     ['remWINtmr'];['remWCNtmr'];['remBCNtmr'];['remBINtmr'];49-52
%     ['remWItmr'];['remWCtmr'];['remBCtmr'];['remBItmr'];53-56
%
%     ['WI'];['WC'];['BC'];['BI'];57-60
%
%     };



%%
%%%%%%%%%%%%% Pre-sleep learning RSA plot %%%%%%%%%%%%%%%%%%%%%%%%
%item-level representations for all items
ifreq = 6;
frex_band_name_tmp = frex_bands_name{1,ifreq};
RSA_Output_folder = ['RSA_proc'];
RSA_Output_subfolder = [task_phase '_tw' num2str(encode_time_window) '_step' num2str((1000/sr*encode_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'encode_corr_conds_final'],'encode_corr_suball');

%item-level for remembered item
cond_selected = [5 6 7];
cond_names = {'WI','WC','BC'};
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWI_remWC']);
cluster_size_tmp = [clusterinfo.pos_clusters.p];
cluster_idx = find(cluster_size_tmp < 0.05);
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
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = encode_corr_suball{subID, cond_selected(icond)};
    end
end

[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
figure;
set(gcf,'color','w');
set(gcf,'position',[0 0 1650 630]);
for il = 1:3
    subplot(2,5,il)
    imagesc(squeeze(mean(data_ttmp(:,il,:,:),1)));caxis([-0.1 0.2]);
    colormap('jet');
    colormap(customcolormap_preset('red-white-blue'));
    xticks([2 11 21 31 41 51]-1);
    set(gca,'xticklabel',[0 1 2 3 4 5]);
    yticks([2 11 21 31 41 51]-1);
    hold on;
    set(gca,'yticklabel',[0 1 2 3 4 5]);
    ylabel('Pre-sleep learning (s)')
    xlabel('Pre-sleep learning (s)');
    set(gca,'ticklength',[0.02 0.02]);
    title({cond_names{il};'Remember'});
    hold on;
    set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
    set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
    hold on;
    colorbar
    axis xy
end
hold on
subplot(2,5,4)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:),1)));caxis([-0.015 0.015]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
% hold on;
% plot([20 20],[1 size(stats_map,2)],'--','color',[128 128 128]./255,'linewidth',2.0);
% hold on;
% plot([1 size(stats_map,2)],[20 20],'--','color',[128 128 128]./255,'linewidth',2.0);
hold on;
set(gca,'yticklabel',[0 1 2 3 4 5]);
ylabel('Pre-sleep learning (s)')
xlabel('Pre-sleep learning (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Item-level (WI - WC)';'Remember'});
hold on;
set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold on;
circle_cluster(clusters_idx{1,1}(:,2),clusters_idx{1,1}(:,1),'k',1.5);
axis xy;
colorbar;
% h = colorbar;
% h.Position(3) = h.Position(3) - 0.002;
% h.Position(1) = h.Position(1) + 0.01;

clear stats_map h_map h p ci stats P_cluster_idx clusters_idx cluster_size_tmp

%category-level representations for remembered items
cond_selected = [6 7];
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWC_remBC'])
cluster_size_tmp = [clusterinfo.pos_clusters.p];
cluster_idx = find(cluster_size_tmp < 0.05);
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
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = encode_corr_suball{subID, cond_selected(icond)};
    end
end


[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
subplot(2,5,5)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:),1)));caxis([-0.015 0.015]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
set(gca,'yticklabel',[0 1 2 3 4 5]);
% hold on;
% plot([20 20],[1 size(stats_map,2)],'--','color',[128 128 128]./255,'linewidth',2.0);
% hold on;
% plot([1 size(stats_map,2)],[20 20],'--','color',[128 128 128]./255,'linewidth',2.0);
hold on;
ylabel('Pre-sleep learning (s)')
xlabel('Pre-sleep learning (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Category-level (WC - BC)';'Remember'});
hold on;
set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold on;
circle_cluster(clusters_idx{1,1}(:,2),clusters_idx{1,1}(:,1),'k',1.5);
axis xy;
colorbar;
clear stats_map h_map h p ci stats P_cluster_idx clusters_idx


%%%%%%%%%%%%% Post-sleep mental retrieval RSA plot %%%%%%%%%%%%%%%%%%%%%%%%
clc,clear
suball_incld = 1:35;
task_phase = 'closeeye';
closeeye_epoch_limits_tf = [-1 6];
sr = 100;
frex_bands = {2 4; 4 8; 9 12; 11 19; 25 40; 2 40};
frex_bands_name = {'delta','theta','alpha','sigma','gamma','allbands'};
normalization_method = 'percent_pre';%'percent_pre';%Ztrans_all
RSA_chan_type = 'global'; %'single': do RSA on single chan; 'global' do RSA on all chans
closeeye_time_window = 500;%ms%200
closeeye_sample_step = 10; %10?1
ifreq=6;
frex_band_name_tmp = frex_bands_name{1,ifreq};
RSA_Output_folder = ['RSA_proc'];
RSA_Output_subfolder = [task_phase '_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];

sub_selected =  setdiff(18:61,[43 50 55 58 59 21 24 40 46]); %rem:24 40 46;
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
J_tstats = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#FAD9D9','#FCEFEF','#ffffff','#E7E7EE','#ECEDF7','#857FDF','#6A70BC','#4B4F8A'});

load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'closeeye_corr_conds_final']);

%item-level for remembered item
cond_selected = [5 6 7];
cond_names = {'WI','WC','BC'};
load([pwd filesep RSA_Output_folder filesep RSA_Output_subfolder filesep 'cluster_diff_remWI_remWC'])

clear data_ttmp
for isub = 1:length(suball_incld)
    subID = suball_incld(isub);
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = closeeye_corr_suball{subID, cond_selected(icond)};
    end
end

for il = 1:3
    subplot(2,5,il+5)
    imagesc(squeeze(mean(data_ttmp(:,il,:,:),1)));caxis([0 0.04]);
    colormap('jet');
    colormap(customcolormap_preset('red-white-blue'));
    xticks([2 11 21 31 41 51]-1);
    set(gca,'xticklabel',[0 1 2 3 4 5]);
    yticks([2 11 21 31 41 51]-1);
    set(gca,'yticklabel',[0 1 2 3 4 5]);
    ylabel('Post-sleep retrieval (s)')
    xlabel('Post-sleep retrieval (s)');
    set(gca,'ticklength',[0.02 0.02]);
    title({cond_names{il};'Remember'});
    hold on;
    set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
    set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
    hold on;
    colorbar
    axis xy
end
hold on

[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
subplot(2,5,9)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:),1)));caxis([-0.005 0.005]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
set(gca,'yticklabel',[0 1 2 3 4 5]);
ylabel('Post-sleep retrieval (s)')
xlabel('Post-sleep retrieval (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Item-level (WI - WC)';'Remember'});
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
cluster_size_tmp = [clusterinfo.pos_clusters.p];
cluster_idx = find(cluster_size_tmp < 0.05);
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
    
    for icond = 1:length(cond_selected)
        data_ttmp(isub,icond,:,:) = closeeye_corr_suball{subID, cond_selected(icond)};
    end
end

[h p ci stats] = ttest(data_ttmp(:,1,:,:),data_ttmp(:,2,:,:));
h_map = squeeze(h);
p_map = squeeze(p);
stats_map = squeeze(stats.tstat);
subplot(2,5,10)
imagesc(squeeze(mean(data_ttmp(:,1,:,:)-data_ttmp(:,2,:,:),1)));caxis([-0.005 0.005]);
colormap(J_deep_contrast);
xticks([2 11 21 31 41 51]-1);
set(gca,'xticklabel',[0 1 2 3 4 5]);
yticks([2 11 21 31 41 51]-1);
set(gca,'yticklabel',[0 1 2 3 4 5]);
ylabel('Post-sleep retrieval (s)')
xlabel('Post-sleep retrieval (s)');
set(gca,'ticklength',[0.02 0.02]);
title({'Category-level (WC - BC)';'Remember'});
hold on;
set(gca,'fontsize',13,'fontname','Arial','linewidth',1.5);
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold on;

circle_cluster(clusters_idx{1,1}(:,2),clusters_idx{1,1}(:,1),'k',1.5);
hold on;
axis xy;
colorbar;
clear stats_map h_map h p ci stats P_cluster_idx clusters_idx