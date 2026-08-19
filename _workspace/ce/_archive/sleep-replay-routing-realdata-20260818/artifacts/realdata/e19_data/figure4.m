%% figure 4: interactive role of REM and SWS duration in representational transformation

clc,clear;close all
addpath(genpath([pwd filesep 'functions']))
suball_incld = 1:35;
sub_selected =  setdiff(suball_incld,[5]); %one participant with disconnected sleep EEG was further excluded

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

%get the ERS froml the 3-way interaction cluster
RSA_Input_folder = ['RSA_proc'];
ERS_Input_subfolder = [ 'EM_closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
EES_Input_subfolder = ['encode_tw' num2str(encode_time_window) '_step' num2str((1000/sr*encode_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
RRS_Input_subfolder = ['closeeye_tw' num2str(closeeye_time_window) '_step' num2str((1000/sr*closeeye_sample_step)) '_freqchan_'  frex_band_name_tmp '_' normalization_method '_2D_fill2'];
load([pwd filesep RSA_Input_folder filesep ERS_Input_subfolder filesep 'EM_closeeye_corr_suball_final'],'EM_closeeye_corr_suball');
output_folder = 'Sleep_RSA_corr';
mkdir([pwd filesep output_folder]);

load([pwd filesep 'Sleep_staging_data\suball_neural_sleep_param.mat'])
SWS_ratio = suball_neural_sleep_rem_cluster(:,3);
REM_ratio = suball_neural_sleep_rem_cluster(:,2);

% SWS*REM and item-level & category-level strength correlation
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


for il1 = 1:size(RRS_item_adjusted,2)
    for il2 = 1:size(RRS_item_adjusted,3)
        [coef,p] = corr(squeeze(RRS_item_adjusted(:,il1,il2)),REM_ratio.*SWS_ratio,'type','spearman');%suball_slope_rem
        RRS_item_rembysws_corr_coef(il1,il2) = coef;
        RRS_item_rem_corr_p(il1,il2) = p;
        if p <= 0.05
            RRS_item_rem_corr_h(il1,il2) = 1;
        else
            RRS_item_rem_corr_h(il1,il2) = 0;
        end
        clear coef p;
    end
end

figure;
set(gcf,'color','w');
set(gcf,'position',[0 0 1750 1020]);
subplot(3,4,1)
imagesc(RRS_item_rembysws_corr_coef);axis xy;colormap('jet');caxis([-0.4 0.4])
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
colormap(J_deep_contrast);
xticks([1 11 21 31 41]);
set(gca,'xticklabel',[0 1 2 3 4]);
yticks([1 11 21 31 41]);
set(gca,'yticklabel',[0 1 2 3 4]);
ylabel('Post-sleep retrieval (s)');
xlabel('Post-sleep retrieval (s)');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
colorbar('Ticks',[-0.4 0 0.4]);
hold on;
title([{'Item-level strength change '}; ...
    {'& SWS*REM correlation'}],'fontsize',14);


for il1 = 1:size(RRS_cate_adjusted,2)
    for il2 = 1:size(RRS_cate_adjusted,3)
        [coef,p] = corr(squeeze(RRS_cate_adjusted(:,il1,il2)),REM_ratio.*SWS_ratio,'type','spearman');
        RRS_cate_rembysws_corr_coef(il1,il2) = coef;
        RRS_cate_rem_corr_p(il1,il2) = p;
        if p <= 0.05
            RRS_cate_rem_corr_h(il1,il2) = 1;
        else
            RRS_cate_rem_corr_h(il1,il2) = 0;
        end
        clear coef p;
    end
end

subplot(3,4,5)
imagesc(RRS_cate_rembysws_corr_coef);axis xy;colormap('jet');caxis([-0.4 0.4])
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
colormap(J_deep_contrast);
xticks([1 11 21 31 41]);
set(gca,'xticklabel',[0 1 2 3 4]);
yticks([1 11 21 31 41]);
set(gca,'yticklabel',[0 1 2 3 4]);
ylabel('Post-sleep retrieval (s)');
xlabel('Post-sleep retrieval (s)');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
% title({'Correlation:'; '(Cate-level RRS & REM/SWS ratio)'});
colorbar('Ticks',[-0.4 0 0.4]);
hold on;
title([{'Category-level strength change'}; ...
    {'& SWS*REM correlation'}],'fontsize',14);
hold on;


%REM./SWS item-level and category-level strength change
for il1 = 1:size(RRS_item_adjusted,2)
    for il2 = 1:size(RRS_item_adjusted,3)
        [coef,p] = corr(squeeze(RRS_item_adjusted(:,il1,il2)),REM_ratio./SWS_ratio,'type','spearman');%suball_slope_rem
        RRS_item_rem2sws_corr_coef(il1,il2) = coef;
        RRS_item_rem_corr_p(il1,il2) = p;
        if p <= 0.05
            RRS_item_rem_corr_h(il1,il2) = 1;
        else
            RRS_item_rem_corr_h(il1,il2) = 0;
        end
        clear coef p;
    end
end
load([pwd filesep output_folder filesep 'cluster_permute_test_RRS_item_rem2sws'],'p_max_cluster','p_min_cluster','clusters_idx','p_max_idx','p_min_idx','P_cluster_idx')

subplot(3,4,2)
imagesc(RRS_item_rem2sws_corr_coef);axis xy;colormap('jet');caxis([-0.4 0.4])
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
colormap(J_deep_contrast);
xticks([1 11 21 31 41]);
set(gca,'xticklabel',[0 1 2 3 4]);
yticks([1 11 21 31 41]);
set(gca,'yticklabel',[0 1 2 3 4]);
ylabel('Post-sleep retrieval (s)');
xlabel('Post-sleep retrieval (s)');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
colorbar('Ticks',[-0.4 0 0.4]);
hold on;
for il = 1:length(P_cluster_idx)
    circle_cluster(clusters_idx{P_cluster_idx(il),1}(:,2),clusters_idx{P_cluster_idx(il),1}(:,1),'k',1.5);
    hold on;
end
title([{'Item-level strength change '}; ...
    {'& REM/SWS correlation'}],'fontsize',14);


%line plot for the correlation
for icluster = 1:length(P_cluster_idx)
    cluster_ttmp = clusters_idx{P_cluster_idx(icluster),1};
    
    for il = 1:length(cluster_ttmp)
        RRS_item_cluster_tmp(:,il)=RRS_item_adjusted(:,cluster_ttmp(il,1),cluster_ttmp(il,2));
    end
    
    RRS_item_clusters(:,icluster) = squeeze(mean(RRS_item_cluster_tmp,2));
    clear RRS_item_cluster_tmp cluster_ttmp
end

x = REM_ratio./SWS_ratio;
y = RRS_item_clusters(:,1);
% y = squeeze(mean(mean(RRS_item_adjusted,3),2));
[coef,p] = corr(x,y,'type','spearman')
x_pred = linspace(min(x), max(x), 100);
lm_cluster = fitlm(x,y,'RobustOpts','on')
[y_pred, y_CI] = predict(lm_cluster, x_pred');
color_rem_scatter = [0 0 255]./255;
color_rem = [0 0 255]./255;
color_scatter = [60 60 60]./255;
% Plot the data points
subplot(3,4,3)
scatter(x, y, 'MarkerFaceColor', color_scatter, 'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_scatter, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_scatter, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel({'Item-level strength change';'(Post minus Pre)'});
xlabel('REM/SWS');
xlim([0 1.6]);
% title('Linear Regression with Standard Deviation Shaded Area');
% Show legend
% legend('Data', 'Regression Line', 'Confidence Interval', 'Location', 'northwest');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
clear x y lm_cluster coef p x_pred y_pred y_CI p_max_cluster p_min_cluster clusters_idx p_max_idx p_min_idx P_cluster_idx


for il1 = 1:size(RRS_cate_adjusted,2)
    for il2 = 1:size(RRS_cate_adjusted,3)
        [coef,p] = corr(squeeze(RRS_cate_adjusted(:,il1,il2)),REM_ratio./SWS_ratio,'type','spearman');
        RRS_cate_rem2sws_corr_coef(il1,il2) = coef;
        RRS_cate_rem_corr_p(il1,il2) = p;
        if p <= 0.05
            RRS_cate_rem_corr_h(il1,il2) = 1;
        else
            RRS_cate_rem_corr_h(il1,il2) = 0;
        end
        clear coef p;
    end
end
load([pwd filesep output_folder filesep 'cluster_permute_test_RRS_cate_rem2sws'],'p_max_cluster','p_min_cluster','clusters_idx','p_max_idx','p_min_idx','P_cluster_idx')

subplot(3,4,6)
imagesc(RRS_cate_rem2sws_corr_coef);axis xy;colormap('jet');caxis([-0.4 0.4])
J_deep_contrast = customcolormap(linspace(0,1,11), {'#68011D','#BA4949','#E95B5B','#F99999','#F0D3D3','#ffffff','#F2F2F2','#CCCCCC','#A5A5A5','#7F7F7F','#595959'});
colormap(J_deep_contrast);
xticks([1 11 21 31 41]);
set(gca,'xticklabel',[0 1 2 3 4]);
yticks([1 11 21 31 41]);
set(gca,'yticklabel',[0 1 2 3 4]);
ylabel('Post-sleep retrieval (s)');
xlabel('Post-sleep retrieval (s)');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
% title({'Correlation:'; '(Cate-level RRS & REM/SWS ratio)'});
colorbar('Ticks',[-0.4 0 0.4]);
hold on;
for il = 1:length(P_cluster_idx)
    circle_cluster(clusters_idx{P_cluster_idx(il),1}(:,2),clusters_idx{P_cluster_idx(il),1}(:,1),'k',1.5);
    hold on;
end
title([{'Category-level strength change'}; ...
    {'& REM/SWS correlation'}],'fontsize',14);

%line plot for the correlation
for icluster = 1:length(P_cluster_idx)
    cluster_ttmp = clusters_idx{P_cluster_idx(icluster),1};
    
    for il = 1:length(cluster_ttmp)
        RRS_cate_cluster_tmp(:,il)=RRS_cate_adjusted(:,cluster_ttmp(il,1),cluster_ttmp(il,2));
    end
    
    RRS_cate_clusters(:,icluster) = squeeze(mean(RRS_cate_cluster_tmp,2));
    clear RRS_cate_cluster_tmp cluster_ttmp
end

clear x y lm_cluster coef p x_pred y_pred y_CI p_max_cluster p_min_cluster clusters_idx p_max_idx p_min_idx P_cluster_idx

x = REM_ratio./SWS_ratio;
y = RRS_cate_clusters(:,1);
% y = squeeze(mean(mean(RRS_cate_clusters,3),2));

[coef,p] = corr(x,y,'type','spearman')
x_pred = linspace(min(x), max(x), 100);
lm_cluster = fitlm(x,y,'RobustOpts','on')
[y_pred, y_CI] = predict(lm_cluster, x_pred');
color_rem_scatter = [0 0 255]./255;
color_rem = [0 0 255]./255;
color_scatter = [60 60 60]./255;

% Plot the data points
subplot(3,4,7)
scatter(x, y, 'MarkerFaceColor', color_scatter, 'MarkerEdgeColor', 'none');
hold on;
% Plot the regression line
plot(x_pred, y_pred, 'color',color_scatter, 'LineWidth', 2);
% Plot the shaded area for confidence intervals
fill([x_pred, fliplr(x_pred)], [y_CI(:,1)', fliplr(y_CI(:,2)')], color_scatter, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Add labels and title
ylabel( {'Category-level strength change' ;'(Post minus Pre)'});
xlabel('REM/SWS');
xlim([0 1.6]);
% title('Linear Regression with Standard Deviation Shaded Area');
% Show legend
% legend('Data', 'Regression Line', 'Confidence Interval', 'Location', 'northwest');
set(gca,'linewidth',1.5,'fontsize',13,'fontname','arial','ticklength',[0.01 0.02]);
hold off;
clear x y lm_cluster coef p x_pred y_pred y_CI p_max_cluster p_min_cluster clusters_idx p_max_idx p_min_idx P_cluster_idx