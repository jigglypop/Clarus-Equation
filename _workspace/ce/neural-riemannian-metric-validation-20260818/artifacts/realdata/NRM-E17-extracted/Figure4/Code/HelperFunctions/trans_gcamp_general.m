function [p_negpos,p_abs]=trans_gcamp_general(selec_start,prob_start,ConSum,trial_duration)
% Step 1: Find the indices of columns with all zeros in 'selec' field of ConSumm
zeroColumns = all(ConSum.selec == 0, 1);
x_df=linspace(-3,3,trial_duration);

% Extract the indices of those columns
indicesToRemove = find(zeroColumns);

% Step 2: Remove the zero columns from all fields of ConSumm
fieldNames = fieldnames(ConSum);
for i = 1:numel(fieldNames)
    ConSum.(fieldNames{i})(:, indicesToRemove) = [];
end

% Step 3: Remove the elements at these indices from prob_start and selec_start
fieldNamesProb = fieldnames(prob_start);
for i = 1:numel(fieldNamesProb)
    prob_start.(fieldNamesProb{i})(indicesToRemove) = [];
end

fieldNamesselec = fieldnames(selec_start);
for i = 1:numel(fieldNamesselec)
    selec_start.(fieldNamesselec{i})(indicesToRemove) = [];
end

p_thresh=0.05;
err_resp=selec_start.err(prob_start.resp<p_thresh);%length(find(prob_start.resp<p_thresh&prob_start.err<p_thresh));
err_stim=selec_start.err(prob_start.stim<p_thresh);%length(find(pieA2_Summ.resp<p_thresh&pieA2_Summ.stim>p_thresh));
err_none=selec_start.err(prob_start.stim>p_thresh&prob_start.resp>p_thresh);

err_Lresp=selec_start.err(selec_start.err'>0);
err_Rresp=selec_start.err(selec_start.err'<0);


ovl=min([length(err_Rresp) length(err_Lresp)]);

figure()
hold on

violinplot([err_Lresp(1:ovl)' -err_Rresp(1:ovl)'],{'Positive','Negative'},'ShowMean',true,'ShowData',false)
ylim([0 1])
xlim([0.5 2.5])

[p_negpos,~]=ranksum(-err_Rresp,err_Lresp);
%%
idx_start=find(x_df>0,1,'first');
idx_end=find(x_df>2,1,'first');
[~,peak_sel_sort]=sort(mean(ConSum.selec(idx_start:idx_end,:)));
sort_sel=ConSum.selec(:,peak_sel_sort);

    figure();
colormap default
s_ee = pcolor(x_df,1:size(sort_sel,2),sort_sel');
s_ee.LineStyle = 'none';
s_ee.MeshStyle = 'row';
colorbar
plot_epochs() 
clim([-1 1])
xlim([-2.95 2.95])



   figure()
    hold on
    stdshade(ConSum.selecabs',0.3,'k',x_df)

   figure()
    hold on
    stdshade(ConSum.selecabs(:,selec_start.err'>0)'-0.05,0.3,'y',x_df)
    stdshade(-ConSum.selecabs(:,selec_start.err'<0)'+0.05,0.3,'b',x_df)

    plot_epochs
    ylim([-0.1 0.15])
    xlim([-2.9 2.9])
    idx_initial=find(x_df>-1.8,1,'first'):find(x_df>-0.2,1,'first');
    nu_init=mean(ConSum.selecabs(idx_initial,:));
    nu_resp=mean(ConSum.selecabs(idx_start:idx_end,:));
    [p_abs,~]=ranksum(nu_init,nu_resp);
end
