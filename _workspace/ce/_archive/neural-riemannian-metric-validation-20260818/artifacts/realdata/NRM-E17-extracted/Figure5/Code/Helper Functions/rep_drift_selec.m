function [p_AB p_AA2 p_A2B p_friedman p_anova, kendall_mat]=rep_drift_selec(Z,coding)

Z_cod=Z;
for ii =1:length(Z)
Z_cod(ii).srt=Z(ii).srt(:,coding);
end
v = 1:length(Z);
C = nchoosek(v,2);
% kendall_mat=NaN(size(theta_code,1),size(theta_code,1),size(theta_code,2));
kendall_mat=NaN(length(Z_cod)+1,length(Z_cod)+1,size(Z_cod(1).srt,2));
for kk =1:size(Z_cod(1).srt,2)%size(type_sum_all(1).sel_types,2)
    
    for ii = 1:size(C,1)
        
        [kendall_mat(C(ii,1),C(ii,2),kk),~] = corr(Z_cod(C(ii,1)).srt(:,kk),Z_cod(C(ii,2)).srt(:,kk),'type','Pearson');
        [kendall_mat(C(ii,2),C(ii,1),kk),~] = corr(Z_cod(C(ii,1)).srt(:,kk),Z_cod(C(ii,2)).srt(:,kk),'type','Pearson');
    end
%     figure();
%     colormap default
%     s_ee = pcolor(1-0.5:size(kendall_mat,1)-0.5,1-0.5:size(kendall_mat,1)-0.5,kendall_mat(:,:,kk));
% %     s_ee.FaceColor = 'interp';
%     s_ee.LineStyle = 'none';
%     s_ee.MeshStyle = 'row'; 
%     colorbar
end
    figure();
    colormap default
    s_ee = pcolor(1-0.5:size(kendall_mat,1)-0.5,1-0.5:size(kendall_mat,1)-0.5,nanmean(kendall_mat,3));
%     s_ee.FaceColor = 'interp';
    s_ee.LineStyle = 'none';
    s_ee.MeshStyle = 'row';
    colorbar
    xlabel('Sessions')
    ylabel('Sessions')
%     clim([0.4 0.6])

    AAcomp=[1 2;1 3;2 3];
    ABcomp=[1 4;4 3;2 4];
    AA2comp=[1 5;5 3;2 5];
    
    AAcorr=zeros(size(AAcomp,1),size(kendall_mat,3));
    ABcorr=zeros(size(AAcomp,1),size(kendall_mat,3));
    AA2corr=zeros(size(AAcomp,1),size(kendall_mat,3));

    for tt=1:size(AAcomp,1)
        AAcorr(tt,:)=kendall_mat(AAcomp(tt,1),AAcomp(tt,2),:);
        ABcorr(tt,:)=kendall_mat(ABcomp(tt,1),ABcomp(tt,2),:);
        AA2corr(tt,:)=kendall_mat(AA2comp(tt,1),AA2comp(tt,2),:);
    end
    AAcorr=reshape(AAcorr,size(AAcorr,2)*3,1);
    ABcorr=reshape(ABcorr,size(ABcorr,2)*3,1);
    AA2corr=reshape(AA2corr,size(AA2corr,2)*3,1);
%     y=[nanmean(AAcorr)' nanmean(ABcorr)' nanmean(AA2corr)'];
    y=[AAcorr ABcorr AA2corr];

    y(any(isnan(y), 2), :) = [];
    p_friedman = friedman(y,1);
    hold on
    ylabel('Coding Correlation')

    p_anova=anova1(y);
    hold on
    ylabel('Coding Correlation')


   p_AA2=signrank(y(:,1),y(:,3))
   p_A2B=signrank(y(:,2),y(:,3))
   p_AB=signrank(y(:,1),y(:,2))
end
