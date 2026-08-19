function [Z,B,sigs]= SortROIsByPeak(type_sum_all,sig_all,aligned,x_df,maxcol)

f=figure('Name','Sorted each session');
f.Units='centimeters';
f.Position = [0 10 40 6];hold on
for ii =1:length(type_sum_all)
[~,maxA] = max(type_sum_all(ii).sel_types, [], 1);
[dummy, index] = sort(maxA);
B    = type_sum_all(ii).sel_types(:, index);
subplot(1,length(type_sum_all),ii)
% colormap gray
colormap default
% s_ee = pcolor(x_df,1:length(type_sum_all(1).sel_types(1,:)),type_sum_all(1).sel_types');
s_ee = pcolor(x_df,1:length(type_sum_all(ii).sel_types(1,:)),B');

% s_ee.FaceColor = 'interp';
% s_ee.LineStyle = 'none';
s_ee.MeshStyle = 'none';
clim([0 maxcol])

% if ii ==5
% colorbar
% end
xlabel('Trial duration')
title('Sorted each session')
if ii==1
ylabel('ROI ID')
end
title(strcat('Session',num2str(ii)))
plot_epochs
end
%%
Z=[];
sigs=[];
% sigs=sig_all;
f=figure('Name','Sorted by session X');
f.Units='centimeters';
f.Position = [0 10 40 6];
hold on
for ii =1:length(type_sum_all)
[~,maxA] = max(type_sum_all(aligned).sel_types, [], 1);
[dummy, index] = sort(maxA);
Z(ii).srt=[type_sum_all(ii).nu_type1(:, index) ;type_sum_all(ii).nu_type2(:, index)] ;
sigs.Choice(ii,:)=sig_all.Choice(ii,index);
sigs.Stimuli(ii,:)=sig_all.Stimuli(ii,index);
sigs.Delay(ii,:)=sig_all.Delay(ii,index);

B    = type_sum_all(ii).sel_types(:, index);
subplot(1,length(type_sum_all),ii)
% colormap gray
colormap default
% s_ee = pcolor(x_df,1:length(type_sum_all(1).sel_types(1,:)),type_sum_all(1).sel_types');
s_ee = pcolor(x_df,1:length(type_sum_all(ii).sel_types(1,:)),B');

% s_ee.FaceColor = 'interp';
% s_ee.LineStyle = 'none';
s_ee.MeshStyle = 'none';
% if ii ==5
% colorbar
% end
xlabel('Trial duration (s)')
title('Sorted by Session X')

if ii==1
ylabel('ROI ID')
end
title(strcat('Session',num2str(ii)))
clim([0 maxcol])
plot_epochs
end
end