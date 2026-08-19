function [l,p]=plot_bar_errors(A,B,plotdots,ymin,ymax)

figure()
hold on
bar(1, nanmean(A))
bar(2, nanmean(B))
errorbar([1 2], [nanmean(A) nanmean(B)],[nanstd(A)/sqrt(length(A)),nanstd(B)/sqrt(length(B))])
if plotdots==1
for ii=1:length(A)
    plot(0.8 + (1.2-0.8) .* rand(1,1),A(ii),'k.')
end
for ii=1:length(B)
    plot(1.8 + (2.2-1.8) .* rand(1,1),B(ii),'k.')
end
end
if exist('ymin','var') && exist('ymax','var')
ylim([ymin ymax])
end
xlim([0.5 2.5])

[l,p]=ranksum(A, B);