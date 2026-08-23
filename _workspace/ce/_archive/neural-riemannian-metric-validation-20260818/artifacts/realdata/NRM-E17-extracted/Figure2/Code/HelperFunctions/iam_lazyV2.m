function [output]= iam_lazyV2(tut,condi,spORb,actimode)

lul=arrayfun(@(x) x.(condi).(spORb).(actimode), tut, 'UniformOutput', false);
% for ii=1:length(lul)
spineSal{1,1}=cell2mat(lul);
% end
output=cell2mat(spineSal');
end