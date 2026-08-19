function [output]= iam_lazy(tut,condi,actimode)

lul=arrayfun(@(x) x.(condi).(actimode), tut, 'UniformOutput', false);
for ii=1:length(lul)
spineSal{ii,1}=cell2mat(avgCell(lul{1,ii}));
end
output=cell2mat(spineSal');
end