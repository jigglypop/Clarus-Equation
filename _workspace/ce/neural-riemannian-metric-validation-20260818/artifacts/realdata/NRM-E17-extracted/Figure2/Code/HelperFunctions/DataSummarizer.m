function [lul, looking] = DataSummarizer(out_type,var_name,VerOrHorz,looking)
%%
if ~exist('looking','var')
looking = uigetfile('MultiSelect', 'on','*.mat');
end
if strcmp(out_type,'struct') && isstruct(load(looking{1,1},var_name))
    
    for ii = 1:length(looking)
        ii
    a = load(looking{1,ii},var_name);
    fields=fieldnames(a.(var_name));
    name2{ii,1} = looking{1,ii};
        for tt = 1:length(a.(var_name))% OR size(a.(var_name),1)
            for jj=1:length(fields)
                if ii ==1
                lul(tt).(fields{jj,1})=[];
                end
                if strcmp(VerOrHorz,'horz')
                lul(tt).(fields{jj,1})=horzcat(lul(tt).(fields{jj,1}),a.(var_name)(tt).(fields{jj,1}));
                elseif strcmp(VerOrHorz,'vert')
                lul(tt).(fields{jj,1})=vertcat(lul(tt).(fields{jj,1}),a.(var_name)(tt).(fields{jj,1}));
                end
            end
        end
    end
end
%%
if strcmp(out_type,'matrix') && ismatrix(load(looking{1,1},var_name))
    lul=[];
    for ii = 1:length(looking)
    a = load(looking{1,ii},var_name);
    name2{ii,1} = looking{1,ii};
    if strcmp(VerOrHorz,'vert')
        lul=vertcat(lul,a.(var_name));
    elseif strcmp(VerOrHorz,'horz')
        lul=horzcat(lul,a.(var_name));
    end
    end
end
%%
if strcmp(out_type,'cellcat') %&& iscell(load(looking{1,1},var_name))
    a = load(looking{1,1},var_name);
    lul=cell(length(a.(var_name)),1);
    for ii = 1:length(looking)
    a = load(looking{1,ii},var_name);
    
    name2{ii,1} = looking{1,ii};
        for tt=1:length(a.(var_name))
        lul{tt,1}=horzcat(lul{tt,1},a.(var_name){tt,1});
        end
    end
end
end








