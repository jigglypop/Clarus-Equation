function [type_sum_all,sig_all,cd_choice_all,cd_stim_all] = DeleteFullZero(type_sum_all,sig_all,cd_choice_all,cd_stim_all)
% Find columns with all zeros
for ii = 1:length(type_sum_all)
zero_columns{1,ii} = find(all(type_sum_all(ii).sel_types== 0, 1));
end

todelete=unique(cell2mat(zero_columns));

f=fieldnames(sig_all);
for jj = 1:length(f)

            sig_all.(f{jj,1})(:,todelete)=[];

end


f=fieldnames(type_sum_all);
% for ii = 1:length(type_sum_all)
    for jj = 1:length(f)
%         zz=1;
%         while zz <=size(type_sum_all(ii).(f{jj,1}),2)
%             if ~any(type_sum_all(ii).(f{jj,1})(:,zz))
                if ~strcmp(f{jj,1},'nu_sel')
                    for tt = 1:length(type_sum_all)
                    type_sum_all(tt).(f{jj,1})(:,todelete)=[];
                    end
                end
                
%             end
%             zz=zz+1;
%         end
    end
% end

f=fieldnames(cd_choice_all);
% for ii = 1:length(type_sum_all)
    for jj = 1:length(f)
%         zz=1;
%         while zz <=size(cd_choice_all(ii).(f{jj,1}),2)
%             if ~any(cd_choice_all(ii).(f{jj,1})(:,zz))
            for tt = 1:length(type_sum_all)
                cd_choice_all(tt).(f{jj,1})(:,todelete)=[];
                cd_stim_all(tt).(f{jj,1})(:,todelete)=[];
            end
%             end
%             zz=zz+1;
%         end
    end
% end
