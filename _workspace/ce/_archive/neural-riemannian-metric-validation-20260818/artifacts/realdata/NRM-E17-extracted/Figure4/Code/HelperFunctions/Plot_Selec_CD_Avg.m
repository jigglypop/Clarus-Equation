function Plot_Selec_CD_Avg(type_sum_all,cd_choice_all,cd_stim_all,x_df,ylim_avg,ylim_sel,ylim_cd)
% for uu=1:size(type_sum_all(1).nu_type1,2)
f=figure();
%     f.Units='centimeters';
%     f.Position = [0 0 16 10]; 
%     f.PaperUnits='centimeters';
%     f.PaperPosition=[0 0 16 10];
    for kk=1:length(type_sum_all)
    subplot(4,length(type_sum_all),kk)
    hold on
%         plot(x_df,type_sum_all(kk).sel_types(:,uu),'Color',[0.5 0.5 0.5],'LineWidth',2)
%         plot(x_df,nanmean(type_sum_all(kk).nu_type1'),'r','LineWidth',2)
%         plot(x_df,nanmean(type_sum_all(kk).nu_type2'),'b','LineWidth',2)
%         if kk==4
%             stdshade(type_sum_all(kk).nu_type1',0.2,'m',x_df);
%         else
            stdshade(type_sum_all(kk).nu_type1',0.2,'r',x_df);
%         end     
        stdshade(type_sum_all(kk).nu_type2',0.2,'b',x_df);
        title(strcat('Stimulus Task',num2str(kk)))
        plot_epochs();
        ylim([0 ylim_avg]);
        xlim([-2.9 2.99])
        xticklabels({})
        if kk~=1
            yticklabels({})
        end
%         yline(0,'k');
        if kk==1
            ylabel('Events/Sec')
        end
    end

    for kk=1:length(type_sum_all)
    subplot(4,length(type_sum_all),kk+length(type_sum_all))
    hold on
%         plot(x_df,nanmean(type_sum_all(kk).sel_types'),'Color',[0.5 0.5 0.5],'LineWidth',2)
        stdshade(type_sum_all(kk).sel_types',0.2,'k',x_df);
%         plot(x_df,nanmean(type_sum_all(kk).nu_type1'),'r','LineWidth',2)
%         plot(x_df,nanmean(type_sum_all(kk).nu_type2'),'b','LineWidth',2)
        
%         title(strcat('Stimulus Task',num2str(kk)))
        plot_epochs();
        ylim([0 ylim_sel]);
        xlim([-2.9 2.99]);
        xticklabels({})
        if kk~=1
            yticklabels({})
        end
%         yline(0,'k');
        if kk==1
            ylabel('Selectivity')
        end
    end

    for kk=1:length(type_sum_all)
    subplot(4,length(type_sum_all),kk+length(type_sum_all)*2)
    hold on
%         plot(x_df,type_sum_all(kk).sel_types(:,uu),'Color',[0.5 0.5 0.5],'LineWidth',2)
        stdshade(cd_stim_all(kk).Lcd_proj',0.2,'b',x_df);
        if kk==4
            stdshade(cd_stim_all(kk).Rcd_proj',0.2,'m',x_df);
        else
            stdshade(cd_stim_all(kk).Rcd_proj',0.2,'r',x_df);
        end
%         plot(x_df,nanmean(type_sum_all(kk).nu_type1'),'r','LineWidth',2)
%         plot(x_df,nanmean(type_sum_all(kk).nu_type2'),'b','LineWidth',2)
        
%         title(strcat('Stimulus Task',num2str(kk)))
        plot_epochs();
        ylim([-ylim_cd ylim_cd]);
        xlim([-2.99 2.99])
        xticklabels({})
        if kk~=1
            yticklabels({})
        end
%         yline(0,'k');
        if kk==1
            ylabel('CD_{Sample}')
        end
    end


    for kk=1:length(type_sum_all)
    subplot(4,length(type_sum_all),kk+length(type_sum_all)*3)
    hold on
%         plot(x_df,type_sum_all(kk).sel_types(:,uu),'Color',[0.5 0.5 0.5],'LineWidth',2)
        stdshade(cd_choice_all(kk).Lcd_proj',0.2,'b',x_df);
        if kk==4
            stdshade(cd_choice_all(kk).Rcd_proj',0.2,'m',x_df);
        else
            stdshade(cd_choice_all(kk).Rcd_proj',0.2,'r',x_df);
        end     
        
%         plot(x_df,nanmean(type_sum_all(kk).nu_type1'),'r','LineWidth',2)
%         plot(x_df,nanmean(type_sum_all(kk).nu_type2'),'b','LineWidth',2)
        
%         title(strcat('Stimulus Task',num2str(kk)))
        plot_epochs();
        ylim([-ylim_cd ylim_cd]);
        xlim([-2.99 2.99])
        if kk~=1
            yticklabels({})
        end
%         yline(0,'k');
        if kk==1
            ylabel('CD_{Report}')
        end
    end