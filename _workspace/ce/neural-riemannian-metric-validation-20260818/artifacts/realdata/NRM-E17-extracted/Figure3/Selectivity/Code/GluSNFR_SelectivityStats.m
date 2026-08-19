

%%
    f=figure();

    subplot(2,3,1)
    hold on

        stdshade(ConA_Summ.type1',0.2,'r',x_df);   
        stdshade(ConA_Summ.type2',0.2,'b',x_df);
        title('Context A')
        plot_epochs();
        ylim([0. 0.2]);
        xlim([-2.9 2.99])

        ylabel('norm. dF/F0')
    subplot(2,3,2)
        hold on

        stdshade(ConB_Summ.type1',0.2,'m',x_df);   
        stdshade(ConB_Summ.type2',0.2,'b',x_df);
        title('Context B')
        plot_epochs();
        ylim([0. 0.2]);
        xlim([-2.9 2.99])
        yticklabels({})
    subplot(2,3,3)
        hold on

        stdshade(ConA2_Summ.type1',0.2,'r',x_df);   
        stdshade(ConA2_Summ.type2',0.2,'b',x_df);
        title('Context A')
        plot_epochs();
        ylim([0.0 0.2]);
        xlim([-2.9 2.99])
        yticklabels({})
    subplot(2,3,4)
    hold on

        stdshade(ConA_Summ.selec',0.2,'k',x_df);   
        
        plot_epochs();
        ylim([0.0 0.13]);
        xlim([-2.9 2.99])
        ylabel('Selectivity')
    subplot(2,3,5) 
        hold on

        stdshade(ConB_Summ.selec',0.2,'k',x_df);   
        
        plot_epochs();
        ylim([0.0 0.13]);
        xlim([-2.9 2.99])
        yticklabels({})
        xlabel('Time (s)')
    subplot(2,3,6)
        hold on

        stdshade(ConA2_Summ.selec',0.2,'k',x_df);
        
        plot_epochs();
        ylim([0.0 0.13]);
        xlim([-2.9 2.99])
        yticklabels({})


