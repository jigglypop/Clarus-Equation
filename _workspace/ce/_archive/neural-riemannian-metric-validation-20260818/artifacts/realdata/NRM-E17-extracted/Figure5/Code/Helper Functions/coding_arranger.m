function [Z_choi,Z_stim,Z_delay,coding_choice,coding_stim,coding_delay] = coding_arranger(Z,x_df,sigs,tested,mindays)
% coding=find(sum(sigs.Choice(1:3,:)+sigs.Stimuli(1:3,:))>2);
coding_choice=find(sum(sigs.Choice(tested,:))>mindays);
% coding=find(sum(sig_all.Choice(1:3,:))>0);
% 
coding_stim=find(sum(sigs.Stimuli(tested,:))>mindays);

coding_delay=find(sum(sigs.Delay(tested,:))>mindays);

% coding=find(sum(sig_all.Stimuli(1:3,:))>0);
%%
Z_choi=Z;
Z_stim=Z;
Z_delay=Z;


% choi=horzcat(30:50,90:110);
% stims=horzcat(12:24,42:54);
idx_s_stim=find(x_df>-1.8,1,'first');
idx_e_stim=find(x_df>-0.8,1,'first');

idx_s_choi=find(x_df>-0,1,'first');
idx_e_choi=find(x_df>2,1,'first');

idx_s_delay=find(x_df>-0.9,1,'first');
idx_e_delay=find(x_df>-0.2,1,'first');


choi=horzcat(idx_s_choi:idx_e_choi,length(x_df)+(idx_s_choi:idx_e_choi));
stims=horzcat(idx_s_stim:idx_e_stim,length(x_df)+(idx_s_stim:idx_e_stim));
delay=horzcat(idx_s_delay:idx_e_delay,length(x_df)+(idx_s_delay:idx_e_delay));

for ii=1:length(Z)
    Z_choi(ii).srt=Z(ii).srt(choi,:);
    Z_stim(ii).srt=Z(ii).srt(stims,:);
    Z_delay(ii).srt=Z(ii).srt(delay,:);
end