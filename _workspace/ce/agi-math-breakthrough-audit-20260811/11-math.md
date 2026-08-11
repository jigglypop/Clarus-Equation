# Mathematical lane

Status: COMPLETE

The current runtime defines internal relaxation but lacks four objects required for model-based agency:

\[
p(x_{t+1}\mid x_t,a_t),\qquad
b_t=p(x_t\mid o_{\le t},a_{<t}),\qquad
r_t\text{ or }\ell(g^*),\qquad
\arg\min_{a_{t:t+H}}\mathbb E[J].
\]

The present action is an immediate similarity decode. The critic is a post-observation error magnitude, not a value function. The cerebellar predictor is action-independent. Therefore the agent cannot compare counterfactual futures.

The current STDP gate is also directionally incomplete:

\[
g_t=\alpha\,\Delta \bar c_t+(1-\alpha)\|p_t-p^*\|^2.
\]

The squared homeostatic term loses the sign of over- versus under-activation and carries no synapse-specific direction. The critic difference is not a task return or TD advantage. This explains why weights can move without improving held-out prediction.

Minimum closure:

\[
b_t=\operatorname{Update}(T(b_{t-1},a_{t-1}),o_t),
\]

\[
\delta_t=r_t+\gamma V(b_{t+1})-V(b_t),\qquad
\Delta W_{ij}=\eta\delta_t e_{ij}-\eta_h(\bar f_i-f_i^*)a_j,
\]

\[
a_t=\operatorname{first}\arg\min_{a_{t:t+H-1}}
\mathbb E\sum_h\gamma^h
[\ell_{goal}+\lambda_u\operatorname{tr}\Sigma+\lambda_e\ell_{energy}].
\]

