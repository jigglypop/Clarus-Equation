# Implementation direction

Status: COMPLETE

One-model checkpoint: `ActionConditionedBeliefRuntime`.

Minimal state:

\[
B_t=(S_t^{runtime},o_t,\mu_t,\Sigma_t,g^*,D).
\]

Minimal transition:

\[
z_{t+1}=Az_t+w_t,
\qquad
\hat o_{t+1}^{(a)}=f_\theta(o_t)+De(a)+\tau_tL\mu^-_{t+1},
\]

\[
\tau_t=\frac{\|L\mu_t\|^2}{\|L\mu_t\|^2+\operatorname{tr}(L\Sigma_tL^\top+R)+\epsilon}.
\]

Use a Huber or Student-t innovation update, H=2/3 MPC, and transition tuples `(o_t,a_t,o_{t+1})`. All new paths require feature-off equivalence and future-poison tests.

Do not add metacognition, diffusion, CloudCell, rank-2 latent modes, or fresh validation seeds before the model demonstrates action-dependent prediction and held-out goal gain.

