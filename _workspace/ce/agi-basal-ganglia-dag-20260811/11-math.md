# Mathematical candidate

Status: PROPOSED ENGINEERING MODEL

Let `G=(V,E)` be a finite DAG inside one microstep, with every edge satisfying
the topological order `u < v`. Observation `o_t`, MD context `c_t`, and previous
state `s_t` are normalized before entering the graph.

## Forward proposal

\[
h_{v,t}=\phi_v\!\left(W_vx_t+C_vs_t+
\sum_{u\in pa(v)}A_{vu}h_{u,t}+b_v\right).
\]

Action leaves emit promoting evidence `q^+_{a,t}=w_a^T h_{a,t}`.

## Reverse competitive inhibition

After the forward pass, an inverse topological reduction aggregates competing
children. Using log-mean-exp avoids an automatic penalty for high branching:

\[
\iota_{v,t}=\tau_I\log\left({1\over |ch(v)|}
\sum_{c\in ch(v)}e^{(\iota_{c,t}+\kappa_cq^+_{c,t})/\tau_I}\right).
\]

For action `a`, sibling inhibition is accumulated along its ancestral routes:

\[
I_{a,t}=\sum_{v\in anc(a)}\alpha_v\,
\operatorname{LME}_{\tau_I}\{\iota_{c,t}:c\in sib_v(a)\}.
\]

\[
z_{a,t}=q^+_{a,t}-q^-_{a,t}-I_{a,t},\qquad
\pi(a|x_t,s_t)={e^{z_{a,t}/\tau_\pi}\over\sum_be^{z_{b,t}/\tau_\pi}}.
\]

The reverse reduction is neither time reversal nor backpropagation. It is one
finite inhibitory message pass over an already evaluated DAG.

## Cross-pass recurrence

Only after action and environmental feedback:

\[
s_{t+1}=(1-\beta_t)s_t+\beta_t\tanh
\left(Rs_t+U_oo_{t+1}+U_aa_t+U_\delta\delta_t\right),
\]

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
\]

Eligibility-gated policy update:

\[
e_{ij,t}=\gamma\lambda e_{ij,t-1}+
\partial_{\theta_{ij}}\log\pi(a_t|x_t,s_t),\qquad
\theta_{ij}\leftarrow\Pi_\Theta[\theta_{ij}+\eta\delta_te_{ij,t}].
\]

The sign of `delta_t` must be preserved.

## Finiteness and stability

One pass terminates in `O(|V|+|E|+|A|)`. Across time, a sufficient deterministic
closed-loop condition is

\[
L_s+L_aL_\pi<1.
\]

Implementation must project the recurrent operator below a declared spectral
bound and cap state norm. This bounds updates but does not prove convergence of
nonlinear recurrent TD learning.

All exponent arguments are dimensionless: logits, inhibition, value, reward,
prediction error, and temperatures must use one normalized utility scale.
