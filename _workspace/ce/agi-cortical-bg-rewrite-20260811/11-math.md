# Rewritten mathematical core

## 1. Cortical predictive state

Let all observations, actions, rewards, hidden activities, values, and time be
normalized by declared reference scales. The cortical state is

\[
b_t=\mathcal B_\phi(b_{t-1},\bar o_t,a_{t-1},\bar r_{t-1}),
\qquad
\hat p_\phi(o_{t+1},r_t\mid b_t,a_t).
\]

It is trained by the dimensionless predictive objective

\[
\mathcal L_{\rm pred}
=-\log \hat p_\phi(\bar o_{t+1},\bar r_t\mid b_t,a_t).
\]

This defines a predictive state, not an oracle context label and not a fixed
hazard grid. Interpreting it as a biological belief state is a hypothesis.

## 2. Option DAG and conserved routing

Within one tick, let \(G=(V,E)\) be finite and acyclic. Root mass is
\(F_{r,t}=1\). Every internal node distributes all incoming mass between its
children and HOLD:

\[
g_{uH,t}+\sum_{v\in\mathrm{ch}(u)}g_{uv,t}=1,
\qquad g_{uv,t}\ge 0.
\]

At a multi-parent node,

\[
F_{v,t}=\sum_{u\in\mathrm{pa}(v)}F_{u,t}g_{uv,t}.
\]

Leaf actions and local HOLD events collect their path mass. Consequently total
terminal mass is exactly one. Multiplying parent probabilities is forbidden: it
double-counts shared ancestors unless an unproved conditional-independence model
is supplied.

## 3. D1/D2, STN, GPi, and a real HOLD effect

For each candidate edge \(u\to v\), cortical state creates dimensionless local
features \(x_{uv,t}\). Nonnegative co-active striatal drives are

\[
D^1_{uv,t}=\operatorname{softplus}((w^1_{uv})^Tx_{uv,t}),\qquad
D^2_{uv,t}=\operatorname{softplus}((w^2_{uv})^Tx_{uv,t}).
\]

With normalized proposal entropy \(C_{u,t}\in[0,1]\), a testable STN hypothesis is

\[
S_{u,t}=\operatorname{softplus}(\alpha_0+\alpha_C C_{u,t}).
\]

Dimensionless GPi inhibition is

\[
G_{uv,t}=G_0+\omega_S S_{u,t}-\omega_1D^1_{uv,t}+\omega_2D^2_{uv,t}.
\]

HOLD is an explicit reference channel, not a common offset inside an action-only
softmax:

\[
P(H\mid u)=\frac{1}{1+\sum_v\exp(-G_{uv,t}/\tau_G)},
\]

\[
P(v\mid u)=\frac{\exp(-G_{uv,t}/\tau_G)}
{1+\sum_c\exp(-G_{uc,t}/\tau_G)}.
\]

Therefore \(P(H)+\sum_vP(v)=1\). Increasing the common STN term increases HOLD
while preserving the conditional action ranking. By contrast,
\(\operatorname{softmax}(z-c\mathbf1)=\operatorname{softmax}(z)\), so any model
that inserts STN only as an action-softmax common offset has exactly zero STN
effect and must fail the gate.

## 4. Dopamine credit without path multiplication

The critic uses the cortical state:

\[
\delta_t=\bar r_t+\gamma(1-d_t)V_\psi(b_{t+1})-V_\psi(b_t).
\]

For realized action \(a\), define the backward probability of reaching that
action from node \(v\):

\[
B_\ell^{(a)}=\mathbf1[\mathrm{label}(\ell)=a],\qquad
B_u^{(a)}=\sum_vg_{uv}B_v^{(a)}.
\]

The posterior responsibility of edge \(u\to v\) is

\[
\xi_{uv,t}^{(a)}=
\frac{F_{u,t}g_{uv,t}B_v^{(a)}}{P(a\mid b_t)}.
\]

Eligibility and the bounded three-factor update are

\[
e_{uv,t}=\gamma\lambda e_{uv,t-1}
+\xi_{uv,t}^{(a_t)}\nabla\log g_{uv,t},
\]

\[
w_{t+1}=\Pi_{\mathcal W}\left(w_t+\eta\delta_te_t\right).
\]

Responsibility normalization is a mathematical construction. Equating the
scalar \(\delta_t\) and signed update exactly with dopamine and receptor-specific
plasticity remains a falsifiable biological hypothesis.

## 5. Option persistence and finite execution

An option terminates with \(\beta_o(b_t)\). Require either
\(\beta_o(b)\ge\beta_{\min}>0\) or duration \(d_o\le D_{\max}\). Then
\(E[d_o]\le1/\beta_{\min}\), and a within-tick topological pass costs
\(O(|V|+|E|)\). Feedback from an outcome can affect only later ticks.

## 6. Dimensional audit

All variables above are normalized: \(\bar r=r/r_0\), \(\bar t=t/t_0\), and
\(\bar o=o/o_0\) for continuous observations. Thus inputs of exp/log/softplus,
\(G/\tau_G\), probabilities, entropy, \(\gamma\), \(\lambda\), \(\beta\), and
\(\xi\) are dimensionless. If physical firing rates are retained, they must be
divided by a declared rate scale before entering the core. This audit establishes
dimensional consistency, not biological truth or convergence.
