# Loop 8 — unified executive inference and control equation

Status: MATHEMATICAL RESET — IMPLEMENTATION LOCKED

**SUPERSEDED — IMPLEMENTATION FORBIDDEN.** This generic POMDP candidate does
not structurally encode the PFC–MD thalamus–BG/STN–hippocampus circuit. It is
retained only as a comparison and counterexample record. The next candidate is
`loop8b-brain-geometric-executive-equation.md`.

## 1. Scope and formal status

- `[정의]`: latent state, generative model, posterior, policy functional.
- `[공리: 모델 선택]`: factorization and bounded dimensionless coefficients.
- `[정리]`: action-independent sensing no-go and branch-free probe criterion.
- `[미완성]`: model identification, approximation class, scalable solver,
  runtime bridge, and empirical efficacy.

## 2. Executive latent state

`[정의]` The executive does not store separate hard modes such as INSPECT,
WAIT, SWITCH, and COMMIT. It maintains one posterior over

\[
z_t=(x_t,c_t,v_t,\kappa_t).
\]

- \(x_t\): external latent world state;
- \(c_t\): latent task context/rule;
- \(v_t\in[0,1]\): context volatility;
- \(\kappa_t>0\): reliability/precision of the agent's observation-feedback
  model.

The history is \(h_t=(o_{1:t},r_{1:t},a_{0:t-1},g_{1:t})\), and the executive
state is the normalized posterior

\[
q_t(z_t)=p_\psi(z_t\mid h_t).
\]

Goal \(g_t\) is an explicit conditioning variable, not silently mixed into the
self state.

## 3. One generative model

`[공리: 모델 선택]` Use the factorization

\[
\begin{aligned}
&p_\psi(o_{t+1},r_{t+1},z_{t+1}\mid z_t,a_t,g_t)\\
&=p_\psi(o_{t+1}\mid x_{t+1},c_{t+1},a_t,\kappa_{t+1})
  p_\psi(r_{t+1}\mid x_{t+1},c_{t+1},a_t,g_t,\kappa_{t+1})\\
&\quad\times p_\psi(x_{t+1}\mid x_t,c_t,a_t)
  p_\psi(c_{t+1}\mid c_t,v_t)
  p_\psi(v_{t+1}\mid v_t)
  p_\psi(\kappa_{t+1}\mid\kappa_t).
\end{aligned}
\]

Context switching is a continuous mixture, not a threshold branch:

\[
p(c_{t+1}\mid c_t,v_t)
=(1-v_t)\,\delta_{c_t}(c_{t+1})+v_t\,\rho_0(c_{t+1}).
\]

Observation noise raises posterior mass on lower reliability \(\kappa\);
persistent evidence for a new rule raises posterior mass on changed \(c\) or
higher \(v\). A scalar surprise threshold is not allowed to decide which one.

## 4. One posterior update

`[정의]` Prediction:

\[
\bar q_{t+1}(z')
=\int p_\psi(z'\mid z,a_t)q_t(z)\,dz.
\]

Correction:

\[
q_{t+1}(z')
=\frac{
p_\psi(o_{t+1},r_{t+1}\mid z',a_t,g_t)\bar q_{t+1}(z')
}{
\int p_\psi(o_{t+1},r_{t+1}\mid \zeta,a_t,g_t)
\bar q_{t+1}(\zeta)\,d\zeta
}.
\]

The model evidence and surprise are

\[
Z_{t+1}=p_\psi(o_{t+1},r_{t+1}\mid h_t,a_t,g_t),
\qquad s_{t+1}=-\log Z_{t+1}.
\]

`[산출]` Surprise is an observable diagnostic derived from the posterior
normalizer. It is not an independent state-transition command.

## 5. One branch-free policy functional

Let \(\Pi_H\) be distributions over action sequences of horizon \(H\). Define
dimensionless task loss \(\tilde\ell=\ell/L_0\) and dimensionless resource cost
\(\tilde d=d/D_0\). For \(\pi\in\Pi_H\),

\[
\begin{aligned}
\mathcal J_t(\pi)
={}&\mathbb E_{q_\psi^\pi}
\left[\sum_{k=0}^{H-1}\gamma^k
\big(\tilde\ell(z_{t+k},a_{t+k};g_t)
+\lambda_d\tilde d(a_{t+k})\big)\right]\\
&+\lambda_R\,
\operatorname{Var}_{q_\psi^\pi}
\left(\sum_{k=0}^{H-1}\gamma^k\tilde\ell_{t+k}\right)\\
&-\beta\sum_{k=0}^{H-1}\gamma^k
I_{q_\psi^\pi}
\left(z_{t+k+1};o_{t+k+1},r_{t+k+1}\mid h_t\right)\\
&+\tau\,D_{\mathrm{KL}}\!\left(\pi\,\|\,\pi_0\right).
\end{aligned}
\]

The executive policy is

\[
\boxed{\pi_t^*=\arg\min_{\pi\in\Pi_H}\mathcal J_t(\pi)}.
\]

- exploitation is the expected task loss;
- caution is posterior risk, not an `if confidence < threshold` branch;
- probing is positive expected information gain caused by the action-dependent
  observation model;
- waiting is any low-resource action whose predictive value is favorable;
- switching is posterior flow between contexts through \(v_t\);
- habitual/default behavior is the KL prior \(\pi_0\).

These labels describe solutions after optimization; they are not controller
branches.

## 6. Theorem: why Loop 7 could not work

`[정리: action-independent sensing no-go]` Suppose for all candidate actions
\(a\),

\[
p(z_{t+1}\mid z_t,a)=p(z_{t+1}\mid z_t),
\qquad
p(o_{t+1},r_{t+1}\mid z_{t+1},a)
=p(o_{t+1},r_{t+1}\mid z_{t+1}).
\]

Then

\[
I(z_{t+1};o_{t+1},r_{t+1}\mid h_t,a)
\]

is identical for every action. Consequently the epistemic term cannot change
the action ordering in \(\mathcal J_t\).

`[증명]` Under the two assumptions, the predictive joint

\[
q(z',o',r'\mid h_t,a)
=\int p(o',r'\mid z')p(z'\mid z)q_t(z)\,dz
\]

contains no \(a\). Mutual information is a functional only of this joint, so it
is action-invariant. Subtracting the same constant from every action objective
cannot change the argmin. \(\square\)

`[미완성]` This theorem is not a complete explanation of Loop 7. Its card
observation itself was action-independent, but binary correctness feedback was
action-dependent, so the second no-go assumption did not hold for the joint
`(observation, feedback)` channel. The measured conclusion is narrower: the
available one-bit action-dependent feedback did not give the registered active
policy a return advantage. A new task must expose an action-dependent
high-quality observation kernel before epistemic control is retested.

## 7. Theorem: when a probe emerges without a branch

`[정리: branch-free probe criterion]` In the deterministic zero-temperature
two-action limit with equal risk and prior terms, let `probe` have extra
dimensionless cost \(\Delta C>0\) and extra expected information
\(\Delta I>0\) relative to `commit`. Then `probe` has lower objective exactly
when

\[
\boxed{\beta\Delta I>\Delta C.}
\]

`[증명]` The objective difference is
\(\mathcal J(\mathrm{probe})-\mathcal J(\mathrm{commit})
=\Delta C-\beta\Delta I\). Its sign gives the result. \(\square\)

For \(\tau>0\), the KL-regularized policy changes this discontinuous argmin into
a smooth probability ratio; no explicit `if probe` branch is required.

## 8. Reduction and sanity limits

`[산출]`

1. \(\beta=\lambda_R=\lambda_d=0\): ordinary expected-loss planning.
2. \(H=1\): contextual bandit limit.
3. \(v_t=0\): stationary-context Bayesian filter.
4. \(\kappa_t=\kappa_0\) fixed: no metacognitive reliability inference.
5. action-independent sensing: epistemic term cancels by the no-go theorem.
6. posterior point mass: fully observed model-predictive control limit.

## 9. Dimensionless audit

| Core argument | Dimension | Normalization |
|---|---:|---|
| probabilities \(q,p,Z,v\) | \((0,0,0,0)\) | already dimensionless |
| \(-\log Z\) | \((0,0,0,0)\) | log of probability |
| mutual information | \((0,0,0,0)\) | KL in nats |
| \(D_{KL}(\pi\|\pi_0)\) | \((0,0,0,0)\) | probability ratio |
| \(\tilde\ell\) | \((0,0,0,0)\) | \(\ell/L_0\) |
| \(\tilde d\) | \((0,0,0,0)\) | \(d/D_0\) |
| \(\operatorname{Var}(\sum\tilde\ell)\) | \((0,0,0,0)\) | normalized loss squared |
| \(\gamma,\lambda_d,\lambda_R,\beta,\tau\) | \((0,0,0,0)\) | model coefficients |

Dimension status: dimensionless. This is dimensional consistency only, not
empirical justification.

## 10. Implementation lock and first falsification problem

No new branch handler may be added to `RuntimeAgent` from this document. The
first implementation must be an isolated exact finite-state solver with:

- at least two latent contexts and two reliability levels;
- an action-dependent observation kernel with one costly high-information
  action and ordinary task actions;
- a context transition driven by latent \(v\), not an observed switch flag;
- exhaustive horizon-2 policy evaluation using the single \(\mathcal J\);
- ablations \(\beta=0\), fixed \(v\), fixed \(\kappa\), shuffled observation
  kernel, action-independent sensing, and oracle;
- parameter recovery checks before return comparison;
- no learned neural baseline until the exact solver and causal identifiability
  gates pass.

The route is rejected if the epistemic advantage remains under the
action-independent no-go control, if inferred volatility cannot separate noise
from context change, or if the candidate needs action labels hard-coded into
the optimizer.

## 11. Closure audit

| Item | Status | Reason |
|---|---|---|
| latent-state and generative factorization | `[공리: 모델 선택]` | one admissible executive model, not derived from CE axioms |
| Bayesian prediction/correction | `[정의]` / `[산출]` | normalized posterior under the selected model |
| policy functional | `[정의]` | optimization target chosen for the next falsification |
| action-independent sensing no-go | `[정리]` | exact under both stated conditional-independence assumptions |
| branch-free probe criterion | `[정리]` | exact only in the stated two-action, equal-risk, zero-temperature limit |
| explanation of all Loop 7 failure | `[미완성]` | feedback violated full action-independence |
| biological PFC mapping | `[미완성]` | no neural measurement or intervention bridge |
| AGI efficacy | `[미완성]` | no general task family or scaling result |

Gate result: the two narrow mathematical theorems are closed. The generative
model and policy are registered model choices; empirical and biological claims
remain open.
