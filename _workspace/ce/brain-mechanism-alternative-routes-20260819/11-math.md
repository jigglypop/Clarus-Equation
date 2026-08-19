# Mathematical and protocol verification

Status: COMPLETE

## 1. Runtime convention and causal orientation

For activation $a_t$ and the remaining native state $z_t$, the relevant deterministic transition is
written schematically as

$$
a_{t+1}=F_W(z_t,e_t),
\qquad
F_W\supset \tanh(Wq_t+g_{\mathrm{ext}}e_t),
$$

where $q_t$ is the active presynaptic vector. The runtime uses `W @ q`, so a causal transition from
cue coordinate $j$ to target coordinate $i$ requires a positive applied change in $W_{ij}$.
Eligibility and weight matrices must therefore use the row-post, column-pre convention.

## 2. Alternative memory updates

### M1: delayed three-factor rule

During an episode block, causal eligibility is accumulated without changing weight:

$$
E_b=\sum_{t\in b}
\left(a_{t}^{\mathrm{post}}(p_t^{\mathrm{pre}})^{\mathsf T}
-\lambda_{\mathrm{LTD}}p_t^{\mathrm{post}}(a_t^{\mathrm{pre}})^{\mathsf T}\right).
$$

Only after the block is complete is the scalar modulation $m_b$ applied:

$$
W_{b+1}=\Pi_{\mathcal W}\left(W_b+\eta m_bE_b\right).
$$

$m_b$ is frozen here to a `+1.0` block-end clock pulse. It does not depend on whether replay occurred
or on any runtime, reward, target, decoder, or memory value. Zero and sign lesions use `0.0` and
`-1.0`; the remaining arms retain `+1.0`. Because the structural
projection $\Pi_{\mathcal W}$ can erase or redistribute a raw eligibility increment, the primary
mechanistic audit is the *applied* block contrast after projection, not merely $E_b\ne0$.

For independent cue and target codewords, define that applied contrast as

$$
B(\Delta W)=\frac1K\sum_k\left[v_k^{\mathsf T}\Delta Wc_k-
\frac1{K-1}\sum_{\ell\ne k}v_\ell^{\mathsf T}\Delta Wc_k\right].
$$

It is evaluated on the actual post-install matrix with tolerance $10^{-6}$ and paired against the
target-shuffled arm. The time lesion reverses target/cue order without changing pair identity or
event counts; the assignment lesion changes only target identity.

Every arm clears native transient dynamics between the two phases. Recurrent weight remains, while
activation, refractory state, adaptation, STP, lifecycle, and delay state are reset. Forward-order
arms retain the staged row for the value phase. The time-reversed arm removes it before the cue phase,
preventing the hippocampus from injecting the already presented target during cue delivery.
Therefore the fixed-clock route can bridge cue to value only through the external local eligibility
trace. The eligibility-reset lesion clears that final bridge as well.

The M1 decoder abstains below cosine $0.20$. This boundary was frozen on development after a strict
gap between the maximum deleted-cue score $0.170074$ and minimum known-target score $0.529764$.
It is not shared with M0 and cannot be tuned on confirmation.

### M2: contrastive settling

For seed- and schedule-matched positive and negative phases, use a bounded difference of local
correlations:

$$
\Delta W_{\mathrm{CHL}}
=\eta\left(C^+-C^-\right),
\qquad
C^{\pm}=\frac{1}{T_{\pm}}\sum_t a_t^{\pm}(a_{t-1}^{\pm})^{\mathsf T}.
$$

The directional lag avoids silently replacing the runtime with a symmetric Hopfield convention.
The positive phase may contain the training target and is therefore supervised. Evaluation never
contains it. Identical positive/negative phases imply $\Delta W=0$ up to numerical tolerance and
serve as an invariant test.

### M3: predictive-error plasticity

Let $\widehat a_{t+1}=f_\theta(z_t,e_t)$ be a deterministic ridge model fitted on the first 64
native transitions of each circuit, from pre-transition observables only. Ridge is $10^{-4}$ and
the model freezes before held-out scoring or recurrent writing. The local
delta candidate is

$$
\Delta W_t=\eta\left(a_{t+1}-\widehat a_{t+1}\right)q_t^{\mathsf T}.
$$

The target is the actual next native activation, never a teacher-forced codeword. This can test
transition learning without a symbolic target. It does not imply association binding.
Consequently its verdict is hierarchical: first held-out next-state prediction, then independent
zero-store recall, then held-out factor transfer. Failure at a later gate narrows rather than erases
an earlier predictive result.

### M0: capacity ceiling

For singular values $\sigma_1\ge\cdots$ of the raw Route-B desired matrix

$$
W^*=\sum_k(v_kc_k^{\mathsf T}+0.65v_kv_k^{\mathsf T}),
$$

the rank-$r$ capacity write is

$$
\Delta W_r=U_{1:r}\operatorname{diag}(\sigma_{1:r})V_{1:r}^{\mathsf T},
$$

installed with the same declared Frobenius bound as its random low-rank control. The random control
uses the same singular spectrum with independent orthonormal vectors; cue-only is norm matched.
Truncation precedes subtraction of the random initial runtime matrix. Dale and structural projection
are disabled for this capacity diagnostic. This identifies the rank/structure needed by a supervised
writer. It supplies no evidence for acquisition.

## 3. Geometry without conflation

Around a frozen calibration trajectory, let

$$
\delta x_{t+1}=A_t\delta x_t+B_tu_t+\varepsilon_t,
\qquad
\varepsilon_t\sim(0,Q_t).
$$

Two matrices with different meanings must not be merged:

$$
C_H^{u}=\sum_{k=0}^{H-1}\Phi_{H,k}B_kB_k^{\mathsf T}\Phi_{H,k}^{\mathsf T},
\qquad
\Sigma_H=\sum_{k=0}^{H-1}\Phi_{H,k}Q_k\Phi_{H,k}^{\mathsf T},
$$

where $\Phi_{H,k}=A_{H-1}\cdots A_{k+1}$. The first is a finite-horizon reachability Gramian for a
declared control interface; the second is predictive covariance. Their regularized inverse metrics
are respectively

$$
g_H^u=(C_H^u+\lambda R_0)^{-1},
\qquad
g_H^x=(\Sigma_H+\lambda R_0)^{-1}.
$$

Calibration data alone determine $A,B,Q,\lambda,R_0$. A coordinate change $x'=Px$ requires

$$
g'=P^{-\mathsf T}gP^{-1}.
$$

Transformation-law residuals verify algebra, not curvature or biological ontology. Direct dynamics
remains an adverse baseline: if $(A,B,Q)$ predicts held-out trajectories as well as or better than a
metric readout, the permissible conclusion is covariance/reachability summarization, not a distinct
geometric mechanism.

## 4. Identification of the two arrows

G1 randomizes a known edge-block intervention $T$ before calibration. Let $M$ be a frozen metric
contrast and $Y$ a held-out trajectory or first-passage score. The separate questions are

$$
T\to \Delta W\to M,
\qquad
T\to M\to Y.
$$

Norm/sign/density-matched scrambled edges, gain-only, noise-only, sham, time reversal, and label
permutation are complete counterexample arms. A post-hoc association among $\Delta W$, $M$, and $Y$
without randomized $T$ does not identify mediation. G3 is evaluated only when a randomized learning
contingency from M1--M3 produces independent recall; supervised M0 is prohibited as its treatment.

## 5. Prediction-guided control

At decision time a frozen risk model supplies $\widehat r(z_t,a)$ before transition. A policy
chooses `commit` or `safe` under a fixed action and energy budget. Let $D=1$ denote deferral and $L$
the realized composite loss. The primary comparison conditions on exactly matched coverage:

$$
\Pr(D=1)_{\mathrm{adaptive}}=\Pr(D=1)_{\mathrm{random}}.
$$

The causal quantity is the paired seed-level difference in $L$, not the adaptive arm's raw loss.
The next state, realized loss, hidden disturbance label, task scorer, and oracle action are forbidden
decision features. The oracle is a ceiling only.

## 6. SCC intervention

Given a precommitted directed effective-edge rule, SCC labels are computed before outcome access.
For an SCC feedback cut $\Delta W_S$ and matched control cut $\Delta W_C$, matching requires at least

$$
|E_S|=|E_C|,
\quad
\|\Delta W_S\|_F=\|\Delta W_C\|_F,
$$

plus sign and pre/post degree-stratum balance and a frozen spectral-radius tolerance. The estimand is
the paired recovery-loss difference. A result explained by removed mass, degree, or spectral radius
does not isolate strong connectivity. A giant SCC without an eligible matched cut is non-evaluable.

## 7. Statistical unit and decision logic

The independent unit is a seeded circuit. Development seeds tune only declared algorithmic
parameters; confirmation seeds receive one frozen pass. Paired intervals and null-family error are
computed over circuits. Time ticks, codebook coordinates, neurons, and multiple cues within a circuit
are repeated measurements, not additional sample size.

Each route can return GO, STOP, or NOT-EVALUABLE. A GO requires all task, adverse-control, leakage,
finite-state, and snapshot-parity predicates. Nonzero weight change, nonzero metric distance, or a
descriptive correlation cannot substitute for the full conjunction.
