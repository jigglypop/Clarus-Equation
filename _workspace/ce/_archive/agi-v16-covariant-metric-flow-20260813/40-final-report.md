# AGI V16 covariant metric flow: final report

Status: COMPLETE

## 1. Question and exact verdict

The run asked whether the five V15 finite readouts could stop depending on an
oracle-supplied metric and instead share one learned Riemannian cost state
$g_t$, with no role-specific persistent weights. The implemented answer is a
qualified yes for finite vector observations:

$$
\boxed{\text{V16 NARROW GO}}
$$

G-MATH, G-NUMERIC, G-LEARN, G-CHART and G-CLOSED-LOOP all passed their
preregistered gates. `AGI GO` is forbidden by the contract and is not claimed.

[Definition] The sole persistent semantic state is an SPD cost metric encoded
by one canonical Cholesky factor. The learner receives an executed nonzero
vector $x_t$ and a positive scalar cost $c_t$; it does not receive the hidden
metric. This is not raw perception.

## 2. Mathematical result

[Theorem] For prediction $p=x^Tgx$, residual $r=\log(p/c)$ and
$0<\eta\le1$, equation V16.1 is an SPD-preserving rank-one congruence. It is
covariant under every affine chart Jacobian $J\in GL(d)$, contracts the same
observation residual exactly by $1-\eta$, and is exactly one AIRM
natural-gradient exponential-map step. No spectral projection is required.

[Theorem] Noiseless quadratic observations identify the full metric iff the
matrices $x_tx_t^T$ span $\operatorname{Sym}(d)$.

[Theorem] For a finite nonzero spanning direction family visited with uniformly
bounded gaps, fixed $0<\eta\le1$ and noiseless costs from $g_*$, the Burg/log-det
divergence has a strict decrement and $g_t\to g_*$. The 12,000-step registered
fixture ended at Frobenius error $8.91\times10^{-16}$.

[No-go theorem] Fixed-rate point convergence under persistent multiplicative
noise is false, even for a bounded-gap spanning schedule. Stationary-risk and
diminishing-rate convergence remain incomplete.

## 3. Implementation result

[Derived] `CovariantMetricFlow.update` uses a factor congruence followed by QR
canonicalization, avoiding a subtractive outer-product update that can destroy
a representable positive eigenvalue. The mathematical rank-one structure is
$O(d^2)$, while this reference binary64 canonicalization is honestly reported
as $O(d^3)$. Nonrepresentable public results fail explicitly.

[Derived] The inherited V15 path, length, target-tie and surprise defects were
repaired. Representative shortest-path relaxation is strict, path counting is
a separate distance-oriented DAG, reconstruction is cycle bounded, lengths use
scaled arithmetic, and the hard surprise decision is made in the log domain.

The factor is an encoding of $g$, not a second state beside $g$. Conversely,
using one $g$ does not prove that world, memory, planning, critic and goal have
the same semantics; it only gives them a shared finite cost substrate and
readout convention.

## 4. Sealed confirmation score

[Prediction protocol] Development seeds 917000--917063 selected rates before
the confirmation block was opened. Six artifacts were SHA-256 sealed. An
exclusive receipt was created before the first confirmation seed, and seeds
918000--918255 were executed once.

[Numerical result] V16 produced:

| Metric | Result | Registered gate |
|---|---:|---:|
| finite episodes | $1.0$ | $1.0$ |
| held-out route accuracy | $0.9642334$ ($15798/16384$) | $\ge0.90$ |
| mean normalized held-out regret | $0.000439384$ | $\le0.05$ |
| median invariant metric error | $0.0339121$ | $\le0.25$ |
| identity/conformal regret improvement | $0.3879170$ | $\ge0.10$ / $\ge0.05$ |
| V16 regret minus additive regret | $-0.0120556$ | $\le0.02$ |
| online regret improvement over identity after step 32 | $0.4872651$ | $\ge0.05$ |
| affine-chart action agreement | $1.0$ | $1.0$ |
| maximum affine prediction defect | $2.6735\times10^{-13}$ | $\le10^{-10}$ |

The route-level Wilson interval $[0.961279,0.966970]$ is descriptive only,
because 64 routes within one episode are clustered. A paired episode-level
test cannot be reconstructed without reopening confirmation because per-seed
summaries were not preregistered for storage.

## 5. Verification and repository state

The focused V16/V15/dimensionless evaluator suite passed 63 tests. The expanded
SCC-related slice passed 296 tests. The deterministic math verifier passed 768
trials, Ruff passed, compileall passed, and the post-confirmation six-file
manifest still matches.

The repository-wide system-Python run recorded `2314 passed, 14 skipped, 28
failed, 41 errors`. The failure/error/skip counts match the prior recorded dirty
baseline and are dominated by unavailable ScienceDB/fusion payloads, Q0 and
neural-tree artifacts, plus an existing policy drift. Therefore the repository
as a whole is not green; the focused and related V16 slices are green. The
preferred `uv` invocation was blocked by host application-control policy before
test collection, so the full run used the installed Python 3.11 interpreter.

## 6. Evidence boundary

This result supports one finite claim: a single affine-covariant SPD state can
learn anisotropic vector costs online and improve immediate route selection in
the registered synthetic environment.

It does not establish any of the following:

- raw sensory representation or autonomous construction of displacement
  features;
- delayed or long-horizon credit assignment;
- a learned compute-matched semantic OOD advantage;
- nonstationary-world adaptation or persistent-noise point convergence;
- integration with BrainRuntime or the nested SCC tower;
- continuum Riemannian geometry or an infinite-agent limit;
- biological fidelity, a brain--cosmos identity, consciousness, tool use, or
  AGI.

Identity and conformal controls are behaviorally identical on the unit-vector
choice protocol, so their two improvement gates are not independent evidence.
The seeds are deterministic and public; the receipt documents procedure but
does not cryptographically prove that nobody inspected them elsewhere.

## 7. Next falsifiable breakthrough

The next run should not increase the number of recursive agents. It should test
whether the learned $g_t$ is a sufficient state for information that actually
matters to an agent:

1. preregister and store per-episode summaries, use a commit--reveal or external
   secret confirmation block, and add a measured compute ledger;
2. replace direct vectors with a learned observation-to-tangent encoder and
   test semantic distribution shift against a compute-matched recurrent model;
3. introduce delayed rewards and first prove or kill the claim that a pure
   metric state can assign credit without an additional eligibility/history
   state;
4. only after that, define and test an explicit map between finite SCC states
   and metric tangent observations, with sampling and direct-limit consistency
   gates.

If two histories can reach the same $g_t$ and current observation but require
different credit updates, a Markov learner whose only state is $g_t$ cannot
distinguish them. That indistinguishability is the likely next no-go theorem and
the correct place to decide whether “only $g$ changes” is sufficient for AGI or
must be relaxed.
