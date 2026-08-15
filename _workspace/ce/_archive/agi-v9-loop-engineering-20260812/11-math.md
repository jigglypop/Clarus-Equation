# Mathematical verification

Status: COMPLETE

## L1. Infinite operator without false exact compatibility

Let $B_X=\{x\in X:\lVert x\rVert_X\le1\}$. For a finite dimensionless input $o$ define
$F_o:B_X\to B_X$ componentwise by the existing Jacobi rule. Componentwise $\tanh$ makes
$B_X$ forward invariant. Since $\tanh$ is one-Lipschitz and every level has at most one
recurrence, upward, and downward block,

$$
\lVert F_o(x)-F_o(y)\rVert_X\le q\lVert x-y\rVert_X,
\qquad q=r+u+d.
$$

Thus $q<1$ makes $F_o$ a contraction on the complete metric space $B_X$, so it has one fixed
point $x^*(o)$.

Let $F_o^{(n)}$ be the zero-boundary prefix through level $n$, $x_n^*$ its unique fixed point,
and $J_nx_n^*$ the append-zero element of $X$. Exact commutation is false when $u>0$, but the
only nonzero one-step residual outside the prefix is the new level $n+1$:

$$
\epsilon_n
=\lVert F_o(J_nx_n^*)-J_nF_o^{(n)}(x_n^*)\rVert_X
\le u\lambda^{n+1}.
$$

The contraction resolvent therefore gives

$$
\lVert x^*(o)-J_nx_n^*\rVert_X
\le \frac{u\lambda^{n+1}}{1-q}.
\tag{L1.1}
$$

This proves convergence of finite fixed points to the infinite fixed point for
$0<\lambda<1$. It does not restore exact direct-limit commutation.

For a common causal input sequence and zero-tail matched initial states, the same boundary
residual yields

$$
E_{t+1}\le qE_t+u\lambda^{n+1},
$$

and hence

$$
E_t\le q^tE_0+rac{1-q^t}{1-q}u\lambda^{n+1}.
\tag{L1.2}
$$

Equations (L1.1)--(L1.2) are uniform domain bounds, not sampled current-state defects.

For the current defaults $r=0.24$, $u=0.16$, $d=0.14$, $\lambda=0.72$, so $q=0.54$ and

$$
\frac{u\lambda^{n+1}}{1-q}=\frac{0.16\cdot0.72^{n+1}}{0.46}.
$$

## L2. Runtime-to-tower cascade

`BrainRuntime` clamps activation to $[-1,1]^d$. A supplied observation is finite-checked, and
the cosine encoder maps every finite observation/action-embedding pair to $[-1,1]^A$ with
zero norm mapped to zero. The tower maps its unit state domain into itself and token policy is
a probability simplex. Therefore the action cascade is bounded under every finite runtime
history.

This is a cascade boundedness theorem. It is not a contraction theorem for the complete
`BrainRuntime` state, because mode switching, sparse selection, plasticity, and the agent
feedback map are not covered by a common Lipschitz constant.

## L3. Empirical identifiability conditions

A performance difference identifies nested state mediation only if all arms share raw inputs,
encoder, decision rule, and seed episodes; lesions operate on the candidate state path; and no
oracle label/posterior enters candidate output. Upper reset and cross-cut losses are necessary
registered conditions. They are not mathematical consequences and remain predictions until
the locked development execution.

## Counterexamples and boundaries

- Exact append-zero compatibility for $u>0$ remains refuted; only the approximate theorem is
  active.
- $q_n<1$ for every finite $n$ without a uniform $q<1$ would not prove the infinite theorem.
- A one-state inspected defect cannot replace the uniform $u\lambda^{n+1}$ bound.
- Bounded cascade output does not imply convergence, intelligence, or utility.
- Several nested maximal SCCs cannot occur in one fixed directed graph; L6 must use typed
  scale-indexed graph views and explicit maps.

## Dimensionless audit

All states, gains, $q$, $\lambda$, defects, cosine evidence, softmax logits, and probability
outputs are dimensionless. Observation units cancel in cosine evidence; any physical neural
quantity would require an external reference scale before entering the core.

## Verdict

L1 and L2 are closed conditional theorems under the stated domains. L3--L6 retain empirical or
design status.
