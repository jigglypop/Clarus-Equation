# Mathematical audit

Status: COMPLETE

## Small gain

In the product norm of the local-bank sup error and cloud sup error, the synchronous map is
bounded by the nonnegative matrix

$$
M=\begin{pmatrix}
\alpha_{\max}+g_{\times}&g_{CL}+g_{\times}\\
g_{LC}&\gamma
\end{pmatrix}.
$$

The extra term comes from the registered bilinear local/cloud interaction
$g_{\times}(h\odot c)$. On the exact state domain $[-1,1]$, its perturbation obeys

$$
\lVert h\odot c-\tilde h\odot\tilde c\rVert_\infty
\leq
\lVert h-\tilde h\rVert_\infty+\lVert c-\tilde c\rVert_\infty.
$$

Therefore the displayed matrix is a global bound, not a sampled Jacobian claim.

For $M\ge0$, $\rho(M)<1$ implies that $w=(I-M)^{-1}\mathbf1$ is positive and

$$
q_w=\max_i\frac{(Mw)_i}{w_i}<1.
$$

Thus the kernel is a contraction in the weighted block-max norm. This proves a unique state
for fixed input and fading dependence under causal input sequences. It does not prove utility.

For the frozen implementation candidate,

$$
M=\begin{pmatrix}0.82&0.26\\0.06&0.72\end{pmatrix},
\qquad q_w=0.935\overline{5}<0.95.
$$

All recurrent states, observations after reference-scale normalization, gains, and every
argument of $\tanh$ are dimensionless. The bilinear product is dimensionless as well.

## Deterministic monadic composition

Each update is a deterministic Markov kernel represented by a Dirac measure. Kernel identity
and Kleisli composition are exact. A two-step composed transition must equal two sequential
calls bitwise under the same frozen inputs. This is a software law, not an ontological claim.

## Comparator boundary

All models must expose equal-dimensional features and fit readout only on train episodes.
Capacity equality is not inferred from feature count alone; coefficient count, effective ridge
degrees of freedom, state count, and MAC remain separate ledgers.

## Verdict

The kernel theorem is implementable. Any empirical increment remains a prediction.
