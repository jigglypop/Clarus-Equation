# BA-TR11 direct metric/curvature contract

Date: 2026-08-22

Input is only the frozen 4x4 learned `H <- S` matrix stored in the BA-TR10
development artifact. No runtime endpoint, output weight, decoder, target, or
reward is opened.

The linear response `F(x)=Bx` has constant pullback `g=B^T B`; full-rank
matrices are intrinsically flat. A rank-deficient matrix is not assigned zero
curvature and is reported `CURVATURE_UNDEFINED_DEGENERATE` without ridge.

For the nonlinear diagnostic, let `P` be the top-two right-singular plane and

\[
F(u)=\tanh(BPu),\quad J=\operatorname{diag}(\operatorname{sech}^2(BPu))BP,
\quad g=J^TJ.
\]

Gaussian curvature is computed from the normal projections of the analytic
second derivatives by the Gauss equation. Every point must pass a fixed
positive-rank/condition gate; no regularization is permitted.

Required controls for every BA-TR10 development matrix: analytic Jacobian
matches central finite difference; linear full-rank curvature certificate is
zero; the uniform rank-one matrix is undefined; a hidden-row permutation
changes the labeled column winners but preserves `g` and `K`; a source-column
permutation accompanied by its chart transform preserves them; and a general
hidden rotation preserves `g(0)` but changes at least one nonzero-state
curvature value. The last control shows that nonlinear curvature can contain
information beyond one origin metric, while the row-permutation control shows
it cannot identify the stored hidden labels.

Claim ceiling: curvature is a derived nonlinear geometric signature of a
learned code, not the memory state itself.

