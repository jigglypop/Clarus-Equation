# Indra causal-orbit closure

`Indra net` is a name for the following proposed AGI construction, not an
independent physical or neuroscientific fact.

## 1. Countable causal network

Let (V) be countable and let the nonnegative row-source kernel satisfy

\[
A_{ij}\ge0,\qquad \sup_i\sum_jA_{ij}<\infty.
\]

The independent-Poisson generating map is

\[
F_A(x)_i=\exp\!\left[-\sum_jA_{ij}(1-x_j)\right].
\]

Starting at zero, (x^{(n+1)}=F_A(x^{(n)})) increases coordinatewise and is
bounded by one. Row summability permits dominated convergence, so its limit is
the componentwise-minimal fixed point. This is the countable extension of the
existing finite CE theorem, under the stated assumptions.

## 2. Finite group-orbit quotient

Let a group \(\Gamma\) act on (V), with finitely many node orbits
\(O_1,\ldots,O_m\), and require

\[
A_{\gamma i,\gamma j}=A_{ij}.
\]

More generally, only the equitable block-sum condition is needed:

\[
\sum_{j\in O_\beta}A_{ij}=\bar A_{\alpha\beta}
\quad\text{for every }i\in O_\alpha.
\]

For the lift \((Lz)_i=z_\alpha\),

\[
AL=L\bar A,
\qquad
\boxed{F_A(Lz)=L F_{\bar A}(z)}.
\]

Thus the orbit-constant sector is exactly invariant and its minimal fixed point
obeys (q_A=Lq_{\bar A}). The full node count may grow without increasing the
quotient dimension (m). A transitive one-orbit action reduces to the existing
common-row-sum scalar theorem.

This does not imply that every fixed point is symmetric. For example, (A=2I)
has common row sum two but also nonsymmetric fixed points. The conclusion applies
to the minimal zero-started branch and orbit-invariant trajectories.

## 3. Why the infinite causal chain changes the SCC result

Consider

\[
V=\mathbb N,\qquad A_{i,i+1}=2.
\]

Every finite open truncation is acyclic and nilpotent; its extinction vector is
one. Nevertheless, the countable process has mean two children per generation
and positive survival probability while its type index moves forever forward.
Translation identifies every bulk node as one orbit and gives

\[
\bar A=[2],\qquad q=e^{-2(1-q)}\approx0.20318787.
\]

Therefore the finite-SCC reachability criterion cannot be applied directly to
an arbitrary countable graph. It is valid after an exact finite quotient, or in
the original finite-type setting. The limits “finite size to infinity” and
“generation to infinity” do not commute for the open truncations.

This is the precise form of an unbounded causal chain moving as one bounded
collective type.

## 4. Finite computation on an infinite carrier

For finite seeds (S), finite horizon (T), and finite outward degree, the
causal cone

\[
C_T(S)=\{j:\operatorname{dist}_{\to}(S,j)\le T\}
\]

is finite. With maximum outward degree \(\Delta\),

\[
|C_T(S)|\le |S|\sum_{k=0}^T\Delta^k.
\]

CE's energy budget additionally enforces \(|A_t|\le B_t\). Exact group execution
and sparse local execution must not be conflated: arbitrary index Top-K can
break equivariance. The proposed state decomposition is

\[
\boxed{x_t=Lz_t+e_t,
\qquad \operatorname{supp}(e_t)\subseteq C_t,
\qquad |C_t|\le B_t.}
\]

Here (z_t\) is the finite collective background and (e_t\) contains only
localized causal deviations. This makes “each node reflects the whole” precise
only to the extent that each node receives its orbit summary (z_t); lossless
reconstruction of the entire network is not implied.

## 5. Stability and approximate symmetry

Equitable defect

\[
\epsilon_{eq}=\sup_{\alpha,i\in O_\alpha}
\sum_\beta\left|\sum_{j\in O_\beta}A_{ij}-\bar A_{\alpha\beta}\right|
\]

bounds one-step closure error because the exponential on the nonnegative axis is
one-Lipschitz:

\[
\|F_A(Lz)-LF_{\bar A}(z)\|_\infty\le\epsilon_{eq}.
\]

Near criticality this error may be strongly amplified. Exact collective motion
requires equivariant inputs, initialization, coupling, and active selection.
For delayed signed neural dynamics, the Poisson Perron radius is not a stability
proof; a separate small-gain condition such as

\[
\operatorname{Lip}(\phi)
\max_q\sum_{r,\delta}|K_{qr\delta}|<1
\]

is sufficient for contraction of the corresponding update.

## 6. Status

- countable minimal fixed point under row summability: conditional theorem;
- equitable/orbit quotient closure: conditional theorem;
- finite causal cone at finite horizon: conditional theorem;
- finite quotient SCC/Perron classification: conditional theorem;
- (x=Lz+e) AGI runtime architecture: implementation hypothesis;
- exact symmetry of a real brain or environment: unsupported assumption;
- every part containing lossless information about the whole: metaphor unless a
  reconstruction/sufficient-statistic theorem is separately supplied.
