# BA-TR10 mathematics note

With `B_hs=1`, equal hidden state, and a deterministic hidden-permutation
equivariant transition and learning rule, equality of all hidden rows is an
invariant. Max-relative competition then maps the tie to zero and any
Hebb/Oja rule updates all rows equally. Deterministic local dynamics therefore
cannot select a unique hidden coordinate from this initial condition.

Revision 1 supplies no fixed edge code. It uses an exchangeable random
variable only at a real packet arrival. Because `tanh(xi)` is symmetric,
`E[zeta]=1`; because `sigma<1`, `zeta` is strictly positive and bounded. If an
occupied row has `r=1` and a free row has `r=0`, then the worst occupied/free
ratio is

\[
e^{-\lambda}\frac{1+\sigma}{1-\sigma}<1
\]

under the contract inequality. This closes the unbounded-jitter counterexample
for one allocation pass. The Oja update is coordinate-local in post activity,
delivered pre activity, and the current synaptic weight. Its finite-run mapping
is a random permutation, not a deterministic identity theorem.

