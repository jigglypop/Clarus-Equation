# Mathematics lane: quantum-neighbor bootstrap and residual dark sector

Status: COMPLETE

## Finding first

For the declared facilitated Lindbladian, an active neighbor can exactly
enable an inactive node's jump.  If the Hamiltonian is diagonal in the
occupation basis, diagonal states remain diagonal and their probabilities are
an exact finite-state CTMC.  The all-inactive state is absorbing; with the
contract's finite graph and every $\gamma_i>0$, it is reached with probability
one.  A directed cycle is necessary for indefinite mutual reactivation in the
support graph, but neither a cycle nor an SCC establishes positive survival or
a Perron threshold.

The CE independent-Poisson branching equation is recovered only after a
separate rare, independent, non-excluding generation limit (or is declared as
a model).  It is not the exact process (C1)--(C2).  In particular, a quantum
network has exclusion and correlations, and a random exponential parent
lifetime produces a mixed-Poisson rather than Poisson offspring law.  The
strongest surviving bridge is therefore a conditional theorem for a declared
classical record/branching approximation; selection and the predecessor's
residual-sector physical map remain additional axioms.

## M1: diagonal sector and the exact CTMC

Let $x=(x_i)\in\{0,1\}^{V}$ label the occupation basis and let
$e_i$ flip its $i$th zero to one.  Assume

$$
H=\sum_x E_x |x\rangle\langle x|.
\tag{M1}
$$

Then $[H,\rho]=0$ for occupation-diagonal $\rho$.  Each $L_{i\leftarrow j}$
maps an occupation basis vector either to zero or to another such vector, and
each $R_i$ does the same.  Its gain and anticommutator terms are diagonal on a
diagonal $\rho$.  Hence this diagonal sector is invariant.  Writing
$p_x=\langle x|\rho|x\rangle$, the exact generator has transitions

$$
x\longrightarrow x+e_i\quad\hbox{at rate}\quad
b_i(x)=(1-x_i)\sum_j\kappa_{ij}x_j,
\qquad
x\longrightarrow x-e_i\quad\hbox{at rate}\quad d_i(x)=\gamma_i x_i.
\tag{M2}
$$

The $j=i$ summand is identically zero because $\sigma_i^+n_i=0$; it can be
retained in (M2), where $(1-x_i)x_i=0$.  Thus

$$
(\mathcal Gf)(x)=\sum_i b_i(x)[f(x+e_i)-f(x)]
+\sum_i d_i(x)[f(x-e_i)-f(x)].
\tag{M3}
$$

This is a classical result about the stipulated diagonal sector, not a claim
that a general quantum state is classical.  If $H$ has a coherence-generating
term, e.g. $H=\Omega\sigma^x_1$, then
$-i[H,|0\rangle\langle0|]$ is off diagonal at first order.  Population closure
is then false (P0 for any unconditional closure claim).

## M2: exact hierarchy, vacuum, and energy

For $p_i=\langle n_i\rangle$ and $C_{ij}=\langle n_i n_j\rangle$ in this
diagonal process, (M3) gives exactly

$$
\dot p_i=\sum_j\kappa_{ij}(p_j-C_{ij})-\gamma_i p_i.
\tag{M4}
$$

The correlations have their own equations containing triples; consequently
the first moments do not close.  The explicit state $x=(1,1)$ gives activation
gain zero for node 1, whereas the false linear equation would contribute
$\kappa_{12}$ (for a two-node example).  This is a complete counterexample to
the mean-field/branching replacement at high occupancy.

At $x=0$, all $b_i(0)$ and $d_i(0)$ vanish, so the vacuum is absorbing.  On a
finite graph with all $\gamma_i>0$, a sequence of positive-rate decay jumps
from every nonvacuum state reaches zero.  Any closed communicating class that
contained a nonvacuum state would therefore contain all its successive decay
states and ultimately zero.  Since zero cannot leave, it is the sole closed
class; finite-CTMC theory gives absorption with probability one.  Long
metastable residence is not permanent bootstrap.

The word `enables` is not `supplies energy`.  For diagonal $H$, the system
energy changes under the dissipators as

$$
\frac{d}{dt}\langle H\rangle=
\sum_{i,j}\operatorname{tr}\!\left[H\mathcal D[L_{i\leftarrow j}]\rho\right]
+\sum_i\operatorname{tr}\!\left[H\mathcal D[R_i]\rho\right].
\tag{M5}
$$

An upward jump can raise $E_x$; its energy is exchanged with the bath, pump,
or drive used to obtain the Markov generator.  Arbitrary positive rates in
(C1) do not specify that source or detailed balance.  Therefore ex-nihilo or
perpetual-bootstrap language is P0 for (C1)--(C2) without an explicit energy
accounting completion.

## M3: graph statements

An edge $j\to i$ with $\kappa_{ij}>0$ says that $j$ can enable $i$ only while
$j$ is active and $i$ is vacant.  An acyclic support graph has a topological
order and cannot support a directed reactivation loop.  Conversely an SCC
provides paths for mutual reactivation, but it does not provide a numerical
growth law.  A two-node cycle with $\kappa_{12}=\kappa_{21}=\epsilon>0$ and
$\gamma_1=\gamma_2=1$ is an SCC for every $\epsilon$, yet for a declared
generation window $\tau$ its independent-branching row mean is
$D=\epsilon\tau<1$ whenever $\epsilon\tau<1$.  It is subcritical in that
approximation and, by M2, still absorbs almost surely when finite.

Thus `SCC implies supercritical persistence` is P0 false.  A cycle is only a
support-graph necessity for recurrent facilitation, not sufficiency for a
survival phase; a supercritical class must additionally be reachable from the
seed in an infinite-volume/generation model.

## M4: controlled branching/Poisson limit

Fix a generation window $\tau$ and orient the next-generation matrix by

$$
A_{ji}=\mathbb E[\hbox{number of type $i$ children of one type $j$ parent}].
\tag{M6}
$$

It follows that $A$ is dimensionless.  In the idealized fixed-window model in
which a type-$j$ parent is externally retained for all $\tau$, each target is
an independent fresh offspring opportunity, and a type-$i$ birth clock has
constant rate $\kappa_{ij}$, then $N_{ji}\sim\operatorname{Poisson}(\kappa_{ij}\tau)$
and

$$
A_{ji}=\kappa_{ij}\tau.
\tag{M7}
$$

For a local many-cell realization this is a controlled approximation when
offspring occupy fresh cells, each collision/exclusion probability tends to
zero, parent--parent overlap tends to zero, bath records decorrelate between
windows, and coherent terms are secularly negligible.  A scaling may take
many potential targets with individually small hazards while holding the row
mean finite.  These are additional limiting hypotheses, not implications of
(C1).

The counterexamples are exact.  A two-level target can activate only once,
so its count is Bernoulli-like rather than Poisson.  Shared targets make two
parents compete; collective jumps such as $\sqrt c\,\sigma_1^+\sigma_2^+$
create correlated children; and collisions make (M4) nonlinear.  If the
parent lifetime $T\sim\operatorname{Exp}(\gamma_j)$ rather than being fixed,
then conditional on $T$ a rate-$\kappa$ offspring count is Poisson, but

$$
\mathbb E N=\frac{\kappa}{\gamma_j},\qquad
\operatorname{Var}N=\frac{\kappa}{\gamma_j}+
\left(\frac{\kappa}{\gamma_j}\right)^2,
\tag{M8}
$$

which is not Poisson.  Different child types share $T$ and are correlated.
For a truncated window, the mean is
$\kappa_{ij}(1-e^{-\gamma_j\tau})/\gamma_j$, but the shared random stopping
time still spoils independent Poisson offspring.  A fixed-window construction
or an explicit age-structured branching process is required.

For an actual multitype Galton--Watson process with independent offspring
vectors and finite mean $A$, extinction from a seed is governed by its
offspring generating functions.  In the independent Poisson case,

$$
q_j=\exp\!\left[\sum_iA_{ji}(q_i-1)\right].
\tag{M9}
$$

For an irreducible, nonsingular class, nonzero survival occurs precisely when
$\rho(A)>1$; in a reducible graph, a seed must reach such a class.  This
Perron theorem does not apply to the finite CTMC of M2 or to a process whose
offspring assumptions fail.

## M5: uniform sector and the CE scalar root

The imposed condition

$$
A\mathbf1=D\mathbf1
\tag{M10}
$$

says every parent has the same total mean number of children.  It makes the
uniform extinction ansatz reduce (M9) to $q=\exp[D(q-1)]$, and for $D>1$ the
minimal root is

$$
q_{\rm ext}=-\frac{W_0(-De^{-D})}{D}.
\tag{M11}
$$

With the independently supplied CE model/readout value
$D=D_{\rm eff}=3.1777584234$, fixed-point iteration from zero gives
$q_{\rm ext}=0.04864671964$.  This verifies only the scalar branching model.
(M7) instead says $D=\tau\sum_i\kappa_{ij}$ under uniform row sums; it does
not select $d+\delta$, Hodge dimension, $D_{\rm eff}$, or a cosmological
parameter.  The identification $D=D_{\rm eff}$ remains a model/readout axiom
(P1 if stated as a microscopic derivation).

## M6: selection and the residual bridge

A Lindblad semigroup gives the unconditional density operator.  To speak of
a selected execution history requires a declared environment/apparatus,
record algebra, and quantum instrument $\{\mathcal I_r\}$ (equivalently a
specified monitored unravelling).  The same master equation admits different
unravellings, so its jump symbols alone do not choose the record or its
selected/nonselected split.  A classical CTMC/branching description further
requires decoherence of the occupation basis, a system--environment split,
and a Markov/secular coarse-graining limit compatible with the locality scale
of the graph.

Local neighbor triggering must arise from a local Hamiltonian/open-system
limit with finite propagation speed (or a lattice Lieb--Robinson-type bound);
an arbitrary long edge is only a graph rule, not a relativistic local theory.
The predecessor's P0 counterexample survives unchanged: conditioning on a
recorded outcome does not add the unrecorded instrument outcome as stress in
the selected branch.  Consequently a nonselected-record-to-residual-field map,
its energy transfer/current, and its no-double-counting rule are still new
physical-map axioms.  Given those axioms, the predecessor's conditional
massive-scalar dust and constant-offset vacuum theorems remain compatible;
neither (C1)--(C2) nor this branching approximation fixes their abundance.

## Severity ledger and independent reproduction

| ID | Finding | Severity | Required closure |
|---|---|---|---|
| M1 | Diagonal $H$ gives the exact CTMC (M2); coherent $H$ breaks it. | P0 for unconditional population-closure wording | Declare diagonal/decohered sector or analyze coherences. |
| M2 | Vacuum absorbs all finite networks with $\gamma_i>0$; energy source is not encoded by enabling. | P0 for perpetual/ex-nihilo wording | State bath/drive and total energy accounting. |
| M3 | Cycles/SCCs give support recurrence only, never a threshold. | P0 for SCC-implies-survival | Supply a valid infinite branching/contact-process limit. |
| M4 | Poisson/Perron requires fresh independent offspring and fixed-window or age structure. | P1 for C1-to-Poisson derivation | Prove the stated scaling/error bounds. |
| M5 | $D_{\rm eff}$ is not fixed by microscopic rates. | P1 for derivation wording | Independently derive or declare the readout axiom. |
| M6 | Instrument, locality, residual map and conservation are separate. | P1 | Give microscopic monitored model and conserved physical map. |

Reproduction: `.codex/hooks/python.cmd python _workspace/ce/quantum-neighbor-bootstrap-dark-sector-20260825/artifacts/verify_quantum_neighbor_bootstrap.py`.
