# Quantum-neighbor bootstrap to residual dark sector

Status: COMPLETE

PREDECESSOR: `_workspace/ce/_archive/nonselected-quantum-dark-sector-20260825`

## 1. Question and claim ceiling

Starting from explicit local quantum dynamics, test the user's proposal that a
quantum next to another quantum can trigger its execution, so mutually coupled
quanta form a bootstrap structure. Determine exactly when this becomes a
positive population process, when that process admits a branching/Poisson
description, and how it can feed the already-audited selection--residual-dark-
sector chain.

The maximal authorized result is a conditional theorem for a declared local
facilitated quantum network plus a rigorous list of approximations required to
recover the CE multitype Poisson bootstrap. It may not claim that every quantum
interaction is execution, that a finite closed network survives forever, that
cycles alone imply supercriticality, that neighbor triggering supplies energy
without a bath/drive, or that this dynamics derives observed dark abundances.

## 2. Microscopic candidate

Let $G=(V,E)$ be a finite or locally finite directed graph. Node $i$ is a
two-level quantum subsystem with inactive state $|0\rangle_i$, active state
$|1\rangle_i$, raising/lowering operators $\sigma_i^\pm$, and number operator
$n_i=\sigma_i^+\sigma_i^-$. Define $\kappa_{ij}\geq0$ so that an active node
$j$ can trigger node $i$, and let $\gamma_i>0$ be the decay rate of node $i$.

The facilitated jump and decay operators are

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\,\sigma_i^+n_j,
\qquad
R_i=\sqrt{\gamma_i}\,\sigma_i^-.
\tag{C1}
$$

For a declared Hamiltonian $H$, the candidate Lindblad generator is

$$
\dot\rho=-i[H,\rho]
+\sum_{i,j}\mathcal D[L_{i\leftarrow j}]\rho
+\sum_i\mathcal D[R_i]\rho,
\qquad
\mathcal D[L]\rho=L\rho L^\dagger
-\frac12\{L^\dagger L,\rho\}.
\tag{C2}
$$

Rates have dimension $T^{-1}$ and jump operators $T^{-1/2}$ in the displayed
master-equation convention. Every exponential or probability object later
introduced must use dimensionless combinations such as $\kappa_{ij}\tau$ or
$\gamma_i\tau$ for a declared time scale $\tau$.

## 3. Registered meaning of execution and bootstrap

`Node $j$ executes node $i$` means only that the transition intensity from an
inactive $i$ to an active $i$ contains the factor $n_j$ and therefore vanishes
when $j$ is inactive. It does not mean that $j$ supplies the transition energy.
The Hamiltonian, bath, drive and total energy bookkeeping must identify the
physical source.

`Mutual execution` means the support graph of the directed rates contains a
strongly connected active block. `Bootstrap persistence` means survival of
the declared stochastic approximation with nonzero probability in its stated
infinite-volume or generation limit. A finite-state process with an accessible
absorbing vacuum is not called permanently persistent merely because it is
metastable for a long time.

## 4. Claims to prove, narrow or reject

### E1. Exact population dynamics

For $H$ diagonal in the occupation basis, determine whether (C2) preserves the
diagonal density-matrix sector and derive the exact classical transition
generator. Derive the exact first-moment equation and show explicitly whether
pair correlations prevent closure.

### E2. Seed and energy no-go

Prove whether the all-inactive vacuum is absorbing. Separate neighbor-enabled
transition logic from the bath/drive that supplies excitation energy. Reject
perpetual or ex-nihilo bootstrap wording if the declared model cannot support
it.

### E3. Graph recurrence

Determine what a directed cycle or strongly connected component guarantees.
Test necessity and sufficiency for mutual reactivation, and show that SCC
structure alone does not determine a survival threshold.

### E4. Branching and Poisson limit

Define a generation window $\tau$ and next-generation matrix. Identify the
conditions under which independent Poisson offspring with mean matrix
$A_{ji}$ can be derived or used as a controlled approximation. Audit exclusion,
collisions, finite lifetime mixing, correlations and coherent Hamiltonian
terms as counterexamples. Establish the Perron threshold only in the exact
branching model where its hypotheses hold.

### E5. CE scalar reduction

Test the uniform-sector reduction $A\mathbf1=D\mathbf1$ and the status of
$D=D_{\rm eff}=d+\delta$. Determine whether the microscopic candidate fixes
$D_{\rm eff}$, or whether this remains a model/readout axiom. Reproduce the
minimal fixed point and its branch choice without using cosmological targets.

### E6. Selection and residual bridge

Determine what additional quantum instrument, record algebra, coarse
graining, system--environment split and Markov/secular limit are required to
turn the facilitated network into the `끼임` stage. Preserve the predecessor's
P0 result: nonselected outcomes do not automatically gravitate in the selected
branch. The residual-field map and DM/DE EFT remain a separate physical-map
axiom followed by conditional theorems.

### E7. Locality and observation

Check finite-speed/locality requirements for neighbor triggering and list
experimental or observational falsifiers. Numerical examples may validate the
declared generator and approximations but cannot establish a universal
quantum ontology or cosmological abundance.

## 5. Controls and falsifiers

The registered negative controls are: no seed, a one-way acyclic graph, an SCC
with rates below threshold, a finite SCC observed for infinite time, coherent
$H$ that breaks population closure, collective jumps, high occupancy with
collisions/exclusion, random exponential parent lifetime instead of a fixed
generation window, and a source rule with missing bath/drive energy.

The claim is narrowed or rejected if complete positivity, locality, population
closure, energy accounting, branch independence, or the stated convergence
limit fails. A Poisson fit at one parameter point is not a derivation of an
offspring law.

## 6. Authorized implementation scope

The source and mathematics lanes may create run-local analytic/numerical
certificates. After the status gate, the ledger writer may update only the
smallest core/quantum/cosmology ledgers needed for the new bootstrap claim.
After those ledgers freeze, the paper writer may update
`docs/5_유도/00_선택과_접힘.md` and, only if required to keep the derivation
self-contained, the existing bootstrap reader guide or bootstrap canonical
document. Production code/tests may be added only for an approved focused
certificate; no full-suite or release workflow is authorized.
