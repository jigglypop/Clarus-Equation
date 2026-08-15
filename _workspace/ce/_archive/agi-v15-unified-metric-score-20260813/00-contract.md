# AGI V15 unified metric: independent scored validation contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v15-unified-metric-20260813

Mode: light follow-up validation. The predecessor's conclusions are referenced,
not silently inherited as evidence for this score.

## 1. Question

Does the implemented finite one-metric core satisfy its exact mathematical and
algorithmic contract on previously unused random seeds, does a supplied metric
improve an independently evaluated navigation cost over an identity-metric
baseline, and does any of that qualify the artifact as an autonomous AGI?

## 2. Frozen system under test

- `reality_stone/python/reality_stone/clarus/unified_metric.py` as present before
  this scored run.
- Frozen file SHA-256:
  `0599FC3B212F924424DE0675266881F8F1A6611D880382533708CD55F2529BE4`.
- No implementation change is allowed before the first scored execution.
- The core receives the metric tensor as an oracle input. It is not credited
  with inferring that tensor from observations.

## 3. Preregistered tests

### 3.1 Formal claims and killing fixtures

F1. Affine tensor transport: for $y=Jx+b$ and
$g_y=J^{-T}g_xJ^{-1}$, local quadratic, edge, and fixed-topology path costs
must agree to relative error at most $10^{-10}$.

F2. Fixed-chart spectral clipping is not generally affine covariant; a concrete
fixture must produce a covariance defect greater than $10^{-3}$.

F3. A static Riemannian cost is symmetric. The implementation must return equal
costs for a path and its reversal to relative error at most $10^{-12}$.

F4. A source-free symmetric diamond has no equivariant singleton goal. The
implementation must preserve the complete two-element minimizer set.

F5. Finite endpoint tensors do not identify a continuum metric between sample
points; two positive interpolants with identical endpoint values must have
different integrated lengths by more than $10^{-2}$.

The proof score is the number of claims whose symbolic argument is complete and
whose killing fixture reproduces, out of five. Numerical reproduction alone is
not promoted to a proof.

### 3.2 Held-out randomized correctness

- Seeds: integers 915000 through 915255 inclusive; 256 trials.
- Trial dimensions: 2--4; graph sizes: 5--9; connected undirected graphs.
- SPD eigenvalues are sampled inside $[0.25,4]$.
- An independent reference implementation computes endpoint-average edge cost,
  all-pairs shortest cost, candidate minimizers, and dimensionless surprise.
- Pass thresholds: 100% finite outputs, at least 99.9% exact path-cost and goal
  agreement, and maximum relative scalar error at most $10^{-10}$.

### 3.3 Affine and permutation OOD metamorphic tests

The same 256 trials are transported by previously unused non-orthogonal affine
maps and node permutations. No reprojection is allowed. Maximum relative cost
error must be at most $10^{-10}$ and minimizer/path costs must be equivariant.

An adversarial review performed before the benchmark runner and first scored
execution identified an omitted scale regime. Therefore the frozen gate also
includes the explicit positive-scale fixture
`[(0,0),(1e-16,0),(2e-16,0)]` on the complete identity-metric graph with source
2 and target 0. The reference optimum is the direct edge with cost $2e-16$.
The implementation must terminate within one second, return an acyclic path
from source to target, and agree with that cost to relative error $10^{-10}$.
This addition and its rationale are part of the preregistration; the SUT remains
frozen and no prior scored execution is discarded.

### 3.4 Oracle-metric navigation utility

- Seeds: integers 916000 through 916255 inclusive; 256 two-route risk tasks.
- For each seed, start and target are $(0,0)$ and $(2,0)$; the two intermediate
  nodes are $(1,h_1)$ and $(1,-h_2)$ with independent
  $h_1,h_2\sim U[0.05,1.2]$. Their isotropic metric multipliers are independent
  log-uniform samples on $[0.25,16]$; endpoint multipliers equal one. The graph
  contains only the two length-two branches. No tie or hard-case resampling is
  allowed.
- Environment cost is fixed before agent evaluation as
  $c_{ij}=\sqrt{(z_j-z_i)^T((g_i+g_j)/2)(z_j-z_i)}$.
- V15 receives $g$ directly. The baseline receives the same graph and points but
  uses $g=I$.
- An independent exhaustive two-route evaluator supplies the optimum.
- Although both systems use the same search algorithm and call count, the V15
  arm receives the oracle metric while the identity arm does not. Therefore
  this is not a compute-matched learned baseline and is intentionally
  insufficient for A4.
- Report exact-choice accuracy and normalized regret
  $(C-C^*)/\max(C^*,10^{-12})$ for both systems.
- Utility GO requires V15 accuracy at least 99%, mean regret at most $10^{-10}$,
  and paired mean-regret improvement over the identity baseline at least 0.05.
- This is explicitly an oracle-geometry score, not learning or AGI evidence.

### 3.5 Autonomous-agent capability gates

A1. infer/update $g_t$ from raw observations without an oracle source tensor;
A2. execute a perception--action--environment closed loop;
A3. receive and solve a temporally delayed credit-assignment task;
A4. pass a compute-matched compositional/OOD task against a learned baseline.

Each gate scores one only if executable code and a scored test exist in the
frozen artifact. An interface name, certificate flag, or proposed equation does
not count.

## 4. Decision rule

- `MATH PASS`: F1--F5 score 5/5.
- `FINITE CORE GO`: randomized correctness, affine/permutation OOD, and the
  positive-scale killing fixture all pass.
- `ORACLE UTILITY GO`: all section 3.4 thresholds pass.
- `AGI GO`: FINITE CORE GO, ORACLE UTILITY GO, and A1--A4 score 4/4.
- Otherwise the AGI verdict is `STOP`, with passed lower-level gates reported
  separately. The internal AGI qualification percentage is
  $100\min(G_{core},G_{utility},A_1,A_2,A_3,A_4)$; it is a preregistered project
  gate, not a standardized intelligence measurement.

## 5. Contamination and interpretation controls

- No external data or literature is required; `10-sources.md` is skipped.
- Seeds and thresholds are written here before the benchmark implementation or
  first execution.
- A score defined by the same metric formula tests numerical realization of the
  declared geometry. It does not establish semantic world understanding.
- Existing repository failures outside the declared test slice are recorded but
  are not silently assigned to V15.
