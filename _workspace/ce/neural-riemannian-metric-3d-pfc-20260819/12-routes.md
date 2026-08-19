# Dependency-ordered route portfolio

Status: COMPLETE

## 1. Execution dependency

The run is a strict directed acyclic graph:

$$
\text{source/schema lock}
\longrightarrow
\text{Gate A numerical geometry}
\longrightarrow
\text{Gate B synthetic recovery and null calibration}
\longrightarrow
\text{Gate C PFC acquisition}
\longrightarrow
\text{PFC feasibility}.
\tag{31}
$$

A downstream node is not executed after an upstream failure. Skipping a sealed
biological route after a failed estimator gate is a correct scientific result,
not an incomplete computation.

## 2. Route matrix

| Order | Route | Required inputs | Independent unit | Primary endpoint | Entry gate | Exit status |
|---:|---|---|---|---|---|---|
| 1 | `R-KERNEL-3D` | analytic $h$, flat pullback, curved $g$, chart transforms | analytic case and fixed mesh | registered numerical errors | contract + math audit | `PASS` or `FAIL_NUMERICAL_GATE` |
| 2 | `R-SYNTH-3D` | evaluator-hidden spatial $W$, six-component field, independent paths | synthetic dataset/circuit | full-field recovery plus held-out NLPD | kernel PASS | `ESTIMATOR_PASS` or `ESTIMATOR_FAIL` |
| 3 | `R-NULL-3D` | six fixed null generators, 200 datasets each | synthetic dataset/circuit | per-family run-level false promotion | kernel PASS | `CALIBRATION_PASS` or `CALIBRATION_FAIL` |
| 4 | `R-PFC-WOJCIK` | official Dryad sessions, license, hashes, trial/session order | animal; sessions nested | frozen held-out trial/later-stage feasibility score | synth + null PASS | `PFC_FEASIBILITY_ONLY`, `ACCESS_BLOCKED`, or `FAILED_EXECUTION` |
| 5 | `R-PFC-CALANGIU` | official dlPFC multi-task archive and unit schema | animal; sessions nested | cross-task transfer score | synth + null PASS | same bounded statuses |
| 6 | `R-PFC-KIANI` | official area-8Ar population files | animal | held-out trajectory/choice score | synth + null PASS | same bounded statuses |
| 7 | `R-PFC-RIBBON` | matched anatomy, depth/layer, $W^s$, activity, perturbation, behavior | independently manipulated animal | full chain (1) | all typed inputs | `UNTESTABLE_MISSING_INPUT` in current sources |

`R-SYNTH-3D` and `R-NULL-3D` are computed in one frozen binary but maintain
separate result ledgers and decisions. The true and null seeds are disjoint.

## 3. Route R-KERNEL-3D

### Build products

- fixed-size `Sym3` and `Metric3` Rust types;
- exact symmetry and SPD rejection;
- eigendecomposition-based exponential, logarithm, square root, inverse square
  root, affine relative deformation, pullback, length, connection, and
  curvature kernels;
- deterministic per-case JSON records;
- an independent Python/NumPy or arbitrary-precision reference fixture which
  does not share the Rust implementation.

### Kill conditions

Any tolerance failure in Section 11 of `11-math.md`, a silent eigenvalue clamp,
different output under thread counts one and maximum, a nonzero curvature call
for the folded Euclidean ribbon or nonlinear flat pullback, or failure to
detect the curved field stops the run before synthetic outcomes.

## 4. Routes R-SYNTH-3D and R-NULL-3D

### Data separation

Each dataset owns independent graph, exact predeclared baseline, fit, validation,
outer-test, and evaluator-truth seeds. The binary serializes generator truth to
a hash-linked evaluator file only after candidate predictions have been
written. The fitter API accepts no truth object or generator label.

### Machine outputs

- one immutable manifest containing source hashes, Cargo.lock hash, release
  binary hash, compiler version, target triple, thread count, and seeds;
- per-dataset/per-circuit candidate scores before aggregation;
- selected regularization and optimizer status for every candidate, including
  the 12-parameter direct response/diffusion alternative;
- six-component truth and estimate fields on the fixed evaluation mesh;
- curvature, length, and first-passage secondary records;
- recomputed sign tests, Holm family, Clopper-Pearson bounds, and route status;
- explicit nonfinite, SPD, rank, path-count, and optimizer failure codes.

The validator independently rebuilds decisions from raw per-circuit scores and
checks exact route counts: 200 true datasets, six times 200 null datasets, 64
circuits per dataset, ten controls, and the fixed path/direction counts. It
recomputes the nine superiority wins, the `0.01`-shifted direct noninferiority
wins, their ten-test Holm family, and every per-null-family promotion count and
Clopper-Pearson upper bound.

### Outcome boundary

Passing both routes means only that the frozen six-component structural metric
family can be recovered and selected under its own generator while avoiding
selection under the six registered alternatives. Failing either route seals
all biological outcomes and reports the exact failing generator/control.

## 5. Route R-PFC-WOJCIK

This is the first biological route because it directly observes learning in
lateral PFC. It is not first computationally because its outcome is sealed by
Gate B.

After Gate B only, acquisition proceeds in three steps:

1. fetch and hash the official 4.56 KB README and repository code/license
   metadata;
2. fetch one smallest session archive for a schema-only parser fixture without
   calculating a neural or behavioral statistic;
3. freeze a PFC-specific amendment, audit it, then fetch only the sessions
   needed by that immutable split.

The default split keeps whole sessions intact and holds out the final
preregistered block within each animal. A training-only task-aligned chart may
use color, shape, and XOR axes. No neuron is aligned by identity across
sessions. The primary target must be a later held-out trial or behavior value,
not a restatement of the same covariance or decoder used to construct the
metric. Both animals must be reported separately; $N=2$ forbids population
confirmation.

## 6. Routes R-PFC-CALANGIU and R-PFC-KIANI

These routes are independent source checks, not substitutes for learning
longitudinality.

- Calangiu can test whether a frozen state-space metric transfers across three
  dlPFC task settings beyond rate, direct decoder, and eye-movement controls.
- Kiani can test simultaneous-population trajectory/choice prediction across
  three animals, but all animals were trained before implantation.

Neither route receives cortical fold, layer, structural-connectivity, or
full-chain status. Large archives are not downloaded if Wójcik already fails a
frozen feasibility estimator, unless the failure is source-specific rather
than estimator-wide.

## 7. Route R-PFC-RIBBON

The smallest decisive biological experiment must register, in the same
animals and region:

1. subject-specific PFC surfaces, ribbon thickness, depth/layer coordinates,
   atlas uncertainty, and valid normal-offset chart;
2. direct or intervention-identified structural connectivity for tracked
   units;
3. baseline activity and $g_0$ before a randomized plasticity manipulation;
4. post-manipulation $W^s$, activity, and $g_t$ for the same units;
5. independent known-force or stimulation responses to separate mobility,
   drift, and noise;
6. later unperturbed trajectories, behavior, and first-passage outcomes;
7. sham, anatomy-only, gain, noise, direct dynamics, W-surrogate, and
   flat-pullback controls.

No current public source in `10-sources.md` provides this package. Combining
unmatched MRI subjects with unmatched electrode animals is forbidden.

## 8. Rust performance route

The implementation is an isolated run-local binary crate. It uses packed
six-component `f64` fields and fixed 3-by-3 matrices in inner loops, preallocated
per-circuit buffers, and Rayon only across independent datasets or circuits.
Each worker writes to its indexed record; aggregation follows stable index
order. Release flags are `opt-level=3`, fat LTO, one codegen unit, and abort on
panic. GPU and `f32` paths are excluded from this run.

Input adapters are added only for the format actually acquired. The synthetic
binary needs no Python runtime. A PFC `.npy` adapter may use `ndarray-npy` after
schema lock; memory mapping is allowed only when array contiguity and endianness
are verified. Runtime benchmarks are descriptive and cannot alter science
parameters.

## 9. Stable-snapshot audits

1. **Formula audit:** `00-contract.md`, `10-sources.md`, `11-math.md`, and this
   file are frozen before any Rust scientific implementation.
2. **Kernel audit:** source, Cargo.lock, fixtures, oracle, and validator are
   frozen before numerical fixture execution.
3. **Outcome audit:** synthetic generator/fitter separation, seeds, counts,
   hashes, and decision recomputation are audited before the first true/null
   outcome.
4. **PFC audit:** a separate schema-derived amendment is required after Gate B
   and before any PFC statistic.

The routine kernel audit stores only source, `Cargo.lock`, one final fixture
JSON, one oracle JSON, and a short validation record. `target/`, copied
executables, intermediate fixture lineages, exhaustive mutation copies, and
release manifests are disposable local products. Exact executable provenance
is rebuilt as Gate A-LOCK only when Gate B is ready for one-shot execution.
This keeps the mathematical gate strict without making ordinary Git changes
depend on release-packaging archaeology.

An audit may narrow or reject a route. It may not inspect its outcome and then
change a formula, threshold, candidate, or split.
