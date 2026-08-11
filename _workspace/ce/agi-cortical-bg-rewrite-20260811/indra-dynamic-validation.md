# Dynamic delayed orbit-quotient validation

## Outcome

The preregistered finite-cover mechanism passed its exact standalone gates.

- covers `N = 32, 64, 128, 256`: full delayed trajectory versus the lifted
  three-orbit quotient had maximum error `0.0`;
- quotient work stayed fixed at `9` scalar accumulation/activation units per
  step while `N` changed;
- a localized initial perturbation reconstructed the dense full reference with
  maximum error `2.7755575615628914e-17`;
- the largest active deviation slice contained `13` `(cell, orbit)` pairs,
  below the preregistered geometric upper bound `51`;
- cyclic translation equivariance error was `0.0` for a nonuniform random
  history and input;
- the sufficient all-spatial-mode small-gain bound was `0.31 < 1`;
- a zero-mean checkerboard perturbation decayed by more than `1e8` over 40
  steps, guarding against the counterexample in which only the homogeneous
  quotient mode is stable;
- snapshot continuation was bit-identical to uninterrupted execution;
- active-budget overflow raised an error; no index Top-K truncation occurred;
- zero-delay/same-tick edges were rejected.

The controls behaved in the required direction.  An untied cell bias, a
spatially varying matched-scale perturbation, and one-cell orbit-label
corruption broke the registered lift.  Open boundaries and index-first Top-K
broke translation equivariance.  These are negative controls, not supported
execution modes.

Validation command:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests\test_option_flow_gate.py `
  tests\test_shared_option_benchmark.py `
  tests\test_stn_hold_benchmark.py `
  tests\test_indra_causal_quotient.py `
  tests\test_orbit_quotient_network.py `
  tests\test_dimensionless.py -q
```

Result: `30 passed`.  Ruff passed on the new module, tests, and benchmark.  The
dimensionless checker completed successfully.

## Formal status

Promoted for this implemented model:

- delay-resolved translation quotient closure: conditional implementation
  theorem;
- finite-horizon sparse causal-cone reconstruction: conditional implementation
  theorem;
- positive-delay causality and fail-closed active budget: implementation
  invariant;
- full-mode contraction under the reported absolute small-gain bound:
  sufficient stability certificate.

Not promoted:

- arbitrary-state compression by spatial averaging;
- literal infinite-carrier execution or uniform-in-time finite work;
- learned discovery or maintenance of the symmetry;
- real-brain translation symmetry;
- task utility or AGI performance;
- BrainRuntime readiness.

The cyclic covers are growing finite periodic models.  They agree with an
unwrapped local lattice only before the causal cone wraps around; they are not
an append-only construction of one literal infinite graph.

## Readiness score

Using the preregistered 100-point rubric, the current result is `73/100`:

- formal/dynamic closure: `20/20`;
- local deviation fidelity: `18/25` (exact lane passed; no approximate-budget
  error curve yet);
- scaling: `12/20` (dimension and counted work are independent of `N`; no
  rigorous wall-time/memory scaling study yet);
- robustness/OOD: `13/15`;
- stability/safety: `10/10`;
- matched task relevance: `0/10`.

Verdict: `GO` as a standalone mechanistic prototype, `STOP` for BrainRuntime
integration.  The integration review threshold remains `80/100` with every
hard gate passing.
