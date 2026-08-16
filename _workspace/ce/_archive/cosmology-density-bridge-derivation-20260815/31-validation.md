# Density-bridge variational audit validation

Status: COMPLETE

Validation date: 2026-08-15

## Verdict boundary

The implementation integrity checks pass. This establishes that the code
reproduces the audited algebra, counterexamples, dimensions, and fail-closed
status policy. It is not a physical proof that the external input `D`, the
chosen potential, a baryon current, freeze-out normalization, or `Omega_m`
follows from CE. In particular, the result remains:

```text
critical_density_bridge = INCOMPLETE
physical_prediction = NONE
```

## Automated test result

Command and raw output:

```text
> uv run --extra dev python -m pytest tests/test_density_bridge_variational_audit.py -q
.............                                                            [100%]
13 passed in 1.01s
```

The tests cover the small-root residual and Hessian, the unstable `x=1`
endpoint, direct evaluation at the closed endpoint, the same-root offset
counterexample, weighted-event equality and inequality cases, the
`q*Omega_m` distinction, the conserved-dust no-go, static-scalar `w=-1`, the
dimension ledger, structured claim status, absence of observational target
data, and CLI fail-closed behavior.

## Default mathematical-audit CLI

Command and raw output (process exit code 0):

```json
{
  "approved_mathematical_checks_pass": true,
  "checks": {
    "action_is_dimensionally_consistent": true,
    "matter_fraction_not_critical_fraction": true,
    "mixed_era_constant_fraction_requires_transfer": true,
    "offset_changes_density_fraction": true,
    "offset_preserves_stationary_data": true,
    "small_root_locally_stable": true,
    "small_root_stationary": true,
    "static_scalar_is_not_dust": true,
    "unit_root_unstable": true,
    "weighted_covariance_identity": true
  },
  "claims": {
    "critical_density_bridge": "INCOMPLETE",
    "local_branch_stability": "THEOREM_LOCAL",
    "matter_composition": "CONDITIONAL_CONSTRUCTION",
    "physical_prediction": "NONE",
    "potential_choice": "MODEL_AXIOM",
    "variational_embedding": "THEOREM_EXISTENCE_CONSTRUCTION",
    "weighted_event_identity": "THEOREM_CONDITIONAL_IFF"
  },
  "counterexamples": {
    "additive_offset": {
      "baseline_fraction": -1.3877787807814457e-17,
      "baseline_offset": 0.04488663169518471,
      "hessian": 17.37861222845491,
      "root": 0.0486467196440282,
      "root_residual": -4.440892098500626e-16,
      "shifted_fraction": 0.19999999999999998,
      "shifted_offset": 0.2948866316951847
    },
    "matter_composition": {
      "branching_probability": 0.0486467196440282,
      "critical_density_bridge_status": "INCOMPLETE",
      "critical_density_fraction": 0.0243233598220141,
      "equals_branching_probability": false,
      "matter_composition_fraction": 0.0486467196440282,
      "status": "CONDITIONAL_CONSTRUCTION",
      "total_matter_fraction": 0.5
    },
    "unequal_conditional_energy": {
      "covariance": 0.046280216311903516,
      "covariance_difference": 0.04413327715134962,
      "direct_difference": 0.044133277151349626,
      "equal_conditional_means": false,
      "mean_weight_complement": 1.0,
      "mean_weight_event": 2.0,
      "probability": 0.0486467196440282,
      "total_mean_weight": 1.0486467196440281,
      "weighted_fraction": 0.09277999679537782
    }
  },
  "dimension_ledger": {
    "d": 0,
    "field_scale": 1,
    "kinetic_action_density": 4,
    "log_argument": 0,
    "passes": true,
    "potential": 0,
    "potential_action_density": 4,
    "potential_scale": 1,
    "required_action_density": 4,
    "x": 0
  },
  "exit_policy": "APPROVED_MATH_ONLY",
  "input": {
    "d": 3.1777584234099736,
    "d_status": "EXTERNAL_INPUT"
  },
  "is_physical_prediction": false,
  "physical_bridge_complete": false,
  "static_scalar_stress": {
    "energy_density": 1.0,
    "equation_of_state": -1.0,
    "pressure": -1.0
  },
  "stationary_branch": {
    "hessian": 17.37861222845491,
    "q": 0.0486467196440282,
    "residual": -4.440892098500626e-16,
    "unit_branch_hessian": -2.1777584234099736
  }
}
```

The tiny negative baseline fraction is binary64 cancellation around the exact
analytic value zero; the test applies an absolute roundoff tolerance.

## Physical-bridge-required CLI

Command:

```text
python examples/physics/density_bridge_variational_audit.py --require-physical-bridge
```

Raw JSON and process result:

```json
{
  "approved_mathematical_checks_pass": true,
  "checks": {
    "action_is_dimensionally_consistent": true,
    "matter_fraction_not_critical_fraction": true,
    "mixed_era_constant_fraction_requires_transfer": true,
    "offset_changes_density_fraction": true,
    "offset_preserves_stationary_data": true,
    "small_root_locally_stable": true,
    "small_root_stationary": true,
    "static_scalar_is_not_dust": true,
    "unit_root_unstable": true,
    "weighted_covariance_identity": true
  },
  "claims": {
    "critical_density_bridge": "INCOMPLETE",
    "local_branch_stability": "THEOREM_LOCAL",
    "matter_composition": "CONDITIONAL_CONSTRUCTION",
    "physical_prediction": "NONE",
    "potential_choice": "MODEL_AXIOM",
    "variational_embedding": "THEOREM_EXISTENCE_CONSTRUCTION",
    "weighted_event_identity": "THEOREM_CONDITIONAL_IFF"
  },
  "counterexamples": {
    "additive_offset": {
      "baseline_fraction": -1.3877787807814457e-17,
      "baseline_offset": 0.04488663169518471,
      "hessian": 17.37861222845491,
      "root": 0.0486467196440282,
      "root_residual": -4.440892098500626e-16,
      "shifted_fraction": 0.19999999999999998,
      "shifted_offset": 0.2948866316951847
    },
    "matter_composition": {
      "branching_probability": 0.0486467196440282,
      "critical_density_bridge_status": "INCOMPLETE",
      "critical_density_fraction": 0.0243233598220141,
      "equals_branching_probability": false,
      "matter_composition_fraction": 0.0486467196440282,
      "status": "CONDITIONAL_CONSTRUCTION",
      "total_matter_fraction": 0.5
    },
    "unequal_conditional_energy": {
      "covariance": 0.046280216311903516,
      "covariance_difference": 0.04413327715134962,
      "direct_difference": 0.044133277151349626,
      "equal_conditional_means": false,
      "mean_weight_complement": 1.0,
      "mean_weight_event": 2.0,
      "probability": 0.0486467196440282,
      "total_mean_weight": 1.0486467196440281,
      "weighted_fraction": 0.09277999679537782
    }
  },
  "dimension_ledger": {
    "d": 0,
    "field_scale": 1,
    "kinetic_action_density": 4,
    "log_argument": 0,
    "passes": true,
    "potential": 0,
    "potential_action_density": 4,
    "potential_scale": 1,
    "required_action_density": 4,
    "x": 0
  },
  "exit_policy": "REQUIRE_PHYSICAL_BRIDGE",
  "input": {
    "d": 3.1777584234099736,
    "d_status": "EXTERNAL_INPUT"
  },
  "is_physical_prediction": false,
  "physical_bridge_complete": false,
  "static_scalar_stress": {
    "energy_density": 1.0,
    "equation_of_state": -1.0,
    "pressure": -1.0
  },
  "stationary_branch": {
    "hessian": 17.37861222845491,
    "q": 0.0486467196440282,
    "residual": -4.440892098500626e-16,
    "unit_branch_hessian": -2.1777584234099736
  }
}
PROCESS_EXIT_CODE=2
```

Thus the executable cannot produce a false-green physical result merely because
the local variational and algebraic checks pass.

## Static checks

Commands and raw outputs:

```text
> uv run --extra dev ruff check examples/physics/density_bridge_variational_audit.py tests/test_density_bridge_variational_audit.py
All checks passed!
```

```text
> python -m compileall -q examples/physics/density_bridge_variational_audit.py tests/test_density_bridge_variational_audit.py
<no stdout; process exit code 0>
```

```text
> powershell.exe -NoProfile -ExecutionPolicy Bypass -File .codex/hooks/run.ps1 check _workspace/ce/cosmology-density-bridge-derivation-20260815 build
OK build
```

## Independent root-agent regression rerun

After reviewing the implementation, the root agent reran the new tests together
with the nearest existing fixed-point, dimensionless, and cosmology-ratio
regressions:

```text
> uv run --extra dev python -m pytest tests/test_density_bridge_variational_audit.py tests/test_bootstrap_solver.py tests/test_dimensionless.py tests/test_cosmology_ratio_audit.py -q
..................................                                       [100%]
34 passed in 8.40s
```

The independent algebra ledger was also rerun:

```text
> python _workspace/ce/cosmology-density-bridge-derivation-20260815/artifacts/verify_density_bridge_math.py
...
ALL DENSITY-BRIDGE MATH CHECKS PASSED
```

The standard CE regression bundle remained green:

```text
> uv run --extra dev python -m pytest tests/test_bootstrap_solver.py tests/test_dimensionless.py tests/test_layer_a.py tests/test_bridge_gates.py -q
..........................................................               [100%]
58 passed, 2 warnings in 19.20s
```

Both warnings are pre-existing PyTorch sparse-CSR warnings from
`tests/test_layer_a.py::TestNoise::test_noise_nonzero_in_wake`; neither concerns
the density bridge.

The standard harness programs all exited zero. Their status strings and key
quantities were:

- `bootstrap_solver.py`: `PASS`, fixed-point residual `2.08e-17`; its printed
  `Interpretation as Omega_b` remains a bridge label and is not promoted here.
- `scorecard.py`: 23 total rows, 12 scored rows, 11 PASS and 1 CAUTION; aggregate
  `CAUTION`. The largest scored caution was `Omega_b h^2` at `-1.80 sigma`.
- `run_validation.py`: bootstrap `PASS`, dimensional analysis `PASS`, scorecard
  `CAUTION`, overall `CAUTION`.
- `proof_completion_attempt.py`: it continued to report explicit obstructions
  for LO `V_cb` (`-6.58 sigma`), tree `V_us` (`+9.84 sigma`), and raw `A_s`
  (`+197.80 sigma`). These are unrelated pre-existing theory gaps and were not
  changed by this implementation.

The targeted Ruff rerun again printed `All checks passed!`.

## What remains physically unvalidated

- No microscopic action derives `D`, the species label, or a reaction law.
- No conserved baryon and partner currents are produced by this scalar alone.
- Equal conditional energy is checked as an iff condition, not derived from a
  symmetry or dynamics.
- No covariant freeze-out surface or total charge/entropy normalization is
  present.
- `Omega_m` remains an independent input, so the critical-density bridge is
  still incomplete.
- No observational quantity was used as a target and no active prediction was
  created.
