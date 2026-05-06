# Self-recursive cosmology closure suite

- passed: `True`
- passed gates: 17/17

| gate | script | return code | seconds | stdout tail |
|---|---|---:|---:|---|
| `research_audit` | `examples/physics/recursive_cosmology_research_audit.py` | 0 | 0.109 | Verdict: More self-recursion remains usable, but mainly as selector/readout recursion rather than a new core fixed point. Top targets: H0 q-selector, residual contraction cascade. / Wrote examples\physics\recursive_cosmology_research_audit_report.md / Wrote examples\physics\recursive_cosmology_research_audit_results.json |
| `h0_selector_self_reference` | `examples/physics/h0_readout/h0_recursive_selector_self_reference_gate.py` | 0 | 0.092 | max algebraic self drift = 1.349e-13 / Wrote examples\physics\h0_readout\h0_recursive_selector_self_reference_report.md / Wrote examples\physics\h0_readout\h0_recursive_selector_self_reference_results.json |
| `residual_cascade` | `examples/physics/residual_cascade_invariant_gate.py` | 0 | 0.090 | \| hemispherical identity \| 0.05963341 \| large-angle amplitude handle \| / Wrote examples\physics\residual_cascade_invariant_report.md / Wrote examples\physics\residual_cascade_invariant_results.json |
| `kernel_no_free_parameter` | `examples/physics/kernel_deformation_no_free_parameter_gate.py` | 0 | 0.132 | minimal kernel delta AIC = +0.000 / Wrote examples\physics\kernel_deformation_no_free_parameter_report.md / Wrote examples\physics\kernel_deformation_no_free_parameter_results.json |
| `d0_measure_transport` | `examples/physics/d0_measure_transport_gate.py` | 0 | 0.123 | \| recursive residual erasure \| 4.17888453e-47 \| closed dimensionless residual suppression \| / Wrote examples\physics\d0_measure_transport_report.md / Wrote examples\physics\d0_measure_transport_results.json |
| `early_late_measure_preservation` | `examples/physics/early_late_measure_preservation_gate.py` | 0 | 0.098 | invariant chi2/dof = 0.379/8 / Wrote examples\physics\early_late_measure_preservation_report.md / Wrote examples\physics\early_late_measure_preservation_results.json |
| `self_recursive_package` | `examples/physics/self_recursive_cosmology_package_gate.py` | 0 | 0.127 | Verdict: Selection/Bridge package, not Exact. / Wrote examples\physics\self_recursive_cosmology_package_report.md / Wrote examples\physics\self_recursive_cosmology_package_results.json |
| `h0_fisher_manifest_validate` | `examples/physics/h0_readout/h0_fisher_manifest_validate_gate.py` | 0 | 0.122 | manifest = C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\examples\physics\h0_readout\h0_fisher_io_examples\manifest.json / status = PASS / Verdict: source manifest is complete and channel files exist. |
| `h0_fisher_io_validate` | `examples/physics/h0_readout/h0_fisher_io_validate_gate.py` | 0 | 0.122 | validated = 4 / failed = 0 / Verdict: all Fisher/covariance JSON inputs passed validation. |
| `h0_fisher_io_regression` | `examples/physics/h0_readout/h0_fisher_io_regression_gate.py` | 0 | 0.125 | Delta q_F = 0.000e+00 / Delta H0 = 0.000e+00 / Verdict: Fisher and covariance JSON inputs are equivalent for the smoke channel. |
| `h0_fisher_io_batch` | `examples/physics/h0_readout/h0_fisher_io_batch_gate.py` | 0 | 0.148 | \| local_endpoint_fisher.json \| Local endpoint Fisher example \| 1.000000 \| 73.180689 \| 73.170 +/- 0.860 \| +0.012 \| / chi2/dof = 0.095160/4 / Verdict: batch Fisher/covariance ingestion path is ready. |
| `h0_fisher_real_readiness` | `examples/physics/h0_readout/h0_fisher_real_readiness_gate.py` | 0 | 0.130 | Verdict: Fisher/covariance IO is ready, but real covariance closure is still a data boundary. / Wrote C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\examples\physics\h0_readout\h0_fisher_real_readiness_report.md / Wrote C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\examples\physics\h0_readout\h0_fisher_real_readiness_results.json |
| `h0_real_covariance_requirements` | `examples/physics/h0_readout/h0_real_covariance_requirements_gate.py` | 0 | 0.109 |   "passed": true, /   "blocking_requirement_count": 6 / } |
| `h0_real_covariance_promotion_decision` | `examples/physics/h0_readout/h0_real_covariance_promotion_decision_gate.py` | 0 | 0.110 |   "passed": true, /   "real_ready_count": 0 / } |
| `prediction_ledger` | `examples/physics/self_recursive_cosmology_prediction_ledger_gate.py` | 0 | 0.099 |   "passed": true, /   "prediction_count": 7 / } |
| `claim_audit` | `examples/physics/self_recursive_cosmology_claim_audit_gate.py` | 0 | 0.110 |   "passed": true, /   "claim_count": 7 / } |
| `artifact_index` | `examples/physics/self_recursive_cosmology_artifact_index_gate.py` | 0 | 0.108 |   "passed": true, /   "artifact_count": 12 / } |
