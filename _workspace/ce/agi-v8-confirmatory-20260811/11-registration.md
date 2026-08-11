# V8 preregistration schema and chronology audit

Status: COMPLETE

## Decision

V8 may extend `sparse_causal_bridge_v7.json`; it does not need to duplicate the
entire V1--V7 merged object.  The existing loader recursively deep-merges
`overrides`, replaces all other top-level fields, and returns both the merged
object and the bytewise concatenation of every ancestor registration.  This is
exactly the mechanism used by V2--V7.

Inheritance is scientifically acceptable only if V8 explicitly says that V7
failed and remains closed.  `supersedes` means only “active protocol successor”;
it must not imply that V7's result was erased.  V8 is a new R1 confirmatory route,
not a second V7 route.  The V7 locked test `78100..78195` remains unopened and is
neither inherited as active data nor reused.

Git chronology does require a distinct preregistration commit.  The minimal
valid sequence is:

1. add and commit only `sparse_causal_bridge_v8.json` before the V8 runner or V8
   tests exist;
2. record that commit ID and the raw-file, ancestor-byte-chain, and canonical
   merged-object SHA-256 values;
3. implement the runner and tests in later commits, with tests asserting the
   three registration digests;
4. run validation once; open test only from the exact passing validation
   artifact with identical registration, implementation, and test hashes.

A self-contained copy would add hundreds of inherited fields, invite drift from
the frozen parent, and provide no stronger chronology.  If a portable snapshot
is desired, emit the canonical merged JSON as a derived artifact after locking;
do not use that snapshot as a second editable registration.

## Minimal exact V8 object

The following is the recommended structure.  Seed arrays are written compactly
here but the canonical JSON must enumerate every integer.

```json
{
  "schema_version": 8,
  "status": "locked_pre_implementation",
  "registered_on": "2026-08-11",
  "experiment": "sparse_causal_bridge_v8",
  "roadmap_stage": "G9-CB-R1-CONFIRMATORY",
  "extends": "sparse_causal_bridge_v7.json",
  "supersedes": "sparse_causal_bridge_v7",
  "runner": "parent_anchored_shrinkage_confirmatory",
  "parent_model_runner": "single_origin_free_rollout",
  "active_models_path": "parent_anchor.models",
  "active_gate": "parent_anchor_gate",
  "predecessor_status": "V7 validation failed its conjunction and remains closed; its locked test remains unopened. V8 is the separately selected R1 successor.",
  "change_reason": "R1 removes the unstable adaptive component and replaces per-episode inverse-RMSE consensus by one training-only scalar projection between the frozen V5 sparse trajectory and persistence. R1 was selected using disclosed V7 validation and fresh development seeds 79100..79355; none count as V8 evidence.",
  "development_data_disclosure": {
    "route_selection_sources": [
      "artifacts/agi/sparse_causal_bridge_validation_v7.json",
      "_workspace/ce/agi-v8-breakthrough-20260811/31-validation.md"
    ],
    "development_seed_block": {"first": 79100, "last": 79355, "count": 256},
    "development_result_role": "route selection and power planning only",
    "v7_test_opened": false,
    "v7_test_reuse_forbidden": true,
    "selected_algorithm": "P + g_sparse * (S - P)",
    "selected_sparse_gain": 0.7868543064870357,
    "selected_dense_control_gain": 0.7835668486813699,
    "selected_zero_bridge_control_gain": 0.882857758971467
  },
  "overrides": {
    "hypothesis": "On wholly fresh episodes from the frozen four-chart same-loading OOD family, training-only parent-anchored sparse shrinkage will improve both its frozen V5 sparse endpoint and persistence, retain positive bridge contribution against an identically fitted zero-bridge ablation, improve the frozen V7 controller checkpoint, and remain noninferior to an identically fitted same-probe dense shrinkage control.",
    "claim_boundary": "This is a confirmatory H20 forecast-controller test in one fully observed four-chart synthetic family. Passing supports only transfer of one training-fitted shrinkage coefficient and bridge contribution within this family. It does not establish sparse superiority over dense models, unseen-environment transfer, open-world causal discovery, a neural or brain mechanism, CE physics, autonomous agency, or AGI.",
    "data_roles": {
      "validation": {
        "environment": "ood",
        "seeds": [80100, 80101, "...", 80355],
        "steps_per_seed": 100,
        "intervention_pairs_per_source_per_seed": 0
      },
      "test": {
        "environment": "ood",
        "seeds": [81100, 81101, "...", 81355],
        "steps_per_seed": 100,
        "intervention_pairs_per_source_per_seed": 0
      }
    },
    "models": [
      "parent_anchored_sparse",
      "v5_sparse_parent",
      "persistence",
      "zero_bridge_shrinkage",
      "symmetric_dense_shrinkage",
      "frozen_v7_consensus",
      "frozen_v7_no_sparse_consensus",
      "stable_adaptive_dense"
    ],
    "parent_anchor": {
      "mode": "training_only_parent_anchored_trajectory_projection",
      "horizon": 20,
      "diagnostic_horizon": 5,
      "h5_is_gating": false,
      "calibration_steps": 80,
      "origin": 80,
      "observed_prefix": "x[0] through x[80]",
      "target_window": "x[81] through x[100]",
      "candidate_formula": "prediction = persistence + sparse_gain * (v5_sparse_parent - persistence)",
      "trajectory_combination": "Generate the complete frozen-parent H20 trajectory first, then combine it leadwise with the constant observed-origin persistence trajectory; the combined output is never recursively fed into either component.",
      "gain_fit_role": "inherited observational_train only",
      "gain_fit_seeds": [45100, 45101, 45102, 45103, 45104, 45105, 45106, 45107],
      "gain_fit_origins": [80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500],
      "gain_fit_windows": 176,
      "gain_fit_independent_trajectories": 8,
      "gain_fit_loss": "sum of squared per-chart training-scale-normalized H20 errors over all registered windows",
      "gain_formula": "clip(sum(<(S-P)/scale,(Y-P)/scale>)/sum(||(S-P)/scale||^2),0,1)",
      "zero_denominator_rule": "return 0.0",
      "gain_interval": [0.0, 1.0],
      "expected_sparse_gain": 0.7868543064870357,
      "gain_match_absolute_tolerance": 1e-15,
      "evaluation_prefix_refits_gain": false,
      "target_window_influence_on_gain": 0,
      "symmetric_dense_control": "Replace only the sparse mechanism by the inherited all-12-edge equal-probe dense latent mechanism; fit its own scalar gain with the identical training seeds, origins, loss, normalization, clipping, and persistence endpoint.",
      "expected_dense_control_gain": 0.7835668486813699,
      "zero_bridge_control": "Set the frozen sparse bridge matrix to zero while retaining its local coefficients; refit the pooled residual AR on the same observational-train episodes, then fit its own scalar gain by the identical rule.",
      "expected_zero_bridge_control_gain": 0.882857758971467,
      "normalization": "inherit V7 training-only chart scales unchanged",
      "metric": "For each seed, sqrt(mean(((truth-prediction)/training_scale)^2)) over 20 leads and four charts; arithmetic mean across seeds.",
      "paired_unit": "independent simulation seed",
      "paired_ci_method": "two-sided Student-t 95 percent endpoint",
      "critical_value_n256_df255": 1.9693105698498752,
      "future_states_available_to_predictor": false,
      "future_hidden_available_to_predictor": false,
      "derivative_convention": "Gate the recurrent dynamic component with the observed origin held fixed. Persistence is a frozen anchor with internal-state Jacobian zero. Report, but do not mislabel, sensitivity to perturbing x[80] as a separate diagnostic.",
      "leakage_contract": [
        "prediction API accepts immutable prefix arrays and frozen fitted objects only",
        "maximum observed state index is instrumented and must be 80",
        "mutating x[81:101] leaves every non-oracle prediction bit-identical",
        "mutating hidden states leaves every prediction bit-identical"
      ]
    },
    "parent_anchor_gate": {
      "validation_seeds_required": 256,
      "test_seeds_required": 256,
      "origins_per_seed_required": 1,
      "primary_horizon": 20,
      "h5_is_gating": false,
      "paired_ci95_lower_improvement_vs_v5_parent": {"operator": ">", "threshold": 0.0},
      "paired_ci95_lower_improvement_vs_persistence": {"operator": ">", "threshold": 0.0},
      "paired_ci95_lower_improvement_vs_zero_bridge": {"operator": ">", "threshold": 0.0},
      "paired_ci95_lower_improvement_vs_frozen_v7_consensus": {"operator": ">", "threshold": 0.0},
      "symmetric_dense_noninferiority_margin": 0.02,
      "paired_log_ratio_ci95_upper_vs_symmetric_dense_max": 0.01980262729617973,
      "stable_adaptive_dense_noninferiority_margin": 0.05,
      "paired_log_ratio_ci95_upper_vs_stable_adaptive_dense_max": 0.048790164169432,
      "expected_sparse_gain_match_required": true,
      "expected_dense_gain_match_required": true,
      "expected_zero_bridge_gain_match_required": true,
      "maximum_candidate_dynamic_component_pathwise_jacobian_radius": 0.98,
      "maximum_candidate_augmented_common_norm_bound": 0.98,
      "maximum_candidate_latent_ar_abs": 0.98,
      "nonfinite_prediction_count_max": 0,
      "maximum_component_prediction_absolute_value": 5.0,
      "maximum_prediction_norm_to_train_q99_ratio": 5.0,
      "maximum_observed_state_index": 80,
      "future_observation_reads_max": 0,
      "finite_metrics_required": true,
      "all_primary_clauses_required": true,
      "validation_and_test_each_must_pass_same_conjunction": true,
      "failure_meaning": "R1 parent-anchored shrinkage is not confirmed in this frozen synthetic family. Favorable means, H5 results, seed-win fractions, or individual clauses cannot override failure."
    },
    "resource_limits": {
      "max_cpu_seconds_target": 180.0,
      "external_download_bytes": 0,
      "write_trajectory_files": false,
      "numpy_only": true,
      "evaluation_probe_pairs": 0,
      "forecast_origins_per_seed": 1
    },
    "test_lock": {
      "open_only_after_validation_pass": true,
      "test_may_change_registration_gate_models_gain_or_code": false,
      "failed_validation_artifact_must_be_preserved": true,
      "failed_test_artifact_must_be_preserved": true,
      "failed_validation_keeps_test_unopened": true,
      "validation_and_test_seeds_must_be_retired_after_use": true,
      "require_identical_ancestor_byte_chain_sha256": true,
      "require_identical_canonical_merged_registration_sha256": true,
      "require_identical_canonical_v8_file_sha256": true,
      "require_identical_implementation_sha256": [
        "parent_anchor_rollout_bridge.py",
        "reliability_rollout_bridge.py",
        "free_rollout_bridge.py",
        "latent_causal_bridge.py",
        "sparse_causal_bridge.py"
      ],
      "require_identical_test_sha256": ["test_parent_anchor_rollout_bridge.py"],
      "require_v7_validation_failure_hash": true,
      "require_v7_test_unopened": true
    }
  }
}
```

## Loader and validator findings

The generic `_validate_registration` function is only a base structural check.
It verifies four charts, SCM array lengths, unique truth edges, contact-matrix
symmetry, disjoint registered seeds, and edge-budget adequacy.  It does not
validate `schema_version`, active runner/gate names, exact seed counts, gain
formula, chronology, hash locks, or conjunction semantics.  V8 therefore needs
a runner-specific validator, as V7 already does.

That V8 validator must additionally assert:

- the merged experiment is V8, active runner/gate names match, and exactly 256
  validation plus 256 test seeds are present;
- all V1--V8 data-role/control seeds and the disclosed `79100..79355`
  development block are disjoint from both V8 blocks;
- episode length is exactly `origin + horizon = 100`;
- the 22 origins are exactly `80..500` with stride 20, producing 176 windows
  from exactly eight inherited train trajectories;
- recomputation gives all three registered gains within `1e-15` and no
  evaluation prefix or target is passed to the gain fitter;
- H5 is exactly the first five rows of the one H20 prediction and is non-gating;
- zero-bridge and dense controls are built by the registered symmetric rules;
- Student-t comparisons use 256 seed-level values and the registered
  `t(255)=1.9693105698498752`, not 20 leads, 80 coordinates, or 176 training
  windows as replicates;
- strict `>` is used for the four superiority lower endpoints, while `<=` is
  used for the two log-ratio noninferiority upper endpoints;
- every primary boolean participates in one `all(checks.values())` conjunction;
- the test unlock checks the exact passing validation artifact plus unchanged
  registration-chain, canonical-merged, implementation, and test hashes.

The ancestor-byte-chain digest produced by `_load_registration` is useful but
is not literally a digest of the merged JSON.  Name it accordingly in V8.  Keep
the canonical merged-object digest as the semantic hash, and separately hash
the V8 file itself.  This removes the ambiguous V7 label
`merged_raw_sha256`.

## Gate interpretation

The four strict paired-superiority clauses preserve the checkpoints that led to
R1: V5 parent, persistence, bridge ablation, and the frozen V7 controller.  The
dense clause is deliberately noninferiority, because development results showed
near equality and do not support sparse-specific superiority.  The adaptive
dense comparison is also noninferiority; its own instability is reported but is
not allowed to contaminate the candidate-component stability gate.

Validation passing only unlocks test; it is not the final confirmation.  V8
passes only if validation and the still-locked test each independently pass the
same full conjunction.  No pooled 512-seed rescue, threshold amendment, model
replacement, or favorable secondary endpoint is permitted after either split
is observed.

CE_RUN=_workspace/ce/agi-v8-confirmatory-20260811
