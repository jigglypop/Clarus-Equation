# External dataset requirements

- data root: `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data`
- ready gates: 0/5
- next action: No external empirical boundary is locally ready; acquire one dataset with the listed required fields, then run the corresponding next_script.

## readiness

| gate | readiness | candidate files | missing fields | next script |
|---|---:|---:|---|---|
| `zebrafish_continuous_movement` | 0.400 | 1 | `turn_or_heading_trace`, `fish_or_session_id`, `alignment_quality_or_sync_marker` | `zebrafish_timestamp_certified_continuous_gate.py` |
| `c_elegans_empirical_trial_behavior` | 0.000 | 0 | `trial_id`, `stimulus_label_or_time`, `behavior_label_or_trace`, `worm_id`, `timebase` | `c_elegans_empirical_trial_behavior_gate.py` |
| `drosophila_trial_dynamics` | 0.000 | 0 | `trial_id`, `timebase`, `neural_activity_or_spikes`, `behavior_label_or_trace`, `stimulus_or_task_epoch`, `celltype_or_region_mapping` | `drosophila_trial_dynamics_gate.py` |
| `life_empirical_origin` | 0.000 | 0 | `reaction_network_or_sequence`, `autocatalysis_or_growth_measure`, `boundary_or_compartment_condition`, `copying_or_template_measure`, `control_or_ablation_condition` | `life_empirical_origin_gate.py` |
| `mammalian_action_replication_or_perturbation` | 0.000 | 0 | `registered_sessions_or_subjects`, `spike_or_activity_matrix`, `action_targets`, `region_or_probe_metadata`, `perturbation_or_larger_panel_indicator` | `mammalian_phi_action_replication_gate.py` |

## details

### zebrafish_continuous_movement

- reason: Discrete bridges pass, but e2-to-continuous tracking alignment is missing.
- ready: `False`
- found fields: `e2_frame_timestamp`, `e2_resampled_speed_or_position`

| candidate file | size bytes | keyword hits |
|---|---:|---|
| `data\evolution\clarus_cell\41593_2022_1180_astrocyte_screen_table2.xlsx` | 2793442 | e2 |

### c_elegans_empirical_trial_behavior

- reason: Connectome proxy is supported, but empirical stimulus-behavior trials are absent.
- ready: `False`
- found fields: none

### drosophila_trial_dynamics

- reason: Adult FlyWire structural loop is closed, but trial dynamics are absent.
- ready: `False`
- found fields: none

### life_empirical_origin

- reason: Toy life triad is positive, but origin-of-life evidence is not tested.
- ready: `False`
- found fields: none

### mammalian_action_replication_or_perturbation

- reason: Mouse Phi_action and carrier split are candidate-panel/mechanism candidates.
- ready: `False`
- found fields: none

