# Clarus cell empirical closure

- passed: `True`
- claim level: `six_branch_empirical_partial_closure`
- weighted closure fraction: `0.910714`
- empirical passed gates: `crisprbrain_neuron_maintenance,depmap_operator_dependency,glia_support_operator,hpa_operator_blueprint,jump_morphology_operator,perturbseq_state_reconstruction`

## percent estimate

| scope | estimate |
|---|---:|
| `overall_clarus_cell` | `60-70%` |
| `postmitotic_neuron_D_Q_R_branch` | `60-70%` |
| `proliferative_cell_recurrence_branch` | `60-70%` |
| `perturbseq_transcriptomic_state_branch` | `55-65%` |
| `glia_tissue_support_context_branch` | `60-70%` |
| `subcellular_operator_blueprint_branch` | `70-80%` |
| `image_morphology_operator_activity_branch` | `50-60%` |
| `jump_direct_mitochondrial_E_branch` | `20-30%` |
| `origin_cell_full_loop` | `30-40%` |
| `human_brain_full_mechanism` | `37-44%` |

## operator scores

| operator | level | fraction | empirical gates |
|---|---|---:|---|
| `B` | `empirical strong` | 1.000 | `depmap_operator_dependency,hpa_operator_blueprint,jump_morphology_operator` |
| `U` | `empirical strong` | 1.000 | `crisprbrain_neuron_maintenance,depmap_operator_dependency,glia_support_operator,hpa_operator_blueprint,jump_morphology_operator` |
| `E` | `empirical strong` | 0.946 | `crisprbrain_neuron_maintenance,depmap_operator_dependency,hpa_operator_blueprint,perturbseq_state_reconstruction` |
| `A` | `empirical strong` | 1.000 | `crisprbrain_neuron_maintenance,depmap_operator_dependency,hpa_operator_blueprint,jump_morphology_operator,perturbseq_state_reconstruction` |
| `I` | `empirical strong` | 0.946 | `depmap_operator_dependency,hpa_operator_blueprint,jump_morphology_operator,perturbseq_state_reconstruction` |
| `D` | `empirical strong` | 0.949 | `crisprbrain_neuron_maintenance,glia_support_operator,hpa_operator_blueprint,jump_morphology_operator,perturbseq_state_reconstruction` |
| `Q` | `empirical strong` | 0.949 | `crisprbrain_neuron_maintenance,glia_support_operator,hpa_operator_blueprint,jump_morphology_operator,perturbseq_state_reconstruction` |
| `S` | `empirical strong` | 1.000 | `glia_support_operator,hpa_operator_blueprint` |
| `R` | `empirical strong` | 1.000 | `crisprbrain_neuron_maintenance,depmap_operator_dependency,glia_support_operator,hpa_operator_blueprint,perturbseq_state_reconstruction` |

## gate statuses

| gate | kind | status | branch |
|---|---|---|---|
| `mechanistic_toy_full_ablation` | `synthetic mechanism` | `passed` | primitive / abstract cell |
| `exact_mechanism_spec` | `formal specification` | `passed` | formal operator specification |
| `human_multiscale_synthetic` | `synthetic human stress model` | `passed` | human proliferative and postmitotic synthetic stress |
| `crisprbrain_neuron_maintenance` | `public empirical phenotype screen` | `passed` | human postmitotic neuron maintenance |
| `depmap_operator_dependency` | `public empirical fitness dependency` | `passed` | human proliferative recurrence |
| `psapko_neuron_rnaseq` | `public empirical transcriptome` | `parsed_no_promotion` | human PSAP-KO neuron transcriptome |
| `glia_support_operator` | `public empirical glia support screens` | `passed` | human glia support context |
| `hpa_operator_blueprint` | `public empirical subcellular atlas` | `passed` | human subcellular operator architecture |
| `jump_morphology_operator` | `public empirical image morphology profiles` | `passed` | human image-based morphology operator activity |
| `jump_mitochondria_channel_gate` | `public empirical image morphology profiles` | `parsed_no_promotion` | human direct mitochondrial image-channel E check |
| `jump_chemical_mito_positive_control` | `public empirical assay control` | `parsed_no_promotion` | human compound direct mitochondrial image-channel assay control |
| `perturbseq_state_reconstruction` | `public empirical transcriptome state` | `passed` | human Perturb-seq transcriptomic operator state |

## bottlenecks

- key operators still weak/open: `none at operator level`
- next gates:
  - `clarus_cell_jump_dose_or_cell_health_mitochondria_gate.py`
  - `clarus_cell_protocell_boundary_recurrence_gate.py`
  - `clarus_cell_neuron_glia_coculture_recurrence_gate.py`
