# Human Clarus cell multiscale dynamics gate

- passed: `True`

## operators

| operator | state variable | primitive source | human role |
|---|---|---|---|
| `membrane_identity` | `membrane` | `boundary_retention` | keeps excitability, receptor state, osmotic identity, and adhesion |
| `mitochondrial_energy` | `energy` | `autocatalytic_core` | pays for biosynthesis, ion gradients, repair, firing, and division |
| `genome_epigenome` | `identity` | `copying_template` | keeps sequence plus chromatin/regulatory state as a cell-type template |
| `organelle_traffic` | `traffic` | `gradient_ports` | routes membrane, protein, nutrient, waste, and vesicle flux |
| `repair_autophagy` | `damage` | `daughter_retention_quality` | keeps damage below the identity-collapse boundary |
| `tissue_support` | `support` | `population_selection` | adds vascular, immune, endocrine, ECM, glial, and neighboring-cell context |
| `recurrence_operator` | `recurrence` | `division_threshold` | chooses division recurrence or postmitotic maintenance recurrence |

## form results

### human_proliferative_clarus_cell

| condition | pass rate | min energy | min identity | min membrane | max damage | recurrences | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| `full` | 1.000 | 0.720 | 0.860 | 0.777 | 0.160 | 3.858 | primary pressure: recurrence |
| `no_membrane_identity` | 0.000 | 0.720 | 0.745 | 0.148 | 0.160 | 0.000 | primary pressure: recurrence |
| `no_mitochondrial_energy` | 0.025 | 0.135 | 0.779 | 0.541 | 0.253 | 0.967 | primary pressure: recurrence |
| `no_genome_epigenome` | 0.000 | 0.720 | 0.538 | 0.777 | 0.160 | 2.158 | primary pressure: recurrence |
| `no_organelle_traffic` | 0.042 | 0.221 | 0.636 | 0.171 | 0.665 | 0.300 | primary pressure: recurrence |
| `no_repair_autophagy` | 0.000 | 0.164 | 0.447 | 0.037 | 1.034 | 0.458 | primary pressure: damage_inverse |
| `no_tissue_support` | 0.067 | 0.216 | 0.404 | 0.268 | 0.582 | 0.758 | primary pressure: recurrence |
| `no_recurrence_operator` | 0.000 | 0.720 | 0.860 | 0.777 | 0.160 | 0.000 | primary pressure: recurrence |

### human_postmitotic_neural_clarus_cell

| condition | pass rate | min energy | min identity | min membrane | max damage | recurrences | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| `full` | 1.000 | 0.720 | 0.860 | 0.772 | 0.174 | 45.000 | primary pressure: energy |
| `no_membrane_identity` | 0.000 | 0.720 | 0.724 | 0.088 | 0.177 | 0.967 | primary pressure: membrane |
| `no_mitochondrial_energy` | 0.000 | 0.033 | 0.480 | 0.206 | 0.555 | 5.233 | primary pressure: energy |
| `no_genome_epigenome` | 0.000 | 0.717 | 0.541 | 0.766 | 0.172 | 27.550 | primary pressure: identity |
| `no_organelle_traffic` | 0.033 | 0.027 | 0.296 | 0.025 | 1.027 | 1.383 | primary pressure: damage_inverse |
| `no_repair_autophagy` | 0.000 | 0.091 | 0.249 | 0.030 | 1.179 | 3.725 | primary pressure: damage_inverse |
| `no_tissue_support` | 0.000 | 0.004 | 0.074 | 0.006 | 1.075 | 0.000 | primary pressure: damage_inverse |
| `no_recurrence_operator` | 0.000 | 0.720 | 0.860 | 0.769 | 0.169 | 0.000 | primary pressure: recurrence |

## verdict

Human Clarus-cell closure is multiscale: a cell-level state remains itself only when membrane identity, mitochondrial energy, genome/epigenome identity, organelle traffic, repair/autophagy, tissue support, and the appropriate recurrence operator are all present.

The advanced rule is: primitive recurrence becomes branch-specific human recurrence.
A proliferative cell must close cell-cycle recurrence; a postmitotic neural cell must close maintenance recurrence.
