# Clarus cell origin timeline gate

- passed: `True`
- structural window: `4.5`-`3.7` Ga
- evidence-by window: `3.8`-`3.4` Ga
- named candidate window: `4.3`-`3.7` Ga
- minimal form: semi-permeable boundary + autocatalytic core + heritable copying template + gradient ports + division threshold

## timeline

| window | Ga | confidence | role | note |
|---|---:|---|---|---|
| `habitable_earth_window` | 4.50-3.90 | `broad_external_review_window` | `outer_possible_window` | Earth habitability is the outer bound, not evidence of a Clarus cell. |
| `structural_clarus_cell_possible` | 4.50-3.70 | `theory_window_inside_habitability_to_biosignature` | `possible_first_closure` | A template-bearing protocell can only be placed as a structural possibility in this interval. |
| `biosignature_boundary` | 3.80-3.40 | `geological_evidence_window` | `life_present_by_then` | Microbial biosphere or biosignature evidence constrains life as present, not the exact first cell. |
| `luca_like_cell_candidate` | 4.30-3.70 | `model_dependent_phylogenetic_window` | `first_named_biological_candidate` | LUCA-like placement is model dependent and is not identical to origin of life. |

## morphology

| term | required | form | equation role | failure if absent |
|---|---|---|---|---|
| `boundary_membrane` | `True` | semi-permeable vesicle or compartment wall | `B_boundary - L_leak` | identity diffuses into open chemistry |
| `autocatalytic_core` | `True` | reaction set that increases its own enabling components | `A_auto(X,E)` | growth under dilution fails |
| `copying_template` | `True` | sequence, polymer, or heritable state copied with bias | `C_copy(X)` | mass may persist but lineage distinction collapses |
| `gradient_ports` | `True` | selective influx/efflux through boundary or surface chemistry | `E_in - L_leak` | the unit cannot remain open while preserving itself |
| `division_threshold` | `True` | growth, instability, budding, or fission threshold | `Pi_C lineage projection` | no selectable recurrence across generations |
| `internal_state_memory` | `False` | chemical concentration or template composition that biases the next cycle | `m_n before neural memory` | minimum life still possible, but adaptive recursion is weak |
| `sensorimotor_surface` | `False` | chemotactic or taxis-like input-output coupling | `U_d -> b_d` | minimum self-reference remains, behavioral recursion absent |

## verdict

- A Clarus cell is the first physical unit that can carry the life triad across cycles.
- It is cell-like before it is neuron-like.
- Its earliest time is a possible interval, not a fossil-dated moment.
- The empirical proof boundary remains the same: reaction, boundary, copying, lineage, ablation.
