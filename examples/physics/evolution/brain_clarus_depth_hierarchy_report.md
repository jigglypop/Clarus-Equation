# Brain Clarus depth hierarchy gate

- passed: `True`
- minimal brain depth: `4`
- minimal brain hypothesis: `four_depth_brain`
- mind candidate depth: `5`

## layers

| depth | layer | recurrence projection | carrier | required | evidence |
|---:|---|---|---|---|---|
| 1 | `cellular_clarus_cell` | X_cell(t) -> same cell-type basin | neuron, glia, vascular, immune, epithelial/stem variants | `True` | human Clarus-cell gates: boundary/metabolism/identity/repair/support/recurrence |
| 2 | `tissue_support_field` | many cells -> stable metabolic/glial/vascular/tissue context | glia, vasculature, ECM, endocrine/immune context | `True` | human postmitotic neural Clarus cell needs tissue/glial support |
| 3 | `neural_circuit_recurrence` | coupled excitable cells -> recurrent activity state | weighted synaptic/electrical/chemical graph | `True` | C. elegans weighted routing and zebrafish recurrent activity closure |
| 4 | `organism_control_loop` | activity state -> behavior/body state -> new sensory and internal input | sensorimotor, autonomic, endocrine, action carrier loops | `True` | cross-species action carrier and mouse speed/wheel action split |
| 5 | `self_model_workspace` | organism-control state -> memory/planning/self-model -> future control policy | human-like workspace, reportability, autobiographical/self-state model | `False` | not closed by current local data; remains higher-cognition candidate |

## hypotheses

| hypothesis | depth | verdict | missing required | unsupported included |
|---|---:|---|---|---|
| `three_depth_brain` | 3 | `underfit_brain` | organism_control_loop | none |
| `four_depth_brain` | 4 | `minimal_brain_closure` | none | none |
| `five_depth_mind_brain` | 5 | `overextended_mind_claim` | none | self_model_workspace |

## interpretation

A brain is not just a pile of Clarus cells.  Current closure needs four nested recurrence projections: cellular self-maintenance, tissue support, neural circuit activity recurrence, and organism-control recurrence.  A fifth self-model/workspace layer is a mind/human-cognition candidate, not yet a closed brain requirement.
