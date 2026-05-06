# Self-reference origin ladder gate

- passed: `True`
- first minimum self-reference: `template_bearing_protocell`
- first named biological candidate: `luca_like_prokaryotic_cell`
- first behavioral self-reference: `chemotactic_bacterium_or_archaeon`
- first local neural self-reference proxy: `c_elegans_primitive_neural_proxy`

## interpretation

Self-reference recursion first closes structurally at a template-bearing protocell, not at a named animal.  If a named organismal candidate is required, the conservative label is LUCA-like prokaryotic cell.  Sensorimotor recursion begins at chemotactic bacteria/archaea-like cells, and neural recursion is only proxied locally at C. elegans.

## stage table

| order | stage | status | score | minimum | behavioral | neural | local evidence |
|---:|---|---|---:|---|---|---|---|
| 0 | `prebiotic_open_chemistry` | `pre_recursive` | 0 | `False` | `False` | `False` | Open reaction chemistry has no local identity term in the current ladder. |
| 1 | `autocatalytic_set_without_compartment` | `proto_recursive_incomplete` | 2 | `False` | `False` | `False` | Autocatalysis gives self-amplification but not protected identity or heredity. |
| 2 | `template_replicator_without_boundary` | `proto_recursive_incomplete` | 3 | `False` | `False` | `False` | Copying can carry sequence distinction, but the open system leaks identity. |
| 3 | `compartment_without_template_copying` | `proto_recursive_incomplete` | 3 | `False` | `False` | `False` | Boundary plus metabolism can persist, but no heritable template distinction closes. |
| 4 | `template_bearing_protocell` | `first_minimum_self_reference` | 4 | `True` | `False` | `False` | This is the first stage matching the local life triad: autocatalysis + boundary + copying. |
| 5 | `luca_like_prokaryotic_cell` | `first_named_biological_candidate` | 4 | `True` | `False` | `False` | A cell-level genotype-metabolism-boundary loop is organismal self-reference. |
| 6 | `chemotactic_bacterium_or_archaeon` | `first_behavioral_self_reference_candidate` | 5 | `True` | `True` | `False` | Action changes the next sensory/input distribution, so the organism loops through environment. |
| 7 | `unicellular_eukaryote_or_ciliate_like_cell` | `rich_single_cell_recursion` | 5 | `True` | `True` | `False` | Internal state and action are richer, but this is not the first recursion threshold. |
| 8 | `c_elegans_primitive_neural_proxy` | `first_local_neural_routing_proxy` | 6 | `True` | `True` | `True` | Local connectome gates support weighted chemical routing as primitive neural control. |

## empirical boundary

This gate is a structural Clarus-ladder audit, not a historical origin-of-life proof.
To promote the first boundary empirically, the next dataset must contain:

- reaction network or sequence table
- autocatalysis or growth measurement
- boundary or compartment retention measurement
- copying/template/heritable-state measurement
- ablation/control showing the triad is jointly necessary

## falsifiers

- A heritable, selectable, self-maintaining unit without boundary retention would lower the boundary.
- A bounded autocatalytic unit with no copying or heritable state would weaken the copying criterion.
- A verified sensorimotor loop before organismal cell closure would split behavioral recursion from life recursion.
