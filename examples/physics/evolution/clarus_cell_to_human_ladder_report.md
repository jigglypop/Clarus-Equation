# Clarus cell to human ladder gate

- passed: `True`
- primitive kernel: boundary + metabolism + heredity + regulated ports + recurrence
- human forms: `human_proliferative_clarus_cell`, `human_postmitotic_neural_clarus_cell`

## ladder

| order | stage | clade | score | recurrence | human Clarus | mechanism note |
|---:|---|---|---:|---|---|---|
| 0 | `template_bearing_protocell` | proto-life | 5 | `division` | `False` | minimum Clarus kernel: boundary, resource flow, autocatalysis, template copying, division |
| 1 | `luca_like_prokaryotic_cell` | cellular ancestor candidate | 6 | `division` | `False` | genome-metabolism-boundary loop gains stronger repair and regulated replication |
| 2 | `bacterial_or_archaeal_clarus_cell` | prokaryotic cell | 7 | `division` | `False` | sensor surfaces and regulatory memory make behavior-ready single cells |
| 3 | `eukaryotic_clarus_cell` | eukaryote | 9 | `division` | `False` | organelles split the primitive core into energy, genome, traffic, and degradation subloops |
| 4 | `metazoan_stem_or_somatic_clarus_cell` | multicellular animal | 10 | `asymmetric_division` | `False` | single-cell recurrence is embedded in tissue signals, differentiation, apoptosis, and stem pools |
| 5 | `vertebrate_specialized_clarus_cell` | vertebrate | 11 | `maintenance_plus_tissue_replacement` | `False` | cell identity is stabilized by endocrine, immune, neural, and tissue context |
| 6 | `human_proliferative_clarus_cell` | human dividing cell | 10 | `asymmetric_division` | `True` | stem, epithelial, immune, or repair-capable cells keep the primitive division loop under tissue control |
| 7 | `human_postmitotic_neural_clarus_cell` | human postmitotic neural cell | 11 | `postmitotic_maintenance` | `True` | neuron-like cells replace division recurrence with membrane excitability, synaptic state, glial support, repair, and long-lived maintenance |

## human operators

| operator | primitive source | human form | failure mode |
|---|---|---|---|
| `plasma_membrane_identity` | `boundary_retention` | lipid membrane, channels, transporters, receptors, adhesion | loss of excitability, osmotic identity, receptor-defined cell state |
| `mitochondrial_energy_closure` | `autocatalytic_core` | ATP/redox/calcium coupling to biosynthesis and maintenance | maintenance, firing, repair, and division cannot be paid for |
| `genome_epigenome_template` | `copying_template` | DNA sequence plus chromatin and regulatory state | cell identity and lineage memory drift |
| `vesicle_organelle_traffic` | `gradient_ports` | ER/Golgi/endosome/lysosome/autophagy traffic | resource flow and waste control decouple from identity |
| `cycle_or_maintenance_recurrence` | `division_threshold` | cell cycle in proliferative cells; repair/autophagy/synaptic turnover in postmitotic cells | no recurrence operator for keeping the cell as itself over time |
| `tissue_context_closure` | `population selection` | ECM, immune, endocrine, vascular, glial, and neighboring-cell signals | the human cell cannot be interpreted as an isolated protocell |

## verdict

Human Clarus cells do not abandon the protocell kernel.  They internalize it into organelles, genome/epigenome regulation, membrane signaling, quality control, and tissue context.  The key human upgrade is that recurrence can be cell division or long-lived maintenance.

The human Clarus cell therefore has two valid forms:

1. proliferative Clarus cell: recurrence by division under tissue control
2. postmitotic neural Clarus cell: recurrence by maintenance, repair, membrane/synaptic turnover, and glial/tissue support
