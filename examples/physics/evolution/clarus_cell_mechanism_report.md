# Clarus cell mechanism gate

- passed: `True`

## mechanism cycle

1. environmental gradient feeds resource ports
2. ports raise internal resource without erasing boundary
3. autocatalytic core converts resource into catalyst and membrane growth
4. template copying preserves lineage bias
5. membrane growth reaches division threshold
6. daughter compartments inherit catalyst/template/membrane state
7. selection acts on recurrence rate and heredity retention

## mechanisms

| mechanism | update | role | measurable proxy |
|---|---|---|---|
| `boundary_retention` | `leak = leak_rate * resource / (membrane + eps)` | keeps inside/outside distinction while allowing an open reactor | retention half-life, permeability, osmotic stability |
| `resource_porting` | `influx = port_rate * membrane * max(external - resource, 0)` | feeds metabolism without erasing compartment identity | monomer or nutrient uptake under gradient |
| `autocatalytic_core` | `dA = k_auto * catalyst * resource / (K + resource)` | turns inflow into self-maintenance and membrane growth drive | growth under dilution, catalytic amplification |
| `template_copying` | `dT = k_copy * template * catalyst * resource / (K + resource)` | preserves lineage-level distinction across cycles | template amplification, heritable sequence/state bias |
| `growth_division` | `divide when catalyst + template + membrane exceeds threshold` | turns a persistent unit into a selectable recurrence | division count, daughter retention, lineage growth rate |

## ablation summary

| condition | pass rate | persistent | heritable | compartmental | recurrent | mean mass | mean heredity gap | mean divisions | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `full_clarus_cell` | 0.869 | 1.000 | 1.000 | 1.000 | 0.869 | 0.533969 | 0.300000 | 0.869 | all required mechanism loops close |
| `no_boundary` | 0.000 | 0.319 | 1.000 | 0.000 | 0.000 | 0.314516 | 0.300000 | 0.000 | primary failure: compartment |
| `no_autocatalytic_core` | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.503197 | 0.300000 | 0.000 | primary failure: recurrence |
| `no_copying_template` | 0.069 | 1.000 | 0.081 | 1.000 | 0.950 | 0.505074 | 0.119330 | 0.950 | primary failure: heredity |
| `no_gradient_ports` | 0.025 | 1.000 | 1.000 | 1.000 | 0.025 | 0.728060 | 0.300000 | 0.025 | primary failure: recurrence |
| `no_division_threshold` | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.072226 | 0.300000 | 0.000 | primary failure: recurrence |

## mechanism verdict

- The Clarus cell is not just a membrane or just a replicator.
- It works only when retention, resource flow, autocatalysis, copying, and division are coupled.
- The first selectable unit is the whole cycle, not any single molecular component.
