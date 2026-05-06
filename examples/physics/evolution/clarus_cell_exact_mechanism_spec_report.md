# Clarus cell exact mechanism specification

- passed: `True`
- mechanism: A Clarus cell is an open bounded identity loop.  Context enters through regulated ports; ports feed energy; energy maintains boundary, identity, metabolism, and repair; repair keeps damage below the closure boundary; identity rebuilds the metabolic machinery; recurrence projects the whole state into either daughters or long-lived maintenance.

## state variables

| symbol | name | primitive form | human form | meaning |
|---|---|---|---|---|
| `B` | `boundary_identity` | semi-permeable protocell boundary | plasma membrane, channels, receptors, adhesion | inside/outside distinction and membrane identity |
| `U` | `regulated_ports` | surface influx/efflux chemistry | transporters, vesicle traffic, ER/Golgi/endosome/lysosome routing | controlled exchange with the outside |
| `E` | `energy_resource` | fed chemical resource pool | ATP/redox/calcium/metabolic state | usable internal free energy and resource state |
| `A` | `autocatalytic_metabolism` | autocatalytic reaction core | mitochondria plus biosynthetic and maintenance metabolism | self-maintaining catalytic production loop |
| `I` | `identity_template` | copying template or heritable chemical state | genome plus epigenome and transcriptional regulatory state | heritable cell identity constraint |
| `D` | `damage_load` | leakage, decay, copying error | ROS/proteotoxic stress/DNA damage/organelle damage | accumulated entropy, waste, misfolding, and injury pressure |
| `Q` | `repair_quality_control` | daughter retention quality | repair, autophagy, proteostasis, lysosomal clearance | damage removal and state restoration capacity |
| `S` | `support_context` | environmental gradient and population selection | ECM, vascular, immune, endocrine, neighboring-cell and glial support | external support that stabilizes the cell unit |
| `R` | `recurrence_operator` | division threshold and daughter inheritance | cell-cycle recurrence or postmitotic maintenance recurrence | projection that makes the next unit count as the same cell type |

## operators

| operator | update | invariant role | failure signature |
|---|---|---|---|
| `context_to_ports` | `U_{t+1}=f_U(U_t,S_t,B_t,D_t)` | outside influence enters only through regulated exchange | uncontrolled exchange or starvation |
| `ports_to_energy` | `E_{t+1}=f_E(E_t,U_t,A_t,S_t)-c_E(D_t)` | resource flow must become usable free energy | energy floor collapse |
| `energy_to_boundary` | `B_{t+1}=f_B(B_t,E_t,U_t,Q_t)-l_B(D_t)` | identity needs active membrane maintenance | membrane identity loss |
| `energy_to_identity` | `I_{t+1}=f_I(I_t,E_t,Q_t,S_t)-n_I(D_t)` | same cell type must be constrained by template and regulatory memory | identity drift |
| `metabolism_to_repair` | `Q_{t+1}=f_Q(Q_t,E_t,U_t,I_t)-c_Q(D_t)` | maintenance must actively reduce accumulated damage | damage accumulation |
| `repair_to_damage` | `D_{t+1}=D_t+g_D(E_t,U_t,S_t)-r_D(Q_t,E_t)` | damage must stay below identity-collapse boundary | damage exceeds closure threshold |
| `identity_to_metabolism` | `A_{t+1}=f_A(A_t,I_t,E_t,U_t)` | template state must rebuild the metabolic machinery | metabolic program decouples from identity |
| `recurrence_projection` | `X_{t+1}=Pi_R[B,E,A,I,U,Q,D,S]` | the whole state is projected into the next same-type unit | no self-continuing unit |

## recurrence branches

### primitive_or_proliferative

- projection: `Pi_R = division/asymmetric-division projection`
- recurrence rule: mass_and_identity_cross_threshold -> daughter inherits B,E,A,I,U,Q
- closure test: division_count >= threshold and daughter identity retained

### human_postmitotic_neural

- projection: `Pi_R = maintenance projection`
- recurrence rule: no division; membrane/synaptic turnover, repair, autophagy, and glial support keep X_t in same identity basin
- closure test: energy, identity, membrane, damage, and maintenance recurrence all stay within thresholds

## invariants

| invariant | variables |
|---|---|
| `bounded_identity` | `B`, `I` |
| `powered_maintenance` | `E`, `A`, `Q` |
| `regulated_openness` | `U`, `S`, `B` |
| `damage_below_boundary` | `D`, `Q`, `E` |
| `same_type_recurrence` | `R`, `B`, `I`, `S` |
