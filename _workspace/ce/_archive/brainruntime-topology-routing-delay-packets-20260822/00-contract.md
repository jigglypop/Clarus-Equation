# Topology routing after event-time delay repair

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-topology-aware-routing-20260822`

The predecessor topology comparison stopped because the mandatory delayed M1
baseline passed `0/16`. This run changes exactly one runtime mechanism: an
axon-delay slot now stores the complete source-qualified presynaptic packet
$e_t=u_t^+\odot x_t^+\odot a_t\odot q_t$ and delivers it after the fixed
two-call delay. Arrival-time lifecycle/STP state cannot erase or create the
packet. `backend=auto` falls back to Torch for delay, and explicit Rust delay
fails closed.

All topology scores, four blocks, exact 25% edge budget, thresholds, learning
schedule, rollout horizon, decoder, development seeds (`97201..97216` and
`97301..97316`), controls, and decision gates are inherited unchanged from
the predecessor contract. Confirmation stays sealed. The maximum claim is a
synthetic `BrainRuntime` routing result; no biological, clinical, anatomical,
physical-energy, curvature-memory, or AGI conclusion is permitted.

The mandatory order is: focused delay/backend tests; full M1 binding admission
at least `15/16`; then the already frozen topology development arms. Any
failed integrity receipt or threshold/seed/horizon/decoder change is STOP.
