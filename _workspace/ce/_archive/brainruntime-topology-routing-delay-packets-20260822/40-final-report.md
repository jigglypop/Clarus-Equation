# Event-time delay repair and topology admission

Status: COMPLETE

The causal runtime defect was real and is repaired: a packet emitted by an
active source now survives source deactivation while in flight, and an
inactive source cannot acquire a past emission merely by becoming active at
arrival. This changed the identical delayed M1 development baseline from
`0/16` to `16/16` without changing its horizon or decoder.

Topology routing itself remains untested. The next registered boundary failed
because a budget defined as 25% of the full learned graph can exceed the
number of edges allowed by the cluster/path support. This is a mathematical
feasibility defect, not a topology performance result. A successor may define
one outcome-blind budget from the minimum admissible support shared across
cues, but it must be a new contract and must preserve this invalid run.

No biological routing, memory-as-curvature, energy, disease, anatomy, or AGI
claim follows from this simulator repair.
