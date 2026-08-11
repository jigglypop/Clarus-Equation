# Mathematical and contract audit

Status: COMPLETE

The central missing closure is not another isolated equation. It is the implemented causal chain

`state -> prediction/belief -> critic -> credit assignment/modulation -> action -> environment -> next state`.

Current components cover pieces of this chain, but no canonical model closes it with one score scale, uncertainty-aware prediction, multi-step planning, and a real task environment.

Required first invariants:

- no future-state access in belief updates;
- feature-off behavior reduces to the current checkpoint;
- critic values and critic derivatives use one typed contract and one scale;
- save/restore preserves next-step dynamics within tolerance;
- full augmented belief-state stability is checked, not only component spectral radii;
- every claimed gain is measured against a matched baseline on held-out tasks.

For ACBSM, the present stability calculation checks `max(rho(J_mechanism), rho(A_latent))`. It does not yet certify the augmented block Jacobian containing the off-diagonal belief correction, so transient amplification and a common Lyapunov/weighted-norm bound remain open.

