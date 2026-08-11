# Loop 8J preregistration — surprise-gated directional context update

Status: LOCKED BEFORE IMPLEMENTATION

Parent Loop 8I remains STOP but supplies the experimental soft-evidence path.
Loop 8J tests whether a high-confidence negative outcome should selectively
labilize the active context direction before the ordinary signed update.

Let `c_t=pi_t(a_t)` be pre-feedback confidence, `r_t in {-1,+1}`, and let `v_t`
be the zero-mean eligibility direction of the selected action. Define

`d_t = 1[r_t<0] c_t`,

`a_t^+ = max(v_t^T s_t,0)/(v_t^T v_t)`,

`s_t^- = s_t - d_t a_t^+ v_t`.

Then apply the unchanged Loop 8H state update to `s_t^-`. No threshold, reset
gain, switch label, future target, or validation-fitted parameter is allowed.
The term is an engineering context-labilization hypothesis, not a claim that
negative RPE directly erases biological memory.

## Comparators

1. soft recurrent with no context-boundary update;
2. candidate confidence-dependent directional update;
3. negative-only directional update (`d=1` on every error);
4. generic whole-state forgetting (`d=c` on error, `d=1-c` on success);
5. negative-triggered full-state reset;
6. soft feedforward and hard recurrent references.

New seeds: training `876000..876015`, ID `877000..877031`, OOD
`877100..877131`, nulls from `877200`. Loop 8H/8I validation seeds are not used.

## Ten gates

1. exact positive-feedback no-labilization and negative `d=c` identities;
2. directional update never increases state norm and preserves the component
   orthogonal to eligibility within `1e-12`;
3. candidate minus no-reset post-switch trials +1..+4 accuracy LCB >= `+0.08`
   ID/OOD;
4. candidate minus hard recurrent accuracy LCB >= `-0.01` ID/OOD;
5. hard recurrent minus candidate NLL LCB > `0` ID/OOD;
6. candidate minus negative-only post-switch accuracy LCB > `0` ID/OOD;
7. candidate minus generic-forgetting post-switch accuracy LCB > `0` ID/OOD;
8. candidate minus full-reset post-switch accuracy LCB > `0` ID/OOD;
9. stationary absolute accuracy difference versus no-reset <= `0.02`, and flat
   candidate minus matched-flat accuracy <= `0.01`;
10. no future read, clone, same-tick commit, pending overwrite, nonfinite event,
    topology cycle, or state-cap violation.

Every gate contributes ten diagnostic points; GO remains conjunctive.
