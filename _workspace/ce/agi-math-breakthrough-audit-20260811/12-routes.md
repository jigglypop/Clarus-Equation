# Route lane

Status: COMPLETE

Recommended route: action-conditioned robust belief MPC on the existing runtime.

1. Separate external task goal from internal self-state.
2. Learn action effect in the transition, initially `f(o_t) + D e(a_t)`.
3. Maintain a rank-1 robust posterior `(mu, Sigma)` for hidden residual causes.
4. Derive correction trust from posterior signal-to-uncertainty rather than a fitted output gain.
5. Compare H=2 or H=3 action sequences and execute only the first action.
6. After control works, replace the present critic derivative with signed TD/RPE eligibility learning.

Rejected immediate routes: more output gains, convex ensembles, extra latent rank, forced sparsity, graph/manifold wrappers, or a nominal V9 without a new causal state.

Separate scientific route: explicit-spike SNN for the CE substrate/natural-emergence claim. It is important to the theory but does not repair missing agency equations by itself.

