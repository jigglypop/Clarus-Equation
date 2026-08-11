# Loop 8I preregistration — uncertainty-preserving recurrent DAG

Status: LOCKED BEFORE IMPLEMENTATION

Loop 8H hard-thresholded three noisy content values into one base action before
the DAG. Audit found that the true action was consequently absent from all DAG
routes on `14.648%` of ID and `35.791%` of OOD trials. Loop 8I changes only this
information bottleneck; Loop 8H code path and coefficients remain available.

For normalized, dimensionless content evidence `e_i`, define

`p_i = sigmoid(e_i / T_e)`, with frozen dimensionless `T_e=1`.

For every base action `b`,

`P(b|e) = product_i p_i^bit_i(b) (1-p_i)^(1-bit_i(b))`.

The conditional DAG proposal becomes

`P(a|e,c) = sum_b sum_k P(b|e) P(k|cue,state) 1[a=b XOR mask_k]`.

Thus every finite-evidence action retains positive support. Reverse inhibition,
state decay, feedback gain, policy temperature, state cap, trial generator, and
all Loop 8H recurrent coefficients remain unchanged.

## Arms

1. Loop 8H hard recurrent DAG;
2. soft-evidence feedforward DAG;
3. soft-evidence recurrent DAG;
4. soft-evidence recurrent feedback derangement;
5. soft-evidence recurrent feedback sign flip;
6. boosted stumps and hard tree as descriptive static references.

New, unopened seeds: training `874000..874015`, ID validation
`875000..875031`, OOD validation `875100..875131`, nulls from `875200`.

## Ten gates, 10 diagnostic points each

1. all action probabilities finite, normalized, and strictly positive;
2. true-action unreachable rate exactly zero;
3. hard minus soft recurrent NLL paired 95% LCB > 0 ID/OOD;
4. soft recurrent minus hard recurrent accuracy LCB >= -0.01 ID/OOD;
5. soft recurrent minus soft feedforward accuracy LCB >= +0.03 ID and +0.02 OOD;
6. soft recurrent minus deranged-feedback accuracy LCB >= +0.05 ID/OOD;
7. soft recurrent minus sign-flip accuracy LCB >= +0.10 ID/OOD;
8. stationary-null absolute soft recurrent/feedforward accuracy difference <= 0.02;
9. flat-null soft recurrent minus matched-flat accuracy <= 0.01;
10. no future reads, environment clones, same-tick commits, nonfinite values,
    pending overwrite, or topology cycles.

All gates are conjunctive for GO. Post-switch recovery is reported but is not a
Loop 8I promotion gate because soft evidence does not itself claim a reset
mechanism. Surprise/context-boundary handling is deferred to a later loop.
