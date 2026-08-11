# Experiment B validation — value-of-information STN HOLD

The locked default was run for 30 seeds and 5,000 episodes per seed.

| Arm | Accuracy | Utility | Mean HOLDs |
|---|---:|---:|---:|
| immediate | 0.663940 | 0.327880 | 0.000000 |
| common-offset STN | 0.663940 | 0.327880 | 0.000000 |
| always wait | 0.827107 | 0.514213 | 4.000000 |
| value-of-information HOLD | 0.818713 | 0.551173 | 2.464393 |
| cost-sign flip | 0.827107 | 0.514213 | 4.000000 |

The candidate improved utility over immediate action by `+0.223293` with paired
95% LCB `+0.218255`. It improved utility over always waiting by `+0.036960` with
LCB `+0.035175`, and improved accuracy over immediate action by `+0.154773` with
LCB `+0.152248`.

The action-only common-offset arm had zero action disagreements and exactly the
same aggregate results as immediate action, confirming the algebraic no-effect
counterexample.

The experiment nevertheless receives the preregistered verdict `STOP`: first-tick
HOLD rate in the top entropy quartile exceeded the bottom quartile by `0.212027`,
below the locked `0.25` gate. This threshold is not lowered after observation.
The failure is localized to the quartile effect-size gate; utility, accuracy,
causal no-effect, sign, and finite-horizon gates passed.

Interpretation: normative HOLD is useful, but the current preregistration does
not establish the demanded degree of conflict selectivity. A later experiment
must use an independently registered absolute posterior-conflict contrast or a
neural perturbation prediction; it may not reuse these seeds to choose a cutoff.
