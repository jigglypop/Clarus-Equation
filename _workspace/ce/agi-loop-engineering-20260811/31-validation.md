# Validation lane

Status: COMPLETE

Results:

- focused runtime/new-feature regression: `76 passed`;
- CE dimensionless/bridge regression: `36 passed`;
- bootstrap fixed point: residual `2.08e-17`, PASS;
- Loop 1a teacher-identified DPC: `66.92/100 HOLD`;
- Loop 1b learned DPC: `86.32/100 GO` within claim limit;
- Loop 2 delayed credit: `100/100 GO` within tabular mechanism claim limit.
- Loop 2b runtime signed credit: `0/100 STOP`; first-update structural
  projection was a confound and signed-off LCB was `-0.41836`.
- Loop 2c matched structural manifold: `0/100 STOP`; signed-off mean
  `-0.02279`, LCB `-0.06021`, signed-shuffle LCB `-0.00208`, and held-out
  guard delta `+0.07479 > 0.02`.
- post-change focused regression: `52 passed`.
- Loop 3 raw-history state discovery: corrected comparator run `85/100 GO`;
  ID/OOD return `0.95879/0.89072`, ECE `0.02021/0.02815`.
- Loop 3 recurrent comparison: ID LCB `0.0`, OOD LCB `-0.00498`; this proves
  noninferiority under the frozen `-0.03` margin, not superiority.
- The first Loop 3 artifact is INVALID and retained only for provenance because
  its supposed recurrent comparator was a flattened MLP. The separately named
  corrected-RNN artifact is the scored result.
- final AGI-focused regression: `61 passed`;
- CE bootstrap residual: `2.08e-17`, PASS;
- CE canonical validation: `53 passed`; dimensionless formulas `7/7`;
- constants scorecard: `23` total, `12` scored, `11 PASS`, `1 CAUTION`,
  `1 INPUT` excluded, `1 OPEN TEST`; aggregate `CAUTION` due to
  `Omega_b h^2 = -1.80 sigma`;
- proof-completion harness retained the explicit LO/tree/raw obstructions and
  labelled the improved readouts as candidates only;
- research-core checks: `OK lanes`, `OK gate`, `OK build`, `OK final`.
- Loop 4 modular reward transfer: `0/100 STOP`; stale LCB failed in four of
  six domain/context cells and maximum oracle gap was `0.020410 > 0.02`.
- Loop 4 context-RNN comparison is non-claimable: train accuracy `68.76%`,
  label counts `7729/269/8002`, prediction counts `3725/0/12275`.
- Loop 5 first artifact is INVALID because the registered no-memory arm was
  omitted; corrected artifact is scored separately.
- Loop 5 corrected episodic memory: `90/100 GO`; latest/evidence/abstention/
  deletion each `1.0`; composite LCB `+0.75` existing, `+0.75` merge-off,
  `+0.25` FIFO; audit and capacity guards pass.
- Loop 6 executive switching: `0/100 STOP` despite ID/OOD accuracy
  `0.8771/0.8444`. It beat hazard-off, shuffled feedback, gap reset, and
  win-stay-shift, but did not beat surprise-off; recovery latency and the OOD
  oracle-gap gates also failed.
- Loop 7 active executive: `0/100 STOP`; active-minus-reward-only LCB
  `-0.00179/-0.00130`, recovery `6.63/6.51`. Information-only accuracy
  `0.6284/0.6174` confirms that the task lacks a useful epistemic action path.

Loop 1b primary per delay:

- return `0.76797` vs reactive `0.15`;
- LCB improvement `0.57344`;
- action-agnostic and H1 controls both lose the effect;
- recurrent sufficient-statistic control ties the candidate;
- success `0.86328`, Brier `0.07232`, ECE `0.04401`.

OOD sigma 1.2:

- return remains above reactive (`0.54102` vs `0.15`);
- success `0.73633`;
- ECE `0.09278`, so the preregistered calibration diagnostic fails.

The locked confirmatory block was not opened. No STDP learning-rate or
threshold sweep was run after the registered Loop 2c failure.

## Loop 8B mathematical preflight

- `[정리]` Pure heat diffusion on a compact connected manifold attenuates mode
  `k` as `exp(-D lambda_k t)`; all nonconstant mnemonic contrast vanishes.
- `[산출]` For the diagnostic values `D=0.2`, `lambda_1=1`, the first mode is
  `1.0, 0.818731, 0.367879, 0.135335` at normalized times `0, 1, 5, 10`.
- `[산출]` The repository dimensionless suite passes in the project virtual
  environment: `10 passed`.
- `[산출]` `git diff --check` found no whitespace error in the authored files;
  it only reported inaccessible pre-existing pytest temporary paths.
- `[미완성]` The `MD = metric modulation` and `replay = residual feedback`
  correspondences remain hypotheses and receive no empirical promotion.

## Loop 8B MD-modulated attractor benchmark

Status: `100/100 GO` on the locked bounded synthetic mechanism test.

- ID accuracy: pure diffusion `0.5104`, fixed attractor `0.7305`, MD attractor
  `0.8215`, shuffled-context MD `0.5114`, oracle MD `0.9530`.
- OOD accuracy: pure diffusion `0.4993`, fixed attractor `0.6867`, MD attractor
  `0.8063`, shuffled-context MD `0.5055`, oracle MD `0.9440`.
- Paired accuracy LCB, MD minus fixed: ID `+0.08382`, OOD `+0.11035`.
- Paired accuracy LCB, MD minus shuffled context: ID `+0.28630`, OOD
  `+0.26790`.
- Post-switch LCB, MD minus fixed: ID `+0.04017`, OOD `+0.02436`.
- All states remained finite and below the registered absolute bound of `4`.
- Exact heat-mode check error: `5.55e-17`.
- Focused tests: `2 passed`; Ruff: all checks passed.

Interpretation is restricted to the registered synthetic task. It supports
context modulation of memory dynamics over context-only readout under a
shared-capacity attractor. It does not establish MD/metric biological identity,
validate replay, or authorize runtime integration.

## Loop 8C feedback-residual replay benchmark

Status: `100/100 GO` on the locked selective-switch benchmark.

- ID accuracy: checkpoint `0.7373`, residual `0.8522`, sign-flip `0.4901`,
  oracle `0.9507`; residual-minus-checkpoint LCB `+0.10758`.
- OOD accuracy: checkpoint `0.7300`, residual `0.8488`, sign-flip `0.4925`,
  oracle `0.9461`; residual-minus-checkpoint LCB `+0.10921`.
- Post-switch trials 2–5: checkpoint/residual `0.0720/0.5572` ID and
  `0.0964/0.5895` OOD; improvement LCB `+0.45396/+0.46090`.
- Stationary accuracy changed only from `0.9520` to `0.9502`, mean difference
  `-0.00179`, within the locked absolute-neutrality bound `0.01`.
- Residual sign causality, oracle ceiling, state/residual bounds, and information
  integrity all passed. Maximum residual magnitude was `2.102 < 4`.

This output validates a narrow action-feedback prediction-error recurrence in
the synthetic switch task. It does not validate spontaneous sequence replay,
episodic reinstatement, or biological hippocampal identity.

## Loop 8D conflict-adaptive STN boundary

Status: `0/100 STOP` on the locked benchmark.

- Adaptive high-conflict accuracy beat the low boundary: LCB `+0.10094` ID and
  `+0.04783` OOD.
- Against the equal-mean matched boundary, the high-conflict LCB was
  `+0.02698` ID but only `+0.01311` OOD, below the locked `+0.015` gate.
- Conflict alignment was not causally identified: adaptive-minus-shuffle
  overall accuracy LCB was `-0.00293` ID and `-0.00977` OOD.
- High-conflict waiting increased by `+13.44/+12.61` steps as intended, but
  low-conflict OOD waiting also increased by `+4.12`, failing the `<=3` gate.
- Utility versus the equal-mean fixed boundary was noninferior in raw means but
  not in paired LCB: `-0.01366/-0.01044`.
- Memory traces were bit-identical, accumulator bounds passed, and timeout was
  below `0.05` in every arm.

The linear conflict-to-boundary map produces a speed–accuracy trade-off but
does not establish efficient conflict-aligned allocation. No coefficient was
changed after observing the result.

## Loop 8E gravitational field mathematical preflight

Status: EQUATION READY / NUMERICAL BENCHMARK NOT YET SCORED.

- The screened field functional is strictly convex for `mu > 0`, giving a
  unique Poisson/Helmholtz potential under the stated self-adjoint boundary
  conditions.
- Deterministic damped motion decreases total mechanical energy and yields a
  saddle-energy capture certificate without an adaptive conflict threshold.
- Gaussian two-source check at `a=1`, `sigma=0.6`: central force is exactly
  `0` for masses `0.5/0.5`, `+0.277058` for `0.7/0.3`, and `-0.277058` for
  `0.3/0.7`.
- Dimensionless regression remains `10 passed`.
- No numerical performance or biological gravity claim is made at this stage.

## Loop 8E quasi-static field benchmark

Status: `0/100 STOP`.

- Field residuals passed: `3.41e-13/2.64e-13` versus `1e-10`.
- Equal masses gave central force `-1.17e-15`; `0.7/0.3` and `0.3/0.7` gave
  `+0.11817/-0.11817`.
- Gravity accuracy was `0.8566/0.8280`, shuffle `0.5150/0.5094`, and sign-flip
  `0.1432/0.1722`; source causality controls passed.
- Gravity beat linear-STN utility by LCB `+0.18742/+0.32771`, but failed fixed
  DDM accuracy by LCB `-0.10872/-0.05680`.
- Capture occurred essentially at the registered five-step minimum for every
  coherence group. Strict ordering passed numerically but is not a meaningful
  graded latency effect.
- Capture rate, flip rate, boundedness, and memory trace identity passed.

The elliptic field changed globally and instantaneously after each evidence
update, making the energy certificate true before reliable evidence had
accumulated. This rejects the quasi-static decision implementation, not the
source-field symmetry theorem.

