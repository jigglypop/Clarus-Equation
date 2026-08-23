<!-- PROVENANCE: 병행 세션(Codex, 2026-08-23)이 stage 파일로 직접 집필한 판본. 단일 작성자 규약 위반으로 artifacts에 이동 보존 (20-audit §6). 정본은 run 루트의 동명 파일. -->
# BA-V3-1 감사 — 20-audit

CE_RUN: _workspace/ce/brainruntime-bio-constrained-l1-20260822
Status: COMPLETE
Gate: BLOCKED
Audit basis: 00-contract.md, 10-sources.md, 11-math.md, 12-routes.md
No source/math rerun performed.

## P0

### BA-V3-1-C01 — CE delta and measurement model absent

Claim in 00-contract §1/§8: the allowed primitive circuit can satisfy biological
observables and reach BIO_EVIDENCE_L1(+L2).

Finding: 11-math §P0-1 states that $F_{\rm bio}$, $\Delta F_{\rm CE}$, and
observable-specific $\mathcal H_d$ are not defined separately. 12-routes states
that without a CE delta the comparison is baseline-only. Therefore no actual
CE-vs-biology residual or falsifier exists.

Status: [미완성], not [예측]; L1/L2 implementation cannot start.
Priority: P0.
Action: delete/replace the L1/L2 success claim from the active contract. Add an
explicit source-locked measurement model $\mathcal H_d$, baseline-only model,
CE delta, residual rule, and matched control before development.

### BA-V3-1-C02 — proposed sleep dynamics has no guaranteed stationary regime

Claim in 00-contract §3.2: $\lambda(w)>0$ globally is inherited as sufficient
homeostatic protection.

Finding: 11-math §P0-2 gives a divergence witness for repeated wake increment
$a>0$ with the declared
$\lambda(w)=\lambda_0/[1+(w/\kappa)^2]$. Positivity alone does not imply
tightness, invariant distribution, or cyclic stationarity.

Status: [반례/미완성]; the claimed normal-state gate is not established.
Priority: P0.
Action: remove “$\lambda(w)>0$ 구조 강제” as a sufficient condition. Either
add and prove a restoring condition (saturating wake gain, nonzero tail loss, or
explicit global homeostatic feedback) or demote all stationary-distribution and
L2 claims to untested hypotheses.

### BA-V3-1-C03 — time/unit bridge is not fixed

Claim in 00-contract §3/§6: tick, milliseconds, day/month, developmental and
adult windows jointly define one biological contract.

Finding: 11-math §P0-3 identifies missing conversions among $w$, $\lambda_0$,
$\tau_{\rm el}$, $\tau_\pm$, $\rho_\infty$, $T_m$, 16:8 ticks, and day/month
windows.

Status: [미완성].
Priority: P0.
Action: delete cross-timescale quantitative gates until a source-locked unit/time
map and acquisition-window mapping are specified.

### BA-V3-1-C04 — source gates are not all verified

Claim in 00-contract §5: R1′–R6′ are usable simultaneous biological gates.

Finding in 10-sources:

- R1′ = `UNVERIFIED_AS_GATE`;
- R2′ = `VERIFIED_N_SERIES_WITH_METADATA_REQUIRED`;
- R3a′ = `VERIFIED_MORPHOLOGY_COMPARISON`;
- R3b′ = `DEFINITION_VERIFIED_GATE_UNVERIFIED`;
- R4′ = `UNVERIFIED`;
- R5′ = `UNVERIFIED_AS_RATIO`;
- R6′ = `VERIFIED_NARROW_CONTRAST`.

Status: only R3a′ is a verified morphology comparison; R2′ and R6′ remain
metadata/narrow-definition dependent. R4′ and R5′ have no locked source
observable.
Priority: P0.
Action: remove R4′ and R5′ from the current contract. Do not score R1′, R3b′,
or R6′ until their exact numerator/denominator, cohort, window, uncertainty, and
preprocessing are source-locked.

## P1

### BA-V3-1-C05 — hidden parameter / design-constant sensitivity

Finding: 11-math reports large gate movement under $w_0$, $\tau_e$,
$K/\tau_{\rm el}$, and homeostatic $\beta$. The declared eight free parameters
do not capture effective degrees of freedom.

Status: [미완성], identifiability unestablished.
Priority: P1.
Action: remove “free 8 < effective conditions 10” as evidence. Add
source-locked $q(\theta)$, Jacobian rank/condition number, and profile likelihood.
Declare $w_0$, $\tau_e$, $K/\tau_{\rm el}$, $\beta$ and scaling locus either
fixed by source or explicitly free before any fit.

### BA-V3-1-C06 — objective function is not fully defined

Finding: 11-math shows R4′/R6′ weighting changes the objective by orders of
magnitude; monotonic maturity/decline statements have no target, range, or
objective term. R3b′ positivity is structurally imposed rather than independently
discriminating.

Status: [미완성].
Priority: P1.
Action: delete the current global-fit protocol. Re-specify targets, likelihood,
uncertainty, inequalities, monotonicity terms, and matched-control scoring;
otherwise classify the run as calibration-only.

### BA-V3-1-C07 — L2 E1/E2 are not genuinely emergent

Finding: 11-math says E2 is controlled by the undeclared homeostatic locus/gain;
full per-neuron correction makes the firing distribution narrow. E1 is only
surrogate-model feasible, not source-confirmed.

Status: [가설], not [창발 통계].
Priority: P1.
Action: remove E1/E2 from the fitted condition count and retain as diagnostic
predictions only after the stationary regime and measurement model are fixed.

### BA-V3-1-C08 — R1/R2 and R3a/R3b have measurement non-identifiability

Finding: 11-math reports R2′(N) values 0.371 versus 0.730 and R1′ numerator
variants 0.040/0.055/0.080/0.110 under the same dynamics. R3b′ lower bound and
R4′ random-box pass rate reduce discrimination.

Status: [측정 자유도], not biological support.
Priority: P1.
Action: delete all gate-pass interpretation based on these quantities until
measurement definitions are locked independently of model fitting.

## P2

### BA-V3-1-C09 — convention and gauge omissions

Findings: 11-math identifies missing tick-to-ms/day/month conversion,
$w$-gauge/$\kappa$/$w_0$/$w_{\min}$ scaling convention, and R4′ net-ratio versus
log-growth choice.

Status: [미완성].
Priority: P2.
Action: add explicit dimensional conventions and gauge fixing to the successor
contract.

## Claim disposition

- Delete from active claims: “Status: COMPLETE” in 00-contract;
  `BIO_EVIDENCE_L1(+L2)` as an attainable current-run outcome; R4′ and R5′
  quantitative gates; any claim that $\lambda(w)>0$ guarantees a normal state;
  any claim that E1/E2 are unforced emergent biological statistics.
- Revise in a successor, not in this run: R2′ into a metadata-complete N-series
  observable; R3a′ into an SBEM/ASI measurement-model comparison; R6′ into the
  exact training/deprivation/eight-hour formation contrast; R1′ and R3b′ only
  after source-level numerator/denominator and uncertainty are fixed.
- Preserve as hypotheses/routes only: RT-S, RT-G, RT-H in 12-routes. None is
  selected by current data; selection requires source-locked development data,
  independent falsifier, and matched controls.

## Contract decision

The pre-development contract cannot be revised in place while retaining its
current L1/L2 objective. It is BLOCKED because the P0 defects precede model
fitting and because several gates are explicitly unverified.

A narrowed successor contract is possible, but it must be a new admission with
the claim ceiling:

> source-locked observable calibration and measurement-model comparison of a
> biologically motivated candidate dynamical family.

The successor may initially retain only R2′ (metadata-complete), R3a′
(measurement-model explicit), and R6′ (narrow contrast), while treating R1′,
R3b′, R4′, R5′, E1, and E2 as excluded or diagnostic. It must define
`BIO_STARTING_MECHANISM`, `CE_DELTA`, `MEASUREMENT_MODEL`, provenance, split,
residual, falsifier, controls, model selection, and revision trigger before any
development run.

Final gate: BLOCKED.

