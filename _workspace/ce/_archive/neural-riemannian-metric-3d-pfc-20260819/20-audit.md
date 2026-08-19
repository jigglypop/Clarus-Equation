# Pre-implementation status audit

Status: COMPLETE

Audited scope: `00-contract.md`, `10-sources.md`, `11-math.md`, and
`12-routes.md`. No numerical fixture, synthetic outcome, neural statistic, or
behavioral outcome was available to or inspected by the auditor.

## Gate

Gate: PASS

The revision-2 snapshot is implementation-ready for the synthetic scope only.
The independent audit found no remaining P0. Its three final P1 wording checks
were applied exactly: the candidate table now matches the per-circuit
surrogate rules, the combined coframe law is $e'=QeJ^{-1}$ with
$S'=QSQ^\top$ restricted to the frame gauge, and full-field recovery explicitly
fails on a zero or nonfinite denominator.

This PASS authorizes source implementation and non-outcome Gate A fixtures. It
does not authorize synthetic true/null outcomes or any PFC data analysis.

## Revision history

The first snapshot was `REVISE` because coordinate principal square roots were
incorrectly treated as tensorial, the raw Itô bridge lacked its connection
correction, controls/surrogates were not finite, and the curved fixture was not
explicit. Revision 1 replaced them with:

- intrinsic $h$-self-adjoint endomorphisms and oriented coframes;
- $A_t=g_0^{-1}\circ g_t$ and its intrinsic logarithm;
- a coordinate-covariant generator using $\nabla^h$ and the coordinate Itô
  correction $-D^{jk}\Gamma(h)^i{}_{jk}$;
- an exact six-parameter metric, 12-parameter direct response/diffusion model,
  and nine other finite controls;
- exact graph, baseline, flat-pullback, curved-field, true, and six null
  generators.

The second snapshot had no P0. Its reproducibility P1s were closed by analytic
flat-map injectivity, typed ambient/coframe conversion, a frozen production
chart, candidate-specific ridge vectors, fit-only/per-circuit surrogate scope,
explicit hidden-graph separation, direct noninferiority hypotheses/ties,
invalid-node failure, exact nearest-neighbor ordering, and validator rules.

During outcome-free Gate A implementation, the raw atlas thresholds exposed a
normalization contradiction: the depth column is $r_w=0.05n$, so the prior
$\sigma_{\min}\ge0.15$ and $|\det Dr|\ge0.20$ were impossible. An independent
math check derived
$|\det Dr|=0.05\sqrt{\det a}(1-0.05wk_1)(1-0.05wk_2)$ and authorized the
pre-outcome correction to raw gates `0.025` and `0.04`, together with the
normalized normal-volume gate $J_\perp\ge0.5625$. The fixture's observed
minimum determinant `0.038424003022341446` was used only to confirm the
analytic bound, not to tune a scientific estimator or outcome threshold.

## Mathematical adjudication

| Item | Status | Boundary |
|---|---|---|
| full 3D SPD field | definition | six independent components and three off-diagonals are mandatory |
| folded anatomical ribbon | definition/control | $h=r^*\delta$ is interior-flat where $Dr$ has rank three; folds are not functional curvature |
| $W\to g$ producer | hypothesis | finite six-tensor synthetic family only |
| metric-to-path bridge | model hypothesis | generator (32), not distance alone, determines paths |
| direct alternative | falsification model | separate six-parameter response and diffusion fields; metric must be noninferior within 0.01 nat/increment |
| synthetic recovery | estimator claim only | 200 independent datasets, at least 180 complete successes |
| null calibration | estimator selection claim | each of six families separately has at most 4/200 false promotions and CP upper bound at most 0.05 |
| public PFC data | feasibility only | no current source supplies continuous ribbon coordinates, direct $W^s$, and same-unit longitudinal activity together |

## Dimension and coordinate gate

Normalized chart variables, $h$, $g$, coframes, graph weights, tensor
coefficients, potential $U$, and every exp/log argument are dimensionless.
Response/diffusion retain inverse-time units. Tensor construction is explicitly

```text
coordinate dq -> ambient vector -> ambient covariant tensor
              -> g0-orthonormal coframe tensor.
```

The production path chart is fixed. Only the continuous generator and tensor
readouts receive chart-covariance claims; finite-step Euler paths do not.

## Statistical gate

Inference uses one score per independent circuit. Paths and increments are
nested estimation replicates. Nine superiority comparisons and the shifted
direct-model noninferiority comparison form one ten-test Holm family. The mean
noninferiority guard is descriptive; the exact circuit-level sign test carries
the registered inference. Null-family error rates are never pooled.

## Implementation boundary

Allowed now:

- create the isolated run-local Rust crate and frozen Cargo.lock;
- implement `R-KERNEL-3D`, the generator/fitter interfaces, result schema, and
  independent validator;
- run compile/tests and Gate A analytic/reference fixtures;
- audit the stable source, lock, seeds, truth separation, and validators.

Not allowed yet:

- run any of the 200 true or 1,200 null scientific datasets;
- inspect aggregate recovery, selection, path, or false-promotion outcomes;
- download large PFC archives or calculate neural/behavioral statistics;
- change a candidate, coefficient, split, tolerance, or decision after an
  outcome is opened.

Gate A must pass and a stable-code audit must authorize the one-shot synthetic
execution. Gate B failure seals every PFC route. Even after Gate B, Wójcik is
`ACCESS_BLOCKED` until its exact reuse license is recorded, and any eventual
result is limited to `PFC_FEASIBILITY_ONLY`.

## Harness policy revision

At the user's request on 2026-08-19, implementation assurance is split into a
fast mandatory kernel gate and an optional publication lock. The kernel gate
retains every mathematical check: full six-component SPD recovery, signed
atlas orientation, pointwise direct-versus-3D Riemann agreement, nonlinear
pullback flatness, a nonflat curved control, deterministic parallel output,
and the independent oracle. Repeated executable copies, exhaustive manifest
lineages, and mutation archives are no longer prerequisites for ordinary
source work or Git commits. They are regenerated only before Gate B outcomes
are opened. This governance change does not relax a scientific threshold and
does not authorize Gate B or PFC analysis.
