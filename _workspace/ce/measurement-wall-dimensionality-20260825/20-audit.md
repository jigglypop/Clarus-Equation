# Formal status audit: measurement-wall dimensionality

Status: COMPLETE

Gate: PASS

Scope of PASS: the gate applies only to the narrowed operational model with complete orthogonal pointer projectors (or a properly specified general Kraus instrument). It does not close an objective-collapse theory, a new physical dimension, an energy map, or a cosmological identity.

## Claim ledger

| claim | status | result | evidence |
|---|---|---|---|
| `MW-01` | conditional theorem | A finite discrete outcome space and each singleton record are topologically 0D. This is outcome-space dimension, not spacetime dimension. | `10-sources.md`; `11-math.md` §§2 |
| `MW-02` | supported model class | Measurement can be a finite-duration interaction supported on a worldline, worldtube, or bounded spacetime region. Instantaneous support is a limiting idealization, not a universal premise. | `10-sources.md`; `11-math.md` §2 |
| `MW-03` | theorem within the model | $\mathcal D_P$ is CPTP and idempotent; $\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P$ is CPTP and scales inter-block coherence by $1-\eta$. | `11-math.md` §3; `artifacts/verify_measurement_wall.py` |
| `MW-04` | conditional theorem and limit | For $\dot\rho=\gamma(t)(\mathcal D_P-I)\rho$, $\eta(t)=1-e^{-\int\gamma dt}$. Finite integrated rate gives a partial wall; a hard wall requires an ideal limit. | `11-math.md` §§4--6; verifier |
| `MW-05` | definition and derivation | Given a time-indexed instrument, $C_I$ and $C_{\rm wall}=\int\dot\eta\,\overline C_I\,dt$ are dimensionless information/opportunity functionals. | `11-math.md` §7; `artifacts/dimensionless-audit.md` |
| `MW-06` | incomplete physical bridge | No derivation maps the record atom, wall strength, or opportunity functional to energy, $T_{\mu\nu}$, dark matter, or dark energy. | all lanes; predecessor opportunity-cost audit |

## Counterexamples and scope corrections

The original `effect/projector` wording was too broad. For $E_0=E_1=I/2$, one has $E_0+E_1=I$ but $E_0^2+E_1^2=I/2$, so the effect-sandwich maps do not sum to a trace-preserving channel. The contract now transparently restricts the displayed formula to complete orthogonal pointer projectors and gives the general Kraus-instrument alternative. This resolves the P1 by narrowing scope; the failed broader formula remains documented in `11-math.md` and the verifier.

Weak/unsharp instruments and the one-outcome identity instrument refute the universal statement that every measurement is a hard wall. A finite-duration point detector has one-dimensional worldline support, refuting the statement that every measurement act is spacetime-0D. These parents are not retained as active claims.

## Premise accounting

The following are explicit model choices, not consequences of standard quantum mechanics:

1. the pointer partition $\{P_r\}$;
2. the Markovian dephasing generator and nonnegative rate $\gamma(t)$;
3. the monotone wall coordinate $\eta(t)$;
4. the time-indexed probability process $p_r(t)$ used in opportunity accounting;
5. any later map from a completed record or measurement history to the persistent carrier measure $\mu_F$.

## Remaining closure requirements

- record-space 0D $\rightarrow$ an additional physical sector or dimension;
- information opportunity cost $\rightarrow$ energy or covariant stress tensor;
- finite-time record formation $\rightarrow$ persistent 0D fold-memory carrier;
- carrier/stress completion $\rightarrow$ dark-matter or dark-energy abundance and perturbations.

Each arrow requires a separately stated physical-map axiom and independent falsifier. None is supplied by dimensional consistency or by the numerical dephasing check.

## Priority findings

- P0: none for the narrowed model.
- P1: resolved by explicit projector/Kraus scope correction.
- P2: none that changes the gate.

The surviving result is internally closed as an operational measurement model and remains incomplete as a cosmological theory.
