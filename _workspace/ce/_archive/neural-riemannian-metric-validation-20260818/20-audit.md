# Neural geometry candidate-universe status audit

Status: COMPLETE
CE_RUN: neural-riemannian-metric-validation-20260818
Snapshot: `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`, `artifacts/candidate-equation-registry.md`, and `artifacts/candidate-equation-registry.json` were read as the stable expanded pre-implementation snapshot. No implementation output was available or used.

## Gate

Gate: PASS

The expanded contract is implementation-ready. The registry contains a finite 27-ID universe split into state SPD, graph metric/quasi-metric, directed action/Finsler, distribution metric, and derived readout/update types. It fixes the E17 chart, animal/session/trial split, horizons, ridges, ranks, penalties, optimizer, initial values, graph mappings, endpoint equations, aggregation, tie rules, controls and seed before multi-candidate output. The earlier single-`S4-H=5` E17 result is disclosed, so all E17 tournament results remain retrospective discovery rather than a lock.

The Markdown registry SHA-256 is `bb6ff3c878c151a3a7986d82791aced840a68f585461b7e3a0682db8cbc8412c`. The JSON eligibility ledger SHA-256 is `258b4b86208508f97f0c10dc9d75d958e8a08f4085fea8be03ef663461b18df1`. Both match `00-contract.md`; the JSON has 27 unique candidate IDs. No P0 or P1 remains in the audited scope.

## Mathematical adjudication

| Object | Actual status | Audit disposition |
|---|---|---|
| `NRM-D1` | [정의] | The local stochastic dynamics, reversible SPD length and directed Gaussian action are well-typed only with the registered inputs and fixed chart. |
| `NRM-D2` | [정의: 모델 선택] | The finite registry, not an unrestricted continuum of formulas, defines "all candidates" for this generation. |
| `NRM-T1` | [정리 후보; fixture required] | Covariance and metric reference tensors now have opposite correct chart laws; fixed identity references restrict E17 to the standardized chart and orthogonal transforms. |
| `NRM-T2` | [정리 후보; fixture required] | Time-varying `S4/S10` correctly transports each innovation/input from its injection time to the common endpoint using `Psi`. |
| `NRM-N1/N2/N3` | deleted parent readings | Raw `W` uniqueness, representation-only causal identification and prediction-to-causal-space promotion remain blocked by complete counterexamples. |
| `NRM-H1A` | [미완성; E17 untestable] | E17 lacks a same-unit direct `W^s`, metric-input and later-trajectory chain. |
| `NRM-H1B/H2--H5` | [미완성] | E17 can supply only limited effective-dynamics or descriptive feasibility and cannot support population confirmation at three opened animals. |
| `NRM-E17D` | [산출 후보: discovery only] | Implementation may calculate all eligible tuples and explicit failure codes under the frozen split. |

The `S4/S10` time direction, covariance-versus-metric ridge law, SPD gates, and separation of symmetric length from directed action agree across the contract and math lane. `S8/S9/S11/S16` are not presumed SPD without their gates. `G1/G2` require connected nonnegative conductance and the registered reversible kernel. `D1` includes log determinant and Gaussian normalization for cross-candidate scoring. `P1/P2` are one Wasserstein-2 equivalence class. Constant session matrices cannot produce nontrivial E17 curvature.

## Reproducibility closure

The first audit returned `REVISE` because `S12/S13` epsilon, objective, initialization, low-dimensional behavior, `S14` stable softplus, graph degeneracy rules, machine eligibility and deformation endpoints were open. The pre-outcome amendment closes each item:

- `S12/S13` fix `epsilon_g`, penalized precision likelihood, initialization, parameter domain, numerical differentiation, optimizer limits, failure handling and `r=1` disposition.
- `S14` uses `logaddexp` rather than direct exponentiation.
- `G1/G2` define degree, connectivity, random-walk kernel, stationary measure, eigennormalization, detailed-balance tolerance and failure codes.
- Pair selection uses released trial order and initial states only; endpoint separation, fit-only nonnegative scale and normalized RMSE are explicit.
- The JSON ledger supplies static status, tuple grid, runtime gate, missing inputs and endpoint for all 27 IDs.

## E17 evidence boundary

The official E17 ZIP, byte hash and CRC are recorded. Figure 3 synapse summaries, Figure 4 tracked dendrites and Figure 2 saline/DCZ branch trials are not a same-unit chain. Figure 2 has 11 sessions from three animals but no verified chronological semantics for released array order. The animal leave-one-out procedure may test transfer of tuple-selection rules with session-local fit calibration; it does not create a new independent cohort or a population winner.

Allowed implementation:

- Generate the 27-ID eligibility ledger and every eligible outer-train raw tuple score.
- Select tuples only from outer-train animal inner blocks, then score the selected tuple once on the held-out animal test block.
- Run the registered uncertainty, separation, condition-information, graph, `D1` and Wasserstein endpoints plus mathematical fixtures.
- Record missing-input, singularity, dimension, graph, pair-count, optimizer and numerical failures without proxy substitution.

Not allowed:

- Adding a candidate, grid point, chart, endpoint, initialization or retry after opening tournament output.
- Combining typed leaderboards, counting sessions/windows/cells/node pairs as independent animals, or calling E17 a locked future test.
- Promoting an effective-`J` metric to structural `W^s`, a decoder to task geometry, a covariance score to an independent spatial mechanism, or any E17 rank to `NRM-H1A`/population success.
- Implementing missing `B,R_u,Q_x`, SDE, Finsler, potential or curvature inputs with an unregistered identity proxy.

## Implementation boundary after PASS

`Gate: PASS` authorizes only the frozen candidate tournament, its focused fixtures, machine ledgers and descriptive E17 reporting. It does not promote a biological claim. A selected formula must later be frozen with code hash and carried unchanged to a new independent cohort; direct `Delta W^s -> Delta g -> Delta x` still requires the Tier A experiment in `artifacts/decisive-experiment.md`.
