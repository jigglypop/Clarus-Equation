# Final report: real brain data for sleep, replay, and routing

Status: COMPLETE

## Answer

Actual public brain data support two narrower biological observations: sleep deprivation reduces hippocampal trajectory replay, and REM/SWS balance covaries in opposite directions with item-level and category-level representational change. Public criticality source tables also reproduce the published predictive advantage of distance-to-criticality measures. They do not yet test a branching-to-replay routing law in the same neural windows and do not identify a dream-generation algorithm.

## Results by claim

| Claim | Final status | Real-data result |
|---|---|---|
| `RR-D1` | PARTIAL PASS | 574 E13/E15/E19 objects hashed; official commits and OSF manifests recorded. E02 metadata reproducible, payload blocked by Dryad OAuth/403. |
| `RR-R1` | REPRODUCED at processed session-label level | E15 SD minus NSD replay count: `-125.29` events at 0--1 h and `-87.83` at 5--6 h; official bootstrap probabilities `0.0161057` and `0.00133728`. The table has 13 session labels but no animal IDs, so independent-unit provenance is not established. |
| `RR-H1` | UNTESTABLE | No released E15 object links unit spike timestamps, SWR events and replay fidelity in the same animal/session/window. |
| `RR-H2` | UNTESTABLE | H1 inputs are absent; no linear-versus-sigmoid held-out animal comparison was fabricated from aggregate windows. |
| `RR-H3` | ACCESS BLOCKED / UNTESTED | E02 Dryad exposes schema, sizes and SHA-256 values, but content download requires OAuth or returned 403 in this environment. |
| `RR-H4` | PARTIAL SUPPORT | E19 participant-level reproduction: item `rho=-0.553`, `p=0.000690`, descriptive slope `-0.0244`; category `rho=0.470`, `p=0.00509`, slope `0.0147`. This reproduces the published direction but the REM/SWS ratio remains composition-confounded. |
| `RR-H5` | REJECTED | E13, E15 and E19 use different species, subjects and measurement chains; they cannot form one observed $\Delta W\to\Delta g\to\Delta x(t)$ trajectory. |
| `RR-H6` | REJECTED as a strong claim | No dataset measures dream content, novel generative recombination, or an REM-specific causal computation. |
| `RR-X1` | OUT OF SCOPE | No matched AGI benchmark was run. |

## Additional E13 check

In the official Figure 2 source table, DCC prediction accuracy averaged `0.6111` versus shuffle `0.5189`, a difference of `0.0923`. Adding DCC to the baseline averaged `0.6801` versus baseline `0.6366`, a difference of `0.0435`. This is a source-table reproduction, not a new raw-broadband branching analysis; the paper states that the raw dataset exceeds 10 TB and is available by author request.

## Artifacts

- `artifacts/analyze_real_brain_data.py`: reproducible analysis and manifest writer.
- `artifacts/realdata-results.json`: machine-readable numerical results.
- `artifacts/realdata-manifest.csv`: byte sizes and SHA-256 for acquired objects.
- `artifacts/acquisition-receipt.md`: URL, version, license/access and failed-download ledger.
- `artifacts/verify_realdata_analysis.py`: focused invariant checks.

## Decision

The empirical core is real but narrower than the proposed narrative. Keep replay reduction and item/category transformation as empirical observations. Keep branching-to-replay and learning geometry open pending linked E15 event data and authenticated E02 files. Remove any wording that treats REM as a verified dream sampler or cross-study agreement as a single routing mechanism.
