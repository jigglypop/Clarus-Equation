# Real-data status audit: sleep, replay, and routing

Status: COMPLETE
CE_RUN: sleep-replay-routing-realdata-20260818
Snapshot: `00-contract.md`, `10-sources.md`, `11-math.md`, and `12-routes.md` were audited as the stable snapshot. No active runtime or source lane was edited.

## Gate

Gate: PASS

The source and route lanes correctly distinguish raw/event-level, processed group, participant-level, source-table, and code-only releases. The math lane correctly blocks window-level pseudoreplication, requires animal/session-held-out confirmation, freezes entropy state maps and branching estimators on training groups, and uses outer group-held-out proper scoring for the generally non-nested H2 comparison. The corrected contract now states the required event-level identifiers and `UNTESTABLE` granularity rule, animal/session independent units, training-only parameter fitting, linear/constant baselines, threshold stability, no window-level confirmatory degrees of freedom, and aggregate-only limits for source tables/group summaries. This is a documentation/preregistration closure gate; it is not an empirical result claim.

## Claim audit

| Claim | Status | Priority | Evidence and disposition |
|---|---|---:|---|
| `RR-D1` | [미완성] | P1 | `10-sources.md` supplies official URLs and broad access classes, but E15 DOI file list/size/license, E19 OSF manifest, and E13 raw-data access remain unverified. A URL or public code repository is not sufficient provenance. Require version, license, size, SHA-256, schema, and acquisition receipt per object. |
| `RR-R1` | [미완성] | P1 | E15 has processed group `.npy` data and public GPL code, but no confirmed public animal/session raw spike/SWR event archive. Selective download can support paper-level processed-group reproduction only after manifest/schema/hash checks; it cannot silently become an event-level replay analysis. |
| `RR-H1` | [예측 후보] | P0/P1 | Allowed only under Route A with same-animal/session/time-window replay and routing data, complete group split, fixed units, timestamps, SWR/replay endpoint, and prespecified branching or entropy estimator. Windows are repeated observations, never iid confirmation units. If event-level E15 is unavailable, mark `UNTESTABLE`, not negative or positive. |
| `RR-H2` | [예측 후보] | P1 | The corrected contract now uses the math-lane protocol: fit all sigmoid parameters inside each training fold, compare against linear/constant (and declared monotone diagnostic) on outer animal/session-held-out proper scores, require an interior/stable threshold, and do not use a nested likelihood-ratio test. |
| `RR-H3` | [미완성] | P1 | E02 Dryad is processed session/trial-level and 32.11 GB; selective session ZIP download may suffice for a predeclared participant/animal-held-out geometry/behavior test if the README schema and behavior linkage are verified. It does not provide raw broadband and cannot support claims beyond its released granularity. |
| `RR-H4` | [미완성] | P1 | E19 article-level statement promises preprocessed data sufficient for main conclusions, but OSF returned 403 and file manifest/license/participant granularity are unverified. No selective download is authorized until the official manifest confirms linked participant-level item/category, REM, SWS, total sleep, baseline, and exclusions. Group means make participant-level prediction `UNTESTABLE`. |
| `RR-H5` | 기각/분리 유지 | P0 | E15/E02/E19/E13 are different datasets/species/subjects and do not form a same-subject `Delta W -> Delta g -> Delta x(t)` chain. Partial findings must remain separate; cross-study concatenation is not an integrated mechanism. |
| `RR-H6` | 기각/미완성 | P0 | Replay, category transformation, or REM/SWS association does not directly measure a generative recombination process or REM-specific causal intervention. Require event-level recombination metric plus suitable control before reopening. |
| `RR-X1` | 활성 제외 | -- | No matched AGI benchmark bridge exists. Real-data results cannot establish AGI architectural superiority. |

## Statistical and granularity gates

1. The independent confirmation unit is an animal; if the design documents one independent session per animal, it may be a session. Adjacent or overlapping windows, SWRs, trials, and items are repeated observations. A mixed model alone does not cure leakage or residual serial dependence; use complete-group-held-out prediction and animal/session clustered uncertainty.
2. E15 H1/H2 require raw or processed event data with animal, session, timestamp, unit quality, condition, SWR/replay endpoint, and enough independent groups to fit the declared model. A processed group summary, figure table, or code-only release cannot support branching-to-replay association.
3. Branching is not an invariant criticality parameter under subsampling. Freeze unit sets, include firing-rate/active-unit and detection-quality covariates, and run the preregistered thinning and nonstationarity sensitivity grid. Entropy requires a training-frozen state map, bin width, smoothing and minimum-count rule.
4. H2 is a predictive model comparison, not a nested test. All parameter and hyperparameter selection must be inside training groups; the confirmation score is outer group-held-out proper score with a prespecified paired uncertainty rule. Window-level p-values cannot confirm H1/H2.
5. E19 REM/SWS ratio alone is insufficient: use REM and SWS directly or a declared composition-plus-total-dose parameterization, with positive-duration handling and baseline covariates. Participant/item rows must be clustered at participant.
6. Source tables and figure data may be downloaded selectively only to reproduce the supplied aggregate statistic. They cannot be expanded into individual-level effects, new predictive claims, or causal claims. Selective E02 session downloads are potentially sufficient for H3; selective E15 group files are potentially sufficient for R1 only; E19 and E13 remain blocked pending manifest/access verification.

## Authorized implementation and data acquisition

Allowed under this PASS gate:

- Create `artifacts/realdata/` manifests and acquisition receipts for official, license-compatible objects only, recording URL, release/version, access date, file class, expected groups, byte size, SHA-256, schema, and local relative path.
- Download a small public E15 processed-group object and its public code solely to reproduce the paper endpoint, after confirming the DOI manifest and usage terms. Do not represent it as raw event-level data.
- Download only predeclared E02 Dryad session ZIPs needed for a training/held-out geometry reproduction, provided the README schema, animal/session IDs, behavior labels, license, and hashes are recorded. Do not download all 32.11 GB by default.
- Attempt official E19 OSF manifest recovery or use an approved browser/manual acquisition path, recording the 403/access limitation. No analysis starts until participant-level granularity and terms are confirmed.
- Use synthetic data only for estimator/unit fixtures and statistical sanity checks; never as empirical replacement.

Not authorized:

- Any login, data-use agreement acceptance, external compute, author-request raw-data retrieval, or redistribution without a new explicit approval for the exact object and terms.
- Any E15 H1/H2 analysis from group summaries, figure values, or windows treated as independent observations.
- Any H2 nested likelihood-ratio test, result-driven sigmoid threshold/bin/state selection, or selection of the better exposure (`B` versus `H`) after seeing replay outcomes.
- Any claim that E13 raw broadband was obtained; the ledger says >10 TB is author-request/restricted and source tables do not replace it.
- Any cross-study integration into RR-H5, REM dream-algorithm claims, or AGI conclusions.

## Contract correction verification

`00-contract.md` §2, §4, §5.1--§5.3 and §6 now satisfy the required correction: `RR-H2` specifies generally non-nested outer animal/session group-held-out proper scoring, training-only fitting, linear/constant baselines and stable interior threshold; §5.1--§5.2 make animal/session the unit and explicitly exclude window-level confirmatory degrees of freedom; §2 makes event-level identifiers and aggregate-only/source-table limits explicit and marks missing granularity `UNTESTABLE`.

`Gate: PASS` applies to the corrected contract and implementation boundary. It does not assert that any acquisition or empirical claim has already succeeded. A successful download alone is not evidence of a claim, and a failed access path is not evidence against the biological hypothesis.

## Re-audit evidence

The current contract was compared with unchanged `10-sources.md`, `11-math.md`, and `12-routes.md`. The former P0 contract defect (“nested training selection” for non-nested models) is removed. The contract now agrees with the math lane on proper-score outer validation, group-level independent units, pseudoreplication controls, frozen estimator/state-map selection, E19 dose/composition controls, and `UNTESTABLE` treatment when the release granularity is insufficient.

## Implementation permissions after PASS

Permitted: create acquisition manifests and receipts; download only official, license-compatible public objects after recording URL, release/version, access date, file class, size, SHA-256, schema and expected independent groups; use small E15 processed-group objects for paper-level RR-R1 reproduction; use predeclared E02 Dryad session ZIPs for subject-held-out geometry reproduction; recover the E19 OSF manifest before deciding whether participant-level analysis is possible; and place analysis code/fixtures under the run's `artifacts/` path. Synthetic data remain unit fixtures only.

Still prohibited: treating E15 group summaries or windows as event-level H1/H2 data; any window-level iid inference; nested likelihood-ratio testing; result-driven bin/state/threshold or exposure selection; claiming E13 raw broadband access without a receipt; accepting logins/data-use agreements/external compute or redistributing data without exact-object approval; cross-study RR-H5 integration; RR-H6 dream-algorithm claims; and any AGI superiority claim.
