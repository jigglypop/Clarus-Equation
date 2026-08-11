# Sparse causal bridge V1--V6: independent source/result audit

Status: COMPLETE

## 1. Scope and method

This lane audited repository evidence only.  No canonical document, source file,
test, registration, or historical artifact was edited.

- Historical evidence point: `itself` at
  `33836b89855d86c73e2ddf271b2c5eee6e1191b3`.
- Current canonical point: `main` at
  `fcb754ee5b1f35324e9502d3b3f146387eb25823`.
- Their merge base is
  `a0d0e3a9fd6489ba51fdcd5aa2da1931979504d0`; the historical line was never
  merged into the audited `main`.
- Registrations were recursively merged using the exact rule in
  `33836b8:reality_stone/python/reality_stone/clarus/sparse_causal_bridge.py::_load_registration`:
  the gate SHA-256 is over the byte concatenation `V1 || ... || Vn`, while the
  semantic config is a recursive deep merge.
- JSON verdicts and metrics were parsed directly from Git blobs.  Git object
  SHA-1 values and independent SHA-256 values were recomputed from blob bytes.
- No web sources were required.  The audited claims are synthetic, generated
  wholly by repository code, and make no external empirical claim needing a
  current observational baseline.

`PASS` below means only "the stored artifact says all preregistered checks for
that synthetic gate passed."  It is not a theorem, an external replication, or
an AGI result.

## 2. Bottom-line result

| Version | Stored validation | Locked test | What the repository actually supports |
| --- | --- | --- | --- |
| V1 | **FAIL** | unopened / absent | Development evidence that the paired-do estimator recovered the programmed two-edge graph, but the predictor failed under target confounding and one negative control was defective. |
| V2 | **PASS** | **PASS** | In a narrowed four-chart family with no direct hidden loading on either bridge target, the ideal paired-do procedure recovered the programmed graph and matched or beat the registered baselines. |
| V3 | **FAIL** | unopened / absent | Development evidence that graph, diagonal mechanism, and rank-one loading direction were recoverable; per-seed 80-step AR estimation was too unstable to pass prediction gates. |
| V4 | **PASS** | **PASS** | In the same known rank-one, matched-basis, fixed-loading-sign-shift family, pooling the latent AR on training residuals passed a **sequential one-step** gate that reread the true chart state at every step. |
| V5 | **FAIL** | unopened / absent | A genuine prefix-only single-origin H20 rollout was finite and used zero future reads, but failed four registered robustness/comparator checks.  It is preserved development evidence, not a passed endpoint. |
| V6 | **not run** | not run | Registration and V5-development pilot disclosure only.  No V6 implementation, runner, unit test, validation artifact, test artifact, or integrity artifact exists in the audited branch. |

The strongest passed historical statement is therefore V4's narrow
**sequential one-step synthetic forecasting** statement.  V5 is the first
genuine free-rollout attempt, and it failed.  V6 established no empirical
claim.

## 3. Version-by-version evidence

### 3.1 V1 -- graph recovery did not close the forecasting claim

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v1.json`.
Stored result:
`33836b8:artifacts/agi/sparse_causal_bridge_validation_v1.json`.

The programmed SCM had `A -> C = +0.52`, `C -> D = -0.48`, and direct hidden
loading on `A`, `B`, and bridge target `C`.  The geometry stage proposed four of
the six possible undirected pairs, including both true pairs by construction;
it was a positive-control candidate prior, not causal evidence.  The causal
model and `dense_probe` each received 512 extra paired interventions.

Stored positive observations:

- selected edges `A->C`, `C->D`;
- precision/recall `1.0/1.0`;
- bridge coefficient MAE `0.00148372`;
- intervention NRMSE `0.0848453`;
- finite metrics and spectral checks passed.

Stored failures:

- causal global RMSE `0.194811` versus local `0.168985`, a registered reduction
  of `-15.2828%`;
- downstream reduction versus local `-41.7815%`;
- paired 95% lower bound versus the best observational model `-0.0323688`;
- lesioning a selected bridge reduced rather than increased the direct-target
  error (`minimum_direct_target_mse_increase_fraction = -0.875845`);
- the permuted intervention control still selected both true edges, which the
  historical document attributes to a label-disconnection implementation bug.

The exact false check names are:
`downstream_vs_dense_observational`, `downstream_vs_local`,
`global_vs_local`, `lesion_direct_target`,
`paired_ci_vs_best_observational`,
`permuted_intervention_negative_control`, and
`raw_correlation_selects_common_cause`.

Verdict: V1 did not establish the registered predictive bridge claim.  Its
graph recovery is usable only as development evidence inside this programmed,
matched-basis intervention setup.  No V1 test artifact exists.

Replay limit: the defective V1 control implementation is not retained as a
reachable source version.  The only tracked `sparse_causal_bridge.py` already
contains the V2 balanced-arm fix, and V1 recorded no implementation SHA.  The
stored failure is preservable, but its exact buggy executable is not
reconstructible solely from branch history.

### 3.2 V2 -- passed only after removing target confounding

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v2.json`.
Stored results:
`sparse_causal_bridge_validation_v2.json` and
`sparse_causal_bridge_test_v2.json`.

V2 set both bridge-target hidden loadings to zero:
`train = [1.15, 1.25, 0, 0]`,
`OOD = [1.15, -1.25, 0, 0]`.  It therefore isolated sparse edge
identification from the V1 target-confounding problem.  It also used fresh
seeds, fixed the permuted-label control, and changed superiority requirements
against selectors that recovered the same graph to noninferiority checks.

Validation (20 seeds) and locked test (30 seeds) both report `passed = true`
with every stored check true.  Key values are:

| Metric | Validation | Locked test |
| --- | ---: | ---: |
| causal global RMSE | 0.155070 | 0.155941 |
| causal downstream RMSE | 0.100650 | 0.099579 |
| global reduction vs local | 11.1432% | 10.6790% |
| global reduction vs dense observational | 55.2004% | 56.1983% |
| global reduction vs raw correlation | 44.2586% | 45.2480% |
| ratio vs predictive-gain selector | 0.999996 | 0.999998 |
| ratio vs equal-probe dense | 1.000149 | 1.000516 |
| downstream reduction vs local | 33.4947% | 33.1476% |
| bridge coefficient MAE | 0.004031 | 0.004031 |
| intervention NRMSE | 0.088841 | 0.088666 |

Both splits selected exactly `A->C`, `C->D`; false common-cause and reverse
edges were zero; no-bridge and permuted controls selected nothing.  The model
was not meaningfully better than `predictive_gain_top2`, and the equal-probe
dense comparator was essentially identical.  The stored results consequently
support neither a special Laplace--Beltrami discovery advantage nor an
algorithmic advantage at equal information.

The V2 lock was explicitly a soft lock.  The post-run integrity file says it
is not a prospective cryptographic blind.  Git history also first introduces
V1--V4 registrations, code, validations, and tests together, so Git cannot
independently certify the asserted pre-implementation timing.

### 3.3 V3 -- target confounding restored; per-seed AR failed

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v3.json`.
Stored result:
`sparse_causal_bridge_validation_v3.json`.

V3 restored direct hidden loading on bridge target `C`, estimated the diagonal
mechanism as well as sparse bridges from paired interventions, and fitted a
rank-one residual filter from an 80-transition OOD prefix.  It re-estimated
the scalar latent AR separately for every validation seed.

Stored positive observations:

- exact `A->C`, `C->D` graph, precision/recall `1/1`;
- self coefficient MAE `0.0009888`, bridge MAE `0.0047149`;
- mean loading-subspace cosine `0.998759`, minimum `0.994147`;
- mean rank-one residual variance fraction `0.889695`;
- global reduction versus the no-latent mechanism `57.4544%` and versus the V1
  bridge form `13.2978%`.

Stored failures:

- global reduction versus fixed-local `0.5936%`, below the registered 5%;
- downstream reduction versus fixed-local `13.5530%`, below 15%;
- mean scalar-AR absolute error `0.109347`, above 0.08;
- paired 95% lower bound versus fixed-local `-0.00848675`.

The exact false checks are `downstream_vs_fixed_local`,
`global_vs_fixed_local`, `latent_ar_error`, and
`paired_ci_vs_fixed_local`.  No V3 test artifact exists.

Replay limit: the V3 artifact records
`latent_causal_bridge.py = b721ee3e...`, but the only reachable tracked version
of that path is the later V4 file with SHA-256 `40306162...`.  Therefore the V3
result artifact is retained, but its exact latent-filter implementation cannot
be rebuilt from reachable branch source alone.

### 3.4 V4 -- passed a one-step, true-state-reentry gate

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v4.json`.
Stored results:
`sparse_causal_bridge_validation_v4.json` and
`sparse_causal_bridge_test_v4.json`.

V4's registered change was to estimate one shared scalar AR from pooled
observational-training mechanism residuals.  Each OOD prefix estimated only
the residual center, rank-one direction, and scalar intercept.  The stored
shared AR is `0.936926794`, with absolute error `0.0230732` from the generator's
`0.96`.

Validation and locked test both report every check true:

| Metric | Validation (20 seeds) | Locked test (30 seeds) |
| --- | ---: | ---: |
| causal latent global RMSE | 0.157122 | 0.155779 |
| causal latent downstream RMSE | 0.114438 | 0.113340 |
| global reduction vs fixed-local | 7.1382% | 6.5831% |
| global reduction vs no-latent | 58.7286% | 61.3539% |
| global reduction vs V1 bridge form | 18.7328% | 20.2218% |
| global ratio vs adaptive-dense prefix | 0.869012 | 0.864783 |
| downstream reduction vs fixed-local | 15.4799% | 15.0738% |
| paired 95% lower bound vs fixed-local | +0.0110874 | +0.0100112 |
| mean loading-subspace cosine | 0.999184 | 0.999209 |
| minimum loading-subspace cosine | 0.997100 | 0.995359 |
| bridge / self coefficient MAE | 0.002399 / 0.001281 | same frozen model |

This establishes only the registered synthetic **sequential one-step** result.
V4 reread the true chart state on every scored step; it did not freely recurse
predicted chart states.  Validation and test used the same single fixed OOD
loading vector and the same frozen train/probe model, changing only simulation
seeds.  It did not discover a new loading family, unknown basis, hidden rank,
or chart identity.

The V4 test lock checked the two source SHA-256 values in the passing
validation artifact.  It did not pin the Git revision, Python/NumPy binaries,
or OS.  The historical document correctly describes it as stronger than V2
but not a complete reproducible-environment lock.

### 3.5 V5 -- genuine free rollout, overall FAIL

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v5.json`.
Stored result:
`sparse_causal_bridge_validation_v5.json`.

V5 froze the V4 mechanism, observed `x[0:81]`, and made one H20 free rollout
from `x[80]`.  H5 was required to be the first five rows of that same H20
trajectory.  The test suite checks the rollout API signature, future-state and
hidden poisoning invariance, and exact H5/H20 prefix equality.  The result
artifact records zero future observation reads, one forecast origin per seed,
20 free steps, no evaluation probes, and 20 independent validation seeds.

Stored validation values:

| Metric | H5 | H20 |
| --- | ---: | ---: |
| causal-latent path RMSE | 0.206780 | 0.333083 |
| persistence path RMSE | 0.231658 | 0.389130 |
| no-latent path RMSE | 0.394764 | 0.444204 |
| fixed-local path RMSE | 0.236655 | 0.422196 |
| stable adaptive-dense path RMSE | 0.214879 | 0.312696 |
| equal-probe dense+latent path RMSE | 0.206966 | 0.334016 |
| reduction vs persistence | 10.7396% | 14.4028% |
| persistence seed-win fraction | 0.65 | 0.75 |
| paired 95% lower vs persistence | -0.006791 | -0.014814 |
| ratio vs stable adaptive-dense | 0.962304 | 1.065193 |
| ratio vs equal-probe dense+latent | 0.999097 | 0.997205 |

The candidate was finite and stable: maximum absolute prediction `1.049216`,
maximum learned-mechanism Jacobian spectral radius `0.781420`, absolute latent
AR `0.936927`, H20/H5 path-RMSE ratio `1.610811`, and nonfinite prediction count
zero.  The failure was robustness/comparator performance, not numerical
divergence or a detected future read.

The exact four false checks are:

- `h5_seed_wins_persistence`;
- `h5_ci_persistence`;
- `h20_ci_persistence`;
- `h20_vs_stable_adaptive_dense`.

The stored integrity artifact explicitly says `validation_passed = false` and
`locked_test_opened = false`; no V5 test result exists.  Therefore the V5 mean
improvements are development results and cannot be reported as a passed
free-rollout endpoint.

The artifact records the execution environment as Windows 11, Python 3.14.2,
NumPy 2.3.5, wall time `0.360887 s`, zero downloads, and zero trajectory files.
These strings are informative but do not hash the interpreter or NumPy binary.

### 3.6 V6 -- preregistered, not executed

Registration:
`33836b8:experiments/preregistration/sparse_causal_bridge_v6.json`.

V6 froze a prefix-backtested convex consensus of three experts: the V5 sparse
causal-latent rollout, stable adaptive-dense prefix rollout, and persistence.
Its new validation/test seed ranges were `67100--67119` and `68100--68129`.
The registration disclosed V5 validation as development data, rejected a
scalar Kalman candidate, and recorded a V5-development-only consensus pilot
with H5/H20 RMSE `0.204421/0.319301`.  By the registration's own text, those
numbers selected the rule and are not V6 evidence.

Five independent repository facts establish non-execution:

1. `git ls-tree -r 33836b8` finds only
   `experiments/preregistration/sparse_causal_bridge_v6.json` for the V6 or
   consensus-rollout path family.
2. The expected `consensus_rollout_bridge.py`, gate wrapper, and unit test are
   absent.
3. There is no `sparse_causal_bridge_validation_v6.json`, test V6 artifact, or
   integrity V6 artifact.
4. `docs/7_AGI/41_Sparse_Causal_Bridge_World_G9CB.md` says both
   "V6 사전등록 완료ㆍ미실행" and "구현ㆍvalidationㆍtest는 아직 시작하지
   않았다."
5. V6 was added in branch-tip commit `33836b8`; there is no later commit on
   `itself` that could contain an execution.

Conclusion: V6 was not executed in the audited repository line.  It provides a
locked proposal and development-data disclosure, but zero validation or test
evidence.

## 4. Immutable provenance and hashes

### 4.1 Commit chronology

| Commit | Time (`+09:00`) | Relevant first additions |
| --- | --- | --- |
| `5db40b9404cd385dbd8f0ce28a7365d166a73c82` | 2026-08-10 17:50:08 | V1--V4 registrations, V1--V4 validation artifacts, V2/V4 tests, sparse/latent implementations, tests, historical chapter, V2 integrity. |
| `293ba9fcc6617e1a866355ed9962d8c81ad66bce` | 2026-08-10 17:58:35 | V4 post-test integrity attestation. |
| `33836b89855d86c73e2ddf271b2c5eee6e1191b3` | 2026-08-10 18:23:34 | V5 and V6 registrations, V5 implementation/test/result/integrity, updated historical chapter. |

This chronology is immutable, but it does **not** independently prove that V1
through V5 registrations were committed before implementation or outcome
inspection: each relevant registration first appears in Git together with its
code/result.  The `locked_pre_implementation` field is an in-file declaration,
not a Git-timestamp proof.  V6 is different: the branch ends with only its
registration and no implementation/result.

### 4.2 Registration identities

The "merged SHA-256" column is independently recomputed over exact Git-blob
bytes concatenated through that version, matching `_load_registration`.

| V | Git blob SHA-1 | Raw-file SHA-256 | Merged gate SHA-256 |
| --- | --- | --- | --- |
| 1 | `cc7fd461bef0989d0cbeca4c4739ee97786eb0c0` | `a4ab2f9b7ba5049926d77cd7cf90352531c93b02a1af4554a26dbfb6b48dbf0e` | `a4ab2f9b7ba5049926d77cd7cf90352531c93b02a1af4554a26dbfb6b48dbf0e` |
| 2 | `b3088e3ed141f139d6a804542a719fbace25b425` | `ae609efae7af917771082caa42d7df5cfd6c41a1ffb0081203eb8b5213f90f19` | `be128195e8504a09cbf2ec58bfc2191a47fb9c7b7ef895d514729253b3c6c704` |
| 3 | `e9efd2683f1ca425fd3301cdcc36f666a776c3a6` | `28041d1679cd526f0bde39ea5cce2688aac5d92ff5ea478bd74f0d578e4c0c20` | `14e06f276d8866e327e67d7a1225b201068535b5731bb20ff21dfc7e6c0a0dc6` |
| 4 | `a87f40aaeb2e4d60ab90da28dcbb074c0b13179b` | `b116b3ff73d20b8dd3ce25c0db669bfdaf06e077e98eb7f17d13c9d93f8ad8b6` | `f9f8a0d9c3a9203e91a675db01775564f05f360557c814bf31e2770017c98a91` |
| 5 | `dd31283029255dc44d1cdf62a52b71bdc7de343a` | `17a8ba5c7889557d0beaf5ab2613f6afe48ec30c8461228563185c8c6059c90d` | `31e029705a372b622bf5d7109784b89bb0b42d959c4960ef2bdedb3a8c07b78a` |
| 6 | `c22dcf4af0e911ef299d1ba7a12078f945c66d0b` | `579fabfaec662d89d06fada50a1dae9117f5befa059b10e37222d86e6ae82cf6` | `b73245484d1a8ff1e385cceb08cbf99105ef7db7d672531da677daba7fbc4eed` |

### 4.3 Result artifact identities and newline caveat

| Artifact | Verdict | Git blob SHA-1 | SHA-256 of canonical Git LF bytes | SHA-256 attested at run time |
| --- | --- | --- | --- | --- |
| validation V1 | FAIL | `65a03e89a443795416167c637ff86a29c45c8620` | `58c8d1ccadb6805c46455ba9d2f9a5af70497aa7a78a87414e2a0d1b2280f690` | `ca9f09f32957124766ff0a5e9dedaa7ae1b5f3f9a3ecdc092d8fbdd3c3c9d282` |
| validation V2 | PASS | `43bae9b6a4f91eb67bf2dbdcc46c4818f28daa71` | `2b91bd2d883f58b12d173455983c74ef6441fca0ae79ee907b26f49aef397f14` | `4717ab98ff85467f9c521d1c3c65955791466faf5733ddb7c1fd777363be5134` |
| test V2 | PASS | `47529b7f6fc912e04ac04e929537c14d63d056f4` | `67cdae06a77f9b1eceb60f9239d25725714402c2d67cf8b55da21aa35c294c7f` | `f970142e556710c36af0d4401abaf572ac32b4eab41e65c61f02fc8a946757f2` |
| validation V3 | FAIL | `189507725c261e60dec8e833f6fd0240db7122cb` | `62da4ed018a38bbc9cd8b1aff5753dded622f78822f24de9489c6e8b726c3310` | `1f07627c81deaaf906477f5a1ed5b870955084c33c726769637fa7d83d160abd` |
| validation V4 | PASS | `b4c73106a3ea9fae995babaf628695f081294884` | `4a6ec955bbe3b9954b74f7389dba84f30beba99452d0ca1c54f2c88423850421` | `41c17778c7aa2adcd36557ca0042ea0d2de90c817acbd8730bbc97424f553986` |
| test V4 | PASS | `4ea6152675282b6529849cf09781a23df37fcb3e` | `6dc29a4e7df50b9b79fdbbc085fe4bdf7321f62f4bc245e4ea6ffae96989646a` | `938e68518080f7a7b0e9a50bd2756121d64e0952ee2fcd75eec6c33656cd7db9` |
| validation V5 | FAIL | `8847e92ac79a4e4b1d014b962eb6bba6310c60d6` | `d7bb236cc1e47672921e2225ac1a9039d584e6036cf8d92d80e91eda2ba7651b` | `6dd4999e385fc47ea5ccd2e3e1233c60f2d1968554b82dd5cb95a34524f9e9a0` |

The apparent SHA mismatch is fully explained by line endings: for every result
artifact above, converting the Git blob's LF bytes to CRLF produces the
attested run-time SHA exactly.  Thus there is no observed semantic JSON drift,
but the cryptographic lock is byte-serialization and platform sensitive.  The
branch has no `.gitattributes`, while the audited Git installation reports
`core.autocrlf=true`.  Registrations were hashed as LF while generated result
files were attested as CRLF, so a reproducer must deliberately recreate that
mixed byte policy or pin a normalization rule.  A plain cross-platform checkout
is not sufficient evidence that the embedded lock will verify.

Post-run integrity artifact identities themselves are:

- V2: Git blob `05272a128e243b4d5af314658e0754d1faa47613`, SHA-256
  `807414037f2b7eaab0dcd9ec6a8a7532cc0192a6d3aab4811ff9cfa11797273e`;
- V4: Git blob `3e461a0a7c386da01b1bd5a5aa062450a6133518`, SHA-256
  `832030a452a2c45d6669cf7b95ee05950b63513b6015ef950a326094a978d045`;
- V5: Git blob `920e4a35e4530f451dcc0bc00c7294d17ceb75fc`, SHA-256
  `f2cf5e177a2d270d9e5adc0f4ab3a8158c0e04de2935a0cd4e54c74e756d4852`.

### 4.4 Reachable implementation and test identities at `itself` tip

| File | Git blob SHA-1 | SHA-256 |
| --- | --- | --- |
| `sparse_causal_bridge.py` | `55deec8dc42df82cc9f6d4120a1c83e43abdd640` | `0885d7244c3ea35367987ec59538d15c081d3ba6009897e5d0e5e42a24538ca7` |
| `latent_causal_bridge.py` (V4) | `8ad3e461427d3f8f3cf4f267521e9990910cbf5f` | `40306162c5d266a8ecb80d882202afb92fbc45aa5bc467a1004721706e57eef7` |
| `free_rollout_bridge.py` (V5) | `d77243d26adf914d84641d0ac6c65375a1adfb69` | `13d38836f3fef8ef6cbad35bb5b79fc41e15b21ebf00eec5d61604e7b3832cd6` |
| `test_sparse_causal_bridge.py` | `a7569d890d95be8e1b8b850ac5d8af61f17c7e97` | `5a123fe035355b5a1af1bc6501730dfd2a65def61a53d3b2d9614b8b1dcf402d` |
| `test_latent_causal_bridge.py` | `75bd55d421b42ab0ac2865e4d4b902ddf3dce411` | `8d217417446724a87b0e05803342497e0ad2229eb422f07b219186a98a2f8cbb` |
| `test_free_rollout_bridge.py` | `06233acb952f5ba85b68e01297dcab5e04eaf043` | `218ab4d82d7c265a0c1d4b6c5f55c238fb1da5ddbc2e26f31c231bcbd598a4a3` |

No V6 consensus implementation or test identity exists.

## 5. Current `main` AGI claim boundary

At `fcb754e`, every sparse-causal-bridge registration, implementation, runner,
test, result artifact, integrity artifact, and historical chapter is absent.
Accordingly, none of V1--V6 is an active `main` result or canonical AGI claim.
The branch can be used as historical evidence only.

The relevant current-main boundaries are internally explicit:

1. `README.md` calls the repository an unpublished research-hypothesis and
   numerical-experiment repository, with no arXiv registration or peer review.
2. `docs/7_AGI/1_AGI.md` calls the AGI series a `Bridge/Phenomenology` design
   document and separates `supported`, `bridge`, and unbenchmarked `hypothesis`.
3. The same document says large-scale benchmark verification remains
   incomplete and warns that transformer+backprop is not the self-organizing
   substrate assumed by the CE bootstrap proposal.
4. `docs/7_AGI/8_Roadmap.md` records transformer natural-emergence failures,
   STDP `NO-EFFECT`, held-out guard `FAIL`, and says genuine substrate
   verification still requires a separate full SNN experiment.
5. `docs/7_AGI/19_OOD_Generalization.md` bounds its evidence to one length axis,
   labels that step `bridge`, and leaves other OOD axes as unverified
   hypotheses.  Stronger prose in that chapter must be read inside that stated
   one-axis boundary.

Therefore the permissible bridge-line claim remains narrower than AGI even if
a future V7 passes: synthetic conditional forecasting/controller performance
in one fully observed, known-family, matched-basis world.  The historical line
does not establish open-world causal discovery, autonomous agency, brain or
U-fiber correspondence, consciousness, general OOD transfer, or AGI.

Current-main source blob identities for this boundary are:

- `README.md`: `4e27bde77da41687e2cf10d36d906dbe787b866c`;
- `docs/7_AGI/1_AGI.md`: `dc602e6584c89faf8f962e2e5ce6769a8f0f5634`;
- `docs/7_AGI/8_Roadmap.md`: `cc6ea241863478789310f5f338b126f02385fe30`;
- `docs/7_AGI/13_Verification.md`: `bd3b2247c8bd87417b3debdc73b61b7ca176cdcb`;
- `docs/7_AGI/19_OOD_Generalization.md`: `7a8e91ec001559cdf738004a4093816ccf5c412d`.

## 6. Reproducibility and evidence-quality limits

These limits must accompany any reuse of the historical results:

- V1--V5 preregistration timing is asserted by file status but not independently
  established by Git chronology.
- V2's integrity is explicitly post-run and soft; V4/V5 locks are stronger but
  still platform/newline sensitive and do not hash a complete environment.
- V1's defective implementation and V3's recorded implementation SHA are not
  reachable as source versions, so those two artifacts cannot be exactly
  replayed from the audited branch alone.
- The geometry proposal had prior truth-pair coverage by construction and a
  four-of-six pair budget.  It cannot evidence geometry-based causal discovery
  or probe-efficiency gain.
- The causal variants receive 512 ideal paired interventions that share the
  same hidden state and process noise within each pair.  Observation-only
  baselines do not receive that information; only `dense_probe` / the
  same-probe dense variants are equal-information controls.
- Generator and learner share the `tanh(source)` basis; chart identity, latent
  rank, latent AR invariance, and loading family are known design facts.
- V2/V4 locked tests change random seeds under the same synthetic family and
  fixed OOD loading, not the environment family or modeling assumptions.
- Only V5 cut all future chart-state reentry.  It failed, and its test remained
  closed.  V6 did not run.

## 7. Handoff facts for V7

- Preserve V1, V3, and V5 as failures; do not relabel their partial metrics as
  passed endpoints.
- Do not count the V6 registration pilot as V6 evidence.
- Treat V4 as the strongest passed parent only for sequential one-step
  prediction; treat V5 as the relevant failed parent for H20 free rollout.
- Historical validation/test/control seeds through V6 are disclosed in the six
  registrations and must not be reused for V7 evidence.
- Any V7 claim must retain the contract's narrow synthetic forecasting boundary.

This report records the evidence and its limits only.  It proposes no gate or
protocol change.
