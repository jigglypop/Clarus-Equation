# Phase A adversarial red-team checklist

Status: READY_FOR_IMPLEMENTATION_REVIEW

Scope: `00-contract.md`, `11-math.md`, `12-routes.md`, `20-audit.md`, `dimensionless-audit.md`, `implementation-preflight.md`의 구현 경계를 공격적으로 검증한다. 이 문서는 제품 코드, 테스트, manifest, runner를 수정하지 않으며 구현 완료 뒤 동일 체크를 실제 코드와 결과에 적용한다.

## 판정 원칙

- **P0**: confirmation 접근/실행, truth 또는 test leakage, manifest/hash 우회, 잘못된 exact-edge `true`, package initializer 실행, 이미 존재하는 결과 덮어쓰기. 하나라도 있으면 build/final gate는 FAIL이다.
- **P1**: 공동 `B` 설계 오류, rank certificate 불충분, seed 역할 결합, bootstrap 단위 오류, scorer scale 비대칭, STOP 규칙 왜곡, 허용 범위 밖 주장. 해당 구현 claim은 미완성으로 남긴다.
- **P2**: 보고 필드·진단의 비본질적 누락. 핵심 판정을 바꾸지 않지만 수정 또는 명시적 제외가 필요하다.
- 공격 테스트는 단순한 정상경로 assert가 아니라 입력 하나만 바꾼 mutation, hash tamper, API signature/bytecode 관찰, subprocess 격리를 포함한다.

## 1. 데이터·seed·truth 누출

1. 동일 `(schema/version, master seed, role, graph index, replicate index)`는 byte-identical seed/result를 재현하고, role 또는 index 하나만 바꾸면 다른 digest/seed가 나와야 한다.
2. `graph`, `train_trajectory`, `heldout_intervention`, `evaluation_noise`, `shuffle`, `bootstrap`가 각기 독립 namespace인지 전 pair를 검사한다. 정수 `seed + offset` 또는 한 RNG를 순서대로 소비하는 구현은 P1이다.
3. pilot/development/confirmation seed 집합은 pairwise disjoint여야 한다. development API에 confirmation 첫 seed를 넣으면 RNG 생성 전에 예외가 나야 한다.
4. learner 함수 signature와 전달 객체를 검사해 `GroundTruth`, `A_z`, `B`, observation support, scorer sigma, held-out target, test seed/receipt가 들어가지 않는지 확인한다.
5. 같은 training batch에서 truth 행렬, scorer sigma, held-out `x_next`, evaluation-noise seed를 mutation해도 fitted coefficient bytes가 변하지 않아야 한다.
6. held-out batch와 training batch가 동일 object/memory를 공유하지 않는지 `np.shares_memory` 및 seed/shape metadata로 검사한다. 평가 노이즈를 재생성해도 training data/hash는 불변이어야 한다.
7. candidate와 pooled arm이 동일 training batch, 동일 held-out batch, 동일 ridge와 동일 scorer sigma를 받는지 object/content digest로 검증한다.
8. graph seed가 통계 단위인지 확인한다. trajectory/frame을 독립 seed처럼 부풀리거나 한 graph의 여러 frame을 bootstrap하는 경우 P1이다.

## 2. shared-B stacked design와 rank certificate

1. 학습 design row는 `[1(z=0)x; ...; 1(z=K-1)x; u]`이고 shape은 `N x (K*n+m)`이어야 한다. target orientation은 `Y = Phi @ W.T`, `A[target, source]`인지 비대칭 fixture로 검증한다.
2. noiseless known-identity fixture에서 `rank(Phi)=K*n+m`일 때 모든 `A_z`와 공동 `B`의 최대 절대오차가 `<=1e-10`이어야 한다.
3. 각 `X_z`는 full row rank지만 `U_z=R_z X_z`인 fixture를 넣어 joint rank가 부족하고 exact-edge certificate가 false인지 검사한다. per-context `[X_z;U_z]` rank만으로 true가 되면 P0이다.
4. certificate는 required/observed joint rank, tolerance, joint singular values, context별 state rank, residualized-input singular value 또는 Gram 최소 eigenvalue를 보고해야 한다.
5. rank tolerance 경계에서 동일 singular spectrum에 대해 fit과 certificate가 서로 다른 기준을 쓰지 않는지 검사한다.
6. ridge가 역행렬을 만들었다는 이유로 deficient design을 identified로 승격하지 않아야 한다. ridge `>0`이어도 exact-edge는 false다.
7. factorized nominal dof `n(Kn+m)`, pooled dof `n(n+m)`, 차이 `(K-1)n^2`를 여러 `(n,m,K)` 조합으로 property test한다. effective dof를 보고한다면 동일 design/ridge 정의여야 한다.
8. equal-context negative fixture에서는 R1의 엄격 우월을 요구하거나 보고하지 않아야 한다.

## 3. scorer-only 공동 sigma

1. manifest의 normalized generator sigma는 finite이고 `>0`이어야 하며 learner 입력/모델 객체에는 없어야 한다.
2. candidate와 pooled NLL은 정확히 같은 manifest sigma와 같은 샘플 수/차원을 사용해야 한다.
3. 직접 SSE로 계산한 `delta=(SSE_pooled-SSE_factorized)/(2*sigma^2)`와 보고된 NLL 차이가 수치 허용오차 내 일치해야 한다.
4. sigma를 두 배로 바꾸면 fitted coefficients는 그대로이고 NLL의 상수항과 delta scaling만 규약대로 바뀌어야 한다.
5. model별 train/test residual로 sigma를 재추정하거나 test target을 scale calibration에 쓰는 경로가 없어야 한다.
6. noiseless exact-recovery fixture는 sigma 0으로 NLL을 계산하지 않고 coefficient error만 채점해야 한다.

## 4. observation mode와 exact-edge fail-closed

1. exact-edge 값은 문자 그대로 `known_identity AND declared_linear_class AND full_rank AND finite_valid_inputs`의 conjunction과 같아야 한다.
2. 네 조건을 하나씩 false로 바꾸는 4개 mutation에서 모두 false가 되어야 한다. 빈 문자열, unknown enum, NaN/Inf, invalid context/shape도 예외 또는 false여야 한다.
3. `known_mask`와 `unknown_mix`에서 prediction API가 있더라도 exact-edge는 항상 false다. identity-shaped unknown mix도 type이 unknown이면 false여야 한다.
4. similarity transform으로 관측열은 같고 latent off-diagonal support가 다른 정확한 PA-N1 fixture를 재현한다.
5. coefficient/support error는 exact-edge certificate가 true인 known-identity case에서만 claimable해야 한다. 다른 mode의 latent edge metric은 누락 또는 명시적 `not_claimable`이어야 한다.

## 5. confirmation 봉인

1. runner CLI/help와 production public callable을 정적·동적으로 조사해 `confirmation`, `holdout`, `unseal` 실행 mode/function이 없는지 확인한다.
2. manifest에는 confirmation metadata가 있어도 `status=reserved_unopened`, `execution_authorized=false`여야 한다. 결과도 `confirmation_status=reserved_unopened`만 기록하고 score/receipt가 없어야 한다.
3. confirmation seed를 development evaluator/namespace 함수에 넣으면 seed digest/RNG 생성 전에 거부되는지 monkeypatch로 호출 순서를 확인한다.
4. manifest의 authorization을 true로 tamper하거나 confirmation status/mode를 바꾸면 manifest 검증 단계에서 실패해야 한다.
5. repository 전체에서 Phase A confirmation result/receipt/score 산출물을 검색해 없어야 한다. 과거 V9/V15/V16/V17/V18b seed/result를 import하거나 복사하지 않아야 한다.

## 6. canonical manifest·hash·경로 공격

1. duplicate JSON key를 허용하는 기본 `json.loads`만 쓰지 않아야 한다. top-level과 중첩 object의 duplicate key 모두 거부한다.
2. UTF-8, sorted keys, compact separators, self-hash 제외 정책 등 선언된 canonicalization으로 manifest digest를 독립 재계산한다.
3. production/test/runner/contract 등 required locked file set이 정확히 검증되고, 각 파일 한 byte mutation에 실행이 실패해야 한다.
4. manifest self-hash field가 있다면 그 field만 제외한 digest 규칙이 단일하고 재현 가능해야 한다. self-reference를 임의로 두 번 해시하거나 raw formatting에 의존하면 P1이다.
5. locked path에 absolute path, `..`, `.`, backslash, repeated separator, symlink escape, case/Unicode alias를 주입해 repository root 밖 또는 다른 파일을 가리키지 못하게 한다.
6. required path 누락, 추가 path, duplicate normalized path, path/hash 역할 교환을 모두 거부한다.
7. source bytes는 한 번 읽은 buffer를 hash한 뒤 같은 buffer를 `compile/exec`해야 한다. hash 후 재-read하는 TOCTOU 경로는 P0이다.
8. manifest JSON이 `.gitignore`에서 제외되어 추적 가능하고 exact exception이 정확히 한 줄인지 확인한다.

## 7. development one-shot

1. runner는 manifest 하나만 받고 development 외 mode 선택지를 제공하지 않아야 한다.
2. output이 이미 존재하면 내용을 읽거나 수정하기 전에 실패해야 한다. overwrite/force 플래그, 자동 suffix, timestamp 새 결과 경로가 없어야 한다.
3. temporary file은 destination과 같은 디렉터리에 쓰고 `os.replace`로 원자 교체하며, 성공 뒤 임시 파일이 없어야 한다.
4. subprocess 두 개의 동시 실행 race를 시도해 둘 다 결과를 쓰지 못하게 한다. 단순 `exists()` 후 `os.replace()`만으로 두 실행이 모두 계산·덮어쓸 수 있으면 P0; 최소한 exclusive claim/lock 또는 문서화된 단일 실행 증거가 필요하다.
5. serialization은 `allow_nan=False`, deterministic key order와 newline을 사용하며 같은 입력은 동일 bytes다.
6. development 결과는 graph-seed별 NLL/delta, mean, median, paired bootstrap interval, dof, integrity gates, GO/STOP, exclusions, Python/NumPy/manifest/implementation hash를 포함해야 한다.
7. 개발 결과를 본 뒤 manifest/threshold/code가 바뀌지 않았는지 모든 locked hash와 result provenance를 다시 계산한다.

## 8. isolated import와 dirty initializer

1. test와 runner 모두 production source bytes를 직접 읽고 private digest module name으로 `compile/exec`해야 한다.
2. subprocess에서 `reality_stone`, `reality_stone.clarus`, dirty `clarus.__init__`에 sentinel exception/import hook을 두어도 benchmark load가 성공하고 해당 package가 `sys.modules`에 새로 생기지 않아야 한다.
3. production module에 package-relative import, 기존 SCC/metric/runtime/default import가 없는지 AST로 검사한다.
4. isolated loader는 실패 시 임시 module을 `sys.modules`에서 제거하며, 같은 hash를 다시 load해도 stale mutable singleton을 공유하지 않는지 검사한다.
5. manifest의 implementation hash와 실제 실행된 module bytes digest가 동일한지 결과까지 연결한다.

## 9. 통계·negative control

1. primary `delta_s`는 graph seed별 paired 값이며 bootstrap resampling unit도 graph seed다. bootstrap RNG는 별도 namespace이고 development seed 목록만 사용한다.
2. bootstrap interval의 resample count, percentile convention, seed가 manifest와 일치하고 직접 독립 구현한 결과와 재현되어야 한다.
3. frame/transition NLL 합계는 graph별 길이가 동일한지 확인한다. 다르면 합계가 긴 graph를 과대가중하지 않도록 prereg 규약이 있어야 한다.
4. input/time shuffle은 held-out target, state, context, evaluation noise는 고정하고 input 대응만 prereg 방식으로 깨야 한다. 새 noise/trajectory 생성으로 nuisance까지 바꾸면 P1이다.
5. shuffle 결과의 방향과 gate를 독립 계산한다. intact보다 악화되지 않으면 PA-H2는 STOP이어야 하며 구현 성공으로 바꿔 쓰면 안 된다.
6. mean delta가 비양수면 PA-H1은 STOP이다. 일부 seed 승리, median, bootstrap 하한을 사후 선택해 GO로 바꾸면 P1이다.
7. hyperparameter/ridge/context basis가 manifest에서 고정되었는지 확인한다. 여러 후보 중 development endpoint로 선택했다면 selection ledger/nested split 없이는 target-aware P1이다.
8. R2/R4 또는 coefficient/support secondary가 primary R1-vs-R3 결과를 대체하지 않아야 한다.

## 10. 과장·범위 침범

1. result/report claim certificate에 SCC, memory, biological connectome, consciousness, AGI가 true로 등장하지 않아야 한다.
2. 허용 문구는 normalized finite-dimensional observed-context linear synthetic benchmark와 conditional identification/prediction에 한정한다.
3. `dimensionless=true`는 단위 일관성만 뜻하며 causal validity, physicality, AGI 증거로 연결하지 않는다.
4. anatomy, latent causal support, observed-coordinate predictive transition을 typed field 또는 명시적 exclusion으로 분리한다.
5. 기존 runtime, `__init__.py`, SCC/metric/defaults, 정본/과거판본 파일이 변경되지 않았는지 scoped diff와 전체 status로 확인한다.

## 구현 완료 뒤 실행 순서

1. 제품·test·manifest·runner 및 `30-implementation.md`, `31-validation.md`, development result를 EOF까지 읽는다.
2. 정적 AST/signature/hash/path audit를 먼저 수행한다.
3. focused tests를 그대로 실행한 뒤 위 mutation을 별도 임시 사본/프로세스에서 수행한다. 원본 manifest/result는 수정하지 않는다.
4. legacy causal regression, Ruff, compileall, `git diff --check`, `git check-ignore`를 독립 재실행한다.
5. P0/P1/P2를 근거·재현 명령과 함께 `post-implementation-review.md`에 기록하고 parent에 최종 판정을 전달한다.
