# Phase A post-implementation red-team review

Status: PASS_POST_RESULT_WITH_VALIDATION_PRECONDITION

Review stage: PRE_ONE_SHOT_COMPLETE; POST_RESULT_COMPLETE

## 1. 판정

Phase A V1 구현은 등록된 development block을 정확히 한 번 실행했다. 활성 과학·실행 P0는 0개, 활성 과학 P1은 0개다. 다만 잠긴 focused test에는 아래 §8의 combined-process interoperability P1이 남아 있으므로 repository-wide 단일 pytest process가 green이라고 보고하면 안 된다. 이 판정은 synthetic known-identity observed-context linear benchmark에만 적용되며 confirmation, SCC, memory, biology, consciousness 또는 AGI에 대한 승격은 허용하지 않는다.

confirmation marker는 실행 가능한 holdout이 아니다. 현재 지위는 `reservation_only`, `custody_unverified`, `not_executable_holdout`, `reserved_unopened`, `execution_authorized=false`다. 검증된 custodian/reveal artifact 또는 사전등록된 future-public-entropy selection rule이 없으므로 이 경계는 development PASS와 양립하지만 confirmation 실행·승격은 차단한다.

## 2. 잠긴 구현

독립 계산한 잠금 대상 SHA-256은 다음과 같다.

- contract: `a77ef203c5a9614043b5f5223547ef50f214234bde43c68ac8dc70309644fe6e`
- production module: `783b19a3d7e5e167203bfa49b8166b452bdd526729b87940b6033b25e3ebb8c8`
- focused test: `f1870e1ab67c837dcea317138472de235c93d93c564af925a37ba68e4239a8aa`
- development runner: `f311a1ae1ecabbfbb55ac966429f276d782f13629f29759ad685f0d93516bb7c`
- manifest file bytes: `708a664bde65848f3abd748126dfa44e798d9fa333bacd79d535864e3bdf97b1`
- canonical manifest self-hash: `78af635c524e05b35ba4a8b982f712d5174b99298bce1d94f827694c1c8733e2`

manifest self-hash와 네 required artifact hash는 모두 독립 재계산값과 일치했다. production source와 runner는 package initializer를 거치지 않는 private hash-name `compile`/`exec` 경로로 로드됐고, import sentinel 아래에서도 `reality_stone` 또는 `reality_stone.clarus`가 로드되지 않았다.

## 3. split contamination과 rotation

초기 development 후보 block은 focused test가 일부 graph에 대해 소형 config의 R1/R3·shuffle endpoint를 계산한 사실이 발견되어 전체가 `ABANDONED_PRE_REGISTRATION_TEST_CONTAMINATION`으로 폐기됐다. 해당 수치는 Phase A H1/H2의 증거나 반증으로 사용하지 않는다. manifest config의 공식 one-shot, full development aggregate, confirmation은 실행되지 않았다. 상세 범위는 `split-contamination-ledger.md`에 기록됐다.

rotation-2는 ledger에 고정된 outcome-independent domain string의 SHA-256 앞 7 hex를 시작점으로 하는 연속 24개 규칙이다. red-team이 digest와 manifest block의 일치를 독립 재계산했다. 새 development raw seed는 one-shot 직전까지 manifest 외 repository occurrence가 0이었고, focused test는 pilot block만 사용했다. development result 파일은 one-shot 승인 시점까지 존재하지 않았다.

## 4. 공격 결과

다음 문제를 one-shot 전에 재현하고 수정·회귀 고정했다.

1. 등록 식의 total Gaussian NLL 대신 per-coordinate mean NLL을 쓰던 endpoint 단위 불일치.
2. context label의 float, bool, 문자열 및 unsigned overflow를 int64로 손실 변환하던 fail-open.
3. manifest에 없는 `bootstrap_seed + 1` stream을 H2에 쓰던 preregistration 불일치.
4. evaluator 완료 뒤에야 output을 예약해 동시 프로세스가 둘 다 development seed를 열 수 있던 one-shot race.
5. numeric string과 `Decimal` config를 받아 원 타입으로 보존하던 scalar domain 위반.
6. identity fit을 같은 차원의 unknown-mix batch에 predict/NLL하던 observation-chart 혼합.
7. unknown-mix 또는 rank-deficient fit에도 latent coefficient error를 내던 secondary-claim fail-open.
8. 결과에 정확한 `confirmation_status: reserved_unopened`가 없던 schema 불일치.
9. benchmark replay가 아니라 `result == result`만 확인하던 tautological determinism test.
10. STOP predicate와 equal-context R1-vs-R3 no-strict-superiority kill test의 누락.

수정 뒤 joint `Kn+m` rank counterexample, residualized-input rank, exact recovery orientation, scorer-only common sigma, identity-shaped unknown-mix refusal, unclaimable coefficient refusal, nested duplicate JSON, traversal/hash tamper, confirmation authorization tamper, dimensionless refusal, pre-evaluator exclusive reservation, failed-run placeholder 보존을 다시 공격했고 모두 fail closed였다.

## 5. 독립 검증

- focused Phase A: `41 passed`
- legacy causal regression: `30 passed`
- Ruff: PASS
- compileall: PASS
- manifest canonical self-hash: PASS
- required artifact hash set: PASS
- rotation rule recomputation: PASS
- fresh development seed nonuse before one-shot: PASS
- development result absence before one-shot: PASS
- dirty package initializer sentinel: PASS
- confirmation function/CLI/raw namespace path absence: PASS

기본 pytest temp 경로는 sandbox 권한 오류가 있어 workspace 내부의 고유 `--basetemp`를 사용했고, resolved path가 run의 `artifacts` 하위인지 확인한 뒤 그 임시 디렉터리만 삭제했다. 첫 시도의 3개 setup error는 assertion 실패가 아니었고 재실행에서 모두 통과했다.

## 6. one-shot 허용 조건과 후속 감사

이 파일에 기록된 hash-lock 이후 contract, module, test, runner, manifest를 바꾸지 않은 상태에서만 registered development runner를 한 번 실행할 수 있다. runner는 manifest/hash 검증 뒤 evaluator 진입 전에 output을 `xb`로 예약한다. 실패 시 빈 reservation과 임시 증거를 보존하며 자동 재실행하지 않는다.

one-shot 뒤 red-team은 다음을 추가 확인해야 최종 판정을 낸다.

1. result의 manifest/implementation/required artifact hash가 위 잠금과 일치한다.
2. `confirmation_status=reserved_unopened`와 custody 제한이 유지되고 confirmation score, receipt, seed 또는 namespace가 없다.
3. graph-seed별 total NLL, delta, shuffle penalty, rank 진단, dof가 finite이고 집계와 직접 재계산값이 일치한다.
4. PA-H1/PA-H2 GO 또는 STOP이 사전등록된 mean·median·paired-bootstrap lower-bound conjunction과 정확히 일치한다.
5. 결과가 실패해도 이를 구현 성공이나 AGI 증거로 재해석하지 않는다.

## 7. post-result 판정

등록된 development one-shot은 exit 0으로 한 번 완료됐고 result SHA-256은 `7c4b9eb9ba08bed4cfc192262cc47c4a1cf56326526b3848cc6e4d5d89780df9`다. result는 canonical finite JSON bytes이며 임시 파일은 남지 않았다. manifest file/self hash, implementation hash, 네 required artifact hash가 pre-one-shot 잠금과 모두 일치했다. 등록된 24개 graph seed가 정확한 순서로 한 번씩 보고됐고 다른 seed는 없었다.

모든 graph에서 joint rank는 required rank `14`와 같았고 context state rank는 각각 `4`, residualized-input rank는 `2`, exact-edge conjunction은 true였다. 모든 수치는 finite였으며 per-seed `pooled_nll - factorized_nll`과 `shuffled_nll - factorized_nll`이 보고된 delta와 정확히 일치했다. nominal dof는 factorized `56`, pooled `24`, 차이 `32`였고 effective dof는 각 nominal bound 안에 있었다.

PA-H1은 development **GO**다. total-NLL delta의 mean은 `30285.541596706124`, median은 `17949.904754418854`, graph-seed paired bootstrap 95% interval은 `[17565.590274881404, 49970.74491150572]`였으며 nonpositive graph seed는 `0/24`였다. PA-H2도 development **GO**다. shuffle penalty mean은 `137135.5433569638`, median은 `125970.40032055078`, interval은 `[115336.15701641495, 159938.57170912955]`였으며 nonpositive graph seed는 `0/24`였다. red-team이 result 원장의 24개 값과 등록 bootstrap namespace만으로 두 mean, median, interval 및 GO conjunction을 독립 재계산해 모두 일치함을 확인했다; graph generator나 development evaluator는 다시 실행하지 않았다.

결과의 `confirmation_status`와 중첩 status는 모두 `reserved_unopened`였고 `reservation_only / custody_unverified / not_executable_holdout` 경계가 유지됐다. confirmation score, receipt, raw seed 또는 namespace field와 관련 산출물은 없었다. 따라서 최종 판정은 Phase A synthetic development 비교와 조건부 식별 fixture의 성공에 한정된다. 이는 strict 성능 정리, 외부 데이터 재현, 실행 가능한 blinded confirmation, SCC·기억·생물학·의식 또는 AGI의 증거가 아니다. result가 존재하므로 development runner 재실행은 금지된다.

## 8. validation invocation precondition과 release caveat

사전등록된 검증 명령처럼 focused test와 legacy causal regression을 서로 다른 pytest process로 실행하면 각각 `41 passed`, `30 passed`다. 반면 네 test 파일을 한 pytest process에서 함께 collect·실행하면 `69 passed, 2 failed`다. legacy module이 collection 중 `reality_stone.clarus`를 먼저 `sys.modules`에 넣은 뒤 focused test 두 개가 절대 조건인 `"reality_stone.clarus" not in sys.modules`를 검사하기 때문에 생기는 순서 의존 false failure다.

production과 runner의 isolated loader가 parent package를 새로 import했다는 증거는 아니다. red-team은 clean subprocess의 import sentinel과 전후 module snapshot으로 두 loader가 `reality_stone` 또는 `reality_stone.clarus`를 새로 로드하지 않음을 별도로 확인했다. 올바른 test는 절대 부재를 요구하는 대신 loader 호출 전후 snapshot을 비교하거나 clean subprocess에서 sentinel을 사용해야 한다.

현재 test bytes는 manifest와 one-shot result에 hash-lock됐으므로 이 test를 고치거나 manifest를 재봉인하면 완료된 result의 provenance chain이 깨진다. 따라서 V1 result는 그대로 보존하고, 현재 validation은 **focused standalone + legacy standalone이라는 등록된 isolated-invocation precondition 아래 PASS**로 판정한다. 동시에 일반적인 combined/full-suite 호출이 green하지 않다는 사실은 release-quality P1 caveat로 유지한다. 후속 V1.1 또는 새 manifest에서는 snapshot/subprocess 방식으로 test를 고치되 이 development runner나 result를 재실행해서는 안 된다.
