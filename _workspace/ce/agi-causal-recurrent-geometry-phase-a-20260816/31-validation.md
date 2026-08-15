# Phase A V1 검증 기록

Status: COMPLETE

## 1. 최종 판정

구현 certificate `PA-I1`--`PA-I3`는 focused/red-team 검증을 통과했다. 등록된
development 비교에서는 `PA-H1 GO`, `PA-H2 GO`가 나왔다. 이는 좁은 synthetic
development 결과이며 strict 성능 정리, external replication 또는 blinded
confirmation이 아니다.

| endpoint | 결과 |
|---|---:|
| graph seed 수 | `24` |
| PA-H1 total-NLL delta mean | `30285.541596706124` |
| PA-H1 median | `17949.904754418854` |
| PA-H1 graph-seed paired bootstrap 95% | `[17565.590274881404, 49970.74491150572]` |
| PA-H1 nonpositive seed | `0/24` |
| PA-H2 shuffle penalty mean | `137135.5433569638` |
| PA-H2 median | `125970.40032055078` |
| PA-H2 graph-seed paired bootstrap 95% | `[115336.15701641495, 159938.57170912955]` |
| PA-H2 nonpositive seed | `0/24` |
| nominal dof R1 / R3 / 차이 | `56 / 24 / 32` |

모든 graph에서 joint design rank는 required rank `14`, 문맥별 state rank는 `4`,
residualized-input rank는 `2`였다. exact-edge conjunction은 모든 graph에서 true였지만
이는 선언한 known-identity linear class에서 coefficient/support를 평가할 조건이
충족됐다는 뜻이지 noisy finite fit의 support가 자동으로 진실이라는 뜻이 아니다.

## 2. focused와 회귀

기본 pytest temp root는 sandbox 권한 오류를 냈다. assertion 실패가 아니라 setup
permission error였으므로 workspace 내부 고유 `--basetemp`로 재실행했다. 임시 경로는
`.gitignore`의 `.pytest_tmp_*` 범위이고 공식 development seed를 사용하지 않았다.

```text
.venv\Scripts\python.exe -m pytest tests/test_causal_recurrent_geometry_benchmark.py -q --basetemp .pytest_tmp_phasea_focused_postresult
.........................................                                [100%]
41 passed in 0.89s
Exit code: 0
```

focused suite는 noiseless exact recovery, joint rank refusal, residualized input,
similarity no-go, mask/mix와 cross-chart refusal, coefficient claimability, total NLL
algebra와 sigma scaling, common bootstrap indices, byte-identical pilot replay,
equal-context R1/R3 동률, nested duplicate JSON, path/hash tamper, dimensionless refusal,
one-shot reservation/race/failure evidence와 confirmation 봉인을 공격한다.

```text
.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py tests/test_latent_causal_bridge.py tests/test_nested_scc_memory_benchmark.py -q --basetemp .pytest_tmp_phasea_legacy_postresult
..............................                                           [100%]
30 passed in 14.60s
Exit code: 0
```

legacy causal 3묶음에는 회귀가 없었다.

root의 별도 프로세스 최종 재현에서도 focused는 `41 passed in 0.79s`, legacy는
`30 passed in 15.22s`로 각각 통과했다.

### 2.1 단일 프로세스 결합 실행의 release-quality caveat

focused와 legacy를 같은 pytest 프로세스에 넣은 결합 실행은 깨끗하게 통과하지 않았다.

```text
.venv\Scripts\python.exe -m pytest tests/test_causal_recurrent_geometry_benchmark.py tests/test_sparse_causal_bridge.py tests/test_latent_causal_bridge.py tests/test_nested_scc_memory_benchmark.py -q --basetemp <workspace artifacts/pytest-root-final-01>
2 failed, 69 passed in 11.49s
Exit code: 1
```

실패 node는 다음 둘이다.

```text
test_isolated_load_hashes_the_exact_executed_bytes_without_parent_import
test_runner_isolated_load_and_manifest_validate_without_package_initializer
```

두 실패 모두 `assert "reality_stone.clarus" not in sys.modules`에서 발생했다. legacy 파일을
같이 collection하면 legacy 모듈이 `reality_stone.clarus`를 먼저 import하므로, 격리 로더가 새로
parent package를 import하지 않았더라도 절대 부재 assertion은 false가 된다. clean subprocess
sentinel과 전후 snapshot에서는 로더의 신규 parent import 없음이 통과했다. 따라서 이는 production
수치·certificate assertion 실패는 아니며 등록된 isolated-invocation precondition에서 sealed result의
유효성을 바꾸지 않는다. 그러나 combined/full-process regression은 `NOT CLEAN`이고 release-quality
P1 caveat다. 잠긴 V1 test·manifest·result는 사후 수정하지 않았으며, 이 assertion은 V1.1에서만
snapshot 또는 subprocess 기준으로 교체해야 한다. 사용한 resolved basetemp는 run의 `artifacts/`
하위임을 확인한 뒤 삭제했다.

## 3. 정적 검증

```text
.venv\Scripts\ruff.exe check reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
All checks passed!
Exit code: 0
```

```text
.venv\Scripts\python.exe -m compileall -q reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
Exit code: 0
```

```text
git diff --check -- .gitignore reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json examples/agi/causal_recurrent_geometry_development_run.py
Exit code: 0
```

```text
git check-ignore -v --no-index experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
.gitignore:55:!experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json
exact_exception_count=1
Exit code: 0
```

`.gitignore`에는 기존 사용자 변경을 보존한 채 승인된 exact exception을 한 번만
추가했다.

## 4. split contamination provenance

초기 후보 block은 `ABANDONED_PRE_REGISTRATION_TEST_CONTAMINATION`이며 official
development 또는 confirmation으로 세지 않는다. 실제 관찰 범위는 소형 config의 일부
R1/R3·shuffle fixture뿐이었고 manifest config aggregate는 실행하지 않았다.

rotation-2 선택 입력은 다음 문자열이다.

```text
CE-PHASE-A-V1|DEVELOPMENT-ROTATION-2|SEALED-MANIFEST-ONLY|2026-08-16
```

SHA-256은
`c308706f75cb59e37f7b0034fa562e6e9eafcdfb8f641be7fd2cafcbd6163108`이며,
`start=int(digest[0:7],16)`, block=`start+i` for `i=0,...,23`으로 정했다.
선택 당시 결과를 보지 않았고 manifest 삽입 전 nonuse와 삽입 후 manifest 제외 nonuse가
모두 occurrence `0`이었다. raw rotation block은 manifest와 official result 밖의
production/test/runner에 넣지 않았다. 상세 원장은
`artifacts/split-contamination-ledger.md`다.

## 5. one-shot 원문과 독립 사후 감사

실행 전 read-only preflight는 다음이었다.

```text
result_exists=False
required_hashes_match=True
```

정확히 한 번 실행한 stdout은 다음과 같다.

```text
{"PA-H1":"GO","PA-H2":"GO","mode":"development","result_path":"_workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/artifacts/development-results.json"}
Exit code: 0
```

result는 canonical finite JSON 한 줄, `29321` bytes이고 SHA-256은
`7c4b9eb9ba08bed4cfc192262cc47c4a1cf56326526b3848cc6e4d5d89780df9`다.
red-team은 evaluator를 다시 실행하지 않고 result 원장의 24개 seed-level 값에서
NLL delta, shuffle penalty, mean, median, 두 bootstrap interval, dof와 GO conjunction을
독립 재계산했다. 모든 값이 일치했다. post-result red-team의 최신 판정은
`PASS_POST_RESULT_WITH_VALIDATION_PRECONDITION`이며, 과학·실행 P0는 `0`, 과학 P1은
`0`, release-quality P1은 `1`이다. release P1은 root의 단일 프로세스 결합 실행에서
발견된 위 2.1절의 test-harness caveat다. 이는 post-result 수치 재계산 판정을 소급
변경하지 않지만 combined/full-process release gate는 `NOT CLEAN`으로 남긴다.

## 6. confirmation과 해석 제한

outer `confirmation_status`와 nested status는 모두 `reserved_unopened`다. 다음 경계를
그대로 유지한다.

- `reservation_only`
- `custody_unverified`
- `not_executable_holdout`
- `execution_authorized=false`

confirmation score, receipt, raw seed, RNG namespace, 함수, CLI mode와 산출물은 없다.
현재 commitment는 실행 가능한 blind holdout이 아니므로 이번 결과를 confirmation으로
승격하지 않는다. SCC 효능, 기억, 생물학, 의식 또는 AGI evidence field는 모두 false다.
제품 runtime, 정본, 기존 SCC·metric 코드와 package initializer는 수정하지 않았다.
