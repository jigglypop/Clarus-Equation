# 31-validation — 최소 검증 기록

Status: COMPLETE

Date: 2026-08-22

## 0. 실행 환경

`.claude/hooks/python.cmd doctor` → `{"status": "PASS", "python": "...Python311\\python.exe", "python_version": "3.11.9", "bytecode_disabled": true, ...}` (기계 상태 문자열 원문, 이론 지위 아님).

## 1. focused 스크립트 1회 실행

명령 (artifacts/ 안에서 1회):

    .claude/hooks/python.cmd python verify_brain_recursive_bridge.py > verify_brain_recursive_bridge.run-20260822.log 2>&1

결과: **exit 0**, 전 assert green. 로그 전문: `artifacts/verify_brain_recursive_bridge.run-20260822.log`.

로그에 기록된 증인 (원문 값):

| 검사 | 로그 값 | 판정 |
|---|---|---|
| scalar branching (subcritical $D=0.8$) | `q_infinity = 0.9999999999999998` | 기록됨 |
| scalar branching (supercritical $D=1.2$) | `supercritical_small_root = 0.6863016689587826`, 고정점 잔차 assert < 1e-12 | 기록됨 |
| 양자 반례 | `x_unitary_period_two = true`, `dephasing_fixed_states_nonunique = true` | 기록됨 |
| 다형 branching | `spectral_radius = 0.43228756555322956`, `q_infinity = [1.0, 1.0]` | 기록됨 |
| activation projection 반례 | `unprojected_after_two_ticks = 1.4924` (bound 초과), `projected = 1.0` | 기록됨 |
| 기계 상태 문자열 | `status = PASS_MATH_WITNESSES_ONLY`, `implementation_parity = BLOCKED_NOT_TESTED` | 스크립트 자체 출력 (이론 지위 아님) |

### 1.1 기대 증인 대조 — 불일치 1건 (은폐 없이 기록)

20-audit §6이 기대한 벤치마크 수치 증인 $q_{\rm ext}=0.0486467196445741$, multiplier $F_D'(q_{\rm ext})=0.1545875231$ ($D=3.1777584234$)는 **이 스크립트의 로그에 없다**. 스크립트는 $D=0.8/1.2$ toy 평균만 검사하며 $D_{\rm eff}$ 벤치마크 분기를 포함하지 않는다. 명시적 무차원 검사 섹션도 로그에 별도로 없다 (무차원성은 11-math §2.1의 수학 감사로만 확보).

- 처리: 지시("스크립트가 실패하면 고치지 말고 기록")에 따라 스크립트 무수정. 실패는 아니고(전 assert green) **감사 기대 vs artifact 커버리지 불일치**로 기록.
- 원인 분류 (empirical_calibration_loop D→I→P→C→B→T): 수치 분기·잔차 시그니처 없음 — D(차원)/I(구현) 아님. 검사 대상 지정의 문제이므로 **P(프로토콜: 검사 범위 지정) 클래스**. 해당 벤치마크 수치의 정본 위치는 11-math §2.2 (residual $\approx-1.04\times10^{-16}$ 기록)이며 본 run에서 재검증하지 않았다.

## 2. focused 정본 문서 policy test

명령:

    .claude/hooks/python.cmd pytest tests/test_canonical_document_policy.py -p no:cacheprovider -q

(래퍼가 `--basetemp`를 자체 소유 — 호출자 지정 시 exit 2로 거부, 캐시 잔존물 없음 확인.)

결과 원문: **`3 failed, 8 passed in 0.60s`** — red를 green으로 표현하지 않고 그대로 기록한다.

| 실패 | 위치 | 본 레인 변경과의 관계 |
|---|---|---|
| machine verdict `pass` | `docs/9_등호이전/README.md:124` ("기계 pass는") | **pre-existing** — worktree 무수정 파일, HEAD 동일 내용 확인 (`git show HEAD:` 대조) |
| policy mirror drift `ce-validate/SKILL.md` | `.claude`/`.codex` | **pre-existing** — 양쪽 모두 worktree 무수정, HEAD에서 이미 미러 상이 확인 |
| missing link targets (다수) | `docs/6_뇌/11_리만계량_라우팅_논문.md` → 미생성 `_workspace` run 파일 | **pre-existing** — 타 레인의 dirty 파일, 본 레인 무접촉 |

수정한 3개 파일(`14_자기재귀성_대칭.md`, `9_우주론_수식_의미와_후보.md`, `00_선택과_접힘.md`)의 위반은 **0건**. 통과한 8개 항목에 본 변경을 직접 게이트하는 검사 포함: math delimiter 정규화, refuted parent 부재, `x_0\in[0,1/D]` 범위·잔류 측도 어휘, reduced Planck 규약, narrative 서두. 범위 밖 실패 3건은 감사 승인 범위가 아니므로 수정하지 않고 main에 인계한다.

## 3. 실행한 검사 / 생략한 검증

실행:

1. `.claude/hooks/python.cmd doctor` — 상태 문자열 상 이상 없음.
2. `artifacts/verify_brain_recursive_bridge.py` 1회 — exit 0, 로그 저장 (§1).
3. `tests/test_canonical_document_policy.py` focused 1회 — 3 failed(전부 pre-existing·범위 밖)/8 passed (§2).
4. 읽기 전용 `git status`/`git diff --stat`/`git show` — 인계용 스냅샷만, 상태 변경 없음.

생략 (실행하지 않았음을 명시):

- 전체 pytest suite: **SKIPPED (프로덕션 코드 무변경)**.
- guard bench·ASR·release 검증: **SKIPPED (프로덕션 코드 무변경, 사용자 명시 요청 없음)**.
- $D=3.1777584234$ 벤치마크 수치 재계산: 실행하지 않음 — 승인 검증 범위(스크립트 1회) 밖이며 §1.1에 불일치로만 기록.
- 같은 byte 재실행·확대 회귀: 실행하지 않음.

임시 산출물: 신규 cache·`.pytest_tmp_*`·venv 생성 0건 (래퍼 소유 basetemp만 사용).

Status: COMPLETE
