# CE-AGI 최신 연구 비교: 최종 보고서

Status: ABANDONED

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 비교 판정을 새로 수행하지 않았고,
존재하는 lane 파일과 artifacts의 상태만 기록한다. **이 보고서는 새 증거를 만들지
않았다.**

## 계약 질문

2026-08-15 기준 AGI·범용 에이전트 연구의 최신 1차 결과와 CE-AGI를 구조·구현·검증·
benchmark·재현성 단계(CR0--CR4)로 비교해 C1--C4를 판정하고 C5(축별 CR 등급과 우선
실험)를 산출한다.

## 존재하는 산출 (있는 그대로)

- `10-sources.md` (COMPLETE): 외부 1차 기준선 원장 E-R1 등, URL·공개일·제한 조건
  고정. 상세 검증은 `artifacts/external-source-ledger.md`.
- `12-routes.md` (COMPLETE): C4 한정 CR3 연결 후보 4종(R2 guarded continual shell,
  R1 residual-gated nested SCC, R3 interventional causal world model, R4
  verifier-monotone self-correction)과 killing gate 설계.
  상세는 `artifacts/route-design-ledger.md`.
- `artifacts/internal-validation.md`: 내부 conformance 실행 기록 — 541 passed
  (내부 conformance이며 외부 AGI benchmark가 아님, 파일 명문); V18b sealing 상태
  `4 failed, 9 passed`; STDP efficacy bench `NO-EFFECT`/`FAIL` 재현;
  `internal_math_probe.py` 산출.

## ABANDONED 근거

1. **판정 단계 산출 부재**: `11-math.md`, `20-audit.md`, `30-implementation.md`,
   `31-validation.md`가 존재하지 않는다. C1--C4의 판정과 C5의 CR 등급 산출은
   수행된 적이 없고, 어떤 lane 파일도 그것을 대신 담고 있지 않다.
2. **경로 대체**: 이 run 이후 연구 프로그램은 뇌 증거 사다리 트랙으로 이동했다
   (`.claude/CLAUDE.md`의 최우선 과제 규정과
   `_workspace/ce/brain-algorithm-route-ledger.md` ACTIVE 원장). 2026-08-23 기준
   `_workspace/ce`와 `_archive`에 이 비교 질문을 이어받은 후속 run은 없다.

완결된 두 lane과 artifacts는 부분 산출로 보존한다. 이 보존은 CE-AGI가 어떤 CR
등급을 갖는다는 판정이 아니다.

## 재개 조건

- 새 계약으로만 재개한다. 외부 스냅샷(2026-08-15)은 낡았으므로 `10-sources.md`와
  `artifacts/external-source-ledger.md`는 출처 최신성 재검증 후에만 재사용한다.
- `12-routes.md`의 R1--R4 killing-gate 설계를 재사용하려면 경로별 endpoint 독립
  등록 규칙(같은 파일 명문)을 그대로 유지해야 한다.
