# Stable input-eligibility audit

Status: COMPLETE

Gate: PASS

## Snapshot

감사 대상은 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`,
`artifacts/cloudcell-input-audit.md`, `artifacts/inspect_cloudcell_inputs.py`,
`artifacts/cloudcell-input-audit.json`의 안정 snapshot이다. 이 PASS는 실제 자료를
분석할 입력 계약의 정합성만 뜻한다.

## 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| local archive bytes | PASS | 3 archive의 bytes/SHA-256이 markdown과 machine receipt에서 일치 |
| selected recordings | PASS | official logs의 22 recordings: GCaMP 11, GFP control 11 |
| MAT schema | PASS | neural arrays, behavior struct, neural/centerline clock, XYZ/correlation fields 확인 |
| time alignment | PASS_WITH_GUARD | behavior T 정렬; 첫 timestamp 중복과 gap은 고정 guard/window mask로 봉인 |
| output-Fisher input | PASS_INPUT | same-recording GCaMP와 future locomotion output 존재 |
| row-partition prediction | DIAGNOSTIC_ONLY | anatomy가 아닌 고정 row group의 held-out predictability만 가능 |
| anatomical routing | BLOCKED_SOURCE_TARGET_DEFINITION | canonical A/B 및 connectome join 없음 |
| causal routing | BLOCKED_INTERVENTION | randomized source intervention과 matched controls 없음 |
| task-context Xi | BLOCKED_CONTEXT | experimental context 없음; pre-t locomotor regime는 관측 stratifier뿐 |

## 안정 감사에서 닫힌 문제

1. Hallinen local cache와 nonlocal Randi source 상태를 분리하고 세 archive hash를
   provenance 문서와 machine receipt에 결합했다.
2. `AML310_moving.tar.gz`만 `AKS297.51_moving/` 내부 root를 갖는다고 명확히 했다.
3. MATLAB `behavior.pc_3`를 수학 표기 `pc3`에 명시적으로 연결했다.
4. AML18 lag 반례를 기존 사전등록 결과와 독립 계산 파일에 연결했다. 이 반례는
   기존 lag-memory 해석만 기각하며 새 Fisher tensor의 결과를 선취하지 않는다.
5. experimental context 부재와 pre-t locomotor-regime 관측 층화를 분리했다.
6. 기계 재검사에서 발견한 첫 timestamp 중복과 한 recording의 세 gap을 숨기지
   않고, 6-volume history/future, 12-volume leading guard, 60/20/20 split,
   boundary별 12-volume embargo, 3x-median gap rule과 정확한 anchor 목록으로
   고정했다. 모든 recording에서 세 split이 비어 있지 않다.

## 형식 지위

- **[정의/조건부 정리]:** output-relative conditional Fisher tensor와 chart law.
- **[경험 입력 판정]:** 11 GCaMP recording의 `PASS_INPUT`.
- **[진단 전용]:** arbitrary row-group predictive score.
- **[미완성/차단]:** anatomical A-to-B, causal routing, metric mediation, curvature.
- **[반례로 삭제]:** 이전 CloudCell lag score를 calcium memory 또는 neural
  routing으로 해석하는 주장.

## Gate 의미

PASS는 R1 metric-only preregistration 또는 Randi source-acquisition contract를
작성해도 된다는 뜻이다. 아직 $G$, $R$, effect size, biological mechanism을
측정했다는 뜻이 아니다.
