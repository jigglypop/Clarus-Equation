# 안정 스냅샷 감사

Status: COMPLETE

Gate: PASS

Gate scope: schema-only apparatus admission probe

Empirical route: `BLOCKED_SOURCE_JOIN + BLOCKED_ASSIGNMENT`

Audited: 2026-08-20

## 안정 스냅샷

| 파일 | SHA-256 |
|---|---|
| `00-contract.md` | `def4b9e88ae730860dd8a6d18afd99c16cbe1c6e0d200b63a2143b5ca5e2950b` |
| `10-sources.md` | `de2ddab87564af6f723cee5cff38bab3d3b03bf6b5aa710889bb927ece9a38f3` |
| `11-math.md` | `258a46741ea2a50e6a7cb7761fec7425ee9be5465b42a5702bbae34522c36581` |
| `12-routes.md` | `5ccba338201562475b24873a30ee9659472296ca8320f70a1601b6d4eb1392e2` |
| `artifacts/audit_source_join_schema.py` | `597bc82aa7c3ff94f220cdd4ab410d0abe9077f820cbe41932eb9814645975e8` |
| `artifacts/source-join-schema-audit.json` | `26ed26e811d388ac15be61a27293b2dad7f4532c8a4099f9b049ccbe14a9cdc9` |

## 독립 감사 결과

**[산출]** 계약의 `Status: COMPLETE`는 contract freeze만 가리키도록 범위를 명시했다.
schema-only 구현은 동결 input과 금지 경계를 지켰다. response value, neural effect,
geometric match, endpoint와 threshold를 열지 않았으며 출처와 선행 hash도 일치한다.
따라서 제한된 admission probe 구현 gate는 통과한다.

**[산출]** empirical route는 통과하지 않는다. target table에는 source identity 열이나
외부 identity reference가 없고 event table에는 assignment/control receipt가 없다.
response-side `neuropal_ids`만으로 stimulation source를 정본 identity에 연결할 수 없다.

**[산출]** 수학 레인은 단순 full-state 예측 이득의 완전한 반례를 제시했고,
$M_{2,\mathrm{int}}$ 대 $M_{2,\mathrm{add}}$ contrast, animal/session holdout, positivity,
carryover와 optical adverse control을 요구한다. 현재 input은 그 estimand에 입장하지
못하므로 $\tau$, $D_B$, $\Delta_{\mathrm{config}}$ 계산 금지가 타당하다.

## 결함 등급

- P0: 없음.
- P1: 없음. 최초 감사에서 지적한 contract status 범위는 수정했다.
- 입력 blocker: `BLOCKED_EXPLICIT_SOURCE_JOIN`, `BLOCKED_ASSIGNMENT_RECEIPT`.

## 허가와 금지

이번 gate는 schema-only report의 재현과 최종 문서화를 허가한다. response matrix
열람, geometric threshold 선택, state encoder fitting, source-effect 계산과 empirical
GO 표시는 허가하지 않는다. 재개에는 outcome-blind immutable source join과 event-level
assignment/control receipt가 모두 필요하다.

