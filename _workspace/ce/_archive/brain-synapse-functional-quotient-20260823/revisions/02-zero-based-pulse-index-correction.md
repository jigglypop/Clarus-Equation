# Revision 02 — medium pulse index의 0-based 교정

Date: 2026-08-23

Status: `PREOUTCOME_SOURCE_CORRECTION / V1_EMPTY_MANIFEST_PRESERVED / MEDIUM_OUTCOMES_UNREAD / CONFIRMATION_UNTOUCHED`

Parent: `01-medium-event-preaccess-prereg.md`

## 교정 사유

Revision 01은 물리적 순서 “첫 8개 펄스의 과거 이력에서 다음 4개 펄스의 반응을
예측한다”를 데이터베이스의 `pulse_number=1..12`로 옮겼다. 그러나 medium r2.1의
구조 메타데이터만 조회한 결과 실제 저장 인덱스는 `0..11`이었다.

결과값, response QC, fit 성공 여부 및 target 결측을 읽지 않은 구조 진단에서 mouse
V1 coarse matrix와 pre-production의 IC dynamics sequence shape는 다음 두 행뿐이었다.

| synapse type | event rows | distinct pulse | min | max | sequence |
|---|---:|---:|---:|---:|---:|
| inhibitory | 12 | 12 | 0 | 11 | 49,858 |
| excitatory | 12 | 12 | 0 | 11 | 28,028 |

따라서 V1의 `min=1,max=12` 조건이 만든 빈 manifest는 생물학적 무효 결과가 아니라
source-index 변환 오류다. 빈 파일과 receipt는 삭제하거나 덮어쓰지 않고 음성 provenance로
보존한다.

## 허용되는 단일 변경

물리적 ordinal은 바꾸지 않는다. DB 인덱스만 다음처럼 교정한다.

| 의미 | Revision 01 표기 | medium r2.1 실제 index |
|---|---|---|
| causal history, 첫 8개 | pulse 1–8 | `pulse_number=0..7` |
| primary future, 다음 4개 | pulse 9–12 | `pulse_number=8..11` |
| sensitivity history, 첫 4개 | pulse 1–4 | `pulse_number=0..3` |
| sensitivity future, 다음 4개 | pulse 5–8 | `pulse_number=4..7` |

즉 식은 그대로다.

$$
H_8=\sigma\{c,z_{(1)},\ldots,z_{(8)}\}
\longmapsto
Y_8=\left(A_{(r)}/V_0,\ell_{(r)}/T_0,
\rho_{(r)}/T_0,\tau_{(r)}/T_0\right)_{r=9}^{12}
\in\mathbb R^{16},
$$

여기서 괄호 첨자 `(r)`는 1-based 물리적 순서이고 SQL의 실제 저장 index는 `r-1`이다.

## 바뀌지 않는 잠금

- slice hash split, E/I 분리, source cohort, protocol 조건을 바꾸지 않는다.
- target-blind slice-round-robin cap과 cap salt를 바꾸지 않는다.
- sequence key, input field, 16개 target field, dimensionless reference를 바꾸지 않는다.
- KRR/PCA grid, covariance, rank, gauge, control 및 ELPD gate를 바꾸지 않는다.
- V2 manifest도 outcome, response QC, fit value 및 target availability를 읽기 전에 고정한다.
- development와 confirmation outcome은 계속 SELECT 금지다.

V2 manifest schema version은
`BA-SRM2-TRAIN-MANIFEST-V2-ZERO-BASED`, train support auditor version은
`BA-SRM2-TRAIN-SUPPORT-V2-ZERO-BASED`로 구분한다. 이 교정 외의 outcome-driven
표본 변경은 허용하지 않는다.
