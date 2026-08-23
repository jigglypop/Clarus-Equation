# BA-TR30 validation

Status: COMPLETE

## calibration (seed 117001, 1차 통과 — R-loop 미발동)

| 항목 | 결과 |
|---|---|
| status | `DEGREE_ID_CALIBRATION_PASS` |
| 주 fold 9: d̂=d\* / 은행 진리 선택 | 9/9 / 9/9 |
| max e (η=0 / 1e−3 / 1e−2) | 5.19e−15 / 9.07e−4 / 4.65e−3 |
| witness 기권 / 여유 min_d s′/τ | 1/1 / 372.9배 |
| shuffle 기각 / wrong-cue 진리 / 강제 d=1 게이트 실패 | 10/10 / 0 / 6/6 |
| max cond(Φ₃) | 66.6 (상한 1e6) |

## development (seeds 117101..117116, calibration 통과 후 개봉, 재조정 없음)

### endpoint 집계 — 주 fold 144 (48/η), witness 16

| 게이트 (§4.3) | η=0 | η=1e−3 | η=1e−2 |
|---|---:|---:|---:|
| 게이트 상수 e≤ | 1e−10 | 2e−2 | 2e−1 |
| 관측 max e | 4.50e−15 | 1.86e−3 | 1.50e−2 |
| 여유 배수 | ~2.2e4 | 10.7 | 13.3 |
| 게이트 초과 fold | 0/48 | 0/48 | 0/48 |
| 차수 식별 d̂=d\* | 48/48 | 48/48 | 48/48 |

| 항목 | 필수 (§4.3–4.5) | 관측 |
|---|---|---|
| 주 fold 통과 | 144/144 | 144/144 |
| 부 endpoint: 은행 진리 선택 (K=8) | 144/144 | 144/144 |
| 은행 영수증 (SHA-256 + 선행 순서 카운터) | 위반 0 | 위반 0 (`BANK_RECEIPT_FAIL` 0, 전 160 fold bank_counter < decision_counter) |
| witness (d=4, η=1e−3) `CLASS_EXTERNAL_ABSTAIN` | 16/16 | 16/16 (여유 min 169.0배, max 809.3배) |
| association shuffle endpoint 전 기각 | 성공 0/전 fold | 기각 160/160 (셔플 전부 기권) |
| wrong-cue 진리 선택 | 0 | 0/144 |
| 강제 d=1 on d\*∈{2,3} 주 게이트 실패 | 전 fold | 96/96 |
| `CUE_DEGENERATE` | — (fail-closed) | 0 (max cond(Φ₃)=104.3 ≤ 1e6) |
| confirmation 개봉 | 금지 | `confirmation_opened: false` (117201..117232 미개봉) |

### selection ablation (ρ=0, 진단 기록만 — 게이트 아님)

d̂_abl=d\*: 109/144. 불일치 35건 전부 저차 잡음 셀의 과대 차수 선택
((d\*,η)별: (1,1e−3) 11, (1,1e−2) 10, (2,1e−3) 8, (2,1e−2) 6) — ρ=0.5
파시모니 slack이 실제로 기여함을 보이는 진단. 판정에 미사용.

## focused 검증 (원문 결과)

```
.claude/hooks/python.cmd pytest tests/test_runtime_independent_stream_degree_id.py -p no:cacheprovider -q
2 passed in 3.58s
```

- test 1: studentized PRESS hat-matrix 항등식 = 명시 LOO 재적합 (상대차 ≤1e−8).
- test 2: calibration artifact 재분석 — 식별 9/9, 은행 진리 9/9, witness 기권, 영수증 순서, `forced_affine_gate_failure_count == 6`.

회귀 여부: 기존 파일 비수정(신규 3파일뿐)이므로 관련 회귀 확대 불요. 전체
pytest·전체 bench는 실행하지 않았다 (사용자 명시 요청 없음). 실행하지 않은
검증을 실행했다고 쓰지 않는다.

## 판정

계약 §4.3 주 endpoint(예측 게이트 3종 + 차수 식별 필수 + witness 기권 필수),
§4.4 부 endpoint(독립 은행 진리 선택 + 영수증), §4.5 대조군 4종 필수 결과가
calibration·development 전 fold에서 충족되었다. §4.7 STOP 조건 1–5 미발동.

기계 상태 문자열 (이론 지위 아님):

`DEGREE_ID_CALIBRATION_PASS` → `DEGREE_ID_DEVELOPMENT_GO`

**DEVELOPMENT_GO** — 이는 기계 판정이며, 주장 상한은 계약 §6 그대로다
(선언된 유한 다항 족·잡음 수준의 시뮬레이터 모델-클래스 식별·직접 예측까지;
실제 뇌·기억·의식·AGI 증거로 승격하지 않음). 형식 지위 부여·closure 판단은
auditor/오케스트레이터 소관.
