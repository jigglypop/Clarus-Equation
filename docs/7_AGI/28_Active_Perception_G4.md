# 최소 능동 지각 G4 사전등록

> 상태: `PRE-IMPLEMENTATION / LOCKED V1`
>
> 기준 모델: `27_Nonlinear_Object_Permanence.md`
>
> 기계 판독 계약: `../../experiments/preregistration/active_perception_v1.json`

## 0. 목적과 비용 상한

G4는 가려진 객체 중 무엇을 추가 관측할지 선택하는 능력을 검증한다. 실제 카메라ㆍ로봇ㆍ외부 데이터ㆍ유료 API는 사용하지 않는다.

- 외부 다운로드: `0 byte`
- backend: NumPy
- train trajectory: 기존 G3의 20개를 재사용
- validation/test: 각각 5개 메모리 생성 episode
- 대용량 trajectoryㆍ영상ㆍcheckpoint 저장: 금지
- 목표 실행시간: CPU 15초 이내

## 1. 수식

객체 (i)의 스칼라 불확실성은 관측하지 못한 한 단계마다

\[
P^-_{i,t+1}=P^+_{i,t}+Q
\]

로 증가한다. 센서가 객체를 관측하면

\[
K_{i,t}=\frac{P^-_{i,t}}{P^-_{i,t}+R},
\qquad
P^+_{i,t}=(1-K_{i,t})P^-_{i,t}
\]

로 갱신한다. 예상 정보이득은

\[
I(i;t)=\frac12\log\left(1+\frac{P^-_{i,t}}{R}\right)
\]

이다. 허용된 관측 시점에서 정책은

\[
i_t^*=\arg\max_{i\in\mathcal H_t}
\left[I(i;t)-\lambda_q\right]
\]

를 선택한다. (mathcal H_t)는 현재 가려진 객체 집합이다.

평가 비용은

\[
J=\frac1T\sum_t
\frac1{|\mathcal H_t|}\sum_{i\in\mathcal H_t}
\|\hat s^i_t-s^i_t\|_2^2
+\lambda_q\frac{N_{query}}T
\]

이다. 가려진 객체가 없는 시점의 상태항은 0으로 둔다.

## 2. 비교 정책

- `no_sensor`: 추가 관측 없음
- `fixed_round_robin`: 고정 순서로 객체를 조회하며 보이지 않는 경우도 예산을 소모
- `random_hidden`: 가려진 객체를 균등 무작위 선택
- `max_information_gain`: (I(i;t)) 최대 객체 선택
- `oracle_error`: 실제 상태오차 최대 객체 선택; 성능 상한이며 경쟁 기준선 아님

모든 정책은 4 step마다 최대 한 번이라는 동일한 호출 기회를 받는다.

## 3. 데이터 분리

- validation seeds: `6000--6004`
- locked test seeds: `7000--7004`
- 객체 수: 4
- 가림 길이: 10--30 step
- 미모델링 속도 교란 표준편차: `0.008`

validation은 구현 오류와 단위 점검에만 사용한다. 정책 수식이나 임계값을 바꾸면 V2 문서와 JSON을 새로 만들고 V1 실패를 보존한다.

## 4. 통과 조건

다음을 모두 만족해야 G4를 통과한다.

1. max-information-gain 평균 비용이 no-sensor보다 20% 이상 낮다.
2. random-hidden보다 10% 이상 낮다.
3. fixed-round-robin보다 10% 이상 낮다.
4. test 5개 seed 중 random-hidden을 4개 이상 이긴다.
5. query rate가 25% 이하이다.
6. 외부 다운로드와 trajectory 파일 저장이 0이다.
7. G1--G3 회귀 테스트가 유지된다.

## 5. 실패 루프

실패 시 순서대로 판정한다.

1. `수치 실패`: NaN, 잘못된 관측 순서, seed 비결정성은 수식 변경 없이 코드 수정한다.
2. `식별 실패`: (P)와 실제 오차가 무관하면 (Q)를 상태 의존 공분산으로 확장한다.
3. `목적함수 실패`: 정보이득은 크지만 task error가 줄지 않으면 (I(i;t)) 대신 예상 task-loss 감소를 사용한다.
4. `비용 실패`: 과잉 조회면 (lambda_q) 또는 query interval을 V2에서 변경한다.
5. test 결과를 본 뒤 V1 임계값을 고치는 것은 금지한다.

비싼 실험이 필요해지는 지점에서는 해당 분기를 `SKIPPED_COST`로 기록하고 합성 최소 실험으로 돌아온다.

## 6. 루프 1 결과: V1 validation FAIL

V1 validation은 외부 다운로드 0 byte, CPU wall 0.50초로 실행됐다. max-information-gain의 평균 비용은 no-sensor 대비 73.3%, random-hidden 대비 16.9%, fixed-round-robin 대비 52.7% 낮았다. 그러나 random-hidden 대비 seed 승리가 `3/5`여서 사전등록 기준 `4/5`를 통과하지 못했다.

실패 원인은 모든 객체에 동일한 (Q)를 둔 불확실성 모형이다. 이 경우 정책은 가림 나이만 구별하고 객체별 교란 민감도를 구별하지 못한다. V1 임계값과 결과는 변경하지 않는다.

## 7. V2 수식 변경

V2는 관측 가능한 질량에 따라 같은 외력이 만드는 속도 교란이 달라진다는 최소 가정을 추가한다.

\[
\epsilon^i_t\sim\mathcal N\left(0,\frac{Q_0}{m_i^2}I\right),
\qquad
P^-_{i,t+1}=P^+_{i,t}+\frac{Q_0}{m_i^2}.
\]

센서 정책, query interval, 비용, 통과 임계값과 test seed는 바꾸지 않는다. V2 계약은 `../../experiments/preregistration/active_perception_v2.json`에 별도 저장한다.

## 8. 루프 2 결과: V2 test FAIL

V2 validation은 random-hidden 대비 평균 비용을 22.1% 낮추고 seed `4/5`를 이겨 통과했다. 그러나 한 번만 연 locked test에서는 평균 비용을 27.9% 낮추면서도 seed 승리가 `3/5`여서 실패했다. V2 test seed는 이후 설계나 판정에 재사용하지 않는다.

이 실패는 평균효과와 5회 이항 판정의 불일치다. 정책 수식은 expected information gain을 최적화하므로 “모든 seed에서 random realization을 이긴다”가 아니라 기대비용 차이가 양수라는 주장을 검정해야 한다.

## 9. V3 통계 판정식

새로운 paired difference를

\[
d_j=J_{\mathrm{random},j}-J_{\mathrm{information},j}
\]

로 놓고 다음 정규근사 95% 하한을 사용한다.

\[
L_{0.95}=\bar d-1.96\frac{s_d}{\sqrt n}.
\]

V3는 (L_{0.95}>0)을 요구한다. 평균 비용 감소율 조건은 그대로 유지한다. validation 10개와 test 20개의 완전히 새로운 seed를 사용하고, V1ㆍV2 seed는 사용하지 않는다. 이 증가는 CPU 수 초 규모라 비용 상한 안에 있다.

## 10. 루프 3 결과: V3 PASS

V3 validation은 새 seed 10개에서 random-hidden 대비 평균 비용을 14.6% 낮추고, seed `9/10` 승리와 `L_0.95 = 4.66e-6 > 0`을 보였다.

한 번 연 locked test 20개 결과는 다음과 같다.

| 측정 | 결과 |
|---|---:|
| no-sensor 대비 비용 감소 | `75.5%` |
| random-hidden 대비 비용 감소 | `11.4%` |
| fixed-round-robin 대비 비용 감소 | `54.8%` |
| random 대비 seed 승리 | `14/20` |
| paired improvement 평균 | `7.50e-6` |
| paired improvement 95% 하한 | `1.03e-7` |
| query rate | `20%` |
| gate wall time | `1.44 s` |
| 외부 다운로드 / 궤적 파일 | `0 byte / 0개` |
| 최종 G4 | `PASS` |

95% 하한이 0보다 작게나마 양수이므로 사전등록 판정은 통과한다. 그러나 margin이 작아 강한 일반 능동지각 증거로 승격하지 않는다. 현재 지위는 `minimal synthetic active-perception engineering gate`다.

보존 보고서:

- `artifacts/agi/active_perception_validation_v1.json` -- V1 FAIL
- `artifacts/agi/active_perception_validation_v2.json` -- V2 validation PASS
- `artifacts/agi/active_perception_test_v2.json` -- V2 locked test FAIL
- `artifacts/agi/active_perception_validation_v3.json` -- V3 validation PASS
- `artifacts/agi/active_perception_test_v3.json` -- V3 locked test PASS
