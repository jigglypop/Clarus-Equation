# 조합적 인과 OOD G5 사전등록

> 상태: `PRE-IMPLEMENTATION / LOCKED V1`
>
> 기계 계약: `../../experiments/preregistration/compositional_causal_ood_v1.json`

## 0. 비용 경계

외부 데이터, 네트워크, GPU, 실제 장비를 사용하지 않는다. NumPy CPU 합성 episode만 메모리에서 생성하며 보고서 JSON 외의 궤적은 저장하지 않는다. 목표 wall time은 15초 이하다.

## 1. 검증 수식

객체별 동역학은

\[
a_i=-k_1x_i-k_3\lVert x_i\rVert^2x_i-\gamma v_i+\omega R x_i+g\frac{u}{m_i}
\]

로 둔다. (R(x,y)=(-y,x))다. 계수 벡터는

\[
\theta=(k_1,k_3,\gamma,\omega,g)
\]

이며 adaptive-local 모델은 첫 20 step에서 공유 계수를 최소제곱으로 식별한다. 핵심 가정은 객체 수가 바뀌어도 같은 국소 법칙이 각 객체에 반복된다는 것이다.

## 2. OOD 구성

- train: 한 번에 주로 한 계수만 달라지는 8개 환경
- validation: 둘 이상의 계수가 함께 바뀐 6개 환경
- locked test: train에 없던 동시 조합 12개
- test 객체 수: 4개로 고정해 train의 2--3개보다 크게 설정
- calibration: 20 step
- free rollout: 20ㆍ100 step
- 속도 process noise: `0.001`

## 3. 비교 모델

- `pooled_local`: train 전체에서 하나의 평균 국소법칙을 학습
- `adaptive_monolithic`: test episode의 20 step에서 고차원 전역 전이를 직접 적합
- `adaptive_local_basis`: 동일 20 step에서 다섯 공유 국소계수만 적합
- `oracle_coefficients`: 참 계수 상한선, 경쟁 기준선 아님

## 4. 판정

G5는 다음을 모두 만족해야 한다.

1. adaptive-local 100-step RMSE가 pooled-local보다 30% 이상 낮다.
2. adaptive-monolithic보다 30% 이상 낮다.
3. 두 paired improvement의 정규근사 95% 하한이 0보다 크다.
4. 행동 개입효과 오차가 `0.02` 이하이다.
5. 계수 상대오차 중앙값이 15% 이하이다.
6. NaNㆍseed 누수ㆍ외부 다운로드ㆍtrajectory 저장이 0이다.

validation 실패 시 test를 열지 않는다. 함수족 누락이면 V2에서 basis를 추가하고, 짧은 calibration의 식별조건이 나쁘면 새로운 데이터를 받지 않고 action excitation만 설계한다. test를 본 뒤 V1을 고치는 것은 금지한다.

## 5. 루프 1 결과: V1 validation FAIL

V1의 adaptive-local 100-step RMSE는 `0.0519`로 pooled-local `0.0549`보다 5.5%만 낮았고, 계수 상대오차 중앙값은 `22.8%`였다. 한 seed의 local RMSE가 `0.2166`으로 튀었다. 행동 개입효과 오차 `0.00194`는 통과했다. adaptive-monolithic은 한 seed에서 수치 발산했다.

원인은 20-step calibration의 부드러운 사인 action이 다섯 basis를 분리할 persistent excitation을 충분히 제공하지 못한 것이다. V1 test는 열지 않는다.

## 6. V2 시험행동

V2는 데이터나 step 수를 늘리지 않고 첫 20 step action만 독립적인 이진 시험신호로 바꾼다.

\[
u_{t,j}=A\xi_{t,j},\qquad \xi_{t,j}\in\{-1,+1\}.
\]

목표는 calibration design matrix (X)의 최소 특이값을 키우는 것이다. 물리계수, validation/test seed, 통과 임계값은 유지한다. 수치적으로 발산한 monolithic baseline에는 모든 모델에 적용 가능한 유한 가속도 한도만 추가한다.

## 7. 루프 2 결과: V2 validation FAIL

PRBS는 개입오차를 `0.000615`로 낮추고 계수 중앙오차를 `17.3%`로 개선했지만, 충격 전이가 섞인 한 환경에서 참 계수 `[0.14, 0.28, 0.09, 0.06, 1.0]`을 `[-1.80, 4.18, 2.51, -0.068, 1.51]`로 추정했다. 그 결과 local 100-step RMSE가 `0.245`로 튀었다. V2 test는 열지 않는다.

## 8. V3 robust 식별

연속력 회귀와 충돌 impulse를 분리하기 위해 residual 상위 10%를 반복 제거한다.

\[
\theta^{(r+1)}=\arg\min_\theta\sum_{j\in S_r}\lVert y_j-X_j\theta\rVert^2,
\quad S_{r+1}=\{j:|r_j|\le Q_{0.9}(|r|)\}.
\]

마지막으로 등록된 물리적 compact set
`[0,.2] x [0,.4] x [0,.15] x [-.12,.12] x [.7,1.3]`에 사영한다. seed, step 수, test, 기준선과 통과 임계값은 바꾸지 않는다.

## 9. 루프 3 결과: V3 validation FAIL

robust 식별 후 local 100-step RMSE는 `0.0175`로 pooled보다 71.8%, monolithic보다 97.2% 낮아졌고 모든 seed가 안정화됐다. 개입오차도 `7.88e-5`였다. 유일한 실패는 계수 상대오차 중앙값 `17.19% > 15%`였다.

## 10. V4 식별 구간 확장

예측만 맞고 원인계수를 틀리게 분해하는 것을 막기 위해 15% 기준은 유지한다. 외부 자료나 episode를 추가하지 않고 동일 140-step episode의 완전관측 PRBS calibration을 20에서 40 step으로 늘린다. 이후 100-step free rollout은 그대로 유지된다.

## 11. 루프 4 결과와 V5

V4는 계수 중앙오차 `14.10%`, 평균 RMSE 56.2% 개선을 달성했지만 pooled 대비 paired 95% 하한이 `-0.00487`라 실패했다. 원인은 x/y 성분을 따로 trimming해 벡터 force basis가 서로 보상한 것이다.

V5는 한 객체ㆍ한 시점의 2차원 residual norm으로 transition 전체를 함께 선택한다. V4 계약을 상속하며 `robust_group_trim=true`만 변경한다.

## 12. 루프 5 결과와 V6

V5는 local RMSE를 `0.0279`로 낮췄지만 계수오차가 `18.3%`로 악화되어 실패했다. estimator 수정은 여기서 멈추고 더 나은 V4 scalar-robust 식별기로 돌아간다.

V6는 이미 본 환경을 재사용하지 않는다. 새 validation 20개와 locked test 30개에서 계수를 등록된 범위의 독립 균등분포로 결정 생성한다. 기존 30% 개선, paired CI, 계수 15%, 개입 기준은 유지한다.

## 13. V6 최종 결과: G5 PASS

locked test 30개에서 adaptive-local 100-step RMSE는 `0.0191`이었다. pooled `0.0660` 대비 71.0%, monolithic `0.5059` 대비 96.2% 낮았다. pooled 대비 paired 95% 하한은 `0.0342`, 계수 중앙오차는 `14.65%`, 개입오차는 `9.83e-5`였다. 실행은 1.26초, 외부 다운로드와 궤적 저장은 0이다.
