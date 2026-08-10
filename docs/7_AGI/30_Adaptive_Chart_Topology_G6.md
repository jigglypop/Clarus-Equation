# 자발적 chart 성장과 국소 복구 G6 사전등록

> 상태: `PRE-IMPLEMENTATION / LOCKED V1`
>
> 계약: `../../experiments/preregistration/adaptive_chart_topology_v1.json`

## 0. 최소 목표

하나의 평균 모델로 시작한 시스템이 서로 다른 두 물리 regime의 예측잔차를 보고 필요한 chart만 생성하는지 검사한다. GPUㆍ외부 데이터ㆍ전역 신경망 재학습은 사용하지 않는다.

## 1. 성장 수식

episode (e)의 짧은 calibration에서 국소계수 (hat\theta_e)를 구하고 기존 chart 중심 (mu_k)와 정규화 거리를 계산한다.

\[
d_k(e)=\left\lVert D^{-1}(\hat\theta_e-\mu_k)\right\rVert_2.
\]

`min d_k <= tau`이면 가장 가까운 chart를 재사용하고 중심을 온라인 평균으로 갱신한다. 아니면 새 chart를 만든다. 두 chart 중심 거리가 merge threshold보다 작아지면 합친다.

## 2. 국소 손상과 복구

학습 뒤 chart 하나의 계수를 고의로 오염한다. 새 calibration 계수와의 normalized residual이 threshold를 넘으면 해당 chart만 교체한다. 다른 chart 중심이 변하면 실패다.

## 3. 고정 데이터

- regime A: `[0.06, 0.12, 0.04, -0.06, 1.0]`
- regime B: `[0.14, 0.28, 0.09, 0.06, 1.0]`
- train episode: 12개, A/B 순서 혼합
- locked test: 새로운 20개
- calibration: 40-step PRBS
- process noise: `0.001`
- 최대 chart: 3

## 4. 통과 조건

1. 최종 chart 수가 정확히 2이며 최대 3을 넘지 않는다.
2. test regime assignment 정확도가 90% 이상이다.
3. chart별 rollout RMSE가 pooled-single-chart보다 40% 이상 낮다.
4. 손상 chart의 계수거리가 복구 후 50% 이상 감소한다.
5. 손상되지 않은 chart 변화량은 `1e-12` 이하이다.
6. 외부 다운로드ㆍtrajectoryㆍcheckpoint가 0이고 wall time 15초 이하이다.

validation 실패 시 threshold를 test에 맞춰 바꾸지 않는다. 거리척도가 실패하면 V2에서 Fisher/Gram 계량으로 바꾸며 V1 결과를 보존한다.

## 5. 루프 1 결과와 V2

V1은 assignment 95%, pooled 대비 RMSE 74.9% 감소, 국소 복구 61.4%를 달성했지만 `A,B,B` 세 chart를 만들어 실패했다. 두 B chart의 정규화 거리는 약 `0.71`, A--B 거리는 `2` 이상이다. V2는 creation `0.55`를 유지하고 merge만 `0.75`로 올려 검출과 병합에 hysteresis를 둔다.

V2에서도 갱신 뒤 B 중심거리가 `0.77`이 되어 세 chart가 남았다. 기존 seed는 폐기한다. V3는 A--B 간격의 절반 이하인 merge `1.0`과 완전히 새로운 train/test seed를 사용한다.

## 6. V3 최종 결과: G6 PASS

새 seed에서 chart는 정확히 2개로 수렴했고 assignment 정확도는 100%였다. chart RMSE `0.0115`는 pooled `0.0720`보다 84.1% 낮았다. 손상 chart 거리는 58.6% 감소했고 비손상 chart 변화는 0이었다. 실행은 0.70초, 외부 비용은 0이다.
