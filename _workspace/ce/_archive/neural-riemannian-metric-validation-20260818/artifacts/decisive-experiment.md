# 신경 계량 연쇄의 결정적 검증 실험

Status: FROZEN BEFORE E17 OUTCOME INSPECTION

## 판정할 주장

검정 대상은 다음 세 변수를 한 동물의 같은 세포 집단에서 시간 순서대로 잇는 주장이다.

$$
\Delta W^s \longrightarrow \Delta g_H \longrightarrow \Delta x_{0:T}.
$$

여기서 $W^s$는 직접 측정한 시냅스 또는 세포 간 연결, $g_H$는 사전에 고정한 동역학 모형에서 계산한 과정 잡음 도달가능성 계량, $x_{0:T}$는 계량 계산에 쓰지 않은 미래 단일 시행 신경 궤적이다. 표현 공간의 평균 거리 변화나 서로 다른 코호트의 세 측정값을 이어 붙인 결과는 이 주장을 검정하지 않는다.

## 실험 설계

성체 생쥐의 동일한 피질 영역에서 세포 식별자를 고정한 뒤, 두 광자 홀로그래픽 자극과 칼슘 또는 전압 영상을 사용해 세포별 인과 연결 행렬을 학습 전후에 반복 측정한다. 같은 세포 집단에서 자발 활동과 과제 활동을 별도 calibration block으로 측정하여 $J$, $Q$, gain과 지연을 추정한다. 그 뒤의 단일 시행은 미래 검정 자료로 봉인한다.

동물은 plasticity 유도, sham, gain/inhibition 조작의 세 군에 무작위 배정한다. plasticity 유도군 안에서는 연결 변화의 Frobenius norm, 자극 세포 수와 광 에너지를 맞추되 사전 모형이 예측하는 $\lVert\Delta g_H\rVert$가 큰 표적과 작은 표적을 짝짓는다. 이 대비는 단순한 자극량이나 전체 연결 변화량이 아니라 계량의 방향성 있는 예측이 필요한지를 판정한다.

각 동물의 측정 순서는 다음과 같이 고정한다.

1. 세포 등록과 기저 연결 $W^s_{\rm pre}$ 측정
2. 독립 calibration block에서 $F,J,Q,A$ 추정
3. 학습 또는 무작위 plasticity 개입
4. 같은 세포에서 $W^s_{\rm post}$ 재측정
5. 두 번째 calibration block에서 $J,Q,A$ 추정
6. 봉인한 미래 과제 시행과 행동 측정

동물 수는 결과를 본 뒤 정하지 않는다. 본 분석에서 제외할 pilot 동물로 동물별 primary-score 차이의 분산과 측정 실패율을 추정하고, 양측 제1종 오류율 0.05와 검정력 0.90을 만족하는 군별 수를 사전등록한다. 시행, 시간창, 세포와 시냅스는 반복 측정이며 독립 표본 수로 세지 않는다.

## 고정할 추정량

훈련 동물에서만 상태 chart, 차원 $r$, horizon $H$, reference tensor $R_0$, ridge $\lambda$, smoothing과 누락 처리법을 정한다. 시간가변 선형화에는 $J^k$ 대신 상태전이 곱 $\Phi_{t,k}$를 쓴다.

$$
C_H(z_t)=\sum_{k=0}^{H-1}\Phi_{t,k}Q_{t+k}\Phi_{t,k}^{\top},
\qquad
g_H(z_t)=\bigl(C_H(z_t)+\lambda R_0(z_t)\bigr)^{-1}.
$$

$R_0$는 chart와 함께 공변 변환한다. 고정한 $R_0=I$를 쓰는 분석은 직교 chart 변환만 허용한다. 이 $Q$ 기반 객체는 제어가능성 Gramian이 아니다. 제어 해석은 별도로 측정한 입력 channel $B$와 비용 $R$을 사용한 분석으로만 한다.

## 분할과 일차 종말점

최외곽 분할 단위는 동물이다. 세포 정합, chart, 모형 차수와 모든 초매개변수는 outer-train 동물 안에서만 맞추고, outer-test 동물의 미래 시행은 최종 점수 계산 때 한 번만 연다. 일차 종말점은 동물별 미래 궤적의 next-state log predictive density에서 직접 동역학 기준모형 대비 계량 모형의 차이다. 집단 추론은 동물별 차이를 사용한다.

이차 종말점은 사전에 지정한 과제 표적에 대한 endpoint error, hitting time과 drift-dependent path action이다. Riemannian 길이는 대칭이고 path action은 방향성을 가지므로 서로 대체하지 않는다. 연습으로 가까워졌다고 지정한 상태쌍과 정합한 비연습 상태쌍의 상호작용 방향도 미리 고정한다.

## 필수 기준모형

- persistence와 condition-label-only
- Euclidean latent distance
- firing rate와 공분산
- raw $W^s$ 또는 raw effective $W^e$
- 동일한 $F,J,Q$를 쓰는 직접 state-space dynamics
- 매개변수 수를 맞춘 unrestricted SPD metric
- global gain 또는 covariance scaling만 허용한 모형

계량 모형은 직접 state-space dynamics보다 동물별 proper score를 개선해야 한다. raw 연결이나 매개변수 수를 맞춘 SPD 모형과 동률이면 기하 고유 효과로 판정하지 않는다.

## 즉시 기각 조건

1. 같은 세포와 시냅스의 pre/post 식별 신뢰도가 사전 기준에 못 미친다.
2. test trajectory를 본 뒤 chart, $H$, $\lambda$, smoothing, 표적 또는 metric family를 골라야 효과가 난다.
3. bootstrap cell resampling이나 허용한 chart 변환에서 고유값과 길이의 부호가 불안정하다.
4. 예측한 거리 단축 방향과 실제 endpoint, hitting time 또는 action 변화 방향이 어긋난다.
5. geometry shuffle, 조건 label permutation 또는 time reversal에서도 점수가 유지된다.
6. global gain/covariance scaling이나 직접 동역학 모형이 같은 효과를 설명한다.
7. $\Delta W^s$, $\Delta g_H$, $\Delta x$가 같은 개체와 같은 시간축에 있지 않다.
8. $W^s$ 개입 뒤 $g_H$ 변화가 없거나, $g_H$를 크게 다르게 예측한 norm-matched 개입들이 미래 궤적에서 구별되지 않는다.

## 판정 문장

세 단계가 모두 통과하면 이 실험은 고정한 모형과 뇌 영역에서 연결 변화가 계산된 계량 변화와 미래 궤적 변화를 매개한다는 증거를 제공한다. 관측 자료만으로 예측 점수가 좋아진 경우에는 유효 동역학 대리 가설만 지지한다. 어느 경우에도 한 영역과 한 과제의 결과를 뇌 전체의 유일한 핵심 계량으로 자동 승격하지 않는다.
