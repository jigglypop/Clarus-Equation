# Contract — agi-v14-binding-design-20260812

Status: COMPLETE
PREDECESSOR: _workspace/ce/agi-v13b-convex-spectral-20260812

## 질문

v13 계열(learnable gated local–cloud, convex/spectral 변형 포함)이 공통으로 보이는 두 실패를 **원리적으로** 해결하는 전이 방정식 족을 유도한다:

1. **T=8 시간 신용할당 붕괴** — σ 캡 1.0 / 1.25 / 무캡 전부에서 horizon·combined ≈ 0.52–0.56 (gru20 0.889). 게인(스펙트럴) 축은 원인이 아님이 실험으로 배제됨.
2. **held-out 조합 일반화 천장 0.55–0.59** — balanced(식별 가능) split에서 gru20 포함 전 모델 공통. 선형 이상 학습기 상한은 1.0이므로, 병목은 관측 시퀀스에서 latent (context, bits)를 복원해 **합성(compose)** 하는 능력의 부재.

## 정의역·과제 구조

- 과제: bits $b \in \{\pm1\}^3$ (+ distractor 1비트)가 tick 0의 local 채널에, context one-hot $c \in \{e_1,\dots,e_4\}$가 tick 1의 shared 채널에 주입되고, tick $2..T-1$은 순수 노이즈($\sigma \in \{0.04, 0.08\}$, input gain 0.5). 라벨:
$$y = \operatorname{sign}\!\big(w_{ctx}^{\top} b\big) = \operatorname{sign}\!\big(c^{\top} W b\big), \qquad W = \begin{pmatrix}1&1&1\\ 1&-1&-1\\ -1&1&-1\\ -1&-1&1\end{pmatrix}$$
- 즉 라벨은 latent 쌍의 **쌍선형(bilinear) 형식**이다. 관측은 노이즈 섞인 20채널 시계열.
- 평가 패널 5종: id/noise/horizon/combined (iid) + heldout (balanced 비보수쌍 split, 식별 가능성 검증 완료).

## 주장 (레인이 유도·검증할 것)

- C1 (충분조건, 수학): (i) 신호 tick에서만 쓰기가 열리고 닫힌 구간에서 고유값 정확히 1인 **무손실 슬롯 유지**(marginal stability), (ii) bits 슬롯과 context 슬롯의 **분리 저장**, (iii) readout이 두 슬롯의 쌍선형 형식 $c^{\top}\hat W b$을 표현 — 세 조건을 만족하는 전이·판독 족은 임의의 $T$와 heldout split에서 정확도 1(노이즈 유한 오차 내)을 달성한다. 이를 정리로 정식화하고 오차 상계를 노이즈 항으로 유도.
- C2 (필요성 방향, 수학): 판독이 상태의 **선형** 함수이고 상태가 latent의 연결(concatenation)만 저장하면(곱 항 부재) heldout 일반화가 원리적으로 제한됨을 보이거나 반례를 제시.
- C3 (경로, 탐색): C1을 만족하는 최소 구조 후보를 가능한 한 많이 나열하고 수치 평가 — 예: (a) salience-latch 슬롯 + 명시적 bilinear readout(외적 특징 $\hat c \otimes \hat b$ 위 선형), (b) 곱셈적 재귀 binding(상태 $h \leftarrow h \odot \rho(c)$, HRR/TPR형), (c) 게이트가 열릴 때만 갱신되는 key–value 슬롯, (d) 기타 구조적으로 다른 경로. 각 후보의 파라미터 수, 학습 가능성(gradient 경로), marginal stability 유지 여부를 표로.

## 기호

- 상태 $h_t \in \mathbb{R}^{n}$ (기존 예산 $n=20$ 기준, 필요 시 슬롯 구조로 재할당), 쓰기 게이트 $g_t \in [0,1]$, 유지 계수 $r$ (닫힘 구간에서 $r=1$ 허용 — 사용자 승인된 제약 완화, marginal stability).
- 판독 $\hat y = \operatorname{sign}(f_\theta(h_T))$. C1의 $f$는 쌍선형 특징 포함.

## 허용 오차·수치 평가 기준

- 수치 검증: 기존 벤치 러너(`examples/agi/local_cloud_v13_run.py`, seeds 9000–9015)와 동일 규약. 목표 게이트: G1 전 패널 ≥ 0.95×gru20, G3 heldout ≥ 0.90, G4 무결성 0.
- 수학 검증 계산: 부동소수 검산 상대오차 ≤ 1e-9, 정리 진술은 노이즈 항 명시.
- 탐색 레인의 toy 수치 평가는 축약 규모(4~8 seed) 허용 — 본 채점은 구현 단계(30/31)에서만.

## 경계

- 이 run은 **과제 특화 설계가 목적이 아니다**: 후보 구조는 "신호 이벤트를 붙잡아 유지하고, 둘 이상의 latent를 곱셈적으로 합성한다"는 일반 원리로 서술하고, 본 과제는 그 최소 시험대로 취급한다.
- 뇌 대응 주장 금지(Bridge 상한 규율 유지). CE 상수(p*, ε²) 도입은 이 run의 범위 밖 — 성공 구조가 나온 뒤 별도 run에서 검토.
- 기존 동결 파일 무수정. 산출물은 본 run의 artifacts/와 신규 파일로만.
