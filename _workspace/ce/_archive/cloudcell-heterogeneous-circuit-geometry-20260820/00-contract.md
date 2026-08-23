# CloudCell 이질적 회로-계량 실데이터 검증 계약

Status: FROZEN_BEFORE_ENDPOINT

PREDECESSOR:

- `docs/6_뇌/11_리만계량_라우팅_논문.md` 식 (21)–(26)
- `_workspace/ce/cloudcell-real-brain-metric-routing-20260820/artifacts/cloudcell-input-audit.json`
- `_workspace/ce/self-recursive-reference-cosmology-quantum-20260820/artifacts/brain-recursive-bridge-equations.md`

## 1. 목적과 주장 상한

실제 C. elegans 전뇌 칼슘 기록에서 뉴런별 역치와 회로별 강도를 반영한 계량 라우팅 식을 시험한다. 원래 식의 `parent responsibility` \(C_{ij}^n\)은 검증된 물리적 parent receipt나 개입이 있어야 한다. CloudCell에는 둘 다 없으므로 원래 인과 해석은 endpoint를 열기 전에

`BLOCKED_PARENT_RECEIPT`

로 판정한다. 이는 경험적 영가설 기각이 아니라 장치·식별성 실패다.

그 다음 별도의 관측 모형 \(\pi_{ij}\)를 사용한다. 이 모형의 결과는 오직 **held-out 1-step neural prediction에 유용한 관측적 graph feature인지**만 말한다. 실제 시냅스 부모, 물리적 피질 변형, 인과적 회로 형성, AGI 성취는 주장하지 않는다.

## 2. 고정 데이터

공식 공개자료의 로컬 고정본 `data/external/cloudcell/`만 읽는다. archive SHA-256은 predecessor audit에 고정되어 있다. primary는 GCaMP이며, `XYZcoord`의 모든 행이 유한한 다음 다섯 recording만 schema로 선택한다.

- `BrainScanner20200310_141211`
- `BrainScanner20200310_142022`
- `BrainScanner20170424_105620`
- `BrainScanner20170610_105634`
- `BrainScanner20180709_100433`

공간 nuisance falsifier는 유한 XYZ를 가진 GFP `BrainScanner20210503_122703` 하나다. recording row는 같은 recording 안에서만 동일 뉴런으로 취급한다. `acorr`, `cgIdx`, recording 간 row 번호는 구조 연결이나 identity로 쓰지 않는다.

첫 장치 시도에서 선택한 released `Ratio2`는 `BrainScanner20200310_141211`에서 전체 원소의 61.9%만 유한했고 공식 Python loader의 neural signal도 아니었다. endpoint 계산 전 확인한 schema 오류로 보존한다. 수정본은 released `gRaw/rRaw`를 읽고, `K`에서만 뉴런별 \(g_i=\beta_i r_i+a_i+I_i\)를 적합하여 \(I_i=g_i-\beta_i r_i-a_i\)를 신호로 고정한다. 이는 공개 loader의 linear red-channel nuisance subtraction을 누출 없이 재현한 것이다. \(I\)는 spike가 아니라 보정된 칼슘 형광량이다.

## 3. 시간 역할과 누출 방지

predecessor의 leading guard 12 volumes, gap rule, 60/20/20 chronological split, 경계 양쪽 12-volume embargo를 그대로 사용한다.

- `K` calibration: 기존 train anchor의 앞 1/3. 뉴런별 \(\mu_i,s_i,\vartheta_i\)만 산출한다.
- `B` construction/fit: 기존 train anchor의 뒤 2/3. \(\pi,\alpha,\Gamma,k,g^{eff}\)와 predictor를 만든다.
- `V` validation: 기존 validation anchor. 1차 관측식의 수학적 실패 양상만 진단한다.
- `T` confirmation: 기존 test anchor. 정확히 한 번만 최종 동결식을 평가한다.

`T`는 식·역치·cycle·그래프·regularizer·horizon·seed 선택에 사용하지 않는다. history와 future는 각각 1 volume으로 고정하며 clock gap을 가로지르는 쌍은 제거한다.

## 4. 고정 전처리와 이질적 역치

각 뉴런에 대해 `K`에서

\[
\mu_i=\operatorname{median}(x_i),\qquad
s_i=1.4826\operatorname{median}|x_i-\mu_i|,
\]

\[
z_i^n={x_i^n-\mu_i\over\max(s_i,10^{-6})},\qquad
e_i^n=\mathbf 1[z_i^n\ge2.5],\qquad
q_i^n=e_i^n\min\left(1,{[z_i^n-2.5]_+\over2.5}\right).
\]

MAD가 \(10^{-6}\) 이하이거나 `K+B`에서 유한 비율이 75% 미만인 뉴런은 endpoint를 보지 않고 제외한다. 첫 장치 실행에서 모든 `K/B/V/T` 표본이 유한해야 한다는 구현 조건은 released 자료의 산발적 NaN 때문에 한 기록을 8개 미만 뉴런으로 축소했다. endpoint 계산 전 발견된 이 과잉 조건은 장치 오류로 보존한다. 수정본은 finite XYZ와 `K+B` 75% 조건으로 뉴런을 고정하고, 남은 모든 split의 결측 standardized sample을 calibration 기준값 \(z=0\)으로 대입한다. 공통역치 모형은 primary가 아니라 robustness control이다.

## 5. 동결 분석 A와 허용된 수정

### A0: 원래 인과식

검증된 \(C_{ij}^n\), delay, STP receipt가 없으므로 계산하지 않고 `BLOCKED_PARENT_RECEIPT`로 보존한다. 상관·시차·회귀계수로 \(C=1\)을 대입하지 않는다.

### A1: 관측 예측식

`B`에서 ridge \(W^{(0)}\)로 \(z^{n+1}\)을 \(z^n\)에서 예측한다. orientation은 row=receiver, column=sender다. 고정 ridge는 \(\lambda=10^{-2}\)이며 intercept는 별도다.

\[
\pi_{ij}={|W^{(0)}_{ij}|\over\sum_r|W^{(0)}_{ir}|+10^{-12}},
\quad
\omega_{ij}^n=\pi_{ij}q_j^nq_i^{n+1},
\quad
\alpha_{ij}=|B|^{-1}\sum_{n\in B}\omega_{ij}^n.
\]

\(\pi\)는 parent probability가 아니라 fit-split predictive responsibility다.

좌표를 각 recording의 median 6-NN 거리로 무차원화한다. 각 6-NN 무방향 edge에서 \(\alpha_{ij}\)와 \(\alpha_{ji}\) 중 큰 방향 하나만 고정하고(동률은 작은 row index가 sender), 이 directed graph의 크기 2 이상 strongly connected component 내부 edge만 cycle-support \(\Gamma\)로 쓴다. cycle-support edge가 2개 미만이면 해당 recording은 `NO_CYCLE_SUPPORT`이며 임의 cycle을 만들지 않는다.

노드 source \(s_i^\Gamma\)는 cycle-support incident \(\alpha\)의 합을 평균 1로 정규화한 뒤 평균을 뺀 값이다. \(L\)은 symmetric normalized graph Laplacian이며

\[
(I+L)k=-{1\over2}s^\Gamma,
\quad
g_i^{eff}=e^{k_i}I_3,
\]

\[
R_{ij}=\Pi_{[1/2,2]}\exp\left[-{(e^{(k_i+k_j)/2}-1)\ell_{ij}^2\over \sigma_g^2}\right],
\quad \sigma_g=1.
\]

A1은 post-fit \(W^{post}_{ij}=W^{(0)}_{ij}R_{ij}\)를 사용한다. validation proper-score contrast는 같은 frozen Gaussian residual variance로 계산한 baseline 대비 평균 log score 차이 \(\Delta_s^V\)다.

### A2: 실패 뒤 허용되는 단 한 번의 수학 수정

A1 validation이 음수이거나 5개 중 4개 미만에서 양수가 아니면 A1은 실패다. 그때만 validation에서 다음 기계적 실패 영수증을 산출한다: \(\|W^{post}\|_F/\|W^{(0)}\|_F\), spectral radius ratio, prediction variance ratio.

수정 A2는 post-fit 곱을 폐기하고 같은 고정 \(R\)을 ridge penalty에 넣는다.

\[
W^{reg}=\arg\min_W\sum_{n\in B}\|z^{n+1}-Wz^n-b\|_2^2
+10^{-2}\sum_{ij}{W_{ij}^2\over R_{ij}}.
\]

이 수정은 `V`에서 구현·검산한 뒤 파라미터 변경 없이 `T`에서 한 번 평가한다. A2도 음수이면 종료한다. 역치, \(k\), \(\lambda\), cycle 정의, 거리척도, endpoint를 결과에 맞춰 바꾸는 추가 수정은 금지한다.

## 6. 대조군과 판정

필수 대조군:

1. frozen geometry \(R=1\);
2. directed-strength shuffle: 공간 그래프와 \(alpha\) multiset을 보존하되 endpoint와 무관한 seed `20260820`으로 source labels를 순열;
3. matched noncycle: 같은 6-NN edge를 첫 공간좌표가 작은 node에서 큰 node로 향하게 만든 DAG에서 \(\Gamma\)와 같은 edge 수를 고르고 \(\alpha\) multiset을 그대로 재배정;
4. common calibration threshold;
5. time shift: `B` source trace를 37 volumes 순환 이동;
6. GFP spatial falsifier.

확인 단위는 recording/animal이다. primary는 다섯 GCaMP recording의 \(\Delta_s^T\) 평균이다. `PASS_PREDICTIVE_FEATURE`는 (i) 5개 중 4개 이상에서 \(Delta_s^T>0\), (ii) 평균 \(Delta^T>0\), (iii) one-sided exact paired sign-flip p<0.05, (iv) directed-strength shuffle보다 평균이 크고, (v) GFP 효과가 GCaMP 평균 이상이 아님을 모두 요구한다. 표본이 작아 이 조건을 못 채우면 효과와 불확실성만 보고하고 승격하지 않는다.

## 7. 금지 주장

- `C`를 관측 회귀로 복구했다.
- 주름 자체가 학습으로 새로 생성됐다.
- 계산된 \(g^{eff}\)가 MRI에서 관측된 물리 metric이다.
- cycle strength와 metric response의 기계적 상관이 독립 증거다.
- bin, neuron, edge, cycle, seed를 독립 표본으로 센다.
