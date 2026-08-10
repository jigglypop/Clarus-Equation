# G9-F: 기능 경계와 해부 경계의 기하학적 분리

## 질문

고랑과 이랑의 리만 기하가 대규모 기능 영역의 경계를 직접 가르는가? 기존
sulcal-depth 회귀와 다른 목표를 쓰기 위해 TemplateFlow fsaverage 10k의
Yeo2011 7-network 기능 atlas와 Desikan2006 해부 atlas를 사용했다.

삼각 mesh edge의 양 끝 레이블이 다르면 boundary, 같으면 non-boundary다. label 0
medial wall에 닿는 edge는 제외했다. edge midpoint 방위각 여덟 sector 중 전체
sector를 하나씩 holdout하며, 약 8%인 boundary class에는 balanced weighted ridge를
사용했다. AUC가 주 평가량이다.

## V1: 국소 LB로 Yeo 기능경계 예측 — 실패

기준 특징은 edge 중점 위치·반경, edge 길이, 양 끝 scalar curvature의 평균,
절댓값 평균과 차이다. 후보에는 등방성 및 경계보존 LB 장의 양 끝 평균과 차이를
추가했다.

- 좌반구 기준 AUC: 0.5168
- 좌반구 후보 AUC: 0.5179
- 이득: 0.0011
- 개선 sector: 2/4
- permutation AUC: 0.5011
- 상태: `FAIL`

국소 리만 기하만으로 Yeo 경계를 예측하지 못했다.

## V2: 전역 heat-kernel landmark 좌표 — 실패

기능망이 비국소적이라는 우회 가설에 따라, label을 사용하지 않고 pial 좌표에서
24개 farthest-point landmark를 선택했다. 각 impulse를

\[
\Phi_{k,t}=e^{t\Delta_M}\delta_{p_k}
\]

로 16, 64, 256 step 확산해 global intrinsic positional encoding을 만들었다.
전체 Laplace 고유분해는 사용하지 않았다.

- 이전 국소 기준 AUC: 0.5179
- heat-landmark 후보 AUC: 0.5289
- 이득: 0.0110 — 등록 기준 0.02 미달
- 개선 sector: 2/4
- permutation AUC: 0.5008
- 상태: `FAIL`

전역 리만 위치를 추가해도 Yeo 경계의 공간 holdout 일반화는 약했다.

## Desikan 해부경계 양성대조

같은 edge 분류기가 fold landmark를 사용해 정의된 Desikan 경계에는 반응하는지
확인했다.

좌반구:

- 점별 위치·곡률 기준 AUC: 0.6582
- LB 추가 AUC: 0.6519
- LB 추가 이득: -0.0063
- 전체 LB gate: `FAIL`

LB의 증분효과는 실패했지만 단순 기하 기준선 자체는 민감했다. 이 좁은 결과를
아직 열지 않은 우반구에서 다시 잠갔다.

우반구:

- Desikan 점별 기하 AUC: 0.6853
- 각 sector AUC 범위: 0.6078–0.7455
- permutation AUC: 0.4975
- 상태: `PASS`

## 우반구 기능–해부 분리

우반구 Yeo label을 열기 전에 점별 기하 Yeo AUC 상한 0.60과 이미 잠긴 Desikan
AUC와의 차이 0.08을 등록했다.

- 우반구 Yeo AUC: 0.5221
- 우반구 Desikan AUC: 0.6853
- Desikan − Yeo: 0.1633
- Yeo permutation AUC: 0.5024
- 상태: `PASS`

좌반구에서도 Desikan 0.6582 대 Yeo 0.5168로 같은 방향이다. 따라서 이
group-average atlas와 선형 공간 holdout 조건에서는 표면 위치·곡률이 해부학적
구획 경계를 예측하지만 Yeo 대규모 기능망 경계는 거의 예측하지 못한다.

## 해석 제한

Desikan은 folding landmark를 사용하는 해부 atlas이므로 그 성공은 부분적으로
정의상 예상되는 양성대조다. Yeo 7-network는 매우 거친 기능 구획이며, 낮은 AUC가
모든 기능–기하 관계의 부재를 증명하지 않는다. 특히 일차 시각·청각 피질의 국소
경계, 개인별 fMRI, 발달 시계열, cytoarchitecture와 장거리 연결성은 시험하지
않았다.

현재 결과는 단순 명제인 `고랑이 대규모 기능망을 직접 분할한다`에는 반대된다.
다음 모델에는 표면 기하만 더 넣기보다 장거리 연결 그래프, 발생 gradient 또는
기능 신호를 별도 장으로 추가해야 한다.
