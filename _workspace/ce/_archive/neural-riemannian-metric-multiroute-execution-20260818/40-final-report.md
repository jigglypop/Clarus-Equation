# 신경 리만 계량 다중 경로 검증 보고서

Status: COMPLETE

## 초록

이 연구는 연결 변화가 기능적 기하를 바꾸고 그 기하가 미래 신경 궤적을 제약한다는 통합 가설을 13개 서로 다른 경로로 시험했다. 물리적 피질 주름 $h$, 이미 형성된 기준 기능기하 $g_0$, 상태 또는 학습 뒤의 $g_t$를 분리하고, 직접 동역학·이득·잡음·평탄 좌표변환을 필수 대조군으로 고정했다. 합성 one-shot에서는 정답 생성기 예측 대조 5/5가 통과했지만 계수 회복 15/20, 평탄 pullback 회복 15/20, 비표적 생성기 거짓선택 72/100으로 전체 추정기 게이트가 실패했다. 실제 GRID 자료는 여섯 모듈을 원자료에서 완전 재계산했지만 엄격한 위상-이동성 패턴은 12개 REM/SWS 비교 중 R3-REM 하나뿐이었고, 이동성 거리 제곱의 중앙값 97.6%는 방향별 모양보다 공통 스케일이었다. E17·수면·구조 자료는 유용한 기술통계와 방향을 주었지만 독립적인 $\Delta W^s\to\Delta g\to\Delta x$ 사슬을 제공하지 않았다. 따라서 현재 형식 지위는 “수학적으로 가능한 통합 가설, 현 추정기 기각, 생물학적 핵심 사슬 미검증”이다.

## 세 종류의 주름

[정의] 물리적 피질 주름은 표면 매장 $r(s)$가 만드는 해부학 계량이다.

$$
h_{ab}(s)=\partial_a r(s)\cdot\partial_b r(s).
\tag{1}
$$

이 계량에는 이랑·고랑, 표면 지오데식, 피질 깊이와 영역 경계가 들어간다. 이것은 신경 활동에서 추정한 상태공간 계량과 자동으로 같지 않다.

[정의] $g_0(z,c)$는 발달과 과거 학습이 이미 만든 기준 기능기하이고, $g_t(z,c)$는 새로운 학습·문맥·약물·수면 상태에서의 후보 기능기하다. 같은 좌표와 단위를 정당화한 경우 상대 변형은

$$
D_{\rm rel}(g_0,g_t)
=\left\|\log\!\left(g_0^{-1/2}g_tg_0^{-1/2}\right)\right\|_F
\tag{2}
$$

로 쓸 수 있다. 고정된 해부학 $h$와 기능 계량 $g$가 서로 다른 공간에 살면 $g-h$는 정의되지 않는다. 표면에서 상태공간으로 가는 사상 $F$를 먼저 정해 $F^*g$를 $h$와 비교하거나, 표면 지오데식·깊이·영역·배선 길이를 별도 공변량으로 넣어야 한다.

[산출] 이번 실행은 세 종류를 개념적으로 분리했지만, 실제 분석에는 물리적 $h$와 같은 단위의 종단 $g_0\to g_t$가 없었다. GRID의 wake는 상태 기준일 뿐 발달 기준기하가 아니며, E17의 세션별 상수 SPD 행렬도 물리적 주름이나 국소 곡률장을 측정하지 않는다. 따라서 “기존 주름을 고려했는가”에 대한 정확한 답은 **이론과 대조 설계에는 넣었지만, 현재 실행 자료에서는 직접 통제하지 못했다**이다.

## 핵심 가설과 수학적 경계

[미완성] 시험 대상인 가장 강한 사슬은

$$
\Delta W^s
\xrightarrow{\ \Phi\ }
\Delta g(z,c)
\longrightarrow
\Delta p(x_{0:T},\tau_B\mid x_0,c)
\tag{3}
$$

이다. $W^s$는 구조 연결, $\Phi$는 사전에 고정할 연결-계량 생산자, $p$는 미래 경로와 first-passage 분포다. 현재 공개 자료에는 같은 단위에서 이 세 화살표를 동시에 관측한 경로가 없었다.

[정리] $W^s$만으로 $g$는 유일하게 정해지지 않는다. 같은 구조 연결에 서로 다른 국소 동역학, 과정잡음, 관측사상 또는 좌표 gauge를 결합하면 서로 다른 SPD 계량 후보가 생긴다. 따라서 $\Phi$는 발견되는 항등식이 아니라 이름과 자유도를 고정해 경쟁시켜야 하는 모형이다.

[정리] 비상수 $g(z)$ 또는 0이 아닌 Christoffel 기호는 곡률을 뜻하지 않는다. 유클리드 공간의 비선형 좌표 pullback은 위치에 따라 변하는 계량과 Christoffel 기호를 만들지만 Riemann 곡률은 0이다. 곡률 주장은 $C^2$ SPD 장, tensor 계산, 좌표 재표본화와 평탄 pullback 대조가 모두 필요하다.

[정리] 짧은 지오데식은 빠른 도달시간을 뜻하지 않는다. 경로시간은

$$
dz=v(z,c)\,dt+B(z,c)\,dB_t,
\qquad Q=BB^\top
\tag{4}
$$

의 drift, 잡음, 경계와 초기분포에 달려 있다. $v=-g^{-1}\nabla V$ 같은 결합법칙과 $V$를 별도 공리로 선언하지 않으면 $g\to v\to\gamma$는 유도되지 않는다.

[정리] 시간가변 과정잡음의 $H$-step 도달 공분산은 혁신이 공통 끝점으로 전파되는 순서로

$$
C_{Q,H}(t)=\sum_{k=0}^{H-1}
\Psi_{t,H,k}Q_{t+k}\Psi_{t,H,k}^{\top},
\quad
\Psi_{t,H,k}=J_{t+H-1}\cdots J_{t+k+1}
\tag{5}
$$

이다. 이 역공분산 후보는 활동 동역학에서 얻은 기술적 계량이며, 구조 $W^s$에서 유도한 생물학적 계량과 동일하지 않다.

## 실행 결과

### 합성 추정기

[산출] 120개의 독립 train/test 회로, 여섯 생성기, e1/e2 학습과 e3 검사를 고정했다. 결과·원시경로·구조행렬·seed·first-passage를 잠근 뒤 독립 감사가 모든 점수와 Holm 결정을 다시 계산했다.

| 사전 게이트 | 결과 | 판정 |
|---|---:|---|
| G1 계량계수 회복 | 15/20, 기준 18/20 | 실패 |
| G1 대 다섯 대조군 예측 | Holm 5/5 | 부분 통과 |
| G2--G6 any-Holm 거짓선택 | 72/100, 기준 최대 5 | 실패 |
| G3 pullback 계수 회복 | 15/20, 기준 18/20 | 실패 |
| 수치 곡률 fixture | 0/20 거짓양성 | 수치 검산 통과 |

G6 null에서도 16/20 회로가 하나 이상의 잘못된 계량 우위를 냈다. 이는 현재 한 회로 적합과 trajectory 중첩 검정이 이론을 가려낼 만큼 보정되지 않았음을 뜻한다. G1에서 올바른 방향의 예측 이득이 나온 사실은 보존하지만 전체 추정기 검증을 대체하지 않는다.

### 실제 GRID wake/REM/SWS

[산출] 공식 Gardner 자료의 Q1/Q2/R1/R2/R3/S1을 10초 블록으로 나누고 wake 전용 C 블록에서만 여섯 차원 chart를 학습했다. 위상 A·이동성 B와 그 반대 배치를 모두 계산했고, 검증기가 원시 NPZ에서 전 결과를 다시 만들었다.

엄격한 호환 판정은 REM 1/6, SWS 0/6, 전체 1/12였다. 유일한 R3-REM은 primary 위상/계량 비가 0.9852/27.5194, swap이 0.9966/26.6398이었다. 그러나 primary 위상만 기준 이하인 경우는 2/12, swap만 보면 8/12여서 분할 민감성이 컸다.

[산출] generalized-eigenvalue log를 공통 scale과 determinant-normalized shape로 정확히 분해했을 때 24개 상태-role 행렬의 shape 제곱 비중 중앙값은 0.02436이었다. 즉 큰 raw AIRM의 중앙값 약 97.6%가 공통 분산·잡음 scale이었다. 이 결과는 상태 변화는 강하다는 뜻이지만 방향별 공간 변형이나 곡률의 증거는 아니다.

### E17·수면·구조 경로

[산출] E17 Figure 3의 공식 거리 변환 뒤 Spearman 결과는 Rule-A noise $\rho=-0.347$ ($n=124$), Rule-B noise $-0.264$ ($n=38$), Rule-A signal $-0.293$ ($n=124$), Rule-B signal $-0.106$ ($n=38$)였다. 행은 동물 독립표본이 아닌 중첩 synapse pair이므로 p값은 기술통계다. Figure 4/5에는 종단 요약이 있지만 동물 ID와 독립적인 이전 계량-미래 표적 연결이 없어 승격하지 않았다.

[산출] E19의 고정 선행 재현은 34명에서 REM/SWS 비와 source-selected cluster의 item $\rho=-0.553$ 및 category $\rho=0.470$ 연관을 보였다. E15는 104개 처리 행과 13개 세션에서 수면박탈 뒤 replay 감소 방향을 보였지만, 세션-동물 독립성과 별도 미래 표적이 확립되지 않았다. 둘 다 수면 연관이지 $\Delta g$ 또는 수면 재정규화의 인과 검증이 아니다.

[산출] C. elegans 7,379개 edge는 구조 추정기 fixture로만 남겼다. MICrONS는 같은 root의 기능값과 구조 edge join이 없고, BCI는 원시 38.8 GB payload가 확보되지 않아 파생표 기술통계만 가능했다. 잘못 지정됐던 DANDI 000037은 종단 M1이 아니라 시각피질 Openscope 자료였고, all-optical 자료는 공개 원시 trial payload가 없어 실행하지 않았다.

## 주장 지위

| 주장 | 현재 지위 | 이유 |
|---|---|---|
| 연결 변화가 기능기하를 바꿀 수 있다 | [미완성] 수학적으로 가능한 모형군 | $W^s\mapsto g$가 비유일하고 실제 동일 단위 사슬이 없음 |
| 현 합성 추정기가 계량을 신뢰성 있게 찾는다 | 기각 | 회복과 거짓선택 게이트 실패 |
| 수면에서 위상은 보존되고 기능계량만 바뀐다 | [경험식] 제한적 기술 신호 | 1/12만 양분 반복, raw 거리는 scale 지배 |
| NREM/REM이 기하를 재정규화한다 | [예측] 미검증 | E15/E19는 연관·집계 자료뿐 |
| 학습된 지오데식이 행동·도달시간을 예측한다 | [예측] 미검증 | 명시적 SDE 결합과 독립 표적 부재 |
| 기존 물리적·발달적 주름을 넘는 효과가 있다 | [예측] 미검증 | 현 실행에서 $h$ 또는 등록된 $g_0\to g_t$를 측정하지 않음 |

이론의 감각적 핵심인 “고정 골격 위에서 유효거리와 접근성이 바뀐다”는 문장은 여전히 연구 프로그램으로 가치가 있다. 그러나 현재 결과는 그 문장을 사실로 승격하지 않는다. 특히 “기능 상태가 크게 달라진다”와 “리만 계량의 방향별 모양이 바뀐다”를 구분해야 한다.

## 다음 결정적 실험

[예측] 가장 작은 결정적 설계는 같은 개체·같은 세포에서 다음 다섯 묶음을 얻는다.

1. 구조 표면과 기존 주름 $h$, 깊이·층·영역·배선 길이를 baseline으로 고정한다.
2. 같은 synapse/cell의 전후 구조 연결 $W^s_0,W^s_t$를 측정한다.
3. 별도 calibration trial에서 $J,Q$와 사전 고정한 $\Phi(W^s)$를 적합한다.
4. 사용하지 않은 trial에서 미래 경로와 first-passage 분포를 점수화한다.
5. plasticity 또는 연결을 무작위 교란하고 sham, 직접 $v,Q$, gain/noise, Euclidean, persistence, flat-pullback과 자유도 맞춘 SPD를 함께 비교한다.

주 비교는 해부학 baseline $M_h$와 기능기하를 추가한 $M_{h+g}$의 subject-held-out proper-score 차이다. $h$를 통제한 뒤에도 $\Delta W^s$가 $\Delta g$를 예측하고, 그 $\Delta g$가 직접 동역학보다 미래 경로를 더 잘 예측하며, 개입에 따라 함께 움직여야 강한 사슬이 살아남는다.

해부학을 먼저 넣을 수 있는 공개 후속 자료는 세 가지다. [HCP-YA 2025](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release)는 1,071명과 45명 retest로 $h$를 넘는 반복 활동 예측을 시험할 수 있다. [ABCD MRI](https://abcdstudy.org/scientists/protocols-mri/)는 T1/T2·HARDI·rest/task fMRI 종단 자료로 발달 baseline을 통제할 수 있다. [OpenNeuro ds006072 v1.1.0](https://openneuro.org/datasets/ds006072/versions/1.1.0)은 CC0 반복 구조·확산·rest/task fMRI와 약물 전후 세션을 제공해 작은 $N$의 고해상도 $H_G$ 탐색에 적합하다. 세 자료 모두 구조 synapse 변화는 없으므로 강한 $H_W$가 아니라 anatomy-aware $H_G$까지만 시험한다.

## 재개 조건

합성 v4는 이번 결과를 덮어쓰지 않는 새 사전등록이어야 한다. 여러 calibration 회로가 $\phi=0.28$ 양쪽을 덮도록 하고, 회로를 추론 단위로 쓰며, direct v/Q를 1차 대조로 고정한 뒤 독립 100-dataset null에서 familywise calibration을 먼저 통과해야 한다. GRID 후속은 determinant-normalized SPD shape, rate/variance nuisance, occupancy-matched bootstrap과 held-out transition proper score를 결과 전에 고정해야 한다. 이 두 조건이 충족되어도 생물학적 승격은 동일 단위의 $W^s$, $h$, $g_0\to g_t$와 미래 경로가 확보될 때까지 보류한다.

## 재현 기록

- 실행·검증 명령: `30-implementation.md`, `31-validation.md`
- one-shot 결과 해시: `artifacts/one-shot-result-lock.json`
- 합성 감사: `artifacts/synthetic-v3-postexecution-audit.md`
- GRID 감사: `artifacts/grid-v3-postexecution-audit.md`
- 13개 경로 원장: `artifacts/route_dispositions.json`
- 공식 출처와 접근 경계: `10-sources.md`

최종 판정은 단순한 “맞음/틀림”이 아니다. 현 데이터는 핵심식을 확인하지 못했고 현 추정기는 기각됐지만, 어떤 관측과 대조가 있어야 이론을 실제 계산이론으로 승격할 수 있는지는 이전보다 훨씬 좁고 명확해졌다.
