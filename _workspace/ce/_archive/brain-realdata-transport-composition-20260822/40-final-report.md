# 실제 칼슘 영상에서의 전이 연산자 합성 검정

Status: COMPLETE

## 초록

본 검정은 신경 연결이 국소 전이 연산자를 저장한다는 후보를 실제 시행별
칼슘 영상에서 가장 좁은 형태로 시험했다. 외부 사건이 없는 지연 구간에서 세
상태를 같은 간격으로 잡고, 두 한-단계 연산자의 합성이 직접 두-단계 연산자와
같은 예측을 내는지 통째로 제외한 시행에서 비교했다. 합성은 직접 회귀보다
오차가 작았지만 train mean, persistence, 시행-derangement를 모두 이기지 못해
사전 판정에서 기각됐다. DCZ 조건에서만 두 신호 표현이 같은 방향의 양성
진단을 보였으나, 이는 사후 조건 비교이며 보편 전이 법칙이나 시냅스 shortcut
증거가 아니다.

## 1. 문제와 자료

앞선 합성 실험에서 곡률은 기억의 정체성이 아니라 동역학에서 파생되는 관측
서명으로 남았다. 그래서 이번에는 곡률을 전혀 쓰지 않고, 간선이 저장할 수
있는 가장 단순한 대상인 상태 전이 연산자 자체를 검사했다. 독자는 선형대수와
교차검증의 기본 개념을 안다고 가정한다.

[공리: 외부 입력] Maristany de las Casas et al.의 Figure 2 공개 자료를
사용했다. 자료에는 세 동물의 11개 세션, saline/DCZ 조건, 시행마다 180 frame,
같은 세션 안에서 정렬된 칼슘 ROI가 있다. Figure 2와 다른 figure의 시냅스 또는
종단 자료는 같은 unit으로 연결되지 않으며, 공개 배열 순서는 실제 취득 시각의
연대기라고 검증되지 않았다.

## 2. 상태와 합성식

외부 cue는 약 $-1.8\,\mathrm{s}$, Go 사건은 $0\,\mathrm{s}$에 있다. 새 외부
사건을 사이에 끼우지 않기 위해 지연 구간의 $-1.5$, $-0.9$, $-0.3\,\mathrm{s}$
주변에서 폭 $0.2\,\mathrm{s}$인 세 창을 고정했다. 각 시행의 창 평균을 차례로
$x_0$, $x_1$, $x_2$라 정의했다. dF/F는 비율이고, 모든 ROI 표준화와 PCA는
train 시행에서만 적합했으므로 최종 상태와 오차비는 무차원이다.

[정의] 공통 train-only 잠재 좌표에서 세 affine map을 다음처럼 둔다.

$$
x_1\approx x_0A_{01}+b_{01},\qquad
x_2\approx x_1A_{12}+b_{12},\qquad
x_2\approx x_0A_{02}+b_{02}. \tag{1}
$$

[정리] 두 한-단계 affine map의 합성 예측은

$$
\widehat x_2^{\mathrm{comp}}
=(x_0A_{01}+b_{01})A_{12}+b_{12} \tag{2}
$$

이다.

증명. 첫 번째 map의 출력 $x_0A_{01}+b_{01}$을 두 번째 map의 입력에
대입하면 식 (2)가 바로 나온다. 따라서 합성 linear part는
$A_{01}A_{12}$이고 intercept는 $b_{01}A_{12}+b_{12}$다. □

## 3. 사전 판정과 대조군

[예측] 합성이 실제 전이 구조를 보존한다면 held-out 오차가 직접 map과 가까워야
할 뿐 아니라, 단순 평균 회귀와 시간적 자기상관보다 실제 예측력이 커야 한다.
이를 위해

$$
G=\frac{\operatorname{SSE}_{\mathrm{comp}}-
\operatorname{SSE}_{\mathrm{direct}}}
{\max(\operatorname{SSE}_{\mathrm{persistence}},\epsilon)} \tag{3}
$$

를 정의했다. 1차 `dff` 판정은 $G\le0.10$이고, 합성이 persistence,
train-fold mean, 시행-deranged 합성을 모두 이길 때만 한 animal을 일관된 것으로
세었다. 전체와 세 animal 모두 이 조건을 만족해야 보편적인 관측상 합성 후보를
남기도록 고정했다. `branch` 신호는 독립적인 민감도 검사일 뿐 1차 실패를
구제하지 못한다.

직접 $T_{02}$, persistence, train mean, 시행 사이의 successor를 바꾼 합성,
중간 좌표를 뒤집은 합성, pooled stationary map의 제곱, 시간역방향 합성을 같은
fold와 rank에서 계산했다. 이 대조군들은 각각 직접 예측, 매끄러운 칼슘 잔상,
평균 수축, 시행 독립 phase mean, ROI 대응, 시간 불변성, 방향성의 대안을
검사한다.

## 4. 관측 결과

[경험식] 11개 세션의 saline/DCZ 22개 block, 모두 1,532 시행에서 1차 결과는
다음과 같았다.

| 무차원 held-out 지표 | `dff` 전체 | `branch` 전체 |
|---|---:|---:|
| $G$ | -0.251052 | 0.088917 |
| 합성 skill 대 train mean | -0.124837 | -0.061956 |
| 합성 skill 대 persistence | -0.260775 | -0.425849 |
| 합성 advantage 대 deranged | -0.145066 | -0.060260 |
| 합성 advantage 대 중간좌표 permutation | 0.091813 | 0.054635 |

음의 $G$만 보면 합성이 직접 두-단계 회귀보다 좋아 보인다. 그러나 `dff`에서
합성 SSE는 52,782.615였고 train mean은 46,924.673, persistence는
41,865.214, deranged는 46,095.704였다. 즉 두 map을 연속 적용하면서 생긴
수축 또는 regularization이 불안정한 직접 회귀보다 나았을 뿐, 실제 미래
상태를 평균이나 현재 상태보다 잘 예측하지 못했다. 이 차이가 이번 검정에서
가장 중요한 반례다.

동물별로는 `DCO2`만 모든 조건을 만족했다. `DCO1`은 mean, persistence,
deranged를 모두 이기지 못했고, `DCO4`는 persistence는 이겼지만 mean과
deranged를 이기지 못했다. 세션 단위에서는 4/11, condition block 단위에서는
8/22만 일관 조건을 만족했다. 독립 표본은 세 동물뿐이므로 이 비율을
population 유의성으로 읽을 수 없다.

조건별 분해는 다음 후속 후보를 제시했다. 이 비교는 1차 판정을 바꾸지 않는
진단이다.

| 조건·신호 | $G$ | skill 대 mean | skill 대 persistence | advantage 대 deranged |
|---|---:|---:|---:|---:|
| DCZ `dff` | -0.045313 | 0.089385 | 0.238994 | 0.066601 |
| saline `dff` | -0.338952 | -0.186388 | -0.474296 | -0.205346 |
| DCZ `branch` | -0.046077 | 0.099292 | 0.224979 | 0.094521 |
| saline `branch` | 0.134861 | -0.093295 | -0.647356 | -0.090097 |

DCZ aggregate는 두 표현에서 모두 모든 대조를 이겼고 saline aggregate는
그렇지 않았다. 그러나 세션쌍에서 DCZ-minus-saline 개선 방향은 mean 기준
6/11, persistence 기준 9/11, deranged 기준 7/11로 완전하지 않았다. 따라서
NDNF 조작이 합성 가능한 동역학을 만들었다고 결론내릴 수 없다. 억제/gain,
신호대잡음비, 상태 안정화가 모두 같은 패턴을 만들 수 있다.

## 5. 결론과 다음 반례

이번 자료와 고정 시간척도에서 보편적인 선형 $T_{02}\simeq
T_{12}\circ T_{01}$은 기각됐다. 더 정확한 현재 후보는 고정 공간을 저장한
간선이 아니라, 관측된 입력과 상태에 따라 바뀌는 조건부 전이 규칙이다.
다음 검정에서는 이미 cue 전에 주어진 left/right instruction $c$만 사용해

$$
A_{ab}^{(c)}=A_{ab}+(2c-1)\Delta A_{ab} \tag{4}
$$

인 shared-base/cue-residual operator를 사전에 고정해야 한다. cue-blind,
train cue-shuffle, 반대 cue map, 같은 cue 안의 successor derangement, conditional
mean, persistence, direct $T_{02}^{(c)}$를 모두 이겨야 한다. 행동 결과,
정답 여부, decoder, 곡률은 입력으로 넣지 않는다.

[미완성] 식 (4)가 통과해도 알려진 instruction에 조건화된 유효동역학만
남는다. 간선이 실제 shortcut을 저장한다는 결론에는 같은 unit을 학습 전후로
추적한 자료와 중간 상태를 교란하는 개입이 필요하다. 학습 뒤 중간 상태를
방해해도 직접 전이가 살아 있고, 동시에 필요한 rank나 경로 비용이 줄어야
비로소 구조적 shortcut 후보를 논할 수 있다.

## 6. 재현성

구현은 `reality_stone/python/reality_stone/clarus/realdata_transport_composition.py`,
집중 검사는 `tests/test_realdata_transport_composition.py`, machine result는
`artifacts/e17-transport-composition-results.json`이다. 실행 명령과 SHA-256은
`31-validation.md`에 기록했다. 전체 test suite는 실행하지 않았으며, 변경된
계산 경계에 대응하는 집중 검사만 수행했다.

## 참고문헌

Maristany de las Casas et al., *Science* (2026), DOI
`10.1126/science.adx4358`; official G-Node archive DOI
`10.12751/g-node.etlk5k`, accessed 2026-08-22.
