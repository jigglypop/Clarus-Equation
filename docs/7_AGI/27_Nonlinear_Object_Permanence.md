# 비선형 물체 영속성 G2--G3 사전등록

> 상태: `PRE-IMPLEMENTATION / LOCKED V1`
>
> 상위 로드맵: `8_Roadmap.md`
>
> 선형 기준: `26_Causal_World_Simulator.md`
>
> 기계 판독 설정: `../../experiments/preregistration/nonlinear_object_permanence_v1.json`

## 0. 검증 주장

이번 실험이 검증하는 주장은 하나다.

> 부분적으로 가려진 비선형 다중물체 세계에서, 국소 chart 모델은 보이지 않는 동안의 객체 상태와 행동별 미래를 유지하며 강한 비국소 기준선보다 긴 홀드아웃 rollout을 더 정확히 예측할 수 있다.

통과해도 실제 뇌의 물체 영속성, 의식, AGI를 증명하지 않는다. `synthetic nonlinear engineering evidence`만 추가한다.

## 1. 환경 계약

### 1.1 잠재상태

물체 (i\in\{1,\dots,N\})의 상태는

\[
s_t^i=(x_t^i,y_t^i,v_{x,t}^i,v_{y,t}^i,r_i,m_i,q_i)
\]

로 둔다. (q_i)는 궤적 내에서 보존되는 정체성이다. 전체 상태에는 관찰자 위치와 행동 복사본을 추가한다.

### 1.2 동역학

- semi-implicit Euler
- 벽 반사와 탄성 물체 충돌
- 질량ㆍ반지름ㆍ마찰의 물체별 차이
- 약한 위치 의존 비선형력
- 제한된 관찰자/센서 행동

oracle은 가림과 무관한 완전 상태를 기록하지만 모델 입력에는 노출하지 않는다.

### 1.3 관측과 가림

관측은 visible object feature, visibility mask, ego state, 직전 action으로 구성한다. 가림판 뒤 객체의 위치ㆍ속도ㆍ정체성은 누락한다. 재등장 시에는 입력 배열 위치를 무작위화해 slot 순서를 정체성 단서로 사용할 수 없게 한다.

## 2. 고정 데이터 분할

| split | seed | 물체 수 | 가림 길이 | 물리 파라미터 |
|---|---:|---:|---:|---|
| train | 1000--1019 | 2--3 | 4--12 | 기본 범위 |
| validation | 2000--2004 | 2--3 | 4--12 | 기본 범위 |
| ID test | 3000--3004 | 2--3 | 4--12 | 기본 범위 |
| long-occlusion OOD | 4000--4004 | 2--3 | 16--32 | 기본 범위 |
| composition OOD | 5000--5004 | 4 | 8--24 | 질량ㆍ마찰 조합 이동 |

각 seed의 생성은 결정적이어야 한다. validation은 하이퍼파라미터 선택에만 사용하고 test는 최종 gate에서 한 번 읽는다.

## 3. 비교 모델과 공정성

1. `persistence`: 마지막 관측 상태 유지
2. `global_linear`: 전체 상태의 선형 controlled transition
3. `monolithic_nonlinear`: 작은 단일 비선형 상태모델
4. `local_chart_nonlinear`: 객체/상호작용 chart와 전이 glue
5. `oracle_state`: 완전 상태 입력의 상한선, 경쟁 기준선은 아님

학습 trajectory, optimizer step, 상태 차원, 허용 parameter budget을 보고한다. local-chart 모델만 identity label 또는 oracle visibility를 받으면 실험은 무효다.

## 4. 측정량

- horizon (h\in\{1,5,20,100\})의 위치ㆍ속도 RMSE
- 가림 중 hidden-state RMSE
- 재등장 위치 오차
- identity switch rate
- 충돌 발생 시각 오차
- 행동 개입에 대한 평균 효과 오차
- calibration 또는 ensemble spread 대비 실제 오차
- seed별 결과와 bootstrap 95% CI

## 5. G2 통과 기준

G2는 다음을 모두 만족할 때만 통과한다.

1. 모든 필수 모델이 같은 train/validation/test 분할을 사용한다.
2. local-chart의 ID-test 20-step RMSE가 persistence와 global-linear보다 각각 20% 이상 낮다.
3. local-chart의 ID-test 100-step RMSE가 monolithic nonlinear보다 낮고, 5개 test seed 중 4개 이상에서 같은 방향이다.
4. 행동 개입 효과의 부호 정확도가 90% 이상이다.
5. NaN, 발산, 데이터 누수 검사가 모두 0건이다.

## 6. G3 통과 기준

G3는 G2 통과 후 다음을 모두 만족할 때만 통과한다.

1. long-occlusion OOD의 가림 중 RMSE가 persistence보다 25% 이상 낮다.
2. 재등장 위치 오차가 monolithic nonlinear보다 10% 이상 낮다.
3. identity switch rate가 5% 이하이다.
4. 가림 길이가 증가할 때 uncertainty가 평균적으로 증가한다.
5. composition OOD 결과를 별도로 보고하며, 실패해도 숨기지 않는다. 이 항목은 G5 판정 전에는 진단 지표다.

## 7. 자동 실패 조건

- test를 보고 임계값ㆍ특징ㆍseed를 수정
- oracle state 또는 미래 visibility 누수
- 배열 slot을 고정해 identity를 암기
- teacher-forced one-step만 평가하고 free rollout을 누락
- 성공 seed만 평균에 포함
- 단일 모델의 계산량이 비교군보다 2배 이상인데 비용 보정을 누락

실패하면 G2/G3를 `FAIL`로 기록하고, 변경된 가설은 V2 사전등록 파일을 새로 만든다. V1 결과를 덮어쓰지 않는다.

## 8. 구현 경계

첫 구현은 NumPy만 사용한다. 학습 모델이 추가 의존성을 요구하면 별도 승인 없이 패키지를 설치하지 않는다. 렌더링은 판정 경로에서 제외하고 디버그 선택사항으로 둔다.

모듈 API 목표:

```python
world = NonlinearObjectWorld(config, seed=seed)
observation, oracle = world.reset()
observation, oracle, info = world.step(action)

model.fit(train_episodes, validation_episodes)
rollout = model.rollout(initial_observation, actions, horizon=100)
report = evaluate_object_permanence(model, test_episodes)
```

## 9. 완료 정의

- 환경 불변량ㆍ충돌ㆍ가림ㆍseed 결정성 단위 테스트
- 모델별 독립 테스트와 공정성 검사
- 설정 JSON hash가 보고서에 포함
- 5개 seed의 G2/G3 결과와 CI 저장
- 기존 G1 및 리만 톱니 회귀 테스트 유지
- 문서의 수치와 생성된 JSON의 수치 자동 대조
