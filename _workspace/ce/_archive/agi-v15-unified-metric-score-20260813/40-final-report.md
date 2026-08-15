# AGI V15 unified metric: 실제 증명·테스트 채점 보고서

Status: COMPLETE

## 1. 최종 판정

동결된 V15에 대한 최종 판정은 다음과 같다.

> **수학 명제 F1--F5는 5/5로 완결됐다. 보통 수치 범위의 미사용 256 seed와 좌표·순열 metamorphism은 통과했다. 그러나 작은 양의 edge에서 최단경로가 종료하지 않는 완전 반례가 있으므로 finite core 종합 판정은 `STOP`이다. Oracle metric 목적정렬 과제는 통과했지만 자율 학습 증거가 아니다. A1--A4는 0/4이며 AGI 판정은 0%, `STOP`이다.**

여기서 0%는 이 run이 사전 고정한 내부 자격 게이트의 최솟값이다. 표준화된 지능 백분율이나 인간 대비 능력 측정값이 아니다.

## 2. 점수표

| 층 | 실제 결과 | 판정 |
|---|---:|---:|
| 수학 증명·완전 반례 F1--F5 | 5/5 | `MATH PASS` |
| ordinary-scale held-out correctness | 256/256 seed | `PASS` |
| affine/permutation coordinate metamorphism | 각 256/256 | `PASS` |
| positive-scale unique-path termination | 1초 내 미종료 | `FAIL` |
| 추가 finite-input 적대검사 | 0/8 | `FAIL` |
| finite core 종합 | scale 반례 존재 | `STOP` |
| oracle metric route choice | 256/256, regret 0 | 제한된 `ORACLE UTILITY GO` |
| identity metric route choice | 168/256, regret 0.190988 | 비교값 |
| 자율 agent A1--A4 | 0/4 | `STOP` |
| AGI 내부 자격 | 0% | `STOP` |

## 3. 무엇이 증명됐는가

[정리] 점과 metric tensor를 함께

$$
y=Jx+b,\qquad g_y=J^{-T}g_xJ^{-1}
$$

로 운반하면 국소 이차 길이, 고정 topology의 edge 비용과 path 비용이 보존된다. 실제 fixture의 최대 상대오차는 $6.868\times10^{-16}$이었다.

[정리: no-go] 현재 좌표계의 eigenvalue clipping은 일반 affine 공변이 아니다. 완전 반례의 defect는 $5/9$였다.

[정리: no-go] 정적 무향 Riemannian 비용은 방향 대칭이다. 따라서 그 비용만으로 비가역 세계동역학을 만들 수 없다.

[정리: no-go] 외생 symmetry-breaking source가 없는 대칭 diamond에서는 equivariant singleton 목표를 선택할 수 없다. 구현은 두 최소점 $(1,2)$를 함께 반환했다.

[정리: no-go] finite endpoint tensor만으로 그 사이의 continuum metric을 식별할 수 없다. 동일 endpoint를 갖는 두 smooth SPD metric의 길이는 $1$과 $3/2$로 달라졌다.

따라서 수학 5/5 중 네 항목은 구조의 한계를 정확히 증명한 점수다. AGI 능력 5개를 획득했다는 뜻이 아니다.

## 4. 실제 held-out 계산

[산출: 수치] seed 915000--915255의 연결 무향 그래프에서 별도 Floyd--Warshall 참조값과 비교했다. 최단비용, 단순 path 유효성, 목표 최소점과 surprise가 모두 일치했고 최대 상대오차는 $3.5955\times10^{-16}$이었다.

[산출: 수치] 같은 256개 instance의 비직교 affine coordinate change와 node permutation에서도 path cost와 goal readout이 모두 일치했다. 최대 edge 상대오차는 $1.8728\times10^{-14}$로 사전 한계 $10^{-10}$보다 작았다. 이 검사는 동일 instance의 좌표 표현을 바꾼 것이므로 semantic OOD로 승격하지 않는다.

## 5. finite core를 중단시킨 반례

[산출: 완전 구현 반례] 점 $(0,0),(10^{-16},0),(2\times10^{-16},0)$을 chain $0-1-2$로 연결하고 source를 2로 두면 유일 path는 $(2,1,0)$이고 비용은 $2\times10^{-16}$이다. 현재 Dijkstra의 tolerance tie branch는 거리 방향을 검사하지 않고 predecessor를 갱신한다. 그 결과

$$
D=(2\times10^{-16},10^{-16},0),\qquad P=(1,0,1)
$$

가 되어 $0\leftrightarrow1$ predecessor cycle이 생긴다. 공개 `shortest_path`는 1초 내 끝나지 않았다. 보통 scale 무작위 시험이 전부 통과해도 이 한 반례로 일반 finite-input 종료 주장은 성립하지 않는다.

추가 적대검사에서는 작은 비용의 goal 동점 오판, source-to-self uniqueness 오판, reference scale 제곱의 underflow·overflow, 큰 유한 좌표의 NaN edge, 큰 유한 metric projection의 nonfinite 출력, 극단 affine 운반의 zero eigenvalue·Inf가 재현됐다. 강한 finite robustness 점수는 0/8이다.

## 6. Oracle utility의 정확한 의미

[산출: 제한된 수치] seed 916000--916255의 두 경로 과제에서 V15는 256/256을 선택했고 identity metric은 168/256을 선택했다. 평균 normalized regret은 각각 0과 0.1909883315였다. 사후 기술 통계로 V15 정확도의 Wilson 95% 구간은 $[98.52\%,100.00\%]$, identity는 $[59.61\%,71.17\%]$이며 discordant $(88,0)$의 단측 exact sign 값은 $2^{-88}=3.23117\times10^{-27}$이다.

이 비교에서 환경 비용은 oracle $g$로 정의되고 V15에도 같은 $g$가 직접 주어졌다. 반면 identity baseline에는 그 정보가 없다. 따라서 결과는 주어진 목적식의 계산과 privileged metric의 유용성을 확인하지만, 관측에서 $g$를 배우는 능력이나 공정한 learned-model 우위를 확인하지 않는다. 해당 $p$ 값도 이 정보비대칭 synthetic 비교 밖으로 일반화되지 않는다.

## 7. AGI 게이트

[미완성] 동결 artifact에는 다음 실행 증거가 없다.

- A1: raw observation에서 $g_t$를 학습하는 과정;
- A2: 지각, 행동, 환경 transition, 새 관측을 잇는 폐루프;
- A3: 지연 보상의 temporal credit assignment;
- A4: learned compute-matched baseline과 compositional task OOD 채점.

`apply_source_metric`은 외부 tensor를 projection하고 보간할 뿐 학습기가 아니다. 정적 path, surprise와 argmin readout은 환경과 상호작용하는 agent loop가 아니다. 따라서 A1--A4는 0/4이고 AGI는 `STOP`이다.

## 8. 다음 구현 순서

먼저 V15.1 numeric repair를 별도 hash와 새 확인 seed로 시험해야 한다.

1. Dijkstra 첫 pass에서는 strict distance relax만 수행한다.
2. 둘째 pass에서 $D(u)<D(v)$이고 $D(u)+w_{uv}\approx D(v)$인 shortest-path DAG만 만든다.
3. 모든 predecessor가 엄격히 작은 거리에서 오도록 강제한다.
4. path 복원에 visited set과 $N-1$ hop guard를 둔다.
5. 입력 scale contract, rescaling과 모든 공개 출력의 finite/SPD 후조건을 고정한다.

그 다음에만 V16 agent 시험을 연다. 필요한 최소 loop는 raw observation에서 common metric source를 학습하고, 그 metric으로 행동을 고른 뒤, 환경의 새 관측과 지연 보상으로 같은 updater를 수정하는 과정이다. 성공 조건은 별도 역할별 persistent state를 몰래 추가하지 않으면서 learned compute-matched baseline을 compositional/OOD 과제에서 이기는 것이다.

## 9. 재현성과 경계

동결 hash, 사전 계약, 독립 증명, 첫 scored JSON, 적대검사와 모든 명령은 이 run에 보존했다. scored runner의 제한도 `20-audit.md`에 기록했다. 관련 unit/회귀는 17, 72, 114개 slice가 각각 통과했고 scored replay는 captured JSON과 동일했다.

저장소 전체 suite는 기존 dirty fixture와 policy 실패가 알려져 있어 재실행하지 않았다. 별도 CE constants 하네스의 overall `CAUTION`은 재현했으나 V15의 수학·AGI 점수와 독립이다. 외부 데이터와 외부 문헌은 이번 채점에 사용하지 않았다.

