# 학습된 계산 기하와 수면 재정렬: 수학 독립 검산

Status: COMPLETE

## 결론

`LGS-T1`과 거리 변화 부분의 `LGS-T2`는 양의 비용 유향 그래프에서 성립한다. 이 결과는 새 전이가 최단경로 비용을 바꾼다는 정리일 뿐, 그래프 추가가 공간의 계량 변형과 구조적으로 같거나, raw 연결 강도에서 유일한 기하가 나오거나, 인지시간이 단일 길이/속도 비라는 결론은 주지 않는다. `LGS-N1`, `LGS-N2`, `LGS-N3`은 각각 완전 반례가 있어 강한 부모 주장으로는 삭제해야 한다.

## 정의와 범위

$G=(V,E,w)$는 유한 유향 multigraph이고 모든 arc 비용은 양수다. 같은 $u\to v$ arc가 이미 있어도 비용 $a$의 새 labelled 병렬 arc를 더할 수 있다. $d(i,j)\in[0,\infty]$는 확장 유향 최단거리이며, $\infty$가 포함된 합은 $\infty$다. 이 범위에서는 음의 cycle 가정이 불필요하다. 음수 arc를 허용하는 일반화에서는 reachable negative cycle이 있으면 최단거리가 $-\infty$가 되어 아래 식의 대상 자체가 무너진다.

## `LGS-T1`: one-edge APSP 정리

**정리 T1.** 비용 $a>0$의 새 arc $e=(u,v)$를 추가한 그래프를 $G'$라 하면 모든 $i,j$에 대해

$$
d_{G'}(i,j)=\min\{d_G(i,j),\ d_G(i,u)+a+d_G(v,j)\}.
\tag{1}
$$

특히 $d_{G'}(i,j)\le d_G(i,j)$다.

**증명.** 우변의 첫 항은 $G$ 안의 경로이고 둘째 항은 $G$의 $i\leadsto u$, 새 arc, $G$의 $v\leadsto j$를 이어 붙인 $G'$ 경로이므로 $d_{G'}$는 우변 이하이다. 반대로 $G'$의 임의의 최단 walk가 $e$를 쓰지 않으면 비용은 $d_G(i,j)$ 이상이다. $e$를 두 번 이상 쓴다면 첫 $e$ 뒤와 다음 $e$ 전 사이에는 $v\leadsto u$의 양의 비용 closed subwalk가 있다. 이를 지우면 비용이 엄격히 감소하므로 최단 walk일 수 없다. 따라서 $e$를 정확히 한 번 쓰는 최단 walk는 $G$ 안의 $i\leadsto u$와 $v\leadsto j$ 부분을 가지며 비용이 둘째 항 이상이다. 두 하한을 합치면 (1)이다.

기존에 더 싼 $u\to v$ arc가 있어도 식은 유효하다. 그 경우 $d_G(i,v)\le d_G(i,u)+b$ ($b<a$)이므로 둘째 항은 기존 경로보다 개선하지 못한다. 반면 “간선 추가”가 기존 arc의 비용을 올리는 덮어쓰기를 뜻하면 단조성도 (1)도 적용되지 않으며 다른 연산이다.

## `LGS-T2`: 영향 집합과 최적경로 경계

$$
S_{uv}(a)=\{(i,j):d(i,u)+a+d(v,j)<d(i,j)\}.
\tag{2}
$$

식 (1)에서 즉시

$$
d_{G'}(i,j)<d_G(i,j)\ \Longleftrightarrow\ (i,j)\in S_{uv}(a).
\tag{3}
$$

따라서 모든 거리 쌍이 변하는 필요충분조건은 $S_{uv}(a)$가 도달가능 ordered pair 전체와 같다는 것이고, 전혀 변하지 않는 필요충분조건은 $S_{uv}(a)=\varnothing$이다. 도달 불가능한 $(i,j)$도 $i\leadsto u\to v\leadsto j$가 새로 연결하면 (2)에 들어간다.

거리와 최적 경로는 구별해야 한다. 새 labelled arc를 포함한 optimal path-set이 늘어나는 조건은

$$
d(i,u)+a+d(v,j)\le d(i,j).
\tag{4}
$$

이고, 엄격부등식은 거리도 바뀌게 하는 더 강한 조건이다. 등식인 경우 새 최적 route가 생겨도 거리값은 안 바뀐다. 양의 비용 예로 기존 $i\to j$ 비용 2, $i\to u=1/2$, $u\to v=1$, $v\to j=1/2$는 tie를 만든다.

6-vertex fixture $P\to A\to X\to Y\to B\to Q$ (모든 비용 1)에 $A\to B$ 비용 1을 더하면 $d(A,B):3\to1$, $d(P,Q):5\to3$이지만 $d(X,Y)=1$은 그대로다. 따라서 많은 쌍이 바뀔 수는 있지만 반드시 모든 쌍이 바뀌지는 않는다.

## 동치의 네 수준

| 수준 | 보존 대상 | 역함의가 깨지는 정확한 예 |
|---|---|---|
| cost 동치 | 고정 상태집합의 모든 $d(i,j)$ | 비용 1의 parallel $u\to v$ arc를 하나 더하면 모든 거리값은 같아도 가능한 optimal action/path-set은 달라진다. |
| policy 동치 | 등록 상태·문맥에서 선택 policy | 정책이 절대 사용하지 않는 고비용 arc를 더하면 policy는 같지만 topology는 다르다. |
| trajectory-law 동치 | 같은 intervention 아래 $P(x_{0:T}\mid c)$ | 같은 결정 policy라도 transition noise kernel을 바꾸면 거리·policy 표기만으로 law가 정해지지 않는다. |
| topology 동치 | vertex/arc incidence 및 multiplicity | 같은 topology에서 arc 비용 1을 2로 바꾸면 topology는 같아도 $d$와 policy가 달라진다. |

그러므로 `LGS-T3`은 고정 $V$, cost observable, action set, tie rule, policy class와 intervention protocol 아래 pairwise cost와 induced policy가 같은 두 모델이라는 **정의**로는 일관된다. 그것은 trajectory law나 topology 동치를 함의하지 않는다. trajectory-law 동치는 transition kernel과 초기분포까지 같은 동치류에 넣어야 한다.

## P0 반례: 그래프 추가는 일반 계량 변형과 구조적으로 동일하지 않다

`LGS-N1`은 다음 독립 반례 중 하나만으로도 무너진다.

1. **Topology:** $A\to X\to B$에 새 $A\to B$를 더하면 arc incidence가 바뀐다. 같은 fixed graph 위의 weight/metric field 변경은 이 combinatorial topology를 보존한다.
2. **Direction:** 두 상태에서 $A\to B$ 비용 1이고 $B\not\leadsto A$이면 $d(A,B)=1$, $d(B,A)=\infty$. Riemannian distance는 대칭이므로 이를 표현할 수 없다.
3. **Multiplicity:** 동일 비용 parallel $u\to v$ 두 개는 pairwise APSP를 바꾸지 않지만 action identity, fault tolerance, stochastic route choice를 바꾼다. distance-only metric에는 이 정보가 없다.
4. **Dynamics:** 동일 shortest-path matrix 및 greedy policy에 transition noise 또는 edge execution delay가 다른 두 시스템은 trajectory law가 다르다.

살아남는 문장은 더 좁다. 고정된 cost observable에 투영하면 특정 edge addition의 APSP 효과를 같은 pairwise cost 행렬을 산출하는 다른 representation으로 operationally emulate할 수 있다. 이는 구조적 동일성이 아니다.

## P0 반례: $W\mapsto g$의 비식별성과 type 경계

`LGS-N2`는 $\Phi_c$ 없이는 함수조차 정의되지 않는다. $W=1$ 하나와

$$
w_c=\Phi_c(W,A,c)=\frac{1}{WA_c}
$$

를 두면 $A_{c_1}=1$에서는 $w_{c_1}=1$, $A_{c_2}=2$에서는 $w_{c_2}=1/2$다. 같은 $W$가 서로 다른 effective distance를 만들며, $A$를 $W$ 변화와 역비례시켜 $WA$를 고정하면 $\Delta W\ne0$인데 $\Delta g=0$도 가능하다.

추가로 latent coordinate $S\in GL(r)$에서 $W'=SW$, $g'=S^{-T}gS^{-1}$이면

$$
(W')^Tg'W'=W^TgW.
$$

관측이 이 quadratic form만 보이면 $W$와 $g$의 개별 변화를 분리할 수 없다. 유향 shortest-path cost는 extended quasi-metric이고 SPD/Riemannian metric은 대칭 양의 정부호 tensor라는 별도 객체다. 전자를 후자로 부르려면 대칭화 또는 별도 drift/control 구조를 명시해야 하며, 그 과정은 방향 정보를 버리거나 추가한다.

## P0 반례와 좁은 시간 정리

`LGS-N3`의 보편식 $T_{\rm cognition}=L_{\rm effective}/v_{\rm neural}$은 거짓이다. 길이 1, $v=1$인 한 edge라도 synaptic delay 9와 decision/integration overhead 3이 있으면 관측시간은 13이지 1이 아니다. 두 독립 branch가 병렬로 끝난 뒤 join하는 과제는 두 branch 길이의 합이 아니라 critical-path maximum과 join delay에 의해 끝난다.

등록 가능한 좁은 모델은

$$
T=t_0+\max_{p\in\mathcal P_{\rm required}}\left(\sum_{e\in p}\tau_e\right)+t_{\rm integrate},
\tag{5}
$$

이며 $\tau_e=\ell_e/v_e+s_e+q_e$는 conduction, synaptic/queue delay를 포함한다. 단일 직렬 경로, 균일 $v_e=v$, $s_e=q_e=0$, $t_0=t_{\rm integrate}=0$일 때만 (5)가 $L/v$로 축소된다. `LGS-H2`는 이 제한된 모형의 held-out 독립 예측 가설로 남는다.

## 주장별 판정

| ID | 판정 | 등급 | 근거와 잔존 범위 |
|---|---|---|---|
| LGS-T1 | [정리] | 없음 | 식 (1), 양의 유향 multigraph와 병렬/기존 싼 arc까지 전수 검산. |
| LGS-T2 | [정리] | 없음 | 거리 변화는 정확히 (2); optimal-path 변화는 (4)로 별도. |
| LGS-N1 | 삭제 | P0 | topology, direction, multiplicity, dynamics의 완전 반례. cost-observable 투영의 emulate만 잔존. |
| LGS-T3 | [정의] | 없음 | cost/policy 동치의 scope는 명시 가능하나 trajectory/topology와 다름. |
| LGS-N2 | 삭제 | P0 | activity/context 및 $GL(r)$ gauge 반례. preregistered $\Phi_c$ 아래의 추정 가설만 가능. |
| LGS-H1, H3--H6 | [미완성] | P1 | 수학적 strict 우위는 나오지 않는다; freeze, matched baseline, held-out/intervention이 필요. |
| LGS-N3 | 삭제 | P0 | delay, overhead, parallel critical-path 반례. 식 (5)의 제한 모형만 생존. |
| LGS-H2 | [미완성] | P1 | (5)의 fixed-cost/parallelism 모델에서만 시험 가능. |
| LGS-N4 | 활성 주장 아님 | P1 | NREM/REM, 곡률, 조합의 조작적 정의와 직렬성 인과 증명이 계약에 없다. local asynchronous maintenance + periodic synchronization은 경쟁 모형이다. |
| LGS-X1 | 활성 제외 | 없음 | graph/trajectory 예측이 AGI 충분조건을 제공하는 bridge는 없다. |

## 재현

```powershell
python _workspace/ce/agi-learning-geometry-sleep-20260818/artifacts/verify_lgs_math.py
```

`artifacts/verify_lgs_math.py`는 3개 labelled vertex의 모든 directed non-loop arc가 absent/1/2인 $3^6=729$ base graph, 모든 새 arc와 $a\in\{1,2\}$를 검사했다. 총 8,748 cases 중 기존 더 싼 $u\to v$ arc가 이미 있는 1,458 cases도 포함하며, 원시 출력은 `artifacts/verify_lgs_math.log`에 있다.
