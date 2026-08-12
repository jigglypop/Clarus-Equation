# 11-math — AGI V15 Unified Metric Agent

Status: COMPLETE

## 1. 대상과 판정 요약

검산 대상은 `00-contract.md`의 UM-1부터 UM-6이다. 선행 Clarus-field의 유계 기억 정리는 재유도하지 않았다. 상세 독립 계산은 `artifacts/verify_unified_metric_math.py`와 `verify_unified_metric_math.log`에 있다.

| Claim | 판정 | P 등급 | 정확한 범위 |
|---|---|---|---|
| UM-1 좌표 공변성 | **정리** | 없음 | tensor와 점·벡터 및 고정 adjacency를 함께 운반한 affine chart change에서 local quadratic length와 graph path cost 불변 |
| UM-2 SPD·유계 | **산출/조건부 정리** | P1 경계 | $LL^T+\varepsilon I$ 또는 한 chart의 spectral projection은 SPD·condition bound를 보장하지만 spectral clipping은 일반 affine-covariant가 아님 |
| UM-3 fixed-metric field | **수정 후 조건부 정리** | P0 부모 범위 수정 | compact finite-volume manifold 또는 $r\in L^2(g)$가 필요; 완비+점wise bounded source만으로 $L^2$ bound는 거짓 |
| UM-4 goal no-go | **정리** | 없음 | source-free isometry-equivariant singleton selector는 후보를 고정점 없이 교환하는 isometry 아래 존재 불가 |
| static one-$g$ world direction | **no-go 정리** | P1 계약 해석 경계 | 정적 Riemannian distance는 대칭이므로 비가역 world transition을 고유하게 결정하지 못함 |
| UM-5 SCC 연속체 | **미완성** | P1 | sampling/overlap/operator consistency 없이는 finite graph가 continuum metric/Laplacian을 식별하지 않음 |
| UM-6 다섯 readout | **구현 예측** | P1 | finite invariant readout의 공동 작동 시험일 뿐 의미론·AGI 정리 아님 |

## 2. UM-1 — affine tensor 공변성

**정리 UM-1.** $y=Jx+b$, $J\in GL(d)$이고 $g_y=J^{-T}g_xJ^{-1}$라 하자. $v_y=Jv_x$이면

$$
v_y^Tg_yv_y
=v_x^TJ^TJ^{-T}g_xJ^{-1}Jv_x
=v_x^Tg_xv_x.
$$

따라서 local quadratic length가 불변이다. curve와 metric을 함께 운반하면 적분 길이와 geodesic distance도 불변이다. 유한 graph에서 endpoint 평균

$$
\bar g_{ij}=\frac{g_i+g_j}{2},
\qquad
\ell_{ij}^2=(z_j-z_i)^T\bar g_{ij}(z_j-z_i)
$$

를 쓰고 adjacency를 그대로 운반하면 각 edge length가 불변이므로 그 합의 최솟값인 shortest-path cost도 불변이다. $square$

독립 수치 검산의 최대 상대오차는 $2.339\times10^{-14}$였다. 이는 부동소수 구현 검산이지 정리의 증명 근거는 아니다.

**경계.** 변환 후 Euclidean $k$-NN을 다시 계산하면 adjacency가 바뀔 수 있으므로 graph 경로 불변성이 자동으로 따라오지 않는다. 또한 한 chart에서 고유값을 $[m,M]$로 clipping하는 연산 $P$는 일반적으로

$$
P(J^{-T}gJ^{-1})\ne J^{-T}P(g)J^{-1}
$$

이다. $g=I$, $J=\operatorname{diag}(10,1)$, $[m,M]=[0.1,2]$에서 $x$ chart의 길이제곱 1이 변환 후 clipping 전에는 1이지만 clipping 뒤 10이 된다. 따라서 UM-1은 **projection을 적용하지 않은 읽기 연산**의 정리다.

## 3. UM-2 — SPD와 condition bound

**정리 UM-2A.** 임의의 실수행렬 $L$과 $\varepsilon_g>0$에 대해

$$
g=LL^T+\varepsilon_g I
$$

는 SPD이고 $\lambda_{\min}(g)\ge\varepsilon_g$다. 임의의 대칭행렬을 고유분해해 고유값을 $[m,M]$, $0<m\le M$로 clipping하면 한 고정 chart에서 결과는 SPD이고 condition number는 $M/m$ 이하이다.

**증명.** $v\ne0$에 대해 $v^Tgv=\lVert L^Tv\rVert^2+\varepsilon_g\lVert v\rVert^2>0$이다. clipping 결과의 고유값이 정의상 $[m,M]$에 있으므로 비율 상계가 따른다. $square$

**산출 UM-2B.** $g$와 SPD source metric $g_s$가 모두 Loewner 의미에서 $mI\preceq g,g_s\preceq MI$이고 $0\le\alpha\le1$이면

$$
g^+=(1-\alpha)g+\alpha g_s
$$

도 같은 경계를 만족한다. 이는 bounded metric deformation 구현의 승인 후보다.

**P1 경계.** $mI\preceq g\preceq MI$ 자체는 coordinate-dependent 수치 인증서다. 일반 affine chart에서 같은 수치 $m,M$을 tensor 불변 상수처럼 쓰지 않는다. affine covariance fixture에서는 변환 뒤 재투영하지 않아야 한다.

## 4. UM-3 — 고정 metric 장의 정확한 범위

계약의 “완비 또는 compact, bounded source” 문구는 $L^2$ 정리로는 너무 넓다. 완비 비콤팩트 $\mathbb R^d$에서 $r\equiv1$, $\phi(0)=0$이면

$$
\phi(t)=\frac{1-e^{-\lambda t}}{\lambda}
$$

는 공간상 상수이고 모든 $t>0$에서 $\lVert\phi(t)\rVert_{L^2(\mathbb R^d)}=\infty$다. 점wise bounded source가 $L^2$ source임을 뜻하지 않으므로 원 부모 범위는 반례로 제거한다.

**정리 UM-3F.** 경계가 없거나 energy flux가 0인 고정 Riemannian manifold에서 $\phi_0,r\in L^2(g)$, $\lambda>0$이고

$$
\partial_t\phi=\kappa\Delta_g\phi-\lambda\phi+r
$$

라 하자. 충분히 정칙한 해는

$$
\frac12\frac{d}{dt}\lVert\phi\rVert_{L^2(g)}^2
=-\kappa\lVert\nabla_g\phi\rVert_{L^2(g)}^2
-\lambda\lVert\phi\rVert_{L^2(g)}^2
+\langle r,\phi\rangle_g
$$

를 만족한다. Young 부등식으로

$$
\frac{d}{dt}\lVert\phi\rVert_2^2
\le-\lambda\lVert\phi\rVert_2^2+\lambda^{-1}\lVert r\rVert_2^2
$$

이고, $r$의 $L^2$ 상계가 있으면 Gronwall로 전역 energy bound가 나온다. compact manifold에서는 bounded source가 자동으로 $L^2$다. scalar heat semigroup의 positivity로 비음 초기값과 비음 source의 positivity도 보존된다. $square$

**시간가변 metric 경계.** $g=g_t$이면 volume form이

$$
\partial_t d\mu_{g_t}
=\frac12\operatorname{tr}_{g_t}(\dot g_t)d\mu_{g_t}
$$

로 변하므로 energy derivative에

$$
\frac14\int\operatorname{tr}_{g_t}(\dot g_t)\phi^2d\mu_{g_t}
$$

가 추가된다. $\operatorname*{ess\,sup}\operatorname{tr}_{g_t}(\dot g_t)\le\beta$이면 감쇠 계수는 대략 $2\lambda-\beta/2$가 되어 $\beta<4\lambda$ 같은 metric-rate 조건이 필요하다. $g_t=e^{2ct}I$와 spatially constant mode에서는 diffusion이 0인데 volume이 $e^{dct}$로 증가하므로, $dc>2\lambda$이면 $L^2(g_t)$ energy가 증가한다. fixed-$g$ 정리는 time-varying update에 자동 상속되지 않는다.

## 5. UM-4 — 목표 선택과 방향성 no-go

**정리 UM-4.** 후보집합 $C$에서 한 점을 반환하는 goal selector $F(g)\in C$가 metric isometry $\varphi$에 대해 equivariant라고 하자. $\varphi^*g=g$이면

$$
F(g)=F(\varphi^*g)=\varphi^{-1}F(g).
$$

따라서 $\varphi$가 $C$의 어느 점도 고정하지 않고 후보를 교환하면 singleton selector는 존재하지 않는다. 대칭 상황에서 허용되는 invariant 출력은 전체 minimizer orbit 또는 tie 집합이다. 한 점을 고르려면 외부 source/boundary, 현재 상태 또는 hidden label tie-break가 필요하다. $square$

이는 계약의 “외부 목표가 source로 들어간다”는 공리가 필요한 이유다. $g$가 source를 기억한 뒤 goal basin을 읽는 것은 가능하지만, source-free metric이 의미 목표를 무에서 생성한다는 결론은 불가능하다.

**정리 UM-4D (방향성 no-go).** 정적 Riemannian metric에 대해 역경로 $\bar\gamma(t)=\gamma(1-t)$는 원 경로와 같은 길이를 가지므로

$$
d_g(x,y)=d_g(y,x).
$$

따라서 정적 $g$의 거리만으로 forward/backward가 다른 비가역 world transition 또는 행동비용을 고유하게 산출할 수 없다. drift vector field, time-dependent $g_t$, asymmetric control set, Finsler/Randers 항 또는 외생 시간 orientation 가운데 적어도 하나가 필요하다. 이 추가 객체를 넣으면 “정적 $g$ 하나만으로 world dynamics가 결정된다”는 부모 주장은 중단해야 한다. $square$

## 6. UM-5 — finite SCC와 continuum의 비식별성

$N$개의 점과 그 점에서의 $g_i$만으로 점 사이 metric field는 유일하게 정해지지 않는다. 1차원 또는 직선 위의 두 endpoint에서 $g=I$인 두 smooth conformal metric을 구성하되, 내부에서 하나만 작게 만들 수 있다. endpoint 표본은 같지만 두 점 사이 연속 geodesic 길이는 다르다. 독립 fixture에서 endpoint 값이 같은 두 metric의 거리는 각각 2.0000과 1.02936이었다.

따라서 finite SCC node를 chart/sample로 읽는 것은 [정의/모델 선택]까지 가능하지만 다음이 없으면 continuum 주장은 [미완성]이다.

1. fill distance와 sampling density 조건,
2. chart overlap transition과 cocycle,
3. adjacency·quadrature 운반 규칙,
4. graph operator consistency와 수렴 topology,
5. direct-limit embedding compatibility.

선행 NISCC-5A/5B의 exact compatibility와 uniform Lipschitz 조건도 계속 필요하다. SCC가 directed인 사실은 symmetric Laplace–Beltrami operator의 근사를 자동으로 주지 않는다. directed world relation과 symmetric metric substrate를 동일시하지 않는다.

## 7. UM-6 — 다섯 기능 통합의 정확한 지위

같은 finite metric state에서 local distance, graph path, surprise, deformation, tie-preserving goal readout을 계산하는 것은 구현 가능한 [산출] 후보다. 하지만 다음은 $g$의 정리가 아니다.

- future observation을 만드는 비가역 transition law,
- 무엇을 기억할지 정하는 source law,
- 의미 있는 목표의 외부 기원,
- task distribution에 대한 일반화와 continual learning,
- 무한 SCC refinement가 지능을 증가시킨다는 결론.

따라서 이번 baseline에서 “world”는 **고정 graph의 metric cost substrate**, “memory”는 **source 전후 metric deformation**, “planning”은 **metric graph shortest path**, “critic”은 **local metric surprise**, “goal”은 **source가 만든 cost에 대한 tie-preserving minimizer readout**으로 좁힌다. 이는 다섯 독립 persistent state를 없애는 구조 실험이지 AGI 완결이다.

## 8. 무차원 감사

정보좌표 $z$가 기준척도 $\ell_0$로 정규화됐다고 두면 $g$, $m$, $M$, $\varepsilon_g$, $\alpha$는 무차원이다. 물리 차원을 부여하는 경우 local length $d_g$는 정보길이 차원을 갖고 hard gate에는 반드시

$$
\delta_g^2/\ell_0^2
$$

만 들어가야 한다. sigmoid/exp를 추가할 경우 그 인자도 같은 무차원 비율과 무차원 gain/bias로 구성해야 한다. 무차원 정합은 물리적 타당성이나 AGI 효능을 뜻하지 않는다.

## 9. P0 / P1 / P2

- **P0-1 해소 범위 제안:** UM-3의 “완비+bounded source면 $L^2$ bound” 부모 범위는 $\mathbb R^d$, $r=1$ 반례로 제거하고, compact 또는 $r\in L^2$인 UM-3F로 교체해야 한다.
- **P1-1:** spectral projection은 fixed-chart numerical certificate이며 affine-covariant update가 아니다. 구현 인증서에서 두 성질을 분리해야 한다.
- **P1-2:** 정적 one-$g$는 비가역 dynamics를 정하지 못한다. 구현의 world 명칭을 metric cost substrate로 좁혀야 한다.
- **P1-3:** finite SCC-to-manifold continuum 및 Laplace–Beltrami 수렴은 미완성이다. 현재 구현은 finite metric graph로만 인증해야 한다.
- **P1-4:** 다섯 역할의 공동 readout은 합성 구현 예측이다. AGI 효능과 의미론은 별도 과제·외부 source law가 필요하다.
- **P2-1:** “리만기하학 평면”은 $d=2$로 오해될 수 있으므로 본체는 $d$차원 다양체, 2D는 analytic/visual fixture로 쓴다.

## 10. 구현 승인 제안

P0-1의 부모 범위를 활성 주장에 남기지 않고 P1 경계를 공개 certificate에 고정한다는 조건으로 다음 finite 범위는 구현 가능하다.

1. SPD validation과 fixed-chart spectral condition certificate,
2. projection을 끈 affine tensor/readout covariance,
3. endpoint-average metric graph edge와 shortest path,
4. 무차원 local surprise hard gate,
5. bounded convex source deformation,
6. symmetric goal tie/no-go regression,
7. full geodesic, curvature, continuum, directed world dynamics, AGI/뇌/우주 증거를 `False`로 명시.

## 11. 재현

```powershell
.venv\Scripts\python.exe _workspace/ce/agi-v15-unified-metric-20260813/artifacts/verify_unified_metric_math.py
```

기록: `artifacts/verify_unified_metric_math.log`.

Status: COMPLETE
