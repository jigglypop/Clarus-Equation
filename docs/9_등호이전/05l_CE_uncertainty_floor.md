# 05l. 불확정성·Gaussian 모드·경로 정칙성의 분리

이 문서는 Kennard 불확정성, 독립 Gaussian mode noise, 경로 regularity를 서로 다른 가정과 결론으로 분리한다. uncertainty floor는 측정 noise·model discrepancy·식별성의 외부 입력을 필요로 하며, 수학적 부등식만으로 CE의 관측 바닥을 산출하지 않는다.

독자는 05k의 constraint/penalty 경계와 05i의 prior route를 먼저 읽는다. Kennard 부등식, Gaussian toy model, Brownian–Sobolev 경계, mean-field bound 및 현재 보존 결론 순서로 읽는다.

## 0. 범위

불확정성, prior의 regularity, 추정 오차 floor는 같은 단어로 섞이기 쉽지만 정의역과 데이터 생성 가정이 다르다. 이 절은 각 층의 정리·경험 입력·미완성 해석을 구분한다.

Kennard 부등식, 한 Gaussian action 모형의 모멘트, Brownian path의
quadratic variation은 각각 참인 결과지만 서로를 자동으로 유도하지 않는다.
이 문서는 세 층을 분리하고 threshold·mean-field에 실제로 필요한 조건만
남긴다.

| 항목 | 형식 출처 |
|---|---|
| Kennard 부등식과 조화진동자 바닥에너지 | **[정리]** |
| 독립 Gaussian action의 Gamma law | **[정리]**; prior는 **[공리: 모델 선택]** |
| Brownian과 $H^1$ path의 quadratic variation 경계 | **[정리]** |
| Brownian variance와 $\hbar$의 식별 | 일반적으로 성립하지 않음 |
| $N_{\rm eff}$ 독립모드 mean-field bound | 조건부 **[정리]** |
| CE residual의 실제 mode 분해 | **[미완성]** |

## 1. Kennard 부등식

Kennard 부등식은 지정한 양자 상태와 공액 관측량의 분산에 관한 형식 정리다. 이는 path prior의 support, Gaussian model discrepancy, 또는 실험 장치의 noise floor를 직접 정하지 않는다.

**[정리]** Hilbert space의 self-adjoint operators $X,P$가 공통
dense invariant domain에서
$[X,P]=i\hbar\mathbf1$을 만족한다고 하자. 단위벡터 $\psi$가
$XP$, $PX$와 두 분산이 정의되는 domain에 있으면

$$
\operatorname{Var}_\psi(X)\operatorname{Var}_\psi(P)
\geq\frac{\hbar^2}{4}.
$$

**증명.** 평균을 뺀
$\widetilde X=X-\langle X\rangle$,
$\widetilde P=P-\langle P\rangle$에 대해
$$
\frac{\hbar}{2}
=
\left|\operatorname{Im}
\langle\widetilde X\psi,\widetilde P\psi\rangle\right|
\leq
\|\widetilde X\psi\|\,\|\widetilde P\psi\|.
$$
제곱하면 결론을 얻는다. $\square$

**[정리]** 질량 $m>0$, 주파수 $\omega>0$인 조화진동자
$$
H=\frac{P^2}{2m}+\frac{m\omega^2X^2}{2}
$$
는 모든 허용 단위상태에서
$$
\langle H\rangle\geq\frac{\hbar\omega}{2}
$$
를 만족한다.

**증명.** 평균항을 버리고 AM--GM과 Kennard 부등식을 적용하면
$$
\langle H\rangle
\geq
\sqrt{\operatorname{Var}(P)\,\omega^2\operatorname{Var}(X)}
\geq\frac{\hbar\omega}{2}.
\quad\square
$$

이 결과는 canonical pair와 Hamiltonian을 고정한 뒤의 에너지 하한이다.
임의의 확률과정의 diffusion coefficient나 임의의 Euclidean functional의
분산을 결정하지 않는다.

## 2. 독립 Gaussian action 모형

독립 Gaussian 모형은 각 mode의 분산과 action 비용을 계산하는 기준 사례다. 독립성·Gaussian성·covariance는 경험적으로 추정하거나 설계로 택해야 하며 보편 물리 법칙이 아니다.

**[공리: 모델 선택]** $z_1,\dots,z_N$을 독립 표준정규 변수로 두고
무차원 action
$$
U_N:=\frac12\sum_{k=1}^Nz_k^2
$$
를 택한다.

**[정리]**
$$
U_N\sim\operatorname{Gamma}\!\left(\frac N2,1\right),
\qquad
\mathbb E U_N=\frac N2,
\qquad
\operatorname{Var}(U_N)=\frac N2.
$$

**증명.** $\sum_kz_k^2$는 자유도 $N$인 chi-square 분포이므로
그 절반은 표시한 Gamma 분포다. 또는 moment-generating function
$\mathbb E e^{tU_N}=(1-t)^{-N/2}$, $t<1$을 두 번 미분한다.
$\square$

작용 차원을 복원해 $S_N=\hbar U_N$로 **정의**하면
$$
\mathbb ES_N=\frac{N\hbar}{2},
\qquad
\operatorname{Var}(S_N)=\frac{N\hbar^2}{2}.
$$
이는 선택한 Gaussian density의 산출이지 Kennard 부등식의 산출이 아니다.

**[정리]** $q\in(0,1)$, $z_q=\Phi_{\rm normal}^{-1}(q)$이고
$u_N(q)$를 $U_N$의 $q$-quantile이라 하면
$$
u_N(q)
=
\frac N2+z_q\sqrt{\frac N2}+o(\sqrt N).
$$

**증명.** 중심화한 독립 변수
$(z_k^2-1)/2$에 central limit theorem을 적용하고, 극한 정규분포의
분포함수가 연속이고 엄격히 증가한다는 quantile convergence를 쓴다.
$\square$

따라서 $S_N$의 같은 quantile은
$$
S_N(q)
=
\frac{N\hbar}{2}
+z_q\hbar\sqrt{\frac N2}
+o(\hbar\sqrt N)
$$
다. $q$와 Gaussian independence는 **[공리: 모델 선택]**이며,
불확정성만으로 정해지지 않는다.

## 3. Brownian과 Sobolev path의 경계

Brownian과 Sobolev 경로는 regularity와 거의확실한 support가 달라 동일한 kinetic constraint를 공유하지 않는다. 이 경계는 prior choice와 uncertainty interpretation을 구분하는 반례 조건이다.

**[정리]** $\gamma\in H^1([0,1];\mathbb R^d)$이면 mesh가 0으로 가는
모든 partition $\Pi_n$에 대해
$$
\sum_{[u,v]\in\Pi_n}
|\gamma(v)-\gamma(u)|^2\longrightarrow0.
$$

**증명.** $H^1\subset W^{1,1}$이므로 $\gamma$는 finite variation이고
$$
\sum|\Delta\gamma|^2
\leq
\max|\Delta\gamma|\,\operatorname{Var}(\gamma)\to0.
\quad\square
$$

**[정리]** $\mathbb R^d$의 bridge
$$
B_t^{x_i,x_f}
=(1-t)x_i+tx_f+\sqrt{\sigma}\,\widetilde B_t,
\qquad \sigma>0,
$$
는 균등 partition에서 각 성분의 quadratic variation이 $\sigma$,
Euclidean 제곱합의 극한이 $d\sigma$다. 따라서 거의 모든 sample path는
$H^1$에 속하지 않는다.

이 정리는 $\sigma>0$인 모든 scale에서 성립한다. $\sigma$의 차원은
좌표 제곱이며 $\hbar$의 차원은 작용이므로, 질량·시간·길이 기준을 포함한
추가 사상 없이 $\sigma=\hbar$라고 둘 수 없다. Kennard 부등식도
$\sigma$의 유일한 값을 고르지 않는다.

## 4. Intensive 독립모드의 mean-field bound

mean-field bound는 모드 수·독립성·uniform moment 등 명시한 가정 아래의 점근 결과다. 상관, heavy tail, 식별 불가능한 model discrepancy에서는 uncertainty floor로 승격할 수 없다.

**[공리: 모델 선택]** $N=N_{\rm eff}$이고
$$
\Phi_N=\sum_{k=1}^NY_k,
\qquad
Y_k\ \text{독립},\qquad
0\leq Y_k\leq\frac CN,\qquad
\operatorname{Var}(Y_k)=\frac{v_k}{N^2},
$$
여기서 $0\leq v_k\leq v_{\max}<\infty$라 하자. 그러면
$0\leq\Phi_N\leq C$.

**[정리]**
$$
\operatorname{Var}(\Phi_N)
=
\frac1{N^2}\sum_{k=1}^Nv_k
\leq\frac{v_{\max}}N
$$
이고
$$
1
\leq
\frac{\mathbb E e^{-\Phi_N}}
{e^{-\mathbb E\Phi_N}}
\leq
\exp\!\left(
\frac{e^C v_{\max}}{2N}
\right).
$$
추가로 $v_k\geq v_{\min}>0$이면
$$
\operatorname{Var}(\Phi_N)\geq\frac{v_{\min}}N.
$$

**증명.** 독립성으로 분산은 합산된다. 아래쪽 비는 $e^{-x}$의
convexity에 대한 Jensen 부등식이다. Taylor 정리와
$\sup_{[0,C]}(e^{-x})''\leq1$로
$$
\mathbb Ee^{-\Phi_N}
\leq
e^{-\mathbb E\Phi_N}
+\frac12\operatorname{Var}(\Phi_N).
$$
$e^{-\mathbb E\Phi_N}\geq e^{-C}$로 나누고 $1+x\leq e^x$를 쓰면
위쪽 비를 얻는다. $\square$

상관된 모드에서는 covariance 합이 추가되므로 이 $N^{-1}$ bound는
자동이 아니다. 실제 CE residual이 위 독립·유계·intensive 분해를 갖는지는
**[미완성]**이다.

## 5. 현재 보존되는 결론

현재 닫힌 것은 각 수학 모형의 조건부 bound이며, 실제 CE noise·calibration·data provenance와 model discrepancy를 합친 불확정성 floor는 아직 미완성이다. 관측 설계는 baseline·holdout·반증 기준을 별도로 제공해야 한다.

- Kennard와 조화진동자 바닥에너지는 operator domain 아래의 정리다.
- Gamma action moment와 threshold 폭은 선택한 독립 Gaussian 모형의
  정리다.
- Brownian prior와 $H^1$ variational pathspace는 서로 다른
  measure/topology route다.
- 독립 intensive 분해 아래에서만 mean-field 오차가
  $O(N_{\rm eff}^{-1})$로 통제된다.

분위수, diffusion scale, mode 독립성 또는 이를 특정 물리장에 대응시키는
단계는 각각 별도의 **[공리]** 또는 **[미완성]** 항목이다.
