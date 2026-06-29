# 05k. CE Hard Constraint Package

## 0. 목표

[05j_CE_supp_scaling_audit.md](05j_CE_supp_scaling_audit.md)는 분율 체계(bootstrap, \(\Omega_b\))가 실제로 쓰는 suppression이 hard constraint

$$
A_{\mathrm{th}}
=
\{\gamma:S_E[\gamma]<S_{\mathrm{th}}\},
\qquad
S_{\mathrm{supp}}
=
\infty\cdot\mathbf 1_{A_{\mathrm{th}}^c}
$$

임을 확인했다(`supp:hard`, [../경로적분.md](../경로적분.md) 정리 3.2.1). 이 문서는 hard constraint 조건화를 세 층으로 닫는다.

1. 조건화 측도 \(\mu(\cdot\mid A_{\mathrm{th}})\)의 존재와 recovery.
2. threshold \(S_{\mathrm{th}}\)의 scale: continuum에서는 정의 불능, finite-\(N\)에서는 \(N\)-dependent.
3. tilt-threshold 대응: \(\langle e^{-\Phi}\rangle\)와 \(Z_{\mathrm{surv}}/Z\)의 정확한 관계.

핵심 결론:

> hard constraint는 continuum kinetic action 위에서는 null set 조건화라 정의되지 않는다. 유효한 자리는 finite-\(N\) 모드 모델이고, 거기서 고정 분율 \(q\in(0,1)\)을 주는 threshold는 절대 작용 scale이 아니라
> \[
> u_{\mathrm{th}}(N)=\langle S_E\rangle+z_q\sqrt{\operatorname{Var}(S_E)}+o(\sqrt N)
> \]
> 즉 **평균 작용 주변의 요동 scale**로만 존재한다. 또한 smooth tilt \(\langle e^{-\Phi}\rangle\)는 threshold 분율들의 \(e^{-t}\)-가중 평균과 정확히 같고(layer-cake 항등식), mean-field 근사 \(e^{-\langle\Phi\rangle}\)는 Jensen에 의해 **항상 과소평가**이며 오차는 \(\operatorname{Var}(\Phi)\)로 통제된다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| continuum kinetic hard constraint | `False` (정의 불능) | 정리 2.1 |
| finite-\(N\) 조건화 존재 | `Exact` | 정리 3.1 |
| threshold의 \(N\)-dependence | `Exact under assumptions` | 정리 3.2 |
| 경로분율-에너지분율 일치 (정리 3.2.1 재검) | `Exact under assumptions`, 수렴율 보정 | 정리 3.3 |
| 조건화 recovery와 manifest 불변 | `Exact under assumptions` | 정리 4.1, 4.2 |
| tilt-threshold layer-cake 항등식 | `Exact` | 정리 5.1 |
| 단일 threshold 대응 | `Exact under assumptions` | 정리 5.2 |
| mean-field 부등식과 오차 통제 | `Exact` | 정리 5.3 |
| threshold 값의 물리적 선택 | `Selection/Open` | 6절 |

## 1. 세팅과 가측성

\(\Gamma\)를 Polish space, \(\mu\in\mathcal P(\Gamma)\), \(S_E:\Gamma\to[0,\infty]\)를 l.s.c.라 하자. l.s.c.에서 \(\{S_E\le c\}\)는 닫힌집합이므로

$$
A_{\mathrm{th}}
=
\{S_E<S_{\mathrm{th}}\}
=
\bigcup_{n\ge1}\{S_E\le S_{\mathrm{th}}-1/n\}
$$

는 \(F_\sigma\), 따라서 Borel 가측이다. 조건화 측도는 \(\mu(A_{\mathrm{th}})>0\)일 때만

$$
\mu_{\mathrm{th}}
:=
\mu(\cdot\mid A_{\mathrm{th}})
=
\frac{\mu(\cdot\cap A_{\mathrm{th}})}{\mu(A_{\mathrm{th}})}
$$

로 정의된다. 모든 문제는 \(\mu(A_{\mathrm{th}})>0\)에 걸린다.

## 2. Continuum 장애물

### 정리 2.1: continuum kinetic hard constraint는 정의 불능

\(\mu=\mathbb B_{x_i,x_f}\)(Brownian bridge) 또는 [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md)의 scaled bridge \(\mu^\hbar\)라 하자. 임의의 \(S_{\mathrm{th}}<\infty\)에 대해

$$
\mu(A_{\mathrm{th}})
\le
\mu(\{S_E<\infty\})
=
\mu(H^1)
=0.
$$

따라서 조건화 측도가 존재하지 않는다.

증명:

\(\{S_E<S_{\mathrm{th}}\}\subset\{S_E<\infty\}=H^1_{x_i,x_f}\)이고 05i 정리 2.1에 의해 bridge는 \(W^{1,p}\)에 질량 0을 준다. scaled bridge도 같은 Gaussian class라 동일하다. 끝.

자유장 버전도 같다. \(N\)개 모드의 자유장에서 \(\langle S_E\rangle=N/2\to\infty\)이므로, 고정된 \(S_{\mathrm{th}}\)에 대한 생존 분율은 \(N\to\infty\)에서 0으로 간다(정리 3.2의 따름). 즉 **연속체에서 절대 작용 threshold는 살아남지 못한다.** hard constraint의 유효한 집은 finite-\(N\) 층이다.

## 3. Finite-\(N\) 모델

[../경로적분.md](../경로적분.md) 정리 3.2.1의 세팅을 그대로 쓴다. \(N\)개 모드 자유장에서 prior가 \(e^{-S_E}\mathcal D\phi\)의 정규화이고, \(u=S_E\)의 law는 shape \(N/2\)의 Gamma 분포다.

$$
u\sim\mathrm{Gamma}(N/2,1),
\qquad
\langle u\rangle=\frac N2,
\qquad
\operatorname{Var}(u)=\frac N2.
$$

### 정리 3.1: finite-\(N\) 조건화 존재

임의의 \(u_{\mathrm{th}}>0\)에 대해

$$
\mu_N(A_{\mathrm{th}})
=
\frac{\gamma(N/2,u_{\mathrm{th}})}{\Gamma(N/2)}
>0
$$

이므로 조건화 측도는 항상 존재한다.

증명:

Gamma 밀도 \(u^{N/2-1}e^{-u}/\Gamma(N/2)\)는 \((0,u_{\mathrm{th}})\)에서 양수이므로 적분이 양수다. 끝.

### 정리 3.2: 고정 분율은 \(N\)-dependent threshold를 강제한다

\(q\in(0,1)\)이라 하자. 다음은 동치다.

$$
\frac{\gamma(N/2,u_{\mathrm{th}}(N))}{\Gamma(N/2)}
\to q
\quad\Longleftrightarrow\quad
\frac{u_{\mathrm{th}}(N)-N/2}{\sqrt{N/2}}
\to z_q,
$$

여기서 \(z_q\)는 표준정규 분위수(\(\Phi_{\mathcal N}(z_q)=q\))다.

증명:

\(u\sim\mathrm{Gamma}(N/2,1)\)에 대해 \((u-N/2)/\sqrt{N/2}\Rightarrow\mathcal N(0,1)\)이다(Gamma CLT, 외부 import).

(\(\Leftarrow\)) 정규화 변수의 분포 수렴과 극한 분포함수의 연속성으로

$$
P\big(u<u_{\mathrm{th}}(N)\big)
=
P\Big(\tfrac{u-N/2}{\sqrt{N/2}}<\tfrac{u_{\mathrm{th}}(N)-N/2}{\sqrt{N/2}}\Big)
\to\Phi_{\mathcal N}(z_q)=q.
$$

(\(\Rightarrow\)) \(t_N:=(u_{\mathrm{th}}(N)-N/2)/\sqrt{N/2}\)의 부분열 극한 \(t\in[-\infty,\infty]\)를 잡으면 위 수렴으로 그 부분열에서 분율은 \(\Phi_{\mathcal N}(t)\)로 간다(\(\pm\infty\)이면 0 또는 1). 분율 전체가 \(q\in(0,1)\)로 수렴하므로 모든 부분열 극한이 \(z_q\)여야 한다. 끝.

해석:

> 고정 분율 \(q=\varepsilon^2\approx0.0486\)을 주는 threshold는
>
> $$
> u_{\mathrm{th}}(N)
> =
> \frac N2+z_{0.0486}\sqrt{\frac N2}+o(\sqrt N),
> \qquad
> z_{0.0486}\approx-1.66
> $$
>
> 뿐이다. 즉 "접힘 통과 조건 \(S_E<S_{\mathrm{th}}\)"는 절대 작용 scale의 조건이 아니라 **평균 작용에서 표준편차 1.66배 아래**라는 요동 조건이다. \(S_{\mathrm{th}}\)는 상수가 아니라 모드 수를 따라가는 양이며, 이는 `Selection`으로 선언되어야 한다.

### 정리 3.3: 경로분율-에너지분율 일치의 수렴율 보정

정리 3.2의 central scaling \(u_{\mathrm{th}}(N)=N/2+z\sqrt{N/2}\)에서

$$
\left|
\frac{\gamma(N/2,u_{\mathrm{th}})}{\Gamma(N/2)}
-
\frac{\gamma(N/2+1,u_{\mathrm{th}})}{\Gamma(N/2+1)}
\right|
=
O(N^{-1/2}).
$$

증명:

불완전 감마의 점화식 \(\gamma(k+1,u)=k\gamma(k,u)-u^ke^{-u}\)을 \(\Gamma(k+1)=k\Gamma(k)\)로 나누면

$$
\frac{\gamma(k+1,u)}{\Gamma(k+1)}
=
\frac{\gamma(k,u)}{\Gamma(k)}
-
\frac{u^ke^{-u}}{\Gamma(k+1)}.
$$

마지막 항은 shape \(k+1\) Gamma 밀도의 \(u\)에서의 값이다. central regime \(u=k+O(\sqrt k)\)에서 이 밀도는 \(O(k^{-1/2})\)이다(Stirling, 외부 import). \(k=N/2\)로 끝.

주의:

- [../경로적분.md](../경로적분.md) 정리 3.2.1은 차이를 \(O(1/N)\)으로 적었다. 고정 \(u_{\mathrm{th}}\) regime(두 분율이 모두 0으로 가는 영역)에서는 그 표기가 가능하지만, 분율이 \(q\in(0,1)\)로 살아남는 central regime에서의 올바른 수렴율은 \(O(N^{-1/2})\)다. 정리의 결론(극한에서 경로분율 = 에너지분율)은 유지되고 수렴율만 보정된다.

## 4. 조건화 recovery와 manifest 불변

\(\mu(A_{\mathrm{th}})>0\)인 층(finite-\(N\), 또는 support가 \(H^1\)에 들어가는 별도 prior)에서 조건화가 농축 구조를 보존하는지 확인한다. 이 절에서는 CE 사용처와 같게 조건화 functional과 Gibbs 에너지가 동일한 \(E\)라고 두고 \(A_{\mathrm{th}}=\{E<S_{\mathrm{th}}\}\)로 쓴다.

### 정리 4.1: recovery는 조건화에서 살아남는다

[05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md)의 세팅(\(E\) good rate, \(m=\inf_{\operatorname{supp}\mu}E\), recovery mass)을 두고

$$
S_{\mathrm{th}}>m
$$

이라 하자. 그러면 \(\mu(A_{\mathrm{th}})>0\)이고, 조건화 측도 \(\mu_{\mathrm{th}}\)도 recovery mass를 만족한다.

증명:

\(\eta_0:=S_{\mathrm{th}}-m>0\)으로 두면 near-minimum set은

$$
R_{\eta_0}
=
\{E<m+\eta_0\}
=
\{E<S_{\mathrm{th}}\}
=
A_{\mathrm{th}}
$$

이고 recovery mass로 \(\mu(A_{\mathrm{th}})=\mu(R_{\eta_0})>0\)이다.

이제 \(\eta\in(0,\eta_0)\)에 대해 \(R_\eta\subset A_{\mathrm{th}}\)이므로

$$
\mu_{\mathrm{th}}(R_\eta)
=
\frac{\mu(R_\eta)}{\mu(A_{\mathrm{th}})}
>0.
$$

\(\eta\ge\eta_0\)인 경우는 \(R_\eta\supset R_{\eta_0/2}\)로 환원된다. 끝.

### 정리 4.2: \(S_{\mathrm{th}}>m\)이면 manifest set은 불변

위 가정에 더해 \(\operatorname{supp}\mu_{\mathrm{th}}\) 위의 최소값이 \(m\)과 같다고 하자(예: 어떤 minimizer가 \(A_{\mathrm{th}}\)의 closure 안에서 접근 가능). 그러면 조건화 Gibbs 측도

$$
\mu_{\mathrm{th},\beta}
\propto
e^{-\beta E}\mu_{\mathrm{th}}
$$

는 \(\beta\to\infty\)에서 **원래와 같은** minimizer set \(\Gamma_*\)의 근방으로 농축한다.

증명:

05e 정리 1.1의 증명을 \(\mu_{\mathrm{th}}\)에 적용한다. 필요한 것은 두 가지다. (i) recovery mass: 정리 4.1. (ii) outer gap: \(E\)의 goodness는 prior와 무관한 성질이므로 그대로다. 분자(바깥 질량)는 \(\mu_{\mathrm{th}}\le\mu/\mu(A_{\mathrm{th}})\)로 같은 지수 상한을 갖고, 분모는 (i)로 같은 지수 하한을 갖는다. 끝.

해석:

> hard constraint는 bounded tilt(05i 정리 5.1)와 같은 쪽에 선다. \(S_{\mathrm{th}}>m\)인 한 **manifest 선택을 바꾸지 못하고** finite-\(\beta\) 분율만 바꾼다. 선택을 바꾸는 유일한 경우는 threshold가 minimizer를 잘라내는 \(S_{\mathrm{th}}\le m\)인데, 이때는 \(A_{\mathrm{th}}\) 안의 새로운 최소값으로 농축이 옮겨간다. CE 분율 체계는 전자에 해당하므로 05j의 결론(분율 주장은 manifest 주장이 아니다)이 hard constraint에서도 유지된다.

## 5. Tilt-threshold 대응

### 정리 5.1: layer-cake 항등식

\(\Phi:\Gamma\to[0,\infty]\) 가측이면

$$
\langle e^{-\Phi}\rangle_\mu
=
\int_0^\infty e^{-t}\,\mu(\Phi\le t)\,dt.
$$

증명:

\(e^{-\Phi}\in[0,1]\)이므로

$$
\langle e^{-\Phi}\rangle
=
\int_0^1\mu\big(e^{-\Phi}\ge s\big)\,ds.
$$

\(s=e^{-t}\), \(ds=-e^{-t}dt\)로 치환하면 \(\{e^{-\Phi}\ge e^{-t}\}=\{\Phi\le t\}\)이고

$$
\langle e^{-\Phi}\rangle
=
\int_0^\infty e^{-t}\mu(\Phi\le t)\,dt.
$$

검산: \(\Phi\equiv c\)이면 우변은 \(\int_c^\infty e^{-t}dt=e^{-c}\). 끝.

해석:

> smooth tilt 생존확률은 단일 threshold 분율이 아니라 **모든 threshold 분율의 \(e^{-t}\)-가중 평균**이다. 경로적분.md가 3.2절(tilt)과 3.2.1(threshold)에서 쓰는 두 독법은 이 항등식으로 정확히 연결된다.

### 정리 5.2: 단일 threshold 대응

\(\Phi\)의 law가 atomless이고 \(0<\langle e^{-\Phi}\rangle<1\)이면, 어떤 \(t_*\in(0,\infty)\)가 존재해서

$$
\langle e^{-\Phi}\rangle
=
\mu(\Phi\le t_*).
$$

증명:

\(g(t)=\mu(\Phi\le t)\)는 단조증가이고 atomless 가정에서 연속이며 \(g(0)=0\), \(g(\infty)=\lim_{t\to\infty}g(t)\ge\langle e^{-\Phi}\rangle\)... 정확히는 \(\langle e^{-\Phi}\rangle\le\mu(\Phi<\infty)=g(\infty^-)\)이고 \(\langle e^{-\Phi}\rangle>0=g(0)\)이므로 중간값 정리로 \(t_*\)가 존재한다. 끝.

주의:

- \(t_*\)는 \(\Phi\)의 분포 전체로 결정되는 양이지 자유 파라미터가 아니다. tilt 독법과 threshold 독법을 바꿔 쓸 수는 있지만, threshold 값은 그 교환에서 **유도되는 출력**이다.

### 정리 5.3: mean-field 부등식과 오차 통제

(i) 모든 가측 \(\Phi\ge0\)에 대해 (Jensen)

$$
\langle e^{-\Phi}\rangle
\ge
e^{-\langle\Phi\rangle}.
$$

등호는 \(\Phi\)가 a.s. 상수일 때뿐이다.

(ii) \(\Phi\in[0,M]\)이면

$$
\langle e^{-\Phi}\rangle
\le
e^{-\langle\Phi\rangle}
\exp\!\Big(\frac{e^{M}}{2}\operatorname{Var}(\Phi)\Big).
$$

증명:

(i) \(x\mapsto e^{-x}\)는 convex이므로 Jensen 그대로다.

(ii) \(Y=\Phi-\langle\Phi\rangle\in[-M,M]\)으로 둔다. Taylor-Lagrange로 \(|y|\le M\)에서

$$
e^{-y}=1-y+\frac{y^2}2e^{-\xi_y},
\qquad
|\xi_y|\le|y|\le M
$$

이므로 \(e^{-y}\le1-y+\frac{y^2}2e^{M}\)이다. 기대값을 취하면 \(\langle Y\rangle=0\)이라

$$
\langle e^{-Y}\rangle
\le
1+\frac{e^M}2\operatorname{Var}(\Phi)
\le
\exp\!\Big(\frac{e^M}2\operatorname{Var}(\Phi)\Big).
$$

양변에 \(e^{-\langle\Phi\rangle}\)를 곱하면 끝.

해석:

> bootstrap의 평균장 단계 \(\langle e^{-\Phi}\rangle\approx e^{-\langle\Phi\rangle}=e^{-\sigma D_{\mathrm{eff}}}\)는 방향이 정해진 근사다. \(e^{-\sigma D_{\mathrm{eff}}}\)는 생존확률의 **하한**이고, 식별 \(\varepsilon^2=e^{-\sigma D_{\mathrm{eff}}}\)가 \(\langle e^{-\Phi}\rangle\)와 1% 안에서 일치하려면 대략 \(\operatorname{Var}(\Phi)\lesssim0.02\,e^{-M}\cdot2\) 수준의 요동 통제가 필요하다. 이것은 경로적분.md가 "누적량 고차항을 유효계수에 흡수"라고 적어 둔 자리에 정확한 반증 가능 조건을 주는 것이다. \(\operatorname{Var}(\Phi)\)의 실제 크기는 `Open/Experiment`다.

## 6. Threshold 선택의 지위

| 선택 | 내용 | 지위 |
|---|---|---|
| \(u_{\mathrm{th}}\)의 절대값 | continuum에서 정의 불능, finite-\(N\)에서 \(N/2+z_q\sqrt{N/2}\) | 정리 2.1, 3.2로 형태는 `Exact under assumptions` |
| \(z_q\)의 값 | \(q=\varepsilon^2\)를 넣어야 \(z_q\approx-1.66\) | bootstrap 출력을 입력으로 쓰는 `Selection`. threshold가 \(\varepsilon^2\)를 독립적으로 예측하는 것이 아님 |
| "접힘 통과"의 물리적 의미 | 평균 작용 대비 \(-1.66\sigma\) 요동 조건 | `Bridge/Open`. 왜 이 분위수인지의 물리는 닫히지 않음 |

중요한 순환 경고:

> 정리 3.2는 threshold 독법이 bootstrap 분율을 **재생산**할 수 있음을 보이지만, threshold 자체가 \(q\)를 결정하지는 않는다. \(z_q\)는 \(q\)에서 역산된다. 따라서 등분배 정리(3.2.1)는 \(\Omega_b\)의 독립 유도가 아니라 두 분율 독법(경로/에너지)의 일치 진술로만 읽어야 한다. 이는 마스터 문서들이 A3b를 `Bridge`로 분류한 것과 정합한다.

## 7. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| continuum hard constraint 정의 불능 | 정리 2.1 |
| finite-\(N\) 조건화 존재 | 정리 3.1 |
| 고정 분율 \(\Leftrightarrow\) \(N\)-dependent threshold | 정리 3.2 |
| 등분배 수렴율 \(O(N^{-1/2})\) 보정 | 정리 3.3 |
| 조건화 recovery 보존 | 정리 4.1 |
| \(S_{\mathrm{th}}>m\)에서 manifest 불변 | 정리 4.2 |
| tilt = threshold 분율의 \(e^{-t}\)-평균 | 정리 5.1 |
| atomless 단일 threshold 대응 | 정리 5.2 |
| mean-field 하한과 \(\operatorname{Var}\) 오차 통제 | 정리 5.3 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| \(\operatorname{Var}(\Phi)\)의 실제 크기 | 모형 또는 수치로 추정. 정리 5.3의 반증 조건 가동 |
| \(z_q\approx-1.66\)의 물리 | 왜 이 분위수인지. 현재는 \(q\)에서 역산되는 `Selection` |
| \(H^1\)-supported continuum prior | 더 매끄러운 Gaussian으로 continuum hard constraint를 살릴 수 있는지 |
| finite-\(N\)과 05h mesh package의 합류 | 모드 절단 \(N\)과 mesh \(N\)의 대응 |

## 8. 결론

$$
\boxed{
\text{supp:hard}
=
\text{finite-}N\ \text{전용}.
\quad
u_{\mathrm{th}}(N)=\frac N2+z_q\sqrt{\frac N2},
\quad
S_{\mathrm{th}}>m
\Rightarrow
\text{manifest 불변}.
}
$$

$$
\boxed{
\langle e^{-\Phi}\rangle
=
\int_0^\infty e^{-t}\mu(\Phi\le t)dt
\ \ge\
e^{-\langle\Phi\rangle},
\qquad
\text{오차}\le\frac{e^M}2\operatorname{Var}(\Phi).
}
$$

분율 체계의 두 독법(tilt, threshold)은 layer-cake로 정확히 연결되고, 둘 다 manifest 선택과는 분리된 finite-\(\beta\)/finite-\(N\) 층의 수학이다. 남은 반증 가능 지점은 \(\operatorname{Var}(\Phi)\)이고, 남은 `Selection`은 분위수 \(z_q\)의 물리적 근거다.
