# 05l. CE Uncertainty Floor Package

## 0. 목표

[05k_CE_hard_constraint.md](05k_CE_hard_constraint.md)는 두 가지를 `Open`으로 남겼다.

1. mean-field 오차를 통제하는 \(\operatorname{Var}(\Phi)\)의 실제 크기.
2. threshold 요동 창 \(u_{\mathrm{th}}=\langle S_E\rangle+z_q\sqrt{\operatorname{Var}(S_E)}\)에서 요동 scale의 근거.

이 문서는 불확정성 원리(Kennard 부등식)를 수학 도구로 들여와 이 두 자리의 **scale을 하한으로 고정**한다.

핵심 결론:

> 불확정성은 모드당 작용 요동이 \(\hbar\) 아래로 내려갈 수 없게 만드는 **floor**다. 이 floor 위에서 (i) 05i의 kinetic 장애물(\(S_E=\infty\) a.s.)은 병리가 아니라 불확정성의 경로공간 그림자임이 드러나고, (ii) 05k threshold 창의 폭 \(\hbar\sqrt{N/2}\)는 자유 선택이 아니라 강제된 scale이 되며, (iii) mode 분해 가정 아래 \(\operatorname{Var}(\Phi)=\Theta(1/N_{\mathrm{eff}})\)로 mean-field 오차가 조건부로 닫힌다. 단, 분위수 \(z_q\)의 값은 불확정성에서 나오지 않는다. 불확정성은 요동의 크기를 고정할 뿐 어느 분위에서 자르는지는 결정하지 못한다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| Kennard 부등식 | `Exact` (domain 조건 import) | 정리 1.1 |
| zero-point 하한 \(\langle H\rangle\ge\hbar\omega/2\) | `Exact` | 정리 2.1 |
| 모드당 Euclidean 작용 모멘트 \((\hbar/2,\hbar^2/2)\) | `Exact` | 정리 2.2 |
| 05i 장애물의 불확정성 독법 | 수학 `Exact`, 독법 `Bridge` | 정리 3.1 |
| threshold 창 scale \(\hbar\sqrt{N/2}\) 고정 | `Exact under assumptions` | 4절 |
| \(\operatorname{Var}(\Phi)=\Theta(1/N_{\mathrm{eff}})\)와 mean-field 수렴 | `Exact under assumptions` | 정리 5.1 |
| bootstrap \(\Phi\)의 mode 분해 가능성 | `Bridge/Open` | 6절 |
| 분위수 \(z_q\)의 값 | `Selection/Open` 유지 | 불확정성으로 닫히지 않음 |

## 1. Kennard 부등식

### 정리 1.1

Hilbert space의 self-adjoint 연산자 \(X,P\)가 공통 dense domain에서 \([X,P]=i\hbar\)를 만족하고, 단위 벡터 \(\psi\)가 그 domain에 있다고 하자(정의역 조건은 표준 import). 그러면

$$
\operatorname{Var}_\psi(X)\cdot\operatorname{Var}_\psi(P)
\ge
\frac{\hbar^2}4.
$$

증명:

평균을 빼서 \(\langle X\rangle=\langle P\rangle=0\)으로 둘 수 있다(\(X\to X-\langle X\rangle\)은 교환자를 바꾸지 않는다). 내적 \(\langle X\psi,P\psi\rangle\)의 허수부는

$$
2\,\mathrm{Im}\langle X\psi,P\psi\rangle
=
\frac1i\big(\langle X\psi,P\psi\rangle-\langle P\psi,X\psi\rangle\big)
=
\frac1i\langle\psi,[X,P]\psi\rangle
=
\hbar.
$$

Cauchy-Schwarz로

$$
\frac\hbar2
=
|\mathrm{Im}\langle X\psi,P\psi\rangle|
\le
|\langle X\psi,P\psi\rangle|
\le
\|X\psi\|\,\|P\psi\|
=
\sqrt{\operatorname{Var}(X)\operatorname{Var}(P)}.
$$

끝.

## 2. Zero-point floor와 Gaussian 포화

### 정리 2.1: zero-point 하한

단위 질량 조화 모드 \(H=\frac12(P^2+\omega^2X^2)\)에 대해 모든 상태에서

$$
\langle H\rangle\ge\frac{\hbar\omega}2.
$$

증명:

\(\langle X^2\rangle\ge\operatorname{Var}(X)\), \(\langle P^2\rangle\ge\operatorname{Var}(P)\)이고 AM-GM으로

$$
\langle H\rangle
=
\frac{\langle P^2\rangle+\omega^2\langle X^2\rangle}2
\ge
\omega\sqrt{\langle X^2\rangle\langle P^2\rangle}
\ge
\omega\sqrt{\operatorname{Var}(X)\operatorname{Var}(P)}
\ge
\frac{\hbar\omega}2.
$$

마지막 부등식이 정리 1.1이다. 끝.

등호는 centered minimum-uncertainty Gaussian(바닥상태)에서만 성립한다. 즉 **모드 하나의 에너지/작용 scale \(\hbar\)는 어떤 상태 선택으로도 제거할 수 없는 floor**다.

### 정리 2.2: 모드당 Euclidean 작용의 정확한 모멘트

05k의 자유장 모델에서 모드 \(\phi_k\)의 prior는 \(e^{-S_k/\hbar}\) Gibbs weight, \(S_k=\omega_k\phi_k^2/2\)다. 그러면 \(\phi_k\sim\mathcal N(0,\hbar/\omega_k)\)이고

$$
\langle S_k\rangle=\frac\hbar2,
\qquad
\operatorname{Var}(S_k)=\frac{\hbar^2}2.
$$

증명:

\(z:=\phi_k\sqrt{\omega_k/\hbar}\sim\mathcal N(0,1)\)로 치환하면 \(S_k/\hbar=z^2/2\)다. \(\langle z^2\rangle=1\), \(\operatorname{Var}(z^2)=2\)이므로

$$
\langle S_k\rangle=\frac\hbar2\langle z^2\rangle=\frac\hbar2,
\qquad
\operatorname{Var}(S_k)=\frac{\hbar^2}4\operatorname{Var}(z^2)=\frac{\hbar^2}2.
$$

끝.

해석:

> 정리 2.1은 모드당 scale \(\hbar\)가 하한임을 말하고, 정리 2.2는 path-integral prior가 그 floor scale에서 정확히 작동함을 말한다. 모드당 평균도 분산도 모두 \(\hbar\) 단위로 고정되어 있고, 이것이 05k의 \(u=S_E/\hbar\sim\mathrm{Gamma}(N/2,1)\)에서 mean \(=\) Var \(=N/2\)가 나온 근원이다. **요동의 크기는 모델의 자유 파라미터가 아니다.**

## 3. 05i 장애물의 불확정성 독법

### 정리 3.1: 유한 kinetic action은 무요동을 강제한다

(i) \(\gamma\in H^1\)이면 quadratic variation은 0이다(05i 정리 2.1의 (i)).

(ii) 05i의 physical prior \(\mu^\hbar\)(scaled bridge)의 path는 dyadic partition을 따라 a.s.

$$
\sum_k|\gamma(t_{k+1})-\gamma(t_k)|^2
\to
\hbar\,d
$$

를 만족한다. 즉 시간당 quadratic variation이 정확히 \(\hbar\)다.

증명:

(i)은 05i에서 증명했다. (ii)는 \(\sqrt\hbar B^0\)의 quadratic variation이 \(\hbar\cdot t\)인 것으로, Brownian quadratic variation의 scaling이다(외부 import). 끝.

독법 (`Bridge`):

> 짧은 시간 \(\Delta t\)에서 \(|\Delta\gamma|^2\approx\hbar\Delta t\), 즉 \(\Delta x\sim\sqrt{\hbar\Delta t}\)는 위치-운동량 요동 \(\Delta x\cdot\Delta p\sim\hbar\)의 경로공간 형태다. 유한 kinetic action 경로는 quadratic variation 0, 즉 모든 시간 scale에서 요동이 사라진 경로인데, 이는 불확정성 floor 아래로 내려간 경로라 prior가 질량을 줄 수 없다. **05i 정리 2.1은 측도론적 병리가 아니라 Kennard floor의 그림자다.** 동시에 A2''' prior(scaled bridge)가 시간당 정확히 \(\hbar\)의 요동을 갖는다는 것은, 그 prior 선택이 불확정성과 정합하는 유일한 scale임을 뜻한다.

## 4. Threshold 창의 scale 고정

05k 정리 3.2는 고정 분율 \(q\)의 threshold가

$$
u_{\mathrm{th}}(N)=\frac N2+z_q\sqrt{\frac N2}+o(\sqrt N)
$$

임을 보였다. 작용 단위로 돌리면 창의 중심과 폭은

$$
S_{\mathrm{th}}
=
\underbrace{\frac{N\hbar}2}_{\text{zero-point 합}}
+
z_q
\underbrace{\hbar\sqrt{\frac N2}}_{\text{불확정성 요동 폭}}
+o(\sqrt N\,\hbar).
$$

정리 2.1, 2.2에 의해 두 밑줄 항은 모두 강제된 scale이다.

| 양 | 값 | 지위 |
|---|---|---|
| 창의 중심 \(N\hbar/2\) | 모드당 zero-point \(\hbar/2\)의 합 | `Exact` (floor 포화) |
| 창의 폭 \(\hbar\sqrt{N/2}\) | 모드당 \(\operatorname{Var}=\hbar^2/2\)의 CLT 합산 | `Exact under assumptions` |
| 분위수 \(z_q\) | \(q\)에서 역산 | `Selection/Open` 유지 |

해석:

> 05k 시점에서는 threshold의 "scale 선택"과 "분위수 선택"이 모두 열려 있었다. 이제 scale은 닫혔다. 남은 자유도는 무차원 분위수 \(z_q\) 하나뿐이다. 불확정성은 자(尺)를 고정했고, 어디서 자를지는 여전히 `Selection`이다.

## 5. \(\operatorname{Var}(\Phi)\)의 조건부 통제

### 정리 5.1: intensive mode 분해 아래 mean-field 수렴

\(\Phi\)가 독립 모드 기여의 합

$$
\Phi=\sum_{k=1}^{N_{\mathrm{eff}}}Y_k,
\qquad
Y_k\ \text{독립},
\quad
0\le Y_k\le\frac C{N_{\mathrm{eff}}},
\quad
\operatorname{Var}(Y_k)=\frac{v_k}{N_{\mathrm{eff}}^2}
$$

로 쓰인다고 하자(\(\langle\Phi\rangle=O(1)\)인 intensive 정규화, \(v_k\le v_{\max}\)). 그러면

$$
\operatorname{Var}(\Phi)
=
\sum_k\frac{v_k}{N_{\mathrm{eff}}^2}
\le
\frac{v_{\max}}{N_{\mathrm{eff}}},
$$

이고 05k 정리 5.3과 결합하면(\(M=C\) 유계)

$$
1
\le
\frac{\langle e^{-\Phi}\rangle}{e^{-\langle\Phi\rangle}}
\le
\exp\!\Big(\frac{e^{C}v_{\max}}{2N_{\mathrm{eff}}}\Big)
=
1+O\!\big(N_{\mathrm{eff}}^{-1}\big).
$$

또한 모드 분포가 비퇴화(\(v_k\ge v_{\min}>0\), 정리 2.2의 Gaussian 모드면 \(v_{\min}\) 자동)라면

$$
\operatorname{Var}(\Phi)\ge\frac{v_{\min}}{N_{\mathrm{eff}}}>0.
$$

증명:

독립성에서 분산은 합산된다. 상한은 05k 정리 5.3 (ii)에 \(M=C\), \(\operatorname{Var}(\Phi)\le v_{\max}/N_{\mathrm{eff}}\)를 대입한 것이고, 하한은 05k 정리 5.3 (i)의 Jensen이다. floor는 분산 합산에서 즉시 나온다. 끝.

해석:

> mean-field 단계 \(\langle e^{-\Phi}\rangle\approx e^{-\langle\Phi\rangle}\)는 mode 분해 가정 아래 \(N_{\mathrm{eff}}\to\infty\)에서 **점근적으로 정확**하고, 유한 \(N_{\mathrm{eff}}\)에서는 불확정성 floor 때문에 **절대 정확해질 수 없다**. bootstrap 방정식
>
> $$
> \varepsilon^2=e^{-(1-\varepsilon^2)D_{\mathrm{eff}}}
> $$
>
> 은 따라서 \(O(1/N_{\mathrm{eff}})\) 보정항을 갖는 점근식으로 읽어야 하며, 보정의 부호는 Jensen에 의해 정해져 있다: 참값 \(\langle e^{-\Phi}\rangle\)는 우변보다 크고, 따라서 mean-field \(\varepsilon^2\)는 생존분율의 **하한 추정**이다.

## 6. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| Kennard 부등식 | 정리 1.1 |
| zero-point floor | 정리 2.1 |
| 모드당 작용 모멘트 \((\hbar/2,\hbar^2/2)\) | 정리 2.2 |
| 05i 장애물 = 불확정성 floor의 그림자 | 정리 3.1 + 독법 |
| threshold 창의 중심/폭 scale | 4절 |
| mode 분해 아래 \(\operatorname{Var}(\Phi)\) 상하한과 mean-field 수렴 | 정리 5.1 |
| mean-field \(\varepsilon^2\)가 하한 추정이라는 부호 확정 | 정리 5.1 + 05k 정리 5.3 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| bootstrap \(\Phi\)의 mode 분해 실재성 | \(\Phi\)가 실제로 독립 intensive 합인지. CE 문서의 \(\langle\Phi\rangle=\sigma D_{\mathrm{eff}}\) 구성 감사 |
| \(N_{\mathrm{eff}}\)의 값 | 보정 \(O(1/N_{\mathrm{eff}})\)의 실제 크기. 수치 회귀 대상 |
| 분위수 \(z_q\approx-1.66\)의 물리 | 불확정성으로 닫히지 않음. 독립 원리 필요 |
| domain 조건의 완전한 처리 | 정리 1.1의 self-adjointness/domain은 표준 import로 둠 |

경고:

> 불확정성 원리로 \(z_q\)나 \(\varepsilon^2\) 값 자체를 유도했다고 주장하면 안 된다. 이 package가 닫은 것은 요동의 **scale**(자)이지 분위수(눈금 위의 위치)가 아니다. 후자를 불확정성에서 끌어내는 시도는 [08_수학도구_진행지도.md](08_수학도구_진행지도.md)의 "증명하면 안 되는 자리"에 추가한다.

## 7. 결론

$$
\boxed{
\Delta X\,\Delta P\ge\frac\hbar2
\;\Longrightarrow\;
\langle S_k\rangle=\frac\hbar2,\ \operatorname{Var}(S_k)=\frac{\hbar^2}2
\;\Longrightarrow\;
S_{\mathrm{th}}=\frac{N\hbar}2+z_q\,\hbar\sqrt{\frac N2}
}
$$

$$
\boxed{
\text{mode 분해 가정 아래}\quad
1\le\frac{\langle e^{-\Phi}\rangle}{e^{-\langle\Phi\rangle}}
\le1+O(N_{\mathrm{eff}}^{-1}),
\qquad
\operatorname{Var}(\Phi)\ge\frac{v_{\min}}{N_{\mathrm{eff}}}>0.
}
$$

불확정성은 fraction layer의 자를 고정하고 mean-field의 오차 부호와 크기를 통제한다. 남은 `Selection`은 \(z_q\), 남은 `Bridge`는 \(\Phi\)의 mode 분해 실재성이다.
