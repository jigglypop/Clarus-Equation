# 05m. bootstrap Φ의 mode 분해 실재성 감사: N_eff 추정과 오차 예산

## 0. 위치와 목표

05l 정리 5.1은 mean-field 수렴 \(\langle e^{-\Phi}\rangle/e^{-\langle\Phi\rangle}=1+O(1/N_{\mathrm{eff}})\)을 **mode 분해 가정** 아래에서 증명했다. 이 문서는 그 가정 자체를 감사한다. 질문은 세 가지다.

1. CE 문서에서 \(\langle\Phi\rangle=\sigma D_{\mathrm{eff}}\)는 유도인가 선언인가. 분해되는 것은 평균인가 요동인가.
2. bootstrap \(\Phi\)는 실제로 독립 intensive 모드 합으로 읽을 수 있는가. 그렇다면 \(N_{\mathrm{eff}}\)는 얼마인가.
3. \(\Omega_b\)의 관측 오차 예산은 \(N_{\mathrm{eff}}\)와 모드 상관에 어떤 정량 하한/상한을 강제하는가.

핵심 결론:

> CE 문서의 차원 분해(A1)는 **평균의 분해**이지 요동 통계의 분해가 아니다. \(D_{\mathrm{eff}}\approx3.18\)은 \(\langle\Phi\rangle\)의 채널 수이고, mean-field 오차를 통제하는 \(N_{\mathrm{eff}}\)는 별도의 미시 모드 수다. 둘을 동일시하는 독법(\(N_{\mathrm{eff}}=D_{\mathrm{eff}}\))은 \(\Omega_b\) 관측과 약 \(135\sigma\)로 충돌하여 **배제**된다. 역으로, 주장된 \(0.05\sigma\) 일치가 성립하려면 Gaussian-mode benchmark에서 \(N_{\mathrm{eff}}\gtrsim4.5\times10^2\), 모드 평균 상관 \(\bar\rho\lesssim2.2\times10^{-3}\)이 **유도된 필요조건**으로 따라 나온다. 이것은 mean-field 단계의 반증 가능 조건이다.

현재 판정:

| 항목 | 지위 | 근거 |
|---|---|---|
| A1 차원 분해 = 평균의 분해 | `Audit/Exact` | 1절, U1 |
| \(\Phi\)의 요동 통계 선언 | 문서군에 **부재**, `Open` | 1절 |
| \(N_{\mathrm{eff}}=D_{\mathrm{eff}}\) 독법(R2) 배제 | `Exact under assumptions` | 정리 3.3 |
| \(N_{\mathrm{eff}}\gtrsim4.5\times10^2\) 필요조건 | `Exact under assumptions` | 정리 3.2 |
| 모드 상관 허용치 \(\bar\rho\lesssim2.2\times10^{-3}\) | `Exact under assumptions` | 정리 4.1 |
| 상관의 실제 크기, \(N_{\mathrm{eff}}\)의 물리값 | `Open/Experiment` | 4, 7절 |

표기: \(\varepsilon^2=0.04865\), \(\sigma=1-\varepsilon^2=0.95135\), \(D_{\mathrm{eff}}=3.17776\), \(m\equiv\langle\Phi\rangle=\sigma D_{\mathrm{eff}}=3.0232\). 관측은 Planck \(\Omega_b=0.0486\pm0.0010\), 상대 \(1\sigma\) 폭 \(\delta_{\mathrm{rel}}=0.0206\).

## 1. 감사 대상 수집

\(\langle\Phi\rangle=\sigma D_{\mathrm{eff}}\)와 mean-field가 등장하는 자리를 모은다.

| # | 위치 | 내용 | 감사 소견 |
|---|---|---|---|
| U1 | 경로적분.md 3.2 형식적 유도 | (A1) "경로적분 \(D_{\mathrm{eff}}\) 차원 곱적 분해", 3단계 "\(\langle\Phi\rangle=\sigma D_{\mathrm{eff}}\) (연장적)" | A1이 분해하는 것은 \(\langle\Phi\rangle\)뿐이다. \(\Phi\)의 분포·분산·모드 수는 선언되지 않음 |
| U2 | 1_강의/A_연역적_유도.md "평균장(leading cumulant) 근사" | \(\langle e^{-\Phi}\rangle=\exp(-\langle\Phi\rangle+\tfrac12\operatorname{Var}(\Phi)-\cdots)\approx e^{-\langle\Phi\rangle}\), "분산/고차 누적량 항은 유효 계수 보정으로 흡수" | 누적량 전개는 명시했으나 "흡수 가능"의 정량 조건이 없음. 본 문서 3절이 그 조건이다 |
| U3 | 경로적분.md 3.2.2 | \(\Phi=\delta^2S/\delta\gamma^2\), \(D_{\mathrm{eff}}\)는 "헤시안 대각합의 독립 접힘 채널 수" | 채널 독립성은 평균 수준 가산성 주장. 요동 독립성과 구분 안 됨 |
| U4 | 경로적분.md 3.2 끝, 9절 | "각 차원이 독립적으로 \(e^{-1}\) 확률로 접히지 않음", \(e^{-3}\) 직관 | 차원당 생존확률 독법. 확률 곱이 성립하려면 차원 간 요동 독립이 추가로 필요 |
| U5 | 경로적분.md 정리 3.2.1 | 자유장 \(N\)-모드 Gamma 모델, \(N\to\infty\), 상호작용 \(O(g^2)\approx2\%\) | 모드 독립성이 자유장 한계에서 **유도됨**. 단 대상이 \(\Phi\)가 아니라 총 작용 \(S_E\)다(5절) |

감사 결과 첫 판정:

> A1의 "곱적 분해"는 \(\langle\Phi\rangle=\sum_{d=1}^{D_{\mathrm{eff}}}\sigma\)라는 **평균의 가산 분해**다. 05l 정리 5.1이 요구하는 것, 즉 \(\Phi=\sum_k Y_k\)에서 \(Y_k\)들의 **독립성과 intensive 정규화**는 어느 문서에도 선언되어 있지 않다. mean-field의 오차를 정하는 것은 평균이 아니라 \(\operatorname{Var}(\Phi)\)이므로, 이 빈자리가 bootstrap 정밀도 주장의 마지막 미선언 가정이다.

## 2. Φ의 세 가지 독법

문서군과 정합 가능한 독법을 모두 나열한다.

**R1 (결정론적).** \(\Phi\)는 상수, \(\operatorname{Var}(\Phi)=0\). mean-field는 자명하게 정확하다.

판정: 배제. 05l 정리 5.1의 불확정성 floor \(\operatorname{Var}(\Phi)\ge v_{\min}/N_{\mathrm{eff}}>0\)와 모순이고, U5의 요동하는 Gamma 모델과도 layer-cake 대응(05k 정리 5.1) 아래 비정합이다. \(\langle e^{-\Phi}\rangle\) 표기 자체가 공허해진다. 고전 극한의 관용 표기로만 허용된다.

**R2 (차원 = 모드).** \(\Phi=\sum_{d=1}^{D_{\mathrm{eff}}}\Phi_d\), 각 차원 기여가 \(O(1)\) 상대 요동을 갖는 독립 확률변수. 즉 \(N_{\mathrm{eff}}=D_{\mathrm{eff}}\approx3.18\). U4의 "각 차원이 독립적으로" 문구를 요동 수준까지 밀어붙인 독법이다.

판정: 정리 3.3에서 관측과 \(\sim135\sigma\)로 충돌하여 배제.

**R3 (미시 모드 집계).** \(\Phi\)는 \(N_{\mathrm{eff}}\gg1\)개 미시 장 모드의 intensive 집계이고, \(D_{\mathrm{eff}}\)는 그 평균이 분해되는 채널 수일 뿐이다. \(\operatorname{Var}(\Phi)=\Theta(1/N_{\mathrm{eff}})\).

판정: 유일하게 생존하는 독법. U5의 자유장 모드 그림과도 정합한다. 단 \(N_{\mathrm{eff}}\)의 값은 문서군에 없으므로, 3절에서 관측이 강제하는 하한을 유도한다.

## 3. 오차 예산이 강제하는 N_eff 하한

**정리 3.1 (분산 예산).** 식별 \(\varepsilon^2=e^{-\langle\Phi\rangle}\)이 참 생존분율 \(\langle e^{-\Phi}\rangle\)과 상대오차 \(\delta_{\mathrm{rel}}\) 이내로 일치하려면, 누적량 전개의 2차 절단에서

$$
\operatorname{Var}(\Phi)\le2\ln(1+\delta_{\mathrm{rel}}).
$$

Planck \(1\sigma\) 예산 \(\delta_{\mathrm{rel}}=0.0206\)을 넣으면 \(\operatorname{Var}(\Phi)\le0.0407\)이다.

증명: 05k 정리 5.3의 Jensen 하한에 의해 비율 \(\langle e^{-\Phi}\rangle/e^{-\langle\Phi\rangle}\ge1\)이고, 2차 누적량 절단에서 비율은 \(e^{\operatorname{Var}(\Phi)/2}\)이다. \(e^{\operatorname{Var}/2}\le1+\delta_{\mathrm{rel}}\)을 풀면 된다. 끝.

해석:

> U2가 "분산 항을 유효 계수에 흡수"라고 적은 자리의 정량 내용이다. 흡수가 허용되는 것은 \(\operatorname{Var}(\Phi)\le0.04\)일 때뿐이다. 그보다 크면 보정은 \(D_{\mathrm{eff}}\)의 재정의로만 흡수될 수 있는데, 그 경우 \(D_{\mathrm{eff}}=3+\delta\)의 제1원리 주장(전자약 보정에서 유도)이 무너진다. 즉 **누적량 흡수와 \(D_{\mathrm{eff}}\) 제1원리성은 동시에 공짜로 가질 수 없고**, 둘의 양립 조건이 정확히 정리 3.1이다.

**정리 3.2 (Gaussian-mode benchmark의 N_eff 하한).** \(\Phi\)가 \(N_{\mathrm{eff}}\)개 iid Gaussian-mode 작용(05l 정리 2.2: 모드당 \(u_k\sim\mathrm{Gamma}(1/2)\), 상대분산 \(r=\operatorname{Var}/\text{mean}^2=2\))의 intensive 합으로 평균 \(m\)에 맞춰 정규화되면, \(\Phi\sim\mathrm{Gamma}(k,\theta)\), \(k=N_{\mathrm{eff}}/2\), \(k\theta=m\)이고 비율은 닫힌 형태

$$
\frac{\langle e^{-\Phi}\rangle}{e^{-\langle\Phi\rangle}}
=\exp\!\Big(m-k\ln\big(1+\tfrac mk\big)\Big)
=1+\frac{m^2}{N_{\mathrm{eff}}}+O(N_{\mathrm{eff}}^{-2})
$$

를 갖는다. 정리 3.1의 예산과 결합하면

$$
N_{\mathrm{eff}}\ \ge\ \frac{r\,m^2}{2\ln(1+\delta_{\mathrm{rel}})}\ \approx\ \frac{2\times9.140}{0.0407}\ \approx\ 4.5\times10^2.
$$

닫힌 형태로 정확히 풀면 \(N_{\mathrm{eff}}\ge445\)다.

증명: \(\mathrm{Gamma}(k,\theta)\)의 Laplace 변환 \(\langle e^{-\Phi}\rangle=(1+\theta)^{-k}\)에 \(\theta=m/k\)를 대입하고 \(e^{-m}\)으로 나눈다. 전개 \(m-k\ln(1+m/k)=m^2/2k-m^3/3k^2+\cdots\)에 \(k=N_{\mathrm{eff}}/2\)를 넣으면 선도항 \(m^2/N_{\mathrm{eff}}\)다. 끝.

**정리 3.3 (R2 배제).** \(N_{\mathrm{eff}}=D_{\mathrm{eff}}=3.17776\)이면(\(k=1.589\)) 정리 3.2의 닫힌 형태가

$$
\frac{\langle e^{-\Phi}\rangle}{e^{-\langle\Phi\rangle}}=e^{1.330}=3.78
$$

을 준다. 참 생존분율은 \(3.78\times0.04865=0.184\)가 되어 관측 \(0.0486\pm0.0010\)에서 약 \(135\sigma\) 벗어난다.

증명: 수치 대입. \(m/k=1.9027\), \(k\ln(1+m/k)=1.6932\), \(m-1.6932=1.3300\). 끝.

해석:

> "3차원 각각이 \(O(1)\)로 요동하는 독립 접힘 기여" 독법은 관측이 직접 배제한다. \(0.05\sigma\) 일치 주장은 그 자체로 \(\Phi\)의 요동이 미시 모드 수백 개 이상에 분산되어 있어야 함을 함의한다. 이것은 CE에 불리한 결과가 아니라, **문서가 선언하지 않은 가정을 관측이 대신 고정해 주는** 결과다. 다만 이제 그 가정(R3, \(N_{\mathrm{eff}}\gtrsim450\))을 명시적으로 적어야 한다.

## 4. 모드 상관 허용치

독립성 가정도 같은 예산의 감사를 받는다.

**정리 4.1 (상관 상한).** \(\Phi=\sum_{k=1}^{N}Y_k\)가 동일 분포(상대분산 \(r=2\)), 평균 \(m/N\), 쌍별 상관 \(\bar\rho\ge0\)의 equicorrelated 모드 합이면

$$
\operatorname{Var}(\Phi)=\frac{r\,m^2}{N}\big(1+(N-1)\bar\rho\big)
\ \xrightarrow{N\to\infty}\ r\,m^2\,\bar\rho,
$$

이고 정리 3.1의 예산은 모드 수와 무관한 상한

$$
\bar\rho\ \le\ \frac{2\ln(1+\delta_{\mathrm{rel}})}{r\,m^2}\ \approx\ 2.2\times10^{-3}
$$

을 강제한다. 동치로, 유효 독립 모드 수 \(N^{\mathrm{eff}}=N/(1+(N-1)\bar\rho)\)는 \(1/\bar\rho\)에서 포화하므로, \(N^{\mathrm{eff}}\ge445\)와 \(\bar\rho\le2.2\times10^{-3}\)는 같은 조건의 두 표현이다.

증명: 분산의 쌍별 전개와 극한. 끝.

긴장 1건 (Open):

> U5는 상호작용 보정을 \(O(g^2)=O(\alpha_s/2\pi)\approx1.9\%\)로 인용한다. 만약 이 보정이 모드 간 쌍별 상관 \(\bar\rho\sim0.019\)로 들어온다면 \(\operatorname{Var}(\Phi)\approx2m^2\bar\rho\approx0.35\), 비율 \(e^{0.17}\approx1.19\), 즉 \(\Omega_b\)가 \(+19\%\) (\(\sim9\sigma\)) 밀린다. 허용치 \(2.2\times10^{-3}\)보다 한 자릿수 크다. 다만 \(O(g^2)\)는 분율 자체의 보정 크기이지 쌍별 상관 계수가 아니며, 상관의 부호와 구조(연결 상관의 거리 감쇠)는 별도 계산 대상이다. 이 간극은 `Open/Experiment`로 둔다. 감사가 주는 것은 명확한 반증선이다: **접힘 기여의 평균 쌍별 상관이 \(0.2\%\)를 넘으면 mean-field 단계가 \(\Omega_b\) 예산을 깬다.**

## 5. 두 경로의 대상 불일치

tilt 경로(U1)와 threshold 경로(U5)는 같은 \(P_{\mathrm{survive}}\)를 겨냥하지만 지수의 대상이 다르다.

| | tilt 경로 (3.2) | threshold 경로 (3.2.1) |
|---|---|---|
| 확률 형식 | \(\langle e^{-\Phi}\rangle\) | \(\mu(S_E<S_{\mathrm{th}})\) |
| 지수 대상 | 접힘 지수 \(\Phi\), \(\langle\Phi\rangle=m\approx3.02\) | 총 작용 \(u=S_E/\hbar\), \(\langle u\rangle=N/2\) |
| 분산 scaling | \(\Theta(1/N_{\mathrm{eff}})\) (intensive, R3) | \(N/2\) (연장적) |
| mean-field 가능 여부 | 가능 (\(\operatorname{Var}\to0\)) | 불가능: \(\langle e^{-u}\rangle=2^{-N/2}\)인데 \(e^{-\langle u\rangle}=e^{-N/2}\), 비율 \((e/2)^{N/2}\to\infty\) |

**판정.** \(\Phi\)는 총 작용이 아니다. \(\Phi\)에 mean-field를 적용하는 순간 \(\Phi\)는 intensive 정규화된 집계량이어야 하며(R3), 총 작용 \(S_E\)에는 threshold 독법(05k)만 유효하다. 두 경로가 같은 \(\varepsilon^2\)를 주는 것은 우연이 아니라 layer-cake 대응(05k 정리 5.1)과 창 scale 고정(05l 4절)의 결과지만, 문서군은 \(\Phi\)와 \(S_E\)를 표기상 구분하지 않는 곳이 있다. 09 용어사전에 구분 항목이 필요하다.

## 6. 권장 규약

| 규약 | 내용 |
|---|---|
| \(\Phi\) 선언 | \(\Phi\)는 intensive 접힘 지수(미시 모드 집계, R3)로 선언. 총 작용 \(S_E\)와 기호·역할 구분 |
| A1 주석 | "곱적 분해"에 "평균의 채널 분해이며 요동 통계의 분해가 아님"을 명기 |
| \(N_{\mathrm{eff}}\) 명기 | mean-field를 쓰는 모든 자리에 \(N_{\mathrm{eff}}\gtrsim4.5\times10^2\) 가정을 명기. 위반 시 \(D_{\mathrm{eff}}\) 제1원리성과 양자택일 |
| 상관 반증선 | \(\bar\rho\le2.2\times10^{-3}\)을 반증 가능 조건으로 등록. \(O(g^2)\) 상관 구조 계산은 `Open` |

## 7. 닫힌 것과 남은 것

닫힌 것:

- A1 차원 분해가 평균의 분해임을 확정했다(U1-U4 감사).
- R1(결정론)과 R2(\(N_{\mathrm{eff}}=D_{\mathrm{eff}}\)) 독법을 각각 불확정성 floor와 관측 \(135\sigma\)로 배제했다.
- \(0.05\sigma\) 일치 주장으로부터 \(N_{\mathrm{eff}}\ge445\) (Gaussian-mode benchmark), \(\operatorname{Var}(\Phi)\le0.041\), \(\bar\rho\le2.2\times10^{-3}\)을 유도된 필요조건으로 닫았다.
- tilt의 \(\Phi\)와 threshold의 \(S_E\)가 다른 scaling의 다른 대상임을 분리했다.

남은 것:

| 항목 | 내용 |
|---|---|
| \(N_{\mathrm{eff}}\)의 물리값 | 미시 모드 수의 제1원리 추정(컷오프, 부피). 하한 445는 관측 유도일 뿐 |
| 상호작용 상관 구조 | \(O(g^2)\) 보정이 \(\bar\rho\)로 환산되는 크기와 부호. 4절 긴장의 해소 |
| 분위수 \(z_q\approx-1.66\) | 여전히 `Selection/Open`. 본 감사 범위 밖 |
| benchmark 의존성 | 하한 445는 \(r=2\) (Gaussian-mode) 기준. 일반 모드는 \(N_{\mathrm{eff}}\ge r\,m^2/0.0407\) |

## 8. 결론

$$
\boxed{
\langle\Phi\rangle=\sigma D_{\mathrm{eff}}\ \text{는 평균의 분해},\qquad
\text{관측 정합 조건:}\quad
N_{\mathrm{eff}}\ \gtrsim\ 4.5\times10^2,\quad
\bar\rho\ \lesssim\ 2.2\times10^{-3}.
}
$$

bootstrap의 mean-field 단계는 \(N_{\mathrm{eff}}=D_{\mathrm{eff}}\) 독법에서는 거짓이고, 미시 모드 집계 독법(R3)에서만 살아남는다. 관측 정밀도가 거꾸로 미시 구조에 정량 하한을 강제한다는 점에서, 이 감사는 mean-field를 가정에서 **반증 가능한 예측**으로 바꾼다.
