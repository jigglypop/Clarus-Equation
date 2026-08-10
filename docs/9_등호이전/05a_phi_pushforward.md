# 05a. \(\phi\) Pushforward 도구

## 0. 목표

05장은 CE bridge에서 잔류 측도를

$$
\phi_\beta(x)=\int_{\mathcal P_{\mathrm{ns}}}K(x,\gamma)\,\mu_{\mathrm{ns},\beta}(d\gamma)
$$

로 쓸 수 있다고 했다. 이 문서는 그 식을 실제 도구로 쓰기 위한 조건과 규약을 고정한다.

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| 측도 pushforward 형식 | `[정리]` | measurable kernel과 finite measure가 있으면 닫힘 |
| CE 물리 \(\Phi\)와 동일시 | `[미완성]` | 어떤 커널을 고르는지에 달림 |
| 잔류 질량 보존 규약 | `[공리: 모델 선택]` | raw/normalized 선택 필요 |

## 1. 기본 데이터

필요한 데이터는 다섯 개다.

| 기호 | 의미 |
|---|---|
| \(\Gamma=\mathcal P_I\) | CE 경로 후보공간 |
| \(\mu_\beta\) | 조건 재가중 후 경로측도 |
| \(\Gamma_*\) | manifest 또는 선택 경로 집합 |
| \(\Gamma_{\mathrm{ns}}\) | 비선택 경로 집합 |
| \(K_\phi(x,\gamma)\) | 경로를 시공간/상태공간의 잔류장으로 읽는 커널 |

여기서 \(x\in X\)이고, \(X\)는 시공간 격자, 연속 시공간, AGI hidden-state index, 또는 관측 채널일 수 있다.

## 2. 잔류 측도 규약

비선택 raw measure를

$$
\nu_{\mathrm{ns},\beta}
=
\mathbf 1_{\Gamma_{\mathrm{ns}}}\mu_\beta
$$

로 둔다.

잔류 질량은

$$
q_{\mathrm{ns},\beta}
=
\nu_{\mathrm{ns},\beta}(\Gamma)
=
\mu_\beta(\Gamma_{\mathrm{ns}})
$$

이다.

조건부 잔류 측도는 \(q_{\mathrm{ns},\beta}>0\)일 때

$$
\widehat\mu_{\mathrm{ns},\beta}
=
\frac{\nu_{\mathrm{ns},\beta}}{q_{\mathrm{ns},\beta}}
$$

로 둔다.

두 규약:

| 규약 | 식 | 의미 |
|---|---|---|
| raw residual | \(\phi^{\mathrm{raw}}_\beta(x)=\int K_\phi(x,\gamma)d\nu_{\mathrm{ns},\beta}\) | 선택되지 않은 총 질량까지 보존 |
| conditional residual | \(\phi^{\mathrm{cond}}_\beta(x)=\int K_\phi(x,\gamma)d\widehat\mu_{\mathrm{ns},\beta}\) | 비선택 경로의 모양만 보존 |

CE의 \(\phi\)를 에너지 저장량으로 읽으면 raw가 자연스럽다. 비선택 패턴의 방향만 읽으면 conditional이 자연스럽다.

## 3. 닫힌 정의

**정의 3.1**  
\((\Gamma,\mathcal B_\Gamma)\), \((X,\mathcal B_X)\)가 measurable space이고

$$
K_\phi:X\times\Gamma\to\mathbb R^r
$$

가 measurable이며 bounded라고 하자. 그러면 raw residual field는

$$
\phi^{\mathrm{raw}}_\beta(x)
=
\int_\Gamma K_\phi(x,\gamma)\nu_{\mathrm{ns},\beta}(d\gamma)
$$

로 정의된다.

bounded 조건은 integrability를 보장하기 위한 가장 쉬운 조건이다. 실제 물리 모델에서는 \(K_\phi\in L^1(\nu_{\mathrm{ns},\beta})\)이면 충분하다.

## 4. 유한 경로공간 공식

\(\Gamma=\{\gamma_1,\dots,\gamma_N\}\)이면

$$
\phi^{\mathrm{raw}}_\beta(x)
=
\sum_{\gamma_i\in\Gamma_{\mathrm{ns}}}
K_\phi(x,\gamma_i)\mu_\beta(\gamma_i)
$$

이다.

conditional 버전은

$$
\phi^{\mathrm{cond}}_\beta(x)
=
\frac{
\sum_{\gamma_i\in\Gamma_{\mathrm{ns}}}
K_\phi(x,\gamma_i)\mu_\beta(\gamma_i)
}{
\sum_{\gamma_i\in\Gamma_{\mathrm{ns}}}\mu_\beta(\gamma_i)
}
$$

이다.

## 5. 약수렴 안정성

**정리 5.1**  
\(\Gamma\)가 compact metric space이고, \(\nu_n\Rightarrow\nu\)가 finite measure의 weak convergence라고 하자. 고정된 \(x\)에 대해 \(K_\phi(x,\cdot)\)가 bounded continuous이면

$$
\int_\Gamma K_\phi(x,\gamma)\nu_n(d\gamma)
\to
\int_\Gamma K_\phi(x,\gamma)\nu(d\gamma)
$$

이다.

**증명.**

bounded continuous test function에 대한 weak convergence의 정의다. \(\square\)

이 정리는 \(\mu_\beta\)의 농축 극한이 있을 때 \(\phi_\beta(x)\)도 안정적으로 읽힌다는 최소 조건이다.

## 6. 커널 선택 후보

CE에서 \(K_\phi\)는 자동으로 주어지지 않는다. 후보는 다음과 같다.

| 커널 | 식 | 해석 |
|---|---|---|
| endpoint kernel | \(K_\phi(x,\gamma)=k(x,\gamma(T))\) | 비선택 경로의 도착점 잔류 |
| occupation kernel | \(K_\phi(x,\gamma)=\int_0^T k(x,\gamma(t))dt\) | 경로가 지나간 위치의 잔류 |
| curvature kernel | \(K_\phi(x,\gamma)=\mathcal H_\gamma(x)\) | 경로 헤시안/곡률 잔류 |
| AGI embedding kernel | \(K_\phi(i,\gamma)=P h_\gamma(i)\) | 후보 trace의 hidden residual |

05장의 `[미완성]`은 바로 이 표에서 어느 커널을 CE의 물리 \(\Phi\)와 연결할지 아직 선택하지 않았다는 뜻이다.

## 7. \(\Phi\)와 \(\phi\)의 구분

이 문서에서는 소문자 \(\phi\)를 잔류장 readout으로 쓴다.

대문자 \(\Phi\)는 기존 CE 문서에서 다음 의미로 쓰인다.

$$
\Phi \equiv \frac{\delta^2S}{\delta\gamma^2}
$$

즉 경로 접힘을 매개하는 유효 억압 자유도다.

둘은 자동으로 같지 않다. 동일시하려면 별도 bridge가 필요하다.

가능한 동일시:

$$
K_\phi(x,\gamma)=\frac{\delta^2S}{\delta\gamma^2}(x;\gamma)
$$

를 택하고

$$
\phi^{\mathrm{raw}}_\beta(x)
=
\int_{\Gamma_{\mathrm{ns}}}
\frac{\delta^2S}{\delta\gamma^2}(x;\gamma)
\mu_\beta(d\gamma)
$$

를 CE의 effective residual \(\Phi_{\mathrm{res}}\)로 읽는다.

이때도 \(\Phi_{\mathrm{res}}\)는 전체 \(\Phi\)가 아니라 비선택 경로에서 온 잔류 성분이다.

## 8. 물리 사상을 닫는 체크리스트

05장의 \(\phi\) pushforward에 조건부 `[정리]`를 적용하려면 다음을 채운다. 물리적 동일시는 별도의 `[공리: 물리 사상]`이다.

| 항목 | 필요 조건 |
|---|---|
| 경로공간 | \(\Gamma\)의 measurable/topological 구조 |
| 선택집합 | \(\Gamma_*\)와 \(\Gamma_{\mathrm{ns}}\)의 measurability |
| 측도 | \(\mu_\beta\)와 \(\nu_{\mathrm{ns},\beta}\)의 finite measure 성질 |
| 커널 | \(K_\phi\)의 measurability와 integrability |
| 규약 | raw 또는 conditional 중 선택 |
| 물리 대응 | \(\phi\), \(\Phi\), \(\Phi_{\mathrm{res}}\) 중 무엇을 말하는지 명시 |

## 9. 한 줄 결론

\(\phi\) pushforward는 수학적으로 어렵지 않다. 어려운 부분은 커널 선택이다. 즉 문제는 적분식이 아니라 "무엇을 잔류장으로 읽을 것인가"다.
