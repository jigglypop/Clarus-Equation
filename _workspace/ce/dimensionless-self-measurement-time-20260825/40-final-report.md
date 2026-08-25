# 무차원 자기측정 깊이, 자기비동일성 흐름과 기회비용의 조립

Status: COMPLETE

## 초록

이 연구는 완료 기록의 0차원성, 무수한 약한 자기측정, 선택되지 않은 경로의 기회비용, 그리고 “흐름은 이전의 자기와 달라짐의 반복”이라는 네 생각을 한 조건부 양자채널 모형으로 조립했다. 단일 고정 dephasing projector에서는 $\theta=-\ln(1-\eta)$가 정확히 additive하며, 특정 $\theta_*$는 같은 unconditional channel을 내는 모든 유한 weak partition의 동치류를 표지한다. object와 record 또는 앞뒤 시간절편을 분리하고 현재 instrument가 과거 기록에만 의존할 때 자기측정은 비순환 operational recursion으로 정의된다. 초기 coherence $A=(I-\mathcal D_P)\rho_0$가 0이 아니면 모든 finite $\theta,h>0$에서 상태가 달라지고, 그 trace-distance 속도와 누적 길이 및 logarithmic residual clock을 닫힌식으로 얻는다. 기회비용은 같은 좌표에서 유계인 $C_{\rm self}$로 쓸 수 있고 선택한 nonstationary path에서는 $dC_{\rm self}$와 $dL$을 연결할 수 있지만, 두 양은 일반적으로 동일하지 않다. stationary, periodic, noncommuting 및 non-Markovian 완전 반례 때문에 이 결과는 물리시간·에너지·0D ontology·암흑물질·암흑에너지의 증명이 아니다.

## 1. 네 개념을 먼저 분리한다

여기서 서로 다른 대상을 같은 “0차원”이라는 말로 합치지 않는다.

1. $Z_{\rm phys}=\{\star\}$는 가정된 strict external preparation boundary다.
2. $R=\{r_1,\ldots,r_n\}$은 완료된 finite record alphabet이다. 위상적으로
   0차원이지만 $n>1$이면 singleton이 아니다.
3. $\mu_F=\sum_jw_j\delta_{X_j}$의 support는 공간절편의 point carrier다.

informative measurement의 최소 channel type은

$$
Z_{\rm phys}\longrightarrow M,
\qquad
M\longrightarrow R,
\qquad
R\ne Z_{\rm phys}
$$

이다. $R=Z_{\rm phys}$라고 동일시하면서 informative record와 strict
no-$M\to Z_{\rm phys}$를 동시에 요구하면 record channel이 input-independent여야
하므로 모순이다. $R\to\mu_F$의 영구 보존은 별도 retention 공리이며 양자
instrument에서 자동으로 나오지 않는다.

## 2. 측정깊이 정리

직교 record partition의 dephasing projector를

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r,
\qquad \mathcal D_P^2=\mathcal D_P
$$

로 두고 partial wall을

$$
\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P
$$

로 둔다. $Q=\operatorname{Id}-\mathcal D_P$이면
$\Phi_\eta=\mathcal D_P+(1-\eta)Q$이고 $\mathcal D_PQ=Q\mathcal D_P=0$이다.
그러므로

$$
\Phi_{\eta_2}\Phi_{\eta_1}
=\mathcal D_P+(1-\eta_2)(1-\eta_1)Q.
$$

여기서

$$
\boxed{\theta=-\ln(1-\eta)}
$$

를 정의하면

$$
\boxed{
\Phi_\theta
=e^{\theta(\mathcal D_P-operatorname{Id})}
=\mathcal D_P+e^{-\theta}Q,
\qquad
\Phi_{\theta_2}\Phi_{\theta_1}=\Phi_{\theta_1+\theta_2}
}
$$

이다. 따라서 $\sum_k\delta\theta_k=\theta_*$인 모든 유한 분할은 정확히
같은 $\Phi_{\theta_*}$를 준다. 이것이 “한 특정 무차원 측정시간은 무수한
자기측정 변형과 같다”의 증명 가능한 내용이다. 단, 같은 것은 unconditional
state channel이며 실제 outcome history, conditional trajectory와 feedback law는
서로 다를 수 있다.

$\theta$는 physical time이 아니다. Markov rate $[\gamma]=T^{-1}$를 독립적으로
지정한 경우에만

$$
\theta(t)=\int_{t_0}^{t}\gamma(s)ds
$$

라는 clock bridge를 얻는다.

## 3. 자기측정 bootstrap의 인과적 정의

전체를 $U=S+R$라고 부르되 object $S$와 내부 record/controller $R$, 또는
앞선 시간절편과 뒤 시간절편을 구분한다. 기록 filtration
$\mathcal F_n=\sigma(r_0,\ldots,r_n)$에 대해 현재 설정
$m_n$이 $\mathcal F_{n-1}$에만 의존하면

$$
p(r_n\mid\mathcal F_{n-1})
=\operatorname{tr}\mathcal I_{r_n}^{(m_n)}(\rho_n),
$$

$$
\rho_{n+1\mid r_n}
=\frac{\mathcal I_{r_n}^{(m_n)}(\rho_n)}
{\operatorname{tr}\mathcal I_{r_n}^{(m_n)}(\rho_n)},
\qquad
m_{n+1}=F(m_n,r_n)
$$

라는 causal loop를 얻는다. 이를 구조적으로 쓰면

$$
S_n\longrightarrow R_n,
\qquad
(S_n,R_{\le n})\longrightarrow S_{n+1}.
$$

즉 옆의 subsystem 또는 직전 시간절편이 만든 record가 다음 quantum operation을
실행시키는 반복 bootstrap이다. 현재 결과를 현재 설정에 미리 넣거나 미래 record를
참조하면 circular rule이 된다. 또한 이 구조는 한 copy가 자기의 arbitrary unknown
state 전체를 완전 복제한다는 뜻이 아니므로 no-cloning을 우회하지 않는다.

feedback가 측정축이나 generator를 바꾸면 실제 conditional dynamics는 일반적으로
위의 단일 fixed-$\mathcal D_P$ semigroup가 아니다. 그 경우 이 절의 trajectory
모형은 유지할 수 있지만 제2절의 scalar $\theta$ 정리는 별도로 재증명해야 한다.

## 4. 자기비동일성 흐름 정리

연속 parameter에는 “바로 직전 점”이 없다. 따라서 모든 양의 increment $h$에
대한 비동일성과 trace-distance metric speed로 문장을 바꾼다. 다음을 두자.

$$
A=(\operatorname{Id}-\mathcal D_P)\rho_0,
\qquad
\rho_\theta=\mathcal D_P\rho_0+e^{-\theta}A.
$$

그러면

$$
\rho_{\theta+h}-\rho_\theta
=-e^{-\theta}(1-e^{-h})A.
$$

따라서 정확히

$$
\boxed{
A\ne0
\Longleftrightarrow
\rho_{\theta+h}\ne\rho_\theta
\quad\text{for every finite }\theta\ge0, h>0
}
$$

이다. trace distance $D_{\rm tr}(\rho,\sigma)=\frac12\|\rho-\sigma\|_1$로
보면

$$
D_{\rm tr}(\rho_{\theta+h},\rho_\theta)
=\frac12e^{-\theta}(1-e^{-h})\|A\|_1,
$$

$$
v(\theta)=\frac12e^{-\theta}\|A\|_1,
\qquad
L(\theta_*)=\frac12(1-e^{-\theta_*})\|A\|_1.
$$

무한히 잘게 나눈 각 차이는 0으로 가지만 그 합은 유한한 $L$로 수렴한다.
fixed point까지 남은 residual을

$$
\mathscr R(\theta)
=D_{\rm tr}(\rho_\theta,\mathcal D_P\rho_0)
=\frac12e^{-\theta}\|A\|_1
$$

로 두면 $A\ne0$에서

$$
\boxed{
\theta
=\ln\frac{\mathscr R(0)}{\mathscr R(\theta)}
=-\ln\left(1-\frac{L(\theta)}{L(\infty)}\right)
}
$$

이다. 따라서 이 고정 수축모형 안에서는 “현재가 이전과 달라지는 누적량”과
“완료상태까지 남은 자기차이”가 하나의 방향 있는 내부 좌표를 복원한다.
이 결과의 지위는 conditional distinguishability clock이지 보편 physical time이 아니다.

## 5. 기회비용과의 조립

finite outcome probabilities $p_a(\theta)$에서

$$
\overline C_I(\theta)
=\sum_ap_a(\theta)[1-p_a(\theta)][-\ln p_a(\theta)]
$$

를 두면

$$
\boxed{
C_{\rm self}(\theta_*)
=\int_0^{\theta_*}e^{-\theta}\overline C_I(\theta)d\theta
}
$$

이다. $\overline C_I\le H(p)\le\ln n$이므로

$$
0\le C_{\rm self}(\theta_*)
\le(1-e^{-\theta_*})\ln n
\le\ln n.
$$

$A\ne0$인 같은 fixed path에서는

$$
dC_{\rm self}
=\frac{2\overline C_I(\theta)}{\|A\|_1}dL.
$$

이 식이 네 생각의 최소 조립이다.

$$
\text{과거 내부 기록}
\longrightarrow
\text{다음 측정 실행}
\longrightarrow
\theta\text{ 증가와 자기차이 수축}
\longrightarrow
C_{\rm self}\text{ 회계}.
$$

그러나 이것은 존재론적 동일식이 아니다. 이미 diagonal인
$\rho_0=\operatorname{diag}(0.8,0.2)$는 $A=0$이라 $L=0$이지만 같은 확률로
계산한 $\overline C_I=0.2932130342\ldots$는 양수다. 따라서 opportunity cost는
가능한 대안의 정보 회계이고 $L$은 실제 coherence motion이다. 둘 중 어느 것도
독립 scale과 action 없이 energy가 아니다.

spatial point carrier 후보에는

$$
d\mu_{\rm self}(\theta,B)
=e^{-\theta}\sum_ap_a(\theta)[1-p_a(\theta)][-\ln p_a(\theta)]
\mathbf1_B(F_\theta(a))d\theta
$$

라는 positive finite measure를 쓸 수 있다. 하지만 $F_\theta$, retention,
covariance, conservation과 backreaction은 아직 미도출이다.

## 6. 완전 반례와 적용 경계

| 더 강한 부모 주장 | counterexample | 결론 |
|---|---|---|
| 모든 상태는 계속 자기와 달라진다 | $A=0$인 dephased stationary state | 초기 coherence 조건 필요 |
| 국소 변화가 시간의 화살을 만든다 | 한 주기 뒤 돌아오는 unitary orbit | monotone Lyapunov 조건 필요 |
| 모든 측정을 하나의 scalar $\theta$로 센다 | noncommuting measurement axes의 order dependence | path ordering 또는 다중 좌표 필요 |
| memory가 있어도 $\theta$는 전역 단조다 | $\lambda(t)=\cos^2(gt/2)$ recoherence | process tensor 또는 memory kernel 필요 |
| 분할 수가 무한이면 정보·에너지도 무한이다 | $C_{\rm self}\le\ln n$, $L\le\|A\|_1/2$ | refinement와 total resource를 구별 |

수치 반례에서 $z$와 $(x+z)/\sqrt2$ dephasing의 순서를 바꾼 출력 차이는
$0.049497474683058$이었다. periodic unitary의 한 주기 path length는
$3.141592601912349$인데 endpoint distance는 0이었다.

## 7. 우주론적 지위와 다음 falsifier

현재 결과는 “비선택 경로가 암흑물질·암흑에너지다”라는 동일성을 증명하지
않는다. 현재 닫힌 것은 측정채널과 bounded information bookkeeping이다.
암흑 표현으로 가려면 적어도 다음을 독립적으로 닫아야 한다.

1. actual record에서 persistent spatial carrier로 가는 quantum retention map.
2. carrier와 한 환경장 $\chi$ 사이의 causal backreaction 및 non-Markov 처리.
3. 독립 energy-density scale을 가진 covariant action과 total conservation law.
4. 암흑물질 후보의 $a^{-3}$ scaling, clustering, sound speed, Jeans scale와 lensing.
5. 암흑에너지 후보의 abundance, pressure, perturbations와
   Einstein--Boltzmann observables.

특히 $C_{\rm self}$를 단순히 $\rho_{\rm dark}$라고 놓는 것은 차원도 물리도
완성하지 못한다. $epsilon_* C_{\rm self}$는 차원상 energy density가 될 수 있지만
$\epsilon_*$의 기원과 action이 없으면 임의 scale 선택일 뿐이다.

## 8. 재현

```powershell
& '.codex\hooks\python.cmd' python `
  '_workspace\ce\dimensionless-self-measurement-time-20260825\artifacts\verify_self_measurement_time.py'
```

검증기는 channel CPTP certificate, exact partition law, $C_{\rm self}$ 상계,
self-flow length와 logarithmic clock 및 네 반례를 모두 통과했다. canonical
우주론 집중 회귀는 `58 passed in 1.28s`였다.

## 참고문헌

- V. P. Belavkin, “Quantum continual measurements and a posteriori collapse on CCR,” *Commun. Math. Phys.* 146 (1992), [doi:10.1007/BF02097018](https://doi.org/10.1007/BF02097018), accessed 2026-08-25.
- H. M. Wiseman, “Quantum theory of continuous feedback,” *Phys. Rev. A* 49 (1994), [doi:10.1103/PhysRevA.49.2133](https://doi.org/10.1103/PhysRevA.49.2133), accessed 2026-08-25.
- S. Attal and Y. Pautrat, “From repeated to continuous quantum interactions,” *Ann. Henri Poincaré* 7 (2006), [doi:10.1007/s00023-005-0242-8](https://doi.org/10.1007/s00023-005-0242-8), accessed 2026-08-25.
- H.-P. Breuer, E.-M. Laine and J. Piilo, “Measure for the Degree of Non-Markovian Behavior of Quantum Processes in Open Systems,” *Phys. Rev. Lett.* 103 (2009), [doi:10.1103/PhysRevLett.103.210401](https://doi.org/10.1103/PhysRevLett.103.210401), accessed 2026-08-25.
- W. K. Wootters and W. H. Zurek, “A single quantum cannot be cloned,” *Nature* 299 (1982), [doi:10.1038/299802a0](https://doi.org/10.1038/299802a0), accessed 2026-08-25.
- B. Misra and E. C. G. Sudarshan, “The Zeno’s paradox in quantum theory,” *J. Math. Phys.* 18 (1977), [doi:10.1063/1.523304](https://doi.org/10.1063/1.523304), accessed 2026-08-25.

