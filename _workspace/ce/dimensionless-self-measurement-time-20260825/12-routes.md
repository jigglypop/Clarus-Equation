# 세 아이디어를 결합하는 대안 경로

Status: COMPLETE

| route | 측정 구조 | 필요한 좌표 | 기회비용 | 판정 |
|---|---|---|---|---|
| R0 | 고정 QND/dephasing partition의 Markov semigroup | 단일 additive $\theta$ | bounded $C_{\rm self}$ | 채택: 조건부 정리 |
| R1 | 실제 stochastic record와 past-adapted internal feedback | trajectory $r_{0:n}$와 $\theta$ | conditional expectation | 채택: operational 자기측정 |
| R2 | 시간에 따라 변하지만 서로 commuting인 generator | 누적률 또는 여러 commuting 좌표 | 경로별 계산 | 제한적 확장 |
| R3 | noncommuting/time-dependent partition | path-ordered superoperator | instrument-order dependent | 단일 $\theta$ 기각 |
| R4 | non-Markovian environment와 recoherence | process tensor 또는 memory kernel | history/reference dependent | 전역 monotone $\theta$ 기각 |

## R0. 가장 좁고 정확한 코어

고정된 record partition $P$에 대해

$$
\Phi_\theta
=\mathcal D_P+e^{-\theta}(\operatorname{Id}-\mathcal D_P),
\qquad
\Phi_{\theta_2}\Phi_{\theta_1}=\Phi_{\theta_1+\theta_2}
$$

를 코어로 삼는다. 이때 특정 $\theta_*$는 물리적 한 순간이 아니라
같은 총 채널을 만드는 모든 weak partition의 동치류이다. 이 문장은
유한 분할에서 이미 정확하므로 continuum limit의 해석에 의존하지 않는다.

## R1. 자기 자신을 보는 구조

$$
S_n\longrightarrow R_n,
\qquad
(S_n,R_{\leq n})\longrightarrow S_{n+1}
$$

처럼 object가 만든 과거 record가 동일한 전체 시스템의 다음 dynamics를
설정하도록 한다. 전체를 $U=S+R$라고 부르면 이는 operational한
self-measurement이지만, 동시적인 완전 자기복제는 아니다. 설정 $m_n$이
과거 filtration에만 의존한다는 adaptedness가 순환을 막는다.

unconditional semigroup $\Phi_\theta$는 평균상태의 분할 불변성을 말하고,
conditional trajectory는 실제 결과열과 feedback을 보존한다. 둘은 서로
대체되는 설명이 아니며, 자기측정이라는 물리적 해석에는 R1의 record가
추가로 필요하다.

## R2. commuting 확장

시간별 generator들이 서로 commute하면

$$
\Phi=exp\left(\int d\theta^i\,\mathcal L_i\right)
$$

형태의 누적 좌표를 쓸 수 있다. 그러나 generator가 하나가 아니면 일반적으로
한 개의 scalar로 압축할 이유는 없다. 이 경로는 모형별 spectral 조건을
따로 증명해야 한다.

## R3. noncommuting 경로

측정축이 바뀌면

$$
\Phi=
\mathcal T\exp\left(\int\mathcal L(\theta)d\theta\right)
$$

가 필요하고, 같은 총 강도라도 순서가 결과를 바꾼다. 따라서
“모든 자기측정 변형이 하나의 특정 무차원 시간과 같다”는 문장은
고정 또는 적절히 commuting한 partition으로 범위를 제한해야 한다.

## R4. non-Markov 경로

환경에 남은 정보가 다시 계로 돌아오면 coherence가 revival할 수 있다.
이때 $\eta$와 $\theta$를 전역 단조 좌표로 둘 수 없다. CE의 persistent
fold-memory를 실제 feedback dynamics로 만들려는 순간 이 경로가 중요해진다.
즉, 현재의 scalar $\theta$ 정리는 memory field가 역으로 계에 작용하지 않는
Markov 또는 quenched 구간에만 그대로 적용된다.

## 세 아이디어의 최소 통합

1. **기회비용:** 선택되지 않은 outcome weight를 $\overline C_I$로 센다.
2. **무차원 측정깊이와 0D 기록:** $\theta=-\ln(1-\eta)$가 기록 형성의
   누적 강도를 세고, completed outcome alphabet은 topological 0D로 분리한다.
3. **자기측정:** 과거 내부 record가 다음 instrument를 설정하는
   past-adapted loop로 정의한다.
4. **자기비동일성 흐름:** 현재 상태와 임의로 가까운 미래 상태의
   trace distance를 누적해 $L$을 정의하고, fixed dephasing 경로에서는
   감소 residual로 $\theta$의 방향을 정한다.

세 요소를 결합하면

$$
C_{\rm self}(\theta_*)
=\int_0^{\theta_*}e^{-\theta}\overline C_I(\theta)d\theta
$$

가 나온다. 이것은 현재 단계에서 bounded dimensionless information
functional이다. 공간 point carrier로의 pushforward, 영구 보존, 에너지와
stress, dark matter clustering 및 dark energy pressure는 각각 별도의
미완성 bridge이다.

fixed dephasing 경로의 $A=(I-\mathcal D_P)\rho_0\ne0$ 구간에서는

$$
L(\theta_*)=\frac12(1-e^{-\theta_*})\|A\|_1,
\qquad
dC_{\rm self}
=\frac{2\overline C_I(\theta)}{\|A\|_1}dL
$$

로 네 요소를 한 식에 조립할 수 있다. 그러나 diagonal stationary state는
$L=0$이면서도 probability-based opportunity cost가 양수일 수 있다. 따라서
$C_{\rm self}$를 상태운동이나 에너지와 보편적으로 동일시하지 않는다.

## 선택

R0과 R1을 함께 채택한다. R0은 정확한 정리와 계산 좌표를 제공하고,
R1은 “자기가 자신을 본다”를 인과적으로 해석한다. R3과 R4는 실패가 아니라
적용범위를 정하는 완전 반례로 유지한다. 0D는 strict external singleton,
finite discrete record, spatial point carrier를 하나의 대상으로 합치지 않고
각기 다른 type으로 보존한다.

“흐름”은 R0에서는 strict residual contraction으로 조건부 채택한다.
일반 경로에서는 positive local speed를 path motion이라고만 부르고,
recurrence가 없는 시간의 화살이라는 더 강한 주장은 채택하지 않는다.
