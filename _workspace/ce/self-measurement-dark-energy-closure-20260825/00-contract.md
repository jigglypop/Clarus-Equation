# 자기측정 기회잔여에서 암흑에너지까지의 공변 완결 연구 계약

Status: COMPLETE

PREDECESSOR: `_workspace/ce/dimensionless-self-measurement-time-20260825`

## 1. 질문과 완결의 의미

선행 run의 무차원 측정깊이와 기회비용을 실제 중력 source로 승격하려면 어떤
최소 공리를 추가해야 하며, 그 공리에서 배경 FLRW, 방정식상태, 안정성, 선형
성장과 DESI BAO observable까지 재현 가능하게 계산할 수 있는가를 묻는다.

이 run에서 “끝까지 유도”는 다음 구조적 폐쇄를 뜻한다.

1. operational measurement variable과 local covariant field의 사상을 명시한다.
2. action, stress tensor와 보존법칙을 명시한다.
3. FLRW background와 late-time fixed point를 유도한다.
4. ghost, gradient, tachyonic stability와 smooth-DE growth 범위를 판정한다.
5. 실제 로컬 DESI DR2 13-vector covariance로 background shape를 비교한다.

절대 dark-energy scale을 양자정보만으로 무입력 예측하거나 자연에서의 dark
identity를 증명했다는 뜻은 아니다. 그 둘은 별도의 claim으로 감사한다.

## 2. 선행 정리와 고정 정의

단일 fixed dephasing partition에서

$$
\theta=-\ln(1-\eta),
\qquad
u=e^{-\theta},
\qquad
c=1-u=1-e^{-\theta}
$$

로 둔다. $u$는 미완료 self-distinguishability fraction, $c$는 완료 fraction이다.
outcome probabilities가 measurement-depth 구간에서 일정하면

$$
C_{\rm self}(\theta)=\overline C_I(1-e^{-\theta})
=C_\infty c,
\qquad C_\infty=\overline C_I.
$$

이 식은 dimensionless information theorem이다. 아직 energy가 아니다.

## 3. 공리 A: operational-to-local retention

**[후보 공리: 물리 사상]** coarse-graining scale $L_c$와 Cauchy surface
$\Sigma_*$를 고정하고, operational depth를 dimensionless covariant scalar
$\Theta(x)$의 initial data로 보낸다.

$$
\Theta|_{\Sigma_*}=\mathcal R_\Theta[\theta,\mu_F],
\qquad
n^\mu\nabla_\mu\Theta|_{\Sigma_*}\ge0.
$$

$\mathcal R_\Theta$, $L_c$, no-double-counting과 covariance는 이 run에서
구체 microscopic derivation이 아니라 명시적 model axiom이다. source는 matching
뒤 꺼지고, 이후 matter, radiation과 $\Theta$ sector는 전체 공변 action 아래
보존된다고 고정한다.

## 4. 공리 B: self-measurement residual quintessence action

metric signature $(-,+,+,+)$와 reduced Planck mass $M_{\rm Pl}$을 쓰고
$f>0$, $\rho_*>0$를 둔다.

$$
S_{\rm SMQ}=
\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac{f^2}{2}g^{\mu\nu}\nabla_\mu\Theta\nabla_\nu\Theta
-\rho_*e^{-\Theta}
\right]+S_m+S_r.
$$

canonical field와 slope를

$$
\phi=f\Theta,
\qquad
\lambda=\frac{M_{\rm Pl}}{f}>0
$$

로 정의한다. 그러면 $V(\phi)=\rho_*e^{-\lambda\phi/M_{\rm Pl}}$다.
$M_{\rm Pl}$은 gravitational action의 scale이고 $\rho_*$는 독립 potential
normalization이다. $\rho_*$ 또는 동등한 present $\Omega_{\Theta0}$는 외부
normalization이며 선행 information theorem에서 나오지 않는다.

## 5. 비교 경로 L: 누적비용의 literal potential

다음 경로를 부모 후보로 함께 감사한다.

$$
V_L(\Theta)=\rho_*c(\Theta)=\rho_*(1-e^{-\Theta}).
$$

source-free canonical equation과 $\dot\Theta=0$ 초기조건에서
$V_{L,\Theta}>0$가 $\Theta$를 감소시키는지 검사한다. 그렇다면 monotone
self-measurement depth와 accumulated cost energy가 같은 autonomous field라는
부모 주장은 제거한다. nonzero initial drive, interacting reservoir, phantom 또는
modified gravity로 전제를 바꾸는 경로는 별도 대안으로만 남긴다.

## 6. 유도할 방정식과 정리 후보

flat FLRW와 homogeneous $\Theta$에서

$$
\rho_\Theta=\frac{f^2}{2}\dot\Theta^2+\rho_*e^{-\Theta},
\qquad
p_\Theta=\frac{f^2}{2}\dot\Theta^2-\rho_*e^{-\Theta}
$$

및 Friedmann, Raychaudhuri, Klein--Gordon 식을 유도한다. exponential-potential
autonomous variables

$$
x=\frac{\dot\phi}{\sqrt6M_{\rm Pl}H},
\qquad
y=\frac{\sqrt V}{\sqrt3M_{\rm Pl}H}
$$

에서 scalar-dominated fixed point, 존재·안정 조건과

$$
w_\Theta=-1+\frac{\lambda^2}{3},
\qquad
\frac{d\Theta}{d\ln a}=\lambda^2
$$

를 검산한다. acceleration 조건이 $\lambda^2<2$인지 확인한다.

linear scalar perturbation의 kinetic coefficient, rest-frame sound speed와
$V_{,\phi\phi}$를 계산한다. subhorizon, negligible-DE-clustering 범위에서

$$
D''+\left(2+\frac{H'}H\right)D'
-\frac32\Omega_m(a)D=0
$$

를 구현한다. prime은 $d/d\ln a$다.

## 7. 수치 initial-value prescription

thawing background control은 다음처럼 사전 고정한다.

- $a_i=10^{-4}$.
- $\Theta(a_i)=0$, $d(\phi/M_{\rm Pl})/d\ln a|_{a_i}=0$.
- flat boundary에서 $\Omega_{r0}$, $\Omega_{m0}$를 외부 입력으로 둔다.
- 각 $\lambda$에서 potential amplitude를 bisection하여
  $\Omega_{\Theta0}=1-\Omega_{m0}-\Omega_{r0}$를 맞춘다.
- ODE constraint와 conservation residual tolerance는 $10^{-9}$,
  grid-refinement observable tolerance는 $10^{-5}$로 둔다.
- 물리적 검사 domain은 $0<\lambda\le1.4<\sqrt2$다. $\lambda=0$은
  $f\to\infty$인 모형 폐포의 cosmological-constant limit control이며,
  유한 $f$ 작용의 원소라고 부르지 않는다.

이 initial prescription은 operational retention map의 microscopic derivation이
아니라 한 conditional cosmological branch다.

## 8. 관측 비교 계약

로컬에서 covariance까지 고정된 자료는 DESI DR2 compressed BAO 13-vector만
사용한다.

- mean SHA-256:
  `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585`
- covariance SHA-256:
  `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509`
- upstream: CobayaSampler/`bao_data` v2.6 commit
  `b7b8a36e9bccb063081f811f323cada21ab5fbdd`.

각 background shape에서 $s=c_{\rm light}/(H_0r_d)>0$ 하나를 analytic하게
profile한다. model vector가 $m=s g(\lambda)$이면

$$
\widehat s=
\frac{g^TC^{-1}d}{g^TC^{-1}g}
$$

를 사용한다. $\lambda=0$은 fitted scale 1개의 closure-limit baseline이고, 전체
$\lambda$ scan은 scale과 slope 2개의 calibration으로 센다. $\chi^2$, degrees of
freedom, $\Delta\chi^2$, AIC와 BIC를 중립적으로 보고한다.

background boundary는 두 개를 분리한다.

1. `B-DESI`: $\Omega_{m0}=0.3027$, $\Omega_{r0}=9\times10^{-5}$.
2. `B-PLANCK`: $\Omega_{m0}=0.3153$, $\Omega_{r0}=9\times10^{-5}$.

첫 경계는 같은 release의 compressed result를 사용하므로 target-aware이고,
두 번째도 Planck $\Lambda$CDM posterior를 dynamic-DE 모형에 옮긴 조건부 control이다.
어느 것도 blind prediction이 아니다. SN, CMB full likelihood, $H(z)$와 growth
likelihood는 로컬 covariance가 없으므로 채점하지 않는다.

## 9. falsifier와 완전 반례

다음 중 하나면 해당 claim을 제거하거나 범위를 좁힌다.

1. action variation이 선언한 stress/KG 식과 일치하지 않는다.
2. $f^2\le0$, $c_s^2\le0$ 또는 $V_{,\phi\phi}<0$가 허용 domain에 생긴다.
3. scalar fixed point나 acceleration 조건이 autonomous-system Jacobian과 다르다.
4. literal $V_L$이 source-free rest initial data에서 $\Theta$를 증가시키는
   attractor를 실제로 준다.
5. numerical Friedmann constraint, conservation 또는 refinement tolerance가 깨진다.
6. BAO loader의 hash/order/positive-definite covariance가 깨진다.

## 10. 주장 상한

통과 가능한 최강 결론은 다음이다. operational $u=e^{-\theta}$를 local scalar
initial data로 보내고 exponential canonical action을 채택하면, stable하고
공변 보존되는 quintessence dark-energy sector와 그 background, smooth-growth 및
BAO prediction을 조건부로 완결할 수 있다. 측정기록이 실제로 그 field를 만든다는
microscopic retention, $\rho_*$의 절대값, 양자복사 안정성, full
Einstein--Boltzmann likelihood와 자연에서의 dark identity는 별도 미완성 또는
외부 calibration이다. 로컬 DESI 비교는 사후 empirical calibration이며 예측으로
승격하지 않는다.
