# CE 귀납 강의: 관측 패턴에서 반증 가능한 모형까지

## 0. 귀납의 원칙

수치가 가까운 식을 찾는 일은 가설 생성이지 증명이 아니다. 이 강의는
관측 패턴을 발견한 뒤 다음 순서로 모형을 고정한다.

$$
\text{패턴 발견}
\to\text{작용·측도·branch 지정}
\to\text{입력/보정 분리}
\to\text{새 출력 계산}
\to\text{공동 likelihood와 holdout}.
$$

현 checkout의 관측 입력, 수치 사슬과 주장 지위는
[D 정합성 원장](D_정합성_원장.md)을 따른다. 삭제된 전역 JSON이나 과거
inventory를 전제로 하지 않는다.

이 장의 논리 지위 `Definition`, `Exact`, `Exact conditional`, `Convention`,
`Selection`, `Bridge`, `Phenomenology`와 validation 상태 `Open`, `Rejected`는
[A 강의의 지위 계약](A_연역적_유도.md#0-이-강의의-계약)과 같은 뜻이다.
`Calibration input`, `Calibration output`은 별도 provenance 역할이다.
관측값과 가까운 수치가 나와도 `Bridge`를 통과하지 않았으면 물리적
예측으로 승격하지 않는다.

## 1. 첫 패턴: 약 5%의 바리온 분율

평탄 $\Lambda$CDM chain에서 $\omega_b=\Omega_bh^2$와 $h$를 함께 추정하면
$\Omega_b$가 약 $0.049$라는 사실을 얻는다. 단순 지수

$$
e^{-3}=0.049787\ldots
$$

가 가까운 것은 가설의 단서다. 여기서 곧바로
$e^{-3}=\Omega_b$라고 쓰지 않는다. 먼저 다음 함수형 정리를 증명한다.

**지위: `Exact conditional` — S1--S4 아래의 지수 character 정리.**

$S:[0,\infty)\to(0,1]$가

1. **S1:** $S(0)=1$,
2. **S2:** $S(D_1+D_2)=S(D_1)S(D_2)$,
3. **S3:** 측정가능 또는 연속,
4. **S4:** 어떤 $D_0>0$에서 $S(D_0)<1$

을 만족한다고 하자. $f=-\log S$는 정칙한 가법함수이므로
$f(D)=\kappa_{\rm surv}D$다. 범위가 $(0,1]$이므로
$\kappa_{\rm surv}\ge0$이고 S4 때문에 $\kappa_{\rm surv}>0$이다. 따라서

$$
\boxed{S(D)=e^{-\kappa_{\rm surv}D}},\qquad \kappa_{\rm surv}>0.
$$

**지위: `Convention`.** $\widetilde D:=\kappa_{\rm surv}D$를 optical
depth로 정의하고 재매개화 함수
$\widetilde S(\widetilde D):=S(\widetilde D/\kappa_{\rm surv})$를 두면
$\widetilde S(\widetilde D)=e^{-\widetilde D}$다. 즉
$\kappa_{\rm surv}=1$은 단위 선택이지
새 자연상수의 예측이 아니다.

**지위: `Open Bridge`.** 실제 CE 구간 비용의 가법성, 독립 구간 생존의
곱성, 그리고 깊이를 계산하는 물리 operator는 함수형 정리가 정하지 않는다.
마지막으로 바리온은 path-count가 아니라 3절의 양의 energy readout으로
식별해야 한다.

## 2. 전자약 패턴을 invariant로 바꾸기

과거에는 질량행렬의 특정 비대각 원소를 “혼합량”으로 사용했지만, 그 값은
기저 회전으로 바뀐다. 최신 가설은 gauge projector와 spectral projector의
overlap을 쓴다.

$$
p_W=\operatorname{Tr}(P_WP_Z),
\qquad
p_B=\operatorname{Tr}(P_BP_Z),
$$

$$
\boxed{\delta_{\rm proj}:=p_Wp_B=s_W^2(1-s_W^2)}.
$$

**지위: `Exact conditional on SM EWSB`.** 이 값은 지정한 $W^3/B$
부분공간의 normalized coherence intensity다. decay rate나 CE fold depth는
아니다.

**논리 지위: `Definition`. provenance: `Calibration output`.** Track A 입력
$\alpha_s(M_Z)=0.1180$에서

$$
s_A^2=0.2315097758,
\qquad
\delta_A:=s_A^2(1-s_A^2)=0.1779129995
$$

를 계산한다. canonical Track-A는 별도 `Selection`

$$
\delta_{\rm fold}:=\delta_A
$$

를 채택한다. 양의 additive fold operator의 trace 아래에서

$$
\boxed{D_A:=3+\delta_A=3.1779129995}
$$

가 따른다. $s_A^2$를 on-shell, $\overline{\rm MS}$ 또는 effective
약혼합각으로 보내는 map과
$\delta_A\stackrel{?}{=}\delta_{\rm proj}$는 아직 `Open Bridge`다. 관측
$s_W^2$를 먼저 넣어 $\alpha_s$를 역산하면 calibration이며 다시 예측으로
세지 않는다.

## 3. 패턴에서 고정점 정리로

상위 모형은 scalar가 아니라 다형 Poisson 재귀다. type $i$가 다음 세대
type $j$를 만드는 평균수를 $\mathsf K_{ij}\ge0$로 두면 최소 소멸확률은

$$
x_i=\exp\!\left[-\sum_j\mathsf K_{ij}(1-x_j)\right]
$$

을 만족한다. 이는 유한 비음수 next-generation 행렬과 type별 독립 Poisson
offspring를 조건으로 한 `Exact conditional`이다.

한 type이거나 모든 행합이

$$
\sum_j\mathsf K_{ij}=D_A\quad\text{for every }i
$$

이고 균일 부분공간 $x_i=x$를 선택할 때에만

$$
x=e^{-D_A(1-x)}
$$

로 정확히 축약된다. 행합이 다르면 전체 벡터를 풀어야 하며 평균 하나가
같은 scalar 식을 따른다고 가정하지 않는다. $D_A>1$에서 저분율 branch는

$$
x_\star=-\frac{W_0(-D_Ae^{-D_A})}{D_A}
=0.0486382585
$$

이고 multiplier $D_Ax_\star=0.1545681540<1$이다. 다른 Lambert 가지는
경계해 $x=1$을 준다. 이 안정성은 고정점 반복의 안정성이지 우주 시간진화의
안정성이 아니다. CE+SM 작용에서 $\mathsf K$, 독립 Poisson 계보와 공통
행합을 얻는 단계는 `Open Bridge`다.

이제 수학적 근 $x_\star$와 물리 에너지분율 $x_E$를 분리한다. 깊이 $D$의
양의 측도 $\mu_D$, 생존 사건 $\mathcal A_D$, 총 comoving-energy readout
$H_D\ge0$에 대해

$$
x_E(D):=
\frac{\int_{\mathcal A_D}H_Dd\mu_D}{\int H_Dd\mu_D},
\qquad
0<\int H_Dd\mu_D<\infty
$$

로 둔다. 다음 E1--E4를 모두 요구한다.

1. **E1:** $H_{b,D}=H_D\mathbf1_{\mathcal A_D}$.
2. **E2:** $\rho_b$와 $\rho_{\rm tot}$를 같은 초곡면·comoving 영역에서
   위 두 적분으로 계산한다.
3. **E3:** $x_E:[0,\infty)\to(0,1]$, $x_E(0)=1$이고
   $x_E(D_1+D_2)=x_E(D_1)x_E(D_2)$이며 연속이다. 비자명성은 어떤
   $D>0$에서 $x_E(D)<1$이라는 뜻이다.
4. **E4:** $c_E>0$를 fold-to-energy depth 변환으로 두고 실제 깊이는
   $D_{\rm act}=c_E(1-x_E)D_A$다.

E3에 같은 함수형 정리를 독립 적용하면
$x_E(D)=e^{-\kappa_E D}$인 고유 rate $\kappa_E>0$를 얻는다. 이는 S1--S4의
$\kappa_{\rm surv}$와 자동으로 같은 상수가 아니다. 따라서

$$
x_E=e^{-\beta_E(1-x_E)D_A},
\qquad \beta_E:=\kappa_E c_E.
$$

$\kappa_E=1$은 energy-depth 좌표 convention일 뿐 $c_E$를 없애지 않는다.
추가 matching `Selection` $\beta_E=1$을 채택하고 저분율 branch를 선택하면
유일성으로 $x_E=x_\star$가 된다. $\beta_E=1$의 동역학적 validation은
`Open Bridge`다. E1--E2는

$$
\frac{\Omega_b}{\Omega_{\rm phys}}=x_E
$$

를 준다. 여기서 $\Omega_{\rm phys}:=\rho_{\rm tot}/\rho_c$는 곡률을
제외한다. flat-slice $\Omega_k=0$과
$1=\Omega_{\rm phys}+\Omega_k$를 추가할 때만

$$
\boxed{\Omega_b=x_E=x_\star}
$$

라고 쓴다. 이것은 현재 validation 상태가 `Open`인 `Bridge`이지 같은 $x$를
두 번 정의한 항등식이 아니다.

## 4. 암흑성분 패턴

관측상 $\Omega_{\rm cdm}/\Omega_{\rm DE}$는 약 $0.38$이다. 후보식을 여러 개
사후 비교한 뒤 가장 가까운 것을 택하면 look-elsewhere 효과가 생긴다.
최신 CE는 하나의 functional을 사전 고정한다.

$$
\boxed{R_{\rm dark}:=
\alpha_sD_A(1+x_\star\delta_A)=0.3782386966}.
$$

이 식은 `Phenomenology/Bridge`다. 완전한 Friedmann 장부는 neutrino의
중복 계상을 피한 convention에서

$$
\Omega_{\rm phys}:=\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm DE}
+\Omega_r+\Omega_\nu,
\qquad
1=\Omega_{\rm phys}+\Omega_k
$$

다. 따라서

$$
\Omega_{\rm rem}:=\Omega_{\rm phys}-\Omega_b-\Omega_r-\Omega_\nu
=1-\Omega_k-\Omega_b-\Omega_r-\Omega_\nu
$$

를 먼저 정의하고

$$
\Omega_{\rm cdm}=\Omega_{\rm rem}
\frac{R_{\rm dark}}{1+R_{\rm dark}},
\qquad
\Omega_{\rm DE}=\Omega_{\rm rem}\frac1{1+R_{\rm dark}}
$$

로 분할한다. $\Omega_r=\Omega_\nu=\Omega_k=0$을 선택한 late-time
truncated 3-sector benchmark에서만

$$
(\Omega_b,\Omega_{\rm cdm},\Omega_{\rm DE})
=(0.0486382585,0.2610881744,0.6902735671)
$$

가 된다. 합이 1인 것은 이 절단 안의 대수 검산이지 데이터 적합 또는
복사·중성미자를 포함한 완전한 flatness 증거가 아니다.

## 5. marginal 일치와 full likelihood의 차이

개별 밀도 중앙값만 보면 CE 벡터는 표준 우주론 값에 가깝다. 최종 판정에는
같은 parameter vector로 CMB·BAO·SN·성장 자료를 함께 계산해야 한다. 다만
현재 실제로 실행한 것은 그 공동 likelihood가 아니라, 로컬
`desi-dr2-all`의 13-component compressed DESI DR2 BAO mean/covariance만 쓴
**BAO-only partial gate**다. external-$r_d$ branch에는 위 late-time 3-sector
벡터를 넣는다. EH branch에서는 그 합 1 위에 radiation을 다시 더하지 않고

$$
\Omega_{{\rm rad},0}^{({\rm EH})}
:=\Omega_{\gamma,0}+\Omega_{\nu,{\rm rel},0}
=9.192332266\times10^{-5},
$$

$$
\Omega_{\rm rem}^{({\rm EH})}:=1-\Omega_b-\Omega_{{\rm rad},0}^{({\rm EH})},
$$

$$
\Omega_{{\rm cdm},0}^{({\rm EH})}
:=\Omega_{\rm rem}^{({\rm EH})}\frac{R_{\rm dark}}{1+R_{\rm dark}}
=0.2610629473,
\qquad
\Omega_{{\rm DE},0}^{({\rm EH})}
:=\frac{\Omega_{\rm rem}^{({\rm EH})}}{1+R_{\rm dark}}
=0.6902068709
$$

로 flat closure를 다시 맞춘다. 두 branch 모두
$w_0=-1,w_a=0$, $H_0=67.4$를 넣어 parameter fit 없이 13개 성분을 평가했다.
external branch의 저적색편이 배경은 복사를 버린 3-sector 절단이고, EH
branch에서는 sound horizon과 BAO 거리 모두에 같은 4-sector 식

$$
E_{\rm EH}^2(z)
=\Omega_{{\rm rad},0}^{({\rm EH})}(1+z)^4
+\bigl(\Omega_b+\Omega_{{\rm cdm},0}^{({\rm EH})}\bigr)(1+z)^3
+\Omega_{{\rm DE},0}^{({\rm EH})}
$$

을 쓴다. 따라서 EH branch에서는 $E_{\rm EH}(0)=1$이며 조기·후기 배경이
같은 closure를 공유한다. 적합한 파라미터가 0개이므로 dof는 13이다.
사전 고정 판정 규칙은 $p\geq0.05$이면 통과 구간,
$0.0027\leq p<0.05$이면 긴장 구간, $p<0.0027$이면 기각 구간이며,
마지막 구간을 validation 상태 `Rejected`로 기록한다.
이 결과는 이미 모형 감사와 문서 수정에 사용됐으므로 새 untouched
holdout으로 다시 세지 않는다.

두 registered background·$r_d$ branch의 explicit-canonical 결과는 다음과 같다.

| 설정 | $\chi^2$ | dof | $p$ |
|---|---:|---:|---:|
| external $r_d=147.09$ Mpc | 40.20145086 | 13 | $1.28283168\times10^{-4}$ |
| 4-sector EH $r_d=151.50842877$ Mpc | 41.90607733 | 13 | $6.78476334\times10^{-5}$ |

external $r_d$는 입력이고, 4-sector EH branch는
$T_{\rm CMB}=2.7255$ K,
$N_{\rm eff}=3.044$와 Eisenstein--Hu drag fit을 채택한 derived
`Selection`이며 precision recombination 결과가 아니다. 여기서
$\Omega_{{\rm rad},0}^{({\rm EH})}$는 장부의 photon과 relativistic-neutrino
항의 합이므로 neutrino를 다시 더하지 않는다. 두 고정 background
benchmark는 이 선언된 BAO-only partial gate에서 `Rejected`다. CMB·SN·growth를
포함한 공동 likelihood는 아직 실행하지 않았으므로 `Open`이다. 다음 확장은
공변 scalar-tensor stress와 perturbation을 같은 action에서 풀고, 새 파라미터
수의 penalty까지 포함해 그 공동 gate를 실제로 계산해야 한다.

이것은 모형을 “강등”한 결과가 아니라, 숫자 중앙값의 인상비평을 실제
likelihood 판정으로 완성한 결과다.

## 6. 결합상수의 교차 가설

Track B에서는 독립 $\alpha_{em}(M_Z)=1/127.95$를 넣어

$$
C_{\rm CE}:=\alpha_s+\alpha_w+\alpha_{em}=\frac1{2\pi}
$$

라는 같은-scale·same-scheme 경계 ansatz를 별도 `Selection`으로 둔다. 이를
한 변수 문제로 닫기 위해 다시

$$
s_{W,B}^2:=4(\alpha_s^{(B)})^{4/3},
\qquad
\alpha_{w,B}:=\frac{\alpha_{em}}{s_{W,B}^2}
$$

라는 matching을 조건부 채택한다. 이 matching의
validation 상태는 `Open Bridge`이므로 Track B 전체는 조건부 benchmark다.
그때

$$
\alpha_s^{(B)}
+\frac{\alpha_{em}}{4(\alpha_s^{(B)})^{4/3}}
+\alpha_{em}=\frac1{2\pi}
$$

를 푼다. 양의 근 두 개 가운데 SM-like hierarchy를 사전 선택하면

$$
\alpha_{s,{\rm SM}}^{(B)}=0.1173186647,
\quad s_{W,B}^2=0.2297291680,
\quad\alpha_{w,B}=0.0340207254
$$

를 얻는다. Track A와의 residual은 서로 다른 calibration input,
boundary·matching·truncation 가설과 scheme/RG 효과를 함께 진단하는 양이지
독립 성공이 아니다. 두 트랙의 입력을 섞어 더 좋은 숫자만 취하지 않는다.

## 7. flavour 패턴의 완성 방법

CKM 또는 PMNS의 원소마다 서로 다른 거듭제곱을 맞추는 방식은 unitary
행렬 하나를 보장하지 않는다. 최신 benchmark는 하나의 Yukawa matrix를
먼저 정의한다.

$$
Y_f=U_{fL}\operatorname{diag}(y_{f1},y_{f2},y_{f3})U_{fR}^\dagger.
$$

그 결과로

$$
V_{\rm CKM}=U_{uL}^\dagger U_{dL}
$$

를 얻는다. 최신 seed의 예는

$$
(s_{12},s_{23},s_{13})
=(0.22724210,0.04168209,0.00372494)
$$

이고 $\delta_q=1.2\,{\rm rad}$에서
$J=3.199594285\times10^{-5}$다. 세 sine seed와 $\delta_q$는
`Calibration input`인 benchmark이지 이 절의 예측이 아니다. $J$는 표준 CKM
매개화로 재계산한 조건부 `Calibration output`이다. 그 매개화로 만든 행렬의
unitarity는 해석적으로 정확하며, 부동소수점 residual은 구현·정밀도·norm을
정하지 않고 물리 숫자로 인용하지 않는다. 질량·혼합·CP의 실제 판정은 같은
covariance를 쓰는 별도 likelihood다.

## 8. strong CP 패턴의 올바른 보강

$e^{-8\pi^2/\alpha_s}$ 같은 식은 instanton convention부터 틀리며,
sector 억압만으로 $\bar\theta$를 0으로 만들지도 못한다. 올바른 semiclassical
action은 $2\pi/\alpha_s$이고, 해결 branch는 pseudoscalar $a_{\rm ax}$와

$$
\left(\bar\theta+\frac {a_{\rm ax}}{f_a}\right)G\widetilde G
$$

coupling을 둔다. QCD potential의 최소가 effective theta를 0으로 만든다.
nEDM과 axion search가 독립 holdout이다.

## 9. 인플레이션 패턴의 실제 계산

$n_s\simeq1-2/N$, $r\simeq12/N^2$가 관측과 가깝다는 사실만으로 CE
인플레이션이 되지 않는다. 이 절은 core의 minus-sign 장 $\phi$와 다른
inflation 장 $\varphi$를 쓰고

$$
F_{\rm inf}(\varphi)=M_P^2+\xi_{\rm inf}\varphi^2,
\qquad
\xi_{\rm inf}:=\alpha_s^{1/3}=0.4904868132
$$

인 plus-sign Jordan action을 선택한다. $\xi_{\rm inf}$의 등록식은
`Selection`이며, [A 강의 2절](A_연역적_유도.md#2-공변-eft)의
$\xi_{\rm core}$와 동일한 coupling이 아니다.
finite-$\xi_{\rm inf}$ 배경 적분 뒤 관측량에는 leading-order slow-roll 식을
쓴다.

$N_*=57.1999$, $A_s=2.10\times10^{-9}$인 quartic benchmark는

$$
n_s=0.96617114,
\qquad
r=0.00434561,
\qquad
\lambda_4=1.3434991\times10^{-10}
$$

을 준다. $A_s$는 $\lambda_4$ calibration이고, $n_s,r$는 지정 branch의
조건부 출력이다. reheating·RG·고차 slow-roll 보정은 `Open`이다.

## 10. 바리온 비대칭의 교훈

$\Omega_b$와 $\eta_b$의 수치가 함께 맞는다는 것만으로 생성 메커니즘이
생기지 않는다. 현재 밀도 변환은

$$
\eta_b^{\rm density}
=\frac{\Omega_b\rho_c}{m_bn_\gamma}
=6.0412\times10^{-10}
$$

을 주지만 $H_0,T_{\rm CMB},m_b$ 입력을 포함한다. 실제 생성은 CP-odd
Wilson operator, bubble bounce, transport와 sphaleron washout을 풀어야
한다. density 변환과 transport output의 일치가 closure test다.

## 11. 귀납 추론의 통계 계약

후보식이 $N$개였다면 최종 하나만 보고한 residual은 유효하지 않다.
최신 절차는 다음을 기록한다.

| 단계 | 기록 |
|---|---|
| 후보 생성 | 모든 식과 계수 범위 |
| 선택 | 선택 자료와 objective |
| calibration | 사용한 관측량·covariance |
| validation | 겹치지 않는 자료 |
| holdout | 분석 전 잠근 결과 |
| 복잡도 | 파라미터 수와 prior volume |

관측 벡터 $\mathbf d$, 이론 벡터 $\mathbf m(\theta)$에 대해

$$
\chi^2=(\mathbf d-\mathbf m)^TC^{-1}(\mathbf d-\mathbf m)
$$

을 사용한다. 동일 자료의 여러 marginal 표를 독립 측정처럼 합하지 않는다.

## 12. 귀납에서 연역으로 넘어가는 기준

다음 조건을 만족할 때 패턴은 계산 모형이 된다.

1. 차원과 기호 타입이 완결돼 있다.
2. action 또는 normalized measure가 있다.
3. EOM·stress·branch가 재현된다.
4. 모든 입력과 calibration이 공개된다.
5. 선택에 쓰지 않은 holdout이 있다.
6. 실패한 full likelihood도 그대로 기록한다.

이 기준으로 벡터 Poisson 모형의 공통 행합 고정점과 별도
finite-$\xi_{\rm inf}$ benchmark의 계산 절차는 재현 가능하다. 그러나
CE+SM action에서 next-generation 행렬을 얻는 단계와 $x_E\leftrightarrow
\Omega_b$는 `Open Bridge`이고, 고정 cosmology benchmark는 지정 BAO-only
partial gate에서 `Rejected`다. CMB·SN·growth 공동 gate는 `Open`이다. 귀납
루프가 닫혔다는 말은 이 성공·미완성·기각 상태를 모두
남겼다는 뜻이지 전체 이론이 증명됐다는 뜻이 아니다.
