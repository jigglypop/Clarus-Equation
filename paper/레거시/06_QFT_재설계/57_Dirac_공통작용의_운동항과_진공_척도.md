# 57. Dirac 공통작용의 운동항과 진공 척도 — CE-SA1

## 57.1 계산 전 등록

**[새 공통 작용 후보]** 2026-09-10. CE-MR1의 물질 표현을 공통 Dirac 연산자에
넣고, 같은 스펙트럼 함수가 중력과 세 gauge 운동항을 주는지 검사한다. 이는 기존
spectral action 방식의 특수 후보이며, CE-AM5 scalar 작용이나 그 한 루프
determinant가 이 작용을 유도했다고 가정하지 않는다.

- 시공간은 경계 없는 compact Euclidean spin 4-manifold, 내부 공간은 CE-MR1의
  한 세대 15성분을 세 번 복사한 $N_F=45$차원이다. hypercharge는
  $Y=(1,-4,2,-3,6)/6$, 나머지 발생자는 기존 표준 정규화를 유지한다.
- 질량·Yukawa·Higgs 없이 $D_{g,A}=i\gamma^\mu(\nabla^S_\mu+A_\mu)$를 쓴다.
  $A_\mu$는 anti-Hermitian gauge 연결이다. bosonic 작용을
  $S_B=\kappa\operatorname{Tr}(e^{-D_{g,A}^2/\Lambda^2})/2$로 고정한다.
  $\kappa,\Lambda>0$, 계산은 $\kappa=\Lambda=1$ 또는 이들을 나눈 값이다.
  $1/2$는 parity-even Weyl 수를 세는 정의이며 chiral 양자 측도의 증명이 아니다.
- 독립적인 진공항·Einstein–Hilbert 항·gauge kinetic 항은 추가하지 않는다.
  $a_0,a_2$와 $a_4$의 gauge 부분을 같은 Dirac heat trace에서 구하고, 세 힘의
  정준 결합비와 중력·진공 척도의 관계를 기록한다.
- Euclidean 4×4 Clifford 행렬 항등식, 12개 물질 발생자의 이차 trace,
  12개 단일 곡률과 seed 1741의 혼합 곡률 4개에서 spin trace의 gauge 항을
  독립 수축한다. scalar curvature 대조값은 $R=-0.3,0,0.2,1$이다.
- $A=0$인 반지름 $a$의 round $S^4$에서 알려진 Dirac 스펙트럼의 정확한 heat
  합과 반지름 미분을 계산한다. $z=a\Lambda=0.5,1/\sqrt2,1,2,5,10,20$,
  스펙트럼 정수 $m\le128,256$을 고정한다. 생략 꼬리는 Gaussian 적분 상계로
  통제한다.
- 큰 $z$에서 정확한 합을 $a_0+a_2+a_4$ 전개와 대조하고, $a_0+a_2$가 예측하는
  round-sphere 정지점이 전체 작용에서도 정지점인지 검사한다.
  $z=5,10,20$에서 첫 생략항의 $z^{-2}$ scaling을 확인한다.
- 반지름 log 미분은 정확한 spectral 미분과 간격 $10^{-3},5\times10^{-4},
  2.5\times10^{-4}$의 중앙 차분으로 대조한다. 대수 허용오차 $10^{-10}$,
  마지막 차분의 상대 오차 $10^{-5}$, 꼬리 상대 상계 $10^{-10}$을 고정한다.
- 이 후보의 근사 운동항 존재를 Lorentzian 진화·양의 양자 중력 Hilbert 공간,
  원래 세 투영의 동적 유도, 비관측 기록의 구현이나 공동 RMSE로 승격하지 않는다.
  관측 자료·피팅 없음.

## 57.2 같은 연산자의 저곡률 전개

**[정의·기존 방법]** 이 후보는 Chamseddine–Connes의
[spectral action](https://arxiv.org/abs/hep-th/9606001)을 특정 heat profile에
제한한다. spin 기하, 연결, 물질 표현, 세대 수와 profile은 입력이다. 같은 $D$로
fermion 운동항을 쓸 수 있다는 사실만으로 chiral 양자 측도나 비관측 기록의
상태공간까지 구성되지는 않는다.

연결 곡률을 $F_{\mu\nu}=[\nabla_\mu,\nabla_\nu]_{\rm gauge}$,
Hermitian 곡률을 $\mathcal F_{\mu\nu}=iF_{\mu\nu}$로 둔다. spin 곡률과
gauge 곡률을 구분하면 Lichnerowicz 식과 Laplace형 표기는

$$D^2=-\nabla^2+\frac R4-\frac12\gamma^{\mu\nu}F_{\mu\nu}
=-(\nabla^2+E),\qquad
E=-\frac R4+\frac12\gamma^{\mu\nu}F_{\mu\nu},\qquad
\gamma^{\mu\nu}=\frac12[\gamma^\mu,\gamma^\nu].$$

Euclidean Clifford trace는

$$\operatorname{tr}_S I=4,\quad
\operatorname{tr}_S\gamma^{\mu\nu}=0,\quad
\operatorname{tr}_S(\gamma^{\mu\nu}\gamma^{\rho\sigma})
=4(\delta^{\mu\sigma}\delta^{\nu\rho}
-\delta^{\mu\rho}\delta^{\nu\sigma}).$$

[표준 heat 계수](https://arxiv.org/abs/hep-th/0306138)의 $E+R/6$과
$E^2/2+\Omega_{\mu\nu}\Omega^{\mu\nu}/12$에 대입한다. parity-even
$1/2$를 포함한 $a_2$는 내부 성분당 $-R/6$이다. gauge 부분은
$-\operatorname{tr}_{\mathcal R}F_{\mu\nu}F^{\mu\nu}/3
=\operatorname{tr}_{\mathcal R}\mathcal F_{\mu\nu}\mathcal F^{\mu\nu}/3$이다.
따라서 같은 작용의 전개는

$$S_B\sim\frac{\kappa}{16\pi^2}\int d^4x\sqrt g\left[
2N_F\Lambda^4-\frac{N_F}{6}\Lambda^2R
+\frac13\operatorname{tr}_{\mathcal R}
\mathcal F_{\mu\nu}\mathcal F^{\mu\nu}
+\frac{N_F}{720}(5R^2-8R_{\mu\nu}R^{\mu\nu}
-7R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma})
+O(\Lambda^{-2})\right].$$

경계 없는 공간에서 적분되는 총미분은 생략했다. 이것은 곡률·장 변화의 척도가
$\Lambda$에 비해 작은 영역의 점근 전개다. 뒤의 항을 버린 작용의 해가 원래
작용의 해인지 별도로 확인해야 한다. 양의 gauge 항과 Euclidean EH 부호가
나온다는 결과를 Lorentzian 전체 이론의 안정성으로 확대하지 않는다.

**[조건부 계수]** $S_{\rm EH}=\int\sqrt g(\rho_{\rm vac}-M_{\rm EH}^2R/2)$와
비교하면

$$M_{\rm EH}^2=\frac{\kappa N_F\Lambda^2}{48\pi^2},\qquad
\rho_{\rm vac}=\frac{\kappa N_F\Lambda^4}{8\pi^2},\qquad
\frac{\rho_{\rm vac}}{M_{\rm EH}^2\Lambda^2}=6.$$

이는 중력계수와 진공계수를 함께 정하는 관계다. $M_{\rm EH}$를 관측 Planck
질량과 이미 일치시켰다는 뜻은 아니다.

## 57.3 세 gauge 운동항의 상대 정규화

**[조건부 산출]** CE-MR1의 한 세대 표현에서 $Y$의 이차 trace는

$$k_Y=6(1/6)^2+3(-2/3)^2+3(1/3)^2
+2(-1/2)^2+1^2=\frac{10}{3}.$$

$\operatorname{tr}_{\rm fund}T_aT_b=\delta_{ab}/2$를 쓰면 weak doublet은
색깔 세 개와 lepton 하나이므로 $k_2=(3+1)/2=2$, color fundamental 또는
antifundamental은 weak 두 성분과 두 singlet이므로 $k_3=(2+1+1)/2=2$다.
다른 부문 사이 이차 trace는 0이다. 세 세대를 합치면

$$\operatorname{tr}_{\mathcal R}T_aT_b=k_a\delta_{ab},\qquad
(k_Y,k_2,k_3)=(10,6,6).$$

연결에 결합을 흡수한 곡률 성분 $f^a_{\mu\nu}$의 정준 계수를
$\sum_a f^a_{\mu\nu}f^{a\mu\nu}/(4g_a^2)$로 정의하면

$$\frac1{g_a^2}=\frac{\kappa k_a}{12\pi^2},\qquad
g_2^2=g_3^2=\frac53g_Y^2,\qquad
\sin^2\theta_W=\frac{g_Y^2}{g_Y^2+g_2^2}=\frac38.$$

이 식은 같은 cutoff에서 정의한 bare matching 관계다. 저에너지 관측값과
비교하려면 threshold와 RG 진화를 고정해야 한다. $3/8$은 관측 weak 각도의
직접 예측이 아니며, SU(5) 군이나 원래의 세 각도 구조를 유도한 결과도 아니다.
표현을 입력했기 때문에 그 이차 trace가 결합비에 나타난다. 같은 $\kappa$를
제거한 추가 관계는

$$\frac{M_{\rm EH}^2g_2^2}{\Lambda^2}
=\frac{N_F}{4k_2}=\frac{15}{8}.$$

## 57.4 절단 작용의 진공과 정확한 구면 합

**[근사의 적용범위 반례]** $a_0+a_2$만 남긴 진공 Einstein 식은

$$R=\frac{4\rho_{\rm vac}}{M_{\rm EH}^2}=24\Lambda^2.$$

이 곡률은 저곡률 전개의 영역 안에 있지 않다. round $S^4$의 반지름을 $a$,
$z=a\Lambda$라 하면 $R=12/a^2$이므로 절단 작용의 정지점은
$z=1/\sqrt2$다. 이를 근사 전 작용으로 직접 검사한다.

[구면 Dirac 스펙트럼](https://arxiv.org/abs/gr-qc/9505009)은
$\pm(n+2)/a$, $n\ge0$이며 각 부호의 중복도는
$4\binom{n+3}{3}$이다. 두 부호와 작용의 $1/2$를 함께 세고 $m=n+2$로
바꾸면 내부 성분당 정확한 합은

$$\mathcal S(z)\equiv\frac{S_B}{\kappa N_F}
=\frac23\sum_{m=2}^{\infty}(m^3-m)e^{-m^2/z^2},$$

$$\frac{d\mathcal S}{d\log z}
=\frac{4}{3z^2}\sum_{m=2}^{\infty}(m^5-m^3)e^{-m^2/z^2}>0.$$

모든 양의 유한 $z$에서 미분이 양수다. 따라서 round sphere 반지름의
정지점은 없다. 실제 절단 정지점에서

| 양 | $z=1/\sqrt2$의 값 |
|---|---:|
| $R/\Lambda^2$ | 24 |
| $a_0+a_2$의 $d\mathcal S/d\log z$ | 0 |
| 정확한 $\mathcal S$ | 0.00134209419179 |
| 정확한 $d\mathcal S/d\log z$ | 0.0214783806865 |

$a_0+a_2$는 그 점에서 $\mathcal S=-1/12$, $a_4$까지 포함해도 $-1/45$를
주므로 양의 정확한 heat trace와도 다르다. 반면 큰 반지름에서는

$$\mathcal S(z)\sim\frac{z^4-z^2}{3}+\frac{11}{180}
+\frac{31}{3780z^2}+O(z^{-4}).$$

unit $S^4$의 $R=12$, $R_{\mu\nu}^2=36$, $R_{\mu\nu\rho\sigma}^2=24$를
국소 $a_4$에 넣어 적분해도 $11/180$을 얻는다. 정확한 합에서 $a_4$까지
뺀 나머지는 $z=5\to10\to20$에서 각각 0.2474927413, 0.2493783324배가
되어 첫 생략항의 $z^{-2}$ 거동과 일치한다. 따라서 실패 원인은 저곡률 계수의
부재가 아니라, 이 후보의 진공을 그 근사 영역에서 얻지 못한다는 점이다.

## 57.5 일반적인 전체 크기 변분의 장애

**[조건부 정리]** 경계 없는 compact spin 4-manifold에서 이 질량 없는
Dirac heat 작용만 쓰고 부피를 자유롭게 변분하면, bosonic 진공의 정지 계량은
존재하지 않는다.

**[유도]** 임의의 기준 계량 $g$와 고정 gauge one-form $A$에 대해
$g_a=a^2g$로 둔다. 자연스러운 spinor Hilbert 공간 식별 아래
$D_{g_a,A}=a^{-1}D_{g,A}$이므로

$$\frac{dS_B[g_a,A]}{d\log a}
=\frac{\kappa}{a^2\Lambda^2}\operatorname{Tr}\left[
D_{g,A}^2e^{-D_{g,A}^2/(a^2\Lambda^2)}\right]>0.$$

elliptic Dirac 연산자는 무한히 많은 영이 아닌 고유값을 갖는다. 그 항들은
모두 양수이고 heat 감쇠로 합과 미분이 수렴한다. 영모드는 상수만 더해 이
미분을 상쇄하지 못한다. 따라서 허용된 전체 크기 변분 하나에서 이미 1차
변분이 0이 아니므로, 모든 계량 변분에 정지할 수 없다. 이는 구면 수치 검사의
일반화이며, 특정 반지름 표본만으로 전체 공간을 추정한 결과가 아니다.

이 정리는 고정 부피, 경계 또는 비compact 공간, 추가 진공 반대항, 질량·Higgs
배경, 다른 profile, Lorentzian 작용에 그대로 적용되지 않는다. 그런 요소를
추가하려면 다른 가정 판본으로 등록해야 한다. 특히 진공항을 사후에 빼고도
‘독립 진공항 없이 같은 작용으로 해결했다’고 판정할 수 없다.

## 57.6 검산과 후보 판정

**[수치 검산]** 재계산 코드 (과거 기록: `verify/ce_dirac_spectral_action.py`)와
JSON 영수증 (과거 기록: `verify/ce_dirac_spectral_action.json`)은 Clifford 전수
항등식, 발생자 Gram 행렬, 단일·혼합 곡률의 행렬 trace, 국소 구면 $a_4$,
정확한 스펙트럼 합과 독립 중앙 차분을 기록한다. 코드·의존 파일 SHA256과
Python·NumPy 판본을 함께 동결한다.

스펙트럼을 $m\le N$에서 자를 때, 등록된 $N>\sqrt{5/2}\,z$에서 단조
감소하는 Gaussian 적분으로 다음 꼬리 상계를 얻는다.

$$0\le\mathcal S-\mathcal S_N
\le\frac{z^2(N^2+z^2)}3e^{-N^2/z^2},$$

$$0\le\dot{\mathcal S}-\dot{\mathcal S}_N
\le\frac23(N^4+2N^2z^2+2z^4)e^{-N^2/z^2},
\qquad \dot{\mathcal S}=\frac{d\mathcal S}{d\log z}.$$

최대 대수 잔차는 $2.6646\times10^{-15}$, 마지막 중앙 차분의 최대 상대
오차는 $8.7084\times10^{-6}$, $N=128$의 최대 상대 꼬리 상계는
$1.4346\times10^{-15}$다. 등록한 허용치 안에서 $N=128,256$ 합의 일치와
정확한 미분의 양수를 확인했다. 차분 오차와 대수 잔차는 서로 다른 검사이며,
수치 통과 자체가 물리적 실재의 증거는 아니다.

**[생존하는 좁은 주장]** 주어진 spin 기하와 물질 표현에서 하나의 Dirac
heat 작용이 저곡률 EH 항과 세 gauge 운동항을 함께 주며, 상대 정규화를
조건부로 정한다. 이 구조는 기존 spectral action 방법의 명시적 특수 계산이다.

**[기각하는 주장]** 독립 진공항 없이 이 양의 heat 작용만으로 compact
Euclidean 진공까지 자기일관적으로 얻었다는 주장은 전체 크기 변분에서 실패한다.
CE-SA1을 완성된 공통장 이론으로 올리지 않는다.

**[미완성]** Lorentzian 양자 진화와 GR 극한, 게이지군·키랄 물질·Higgs·질량의
상태 기원, CE-AM6의 비관측 Hilbert 공간 및 기록 동역학과의 단일 작용 결합,
RG를 포함한 공동 관측 RMSE는 아직 구성하지 않았다. 이번 후보에는 관측 자료나
피팅이 없으므로 VAL-01의 공동 개선 판정도 없다.
