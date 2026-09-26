# 11. 무한 Fock 비관측공간과 중력 중복도 — CE-AM6

## 11.1 계산 전 등록

**[입자 수 절단을 제거하는 후보]** 2026-09-10. CE-AM5의 열린 질량 판본
$M=6$, $m_\phi=1$, $m_\chi=4$, $g_3=0.4$, $\lambda_4=0.02$를 고정한다.
유한한 공간 격자를 유지하되 각 장의 점유 수와 SU(3) 링크의 표현은 절단하지
않는다. 이 공간 regulator 안에서 비관측 상태·정규화·Hamiltonian·관측 사상을
정의하고, 무한 Fock 상태가 무한한 내부 장 종류와 같은지 검사한다.

- 각 꼭짓점의 장 좌표는 $z=(q,\phi,\chi)\in\mathbb C^7$,
  실수 정준 좌표는 $z=(x+iy)/\sqrt2$다. 유한 그래프의 운동 공간을
  $L^2(\mathbb R^{14|V|}\times SU(3)^{|E|},dx\,d\mu_H)$로 둔다.
  Haar 측도는 링크마다 1로 정규화한다.
- 같은 onsite potential, 양의 공변 이웃 차분, 양의 gauge 전기 운동항과
  비음 plaquette potential을 공급한다. 격자 단위에서 계수는 양의 상수다.
  비음 이차형식을 닫아 선택하는 Friedrichs Hamiltonian의 자기수반성·보존을
  증명한다. 연속 공간 극한이나 기본 시공간의 유도를 주장하지 않는다.
- 두 꼭짓점·한 링크에서 차분 계수 1, seed 1731의 장·링크·국소 gauge 변환
  12개를 사용해 potential의 강제 하한과 링크 gauge 불변성을 검산한다.
- 한 셀의 $\chi$ 자유 oscillator 진공 사영을 $Q$로 읽는 관측 후보를 둔다.
  나머지 변수는 보존하고 $P=I-Q$다. 이 사영의 gauge 불변성과 두 결과 instrument를
  정의하되, 안정 장치나 에너지 보존 측정 상호작용을 유도했다고 하지 않는다.
- $q$의 입자·반입자 모드에
  $|\psi_r\rangle=\sqrt{1-r}\sum_{n\ge0}r^{n/2}|n,n\rangle_q
  \otimes|0\rangle_\phi\otimes|0\rangle_\chi$를 둔다.
  $r=1/4,1/2,3/4$에서 정규화·비관측 사영·유한 에너지를 증명한다.
  $n\le8,16,32$의 부분합과 정확한 꼬리를 독립 대조한다.
- 자유 한 셀의 14개 실수 oscillator 주파수는 $6$ 두 개, $1$ 여섯 개,
  $4$ 여섯 개다. $\beta=0.25,0.5,1,2$에서 vacuum 에너지를 뺀 Fock partition을
  oscillator 곱, 총 에너지 $80,160,320$ 이하의 정수 상태 계수, cycle 합
  $32,64,128$로 대조한다. 이는 상호작용·Gauss 사영 뒤 partition의 정확한 값이
  아니라 자유 운동 공간의 기준이다.
- 같은 일곱 복소 scalar의 영장 배경 Hessian에 대해
  $K_{\rm int}(s)=e^{-36s}+3e^{-s}+3e^{-16s}$를 쓴다.
  $s=0.01,0.1,1$과 proper-time cutoff $L=\Lambda^2=10,100,1000$,
  공통 곡률 결합 $\xi=0,1/6,1/3$을 고정한다.
  CE-GR1의 두 적분 치환과 닫힌 식으로 scalar 한 루프의 $R$ 계수를 비교한다.
  gauge·ghost·graviton loop는 이 scalar 블록 계산에 포함하지 않는다.
- 대수 허용오차 $10^{-10}$, proper-time 적분 상대 오차 $10^{-8}$,
  마지막 총 에너지 절단의 자유 partition 상대 오차 $10^{-10}$를 고정한다.
  불변성 표본이나 수치 합으로 무한 공간의 자기수반성 정리를 대신하지 않는다.
  관측 자료·피팅 없음.

## 11.2 같은 potential의 강제 하한과 무한 Hilbert 공간

**[정의·정리의 전제]** 격자 간격과 셀 부피를 고정하고 정준 좌표에 흡수한다.
그래프 $\mathcal G=(V,E)$는 유한하며, 장의 값과 링크 변수는 연속이다.
운동 공간은

$$
\mathcal H_{\rm kin}
=L^2\left(\mathbb R^{14|V|}\times SU(3)^{|E|},dX\prod_e d\mu_H(U_e)\right).
\tag{11.1}
$$

유한한 수의 좌표를 가진 $L^2$ 공간도 무한차원이다. 각 복소 scalar의 자유
oscillator 기저를 택하면 입자·반입자 두 bosonic 모드의 Fock 기저와 동등하다.
링크에도 $L^2(SU(3))$ 전체를 보유하므로 표현 차원의 절단이 없다. 이 기저
선택은 장 종류를 추가하지 않는다.

이전 potential을 다른 순서로 완전제곱하면

$$
V=\lambda_4\left|B-\frac{g_3}{\lambda_4}q\right|^2
+\left(M^2-\frac{g_3^2}{\lambda_4}\right)|q|^2
+m_\phi^2\phi^\dagger\phi+m_\chi^2\chi^\dagger\chi.
\tag{11.2}
$$

등록 값에서는 $M^2-g_3^2/\lambda_4=28$이므로

$$
V\ge28|q|^2+\phi^\dagger\phi+16\chi^\dagger\chi
\ge |q|^2+\phi^\dagger\phi+\chi^\dagger\chi.
\tag{11.3}
$$

이 부등식은 모든 장 값에서 성립한다. 앞선 leading Fock 모형의 한 입자·두
입자 부분공간에 국한되지 않는다. $z=(x+iy)/\sqrt2$이므로 실수 좌표에서는
아래쪽에 양의 harmonic oscillator potential이 놓인다.

링크 $e:x\to y$의 공간 차분은 양의 계수와 함께

$$
|q_y-q_x|^2+\|\phi_y-U_e\phi_x\|^2
+\|\chi_y-U_e^*\chi_x\|^2
$$

로 둔다. $U_e\mapsto G_yU_eG_x^\dagger$ 아래 불변이다. gauge 전기 운동항은
링크 Laplace–Beltrami 연산자의 양의 부호이고, plaquette potential은
$\kappa_B(3-\operatorname{Re}\operatorname{tr}U_\square)\ge0$이다.
이는 기존 Hamiltonian 격자 gauge 방식의 선택을 사용하는 것이다.
[Kogut–Susskind, 1975](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.11.395).

## 11.3 닫힌 이차형식으로 정의하는 보존 동역학

**[유한 공간 regulator에서의 조건부 증명]** onsite·차분·plaquette를 합한
비음 곱셈 potential을 $W(X,U)$로 쓴다. 양의 계수 $\kappa_E$와 링크의
Lie 미분 $\mathcal L_e^a$를 사용해

$$
\mathfrak h[\Psi]
=\frac12\|\nabla_X\Psi\|^2
+\kappa_E\sum_{e,a}\|\mathcal L_e^a\Psi\|^2
+\|W^{1/2}\Psi\|^2
\tag{11.4}
$$

를 정의한다. 정의역은 $\Psi\in\mathcal H_{\rm kin}$ 중 세 항이 유한한 함수의
교집합이다. 미분은 약한 미분으로 정의한다. 운동 미분과 곱셈 연산자가 닫혀
있으므로 $\|\Psi\|^2+\mathfrak h[\Psi]$의 norm에서 이 정의역은 완비다.
실수 장 좌표에서 compact support를 가진 매끄러운 함수와 매끄러운 링크 함수가
조밀하게 들어가므로 이차형식은 조밀 정의되고 닫혀 있다.

닫힌 비음 이차형식의 표현 정리에 따라 이에 대응하는 비음 자기수반 연산자
$H_{\mathfrak h}$가 존재한다. 여기서 선택한 것은 이 이차형식에 대응하는
Friedrichs Hamiltonian이다. 단순히 형식적 미분식을 쓴 뒤 정의역을 생략한
자기수반성 주장이 아니다. 따라서

$$
U(t)=e^{-itH_{\mathfrak h}},\qquad
\|U(t)\Psi\|=\|\Psi\|,
\qquad
\mathfrak h[U(t)\Psi]=\mathfrak h[\Psi]
\quad(\Psi\in D(\mathfrak h))
\tag{11.5}
$$

가 성립한다. 마지막 등식은 $H_{\mathfrak h}^{1/2}$의 spectral calculus로
따른다. 시간은 이 Hamiltonian에 공급한 연속 매개변수이며, 기본 비동기 틱이나
관측자의 proper time을 새로 유도한 것은 아니다.

(11.3)은 실수 장 좌표의 꼬리를 제어하고, 링크 공간은 compact하다. 유계한
장 좌표 영역에서 운동 미분의 Rellich compactness를 적용하고, 바깥쪽 norm을
$W\ge c|X|^2$로 억제하면 이차형식 정의역의 $L^2$ 매장이 compact하다.
따라서 이 유한 그래프 Hamiltonian에는 compact resolvent가 있으며, 무한한
상태 수가 있다는 사실과 바닥 없는 에너지 발산은 구별된다.

이 증명은 공간 격자와 couplings를 고정한 채 입자 수 절단을 제거한 결과다.
그래프의 무한 부피·격자 간격 0의 극한, Lorentz 대칭 복원, 비섭동적 연속 QFT와
양자중력의 존재를 증명하지 않는다.

## 11.4 Gauss 제약과 비관측 사영

**[물리 부분공간의 정의]** 꼭짓점마다 독립인 $G_x\in SU(3)$는 장 좌표와
링크에 작용한다. 실수 장 측도의 Jacobian은 1이고 Haar 측도는 좌우 불변이므로
이 작용은 $\mathcal H_{\rm kin}$ 위의 유니터리 표현 $\mathcal U(G)$가 된다.
같은 변환은 (11.4)의 모든 항과 그 정의역을 보존한다. 따라서

$$
\Pi_G=\int\prod_xd\mu_H(G_x)\,\mathcal U(G),\qquad
\Pi_G^2=\Pi_G=\Pi_G^\dagger,
\qquad [\Pi_G,U(t)]=0
$$

이고 $\mathcal H_{\rm phys}=\Pi_G\mathcal H_{\rm kin}$은 보존된다.
이는 finite graph의 Gauss 불변 부분공간이다. 모든 장의 gauge 불변 Gaussian과
상수 링크 함수의 곱은 정규화 가능한 예이므로 이 부분공간은 비어 있지 않다.

**[관측 후보]** $\chi$의 자유 기준 oscillator 진공에 대한 사영을 $Q$로 두고
다른 모든 인자에는 항등 연산자를 둔다. $P=I-Q$다. 이 진공 Gaussian은 국소
SU(3) 불변이므로 $Q$와 $P$는 $\Pi_G$와 가환한다. 따라서 물리 부분공간에서도

$$
\mathcal I_0(\rho)=Q\rho Q,\qquad
\mathcal I_1(\rho)=P\rho P,\qquad
\operatorname{tr}\mathcal I_0(\rho)+\operatorname{tr}\mathcal I_1(\rho)=1
\tag{11.6}
$$

이라는 완전양성 두 결과 instrument를 정의할 수 있다. $Q$를 ‘비관측’으로 읽는
것은 공급한 관측 대수의 선택이다. 상호작용 진공이나 안정한 검출 장치와 동일하다고
가정하지 않는다.

수학적 flag를 붙인
$\Psi\mapsto Q\Psi\otimes|0\rangle+P\Psi\otimes|1\rangle$는 등거리 사상이며
두 진폭을 함께 보존한다. 하지만 $Q$는 일반적으로 $H_{\mathfrak h}$와 가환하지
않는다. degenerate flag의 단순 제어 유니터리
$W_f=Q\otimes I+P\otimes\sigma_x$는 공통 Schwartz 정의역에서
$[W_f,H_{\mathfrak h}\otimes I]=[P,H_{\mathfrak h}]\otimes(\sigma_x-I)$를
남길 수 있다. 따라서 이 instrument의 존재를 에너지 보존 측정 장치나 실제 단일
결과의 유도로 승격하지 않는다. 에너지 공급·장치 동역학은 별도로 닫아야 한다.

## 11.5 무한 지지와 유한 에너지를 가진 비관측 상태

**[명시적 존재 증명]** 한 셀에서 $q$의 입자·반입자 oscillator 상태
$|n,n\rangle_q$를 사용한다. 이 상태들은 서로 직교하고 색깔 singlet이며,
global 위상 전하도 0이다. $\phi,\chi$는 각각 자유 기준 진공에 둔다.
등록한 $0<r<1$에서

$$
|\psi_r\rangle=\sqrt{1-r}\sum_{n=0}^{\infty}r^{n/2}|n,n\rangle_q
\otimes|0\rangle_\phi\otimes|0\rangle_\chi
\tag{11.7}
$$

는 무한히 많은 Fock 성분을 가진다. 그런데

$$
\langle\psi_r|\psi_r\rangle=(1-r)\sum_{n\ge0}r^n=1,
\qquad Q|\psi_r\rangle=|\psi_r\rangle,
\qquad \langle n\rangle=\frac r{1-r}<\infty.
$$

한 셀의 자유 진공 에너지는 $E_0=M+3m_\phi+3m_\chi=21$이다.
삼차항의 기대값은 $\phi,\chi$ 진공에서 0이고,
$\langle|B|^2\rangle=3/(4m_\phi m_\chi)$다. 여기서는 (11.2)의 좌표 potential을
그대로 양자화했으며 사차항을 normal order하여 이 진공 항을 지우지 않았다.
따라서 전체 한 셀 이차형식 에너지는 정확히

$$
\mathfrak h[\psi_r]
=21+12\frac r{1-r}+\frac{3\lambda_4}{4m_\phi m_\chi}.
\tag{11.8}
$$

| $r$ | $\langle n\rangle$ | $\|\psi_r\|^2$ | $Q$ 확률 | 전체 한 셀 에너지 |
|---|---:|---:|---:|---:|
| $1/4$ | $1/3$ | 1 | 1 | 25.00375 |
| $1/2$ | 1 | 1 | 1 | 33.00375 |
| $3/4$ | 3 | 1 | 1 | 57.00375 |

입자 수 $n\le N$의 부분합에서 norm 꼬리는 $r^{N+1}$이고, 에너지 꼬리는

$$
r^{N+1}\left[E_b+2M\left(N+1+\frac r{1-r}\right)\right],
\qquad E_b=21.00375
$$

다. 이 식은 무한 상태를 수치 절단으로 가정하는 대신, 절단 밖의 기여까지 정확히
통제한다. 다른 유한 그래프에는 나머지 꼭짓점의 불변 Gaussian과 상수 링크 함수를
곱해 확장할 수 있다. 양의 이웃 차분의 기대값도 유한하므로 form domain에 남는다.
이로써 $Q\mathcal H_{\rm phys}$가 무한차원이고 유한 에너지 상태를 포함함을
명시적으로 보였다. 이것이 자연의 실제 비관측 상태라는 경험적 판정은 별도다.

## 11.6 Fock trace와 내부 장 trace의 구별

**[두 trace의 정의]** 한 셀의 자유 운동 공간은 14개 실수 oscillator와 동등하다.
진공 에너지를 뺀 자유 Hamiltonian의 Fock partition은

$$
Z_{\rm Fock}(\beta)
=\frac1{(1-e^{-6\beta})^2(1-e^{-\beta})^6(1-e^{-4\beta})^6}<\infty
\qquad(\beta>0).
\tag{11.9}
$$

무한한 각 모드의 점유 수를 모두 합한 기하급수의 곱이다. 동등하게

$$
\log Z_{\rm Fock}(\beta)
=\sum_{\ell=1}^\infty\frac1\ell
\operatorname{tr}_{1p}e^{-\ell\beta h_0},\qquad
\operatorname{tr}_{1p}e^{-\beta h_0}=2e^{-6\beta}+6e^{-\beta}+6e^{-4\beta}.
\tag{11.10}
$$

이것은 Gauss 사영 전 자유 기준의 trace다. 상호작용하는 전체 Hamiltonian의
partition과 수치적으로 같다고 하지 않는다. 그러나 전체 trace의 존재도 고정
regulator에서 보일 수 있다. (11.3)의 harmonic 하한과 gauge 전기 운동항으로
만든 비교 연산자는 oscillator와 compact 군 Laplacian의 유한 곱이다. 그 heat
trace는 양의 $\beta$에서 유한하다. min–max 고유값 비교로
$\operatorname{Tr}e^{-\beta H_{\mathfrak h}}$도 그 값 이하이고, 보존되는 Gauss
부분공간의 trace는 더 작거나 같다.

검산에서는 자유 partition을 직접 oscillator 곱과 비교하는 것에 그치지 않았다.
정수 생성함수

$$
\prod_{\omega\in\{6,6,1,1,1,1,1,1,4,4,4,4,4,4\}}(1-x^\omega)^{-1}
=\sum_{E\ge0}d_Ex^E
$$

의 계수 $d_E$를 동적 계획법으로 세고 $\sum_{E\le E_{\max}}d_Ee^{-\beta E}$를
계산했다. 에너지 꼬리는
$e^{-\beta(E_{\max}+1)/2}Z_{\rm Fock}(\beta/2)$ 이하이고,
(11.10)의 $\ell>K$ 꼬리는

$$
\sum_\omega\frac{e^{-(K+1)\beta\omega}}{(K+1)(1-e^{-\beta\omega})}
$$

이하이다. 두 상계는 양의 항을 직접 비교해 얻으며 피팅하지 않는다. 마지막
$E_{\max}=320$의 상대 오차는 네 온도에서 최대 $4.441\times10^{-16}$이었다.

반면 scalar 한 루프 determinant의 내부 trace는 **한 입자 이차 연산자의 장 성분**에
대한 trace다. 영장 배경에서 potential의 Hessian은 일곱 복소 scalar 질량
$6,1,1,1,4,4,4$를 가지므로

$$
K_{\rm int}(s)=e^{-36s}+3e^{-s}+3e^{-16s},\qquad K_{\rm int}(0)=7.
\tag{11.11}
$$

14개의 실수 scalar를 쓴다면 determinant의 $1/2$ 계수가 같은 결과를 준다.
(11.9)는 여러 입자 점유의 thermal trace이고 (11.11)은 Gaussian 이차 연산자의
내부 trace다. $\beta$와 $s$의 차원도 각각 에너지의 역수와 역제곱으로 다르다.
한 루프 trace의 이런 정의는 표준 heat-kernel 전개의 출발점이다.
[Vassilevich, 2003, §1·3.1](https://arxiv.org/html/hep-th/0306138).

따라서 (11.11)에 무한 Fock 차원을 다시 곱하지 않는다. 이는 CE-GR1의 반례를
취소하는 재명명이 아니다. CE-GR1은 같은 유한 질량을 가진 내부 성분을 무한히
추가한 별도 후보였으며, 그 trace는 여전히 발산한다. 이번 ASM-01 판본은 유한한
장 종류의 점유 수를 늘리는 구조이고 Hamiltonian의 스펙트럼도 다르다.

## 11.7 유한 장 종류에도 남는 중력 계수의 한계

**[공급한 배경의 scalar 한 루프 비교]** 약한 곡률의 외부 Euclidean 계량에 공통
곡률 결합 $\xi R$을 허용한다. gauge·ghost·graviton loop는 포함하지 않은 scalar
블록에서

$$
\Gamma_R=-\frac{1/6-\xi}{16\pi^2}
\left[I(36,L)+3I(1,L)+3I(16,L)\right]\int\sqrt g\,R,
$$

$$
I(c,L)=\int_{1/L}^{\infty}\frac{e^{-cs}}{s^2}\,ds
=Le^{-c/L}-cE_1(c/L),\qquad L=\Lambda^2
\tag{11.12}
$$

를 얻는다. 이는 [CE-GR1의 동일한 proper-time 규약](../06_QFT_재설계/42_무한_보완공간의_곡률_응답_검사.md)에
새 장 내용을 넣은 것이다. Euclidean EH 부호를
$-M_{\rm ind}^2\int\sqrt g R/2$로 **비교 정의**하면

$$
M_{\rm ind}^2=\frac{2(1/6-\xi)}{16\pi^2}
\left[I(36,L)+3I(1,L)+3I(16,L)\right].
$$

| $L$ | $\xi=0$의 scalar $M_{\rm ind}^2$ | $M_{\rm ind}^2/L$ |
|---|---:|---:|
| 10 | 0.04990471151 | 0.004990471151 |
| 100 | 1.08664928112 | 0.010866492811 |
| 1000 | 13.9808570992 | 0.013980857099 |

각 유한 $L$에서 내부 중복도 발산은 없다. 그러나

$$
\lim_{L\to\infty}\frac{M_{\rm ind}^2}{L}
=\frac7{48\pi^2}\quad(\xi=0)
$$

이므로 시공간 UV cutoff를 제거한 유한 Planck 질량을 얻은 것은 아니다.
$\xi=1/6$이면 이 선형 $R$ 항은 0이고 $\xi=1/3$이면 부호가 반대다.
따라서 최소 결합을 공급한 경우의 양의 계수를 물리적 중력 척도의 고유예측으로
해석하지 않는다. 동일한 평탄 장 작용에 추가할 수 있는 곡률 결합·bare EH 항과
재규격화 조건은 여전히 선택해야 한다.

이 continuum heat-kernel 비교를 앞선 유한 공간 격자의 regulator 제거 증명으로
합치지도 않는다. 같은 장 종류를 사용하지만, (11.12)는 공급한 매끄러운 배경에서의
섭동적 scalar 블록이고 (11.4)는 고정 격자의 비섭동적 연산자 정의다.

## 11.8 검산과 연구 지위

**[수치 구현의 증거]** 실행 코드 (과거 기록: `verify/ce_infinite_fock_source.py`)와
산출 JSON (과거 기록: `verify/ce_infinite_fock_source.json`)은 두 꼭짓점의 12개 국소
gauge 변환에서 강제 하한·potential·공변 차분·기준 Gaussian 불변성을 검사했다.
세 무한 지지 상태의 유한 부분합과 정확한 norm·에너지 꼬리를 대조하고, 자유
partition을 oscillator 곱·정수 상태 계수·cycle 합으로 계산했다.

최대 대수 잔차는 $2.274\times10^{-13}$, 최종 자유 partition의 상대 오차는
$4.441\times10^{-16}$이다. 아홉 질량·cutoff 쌍의 proper-time 적분을 각각
두 치환과 256·512점으로 대조한 최대 상대 오차는 $9.769\times10^{-9}$로
등록 허용오차 $10^{-8}$ 안이다. 해시와 환경 판본을 저장했다. 수치 표본은
자기수반성·Gauss 보존·무한 상태 존재의 해석적 증명과 별개다.

**[조건부 진전]** 입자 수 절단 없이도 비관측 공간, 유한 에너지 정규화 상태,
자기수반 Hamiltonian과 관측 instrument를 한 유한 공간 격자 모형에서 정의했다.
이 모형의 유니터리 진화는 전체 진폭을 보존한다. 무한 상태 수와 무한 내부 장
종류를 구별하면 이전 종류 중복도의 발산을 이 새 판본에 자동으로 적용하지 않아도
되지만, 중력의 UV와 계수 선택 문제가 사라지는 것은 아니다.

**[미완성]** 이 비관측 사영과 격자·장 종류·작용은 후보 입력이다. 자연의 숨은
성분이 이 공간에 존재한다는 실험 증명, 공간 연속 극한과 시계의 기원, 자율적인
에너지 보존 검출·실제 결과 선택, 하나의 동적 양자 중력 작용과 전자기·약력·강력의
선택 및 공동 RMSE는 여전히 남는다. 따라서 OBJ-03의 수학적 구성 범위를 넓힌
결과이며 전체 네 목표의 완료 선언이 아니다.

관련 선행 문서: [국소 기록장과 문턱](10_국소_기록장의_공변_원천과_붕괴_문턱.md),
[유한 기준과 측정 에너지](07_에너지_보존_기록장치와_유한_기준상태.md),
[기록과 중력 원천](../06_QFT_재설계/56_비관측_기록의_중력_원천과_평균장_한계.md).
