# 10. 농축·커널·Born 수학 감사

## 1. 계산 전 판정 계약 — CE-MATH-E1

2026-09-11. [수학 도구 원장](../../검증_원장/수학_도구_확정_원장.md)의
M-05와 M-06을 위한 추가 감사다. 아래 항목의 정의·가정·반증 조건을
먼저 고정하고, 다음 절에 증명과 정확 반례를 기록한다.

| 번호 | 입력과 검토 명제 | 반증 또는 범위 축소 조건 |
|---|---|---|
| E1 | Polish 공간, l.s.c. good-rate 비용, 확률 prior, 양의 recovery mass에서 농축 | 최소값·정규화·유한 gap 처리가 누락되면 보완 |
| E2 | Gamma liminf·equicoercivity와 prior mass의 역할, 움직이는 recovery | Gamma만으로 확률 농축이 실패하는 명시적 함수열; 충분조건을 필요조건으로 읽지 않음 |
| E3 | 고정 끝점 Sobolev 경로, Gaussian prior, Brownian 비교 | topology 불일치, 영 분모, 확률급수의 존재를 전제한 순환 증명 |
| E4 | mesh 보간·고정 온도·온도와 mesh의 공동 극한 | 국소 균등 오차를 전역 오차로 올리거나 임의의 극한 교환을 주장하면 기각 |
| E5 | 유한 양의 row 커널 합성·정규화·tropical 경계 | row별 정규화와 전체 상태 정규화 혼동; 무한대 차·잘못된 범주 |
| E6 | joint·조건부·잔류 readout, bounded continuous 관측 | 영확률 조건화, 유일성 없는 Dirac 주장, raw 잔류의 무조건 보존 |
| E7 | 모든 유한 refinement를 연결하는 B0–B5 아래 Born 유일성 | B4 없이도 결론이 따른다는 주장; 순환 물리 유도 |
| E8 | Gibbs 확률과 선형 CP instrument의 관계 | 조건부 비선형 사상을 무조건 CPTP라고 부르는 주장 |

새 경험 자료·피팅·물리 가정은 추가하지 않는다. 일반 증명은 유한 행렬
검산과 구별한다. 표준 보조정리의 사용은 의존도를 명시하며 공리로부터의
proof-assistant 형식 증명이라고 부르지 않는다.

## 2. 정리와 반례

### E1. good-rate 농축의 정확한 가정

**[정리]** Polish 공간 $X$, Borel 확률 $\mu$, $S=\operatorname{supp}\mu$,
l.s.c. 함수 $E:X\to[0,\infty]$를 택한다. 모든 유한 sublevel이 compact,
$m=\inf_SE<\infty$, 모든 $\eta>0$에 대해
$r_\eta=\mu\{E<m+\eta\}>0$라고 하자. $\beta>0$에서
$0<Z_\beta=\int e^{-\beta E}d\mu\le1$이고
$M=\{x\in S:E(x)=m\}$는 공집합이 아닌 compact 집합이다.
모든 열린 $U\supset M$에 대해 $\mu_\beta(U)\to1$이다.
유일 최소점일 때만 이 결론에서 해당 Dirac 약수렴이 따라온다.

**증명.** 최소화열은 결국 compact $\{E\le m+1\}$ 안에 들어간다.
닫힌 $S$와 l.s.c.를 사용하면 그 부분열의 극한에서 최소값이 달성된다.
$M=S\cap\{E\le m\}$는 compact다. $F=S\setminus U$가 비면 끝이다.
$F$에 높이가 $m$으로 가는 열이 있으면 같은 논증으로 $F$ 안의
최소점이 생겨 모순이다. 따라서 어떤 **유한** $\delta>0$에 대해
$E|_F\ge m+\delta$다. $\inf_FE=\infty$인 경우에도 임의의 유한
$\delta$를 택하면 된다. $r_{\delta/2}>0$로

$$
Z_\beta\ge e^{-\beta(m+\delta/2)}r_{\delta/2},\qquad
\mu_\beta(F)\le r_{\delta/2}^{-1}e^{-\beta\delta/2}\to0.
$$

Polish 공간의 Borel 확률은 $S$ 바깥 질량이 0이다. 이는 가산 기저의
영질량 열린집합 합집합으로 $X\setminus S$를 덮어 확인한다.
유일 최소점 $x_*$이면 bounded continuous $f$를 그 작은 근방과
여집합으로 나누어
$|\int f\,d\mu_\beta-f(x_*)|\le\epsilon+2\|f\|_\infty\mu_\beta(U^c)$를
얻는다. $\square$

**[반례: recovery 생략]** $X=[0,1]$, 균등 $\mu$,
$E(0)=0$, $E(x)=1$ for $x>0$는 full support, l.s.c., compact sublevel,
유일 최소점을 모두 만족한다. 그러나 $Z_\beta=e^{-\beta}$,
$\mu_\beta=\mu$이므로 농축하지 않는다. 낮은 한 점은 영질량이다.

**[정의역 보완]** $W$의 support 최소값을 빼면 $W-m\ge0$는 $S$ 위의
명제다. $X\setminus S$의 더 낮은 값까지 비음수라고 주장하지 않는다.
닫힌 Polish 부분공간 $S$로 제한하여 E1을 적용하면 충분하다.
재가중 $Z_W^{-1}e^{-W}\mu$에는 반드시 $0<Z_W<\infty$가 필요하다.
$W=0$ at $0$, $W=\infty$ elsewhere인 위 prior에서는 $Z_W=0$이다.

### E2. Gamma 수렴과 recovery 속도

**[정리]** 고정 Polish 공간에서 비음수 Borel $E_n$과 확률 $\mu$를
택한다. $E_n$이 equicoercive이고 $E_n\xrightarrow{\Gamma}E_0$이며,
$E_0|_S$의 유일 최소점 $x_*$와 유한 최소값 $m$이 있다고 하자.
모든 $\eta>0$에 대해 Borel 집합 $V_{n,\eta}$가 결국

$$
\sup_{V_{n,\eta}}E_n\le m+\eta,\quad
r_{n,\eta}=\mu(V_{n,\eta})>0,\quad
\beta_n^{-1}\log(1/r_{n,\eta})\to0
$$

를 만족하고 $\beta_n\to\infty$이면 $\mu_n\Rightarrow\delta_{x_*}$다.

**증명.** $U\ni x_*$를 고정한다. 만일 $S\setminus U$ 위의 uniform
gap이 없으면 부분열 $n_j$와 $x_j\in S\setminus U$를
$E_{n_j}(x_j)\le m+1/j$로 잡을 수 있다. Equicoercivity로
부분열이 $x\in S\setminus U$로 수렴한다. Gamma liminf를 부분열에
적용하면 $E_0(x)\le m$, 유일성에 모순이다. 부분열 liminf는 빠진
지수에 $x$를 넣어 전체 수렴열을 만들면 원 정의에서 따른다.
따라서 유한 $\delta>0$에 대해 결국 $E_n|_{S\setminus U}\ge m+\delta$다.
$\eta=\delta/2$이면

$$
\log\mu_n(U^c)\le-\beta_n\delta/2+\log(1/r_{n,\delta/2})\to-\infty.
$$

분모도 이 recovery로 양수다. E1의 test-function 논증으로 끝난다.
이 증명은 Gamma **liminf**를 쓰며 점 recovery 자체 대신 양질량
recovery를 요구한다. $\square$

고정 양질량 집합을 주는 02c의 가정 G는 이 정리의 충분조건이다.
연속 $E_0$, 국소 균등 수렴, equicoercivity인 유클리드 경우에는
최소점 근처 작은 닫힌 공 안에서 $\sup_V E_n\le m+\eta$인 열린
근방을 잡아 같은 결과를 얻는다. $\beta_n$ 곱 오차가 0일 필요는 없다.
02b의 움직이는 jet 중심과 고정 양정치 가중치도 이 조건을 만족한다.

**[연속 함수열 반례]** $X=[0,1]$, 균등 prior,
$a_n=e^{-n^2}$, $\beta_n=n$,

$$
E_n(x)=\min(1,x/a_n)
$$

를 택한다. 각 $E_n$은 연속이고 compact 공간에서 equicoercive다.
Gamma 극한은 $E_0(0)=0$, $E_0(x)=1$ for $x>0$다.
$x>0$로 가는 열은 결국 $a_n$보다 크고 에너지가 1이다.
$x=0$에서 liminf는 비음성, recovery는 상수 0 열로 확인한다.
그러나

$$
Z_n=\frac{a_n(1-e^{-n})}{n}+(1-a_n)e^{-n},\qquad
\mu_n([0,a_n])\le\frac{a_ne^n}{n(1-a_n)}\to0.
$$

나머지 구간에서 밀도는 상수이므로 $\mu_n$은 균등측도로 약수렴한다.
유일 Gamma 최소점으로도 농축하지 않는 정확 반례다.

**[충분조건의 한계]** 두 후보 $a,b$에 $E(a)=0,E(b)=1$,
prior odds $\mu_n(a)/\mu_n(b)=e^{-c\beta_n}$, $0<c<1$을 주면
Gibbs odds는 $e^{(1-c)\beta_n}\to\infty$다. 따라서 $a$에 농축하지만
$\beta_n^{-1}\log(1/\mu_n(a))\to c\ne0$다. Subexponential recovery는
보편적으로 쓰기 편한 **충분조건**이며 필요한 조건은 아니다.
특정 바깥 gap $\delta$에 대해서는
$\beta_n(\delta-\eta_n)+\log r_n\to+\infty$라는 직접 bound로 충분하다.

### E3. 경로 공간과 prior를 동시에 닫는 한 패키지

**[정리]** $X=C^0_{x_i,x_f}([0,1];\mathbb R^d)$에 sup norm을 주고,
$H=H^1_0([0,1];\mathbb R^d)$와 직선 $\ell(t)=(1-t)x_i+tx_f$를 택한다.

$$
E(\gamma)=
\begin{cases}
\frac12\int_0^1|\dot\gamma|^2dt,&\gamma\in\ell+H,\\
\infty,&\text{그 밖}
\end{cases}
$$

는 $X$에서 l.s.c.이고 compact sublevel을 가진다. 최소점은 $\ell$이다.
$H$의 정규직교 기저 $e_j$, $\lambda_j>0$, $\sum_j\lambda_j<\infty$,
독립 표준정규 $\xi_j$에 대해
$\mu=\operatorname{Law}(\ell+\sum_j\sqrt{\lambda_j}\xi_je_j)$는
$H^1$에 집중하고 $X$에서 full support이며 E1의 recovery를 만족한다.
따라서 이 명시적 **수학 모형**의 Gibbs 측도는 $\delta_\ell$로 농축한다.

**증명: 에너지와 topology.** 유계 kinetic sublevel은
$|\gamma(t)-\gamma(s)|\le\|\dot\gamma\|_2|t-s|^{1/2}$와 끝점 고정으로
균일 유계·등연속이다. Arzelà–Ascoli로 균일 수렴 부분열이 있다.
도함수는 유계 $L^2$ 열이므로 약수렴 부분열을 잡는다. 분포 미분의
정의에서 극한 도함수는 균일 극한 경로의 도함수이며, $L^2$ norm의
약한 하반연속성으로 energy liminf를 얻는다. 일반 liminf가 유한한
열에도 이 논증을 적용하므로 $X$에 무한대로 확장한 $E$가 l.s.c.다.
특히 극한 경로는 $H^1$에 남아 compactness가 완성된다.

$\gamma=\ell+h$이면 $\int\dot h=0$이므로
$E(\ell+h)=|x_f-x_i|^2/2+\|\dot h\|_2^2/2$다. 최소점은 $h=0$뿐이다.

**증명: prior의 존재와 support.** 먼저 비음수 급수
$Y=\sum_j\lambda_j\xi_j^2$를 정의한다. 단조수렴으로
$\mathbb EY=\sum_j\lambda_j<\infty$이므로 $Y<\infty$ a.s.다.
정규직교성에 의해 부분합들이 a.s. $H$에서 Cauchy이므로 급수의
극한이 존재한다. 가측 부분합의 극한이므로 Borel 확률을 준다.
임의의 $h\in H$, $r>0$에 대해 큰 $k$를 잡아
$\|h-h_{\le k}\|_H<r/4$, $\sum_{j>k}\lambda_j<r^2/8$로 만든다.
Markov 부등식으로 tail norm이 $r/2$보다 작을 확률은 양수다.
첫 $k$개 계수는 양의 밀도의 $\mathbb R^k$ Gaussian이므로
$h_{\le k}$의 $r/4$ 근방을 양의 확률로 맞힌다. 독립성으로 두 사건의
교집합도 양수다. 따라서 모든 $H$ 열린 공에 양의 질량을 준다.
연속 포함 $H^1\hookrightarrow C^0$와 polygonal 경로의 조밀성으로
$X$에서도 full support다. $E$가 $H^1$ norm에서 연속이므로
최소점의 충분히 작은 $H^1$ 공은 모든 $\{E<m+\eta\}$ 안에 들어간다.
따라서 recovery가 성립하고 E1을 적용한다. $\square$

**[Brownian 분모 반례]** 분산 $\sigma>0$의 Brownian bridge는
$\ell+\sqrt{\sigma}(B_t-tB_1)$로 표현된다. Dyadic 분할의 Brownian
제곱증분 합 $Q_n$은 $\mathbb EQ_n=1$,
$\operatorname{Var}Q_n=2^{1-n}$이다. Chebyshev와 Borel–Cantelli로
$Q_n\to1$ a.s.다. 선형 보정의 제곱증분 합은
$B_1^2/2^n\to0$이고 교차항은 Cauchy–Schwarz로 0으로 간다.
따라서 bridge의 제곱증분 합은 성분당 $\sigma$다.
연속 유한변동 경로의 합은
$\max|\Delta\gamma|\operatorname{Var}(\gamma)\to0$이므로
Brownian bridge는 유한변동일 수 없다. $W^{1,p}\subset W^{1,1}$
for $p\ge1$이므로 그 공간의 질량은 0이다.
위 kinetic energy로 재가중하면 $\beta>0$에서 $Z_\beta=0$이다.
Brownian prior에 kinetic action을 다시 넣는 구성은 이 방식으로 정의되지 않는다.

기존 05f의 일반 Tonelli 모형은 $C^0$ l.s.c.를 **추가 가정**으로
유지한다. $W^{1,p}$의 유도 $C^0$ 거리 자체를 완비 Polish라고
전제하지 않고, 여기처럼 완비 $C^0$ 공간에 비용을 $\infty$로
확장하거나 각각의 compactness·측도 가정을 별도로 확인한다.
이 패키지는 실제 CE 작용·게이지·인과율의 유도가 아니다.

### E4. mesh와 온도의 극한

05h의 J3 outer gap과 J4 recovery가 있으면 E2 마지막 부등식을
보간된 후보에 그대로 적용해 최소집합 근방 농축을 얻는다.
J3를 직접 가정할 때 별도의 equicoercivity는 그 부등식의 증명에
필요하지 않다. Equicoercivity는 J3를 유도하는 한 방법이다.

**[반례]** $E(x)=x^2$ on $\mathbb R$,
$E_n(x)=x^2$ for $|x|<n$, $E_n(x)=x^2/2$ for $|x|\ge n$은
국소 균등 수렴하고 $E_n\ge x^2/2$로 equicoercive다.
그러나 모든 $n$에서 $\sup_x(E-E_n)=\infty$다.
따라서 국소 균등+equicoercivity가 전역 부등식
$E_n\ge E-\epsilon$을 주지는 않는다. 최소점 밖의 **에너지 gap**은
E2와 같은 compactness/liminf 논증으로 따로 얻을 수 있다.

**[정리: 고정 온도]** Compact metric $X$, 연속 유한 $E$,
$(\iota_N)_*\mu_N\Rightarrow\mu$,
$\epsilon_N=\sup_a|E_N(a)-E(\iota_N(a))|\to0$이면 고정 $\beta>0$에서
보간 Gibbs 측도는 $Z_\beta^{-1}e^{-\beta E}\mu$로 약수렴한다.
실제로 $|e^{-\beta(E_N-E)}-1|\le e^{\beta\epsilon_N}-1$로 분자·분모
오차가 0이고, $fe^{-\beta E}$는 bounded continuous이므로 prior의
약수렴을 적용한다. 분모 극한은 양수이므로 비율도 수렴한다. $\square$

$\beta$가 함께 커지면 이 bound는 자동으로 0이 아니다.
각 정수 $k$에 대해 고정 온도 수렴을 확보했을 때만 약수렴 거리 오차를
$1/k$보다 작게 하는 **증가하는** $N(k)$를 귀납적으로 골라 diagonal
수렴을 얻는다. 이는 임의의 mesh–temperature 일정에 대한 정리가 아니다.

### E5. 정규화·커널·tropical 도구

**[정리]** 유한 비음수 행렬 $K,L$의 각 row 합이 양수이면 곱 $KL$도
같은 성질을 가진다. 상태 row vector $\mu$에 대해
$T_K(\mu)=\mu K/(\mu K\mathbf1)$라 두면

$$
T_L(T_K(\mu))=T_{KL}(\mu).
$$

분자·분모의 중간 양의 scalar를 약분하면 바로 성립한다.
이는 일반적으로 비선형인 **상태 함수**의 합성이다.
Row-normalized 행렬을 먼저 곱하는 연산과 다르다.

**[반례]** $K=(1,1)$, $L=\operatorname{diag}(1,2)$이면
$\overline{KL}=(1/3,2/3)$인 반면
$\bar K\bar L=(1/2,1/2)$다.
State normalization은 합성과 양립해도 row normalization은 일반적으로
functor가 아니다.

Weight monad는 Set 위의 유한 support 함수
$W(X)=\{w:X\to[0,\infty):|\operatorname{supp}w|<\infty\}$로 둔다.
단위는 Dirac, 곱셈은 $\sum_w\Phi(w)w(x)$다. 유한합 재배열로 단위와
결합법칙이 성립한다. 그 Kleisli 범주에서 유한 대상만 취하면 모든
유한 비음수 행렬을 얻고, nonzero row로 제한하면 PreEq를 얻는다.
$W(A)$는 보통 무한집합이므로 FinSet의 endofunctor라고 부르지 않는다.

**[정리]** 유한 값 $u_1,\ldots,u_N$, $\beta>0$이면

$$
m-\frac{\log N}{\beta}
\le -\beta^{-1}\log\sum_j e^{-\beta u_j}\le m,\quad m=\min_j u_j.
$$

$e^{-\beta m}$으로 나눈 합이 $[1,N]$에 있다는 것으로 증명된다.
유한 경로가 하나도 없으면 합은 0, 두 energy 값은 $\infty$이며
위 **차에 대한 부등식은 적용하지 않는다**. $\infty-\infty$는 무정의다.
또 $K>1$인 성분의 $-\log K/\beta$는 음수다.
Tropicalization의 일반 공역은 $\mathbb R\cup\{\infty\}$이지
비음수 energy 범주가 아니다. 유한 온도 Gibbs 합성은
비음수 energy의 min-plus 합성과 같지 않으며, 위 오차를 가진 극한만 보존된다.

### E6. 조건화와 잔류

**[정리]** 유한 prior $\rho_0$와 유한 비용 $E$에 대해
$S=\{\rho_0>0\}$, $M=\arg\min_SE$라 두면

$$
\rho_\beta(x)\longrightarrow
\frac{\rho_0(x)\mathbf1_M(x)}{\rho_0(M)}.
$$

**증명.** 최소 에너지 $m$을 지수에서 빼면 $M$에서 가중치는 1,
$S\setminus M$에서는 0으로 수렴한다. 유한합 분모는
$\rho_0(M)>0$로 수렴한다. $\square$

Joint 조건–값 공간에서도 같다. 유일 최소쌍 없이 Dirac를 쓰면
틀린다. 양의 marginal에서만 조건부 확률을 나눗셈으로 정의한다.
연속 영질량 사건의 조건부 확률은 이 비율식으로 결정되지 않는다.
예를 들어 독립 균등 $X,Y\in[0,1]$의 조건부 $Y|X=x$를 모든 $x$에서
균등으로 둘 수도 있고, $x=0$에서만 $\delta_0$로 바꿀 수도 있다.
둘 다 같은 joint law를 재구성하는 가측 kernel이다.
따라서 영질량 점에서의 판본은 joint law만으로 유일하지 않다.

Raw 잔류 $\rho_\beta|_{M^c}$의 총질량은 0으로 간다.
$K$가 유계이면
$|\int K\,d(\rho_\beta|_{M^c})|\le\|K\|_\infty\rho_\beta(M^c)\to0$이다.
조건부 잔류는 양의 잔류 질량이 있는 유한 $\beta$에서만 나눈다.
이 정규화로 남은 모양을 raw 에너지 보존이라고 해석할 수 없다.

약수렴 $\nu_n\Rightarrow\nu$를 readout에 옮기려면 bounded continuous
kernel이 충분하다. 단순 measurable kernel에는 반례가 있다:
$\nu_n=\delta_{1/n}\Rightarrow\delta_0$, $K=\mathbf1_{\{0\}}$이면
적분은 $0\not\to1$이다. 비유계 kernel도 별도 적분 조건이 필요하다:
$(1-1/n)\delta_0+(1/n)\delta_n\Rightarrow\delta_0$이지만
$K(x)=x$ 적분은 항상 1이다.

### E7. Born 정리의 전제와 독립성 반례

06a의 B0–B5 아래 제곱진폭 규칙은 조건부로 성립한다.
그 일반 증명은 다음과 같다. 동일한 $N$개 nonzero 진폭은 B3와
정규화로 확률 $1/N$을 갖는다. $|c_i|^2=n_i/N$인 유리 진폭은
분지 $i$를 $n_i$개로 동일 분할하는 B4 isometry를 적용한다.
영 진폭은 하나의 영 microbranch로 남기고 B2를 쓴다.
모든 nonzero microbranch 진폭 크기가 $1/\sqrt N$이므로
coarse 확률은 B4에 의해 $n_i/N$이다.
임의 진폭은 확률 simplex의 유리 근사와 원래 위상을 유지한
$c_i^{(k)}=\sqrt{p_i^{(k)}}e^{i\arg c_i}$로 근사한다.
영 진폭의 위상은 임의로 둔다. Norm 수렴과 B5로
$p_i=|c_i|^2$를 얻는다. 이는 모든 유한 차원에 걸친 assignment에 대한
유일성 증명이며 B0는 이 증명에서 중복 전제다.

**[반례: B4 제거]** $\alpha>0$에 대해

$$
p_i^{(\alpha)}=\frac{|c_i|^{2\alpha}}{\sum_j|c_j|^{2\alpha}}
$$

는 B0, B1, B2, B3, B5를 모두 만족한다.
$|c|^2=(3/4,1/4)$, $\alpha=2$이면 원 확률은 $(9/10,1/10)$이다.
첫 분지를 3개로 분할하면 네 microbranch가 같은 크기이므로
다시 합한 확률은 $(3/4,1/4)$다. 따라서 B4를 위반하며 Born과 다르다.
대칭·연속성만으로는 Born 결론이 나오지 않는다.
반대로 Born assignment는 제곱진폭의 유한합으로 B0–B5를 모두
만족하므로 이 공리계는 해당 모형을 갖는다.

### E8. Gibbs 조건화의 CP 구현과 단일 사건의 한계

**[정리]** 유한 outcome PVM $\{P_i\}$와 유한 비용 $c_i$,
$\beta>0$를 고정하고 $m=\min_i c_i$,
$w_i=e^{-\beta(c_i-m)}\in(0,1]$라 두자.

$$
M_i=\sqrt{w_i}P_i,\qquad
M_{\rm fail}=\sqrt{I-\sum_iw_iP_i}
$$

는 $\sum_iM_i^\dagger M_i+M_{\rm fail}^\dagger M_{\rm fail}=I$이므로
CP instrument를 만든다. 성공 결과만 조건화하면

$$
p(i\mid{\rm success})
=\frac{w_i\operatorname{Tr}(\rho P_i)}
{\sum_jw_j\operatorname{Tr}(\rho P_j)}
$$

로 정확히 Gibbs 재가중이다. **성공 확률과 실패 결과를 포함한**
선형 map이 CPTP이며 성공 조건부 map 자체는 일반적으로 비선형이다.
성공 여부만 기록할 때 단일 Kraus $\sum_i\sqrt{w_i}P_i$를 쓰는 모형과
$i$까지 기록하는 위 모형은 성공 확률은 같아도 coherence 보존이 다르다.
따라서 확률 재가중식 하나로 instrument가 유일하게 정해지지 않는다.

이 구성은 Born trace 규칙과 PVM을 입력으로 사용하므로 Born의
무가정 유도가 아니다. 경계를 통과하는 unitary 궤적과도 다르다.
Outcome 확률의 수학적 정의가 실제 사건의 발생 원리까지 설명하지 않는다.
또 $[H,P_i]=0$이면 모든 Kraus가 $H$와 가환하므로
$\sum_kM_k^\dagger H M_k=H$로 비선택 에너지 기대값이 보존된다.
가환 가정이 없으면 이 결론을 승계하지 않는다.

## 3. 증명 의존도와 적용 한계

| 결과 | 사용한 표준 도구 | 아직 포함하지 않는 결론 |
|---|---|---|
| E1–E2 | compact 부분열, l.s.c., Gamma liminf, bounded continuous test | 자연의 선택 원리, 실제 CE 작용의 coercivity |
| E3 | Hölder, Arzelà–Ascoli, Hilbert 약한 compactness·norm l.s.c., 단조수렴, Borel–Cantelli | Lorentzian 경로적분, 게이지 quotient, 인과율 |
| E4 | 약수렴, 연속 함수, 양의 분모, 대각 선택 | 임의 극한 교환, 계산 가능한 수렴률 |
| E5–E6 | 유한합·행렬 곱, 로그 bound, 측도 조건화 | 새 논리적 등호의 발생, raw 잔류의 물리 보존 |
| E7 | 유한 refinement **공리**, 유리 근사, 연속성 | 실제 장치가 공리를 따르는 이유 |
| E8 | Kraus 완전양성, trace 규칙, PVM | 사건의 존재론, 독립 Born 유도 |

이 결과는 명시한 수학적 전제 아래 재사용 가능하다. “증명이 완벽하다”는
무제한 선언이나 리만가설 해결을 하지 않는다. 형식 검증 여부와 검산은
다음 절에 별도로 기록한다.

## 4. 검산과 판정

[별도 계산 코드](../../../verify/ce_mathematical_boundary_audit.py)와
[결과 JSON](../../../verify/ce_mathematical_boundary_audit.json)은
연속 well 적분·서로 다른 recovery 속도·경로 kinetic 분해·Gaussian
증분 moment·row 정규화 반례·joint 극한·Born refinement 반례·CP
instrument를 정확식으로 검산한다. B·R·E 전체 정확 항등식은 69개,
반례 기록은 30항목이다. 부동소수 최대 절대오차는
$2.5121479338940403\times10^{-15}$, 허용치는 $10^{-10}$이다.

§1 끝의 빈 줄을 제거하고 LF 하나로 끝내는 사전등록 prefix의 SHA256:
27b11733f2beec5c45f9afc524acd645c7ab8d54f18d6094b6e342060c670904.
기존 B·R 사전등록도 함께 검사했다. 계산 뒤 전제를 바꾸지 않았다.

새 일반 정리의 증명과 원문에서 보완한 조건을 모두 보존한다.
기계 검산은 유한 증인의 대수·수치를 확인하며 a.s. 수렴이나 모든
Polish 공간의 정리를 유한 표본으로 증명한 것이 아니다.
형식 proof assistant는 사용하지 않았다. 실제 CE prior·action·
Born 공리의 물리적 정당화와 전체 공동 RMSE는 다음 물리 검증의 입력이다.
