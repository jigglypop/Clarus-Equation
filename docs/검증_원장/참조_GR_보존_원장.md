# CE GR 보존 원장

<!-- 출처: docs/검증_원장/참조_이론물리_보존_원장.md — §6 이동 | 이동일: 2026-08-23 (MULTIREPO_PLAN.md P2-6) -->

이 문서는 CE의 특정 물리 모형을 완성했다고 주장하지 않는다. 얇은 껍질
중력과 exotic source에 대한 조건부 정리를 전제와 함께 보존한다. 이 결과를
CE의 장, 입자, 측정 장치에 대응시키는 단계는 별도의 물리 사상이다. 절
번호는 원파일 `참조_이론물리_보존_원장.md`의 번호를 유지한다. 무차원 지수
규약(§7)과 남은 물리 사상(§8)은 원파일에 있다.

## 6. Wormhole과 얇은 껍질의 조건부 정리

### 6.1 기하학적 shortcut과 NEC

**[정리]** throat를 통과하는 내부 길이가 $\ell>0$, 국소 속력이
$v=\beta c$, $0<\beta\leq1$이면 통과 시간은
$$
t_{\rm int}=\frac{\ell}{\beta c}>0.
$$
외부 기준 거리 $L$로 정의한 겉보기 속력은
$$
\frac{L}{t_{\rm int}}=\beta c\,\frac L\ell
$$
이다. $L>\ell/\beta$이면 이 비는 $c$보다 클 수 있지만 국소
초광속 운동이나 순간 전달을 뜻하지 않는다.

**[정리]** zero-redshift Morris--Thorne metric에서
$b(r_0)=r_0$, $b'_0<1$인 regular throat의 orthonormal
energy density와 radial pressure는
$$
\rho_0+p_{r0}
=\frac{c^4}{8\pi G r_0^2}(b'_0-1)<0.
$$
따라서 radial null vector에 대한 NEC가 throat에서 위반된다.

**[정리: source no-go]** 최소결합 canonical scalar의 stress tensor는
모든 null vector $k^\mu$에 대해
$$
T_{\mu\nu}k^\mu k^\nu=(k^\mu\partial_\mu\varphi)^2\geq0.
$$
그러므로 고전 Einstein 방정식에서 canonical scalar 하나만으로 위
throat의 필요한 NEC 위반을 만들 수 없다. 비최소결합, higher derivative,
양자 기대값 또는 phantom 부호는 이 정리의 정의역 밖이며 각각 별도의
ghost·안정성·재규격화 검사가 필요하다.

### 6.2 대칭 Schwarzschild thin shell

**[공리: 기하 모형]** 두 Schwarzschild exterior를 정적 반지름 $a$에서
대칭 접합하고
$$
f=1-\frac{2GM}{c^2a}\in(0,1]
$$
라 하자.

**[정리]** Israel junction condition이 주는 표면 에너지 밀도와 등방
압력은
$$
\sigma_s=-\frac{c^4}{2\pi Ga}\sqrt f<0,\qquad
p_s=\frac{c^4}{8\pi Ga}\frac{1+f}{\sqrt f}>0.
$$
$f\downarrow0$이면 $\sigma_s\to0$이지만 $p_s\to+\infty$다.

**[정리: 방정식상태 no-go]** 등방 $2+1$차원 traceless source는
$-\sigma_s+2p_s=0$, 즉 $p_s=\sigma_s/2$를 만족해야 한다.
위 shell은 $\sigma_s<0<p_s$이므로 어떤 양의 amplitude 재조정으로도
이 방정식상태와 일치하지 않는다.

**[정리: 선형 안정성 no-go]** shell 물질이 국소 barotropic 기울기
$\eta=dp_s/d\sigma_s$를 갖는다고 하자. 표준 radial potential의
정적점에서
$$
a^2V''(a)
=2\eta(1-3f)-\frac{1+3f^2}{2f}.
$$
$0<f<1/3$에서 $V''>0$이려면
$$
\eta>\frac{1+3f^2}{4f(1-3f)}>1,
$$
$f=1/3$에서는 $a^2V''=-2$, $f>1/3$에서는 안정하려면
$\eta<0$이어야 한다. 따라서 전 정의역에서
$0\leq\eta\leq1$인 causal-stable barotropic 구간과 겹치는
radial stable branch가 없다.

**[정리: 수동 모드 no-go]** radial displacement $x$와 내부 모드
$y$의 이차 potential이
$$
V_2=\frac12(K_{rr}+D)x^2+xB^{\mathsf T}y+\frac12y^{\mathsf T}Cy,
\qquad C>0
$$
이면 $y$를 수동적으로 최소화한 유효 radial stiffness는
$$
K_{\rm eff}=K_{rr}+D-B^{\mathsf T}C^{-1}B\leq K_{rr}+D.
$$
따라서 안정한 내부 모드의 수동 완화만으로 음의 radial stiffness를
증가시킬 수 없다.

### 6.3 Casimir형 source의 국소·전역 경계

**[정리]** 일반 redshift Morris--Thorne throat에서
$$
C_0:=\frac{c^4}{8\pi Gr_0^2},\qquad u:=r_0\Phi'(r_0)
$$
라 하면 regular throat의 국소 자료는
$$
\frac{(\rho_0,p_{r0},p_{t0})}{C_0}
=
\left(b'_0,-1,\frac{(1-b'_0)(1+u)}2\right).
$$
이를 이상화한 radial Casimir 비
$(-1/3,-1,+1/3)$와 맞추면 유일하게
$$
b'_0=-\frac13,\qquad u=-\frac12
$$
를 얻는다. 이는 throat의 국소 Taylor 자료일 뿐 전역 해의 존재,
점근평탄성이나 안정성을 증명하지 않는다.

**[정리: 전역 no-go]** 어떤 $R_*>0$ 이후에 고정 방정식상태와
정확한 nonzero power-law tail
$$
p_r=3\rho,\qquad p_t=-\rho,\qquad
\rho(r)=\rho_*\left(\frac r{r_*}\right)^{-n},
\qquad \rho_*\ne0,\quad r_*>0
$$
를 유지하고 amplitude envelope만 허용한다고 하자. 정적 구대칭 보존식
$$
p_r'=-(\rho+p_r)\Phi'+\frac2r(p_t-p_r)
$$
은
$$
\Phi(r)=\left(\frac{3n}{4}-2\right)\log\frac r{r_*}+O(1)
$$
을 강제한다. 유한한 점근 redshift에는 $n=8/3$이 필요하지만,
유한한 $\int^\infty r^2|\rho(r)|\,dr$에는 $n>3$이 필요하다.
따라서 이 고정 방정식상태와 amplitude-only 꼬리로 두 조건을 동시에
만족시킬 수 없다. 유한 반지름 cutoff나 방정식상태 변화는 이 정리의
정의역 밖이다.
