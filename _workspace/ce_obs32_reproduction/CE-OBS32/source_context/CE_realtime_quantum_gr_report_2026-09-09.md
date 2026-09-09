# 31. 공통 스펙트럼의 실시간 양자상태와 중력 되먹임

2026-09-09 · CE-RT31 · 출발 `05013ce3c653fc68c2fe5a4ab6294e95bd6c0a30`

이 장은 29장의 상태–내부장 공동 계산과 30장의 4차원 상대 곡률 응력 사이를 연결한다. 같은 세 복소 스칼라를 유지하면서 정적인 행렬식을 인과적 실시간 반응으로 확장하고, 생성되는 공간모드가 집단장과 계량에 주는 힘을 되돌려 넣는다. 순서는 지연 응답, 연속 운동량 모드, 보존 응력, 기준 진공의 재규격화, 자기정합 팽창, 순환 조건의 공간모드 검사다.

이번에 새로 실행한 것은 지정한 한 루프/가우스 평균장 절단의 계산이다. 계량과 집단장은 고전 평균장, 세 복소 물질장은 양자 가우스장이다. 양자중력, 실제 우주의 초기상태 유일성, 절대 진공밀도, 관측 H0와 CMB 우도의 완성으로 보고하지 않는다. 특히 뒤의 팽창 계산은 xi=1/6과 명시한 유한 R^2 matching을 사용한다. 이 입력을 CE의 무입력 귀결로 승격하지 않는다. 서로 다른 초기상태와 곡률결합의 이전 장들을 자동으로 합치지 않는다.

## 1. 보존한 스펙트럼과 공통 함수

기존과 같이 질량제곱은

$$x_j(\theta)=s+2\epsilon\cos[(\theta+2\pi j)/3],\qquad s>2\epsilon>0,$$

이고 j=0,1,2다. x는 질량제곱이고 theta는 무차원이다. 이번 문서에서 A_j=partial_theta x_j, B_j=partial_theta^2 x_j로 쓴다. sum x_j=3s와 sum x_j^2=3s^2+6epsilon^2를 유지한다. 중성 암흑장에 새 게이지 전하나 Yukawa를 붙이지 않는다.

정적인 상대 퍼텐셜은

$$U(\theta)={1\over32\pi^2}\Delta_\pi\sum_jx_j^2[\ln(x_j/\mu^2)-3/2].$$

이 U는 theta=pi를 기준으로 한 루프 기여다. 전체 상수 Lambda_R, 허용되는 유한 국소항과 그 matching은 별도다. 모드 전체를 다시 보유할 때는 이전에 적분해 얻은 delta Z를 독립 운동항으로 한 번 더 더하지 않는다. 아래에서 delta Z는 정확한 모드 반응의 느린 극한으로 재현된다.

## 2. 정적 반응과 입자 생성의 공통 지연 커널

먼저 평탄공간의 일정한 배경 theta_bar와 진공을 택한다. h=theta-theta_bar의 이차 반응에서 정적인 U''를 분리하면, 복소 스칼라 한 루프의 지연 커널은

$$\boxed{\widehat\Pi_R(\omega,\boldsymbol q)={1\over16\pi^2}\sum_j A_j^2\int_0^1 dz\,\ln\left[1-{z(1-z)[(\omega+i0)^2-\boldsymbol q^2]\over x_j}\right].}\tag{1}$$

이는 유클리드 bubble을 인과적으로 계속한 식이다. Tr log의 이차 전개에서 국소 B_j tadpole과 영운동량 bubble은 U''에 포함된다. 나머지는 -A_j^2[I_2(p)-I_2(0)]이며 Feynman parameter 적분이 식(1)을 준다. CTP/in-in 및 Gaussian backreaction 자체는 기존 방법 [1,2]이다. 이번 합성은 그 커널을 CE의 고정된 순환 스펙트럼에 연결한 것이다.

양의 시간주파수 nu에서 흡수 스펙트럼은

$$\boxed{\mathcal D(\nu):=-\operatorname{Im}\widehat\Pi_R(\nu,0)={1\over16\pi}\sum_j A_j^2\sqrt{1-{4x_j\over\nu^2}}\,\Theta(\nu-2\sqrt{x_j})\ge0.}\tag{2}$$

로그가 음의 실수축을 만나는 z 구간의 길이가 제곱근을 준다. sum A_j=0이라도 sum A_j^2의 흡수는 일반적으로 사라지지 않는다. 정적인 상쇄를 실제 쌍생성의 상쇄로 해석할 수 없다.

한 번 감산한 분산식과 그 낮은 주파수 극한은

$$\widehat\Pi_R(\omega,0)=-{2\omega^2\over\pi}\int_0^\infty {\mathcal D(\nu)\,d\nu\over\nu[\nu^2-(\omega+i0)^2]},$$

$$\boxed{\delta Z={2\over\pi}\int_0^\infty{\mathcal D(\nu)\over\nu^3}d\nu={1\over96\pi^2}\sum_j{A_j^2\over x_j}.}\tag{3}$$

따라서 기존 22장의 운동계수와 새 입자 생성 스펙트럼은 독립 계수가 아니다. 한 종에 대해 int_(2sqrt(x))^infinity sqrt(1-4x/nu^2) dnu/nu^3=1/(12x)이므로 정규화도 직접 확인된다.

실시간으로는 K(tau)=(2/pi)int D(nu)sin(nu tau)dnu/nu^2, tau>=0를 두어 int_0^infinity K(tau) h''(t-tau)dt로 쓸 수 있다. 과거 반응이 현재에 들어가며 미래 신호를 사용하는 법칙이 아니다. 작은 주파수에서 이 항이 delta Z h''로 돌아온다.

0<omega<2sqrt(x_min)에서는 이 평탄 진공 선형반응의 흡수가 0이고,

$$0\le{\left|\widehat\Pi_R+\delta Z\omega^2\right|\over\delta Z\omega^2}\le{\omega^2\over10x_{\min}[1-\omega^2/(4x_{\min})]}.\tag{4}$$

이 경계는 로그의 나머지 부등식과 int z^2(1-z)^2 dz=1/30에서 나온다. 이는 곡률, 유한밀도, 큰 진폭이나 비선형 고조파의 모든 입자 생성을 금지하는 명제가 아니다. 느린 진공 응답에 임의의 큰 마찰을 붙이지 못한다는 제한이다.

## 3. 실제 생성 에너지를 두 독립 경로로 계산

h(t)=a_h sech^2(t/tau)의 작은 펄스를 공급한다. a_h는 척도인자 a와 다른 펄스 진폭이다. Fourier transform은 h_tilde(nu)=a_h pi nu tau^2/sinh(pi nu tau/2)다. 흡수 일은

$$\boxed{E_{\rm abs}=\int_0^\infty{d\nu\over\pi}\nu\mathcal D(\nu)|\widetilde h(\nu)|^2.}\tag{5}$$

한편 실제 모드의 일차 Bogoliubov 계수는 beta_jk=-i A_j h_tilde(2omega_jk)/(2omega_jk), omega_jk=sqrt(k^2+x_j)이므로

$$E_{\rm pair}=\sum_j\int {d^3k\over(2\pi)^3}\,2\omega_{jk}|\beta_{jk}|^2=E_{\rm abs}.$$

복소장의 입자–반입자 때문에 2omega가 들어간다. 생성량을 이중계산하지 않는다. 양자 kinetic 표현과 모드 산란의 일반적 동등성은 [3]을 따른다.

s=1, epsilon=.35, theta_bar=.7, tau=.7에서 두 적분의 E/a_h^2는 각각 0.00041436184863791366과 0.00041436184863791377이었다. 전체 비선형 질량 x(theta)를 사용하는 모드 진화도 a_h=.02,.01,.005,.0025를 줄일 때 이 값으로 접근했다. a_h=.02의 정확 kinetic 에너지와 별도의 복소 2차 모드 방정식 에너지는 상대 5.44e-10 이내였다. 외부 펄스가 공급한 에너지이며 무에서 얻은 에너지가 아니다.

## 4. 공간 연속모드와 상관을 함께 보유하는 실시간 식

이제 공간적으로 평탄한 FLRW에서 xi=1/6을 명시적으로 고정한다. eta는 등각시간이고 prime=d/deta다. chi=aX의 정준 모드는

$$\chi_{jk}''+\Omega_{jk}^2\chi_{jk}=0,\qquad\Omega_{jk}^2=k^2+a^2x_j(\theta).$$

n_jk=|beta_jk|^2, c_jk=alpha_jk beta_jk^* exp[-2i int Omega_jk deta]를 두고 u=Re c, v=Im c라 하면

$$\boxed{n'=2\mathscr W u,\quad u'=\mathscr W(1+2n)+2\Omega v,\quad v'=-2\Omega u,\quad\mathscr W={\Omega'\over2\Omega}.}\tag{6}$$

순수 가우스 초기상태의 (1+2n)^2-4(u^2+v^2)=1을 보존한다. 생성량 n만 남기고 u,v를 버리지 않는다. u,v는 진폭과 위상의 양자상관이며 객관적인 단일결과 선택법칙이 아니다. 각 양자의 비동기 틱을 공통 우주 갱신으로 바꾸지도 않는다.

dPi_k=k^2 dk/(2pi^2)로 정의하면 순간 진공을 뺀 상태 응력과 위상 원천은

$$\rho_{\rm ex}=a^{-4}\sum_j\int d\Pi_k\,2\Omega n,$$
$$p_{\rm ex}={a^{-4}\over3}\sum_j\int d\Pi_k\left[2\Omega n-{2a^2x_j\over\Omega}(n+u)\right],$$
$$\boxed{J_{\rm ex}=a^{-2}\sum_j\int d\Pi_k\,{A_j\over\Omega}(n+u).}\tag{7}$$

직접 미분하면

$$\boxed{\rho_{\rm ex}'+3{a'\over a}(\rho_{\rm ex}+p_{\rm ex})=\theta'J_{\rm ex}.}\tag{8}$$

양자 생성과 에너지 교환을 별도 함수로 공급하지 않는다. 순간 입자수의 해석은 기저 의존적이지만 보유한 상태 응력은 같은 모드와 metric 변분으로 계산한다. 비단열 구간에 압력을 단순한 입자 기체 식으로 바꾸면 u 항이 빠진다.

## 5. 같은 계량에서 유한한 상대 응력

theta=pi인 기준 종도 같은 a(eta) 위에서 실제로 진화시킨다. 두 상태는 적절한 공통 단열/Hadamard 자외선 조건을 갖고 경계 특이성은 별도로 다룬다고 한다. 질량 합 규칙 때문에 두 작용의 국소 자외선 계수는 30장처럼 상쇄된다. 새 실시간 표현은

$$\boxed{\Delta\rho=U+\rho_{\rm ex}-\rho_{{\rm ex},\pi},\quad\Delta p=-U+p_{\rm ex}-p_{{\rm ex},\pi},\quad J=U_\theta+J_{\rm ex}.}\tag{9}$$

식(8)로 Delta rho'+3(a'/a)(Delta rho+Delta p)=theta' J다. 이는 기준 진공의 응력이 존재하지 않는다는 뜻이 아니다. 일반 xi에서의 유한성을 순간 진공 감산 하나로 보장하지도 않는다 [4].

고정된 매끄러운 팽창 a(eta)=1+.2[1+tanh(eta/2)]과 위상 펄스 theta=.7+.4 sech^2(eta/.7)를 계산했다. 결과에는 이전 이력에 따른 잔여 상태 에너지가 남는다. 최종 Delta rho-U(.7)=2.39149e-5였고, 같은 시점의 정적 U만으로는 이를 알 수 없다. 이것은 외부에서 지정한 배경의 반응 검사이지 자기정합 팽창이나 관측 암흑밀도는 아니다.

### 30장의 유클리드 중력원과 실시간 모드가 연결되는가

일정 H, 일정 theta, Euclidean/Bunch–Davies 상태를 선택하면 Lorentzian flat patch의 정확 Hankel 모드를 사용할 수 있다. nu_j=sqrt(1/4-x_j/H^2)에서 a=1인 시점의 모드와 도함수를 직접 구성하고 int dPi_k sum(E_j-E_j,pi)를 무한구간 적분했다. 30장의 별도 digamma 식과 비교했다.

H=1, epsilon=.35, theta=.7에서 실시간 적분은 0.00046846791500578753, 유클리드 중력원은 0.0004684679150052514였다. 192점 결과의 상대차는 1.15e-12 이하다. 96점에서 192점으로 늘린 수렴도 함께 기록했다. 임의 비평형 상태를 이 진공으로 바꾼 것이 아니라, 같은 상태가 되는 공통 극한에서 두 표현을 연결한 것이다.

## 6. 공간모드가 집단장에 되먹이는 닫힌 평탄공간 계산

a=1에서 bare collective kinetic f^2를 유지하고

$$f^2\ddot\theta+U_\theta+J_{\rm ex}=0$$

와 식(6)을 함께 풀었다. 따라서

$$\boxed{E_{\rm tot}=f^2\dot\theta^2/2+U+\rho_{\rm ex},\qquad\dot E_{\rm tot}=0.}\tag{10}$$

를 얻는다. 모드 자기장을 따로 소거해 delta Z를 다시 더하지 않았으며, U는 순간 진공을 감산한 뒤 복구한 같은 상대 정적 기여다. 이 분해는 원래 전체 상태 에너지에서 기준 진공 에너지를 한 번 뺀 것과 같다.

s=1, f=.1, theta(0)=.7, theta_dot(0)=6, 초기 순간 진공의 지정된 가우스 상태, t=25를 사용했다. r=.15에서 모드로 전달된 에너지는 초기 총에너지의 약 11.93%였고, r=.35에서는 약 34.07%였다. 후자의 위상 속도는 약 4.87862이며, 같은 U지만 상태 힘을 끈 대조는 약 5.99887이다. 기존 마찰계수를 피팅하지 않았다.

이는 cosmological dark matter fraction이 아니다. 강한 회전으로 준비한 무차원 내부 검산이며 고전 평균장의 에너지가 quantum spatial modes로 이동한 비율이다. 초기 에너지는 입력이다. 특히 f=.1 sqrt(s)의 강한 회전 시험이 집단장 양자루프까지 작다는 제어된 매개변수 영역임을 증명하지 않았다. 수치 수렴은 지정한 가우스 평균장 절단 안에서의 수렴이며, 재산란과 집단장 양자요동의 절단오차는 별도다. 유한 momentum quadrature의 수치 보존과 continuum cutoff 수렴을 구분한다. K=24에서48로 늘렸을 때 최종 모드에너지 상대변화는 5.70e-6이며, 같은 cutoff의 보존오차 약 8e-15를 continuum 오차라고 보고하지 않는다.

## 7. 기준 응력을 되살려 실제 팽창도 함께 계산

상대 응력만 Einstein 식에 넣으면 기준 부문을 버리게 된다. 이 장에서는 제한된 conformal branch에서 기준 부문의 단열 감산을 명시적으로 계산한다. WKB 빈도를 omega+W2+...로 쓰면 W2=-omega''/(4omega^2)+3omega'^2/(8omega^3)이고, 복소장 한 종의 에너지 2차·4차 항은

$$E_2={y'^2\over32\omega^5},\qquad y=a^2m^2,$$
$$E_4={y''^2-2y'y'''\over128\omega^7}+{7y'^2y''\over128\omega^9}-{105y'^4\over2048\omega^{11}}.$$

이를 운동량 적분하고 cosmic time으로 쓰면

$$\rho_2={m^2H^2\over48\pi^2},\qquad\rho_4=-{H^4+6H^2\dot H-\dot H^2+2H\ddot H\over480\pi^2}.\tag{11}$$

이 계수는 유한하며 단순한 normal ordering에서 빠지는 국소 기하항을 포함한다. WKB 기호전개와 beta 적분, 별도 momentum quadrature로 검사했다. 참조질량 세 종의 합이 3s이고 세 복소장이므로 같은 relative prescription을 복원한 양자 에너지에는

$$\rho_{q,R}=U+\rho_{\rm ex}-{sH^2\over16\pi^2}+{H^4+\mathcal D_H\over160\pi^2},\quad\mathcal D_H=6H^2\dot H-\dot H^2+2H\ddot H$$

가 나타난다. Lambda_R는 별도로 남긴다. 임의로 H를 시간에 따른 정적 de Sitter 에너지에 대입한 결과가 아니다.

### 명시한 유한 matching 가지

재규격화된 R^2 계수는 입력이다. 실제 실행에서는 그 유한 계수를 한 번 정해 H^(1) 형태의 D_H 항을 상쇄하는 가지를 사용했다. Euler anomaly의 H^4는 지우지 않는다. 이는 새 자유함수를 관측에 맞춘 것이 아니라, 알려진 국소 기하 ambiguity [4]의 한 물리적 matching 선택이며 다른 선택에 대한 해를 증명하지 않는다. 같은 조건을 모든 실행에 고정했다.

L=3M_R^2+s/(16pi^2), c_A=1/(160pi^2), rho_B=f^2 theta_dot^2/2+U+rho_ex+Lambda_R라 하면

$$\boxed{LH^2-c_AH^4=\rho_B,\qquad H^2={2\rho_B\over L+\sqrt{L^2-4c_A\rho_B}}.}\tag{12}$$

GR로 연속 연결되는 낮은 곡률 가지를 택한다. 분모의 discriminant가 음수가 되는 영역은 실행 중단 조건이다. p_B=f^2 theta_dot^2/2-U+p_ex-Lambda_R에서 독립적인 시간진화식은

$$\boxed{\dot H=-{3(\rho_B+p_B)\over2L-4c_AH^2},\quad\dot a=aH,\quad f^2(\ddot\theta+3H\dot\theta)+U_\theta+J_{\rm ex}=0.}\tag{13}$$

식(6), (7), (12), (13)이 양자상태–집단장–계량의 닫힌 semiclassical forward problem이다. 중력을 양자화한 것은 아니며 초기상태를 유일하게 골라 주는 boundary theory도 아니다.

초기 상태는 q=1/sqrt(2Omega), q'=(-iOmega-Omega'/(2Omega))q인 유한에너지 단열형 Gaussian preparation이다. 따라서 a(0)=1에서도 H(0)가 covariance 안에 들어가며 초기 Friedmann constraint와 함께 root solve했다. 이 간단한 준비의 all-order Hadamard 성질이나 모든 stress derivative의 정칙성을 입증했다고 하지 않는다. 실행은 유한에너지 상태와 cutoff 수렴을 검사하며 더 강한 경계 정칙성은 별도 의무다.

s=1, epsilon=.35, f=.1, M_R=3, theta0=.7, theta_dot0=6, Lambda_R=0의 시험에서 K=48 결과는

| 양 | 처음 | t=25 |
|---|---:|---:|
| a | 1 | 1.9756954 |
| H | 0.0822793 | 0.0128174 |
| theta_dot | 6 | 0.7501954 |
| rho_ex | 0.00231475 | 0.00144007 |

이다. theta와 a를 외부 함수로 주지 않았다. 최초 상태에는 이미 양의 준비 에너지가 있으며, 팽창에 따른 희석과 추가 생성은 함께 작용한다. M_R=10의 별도 사례도 실행했다. 이 M_R/sqrt(s), f/sqrt(s)와 시간은 실제 우주의 보정값이 아니라 계산 가능한 무차원 진단이다. 25장의 현재 우주 질량척도를 이 숫자에 몰래 결합하지 않는다.

Friedmann algebraic constraint로 H를 구하는 실행과 Hdot를 직접 적분한 실행은 K=24에서 H,a,theta,velocity가 상대 1e-8보다 잘 일치했다. K=24에서48로 늘린 최종 H 변화는 약 8.63e-6이었다. 따라서 표는 필요한 자릿수만 쓰며 machine-precision conservation을 실제 물리 정밀도로 주장하지 않는다. Lambda_R=0은 시험 가지이지 절대 진공 문제의 해결이 아니다.

## 8. 순환 조건의 공간모드 병목과 생산적인 선회

유한계의 k=0 모드가 안정해도 continuum 전체가 안정한 것은 아니다. 이를 기존 29장의 네 궤도를 재판정한 것처럼 쓰지 않고, 같은 질량구조의 별도 진단으로 검사했다. theta_dot=7.2를 외부에서 일정하게 공급하면 x=1+.7 cos(2.4t)이고 각 공간모드는 Mathieu 문제다. 재가열/parametric resonance의 일반적 구조는 [5]에 있다.

한 주기 전달행렬 S_k에서 k=0은 |tr S_k/2|=.956996<1로 안정하지만 k=.6은 1.071926>1이었다. 첫 불안정 구간의 k^2 경계는 0.0796858과0.7790404다. 독립적인 Mathieu characteristic values와 실제 전달행렬 적분이 일치한다.

양의 정부호 공분산 Gamma=S Gamma S^T가 있으면 Gamma^(-1/2) S Gamma^(1/2)가 orthogonal이다. 따라서 고유값 절댓값이1보다 큰 이 모드에는 양의 유한 invariant covariance가 없다. 이는 공급한 회전 배경의 반례이며 모든 순환우주의 불가능성 명제가 아니다.

생산적인 다음 경로는 이런 모드를 삭제하는 것이 아니라, 그 모드의 생성과 반작용을 같은 식(6)–(13)에 남겨 실제 전체 이력을 구하는 것이다. 주기성을 요구한다면 모든 k의 covariance와 경계 응력까지 같은 해에서 검사해야 한다. 균일모드 반복만으로 공간 전체의 반복을 선언하지 않는다.

## 9. 이번에 줄인 입력과 그대로 남은 최종 목표

이번에 독립 입력에서 제거한 것은 주어진 작용·초기상태 아래의 마찰함수, 생성량 함수, 별도 초기 이후 점유곡선, 별도 물질 교환함수와 공급된 팽창 이력이다. 실시간 양자 응답과 응력을 보유하므로 그 값들이 함께 산출된다. 기존 delta Z와 새로운 흡수 스펙트럼도 식(3)으로 연결된다.

여전히 남은 것은 s,epsilon,f,M_R,xi 및 유한 local matching의 선택, Lambda_R와 원시 상태의 선택이다. 특히 상수 Lambda_R를 바꾸면 theta 미분이 아니라 metric 식이 바뀐다. 실시간 계산을 추가했다고 이 독립 상수가 자동으로 고정되지는 않는다. CMB/recombination, 원시 perturbation, 실제 H0 likelihood는 실행하지 않았다. 전체 QFT–GR 및 네 힘의 미시 기원 목표도 유지한다.

따라서 현재 전진은 “상태를 준 정적 반응”에서 “상태·집단장·계량이 서로 반응하는 4차원 계산”으로 이동한 것이다. 다음 공통 경계문제는 이 forward solver의 초기상태와 matching을 선택하면서 continuum의 모든 물리 모드를 통과해야 한다. 그 다음에 동일 입력으로 초기 생성–재결합–거리–군집을 비교한다.

## 10. 검증과 재현

새 코드에는 모델 API, 관측 피팅 또는 remote write가 없다. numpy/scipy/mpmath/sympy가 필요하며 실행 환경은 results에 기록한다. 다음은 이번 실제 실행 명령이다.

```bash
OPENBLAS_NUM_THREADS=1 python -W error causal_spectrum_check.py --section analytic --out analytic_results.json
OPENBLAS_NUM_THREADS=1 python -W error causal_spectrum_check.py --section pulse --out pulse_results.json
OPENBLAS_NUM_THREADS=1 python -W error causal_spectrum_check.py --section joint --out joint_results.json
OPENBLAS_NUM_THREADS=1 python -W error causal_spectrum_check.py --section flrw --out flrw_results.json
OPENBLAS_NUM_THREADS=1 python -W error geometry_extension.py
OPENBLAS_NUM_THREADS=1 python -W error independent_checks.py
OPENBLAS_NUM_THREADS=1 python -W error verify_realtime.py
```

모든 집중 검사가 통과했다. 이는 수학/수치 정합성 게이트이며, 발견 개수·완성률·자연의 검증을 뜻하지 않는다. 비교적 거친 시간표본의 미분 오차도 숨기지 않고 결과에 둔다. 전체 저장소 회귀와 고차 loop, full quantum-state stability는 실행하지 않았다. 바이너리 시간 이력은 배포 묶음에 포함한다. 이 응답에서는 원격 Git을 변경하지 않았다. 향후 저장소에 반영할 텍스트 후보는 소스·텍스트 결과·문서이며, 이번 manifest에 파일별 해시를 남긴다.

## 1차 출처

[1] Ramsey & Hu, O(N) Quantum Fields in Curved Spacetime, Phys. Rev. D56, 661 (1997), https://arxiv.org/abs/gr-qc/9706001 . CTP/2PI와 상태–배경의 causal backreaction 방법.

[2] Kainulainen & Koskivaara, Non-equilibrium dynamics of a scalar field with quantum backreaction, JHEP 12 (2021) 190, https://arxiv.org/abs/2105.09598 . 비평형 모드와 평균장의 자기정합 처리. 해당 모형을 CE로 채택하지 않는다.

[3] Dumlu, On the Quantum Kinetic Approach and the Scattering Approach to Vacuum Pair Production, Phys. Rev. D79, 065027 (2009), https://arxiv.org/abs/0901.2972 . Bogoliubov/kinetic/scattering의 일반적 방법. 전기장 입자생성 실험을 이번 CE 검증으로 인용하지 않는다.

[4] Ferreiro, Monin & Torrenti, Physical scale adiabatic regularization in cosmological spacetimes, Phys. Rev. D109, 045015 (2024), https://arxiv.org/abs/2311.08986 ; del Rio & Navarro-Salas, Equivalence of Adiabatic and DeWitt-Schwinger renormalization schemes, https://arxiv.org/abs/1412.7570 . 4차 단열 감산, 국소 기하 matching, 보존과 anomaly. 식(11)의 특정 정규화는 여기서 직접 재유도했다.

[5] Kofman, Aspects of Preheating after Inflation, https://arxiv.org/abs/astro-ph/9802221 . Mathieu 안정도와 parametric resonance의 선행 방법. 순환우주의 관측 증거가 아니다.

전단계: 첨부 CE식.txt, 저장소20–22장의 공통 반응과 운동계수, 29장의 유한 공동 상태, 30장의 relative curvature source. 기존 입력 회계와 기각된 직접 질량 대응을 보존한다.
