# 7. CE scalar–tensor 블랙홀: 측정된 \(G_N\)과 profile 경계값 문제

## 7.1 교정할 핵심

과거 문서는 상수

\[
F=1+\alpha_sD_{\rm eff}\simeq1.3746
\]

를 Einstein–Hilbert 항 앞에 곱한 뒤 \(G_{\rm eff}=G/F\)라 두고, 같은
질량의 모든 블랙홀 길이가 \(1/F\)로 작아진다고 주장했다. 상수 \(F\)는
bare Newton 계수와 완전히 재매개화된다. 무한대에서 측정한 Newton 상수와
bare \(G\)를 동시에 같은 기호로 쓰고 다시 \(F\)로 나누면 같은 효과를 두 번
센 것이다.

블랙홀에서 관측 가능한 수정은 상수 재정의가 아니라

- 무한대와 강곡률 영역 사이에서 변하는 scalar profile,
- 그 profile의 stress tensor와 \(f(\phi)\) 미분항,
- 같은 작용을 만족하는 계량

에서 나와야 한다.

## 7.2 Jordan-frame 정규화

물질이 최소결합하는 physical metric을 \(g_{\mu\nu}\)라 하고, 무한대에서
측정한 Newton 상수를 \(G_N\)으로 고정한다. 이 절부터 7.3절의
작용·장방정식은 \(c=\hbar=1\) 단위를 쓴다. 7.4절 이후 길이·온도·엔트로피
관측식에서는 \(c,\hbar,k_B\)를 명시적으로 복원한다. 최소 후보 작용은

\[
\boxed{
S=\int d^4x\sqrt{-g}\left[
\frac{f(\phi)}{16\pi G_N}R
-\frac12 Z(\phi)(\nabla\phi)^2-U(\phi)
\right]+S_m[g,\Psi].
}
\]

무한대 정규화와 건강한 kinetic branch는

\[
f(\phi_\infty)=1,\qquad
f_{,\phi}(\phi_\infty)=0,\qquad
Z(\phi)>0
\]

로 둔다. 두 번째 조건은 무한대 배경에서 선형 scalar fifth force가
Newton 측정을 다시 바꾸지 않도록 하는 \(Z_2\)-호환 최소 조건이다.
그 조건을 쓰지 않으면 Cavendish \(G_N\)과 action parameter의 표준
scalar–tensor 변환을 별도로 포함해야 한다.

기존 \(R\phi^2\) 항을 이 표기에 옮기는 한 후보는

\[
f(\phi)=1+16\pi G_N\xi_R
\left(\phi^2-\phi_\infty^2\right)
\]

이다. 여기서 \(\xi_R\)은 curvature coupling이며 correlation length와
다른 기호다. 이 함수, \(Z\), \(U\)의 정규화와 renormalization prescription이
고정되기 전에는 CE 블랙홀 이론이 완전히 지정된 것이 아니다.

## 7.3 coupled field equations

위 작용을 변분하면

\[
\boxed{
fG_{\mu\nu}
=8\pi G_N\left(T_{\mu\nu}^{(m)}+T_{\mu\nu}^{(\phi)}\right)
+\nabla_\mu\nabla_\nu f-g_{\mu\nu}\Box f
}
\]

이고

\[
T_{\mu\nu}^{(\phi)}
=Z\,\nabla_\mu\phi\nabla_\nu\phi
-g_{\mu\nu}\left[\frac12Z(\nabla\phi)^2+U(\phi)\right].
\]

scalar 방정식은

\[
\boxed{
Z\Box\phi+\frac12Z_{,\phi}(\nabla\phi)^2
-U_{,\phi}+\frac{f_{,\phi}}{16\pi G_N}R=0.
}
\]

상수 \(f\)를 먼저 넣고
\(fG_{\mu\nu}=8\pi G_NT_{\mu\nu}^{(\phi)}\)만 남기는 것은 일반
scalar–tensor EOM이 아니다. profile이 있으면
\(\nabla_\mu\nabla_\nu f-g_{\mu\nu}\Box f\)가 반드시 함께 들어간다.

이 EOM은 **위 후보 작용을 채택했을 때 조건부로 exact**하다. CE가
\(f,Z,U\)를 유일하게 유도했다는 뜻은 아니다.

## 7.4 정적 구면대칭 경계값 문제

Schwarzschild-like 좌표에서

\[
ds^2=-e^{2\delta(r)}N(r)c^2dt^2
+\frac{dr^2}{N(r)}+r^2d\Omega^2,
\]

\[
N(r)=1-\frac{2G_Nm(r)}{c^2r},\qquad \phi=\phi(r)
\]

로 둔다. 미지함수는 \(m(r),\delta(r),\phi(r)\)다. 지평선과 무한대
경계조건은

\[
N(r_h)=0,\qquad
m(r_h)=\frac{c^2r_h}{2G_N},
\]

\[
\phi(r_h)\ {\rm finite},\qquad
\delta(r_h)\ {\rm finite},
\]

\[
\phi(\infty)=\phi_\infty,\qquad
\delta(\infty)=0,\qquad
m(\infty)=M_{\rm ADM},\qquad f(\phi_\infty)=1.
\]

지평선에서의 \(\phi'(r_h)\)는 임의 입력이 아니라 scalar EOM의 regularity
조건이 정한다. \(\phi(r_h)\) 또는 동등한 shooting datum을 바꾸어 무한대
조건을 만족시키는 해를 찾아야 한다.

따라서 문제는 “\(G\)를 \(G/F\)로 치환”하는 대수 문제가 아니라

\[
\mathcal B[f,Z,U;M_{\rm ADM}]
\longrightarrow
\{m(r),\delta(r),\phi(r)\}
\]

라는 nonlinear boundary-value problem이다.

## 7.5 상수 scalar branch

\[
\phi(r)=\phi_\infty,\qquad
f(\phi_\infty)=1,\qquad
U(\phi_\infty)=U_{,\phi}(\phi_\infty)=0
\]

이면 scalar stress와 \(f\) 미분항이 사라지고

\[
G_{\mu\nu}=0
\]

이 된다. 이 branch의 외부해는 측정된 \(G_N\)을 쓰는 표준 GR
Schwarzschild/Kerr 해다.

bare 표기에서 \(f_0\ne1\)인 상수로 시작해도

\[
G_N=\frac{G_{\rm bare}}{f_0}
\]

로 측정량을 정의하면 같은 결론이다. 따라서 상수 \(F=1.3746\)만으로

\[
r_h\to r_h/F,\quad
b_{\rm sh}\to b_{\rm sh}/F,\quad
\omega_{\rm QNM}\to F\omega_{\rm QNM}
\]

라는 관측 이동은 생기지 않는다.

## 7.6 profile이 생길 조건

\(Z_2\) 배경 \(\phi_\infty=0\) 근방에서 선형화하면

\[
Z_0\Box\,\delta\phi
-\left[
U_{,\phi\phi}(0)
-\frac{f_{,\phi\phi}(0)}{16\pi G_N}R_{\rm GR}
\right]\delta\phi=0.
\]

vacuum Schwarzschild와 Kerr 배경은 \(R_{\rm GR}=0\)이다. 따라서
\(U_{,\phi\phi}(0)>0\)인 단순 massive \(R\phi^2\) 후보는 선형 차수에서
곡률 유도 scalarization을 만들지 않는다. 비자명 profile을 얻으려면

- 다른 curvature invariant와의 결합,
- matter 환경,
- tachyonic eigenmode,
- 또는 명시적 boundary/source

중 무엇이 작동하는지 같은 작용에서 보여야 한다. 이는 단순히
\(D_{\rm eff}\)나 \(\alpha_s\)를 \(F\)에 대입하는 것으로 대체되지 않는다.

## 7.7 관측량의 올바른 정의

profile 해를 얻은 뒤에 다음을 계산한다.

- 지평선: \(N(r_h)=0\)
- photon sphere: 해당 해의 null effective potential의 극값
- ISCO: timelike effective potential의 안정성 경계
- shadow: 무한대 정규화가 고정된 null geodesic impact parameter
- QNM: coupled metric–scalar perturbation operator의 outgoing spectrum
- ADM mass: \(m(\infty)\)

Wald entropy는 이 후보 작용 안에서

\[
\boxed{
S_{\rm Wald}
=\frac{k_Bc^3}{4\hbar G_N}
f(\phi_h)A_h.
}
\]

온도는 실제 해의 surface gravity로

\[
T_H=\frac{\hbar\kappa_h}{2\pi k_Bc}
\]

를 계산한다. \(f(\phi_h)\), \(A_h\), \(\kappa_h\)가 profile EOM을 통해
서로 얽혀 있으므로 하나의 상수 \(F\) 배율표로 대체할 수 없다.

## 7.8 회전·전하 해

Reissner–Nordström, Kerr, Kerr–Newman도 상수 scalar branch에서는 표준
GR 식을 측정된 \(G_N\)으로 그대로 쓴다. 비자명 회전 profile은
\(\phi(r,\theta)\)와 stationary axisymmetric metric의 결합 PDE 문제다.
구면해의 \(G\to G/F\) 치환으로 얻을 수 없다.

## 7.9 수치 해법과 acceptance gate

정적 구면 branch의 최소 검증 순서는 다음과 같다.

1. \(f,Z,U\)와 단위·renormalization scale 고정
2. horizon series로 regular initial data 생성
3. collocation 또는 shooting으로 무한대 경계조건 해결
4. Hamiltonian/constraint residual과 grid convergence 확인
5. ghost \(Z>0\), hyperbolicity, radial·angular perturbation spectrum 검사
6. 같은 \(G_N\), \(M_{\rm ADM}\)에서 GR와 shadow/QNM 비교
7. weak-field fifth-force, binary-pulsar, GW propagation 제약과 joint fit

profile을 역으로 골라 metric을 맞추는 것은 해 구성일 수 있지만, 고정된
작용의 EOM과 경계조건을 통과하기 전에는 물질 모형의 증명이 아니다.

## 7.10 상태표

| 명제 | 현재 상태 |
|---|---|
| 상수 \(F\)는 bare \(G\)에 흡수됨 | Exact reparameterization |
| measured-\(G_N\) Jordan-frame EOM | 후보 작용에 조건부 Exact |
| 상수 \(Z_2\) scalar branch | GR branch |
| \(F=1.3746\)에 의한 보편 \(0.727\) 길이 이동 | Refuted |
| 비자명 정적 CE profile | Open boundary-value problem |
| 회전 profile과 coupled QNM | Open PDE/spectral problem |
| CE가 \(f,Z,U\)를 유일하게 고정 | Open |

현재 닫힌 물리 결론은 상수 form factor가 독립 블랙홀 예측이 아니라는
것이다. 새로운 블랙홀 현상은 측정된 \(G_N\)을 고정한 뒤 비자명
\(\phi(r,\theta)\) profile을 실제 EOM으로 풀 때만 정의된다.
