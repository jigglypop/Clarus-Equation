# CE 우주 밀도 사상 1단계 수학 검산

Status: COMPLETE

검산 기준일: 2026-08-15  
계약: [00-contract.md](00-contract.md)  
독립 계산: [verify_density_bridge_math.py](artifacts/verify_density_bridge_math.py)

## 0. 판정 요약

| 계약 명제 | 형식 지위 | 수학 판정 | 등급 |
|---|---|---|---|
| B1 변분 임베딩 | [정리: 존재구성] (작용 선택은 [공리]) | 주어진 \(v_D\)의 균일 Euler--Lagrange 조건은 정확히 고정점 식이다 | PASS |
| B2 가지와 안정성 | [정리] | \(q_*\)는 허용구간의 유일한 전역 최소, \(x=1\)은 한쪽 최대이자 tachyonic 정지점이다 | PASS |
| B3 공변 동역학 | [산출]과 국소 [정리] | 변분, stress, FLRW 식, 국소 ghost/gradient/tachyon 검사는 성립한다 | PASS (국소) |
| B4 \(x=q_*\Rightarrow\Omega_b\) | 긍정 명제는 [미완성]; 불가능성은 [정리: no-go] | 같은 \(q_*\)에 서로 다른 stress와 \(\Omega_x\)가 가능하다 | P0 반례 |
| B5 conserved dust와 상수 attractor | [정리: no-go] | 상수 \(q_*\)와 \(\Omega_b\)가 구간에서 같으려면 \(w_{\rm tot}=0\)이어야 한다 | P0 반례 |
| B6 물리 지위 회계 | [산출] | 새 관측 [예측]은 없고, 현재값 동일시는 경계 [공리]로만 존재구성 가능하다 | P1 공백 |

따라서 첫 수학 다리에서 성립한 것은 “Poisson 고정점 식을 한 국소
canonical scalar의 안정한 정지조건으로 구현할 수 있다”는 존재구성이다.
그 장의 값, 그 장의 stress, conserved baryon number와 오늘의
critical-density fraction은 서로 다른 객체다. 주어진 작용만으로 이들을
동일시하는 정리는 나오지 않는다.

선행 run에서 이미 검산한 \(D>1\) 고정점의 존재·유일성과 Lambert-\(W\)
표현은 [선행 수학 검산 §2](../cosmology-theory-repository-audit-20260815/11-math.md)에
의존하며 여기서 재증명하지 않았다. 이번 계산은 그 근을 입력받은 뒤의
새 B1--B6만 독립 검산한다.

## 1. 정의역, 전제와 자유 자료

자연단위 \(c=\hbar=1\), metric \((-+++)\)와
\(M_{\rm Pl}^{-2}=8\pi G\)를 쓴다. 중력과 다른 물질까지 필요할 때의 전체
작용은

\[
S=S_{\rm EH}+S_x+S_{\rm other},\qquad
S_{\rm EH}=\frac{M_{\rm Pl}^2}{2}\int d^4x\sqrt{-g}\,R
\]

로 둔다. 계약의 후보는

\[
S_x=\int d^4x\sqrt{-g}\left[-\frac{F^2}{2}(\nabla x)^2
-M^4v_D(x)\right],
\]

\[
v_D=x\log x-x+D\left(x-\frac{x^2}{2}\right)+C,
\quad D>1,\quad 0<x\le1
\]

이다. \(D\)는 고정점 family의 외부 입력이고 \(F>0,M>0,C\in\mathbb R\)는
이 작용이 추가한 자유 자료다. \(C\)는 \(x\) 운동방정식에는 없지만
metric과 결합하면 진공 stress 및 Friedmann 식에 나타나므로 물리적으로
무해한 상수가 아니다.

차원은

\[
[x]=[D]=[v_D]=0,\quad [F]=[M]=[M_{\rm Pl}]=[H]=1.
\]

따라서 \(F^2(\nabla x)^2\)와 \(M^4v_D\)의 차원은 4이고,
\(\log x\)의 인자는 무차원이다. canonical 변수
\(\varphi=Fx\)로 쓰면 로그는 \(\log(\varphi/F)\)이므로 역시 무차원이다.

## 2. B1: 장 변분과 균일 정지조건

compact-support variation 또는 경계에서 \(\delta x=0\)을 두면

\[
\delta S_x=\int d^4x\sqrt{-g}
\left(F^2\Box x-M^4v_D'(x)\right)\delta x
\]

이고

\[
F^2\Box x-M^4v_D'(x)=0,\qquad
v_D'(x)=\log x+D(1-x).
\]

균일하고 시공간적으로 상수인 해에서는 \(\Box x=0\)이므로

\[
v_D'(x)=0
\iff \log x=-D(1-x)
\iff x=e^{-D(1-x)}.
\]

즉 계약의 potential은 관측 \(\Omega_b\) 숫자를 넣지 않고 요구된 두
고정점의 정지조건을 구현한다. 다만 이는 식을 적분해 potential을 만든
명시적 존재구성이다. 임의의 양의 함수 \(w(x)\)에 대해

\[
\widetilde v'(x)=w(x)\,[\log x+D(1-x)]
\]

도 같은 정지점 집합을 보존한다. 따라서 고정점 식은 kinetic term,
potential의 곡률과 상호작용을 유일하게 고르지 않는다.

## 3. B2: 두 가지, Hessian과 경계

두 번째와 세 번째 미분은

\[
v_D''(x)=\frac1x-D,\qquad v_D'''(x)=-\frac1{x^2}.
\]

선행 고정점 정리에 의해 작은 근은 \(0<q_*<1/D\)이고 다른 근은
\(x=1\)이다. 따라서

\[
v_D''(q_*)=\frac1{q_*}-D>0,\qquad
v_D''(1)=1-D<0.
\]

\(v_D'\)는 \((0,q_*)\)에서 음수, \((q_*,1)\)에서 양수다. 그러므로
\(q_*\)는 허용구간의 유일한 전역 최소이고 \(x=1\)은 한쪽 국소 최대다.
또한

\[
\lim_{x\to0^+}v_D(x)=C,\qquad
v_D(q_*)=C-q_*+\frac D2q_*^2<C,
\]

여서 \(x\to0^+\) 경계도 최소가 아니다.

독립 수치는 다음과 같다.

| \(D\) 입력 | \(q_*\) | 고정점 잔차 | \(v''(q_*)\) | \(v''(1)\) |
|---:|---:|---:|---:|---:|
| 3.1777584234099736 | 0.04864671964402821 | \(-1.39\times10^{-17}\) | 17.3786122285 | -2.17775842341 |
| 3.17776 | 0.04864663333721407 | \(-6.94\times10^{-18}\) | 17.3786471221 | -2.17776 |

여기서 “안정”은 \(q_*\) 근방의 국소 장론 안정성이다. 선언한
\((0,1]\)은 Lorentzian 시간발전에 대해 자동으로 불변인 field space가
아니다. \(x\to0^+\)에서 potential 높이는 유한하고, \(x>1\)로 단순
연장하면 \(-Dx^2/2\) 때문에 아래로 무한해진다. 충분히 큰 초기 운동에너지를
가진 해를 막는 벽이나 UV completion이 없으므로 전역 안정·모든 초기값의
attractor라는 명제는 성립하지 않는다.

## 4. B3: metric 변분, FLRW와 국소 섭동

metric 변분은

\[
T^{(x)}_{\mu\nu}
=F^2\nabla_\mu x\nabla_\nu x
-g_{\mu\nu}\left[\frac{F^2}{2}(\nabla x)^2+M^4v_D(x)\right]
\]

를 준다. \(ds^2=-dt^2+a^2(t)d\boldsymbol x^2\)의 균일장에서는

\[
\ddot x+3H\dot x+\frac{M^4}{F^2}v_D'(x)=0,
\]

\[
\rho_x=\frac{F^2}{2}\dot x^2+M^4v_D(x),\qquad
p_x=\frac{F^2}{2}\dot x^2-M^4v_D(x),
\]

\[
\dot\rho_x+3H(\rho_x+p_x)=0.
\]

최소결합된 다른 sector와 함께라면 평탄 Friedmann 식은

\[
3M_{\rm Pl}^2H^2=\rho_{\rm other}+\rho_x,\qquad
-2M_{\rm Pl}^2\dot H=\rho_{\rm other}+p_{\rm other}+F^2\dot x^2
\]

이다. 전체 stress 보존은 Bianchi identity에서, 분리된 \(x\) stress
보존은 위 운동방정식에서 따른다.

### 4.1 ghost, gradient와 tachyon

\(x=q_*+\delta x\), \(\phi=F\delta x\)로 두면 고정 metric에서 이차작용은

\[
S_x^{(2)}=\int\sqrt{-g}\left[-\frac12(\nabla\phi)^2
-\frac12m_*^2\phi^2\right],
\qquad
m_*^2=\frac{M^4}{F^2}\left(\frac1{q_*}-D\right)>0.
\]

- \(F^2>0\): 시간 kinetic의 부호가 양수이므로 ghost가 없다.
- 공간 gradient도 같은 canonical 계수여서 \(c_s^2=1>0\)이다.
- \(m_*^2>0\): 작은 가지에는 tachyon이 없다.
- \(x=1\)에서는 \(m_1^2=(M^4/F^2)(1-D)<0\)이므로 tachyonic이다.

이 판정은 \(q_*\) 주위의 고전적·국소적 판정이다. 비다항 EFT의 cutoff,
loop 안정성과 field-space 경계의 양자 처방은 주어지지 않았다.

### 4.2 attraction 시간척도

\(H>0\)가 한 relaxation 동안 상수라고 근사하면

\[
\delta\ddot x+3H\delta\dot x+m_*^2\delta x=0,
\quad
\lambda_\pm=\frac{-3H\pm\sqrt{9H^2-4m_*^2}}2.
\]

\(m_*<3H/2\)에서는 느린 amplitude 시간척도가

\[
\tau_{\rm slow}^{-1}
=\frac{3H-\sqrt{9H^2-4m_*^2}}2
\simeq\frac{m_*^2}{3H},
\]

\(m_*>3H/2\)에서는 진동 envelope가 \(e^{-3Ht/2}\)라서
\(\tau_{\rm amp}=2/(3H)\)다. 예를 들어 \(m_*/H=0.1,1,2\)이면
\(\tau H=299.666,2.618,0.667\)이다. \(H=0\)에서는 damping이 없어
국소 최소일 뿐 attractor가 아니다. 또한 \(m_*/H\)는

\[
\frac{m_*}{H}=\frac{M^2}{FH}\sqrt{1/q_*-D}
\]

로서 \(F,M\)과 배경 \(H\)에 의존하는 자유 비다.

빠른 작은 진동 \(m_*\gg H\)의 평균 에너지는

\[
\langle\rho_{\rm osc}\rangle\simeq
\frac12M^4v_D''(q_*)A_x^2\propto a^{-3}
\]

가 될 수 있지만 진폭 \(A_x\)는 초기조건이다. 최소점 위치 \(q_*\)가
진동 에너지의 정규화를 고정하지 않는다.

## 5. B4: \(x=q_*\)에서 \(\Omega_b\)가 나오지 않는 정확한 이유

### 5.1 같은 \(q_*\), 서로 다른 stress

정지해에서

\[
\rho_x=M^4v_D(q_*),\qquad p_x=-\rho_x,\qquad
\Omega_x=\frac{M^4v_D(q_*)}{3M_{\rm Pl}^2H^2}.
\]

반면 \(q_*\)는 \(D\)만으로 정해지고 \(F,M,C,H\)를 모른다. 특히

\[
C_0=q_*-\frac D2q_*^2
\]

를 택하면 \(v_D(q_*)=0\)이다. 같은 \(D,q_*,F,M\)에서
같은 양의 다른 물질밀도 \(\rho_{\rm other}=M^4\)를 둔 두 Friedmann
배경에서 \(C=C_0+1/4\)로만 바꾸면

\[
\Omega_x(C_0)=0,\qquad
\Omega_x(C_0+1/4)=\frac{M^4/4}{M^4+M^4/4}=0.2.
\]

두 경우의 정지점과 Hessian은 완전히 같다. 이것은 “\(q_*\)만으로
energy fraction이 정해진다”는 부모 명제의 완전 반례다.

더 직접적으로, 비영 정지 stress의 상태방정식은 \(w_x=-1\)인 반면
표준 비상대론적 baryon dust는 \(w_b=0\)이다. 따라서 \(S_x\)의 상수
정지 stress 자체를 baryon stress라고 동일시할 수도 없다.

### 5.2 current가 추가로 필요한 이유

이 real scalar의 potential은 연속 shift symmetry를 깨므로 baryon number로
읽을 Noether current가 없다. 상수해에서는 \(\nabla_\mu x=0\)이므로
\(x\)의 도함수로 만든 국소 벡터도 0이다. nonzero timelike baryon current를
얻으려면 독립 fluid 변수나 최소한 추가 phase 자유도가 필요하다.

표준 dust sector를 외부에서 공급한다고 하면 최소 covariant 자료는

\[
J_b^\mu=n_bu^\mu,\quad u^\mu u_\mu=-1,\quad
\nabla_\mu J_b^\mu=0,\quad
T_b^{\mu\nu}=m_bn_bu^\mu u^\nu
\]

다. 그래도 \(n_b\)의 적분상수는 \(q_*\)로 정해지지 않는다. 현재값을
연결하려면 별도의, covariantly 지정된 spacelike hypersurface
\(\Sigma_*\)에서

\[
\left.\frac{m_b(-u_\mu J_b^\mu)}{3M_{\rm Pl}^2H^2}\right|_{\Sigma_*}
=\left.x\right|_{\Sigma_*}=q_*
\tag{boundary bridge}
\]

를 부과해야 한다. 표준 baryon sector가 이미 주어졌다는 전제에서는 이
한 줄이 최소 추가 [공리: 물리 사상+경계조건]이다. \(S_x\)만에서
시작하면 current-bearing 자유도, 그 작용, 그리고 위 정규화 경계조건이
모두 추가로 필요하다. \(\Sigma_*\)를 오늘이라고 고르는 것은 유도가 아니라
현재-time boundary 선택이며, 온도나 곡률로 고른다면 그 기준 scale이 새
자유 자료다.

### 5.3 확률과 energy fraction 사이의 정확한 항등식

고정점 확률을 \(q=P(E)\), 각 event가 갖는 양의 에너지 weight를 \(W\)라
하고 energy-weighted fraction을

\[
\Omega_E:=\frac{\mathbb E[W\mathbf1_E]}{\mathbb E[W]}
\]

로 정의하면

\[
\Omega_E-q
=\frac{\operatorname{Cov}(\mathbf1_E,W)}{\mathbb E[W]}
=\frac{q(1-q)
\left(\mathbb E[W\mid E]-\mathbb E[W\mid E^c]\right)}
{\mathbb E[W]}.
\]

\(0<q<1\), \(0<\mathbb E[W]<\infty\)에서

\[
\Omega_E=q
\iff \mathbb E[W\mid E]=\mathbb E[W\mid E^c].
\]

따라서 확률과 energy fraction의 동일시는 일반 정리가 아니다. 두
조건부 평균 에너지가 같다는 독립 가정이 정확히 필요하다. 독립성
\(W\perp\mathbf1_E\)은 충분하지만 필요조건보다 강하다. 이 항등식은
확률-to-energy bridge의 숨은 가정을 드러낼 뿐 spacetime current,
Friedmann normalization이나 현재 hypersurface를 제공하지 않는다.

## 6. B5: 보존법칙의 no-go

FLRW에서 \(\nabla_\mu J_b^\mu=0\)은

\[
\dot n_b+3Hn_b=0,\qquad
\rho_b=m_bn_b\propto a^{-3}
\]

를 준다. 평탄 Friedmann 배경에서
\(x_b:=\rho_b/\rho_{\rm tot}=\Omega_b\)라 두면 총 연속방정식과 함께

\[
\frac{\dot x_b}{x_b}
=\frac{\dot\rho_b}{\rho_b}
-\frac{\dot\rho_{\rm tot}}{\rho_{\rm tot}}
=-3H+3H(1+w_{\rm tot})
=3Hw_{\rm tot}.
\tag{no-go}
\]

따라서 \(x_b=q_*\)가 열린 시간구간에서 상수이면 정확히
\(w_{\rm tot}=0\)이어야 한다. 같은 결과는

\[
\frac{d\log\Omega_b}{d\log a}
=-3-2\frac{d\log H}{d\log a}=0
\]

에서 \(d\log H/d\log a=-3/2\)로도 나온다. radiation--matter--dark-energy
혼합 우주에서는 conserved dust fraction과 상수 attractor를 동일시할 수
없다.

명시적 반례로
\((\Omega_{r0},\Omega_{m0},\Omega_{\Lambda0})=
(0.000092,0.310968903,0.688939097)\)인 평탄 배경에서 오늘
\(\Omega_b(1)=q_*=0.0486466333372\)로 정규화하면

| \(a\) | conserved \(\Omega_b(a)\) | \(\Omega_b-q_*\) |
|---:|---:|---:|
| \(10^{-6}\) | 0.000526986488 | -0.048119646849 |
| \(10^{-3}\) | 0.120720564023 | +0.072073930686 |
| 0.5 | 0.122452231004 | +0.073805597667 |
| 1 | 0.0486466333372 | 0 |
| 2 | 0.0083548994559 | -0.040291733881 |

즉 한 시점의 일치는 경계 정규화로 언제든 만들 수 있지만 attractor가
현재 시점이나 current 적분상수를 선택하지 않는다.

상수 fraction을 강제로 추적시키는 reacting fluid를 생각하면

\[
\dot\rho_b+3H\rho_b=Q_b,\qquad
\rho_b=q_*\rho_{\rm tot}
\]

에서 필요한 source는 유일하게

\[
Q_b=-3Hq_*p_{\rm tot}
=-3Hw_{\rm tot}\rho_b.
\]

질량이 일정하면
\(\nabla_\mu J_b^\mu=Q_b/m_b=-3Hw_{\rm tot}n_b\)이므로
\(w_{\rm tot}\ne0\)에서 current 보존을 정면으로 위반한다. 위 수치
배경에서 \(a=10^{-3}\)에는 \(-0.228305\,H\rho_b\)의 소멸,
\(a=1\)에는 \(+2.066725\,H\rho_b\)의 생성이 필요하다.

## 7. B6: 숨은 공리, 자유도와 관측 독립성

| 항목 | 고정점 위치에 영향 | density/stress에 영향 | 현재 지위 |
|---|---:|---:|---|
| \(D\) | 있음 | 간접 | 외부 [공리]/입력 |
| potential 함수형 선택 | roots를 보존하게 만들 수 있음 | 곡률·상호작용 변경 | 모델 [공리] |
| \(F,M\) | 없음 | 질량, relaxation, 에너지 scale | 연속 자유 매개변수 2개 |
| \(C\) | 없음 | 진공 stress 직접 변경 | 연속 자유 매개변수 1개 |
| \(x_i,\dot x_i\) | 없음 | basin, relaxation, 진동 밀도 | 초기조건 2개 |
| baryon current/action | 없음 | 입자수·dust stress를 정의 | 추가 물리 sector |
| current 정규화 | 없음 | \(\rho_b\) 절대값 | 적분상수/경계조건 |
| \(\Sigma_*\) | 없음 | 어느 시점의 fraction인지 선택 | freeze/current-time 공리 |
| \(x\leftrightarrow\Omega_b\) | 없음 | 물리 의미를 부여 | bridge [공리] |
| event energy \(W\)의 조건부분포 | 없음 | probability-to-energy readout | 추가 확률 법칙 |

potential은 관측 \(\Omega_b\)의 숫자를 literal로 포함하지 않지만, 알려진
고정점-밀도 대응을 만들기 위해 선택된 식이다. 특히 §5.2의 bridge와
[12-routes.md](12-routes.md)의 모든 후보는 이미 목표 대응을 안 뒤 만든
target-aware 구성이다. 독립 holdout이나 부수 관측량을 사전 고정하지
않았으므로 [예측]으로 분류할 항목은 없다.

## 8. P0/P1/P2 원장과 부모 명제 범위

### P0

1. **\(x=q_*\Rightarrow\Omega_b\) 반례.** 같은 \(q_*\)에서
   \(C=C_0\)와 \(C_0+1/4\)가 같은 \(\rho_{\rm other}=M^4\)에서 각각
   \(\Omega_x=0,0.2\)를 준다.
   무너지는 범위는 “고정점 값만으로 density fraction이 정해진다”이다.
   potential의 정지점 존재는 보존된다.
2. **stress 종류 반례.** 상수 \(q_*\)의 비영 stress는 정확히
   \(p_x=-\rho_x\)이고 baryon dust는 \(p_b=0\)이다. 무너지는 범위는
   “정지 scalar stress 자체가 baryon stress다”이다. 별도 current에 대한
   경계 readout 가능성은 이 반례의 대상이 아니다.
3. **시간의존성 반례.** conserved dust를 \(a=1\)에서 \(q_*\)로 맞춰도
   \(a=0.5\)에서 \(\Omega_b=0.122452231004\ne q_*\)다. 무너지는 범위는
   “상수 attractor가 혼합 우주에서 \(\Omega_b(t)\)를 계속 고정한다”이다.
   한 hypersurface의 명시적 경계 공리는 가능하다.
4. **probability-to-energy 반례.** 조건부 평균이 다르면
   \(\Omega_E-q=q(1-q)(\mu_E-\mu_{E^c})/\mathbb E[W]\ne0\)다.
   무너지는 범위는 “event 확률이 자동으로 같은 event의 energy fraction과
   같다”이다.

### P1

1. \(F,M,C\), 두 초기조건과 field-space completion이 고정되지 않았다.
2. real scalar에는 nonzero conserved baryon current가 없고 current 정규화와
   covariant \(\Sigma_*\) 선택이 빠져 있다.
3. 같은 stationary set을 갖는 무한한 \(w(x)\) potential family가 있어
   주어진 \(v_D\)를 고르는 원리가 없다.
4. \(x\in(0,1]\) 경계가 전역 시간발전에 대해 불변이라는 증명이 없고,
   단순 \(x>1\) 연장은 아래로 무한하다.
5. energy weight의 조건부 평균 동일성, target-aware 후보 선택,
   freeze-out과 blind cross-prediction이 없다.

### P2

별도 수치 표기 오류는 발견하지 않았다. 다만 “attractor”라는 말은
\(H>0\), 국소 basin과 자유로운 \(m_*/H\)를 명시할 때만 사용해야 한다.

## 9. 재현

저장소 루트에서 다음을 실행한다.

    python "_workspace/ce/cosmology-density-bridge-derivation-20260815/artifacts/verify_density_bridge_math.py"

결과는 exit 0, ALL DENSITY-BRIDGE MATH CHECKS PASSED였다. 두 \(D\)
원장의 정지 잔차, Hessian 부호, 같은-\(q\) stress 반례, constant-\(H\)
감쇠 지수, conserved-dust 시간의존성과 모든 차원 항목을 독립 계산한다.

## 10. 종료 체크

- [x] B1--B6 모두 형식 지위와 수학 판정이 있다.
- [x] 모든 P0에 정확한 반례 값과 무너지는 부모 범위가 있다.
- [x] 수치가 artifacts의 독립 표준-library 스크립트로 재현된다.
- [x] 두 stationary branch, \(x\to0^+\)와 \(x=1\) 경계를 검사했다.
- [x] ghost, gradient, tachyon, stress, FLRW EOM과 attraction scale을 검사했다.
- [x] 12-routes.md에 다섯 구조적 후보와 dof/target-awareness/cross-test를 기록했다.
