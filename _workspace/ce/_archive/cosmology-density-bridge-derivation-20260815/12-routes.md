# CE 우주 밀도 사상 우회 경로

Status: COMPLETE

목표량은 “관측 중심값을 coupling에 넣지 않고, \(D>1\) 고정점의 작은 근
\(q_*\)를 conserved baryon current가 만드는 현재의 무차원 fraction
\(\Omega_b(t_0)\)와 연결하는 것”이다. 기본 \(S_x\)가 강제하는 것은
정지점 위치와 국소 scalar 질량의 부호뿐이다. 자유 부분은 current-bearing
자유도, energy weight, density 정규화, \(F,M,C\), 초기조건과 freeze
hypersurface다.

모든 후보는 알려진 \(q_*\leftrightarrow\Omega_b\) 목표를 본 뒤 구성했으므로
target-aware는 “예”다. 어느 후보도 관측 \(\Omega_b\) 숫자를 action 계수에
literal로 넣지는 않지만, 그것만으로 blind prediction이 되지는 않는다.

## 1. 후보 비교

| 순위 | 구조적 경로 | 후보당 새 공리 | 추가 dof/선택 회계 | target-aware | 수치 또는 조건부 교차 산출 | 죽이는 시험 |
|---:|---|---|---|---|---|---|
| 1 | A. 독립 conserved dust + 한 hypersurface 경계 bridge | \(\Sigma_*\)에서 \(\Omega_b=x=q_*\)라는 경계 사상 1개 | 표준 baryon sector를 외부 공급하면 연속 선택 \(\ge1\): \(\Sigma_*\)의 clock/scale; 모델 선택 1 | 예 | \(a_*=1\) 정규화 시 \(\Omega_b(0.5)=0.1224522\), \(\Omega_b(10^{-3})=0.1207206\) | 경계 밖에서도 \(\Omega_b=q_*\)라고 주장하거나, \(\Sigma_*\)가 독립적으로 지정되지 않으면 실패 |
| 2 | E. energy-weighted event theorem | \(\mathbb E[W\mid E]=\mathbb E[W\mid E^c]\)라는 weight 대칭 1개 | 일반 joint law는 함수형 dof; equality만 쓰면 평균차 1개를 0으로 제한, 모델 선택 1 | 예 | 평균 weight가 2 대 1이면 \(q=0.0486466\)에서 \(\Omega_E=0.0927798\), 동일 평균이면 정확히 \(q\) | 두 conditional mean energy가 다르거나 spacetime current가 요구되면 실패 |
| 3 | B. \(U(1)\) phase를 가진 complex-scalar current | real \(x\)를 radial mode로 갖는 \(U(1)\) completion 1개 | 추가 phase 1개, conserved charge \(Q\) 1개, radial 초기자료; 모델 선택 1 | 예 | \(Q\ne0\)이면 \(x=q_*\)에서 radial EOM 잔차 \(-q_*\dot\theta^2\ne0\); 고정 radius의 charge energy는 \(a^{-6}\) | nonzero current와 정확한 \(x=q_*\)를 동시에 요구하거나, dust/SM baryon 섭동·상호작용을 못 맞추면 실패 |
| 4 | C. conserved dust의 scalar-dependent mass | \(m_b(x)\) coupling 함수 1개 | 최소 지수형 \(m_b=m_0e^{\beta(x-q_*)}\)에서도 \(\beta\)와 \(n_{b0}\) 2개; 일반 함수면 함수형 dof | 예 | \(\dot\rho_b+3H\rho_b=(d\ln m_b/dx)\dot x\rho_b\); \(x=q_*\) 뒤에는 dust지만 \(n_{b0}\)는 자유 | equivalence-principle/fifth-force·질량변화 제한, 또는 \(n_{b0}\)가 \(q_*\)에서 유도되지 않으면 실패 |
| 5 | D. reacting two-fluid constant-fraction tracker | \(Q_b=-3Hq_*p_{\rm tot}\) transfer law 1개 | exact law면 새 연속계수 0, exchange 방향/sector 모델 선택 1; 초기 transient 1 | 예 | \(a=10^{-3}\): \(Q/(H\rho_b)=-0.228305\); \(a=1\): \(+2.066725\) | \(\nabla_\mu J_b^\mu=0\) 또는 표준 \(\rho_b\propto a^{-3}\)를 요구하는 즉시 실패 |

순위는 “목표를 맞추는 정도”가 아니라 새 구조가 적고 실패 조건이 투명한
순서다. A는 현재값 동일시를 솔직한 경계 공리로 격리한다. E는 확률과
energy fraction 사이에 정확히 빠진 통계 조건을 보인다. B와 C는 current를
공변적으로 만들지만 정규화를 유도하지 못한다. D는 fraction을 동적으로
추적할 수 있으나 계약 B5의 conserved-current 전제를 버린다.

## 2. Route A — snapshot boundary current

표준 dust를

\[
J_b^\mu=n_bu^\mu,\quad \nabla_\mu J_b^\mu=0,\quad
T_b^{\mu\nu}=m_bn_bu^\mu u^\nu
\]

로 두고, covariant clock \(\chi\)의 level set
\(\Sigma_*:\chi=\chi_*\)에서 한 번만

\[
\Omega_b|_{\Sigma_*}=x|_{\Sigma_*}=q_*
\]

를 부과한다. 이는 total stress와 current 보존을 함께 만족하는
존재구성이다. 그 뒤에는

\[
\Omega_b(a)=q_*\left(\frac{a_*}{a}\right)^3
\frac{H_*^2}{H(a)^2}
\]

가 강제된다. 따라서 \(\Omega_b=x\)는 일반적으로 \(\Sigma_*\) 밖에서
깨진다. \(\chi_*\) 또는 그 물리 scale을 이론이 고르지 않으면 현재 시점의
일치는 경계 선택이다. 이 경로에는 독립 수치 cross-prediction이 없고,
주어진 배경에서 fraction의 이후/이전 변화를 재생하는 조건부 산출만 있다.

## 3. Route E — probability에서 energy fraction으로

event \(E\)의 확률이 \(q=P(E)\), 각 event의 양의 energy weight가 \(W\)일
때 자연스러운 energy fraction은

\[
\Omega_E=\frac{\mathbb E[W\mathbf1_E]}{\mathbb E[W]}
\]

이다. 정의만 전개하면

\[
\Omega_E-q
=\frac{\operatorname{Cov}(\mathbf1_E,W)}{\mathbb E[W]}
=\frac{q(1-q)(\mu_E-\mu_{E^c})}{\mathbb E[W]},
\]

\[
\mu_E=\mathbb E[W\mid E],\qquad
\mu_{E^c}=\mathbb E[W\mid E^c].
\]

\(0<q<1\)이고 평균 energy가 유한·양수이면

\[
\Omega_E=q\iff\mu_E=\mu_{E^c}.
\]

독립성은 충분조건이지만 필요한 것은 conditional mean equality뿐이다.
\(\mu_E=2,\mu_{E^c}=1\)인 완전 반례에서는

\[
q=0.0486466333372,\qquad
\Omega_E=\frac{2q}{1+q}=0.0927798398254,
\]

즉 차이는 0.0441332064881이다. weight 대칭을 새 공리로 두면 snapshot의
확률-to-energy 비는 같아지지만, 왜 실제 extinction/event class의 평균
에너지가 같은지와 \(W\)의 microphysics는 남는다. 또한 이 route만으로는
covariant baryon current, \(a^{-3}\) scaling, Friedmann denominator 또는
현재 hypersurface가 생기지 않는다.

교차 시험은 event label을 blind하게 유지한 energy distribution에서 두
conditional mean을 비교하는 것이다. 평균차가 하나라도 검출되면 equality
bridge는 즉시 죽는다.

## 4. Route B — charged complex completion

한 real scalar에는 nonzero charge current가 없으므로 phase를 추가해

\[
\mathcal L=-\frac{F^2}{2}
\left[(\nabla x)^2+x^2(\nabla\theta)^2\right]-M^4v_D(x)
\]

를 후보로 둔다. shift \(\theta\mapsto\theta+\text{const}\)의 current와 FLRW
charge는

\[
j^\mu=F^2x^2\nabla^\mu\theta,\qquad
a^3F^2x^2\dot\theta=Q
\]

다. 그러나 radial EOM은

\[
\ddot x+3H\dot x-x\dot\theta^2+\frac{M^4}{F^2}v_D'(x)=0.
\]

\(x=q_*\)에서는 potential force가 0이므로 \(Q\ne0\)인 정확한 상수해가
아니다. 작은 charge의 준정적 shift는

\[
\delta x\simeq\frac{q_*\dot\theta^2}{m_*^2}\propto a^{-6}
\]

이고, radius를 억지로 고정하면 phase energy도 \(a^{-6}\)이라 dust가
아니다. 다른 potential regime에서 빠른 complex-scalar oscillation을
dust처럼 만들 수는 있지만 \(Q\)와 진폭이 새 초기조건이며 \(q_*\)가 그
정규화를 고르지 않는다. 또한 이 bosonic current를 SM baryon number와
연결하는 coupling이 별도로 필요하다.

## 5. Route C — conserved number, varying mass

이미 존재하는 baryon current의 입자수는 보존하면서

\[
\rho_b=m_b(x)n_b,\qquad
m_b(x)=m_0e^{\beta(x-q_*)}
\]

를 한 coupling 후보로 둔다. 그러면

\[
\nabla_\mu J_b^\mu=0,\qquad
\dot\rho_b+3H\rho_b=\beta\dot x\rho_b
\]

이고 반대 부호의 에너지 교환을 \(x\) sector에 넣으면 total stress는
보존된다. \(x\to q_*\) 뒤에는 \(m_b\to m_0\)이고 표준 dust scaling을
회복한다.

그러나 보존식의 적분상수 \(n_{b0}\)는 그대로 남는다. \(\beta\)는 relaxation
동안 fifth force와 시간가변 질량을 만들지만 density normalization을
고르지 않는다. 따라서 이 경로의 부수 시험은 equivalence-principle,
fifth-force, baryon-mass drift와 scalar perturbation이고, 핵심 kill test는
그 제한을 만족한 뒤에도 \(n_{b0}\)가 \(q_*\)에서 나오는지다.

## 6. Route D — reacting tracker

정말로 \(\Omega_b=q_*\)를 모든 epoch에서 유지하려면 total continuity가
transfer law를 역으로 고정한다.

\[
\dot\rho_b+3H\rho_b=Q_b,\qquad
Q_b=-3Hq_*p_{\rm tot}=-3Hw_{\rm tot}\rho_b.
\]

공유 four-velocity의 expansion \(\Theta=\nabla_\mu u^\mu=3H\)를 쓰면
\(Q_b=-\Theta q_*p_{\rm tot}\)라는 배경 공변형으로 쓸 수 있고, 다른
fluid가 \(-Q_b\)를 받게 해 total stress는 보존할 수 있다. 하지만 일정한
baryon mass에서는

\[
\nabla_\mu J_b^\mu=-3Hw_{\rm tot}n_b
\]

라서 baryon current는 matter-only epoch 외에는 보존되지 않는다. 이
경로는 exact tracker의 존재 예일 뿐 B5의 요구를 만족하는 후보가 아니다.
BBN--recombination--today 사이 baryon-to-photon ratio의 변화가 직접적인
부수 시험이다.

## 7. 경로별 남은 구조와 종료 체크

- A에 필요한 빠진 구조: \(\Sigma_*\)를 관측 현재시간과 무관하게 정하는
  clock law. 이를 주지 않으면 boundary normalization 이상의 내용이 없다.
- E에 필요한 빠진 구조: event energy \(W\)의 microphysics와 조건부 평균
  equality의 대칭 원리, 그리고 spacetime current.
- B에 필요한 빠진 구조: charge normalization과 SM baryon quantum number,
  dust-like perturbation을 함께 만드는 action.
- C에 필요한 빠진 구조: \(n_{b0}\)를 선택하는 생성/초기조건 법칙. coupling
  함수만으로는 conservation integration constant를 없애지 못한다.
- D는 conserved-current 요구와 논리적으로 양립하지 않는다.

- [x] 구조적으로 다른 후보가 5개다.
- [x] 후보마다 새 공리는 1개 이하로 분리했다.
- [x] 각 후보에 연속 dof/모델 선택 수와 target-aware 여부가 있다.
- [x] 각 후보에 조건부 cross-output과 죽이는 반증 시험이 있다.
- [x] 유망하지 않은 경로와 정확한 실패 이유도 보존했다.
