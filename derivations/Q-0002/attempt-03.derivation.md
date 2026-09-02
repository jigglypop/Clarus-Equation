---
question: Q-0002
attempt: 3
pivot_step: alt_derivation
claim: "시계 2–플럭스 교환 결합 모형 C=H_1+H_2+H_S+λ(T_2⊗σ_-^Q+h.c.), H_S=E_*n_Q+gσ_x^R (E_*=1,g=2), Z_N 군평균 Π, 정수 스펙트럼 (N,λ)∈{(8,3),(6,4),(24,5)} 에서: K6(i) 관측자 2 의 정규화 연산자 O^{(2)}_1=V_2^†V_2 는 k_1 잔류류별 블록대각이고 각 블록은 시계-2 τ-단면 벡터의 Gram 행렬이며, 그 스펙트럼은 일반 보조정리로 {0,1,2,1±c,1±c'}⊂ (c=√((1+sin2φ)/2), c'=√((1−sin2φ)/2), tan2φ=2λ/N) 안에 있고 세 경우에는 정확히 (8,3),(24,5): {0,1,2}, (6,4): {1−c,1,1+c} 이다. K6(ii) 정규화 장부 E^{(2)}=ω(O^{(2)}_{H_S})/ω(O^{(2)}_1) 는 λ≠0 에서 상태의 이차형식이 아니고(polarization 위반, 세 경우 최대 0.10, 0.31, 0.023), λ=0 에서는 정확히 이차형식이며, 따라서 E^{(1)}−E^{(2)}=ω(X) 를 만족하는 연산자 X 는 (Dirac 여부와 무관하게) 존재하지 않는다 — 부모 (B) 문구 그대로 기각. K8(a) V_i 가 유니터리인 경우(관측자 1 전부, λ=0 의 관측자 2) O_{P_b}O_{H_S}O_{P_b}=O_{P_bH_SP_b}. K8(b) 관측자 2, λ≠0: O^{(2)}_{P_b}O^{(2)}_{H_S}O^{(2)}_{P_b}−O^{(2)}_{P_bH_SP_b}=V_2^†P_b(Q_2H_SQ_2−H_S)P_bV_2, Q_2:=V_2V_2^†=NΠ 의 시계-2 τ 부분행렬원소 =Σ_n e^{2πinH_1/N}⊗⟨τ|e^{2πinH_rest/N}|τ⟩_2 ≠1 이며, 괄호는 R:=1−Q_2 의 함수 RH_SR−RH_S−H_SR 로만 쓰이고 R=0 이면 소멸한다; 관계적 레지스터 장부와 운동학적 장부의 차이는 세 경우 모두 >SEP (0.69, 1.11, 0.26). 단 Q_2 는 사영이 아니다(고윳값 2 또는 1+c 존재): '비선택 성분'은 ker Q_2 (=(8,3),(24,5) 형에만 존재) 이고 (6,4) 형에서는 저가중(1−c)·과가중(1+c) 방향만 있다. K9 부정 보조정리: 적색편이형 결합 H_2(1+λn_Q) 에 [H_S,n_Q]=0 이면 H_S 는 Dirac, gcd(1+λ,N)=1 이면 V_2 유니터리·Δ=0 이라 관측자 의존이 생기지 않는다."
assumptions:
  - "시계 C_i=C^N, H_i=diag(0..N−1), |τ⟩=N^{−1/2}Σ_k e^{−ikτ}|k⟩, 격자 τ_n=2πn/N 에서만 진술 (K6(i) 스펙트럼 τ-불변성은 격자 위상 e^{i(N−1)τ_n}=e^{−iτ_n} 에 의존)"
  - "T_2|k⟩=|k+1 mod N⟩ (순환 올림), σ_-^Q=|0⟩⟨1|_Q; S=R⊗Q, 인덱스 s=2r+q; E_*=1 (섹터 블록이 [[j,λ],[λ,j]] 로 퇴화하는 데 필요), g=2"
  - "Π=(1/N)Σ_n e^{2πinC/N}; 사영이 되려면 spec C⊂Z, 즉 ρ:=√((N/2)²+λ²) 에 대해 N/2±ρ∈Z. 검사한 세 점 (8,3)ρ=5, (6,4)ρ=5, (24,5)ρ=13"
  - "물리 기저는 H_rest 고유벡터 |μ⟩ 마다 |k_1=(−μ) mod N⟩⊗|μ⟩ 로 명시 구성; 이 Π 가 군평균과 일치함을 검사 (*_Pi_equals_group_average)"
  - "V_i(τ)=√N⟨τ|_i|_{H_phys}, O^{(i)}_A=V_i^†AV_i, ω_Ψ(O)=⟨Ψ|O|Ψ⟩; E^{(2)} 는 V_2Ψ≠0 인 상태에서만 정의 (ker V_2(τ) 는 (8,3),(24,5) 에서 비자명)"
  - "레지스터 사영 P_b=|b⟩⟨b|_R⊗1_Q 운동학적 고정 (attempt-02 와 동일)"
  - "K6(ii)·K8(b) 비자명성은 세 매개변수 점에서의 수치 증인(씨앗 20260902, 20 쌍/20 상태)이며 일반 (N,λ) 정리가 아님"
symbols:
  N: positive integer
  lam: real
  phi: real
  R: real
  h: real
  n: integer
  x: real
verify:
  - type: identity
    lhs: "(N/2+sqrt((N/2)**2+lam**2))*(N/2-sqrt((N/2)**2+lam**2))"
    rhs: "-lam**2"
  - type: identity
    lhs: "((sin(phi)+cos(phi))/sqrt(2))**2"
    rhs: "(1+sin(2*phi))/2"
  - type: identity
    lhs: "((cos(phi)-sin(phi))/sqrt(2))**2"
    rhs: "(1-sin(2*phi))/2"
  - type: identity
    lhs: "(1+sin(2*phi))/2+(1-sin(2*phi))/2"
    rhs: "1"
  - type: identity
    lhs: "(1+lam/sqrt((N/2)**2+lam**2))/2"
    rhs: "(1+2*lam/sqrt(N**2+4*lam**2))/2"
  - type: identity
    lhs: "(1-R)*h*(1-R)-h"
    rhs: "R*h*R-R*h-h*R"
  - type: identity
    lhs: "cos((N-1)*2*pi*n/N)-cos(2*pi*n/N)"
    rhs: "0"
  - type: identity
    lhs: "sin((N-1)*2*pi*n/N)+sin(2*pi*n/N)"
    rhs: "0"
  - type: identity
    lhs: "(1-(1+x))**2-x**2"
    rhs: "0"
  - type: numeric
    expr: "(8/2)**2+3**2-5**2"
    tol: 1e-12
  - type: numeric
    expr: "(6/2)**2+4**2-5**2"
    tol: 1e-12
  - type: numeric
    expr: "(24/2)**2+5**2-13**2"
    tol: 1e-12
  - type: numeric
    expr: "2*(1/3)/(1-(1/3)**2)-2*3/8"
    tol: 1e-12
  - type: numeric
    expr: "sqrt((1+3/5)/2)-2/sqrt(5)"
    tol: 1e-12
  - type: numeric
    expr: "sqrt((1+4/5)/2)-3/sqrt(10)"
    tol: 1e-12
---

# Q-0002 attempt-03 — 후보 K6 + K8 (교환 결합 모형에서 관측자 2 의 정규화 장부와 사영 준동형 결함)

기계 검사: 프론트매터 verify 블록(스칼라 항등식 9개·수치 6개, sympy 없음 → symbolic skipped)과
`verify/Q-0002/attempt-03/check_k6k8.py`(numpy, 씨앗 20260902, TOL=1e-10, SEP=1e-3, TOL_SPEC=1e-8 선선언, 102개 검사, 102 통과).
행렬 수준 주장의 실질 증거는 후자다. attempt-02(E-20260902-002)의 비결합 모형에서는 V_2 가 유니터리라 관측자 2 의
정규화가 자명했다(ω(O_1)=1). 이번 alt_derivation 은 시계 2 와 플럭스를 교환 결합시켜 V_2 를 비등거리로 만들고,
그때 정규화 장부와 사영 장부에 무엇이 생기는지를 닫힌 식으로 쓴다.

## (S0) 기호와 섹터 분해

$$ \mathcal C_i=\mathbb C^N,\ H_i|k\rangle=k|k\rangle,\ |\tau\rangle=N^{-1/2}\sum_ke^{-ik\tau}|k\rangle,\ \tau_n=2\pi n/N,\qquad T_2|k\rangle=|k+1\bmod N\rangle,\ \sigma_-^Q=|0\rangle\langle1|_Q $$  (S0.1) 시계·순환 올림·플럭스 내림 정의
$$ H_S=E_*n_Q+g\,\sigma_x^R,\qquad H_{\rm rest}:=H_2+H_S+\lambda(T_2\otimes\sigma_-^Q+T_2^\dagger\otimes\sigma_+^Q)\ \text{on }\mathcal C_2\otimes\mathcal S,\qquad C=H_1\otimes1+1\otimes H_{\rm rest} $$  (S0.2) 제약; [H_1,H_rest]=0
$$ [n_Q,\,T_2\otimes\sigma_-^Q]=-T_2\otimes\sigma_-^Q\ \Rightarrow\ [H_S,C]=\lambda E_*(T_2^\dagger\otimes\sigma_+^Q-T_2\otimes\sigma_-^Q),\quad \|[H_S,C]\|=\lambda $$  (S0.3) H_S 는 Dirac 이 아님 (check *_norm_[HS,C]_equals_lam: 3, 4, 5)
$$ e^{2\pi i(H_2+n_Q)/N}\ \text{commutes with }H_{\rm rest}:\quad \mathcal C_2\otimes\mathcal Q=\bigoplus_{j\in\mathbb Z_N}\mathcal V_j,\ \mathcal V_j=\mathrm{span}\{|j,q{=}0\rangle,\ |j-1\bmod N,\ q{=}1\rangle\} $$  (S0.4) T_2⊗σ_- 는 (k_2,q)=(j−1,1)↦(j,0): k_2+q mod N 보존
$$ (H_2+n_Q+\lambda(T_2\sigma_-+{\rm h.c.}))\big|_{\mathcal V_j}=\begin{pmatrix}j&\lambda\\\lambda&j\end{pmatrix}\ (j\ge1),\qquad \begin{pmatrix}0&\lambda\\\lambda&N\end{pmatrix}\ (j=0) $$  (S0.5) E_*=1 이라 j≥1 블록 대각이 퇴화; j=0 은 순환 경계에서 k_2=N−1 의 값 N−1+1=N 이 남음
$$ \mathrm{spec}\,H_{\rm rest}=\{j\pm\lambda\pm g\}_{j=1}^{N-1}\cup\{\tfrac N2\pm\rho\pm g\},\qquad \rho:=\sqrt{(N/2)^2+\lambda^2},\qquad (\tfrac N2+\rho)(\tfrac N2-\rho)=-\lambda^2 $$  (S0.6) 2×2 고윳값과 σ_x^R 의 ±g (verify 1)
$$ \mathrm{spec}\,C\subset\mathbb Z\ \Leftrightarrow\ \tfrac N2\pm\rho\in\mathbb Z:\quad (8,3)\,\rho=5,\ (6,4)\,\rho=5,\ (24,5)\,\rho=13;\qquad (8,2)\,\rho=\sqrt{20}\notin\mathbb Z $$  (S0.7) 정수 스펙트럼 조건 (verify 10–12; check *_integer_spectrum ≤6e-14, NEG_N8_lam2_spectrum_not_integer 0.47)
$$ \Pi:=\tfrac1N\sum_{n=0}^{N-1}e^{2\pi inC/N},\qquad \Pi^2=\Pi\ \Leftrightarrow\ \mathrm{spec}\,C\subset\mathbb Z $$  (S0.8) 기하급수 (check NEG_N8_lam2_group_average_not_projector: ‖Π²−Π‖=0.71)
$$ u_+:=(\sin\varphi,\cos\varphi)^T,\ u_-:=(\cos\varphi,-\sin\varphi)^T\ \text{in }(|q{=}0\rangle,|q{=}1\rangle),\qquad \tan\varphi=\frac{\rho-N/2}{\lambda}\ \Leftrightarrow\ \tan2\varphi=\frac{2\lambda}N $$  (S0.9) j=0 블록의 고유벡터 (N/2±ρ 순); 배각 공식 (verify 13: (8,3) 에서 tanφ=1/3)
$$ |\pm_q\rangle:=(|0\rangle\pm|1\rangle)/\sqrt2:\quad \langle+_q|u_+\rangle=\langle-_q|u_-\rangle=\frac{\sin\varphi+\cos\varphi}{\sqrt2}=:c,\quad \langle+_q|u_-\rangle=-\langle-_q|u_+\rangle=\frac{\cos\varphi-\sin\varphi}{\sqrt2}=:c' $$  (S0.10) c²=(1+sin2φ)/2, c'²=(1−sin2φ)/2, c²+c'²=1, sin2φ=λ/ρ (verify 2–5, 14–15)

## (S1) 물리 공간과 두 환원 사상

$$ H_{\rm rest}|\mu\rangle=\mu|\mu\rangle\ \Rightarrow\ \mathcal H_{\rm phys}=\mathrm{ran}\,\Pi=\mathrm{span}\{|k_1(\mu)\rangle\otimes|\mu\rangle\},\ k_1(\mu):=(-\mu)\bmod N,\qquad \dim\mathcal H_{\rm phys}=4N $$  (S1.1) (S0.2)(S0.8): C 고윳값 k_1+μ≡0 (mod N) 이 각 μ 에 k_1 하나 (check *_phys_dim_equals_4N: 32, 24, 96; *_Pi_equals_group_average ≤1.2e-13)
$$ V_1(\tau)\,|k_1\rangle\otimes|\mu\rangle=e^{ik_1\tau}|\mu\rangle\ \Rightarrow\ V_1^\dagger V_1=1_{\mathcal H_{\rm phys}},\ V_1V_1^\dagger=1_{\mathcal C_2\otimes\mathcal S} $$  (S1.2) {|μ⟩} 가 C_2⊗S 의 정규직교기저이므로 유니터리 (check *_V1_unitary ≤3.6e-15)
$$ V_2(\tau)\,|k_1\rangle\otimes|\mu\rangle=|k_1\rangle\otimes w_\mu(\tau),\qquad w_\mu(\tau):=\sqrt N\,\langle\tau|_2|\mu\rangle\in\mathcal S $$  (S1.3) 정의; |μ⟩ 가 시계 2 와 S 에 얽혀 있어 w_μ 는 일반적으로 서로 직교하지 않음
$$ \langle k_1\mu|V_2^\dagger V_2|k_1'\mu'\rangle=\delta_{k_1k_1'}\,\langle w_\mu|w_{\mu'}\rangle $$  (S1.4) (S1.3): O^{(2)}_1=V_2^†V_2 는 k_1 잔류류별 블록대각, 블록 = 그 류의 τ-단면 Gram 행렬 (check K6i_*_V2dV2_block_diagonal_in_k1: 0; *_block_equals_Gram_of_slices ≤7e-16)

## (S2) K6(i): Gram 블록의 스펙트럼

$$ j\ge1:\ |\mu\rangle=\tfrac1{\sqrt2}(|j,0\rangle\pm|j-1,1\rangle)\otimes|r_\pm\rangle\ \Rightarrow\ w_\mu(\tau_n)=e^{ij\tau_n}\,\mathrm{diag}(1,e^{-i\tau_n})|\pm_q\rangle\otimes|r_\pm\rangle $$  (S2.1) (S0.5)(S1.3); |r_±⟩ 는 σ_x^R 고유벡터
$$ j=0:\ |\mu\rangle=(\cos\theta|0,0\rangle+\sin\theta|N-1,1\rangle)\otimes|r_\pm\rangle\ \Rightarrow\ w_\mu(\tau_n)=\mathrm{diag}(1,e^{-i\tau_n})\,u_\pm\otimes|r_\pm\rangle $$  (S2.2) e^{i(N−1)τ_n}=e^{−iτ_n} 격자 위상 (verify 7–8); 격자 밖 τ 에서는 성립하지 않음
$$ \text{Gram}\{w_\mu\}_{\rm class}\ \simeq\ \text{Gram}\{v_\mu\otimes|r_{t(\mu)}\rangle\},\quad v_\mu\in\{|+_q\rangle,|-_q\rangle,u_+,u_-\} $$  (S2.3) (S2.1)(S2.2): 공통 유니터리 diag(1,e^{−iτ_n}) 와 벡터별 위상은 Gram 을 대각 유니터리로 켤레할 뿐 → 스펙트럼은 격자 τ 에 무관 (check tau0/tau1 동일)
$$ \text{보조정리: } \mathbb C^2\text{ 의 벡터 집합 }W\subset\{|+\rangle,|-\rangle,u_+,u_-\}\text{ 의 Gram 스펙트럼}\subset\{0,1,2,1\pm c,1\pm c'\} $$  (S2.4) |W|=1: {1}; |W|=2 직교쌍({|±⟩} 또는 {u_±}): {1,1}; |W|=2 비직교쌍: 1±|⟨·|·⟩|∈{1±c,1±c'} (verify 9); |W|=3: 항상 정규직교쌍 포함 → [[1,0,a],[0,1,b],[a,b,1]], |a|²+|b|²=1 → {0,1,2}; |W|=4: {0,0,2,2}
$$ \text{류 }\nu:=\mu\bmod N\text{ 에 속하는 } j\ge1\text{ 패턴}:\ (+_q,r_+)\ j\equiv\nu-\lambda-g,\ (+_q,r_-)\ j\equiv\nu-\lambda+g,\ (-_q,r_+)\ j\equiv\nu+\lambda-g,\ (-_q,r_-)\ j\equiv\nu+\lambda+g;\quad j=0\ \text{패턴}:\ (u_\pm,r_\pm)\ \nu\equiv\tfrac N2\pm\rho\pm g $$  (S2.5) (S0.6): j≡0 인 패턴은 그 류에서 빠지고, j=0 섹터 벡터는 자신의 ν 에 들어감 (류 크기는 4 가 아닐 수 있음)
$$ (8,3):\ j\equiv\nu+3,\nu-1,\nu+1,\nu-3;\ u\text{-패턴 }\nu\in\{3,7,1,5\};\quad \nu\text{ 홀수: 한 }r\text{-블록에 }\{|+\rangle,|-\rangle,u_\pm\}\to\{0,1,2\},\ \text{다른 블록 }\{1\};\ \nu\text{ 짝수: }\{1,1,1,1\} $$  (S2.6) 스펙트럼 집합 {0,1,2} (check K6i_N8_lam3_*: max_dist 0, 세 값 모두 도달); 유도 다중도 0×4, 2×4 (수치 미검사)
$$ (6,4):\ j\equiv\nu,\nu-2,\nu+2,\nu;\ u\text{-패턴 }\nu\in\{4,0,0,2\};\quad \nu=0:\ r_+\{|-\rangle,u_-\},\ r_-\{|+\rangle,u_+\}\to\{1\pm c\}^2;\ \nu=4:\ r_+\{|+\rangle,u_+\};\ \nu=2:\ r_-\{|-\rangle,u_-\};\ \nu\in\{1,3,5\}:\ \{1\}^4 $$  (S2.7) 모든 비직교쌍이 (S0.10) 의 c 쌍이라 스펙트럼 집합 {1−c,1,1+c}, c=√0.9=0.9487 (check K6i_N6_lam4_*: 0.051317, 1, 1.948683, max_dist 0); c' 쌍은 나타나지 않음
$$ (24,5):\ j\equiv\nu-7,\nu-3,\nu+3,\nu+7;\ u\text{-패턴 }\nu\in\{3,23,1,21\};\quad \nu\in\{3,21\}:\ \{|+\rangle,|-\rangle,u\}\ \text{블록}\to\{0,1,2\};\ \nu\in\{1,23\}:\ \text{류 크기 5, 같은 구조};\ \nu\in\{7,17\}:\ \text{류 크기 3},\ \{1\}^3 $$  (S2.8) 스펙트럼 집합 {0,1,2} (check K6i_N24_lam5_*); 4·18+4+4+3+3+5+5=96=4N
$$ \mathrm{spec}\,O^{(2)}_1(\tau_n)=\{0,1,2\}\ ((8,3),(24,5)),\quad \{1-c,1,1+c\}\ ((6,4)),\qquad \ker V_2(\tau_n)\neq0\ \text{iff 고윳값 }0 $$  (S2.9) K6(i) 증명 끝. ker V_2: 관측자 2 가 τ_n 에서 전혀 보지 못하는 물리 상태 (Σ_nV_2(τ_n)^†V_2(τ_n)=N·1 이므로 다른 τ 에서는 보임)

## (S3) K6(ii): 정규화 장부는 이차형식이 아니다 → 부모 (B) 기각

$$ E^{(2)}(\Psi):=\frac{\omega_\Psi(O^{(2)}_{H_S})}{\omega_\Psi(O^{(2)}_1)}=\frac{\langle\Psi|A|\Psi\rangle}{\langle\Psi|B|\Psi\rangle},\quad A:=V_2^\dagger(1\otimes H_S)V_2,\ B:=V_2^\dagger V_2,\qquad g_2(c):=\|c\|^2\,E^{(2)}(c) $$  (S3.1) 정의 (V_2c≠0); g_2 는 2차 동차
$$ \exists Y:\ E^{(2)}(\Psi)=\langle\Psi|Y|\Psi\rangle\ \forall\|\Psi\|=1\ \Rightarrow\ g_2(c)=\langle c|Y|c\rangle\ \Rightarrow\ g_2(a+b)+g_2(a-b)=2g_2(a)+2g_2(b)\ \forall a,b $$  (S3.2) 이차형식의 polarization 항등식
$$ \max_{20\text{ 쌍}}|g_2(a+b)+g_2(a-b)-2g_2(a)-2g_2(b)|=0.103\ (8,3),\ 0.310\ (6,4),\ 0.0229\ (24,5)\ >\text{SEP} $$  (S3.3) 수치 증인: (S3.2) 의 Y 는 존재하지 않음 (check K6ii_*_E2_normalized_polarization_violation_max_gt_SEP; SEP 초과 쌍 수 20/20, 19/20, 15/20 — 판정 기준은 최대값, 존재 논증에는 한 쌍이면 충분)
$$ \lambda=0:\ B=1\ \Rightarrow\ g_2(c)=\langle c|A|c\rangle\ \text{정확히 이차형식} $$  (S3.4) (S1.3) 에서 w_μ 정규직교 → V_2 유니터리 (check K6ii_N*_lam0_E2_normalized_exactly_quadratic ≤1.2e-15; *_lam0_V2_unitary 2.4e-16)
$$ E^{(1)}(\Psi)=\langle\Psi|V_1^\dagger(1\otimes H_S)V_1|\Psi\rangle\ \text{이차형식 (분모 }\|V_1\Psi\|^2=1) $$  (S3.5) (S1.2) (check K6ii_*_E1_exactly_quadratic 8.9e-16)
$$ \exists X:\ E^{(1)}-E^{(2)}=\omega(X)\ \Rightarrow\ E^{(2)}=\omega(V_1^\dagger H_SV_1-X)\ \text{이차형식}\ \Rightarrow\ \text{(S3.3) 모순} $$  (S3.6) 어떤 연산자 X 도 없음; 특히 Dirac X 없음 (check K6ii_*_E1_minus_E2_polarization_violation_gt_SEP: 0.21, 0.29, 0.059). 부모 (B) "관측자 간 에너지 차이는 Dirac 관측량의 기댓값" 은 이 모형에서 문구 그대로 거짓

## (S4) K8(a): 환원이 유니터리면 사영 장부는 준동형

$$ V^\dagger V=1,\ VV^\dagger=1\ \Rightarrow\ O_{P_b}O_{H_S}O_{P_b}=V^\dagger P_b(VV^\dagger)H_S(VV^\dagger)P_bV=V^\dagger P_bH_SP_bV=O_{P_bH_SP_b} $$  (S4.1) attempt-02 (S2.6) 과 같은 논증; 관측자 1 (S1.2), λ=0 의 관측자 2 (S3.4) (check K8a_*_observer1 ≤7.2e-15; K8a_N*_lam0_observer2 4.4e-16)

## (S5) K8(b): 관측자 2, λ≠0 — 결함은 R=1−Q_2 로만 쓰인다

$$ Q_2:=V_2V_2^\dagger\ \text{on }\mathcal C_1\otimes\mathcal S,\qquad O^{(2)}_{P_b}O^{(2)}_{H_S}O^{(2)}_{P_b}-O^{(2)}_{P_bH_SP_b}=V_2^\dagger P_b\,(Q_2H_SQ_2-H_S)\,P_bV_2 $$  (S5.1) V_2V_2^† 를 삽입 (check K8b_*_defect_identity_residual ≤1.9e-15)
$$ R:=1-Q_2\ \Rightarrow\ Q_2H_SQ_2-H_S=RH_SR-RH_S-H_SR,\qquad R=0\ \Rightarrow\ 0 $$  (S5.2) (1−R)H_S(1−R)−H_S 전개 (verify 6; check K8b_*_defect_depends_only_on_1_minus_Q2 ≤1.4e-15, *_defect_zero_when_Q2_replaced_by_1 = 0)
$$ V_2(\tau)=\sqrt N(\langle\tau|_2\otimes1)\,\mathcal B,\ \mathcal B\mathcal B^\dagger=\Pi\ \Rightarrow\ Q_2=N\,\langle\tau|_2\,\Pi\,|\tau\rangle_2 $$  (S5.3) 시계 2 에 대한 부분 행렬원소
$$ [H_1,H_{\rm rest}]=0\ \Rightarrow\ e^{2\pi inC/N}=e^{2\pi inH_1/N}\otimes e^{2\pi inH_{\rm rest}/N}\ \Rightarrow\ Q_2=\sum_{n=0}^{N-1}e^{2\pi inH_1/N}\otimes\langle\tau|e^{2\pi inH_{\rm rest}/N}|\tau\rangle_2 $$  (S5.4) (S0.8) 대입: Q_2 폐형 (check K8b_*_Q2_closed_form ≤9.7e-14, τ_1 에서도 ≤9.7e-14)
$$ \lambda=0:\ \langle\tau|e^{2\pi in(H_2+H_S)/N}|\tau\rangle_2=\langle\tau|\tau-\tau_n\rangle\,e^{2\pi inH_S/N}=\delta_{n0}\ \Rightarrow\ Q_2=1 $$  (S5.5) e^{iθH_2}|τ⟩=|τ−θ⟩ 와 격자 직교성 (check K8b_N*_lam0_Q2_equals_1 2.4e-16)
$$ \lambda\neq0:\ \mathrm{spec}\,Q_2=\mathrm{spec}\,V_2^\dagger V_2=\{0,1,2\}\ \text{또는}\ \{1\pm c,1\}\ \Rightarrow\ Q_2\neq1,\ Q_2^2\neq Q_2 $$  (S5.6) V_2V_2^† 와 V_2^†V_2 는 같은 차원 4N 에서 같은 스펙트럼 (check K8b_*_Q2_ne_1: 1, 0.95, 1; *_Q2_not_a_projector: 2, 1.85, 2)
$$ \text{해석 교정: }R=1-Q_2\text{ 는 사영이 아니다. }\ker Q_2\ (\text{고윳값 }0)=\text{어떤 }V_2\Psi\text{ 도 닿지 않는 방향}=\text{'비선택 성분'};\ \text{고윳값 }2\ (1+c)=\text{두 물리 방향이 한 읽기 방향에 겹치는 과가중};\ 1-c=\text{저가중} $$  (S5.7) 부모 해석 "비선택 성분이 지배" 는 (8,3),(24,5) 형(ker Q_2≠0)에서만 문자 그대로이고 (6,4) 형에는 정확한 비선택 방향이 없다 — 옳은 일반 진술은 "읽기 가중 연산자 Q_2≠1 의 함수"
$$ \max_{b,\ 20\text{ 상태}}\frac{|\omega(O^{(2)}_{P_b}O^{(2)}_{H_S}O^{(2)}_{P_b})-\omega(O^{(2)}_{P_bH_SP_b})|}{\omega(O^{(2)}_1)}=0.686\ (8,3),\ 1.112\ (6,4),\ 0.261\ (24,5)\ >\text{SEP} $$  (S5.8) 관계적 레지스터 장부 E^rel_b 와 운동학적 장부 E^kin_b 의 차이가 비자명 (check K8b_*_defect_nontrivial); 영감 ② 의 유한 정식화 후보: 두 장부의 차 = V_2^†P_b F(R) P_bV_2, F(R)=RH_SR−RH_S−H_SR

## (S6) K9 부정 보조정리: 적색편이형 결합은 관측자 의존을 만들지 못한다

$$ C'=H_1+H_2(1+\lambda n_Q)+H_S,\ H_S=n_Q+g\sigma_x^R\ \Rightarrow\ [H_S,C']=0 $$  (S6.1) [n_Q,H_2n_Q]=0, σ_x^R 은 R 이외에 작용하는 항이 없음: H_S Dirac (check K9_rate_N8_lam2_gcd1_HS_Dirac: 0)
$$ C'|k_1k_2r_\pm q\rangle=(k_1+k_2(1+\lambda q)+q\pm g)|\cdot\rangle\in\mathbb Z\ \Rightarrow\ \text{정수 조건 불필요};\quad k_1\equiv-k_2(1+\lambda q)-q\mp g $$  (S6.2) 곱기저에서 대각; 각 (k_2,r,q) 에 k_1 하나 → dim 4N, V_1 유니터리
$$ V_2|k_1k_2e\rangle=e^{ik_2\tau}|k_1(k_2,e),e\rangle,\quad k_2\mapsto k_1\ \text{전단사}\ \Leftrightarrow\ \gcd(1+\lambda,N)=1\ \Rightarrow\ V_2\ \text{유니터리} $$  (S6.3) q=1 에서 곱셈 1+λ 가 Z_N 의 순열이어야 함 (check K9_rate_N8_lam2_gcd1_V2_unitary 2.4e-16; gcd(3,8)=1)
$$ V_i^\dagger(1\otimes H_S)V_i=\sum_{k_2,e}|c_{k_2e}|^2\epsilon_e\ (i=1,2)\ \Rightarrow\ \Delta:=O^{(1)}_{H_S}-O^{(2)}_{H_S}=0 $$  (S6.4) attempt-02 (S4.3) 논증 그대로: H_S 가 e-대각이고 상은 정규직교 (check K9_rate_N8_lam2_gcd1_Delta_zero 8.9e-17)
$$ \gcd(1+\lambda,N)\neq1\ ((8,1):\ \gcd(2,8)=2)\ \Rightarrow\ k_2\mapsto k_1\ \text{2-대-1}\ \Rightarrow\ V_2^\dagger V_2\neq1 $$  (S6.5) 순환 겹침으로 인한 비유니터리 (check K9_rate_N8_lam1_gcd2_V2_not_unitary: ‖V_2^†V_2−1‖=1, Δ=3.0 — 이 Δ 는 라벨 충돌 인공물이며 본 attempt 는 해석하지 않음)
$$ \text{결론: } [H_S,n_Q]=0\text{ 인 적색편이형 결합에서 관측자 의존 }\Delta\neq0\text{ 은 gcd 조건 아래 불가능; 교환 결합 (S0.2) 처럼 }[H_S,C]\neq0\text{ 이 필요} $$  (S6.6) K9 경로 사망 (부정 보조정리)

## (S7) 증명한 것과 증명하지 않은 것

$$ \text{증명(해석+수치): (S1.1)–(S1.4), (S2.3)–(S2.9), (S3.4)–(S3.6) 의 함의, (S4.1), (S5.1)–(S5.6), (S6.1)–(S6.5)} $$  (S7.1) 범위: 세 정수 스펙트럼 점, 격자 τ, Z_N 군평균
$$ \text{수치 증인만: (S3.3) 비이차성, (S5.8) 비자명성 — 세 매개변수 점에서의 존재 진술이지 일반 }(N,\lambda)\text{ 정리가 아님} $$  (S7.2) polarization 결함의 해석적 하한 없음
$$ \text{증명하지 않은 것 1: 일반 }(N,\lambda,g)\text{ 에서 }\{0,1,2\}\text{ 형과 }\{1\pm c\}\text{ 형의 이분법이 완전한지 — (S2.4) 는 }\{1\pm c'\}\text{ 도 허용하며 배제하지 않았다} $$  (S7.3) 잔류류 조합론이 매개변수마다 달라짐
$$ \text{증명하지 않은 것 2: (S2.6)–(S2.8) 의 다중도(0×4, 2×4 등)는 유도값이며 검사는 스펙트럼 집합만 확인했다} $$  (S7.4) 검사 범위 명시
$$ \text{증명하지 않은 것 3: 결함이 모든 }b\text{·모든 상태에서 }\neq0\text{ (최대값만 검사), 격자 밖 }\tau\text{, 연속 극한, }E_*\neq1 $$  (S7.5) 비범위
$$ \text{증명하지 않은 것 4: (S5.7) 과가중 방향(고윳값 2, 1+c) 의 물리적 의미 — 해석이며 정리가 아님} $$  (S7.6) 비범위
