---
question: Q-0002
attempt: 2
claim: "Z_N 군평균 사영 Π=(1/N)Σ_n e^{2πinH/N} 로 정의한 유한 이상 시계 둘+S=R⊗Q 제약 모형에서 (1) E_total:=ω(O^{(i)}_{H_S})/ω(O^{(i)}_1)=⟨Ψ|H_S|Ψ⟩ 는 관측자 i·τ 무관, (2) E^{(i)}_unseen(τ)=E_total−Σ_b ω(O^{(i)}_{P_bH_SP_b}) = Σ_{b≠b'} ω(O^{(i)}_{P_bH_SP_b'}) (정확식), (3) [P_b,H_S]=0 이면 E_unseen≡0 이고 Σ_b p_bE_b=E_total 이 관측자·τ 무관하게 성립(E39-C 한 번 영수증 재현), (4a) E^{(i)}_unseen 이 프레임 의존인 정확한 조건은 [P_b,H_S]≠0 이 아니라 [D(H_S),H_S]≠0 (D(H_S):=Σ_bP_bH_SP_b) 이며 — 과제의 H_S=E_*n_Q+gσ_x⊗n_Q 는 [P_b,H_S]≠0 이지만 D(H_S)=E_*n_Q 가 Dirac 이라 프레임 무관(반례) — [D,H_S]≠0 이면 E^{(1)}_unseen≠E^{(2)}_unseen 인 물리 상태가 존재하고, (4b) 수송 사영 Φ_12P_bΦ_12^† 를 쓰면 관측자 2의 장부가 관측자 1과 정확히 일치, (5) 감김 회계: Π는 H_S↦H_S+NK (K 정수 스펙트럼, [K,H_S]=0) 아래 불변이고 ΔE_total=N⟨K⟩_Ψ — 균일 이동(K=m·1)에서만 정확히 N의 정수배 — 시계 대표원 이동은 E_total 을 바꾸지 않고 감김수 W=H/N 을 1 옮긴다. 부가: mod-N 에서 V_i 는 전체 C_j⊗S 로의 유니터리라 A↦O^{(i)}_A 는 *-준동형이고 Φ_12 는 전체 공간 유니터리(경계 인공물 없음)."
assumptions:
  - "시계 C_i=C^N, H_i=diag(0..N−1) (라벨 mod N; 대표원 −M..M 과 동치), 제약 군 Z_N={e^{2πinH/N}}, Π=(1/N)Σ_{n=0}^{N−1}e^{2πinH/N} (de la Hamette 등 2021 유한군 정칙표현 프레임)"
  - "S=R⊗Q, R 큐비트(맨 에너지 0), Q 큐비트 n_Q=diag(0,1); H_S 는 정수 스펙트럼이며 고윳값 대표원 |ε_e|<N/2 (이 조건이 ε_e≢ε_{e'} (mod N) ⇔ ε_e≠ε_{e'} 를 보장)"
  - "구성 A0: N=7, H_S=E_*n_Q (g=0); A2: N=7, H_S=E_*n_Q+2σ_x^R⊗n_Q (과제 문구, 고윳값 0,0,3,−1); B: N=13, H_S=(E_*+3σ_x^R+4σ_z^R)⊗n_Q (고윳값 0,0,6,−4; 정수 스펙트럼을 위해 (g,h)=(3,4) 피타고라스 선택, |6|<6.5 를 위해 N=13); E39: N=7, H_S'=E_*(n_Q+1−n_R); E_*=1"
  - "환원은 V_i(τ):=√N⟨τ|_i|_{H_phys}, τ∈2πZ_N/N 격자 (격자 밖 τ 에서는 (S3.2) 위상 항등식이 깨진다)"
  - "관계적 관측량 O^{(i)}_A(τ):=NΠ(|τ⟩⟨τ|_i⊗A)Π, A∈End(C_j⊗S); 비율 ω(O_A)/ω(O_1) 로 정규화 (mod-N 에서는 ω(O_1)=1)"
  - "레지스터 사영 P_b=|b⟩⟨b|_R⊗1_Q 는 σ_z^R 고유기저 고정 (프레임 무관한 기준 분할)"
  - "(4a) 존재 논증은 D(H_S) 의 H_S 고유기저 비대각 성분 D_{ee'}≠0, ε_e≠ε_{e'} 인 쌍 하나를 쓴다 ([D,H_S]≠0 과 동치)"
symbols:
  Es: real
  g: real
  h: real
  k1: integer
  k2: integer
  e: integer
  m: integer
  n: integer
  N: positive integer
  ar: real
  ai: real
  br: real
  bi: real
  d: real
  x: real
  z: real
verify:
  - type: identity
    lhs: "(Es+sqrt(g**2+h**2))*(Es-sqrt(g**2+h**2))"
    rhs: "Es**2-g**2-h**2"
  - type: identity
    lhs: "cos((k1+N)*2*pi*n/N)-cos(k1*2*pi*n/N)"
    rhs: "0"
  - type: identity
    lhs: "sin((-k2-e+N*m)*2*pi*n/N)+sin((k2+e)*2*pi*n/N)"
    rhs: "0"
  - type: identity
    lhs: "cos((-k2-e+N*m)*2*pi*n/N)-cos((k2+e)*2*pi*n/N)"
    rhs: "0"
  - type: identity
    lhs: "(Es+g*x+h*z)-(Es+h*z)"
    rhs: "g*x"
  - type: identity
    lhs: "(ar*br+ai*bi)*d+(br*ar+bi*ai)*d"
    rhs: "2*(ar*br+ai*bi)*d"
  - type: identity
    lhs: "((Es+N*m)-Es)-N*m"
    rhs: "0"
  - type: numeric
    expr: "(1+sqrt(3**2+4**2))-6"
    tol: 1e-12
  - type: numeric
    expr: "(1-sqrt(3**2+4**2))+4"
    tol: 1e-12
---

# Q-0002 attempt-02 — 후보 K3: 제약 수준 에너지 장부 (Z_N 군평균 사영)

기계 검사: 프론트매터 verify 블록(스칼라 항등식 7개·수치 2개, sympy 없음 → symbolic skipped)과
`verify/Q-0002/attempt-02/check_k3.py`(행렬 수준, numpy, 씨앗 20260902, TOL=1e-10, SEP=1e-3, 62개 검사).
행렬 수준 주장 (S2)(S4)–(S9)의 실질 증거는 후자다. attempt-01(E-20260902-001)의 exact-kernel Π 를 버리고
Z_N 군평균 Π 로 바꾼 이유: 경계 k=±M 에서 (S5.7) 에너지 이동이 소멸하고 A↦O_A 가 곱에 닫히지 않던 두 결함이
mod-N 에서 동시에 사라진다((S2.6)(S9)).

## (S0) 기호와 설정

$$ \mathcal C_i=\mathbb C^N,\quad H_i|k\rangle=k|k\rangle,\ k\in\{0,\dots,N-1\},\qquad |\tau\rangle=N^{-1/2}\sum_k e^{-ik\tau}|k\rangle,\ \tau_n=2\pi n/N $$  (S0.1) 시계 정의, 이산 푸리에 직교기저
$$ \langle k|\tau\rangle=N^{-1/2}e^{-ik\tau},\qquad e^{i(k+N)\tau_n}=e^{ik\tau_n} $$  (S0.2) 격자 τ 에서 라벨 k 는 mod N 만 의미 (verify 2)
$$ \mathcal S=\mathcal R\otimes\mathcal Q=\mathbb C^2\otimes\mathbb C^2,\quad n_Q=1_R\otimes\mathrm{diag}(0,1),\quad P_b=|b\rangle\langle b|_R\otimes 1_Q,\ b\in\{0,1\} $$  (S0.3) 레지스터·플럭스·레지스터 사영
$$ H_S=W\,\mathrm{diag}(\epsilon_0,\dots,\epsilon_3)\,W^\dagger,\quad \epsilon_e\in\mathbb Z,\ |\epsilon_e|<N/2 $$  (S0.4) H_S 고유분해, 정수 스펙트럼 가정
$$ \text{B: }H_S=(E_*+g\sigma_x+h\sigma_z)\otimes n_Q\ \Rightarrow\ \epsilon\in\{0,0,E_*\pm\sqrt{g^2+h^2}\},\quad (g,h)=(3,4),\ E_*=1\ \Rightarrow\ \epsilon\in\{0,0,6,-4\} $$  (S0.5) n_Q=1 블록의 특성다항식 (verify 1, 8, 9); 정수 스펙트럼을 위한 피타고라스 선택
$$ H=H_1\otimes1\otimes1+1\otimes H_2\otimes1+1\otimes1\otimes H_S,\qquad H|k_1k_2e\rangle=(k_1+k_2+\epsilon_e)|k_1k_2e\rangle $$  (S0.6) 제약, 기저 |k_1k_2e⟩ (e: H_S 고유벡터) 에서 대각
$$ U:=e^{2\pi iH/N},\qquad U^N=e^{2\pi i(k_1+k_2+\epsilon_e)}=1 $$  (S0.7) 정수 스펙트럼 ⇒ U 는 Z_N 표현
$$ \Pi:=\frac1N\sum_{n=0}^{N-1}U^n $$  (S0.8) Z_N 군평균

## (S1) H_phys = ker(H mod N)

$$ \Pi|k_1k_2e\rangle=\Big(\frac1N\sum_{n=0}^{N-1}e^{2\pi in(k_1+k_2+\epsilon_e)/N}\Big)|k_1k_2e\rangle=\delta_{k_1+k_2+\epsilon_e\equiv0\ (N)}\,|k_1k_2e\rangle $$  (S1.1) 기하급수 합: 정수 m 에 대해 (1/N)Σ_n e^{2πinm/N}=δ_{m≡0 (N)}
$$ \Pi^2=\Pi=\Pi^\dagger,\qquad \mathcal H_{\rm phys}:=\mathrm{ran}\,\Pi=\mathrm{span}\{|k_1k_2e\rangle: k_1+k_2+\epsilon_e\equiv0\ (N)\} $$  (S1.2) (S1.1)은 직교사영 (check S1_*_Pi_group_average, Pi_idempotent)
$$ \forall(k_2,e)\ \exists!\,k_1=(-k_2-\epsilon_e)\bmod N\ \Rightarrow\ \dim\mathcal H_{\rm phys}=N\cdot4=\dim(\mathcal C_j\otimes\mathcal S) $$  (S1.3) 라벨 (k_2,e) 또는 (k_1,e) 로 전단사; attempt-01 의 Σ_e(N−|ε_e|) 와 달리 결손 없음 (check phys_dim_equals_4N: 28, 28, 52)
$$ H|_{\mathcal H_{\rm phys}}=N\,W,\qquad W|k_1k_2e\rangle=\frac{k_1+k_2+\epsilon_e}{N}|k_1k_2e\rangle\in\mathbb Z $$  (S1.4) 감김수 연산자 정의: H_phys 는 ker H 가 아니라 ker(H mod N) (check K3_5_*_clock_representative_shift: 스펙트럼 {0,1,2})

## (S2) V_i 는 전체 공간 유니터리, O^{(i)}_A=V_i^†AV_i 는 *-준동형

$$ V_i(\tau):=\sqrt N\,\langle\tau|_i\big|_{\mathcal H_{\rm phys}},\qquad V_1(\tau)|k_1k_2e\rangle=e^{ik_1\tau}|k_2e\rangle,\quad V_2(\tau)|k_1k_2e\rangle=e^{ik_2\tau}|k_1e\rangle $$  (S2.1) (S0.2) 대입: 정규직교기저 → 위상 붙은 정규직교기저
$$ V_i(\tau)^\dagger V_i(\tau)=1_{\mathcal H_{\rm phys}},\qquad V_i(\tau)V_i(\tau)^\dagger=1_{\mathcal C_j\otimes\mathcal S} $$  (S2.2) (S1.3) 전단사 + (S2.1) ⇒ 유니터리 (check S2_*_V_i_unitary_onto_full)
$$ O^{(i)}_A(\tau):=N\,\Pi\,(|\tau\rangle\langle\tau|_i\otimes A)\,\Pi\ \Rightarrow\ \langle k_1k_2e|O^{(1)}_A|k_1'k_2'e'\rangle=N\cdot N^{-1/2}e^{-ik_1\tau}\,N^{-1/2}e^{ik_1'\tau}\langle k_2e|A|k_2'e'\rangle $$  (S2.3) 정의와 (S0.2), 두 상태 모두 물리적
$$ O^{(i)}_A(\tau)\big|_{\mathcal H_{\rm phys}}=V_i(\tau)^\dagger A\,V_i(\tau) $$  (S2.4) (S2.3)을 (S2.1)로 다시 씀 (check S2_*_O_full_definition_equals_B_Vdag_A_V_Bdag)
$$ \omega_\Psi(O^{(i)}_A):=\langle\Psi|O^{(i)}_A|\Psi\rangle=\langle V_i\Psi|A|V_i\Psi\rangle,\qquad \omega_\Psi(O^{(i)}_1)=\|V_i\Psi\|^2=\|\Psi\|^2=1 $$  (S2.5) (S2.2)(S2.4) (check K3_1_*_omega_O_1_equals_1)
$$ O^{(i)}_{AB}=V_i^\dagger AB\,V_i=V_i^\dagger A\,(V_iV_i^\dagger)\,B\,V_i=O^{(i)}_AO^{(i)}_B,\qquad O^{(i)}_{A^\dagger}=(O^{(i)}_A)^\dagger $$  (S2.6) (S2.2)의 V_iV_i^†=1 이 핵심: mod-N 에서 A↦O^{(i)}_A 는 *-준동형 (check S2_*_A_to_O_A_preserves_products: 잔차 4e-16; exact-kernel 에서는 V_iV_i^†=P_{𝒱_i}≠1 이라 실패, (S9.2))
$$ \Phi_{12}(\tau_1,\tau_2):=V_2(\tau_2)V_1(\tau_1)^\dagger,\qquad \Phi_{12}^\dagger\Phi_{12}=\Phi_{12}\Phi_{12}^\dagger=1_{\mathcal C\otimes\mathcal S} $$  (S2.7) (S2.2) 두 번: 전체 공간 유니터리, 경계 인공물 없음 (check S5_*_Phi_dag_Phi_eq_1_full_space, X_boundary_annihilation: mod-N 최소 노름 1, exact 4개 소멸)
$$ O^{(2)}_{\Phi_{12}A\Phi_{12}^\dagger}=V_2^\dagger V_2V_1^\dagger AV_1V_2^\dagger V_2=V_1^\dagger AV_1=O^{(1)}_A $$  (S2.8) (S2.2)(S2.7): 공변성 연산자 항등식 (attempt-01 (S5.5)와 같은 논증, 이번엔 사영 없이)

## (S3) 격자 τ 에서의 환원 상태 발전

$$ \Psi\in\mathcal H_{\rm phys}\ \Rightarrow\ k_1=-k_2-\epsilon_e+Nm\ (m\in\mathbb Z)\ \Rightarrow\ e^{ik_1\tau_n}=e^{-i(k_2+\epsilon_e)\tau_n} $$  (S3.1) (S1.2)와 e^{iNm\tau_n}=1 (verify 3, 4)
$$ V_1(\tau_n)\Psi=e^{-i(H_2+H_S)\tau_n}\,V_1(0)\Psi,\qquad V_2(\tau_n)\Psi=e^{-i(H_1+H_S)\tau_n}\,V_2(0)\Psi $$  (S3.2) (S2.1)에 (S3.1) 대입: 유한 Page–Wootters, 격자 τ 에서만 정확 (check S3_*_reduced_state_evolves: 잔차 ≤2e-14)

## (S4) 주장 (1): E_total 은 관측자·τ 무관

$$ E^{(i)}_{\rm total}(\tau):=\frac{\omega_\Psi(O^{(i)}_{1\otimes H_S}(\tau))}{\omega_\Psi(O^{(i)}_1(\tau))}=\langle V_i(\tau)\Psi|1\otimes H_S|V_i(\tau)\Psi\rangle $$  (S4.1) 정의, (S2.5)로 분모 1
$$ =\langle V_i(0)\Psi|e^{i(H_j+H_S)\tau}(1\otimes H_S)e^{-i(H_j+H_S)\tau}|V_i(0)\Psi\rangle=\langle V_i(0)\Psi|1\otimes H_S|V_i(0)\Psi\rangle $$  (S4.2) (S3.2)와 [1⊗H_S, H_j+H_S]=0 ⇒ τ 무관
$$ \Psi=\sum_{k_2,e}c_{k_2e}|(-k_2-\epsilon_e)\bmod N,\,k_2,\,e\rangle\ \Rightarrow\ \langle V_1(0)\Psi|1\otimes H_S|V_1(0)\Psi\rangle=\sum_{k_2,e}|c_{k_2e}|^2\epsilon_e=\langle\Psi|1\otimes1\otimes H_S|\Psi\rangle $$  (S4.3) (S2.1)과 H_S 의 e-대각성; 같은 계수를 (k_1,e) 로 재라벨해도 합은 같으므로 i=2 도 동일 ⇒ i 무관
$$ E_{\rm total}=\langle\Psi|H_S|\Psi\rangle\quad\forall i,\ \forall\tau\in2\pi\mathbb Z_N/N $$  (S4.4) 주장 (1) 증명 끝 (check K3_1_{A0,A2,B}: 20 상태×2 관측자×N τ, 잔차 ≤1e-15). 일반화: [A,H_S]=0 이면 ω(O^{(i)}_{1⊗A}) 도 i·τ 무관 (같은 논증)
$$ \Phi_{12}(1\otimes H_S)\Phi_{12}^\dagger=V_2V_1^\dagger(1\otimes H_S)V_1V_2^\dagger=V_2\,(H_S|_{\mathcal H_{\rm phys}})\,V_2^\dagger=1\otimes H_S $$  (S4.5) (S4.3)의 V_i^†(1⊗H_S)V_i=H_S|_{H_phys} 가 i 무관 ⇒ Φ_12 는 1⊗H_S 와 교환 (check S5_*_Phi_commutes_with_1xHS)

## (S5) 주장 (2): 분할의 정확식

$$ H_S=\sum_{b,b'}P_bH_SP_{b'}=D(H_S)+X(H_S),\qquad D:=\sum_bP_bH_SP_b,\ X:=\sum_{b\neq b'}P_bH_SP_{b'} $$  (S5.1) 완전성 Σ_bP_b=1 두 번
$$ E^{(i)}_{\rm seen}(\tau):=\sum_b\frac{\omega_\Psi(O^{(i)}_{1\otimes P_bH_SP_b})}{\omega_\Psi(O^{(i)}_1)}=\langle V_i\Psi|1\otimes D|V_i\Psi\rangle $$  (S5.2) 정의, A↦O_A 선형
$$ E^{(i)}_{\rm unseen}(\tau):=E_{\rm total}-E^{(i)}_{\rm seen}(\tau)=\langle V_i\Psi|1\otimes X|V_i\Psi\rangle=\sum_{b\neq b'}\frac{\omega_\Psi(O^{(i)}_{1\otimes P_bH_SP_{b'}})}{\omega_\Psi(O^{(i)}_1)} $$  (S5.3) (S5.1)(S4.1)(S5.2): 주장 (2) 정확식 (check K3_2_*: 잔차 ≤6e-16)
$$ p^{(i)}_b(\tau):=\omega_\Psi(O^{(i)}_{1\otimes P_b}),\qquad E^{(i)}_b(\tau):=\frac{\omega_\Psi(O^{(i)}_{1\otimes P_bH_SP_b})}{p^{(i)}_b}\ \Rightarrow\ \sum_bp^{(i)}_bE^{(i)}_b=E^{(i)}_{\rm seen} $$  (S5.4) 조건부 에너지 정의; 항등식은 정의로부터 즉시
$$ \text{B: }D=(E_*+h\sigma_z)\otimes n_Q,\quad X=g\sigma_x\otimes n_Q;\qquad \text{A2: }D=E_*n_Q,\quad X=g\sigma_x\otimes n_Q $$  (S5.5) 명시 계산 (P_bσ_xP_b=0, P_bσ_zP_b=(−1)^bP_b); n_Q=1 섹터 블로흐 벡터 (x,z) 에 대해 E_total−E_seen=gx (verify 5)

## (S6) 주장 (3): [P_b,H_S]=0 이면 장부가 닫히고 관측자 무관

$$ [P_b,H_S]=0\ \Rightarrow\ P_bH_SP_{b'}=H_SP_bP_{b'}=0\ (b\neq b')\ \Rightarrow\ X=0\ \Rightarrow\ E^{(i)}_{\rm unseen}(\tau)\equiv0 $$  (S6.1) (S5.3) (check K3_3_A0_E_unseen_zero: 1e-16)
$$ [P_b,H_S]=0\ \Rightarrow\ [P_b,H]=0,\ [P_bH_SP_b,H]=0\ \Rightarrow\ p^{(i)}_b,\ E^{(i)}_b\ \text{는 (S4.4) 일반화로 } i,\tau\text{ 무관} $$  (S6.2) Dirac 관측량 (check K3_3_A0_pb_and_Eb_observer_and_tau_independent: 3e-16)
$$ \sum_bp_bE_b=E_{\rm seen}=E_{\rm total}=\langle\Psi|H_S|\Psi\rangle $$  (S6.3) (S5.4)(S6.1)(S4.4): 제약 수준 장부 닫힘 (check K3_3_A0_sum_pb_Eb_equals_E_total)
$$ \text{E39 형: }H_S'=E_*(n_Q+1-n_R),\ \Psi_S=\cos\theta|r{=}1,q{=}1\rangle+\sin\theta|r{=}0,q{=}0\rangle\ \Rightarrow\ E_0=E_1=E_*,\ p_1=\cos^2\theta,\ p_0E_*+p_1E_*=E_*=E_{\rm total} $$  (S6.4) E39-B 의 n_b=1−n_s 저장 구조를 레지스터 0(이동 가지)에 E_* 한 개로 옮긴 것: "한 번 영수증" p_mE_*+p_LE_*=E_* 가 제약 수준에서 i·τ 무관 재현 (check K3_3_E39C: 잔차 0, p_L=0.585)
$$ \text{A0 상관 상태 (H_S=E_*n_Q): }E_1=E_*,\ E_0=0,\ E_{\rm total}=p_LE_* $$  (S6.5) 플럭스 에너지만 셀 때의 장부 (check K3_3_A0_correlated_state)

## (S7) 주장 (4): 프레임 의존의 정확한 조건과 수송 공변성

$$ [D,H_S]=0\ \Rightarrow\ E^{(i)}_{\rm seen}(\tau)=\langle V_i\Psi|1\otimes D|V_i\Psi\rangle\ \text{는 (S4.4) 일반화로 } i,\tau\text{ 무관}\ \Rightarrow\ E^{(i)}_{\rm unseen}\ \text{도 무관} $$  (S7.1) 프레임 무관의 충분조건: [P_b,H_S]≠0 이어도 D 가 Dirac 이면 분할은 프레임 무관
$$ \text{A2 (과제 문구): }H_S=E_*n_Q+g\sigma_x\otimes n_Q\ \Rightarrow\ [P_b,H_S]=\pm ig\sigma_y\otimes n_Q\neq0,\ \text{그러나 }D=E_*n_Q,\ [D,H_S]=0 $$  (S7.2) 반례: 과제의 (4a) "g≠0 ⇒ 프레임 의존 상태 존재" 는 이 H_S 에서 거짓 (check NEG_A2_literal: max|E^{(1)}−E^{(2)}_unseen|=3e-16, ‖[P_b,H_S]‖=2, ‖[D,H_S]‖=0)
$$ [D,H_S]\neq0\ \Leftrightarrow\ \exists\,e\neq e':\ D_{ee'}:=\langle e|D|e'\rangle\neq0,\ \epsilon_e\neq\epsilon_{e'} $$  (S7.3) H_S 고유기저에서 [D,H_S]_{ee'}=D_{ee'}(ε_{e'}−ε_e)
$$ \Psi=a|k_1,k_2,e\rangle+b|k_1',k_2,e'\rangle,\quad k_1=(-k_2-\epsilon_e)\bmod N,\ k_1'=(-k_2-\epsilon_{e'})\bmod N,\ |a|^2+|b|^2=1 $$  (S7.4) 두 준위 물리 상태; |ε|<N/2 ⇒ ε_e≢ε_{e'} (N) ⇒ k_1≠k_1'
$$ E^{(1)}_{\rm seen}(0)=|a|^2D_{ee}+|b|^2D_{e'e'}+2\,\mathrm{Re}(\bar abD_{ee'}),\qquad E^{(2)}_{\rm seen}(0)=|a|^2D_{ee}+|b|^2D_{e'e'} $$  (S7.5) (S2.1)(S5.2): 관측자 1 은 |k_2e⟩,|k_2e'⟩ 간섭항을 보고, 관측자 2 는 ⟨k_1|k_1'⟩=0 으로 간섭항 소거 (verify 6)
$$ E^{(1)}_{\rm unseen}-E^{(2)}_{\rm unseen}=-2\,\mathrm{Re}(\bar abD_{ee'})\neq0\ \text{(a,b 적당히)} $$  (S7.6) 주장 (4a) 다듬은 형태 증명 끝: [D,H_S]≠0 ⇒ 프레임 의존 상태 존재 (check K3_4a_B_two_level: 예측 2.1359 = 관측 2.1359, D_{ee'}=2.4, k_1=2≠k_1'=12; K3_4a_B_E_unseen_frame_dependent_exists: 무작위 20 상태 모두 차이 >SEP, 최대 1.004)
$$ E^{(2)}_{\rm unseen}[\Phi]:=\sum_{b\neq b'}\frac{\omega_\Psi(O^{(2)}_{\Phi_{12}(1\otimes P_bH_SP_{b'})\Phi_{12}^\dagger})}{\omega_\Psi(O^{(2)}_1)}=\sum_{b\neq b'}\omega_\Psi(O^{(1)}_{1\otimes P_bH_SP_{b'}})=E^{(1)}_{\rm unseen} $$  (S7.7) (S2.8): 주장 (4b) 수송 공변성, τ_1≠τ_2 도 포함 (check K3_4b_B_transported: 1e-15; tau1_ne_tau2: 3e-16; NEG_B_identity_transport_fails: 1.004)
$$ \Phi_{12}(1\otimes P_b)\Phi_{12}^\dagger\neq1\otimes P_b'\ \text{일반적으로: 수송된 사영은 시계 라벨과 얽힌다 — 분할은 프레임 불변이 아니라 프레임 공변} $$  (S7.8) (S2.1)로 Φ_12|k_2e⟩=|(-k_2-ε_e) mod N, e⟩ 는 e 에 따라 시계 이동이 다름 (attempt-01 (S5.7) 의 경계 없는 판)

## (S8) 주장 (5): 감김 회계

$$ K=K^\dagger,\ \mathrm{spec}(K)\subset\mathbb Z,\ [K,H_S]=0\ \Rightarrow\ e^{2\pi i(H+NK)/N}=e^{2\pi iH/N}e^{2\pi iK}=U\ \Rightarrow\ \Pi[H_S+NK]=\Pi[H_S] $$  (S8.1) Π 는 H_S 의 mod-N 류에만 의존 (check K3_5_*: Pi_residual 0)
$$ V_i,\ \mathcal H_{\rm phys}\ \text{불변}\ \Rightarrow\ E_{\rm total}[H_S+NK]-E_{\rm total}[H_S]=N\langle V_i\Psi|1\otimes K|V_i\Psi\rangle=N\langle\Psi|K|\Psi\rangle $$  (S8.2) (S4.3)의 선형성; 정확한 회계 규칙 (check K3_5_*_single_eigenvalue_shift: accounting_residual 9e-16)
$$ K=m\cdot1\ \Rightarrow\ \Delta E_{\rm total}=Nm\in N\mathbb Z $$  (S8.3) 균일 대표원 이동에서만 "N의 정수배" (verify 7; check K3_5_*_uniform: 7, −14, 13, −26 정확)
$$ K=P_e\ (\text{단일 고윳값 이동})\ \Rightarrow\ \Delta E_{\rm total}=N\,p_e,\quad p_e=\langle\Psi|P_e|\Psi\rangle\in[0,1] $$  (S8.4) 과제 문구 (5) "정수배" 는 일반 상태에서 거짓; H_S 고유상태(p_e∈{0,1})에서만 정수배 (check: 무작위 상태 ΔE/N 소수부 0.16–0.44)
$$ |\epsilon_e|<N/2\ \text{대표원 선택이 } K \text{ 를 고정}\ \Rightarrow\ E_{\rm total}=\langle\Psi|H_S^{\rm can}|\Psi\rangle\ \text{가 유일} $$  (S8.5) 정확한 에너지는 대표원 선택으로 정의된다는 회계 명시
$$ H_i\mapsto H_i+N\ \Rightarrow\ \Pi,\ V_i(\tau_n),\ E_{\rm total}\ \text{불변},\quad W\mapsto W+1 $$  (S8.6) (S0.2)(S1.4): 시계 대표원은 S-에너지에 무관, 총 제약 에너지(감김수)만 옮김 (check K3_5_*_clock_representative_shift: dE=−1e-16, 감김 스펙트럼 {0,1,2}, 이동 잔차 4e-15)

## (S9) exact-kernel (attempt-01) 과의 대조

$$ \text{exact: }\dim\mathcal H_{\rm phys}=\sum_e(N-|\epsilon_e|)=24<28,\quad V_iV_i^\dagger=P_{\mathcal V_i}\neq1 $$  (S9.1) N=7, A2 의 H_S (check X_exact_kernel_phys_dim)
$$ \text{exact: }\min_{A,B}\frac{\|O_{AB}-O_AO_B\|}{1+\|AB\|}=0.43>\text{SEP},\qquad \|\Phi^\dagger\Phi-1\|=1,\ \mathrm{rank}\,\Phi=24 $$  (S9.2) (S2.6)(S2.7)의 mod-N 결론이 exact 에서 깨짐 (check X_exact_kernel_A_to_O_A_fails, X_exact_kernel_Phi_dag_Phi_ne_1)
$$ \text{exact: }\#\{(k_2,e):\Phi|k_2e\rangle=0\}=4,\qquad \text{mod-}N:\ \|\Phi|k_2e\rangle\|=1\ \forall(k_2,e) $$  (S9.3) 경계 소멸 (Q-0004 의 출발 결함) 이 mod-N 에서 부재 (check X_boundary_annihilation_exact_vs_modN)
$$ \text{exact 에서도 (1) 은 성립: }E_{\rm total}=\langle\Psi|H_S|\Psi\rangle $$  (S9.4) (S4.3) 논증이 V_i 등거리성만 쓰기 때문 (check X_exact_kernel_claim1_still_holds: 3e-16). 차이는 (2)–(4) 의 곱·수송 구조에만 있음

## (S10) 증명한 것과 증명하지 않은 것

$$ \text{증명: (S1.2)(S1.3)(S2.2)(S2.6)(S2.7)(S2.8)(S4.4)(S4.5)(S5.3)(S6.1)–(S6.3)(S7.1)(S7.2)(S7.6)(S7.7)(S8.1)–(S8.4)(S8.6)} $$  (S10.1) 범위: 유한 정수 스펙트럼, Z_N 군평균, 격자 τ
$$ \text{증명하지 않은 것 1: 과제 문구 (4a)·(5) 그대로는 거짓 — (S7.2)(S8.4) 가 반례이며 본 attempt 는 다듬은 진술만 증명} $$  (S10.2) 주장 축소를 명시
$$ \text{증명하지 않은 것 2: }[D,H_S]\neq0\ \Rightarrow\ \text{"거의 모든" 상태가 프레임 의존 (존재만 증명, 20 상태 전부 의존은 관측)} $$  (S10.3) 관측 근접은 근거 아님
$$ \text{증명하지 않은 것 3: 격자 밖 }\tau\text{, 연속 }U(1)\text{ 이상 시계 }(\ell^2(\mathbb Z))\text{ 극한, }N_1\neq N_2\text{ (군이 }Z_{\mathrm{lcm}}\text{ 로 바뀜)} $$  (S10.4) 비범위
$$ \text{증명하지 않은 것 4: 레지스터 기저 }\{P_b\}\text{ 자체를 관계적으로(시계에 조건화해) 정의했을 때의 장부 — 본 attempt 는 }P_b\text{ 를 운동학적으로 고정} $$  (S10.5) 비범위
$$ \text{증명하지 않은 것 5: 36장 M3(양성·Hadamard·foliation 독립)·중력 결합 — 0+0차원 유한계} $$  (S10.6) 비범위
