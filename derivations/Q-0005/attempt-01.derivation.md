---
question: Q-0005
attempt: 1
pivot_step: null
claim: "유한 모형 H_kin=C^16_T⊗C^10_g⊗C^7_r, Λ=T 의 DFT 켤레(Λ_k=k mod 16), H_0=H_g⊗1+1⊗H_r (H_g=diag(0,1,1,2,2,2,3,3,4,5), H_r=diag(0..6), spec H_0⊂[0,11]⊂[0,15] 감김 없음), C=Λ+H_0 (mod 16), Π=(1/16)Σ_t e^{2πitC/16} 에서: C1 [(i) 기각] 군평균은 껍질 제한 Π=Σ_k|k⟩⟨k|⊗Q_{−k} 이며, 임의의 곱상태 χ⊗φ 에서 물리 상태의 Λ 분포는 p(Λ_k)=|χ̂_k|²‖Q_{−k}φ‖²/Z 로 운동학적 상태에서 그대로 상속된다(60 상태 최대 잔차 2.2e−16); 무작위 ψ_kin 200개의 p-벡터 행렬 수치 랭크는 도달 가능 섹터 수 12 (k∈{0,5..15}) 와 같고 d_k/70 에서 최대 0.13 벗어난다 — 제약이 고정하는 것은 지지집합(k∈{1,2,3,4} 는 p=0)과 물리 내적뿐이고 측도는 고정하지 않는다. Λ 는 Dirac 관측량이지만 고정-τ 관계적 관측량 Π(|τ⟩⟨τ|⊗A⊗1)Π 와의 교환자 노름은 무작위 A 10개에서 2.18–2.88 (>SEP) 이라 Λ 는 초선택되지 않으며(사전 규칙: 초선택 문구 강화 없음), [H_g,A]=0 인 A 와 τ-평균 관측량에서만 교환한다(<2.9e−15). C2 [Page 환원] ρ_kin=1/1120 의 군평균 ρ_phys=Π/70 은 p(Λ_k)=d_k/70 (d_k=(1,0,0,0,0,1,2,4,7,9,10,10,9,8,6,3)), 기록 사영 E=1⊗1⊗P_{H_r≥ε} 로 Lüders 조건화하면 p(Λ_k|E)=Tr(EP_k)/Σ_jTr(EP_j) (ε=1..4 최대 잔차 2.5e−16) — 관측자 조건화 분포는 Page 의 관측자 계수 측도이고 그 입력은 Haar 사전분포다. C3 [(ii) 기각] 선택함수 f_k=Tr(EP_k)/d_k 는 도달 가능 섹터에서 상수가 아니라(ε=1 최대 편차 0.78) 조건화가 봉우리를 옮기지만, 최빈 E*=−Λ* 는 ε=1,2,3,4 에서 6,6,6,7 (동률 집합 {6},{6,7},{6,7},{7}) 로 ε 를 추적하고, H_g 다중도를 B=diag(0,0,0,1,2,3,4,5,5,5) 로 바꾸면 ε=4 에서 7→6 으로 한 섹터 이동하며 조건화 평균 ⟨E⟩ 는 모든 ε 에서 0.2 이동한다(>SEP) — 봉우리 위치는 기록 문턱 에너지와 상태 다중도의 함수이므로 사전등록 규칙에 따라 Weinberg 입력을 에너지 언어로 바꿔 쓴 것이며 CE 고유 내용이 아니다. 부호 Λ*=−E* (대표원 −6, −7) 는 제약 C=Λ+H_0 의 부호 관례이고, 척도는 입력 에너지 O(1) 이라 위계가 생기지 않는다."
assumptions:
  - "유한 Z_16 시계: Λ 고유기저를 계산 기저로 택하고 |τ_n⟩=16^{−1/2}Σ_k e^{−ikτ_n}|k⟩, τ_n=2πn/16; C 는 mod 16 으로 정의(감김은 spec H_0⊂[0,11] 이라 없음)"
  - "H_0=H_g⊗1+1⊗H_r 상호작용 없음 (C3 의 전제); 기록 사영 E=1⊗1⊗P_{H_r≥ε} 는 운동학적 고정이며 [E,H_0]=[E,Λ]=0"
  - "C2 의 사전분포 ρ_kin=1/dim (Haar) 는 입력이지 유도가 아님; 다른 ρ_kin 은 C1 에 의해 다른 p 를 준다"
  - "최빈값 동률은 가장 작은 E 로 정의(사전 선언, 동률 집합도 보고); 보조 통계로 조건화 평균 ⟨E⟩ 사용"
  - "C1(c) 의 '초선택' 판정은 고정-τ 관계적 관측량 10개와 τ=τ_0,τ_5 에서의 수치 증인; 다른 관측량 대수를 택하면 결론이 바뀔 수 있음"
  - "Giulini–Marolf 유일성은 유한 차원에서 자명(군평균=사영, rigging 내적=제한 운동학 내적); 연속 미니초공간의 수렴·유일성은 가정하지 않음"
  - "수치 상수: 씨앗 20260902, TOL=1e−10, SEP=1e−3, SVD 상대 tol 1e−8 (결과 전 선언)"
symbols:
  n: integer
  k: integer
  kp: integer
  x: real
  a: positive
  b: positive
  d: positive
  f: positive
  g: positive
verify:
  - type: identity
    lhs: "cos(2*pi*n)"
    rhs: "1"
  - type: identity
    lhs: "sin(2*pi*n)"
    rhs: "0"
  - type: identity
    lhs: "(kp-k)*x"
    rhs: "kp*x-k*x"
  - type: identity
    lhs: "a/(a+b)+b/(a+b)"
    rhs: "1"
  - type: identity
    lhs: "(d*f)/(d*f+g)"
    rhs: "1/(1+g/(d*f))"
  - type: numeric
    expr: "1+3+6+8+9+10+10+9+7+4+2+1-70"
    tol: 1e-12
  - type: numeric
    expr: "16*10*7-1120"
    tol: 1e-12
  - type: numeric
    expr: "16-12-4"
    tol: 1e-12
  - type: numeric
    expr: "1+1+2+3+2+1-10"
    tol: 1e-12
  - type: numeric
    expr: "(10-1)-9"
    tol: 1e-12
  - type: numeric
    expr: "(-6) % 16 - 10"
    tol: 1e-12
  - type: numeric
    expr: "(((-7)+8) % 16) - 8 - (-7)"
    tol: 1e-12
  - type: numeric
    expr: "(4*1+5*3+6*6+7*7+8*6+9*4+10*2+11*1)/30-7.3"
    tol: 1e-12
  - type: numeric
    expr: "(4*3+5*4+6*5+7*3+8*3+9*5+10*4+11*3)/30-7.5"
    tol: 1e-12
  - type: numeric
    expr: "(1+3+6+7+6+4+2+1)-30"
    tol: 1e-12
---

# Q-0005 attempt-01 — 후보 C1 + C3 + C2 (Λ 섹터 측도: 군평균 상속, Page 환원, 기록 선택함수)

기계 검사: 프론트매터 verify 블록(스칼라 항등식 5개·수치 10개, sympy 없음 → symbolic skipped)과
`verify/Q-0005/attempt-01/check_lambda_measure.py`(numpy, 씨앗 20260902, TOL=1e-10, SEP=1e-3, SVD 상대 tol 1e-8 선선언,
필수 검사 30개 전부 통과, 조건부 문구 검사 2개는 사전 규칙대로 "강화 없음"·"C3 존속"). 행렬 수준 주장의 실질 증거는 후자다.
표적: 진전 원장 §2 Q-0005 의 (i) "제약이 Λ 섹터 위 측도를 고정하는가", (ii) "관측자 조건화가 인류원리 입력 없이 봉우리를 만드는가".
사전등록(주차장 2026-09-02): (i) 자유 측도가 남으면 표준 측도 문제로 환원 → 반례; (ii) 봉우리가 표준 인류원리 입력을 넣어야만 생기면 CE 고유 내용 없음.

## (S0) 기호

$$ \mathcal H_{\rm kin}=\mathbb C^{16}_T\otimes\mathbb C^{10}_g\otimes\mathbb C^7_r,\quad \Lambda|k\rangle=k|k\rangle\ (k\in\mathbb Z_{16}),\quad |\tau_n\rangle=16^{-1/2}\sum_ke^{-ik\tau_n}|k\rangle,\ \tau_n=2\pi n/16 $$  (S0.1) Λ 고유기저를 계산 기저로; T 의 DFT 켤레
$$ H_g={\rm diag}(0,1,1,2,2,2,3,3,4,5),\ H_r={\rm diag}(0,\dots,6),\ H_0=H_g\otimes1+1\otimes H_r,\quad {\rm spec}\,H_0=\{0,\dots,11\}\subset\{0,\dots,15\} $$  (S0.2) 상호작용 없음; 감김 없음 (check S1_no_wrap_max_specH0_lt_N: 11<16)
$$ C=\Lambda\otimes1+1\otimes H_0\ (\bmod 16),\qquad \Pi=\frac1{16}\sum_{t=0}^{15}e^{2\pi itC/16},\qquad m_g(E):=\dim\ker(H_g-E)=(1,2,3,2,1,1)_{E=0..5} $$  (S0.3) 제약·군평균·H_g 다중도
$$ Q_E:=\text{spectral projector of }H_0\text{ onto }E,\qquad d(E):=\mathrm{rank}\,Q_E=\sum_{E_r=0}^6m_g(E-E_r) $$  (S0.4) H_0 스펙트럼 사영과 다중도(합성곱)

## (S1) 물리 공간 기저와 d_k 표

$$ \frac1{16}\sum_{t=0}^{15}e^{2\pi it(k+E)/16}=\mathbb 1[(k+E)\equiv0\ (16)] $$  (S1.1) 기하급수; 곱기저에서 C 대각 (verify 1–2: 정수 n 에 e^{2πin}=1; check S1_group_average_equals_shell_indicator 1.1e-15, S1_Pi_is_projector 1.1e-15)
$$ \Pi=\sum_{k\in\mathbb Z_{16}}|k\rangle\langle k|\otimes Q_{(-k)\bmod16},\qquad \mathcal H_{\rm phys}=\mathrm{ran}\,\Pi=\bigoplus_k|k\rangle\otimes\mathrm{ran}\,Q_{-k} $$  (S1.2) (S1.1) 을 (S0.3) 에 대입: 군평균 = 껍질 Λ≡−H_0 제한
$$ d(E)_{E=0..11}=(1,3,6,8,9,10,10,9,7,4,2,1),\quad \sum_Ed(E)=70=\dim\mathcal H_{\rm phys},\qquad d(5)=1+1+2+3+2+1=10 $$  (S1.3) (S0.4) 손계산 (verify 6, 9; check S1_d_of_E_table 일치, S1_dim_phys_70)
$$ P_k:=|k\rangle\langle k|\otimes Q_{-k},\quad d_k:=\mathrm{rank}\,P_k=d((-k)\bmod16):\ (d_0,d_5,\dots,d_{15})=(1,\,1,2,4,7,9,10,10,9,8,6,3),\quad d_{1..4}=0 $$  (S1.4) 도달 가능 섹터 12개 k∈{0,5..15}, 불가 4개 (verify 8; check S1_d_k_unreachable_zero)

## (S2) 군평균은 Λ-선형 제약에서 껍질 제한이다 — C1 폐형식

$$ \psi_{\rm kin}=\chi\otimes\varphi,\ \hat\chi_k:=\langle k|\chi\rangle\ \Rightarrow\ \Pi\psi_{\rm kin}=\sum_k\hat\chi_k\,|k\rangle\otimes Q_{-k}\varphi $$  (S2.1) (S1.2) 적용
$$ \|(|k\rangle\langle k|\otimes1)\Pi\psi_{\rm kin}\|^2=|\hat\chi_k|^2\|Q_{-k}\varphi\|^2\ \Rightarrow\ p(\Lambda_k)=\frac{|\hat\chi_k|^2\|Q_{-k}\varphi\|^2}{\sum_j|\hat\chi_j|^2\|Q_{-j}\varphi\|^2} $$  (S2.2) k-블록 직교성; Z 로 규격화 (verify 4; check C1a_closed_form_p_Lambda_60_states 최대 잔차 2.2e-16: χ∈{δ_{τ_0}, 폭4 창, τ-균일} × 무작위 φ 20)
$$ \chi=|\tau_0\rangle\ \Rightarrow\ |\hat\chi_k|^2=\tfrac1{16}\ \Rightarrow\ p(\Lambda_k)\propto\|Q_{-k}\varphi\|^2;\qquad \chi=16^{-1/2}\sum_n|\tau_n\rangle=|k{=}0\rangle\ \Rightarrow\ p(\Lambda_k)=\delta_{k0} $$  (S2.3) 두 극단: 날카로운 시계 → Λ 측도는 φ 의 에너지 분포, 균일 시계 → Λ=0 한 점 (check C1a_delta_tau0_has_flat_Lambda_amplitudes 0, C1a_uniform_tau_equals_Lambda_eigenvector_k0 2.7e-15)
$$ \mathrm{rank}\{p^{(i)}\}_{i=1}^{200}=12=\#\{k:d_k>0\},\qquad \max_{i,k}|p^{(i)}_k-d_k/70|=0.13>\text{SEP} $$  (S2.4) 무작위 ψ_kin (곱상태 아님) 200개의 측도가 도달 가능 섹터 전체를 펼침; 특정 측도로 수렴하지 않음 (check C1b_rank_of_200_random_measures_equals_12: 유지 최소 특이값 0.169, 버린 최대 1.4e-16; C1b_measures_not_all_equal_to_dk_over_70)
$$ \text{C1 결론: 제약이 고정하는 것 }=\ \{k:p_k\neq0\}\subset\{0,5,\dots,15\}\ \text{(지지집합)};\quad \text{고정하지 않는 것 }=\ p\ \text{자체 (}\psi_{\rm kin}\text{ 상속)} $$  (S2.5) (i) 유한 모형 반례: 사전등록 시험 1 발동 → 표준 측도 문제로 환원 (check C1b_unreachable_sectors_zero 3e-31)

## (S3) 측도는 운동학적 상태의 함수이며, Giulini–Marolf 유일성은 내적을 고정하고 상태를 남긴다

$$ [\Lambda,C]=[\Lambda,H_0]=0\ \Rightarrow\ \Lambda\ \text{는 Dirac 관측량},\qquad \Lambda|_{\mathcal H_{\rm phys}}=-H_0\ (\bmod16) $$  (S3.1) 서로 다른 인자에 작용; 껍질 위에서 Λ 는 −(계+기록 에너지)
$$ O_A(\tau):=\Pi(|\tau\rangle\langle\tau|\otimes A\otimes1)\Pi=\sum_{k,k'}\langle k|\tau\rangle\langle\tau|k'\rangle\,|k\rangle\langle k'|\otimes Q_{-k}(A\otimes1)Q_{-k'} $$  (S3.2) 고정-τ 관계적 관측량의 블록 전개 ((S1.2) 양쪽 대입)
$$ [O_A(\tau),\Lambda]_{kk'}=(k'-k)\,O_A(\tau)_{kk'}\ \Rightarrow\ [O_A(\tau),\Lambda]=0\ \Leftrightarrow\ Q_{-k}(A\otimes1)Q_{-k'}=0\ \forall k\neq k' $$  (S3.3) 블록별 (verify 3); [A,H_g]≠0 이면 서로 다른 E 를 잇는 블록이 살아남음
$$ \max_{10\,A}\|[O_A(\tau_0),\Lambda]\|=2.88,\ \min=2.18>\text{SEP};\quad [A,H_g]=0:\ \le2.7\times10^{-15};\quad \bar O_A:=\tfrac1{16}\sum_nO_A(\tau_n)=\tfrac1{16}\Pi(1\otimes A\otimes1)\Pi:\ \le2.9\times10^{-15} $$  (S3.4) 사전 규칙 "전부 <TOL 이면 초선택 문구 강화" 는 발동하지 않음: Λ 는 고정-τ 관계적 관측량 대수에서 중심이 아님 (check C1c_generic_A_all_commutators_below_TOL: fail(조건부), C1c_generic_A_commutators_above_SEP, C1c_tau5_generic_A_commutator_above_SEP 2.70, C1c_diag_A_commutes_with_Lambda, C1c_tau_averaged_O_commutes_with_Lambda)
$$ \text{Giulini–Marolf: rigging 사상 }\eta\text{ 가 수렴하면 관측량 대수와 얽힌 }\eta\text{ 는 초선택 섹터마다 양의 상수 배까지 유일} $$  (S3.5) 문헌 진술 (증명은 인용); 유한 차원에서는 η=Π, 물리 내적 = 제한된 운동학 내적으로 자명
$$ (S3.4)\ \Rightarrow\ \Lambda\text{ 섹터는 초선택 섹터가 아님}\ \Rightarrow\ \text{섹터 간 상대 규격화는 GM 이 고정}\ (\text{전체 상수 하나만 자유}) $$  (S3.6) 내적은 고정됨; 만약 초선택이었다면 상대 규격화까지 자유여서 (i) 는 더 강하게 실패했을 것
$$ \text{고정된 내적 아래서도 }p(\Lambda_k)=\|P_k\Psi\|^2/\|\Psi\|^2,\ \Psi=\Pi\psi_{\rm kin}\text{ 은 (S2.2) 대로 }\psi_{\rm kin}\text{ 의 함수} $$  (S3.7) 유일성이 고정하는 것 = 내적(상대 규격화 포함); 남기는 것 = 상태 = 측도. (i) 기각의 근거는 (S2.5)+(S3.7)
$$ \text{문헌 C1: Unruh–Wald 1989·Smolin 2009 와 같음 — }\Lambda\text{ 는 시계 켤레 보존량이고 제약은 그 값을 고정하지 않는다; 다름 — 상속 폐형식 (S2.2), 도달 섹터 랭크 12 (S2.4), 고정-}\tau\text{ 관계적 관측량이 섹터를 섞어 초선택이 아님 (S3.4)} $$  (S3.8) 한 줄 대조

## (S4) C2: Haar 사전분포의 군평균은 Page 관측자 계수 측도로 환원된다

$$ \rho_{\rm kin}=\tfrac1{1120}\ \Rightarrow\ \rho_{\rm phys}:=\frac{\Pi\rho_{\rm kin}\Pi}{\mathrm{Tr}(\Pi\rho_{\rm kin})}=\frac{\Pi}{70}\ \Rightarrow\ p(\Lambda_k)=\mathrm{Tr}(P_k)/70=d_k/70 $$  (S4.1) (S1.2)(S1.4) (verify 7; check C2_haar_p_equals_dk_over_70 1.5e-16; 최댓값 k=10,11 동률 10/70)
$$ E:=1\otimes1\otimes P_{H_r\ge\epsilon},\quad [E,\Pi]=[E,P_k]=0\ \Rightarrow\ \rho'=\frac{E\rho_{\rm phys}E}{\mathrm{Tr}(E\rho_{\rm phys})}\ \Rightarrow\ p(\Lambda_k|E)=\frac{\mathrm{Tr}(EP_k)}{\sum_j\mathrm{Tr}(EP_j)} $$  (S4.2) Lüders 조건화; E 가 H_0·Λ 와 교환하므로 트레이스가 계수로 환원 (verify 4; check C2_lueders_equals_TrEPk_over_sum_eps1to4 최대 잔차 2.5e-16, ε=1..4)
$$ \mathrm{Tr}(EP_k)=\#\{(g,r):E_g+E_r\equiv-k,\ E_r\ge\epsilon\}=\sum_{E_r=\epsilon}^{6}m_g(-k-E_r)=:n_\epsilon(-k) $$  (S4.3) 계수식; n_1(E)=d(E−1)−m_g(E−7) (verify 10: n_1(7)=10−1=9)
$$ \text{C2 결론: 관측자 조건화 분포 }=\text{ 관측자(기록)를 포함하는 껍질 상태의 개수 비 }=\text{ Page 의 관측자 계수 측도; 그 입력은 }\rho_{\rm kin}=1/\dim $$  (S4.4) 사전분포 없이는 (S2.2) 에 의해 어떤 p 도 가능 → "입력 없는 봉우리" 는 정의상 존재하지 않음
$$ \text{문헌 C2: Page 2008 과 같음 — 관측자 존재 조건화가 관측 계수 가중이 된다; 다름 — 여기서는 가중이 공준이 아니라 Haar+Lüders 로부터의 등식이며, 따라서 가중의 자의성은 사전분포의 자의성으로 옮겨진다} $$  (S4.5) 한 줄 대조

## (S5) C3: 선택함수 폐형식과 봉우리의 ε·다중도 의존, 부호·척도

$$ f_k:=\frac{\mathrm{Tr}(EP_k)}{d_k}=\frac{n_\epsilon(-k)}{d(-k)},\qquad p(\Lambda_k|E)=\frac{d_kf_k}{\sum_jd_jf_j} $$  (S5.1) 선택함수 정의; Bayes 형 (verify 5)
$$ \epsilon=1:\ f=(0,\tfrac13,\tfrac12,\tfrac34,\tfrac89,\tfrac9{10},1,1,1,1,1,1)_{E=0..11},\quad \max_k|f_k-\bar f|=0.78>\text{SEP} $$  (S5.2) 상수가 아님 → 사전 규칙 "(a) 상수면 C3 기각" 은 발동하지 않음; 조건화는 낮은 E 를 억제해 봉우리를 옮긴다 (check C3a_selection_function_constant_eps1: fail(조건부); C3a_selection_function_nonconstant_all_eps 0.78, 0.66, 0.56, 0.53)
$$ n_\epsilon(E)\ \text{최빈값 }E^*(\epsilon):\ \epsilon=1,2,3,4\ \mapsto\ 6,6,6,7\ (\text{동률 }\{6\},\{6,7\},\{6,7\},\{7\});\qquad \Lambda^*=(-E^*)\bmod16=10,10,10,9 $$  (S5.3) ε 에 단조 비감소, 총 이동 1 섹터 (check C3b_mode_nondecreasing_in_eps_A, C3b_mode_shifts_with_eps_A)
$$ H_g\to B={\rm diag}(0,0,0,1,2,3,4,5,5,5):\ d^B(E)=(3,4,5,6,7,10,10,7,6,5,4,3),\ E^*_B(\epsilon)=6,6,6,6\ (\text{동률 }\{6\},\{6,7\},\{6,8\},\{6,9\}) $$  (S5.4) ε=4 에서 7→6 한 섹터 이동 (check C3b_mode_shifts_with_multiplicity_B_some_eps; S1_d_of_E_table_B_sums_to_70)
$$ \langle E\rangle_{\epsilon=4}=\frac{4\cdot1+5\cdot3+6\cdot6+7\cdot7+8\cdot6+9\cdot4+10\cdot2+11\cdot1}{30}=7.3,\quad \langle E\rangle^B_{\epsilon=4}=\frac{4\cdot3+5\cdot4+6\cdot5+7\cdot3+8\cdot3+9\cdot5+10\cdot4+11\cdot3}{30}=7.5 $$  (S5.5) 동률 없는 보조 통계: 모든 ε 에서 이동 0.2>SEP (verify 13–15; check C3b_mean_shifts_with_multiplicity_B_gt_SEP: 평균 A=(5.8,6.3,6.8,7.3), B=(6.0,6.5,7.0,7.5))
$$ \text{C3 결론: 봉우리 위치 }E^*=E^*(\epsilon,\ m_g)\ \Rightarrow\ \text{사전등록 시험 2 발동: 기록 문턱(은하 형성 문턱의 에너지 판)과 상태 계수를 넣어야 위치가 정해진다 — (ii) 기각} $$  (S5.6) (S5.3)(S5.4)(S5.5); 인류원리 입력 없는 봉우리 아님
$$ \text{부호: }\Lambda|_{\rm phys}=-H_0\ \Rightarrow\ \Lambda^*=-E^*\ \text{(대표원 }((-E^*+8)\bmod16)-8=-6,-7<0) $$  (S5.7) C=Λ+H_0 의 부호 관례; C=Λ−H_0 였다면 +E*. 예측이 아니라 관례 (verify 11–12; check C3b_sign_Lambda_star_negative_representative)
$$ \text{척도: }E^*\in\{6,7\}\subset{\rm spec}\,H_0,\ \epsilon\in\{1,..,4\}\ \text{입력 단위 O(1)}\ \Rightarrow\ |\Lambda^*|\sim\text{입력 에너지},\ \text{위계(작은 수) 생성 없음} $$  (S5.8) 관측 Λ 의 10^{−120} 은 이 모형이 말할 수 있는 것이 아님 (check C3b_scale_peak_is_order_input_energies)
$$ \text{문헌 C3: Weinberg 1987 과 같음 — 기록(구조) 형성 문턱이 창을 고르고 봉우리가 그 문턱을 추적한다; 다름 — 문턱이 기록 부분계의 에너지 문턱 }\epsilon\text{ 이며, 유한 모형에서 봉우리 이동을 정확히 셀 수 있어 '문턱 없이도 봉우리' 가 없음을 확인} $$  (S5.9) 한 줄 대조

## (S6) 가정과 증명하지 않은 것

$$ \text{증명(해석+수치): (S1.1)–(S1.4), (S2.1)–(S2.3), (S3.1)–(S3.3), (S3.6)–(S3.7) 의 함의, (S4.1)–(S4.3), (S5.1), (S5.7)} $$  (S6.1) 범위: Z_16 시계, 무상호작용 H_0, 두 다중도 표
$$ \text{수치 증인만: (S2.4) 랭크 12, (S3.4) 비교환 노름, (S5.2)–(S5.5) 봉우리 이동 — 이 모형·이 씨앗의 존재 진술} $$  (S6.2) 일반 (N, m_g, m_r) 정리 아님
$$ \text{증명하지 않은 것 1: 연속 미니초공간 — }\Lambda\in\mathbb R,\ \text{비콤팩트 군평균의 수렴·GM 유일성·rigging 내적의 존재} $$  (S6.3) (S3.5) 는 인용이며 유한 차원 밖에서 검증 안 함
$$ \text{증명하지 않은 것 2: 실제 관측 }\Lambda\text{ 값·부호·척도 — (S5.7)(S5.8) 은 관례와 입력 단위의 재진술} $$  (S6.4) 위계 문제는 건드리지 않음
$$ \text{증명하지 않은 것 3: 인류원리 논증 자체의 타당성 — 본 attempt 는 '입력 없는 봉우리' 의 부재만 보였고 입력 있는 논증의 옳고 그름은 판정하지 않음} $$  (S6.5) 비범위
$$ \text{증명하지 않은 것 4: 상호작용 있는 }H_0\text{ 나 시계–기록 결합에서 (S4.2) 의 교환 가정 }[E,\Pi]=0\text{ 이 깨질 때의 조건화 분포} $$  (S6.6) Q-0002 attempt-03 형 결합은 여기서 다루지 않음
$$ \text{증명하지 않은 것 5: 최빈값 동률(ε=2,3 에서 \{6,7\})은 모형의 산술적 우연이며, 판정은 사전 선언한 동률 규칙과 평균 이동 (S5.5) 에 의존} $$  (S6.7) 규칙을 바꾸면 (S5.3)(S5.4) 의 정수 이동 진술은 달라질 수 있음(평균 이동은 불변)

```json
{"derivation_path":"C:/Users/22310326/Desktop/Clarus-Equation/derivations/Q-0005/attempt-01.derivation.md","check_paths":["C:/Users/22310326/Desktop/Clarus-Equation/verify/Q-0005/attempt-01/check_lambda_measure.py","C:/Users/22310326/Desktop/Clarus-Equation/verify/Q-0005/attempt-01/result.json"],"all_pass":true,"verdict_i":"기각(유한 모형 반례): 군평균은 껍질 제한이고 Λ 측도 p(Λ_k)=|χ̂_k|²‖Q_{-k}φ‖²/Z 는 ψ_kin 상속(잔차 2.2e-16), 200 상태 랭크 12=도달 섹터 수; 제약이 고정하는 것은 지지집합과 내적뿐. Λ 는 고정-τ 관계적 관측량과 비교환(노름 2.2–2.9)이라 초선택 아님","verdict_ii":"기각(유한 모형): 조건화 분포는 Page 계수 측도 Tr(EP_k)/ΣTr(EP_j)(Haar 사전분포 입력), 선택함수 비상수(0.78)라 봉우리는 있으나 위치 E*=6,6,6,7 이 ε 를 추적하고 다중도 B 에서 7→6·평균 0.2 이동 → 기록 문턱+계수 입력의 함수 = Weinberg 입력의 에너지 판; 부호 Λ*=−E* 는 제약 부호 관례, 척도 O(입력 에너지)","assumptions":["Z_16 시계·mod 16 제약·감김 없음","H_0 무상호작용, 기록 사영 E 운동학 고정·[E,Π]=0","C2 의 Haar 사전분포는 입력","최빈값 동률은 최소 E(사전 선언), 보조 통계 평균","초선택 판정은 관측량 10개·τ_0,τ_5 수치 증인","GM 유일성은 유한 차원에서 자명, 연속은 인용"],"not_proved":["연속 미니초공간(Λ∈R, 비콤팩트 군평균 수렴·rigging 존재)","실제 관측 Λ 값·10^-120 위계","인류원리 논증 자체의 타당성","상호작용/시계–기록 결합에서 [E,Π]≠0 인 조건화","일반 (N,m_g,m_r) 정리(수치 증인은 이 모형·이 씨앗)"],"parking":["Λ 초선택이 성립하는 관측량 대수(τ-평균·H_g-보존)만으로 관측을 제한하면 GM 상대 규격화 자유가 살아나 (i) 가 더 강하게 실패 — 그 대수의 물리적 정당성 검토","시계–기록 결합(Q-0002 attempt-03 형)에서 [E,Π]≠0 일 때 Lüders 조건화가 Page 계수에서 얼마나 벗어나는지"]}
```
