---
question: Q-0008
attempt: 6
ladder_step: 3
claim: '커널 법칙(카드 F-02 사다리 3단): 가우스 등방 label(공분산 κ⊗I_16)·polar 정렬 정확 simple cell에서, 2단 정확 항등식과 Isserlis/Wick 전개로 $\bar\epsilon^2:=E[\epsilon^2]=\epsilon_\star^2\,\lVert H\kappa H\rVert_F^2/n^2\,(1+O(\delta^2))$, $\epsilon_\star^2=2T_2\delta^4/\lVert G_0\rVert_F^2=10\,\delta^4$ (T_2=60, ‖G_0‖²=12는 정확 구조상수). 평균 바닥 없음은 등방성 보조정리 Σ_a M_aa=0(SO(4) 켤레 등변성 + Schur)으로 모든 δ에서 정확. 닫힌 식 4종 ‖HIH‖²=n−1, 두 종 4n²p²(1−p)², chain (n²−1)(2n²+7)/180, star n−2+1/n²과 교차항 tr(Hκ)=Σ_u s_u(1−s_u/n)을 기호로 증명. γ_her=1/d_tree는 조건 (C) D/n²≍depth²인 족(chain·caterpillar: D/(n²depth²)→1/90 정확·수치, Cayley: E D/n³ 안정)에서의 조건부 따름정리이고 star-of-chains(D/n²≍depth, γ=1/4)·balanced binary(depth≍log n)·star(D/n²→0)는 (C) 밖. 절단: 홀수 차수는 대칭으로 소멸해 첫 보정은 상대 cδ²이며, δ 3점 시험(씨앗 20260902, 공통난수 512 trial, 보고만)의 함의 절단은 δ=0.005에서 RMS의 +0.026±0.014%(her n=8)·+0.018±0.021%(her n=32)·−0.006±0.004%(iid) — attempt-04 adversary의 상계 0.17–0.23%(SE 미보고)보다 한 자리 작고 기울기 영향 ≈−6e−5(±2e−4); 7단 진폭비의 −2% 계통 결손은 절단 항이 아니다(판정 라벨 "잡음").'
assumptions:
  - 'P_micro 등방성(카드 scope 가정): label ξ_v∈R^{16}의 결합 분포는 중심 가우스이고 공분산이 κ⊗I_16이다. (S4)(S5)에서 실제로 쓰는 성질은 (i) 분포가 동시 켤레 ξ_v↦Λξ_vΛ^T (Λ∈SO(4))에 불변, (ii) ξ↦−ξ 대칭, (iii) 4점 함수가 Isserlis 쌍 축약이다. (i)(ii)는 등방 비가우스에도 성립하지만 (iii)은 가우스 전용(비가우스 첨도 항은 Q-0012)'
  - '정확 simple cell과 polar 정렬(attempt-03 전제 그대로): tl gram(X_v)=0, X_v=polar(G(Σ_0,Σ(e_v)))·Σ(e_v), 양의 Euclidean branch(정렬 거부·MIN_DET=0.05 재추출은 δ=0.005에서 발동 0)'
  - '선형화: 정렬된 cell의 δ-급수 η_v=δLξ_v+δ²Q(ξ_v,ξ_v)+…에서 각 차수 δ^k의 계수는 ξ의 k차 동차 다항식이다(polar 인자와 Σ(e)가 δ의 해석함수, Q-0012 a1 해석식). 절단은 E[ε²]의 상대 O(δ²) 항 하나뿐이며 그 크기는 (S9)에서 실측 상계로 적는다(닫힌 상계 아님)'
  - '구조상수 T_2=Σ_ab‖M_ab‖²=60, T_4=Σ_a‖M_aa‖²=2, Σ_a M_aa=0, G_0=2I_3(‖G_0‖²=12), 다중도 {0:52,1/8:96,1/6:24,1/2:72,2/3:12}는 Q-0012 adversary a1의 sympy 정확 유리수 계산을 인용하고 검사 A가 수치(Richardson, 1e−13)로 재확인한다'
  - '2단 항등식 tl gram(Y)=−nΣ_v tl gram(η_v−η̄)은 E-20260902-008(강한 형태)을 인용한다'
  - '분모 ‖gram Y‖_F의 O(δ) 요동은 ξ의 1차 동차이므로 (ii)로 E[ε²]에 O(δ²)로만 들어간다((S6.3)); RMS 통계는 √E[ε²]이며 ε의 정의는 코드 simplicity_residual(‖tl gram‖_F/‖gram‖_F)과 같다'
  - '조건부 따름정리 (S8)의 조건 (C)는 증명하지 않는다: 족 F에서 D(n)/n²≍depth(n)²이고 depth≍n^{1/d_tree}. 소속·제외는 정확 계산(chain·star 닫힌 식, Cayley 정확 조합론, caterpillar·star-of-chains·binary 정확 driver)으로 확인한 사실이지 정리가 아니다'
  - '검사 C-b(물리 MC 2000 trial, 5% 창)의 표본 상대 SE는 설계 파일럿(씨앗 1, 증거 아님)에서 1.5–2.6%(CV 0.7–1.2)라 5% 창은 2–3σ이며 12개 사례 중 잡음 초과가 가능하다; 그 경우도 결과를 본 뒤 창을 바꾸지 않고 z와 함께 그대로 보고한다. 결정적 검사는 공통난수 비(C-c, 1%)와 대리모형 10⁵ trial(C-a, 2%)이다'
symbols:
  n: positive integer
  n_B: positive integer
  p: nonnegative real
  k: integer
  v: integer
  u: integer
  a: real
  b: real
  c: real
  d: positive real
verify:
  # [0] (S7.2) i.i.d.: ‖HIH‖_F² = n(1−1/n)² + n(n−1)(1/n)² = n−1
  - type: identity
    lhs: "n*(1-1/n)**2 + n*(n-1)*(1/n)**2"
    rhs: "n-1"
  # [1] (S7.3) 두 종: D = 4‖u‖⁴, ‖u‖² = n_B(1−p)² + (n−n_B)p², p=n_B/n  ⇒ 4n²p²(1−p)²
  - type: identity
    lhs: "4*(n_B*((n-n_B)/n)**2 + (n-n_B)*(n_B/n)**2)**2"
    rhs: "4*n**2*(n_B/n)**2*(1-n_B/n)**2"
  # [2] (S7.1) 일반식 ‖HκH‖² = trκ² − (2/n)1ᵀκ²1 + (1ᵀκ1)²/n², n=2 기호 κ=[[a,b],[b,c]]: 좌변은 직접 (a−2b+c)²/4
  - type: identity
    lhs: "(a-2*b+c)**2/4"
    rhs: "(a**2+2*b**2+c**2) - (a**2+2*b**2+c**2+2*a*b+2*b*c) + (a+2*b+c)**2/4"
  # [3] (S7.4) chain tr κ² = Σ_k k²(2(n−k)+1)
  - type: identity
    lhs: "Sum(k**2*(2*(n-k)+1),(k,1,n)).doit()"
    rhs: "n*(n+1)*(n**2+n+1)/6"
  # [4] (S7.4) chain 1ᵀκ1 = Σ_v r_v = Σ_v v(2n+1−v)/2 = Σ s_u² = n(n+1)(2n+1)/6
  - type: identity
    lhs: "Sum(v*(2*n+1-v)/2,(v,1,n)).doit()"
    rhs: "n*(n+1)*(2*n+1)/6"
  # [5] (S7.4) chain Σ_v r_v² (멱합으로)
  - type: identity
    lhs: "Sum((v*(2*n+1-v))**2/4,(v,1,n)).doit()"
    rhs: "((2*n+1)**2*(n*(n+1)*(2*n+1)/6) - 2*(2*n+1)*(n**2*(n+1)**2/4) + n*(n+1)*(2*n+1)*(3*n**2+3*n-1)/30)/4"
  # [6] (S7.4) chain D 조립 = (n²−1)(2n²+7)/180 (Sum 세 개로 직접)
  - type: identity
    lhs: "Sum(k**2*(2*(n-k)+1),(k,1,n)).doit() - (2/n)*Sum((v*(2*n+1-v))**2/4,(v,1,n)).doit() + (Sum(v*(2*n+1-v)/2,(v,1,n)).doit())**2/n**2"
    rhs: "(n**2-1)*(2*n**2+7)/180"
  # [7] (S7.5) star: ‖H − uuᵀ‖² = (n−1) − 2(1−1/n) + (1−1/n)² = n − 2 + 1/n²
  - type: identity
    lhs: "(n-1) - 2*(1-1/n) + (1-1/n)**2"
    rhs: "n - 2 + 1/n**2"
  # [8] (S7.6) 교차항 chain: Σ_u s_u(1−s_u/n), s_u=u  = (n²−1)/6
  - type: identity
    lhs: "Sum(u*(1-u/n),(u,1,n)).doit()"
    rhs: "(n**2-1)/6"
  # [9] (S7.6) 같은 값을 tr κ − 1ᵀκ1/n 으로: Σ_v v − (Σ v²)/n
  - type: identity
    lhs: "n*(n+1)/2 - (n*(n+1)*(2*n+1)/6)/n"
    rhs: "(n**2-1)/6"
  # [10] (S7.7) 혼합 커널 가법성 n=2 기호: ‖H(I+κ)H‖² = ‖H‖² + ‖HκH‖² + 2 tr(Hκ)
  - type: identity
    lhs: "4*(1/2 + (a-2*b+c)/4)**2"
    rhs: "1 + (a-2*b+c)**2/4 + 2*((a+c) - (a+2*b+c)/2)"
  # [11] (S5.2)(S5.4) Isserlis 쌍 축약의 2차원 표본: Var(ξᵀMξ) = 2‖M‖_F², M=[[a,b],[b,c]]
  - type: identity
    lhs: "3*a**2 + 3*c**2 + 2*a*c + 4*b**2 - (a+c)**2"
    rhs: "2*(a**2 + 2*b**2 + c**2)"
  # [12] (S8.3) chain: D/(n² depth²) → 1/90, depth = n−1
  - type: limit
    expr: "(n**2-1)*(2*n**2+7)/(180*n**2*(n-1)**2)"
    var: n
    point: oo
    expected: "1/90"
  # [13] (S8.3) chain γ=1: √D/n² → √10/30
  - type: limit
    expr: "sqrt((n**2-1)*(2*n**2+7)/180)/n**2"
    var: n
    point: oo
    expected: "sqrt(10)/30"
  # [14] (S8.4) star: D/n² → 0 (조건 (C) 밖, γ=−1/2)
  - type: limit
    expr: "(n-2+1/n**2)/n**2"
    var: n
    point: oo
    expected: 0
  # [15] (S6.4) i.i.d. 극한 √(n−1)/n → 0 (13.6 CLT 복원)
  - type: limit
    expr: "sqrt(n-1)/n"
    var: n
    point: oo
    expected: 0
  # [16] (S6.5) ε_★² = 2T_2/‖G_0‖² · δ⁴ = 2·60/12 δ⁴ = 10 δ⁴
  - type: numeric
    expr: "2*60/12 - 10"
    tol: 1.0e-12
  # [17] (S1.2) ‖G_0‖² = 3·2² = 12
  - type: numeric
    expr: "3*2**2 - 12"
    tol: 1.0e-12
  # [18] (S6.5) F-02 보정점 규약: n=2 i.i.d.(D=1)에서 ε̄_2 = ε_★/2, ε_★=√10 δ²
  - type: numeric
    expr: "(sqrt(10)*d**2/2)**2 - 10*d**4*(2-1)/2**2"
    tol: 1.0e-12
  # [19] (S7.3) 두 종 p=1/2: ε̄/ε_★ = √D/n = 1/2, n에 무관 (13.5 가우스-종 복원)
  - type: numeric
    expr: "sqrt(4*n**2*(1/2)**2*(1-1/2)**2)/n - 1/2"
    tol: 1.0e-12
  # [20] (S7.7) K2 사전등록 숫자의 출처: X(32) = 2E tr(Hκ)/√(31·E D_C(32))
  - type: numeric
    expr: "2*92.3847/sqrt(31*2008.0806) - 0.7406"
    tol: 1.0e-4
  # [21] (S1.4) 극분해 1차항: C=cI+δC_1 ⇒ Ω=(C_1−C_1ᵀ)/(2c), c=2 ⇒ 계수 1/4
  - type: numeric
    expr: "1/(2*2) - 1/4"
    tol: 1.0e-12
  # [22] (S9.2) 절단 상계 산술: RMS(δ)/RMS(δ/2) = 4(1+x/2)/(1+x/8) ⇒ 4(1+3x/8)+O(x²), x=cδ²; 비 4.0051 ⇒ x=0.0034, RMS 상대 x/2=0.17%
  - type: numeric
    expr: "(4.0051/4 - 1)*8/3 - 0.0034"
    tol: 2.0e-4
---

# Q-0008 attempt-06 — 사다리 3단 보조정리: 커널 법칙

기계 검사: 프론트매터 verify 블록(항등식 12·극한 4·수치 7)과
`verify/Q-0008/attempt-06/check_kernel.py` → `result.json`(씨앗 20260902; 모든 허용오차·격자·trial 수는
스크립트 상단에 실행 전 선언). 검사는 여섯 부분이다 — A 구조상수, B 등방성(SO(4) 켤레 등변성, 모든 δ),
C Wick 법칙(대리모형 10⁵ trial·물리 블록 2000 trial·공통난수 비), D 닫힌 식·교차항·일반식, E γ_her scope 표(보고),
F δ 3점 시험(보고). 사전등록 kill 스크립트(`check_modes.py`)는 실행하지 않았다.

## (S1) 설정과 선형화

$$ e_v = I_4 + \delta\,\xi_v,\qquad \xi_v\in\mathbb R^{4\times4}\cong\mathbb R^{16},\qquad E[\xi_v^a\xi_w^b]=\kappa_{vw}\,\delta^{ab}\quad(a,b=1..16) $$  (S1.1) 정의 — 가우스 등방 라벨, 공분산 $\kappa\otimes I_{16}$ (카드 scope 가정)

$$ T_v:=\Sigma(e_v),\quad \mathrm{tl\,gram}(T_v)=0,\qquad \Sigma_0:=\Sigma(I_4),\quad G_0:=\mathrm{gram}(\Sigma_0)=2I_3,\quad \lVert G_0\rVert_F^2=12 $$  (S1.2) 12.4 geometric triple은 정확 simple; $G_0$는 Q-0012 a1 정확값 (verify[17], 검사 A)

$$ C_v:=G(\Sigma_0,T_v),\qquad R_v:=\mathrm{polar}(C_v)\in SO(3),\qquad X_v:=R_vT_v,\qquad \mathrm{tl\,gram}(X_v)=0 $$  (S1.3) 13.2 polar 정렬(`optimal_internal_alignment`); simplicity는 켤레에 불변 (attempt-03 S1.6)

$$ \eta_v:=X_v-\Sigma_0=\delta\,L\xi_v+\delta^2Q(\xi_v,\xi_v)+O(\delta^3),\qquad L\xi=\Sigma'(\xi)+\Omega(\xi)\Sigma_0,\quad \Omega(\xi)=\tfrac{1}{2c}\bigl(C_1-C_1^{\top}\bigr),\ C_1=G(\Sigma_0,\Sigma'(\xi)),\ c=2 $$  (S1.4) $\delta$-급수 — $C_v=cI+\delta C_1+O(\delta^2)$의 극분해 1차항은 $\mathrm{skew}(C_1)/c$ (verify[21]; Q-0012 a1 해석식, 수치 Richardson과 1.1e−13 일치). 차수 $\delta^k$의 계수는 $\xi$의 $k$차 동차

$$ Y:=\sum_vX_v=n\Sigma_0+S,\quad S:=\sum_v\eta_v,\qquad \zeta_v:=\eta_v-\bar\eta,\qquad \tilde\xi:=H\xi\ (H=I-\tfrac1nJ),\qquad \zeta_v=\delta\,L\tilde\xi_v+\delta^2\widetilde Q_v+O(\delta^3) $$  (S1.5) 중심화 — $L$이 선형이므로 평균이 $L$을 통과한다($\widetilde Q_v:=Q(\xi_v,\xi_v)-\overline{Q}$)

## (S2) 2단 항등식 대입

$$ \mathrm{tl\,gram}(Y)=-\,n\sum_v\mathrm{tl\,gram}(\zeta_v) $$  (S2.1) E-20260902-008 (강한 형태: 기준점·정렬·가중·signature 무관, 모든 $\delta$) 인용

## (S3) 2차 전개

$$ \mathrm{gram}(\zeta_v)=\delta^2\,G(L\tilde\xi_v,L\tilde\xi_v)+2\delta^3\,\Sigma(L\tilde\xi_v,\widetilde Q_v)+O(\delta^4) $$  (S3.1) (S1.5)를 gram의 쌍선형성(attempt-03 S1.4)에 대입

$$ M_{ab}:=\mathrm{tl}\,\tfrac12\bigl[G(Le_a,Le_b)+G(Le_b,Le_a)\bigr],\qquad M_{ab}=M_{ba},\quad \mathrm{tr}\,M_{ab}=0 $$  (S3.2) 정의(Q-0012) — $\{e_a\}$는 $\mathbb R^{16}$의 표준 정규직교 기저

$$ \mathrm{tl\,gram}(\zeta_v)=\delta^2\sum_{a,b}\tilde\xi_v^a\tilde\xi_v^b\,M_{ab}+\delta^3\Psi_v+O(\delta^4),\qquad \Psi_v:\ \xi\text{의 3차 동차} $$  (S3.3) $L\tilde\xi=\sum_a\tilde\xi^aLe_a$를 (S3.1)에 넣고 tl(선형) 적용

$$ \mathrm{tl\,gram}(Y)=-\,n\bigl[\delta^2\Phi+\delta^3\Psi+\delta^4\Xi+O(\delta^5)\bigr],\qquad \Phi:=\sum_v\sum_{a,b}\tilde\xi_v^a\tilde\xi_v^b M_{ab},\quad \Psi:=\sum_v\Psi_v\ (3\text{차}),\quad \Xi\ (4\text{차}) $$  (S3.4) (S3.3)을 (S2.1)에 합산

## (S4) 등방성 보조정리 — 평균 바닥 없음

$$ E[\tilde\xi_v^a\tilde\xi_w^b]=(H\kappa H)_{vw}\,\delta^{ab}=:K_{vw}\,\delta^{ab} $$  (S4.1) (S1.1)의 양쪽에 $H$(대칭)를 곱함 — 중심화 라벨의 공분산은 $K\otimes I_{16}$

$$ E[\Phi]=\sum_vK_{vv}\sum_aM_{aa}=\mathrm{tr}(K)\,\sum_aM_{aa} $$  (S4.2) (S4.1)을 $\Phi$의 정의에 대입

$$ \Lambda\in SO(4):\quad \xi\mapsto\Lambda\xi\Lambda^{\top}\ \text{는 }\mathbb R^{16}\text{의 직교변환이고 }I_4\text{를 고정};\qquad \Sigma(\Lambda e\Lambda^{\top})=R_L(\Lambda)\,\rho_6(\Lambda)\,\Sigma(e) $$  (S4.3) 켤레 대칭 — frame 지표의 자기쌍대 (3,1) 회전 $R_L$과 성분 지표의 2-form 회전 $\rho_6$ (12.4 't Hooft 구성의 SO(4) 등변성)

$$ \rho_6^{-1}\Sigma_0=R_0\Sigma_0\ (R_0\in SO(3)),\quad G(\rho_6A,\rho_6B)=G(A,B)\ \Longrightarrow\ C\mapsto R_0\,C\,R_L^{\top},\quad \mathrm{polar}(R_0CR_L^{\top})=R_0\,\mathrm{polar}(C)\,R_L^{\top} $$  (S4.4) wedge는 $SO(4)$ 불변, $\rho_6$은 자기쌍대 부분공간을 보존($R_0=G(\rho_6^{-1}\Sigma_0,\Sigma_0)G_0^{-1}$), 극분해는 직교 켤레에 등변

$$ \boxed{\ \mathrm{gram}\bigl(Y(\Lambda\xi\Lambda^{\top})\bigr)=R_0\,\mathrm{gram}\bigl(Y(\xi)\bigr)R_0^{\top}\quad\text{모든 }\delta,\ n\ } $$  (S4.5) (S4.3)(S4.4)를 $Y=\sum_vX_v$에 적용 — 검사 B: $\delta=0.2$, $n=3$, 20회, 상대오차 $\le 1.2\times10^{-15}$

$$ S:=\sum_aM_{aa}\ \text{는 정규직교 기저 합이라 }\xi\mapsto\Lambda\xi\Lambda^{\top}\text{에 불변}\ \Longrightarrow\ S=R_0SR_0^{\top}\ \ \forall R_0\in SO(3) $$  (S4.6) (S4.5)의 $\delta^2$ 계수; $\rho_6|_{\rm SD}:SO(4)\to SO(3)$는 전사

$$ \boxed{\ \sum_aM_{aa}=0\ } $$  (S4.7) Schur — $SO(3)$ 불변 대칭 $3\times3$은 $\lambda I_3$뿐이고 $\mathrm{tr}S=0$ ⇒ $\lambda=0$; Q-0012 a1 정확 유리수 계산(16방향 합 정확 0)·검사 A(1.8e−15)와 일치

$$ E[\Phi]=0,\qquad\text{더 강하게}\quad E[\mathrm{tl\,gram}(Y)]=0\ \text{모든 }\delta\text{에서 정확} $$  (S4.8) (S4.2)(S4.7); 라벨 분포가 동시 켤레에 불변(가우스 $\kappa\otimes I_{16}$이 그 예)이면 (S4.5)로 $E[\mathrm{tl\,gram}Y]=R_0E[\mathrm{tl\,gram}Y]R_0^{\top}$ ⇒ 0 — 이것이 "traceless 2차 평균 0"의 정확한 내용이며 K5 통제의 근거. 비등방 라벨은 (S4.6)의 불변성이 깨져 $O(\delta^2)$ 바닥이 생긴다(Q-0013, S10)

## (S5) 2차 모멘트 — Wick

$$ E\lVert\Phi\rVert_F^2=\sum_{v,w}\sum_{a,b,c,d}E\bigl[\tilde\xi_v^a\tilde\xi_v^b\tilde\xi_w^c\tilde\xi_w^d\bigr]\,\langle M_{ab},M_{cd}\rangle $$  (S5.1) Frobenius 내적 전개

$$ E\bigl[\tilde\xi_v^a\tilde\xi_v^b\tilde\xi_w^c\tilde\xi_w^d\bigr]=K_{vv}K_{ww}\,\delta^{ab}\delta^{cd}+K_{vw}^2\bigl(\delta^{ac}\delta^{bd}+\delta^{ad}\delta^{bc}\bigr) $$  (S5.2) Isserlis(1a, cited) — 세 쌍 축약에 (S4.1) 대입 (verify[11]은 2차원 표본)

$$ \sum_{v,w}K_{vv}K_{ww}\sum_{a,c}\langle M_{aa},M_{cc}\rangle=(\mathrm{tr}K)^2\,\Bigl\lVert\sum_aM_{aa}\Bigr\rVert_F^2=0 $$  (S5.3) 첫 축약 — (S4.7)로 소멸 (등방성이 $(\mathrm{tr})^2$ 항을 죽인다)

$$ \sum_{v,w}K_{vw}^2\sum_{a,b}\bigl(\langle M_{ab},M_{ab}\rangle+\langle M_{ab},M_{ba}\rangle\bigr)=\lVert K\rVert_F^2\cdot 2T_2,\qquad T_2:=\sum_{a,b}\lVert M_{ab}\rVert_F^2=60 $$  (S5.4) 교차 축약 — $M_{ab}=M_{ba}$; $T_2$는 정확 구조상수(검사 A)

$$ \boxed{\ E\lVert\Phi\rVert_F^2=2T_2\,\lVert H\kappa H\rVert_F^2=120\,D_\kappa\ } $$  (S5.5) (S5.3)+(S5.4). 비가우스 라벨이면 여기에 첨도 항 $\kappa_4T_4S_{\rm gen}$이 더해진다(Q-0012; 가우스에서 0 — 언급만). 검사 C-a: 대리모형 $10^5$ trial, 12 사례

$$ E\lVert\mathrm{tl\,gram}Y\rVert_F^2=n^2\Bigl[\delta^4E\lVert\Phi\rVert^2+2\delta^5E\langle\Phi,\Psi\rangle+\delta^6E\bigl(\lVert\Psi\rVert^2+2\langle\Phi,\Xi\rangle\bigr)+O(\delta^7)\Bigr] $$  (S5.6) (S3.4)의 제곱 노름

$$ E\langle\Phi,\Psi\rangle=0\ \Longrightarrow\ E\lVert\mathrm{tl\,gram}Y\rVert_F^2=120\,n^2\delta^4D_\kappa\,\bigl(1+c_1\delta^2+O(\delta^4)\bigr),\qquad c_1:=\frac{E(\lVert\Psi\rVert^2+2\langle\Phi,\Xi\rangle)}{120\,D_\kappa} $$  (S5.7) 홀수 차수 소멸 — $\langle\Phi,\Psi\rangle$는 $\xi$의 5차 동차, 중심 가우스는 $\xi\mapsto-\xi$ 대칭. $c_1$은 $\kappa,n$에 의존하는 절단 상수(정의만; 크기는 S9)

## (S6) 분모와 조립

$$ \mathrm{gram}(Y)=n^2G_0+2n\,\Sigma(\Sigma_0,S)+\mathrm{gram}(S),\qquad S=\delta\,L\sum_v\xi_v+O(\delta^2) $$  (S6.1) attempt-03 (S5.1)

$$ \lVert\mathrm{gram}Y\rVert_F^2=n^4\lVert G_0\rVert_F^2\bigl(1+\delta\beta_1(\xi)+\delta^2\beta_2(\xi)+O(\delta^3)\bigr),\qquad \beta_1\ 1\text{차 동차},\ \beta_2\ 2\text{차} $$  (S6.2) (S6.1)의 제곱 노름

$$ \epsilon^2:=\frac{\lVert\mathrm{tl\,gram}Y\rVert_F^2}{\lVert\mathrm{gram}Y\rVert_F^2},\qquad E[\epsilon^2]=\frac{E\lVert\mathrm{tl\,gram}Y\rVert_F^2}{n^4\lVert G_0\rVert_F^2}\bigl(1+c_2\delta^2+O(\delta^4)\bigr) $$  (S6.3) $1/(1+x)$ 전개; $E[\lVert\mathrm{tl\,gram}Y\rVert^2\beta_1]$은 5차 홀수 ⇒ 0 ((S5.7)과 같은 대칭 논거); $c_2$는 $-E[\beta_2]+E[\beta_1^2]$ 형의 분모 보정 상수

$$ \boxed{\ \bar\epsilon^2:=E[\epsilon^2]=\frac{120\,n^2\delta^4D_\kappa}{12\,n^4}\bigl(1+c\,\delta^2\bigr)=\frac{10\,\delta^4}{n^2}\,\lVert H\kappa H\rVert_F^2\,\bigl(1+c\,\delta^2+O(\delta^4)\bigr),\qquad c=c_1+c_2\ } $$  (S6.4) (S5.7)(S6.3) 대입, $\lVert G_0\rVert^2=12$ (verify[17]) — 카드 3단의 커널 법칙. 검사 C-b·C-c: 물리 블록 $n\in\{2,4,8,16\}$, i.i.d.·chain·두 종

$$ \epsilon_\star^2=\frac{2T_2}{\lVert G_0\rVert_F^2}\,\delta^4=10\,\delta^4,\qquad n=2\ \text{i.i.d.}:\ D=1\ \Rightarrow\ \bar\epsilon_2=\epsilon_\star/2 $$  (S6.5) F-02 보정점 규약과 정합 (verify[16][18]); $\epsilon_\star$는 커널·트리에 무관한 순수 기하 상수이므로 her·i.i.d. 분기가 하나의 $\epsilon_\star$를 공유한다(E-015 hidden assumption 해소). Q-0013 물리 재측정 $3.16437\approx\sqrt{10}$은 관측 근접이지 근거가 아니다

$$ \text{복원: }\ \kappa=c^2J\ \text{또는 단일 종}\Rightarrow HJH=0\Rightarrow\bar\epsilon=0\ (13.3);\quad \kappa=I\Rightarrow\bar\epsilon=\epsilon_\star\sqrt{n-1}/n\to0\ (13.6);\quad p=\tfrac12\ \text{두 종}\Rightarrow\bar\epsilon=\epsilon_\star/2\ n\text{-무관}\ (13.5) $$  (S6.6) 카드 recovers 셋 (verify[15][19]; 두 종은 S7.3)

## (S7) 닫힌 식 4종과 교차항

$$ \lVert H\kappa H\rVert_F^2=\mathrm{tr}(H\kappa H\kappa)=\mathrm{tr}\,\kappa^2-\frac2n\,\mathbf 1^{\top}\kappa^2\mathbf 1+\frac1{n^2}\bigl(\mathbf 1^{\top}\kappa\mathbf 1\bigr)^2 $$  (S7.1) $H^2=H$, $H=I-J/n$ 전개 (verify[2]: $n=2$ 기호; 검사 D: 무작위 대칭 $\kappa$, $n=5$)

$$ \lVert HIH\rVert_F^2=\mathrm{tr}\,H=n-1 $$  (S7.2) i.i.d. — $H$는 계수 $n-1$ 사영 (verify[0])

$$ \kappa=\mathbf 1_B\mathbf 1_B^{\top}+\mathbf 1_C\mathbf 1_C^{\top},\quad u:=H\mathbf 1_B=-H\mathbf 1_C\ \Rightarrow\ H\kappa H=2uu^{\top},\quad \lVert u\rVert^2=n_B(1-p)^2+(n-n_B)p^2=np(1-p)\ \Rightarrow\ D=4\lVert u\rVert^4=4n^2p^2(1-p)^2 $$  (S7.3) 두 종 — $\mathbf 1_B+\mathbf 1_C=\mathbf 1$, $H\mathbf 1=0$ (verify[1]; 검사 D: $n\le11$, 모든 $n_B$)

$$ \kappa_{vw}=\min(v,w):\quad \mathrm{tr}\,\kappa^2=\sum_{k=1}^nk^2\bigl(2(n-k)+1\bigr),\quad r_v=\sum_w\kappa_{vw}=\frac{v(2n+1-v)}2,\quad \mathbf 1^{\top}\kappa\mathbf 1=\sum_vv^2\ \Longrightarrow\ D_{\rm chain}=\frac{(n^2-1)(2n^2+7)}{180} $$  (S7.4) chain(root 1, $|\mathrm{path}(v)\cap\mathrm{path}(w)|=\min(v,w)$) — (S7.1)에 대입, Sum 항등식 (verify[3][4][5][6]; 검사 D $n\le11$; $n=8$: 47.25)

$$ \kappa=J+(I-e_1e_1^{\top}),\quad u:=He_1,\ \lVert u\rVert^2=1-\tfrac1n\ \Rightarrow\ H\kappa H=H-uu^{\top},\quad D_{\rm star}=\mathrm{tr}H-2\lVert u\rVert^2+\lVert u\rVert^4=n-2+\frac1{n^2} $$  (S7.5) star(root 1 + $n-1$ 잎: $\kappa_{11}=1$, 잎 대각 2, 비대각 1) — $HJH=0$, $Hu=u$ (verify[7]; 검사 D)

$$ \mathrm{tr}(H\kappa)=\mathrm{tr}\,\kappa-\frac1n\mathbf 1^{\top}\kappa\mathbf 1,\qquad \kappa_{vv}=|\mathrm{path}(v)|\Rightarrow\mathrm{tr}\,\kappa=\sum_us_u,\qquad \mathbf 1^{\top}\kappa\mathbf 1=\lVert A^{\top}\mathbf 1\rVert^2=\sum_us_u^2\ \Longrightarrow\ \mathrm{tr}(H\kappa)=\sum_us_u\Bigl(1-\frac{s_u}n\Bigr) $$  (S7.6) 교차항 — $\kappa=AA^{\top}$, $A_{vu}=[u\preceq v]$, $s_u=|\mathrm{sub}(u)|$; 경로 길이 합은 부분트리 크기 합과 같다 (verify[8][9]: chain $(n^2-1)/6$; 검사 D: chain·star·Cayley 표본)

$$ \kappa_{\rm mix}=I+AA^{\top}\ \Longrightarrow\ D_{\rm mix}=\lVert H+H\kappa_{\rm her}H\rVert_F^2=(n-1)+D_{\rm her}+2\,\mathrm{tr}(H\kappa_{\rm her}) $$  (S7.7) K2 혼합 mode(독립 $\xi$ + 상속 $\zeta$) — $H$ 멱등, 교차항이 (S7.6) (verify[10]: $n=2$ 기호; verify[20]: 사전등록 $X(32)=0.7406$의 출처)

## (S8) 조건부 따름정리 — $\gamma_{\rm her}=1/d_{\rm tree}$

$$ \bar\epsilon(n)=\epsilon_\star\frac{\sqrt{D(n)}}{n}\ \Longrightarrow\ \gamma:=\frac{d\ln\bar\epsilon}{d\ln n}=\frac12\,\frac{d\ln\bigl(D/n^2\bigr)}{d\ln n} $$  (S8.1) (S6.4)의 정의

$$ \text{조건 (C):}\ \ \frac{D(n)}{n^2}\asymp\mathrm{depth}(n)^2\ \text{이고}\ \mathrm{depth}(n)\asymp n^{1/d_{\rm tree}}\quad\Longrightarrow\quad \gamma_{\rm her}=\frac1{d_{\rm tree}} $$  (S8.2) (S8.1)에 대입 — (C)의 내용: 중심화된 전형적 쌍의 공통조상 수 $(H\kappa H)_{vw}$의 RMS가 깊이 차수. (C)는 여기서 증명하지 않는다(카드 scope와 같은 조건부 진술)

$$ \text{chain: }\frac{D}{n^2(n-1)^2}\to\frac1{90}\ (\gamma=1,\ d_{\rm tree}=1);\quad \text{caterpillar }k\text{: }\frac{D}{n^2\mathrm{depth}^2}=0.01231,0.01131,0.01115,0.01112,0.01111\ (k=8..128)\to\frac1{90};\quad \text{Cayley: }\frac{E\,D_C(n)}{n^3}=0.0540,0.0585,0.0613,0.0630,0.0642,0.0649,0.0655,0.0658\ (n=8..1024) $$  (S8.3) 소속 — chain은 닫힌 식(verify[12][13]), caterpillar는 정확 driver(검사 E; b9의 $\gamma\to0.5000$과 정합), Cayley는 정확 조합론(depth$\asymp n^{1/2}$, 1d) ⇒ $D/n^2\asymp n\asymp\mathrm{depth}^2$; 격자 기울기 0.5302(K1 사전등록)

$$ \text{star-of-chains: }\frac{D}{n^2\mathrm{depth}}=0.183,0.176,0.172,0.169\ (k=8..64)\to\frac16\ \Rightarrow\ \frac{D}{n^2}\asymp\mathrm{depth}\ (\gamma=\tfrac14);\quad \text{balanced binary: depth}\asymp\log n;\quad \text{star: }\frac{D}{n^2}\to0\ (\gamma=-\tfrac12) $$  (S8.4) 제외 — (C)의 전제가 깨지는 세 족(각각 depth의 1차·멱법칙 아님·상수 depth; verify[14], 검사 E 격자 0.2416·0.2267·−0.4538). 반례가 아니라 (C) 밖

$$ d_{\rm tree}\text{를 공통조상 수 지수로 재정의하면 (S8.2)는 }D\approx n^2\bar\kappa^2\text{의 동어반복이 된다 — 채택하지 않는다} $$  (S8.5) 카드 scope의 대안 (b) 기각 유지

## (S9) $O(\delta^4)$ 절단의 상계

$$ \bar\epsilon^2=\frac{10\,\delta^4D}{n^2}\bigl(1+c\,\delta^2+O(\delta^4)\bigr)\ \Longrightarrow\ \frac{\mathrm{RMS}(\delta)}{\delta^2}=r_0\Bigl(1+\frac c2\delta^2+O(\delta^4)\Bigr) $$  (S9.1) (S6.4) — 홀수 차수가 (S5.7)(S6.3)의 대칭으로 소멸하므로 첫 보정은 $\delta^2$ 상대(즉 $\epsilon$에서 $O(\delta^4)$ 절대)

$$ \frac{\mathrm{RMS}(\delta)}{\mathrm{RMS}(\delta/2)}=4\,\frac{1+x/2}{1+x/8}=4\Bigl(1+\frac38x\Bigr)+O(x^2),\quad x:=c\,\delta^2;\qquad 4.0051\ (n=8),\ 4.0069\ (n=128)\ \Rightarrow\ x=0.0034,\ 0.0046\ \Rightarrow\ \frac x2=0.17\%,\ 0.23\% $$  (S9.2) attempt-04 adversary 공통난수 시험(seed 555013, 128 trial, $\delta=0.005$ 대 $0.0025$, Cayley 상속)의 재해석 (verify[22]) — trial당 비의 sd 0.027(SE 미보고)이라 비의 잡음이 $\approx0.003$이므로 이 값은 $\delta=0.005$에서 RMS 절단의 **상계** $\lesssim0.23\%$로만 읽는다

$$ \Delta\gamma_{\rm trunc}=\frac{\tau(n_2)-\tau(n_1)}{\ln(n_2/n_1)}:\qquad \text{(S9.2) 상계 }\frac{(0.0046-0.0034)/2}{\ln16}\approx2\times10^{-4};\qquad \text{(S9.5) 3점 시험 }\frac{(0.018-0.026)\%}{\ln4}\approx-6\times10^{-5}\ (\pm2\times10^{-4}) $$  (S9.3) 격자 기울기에 미치는 절단 영향 — K1 창 반폭 0.10의 0.2% 이하; E-015의 "≲1e−4"와 부합($\tau(n):=$ 크기 $n$에서의 RMS 상대 절단)

$$ \text{이 attempt의 }\delta\text{ 3점 시험(검사 F, 씨앗 20260902, 공통난수 512 trial, }\delta\in\{0.02,0.01,0.005\},\ n\in\{8,32\},\ \text{her}\cdot\text{iid): }\ \ln\frac{\mathrm{RMS}}{\delta^2}=\ln r_0+\frac c2\delta^2\ \text{적합} $$  (S9.4) 보고만(사전등록 아님, 판정 아님) — 결과와 결론 표는 아래 검증 요약. 7단 진폭비의 약 2% 계통 결손(E-016 P1-3)이 절단 항인지의 판정 기준은 스크립트에 선언: $|\tau|<0.5\%$면 "잡음"(절단 아님), $\tau\le-1\%$면 "O(δ⁴)", 그 밖은 "미결"

$$ \tau(0.005)=+0.026\pm0.014\%\ (\text{her }n{=}8),\quad +0.018\pm0.021\%\ (\text{her }n{=}32),\quad -0.006\pm0.004\%\ (\text{iid }n{=}8),\quad -0.006\pm0.003\%\ (\text{iid }n{=}32)\ \Longrightarrow\ \text{"잡음"} $$  (S9.5) 결과(부트스트랩 SE, B=2000) — 넷 모두 $|\tau|<0.5\%$; her에서 부호가 양(과잉)이라 −2% 결손을 만들 수 없고, iid의 음의 $\tau$는 실재하지만(≈2σ) 2%의 1/300이다. 7단 결손 −2%는 그 통계의 자체 SE 2.9%(E-016: 6.67±0.19)의 0.7σ, 즉 표본 요동과 부합 — 여기서는 판정하지 않고 보고만 한다

## (S10) 무엇을 주장하지 않는가

$$ \text{비등방 라벨: (S4.6)의 켤레 불변성이 깨져 }E[\mathrm{tl\,gram}Y]\ne0\ (O(\delta^2)\text{ 바닥, Q-0013 }\mathrm{tl}\,\mathcal G(\Sigma)\ne0) $$  (S10.1) 카드 scope 밖

$$ \text{비가우스 첨도: (S5.2)에 4차 누적항이 추가되어 }E\lVert\Phi\rVert^2=2T_2D+\kappa_4T_4S_{\rm gen}\ (\text{Q-0012 F-01, 별도 카드}) $$  (S10.2) 언급만

$$ \text{5단 반집중 }P[\bar\epsilon\le\epsilon_{\rm res}]\to0\text{은 2차 모멘트가 아니라 분포의 하한을 요구 — 이 단의 결과 아님} $$  (S10.3)

$$ \text{조건 (C)를 정리로, }\gamma_{\rm her}=1/d_{\rm tree}\text{를 모든 tree 족에 — 주장하지 않음; Lorentzian·10.10 counting measure 대응 — 카드 밖} $$  (S10.4)

$$ \text{닫힌 절단 상계 }|c|\le c_{\max}(\kappa,n)\text{는 유도하지 않았다 — (S9)는 구조(홀수 소멸, }\delta^2\text{ 상대)와 실측 상계뿐} $$  (S10.5) 정직 기록

## 검증 요약

산출물 `verify/Q-0008/attempt-06/result.json`(본실행 277 s, 재실행 비트 동일), `hook_result.json`, `log_check.txt`.
허용오차·격자·trial 수는 스크립트 상단에 실행 전 선언했고 결과를 본 뒤 바꾸지 않았다(F부에는 결과를 본 뒤
부트스트랩 SE **보고**만 추가, 기준·숫자 불변).

| 검사 | 내용 | 결과 |
|---|---|---|
| 훅 | verify 블록 항등식 12·극한 4·수치 7 | symbolic pass · numeric pass (23/23) |
| A | 구조상수 $T_2,T_4,\sum_aM_{aa},\lVert G_0\rVert^2$, 다중도 | 60(−9e−12)·2(+1e−15)·1.8e−15·12·{0:52, 1/8:96, 1/6:24, 1/2:72, 2/3:12}; $\epsilon_\star^2/\delta^4=10.000$ — 통과 |
| B | (S4.5) 등변성 $\mathrm{gram}\,Y(\Lambda\xi\Lambda^{\top})=R_0\,\mathrm{gram}\,Y(\xi)R_0^{\top}$, $\delta=0.2$, $n=3$, 20회 (tl/gram 0.027–0.114, $\delta^2$ 영역 훨씬 밖) | 최대 상대오차 1.2e−15 — 통과 |
| C-a | (S5.5) 대리모형 $E\lVert\Phi\rVert^2$ vs $120D$, i.i.d.·chain·두 종 × $n\in\{2,4,8,16\}$, $10^5$ trial | 최대 \|rel\| 0.58% (chain $n=8$, $z=-1.7$), 12건 모두 2% 창 안 — 통과 |
| C-b | (S6.4) 물리 블록 $E\lVert\mathrm{tl\,gram}Y\rVert^2$ vs $120\,n^2\delta^4D$, 2000 trial, 5% 창 | 10건 ≤4.0%; i.i.d. $n=2$ −5.58% ($z=-2.47$)·두 종 $n=8$ −5.82% ($z=-2.37$)가 창 밖 → 선언 기준 **미통과(2/12)**; 12건 합산 $z=-1.5$; $E[\epsilon^2]$ 대 $10\delta^4D/n^2$도 같은 값; 재추출 0회 |
| C-c | 공통난수 비 물리/대리모형 (같은 난수, trial별) | 평균 편차 최대 0.066%, 12건 모두 1% 창 안 — 통과; trial당 sd 0.8–3.3% ($O(\delta)$ 홀수항) |
| D | 닫힌 식 4종 $n\le11$ (i.i.d. 11·두 종 77·chain 11·star 11), 교차항 50 트리(chain·star·Cayley), 일반식 (S7.1) 무작위 대칭 κ 10 | 최대 상대오차 8.5e−16 — 통과 |
| E | scope 표 (S8.3)(S8.4); 같은 격자 γ | chain 0.994 · Cayley 0.5302 · caterpillar 0.4885 ($k=8..64$) · star-of-chains 0.2416 · binary 0.2267 · star −0.4538 — 보고 |
| F | δ 3점 시험 (S9.4)(S9.5) | 판정 라벨 "잡음": $\max|\tau|=0.026\%$ — 보고 |

C-b 해석: 창 밖 2건은 물리가 아니라 가우스 표본의 요동이다 — 같은 난수의 대리모형 평균이 Wick 값에서 같은 만큼
(−5.6%·−5.8%) 벗어나고 물리/대리 비는 $1\pm0.07\%$(C-c)이다. 2000 trial의 상대 SE 2.2–2.6%(CV 0.7–1.2, 우꼬리)에서
5% 창은 2σ 안팎이라 설계 시점(assumptions 마지막 항)에 예고한 대로 잡음 초과가 났다. Wick 대수는 C-a($10^5$ trial, 0.6%)가,
선형화·정렬·절단은 C-c(0.07%)가 닫는다; C-b 5% 기준은 선언대로 미통과로 기록하고 창을 바꾸지 않는다.

δ 3점 시험 (F, $r(\delta):=\mathrm{RMS}/\delta^2$, 공통난수 512 trial, 부트스트랩 SE):

| mode, $n$ | $r(0.02)/r(0.005)-1$ | $r(0.01)/r(0.005)-1$ | $c/2$ | $\tau(0.005)$ | trial당 비 sd |
|---|---|---|---|---|---|
| her, 8 | +0.39±0.22% | +0.08±0.07% | 10.4 | +0.026±0.014% | 5.3% |
| her, 32 | +0.25±0.33% | +0.00±0.11% | 7.1 | +0.018±0.021% | 7.1% |
| i.i.d., 8 | −0.10±0.06% | −0.03±0.02% | −2.6 | −0.006±0.004% | 1.9% |
| i.i.d., 32 | −0.09±0.05% | −0.03±0.02% | −2.3 | −0.006±0.003% | 1.6% |

$\delta^2$ 적합 잔차(9e−6…3e−4)는 $\delta^1$ 대안(1.5e−5…5e−4)과 3점으로는 구별되지 않는다 — 홀수 소멸은 (S5.7)의 대칭 논거가
정하고, 시험은 크기만 잰다. 실행하지 않은 것: `verify/Q-0008/F-02/check_modes.py`(사전등록 K1·K2·K3·K5 창, 6·7단은
E-015·E-016으로 이미 닫힘).
