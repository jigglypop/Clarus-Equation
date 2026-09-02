---
question: Q-0008
attempt: 3
ladder_step: 2
claim: "polar-aligned 정확 simple cell $X_v=\\Sigma_0+\\eta_v$ ($v=1..n$)의 합 $Y=\\sum_v X_v$는 모든 $\\delta$에서
  정확히 $\\mathrm{tl}\\,\\mathrm{gram}(Y)=-n\\sum_v\\mathrm{tl}\\,\\mathrm{gram}(\\eta_v-\\bar\\eta)$를 만족한다(1차항 정확 상쇄).
  따름정리 (a) 두 종 결정론 $\\Delta$에서 $\\epsilon=p(1-p)\\|\\mathrm{tl\\,gram}\\,\\Delta\\|_F/\\|\\mathrm{gram}(\\Sigma_0+(1-p)\\Delta)\\|_F$로
  $n$에 무관(13.5 재현), (b) 한 cell 결함에서 $p(1-p)=(n-1)/n^2$ 희석."
assumptions:
  - "gram과 tl은 코드 정의를 따른다: $\\mathrm{gram}(B)_{ij}=B^i\\wedge B^j$(`plebanski_gram`), $\\mathrm{tl}M=M-\\frac{\\operatorname{tr}M}{3}I_3$; tl은 선형"
  - "짝수차 form의 wedge 가환성 $B\\wedge B'=B'\\wedge B$ — 따라서 $G(B',B)=G(B,B')^{\\top}$이고 대칭화 $\\Sigma(B,B')=\\frac12(G(B,B')+G(B',B))$는 값이 대칭행렬인 대칭 쌍선형 형식이며 $\\Sigma(B,B)=\\mathrm{gram}(B)$ (수치 확인 1.6e−16)"
  - "$\\Sigma_0$가 정확히 simple: $\\mathrm{tl\\,gram}(\\Sigma_0)=0$ (12.4 $\\Sigma(e)$ audit)"
  - "각 cell이 정확히 simple: $\\mathrm{tl\\,gram}(X_v)=0$ — geometric triple $\\Sigma(e)$와 polar alignment $R\\in SO(3)$의 합성에서 유지된다((S1.6); 정렬 후 cell 잔차 최대 1.2e−15)"
  - "블록은 정렬 뒤 단순 합 $Y=\\sum_v X_v$(가중치 없음, 13.2 규약)이고 $n$은 유한하다"
  - "따름정리 (a)는 label 값이 정확히 두 개 $\\{0,\\Delta\\}$이고 $pn$이 정수일 때, (b)는 결함 cell이 정확히 1개일 때"
  - "$\\delta$ 크기·signature(Euclid/Lorentz)·label 분포·tree 구조에 대한 가정은 쓰지 않는다(전개도 절단도 없다)"
symbols:
  n: positive integer
  n_B: positive integer
  p: nonnegative real
  v: integer
  x: real
  y: real
  z: real
  a_tot: real
  q_tot: real
verify:
  # [0] (S5.2) 중심화 항등식, n=3에서 완전 검사 (스칼라 쌍선형 형식은 대칭텐서에 대해 충실한 시험이다)
  - type: identity
    lhs: "(x - (x+y+z)/3)**2 + (y - (x+y+z)/3)**2 + (z - (x+y+z)/3)**2"
    rhs: "x**2 + y**2 + z**2 - (x+y+z)**2/3"
  # [1] (S5.2) 일반 n: label족 eta_v = v 에서 sum eta^2 - (sum eta)^2/n = n(n^2-1)/12
  - type: identity
    lhs: "Sum(v**2,(v,1,n)) - (n*(n+1)/2)**2/n"
    rhs: "n*(n**2-1)/12"
  # [2] (S5.1) Y = n*Sigma_0 + S 의 이차 전개 (tl gram(Sigma_0)=0 대입 전)
  - type: identity
    lhs: "(n*x + y)**2"
    rhs: "n**2*x**2 + 2*n*x*y + y**2"
  # [3] (S5.3) 최종 조립: a_tot = sum_v tl gram(eta_v), q_tot = tl gram(S)
  - type: identity
    lhs: "-n*a_tot + q_tot"
    rhs: "-n*(a_tot - q_tot/n)"
  # [4] (S6.1) 두 종 결정론 label의 중심화 제곱합 = n p(1-p)
  - type: identity
    lhs: "n*p*(1-p)**2 + n*(1-p)*p**2"
    rhs: "n*p*(1-p)"
  # [5] (S6.1) 같은 항등식의 n_B 형 (카드 F-02 verify[6]와 동일한 식)
  - type: identity
    lhs: "n_B*((n-n_B)/n)**2 + (n-n_B)*(n_B/n)**2"
    rhs: "n_B*(n-n_B)/n"
  # [6] (S7.1) 결함 하나: p(1-p) 에 p=1/n 대입
  - type: identity
    lhs: "(1/n)*(1-1/n)"
    rhs: "(n-1)/n**2"
  # [7] (S7.2) 분자 법칙의 격자 비 (카드 K4 숫자 0.140625)
  - type: numeric
    expr: "((64-1)/64**2)/((8-1)/8**2) - 0.140625"
    tol: 1.0e-12
  # [8] 복원: 단일 종(p->1)에서 중심화 제곱합이 0 — 13.3 common orbit
  - type: limit
    expr: "p*(1-p)"
    var: p
    point: 1
    expected: 0
  # [9] (S7.2) 결함 희석의 점근: (n-1)/n^2 ~ 1/n
  - type: limit
    expr: "n*(n-1)/n**2"
    var: n
    point: oo
    expected: 1
  # [10] 복원: n=1 이면 eta_1 - etabar = 0 이므로 우변이 정확히 0 (12.4 단일 cell)
  - type: identity
    lhs: "1*(x - x/1)**2"
    rhs: "0"
---

# Q-0008 attempt-03 — 사다리 2단 보조정리: 정확 블록 항등식

기계 검사: 프론트매터 verify 블록(항등식 8·수치 1·극한 2)과
`verify/Q-0008/attempt-03/check_identity.py` → `result.json`(씨앗 20260902, TOL_IDENT=1e−12 선선언).
검사는 세 층이다. (A) 기호층은 임의의 대칭 쌍선형 형식에 대한 자유 모형에서 정리를 $n=2,3,5,8,17$에
대해 다항식 항등식으로 확인하고, **cell simplicity 대입을 빼면 잔차가 0이 아님**도 함께 확인한다(전제가
쓰이는 유일한 자리). (B) 수치층은 실제 tetrad·polar alignment에서 $n\in\{2,5,17\}$, $\delta\in\{0.3,0.05\}$,
20 trial씩 최대 상대오차 4.12e−13을 준다. (C) 따름정리층은 두 종·결함을 정확 값으로 대조한다.
사전등록 kill 스크립트(`check_modes.py`, 6·7단)와 K4 창은 실행하지 않았다.

## (S1) 기호와 전제

$$ G(B,B')_{ij}:=B^i\wedge B'^j,\qquad \mathrm{gram}(B):=G(B,B),\qquad \mathrm{tl}\,M:=M-\frac{\operatorname{tr}M}{3}I_3 $$  (S1.1) 정의 (`plebanski_gram`·`wedge_scalar`와 같은 식)

$$ B\wedge B'=B'\wedge B\ \Longrightarrow\ G(B',B)_{ij}=B'^i\wedge B^j=B^j\wedge B'^i=G(B,B')_{ji} $$  (S1.2) 짝수차 form의 wedge 가환성 (수치 1.6e−16)

$$ \Sigma(B,B'):=\tfrac12\bigl(G(B,B')+G(B',B)\bigr)\ \Longrightarrow\ \Sigma(B,B')=\Sigma(B',B)=\Sigma(B,B')^{\top},\qquad \Sigma(B,B)=\mathrm{gram}(B) $$  (S1.3) (S1.2)로 대칭화 — 이후 모든 계산은 값이 대칭행렬인 대칭 쌍선형 형식 $\Sigma$ 하나로 한다

$$ \mathrm{gram}(aB+a'B')=a^2\,\mathrm{gram}(B)+2aa'\,\Sigma(B,B')+a'^2\,\mathrm{gram}(B') $$  (S1.4) $\Sigma$의 쌍선형성 (수치 2.7e−16)

$$ X_v=\Sigma_0+\eta_v\ (v=1..n),\qquad Y:=\sum_{v}X_v,\qquad S:=\sum_v\eta_v=n\bar\eta,\qquad \mathrm{tl\,gram}(\Sigma_0)=0,\qquad \mathrm{tl\,gram}(X_v)=0 $$  (S1.5) 전제 — 정확 simple reference와 정확 simple cell

$$ R\in SO(3):\ \mathrm{gram}(RB)=R\,\mathrm{gram}(B)\,R^{\top},\quad \mathrm{tl}(RMR^{\top})=R\,(\mathrm{tl}M)\,R^{\top}\ \Longrightarrow\ \mathrm{tl\,gram}(B)=0\Rightarrow\mathrm{tl\,gram}(RB)=0 $$  (S1.6) trace가 켤레에 불변이므로 13.2의 polar alignment는 정확 simplicity를 보존한다 — (S1.5)는 정렬 뒤에도 참 (정렬 후 cell 잔차 최대 1.2e−15)

## (S2) 블록 gram의 이중합

$$ \mathrm{gram}(Y)=\Sigma\Bigl(\sum_vX_v,\sum_wX_w\Bigr)=\sum_{v,w}\Sigma(X_v,X_w) $$  (S2) (S1.3)(S1.4)의 쌍선형성으로 이중합 전개

## (S3) 항별 전개

$$ \Sigma(X_v,X_w)=\mathrm{gram}(\Sigma_0)+\Sigma(\Sigma_0,\eta_w)+\Sigma(\eta_v,\Sigma_0)+\Sigma(\eta_v,\eta_w) $$  (S3) $X_v=\Sigma_0+\eta_v$ 대입

## (S4) cell simplicity가 1차항을 고정한다

$$ 0=\mathrm{tl\,gram}(X_v)=\underbrace{\mathrm{tl\,gram}(\Sigma_0)}_{=0}+2\,\mathrm{tl}\,\Sigma(\Sigma_0,\eta_v)+\mathrm{tl\,gram}(\eta_v) $$  (S4.1) (S1.5)를 (S1.4)에 대입 (b8 증명의 둘째 줄)

$$ 2\,\mathrm{tl}\,\Sigma(\Sigma_0,\eta_v)=-\,\mathrm{tl\,gram}(\eta_v) $$  (S4.2) (S4.1) 이항 — 1차항이 2차항으로 **정확히** 표현된다(근사 아님)

$$ 2\,\mathrm{tl}\,\Sigma(\Sigma_0,S)=-\sum_v\mathrm{tl\,gram}(\eta_v) $$  (S4.3) (S4.2)를 $v$에 대해 합산 ($\Sigma$의 선형성)

## (S5) 합산과 중심화

$$ \mathrm{tl\,gram}(Y)=n^2\underbrace{\mathrm{tl\,gram}(\Sigma_0)}_{=0}+2n\,\mathrm{tl}\,\Sigma(\Sigma_0,S)+\mathrm{tl\,gram}(S) $$  (S5.1) $Y=n\Sigma_0+S$에 (S1.4)를 적용하고 tl의 선형성 사용

$$ \mathrm{tl\,gram}(Y)=-n\sum_v\mathrm{tl\,gram}(\eta_v)+\mathrm{tl\,gram}(S) $$  (S5.2) (S4.3) 대입

$$ \sum_v\mathrm{gram}(\eta_v-\bar\eta)=\sum_v\mathrm{gram}(\eta_v)-2\,\Sigma(S,\bar\eta)+n\,\mathrm{gram}(\bar\eta)=\sum_v\mathrm{gram}(\eta_v)-\frac1n\mathrm{gram}(S) $$  (S5.3) 중심화 항등식 — $S=n\bar\eta$로 $2\Sigma(S,\bar\eta)=2n\,\mathrm{gram}(\bar\eta)$, $n\,\mathrm{gram}(\bar\eta)=\mathrm{gram}(S)/n$ (verify[0][1], 기호층 $n\le17$)

$$ \boxed{\ \mathrm{tl\,gram}(Y)=-\,n\sum_v\mathrm{tl\,gram}(\eta_v-\bar\eta)\ } $$  (S5.4) (S5.3)에 $-n$을 곱해 (S5.2)와 대조 — 모든 $\delta$에서 정확 (verify[3]; 수치 최대 상대오차 4.12e−13, $\delta=0.3$에서 7.4e−15)

## (S6) 따름정리 (a): 두 종 결정론 $\Delta$ — 13.5의 $n$-무관성

$$ \eta_v=0\ (v\le pn),\qquad \eta_v=\Delta\ (v>pn)\qquad\Longrightarrow\qquad \bar\eta=(1-p)\Delta $$  (S6.1) 종 B(label 0)의 비율을 $p$로 두고 대입

$$ \sum_v\mathrm{gram}(\eta_v-\bar\eta)=pn\,(1-p)^2\,\mathrm{gram}(\Delta)+(1-p)n\,p^2\,\mathrm{gram}(\Delta)=n\,p(1-p)\,\mathrm{gram}(\Delta) $$  (S6.2) $\mathrm{gram}(c\Delta)=c^2\mathrm{gram}(\Delta)$와 $p+(1-p)=1$ (verify[4][5])

$$ \mathrm{tl\,gram}(Y)=-\,n^2p(1-p)\,\mathrm{tl\,gram}(\Delta),\qquad Y=n\bigl(\Sigma_0+(1-p)\Delta\bigr)\ \Longrightarrow\ \mathrm{gram}(Y)=n^2\,\mathrm{gram}\bigl(\Sigma_0+(1-p)\Delta\bigr) $$  (S6.3) (S5.4)에 (S6.2) 대입, 분모는 $Y$의 동차성

$$ \epsilon:=\frac{\|\mathrm{tl\,gram}(Y)\|_F}{\|\mathrm{gram}(Y)\|_F}=p(1-p)\,\frac{\|\mathrm{tl\,gram}\,\Delta\|_F}{\|\mathrm{gram}(\Sigma_0+(1-p)\Delta)\|_F} $$  (S6.4) $n^2$ 상쇄 — $\epsilon$은 $n$에 무관 (수치: $p=1/2$·$1/4$, $n=4..64$에서 상대 퍼짐 0.0, 예측 대조 상대오차 2.4e−15)

## (S7) 따름정리 (b): 결함 하나의 $(n-1)/n^2$ 희석

$$ n-1\ \text{cell이}\ \eta=0,\ \text{한 cell이}\ \eta=\Delta\ \Longrightarrow\ p=\frac{n-1}{n},\quad 1-p=\frac1n,\quad p(1-p)=\frac{n-1}{n^2} $$  (S7.1) (S6.4)에 결함 비율 대입 (verify[6]; $p\leftrightarrow1-p$ 대칭이므로 $p=1/n$로 써도 같다)

$$ \epsilon(n)=\frac{n-1}{n^2}\cdot\frac{\|\mathrm{tl\,gram}\,\Delta\|_F}{\|\mathrm{gram}(\Sigma_0+\Delta/n)\|_F},\qquad \frac{(n-1)/n^2\big|_{n=64}}{(n-1)/n^2\big|_{n=8}}=\frac{63/4096}{7/64}=0.140625 $$  (S7.2) 분자는 정확한 조합 인자, 분모는 $1+O(\|\Delta\|/n)$ 보정 (verify[7][9]; 수치 대조 상대오차 1.1e−14, 이 표본에서 분모 표류 최대 19.1%)

## (S8) 무엇에 의존하지 않는가

$$ \text{(S5.4)의 유도에 쓰인 것: }\Sigma\text{의 쌍선형·대칭성((S1.3)(S1.4)), tl의 선형성, 전제 }\mathrm{tl\,gram}(\Sigma_0)=\mathrm{tl\,gram}(X_v)=0\text{ 뿐} $$  (S8.1) 사용된 성질의 완전 목록

$$ \delta\text{ 크기·}\|\eta_v\|\text{에 대한 전개나 절단이 없다:\ (S5.4)는 }O(\delta^2)\text{ 근사가 아니라 항등식이다} $$  (S8.2) $\delta=0.3$에서도 상대오차 7.4e−15 (수치층)

$$ \text{signature(Euclid/Lorentz)·label 분포(가우스·결정론·Rademacher)·라벨 상관구조(tree)·}n\text{에 무관} $$  (S8.3) (S8.1)에 그 정보가 들어가지 않음 — 기호층이 자유 대칭형식으로 $n=2,3,5,8,17$에서 확인

$$ \text{전제를 빼면 거짓: cell simplicity 대입 없이 같은 계산을 하면 잔차 }2n\sum_v\bigl(u_v+\tfrac12\mathrm{tl\,gram}(\eta_v)\bigr)+n^2\,\mathrm{tl\,gram}(\Sigma_0)\neq0 $$  (S8.4) 기호층 `cell_simplicity_is_necessary` ($n=2,3,5$에서 잔차 비영) — 전제가 쓰이는 유일한 자리는 (S4.1)

## (S9) 무엇을 주장하지 않는가

$$ \text{잔차의 분포·기대값(3단 커널 법칙 }\bar\epsilon^2=\epsilon_\star^2\|H\kappa H\|_F^2/n^2)\text{은 (S5.4)에서 따라오지 않는다} $$  (S9.1) 그 단계는 $P_{\rm micro}$ 등방성 가정·1a Wick·$O(\delta^4)$ 절단을 추가로 쓴다

$$ \text{(S5.4)는 }\|\mathrm{tl\,gram}(Y)\|\text{의 상계를 주지 않는다: }\eta_v\text{가 정렬되면 우변은 }n^2\text{ 차수로 커진다} $$  (S9.2) 항등식은 크기가 아니라 구조(중심화)를 고정한다

$$ \text{cell이 정확히 simple하지 않으면}\ \tau_v:=\mathrm{tl\,gram}(X_v)\neq0\ \Longrightarrow\ \mathrm{tl\,gram}(Y)=n\sum_v\tau_v-n\sum_v\mathrm{tl\,gram}(\eta_v-\bar\eta) $$  (S9.3) (S4.1)에 $\tau_v$를 남긴 일반형 — 전제 위반의 정확한 대가 (수치: 비-simple 일반 triple, $\|\tau_v\|$가 $O(1)$일 때 상대오차 2.4e−15)

$$ \text{$\epsilon_\star$·분기 상수·}\gamma_{\rm her}\text{·}\Omega_{\rm align}/\Omega_{\rm mis}\text{ 흐름은 이 단의 결과가 아니다} $$  (S9.4) 사다리 3·5·7단의 몫

## 검증 요약

| 층 | 내용 | 결과 |
|---|---|---|
| 기호 (자유 대칭형식) | (S5.4) 다항식 항등식, $n=2,3,5,8,17$ | 5/5 잔차 0 |
| 기호 | 전제 없이는 잔차 비영, $n=2,3,5$ | 3/3 비영 (전제 필수) |
| 기호 | (S5.3) 중심화, $n=2,3,5,8,17$ + 일반 $n$ (라벨족 $\eta_v=v$) | 통과 |
| 수치 (물리) | (S5.4), $n\in\{2,5,17\}$, $\delta\in\{0.3,0.05\}$, 20 trial | 최대 상대오차 4.12e−13 |
| 수치 (물리) | 정렬 후 cell 정확 simplicity | 최대 1.2e−15 |
| 수치 | (S9.3) 비-simple 일반형 | 최대 2.4e−15 |
| 수치 | (S6.4) 두 종, $p\in\{1/2,1/4\}$, $n=4..64$ | 상대오차 2.4e−15, $n$-퍼짐 0.0 |
| 수치 | (S7.2) 결함 하나, $n=4..64$ | 상대오차 1.1e−14, 분자비 0.140625 |

산출물: `verify/Q-0008/attempt-03/result.json`. 실행하지 않은 것: `verify/Q-0008/F-02/check_modes.py`
(사전등록 K1·K2·K3·K5와 K4 일관성 창 — 사다리 6·7단).

## 사다리 1단(외부기존) 인용 목록

| 번호 | 쓰임 | 서지 |
|---|---|---|
| 1a | 가우스 2차 형식의 분산 $\mathrm{Var}(\xi^{\top}M\xi)=2\|M\|_F^2$ (3단) | L. Isserlis, "On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables", *Biometrika* **12** (1918) 134–139, DOI 10.1093/biomet/12.1-2.134. 물리 표기: G. C. Wick, "The evaluation of the collision matrix", *Phys. Rev.* **80** (1950) 268–272, DOI 10.1103/PhysRev.80.268 |
| 1b | i.i.d. 블록의 $n^{-1/2}$ (3단 복원·K5 통제) | Lindeberg–Lévy 중심극한정리. P. Billingsley, *Probability and Measure*, 3rd ed., Wiley 1995, Thm 27.1; W. Feller, *An Introduction to Probability Theory and Its Applications* II, 2nd ed., Wiley 1971, §VIII.4 |
| 1c | 균등 rooted labelled(Cayley) tree = Poisson(1) 임계 GW를 전체 크기 $n$으로 조건화 (K1 격자의 정의) | D. P. Kennedy, "The Galton–Watson process conditioned on the total progeny", *J. Appl. Probab.* **12** (1975) 800–806, DOI 10.2307/3212735; J. Pitman, *Combinatorial Stochastic Processes* (Saint-Flour XXXII–2002), Springer LNM **1875** (2006), §6.1–6.2; D. Aldous, "The continuum random tree II: an overview", in *Stochastic Analysis* (Durham 1990), LMS Lecture Note Ser. **167**, CUP 1991, 23–70 |
| 1d | 깊이 $\sim n^{1/2}$, $d_{\rm tree}=2$ (3단 $\gamma_{\rm her}=1/d_{\rm tree}$의 지수 입력) | D. Aldous, "The continuum random tree I", *Ann. Probab.* **19** (1991) 1–28, DOI 10.1214/aop/1176990534; "The continuum random tree III", *Ann. Probab.* **21** (1993) 248–289, DOI 10.1214/aop/1176989404; 유한 상수는 A. Rényi, G. Szekeres, "On the height of trees", *J. Austral. Math. Soc.* **7** (1967) 497–507, DOI 10.1017/S1446788700004432 ($E[\text{height}]\sim\sqrt{2\pi n}$) |
| 1e | 부분트리 크기 법칙 $N_k=\binom nk k^{k-1}(n-k)^{n-k}/n^{n-1}$ (K1 정확 조합론) | A. Meir, J. W. Moon, "The distance between points in random trees", *J. Combin. Theory* **8** (1970) 99–103, DOI 10.1016/S0021-9800(70)80012-1; 관련 A. Meir, J. W. Moon, "On the altitude of nodes in random trees", *Canad. J. Math.* **30** (1978) 997–1015, DOI 10.4153/CJM-1978-085-0. **주의(sourcer 확인 필요)**: 이 계수식 자체는 rooted labelled forest 수(Cayley 1889; A. Rényi, "Some remarks on the theory of trees", *Publ. Math. Inst. Hungar. Acad. Sci.* **4** (1959) 73–85)에서 직접 나오며, 이 저장소에서는 `verify/Q-0008/F-02/driver_numbers.py`가 $n\le7$ 전수열거와 2e−13 일치로 독립 확인했다 — 정확한 문헌 위치(1970 vs 1978)는 sourcer가 확정한다 |
| 1f | size-biased GW spine (4단 Q-spine 블록 구조) | R. Lyons, R. Pemantle, Y. Peres, "Conceptual proofs of $L\log L$ criteria for mean behavior of branching processes", *Ann. Probab.* **23** (1995) 1125–1138, DOI 10.1214/aop/1176988176 |

1a·1c·1d·1f는 표준 결과를 그대로 쓰며 이 저장소가 재유도하지 않는다. 1b는 교과서 정리다.
1e만 서지 위치 확정이 남아 있고, 쓰이는 식은 저장소 안에서 전수열거로 독립 확인되어 있다.
