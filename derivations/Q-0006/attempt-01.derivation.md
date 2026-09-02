---
question: Q-0006
attempt: 1
pivot_step: null
claim: "M1 작용(E54-A: 중력+V(χ) 스칼라 χ+무차원 KG 기준 스칼라 X^A 4개, μ_X²>0, 양의 internal metric δ_AB)에 대해 (1) M3 관계적 관측량 대수는 BFR 2016(arXiv:1306.1058) 일반 틀의 특수 사례로 도입할 수 있으나 BFHPR 2016(arXiv:1605.02573) 결과의 직접 이전은 아니다 — X^A는 미분동형 공변 스칼라(BFR partial-observable 조건 충족)이고, 단사 조건 det(∂_μX^A)≠0은 배경 의존이어서 상수-X̄ 평탄 배경(요청의 E58, 정본 E61-D형)에서는 det=0으로 실패하고 비퇴화 FLRW 국소 patch(요청의 E59, 정본 E62-A: X^0=T(t), X^i=βx^i)에서는 det=Ṫβ³=μ_X⁴ua³b³≠0 (uβ≠0)으로 국소 통과한다; BFHPR이 외부 기준장을 '유용하지 않다'고 한 것은 우주론 섭동론 목적에 상대적인 판단(기준장이 최종 게이지 불변식에 남음)이지 불가능 판정이 아니며, CE에서는 X^A가 물질이라 남는 것이 의도된 특징이다. (2) M2 이상항은 공변 BV/QME 정식화에서 AGW 1984 차원 정리(중력 이상항은 4k+2 차원 카이럴 장에만)와 hep-th/0404033의 4D 명시 문장으로 [정리: 문헌, 4D 보손 물질 조건부]에 둘 수 있으나 보손 물질 전용 명시 문장은 미확인이고, 36장 식 (36.7)의 정준 HDA 이상항과 BV 이상항의 동치·M1 field content에서의 QME 해결 가능성(재규격화 조건·EFT·경계항)은 미증명이다. (3) kill 1 부분(BFR 클래스 안, 배경 조건부·국소; BFHPR 특수 구성 밖), kill 2 부분(4D 명시 문장 있음, 보손 전용·BV 범위 미확인), kill 3 부분(E62 선형화계 Hadamard 상태 존재 미검증 — 무효 확정도 통과도 아님). 어느 kill도 발동하지 않았으나 셋 다 조건부다."
assumptions:
  - "문헌 사실은 sourcer 2026-09-02 확인 범위에 한정: BFHPR 2016 원문 인용문 1개('introducing external fields ... not useful ... because these fields would appear in the final gauge-invariant expressions'), BFR 2016 일반 틀(단사 X_Γ·공변 스칼라 4개·BV·QME=재규격화 조건), AGW 1984 차원 정리, hep-th/0404033의 4D 문장. 절·식·정리 번호는 미확보이며 본 유도는 그 번호를 만들어 쓰지 않는다"
  - "E54-A 작용을 그대로 쓴다(χ 유무는 대조에 영향 없음; E62-A는 χ=0). 요청의 'E58'·'E59' 라벨은 정본 QNB-E58-A(one-cell scalar)·QNB-E59-A(유한 cutoff)가 아니라 각각 E61-D형 상수-X̄ 평탄 배경과 E62-A 비퇴화 FLRW 배경을 뜻하는 것으로 읽는다(진전 원장 §2 표기와 동일). E78은 정본에 없다(진전 원장 §2 이력에 명시)"
  - "'클래스 안'의 정의: BFR 틀에서 관계적 관측량을 만들기 위해 요구하는 것은 (a) 장 배치의 국소 공변 스칼라 함수 X^A 4개, (b) 배경 해 근방에서 X_Γ: M→R⁴가 (국소) 단사, (c) 배경 둘레 섭동적 BV 구성(게이지 고정·QME 해결), (d) 선형화 게이지 고정계의 Hadamard 상태. 이 네 항목을 조건으로 채택하며, BFR 원문이 정확히 이 네 항목을 정리로 묶었는지는 확인하지 못했다"
  - "M2의 '이상항'을 (36.7)의 정준 HDA 이상항 𝒜에서 공변 BV 이상항(ghost number 1 국소 범함수의 BRST cohomology class)으로 바꿔 읽는 것은 재정식화(reformulate)이며, 두 이상항의 동치는 가정하지 않는다"
  - "verify 블록의 두 항등식은 문헌 판정이 아니라 본문 (S2.3)·(S4.1)의 산술 확인일 뿐이다(sympy 없음 → symbolic skipped, numeric만)"
symbols:
  mu: positive
  u: real
  a: positive
  b: real
  k: integer
verify:
  - type: identity
    lhs: "(mu*u)*(mu*a*b)**3"
    rhs: "mu**4*u*a**3*b**3"
  - type: identity
    lhs: "4*k+2"
    rhs: "4*(k+1)-2"
---

# Q-0006 attempt 1 — M2·M3 문헌 도입 판정: M1 작용 대 BFHPR/BFR 조건 대조

verify: symbolic skipped(sympy 없음), numeric은 위 두 산술 항등식만. artifacts: `verify/Q-0006/attempt-01/` (대조표 `class_check.md`, `hook_result.json`). 이 attempt는 수치 실험이 아니라 문헌 조건과 작용의 대조이므로 최고 등급은 L2를 넘지 않는다.

## (S0) 기호와 입력

$$ S=\int_M d^4x\sqrt{-g}\Big[\tfrac{M_P^2}{2}(R-2\Lambda)-\tfrac12(\nabla\chi)^2-V(\chi)-\tfrac{\mu_X^2}{2}\delta_{AB}\nabla X^A\!\cdot\!\nabla X^B\Big]+M_P^2\!\int_{\partial M}\!\epsilon\sqrt{|h|}K,\quad \mu_X^2>0 $$  (S0.1) E54-A1 그대로; $X^A$는 무차원 물질 스칼라(정본: "gauge-only label이 아니다")
$$ \mathcal O_f[X=\xi]=\int d^4y\sqrt{-g}\,f(X^A(y)-\xi^A)\,\mathcal O(y) $$  (S0.2) 36장 식 (36.4), M3 산출물 타입
$$ [\hat C[\xi],\hat C[\eta]]=i\hbar\hat C[[\xi,\eta]_{\rm HD}]+\mathcal A[\xi,\eta],\qquad \text{M2 요구: }\mathcal A=0\ \text{또는 조절 가능 소거} $$  (S0.3) 36장 식 (36.7), 정준 HDA 형태
$$ \text{배경 두 종: }\ \bar g=\eta,\ \bar X^A=\text{const}\ (\text{요청 E58; 정본 E61-D형}),\qquad ds^2=-dt^2+a^2\delta_{ij}dx^idx^j,\ X^0=T(t),\ X^i=\beta x^i\ (\text{요청 E59; 정본 E62-A}) $$  (S0.4) 라벨 대응(assumptions 2); E62-A2 무차원화 $u=\dot T/\mu_X,\ b=\beta/(\mu_Xa)$

## (S1) BFR 일반 틀의 관계적 관측량 조건 (문헌 항목화)

$$ \text{(C1) 공변성: } X^A[\Gamma]\ \text{는 장 배치 }\Gamma\text{의 국소 공변 스칼라 범함수 4개 (partial observables), } \phi^*X^A[\Gamma]=X^A[\phi^*\Gamma]\ \forall\phi\in\mathrm{Diff}(M) $$  (S1.1) BFR 2016: 관계적 관측량은 미분동형에 공변인 스칼라 4개로 만든다(sourcer 확인)
$$ \text{(C2) 단사성: } X_\Gamma:=(X^0,\dots,X^3)[\Gamma]:M\to\mathbb R^4\ \text{가 배경 해 }\bar\Gamma\ \text{근방에서 단사} \iff \det(\partial_\mu X^A)\neq0\ \text{(국소 형태)} $$  (S1.2) BFR: 단사 사상 $X_\Gamma$; 36장 M3 행의 "비퇴화 reference patch"와 같은 조건
$$ \text{(C3) 섭동적 BV: } \Gamma=\bar\Gamma+\varphi,\ S_{\rm BV}=S+\text{(ghost·antifield)},\ \text{게이지 고정 후 QME } \tfrac12\{S_{\rm BV},S_{\rm BV}\}_\hbar=i\hbar\Delta S_{\rm BV}\ \text{를 재규격화 조건으로 해결} $$  (S1.3) BFR·Rejzner 2011: QME 해결 = 이상항 class 자명 + 유한 재규격화 선택
$$ \text{(C4) Hadamard: 선형화 게이지 고정계의 2점 함수 }\omega_2\ \text{가 Hadamard 형식 (time-ordered product 구성의 입력)} $$  (S1.4) BFHPR이 FLRW 위에서 Hadamard 상태를 다룸(sourcer 확인); BFR 일반 틀의 표준 입력
$$ \text{(C5, BFHPR 특수) } X^0=X^0[\tilde\phi]\ (\text{inflaton}),\quad X^i=\text{비국소 Green 함수로 만든 공간 조화 좌표},\quad \text{외부 기준장 도입은 "not useful"} $$  (S1.5) BFHPR 2016 원문: FLRW 고대칭 때문에 동역학장만으로 비퇴화 좌표를 못 만들어 비국소 구성; Brown–Kuchař형 외부장은 최종 게이지 불변식에 남아 "유용하지 않다"

## (S2) M1의 $X^A$ 대조 (조건별)

$$ \text{(C1) } X^A\ \text{는 }S\text{의 동역학 스칼라장}:\ \phi^*X^A=X^A\circ\phi,\quad \delta_{AB}\ \text{는 내부 상수 metric}\ \Rightarrow\ \text{(C1) 충족(구성적 함수가 아니라 기본 장)} $$  (S2.1) E54-A1의 항 $\delta_{AB}\nabla X^A\!\cdot\!\nabla X^B$는 스칼라이므로 공변; 내부 $SO(4)$ 회전은 별도 전역 대칭(참고, 조건 아님)
$$ \bar X^A=\text{const}\ \Rightarrow\ \partial_\mu\bar X^A=0\ \Rightarrow\ \det(\partial_\mu X^A)\big|_{\bar\Gamma}=0\ \Rightarrow\ \text{(C2) 실패 (요청 E58)} $$  (S2.2) E61-D 배경; 정본도 "constant $X^A$는 reference Jacobian이 퇴화"라 명시(36장 E61 문단)
$$ X^0=T(t),\ X^i=\beta x^i\ \Rightarrow\ \partial_\mu X^A=\mathrm{diag}(\dot T,\beta,\beta,\beta)\ \Rightarrow\ \det(\partial_\mu X^A)=\dot T\beta^3=\mu_X^4\,u\,a^3b^3\neq0\iff u\beta\neq0\ \Rightarrow\ \text{(C2) 국소 통과 (요청 E59)} $$  (S2.3) E62-A1·A2 대입 ($\dot T=\mu_Xu$, $\beta=\mu_Xab$); verify 1이 이 산술을 확인. 단 E62-A는 local/noncompact chart이며 nonzero clock branch에 유한 과거 singularity bound가 있으므로 "국소 단사"까지만
$$ \text{(C3) M1 BV: 정본 E70-B\ldots J는 bounded jet·유한 CME witness까지;\ full M1 CME·QME·loop ST 미계산}\ \Rightarrow\ \text{(C3) 미검증 (실패 아님)} $$  (S2.4) 36장 E70 계열 문단의 "QME와 M2는 계산하지 않았다" 그대로
$$ \text{(C4) E62 선형화계: E63 frozen principal symbol }c^2=1,\ K,G>0\ (\text{선언 부문});\ E65\ c_V^2=1;\ E67\ \text{finite-time symplectic}\ \Rightarrow\ \text{쌍곡성 징후, Hadamard 상태 존재 증명 아님} $$  (S2.5) 필요조건 일부만; 게이지 고정 후 normally hyperbolic 여부·전역 쌍곡 patch·상태 구성은 없음
$$ \text{(C5) } X^A\ \text{는 BFHPR가 배제한 "external fields as reference coordinates like Brown–Kuchař"의 클래스에 정확히 든다} $$  (S2.6) 추가된 동역학 물질장을 기준 좌표로 씀 — 문헌이 말한 바로 그 경우
$$ \text{BFHPR "not useful"} = \text{목적 상대적(우주론 섭동론: inflaton·metric만으로 쓴 관측량을 원함)},\qquad \neq\ \text{"impossible"} $$  (S2.7) 원문 이유는 "these fields would appear in the final gauge-invariant expressions"이지 구성이 정의되지 않는다는 것이 아님; CE는 $X^A$를 물질로 선언했으므로 최종식에 남는 것이 의도된 특징
$$ \therefore\ X^A\in\text{BFR 클래스 (C1 충족, C2 배경 조건부·국소)},\qquad X^A\notin\text{BFHPR 특수 구성 (C5)},\qquad \text{C3·C4 미검증} $$  (S2.8) (S2.1)–(S2.7) 종합

## (S3) M3 도입의 지위와 적응에서 새로 증명할 것

$$ \text{M3 지위} = \text{"BFR 일반 틀의 특수 사례(적응 필요)"}\ \neq\ \text{"BFHPR 결과의 직접 이전"} $$  (S3.1) (S2.8): BFHPR의 배경(inflaton FLRW)·기준 좌표(비국소 조화)·관측량 대수(inflaton·metric 섭동)가 모두 M1과 다름
$$ \text{P1: } \delta\mathcal O_f=\int\sqrt{-g}\,\partial_Af\,(\delta X^A)\,\mathcal O+\ldots\ \text{— }X^A\text{ 요동이 dressed 관측량에 들어감; 대수 }\mathfrak A_{\rm rel}\ \text{에서 }X\text{-부문의 기여와 고정 배경 극한(M3 산출물)의 억제 조건} $$  (S3.2) BFHPR에는 없는 항: 기준장이 추가 자유도(4개)로 대수에 들어감; 억제가 $(E/\mu_X)$인지 $(\mu_X/M_P)$인지 미정
$$ \text{P2: 강결합 스케일 }\Lambda_{\rm sc}(\mu_X,M_P)\ \text{와 EFT 유효 영역 — }f\text{의 지지 스케일 }\ell\gg\Lambda_{\rm sc}^{-1}\text{에서만 (36.4) 유의미} $$  (S3.3) 36장: "physical strong-coupling scale ... 미완성"; $M_P/\mu_X=10$ power counting은 cutoff 도출이 아님
$$ \text{P3: E62 patch의 전역 쌍곡성 + 게이지 고정 선형화 연산자의 normal hyperbolicity + Hadamard 상태 존재(5 스칼라–metric 혼합계)} $$  (S3.4) (S2.5); 자유 KG의 Hadamard 존재 정리를 혼합계로 확장해야 함(출처 미확보)
$$ \text{P4: 국소 단사 → patch 덮개·gluing (36장 M3 행 "영역 포함", M5 영역), compact }\Sigma\ \text{대 noncompact rod }X^i=\beta x^i\ \text{충돌 해소} $$  (S3.5) E62-A 정의문 자체가 "전역 rod가 아니다"라고 못 박음
$$ \text{P5: dressing·edge/boundary sector 명세 (Donnelly–Giddings 장거리 dressing) — BFR 틀은 이를 자동 제공하지 않음} $$  (S3.6) 36장 M3 행의 네 항목 중 dressing·edge는 문헌 도입으로 채워지지 않음
$$ \text{P6: (C3) — M1 field content에서 QME 해결 (M2와 공유)} $$  (S3.7) (S2.4)

## (S4) M2 도입: 4D 보손 물질 미분동형 이상항 부재

$$ \text{AGW 1984: 순수 중력(미분동형) 이상항은 }d=4k+2\ \text{의 카이럴 장에서만};\qquad 4k+2=4(k+1)-2\ \Rightarrow\ 4\notin\{4k+2\} $$  (S4.1) 차원 정리; verify 2가 산술 확인. M1 field content = metric + 실스칼라 5개(카이럴 장 없음, 게이지장 없음)
$$ \text{hep-th/0404033 (4D): "there are no field-theoretic diffeomorphism anomalies"} $$  (S4.2) 명시 문장(sourcer 확인); 보손 물질 전용 문장은 미확인 — AGW로부터 함의될 뿐
$$ \text{Weyl(trace) 이상항은 존재하나 M1에 Weyl 대칭이 없으므로 QME의 게이지 이상항이 아님; Bardeen–Zumino형 이동은 미분동형 보존 쪽으로 항상 가능} $$  (S4.3) 혼동 차단: "4D에 이상항 없음"은 미분동형에 대한 것
$$ \text{BV 언어: 이상항 }\mathcal A_{\rm BV}\in H^1(s|d)\ \text{(ghost number 1 국소 범함수)};\ \text{(S4.1)–(S4.2)는 이 class의 비자명 후보가 없다는 주장의 근거이나, 그 cohomology 계산 출처는 미확보} $$  (S4.4) 정직 표기: 차원 정리(섭동적 이상항 분류)와 국소 BRST cohomology 정리는 같은 결론을 주지만 후자 문장을 확보하지 못함
$$ (36.7)\ \mathcal A_{\rm HD}\ \text{(정준 HDA)}\ \not\equiv\ \mathcal A_{\rm BV}\ \text{(공변 QME)}\ \text{— 동치는 별도 증명; 문헌 도입은 M2를 공변 BV/QME 형태로 재정식화하는 것} $$  (S4.5) pAQFT는 kinematical Hilbert 위의 제약 연산자를 만들지 않으므로 (36.7) 형태의 이상항을 직접 판정하지 않음
$$ \text{QME 해결 가능성} = \text{이상항 class 자명} + \text{유한 재규격화 선택의 존재(재규격화 조건)} — \text{후자는 M1 field content·GHY 경계·EFT truncation에서 미증명} $$  (S4.6) (S1.3): 문헌은 "이상항 없으면 해결 가능"의 틀만 주고 M1 계산은 없음
$$ \text{M2 지위 문구 초안: [정리: 문헌, 4D 보손 물질 조건부] — 단 QME 해결 가능성은 M1 field content에서 미증명(재규격화 조건, EFT)} $$  (S4.7) 요청 문구 채택, (S4.4)·(S4.5) 단서 부착

## (S5) kill 1·2·3 판정 (사전등록 노트 `_workspace/20260902-qftnext-M4_문헌도입_끼임instrument.md`)

$$ \text{kill 1 (}X^A\text{가 BFHPR/BFR 클래스 밖이면 무효): 부분 — BFR 클래스 안(C1 충족, C2 E62형 배경에서 국소 통과·E61형에서 실패), BFHPR 특수 구성 밖} $$  (S5.1) (S2.8); 발동 안 함, 단 "배경 조건부·국소"가 붙음
$$ \text{kill 2 (4D 보손 미분동형 BRST 이상항 부재 명시 문장 없으면 조건부도 불가): 부분 — 4D 명시 문장 있음(S4.2), 보손 전용·}H^1(s|d)\text{ 범위 문장 미확인(S4.4)} $$  (S5.2) 발동 안 함, 단 "조건부"의 조건에 (S4.4)·(S4.5) 포함
$$ \text{kill 3 (BFHPR Hadamard 구성이 E59에서 적용 조건 불충족이면 M3 무효): 부분 — 불충족 증거 없음, 충족 증명도 없음(S2.5, P3)} $$  (S5.3) 판정 보류; E62 선형화계의 Hadamard 존재가 결정 항목
$$ \text{종합: 어느 kill도 발동하지 않음; 셋 다 "부분"이므로 M2·M3는 [정리: 문헌]이 아니라 [정리: 문헌 틀, 조건부]로만 올릴 수 있음} $$  (S5.4)

## (S6) 36장 M2·M3 행 지위 문구 초안과 M4 전제 조건

$$ \text{M3 초안: [정리: 문헌 틀 — BFR 2016 일반 틀의 특수 사례, 적응 미완성] — }X^A\text{ 4개는 공변 partial observable이고 E62형 비퇴화 국소 배경에서 단사; }X\text{-동역학의 대수 기여·}\Lambda_{\rm sc}\text{ 아래 EFT·E62 Hadamard·patch gluing·dressing/edge는 미증명. BFHPR 직접 이전 아님} $$  (S6.1)
$$ \text{M2 초안: [정리: 문헌, 4D 보손 물질 조건부] — 공변 BV/QME에서 4D 미분동형 이상항 부재(AGW 1984 + hep-th/0404033; 보손 전용 문장 미확인). (36.7) HDA 이상항과의 동치·M1 QME 해결 가능성(재규격화 조건, EFT, GHY)은 미증명} $$  (S6.2)
$$ \text{M4 전제(가정으로 명시): (A1) E62형 국소 비퇴화 patch 고정, (A2) }\ell\gg\Lambda_{\rm sc}^{-1}\text{ EFT 영역, (A3) 선형화계 Hadamard 상태 }\omega\text{ 존재 가정, (A4) 관측량 대수 = BFR 관계적 대수의 }f\text{-국소화(dressing 명세 별도), (A5) 이상항 자명·QME 해결 가정} $$  (S6.3) 프론티어 M4(끼임 CP instrument, Q-0007)는 이 다섯을 가정으로 달고 진행하며, 통과해도 M2·M3 닫힘이 아님

## (S7) 가정과 증명하지 않은 것

- 증명하지 않은 것: BFR 원문의 정확한 정리 진술(번호), BFHPR 인용문의 절 번호, 보손 물질 전용 이상항 부재 문장, $H^1(s|d)$ 계산 출처, (36.7)↔QME 동치, E62 선형화계 Hadamard 존재, $\Lambda_{\rm sc}$, M1 QME 해결.
- 이 attempt의 진전 종류: 축소(M2·M3 도입 경로를 "직접 이전"에서 "특수 사례+적응 목록 P1–P6"으로 좁힘). 닫힘 아님.
