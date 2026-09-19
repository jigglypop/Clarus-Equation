# CE-TH1: 에너지 보존 환경에서 미시적 상태기원과 거시 열역학을 잇는 조건부 증명

2026-09-19. 기준 main: `cbbc1cd930eb6c7ff42e0437aca261334f2a271d` (CE-CS1).
관측 피팅·계수 최적화 없음. 새로운 원시 상수의 수치 선택 없음.

## 0. 결과와 전제의 구분

CE-CS1은 집단 환경에서 기존의 24상태 평균을 동적으로 얻었지만, 그 환경의 준비/구동 에너지와 유한온도 detailed balance는 별도로 남겼다. 본 문서는 그것을 조용히 교체하지 않고 **열적·에너지 분해 충돌이라는 새 환경 가정**을 추가해 검사한다.

완료한 것은 다음이다.

- 매 충돌의 계+환경 Hamiltonian이 정확히 에너지를 보존하는 구현.
- 그 충돌의 연속극한 생성자, 보존 SWAP 가중치에서의 유일한 정상상태.
- 동일한 미시 분배함수에서 사건확률, 에너지분율, 자유에너지, 압력 및 엔트로피 생산의 연결.
- 비보존적 국소 열결합을 추가할 때의 유일 Gibbs 상태와 가중치 선택식.
- 기존 수치관계의 정확한 회수 한계와, 유한온도에서 달라지는 정도의 계산.

**실제 자연의 PMNS/힉스/뮤온/바리온량/진공값을 여기서 도출하지 않았다.** 온도와 진단 내부질량은 외부 조건이고, 관측값을 맞추기 위해 역산하지 않았다. CS1은 비열적 준비모형으로 여전히 가능하다. 여기의 유한온도 보정이 모든 CE 후보에 필수라는 주장은 아니다.

## 1. 보존한 입력

기존 alpha_s=0.11789에서 p=4 alpha_s^(4/3), delta=p(1-p)를 유지한다.

```
p=0.23122206826075514
delta=0.17775842340997383
m=(0.7682397925548171,1.0534740723602700,1.1401754250991380)
```

m은 기존 s=1, epsilon=.15의 내부 질량제곱의 제곱근이다. 실제 렙톤질량이 아니다. h=diag(m), H0=h tensor I+I tensor h다. S는 두 구별 준비계를 교환하는 연산자, Pi±=(I±S)/2이며 dim(Pi+)=6, dim(Pi-)=3이다. 초기 상태는 rho_p tensor rho_p, rho_p=diag(p,1-p,0)다.

beta=1/(k_B T), hbar=k_B=1을 사용한다. 표의 beta는 이 내부 에너지 단위의 진단값이며, beta=0은 무한온도 한계이다. 실제 우주 온도나 열역사를 정한 값이 아니다.

## 2. 미시적 공리: 환경에 전달되는 에너지를 생략하지 않는다

E_i>E_j인 내부 전이에 대해

```
A_ij=|i><j| tensor I+I tensor |i><j|,
[H0,A_ij]=omega_ij A_ij,  omega_ij=E_i-E_j>0.
```

같은 gap을 가진 환경 qubit H_B=omega_ij |1><1|를 Gibbs 상태로 준비한다.

```
p_e=1/(1+exp(beta omega_ij)), p_g=1-p_e,
V_ij=A_ij tensor sigma_-+A_ij^dagger tensor sigma_+.
```

직접 교환자를 계산하면

```
[V_ij,H0+H_B]=0, [V_ij,S tensor I_B]=0.
```

따라서 U=exp[-i sqrt(gamma dt) V_ij]는 유니터리이고 모든 dt에서

```
Delta E_system+Delta E_bath=0,
Delta <Pi->=0.
```

를 만족한다. **새로운 열적 앙실라를 계속 공급한다는 reservoir 조건은 가정**이다. 무한한 환경 준비나 전체 우주의 자율적 열역학을 완성했다는 뜻이 아니다. 다만 한 충돌에서 계가 얻거나 잃은 에너지를 환경 장부에서 실제로 추적한다.

부분추적한 충돌을 2차까지 전개하면

```
L_ij(R)=gamma[p_e D[A_ij]R+p_g D[A_ij^dagger]R],
D[J]R=J R J^dagger-{J^dagger J,R}/2.
```

상향/하향 전이율 비는 exp(-beta omega_ij)이다. 두 대각 생성자 T3,T8에는 gap=0인 열적 qubit를 결합해 gamma D[T_a tensor I+I tensor T_a]를 더한다. 전체 생성자는 -i[H0,R]와 이 항들의 합이다.

**beta=0에서 상향/하향 계수가 각각 gamma/2가 되어 원래 CS1 생성자를 정확히 회수한다.** 감쇠율을 관측값으로 조정하지 않는다. gamma는 계산 시간단위이며 정상상태를 바꾸지 않는다. 반복 충돌에서 연속 생성자를 얻는 일반적 틀은 [R1], 열적 평형의 마스터방정식 틀은 [R2]다.

## 3. 정상상태의 미시적 기원과 유일성

고정된 swap 가중치 delta에서

```
Z±(beta)=Tr[Pi± exp(-beta H0)],
rho±(beta)=Pi± exp(-beta H0)/Z±,
Sigma_(beta,delta)=(1-delta)rho+ +delta rho-.
```

각 유한 충돌도 Sigma_(beta,delta) tensor rho_B를 불변으로 둔다. 그 전체 상태가 각 보존 부문 안에서 exp[-beta(H0+H_B)]에 비례하고 U가 그 에너지와 부문을 보존하기 때문이다. 따라서 연속 생성자에도 정확한 정상상태다.

유일성: 유한 beta와 0<delta<1에서 Sigma는 full rank다. 정상 Heisenberg 연산자 X에 대해 Lindblad의 product identity와 불변상태를 사용하면

```
0=Tr{Sigma[L^*(X^dagger X)-L^*(X^dagger)X-X^dagger L^*(X)]}
 =sum_J Tr{Sigma[J,X]^dagger[J,X]}.
```

따라서 모든 [J,X]=0이다. 상향/하향 집단 생성자와 대각 생성자는 su(3)를 생성하고, 3 tensor 3=6 plus 3bar는 서로 비동등한 두 불가약 표현이다. 그 가환대수는 span(I,S)이다. 정상공간은 2차원이며 trace와 delta가 계수를 유일하게 결정한다. 같은 논리로 영이 아닌 순허수 주변 고유모드도 없으므로 각 보존 부문에서 평형에 접근한다.

beta=0의 정상상태는 기존 (1-delta)Pi+/6+delta Pi-/3이다. beta>0에서는 일반적으로 그 균등 평균이 아니다. **초기 delta는 여전히 보존되는 입력이며, 이 collective 환경만으로 자연상수 delta가 선택되지는 않는다.**

## 4. 핵심 공통식: 미시 순도 변화가 에너지분율의 보정을 정한다

한 사본의 열적 분배함수를 Z1(beta)=Tr exp(-beta h)라고 하자. SWAP trace 항등식으로

```
Z±=[Z1(beta)^2 ± Z1(2beta)]/2.
```

따라서 한 사본 열적상태 r_beta=exp(-beta h)/Z1의 순도

```
P_beta=Tr(r_beta^2)=Z1(2beta)/Z1(beta)^2
```

와 U±=-d ln Z±/d beta를 연결하면

```
U- - U+ = 2 P_beta'/(1-P_beta^2),
P_beta'=2 P_beta[u(beta)-u(2beta)],
u(beta)=Tr(r_beta h).
```

u'(beta)=-Var_beta(h)<0이므로, 비상수 h와 beta>0에서 U->U+다. 실제 사건의 확률은 delta지만, 그 사건이 차지하는 에너지분율은

```
f_E=delta U-/Ubar, Ubar=(1-delta)U+ +delta U-,
f_E-delta = [delta(1-delta)/Ubar] * [2 P_beta'/(1-P_beta^2)] >0.
```

**동일한 질량·온도·미시 상태가 순도, 사건확률, 거시 에너지비를 동시에 제한한다.** 열적 준비의 순도 P_beta는 CS1 초기 rank-two 기록의 순도 1-2delta와 다른 양이다. 혼동해서 넣지 않는다.

고온 전개에서 E0=2 Tr(h)/3, v_h=Tr(h^2)/3-(Tr h/3)^2라 하면

```
U+=E0-(5/2)beta v_h+O(beta^2),
U-=E0-beta v_h+O(beta^2),
f_E/delta-1=(1-delta)(3 v_h/2E0) beta+O(beta^2).
```

기존의 분산비 5/2가 이번에는 **두 부문의 열적 에너지 반응의 차이**로 다시 등장한다. 그러나 유한 beta에서도 분산비가 항상 5/2라는 주장은 틀리다.

엄밀한 보수적 상계도 있다. Delta_h=E_max-E_min, E_min>0이면 대칭 에너지 범위는 2Delta_h, 반대칭 범위는 Delta_h다. 각 분산은 범위제곱/4 이하이고, U- -U+의 beta미분은 Var+ -Var-다. 그러므로

```
0 <= f_E/delta-1 <= (1-delta) beta Delta_h^2/(2 E_min).
```

이는 모든 양의 beta에 유효한 진단 상계이지 관측 신뢰구간은 아니다.

## 5. 고정 진단값의 실제 계산

|beta|U+|U-|사건확률|에너지분율|동일시 상대편향|
|---:|---:|---:|---:|---:|---:|
|0|1.9745928600|1.9745928600|0.1777584234|0.1777584234|0|
|0.1|1.9682432848|1.9720795228|0.1777584234|0.1780432011|+0.16020%|
|1|1.9083435592|1.9505963466|0.1777584234|0.1809818850|+1.81339%|
|5|1.6697740881|1.8849756654|0.1777584234|0.1961737783|+10.35976%|
|10|1.5629096483|1.8531801366|0.1777584234|0.2040364285|+14.78299%|

고온 상대편향의 1차 계수는 0.0157689060991 beta다. beta는 결과에 맞춰 선택하지 않았다. 표는 자연의 질량/온도 예측이 아니라 **같은 미시 구성의 적용조건 검사**다. 기존 정확한 동일시는 beta=0의 한계에서 회수되며, 유한온도 열적 환경에 그대로 연장할 수 없다.

## 6. 같은 미시 상태의 거시 자유에너지·압력·열 보존

swap 가중치가 보존되므로 평형에서 최소화하는 함수는

```
F_delta=-T[H2(delta)+(1-delta)ln Z+ +delta ln Z-],
H2(delta)=-delta ln delta-(1-delta)ln(1-delta).
```

같은 delta를 가진 상태 R에 대해 직접 전개하면

```
F[R]-F[Sigma_(beta,delta)]=T D(R||Sigma_(beta,delta)) >=0.
```

따라서 위 상태는 **에너지와 보존량을 함께 처리하는 유일한 제한 자유에너지 최소상태**다. 같은 CPTP 반군의 상대엔트로피 수축성으로, 고정 부피에서는

```
dS/dt-beta dQ_system/dt >=0.
```

이 성질은 quantum entropy production의 표준 틀[R3]이며, 여기의 구체적인 상태·에너지를 대입해 확인한다.

이제 h=h(V)가 바뀌면 물리적 압력은

```
P_work=-(partial F_delta/partial V)_(T,delta)
      =-(1-delta)<H0,V>_+ -delta<H0,V>_-.
```

동일한 준위 기저에서 [H0,H0,V]=0인 본 시험에서는

```
-(partial Ubar/partial V)_T
 =P_work+beta sum_(s=±) w_s Cov_s(H0,H0,V).
```

**열적 환경과 계속 접촉하는 경우, 평균에너지를 부피로 미분한 값을 곧바로 압력이라고 부르면 열교환 항을 빠뜨린다.** 단열 상태추적을 가정했던 기존 셀 보고서가 틀렸다는 뜻이 아니라, 상태준비의 전제가 바뀌었으므로 미분조건도 바뀐다는 뜻이다.

진단 h_i(V)=sqrt(m_i^2+k^2 V^(-2/3)), k=.3, V=2, beta=1에서

```
P_work=0.0196409698805,
- (partial Ubar/partial V)_T=0.0202253367673,
beta weighted covariance=0.0005843668865.
```

원래 준위의 미분, 자유에너지의 독립 5점 차분, 열항을 포함한 에너지 차분을 대조했다. 같은 모형의 압력은 양수이고 w=0.0198624489다. 이를 암흑에너지라고 부르지 않는다. 실제 우주 셀 수·온도이력과 중력 결합은 여기서 선택하지 않았다.

## 7. delta의 값 자체를 선택하려면 무엇이 달라져야 하는가

집단 결합만 쓰면 [S,H_int]=0이라 delta는 보존된다. 따라서 그 기원은 초기 준비에 남는다. 이를 숨기지 않고 **다른 명시적 후보**를 검사했다.

같은 온도에서 각 사본에 작용하는 국소 에너지분해 열결합을 조금이라도 추가하면, 모든 jump의 가환대수는 I뿐이다. 유일 정상상태는

```
Sigma_eq=r_beta tensor r_beta,
delta_eq=(1-Tr(r_beta^2))/2.
```

가 된다. 국소 결합의 강도가 양수인 한 최종 값은 그 강도와 무관하다. 단, 그 강도가 매우 작으면 평형 도달시간은 길어지고, 무한시간 극한과 강도0 극한은 교환되지 않는다. 원래 conserved-sector 모형과 이 모형을 같은 것으로 취급하지 않는다.

|beta|국소 열결합이 선택하는 delta_eq|
|---:|---:|
|0|0.3333333333|
|0.1|0.3332908918|
|1|0.3288040796|
|5|0.2223861551|
|10|0.0712075098|

**이것은 미시적 기원에 대한 구체적인 선택식이지만, beta와 실제 h를 독립적으로 결정해야 자연상수의 예측이 된다.** delta=0.1777584234가 되도록 beta를 역산하거나 채택하지 않았다. 기존 근접식을 정답으로 넣고 초기상태를 선택한 뒤 기원을 증명했다고 부르지 않는다.

## 8. 진공 및 관측 범위

h에 같은 상수를 더하면 Gibbs 상태와 모든 에너지차·충돌율·delta_eq는 같지만 평균에너지의 기준은 변한다. 실제 시공간 작용의 밀도독립 진공항은 또 별도의 정합계수다. 따라서 이 열적 동역학은 진공 절대량이나 Omega_DE를 유도하지 않는다.

u=delta/8을 PMNS 원소로 읽는 식별, q의 최종 바리온 생성량, H_phase의 오늘 시점, Higgs/Yukawa와 muon g-2는 여전히 원래 증명 의무다. 이 작업에서 BAO/CMB 관측잔차를 줄였거나 과거 전체 계산을 재실행했다고 하지 않는다. **개선한 것은 미시 환경의 에너지 보존과 열적 상태의 물리적 일관성·보정의 명시성이다.**

## 9. 검산·재현·main 반영 범위

24개 unittest: 원래 CS1 스펙트럼 회수, 유한 beta 정상상태, kernel 차원2/1, 충돌의 총에너지 보존, 유한충돌 전개의 2차 잔차, 제한 자유에너지 최소와 감소, 순도–에너지 분할 항등식, 고온 보정·상계, 압력 및 열항, 상수이동 비식별성 등을 실행했다.

- 유한 충돌 에너지 balance 잔차: 1.94e-16 (진단 에너지 단위).
- 81x81 생성자 정상잔차: 검사 beta에서 최대 약3.8e-16.
- matrix exponential과 별도 DOP853: 최대 약2.2e-14 (HS norm).
- beta=1, 진단시간16에서 평형 오차: 약5.80e-11.
- 정확한 무한시간 수렴률 상계를 유한 beta 전체에 대해 새로 인증하지는 않았다. spectral gap 수치는 진단값이다.

```bash
OPENBLAS_NUM_THREADS=1 python thermal_origin.py --output results.json
```

원격 main의 CS1 및 기존 88–89장 연구를 수정하거나 삭제하지 않는다. 본 폴더를 additive commit으로 추가한다. 과거 수 MB급 ZIP 전체가 이미 원격에 있다고 주장하지 않는다. 해당 원본 ZIP은 이 대화의 산출물이고, 그 해시·크기와 선행 커밋은 provenance.json에 기록한다.

## 출처

기존 CE 근거: main의 experiments/ce_state_selection_20260919/REPORT_ko.md, commit cbbc1cd930eb6c7ff42e0437aca261334f2a271d. 특히 §§6–8의 환경 에너지·보존된 delta·비Gibbs 정상상태 구분을 보존했다.

[R1] Attal & Pautrat, From repeated to continuous quantum interactions, math-ph/0311002. https://arxiv.org/abs/math-ph/0311002

[R2] Davies, Markovian master equations, Commun. Math. Phys. 39, 91–110 (1974). https://doi.org/10.1007/BF01608389

[R3] Spohn, Entropy production for quantum dynamical semigroups, J. Math. Phys. 19,1227 (1978). https://doi.org/10.1063/1.523789

[R4] Strasberg, Repeated interactions and quantum stochastic thermodynamics at strong coupling, 1907.01804. https://arxiv.org/abs/1907.01804

표준 thermalization·Gibbs·상대엔트로피 정리의 최초 발견을 주장하지 않는다. 새 CE 계산은 기존 집단 준비모형에 보존적 환경을 구성하고, 같은 준위에서 사건비율·에너지·압력의 차이를 정량화한 것이다.
