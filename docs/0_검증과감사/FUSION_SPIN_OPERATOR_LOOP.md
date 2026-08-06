# 핵융합 spin/operator 반복 감사

코드: `reality_stone/python/reality_stone/clarus/fusion_spin_operator_loop.py`  
테스트: `tests/test_fusion_spin_operator_loop.py`  
실행: `examples/physics/fusion_spin_operator_gate.py`

## 1. 판정과 적용 범위

최신 canonical \(29.6991596174\) MeV CE 매개체의 pseudoscalar, axial-vector,
vector, massive spin-2,
derivative-node 분기를 기존 10 keV D--T scalar 계산과 같은 장거리 Yukawa 세기에
맞췄다. 기존 scalar loop에서 직접 가져오는 목표값은

\[
P_{1\%}=1.826376655\times10^{-3},\qquad
Y(r)=\frac{e^{-Mr}}{4\pi r}.
\]

이 문서의 결합은 \(V=-P_{1\%}Y\)와 같은 크기를 내는 **operator-level 등가값**이다.
정확한 \(^{5}\mathrm{He}\;3/2^+\) coupled-channel NCSMC/R-matrix 재계산과
29.6991596174 MeV 전용 \(\pi/K/\)BaBar likelihood가 없으므로 모든 physical gate는
항상 `False`다.

| 분기 | 수학적 장거리 해 | 빠진 물리 입력 | physical gate |
|---|---|---|---|
| pseudoscalar | quartet에서 존재, \(\lvert g_{PD}g_{PT}\rvert=130.908\) | D/T pseudoscalar form factor와 공명 재계산 | `False` |
| axial-vector | quartet에서 존재, \(g_{AD}g_{AT}=2.73956\times10^{-3}\) | quark--nucleus matching, K likelihood, UV 완결 | `False` |
| vector | attractive Pb \(q=0\) blind 해 존재 | finite-\(q\) Pb, \(\pi/K\), gauge completion | `False` |
| spin-2 | 중앙 인력 존재, \(c/\Lambda=0.0228034\ \mathrm{GeV}^{-1}\) | mass-bin BaBar likelihood 또는 보존된 비보편 UV | `False` |
| derivative node | on-shell node를 쓸 수 있음 | 그 node가 Yukawa pole도 제거함 | `False` |

## 2. raw 비편극 trace와 quartet projector

점핵 spin 정규화를

\[
\Sigma_D=S_D,\qquad \Sigma_T=2S_T,\qquad
O=\Sigma_D\!\cdot\!\Sigma_T
\]

로 두면

\[
O_{S=3/2}=+1\quad(4\text{ states}),\qquad
O_{S=1/2}=-2\quad(2\text{ states}).
\]

따라서 raw 비편극 평균은 정확히

\[
\mathrm{Tr}(\rho_{\rm unpol}O)
=\frac{4(1)+2(-2)}6=0.
\]

반응 커널이 spin-independent이면 이 값 때문에 spin-dependent potential의 1차항이
상쇄된다. 그러나 저에너지 D--T 반응은 \(l=0,J^\pi=3/2^+\) quartet 공명이
지배한다. Quartet projector는

\[
P_{3/2}=\frac{O+2}{3},\qquad
\mathrm{Tr}(\rho_{\rm unpol}P_{3/2}O)=\frac46,
\qquad \langle O\rangle_{3/2}=1.
\]

그러므로 raw trace `0`을 그대로 반응률 보정에 대입하면 안 된다. 코드가 기록하는
`quartet_projected_unpolarized_trace`는 \(2/3\)이다. 이 채널 선택은
[Hupin, Quaglioni, Navrátil](https://arxiv.org/abs/1803.11378)의 D--T ab initio
계산에 근거하지만, 그 논문이 아래 새 매개체 potential을 계산한 것은 아니다.

## 3. Pseudoscalar

핵 수준의 점입자 정규화 작용은

\[
\mathcal L_P=i g_{Pi}\phi\bar N_i\gamma_5N_i
\]

이고, NR potential은

\[
V_{PP}=\frac{g_{PD}g_{PT}}{4m_Dm_T}
(\Sigma_D\!\cdot\nabla)(\Sigma_T\!\cdot\nabla)Y.
\]

\(l=0\) 각평균과 \(r>0\) 장거리 부분은

\[
V^{(0)}_{PP}=+
\frac{g_{PD}g_{PT}M^2}{12m_Dm_T}O\,Y(r).
\]

Quartet attraction에는 반대 부호의 결합이 필요하고,

\[
|g_{PD}g_{PT}|=130.90796,\qquad
|g_P|_{\rm equal}=11.44150,\qquad
\frac{g_P^2}{4\pi}=10.4173.
\]

따라서 이 숫자는 약결합 one-boson/Born 예측이 아니다. D와 T가 기본 Dirac 입자가
아니므로 실제 핵자 결합으로 옮기려면 pseudoscalar nuclear form factor와 접촉항,
tensor-induced partial-wave mixing을 함께 계산해야 한다. 여기서는 이를 공급하지 않아
fail-closed다. NR potential과 contact-term 부호는
[Fadeev et al.](https://arxiv.org/abs/1810.10364)의 수정된 potential을 따른다.

## 4. Axial-vector

작용과 S-wave 장거리 항은

\[
\mathcal L_A=X_\mu\bar N_i\gamma^\mu g_{Ai}\gamma_5N_i,
\]

\[
V^{(0)}_{AA}=-\frac23g_{AD}g_{AT}O\,Y(r).
\]

두 번째 식은 Proca longitudinal mode와 transverse term을 모두 각평균한 결과다.
Quartet attraction에는 같은 부호가 필요하고,

\[
g_{AD}g_{AT}=\frac32P_{1\%}
=2.7395650\times10^{-3},\qquad
g_A|_{\rm equal}=0.0523409.
\]

### 보편 결합에 적용되는 비교

[Di Luzio et al.](https://arxiv.org/abs/2308.05215)의 보편 diagonal light-quark
axial 방향 근사식

\[
g_xx_u^A\lesssim3\times10^{-6}\frac{M}{0.1\ \mathrm{GeV}}
\]

은 이 질량에서 \(8.90975\times10^{-7}\)이다. 핵 유효결합을 이 quark bound와
단순 나눈 비는 \(5.8746\times10^4\)지만, 이것은 정확한 quark--D/T matching
likelihood가 아니라 긴장도를 보여 주는 proxy다.

### 비보편 결합에는 바로 적용되지 않는 비교

비보편 \(u,d,s\) 계수는 특정 K 진폭의 선형조합을 조정할 수 있다. 그러므로 위
보편 숫자를 임의의 flavor-tuned 모델에 그대로 exclusion으로 쓰지 않는다. 대신
비보편 모델은 모든 charged/neutral K 채널, anomaly cancellation, Higgs/gauge
completion을 새로 공급해야 한다. 최신 보완 연구인
[Hostert, Pospelov, Thompson (2026)](https://arxiv.org/abs/2602.19479)은
non-conserved light-quark vector/axial current에 대해 중성 모드가 \(O(10^{-5})\),
charged 모드가 보완적인 \(O(10^{-4})\) 조합까지 탐색한다고 보고한다. 이 order-of-
magnitude도 29.6991596174 MeV 전용 likelihood로 취급하지 않는다.

## 5. Vector와 attractive Pb blind 해

\[
V_{VV}=+(g_p+g_n)(g_p+2g_n)Y.
\]

보편 같은 부호 vector는 반발한다. Attraction을 유지하면서
\(\max(|g_p|,|g_n|)\)를 최소화하면

\[
\frac{g_p}{g_n}=-\frac43,\qquad
g_p=0.1208760,\qquad g_n=-0.0906570.
\]

더 중요한 정확한 대수 결과는 \(^{208}\)Pb의 \(q=0\) coherent charge

\[
82g_p+126g_n=0
\]

를 지우는 방향도 D--T attraction 구간에 있다는 것이다. 목표 세기에 맞추면

\[
g_p=0.1316884,\qquad g_n=-0.0857020,
\]

\[
Q_D=0.0459864,\qquad Q_T=-0.0397156,
\qquad Q_DQ_T=-P_{1\%}.
\]

Quark charge는

\[
g_u=0.1163596,\qquad g_d=-0.1010308,
\qquad g_u-g_d=0.2173904.
\]

이 Pb cancellation은 오직 \(q=0\)이다. 유한 momentum에서 proton/neutron form
factor가 다르므로 mass-specific Pb differential likelihood 없이 gate를 열지 않는다.

### Prompt-visible 보편/anomaly 매핑의 범위

[NA48/2의 직접 검색](https://arxiv.org/abs/1504.00607)은
\(\pi^0\to\gamma X, X\to e^+e^-\)를 9--70 MeV에서 검사했으므로 CE 질량이
범위 안이다. [Hostert--Pospelov](https://arxiv.org/abs/2306.15077)의 17 MeV
prompt-visible anomaly-current 매핑은 물리적 proton coupling 약
\(2.42\times10^{-4}\)를 proxy로 준다. 필요한 Pb-blind \(g_p\)와의 비는 544.17이다.
다만 이것은 29.6991596174 MeV mass-bin likelihood가 아니므로 코드도
`prompt_visible_proxy_is_mass_specific=False`로 기록한다.

Invisible/long-lived 또는 flavor-tuned vector에는 같은 prompt-visible 숫자를 직접
적용하지 않는다. 그런 분기는 \(\pi/K+\)missing-energy, finite-\(q\) Pb, anomaly-free
UV action을 별도로 공급해야 한다.

## 6. Massive spin-2

보편 stress-energy 작용과 선도 중앙 potential은

\[
\mathcal L_2=-\frac{c_i}{\Lambda}G_{\mu\nu}T_i^{\mu\nu},
\qquad
V_2=-\frac23\frac{c_Dc_Tm_Dm_T}{\Lambda^2}Y.
\]

동일 결합이면

\[
\frac c\Lambda=0.02280339\ \mathrm{GeV}^{-1},\qquad
\frac\Lambda c=43.8531\ \mathrm{GeV}.
\]

Ghost-free/dRGT형 낙관적 strong-coupling 척도는

\[
\Lambda_3\sim(M^2\Lambda/c)^{1/3}=0.33819\ \mathrm{GeV}.
\]

### 보편 stress-energy 결합

[Kang--Lee](https://arxiv.org/abs/2001.04868)의 BaBar 재해석 proxy는 visible에서
\(3\times10^{-5}\ \mathrm{GeV}^{-1}\), invisible에서
\(2\times10^{-4}\ \mathrm{GeV}^{-1}\) 수준이다. 요구값은 각각 760.1배와 114.0배다.
이 비교는 전자에도 같은 stress-energy 결합이
있는 보편 모델에만 바로 적용된다.

### 비보편 nucleon-only 결합

Nucleon-only 계수에는 위 BaBar 숫자를 직접 exclusion으로 쓰지 않는다. 대신
비보편 massive spin-2가 보존된 source, longitudinal unitarity, meson-loop matching을
갖는 UV completion을 공급해야 한다. 현재 그런 completion과 mass-specific likelihood가
없으므로 역시 `False`다.

## 7. Derivative on-shell node

단일 analytic vertex의 교환진폭을

\[
\mathcal A(q^2)=\frac{F_D(q^2)F_T(q^2)}{q^2-M^2}
\]

라 하자. 실재 방출을 없애려고 \(F(M^2)=0\)로 두면

\[
F(q^2)=(q^2-M^2)G(q^2)
\]

이므로 같은 vertex의 Yukawa pole residue도 없어진다. 작용의
\((\Box+M^2)\phi\mathcal O\)는 mediator EOM 연산자이며 field redefinition 뒤 contact
operator로 바뀐다. Contact는 장거리 Coulomb barrier를 낮추지 않는다.

수치적 척도는

```text
M^2                    882.0400820 MeV^2
hbar*c/M               6.6441941 fm
mu_DT                   1124.64735 MeV
E_G                     30.92 keV
p_G                     8.3395559 MeV
```

이다. Pb의 특정 spacelike momentum 한 점에만 node를 두는 것은 전체 recoil 구간을
지우지 않는다. 급격한 저에너지 form factor나 두 번째 mediator를 넣으면 새 light pole을
포함한 별도 모델이므로, 그 상태와 전체 differential likelihood를 함께 감사해야 한다.

## 8. Primary sources와 코드가 사용하는 범위

| 출처 | 코드에서 사용하는 내용 | 사용하지 않는 확대해석 |
|---|---|---|
| [Hupin et al.](https://arxiv.org/abs/1803.11378) | D--T \(3/2^+,l=0\) 지배 | 새 매개체의 정확한 1% 응답 |
| [Fadeev et al.](https://arxiv.org/abs/1810.10364) | spin-0/1 NR potential과 contact 구조 | D/T 복합핵 form factor |
| [Di Luzio et al.](https://arxiv.org/abs/2308.05215) | 보편 axial K proxy | 임의의 flavor-tuned exclusion |
| [NA48/2](https://arxiv.org/abs/1504.00607) | 9--70 MeV prompt-visible 검색 범위 | CE 질량 전용 arbitrary-vector likelihood |
| [Hostert--Pospelov](https://arxiv.org/abs/2306.15077) | anomaly-current와 17 MeV proxy | invisible/nonuniversal 모델의 직접 bound |
| [Hostert et al. 2026](https://arxiv.org/abs/2602.19479) | nonconserved K 채널의 최신 민감도 차수 | CE 질량의 완전 likelihood |
| [Kang--Lee](https://arxiv.org/abs/2001.04868) | 보편 spin-2 BaBar proxy와 unitarity | nucleon-only 비보편 직접 exclusion |

## 9. 실행

```bash
uv run --extra dev python -m pytest tests/test_fusion_spin_operator_loop.py -q
uv run --extra dev ruff check \
  reality_stone/python/reality_stone/clarus/fusion_spin_operator_loop.py \
  tests/test_fusion_spin_operator_loop.py \
  examples/physics/fusion_spin_operator_gate.py
uv run python examples/physics/fusion_spin_operator_gate.py
```

최신 canonical 질량 overlay는 thermal/flavor solver와 spin 모듈의 등록질량을 같은 값으로
주입하고 cache를 비운 뒤 재계산했다.

```powershell
.\.venv\Scripts\python.exe -c "import reality_stone.clarus.fusion_equation_iteration_loop as fe; import reality_stone.clarus.fusion_flavor_aligned_loop as fa; import reality_stone.clarus.fusion_spin_operator_loop as sp; m=29.69915961743591; fe.DEFAULT_SCALAR_MASS_MEV=m; fa.REGISTERED_SCALAR_MASS_MEV=m; sp.REGISTERED_SCALAR_MASS_MEV=m; fe.current_fusion_equation_iteration_report.cache_clear(); fa.current_fusion_flavor_aligned_report.cache_clear(); sp.current_fusion_spin_operator_report.cache_clear(); r=sp.current_fusion_spin_operator_report(); print(r.required_dt_charge_product, r.pseudoscalar, r.axial_vector, r.vector, r.spin_two, r.derivative_node, r.any_physical_operator_gate_pass)"
```

핵심 출력은 `P_1%=0.0018263766547363008`, pseudoscalar product `130.9079551`,
axial product `0.002739564982`, spin-2 `c/Lambda=0.02280339026 GeV^-1`,
`M^2=882.04008198 MeV^2`다.

성공 조건은 physical gate가 열리는 것이 아니라, 모든 수학 수치가 재현되고 빠진
NCSMC/R-matrix 및 mass-specific likelihood 때문에 gate가 명시적으로 닫혀 있는 것이다.
