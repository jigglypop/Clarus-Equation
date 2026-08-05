# 핵융합 직접 연산자 대체분기 루프

코드: `reality_stone/python/reality_stone/clarus/fusion_operator_alternatives_loop.py`  
실행: `examples/physics/fusion_operator_alternatives_gate.py`  
테스트: `tests/test_fusion_operator_alternatives_loop.py`

## 1. 판정

flavor-aligned (uds) 후보의 경계 제약을 피하기 위해 세 연산자 계열을 추가로
감사했다.

| 분기 | 1%에 필요한 값 | 가장 가까운 제약 | 판정 |
|---|---:|---:|---|
| pure trace/gluon | (|K_\Theta|v/f=5.48) | (1.38\times10^{-3}) | 약 (4.0\times10^3)배 초과 |
| protophobic (g_p=0) | (g_n=0.03018) | Pb effective 0.0235 vs 0.01758 | neutron `FAIL` |
| neutron-phobic (g_n=0) | (g_p=0.042686) | kaon 조합 | (9.5\times10^3)--(2.75\times10^4)배 초과 |
| Pb charge blind spot | (g_p/g_n=-126/82) | D/T charge product | 부호가 음수, 반발력 |
| massless disformal upper | \(M=180.705\) MeV | massless 200/810 MeV 참고값; light-mediator collider 1.2 TeV | collider `FAIL` |

따라서 이번에 선언한 대체 연산자 중 실험 제약을 통과한 1% CE branch는 없다.

## 2. pure trace/gluon

\[
\mathcal L=-\frac{K_\Theta}{f}\phi\Theta^\mu_{\ \mu}
\]

에서 사용한 핵자 계수는

\[
g_p=0.7844\frac{K_\Theta}{f/\mathrm{GeV}},
\qquad
g_n=0.7817\frac{K_\Theta}{f/\mathrm{GeV}}.
\]

D/T charge product (1.8221\times10^{-3})을 맞추면

\[
f/K_\Theta=44.9215\ \mathrm{GeV},
\qquad |K_\Theta|v/f=5.4762.
\]

[Delaunay et al.](https://arxiv.org/abs/2501.16477) Fig. 4의 one-parameter
trace 방향 한계 약 (1.38\times10^{-3})보다 3968배 크다. 여러 Wilson
coefficient 사이의 별도 cancellation action은 공급되지 않았으므로 이 분기는 닫힌다.

## 3. isospin endpoint와 Pb blind spot

고정해야 하는 것은

\[
(g_p+g_n)(g_p+2g_n)=P_{\rm req}>0
\]

이다.

- (g_p=0)이면 (g_n=\sqrt{P/2}=0.03018)이고 Pb effective coupling은
  0.02349로 외삽 neutron bound의 1.336배다.
- (g_n=0)이면 neutron scattering source는 사라지지만
  (g_p=\sqrt P=0.042686)이다. 이 방향을 quark 계수로 맞출 때 필요한 두
  kaon combination은 198과 1581이고, digitized bound 0.0209와 0.0574를
  각각 9474배, 27544배 넘는다.
- Pb coherent charge를 지우는 (82g_p+126g_n=0)은
  (g_p/g_n=-126/82)이다. 이때 D/T product coefficient는 -0.24866이라
  attraction이 아니라 repulsion이다.

Universal (g_p=g_n)은 (max(|g_p|,|g_n|))을 최소화한다. neutron-only
proxy만 최소화하면 neutron-phobic endpoint가 최적이지만 rare-kaon gate가
그 방향을 닫는다. 이는 제약 하나만 골라 coupling을 최적화하면 안 되는 이유다.

## 4. disformal massless upper bound

감사한 (Z_2)-even 연산자는

\[
\mathcal L_{\rm dis}=\frac{\partial_\mu\phi\partial_\nu\phi}{M^4}T^{\mu\nu}
\]

이고, massless two-scalar one-loop potential의 attraction magnitude는

\[
V_{\rm dis}(r)=\frac{3m_Dm_T}{32\pi^3M^8r^7}.
\]

이를 기존 turning-point 안정화 WKB, Bosch--Hale (sigma(E)), 10 keV
Maxwellian에 그대로 넣었다. 1%에 필요한 가장 큰 scale은

\[
M=180.70494\ \mathrm{MeV}
\]

이다. coarse/default/fine 격자의 gain spread는 (2\times10^{-6})보다 작다.
등록 scalar mass는 pair potential을 더 억제하므로 massless 계산은 낙관적 상한이다.

공개 제약을 같은 식에 넣은 결과는

```text
M = 200 MeV       gain = 4.39587e-3
M = 810 MeV       gain = 6.02449e-8
M = 1.2 TeV       gain = 2.59627e-33
```

이다. 200 MeV 수소분광과 810 MeV stellar-burning 값은
[Brax--Burrage 계열 분석](https://arxiv.org/abs/1407.2376)의 **massless scalar**
결과다. 따라서 이를 29.64757 MeV scalar의 mass-specific 배제로 부르지 않고
참고값으로만 기록한다. 반면 29.65 MeV는 collider 운동량에 비해 light-mediator
극한이며, [ATLAS scalar dark-energy search](https://cds.cern.ch/record/2627837)의
약 1.2 TeV scale은 필요한 180.7 MeV보다 6천 배 이상 높다. 이 적용 가능한
collider bound 하나만으로도 낙관적인 massless-potential upper branch는 닫힌다.
nonlinear screening completion을 새로 도입하면 collider 해석과 10 keV plasma의
unscreening, EFT 일관성을 모두 다시 증명해야 한다. 그런 completion은 현재
공급되지 않았다.

## 5. 최종 gate

```text
trace/gluon physical gate       False
protophobic physical gate       False
neutron-phobic physical gate    False
Pb blind-spot physical gate     False
disformal physical gate         False
maximum supported stage         ALTERNATIVE_OPERATOR_MODEL_CLASS_NO_GO
```

flavor-aligned (uds) 후보가 여전히 가장 가까운 조건부 후보이며, 이를 통과시키는
데 필요한 입력은 별도 문서의 mass-specific neutron likelihood와 full-NLO kaon
likelihood다.

## 6. 실행

```bash
uv run python examples/physics/fusion_operator_alternatives_gate.py
uv run --extra dev python -m pytest tests/test_fusion_operator_alternatives_loop.py -q
```
