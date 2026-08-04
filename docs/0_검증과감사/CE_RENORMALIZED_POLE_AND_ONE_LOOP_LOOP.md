# CE renormalized pole · one-loop 루프

기준일: 2026-08-04  
범위: action provenance → renormalized \(\Gamma^{(2)}\) gate → 선택적 \(Z_2\)
portal의 scalar one-loop 통제 계산

> 결론: 현재 CE 증거 단계는 여전히 `REGISTERED_SCALE`이다. 완전한
> renormalized CE action, counterterm manifest와 실제 kernel 자료가 없기
> 때문이다. 다만 어떤 자료가 들어와야 pole·residue·dispersion 단계가
> 올라가는지는 fail-closed 인증서로 구현했다. 선택적 portal의 two-real-scalar
> one-loop 통제에서는 결합은 섭동적이지만 29.65 MeV 질량은 방사적으로
> 안정하지 않다는 강한 진단이 추가되었다.

## 1. 이번 루프의 판정

| 명제 | 판정 |
|---|---|
| 저장소에 complete renormalized CE action이 있다 | `REFUTED BY PROVENANCE AUDIT` |
| Q0 선택적 portal action이 bare tree control로 일관된다 | `EXACT CONDITIONAL` |
| Q0 action definition을 pole/vertex 계산과 동일 hash로 묶었다 | `IMPLEMENTED` |
| 수치 kernel에서 simple pole·residue·cut·holdout을 재계산할 수 있다 | `CONTROL IMPLEMENTED` |
| 현재 CE 자료가 renormalized kernel stage에 도달한다 | `NO` |
| 선택적 portal scalar one-loop diagram 식 | `DERIVED CONDITIONAL` |
| 그 finite 합이 물리 pole-mass 예측이다 | `NO` |
| 29.65 MeV same-field portal pole이 radiatively stable하다 | `NOT SUPPORTED` |
| physical CE pole·spectral positivity·LSZ | `OPEN` |

## 2. action provenance 감사

| 후보 | 닫히는 범위 | 첫 blocker | 판정 |
|---|---|---|---|
| Q0 Abelian-Higgs+\(Z_2\) | signature, canonical singlet, 고정 Minkowski 배경, Abelian \(R_\xi\), ghost | counterterm·renormalization 미적용; CE·중력·fermion·비가환 sector 제외 | `TREE CONTROL ONLY` |
| \(\epsilon\) 제1원리 FLRW scalar | 고정 배경의 고전 mode equation | field/계수 차원, metric perturbation, gauge, counterterm 미고정 | `CLASSICAL CANDIDATE` |
| 형식적 \(\sigma\) 모델 | 독립 scalar+중력의 형식 | kinetic/EOM 부호, 차원, metric variation 불일치 | `REJECT AS Γ INPUT` |
| `axium` master action | 형식적 Hessian 문법 | 물리 field identity와 action sector 미정, higher-derivative extra pole 미처리 | `FORMAL/OPEN` |
| 경로공간/topology action | rate-functional compactness | 4D Lorentzian local QFT action과 field map 없음 | `NOT A TWO-POINT ACTION` |

따라서 첫 blocker는 loop 적분이 아니라 다음 identity의 부재다.

\[
\Phi_H\text{ (Hessian readout)}
\ne\sigma\text{ (dimensionless suppression)}
\ne S\text{ (FLRW scalar)}
\ne\phi\text{ (optional portal field)}.
\]

### 2.1 형식적 \(\sigma\) action의 정확한 오류

문서는 signature \((-+++ )\)를 선언한 뒤

\[
\mathcal L_\sigma
=+\frac12(\nabla\sigma)^2-V(\sigma)+\sigma f(R)
\]

를 적었다. 이를 문자 그대로 변분하면

\[
\Box\sigma+V'(\sigma)-f(R)=0
\]

이지, 문서에 표시된
\(\Box\sigma-V'(\sigma)-f(R)=0\)가 아니다. 또한 \((-+++ )\)에서
표시된 kinetic term은 canonical scalar의 시간 운동항과 반대 부호다.

\(\sigma\)가 무차원이라면 건강한 최소 seed는 예를 들어

\[
\mathcal L_\sigma
=-\frac{F_\sigma^2}{2}(\nabla\sigma)^2-U(\sigma)
\]

처럼 \([F_\sigma^2]=2\), \([U]=4\)를 가져야 한다. \(\sigma f(R)\)를
유지하면 metric variation에는 최소한

\[
\sigma f_RR_{\mu\nu}
-\frac12g_{\mu\nu}\sigma f
+(g_{\mu\nu}\Box-\nabla_\mu\nabla_\nu)(\sigma f_R)
\]

가 생긴다. 현재 형식 문서의 Einstein 방정식에는 이 항이 없다. 그러므로
해당 문서는 수정 전까지 physical \(\Gamma_R^{(2)}\) 입력에서 격리한다.

## 3. action-definition hash 연결

Q0 control의 field, signature, action/potential convention, background,
gauge fixing, ghost, 제외 sector와 renormalization 상태를 canonical JSON으로
직렬화해 SHA-256을 계산한다.

```text
c6e1f448c388900d3a70f997d2c133580f1a87a0682e6fe2309fe58bf21ed233
```

이 digest는 numerical benchmark가 아니라 **선택한 action definition**을
식별한다. tree two-point와 aggregate portal certificate가 같은 digest를
보존하도록 회귀 테스트를 추가했다. 이는 drift를 막지만 counterterm이나
renormalized CE action을 새로 만들어내지는 않는다.

## 4. renormalized pole 인증 단계

새 인증서는 caller가 `physical_pole=True`를 넣는 방식을 허용하지 않는다.
manifest와 수치 replica에서 다음을 내부 계산한다.

```text
REGISTERED_SCALE
  → RENORMALIZED_KERNEL_CONTROL
  → ISOLATED_SIMPLE_POLE_CONTROL
  → POSITIVE_RESIDUE_CONTROL
  → DISPERSION_CONTROL
```

각 kernel replica는 같은 action/counterterm/background hash와 field ID를
가져야 한다. 서로 다른 gauge parameter와 renormalization scale의 holdout이
모두 있어야 하며,

\[
\Gamma_R^{(2)}(s_*)=0,
\qquad
Z_*=left[\frac{d\Gamma_R^{(2)}}{ds}(s_*)\right]^{-1}>0
\]

를 수치 grid에서 다시 계산한다. 다음 반례는 전부 fail-closed다.

- 음의 derivative/residue
- double zero
- pole에서 비영 \(\operatorname{Im}\Gamma^{(2)}\)
- 음의 \(s_*\)
- first cut과 충돌
- gauge/scale pole drift
- residue drift
- \(E^2\ne m^2+\mathbf p^2\)
- kernel/dispersion action hash 불일치

synthetic scalar control은 `DISPERSION_CONTROL`까지 갈 수 있지만 그 경우에도
spectral positivity, asymptotic state, LSZ와 CE field identity는 hard `False`다.
현재 CE 실행값은 다음과 같다.

```text
maximum stage   REGISTERED_SCALE
first blocker   renormalized action manifest is absent
physical LSZ    False
CE identity     False
```

## 5. 선택적 portal의 scalar one-loop 식

tree action을 전개하면

\[
\mathcal L_{\rm int}
=-\lambda_{HP}vh\phi^2
-\frac{\lambda_{HP}}2h^2\phi^2
-\frac{\lambda_{HP}}2\chi^2\phi^2
-\frac{\lambda_\phi}{4}\phi^4.
\]

\(g=2\lambda_{HP}v\)라 두면 vertex는

\[
h\phi\phi:-ig,
\quad hh\phi\phi:-i2\lambda_{HP},
\quad \chi\chi\phi\phi:-i2\lambda_{HP},
\quad \phi^4:-i6\lambda_\phi.
\]

따라서 scalar diagram 식은

\[
\Sigma_{\rm scal}(s)=\frac1{16\pi^2}
\left[
g^2\bar B_0(s;m_h^2,m_\phi^2)
+\lambda_{HP}\bar A_0(m_h^2)
+\lambda_{HP}\bar A_0(m_\chi^2)
+3\lambda_\phi\bar A_0(m_\phi^2)
\right]+\Sigma_{\rm ct}.
\]

mixed \(h\phi\) bubble은 내부 선이 서로 달라 symmetry factor가 1이다.
tadpole은 동일 loop leg 교환으로 \(1/2\)가 남아 위 계수가 된다.

구현된 숫자 control은 \(h,\phi\) 두 real scalar만 포함한다. \(m_\chi\) 또는
\((\xi,m_A)\) 입력이 없으므로 Goldstone 항을 임의로 0으로 놓지 않았다.
Landau-gauge dimensional-regularization control을 명시할 때만
\(A_0(0)=0\)으로 둘 수 있다. 전체 SM에서는 Goldstone multiplicity도 다시
계산해야 한다.

사용한 finite convention은

\[
\bar A_0(m^2)=m^2\left[1-\ln\frac{m^2}{\mu^2}\right],
\]

\[
\bar B_0(s)=-\int_0^1dx\,
\ln\frac{xm_1^2+(1-x)m_2^2-x(1-x)s-i0}{\mu^2},
\]

\[
\Gamma_R^{(2)}(s)=s-m_R^2+\Sigma_R(s),
\qquad
Z_*=[1+\Sigma_R'(s_*)]^{-1}
\]

이다. Passarino–Veltman 적분과 scale convention의 구현 참고는
[COLLIER](https://arxiv.org/abs/1604.06792), 선택적 singlet portal의 one-loop
vacuum/RG 구조는 [Gonderinger et al.](https://arxiv.org/abs/0910.3167)을
대조했다.

## 6. 수치 결과

입력은

\[
\lambda_{HP}=0.031598052\ldots,
\quad \lambda_\phi=0.1,
\quad v=246.22\,\mathrm{GeV},
\quad m_h=\mu=125.25\,\mathrm{GeV},
\quad m_\phi=0.02964757\,\mathrm{GeV}
\]

이다.

```text
g_hφφ                                  15.560144719 GeV
h-loop seagull finite piece             3.139034221 GeV^2
hφ mixed bubble finite piece            1.533229411 GeV^2
φ-loop seagull finite piece             2.955211409e-5 GeV^2
two-real-scalar finite sum               4.672293184 GeV^2
finite sum / target mass squared         5315.594954
Sigma'(m_phi^2)                          4.886763794e-5
linearized residue control               0.999951135
first hφ cut                             125.279647570 GeV
Im Sigma at light target                 0
```

scale만 \(\mu=m_h/2,m_h,2m_h\)로 바꾸면 raw finite 합은

```text
-1.804843787, 4.672293184, 11.149430155 GeV^2
```

로 부호까지 바뀐다. 이것은 물리 오차막대가 아니다. running parameter와
counterterm이 상쇄해야 할 subtraction-scale dependence다. 따라서
\(4.672\,\mathrm{GeV}^2\)를 그대로 물리 질량 보정이라고 부르면 안 된다.

그럼에도 진단은 강하다.

- \(\lambda_{HP}/(16\pi^2)\)는 작으므로 loop expansion 자체는 섭동적이다.
- finite mass scale은 light target \(m_\phi^2\)보다 약 5316배 크다.
- 같은 convention에서 target을 유지하려면 약
  \(m_\phi^2/|\Sigma|=1.88\times10^{-4}\) 수준의 추가 retuning이 필요하다.
- 반면 momentum derivative와 residue 변화는 약 \(4.9\times10^{-5}\)로 작다.
- light pole은 \(h\phi\) cut보다 훨씬 아래라 이 subset에서
  \(\operatorname{Im}\Sigma=0\)이다.

즉 병목은 강결합이나 즉시 생기는 width가 아니라 **additive light-mass
radiative tuning**이다. 이 결론도 선택적 portal control에만 해당하며 CE core의
반증은 아니다.

## 7. 현재 점수

| 항목 | 점수 | 이유 |
|---|---:|---|
| action 후보 provenance 분리 | 100/100 | 후보별 첫 blocker와 scope 고정 |
| Q0 tree action hash 연결 | 100/100 | pole/vertex certificate와 digest 일치 |
| renormalized pole gate 구현 | 95/100 | 수치 replica control 완결; 실제 CE data 없음 |
| 선택적 portal scalar one-loop diagram 식 | 90/100 | \(h,\phi\) 숫자 control; gauge-complete 아님 |
| 29.65 MeV portal pole의 방사 안정성 | 10/100 | 큰 additive scale과 scheme 의존성 |
| 실제 CE renormalized kernel | 0/100 | complete action·CT·kernel replica 없음 |
| spectral positivity·LSZ | 0/100 | KL density와 asymptotic state 없음 |
| 물질 생성률 | 0/100 | external LSZ와 renormalized vertex/rate 없음 |

## 8. 재현

2026-08-04 검증 결과:

- 관련 집중 회귀: `83 passed`
- 전체 회귀: `1329 passed, 13 skipped, 0 failed`
- 전체 회귀에서는 작업 트리에서 이미 삭제된 fixture에 직접 의존하는 테스트
  파일 5개(`local_memory`, `origin_life_branching`, `origin_life_coupled`,
  `q0_manifest`, `neural_tree_algorithm_census`)만 명시적으로 제외했다.
- Ruff check/format과 `git diff --check`를 통과했다. diff 검사의 출력은
  CRLF 변환 경고뿐이다.

```powershell
uv --cache-dir .uv-cache run python examples/physics/ce_renormalized_pole_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_renormalized_pole_certificate.py `
  tests/test_portal_one_loop_control.py `
  tests/test_ce_two_point_vertex_certificate.py -q
```

핵심 구현:

- `reality_stone/python/reality_stone/clarus/renormalized_pole_certificate.py`
- `reality_stone/python/reality_stone/clarus/portal_one_loop_control.py`
- `examples/physics/ce_renormalized_pole_gate.py`

## 9. 다음 분기

다음 계산 전에 field identity를 하나 선택해야 한다.

1. **CE core readout branch:** \(\Phi_H\)를 local particle field로 간주하지 않고
   실제 connected correlator와 spectral reconstruction부터 수집한다.
2. **optional portal branch:** complete SM+\(phi\) bare action, background,
   gauge/tadpole prescription, counterterm basis와 running input을 하나의 manifest로
   고정한 뒤 gauge-complete one-loop complex pole을 계산한다.

두 branch를 동일한 29.65 MeV 입자 증명으로 합치지 않는다. portal branch가
full pole gate를 통과해도 `CE field identity`는 독립 matching 증거가 들어오기
전까지 계속 `False`다.

첫 번째 branch의 후속 감사와 fail-closed scaffold는
[CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md](CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md)에
기록했다. 실제 CE raw ensemble은 발견되지 않았으며 finite-grid spectrum
유일성은 정규화 보존 nullspace 반례로 기각되었다.
