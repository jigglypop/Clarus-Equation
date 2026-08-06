# CE two-point → pole → vertex 루프

> 갱신: 2026-08-06
> 범위: Q0.4 singlet quadratic block과 Q0.5 국소 vertex의 조건부 통제 증명  
> 결론: 선택적 포탈 EFT의 tree-level pole·vertex는 닫히지만, 29.6991596 MeV
> 물리 CE pole과 실제 물질 생성은 닫히지 않는다.

## 1. 이번 루프의 판정

이번 루프는 세 명제를 분리한다.

1. 등록된 역상관 scale만으로 물리 입자 pole을 알 수 있는가.
2. 저장소가 선언한 선택적 \(Z_2\) singlet portal action에서는 어떤 pole과
   vertex가 실제로 나오는가.
3. \(29.6991596\,\mathrm{MeV}\)와
   \(v_{\rm EW}\sqrt{\lambda_{HP}}=43.8056765\,\mathrm{GeV}\)를 같은 field의 같은
   pole로 읽을 수 있는가.

결과는 다음과 같다.

| 명제 | 판정 |
|---|---|
| \(m_p\delta_n^2=29.6991596\,\mathrm{MeV}\) 산술 | `EXACT SCALE IDENTITY` |
| 역상관 scale 하나 \(\Rightarrow\) 고립 pole·residue·LSZ | `REFUTED` |
| 선택적 canonical portal EFT의 tree kernel·pole·dispersion | `EXACT CONDITIONAL` |
| 같은 EFT의 \(h\Phi^2,h^2\Phi^2\) 국소 vertex | `EXACT CONDITIONAL` |
| A1 Hessian만으로 cubic/quartic vertex 유일 결정 | `REFUTED` |
| \(m_0^2\ge0,\lambda_{HP}=\delta_n^2\)에서 29.699 MeV same-field pole | `REFUTED` |
| 음의 \(m_0^2\)를 역산해 29.699 MeV tree pole 구성 | `CONSTRUCTIBLE, NOT PREDICTED` |
| renormalized CE pole·양의 spectral weight·LSZ | `OPEN` |
| physical SM 물질 생성률 | `OPEN` |

## 2. 역상관길이는 왜 pole 증명이 아닌가

장거리 감쇠율을 \(m_\xi=1/\xi\)로 쓸 수 있다고 하자. 다음 네 모형은
모두 같은 leading \(e^{-m_\xi r}\) scale을 가질 수 있다.

\[
G_{E,1}(q)=\frac{1}{q^2+m_\xi^2},
\qquad
G_{E,2}(q)=\frac{10^{-12}}{q^2+m_\xi^2},
\]

\[
G_{E,3}(q)=-\frac{1}{q^2+m_\xi^2},
\]

그리고

\[
G_{E,4}(q)
=
\int_{m_\xi^2}^{\infty}
\frac{\rho(\mu^2)}{q^2+\mu^2}\,d\mu^2,
\qquad \rho\ge0,
\]

에서 threshold에만 연속 spectrum을 두고 delta pole을 두지 않을 수도 있다.
첫째와 둘째는 pole 위치가 같지만 residue가 다르고, 셋째는 residue가 음수인
ghost 반례이며, 넷째는 같은 leading exponential threshold를 가지면서 고립
pole이 없다.

따라서 \(\xi\) 하나로는 다음을 식별할 수 없다.

- 고립 pole의 존재
- pole residue의 크기와 부호
- reflection positivity
- continuum cut과 pole 사이의 간격
- 안정한 asymptotic state와 LSZ

LSZ가 Green function의 pole에서 S-matrix 외부 상태를 추출한다는 원래의
조건은 [Lehmann–Symanzik–Zimmermann](https://doi.org/10.1007/BF02731765)에
기초한다. 현대적인 scattering-theory 보강도 asymptotic state 조건이 별도임을
강조한다. [Collins, 2019](https://arxiv.org/abs/1904.10923)

그러므로 현재 29.6991596 MeV의 정직한 provenance는

```text
REGISTERED_INVERSE_CORRELATION_SCALE_ANSATZ
```

다. `PHYSICAL_CLARUS_POLE`이 아니다.

## 3. 저장소에서 실제로 선택된 통제 action

`q0_manifest_gate.py`가 선언한 선택적 고전 bare tree-level control은
\((-+++ )\) signature에서

\[
\mathcal L_\Phi
=
-\frac12\partial_\mu\Phi\partial^\mu\Phi
-\frac12m_0^2\Phi^2
-\frac{\lambda_\Phi}{4}\Phi^4
-\lambda_{HP}|H|^2\Phi^2
\]

다. 이 식은 full CE+SM action이 아니라, 독립 \(Z_2\) singlet을 추가한
선택적 EFT control이다. 실수 singlet Higgs-portal 모형에서는 vacuum
stability·perturbativity·running을 별도로 검사해야 한다는 점도 기존 연구와
같다. [Gonderinger et al.](https://arxiv.org/abs/0910.3167)

\[
H=\frac1{\sqrt2}\begin{pmatrix}0\\v+h\end{pmatrix},
\quad\text{또는 }R_\xi\text{ control에서 }H=\frac{v+h+i\chi}{\sqrt2},
\qquad \langle\Phi\rangle=0
\]

를 대입하면

\[
m_{\rm eff}^2=m_0^2+\lambda_{HP}v^2
\]

이고 singlet quadratic action은

\[
S_\Phi^{(2)}
=
\frac12\int\frac{d^4p}{(2\pi)^4}
\Phi(-p)\left[p^2-m_{\rm eff}^2\right]\Phi(p)
\]

다. 여기서 \(p^2=\omega^2-|\mathbf k|^2\)다. 따라서

\[
K_F(p)=p^2-m_{\rm eff}^2+i0,
\qquad
G_F(p)=\frac{i}{p^2-m_{\rm eff}^2+i0}.
\]

공급된 \(m_{\rm eff}^2>0\) 아래에서는

\[
p^2=m_{\rm eff}^2,
\qquad
\omega^2=|\mathbf k|^2+m_{\rm eff}^2,
\]

이며 \(G_F/i\)의 invariant pole residue는 canonical normalization 때문에
\(+1\)이다. 양의 주파수 \(\omega=E_{\mathbf k}\)에 대한 residue는
\(1/(2E_{\mathbf k})>0\)이다.

이것은 지정한 tree EFT의 정확한 결과다. 그러나 \(m_0^2\)가 외부 입력이고
self-energy, counterterm, scheme, RG scale과 spectral density가 없으므로
`RENORMALIZED_CE_POLE`이나 `LSZ_COMPLETED`로 승격하지 않는다.

### 3.1 다른 형식 action은 현재 대체 입력으로 쓸 수 없음

`docs/참조/형식적_수학_모델과_증명.md`는 \((-+++ )\)에서

\[
\mathcal L_\sigma
=+\frac12(\nabla\sigma)^2-V(\sigma)+f(R)\sigma
\]

를 쓰면서 \(\Box\sigma-V'-f=0\)을 적는다. 그러나 표시된 action을 그대로
변분하면

\[
\Box\sigma+V'(\sigma)-f(R)=0
\]

이며, 시간 kinetic sign도 canonical scalar와 반대다. 또한 문서의 Einstein
방정식에는 \(\sigma f_R R_{\mu\nu}\) 및
\((g_{\mu\nu}\Box-\nabla_\mu\nabla_\nu)(\sigma f_R)\) 같은 metric variation
항이 빠져 있다. 따라서 이 형식 action은 부호·차원·metric variation을 고치기
전에는 CE 물리 two-point kernel의 입력으로 사용하지 않는다.

## 4. 29.699 MeV와 43.806 GeV의 양립성 정리

### 4.1 비음수 bare mass 정리

\(\lambda_{HP}>0\), \(v>0\), \(m_0^2\ge0\)이면

\[
m_{\rm pole}^2=m_0^2+\lambda_{HP}v^2
\ge\lambda_{HP}v^2
\]

이므로

\[
\boxed{m_{\rm pole}\ge v\sqrt{\lambda_{HP}}}.
\]

등록값에서

\[
\lambda_{HP}=\delta_n^2=0.0316530353958,
\qquad v_{\rm EW}=246.21965\,\mathrm{GeV}
\]

를 쓰면

\[
v_{\rm EW}\sqrt{\lambda_{HP}}
=43.8056765\,\mathrm{GeV}.
\]

이는 light-bridge 목표 \(0.0296991596\,\mathrm{GeV}\)의

\[
\frac{43.8056765}{0.0296991596}=1474.9803
\]

배다. 따라서 `nonnegative bare mass + same field + lambda_HP=delta_n^2` 분기에서
29.699 MeV pole은 수학적으로 불가능하다. 현재 정본은 이 두 수치를
각각 `light inverse-correlation bridge`와 \(m_0=0\)을 선택한
`portal-dominance benchmark`로 분리한다.

### 4.2 음의 bare mass를 허용하면

목표를 강제로 맞추는 유일한 bare parameter는

\[
m_0^2
=(0.0296991596)^2-\lambda_{HP}(246.21965)^2
=-1918.9364090\,\mathrm{GeV}^2
\]

다. portal shift는

\[
\lambda_{HP}v_{\rm EW}^2=1918.9372910\,\mathrm{GeV}^2
\]

이므로 남는 비율은

\[
\frac{m_{\rm target}^2}{\lambda_{HP}v_{\rm EW}^2}
=4.5965029\times10^{-7}.
\]

즉 약 \(2.17557\times10^6:1\)의 squared-mass 상쇄다. 이는 일반 EFT
대수에서 금지되지는 않지만 다음 두 주장을 동시에 버리게 한다.

- \(m_0^2\ge0\)인 symmetric bare-mass 분기
- \(\lambda_{HP}v^2\gg|m_0^2|\)인 portal-dominance 근사

음의 \(m_0^2\) 자체가 EWSB 뒤 \(Z_2\) 진공을 자동으로 깨는 것은 아니다.
tree potential의 모든 radial stationary point를 열거하면, 공급 control
\(\lambda_\Phi=0.1\)과
\(\lambda_H=m_h^2/(2v^2)\)에서는 \((v,0)\)가 local이면서 global minimum이다.
singlet-only minimum보다 낮기 위한 하한은

\[
\lambda_\Phi>
\frac{(m_0^2)^2}{\lambda_Hv^4}
\simeq0.0077611
\]

다. 따라서 이 분기는 수학적으로 구성 가능하지만, 큰 상쇄의 기원과 loop·thermal
vacuum stability는 여전히 미유도다.

반대로 \(m_0^2\ge0\)을 유지하려면

\[
\lambda_{HP}
\le\frac{m_{\rm target}^2}{v_{\rm EW}^2}
=1.4549327\times10^{-8}
\]

이어야 한다. 이는 \(\delta_n^2\)보다 약 \(2.17557\times10^6\)배 작다.

따라서 현재 정본은 29.6991596 MeV inverse-correlation bridge와
43.8056765 GeV 선택적 EW portal pole을 서로 다른 scale/branch로 유지한다.
같은 field라고 주장하려면 위 상쇄의 동역학과 RG 안정성을 새로 유도해야 한다.

## 5. Hessian만으로 vertex가 나오지 않는 정확한 반례

배경 \(\bar\varphi\) 주위에서 임의 action \(S\)에

\[
\Delta S
=
\frac{C}{3!}(\varphi-\bar\varphi)^3
+\frac{D}{4!}(\varphi-\bar\varphi)^4
\]

를 더하자. 배경에서

\[
\Delta S'=0,
\qquad
\Delta S''=0,
\qquad
\Delta S'''=C,
\qquad
\Delta S''''=D.
\]

따라서 gradient와 Hessian은 완전히 같지만 cubic과 quartic vertex는 임의로
바뀐다. 이 반례로

\[
\boxed{\text{A1 quadratic Hessian alone does not identify a production vertex}}
\]

가 증명된다. vertex를 얻으려면 선택적 portal action 같은 higher action jet을
별도로 채택해야 한다.

## 6. 선택적 portal action에서 실제로 남는 vertex

portal 항을 전개하면

\[
\mathcal L_{HP}
=
-\frac{\lambda_{HP}v^2}{2}\Phi^2
-\lambda_{HP}v h\Phi^2
-\frac{\lambda_{HP}}2(h^2+\chi^2)\Phi^2.
\]

따라서 \((h,\Phi)=(0,0)\)에서

\[
\frac{\partial^2\mathcal L}{\partial h\partial\Phi}=0,
\]

이지만

\[
\frac{\partial^3\mathcal L}{\partial h\partial\Phi^2}
=-2\lambda_{HP}v
=-15.5601447\,\mathrm{GeV},
\]

\[
\frac{\partial^4\mathcal L}{\partial h^2\partial\Phi^2}
=-2\lambda_{HP}
=-0.063196104
\]

이고 \(R_\xi\) control에서는

\[
\frac{\partial^4\mathcal L}{\partial\chi^2\partial\Phi^2}
=-2\lambda_{HP}
\]

도 함께 남는다.

는 0이 아니다. 즉 정확한 \(Z_2\) 아래에서

- single-\(\Phi\) source와 \(h\)-\(\Phi\) bilinear mixing은 없다.
- \(h\to\Phi\Phi\)와 역과정 \(\Phi\Phi\to h^*\)의 bare local pair vertex는
  있다.
- 기존 toy의 직접 \(\Phi^2\chi^2\) daughter vertex는 이 action에서 나오지
  않는다.

그러므로 공명 아이디어에서 살아남는 실제 portal 경로는

\[
\Phi+\Phi\longrightarrow h^*\longrightarrow \mathrm{SM}
\]

후보다. 하지만 physical \(\Phi\) external state, pump amplitude와 분포,
off-shell Higgs propagator, SM final-state matrix element, phase space와 에너지
장부가 없으므로 생성률은 아직 0개도 계산되지 않았다.

## 7. 공급된 비가시폭 gate

정확한 \(Z_2\)로 light \(\Phi\)가 안정하고 \(m_\Phi<m_h/2\)이면

\[
\Gamma(h\to\Phi\Phi)
=
\frac{\lambda_{HP}^2v^2}{8\pi m_h}
\sqrt{1-\frac{4m_\Phi^2}{m_h^2}}.
\]

이 gate에서는 light bridge를 포탈 field 질량으로 넣지 않는다. 정본
portal-dominance 분기의
\(m_0=0\), \(\lambda_{HP}=\delta_n^2=0.0316530354\),
\(m_\Phi=43.8056765\,\mathrm{GeV}\),
\(m_h=125.11\,\mathrm{GeV}\),
\(\Gamma_h^{SM}=4.10\,\mathrm{MeV}\)를 넣으면

\[
\Gamma_{\rm inv}=13.7900417\,\mathrm{MeV},
\qquad
\mathrm{BR}_{\rm inv}=0.7708222.
\]

따라서 PDG 2026 Higgs review §11.4.3이 요약한 ATLAS Run-2 direct observed
\(\mathrm{BR}_{\rm inv}<0.107\) (95% CL) gate를 통과하지 못한다.

29.6991596 MeV light bridge를 같은 portal field의 고정 질량으로 강제하는
**별도 same-field stress-test**에서는
\(\Gamma_{\rm inv}=19.3172016\,\mathrm{MeV}\),
\(\mathrm{BR}_{\rm inv}=0.8249150\)이다. 이 질량을 고정한 채 0.107 gate를
역산하면 \(|\lambda_{HP}|<0.00504779076\)이며, 정본
\(\delta_n^2\)는 이보다 6.27067배 크다. 이는 \(m_0=0\)인 canonical
portal-dominance 관계 \(m_\Phi=v_{\rm EW}\sqrt{\lambda_{HP}}\)와 다른 반사실적
분기이며, 두 판정을 섞지 않는다. 이는 선택적 portal branch의
판정이며, 상한 자체의 최신 global-fit 재분석이나 CE 코어 반증은 아니다.

## 8. 루프 점수

아래 점수는 참일 확률이 아니라 현재 증명 의무의 완료율이다.

| 항목 | 점수 | 근거 |
|---|---:|---|
| 역상관 scale 산술 | 100/100 | 단위와 수치 항등식 |
| 역상관 scale의 physical pole 식별 | 0/100 | pole/ghost/continuum 반례 |
| 선택적 EFT tree kernel·pole | 95/100 | bare tree control exact; loop/RG 제외 |
| 선택적 EFT tree residue·dispersion | 95/100 | canonical control exact; physical holdout 제외 |
| A1에서 vertex 유일 유도 | 0/100 | 동일 Hessian·상이한 higher jet 반례 |
| 선택적 portal local vertex | 95/100 | bare derivative exact; renormalized form factor 제외 |
| 29.699 MeV와 portal dominance의 same-field 양립 | 0/100 | 비음수 bare-mass 정리로 반증 |
| physical CE pole·LSZ | 0/100 | CE \(\Gamma_R^{(2)}\), cut, \(Z_*\) 없음 |
| physical 물질 생성률 | 0/100 | external pole certificate와 matrix element 없음 |

## 9. 현재 hard blocker

물리 pole을 닫으려면 최소한

\[
\det\Gamma^{(2)}_{CE,R}(p_*)=0,
\]

\[
Z_*^{-1}
=
v_*^\dagger
\left.\frac{\partial\Re\Gamma^{(2)}_{CE,R}}
{\partial p^2}\right|_{p_*}
v_*>0
\]

와 pole–cut 분리, gauge/scheme/RG holdout, spectral positivity가 필요하다.
그 뒤 동일 action과 normalization에서 renormalized 1PI vertex를 계산하고
external leg에 같은 pole certificate를 연결해야 한다.

현재는 full CE action과 \(\Gamma^{(2)}_{CE,R}\)가 없으므로 다음 루프는
숫자 fitting이 아니라 action provenance를 먼저 닫아야 한다.
그 후속 감사와 scalar one-loop 방사 안정성 결과는
[CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md](CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md)에
기록했다.

## 10. 재현

2026-08-04 역사적 전체 회귀 기록:

- `1296 passed, 13 skipped, 0 failed`
- 작업 트리에서 이미 삭제되어 있던 fixture에 직접 의존하는 테스트 파일 5개만 명시적으로 제외했다:
  `local_memory`, `origin_life_branching`, `origin_life_coupled`, `q0_manifest`,
  `neural_tree_algorithm_census`.
- 시스템 임시 폴더의 권한 오류를 피하려고 저장소 안의 일회성 `--basetemp`를 사용했고,
  검증 뒤 그 임시 디렉터리는 제거했다.
- `git diff --check`는 공백 오류 없이 통과했다. 표시된 내용은 CRLF 변환 경고뿐이다.

2026-08-06 수치 계약은
`CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`에서 재계산했다. 아래 runtime
gate의 기본 인자는 2026-08-04 legacy fixture를 유지하므로, 현재 문서의
29.6991596 MeV·43.8056765 GeV·13.7900417 MeV 수치를 구 fixture 출력과
혼합하지 않는다.

```powershell
uv --cache-dir .uv-cache run python examples/physics/ce_two_point_vertex_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_ce_two_point_vertex_certificate.py `
  tests/test_a1_q0_action_bridge.py `
  tests/test_resonance_stress_identifiability.py `
  tests/test_clarus_boson_search_gate.py -q
```

핵심 구현:

- `reality_stone/python/reality_stone/clarus/ce_two_point_vertex_certificate.py`
- `examples/physics/ce_two_point_vertex_gate.py`
- `tests/test_ce_two_point_vertex_certificate.py`

## 11. 다음 우선순위

1. 선택적 portal control과 CE core를 같은 field로 볼지 분기부터 결정한다.
2. 같은 field 분기라면 full CE action, background, counterterm와 matching scale을
   명시하고 \(\Gamma_R^{(2)}\)를 계산한다.
3. 별도 field/scale 분기라면 29.699 MeV와 43.806 GeV 기호·API를 완전히 분리한다.
4. physical pole이 실제로 닫힌 뒤 \(\Phi\Phi\to h^*\to\mathrm{SM}\)의
   on-shell/continuum rate와 pump depletion을 계산한다.
5. 생성된 SM matter의 density·lifetime·response를 계산한 뒤에만 Casimir stress
   및 throat backreaction으로 돌아간다.
