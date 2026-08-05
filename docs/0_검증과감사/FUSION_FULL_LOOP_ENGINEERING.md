# CE 핵융합 전분기 루프 엔지니어링

작성일: 2026-08-05  
코드: `reality_stone/python/reality_stone/clarus/fusion_full_loop.py`  
실행: `examples/physics/fusion_full_loop_gate.py`  
테스트: `tests/test_fusion_full_loop.py`

## 1. 범위와 최종 판정

이 문서는 [핵융합 공명 1차 감사](FUSION_RESONANCE_LOOP_ENGINEERING.md)가
남긴 모든 후보 분기를 같은 fail-closed 규칙으로 끝까지 추적한다.

1. 정본 $Z_2$ 보존 portal의 쌍-scalar 분기
2. 명시적 $Z_2$ 파괴와 단일-scalar 정적 교환
3. 외부 source가 만드는 coherent 시간주기 배경
4. D--T 단면적에서 Maxwellian reactivity와 Lawson 장부로 가는 분기
5. NIF 캡슐·방사유체역학·점화에너지 분기

결론은 다음과 같다.

\[
\boxed{
\texttt{STANDARD DT BASELINE + SOURCE ENERGY NEGATIVE CONTROLS}
\neq
\texttt{CE-MODIFIED DT AMPLITUDE}
}
\]

즉, 선언된 계산 분기는 모두 검사했지만 현재 입력으로 물리적인 CE 수정 D--T
진폭, 수정 열반응률, 수정 Lawson 조건, NIF 점화에너지는 하나도 유도되지 않는다.

## 2. 정본 \(Z_2\) 쌍-scalar 분기

정본 작용에는 $h\Phi^2$ tree vertex가 있지만 단일 Φ source는 없다. 기존
Q0.4--Q0.5 certificate와 결합한 수치는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| $h\Phi^2$ tree vertex | `True` |
| 단일 Φ source | `False` |
| 두-scalar cut 문턱 $2m_\Phi$ | `59.29514 MeV` |
| 장거리 감쇠 scale $\hbar c/(2m_\Phi)$ | `3.327877806 fm` |
| $m_0^2=0$ portal pole | `43.76767547 GeV` |
| 공급 benchmark의 Higgs invisible BR | `0.8253120442` |
| 공급 상한 | `0.11` |
| 상한 아래 최대 $|\lambda_{HP}|$ | `0.005110743` |

따라서 pair vertex의 존재는 조건부 tree-level 사실이지만, 29.65 MeV pole이 같은
portal 작용에서 예측되거나 portal-dominated라는 뜻은 아니다. 더구나 두-scalar
exchange의 renormalized 핵자 진폭과 표준 D--T R-matrix에 대한 잔차는 아직 없다.
이 분기는 `TREE_PAIR_VERTEX_ONLY_PHYSICAL_FUSION_BRANCH_NOT_REACHED`에서 멈춘다.

## 3. 명시적 \(Z_2\) 파괴 분기

레거시 입력 $|\sin\theta|=0.04344$를 문서에 공급된 상한 0.0038과 비교하면

\[
\frac{0.04344}{0.0038}=11.4316,
\qquad
\left(\frac{0.04344}{0.0038}\right)^2=130.681.
\]

그러므로 레거시 benchmark는 공급 상한을 통과하지 못한다. 비교를 위해 허용
상한까지 coupling을 낮춘 정적 Yukawa 힘은 핵반경에서 Coulomb 힘의
$1.2671\times10^{-10}$뿐이다. 레거시 입력 자체도
$1.6558\times10^{-8}$이다. 두 값 모두 timelike line의 품질계수 $Q$를
곱하지 않았다. 정적 전달량은 spacelike이므로 그것이 올바른 propagator
판정이다. 이 분기는 `LEGACY_MIXING_REJECTED_BY_SUPPLIED_LIMIT`다.

상한 0.0038은 이 저장소가 공급한 입력이며 최신 전역 likelihood를 새로 맞춘
결과로 승격하지 않는다. 새 benchmark를 주장하려면 동일 질량에서 생산·붕괴
likelihood와 비공명 D--T 진폭을 함께 다시 계산해야 한다.

## 4. coherent 시간주기 배경 분기

외부 source가 실제로 장을 준비한다는 가정 아래, 핵자 질량의 1% 진동을 만드는
데 필요한 최소 prescribed-field 규모를 음의 대조군으로 계산했다.

| 항목 | 결과 |
|---|---:|
| $g_{\Phi NN}$ | `4.970553562e-5` |
| 필요한 장 진폭 $|\Phi_0|$ | `1.887661142e5 MeV` |
| 자유장 에너지밀도 | `3.265576509e38 J/m^3` |
| 양자수 밀도 | `6.874705009e49 m^-3` |
| 진공 수명 | `6.865e-8 s` |
| 수명마다 보충하는 power density | `4.756905614e45 W/m^3` |
| 구동 주파수 / 20 keV D--T 핵반경 통과주파수 | `12.99128948` |

이는 source 설계가 아니라, source를 생략할 수 없음을 보이는 에너지 scale
control이다. source current, 공간 mode, coherent-state preparation, pump work,
decay/backreaction 중 어느 것도 유도되지 않았다. 시간주기 장에서 반응률을
주장하려면 단순 정적 WKB가 아니라 실제 time-dependent scattering 계산이
필요하다. 동적 보조 핵융합 연구도 Floquet/Volkov, Kramers--Henneberger,
Crank--Nicolson 같은 시간의존 방법을 분리해 다룬다
([Phys. Rev. C 109, 044605](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.109.044605)).

따라서 이 분기는 `PRESCRIBED_BACKGROUND_ENERGY_SCALE_CONTROL_ONLY`다.

## 5. Bosch--Hale 열반응률과 Lawson 기준선

표준 D--T 기준선은 Bosch--Hale parametrization
([Nuclear Fusion 32 (1992)](https://www.osti.gov/etdeweb/biblio/5161054))으로
독립 재현했다. $T=10\,\mathrm{keV}$에서

\[
\theta=11.9356225\,\mathrm{keV},\qquad
\xi=2.9146850,
\]

\[
\langle\sigma v\rangle_{DT}
=1.136165471\times10^{-16}\ \mathrm{cm^3/s}.
\]

알파 에너지 $E_\alpha=3.52\,\mathrm{MeV}$, 등온 0차원 손실 장부
$n\tau=12T/(E_\alpha\langle\sigma v\rangle)$를 적용한 baseline은

\[
n\tau=3.000523249\times10^{14}\ \mathrm{cm^{-3}s}.
\]

이 값은 표준 기준선 통과일 뿐이다. CE 후보 σ(E)가 공급되지 않았으므로 WKB
penetrability factor를 σ(E)나 〈σv〉에 곱하지 않는다. 따라서 수정 reactivity와
수정 Lawson 값은 모두 `False`다.

## 6. NIF 캡슐·점화 분기

NIF 기준 입력 2.05 MJ laser와 약 3.1 MJ yield는 target gain 약 1.5를 준다.
해당 결과는 target, laser, hohlraum/capsule 설계와 실험 개선 전체의 결과다
([Phys. Rev. Lett. 132, 065102](https://www.osti.gov/biblio/2283994)).

반사실 WKB factor 40.85172379로 laser 에너지를 선형 나누면 50.1815 kJ라는
숫자가 나오지만, 코드는 이를 명시적으로 `rejected_linear_rescale_energy_kj`에만
기록한다. laser-to-hotspot coupling, implosion symmetry, mix, alpha heating,
radiation/conduction loss와 radiation hydrodynamics가 없기 때문에 이것은 점화
예측이 아니다. 이전 7.4 kJ 주장은 계속 철회 상태다.

## 7. 최종 gate ledger

| Gate | 판정 |
|---|---|
| 정본 $Z_2$ 단일-scalar portal | `CLOSED_OFF` |
| 정본 $Z_2$ pair vertex | `CONDITIONAL_PASS` |
| renormalized pair D--T amplitude | `NOT_REACHED` |
| 레거시 명시적 $Z_2$ 파괴 benchmark | `REJECT` |
| coherent background energy scale | `NEGATIVE_CONTROL` |
| source-normalized Floquet D--T scattering | `NOT_REACHED` |
| Bosch--Hale 표준 D--T baseline | `PASS` |
| 수정 reactivity와 Lawson | `NOT_REACHED` |
| NIF radiation-hydrodynamic gain | `NOT_REACHED` |

다음 승격 gate는 하나다. 실험 제약을 통과하는 renormalized CE vertex와
source-normalized field를 먼저 제시하고, 표준 D--T 진폭에 대한 부호가 고정된
잔차를 계산해야 한다. 그 전에는 thermal 또는 ICF 단계로 승격하지 않는다.

## 8. 재현 명령

```powershell
uv --cache-dir .uv-cache run python `
  examples/physics/fusion_full_loop_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_fusion_full_loop.py -q
```
