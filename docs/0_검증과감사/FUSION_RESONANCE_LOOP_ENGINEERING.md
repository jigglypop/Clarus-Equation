# CE 핵융합 공명 루프 엔지니어링 감사

작성일: 2026-08-05  
코드: `reality_stone/python/reality_stone/clarus/fusion_resonance_loop.py`  
실행: `examples/physics/fusion_resonance_loop_gate.py`  
테스트: `tests/test_fusion_resonance_loop.py`

## 1. 이번 루프의 질문과 판정

기존 핵융합 문서는 다음 사슬을 제안했다.

\[
29.6991596\ {\rm MeV\ inverse\!-\!correlation\ candidate}
\xrightarrow{Q}
Q\times V_{\rm Yukawa}(r)
\rightarrow
\text{D--T 장벽 감소}
\rightarrow
\langle\sigma v\rangle 증가
\rightarrow
\text{NIF 점화에너지 감소}.
\]

루프 엔지니어링에서는 각 화살표를 독립 gate로 분리한다. 이번 루프의 최종
판정은 다음이다.

\[
\boxed{
\texttt{LEGACY COUNTERFACTUAL WKB CONTROL ONLY}
\neq
\texttt{PHYSICAL RESONANT BARRIER REDUCTION}
\neq
\texttt{IGNITION ENERGY PREDICTION}
}
\]

레거시 수치 일부는 재현되지만 핵심 화살표인
`timelike pole Q -> static spacelike potential`이 기각된다. 따라서 기존의
`7.4 kJ`는 현재 지원되는 결론이 아니다.

## 2. 루프 0 — 정본 Z2 분기

현재 정본 portal은

\[
\mathcal L\supset-\lambda_{HP}|H|^2\Phi^2,
\qquad \Phi\to-\Phi,
\qquad \langle\Phi\rangle=0
\]

이다. 이 분기에서는 `h-Phi` 이중선형 혼합과 단일 `Phi`-핵자 Yukawa 결합이
없다.

\[
\sin\theta=0,
\qquad g_{\Phi NN}=0.
\]

따라서 기존 문서의 단일 scalar 교환력은 정본 분기에서 `CLOSED_OFF`다. 가능한
`Phi Phi` 쌍 교환 또는 `Phi Phi -> h* -> SM`은 별도 고차 vertex 문제이며
단일-pole 공명식을 상속하지 않는다. 기존 portal one-loop control도 29.65 MeV
light pole이 방사적으로 안정한 CE 예측임을 입증하지 못했다.

## 3. 루프 1 — 레거시 line 산술의 교정

비정본 비교를 위해 `sin(theta)=0.04344`를 조건부로 공급하면 전자쌍 폭은

\[
\Gamma_{e e}
=\sin^2\theta\frac{m_e^2m_\Phi}{8\pi v^2}
\left(1-\frac{4m_e^2}{m_\Phi^2}\right)^{3/2}
=9.58805\times10^{-15}\ {\rm MeV}
\]

이고 `Q_vac=3.09218e15`, `tau=68.65 ns`는 재현된다. 이 폭의 형태는 실제
Higgs-portal scalar 탐색에서도 사용된다
([MicroBooNE, arXiv:2106.00568](https://arxiv.org/abs/2106.00568)).

그러나 각주파수와 보통 주파수를 구분해야 한다.

| 양 | 교정값 |
|---|---:|
| `omega=m/hbar` | `4.50432e22 rad/s` |
| `nu=m/h` | `7.16885e21 Hz` |
| `Gamma/hbar` | `1.45668e7 rad/s` |
| `Gamma/h` | `2.31838 MHz` |

문서의 충돌 단면적 ansatz를 물리식으로 승인하지 않고 단위만 교정하면

\[
\sigma_{\Phi e}^{\rm ansatz}
=\alpha\sin^2\theta\left(\frac{\hbar c}{m_\Phi}\right)^2
=6.09996\times10^{-34}\ {\rm m^2},
\]

\[
\hbar n_e\sigma v=2.38495\times10^{-15}\ {\rm MeV}
\]

이다. 기존 `1.57e-53 m^2` 및 “진공폭보다 20자릿수 작다”는 단위 오류다. 이
ansatz 아래에서도 `Q_plasma=2.47624e15`로 진공값과 같지 않다. 실제 plasma
self-energy를 유도하지 않았으므로 이 값도 `CONDITIONAL ANSATZ`다.

같은 매개변수의 표준 scalar one-loop 전자 자기모멘트 적분은

\[
\Delta a_e
=\frac{y_e^2}{8\pi^2}\int_0^1dx\,
\frac{(1-x)^2(1+x)}{(1-x)^2+x(m_\Phi/m_e)^2}
=2.1325\times10^{-19},
\]

이며 기존 `7e-9`가 아니다.

## 4. 루프 2 — 결정적 spacelike 반례

폭이 있는 scalar propagator의 pole은 개략적으로

\[
D(q)=\frac{i}{q^2-m_\Phi^2+i m_\Phi\Gamma_\Phi}
\]

에 있다. on-shell 공명에는 `q^2=m_Phi^2>0`인 시간꼴 전달이 필요하다. 반면
정적 핵간 힘은 `q0=0`이므로

\[
q^2=(q^0)^2-|\mathbf q|^2=-|\mathbf q|^2<0.
\]

핵반경 `3.24 fm`의 대표 운동량은 약 `60.90 MeV`이고 구현 gate의 불변량은
`-3.709e3 MeV^2`다. 양의 pole `m_Phi^2=879.0 MeV^2`와 만날 수 없다.

따라서

\[
V_{\rm res}(r)=Q\,V_{\rm static}(r)
\]

는 propagator에서 나오지 않는다. 별도 외부원으로 시간의존 coherent background를
만들 수는 있지만, 그것은 먼저 `Phi_bg(t,x)`와 source work를 풀어야 하는
one-body 질량/힘 배경이다. 두 핵자의 정적 pair potential에 `Q`를 곱한 것과
같지 않다.

## 5. 루프 3 — 반사실 WKB 계산의 정확한 경계

물리 bridge 실패와 별개로 기존 toy potential

\[
V(r)=\frac{\alpha\hbar c}{r}
-Q\alpha_\Phi\frac{\hbar c\,e^{-r/\xi}}r
\]

을 그대로 계산했다. `E_cm=20 keV`에서 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| baseline exponent `gamma_0` | `2.81364784` |
| `Q=1e9` exponent | `0.95867333` |
| `Q=1e9`의 반사실 증폭 | `40.8517` |
| 핵반경에서만 Coulomb 상쇄하는 Q | `6.039e7` |
| 전체 장벽의 최대값이 20 keV가 되는 Q | `6.297e10` |
| 후자의 형식적 분해능 `1/Q` | `1.588e-11` |

전체 장벽 임계값은 수치 fit 없이도 닫힌다. `x=r/xi`에서 `V'=0`과 `V=E`를
동시에 쓰면

\[
x_*=\frac{\alpha\hbar c}{\xi E}-1,
\qquad
Q_{\rm whole}
=\frac{\alpha e^{x_*}}{(1+x_*)\alpha_\Phi}.
\]

기존 `Qcrit=1e9`는 핵반경 부근의 상쇄와 바깥에 남은 Coulomb hump를 혼동했다.
문서 자체의 `Q=1e9, gamma=0.9587, Sigma=40.82` 행이 이 반례를 이미 담고
있었다.

## 6. 루프 4 — WKB에서 점화로의 승격 금지

D--T 반응률은 단일 에너지의 penetrability만이 아니라 핵반응 amplitude와
분포를 평균한

\[
\langle\sigma v\rangle
=\int d^3v_1d^3v_2\,f_1f_2\,\sigma(v_{12})v_{12}
\]

가 필요하다. 표준 실무도 측정과 R-matrix에 기초한 cross section 및 Maxwellian
reactivity parametrization을 사용한다
([Bosch--Hale, Nuclear Fusion 32 (1992)](https://www.osti.gov/etdeweb/biblio/5161054)).

그 뒤에도 ICF 점화에는 laser-hohlraum coupling, capsule symmetry, ablator mix,
compression, alpha heating, radiation/conduction loss가 남는다. 2022 NIF 결과는
`2.05 MJ`의 `351 nm` laser로 `3.1 MJ`, target gain `1.5`를 얻었고 논문은 target,
laser, design 및 experimental advancement 전체를 다룬다
([PRL 132, 065102](https://www.osti.gov/biblio/2283994)).

따라서

\[
E_{\rm laser,new}=E_{\rm NIF}/\Sigma
\]

는 도출식이 아니다. 이번 루프는 `thermal_reactivity_derived`,
`nif_capsule_gain_derived`, `ignition_energy_derived`를 모두 `False`로 잠근다.

## 7. 현재 stage ledger

| Gate | 상태 | 이유 |
|---|---|---|
| canonical Z2 linear nucleon portal | `CLOSED_OFF` | `sin(theta)=0`, 단일 scalar force 없음 |
| legacy scalar line arithmetic | `CONDITIONAL PASS` | 폭·Q·단위·`a_e` 재현 |
| static spacelike resonance | `REJECT` | `q^2<0`이므로 timelike pole 불가 |
| driven background source/energy ledger | `OPEN` | source amplitude·geometry·work·backreaction 없음 |
| `Q times Yukawa` WKB | `CONDITIONAL PASS` | 반사실 toy로만 재현 |
| Maxwellian D--T reactivity | `NOT REACHED` | 수정 scattering amplitude 없음 |
| ICF capsule/ignition energy | `NOT_REACHED` | radiation-hydrodynamic model 없음 |

최대 지원 단계는

```text
LEGACY_COUNTERFACTUAL_WKB_CONTROL_ONLY
```

이다.

## 8. 살아남는 후속 연구 분기

다음 루프는 기존 `Q x static potential`을 더 정밀하게 계산하는 작업이 아니다.
아래 중 하나의 새로운 물리 bridge를 먼저 닫아야 한다.

1. **Z2 쌍 분기**: `Phi Phi -> h* -> D/T`의 renormalized amplitude를 유도하고
   two-scalar exchange가 표준 D--T R-matrix amplitude에 주는 잔차를 계산한다.
2. **명시적 Z2 파괴 분기**: 허용되는 혼합각·수명을 외부 실험 likelihood와 함께
   고정한 뒤, spacelike 교환의 비공명 수정만 계산한다.
3. **구동 배경 분기**: 실제 source current, 공간 mode, coherent occupation,
   linewidth, pump work와 decay/backreaction을 함께 풀고 D--T scattering의
   시간주기 Floquet amplitude를 계산한다.
4. 어느 분기든 `sigma(E)`를 먼저 산출하고 Bosch--Hale 대조군과 같은 thermal
   distribution에서 `delta<sigma v>`를 계산한다. 이 gate 전에는 Lawson/NIF로
   이동하지 않는다.

가장 정보가 큰 다음 질문은 “Q를 얼마나 키울 수 있는가”가 아니라 다음이다.

> CE 작용에서 유도되고 현재 입자 제약을 통과하는 vertex가 표준 D--T 산란
> amplitude에 만드는 부호와 크기가 고정된 잔차는 무엇인가?

## 9. 재현 명령

```powershell
uv --cache-dir .uv-cache run python `
  examples/physics/fusion_resonance_loop_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_fusion_resonance_loop.py -q
```

테스트는 `rad/s`--`Hz`, 충돌 단위, electron `g-2`, spacelike pole 반례,
핵반경 상쇄와 전체 장벽 제거의 분리, `Q=1e9` WKB, 모든 물리 claim-lock 및
NaN/Inf/bool/음수 입력을 검사한다.
