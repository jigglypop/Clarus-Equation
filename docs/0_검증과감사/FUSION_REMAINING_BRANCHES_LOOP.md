# CE 핵융합 잔여분기 최종 루프

작성일: 2026-08-05  
코드: `reality_stone/python/reality_stone/clarus/fusion_remaining_branches_loop.py`  
실행: `examples/physics/fusion_remaining_branches_gate.py`  
테스트: `tests/test_fusion_remaining_branches_loop.py`

## 1. 범위

[식-수정 반복 루프](FUSION_EQUATION_ITERATION_LOOP.md)는 허용 portal 정적 식이
1% 열반응률 목표를 달성하지 못하고, 새 직접 핵자 결합만 수학적 목표에 도달함을
보였다. 이 문서는 남은 항목을 모두 검사한다.

1. 직접 $\Phi\bar NN$ 연산자의 등록 질량 해와 massless 하한
2. 질량비례 gauge-invariant completion의 유효 scale
3. 핵물질 mean-field와 필요한 실험 likelihood
4. 시간의존 강전기장 대조군의 에너지·주파수·D--T 상대 quiver
5. CE scalar source와 published electromagnetic control의 비동일성
6. 허용 반응률 상한의 Lawson·fusion power·NIF 경계 전파

## 2. 직접 핵자 연산자

저에너지 연산자

\[
\mathcal L_{\rm direct}=-g_N\Phi\bar NN
\]

로 10 keV 열반응률 1% 증가를 역산한 결과는 다음과 같다.

| scalar 질량 | 필요한 $g_N$ | Higgs-mixing 등가값 |
|---|---:|---:|
| 0, 낙관적 하한 | `0.00569352` | `4.97583` |
| 29.6991596174 MeV | `0.0174469513` | `15.2477` |

등록 질량에서 결합은 perturbative이지만, 이것만으로 물리적 허용을 뜻하지 않는다.
현재 정본 portal 작용에는 직접 연산자가 없으며 Higgs mixing으로도 만들 수 없다.

질량비례 quark operator를 가정해

\[
\mathcal L\supset-\frac{\Phi}{\Lambda}
\sum_q m_q\bar q q,
\qquad
g_N\simeq f_N\frac{m_N}{\Lambda}
\]

로 matching하면 등록 질량 해는

\[
\Lambda_{\rm req}=16.1336\ {\rm GeV},
\qquad
\frac{\Lambda_{\rm req}}v=0.065584.
\]

이는 scalar 질량보다 543.23배 높아 저에너지 수치 EFT의 scale 분리는 있지만,
electroweak 대칭 위에서 적분해 낸 무거운 completion으로 보기에는 $\Lambda<v$다.
따라서 UV completion을 공급하지 않은 채 gauge-invariant 성공으로 세지 않는다.

핵포화밀도 $n_0=0.16\,\mathrm{fm^{-3}}$에서 uniform scalar mean-field의 에너지
진단값

\[
\frac{|\Delta\mathcal E|}{n_0}
=\frac{g_N^2n_0}{2m_\Phi^2}=0.212129\ {\rm MeV/nucleon}
\]

도 작지 않다. 이는 곧바로 배제를 증명하는 값은 아니지만 기존 NN phase shift와
핵결합에 흡수해도 되는지 재적합이 필수임을 뜻한다. light scalar의 nucleon
coupling은 실제로 neutron scattering·원자·뮤온 계에서 독립 제약을 받는다
([neutron--nucleus 원 논문](https://doi.org/10.1016/0370-2693(75)90073-8),
[model-independent scalar coupling 분석](https://arxiv.org/abs/1605.04612)).

현재 저장소에는 동일 proton/neutron coupling, 29.6991596174 MeV에서의 NN-scattering,
nuclear-binding, rare-decay joint likelihood가 없다. 그러므로 이 연산자는
`LOW_ENERGY_MATH_SOLUTION_UV_AND_EXPERIMENTAL_GATES_FAIL_CLOSED`다.

후속 [저에너지 핵산란 루프](FUSION_DIRECT_SCATTERING_LOOP.md)는 자유 Born
$np$ scattering-length 이동 `-0.00508459 fm`과 Hulthén deuteron 이동
`-2.08101 keV`를 얻었다. 보고된 triplet/singlet 산란길이 오차의 각각
3.3897/1.1825배 규모이므로
무시할 수 있다는 gate는 실패하지만, strong distorted-wave와 few-body 재적합이
없어 exclusion도 아직 유도하지 않는다.

## 3. 시간의존 electromagnetic 대조군

동적 보조 핵융합 연구는 1 keV D--T plasma에서 유의미한 enhancement에
$10^{15}$--$10^{16}$ V/m, photon energy 1 keV 이하가 필요하다고 보고하며
Floquet/Volkov와 Crank--Nicolson을 비교한다
([Phys. Rev. C 109, 044605](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.109.044605)).
이 범위를 CE scalar 결과로 가져오지 않고 electromagnetic 대조군으로만 재현했다.

| 항목 | 값 |
|---|---:|
| $10^{15}$ V/m의 EM 에너지밀도 | `4.42709e18 J/m^3` |
| $10^{16}$ V/m의 EM 에너지밀도 | `4.42709e20 J/m^3` |
| $10^{16}$ V/m plane-wave intensity | `1.32721e29 W/m^2` |
| D--T 상대유효전하 $q_{eff}/e$ | `0.19923` |
| 1 keV, $10^{16}$ V/m 상대 quiver | `68.9787 fm` |

D와 T의 charge는 같지만 질량이 달라 상대좌표에

\[
\frac{q_{\rm eff}}e=\frac{m_T-m_D}{m_T+m_D}
\]

가 남는다. 상대 quiver는

\[
a_{\rm rel}=\frac{q_{\rm eff}E_0}{\mu\omega^2}
\]

로 계산했다.

## 4. CE scalar 주파수와 source 장부

29.6991596174 MeV scalar의 각주파수는 $4.51210\times10^{22}$ rad/s이고
양자에너지는 29,699.1596 keV다. published control의 1 keV ceiling보다
29,699.1596배 높다.

같은 $10^{16}$ V/m electromagnetic 대조군을 이 주파수에 대입하면 상대 quiver는

\[
a_{\rm rel}=7.82036\times10^{-8}\ {\rm fm}
\]

뿐이다. 핵반경 3.24 fm quiver에는

\[
E_0=4.14303\times10^{23}\ {\rm V/m}
\]

가 필요하고 대응 EM 에너지밀도는 약 $7.599\times10^{35}\,\mathrm{J/m^3}$다.
이 계산은 scalar를 photon으로 동일시하지 않는다. 오히려 electromagnetic control을
CE source로 대체할 수 없음을 보이는 주파수 대조다.

별도로 CE scalar가 핵자질량을 1% 변조하는 prescribed free-field 에너지밀도는
$3.26558\times10^{38}\,\mathrm{J/m^3}$이며 published $10^{16}$ V/m EM
에너지밀도의 $7.38\times10^{17}$배다. source geometry, coherent preparation,
pump work, decay/backreaction과 scalar Floquet D--T amplitude가 없으므로
`PUBLISHED_EM_CONTROL_REPRODUCED_CE_SOURCE_AND_FLOQUET_CHAIN_NOT_REACHED`다.

## 5. Lawson·power·ICF 전파

허용 정적 branch의 열반응률 증가율은 $6.16679\times10^{-10}$이므로 Lawson
$n\tau$ 감소율도 사실상 같은 크기다. Higgs-비례 모델계열 전체의 질량 0·unit
mixing 상한은 $4.01944\times10^{-4}$다.

이 상한을 선형 fusion-power 변화로 읽어도 0.0067%뿐이다. 더구나 NIF laser
2.05 MJ에 선형 적용해 얻는 823.65 J 절감은 capsule radiation hydrodynamics가
없는 `rejected upper bound`일 뿐 점화 예측이 아니다. 1% 직접 결합의 Lawson
감소율 0.9901%도 해당 연산자의 물리 gate가 실패하므로 reactor prediction으로
승격하지 않는다. NIF 기준 자체는 target·laser·design·실험 개선 전체로 target
gain 1.5를 달성한 결과다
([Phys. Rev. Lett. 132, 065102](https://www.osti.gov/biblio/2283994)).

## 6. 최종 certificate

| Gate | 판정 |
|---|---|
| 정적 허용 식 → thermal reactivity | `CONDITIONAL PASS` |
| 현재 portal/Higgs-비례 계열의 1% 목표 | `NO-GO` |
| 직접 연산자의 저에너지 수학해 | `PASS` |
| 직접 연산자 UV completion | `FAIL CLOSED` |
| NN·핵결합·rare-decay joint likelihood | `ABSENT` |
| published electromagnetic strong-field control | `REPRODUCED CONTROL` |
| electromagnetic control → CE scalar source | `REJECT` |
| source-normalized scalar Floquet D--T amplitude | `NOT REACHED` |
| 물리적 1% reactivity gain | `False` |
| reactor/ICF upgrade | `False` |

선언된 정적·시간의존 분기는 모두 계산 또는 blocker 인증까지 도달했다. 현재 최대
지지 단계는 `MODEL_CLASS_NO_GO_PLUS_SOURCE_ENERGY_CONTROLS`다. 다음 단계는 식의
추가 변형이 아니라 외부 입력이다. 즉 직접 연산자의 UV 작용과 joint likelihood,
또는 source-normalized spacetime drive와 검증된 Floquet D--T 산란 계산이 필요하다.

## 7. 재현 명령

```powershell
uv --cache-dir .uv-cache run python `
  examples/physics/fusion_remaining_branches_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_fusion_remaining_branches_loop.py -q
```

최신 canonical 질량을 기존 solver 경로에 주입한 재계산 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -c "import reality_stone.clarus.fusion_equation_iteration_loop as fe; import reality_stone.clarus.fusion_remaining_branches_loop as rb; m=29.69915961743591; fe.DEFAULT_SCALAR_MASS_MEV=m; rb.REGISTERED_SCALAR_MASS_MEV=m; fe.current_fusion_equation_iteration_report.cache_clear(); print(rb.current_fusion_remaining_branches_report().to_dict())"
```

핵심 출력은 `g_N=0.01744695128447136`, `Lambda_req=16.133570834265605 GeV`,
`omega=4.51209664396115e22 rad/s`, `E_nuclear-quiver=4.143030018961213e23 V/m`다.
EM control과 scalar source가 같지 않다는 판정 및 모든 physical gate `False`는 유지된다.
