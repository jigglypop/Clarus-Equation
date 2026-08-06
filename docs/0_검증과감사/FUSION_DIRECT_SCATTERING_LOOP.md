# CE 직접 핵자 연산자 저에너지 핵산란 루프

작성일: 2026-08-05  
코드: `reality_stone/python/reality_stone/clarus/fusion_direct_scattering_loop.py`  
실행: `examples/physics/fusion_direct_scattering_gate.py`  
테스트: `tests/test_fusion_direct_scattering_loop.py`

## 1. 목적과 경계

coherent D--T scalar charge $A_DA_T=6$을 포함하면 최신 canonical 질량
29.6991596174 MeV에서 1% 열반응률 증가에 필요한 핵자당 직접 결합은
$g_N=0.0174469513$이다. 이 문서는
그 결합이 저에너지 $np$ 관측량에서 보일 규모인지 두 analytic control로 본다.
강한 NN potential의 distorted-wave 재적합이 아니므로 단독 exclusion으로 세지
않는다.

## 2. free Born scattering-length control

추가 Yukawa potential

\[
V_\Phi(r)=-\alpha_\Phi\frac{\hbar c}{r}e^{-m_\Phi r/(\hbar c)},
\qquad
\alpha_\Phi=\frac{g_N^2}{4\pi}
\]

의 자유 Born zero-momentum scattering-length 이동은

\[
\Delta a_{\rm Born}
=-\frac{2\mu_{np}\alpha_\Phi}{m_\Phi^2}\hbar c
=-0.00508459\ {\rm fm}.
\]

비교 입력은 저에너지 $np$ 분석의
$a_t=5.4112(15)$ fm, $a_s=-23.7148(43)$ fm이다
([분석 원문](https://arxiv.org/abs/0704.1024)). 자유 Born 이동은 보고된 triplet
오차의 3.3897배, singlet 오차의 1.1825배다. 따라서 관측 정밀도보다 작다고 버릴 수
없지만, 강한 상호작용 위의 실제 이동은 distorted-wave Born 또는 전체 phase-shift
fit으로 계산해야 한다.

## 3. Hulthén deuteron control

정규화 radial wavefunction

\[
u(r)=N(e^{-\gamma r}-e^{-\beta r}),
\quad
\gamma=\sqrt{2\mu B_d}/(\hbar c),
\quad \beta=1.4\ {\rm fm^{-1}}
\]

에 대해 Yukawa expectation은 analytic하게

\[
\langle V_\Phi\rangle
=-\alpha_\Phi\hbar c\,N^2
\log\frac{(\gamma+\beta+\kappa)^2}
{(2\gamma+\kappa)(2\beta+\kappa)},
\quad \kappa=m_\Phi/(\hbar c)
\]

이고 결과는 `-2.08101 keV`다. 이는 deuteron binding의 약
$9.3547\times10^{-4}$이다. 고정된 기존 Hamiltonian에서는 0이 아닌 이동이지만,
Hulthén parameter 선택과 strong contact 재적합을 포함하지 않으므로 실험 배제로
승격하지 않는다.

## 4. 판정

| Gate | 판정 |
|---|---|
| 자유 Born 산란길이 산술 | `PASS` |
| Hulthén 1차 섭동 산술 | `PASS` |
| 보고 정밀도보다 완전히 작은 효과 | `False` |
| distorted-wave NN phase-shift fit | `NOT REACHED` |
| deuteron/triton/helium few-body refit | `NOT REACHED` |
| 직접 연산자 experimental exclusion | `NOT DERIVED` |
| 직접 연산자 physical pass | `False` |

최대 지지 단계는 `FREE_BORN_AND_HULTHEN_TENSION_CONTROL_FULL_NUCLEAR_REFIT_REQUIRED`다.

## 5. 최신 질량 재현

wrapper의 legacy 기본질량 대신 canonical 질량을 solver에 주입한 실제 재계산 명령은
다음과 같다.

```powershell
.\.venv\Scripts\python.exe -c "import reality_stone.clarus.fusion_equation_iteration_loop as fe; import reality_stone.clarus.fusion_direct_scattering_loop as ds; fe.DEFAULT_SCALAR_MASS_MEV=29.69915961743591; fe.current_fusion_equation_iteration_report.cache_clear(); print(ds.audit_direct_nuclear_scattering().to_dict())"
```

이 명령은 `range=6.6441940763 fm`, `Delta a=-0.00508459277 fm`,
`Hulthen shift=-2.08101362 keV`를 반환한다. analytic control은 갱신됐지만
distorted-wave·few-body fit이 없으므로 physical gate는 여전히 `False`다.

