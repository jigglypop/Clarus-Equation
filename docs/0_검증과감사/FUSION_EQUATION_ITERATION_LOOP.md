# CE 핵융합 식-수정 반복 루프

작성일: 2026-08-05  
코드: `reality_stone/python/reality_stone/clarus/fusion_equation_iteration_loop.py`  
실행: `examples/physics/fusion_equation_iteration_gate.py`  
테스트: `tests/test_fusion_equation_iteration_loop.py`

> 2026-08-06 canonical override: 이 문서의 scalar/portal 행은
> \(m_{\rm light}=29.6991596\,\mathrm{MeV}\)와 direct
> \(\mathrm{BR}_{\rm inv}<0.107\)으로 재평가했다. 위 실행 코드의 기본
> fixture는 구 snapshot이므로 현행 acceptance runner가 아니다.

## 1. 성공 조건

이 루프는 식을 바꾸어 수치만 크게 만드는 방식이 아니라 다음 조건을 동시에
만족할 때만 “된다”고 판정한다.

1. interaction이 선택한 작용 또는 명시된 EFT 확장에서 추적된다.
2. 공급된 coupling 제약을 통과한다.
3. 정적 퍼텐셜에서 에너지별 D--T 장벽 응답을 계산한다.
4. 수정 단면적을 Maxwellian 평균해 반응률과 Lawson 장부까지 닫는다.
5. 10 keV 열반응률이 선언한 공학 목표인 1% 이상 증가한다.

현재 결과는 **계산 사슬은 조건부로 닫혔지만 1% 공학 목표는 실패**다. 실패한
식을 억지로 승격하지 않고 다음에 필요한 새 연산자까지 역산한다.

## 2. 반복 1: 깨진 \(Z_2\) 단일-scalar 식 교정

잘못된 $Q V_{\rm Yukawa}$를 제거하고 Higgs mixing에서 직접 나오는 핵자 결합만
남겼다.

\[
g_{\Phi NN}=\sin\theta\,f_N\frac{m_N}{v},
\qquad
V_1(r)=-\frac{g_{\Phi NN}^2}{4\pi}
\frac{\hbar c}{r}e^{-m_\Phi r/(\hbar c)}.
\]

공급 상한 $|\sin\theta|=0.0038$과 현행
$m_{\rm light}=29.6991596$ MeV에서 핵반경의
인력/Coulomb 비는 $7.59613\times10^{-10}$이다. deuteron과 triton의 coherent
point-nucleus scalar charge를 각각 2와 3으로 두어 $A_DA_T=6$을 포함한 낙관적
상한이다. 유한 핵 form factor는 1 이하이므로 이 근사는 효과를 과소평가하지
않는다. 에너지별 WKB와 열평균까지
전파하면

\[
\frac{\Delta\langle\sigma v\rangle}{\langle\sigma v\rangle}
=6.16679\times10^{-10}.
\]

즉 식은 정상 작동하지만 공학 효과는 없다.

## 3. 반복 2: Higgs-비례 결합 계열의 상한

양의 scalar 질량은 Yukawa 인력을 감소시키고 $|\sin\theta|\le1$이다. 따라서
$m_\Phi=0$, $|\sin\theta|=1$은 이 결합 계열 전체의 점별 상한이다. 이 극단값도

\[
\frac{V_1(r_N)}{V_C(r_N)}=1.42776\times10^{-5},
\qquad
\frac{\Delta\langle\sigma v\rangle}{\langle\sigma v\rangle}
=4.01944\times10^{-4}
\]

에 그쳐 1% 목표를 실패한다. 그러므로 scalar 질량이나 mixing을 이 식 안에서
계속 조정하는 반복은 종료할 수 있다. 더 낮은 질량도, 더 긴 range도 이 상한을
넘지 못한다.

## 4. 반복 3: 정본 \(Z_2\) 두-scalar 교환

정본 작용의 $h\Phi^2$ vertex와 표준 핵자-Higgs form factor를 사용해 저운동량에서
Higgs를 적분하면

\[
\mathcal L_{N\Phi^2}^{\rm eff}
=-\frac{C_N}{2}\bar N N\Phi^2,
\qquad
C_N=\frac{2\lambda_{HP}f_Nm_N}{m_h^2}.
\]

두 동일 scalar의 Euclidean connected correlator를 적분한 장거리 퍼텐셜은 이
정규화에서

\[
V_2(r)=-\frac{C_N^2m_\Phi}{32\pi^3r^2}K_1(2m_\Phi r),
\]

이며 자연단위가 아닌 코드에서는 모든 $r$에 $\hbar c$ 변환을 명시한다. 질량 0
극한은

\[
V_2(r)\longrightarrow-\frac{C_N^2}{64\pi^3r^3}.
\]

PDG 2026의 ATLAS direct
\(\operatorname{BR}_{\rm inv}<0.107\)을 fixed-light stress-test에서 역산한
$|\lambda_{HP}|=0.00504779076$을 사용하면 결과는 다음과 같다.

| 분기 | $V/V_C$ at $r_N$ | 열반응률 증가율 |
|---|---:|---:|
| 29.6991596 MeV 두-scalar | `2.48554e-17` | `3.72520e-18` |
| 질량 0 상한 | `5.06726e-17` | `1.97233e-17` |

따라서 정본 $Z_2$ 식도 계산은 닫히지만 공학 목표를 실패한다. 이 결과는
point-nucleon, 저운동량 Higgs 적분, 장거리 두-scalar cut의 조건부 EFT
control이며 완전한 핵 R-matrix 계산으로 과장하지 않는다.

## 5. WKB에서 Bosch--Hale 열평균까지

수치 cancellation을 피하기 위해 두 큰 WKB exponent를 뺄셈하지 않고 다음의
양의 차이를 직접 적분한다.

\[
\log R(E)=\frac{2\sqrt{2\mu}}{\hbar c}
\int_{r_N}^{r_C}
\left[
\sqrt{V_C-E}-
\sqrt{\max(V_C-|V_X|-E,0)}
\right]dr.
\]

표준 단면적은 Bosch--Hale 식의 D--T 계수를 그대로 구현했다
([Bosch--Hale 원 논문](https://www.osti.gov/etdeweb/biblio/5161054),
[WarpX의 독립 구현](https://warpx.readthedocs.io/en/latest/_static/doxyhtml/_bosch_hale_fusion_cross_section_8_h_source.html)).
조건부 bridge는 핵 S-factor가 바뀌지 않는다는 가정 아래

\[
\sigma_{\rm mod}(E)=\sigma_{BH}(E)R(E)
\]

로 정의하고, 실제 Maxwellian kernel

\[
\langle\sigma v\rangle\propto
\int_0^\infty dE\,\sigma(E)E e^{-E/T}
\]

를 수치 적분한다. 독립 cross-section 적분과 Bosch--Hale closed reactivity fit은
10 keV에서 비가 1.00522로 0.53% 안에 재현된다. 수정률은 이 수치 kernel의
비로 계산하고 표준 closed fit에 적용한다.

turning point의 수치 spike를 막기 위해
$r=r_N+(r_C-r_N)\sin^2\vartheta$로 변수변환했다. 공급 mixing 분기의 증가율은
`121×601`, `181×1001`, `361×4001` 에너지×방사 격자에서 각각
`6.166589e-10`, `6.166795e-10`, `6.167004e-10`으로 수렴한다.

## 6. 될 때까지 역산한 새 식

질량 0 단일-scalar에 독립 직접 핵자 결합

\[
\mathcal L_{\rm direct}=-g_N\Phi\bar N N
\]

을 새로 허용한다고 가정하고 1% 열반응률 목표를 이분법으로 역산하면

\[
g_N^{\rm req}=5.69352\times10^{-3}.
\]

이 값은 질량 0의 낙관적 하한이다. 현행 light 질량 29.6991596 MeV에서
같은 1% 목표를 다시 풀면 \(g_N=0.01744695\), Higgs-mixing 등가값
약 \(15.2477\)이 필요하다.

질량 0 해를 Higgs-비례 식으로 환산한 mixing은

\[
|\sin\theta|_{\rm equiv}=4.97583>1
\]

이다. 수학적 1% 목표는 이 새 연산자로 도달하지만 다음 두 gate를 실패한다.

- 현재 선택된 portal 작용에는 $\Phi\bar N N$ 직접 연산자가 없다.
- unit mixing보다도 큰 등가 결합이므로 기존 Higgs-mixing 식의 수정으로 얻을 수 없다.

따라서 `MATHEMATICAL_TARGET_ONLY_NEW_DIRECT_OPERATOR_REQUIRED`로 기록하며 물리적
핵융합 성공으로 승격하지 않는다. 다음 반복에는 새 연산자의 대칭성, 재규격화,
핵산란·희귀붕괴 제약 또는 source-normalized 시간의존 작용이 실제 입력으로
필요하다.

직접 연산자의 UV/EFT·핵물질·시간의존 source 후속 판정은
[잔여분기 최종 루프](FUSION_REMAINING_BRANCHES_LOOP.md)에서 닫는다.

## 7. 최종 판정

| 항목 | 판정 |
|---|---|
| 잘못된 $Q\times V$ 제거 | `PASS` |
| 단일-scalar 퍼텐셜 → WKB → 열평균 → Lawson | `CONDITIONAL PASS` |
| 정본 두-scalar 퍼텐셜 → WKB → 열평균 → Lawson | `CONDITIONAL PASS` |
| 허용된 현재 작용에서 1% 반응률 증가 | `FAIL` |
| Higgs-비례 전체 모델 계열에서 1% 증가 | `NO-GO UNDER STATED BOUNDS` |
| 새 직접 결합의 수학적 1% 해 | `PASS / OUTSIDE SELECTED ACTION` |
| 물리적 CE 핵융합 upgrade | `False` |

현재 최대 지지 단계는
`CONDITIONAL_STATIC_POTENTIAL_TO_THERMAL_REACTIVITY_CHAIN`이다.

## 8. 재현 명령

```powershell
uv --cache-dir .uv-cache run python `
  examples/physics/fusion_equation_iteration_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_fusion_equation_iteration_loop.py -q
```
