# 05j. CE Suppression Scaling 감사

## 0. 목표

[05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md)는 suppression의 scale에 따라 결론이 갈린다는 것을 증명했다.

| scale | 정리 | manifest 극한에서의 효과 |
|---|---|---|
| bounded tilt \(e^{-S_{\mathrm{supp}}}\) | 05i 정리 5.1 | 효과 없음. finite-\(\beta\) 가중치만 변경 |
| \(\beta\)-coupled \(e^{-S_{\mathrm{supp}}/\hbar}\) | 05i 정리 5.2 | \(\operatorname{argmin}(S_E+S_{\mathrm{supp}})\)로 선택 변경 |
| hard constraint \(S_{\mathrm{supp}}\in\{0,\infty\}\) | 05i `Open` | 조건부 측도. 별도 recovery 필요 |

이 문서는 [05d_pathspace_audit.md](05d_pathspace_audit.md) 방식으로 실제 CE 문서의 suppression 사용처를 수집하고, 각 사용처가 위 세 regime 중 어디에 속하는지, 그리고 해당 문서의 주장이 그 regime에서 실제로 성립하는 종류의 주장인지 판정한다.

핵심 결론:

> CE 문서군의 suppression은 한 종류가 아니다. 정식 경로공간 공리의 \(S_{\mathrm{supp}}\)는 unscaled, master formula의 Clarus 항은 \(\beta\)-coupled, 우주론 생존분율의 \(\Phi\)는 O(1) bounded tilt, 등분배 정리의 접힘 조건은 hard constraint다. 네 가지가 같은 이름 아래 섞여 있고, 그중 생존분율 주장은 **본질적으로 finite-\(\beta\) 주장**이라 manifest 극한 언어로 읽으면 무효가 된다.

현재 판정:

| 사용처 | regime | 해당 주장의 지위 |
|---|---|---|
| 경로공간 공리의 \(S_{\mathrm{supp}}\) | unscaled, 미지정 | scale 미선언, `Selection` 누락 |
| master formula Clarus 항 | \(\beta\)-coupled | 05i 정리 5.2 적용 가능 |
| 우주론 생존분율 \(\langle e^{-\Phi}\rangle\) | bounded tilt | fixed-\(\beta\) 주장으로만 유효 |
| 등분배 threshold \(S_E<S_{\mathrm{th}}\) | hard constraint | 05i `Open`, 조건부 bridge 필요 |
| 강의의 "비고전 경로 억압" | scale 혼합 | 긴장 지점, 4절 |

## 1. 판정 기준

05i의 두 정리를 감사 기준으로 다시 쓴다.

기준 1 (bounded tilt 무효): \(S_{\mathrm{supp}}\)가 \(\hbar\)와 무관하게 유계이면

$$
\varepsilon\log\frac{d\nu^\varepsilon}{d\mu^\varepsilon}
=O(\varepsilon)\to0
$$

이므로 manifest set은 \(S_{\mathrm{supp}}\) 없이 계산한 것과 같다. 이런 \(S_{\mathrm{supp}}\)에 대한 올바른 주장 형태는 "고정된 \(\hbar\)(또는 고정된 \(\beta\))에서 확률 가중치가 \(e^{-S_{\mathrm{supp}}}\) 비율로 바뀐다"이다.

기준 2 (\(\beta\)-coupled 유효): \(S_{\mathrm{supp}}\)가 \(1/\hbar\)로 결합하면 manifest set이 \(\operatorname{argmin}(S_E+S_{\mathrm{supp}})\)로 바뀐다. 이런 \(S_{\mathrm{supp}}\)에 대한 올바른 주장 형태는 "선택되는 고전 경로 자체가 바뀐다"이다.

기준 3 (hard constraint): \(S_{\mathrm{supp}}=\infty\cdot\mathbf 1_{A^c}\) 꼴이면 측도를 \(A\)로 조건화하는 것과 같다. 05i가 닫지 않은 자리이며, 조건화된 prior의 recovery mass를 별도로 확인해야 한다.

## 2. 사용처 수집

### 2.1 정식 경로공간 공리: unscaled \(S_{\mathrm{supp}}\)

출처: [../참조/epsilon_제1원리_유도.md](../참조/epsilon_제1원리_유도.md) 12-13장.

$$
W[\gamma]=\frac{S_E[\gamma]}{\hbar}+S_{\mathrm{supp}}[\gamma],
\qquad
S_{\mathrm{supp}}\ge0,
\qquad
\int e^{-W}d\mu_0<\infty.
$$

관찰:

- \(S_{\mathrm{supp}}\)에는 \(1/\hbar\)가 붙어 있지 않다. 식 그대로 읽으면 unscaled다.
- 유계성은 선언되어 있지 않다. "클라루스장/비선택 경로 효과를 통합한 추가 양"이라는 서술뿐이다.

판정:

- \(S_{\mathrm{supp}}\)가 유계라면 기준 1에 의해 manifest 선택에 참여하지 못한다. 이 경우 13장의 "선택 기준 공리"(\(F=W+c\)의 minimizer가 선택 경로)에서 \(S_{\mathrm{supp}}\)는 \(\hbar\to0\) 극한의 minimizer를 바꾸지 못하고, 고정 \(\hbar\)에서의 \(F\)-minimizer에만 기여한다.
- \(S_{\mathrm{supp}}\)가 비유계라면 05i 정리 5.1의 가정 밖이며 별도 검증이 필요하다.
- 어느 쪽인지가 문서에 선언되어 있지 않다. 이것은 `Selection` 누락이다.

### 2.2 Master formula: \(\beta\)-coupled Clarus 항

출처: [../axium.md](../axium.md) 제1공리 구체화.

$$
\mathcal K_{\mu\nu}(x)
=
\frac1Z\int\mathcal D\gamma\,
\frac{\delta^2S[\gamma]}{\delta\gamma^\mu\delta\gamma^\nu}
\,e^{-S[\gamma]/\hbar}
$$

및 master formula의 Clarus 항 \(\alpha_C\beta|\nabla\Phi|^2\), \(\lambda|\nabla^2\Phi|^2\)는 작용 밀도 안에 들어 있다.

관찰:

- 억제 항이 작용 \(S\) 내부에 있고 전체가 \(e^{-S/\hbar}\)로 들어가므로, 이 독법에서는 suppression이 자동으로 \(\beta\)-coupled다.
- axium.md의 saddle point 서술(\(\hbar\to0\)에서 고전 경로 지배)도 이 독법과 정합적이다.

판정:

- 기준 2에 해당한다. 05i 정리 5.2가 적용 가능하고, manifest set은 \(\operatorname{argmin}(S_{\mathrm{SM}}+\text{Clarus 항})\)이다.
- 단, 2.1의 정식 공리와 표기가 충돌한다. 2.1은 \(S_{\mathrm{supp}}\)를 \(S_E/\hbar\) **바깥**에 두고, axium은 **안**에 둔다. 같은 이론의 두 정식화가 서로 다른 regime에 있다.

### 2.3 우주론 생존분율: O(1) bounded tilt

출처: [../경로적분.md](../경로적분.md) 3.2절.

$$
P_{\mathrm{survive}}=\langle e^{-\Phi}\rangle,
\qquad
\langle\Phi\rangle=\sigma D_{\mathrm{eff}},
\qquad
\varepsilon^2=e^{-(1-\varepsilon^2)D_{\mathrm{eff}}}.
$$

관찰:

- 억제 지수 \(\Phi\)의 크기는 \(\sigma D_{\mathrm{eff}}\approx0.95\times3.18\approx3\), 즉 O(1)이다. \(1/\hbar\)가 붙어 있지 않다.
- 출력도 분율이다. \(\Omega_b\approx0.0486\)은 0도 1도 아닌 중간값이다.

판정:

- 이것은 기준 1의 bounded tilt이고, **그래야만 한다**. 만약 \(\Phi\)가 \(\beta\)-coupled였다면 manifest 극한에서 \(P_{\mathrm{survive}}\)는 0 또는 1로 퇴화하고 중간 분율 자체가 존재할 수 없다.
- 따라서 bootstrap 고정점과 \(\Omega_b\) 주장은 본질적으로 **finite-\(\beta\) Gibbs 측도의 분율 주장**이다. 올바른 수학적 받침은 05e의 zero-temperature 농축 정리가 아니라 05i 정리 3.2(Route W fixed-\(\beta\) 측도 존재)와 05h 정리 5.1(fixed-\(\beta\) convergence)이다.
- 이 사용처를 "경로가 고전 경로로 농축한다"는 manifest 언어로 읽으면 무효다. 분율 주장과 농축 주장은 서로 다른 극한에 산다.

### 2.4 등분배 정리의 접힘 조건: hard constraint

출처: [../경로적분.md](../경로적분.md) 정리 3.2.1.

$$
Z_{\mathrm{surv}}
=
\int_{\{S_E<S_{\mathrm{th}}\}}\mathcal D\phi\,e^{-S_E[\phi]}.
$$

관찰:

- "접힘 통과"가 indicator 조건 \(S_E<S_{\mathrm{th}}\)로 정의된다. 이는 \(S_{\mathrm{supp}}=\infty\cdot\mathbf 1_{\{S_E\ge S_{\mathrm{th}}\}}\)와 동치다.

판정:

- 기준 3의 hard constraint다. 05i가 `Open`으로 남긴 자리이고, 2.3의 지수형 tilt \(\langle e^{-\Phi}\rangle\)와도 형태가 다르다.
- 같은 문서(경로적분.md) 안에서 생존이 두 가지 다른 수학(지수 tilt와 sharp threshold)으로 정의되고 있다. 정리 3.2.1은 threshold 독법에서만 성립하는 진술이므로, 두 독법의 동치 또는 대응 관계가 별도로 필요하다.

### 2.5 강의 문서의 접힘 서술: scale 혼합

출처: [../1_강의/A_연역적_유도.md](../1_강의/A_연역적_유도.md).

> "이 장이 비고전적 경로를 억압(suppress)하여 고전 경로로 접기 때문이다."

판정:

- "비고전 경로를 접어 고전 경로로 만든다"는 manifest 선택 주장이다. 기준 2가 필요하다.
- 그러나 같은 문서군에서 \(\Phi\)의 크기는 2.3처럼 O(1)로 쓰인다. O(1) 억제는 기준 1에 의해 고전 경로 선택을 만들지 못한다. 표준 그림에서 고전 경로 농축을 만드는 것은 \(S_E/\hbar\) 자체다.
- 따라서 이 서술이 참이 되는 독법은 두 가지뿐이다. (a) \(\Phi\)가 작용 내부 항(2.2 독법)으로서 \(1/\hbar\)를 함께 받는다. (b) "접는다"를 농축이 아니라 finite-\(\beta\) 가중치 변경으로 약화해 읽는다. 문서는 어느 쪽인지 선언하지 않는다.

## 3. 종합 분류표

| # | 출처 | suppression 형태 | regime | 주장 종류 | 정합 여부 |
|---|---|---|---|---|---|
| 2.1 | 참조/epsilon 12-13장 | \(W=S_E/\hbar+S_{\mathrm{supp}}\) | unscaled, 미지정 | 선택 기준 공리 | scale 선언 누락 |
| 2.2 | axium.md | Clarus 항 in \(S\), \(e^{-S/\hbar}\) | \(\beta\)-coupled | manifest 선택 | 05i 정리 5.2로 정합 |
| 2.3 | 경로적분.md 3.2 | \(\langle e^{-\Phi}\rangle\), \(\Phi=O(1)\) | bounded tilt | finite-\(\beta\) 분율 | 분율 주장으로는 정합 |
| 2.4 | 경로적분.md 3.2.1 | \(S_E<S_{\mathrm{th}}\) indicator | hard constraint | 등분배 정리 | `Open`, 조건부 bridge 필요 |
| 2.5 | 1_강의/A | "비고전 경로 억압" | 미지정 | manifest 선택 | 2.3 scale과 긴장 |

## 4. 발견된 긴장 지점

### 긴장 1: 정식 공리(2.1)와 master formula(2.2)의 placement 불일치

같은 이론에서 suppression이 한 번은 \(S_E/\hbar\) 바깥(unscaled), 한 번은 안(\(\beta\)-coupled)에 있다. 두 placement는 05i에 의해 **다른 물리**를 준다. 해소하려면 정식 공리를

$$
W[\gamma]
=
\frac{S_E[\gamma]+S_{\mathrm{supp}}^{\mathrm{phys}}[\gamma]}{\hbar}
\qquad\text{또는}\qquad
W=\frac{S_E}{\hbar}+S_{\mathrm{supp}}^{\mathrm{stat}}
$$

중 하나로 고르고, 전자는 선택 참여형(\([S_{\mathrm{supp}}^{\mathrm{phys}}]=\) 작용 차원), 후자는 분율형(무차원, manifest 무참여)으로 역할을 분리해 선언해야 한다.

### 긴장 2: 분율 주장과 농축 주장의 극한 혼용

bootstrap/\(\Omega_b\) 체계(2.3)는 finite-\(\beta\)에서만 의미가 있고, 고전 경로 접힘(2.5)은 \(\beta\to\infty\)에서만 의미가 있다. 두 주장을 같은 \(\Phi\)의 같은 성질로 서술하면 한쪽이 반드시 무효가 된다. 분리 선언이 필요하다.

### 긴장 3: 지수 tilt와 threshold의 이중 정의

생존이 2.3에서는 smooth tilt, 2.4에서는 sharp threshold다. threshold 독법은 hard constraint recovery(05i `Open`)가 닫혀야 정당화되고, 두 독법 사이의 대응(예: threshold를 tilt의 어떤 극한으로 얻는지)은 어디에도 증명되어 있지 않다.

## 5. 권장 규약

각 suppression 사용처에 다음 label 중 하나를 명시하는 것을 권장한다.

| label | 정의 | 허용되는 주장 형태 | 받침 정리 |
|---|---|---|---|
| `supp:stat` | 무차원, \(\hbar\) 무관 유계 | 고정 \(\beta\)에서의 분율/가중치 | 05i 정리 3.2, 05h 정리 5.1 |
| `supp:dyn` | 작용 차원, \(e^{-S_{\mathrm{supp}}/\hbar}\) 결합 | manifest 선택 변경 | 05i 정리 5.2 |
| `supp:hard` | \(\{0,\infty\}\) indicator | 조건부 측도 | `Open`, 05k 후보 |

이 규약으로 3절 표를 다시 쓰면:

| 출처 | 권장 label |
|---|---|
| 참조/epsilon 12-13장 | 선언 필요. 분율 체계와 일관되려면 `supp:stat` |
| axium.md Clarus 항 | `supp:dyn` |
| 경로적분.md 3.2 | `supp:stat` |
| 경로적분.md 3.2.1 | `supp:hard` |
| 1_강의/A 접힘 서술 | `supp:dyn`으로 고치거나 농축 표현을 약화 |

## 6. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| suppression 사용처 수집과 regime 분류 | 3절 표 |
| 생존분율 주장이 finite-\(\beta\) 전용임 | 2.3, 기준 1 |
| master formula 독법의 \(\beta\)-coupled 정합성 | 2.2 |
| placement/극한/이중정의 긴장 3건 분리 | 4절 |
| label 규약 제안 | 5절 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| hard constraint recovery | threshold 조건화 \(\mu(\cdot\mid S_E<S_{\mathrm{th}})\)의 존재와 recovery mass. `05k` 후보 |
| tilt-threshold 대응 | \(\langle e^{-\Phi}\rangle\)와 \(Z_{\mathrm{surv}}/Z\)가 언제 같은 분율을 주는지 |
| 정식 공리의 scale 선언 | epsilon_제1원리 12-13장에 `supp:stat`/`supp:dyn` 명시 |
| 비유계 \(S_{\mathrm{supp}}\) | 유계 가정이 깨지는 사용처가 있는지 추가 수집 |

## 7. 결론

$$
\boxed{
\text{CE suppression}
=
\underbrace{\text{supp:dyn}}_{\text{선택 변경, }e^{-S/\hbar}}
\;\sqcup\;
\underbrace{\text{supp:stat}}_{\text{분율, finite-}\beta}
\;\sqcup\;
\underbrace{\text{supp:hard}}_{\text{조건화, Open}}
}
$$

세 regime은 05i에 의해 수학적으로 다른 결론을 주므로 같은 기호로 섞어 쓸 수 없다. 우주론 분율 체계(bootstrap, \(\Omega_b\))는 `supp:stat`/`supp:hard`에 살고, 고전 경로 선택 서사는 `supp:dyn`에 산다. 다음 병목은 분율 체계가 실제로 쓰는 hard constraint 조건화의 recovery를 닫는 일이다.
