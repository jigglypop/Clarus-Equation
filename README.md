# Clarus Equation (CE)

> **상태:** 미발표 연구 가설·수치 실험 저장소. arXiv 등록과 동료 심사는 아직 없다.
>
> **현재 결론:** 고정점과 일부 대수는 명시한 전제 아래 완결되어 있다. CE 고유의
> 물리 사상과 관측량 연결은 공리 또는 미완성 항목으로 분리하며, 자연에 대한
> 확증과 수학적 완결성을 같은 말로 쓰지 않는다.

## 1. 무엇을 연구하는가

CE는 적은 수의 무차원 구조로 입자물리와 우주론의 여러 수치 관계를 함께 기술할 수
있는지 시험한다. 다음 세 층을 분리한다.

- 명시한 공리에서 증명되는 정리와 그 직접 산출
- 수학 구조를 물리량에 대응시키는 공리
- 자료를 사용하는 경험식과 아직 증명이 완결되지 않은 항목

전체 주장별 출처와 남은 증명 의무는
[형식 구조 원장](docs/검증_원장/경로적분_전체_진리값_감사.md)에 기록한다.
우주론의 정밀 계산값·과거 재현값·런타임 호환값·관측 자료는
[우주론 판본·주장 원장](docs/검증_원장/상수_우주론_원장.md)에서 서로 다른
`config_id`와 Claim ID로 관리한다.

## 2. 정본 출처 체계

| 표기 | 뜻 | 필요한 기록 |
| --- | --- | --- |
| **[정의]** | 기호·대상·정의역을 선언하는 항목으로, 참거짓 명제가 아님 | 기호와 정의역 |
| **[정리]** | 명시한 전제와 정의만으로 증명된 명제 | 전제, 정의역, 증명 |
| **[공리]** | 모형이 채택한 경계조건·가지·물리 사상 | 선택 내용과 적용 범위 |
| **[산출]** | 정리와 공리를 대입해 얻는 직접 결과 | 사용한 선행 항목 |
| **[경험식]** | 자료·보정·유효 계수를 사용하는 관계 | 자료와 계수의 출처 |
| **[미완성]** | 정의·증명·작용·사상이 아직 완결되지 않은 항목 | 남은 증명 의무 |
| **[예측]** | 입력과 판별 기준을 미리 고정한 관측량 | 독립 자료와 불확도 |

수학적 출처와 관측 비교는 별도 열에 둔다. 허용된 정의역 안의 완전한
반례가 확인된 보편 명제는 정본 문장·식·표에 보존하지 않으며, 재도입 방지는
실행 회귀검사에서 담당한다.

## 3. 입력 회계와 코어

### 3.1 외부 입력

| 항목 | 역할 | 상태 |
| --- | --- | --- |
| \(\alpha_s(M_Z)=0.11789\) | 저장소의 강결합 benchmark | **[공리]**, 외부 입력 |
| \(d=3\) | 현재 물리 branch | **[공리]**, 가지 선택 |
| 관측 자료·공분산·척도 | 각 likelihood와 경험식에 명시 | 외부 비교 자료 |

2026 PDG QCD review의 평균은
\(\alpha_s(M_Z^2)=0.1180\pm0.0009\)이다. 저장소 값 0.11789는 이 측정과
양립하는 **입력값**이지 CE의 독립 예측이 아니다.

### 3.2 전자약 브리지

**[경험식]**

\[
s_W^2 \equiv \sin^2\theta_W = 4\alpha_s^{4/3}
= 0.2312220683 .
\]

PDG 2026의 \(\overline{\mathrm{MS}}\) 기준값 \(0.23122\pm0.00006\)과 비교하면
약 \(+0.03\sigma\)다. 가까운 수치는 관계식의 흥미로운 readout이지만, RG·scheme·
scale 의존성을 작용에서 유도한 것은 아니다.

**[산출]** 아래 대수는 위 경험식과 \(d=3\) branch를 조건으로 한다.

\[
\delta=s_W^2(1-s_W^2),\qquad D_{\mathrm{eff}}=3+\delta .
\]

### 3.3 고정점

**[정리]** 아래 최소 고정점은 \(D_{\mathrm{eff}}>1\)인 Poisson 분지모형에서
정의되며, 해당 정의역의 존재성과 최소성은 고정점 분석으로 닫힌다.

\[
q_{\mathrm{ext}}
=\exp\!\left[-(1-q_{\mathrm{ext}})D_{\mathrm{eff}}\right]
=0.0486467196\ldots .
\]

지수의 인자 \((1-q_{\mathrm{ext}})D_{\mathrm{eff}}\)는 무차원이다.
분기과정 용어에서 최소 고정점 \(q_{\mathrm{ext}}\)는 **소멸 확률**이고,
\(s_{\mathrm{branch}}=1-q_{\mathrm{ext}}\)는 생존 확률이다.

**[정의]** 이 문맥의
\(\Omega_b:=\rho_b(t_0)/\rho_{\rm crit}(t_0)\)는 현재 바리온 밀도분율이다.

**[공리: 과거 경계모형 `C-B-LEGACY-01`]**
\(q_{\mathrm{ext}}\mapsto\Omega_b\)는 별도로 채택했던 호환 사상이다.
따라서 고정점의 정확한 해와 바리온 밀도의 물리적 유도를 같은 주장으로 세지
않는다. 새 경로의 conditioned \(Dq\) 조성과 공변 전이 면 \(1/D\) 합성은
[우주론 판본·주장 원장](docs/검증_원장/상수_우주론_원장.md)에 분리한다.

## 4. 형식 결과와 관측 비교

### 4.1 형식 결과

| 항목 | 결과 | 출처 |
| --- | --- | --- |
| 정밀 최소 고정점 | \(q_{\mathrm{ext}}=0.0486467196\ldots\) | **[정리]**과 **[산출]** |
| 반올림 입력 호환 고정점 | \(q_{\mathrm{ext}}=0.0486466333\ldots\) | **[공리]** 호환 판본 |
| 코어 지수·로그 인자 | 7식의 dimension vector가 0 | **[산출]** |
| \(\delta,D_{\mathrm{eff}}\) | 전자약 사상과 외부 \(\alpha_s\)를 대입한 값 | **[산출]** |
| 과거 \(q_{\mathrm{ext}}\mapsto\Omega_b\) | 확률을 에너지 분율로 읽는 경계규칙 | **[공리]** |

### 4.2 관측 비교 자료

| 경험적 readout | 계산값 | 비교값 | 수학적 출처 |
| --- | ---: | ---: | --- |
| \(V_{cb}\) projector | 0.04162 | 0.04153 | **[경험식]** |
| \(V_{us}\) 보정식 | 0.22696 | 0.22650 | **[경험식]** |
| \(A_s\) 투영 잔차 readout | \(2.104\times10^{-9}\) | \((2.099\pm0.029)\times10^{-9}\) | **[경험식]** |

이 표의 근접도는 정리의 증명에 사용하지 않는다. 자료 선택, 보정 규칙과
공분산을 포함한 독립 우도가 갖춰진 경우에만 **[예측]**으로 분리한다.

## 5. 기호와 의미의 경계

| 기호 | 이 저장소의 정본 의미 | 동일시하지 않는 것 |
| --- | --- | --- |
| \(\Phi[\gamma]\) | 경로 진폭 또는 문맥상 명시한 장 | Hessian, 곡률, 중력 자체 |
| \(H_{ij}\) | 작용의 2차 변분/Hessian | \(\Phi\) |
| \(R\) | 시공간 곡률 스칼라 | 확률 진폭 |
| \(q_{\mathrm{ext}}\) | 최소 소멸 고정점 | 생존 확률 |
| \(s_{\mathrm{branch}}\) | \(1-q_{\mathrm{ext}}\), 생존 확률 | 억압 계수 \(\sigma\) |
| \(\varepsilon^2\) | 과거 문서의 호환 표기 | 문맥 없이 소멸/생존을 동시에 뜻하지 않음 |

## 6. 모형 후보와 차원 규약

**[공리]** 이론물리 branch는 부호·대칭·정의역을 고정한 공변
\(Z_2\) singlet-portal EFT를 사용한다.

\[
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\Lambda_0
-\frac12(\nabla\phi)^2+\mathcal L_{\rm SM}^{\rm kin+gauge+Yuk}
-V(H,\phi)\right],
\]

\[
V(H,\phi)=V_H(H)+\frac12m_\phi^2\phi^2
+\frac{\lambda_\phi}{4}\phi^4
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H.
\]

**[정리]** \(\lambda_H,\lambda_\phi>0\)에서 portal 사차항이 음이
아니기 위한 필요충분조건은

\[
\lambda_{H\phi}\geq-2\sqrt{\lambda_H\lambda_\phi}.
\]

[증명](docs/검증_원장/참조_핵심_정리_증명.md#portal-boundedness) 등호에서는
평평한 사차 방향이 남으므로, 다음 측도 정리에는 엄격한 부등식을 쓴다.

**[정리]** \(\xi=0\)인 고정 Euclidean 배경에서
\(\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}\)와 양의 kinetic term을
만족하는 measurable finite scalar--Higgs bosonic truncation을 유한
격자에 두면 partition function과 모든 다항식 모멘트가 유한하다.
[증명](docs/검증_원장/참조_핵심_정리_증명.md#finite-lattice-measure)

**[정리]** 완결된 공변 작용의 총 stress tensor는 on shell 보존된다.
[증명](docs/검증_원장/참조_핵심_정리_증명.md#noether-stress)

따라서 이 branch는 특정 질량이나 결합을 예측하지는 않지만 대칭, 안정성과
보존법칙을 갖는다. 양의 cutoff 측도가 여기서 증명된 범위는 scalar--Higgs
bosonic truncation이며, gauge fixing·ghost·fermion determinant를 포함한
전체 양자측도는 별도 구성이다.

경로 가중치는 차원 있는 작용 \(S_E\)를 그대로 지수에 넣지 않는다.

\[
P(\gamma)=
\frac{\exp[-S_E(\gamma)/\hbar]}
{\sum_{\gamma'}\exp[-S_E(\gamma')/\hbar]} .
\]

따라서 모든 `exp`, `log`, 확률, 고정점 코어의 인자는 무차원이어야 한다.

### 6.1 추가로 보존된 조건부 이론

| 구조 | 닫힌 범위 |
|---|---|
| Hodge \(\Lambda^2\leftrightarrow\Lambda^1\) | 방향·metric을 가진 \(d=3\) |
| 다형 Poisson 최소 고정점 | 비음수 독립 offspring |
| \(A=dI+\delta B\) | row-stochastic toy family의 정확한 균일 축약 |
| canonical scalar \(w\geq-1\) | Einstein frame 최소 결합·양의 kinetic·\(\rho>0\) |
| 상수 진공항 \(w=-1\) | 공변 GR branch |
| CKM·PMNS 질량행렬 | 주어진 질량과 unitary 입력의 존재구성 |
| Koide \(Q=2/3\) | 제곱근 질량벡터의 \(45^\circ\) 기하 |
| Starobinsky형 \(n_s,r\) | \(V_0>0\), Einstein-frame 정준 단일장 지배와 leading slow-roll |

공통 증명은
[핵심 정리 증명 원장](docs/검증_원장/참조_핵심_정리_증명.md)에 둔다. 공변 Hessian,
국소 양자채널, 스펙트럼·역문제, wormhole·thin-shell·Casimir처럼 반례 뒤에도
전제를 좁혀 남는 구조는
[이론물리 보존 원장](docs/검증_원장/참조_이론물리_보존_원장.md)에 분리했다.

## 7. 열린 bridge와 예측 조건

- CE와 \(\Lambda\)CDM을 같은 데이터·공분산·자유도에서 비교한 완전한 model
  selection 결과는 아직 없다.
- 양자 진폭에서 완전양성 jump process와 Poisson offspring로 가는
  microscopic bridge가 열려 있다.
- \(H_0\), \(S_8\), 재가열과 continuum·UV 완비성은 열려 있다.
- 외부 입력, fitted nuisance, 같은 release 내부 교차검증은 독립 예측으로 세지 않는다.
- 핵융합과 공학 branch는 작용에서 reactor gain이나 점화 에너지까지의 산출이
  아직 완결되지 않았다.
- 물리 주장을 정리로 쓰려면 전제·정의역·증명이 필요하고, 자연에 대한 예측으로
  쓰려면 독립 자료·불확도·공분산과 사전 고정한 판별 기준이 추가로 필요하다.

## 8. 재현

Python 환경이 준비돼 있다면 다음 명령으로 핵심 계산을 재현할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bootstrap_solver.py -q
.\.venv\Scripts\python.exe tests\scorecard.py
.\.venv\Scripts\python.exe tests\run_validation.py
.\.venv\Scripts\python.exe examples\physics\proof_completion_attempt.py
.\.venv\Scripts\python.exe -m pytest tests\test_dimensionless.py -q
.\.venv\Scripts\python.exe docs\2_경로적분과_응용\validate_manuscript.py
```

전체 회귀:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 9. 읽는 순서

1. [문서 지도](docs/README.md)
2. [검증 규약](docs/검증_원장/경로적분_검증_규약.md)
3. [형식 구조 원장](docs/검증_원장/경로적분_전체_진리값_감사.md)
4. [공리·기호 사전](docs/axium.md)
5. [계산 체인](docs/경로적분.md)
6. [상수 후보식과 검증](docs/상수.md)

`docs/2_경로적분과_응용/`은 장별 전제·증명·공리·산출을,
`docs/4_공학적_활용/`은 공학 적용의 입력과 미완성 항목을 담는다.

## 10. 주요 검증 코드

| 코드 | 범위 |
| --- | --- |
| [`tests/scorecard.py`](tests/scorecard.py) | 입력을 제외한 정본 수치 스코어카드 |
| [`tests/run_validation.py`](tests/run_validation.py) | 고정점·스코어카드·실제 차원 검사 통합 |
| [`examples/physics/proof_completion_attempt.py`](examples/physics/proof_completion_attempt.py) | 원 주장과 조건부 후손 분리 |
| [`examples/physics/cosmology_ratio_audit.py`](examples/physics/cosmology_ratio_audit.py) | 후기우주 비율 산술 감사 |
| [`examples/physics/ce_residual_forward_model.py`](examples/physics/ce_residual_forward_model.py) | DESI DR2 공분산 전방검사 |
| [`examples/physics/quantum_jump_bridge_gate.py`](examples/physics/quantum_jump_bridge_gate.py) | 양자점프 bridge gate |
| [`examples/physics/fusion_resonance_loop_gate.py`](examples/physics/fusion_resonance_loop_gate.py) | 핵융합 공명 회귀검사 |
| [`examples/physics/fusion_full_loop_gate.py`](examples/physics/fusion_full_loop_gate.py) | 핵융합 전 분기 감사 |
| [`examples/physics/fusion_equation_iteration_gate.py`](examples/physics/fusion_equation_iteration_gate.py) | 퍼텐셜–열반응률 반복 게이트 |
| [`examples/physics/fusion_remaining_branches_gate.py`](examples/physics/fusion_remaining_branches_gate.py) | UV·source·reactor/ICF 잔여분기 |
| [`examples/physics/fusion_direct_scattering_gate.py`](examples/physics/fusion_direct_scattering_gate.py) | Born·Hulthén 핵물리 대조군 |
| [`examples/physics/fusion_floquet_source_gate.py`](examples/physics/fusion_floquet_source_gate.py) | QED Floquet와 CE scalar 비동일성 |
| [`examples/physics/fusion_flavor_aligned_gate.py`](examples/physics/fusion_flavor_aligned_gate.py) | flavor-aligned 후보 제약 |
| [`examples/physics/fusion_flavor_margin_robustness_gate.py`](examples/physics/fusion_flavor_margin_robustness_gate.py) | finite-size·Pb·NA62 강건성 |
| [`examples/physics/fusion_operator_alternatives_gate.py`](examples/physics/fusion_operator_alternatives_gate.py) | 대체 연산자 no-go |
| [`examples/physics/fusion_spin_operator_gate.py`](examples/physics/fusion_spin_operator_gate.py) | spin 연산자 투영과 제약 |
| [`examples/physics/fusion_spin_polarization_control_gate.py`](examples/physics/fusion_spin_polarization_control_gate.py) | 편극 대조군과 source 장부 |
| [`examples/physics/fusion_polarized_evidence_gate.py`](examples/physics/fusion_polarized_evidence_gate.py) | 편극 evidence 회귀검사 |
| [`examples/physics/fusion_sciencedb_payload_gate.py`](examples/physics/fusion_sciencedb_payload_gate.py) | ScienceDB payload 무결성 |
| [`examples/physics/fusion_sciencedb_reactivity_gate.py`](examples/physics/fusion_sciencedb_reactivity_gate.py) | D–T 표의 Maxwellian 반응률 |
| [`examples/physics/fusion_scalar_current_gate.py`](examples/physics/fusion_scalar_current_gate.py) | scalar current와 공동 공분산 결손 |

기존 실행 산출물의 상수명은 호환성을 위해 유지한다. 현재 문서에는 direct D–T
reaction/source/burn/wall 산출을 물리 예측으로 두지 않는다.

## 11. 엔진

`reality_stone/`은 CE 수치 런타임과 Rust/PyO3 백엔드를 담는다.

```text
reality_stone/
  python/reality_stone/          Python API
  python/reality_stone/clarus/   CE 연산·브리지·공학 gate
  src/                           Rust/PyO3 백엔드
tests/                           최상위 회귀·스코어카드
```

```python
import reality_stone.clarus
```

## 12. 읽는 사람에게

이 저장소는 정리, 공리, 산출, 경험식, 미완성 항목과 예측을 서로 다른 출처로
기록한다. 표의 가까운 값만으로 매개변수 감소나 자연법칙의 유도를 주장하지 않는다.
관측 비교는 형식적 증명과 분리해 독립 데이터와 공분산으로 다룬다.

— *To you, two thousand years from now.*
