# 등호 이전 수학

이 폴더는 등호를 쓰기 전에 후보공간·모호함·조건·manifest를 어떤 공리와 측도로 정의하는지 정리한다. 독자는 기초 집합·확률·해석학의 기본을 아는 독자를 전제로 하며, 공리·정의·정리·채택 공리·미완성 다리는 서로 다른 지위로 읽어야 한다.

먼저 폴더 지위와 문서 의존도를 확인한 뒤 유한 후보공간, 연속 compact 공간, 비compact 확장 도구 순으로 읽는다. 각 문서의 정규화·측도·경계 가정이 정의역이며, 반례 또는 가정 위반은 정리를 넓히지 않고 미완성 다리로 남긴다.


## 폴더 지위

폴더 지위는 어떤 내용이 형식 정의·증명·채택 공리·향후 도구인지 구분한다. 문서 존재나 코드 검사는 수학 정리의 증명을 대체하지 않으며, 지위 변경은 명시한 가정과 증명에 의존한다.

이 폴더는 CE 문서군 안에서 `pre-equality` 또는 `ambiguity dynamics`를 독립 연구 축으로 다룬다.

핵심 질문:

> 등호가 표면에 manifest 되기 전, 후보 상태와 조건은 어떤 구조를 갖고 어떻게 한 값 또는 한 조건-값 쌍으로 수렴하는가?

이 폴더는 `1_강의/`의 해설 문서가 아니다. 또한 `참조/`의 보조 노트도 아니다. 유한 후보공간에서 시작해 공리, 정리, 증명, 범주 후보, CE bridge를 단계적으로 세우는 별도 기초 폴더다.

이 폴더의 순수수학 식에서 $E$, $\mathcal I$와 $\beta$는 별도
선언이 없으면 무차원 cost와 무차원 선택 scale이다. 물리 에너지나 작용을
대입할 때에는 $E/(k_BT)$, $S_E/\hbar$ 또는 명시한 기준량의 비로
먼저 무차원화한다.

## 현재 문서

현재 문서 표는 각 장의 입력 정의와 다음 장이 소비하는 정리·도구를 연결한다. 링크는 의존 관계의 안내이며, 다른 문서를 읽지 않아도 성립 범위와 미완성 경계가 드러나야 한다.

| 문서 | 역할 | 형식 출처 |
|---|---|---|
| [01_공리와증명.md](01_공리와증명.md) | 유한 후보공간의 공리계, Gibbs 재가중, manifest 극한 | `[정의]`, `[정리]` |
| [02_연속공간과측도.md](02_연속공간과측도.md) | compact 연속공간의 농축 정리, 해 없음의 최소잔차 manifest | 가정이 명시된 `[정리]` |
| [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md) | $\mathbb R^n$ coercive 농축 정리, 유일 minimizer Dirac 약수렴, Gamma 위치 | 가정이 명시된 `[정리]`; 추가 적용은 `[미완성]` |
| [02b_미분과Jet농축.md](02b_미분과Jet농축.md) | finite-difference jet 후보공간, fixed-$\beta$ 한계, $\beta_h\to\infty$ jet 농축 | `[정리]`; scale은 `[공리: 모델 선택]` |
| [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md) | scale-dependent energy, 국소균등/Gamma 수렴, positive-mass recovery | `[정리]`; 최소집합 내부 선택은 `[미완성]` |
| [03_조건공간과등호발생.md](03_조건공간과등호발생.md) | 조건 자체가 후보가 되는 joint manifest | `[정리]`; 해석은 `[공리: 물리 사상]` |
| [03a_조건주변화와Bayes.md](03a_조건주변화와Bayes.md) | joint 상태의 조건/값 marginal, conditional readout, projected minimizer | `[정의]`, `[정리]`, `[공리: 모델 선택]` |
| [04_PreEq_범주.md](04_PreEq_범주.md) | `PreEq_fin` 비음수 커널 범주, 상태 작용, tropical 극한 | `[정의]`, `[정리]` |
| [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md) | Markov category, weight Kleisli, row mass 분해 | `[정의]`, `[정리]` |
| [04b_Tropical_Functor.md](04b_Tropical_Functor.md) | Gibbs kernel 합성의 zero-temperature/tropical functorial limit | `[정리]` |
| [04c_PreEq_보편스킴.md](04c_PreEq_보편스킴.md) | metric 등호의 보편 인코딩, 선형 manifest $=A^+b$, defect 분리 정리 | `[정리]`; Gibbs functor는 `[미완성]` |
| [05_CE_브리지.md](05_CE_브리지.md) | 무차원 good-rate 경로 재가중, 선택/비선택 subprobability, kernel pushforward | 조건부 `[정리]`; 물리 선택은 `[공리]`, `[미완성]` |
| [05a_phi_pushforward.md](05a_phi_pushforward.md) | raw/conditional 잔류 측도, $K_\phi$ 커널 조건, operator·scalar 구분 | `[정의]`, 조건부 `[정리]`; 물리 사상은 `[미완성]` |
| [05b_phi_kernel_catalog.md](05b_phi_kernel_catalog.md) | endpoint, occupation, curvature, AGI embedding 커널 분류 | `[정의]`, `[공리: 모델 선택]`, `[미완성]` |
| [../검증_원장/등호이전_pathspace_closure_checklist.md](../검증_원장/등호이전_pathspace_closure_checklist.md) | CE 경로공간 농축 정리 적용 조건 | `[정의]`, `[미완성]` |
| [../검증_원장/등호이전_pathspace_audit.md](../검증_원장/등호이전_pathspace_audit.md) | CE 문서의 $\mathcal P_I$, $\mu_{\mathrm{ref}}$, $W/F$, $\mathcal P_{\mathrm{ns}}$ 감사 | 조건부 `[정리]`, 남은 사상은 `[미완성]` |
| [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md) | $W=S_E/\hbar+S_{\mathrm{supp}}$ good-rate 농축 정리와 충분조건 | `[정리]`; action 선택은 `[공리]` |
| [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md) | CE continuum 위상과 $C^1$ 강화조건 | `[정리]`, `[공리: 모델 선택]` |
| [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md) | recovery mass/support 조건과 full-support no-go | `[정리]`; 물리 prior는 `[미완성]` |
| [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md) | finite mesh와 continuum의 joint limit, scaled recovery mass | `[정리]`; continuum prior는 `[미완성]` |
| [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md) | Sobolev--Gaussian prior와 Brownian $C^0/W^{1,p}$ 경계 | `[정리]`; 물리 route 선택은 `[공리]`, `[미완성]` |
| [../검증_원장/등호이전_CE_supp_scaling_audit.md](../검증_원장/등호이전_CE_supp_scaling_audit.md) | suppression 사용처와 scale 규약 감사 | `[산출]`, `[공리: 모델 선택]`, `[미완성]` |
| [05k_CE_hard_constraint.md](05k_CE_hard_constraint.md) | threshold 조건화 no-go, layer-cake와 mean-field 오차 | `[정리]`; 분위수는 `[공리]` 또는 `[미완성]` |
| [05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md) | Kennard, Gaussian action과 Brownian 정칙성의 분리 | `[정리]`; prior·mode 사상은 `[공리]`, `[미완성]` |
| [../검증_원장/등호이전_CE_mode_decomposition_audit.md](../검증_원장/등호이전_CE_mode_decomposition_audit.md) | Gamma mean-field 비, equicorrelation 분산과 scaling 경계 | 조건부 `[정리]`; 물리 mode 구조는 `[미완성]` |
| [../검증_원장/등호이전_CE_cosmology_modern_audit.md](../검증_원장/등호이전_CE_cosmology_modern_audit.md) | 지수 고정점 정리와 우주론 forward 사상의 경계 | `[정리]`, `[공리]`, `[산출]`, `[미완성]` |
| [05o_CE_residual_cosmology_forward_model.md](05o_CE_residual_cosmology_forward_model.md) | 평탄 FLRW·거리·성장 계산과 BAO 중립 데이터 감사 | `[정의]`, `[공리]`, `[산출]`, `[경험식]`, `[미완성]` |
| [06_측정문제와Born.md](06_측정문제와Born.md) | 양자 측정 후보공간, 측정 조건, Born prior | `[공리: 물리 사상]`, `[미완성]` |
| [06a_Born_prior_유도.md](06a_Born_prior_유도.md) | finite branch refinement 아래 Born prior 조건 | 조건부 `[정리]`, `[공리]` |
| [07_AGI_잔류장.md](07_AGI_잔류장.md) | token/action 후보분포와 residual channel | `[정의]`, `[경험식]`, `[미완성]` |
| [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md) | 후보분포, 잔류 압축, $\phi$ 재주입 toy gate | `[경험식]`, `[예측]` |
| [07b_gate_ablation_protocol.md](07b_gate_ablation_protocol.md) | $\alpha_\phi=0$과 $\alpha_\phi>0$의 사전등록 비교 | `[정의]`, `[예측]` |
| [08_수학도구_진행지도.md](08_수학도구_진행지도.md) | 증명·정의·실험 우선순위 | `[정의]` |
| [09_용어사전.md](09_용어사전.md) | operator, scalar, residual, manifest와 출처 표지 | `[정의]` |

독립적인 공변 Hessian, Ward identity, 유클리드 스펙트럼, 국소 채널과
wormhole no-go는
[../검증_원장/참조_이론물리_보존_원장.md](../검증_원장/참조_이론물리_보존_원장.md)에
전제와 함께 보존한다.

## 읽는 순서

읽기 순서는 정의역과 공리부터 정리·예제로 가는 논리 의존 경로다. 순서를 건너뛰면 측도·정규화·compactness 가정이 빠질 수 있으며, 그 경우 결론을 적용하지 않는다.

1. [01_공리와증명.md](01_공리와증명.md): 작은 방정식에서 후보분포가 답으로 수렴하는 핵심 증명
2. [02_연속공간과측도.md](02_연속공간과측도.md): 실수공간/compact 공간으로 확장
3. [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md): non-compact 후보공간에서 escape를 막는 조건
4. [02b_미분과Jet농축.md](02b_미분과Jet농축.md): 미분을 finite-scale jet 후보분포의 manifest로 읽음
5. [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md): 에너지 자체가 변하는 scale-dependent 농축
6. [03_조건공간과등호발생.md](03_조건공간과등호발생.md): 조건 자체도 후보가 되는 joint manifest
7. [03a_조건주변화와Bayes.md](03a_조건주변화와Bayes.md): 조건/값 marginal과 conditional readout
8. [04_PreEq_범주.md](04_PreEq_범주.md): `PreEq_fin` 범주와 zero-temperature/tropical 극한
9. [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md): 표준 Markov/Kleisli 구조와의 정확한 위치
10. [04b_Tropical_Functor.md](04b_Tropical_Functor.md): zero-temperature에서 tropical/min-plus 합성으로 내려감
11. [04c_PreEq_보편스킴.md](04c_PreEq_보편스킴.md): 등호 수학 전반의 보편 인코딩과 적용 경계
12. [05_CE_브리지.md](05_CE_브리지.md): 조건부 경로 농축과 물리 사상의 경계
13. [05a_phi_pushforward.md](05a_phi_pushforward.md): 잔류 측도에서 장으로 내려가는 pushforward 규약
14. [05b_phi_kernel_catalog.md](05b_phi_kernel_catalog.md): endpoint/occupation/curvature/embedding 커널 후보 분류
15. [../검증_원장/등호이전_pathspace_closure_checklist.md](../검증_원장/등호이전_pathspace_closure_checklist.md): CE 경로공간에 농축 정리를 적용하기 위한 가정표
16. [../검증_원장/등호이전_pathspace_audit.md](../검증_원장/등호이전_pathspace_audit.md): 실제 CE 문서에서 이미 닫힌 것과 남은 병목 감사
17. [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md): CE action이 good-rate이면 선택 경로로 농축한다는 정리
18. [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md): CE 경로공간 위상과 action package 선택
19. [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md): CE prior/support와 recovery mass 조건
20. [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md): finite mesh 농축과 continuum CE 농축의 일관성
21. [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md): Sobolev--Gaussian route와 Brownian route의 정확한 support 경계
22. [../검증_원장/등호이전_CE_supp_scaling_audit.md](../검증_원장/등호이전_CE_supp_scaling_audit.md): 실제 문서의 suppression scale 감사와 label 규약
23. [05k_CE_hard_constraint.md](05k_CE_hard_constraint.md): threshold 조건화와 tilt-threshold 대응
24. [05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md): Kennard·Gaussian·Brownian 결과의 정확한 분리
25. [../검증_원장/등호이전_CE_mode_decomposition_audit.md](../검증_원장/등호이전_CE_mode_decomposition_audit.md): intensive/extensive mode scaling과 조건부 오차
26. [../검증_원장/등호이전_CE_cosmology_modern_audit.md](../검증_원장/등호이전_CE_cosmology_modern_audit.md): 고정점 수학과 우주론 사상의 범위 구분
27. [05o_CE_residual_cosmology_forward_model.md](05o_CE_residual_cosmology_forward_model.md): 조건부 FLRW forward 계산과 BAO 중립 데이터 감사
28. [06_측정문제와Born.md](06_측정문제와Born.md): 양자 측정과 Born prior의 지위
29. [06a_Born_prior_유도.md](06a_Born_prior_유도.md): Born prior에 필요한 finite branch 공리
30. [07_AGI_잔류장.md](07_AGI_잔류장.md): AGI runtime 응용
31. [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md): 실험 가능한 toy gate 절차
32. [07b_gate_ablation_protocol.md](07b_gate_ablation_protocol.md): $\phi$ 재주입의 사전등록 비교 기준
33. [08_수학도구_진행지도.md](08_수학도구_진행지도.md): 이후 증명·정의·실험 우선순위
34. [09_용어사전.md](09_용어사전.md): 문서군 기호와 형식 출처 고정

## 남은 도구 보강 계획

계획은 비compact·수렴·측도 도구의 남은 미완성 다리를 표시한다. 계획 항목은 정리나 산출이 아니며, 반례·증명 공백·정의 누락이 닫히기 전에는 승격하지 않는다.

현재 01-07b까지는 도구를 분리한 2차 구조다. 이후에는 아래 항목을 더 보강한다.

| 예정 작업 | 내용 | 필요한 작업 |
|---|---|---|
| Gibbs 배정 functor 승격 | $\mathbf{Def}$ 위에서 $x\mapsto\mu_\beta$의 functor 조건과 반례 경계 | `04d` 후보, 04a/04b와 정렬 |
| 모드 상관 구조 계산 | microscopic covariance에서 $\bar\rho$의 크기와 부호 산출 | 장론 계산, `[미완성]`; 사전 비교는 `[예측]` |
| manifold target 확장 | $\mathbb R^d$ Brownian bridge를 Riemannian manifold로 올림 | manifold bridge 구성, support 정리 확인 |

## 코드 검증

코드 검증은 예제 계산과 기호·수치 회귀를 재현하는 보조 fixture다. 기계적 실행 성공은 정리의 가정·무한 공간·측도론적 결론을 증명하지 않으며, 코드 범위 밖 반례는 별도 처리한다.

과거 패키지 코어는 제거되었고, 현재 재현 범위는 저장소에 남은 예제와 테스트다.

검증:

```powershell
.codex\hooks\python.cmd source
.codex\hooks\python.cmd pytest tests\test_pre_eq_toy_gate.py -q
python -m pytest tests\test_pre_eq_fraction.py -q
python -m pytest tests\test_pre_eq_universal.py -q
python examples\pre_eq\toy_gate_ablation.py
python examples\pre_eq\fraction_threshold.py
```

현재 포함된 회귀검사:

| 항목 | 검증 |
|---|---|
| Gibbs 농축 | 유일 최소 후보로 posterior 집중 |
| 조건 합성 | 순차 재가중 = 에너지 가중합 한 번 적용 |
| 비선택 잔류 | raw mass와 conditional shape 분리 |
| joint Bayes readout | conditional과 free energy marginal 재구성 |
| tropical 극한 | log-sum-exp 오차 $\le\log N/\beta$ |
| Born refinement | branch count prior = $|c_i|^2$, 위상 불변 |
| toy gate ablation | $\alpha_\phi=0$, $\alpha_\phi>0$, shuffled residual 비교 |
| layer-cake 항등식 | $\langle e^{-\Phi}\rangle=\int e^{-t}\mu(\Phi\le t)dt$ 유한공간 정확 일치 |
| mean-field bounds | Jensen 하한과 Var 상한이 tilt를 bracket |
| hard constraint 조건화 | zero-mass 거부, $S_{\mathrm{th}}>m$에서 manifest 불변 |
| threshold scaling | Gamma 분율이 $N/2+z_q\sqrt{N/2}$에서 정규 분위수로 수렴 |
| 경로-에너지 분율 gap | $O(N^{-1/2})$ 스케일, 상수 $\varphi(z)$ 일치 |
| Gaussian 모드 | $U_N$의 Gamma law, 평균·분산과 quantile scaling |
| 경로 정칙성 | Brownian quadratic variation과 $H^1$ zero variation의 분리 |
| 보편 스킴 선형 등호 | manifest = $A^+b$, kernel 방향 prior 분산 보존, least-squares manifest |
| mode 분해 감사 | Gamma Laplace ratio와 equicorrelation 분산식의 수치 회귀 |

## 형식 출처 규칙

형식 출처 규칙은 공리·정의·정리·산출·미완성의 표기를 보존한다. 관측·직관·코드 계산을 증명으로, 채택 선택을 유도로 바꾸지 않는 것이 이 폴더의 기본 경계다.

이 폴더의 활성 주장은 아래 일곱 표지로 읽는다. 기계 회귀 결과는 이 표지를
대신하지 않는다.

| 표지 | 의미 |
|---|---|
| `[정의]` | 기호, 대상, 함수와 계산 규약을 고정 |
| `[정리]` | 명시한 전제와 정의역에서 증명이 완결된 명제 |
| `[공리]` | 수학 전제, 모델 선택, 물리 사상 또는 외부 입력 |
| `[산출]` | 정의·공리·정리의 직접 대입 결과 |
| `[경험식]` | 자료, 보정, 유효 계수 또는 사후 탐색을 포함하는 관계 |
| `[미완성]` | 필요한 정의, 증명, 작용, 사상 또는 독립 자료가 남은 항목 |
| `[예측]` | 입력·범위·비교 절차를 자료 확인 전에 고정한 관측량 |

## 한 줄 요약

요약은 모든 정리의 정의역·정규화·반례 경계를 생략하지 않는 독해 표지다. 한 문장으로 압축된 서술은 증명·채택 공리·미완성 다리의 지위를 바꾸지 않는다.

등호 이후의 수학은 `x+1=2 -> x=1`을 말한다. 이 폴더는

$$
\mu_0
\xrightarrow{E(x)=(x-1)^2,\;\beta}
\mu_\beta
\xrightarrow{\beta\to\infty}
\delta_1
$$

을 공리와 정리로 다룬다.
