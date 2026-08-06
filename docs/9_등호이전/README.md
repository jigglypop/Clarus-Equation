# 등호 이전 수학

## 폴더 지위

이 폴더는 CE 문서군 안에서 `pre-equality` 또는 `ambiguity dynamics`를 독립 연구 축으로 다룬다.

핵심 질문:

> 등호가 표면에 manifest 되기 전, 후보 상태와 조건은 어떤 구조를 갖고 어떻게 한 값 또는 한 조건-값 쌍으로 수렴하는가?

이 폴더는 `1_강의/`의 해설 문서가 아니다. 또한 `참조/`의 보조 노트도 아니다. 유한 후보공간에서 시작해 공리, 정리, 증명, 범주 후보, CE bridge를 단계적으로 세우는 별도 기초 폴더다.

## 현재 문서

| 문서 | 역할 | 현재 판정 |
|---|---|---|
| [01_공리와증명.md](01_공리와증명.md) | 유한 후보공간의 공리계, Gibbs 재가중, manifest 극한 | `Exact` |
| [02_연속공간과측도.md](02_연속공간과측도.md) | compact 연속공간의 농축 정리, 해 없음의 최소잔차 manifest | 조건부 `Exact` |
| [02a_noncompact_Gamma.md](02a_noncompact_Gamma.md) | \(\mathbb R^n\) coercive 농축 정리, 유일 minimizer Dirac 약수렴, Gamma 위치 | `Exact under assumptions`, `Open` |
| [02b_미분과Jet농축.md](02b_미분과Jet농축.md) | finite-difference jet 후보공간, fixed-\(\beta\) 한계, \(\beta_h\to\infty\) jet 농축 | `Exact under assumptions`, `Selection` |
| [02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md) | scale-dependent energy, 국소균등/Gamma 수렴, positive-mass recovery | `Exact under assumptions`, `Selection/Open` |
| [03_조건공간과등호발생.md](03_조건공간과등호발생.md) | 조건 자체가 후보가 되는 joint manifest | 유한공간 `Exact`, 해석 `Selection/Bridge` |
| [03a_조건주변화와Bayes.md](03a_조건주변화와Bayes.md) | joint 상태의 조건/값 marginal, conditional readout, projected minimizer | `Exact`, `Selection` |
| [04_PreEq_범주.md](04_PreEq_범주.md) | `PreEq_fin` 비음수 커널 범주, 상태 작용, tropical 극한 | `Exact` |
| [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md) | Markov category, weight Kleisli, row mass 분해 | `Exact`, `Selection` |
| [04b_Tropical_Functor.md](04b_Tropical_Functor.md) | Gibbs kernel 합성의 zero-temperature/tropical functorial limit | `Exact` |
| [04c_PreEq_보편스킴.md](04c_PreEq_보편스킴.md) | metric 등호의 보편 인코딩, 선형 manifest = \(A^+b\), defect가 등호보다 많은 정보라는 분리 정리, 적용 경계 | `Exact`, functor 승격 `Open` |
| [05_CE_브리지.md](05_CE_브리지.md) | 세계선 공간, 접힘 에너지, 선택/비선택, \(\phi\) pushforward | `Bridge` |
| [05a_phi_pushforward.md](05a_phi_pushforward.md) | raw/conditional 잔류 측도, \(K_\phi\) 커널 조건, \(\Phi\) 구분 | 조건부 `Exact`, `Bridge` |
| [05b_phi_kernel_catalog.md](05b_phi_kernel_catalog.md) | endpoint, occupation, curvature, AGI embedding 커널 분류 | `Exact under assumptions`, `Selection/Bridge` |
| [05c_pathspace_closure_checklist.md](05c_pathspace_closure_checklist.md) | CE 경로공간 농축 정리 적용 조건 | `Bridge -> Exact under assumptions 후보` |
| [05d_pathspace_audit.md](05d_pathspace_audit.md) | 실제 CE 문서의 \(\mathcal P_I\), \(\mu_{\mathrm{ref}}\), \(W/F\), \(\mathcal P_{\mathrm{ns}}\) 감사 | 조건부 `Exact`, 남은 `Bridge` 분리 |
| [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md) | \(W=S_E/\hbar+S_{\mathrm{supp}}\) good-rate 농축 정리와 CE 충분조건 | `Exact under assumptions`, 남은 `Selection/Bridge` |
| [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md) | CE continuum 기본 위상을 \(W^{1,p}/C^0\)로 고정하고 \(C^1\) 강화조건 분리 | `Exact under assumptions`, `Selection` |
| [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md) | \(\mu_{\mathrm{base}}\) recovery mass/support 조건, full support 반례, CE reweighting support 동치 | `Exact under assumptions`, 남은 `Bridge/Open` |
| [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md) | finite mesh와 continuum CE 농축의 joint limit, scaled recovery mass, fixed-\(\beta\) convergence | `Exact under assumptions`, 남은 `Bridge/Open` |
| [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md) | scaled Brownian bridge prior, kinetic density 장애물 정리, LDP rate \(=S_E-\min S_E\), \(S_{\mathrm{supp}}\) scaling 갈림길 | `Exact under assumptions`, 남은 `Selection/Open` |
| [05j_CE_supp_scaling_audit.md](05j_CE_supp_scaling_audit.md) | 실제 CE 문서의 suppression 사용처를 `supp:stat`/`supp:dyn`/`supp:hard`로 분류, 긴장 3건 분리 | `Audit/Selection`, 남은 `Open` |
| [05k_CE_hard_constraint.md](05k_CE_hard_constraint.md) | threshold 조건화의 존재/불능, \(N\)-dependent threshold, layer-cake tilt 대응, mean-field 오차 통제 | `Exact under assumptions`, 남은 `Selection/Open` |
| [05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md) | Kennard floor, 모드당 작용 모멘트, threshold 창 scale 고정, mean-field \(1+O(1/N_{\mathrm{eff}})\) 수렴 | `Exact`/`Exact under assumptions`, 남은 `Bridge/Open` |
| [05m_CE_mode_decomposition_audit.md](05m_CE_mode_decomposition_audit.md) | A1 분해가 평균의 분해임을 확정, R1/R2 독법 배제, \(N_{\mathrm{eff}}\ge445\)와 \(\bar\rho\le2.2\times10^{-3}\) 유도 | `Audit`/`Exact under assumptions`, 남은 `Open/Experiment` |
| [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md) | Sobolev-Gaussian/Brownian/finite mesh physical path prior route와 support 판정 | `Exact under assumptions`, 남은 `Selection/Bridge` |
| [06_측정문제와Born.md](06_측정문제와Born.md) | 양자 측정 후보공간, 측정 조건, Born prior의 지위 | `Bridge` |
| [06a_Born_prior_유도.md](06a_Born_prior_유도.md) | finite branch refinement 아래 Born prior 유도 조건 | 조건부 `Exact`, `Bridge` |
| [07_AGI_잔류장.md](07_AGI_잔류장.md) | token/action 후보분포, 비선택 residual channel, hallucination gate 연결 | `Bridge` |
| [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md) | 후보분포, 잔류 압축, \(\phi\) 재주입 toy gate | `Tooling`, `Open/Experiment` |
| [07b_gate_ablation_protocol.md](07b_gate_ablation_protocol.md) | \(\alpha_\phi=0\) vs \(\alpha_\phi>0\) 사전등록 ablation 기준 | `Tooling`, `Open/Experiment` |
| [08_수학도구_진행지도.md](08_수학도구_진행지도.md) | 다음에 증명할 자리, 정의를 닫을 자리, 실험으로 보낼 자리의 분류 | `Tooling` |
| [09_용어사전.md](09_용어사전.md) | \(\Phi\), \(\phi\), residual, manifest, readout 용어 고정 | `Tooling/Definition` |

## 읽는 순서

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
12. [05_CE_브리지.md](05_CE_브리지.md): CE 선택/비선택 경로와 \(\phi\) 잔류장
13. [05a_phi_pushforward.md](05a_phi_pushforward.md): 잔류 측도에서 장으로 내려가는 pushforward 규약
14. [05b_phi_kernel_catalog.md](05b_phi_kernel_catalog.md): endpoint/occupation/curvature/embedding 커널 후보 분류
15. [05c_pathspace_closure_checklist.md](05c_pathspace_closure_checklist.md): CE 경로공간에 농축 정리를 적용하기 위한 가정표
16. [05d_pathspace_audit.md](05d_pathspace_audit.md): 실제 CE 문서에서 이미 닫힌 것과 남은 병목 감사
17. [05e_CE_good_rate_theorem.md](05e_CE_good_rate_theorem.md): CE action이 good-rate이면 선택 경로로 농축한다는 정리
18. [05f_CE_action_topology_package.md](05f_CE_action_topology_package.md): CE 경로공간 위상과 action package 선택
19. [05g_CE_prior_support_package.md](05g_CE_prior_support_package.md): CE prior/support와 recovery mass 조건
20. [05h_CE_finite_to_continuum.md](05h_CE_finite_to_continuum.md): finite mesh 농축과 continuum CE 농축의 일관성
21. [05i_CE_physical_path_prior.md](05i_CE_physical_path_prior.md): physical prior를 scaled Brownian bridge로 고정
22. [05j_CE_supp_scaling_audit.md](05j_CE_supp_scaling_audit.md): 실제 문서의 suppression scale 감사와 label 규약
23. [05k_CE_hard_constraint.md](05k_CE_hard_constraint.md): threshold 조건화와 tilt-threshold 대응
24. [05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md): 불확정성 floor로 요동 scale 고정
25. [05m_CE_mode_decomposition_audit.md](05m_CE_mode_decomposition_audit.md): mode 분해 실재성 감사와 \(N_{\mathrm{eff}}\) 관측 하한
26. [06_측정문제와Born.md](06_측정문제와Born.md): 양자 측정과 Born prior의 지위
27. [06a_Born_prior_유도.md](06a_Born_prior_유도.md): Born prior가 닫히려면 필요한 finite branch 공리
28. [07_AGI_잔류장.md](07_AGI_잔류장.md): AGI runtime 응용
29. [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md): 실험 가능한 toy gate 절차
30. [07b_gate_ablation_protocol.md](07b_gate_ablation_protocol.md): \(\phi\) 재주입 실험 판정 기준
31. [08_수학도구_진행지도.md](08_수학도구_진행지도.md): 이후 증명/정의/실험 우선순위
32. [09_용어사전.md](09_용어사전.md): 문서군 기호와 판정 등급 고정

## 남은 도구 보강 계획

현재 01-07b까지는 도구를 분리한 2차 구조다. 이후에는 아래 항목을 더 보강한다.

| 예정 작업 | 내용 | 필요한 작업 |
|---|---|---|
| Gibbs 배정 functor 승격 | \(\mathbf{Def}\) 위에서 \(x\mapsto\mu_\beta\)의 functor 조건과 반례 경계 | `04d` 후보, 04a/04b와 정렬 |
| 모드 상관 구조 계산 | \(O(g^2)\) 보정의 \(\bar\rho\) 환산 크기와 부호, 05m 허용치 \(2.2\times10^{-3}\)와 비교 | 장론 계산, `Open/Experiment` |
| manifold target 확장 | \(\mathbb R^d\) Brownian bridge를 Riemannian manifold로 올림 | manifold bridge 구성, support 정리 확인 |

## 코드 검증

닫힌 유한 코어는 `reality_stone.clarus.pre_eq`로 내려갔다.

검증:

```powershell
python -m pytest tests\test_pre_eq.py -q
python -m pytest tests\test_pre_eq_toy_gate.py -q
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
| tropical 극한 | log-sum-exp 오차 \(\le\log N/\beta\) |
| Born refinement | branch count prior = \(\lvert c_i\rvert^2\), 위상 불변 |
| toy gate ablation | \(\alpha_\phi=0\), \(\alpha_\phi>0\), shuffled residual 비교 |
| layer-cake 항등식 | \(\langle e^{-\Phi}\rangle=\int e^{-t}\mu(\Phi\le t)dt\) 유한공간 정확 일치 |
| mean-field bounds | Jensen 하한과 Var 상한이 tilt를 bracket |
| hard constraint 조건화 | zero-mass 거부, \(S_{\mathrm{th}}>m\)에서 manifest 불변 |
| threshold scaling | Gamma 분율이 \(N/2+z_q\sqrt{N/2}\)에서 정규 분위수로 수렴 |
| 경로-에너지 분율 gap | \(O(N^{-1/2})\) 스케일, 상수 \(\varphi(z)\) 일치 |
| 불확정성 floor | 모드당 작용 모멘트 \((1/2,1/2)\), mean-field \(1+O(1/N_{\mathrm{eff}})\) |
| 보편 스킴 선형 등호 | manifest = \(A^+b\), kernel 방향 prior 분산 보존, least-squares manifest |
| mode 분해 감사 | R2 독법 \(>100\sigma\) 배제, \(N_{\mathrm{eff,min}}=445\), 상관 상한 \(0.00223\)의 역수 일치 |

## 판정 규칙

이 폴더의 주장은 아래처럼 읽는다.

| 층 | 의미 |
|---|---|
| `Exact` | 유한집합, 비음수 커널, 확률분포, log-sum-exp 극한처럼 수학적으로 닫힌 정리 |
| `Exact under assumptions` | compactness, continuity, support 조건 등 표준 해석학 가정 아래 닫힌 정리 |
| `Selection` | 어떤 후보공간/조건공간/readout을 채택할지의 선택 |
| `Bridge` | CE 경로공간, 양자 측정, AGI runtime으로 올리는 연결 |
| `Open` | 아직 공리, 정리, 반증 조건이 정리되지 않은 큰 주장 |

## 한 줄 요약

등호 이후의 수학은 `x+1=2 -> x=1`을 말한다. 이 폴더는

$$
\mu_0
\xrightarrow{E(x)=(x-1)^2,\;\beta}
\mu_\beta
\xrightarrow{\beta\to\infty}
\delta_1
$$

을 공리와 정리로 다룬다.
