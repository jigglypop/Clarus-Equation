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
| [05_CE_브리지.md](05_CE_브리지.md) | 세계선 공간, 접힘 에너지, 선택/비선택, \(\phi\) pushforward | `Bridge` |
| [05a_phi_pushforward.md](05a_phi_pushforward.md) | raw/conditional 잔류 측도, \(K_\phi\) 커널 조건, \(\Phi\) 구분 | 조건부 `Exact`, `Bridge` |
| [06_측정문제와Born.md](06_측정문제와Born.md) | 양자 측정 후보공간, 측정 조건, Born prior의 지위 | `Bridge` |
| [06a_Born_prior_유도.md](06a_Born_prior_유도.md) | finite branch refinement 아래 Born prior 유도 조건 | 조건부 `Exact`, `Bridge` |
| [07_AGI_잔류장.md](07_AGI_잔류장.md) | token/action 후보분포, 비선택 residual channel, hallucination gate 연결 | `Bridge` |
| [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md) | 후보분포, 잔류 압축, \(\phi\) 재주입 toy gate | `Tooling`, `Open/Experiment` |
| [08_수학도구_진행지도.md](08_수학도구_진행지도.md) | 다음에 증명할 자리, 정의를 닫을 자리, 실험으로 보낼 자리의 분류 | `Tooling` |

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
11. [05_CE_브리지.md](05_CE_브리지.md): CE 선택/비선택 경로와 \(\phi\) 잔류장
12. [05a_phi_pushforward.md](05a_phi_pushforward.md): 잔류 측도에서 장으로 내려가는 pushforward 규약
13. [06_측정문제와Born.md](06_측정문제와Born.md): 양자 측정과 Born prior의 지위
14. [06a_Born_prior_유도.md](06a_Born_prior_유도.md): Born prior가 닫히려면 필요한 finite branch 공리
15. [07_AGI_잔류장.md](07_AGI_잔류장.md): AGI runtime 응용
16. [07a_toy_runtime_gate.md](07a_toy_runtime_gate.md): 실험 가능한 toy gate 절차
17. [08_수학도구_진행지도.md](08_수학도구_진행지도.md): 이후 증명/정의/실험 우선순위

## 남은 도구 보강 계획

현재 01-07a까지는 도구를 분리한 2차 구조다. 이후에는 아래 항목을 더 보강한다.

| 예정 문서 | 내용 | 필요한 작업 |
|---|---|---|
| `05b_phi_kernel_catalog.md` | endpoint, occupation, curvature 커널 비교 | CE \(\Phi\)와의 실제 대응 후보 정리 |
| `07b_gate_ablation_protocol.md` | \(\alpha_\phi=0\) vs \(\alpha_\phi>0\) 실험 설계 | toy gate를 코드 실험으로 내림 |
| `09_용어사전.md` | \(\Phi\), \(\phi\), residual, manifest, readout 용어 고정 | 문서군 표기 충돌 방지 |

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
