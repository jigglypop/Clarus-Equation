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
| [02_연속공간과측도.md](02_연속공간과측도.md) | compact 연속공간의 농축 정리, 해 없음의 최소잔차 manifest, non-compact 확장 조건 | 조건부 `Exact`, 확장 `Open/Tooling` |
| [03_조건공간과등호발생.md](03_조건공간과등호발생.md) | 조건 자체가 후보가 되는 joint manifest | 유한공간 `Exact`, 해석 `Selection/Bridge` |
| [04_PreEq_범주.md](04_PreEq_범주.md) | `PreEq_fin` 비음수 커널 범주, 상태 작용, tropical 극한 | `Exact` |
| [05_CE_브리지.md](05_CE_브리지.md) | 세계선 공간, 접힘 에너지, 선택/비선택, \(\phi\) pushforward | `Bridge/Open` |
| [06_측정문제와Born.md](06_측정문제와Born.md) | 양자 측정 후보공간, 측정 조건, Born prior의 지위 | `Bridge/Open` |
| [07_AGI_잔류장.md](07_AGI_잔류장.md) | token/action 후보분포, 비선택 residual channel, hallucination gate 연결 | `Bridge/Open` |

## 읽는 순서

1. [01_공리와증명.md](01_공리와증명.md): 작은 방정식에서 후보분포가 답으로 수렴하는 핵심 증명
2. [02_연속공간과측도.md](02_연속공간과측도.md): 실수공간/compact 공간으로 확장
3. [03_조건공간과등호발생.md](03_조건공간과등호발생.md): 조건 자체도 후보가 되는 joint manifest
4. [04_PreEq_범주.md](04_PreEq_범주.md): `PreEq_fin` 범주와 zero-temperature/tropical 극한
5. [05_CE_브리지.md](05_CE_브리지.md): CE 선택/비선택 경로와 \(\phi\) 잔류장
6. [06_측정문제와Born.md](06_측정문제와Born.md): 양자 측정과 Born prior의 지위
7. [07_AGI_잔류장.md](07_AGI_잔류장.md): AGI runtime 응용

## 다음 도구 보강 계획

현재 01-07장은 도구를 분리한 1차 구조다. 이후에는 아래 항목을 더 보강한다.

| 예정 문서 | 내용 | 필요한 작업 |
|---|---|---|
| `02a_noncompact_Gamma.md` | Polish space, tightness, Gamma-convergence | non-compact 정리 강화 |
| `04a_Markov_Kleisli.md` | Markov category와 확률 monad 비교 | 범주론 표준 용어 정렬 |
| `05a_phi_pushforward.md` | \(\phi(x)=\int K(x,\gamma)d\mu_{\mathrm{ns}}\)의 커널 조건 | CE 잔류장 bridge 정밀화 |
| `06a_Born_prior_유도.md` | \(\mu_0(i)=|c_i|^2\) prior 유일성 | Gleason류 조건 검토 |
| `07a_toy_runtime_gate.md` | token 후보분포와 \(\phi\) 재주입 toy 실험 | 코드 gate |

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
