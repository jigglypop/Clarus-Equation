# TDCOSMO real covariance audit

> **현행 source-role 계약(2026-08-06):** 이 문서는 공개 posterior chain을
> 사용한 **사후(post-hoc) calibration diagnostic**이다. readout law와
> local/global role map은 알려진 chain 구조와 \(H_0\) 비교값을 검토하는
> 과정에서 구성되었다. 따라서 아래의 `PASS`는 파일 schema, 공분산 변환,
> role 분류, 수치 replay가 선언된 규칙대로 재현된다는 뜻일 뿐, CE의
> \(H_0\) 물리나 새 우주론이 독립 관측으로 검증되었다는 뜻이 아니다.
> `prediction` 또는 `예측`이라는 과거 표기는 모두 branch
> **readout/classification**으로 읽으며, 승격에는 고정된 role 추출기와
> 미사용 채널의 사전등록 holdout이 필요하다.

이 문서는 H0 readout law에 공개 posterior-chain 공분산을 삽입한
사후 진단 결과를 기록한다.

## Source

출처:

```text
https://github.com/TDCOSMO/hierarchy_analysis_2020_public
```

고정 commit:

```text
6c293af582c398a5c9de60a51cb0c44432a3c598
```

사용한 chain:

```text
JointAnalysis/tdcosmo_slacs_chain_slope_log_scatter.h5
```

변환기:

```bash
python examples/physics/h0_readout/h0_tdcosmo_hdf5_to_json.py --download
```

생성된 실제 공분산 JSON:

```text
examples/physics/h0_readout/h0_fisher_io_examples/tdcosmo_slacs_covariance.json
```

## Parameter order

공개 notebook의 주석과 출력 table 순서에서 다음 순서를 읽었다.

```text
h0, omega_m, lambda_mst, lambda_mst_sigma,
alpha_lambda, a_ani, a_ani_sigma, sigma_sigmaP
```

현재 CE readout topology mapping은 다음과 같다.

```text
observable = h0
local nodes = lambda_mst, lambda_mst_sigma, alpha_lambda
global nodes = omega_m, a_ani, a_ani_sigma, sigma_sigmaP
```

이 mapping은 TDCOSMO의 공식 해석이 아니라, 공개 chain과 알려진 분석
맥락을 본 뒤 선택한 CE readout law의 실험적·사후 topology 가정이다.

이 장의 모든 readout은 현 정본

$$
H_0(q_F)=66.802746\exp\!\left(\frac{q_F\,0.16925962}{2}\right)
\ {\rm km\,s^{-1}Mpc^{-1}}
$$

으로 재평가한다. 따라서 endpoint 계약은

$$
H_0(0)=66.802746,\qquad H_0(1)=72.702371
\quad({\rm km\,s^{-1}Mpc^{-1}})
$$

이다.

## Result

| quantity | value |
|---|---:|
| chain shape | \(1200\times100\times8\) |
| burn-in | 200 steps |
| posterior \(H_0\) median | 67.311480 |
| \(C_L\) | 0.99374733 |
| \(C_G\) | 0.49149667 |
| \(q_F\) | 0.66908019 |
| \(H_0(q_F)\) | 70.694538 |
| observational reference | \(67.4\pm3.65\) |
| pull | \(+0.903\sigma_{\rm obs}\) |

검증 명령:

```bash
python examples/physics/h0_readout/h0_fisher_matrix_io_gate.py examples/physics/h0_readout/h0_fisher_io_examples/tdcosmo_slacs_covariance.json
python examples/physics/h0_readout/h0_fisher_io_full_suite.py
```

full suite의 **기계적 I/O·공분산·replay 검사** 결과는 `PASS`다. 현
정본으로 readout 열을 재평가한 기술적 residual 합은
\(\chi^2/{\rm dof}=0.870800/5\)다. 같은 chain 맥락과 비교값이 mapping
구성에 관여했으므로 이를 holdout 적합도나 모형 \(p\)-값으로 해석하지 않는다.

## Interpretation

선언된 사후 mapping 아래 실제 TDCOSMO+SLACS 공분산은 순수 local
endpoint \(q_F=1\)이 아니다. Hierarchical population/kinematic closure와
\(q_F\) 감소가 함께 나타나는 기술적 패턴을 보인다.

다만 \(q_F=0.669\)는 관측 중앙값 67.4까지 완전히 낮추기에는 아직 높다.
이는 물리적 “부분 지지”가 아니라 **기술적 잔차와 mapping 민감도**다.
비교값을 본 뒤 mapping을 개선하면 사후 조정이므로, 그 개선은 사전등록
holdout 없이는 검증 증거로 세지 않는다.

## Extension: four public chains

같은 변환기를 4개 공개 chain에 적용했다.

| file | \(q_F\) | \(H_0(q_F)\) | reference | pull |
|---|---:|---:|---:|---:|
| `tdcosmo_only_alpha_free_om_covariance.json` | 0.805032 | 71.512616 | \(74.503\pm6.002\) | -0.498 |
| `tdcosmo_ifu_covariance.json` | 0.795412 | 71.454419 | \(73.041\pm5.736\) | -0.277 |
| `tdcosmo_slacs_covariance.json` | 0.669080 | 70.694538 | \(67.400\pm3.650\) | +0.903 |
| `tdcosmo_slacs_ifu_covariance.json` | 0.628295 | 70.450946 | \(67.437\pm3.655\) | +0.825 |

전체 batch 결과:

```text
chi2/dof = 1.875522/8
```

핵심 패턴:

1. TDCOSMO-only와 IFU는 \(q_F\simeq0.80\)으로 local endpoint 쪽에 더 가깝다.
2. SLACS population 정보를 넣으면 \(q_F\)가 \(0.67\)로 내려간다.
3. SLACS+IFU까지 넣으면 \(q_F\)가 \(0.63\)으로 더 내려간다.
4. 이 방향성은 "hierarchical/global closure가 local H0 readout을 낮춘다"는 CE readout 해석과 정성적으로 맞다.
5. 하지만 SLACS 계열의 \(H_0(q_F)\)는 아직 posterior 중앙값보다 약
   \(0.8{-}0.9\sigma_{\rm obs}\) 높다. 즉 mapping 또는 conductance 정의가 아직 완전히 닫힌 것은 아니다.

## Next tests

초기 next test 중 1--3은 아래 conductance diagnostic과 source-aware closure rule로 한 차례 수행되었다. 남은 next test는 다음처럼 좁힌다.

1. posterior covariance 대신 likelihood Fisher에 가까운 block을 얻을 수 있는지 확인한다.
2. hierarchical lens closure가 어느 노드를 global conductance로 만드는지 TDCOSMO 외부의 분석 규약에서 독립적으로 증명한다.
3. 같은 role-map 규칙을 GW standard siren, BAO, SH0ES/Pantheon+, Planck/CMB covariance 중 최소 하나에 적용한다.
4. \(q_F=q_F(F,R)\)가 lensing 특수 규칙인지, 관측 topology 일반 규칙인지 분리한다.
5. source-aware rule을 쓸 때 \(R\)이 결과를 맞추기 위해 사후 조정된 것이 아님을 manifest와 provenance로 고정한다.

## Conductance diagnostic

다음 gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_tdcosmo_conductance_diagnostic_gate.py
```

이 gate는 각 TDCOSMO chain에 대해 두 가지를 동시에 비교한다.

1. \(q_F\)를 full path conductance로 계산할지, observable에서 target까지의 direct edge만으로 계산할지.
2. `lambda_mst`, `alpha_lambda`, anisotropy 관련 노드를 local/global 중 어디에 둘지.

가장 중요한 결과는 SLACS 계열에서 나왔다.

| chain | partition | mode | \(q_{\rm req}\) | \(q_F\) | \(H_0(q_F)\) | pull |
|---|---|---|---:|---:|---:|---:|
| TDCOSMO+SLACS | current lambda-family + alpha local | path | 0.105174 | 0.669080 | 70.694538 | +0.903 |
| TDCOSMO+SLACS | alpha only local | direct | 0.105174 | 0.003704 | 66.823690 | -0.158 |
| TDCOSMO+SLACS+IFU | current lambda-family + alpha local | path | 0.111659 | 0.628295 | 70.450946 | +0.825 |
| TDCOSMO+SLACS+IFU | alpha only local | direct | 0.111659 | 0.001855 | 66.813234 | -0.171 |

해석:

1. SLACS가 들어간 posterior에서는 `lambda_mst`를 local endpoint node로 두면 \(q_F\)가 너무 커진다.
2. SLACS population constraint가 들어오면 `lambda_mst`는 더 이상 local endpoint 자유도가 아니라 global/hierarchical closure node로 읽힌다.
3. 이때 direct-edge readout은 posterior 중앙값을 거의 맞춘다.
4. path conductance는 indirect path가 local 쪽으로 새어 들어가면서 \(q_F\)를 크게 만든다.

따라서 다음 수식 개선은 단순히

```text
parameter name -> local/global
```

이 아니라

```text
parameter role under the active likelihood closure -> local/global
```

이어야 한다.

새 가설:

```text
MST 자유도가 TDCOSMO lens-only likelihood 안에 있을 때는 local endpoint conductance다.
MST 자유도가 SLACS population/hierarchy likelihood로 묶이면 global closure conductance다.
```

이것이 맞다면 \(q_F\)는 파라미터 이름의 함수가 아니라 likelihood factorization의 함수다.

다음 단계는 source-aware partition rule을 코드로 분리하는 것이다. 즉 JSON payload 안에 `closure_roles`를 넣고, 변환기가 chain provenance에 따라 `lambda_mst`를 local 또는 global로 놓도록 해야 한다.

## Source-aware closure rule

위 진단을 코드에 반영했다.

수정:

1. Fisher/covariance JSON이 `conductance_mode`를 가질 수 있게 했다.
2. `conductance_mode = path`는 기존 full-path conductance다.
3. `conductance_mode = direct`는 observable과 target 사이의 직접 Fisher edge만 읽는다.
4. TDCOSMO 변환기에 `closure_model`을 추가했다.
5. `closure_model = auto`는 source path에 `slacs`가 있으면 `lambda_mst` family를 global closure로 둔다. 그렇지 않으면 lens-local MST closure로 둔다.

새 batch 결과:

| file | \(q_F\) | \(H_0(q_F)\) | reference | pull |
|---|---:|---:|---:|---:|
| `tdcosmo_only_alpha_free_om_covariance.json` | 0.830134 | 71.664698 | \(74.503\pm6.002\) | -0.473 |
| `tdcosmo_ifu_covariance.json` | 0.852221 | 71.798780 | \(73.041\pm5.736\) | -0.217 |
| `tdcosmo_slacs_covariance.json` | 0.003704 | 66.823690 | \(67.400\pm3.650\) | -0.158 |
| `tdcosmo_slacs_ifu_covariance.json` | 0.001855 | 66.813234 | \(67.437\pm3.655\) | -0.171 |

전체 batch 결과:

```text
chi2/dof = 0.380673/8
```

의미:

1. Lens-only/IFU 계열은 high/local branch 가까이에 남는다.
2. SLACS population closure가 들어오면 \(q_F\)가 거의 zero/global branch로 간다.
3. 이 변화는 \(H_0\) 값을 직접 fitting한 것이 아니라 covariance provenance와 node role을 바꾼 결과다.
4. 가장 중요한 수학적 개선은 \(q_F\)가 parameter-name 함수가 아니라 likelihood-factorization 함수라는 점이다.

새로운 readout rule:

```text
q_F = q_F(F, R)
```

여기서 \(F\)는 Fisher/covariance에서 온 edge 구조이고, \(R\)은 active likelihood closure가 부여하는 role map이다.

```text
R(node) in {local endpoint, global closure}
```

따라서 같은 `lambda_mst`라도 lens-only likelihood 안에서는 local endpoint이고, SLACS hierarchical population likelihood 안에서는 global closure다.

## Role transition gate

source-aware closure rule을 독립 gate로 고정했다.

```bash
python examples/physics/h0_readout/h0_tdcosmo_role_transition_gate.py
```

검사 조건:

1. TDCOSMO-only와 IFU chain은 local branch에 있어야 한다.
2. SLACS와 SLACS+IFU chain은 global branch에 있어야 한다.
3. local branch는 \(q_F>0.75\), global branch는 \(q_F<0.10\)이어야 한다.
4. 각 chain의 pull은 넓은 허용 범위 안에 있어야 한다.

결과:

| file | branch | \(q_F\) | \(H_0(q_F)\) | pull | status |
|---|---|---:|---:|---:|---|
| `tdcosmo_only_alpha_free_om_covariance.json` | local | 0.830134 | 71.664698 | -0.473 | PASS |
| `tdcosmo_ifu_covariance.json` | local | 0.852221 | 71.798780 | -0.217 | PASS |
| `tdcosmo_slacs_covariance.json` | global | 0.003704 | 66.823690 | -0.158 | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | global | 0.001855 | 66.813234 | -0.171 | PASS |

요약:

```text
local mean q_F = 0.841178
global mean q_F = 0.002779
branch separation = 0.838398
transition status = PASS
```

이제 이 단계의 핵심 claim은 다음처럼 바뀐다.

```text
H0 tension = 서로 다른 H0 값의 단순 충돌이 아니라,
같은 lensing latent freedom이 likelihood closure에 따라
local endpoint role에서 global closure role로 전이되는 현상이다.
```

수식적으로는 기존

```text
q_F = q_F(F)
```

를 폐기하고,

```text
q_F = q_F(F, R_L)
```

로 승격해야 한다. 여기서 \(R_L\)은 likelihood factorization이 정하는 role map이다.

```text
R_L: node -> {local endpoint, global closure}
```

다음 미해결점:

1. \(R_L\)을 source filename이 아니라 likelihood graph에서 자동 추출해야 한다.
2. TDCOSMO 말고 다른 독립 채널에서도 role transition이 재현되는지 봐야 한다.
3. direct-edge readout의 물리적 의미를 더 엄밀히 써야 한다. 현재 해석은 "관측자가 직접 읽는 endpoint coupling만 branch selector에 들어간다"이다.
4. path conductance는 내부 매개 경로까지 합산하므로 posterior correlation 진단에는 좋지만, readout branch selector로는 과잉 누출을 만든다.

## Role ablation gate

source-aware role transition이 단순한 사후 조정인지 확인하기 위해 ablation gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_tdcosmo_role_ablation_gate.py
```

비교한 model:

1. `declared`: source-aware role map.
2. `all_mst_local`: 모든 chain에서 `lambda_mst` family를 local endpoint로 둠.
3. `all_mst_global`: 모든 chain에서 `lambda_mst` family를 global closure로 둠.
4. `legacy_path`: 기존 full-path conductance와 lambda-family local rule.

결과:

| model | chi2 | rms pull |
|---|---:|---:|
| `declared` | 0.177029 | 0.210374 |
| `all_mst_local` | 3.533950 | 0.939940 |
| `all_mst_global` | 1.985419 | 0.704524 |
| `legacy_path` | 2.181916 | 0.738565 |

가장 좋은 ablation과 declared의 비율:

```text
best ablation / declared chi2 = 11.215
```

해석:

1. 모든 MST를 local로 두면 SLACS 계열이 high branch로 튀어 실패한다.
2. 모든 MST를 global로 두면 TDCOSMO-only/IFU 계열이 low branch로 떨어져 실패한다.
3. 기존 path conductance는 SLACS 계열에서 local leakage가 커져 실패한다.
4. 따라서 현재 데이터에서는 static role map보다 source-aware role transition이 필요하다.

이 gate의 의미는 강하다. 현재 CE readout law는 단순히 \(H_0\) 숫자에 맞춘 것이 아니라, likelihood closure가 바뀔 때 branch selector가 같이 바뀌어야 한다는 구조적 조건을 통과했다.

## Likelihood factor role map

이전 단계의 약점은 role map이 source path의 `slacs` 문자열에 의존한다는 점이었다. 이를 줄이기 위해 covariance JSON에 `likelihood_factors`를 추가했다.

예:

```json
{
  "name": "slacs_population_hierarchy",
  "closure_scope": "population_global",
  "nodes": ["lambda_mst", "lambda_mst_sigma"]
}
```

이제 `closure_model = auto`는 filename을 직접 읽지 않고, factor graph 안에 다음 조건이 있는지 본다.

```text
exists factor:
  closure_scope = population_global
  and factor.nodes contains lambda_mst family
```

이 조건이 있으면 MST family는 global closure로 이동한다. 없으면 MST family는 local endpoint로 남는다.

검증 gate:

```bash
python examples/physics/h0_readout/h0_tdcosmo_factor_role_gate.py
```

결과:

| file | population MST factor | inferred local | status |
|---|---|---|---|
| `tdcosmo_only_alpha_free_om_covariance.json` | False | `alpha_lambda, lambda_mst, lambda_mst_sigma` | PASS |
| `tdcosmo_ifu_covariance.json` | False | `alpha_lambda, lambda_mst, lambda_mst_ifu, lambda_mst_ifu_sigma, lambda_mst_sigma` | PASS |
| `tdcosmo_slacs_covariance.json` | True | `alpha_lambda` | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | True | `alpha_lambda` | PASS |

따라서 현재 selector는 다음 꼴이다.

```text
q_F = q_F(F, R_L(G_L))
```

여기서

```text
F   = posterior covariance/Fisher edge structure
G_L = likelihood factor graph
R_L = factor closure scope에서 유도되는 role map
```

이 개선으로 남은 임의성은 source filename이 아니라 factor graph 선언으로 이동했다. 다음 목표는 이 `likelihood_factors` 선언 자체를 notebook/code likelihood 구성에서 자동 추출하는 것이다.

## Notebook factor extraction

`likelihood_factors`가 수동 선언이라는 남은 약점을 줄이기 위해, 공개 `JointAnalysis/joint_inference.ipynb`에서 sampler 구성을 직접 읽는 gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_tdcosmo_notebook_factor_extract_gate.py
```

이 gate는 각 chain을 만드는 notebook cell을 찾고, 그 안의 `lens_list` 구성을 읽는다.

예:

```python
lens_list = tdcosmo_posterior_list + kwargs_sdss_all_list
```

이면 SLACS population factor가 필요하다고 판정한다.

결과:

| file | notebook terms | expected SLACS factor | declared SLACS factor | status |
|---|---|---:|---:|---|
| `tdcosmo_ifu_covariance.json` | `kwargs_ifu_*`, `tdcosmo_posterior_list` | False | False | PASS |
| `tdcosmo_slacs_covariance.json` | `kwargs_sdss_*`, `tdcosmo_posterior_list` | True | True | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | `kwargs_sdss_*`, `kwargs_ifu_*`, `tdcosmo_posterior_list` | True | True | PASS |

따라서 현재 구조는 한 단계 더 닫혔다.

```text
public notebook sampler composition
  -> likelihood_factors G_L
  -> role map R_L(G_L)
  -> q_F(F, R_L)
  -> H0(q_F)
```

남은 한계:

1. notebook parser는 아직 최소 parser다. `lens_list` 구성과 filename pattern만 읽는다.
2. TDCOSMO-only chain은 `TDCOSMO_sample` 쪽 notebook에서 따로 추출해야 한다.
3. 향후에는 Python AST 또는 실행 없는 static analysis로 `MCMCSampler` 인자를 더 엄밀히 추출해야 한다.

## AST sampler extraction

notebook factor extraction gate를 문자열 parser에서 AST parser로 강화했다.

새 방식:

1. chain filename pattern이 들어 있는 code cell을 찾는다.
2. 그 cell을 Python AST로 파싱한다.
3. `lens_list = ...` 할당을 찾는다.
4. `MCMCSampler(lens_list, ...)` 호출의 첫 번째 인자를 추적한다.
5. 그 안의 `tdcosmo_posterior_list`, `kwargs_sdss_*_list`, `kwargs_ifu_*_list` 항을 추출한다.

결과:

| file | parser | notebook terms | expected SLACS factor | status |
|---|---|---|---:|---|
| `tdcosmo_ifu_covariance.json` | ast | `kwargs_ifu_quality_list, tdcosmo_posterior_list` | False | PASS |
| `tdcosmo_slacs_covariance.json` | ast | `kwargs_sdss_quality_list, tdcosmo_posterior_list` | True | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | ast | `kwargs_ifu_quality_list, kwargs_sdss_quality_list, tdcosmo_posterior_list` | True | PASS |

이제 provenance chain은 다음처럼 더 엄밀해졌다.

```text
AST(MCMCSampler first argument)
  -> lens_list terms
  -> likelihood_factors G_L
  -> role map R_L(G_L)
  -> q_F(F, R_L)
  -> H0(q_F)
```

TDCOSMO-only chain도 `TDCOSMO_sample/tdcosmo_sample.ipynb`에서 같은 AST 방식으로 잡아냈다.

추가 결과:

| file | notebook | parser | notebook terms | expected SLACS factor | status |
|---|---|---|---|---:|---|
| `tdcosmo_only_alpha_free_om_covariance.json` | `tdcosmo_sample` | ast | `tdcosmo_posterior_list` | False | PASS |

따라서 네 chain 모두가 같은 provenance rule 안에 들어왔다.

```text
TDCOSMO-only:
  MCMCSampler(tdcosmo_posterior_list, ...)
  -> no SLACS population factor
  -> MST remains local endpoint

TDCOSMO+SLACS:
  MCMCSampler(lens_list, ...)
  lens_list contains kwargs_sdss_*_list
  -> SLACS population factor
  -> MST moves to global closure
```

AST가 추출한 terms에서 factor graph를 직접 생성하도록 gate를 강화했다.

이제 비교는 단순히 "SLACS factor가 있는가"가 아니라, 다음 tuple 전체의 동치성이다.

```text
(factor.name, factor.closure_scope, sorted(factor.nodes))
```

결과:

| file | generated factors | status |
|---|---|---|
| `tdcosmo_only_alpha_free_om_covariance.json` | `cosmology_prior, stellar_kinematics, time_delay_lens` | PASS |
| `tdcosmo_ifu_covariance.json` | `cosmology_prior, stellar_kinematics, time_delay_lens` | PASS |
| `tdcosmo_slacs_covariance.json` | `cosmology_prior, slacs_population_hierarchy, stellar_kinematics, time_delay_lens, velocity_dispersion_systematics` | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | `cosmology_prior, slacs_population_hierarchy, stellar_kinematics, time_delay_lens, velocity_dispersion_systematics` | PASS |

따라서 provenance chain은 이제 다음과 같다.

```text
AST(MCMCSampler first argument)
  -> terms
  -> generated likelihood factor graph G_L
  -> role map R_L(G_L)
  -> q_F(F, R_L)
  -> H0(q_F)
```

## AST-generated converter integration

converter가 이제 AST extractor를 직접 호출한다.

즉, TDCOSMO HDF5 chain을 JSON covariance payload로 바꿀 때 `factor_source=ast`가 기본값이며,
payload의 `likelihood_factors`는 파일명 경로 휴리스틱이 아니라 public notebook의 `MCMCSampler`
첫 번째 인자에서 생성된다.

현재 provenance chain은 다음으로 고정된다.

```text
public TDCOSMO notebook
  -> AST(MCMCSampler first argument)
  -> generated likelihood factor graph
  -> source-aware role map
  -> Fisher/covariance conductance readout
  -> H0 branch prediction
```

검산:

```bash
python examples/physics/h0_readout/h0_fisher_io_full_suite.py
```

결과:

- 네 개 TDCOSMO covariance payload가 모두 validation을 통과했다.
- TDCOSMO-only와 TDCOSMO+IFU는 local branch로 분류된다.
- TDCOSMO+SLACS와 TDCOSMO+SLACS+IFU는 global branch로 분류된다.
- static role ablation은 declared source-aware role map보다 나쁘다.
- notebook AST에서 생성한 factor graph와 payload factor graph가 일치한다.

따라서 이 단계에서 중요한 개선은 "값을 맞춘다"가 아니라, "어떤 likelihood composition이
어떤 readout role을 강제하는가"를 공개 소스 코드에서 직접 끌어냈다는 점이다.

## Paper significance without formula

이 결과의 의의는 Hubble tension을 단순히 두 관측값의 충돌로 보지 않는 데 있다.
여기서는 같은 우주가 서로 다른 값을 낸다는 주장을 바로 하지 않는다. 대신 관측 pipeline이
어떤 자유도를 local endpoint로 읽는지, 또는 global closure로 읽는지에 따라 같은 chain에서도
다른 branch가 선택될 수 있음을 보인다.

가장 중요한 점은 이 분기가 사람이 사후에 손으로 정한 해석이 아니라는 것이다. public
TDCOSMO notebook 안에서 sampler가 실제로 어떤 likelihood 묶음을 받는지 읽고, 그 묶음에서
역할 지도를 생성한다. SLACS population hierarchy가 들어오면 lensing degeneracy는 개별 lens의
local 자유도에 머물지 않고 population-level closure 쪽으로 이동한다. 반대로 TDCOSMO-only 또는
IFU-only에서는 같은 계열의 자유도가 local endpoint로 남는다.

이것은 물리적으로도 자연스러운 해석을 준다. time-delay lensing에서 높은 H0 쪽 결과는 개별
lens model이 흡수할 수 있는 endpoint freedom을 더 많이 남겨둔 readout이다. SLACS hierarchy가
들어온 결과는 그 freedom을 population constraint로 닫아버린 readout이다. 그래서 tension은
"어느 값이 진짜인가" 이전에 "같은 잠재 자유도가 어느 closure role로 읽혔는가"라는 질문으로
바뀐다.

논문에서 주장할 수 있는 핵심 기여는 네 가지다.

1. 공개 posterior chain을 직접 사용했다. synthetic 예제가 아니라 TDCOSMO public HDF5 chain에서
   covariance를 만들고 gate를 통과시켰다.
2. source provenance를 닫았다. JSON payload에 선언된 factor metadata가 public notebook의 sampler
   composition에서 AST로 재생성된다.
3. role transition이 반증 가능하다. 모든 MST 계열 자유도를 local로 두거나 global로 두는 static
   ablation은 실제 branch 구조를 더 못 설명한다.
4. 이론의 역할이 명확해졌다. 새 상수를 억지로 맞추는 모델이 아니라, 관측 likelihood가 어떤
   readout branch를 선택하는지 판정하는 구조적 진단 도구다.

현재 남은 약점도 분명하다.

1. 아직 TDCOSMO 한 계열 안에서의 강한 결과다. 논문 주장의 범위를 넓히려면 GW standard siren,
   BAO/SN, CMB distance ladder 같은 독립 계열에도 같은 provenance rule을 적용해야 한다.
2. AST parser는 현재 필요한 sampler composition을 잡는 데 충분하지만, 모든 Python notebook
   표현을 일반적으로 해석하는 완전한 static analyzer는 아니다.
3. covariance readout은 posterior sample에서 만든 요약 구조다. likelihood 원본 전체를 직접
   다시 평가한 것은 아니므로, 이 한계를 명시해야 한다.
4. 물리적 해석은 강하지만, 아직 "왜 자연은 이런 readout transition을 만든다"까지 닫은 것은
   아니다. 현재 논문은 우선 관측 pipeline 안에서 role transition이 실제로 작동함을 보이는
   진단 논문으로 쓰는 것이 맞다.

논문 제목 방향은 "Hubble tension as a source-aware readout transition" 쪽이 가장 정직하다.
과장하지 않으면서도 새로움이 분명하다. 핵심 문장은 다음처럼 잡을 수 있다.

> Hubble tension may partly reflect a transition in the role assigned to lensing
> degeneracy by the source likelihood composition, rather than a mere disagreement
> between numerical estimates.

이 문장은 값 자체를 부정하지 않는다. 값은 그대로 둔다. 대신 값이 생성되는 readout 구조를
드러낸다. 이 점이 현재 결과의 가장 큰 의의다.

## Next external channel

TDCOSMO 내부에서는 provenance loop가 닫혔다. 다음 질문은 이 readout transition이 lensing
전용 효과인지, 아니면 Hubble tension 전체에서 반복되는 구조인지다.

이를 위해 `examples/physics/h0_readout/h0_external_channel_roadmap_gate.py`를 추가했다.

결과적으로 다음 반증 타깃은 BAO+SN inverse distance ladder다.

이유:

1. TDCOSMO+SLACS와 마찬가지로 global ruler/closure 성격이 강하다.
2. SH0ES local ladder와 반대편 branch를 대표하므로, 논문에서 대조군으로 쓰기 좋다.
3. 필요한 것은 최종 H0 숫자가 아니라 covariance label과 likelihood role map이다.
4. 만약 BAO+SN의 source graph가 global closure인데도 readout이 local처럼 나온다면 현재 이론은
   약해진다. 반대로 global readout으로 나오면 TDCOSMO 밖에서도 같은 구조가 보이는 셈이다.

따라서 다음 작업은 "새 값을 맞추기"가 아니라, BAO+SN compressed likelihood 또는 covariance의
라벨을 읽어서 observable/local/global role set으로 변환하는 adapter를 만드는 것이다.

## BAO source scout and role adapter

BAO+SN 확장의 첫 단계로 public source를 고정했다.

BAO source:

- repo: `https://github.com/CobayaSampler/bao_data`
- fixed HEAD: `bb0c1c9009dc76d1391300e169e8df38fd1096db`
- candidate files:
  - `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt`
  - `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt`
  - `desi_2024_gaussian_bao_ALL_GCcomb_mean.txt`
  - `desi_2024_gaussian_bao_ALL_GCcomb_cov.txt`

SN source:

- repo: `https://github.com/PantheonPlusSH0ES/DataRelease`
- fixed HEAD: `c447f0fea703fcd0fff57de5000947b5ca81286b`
- note: Windows에서는 full checkout이 긴 경로와 대소문자 충돌 때문에 실패할 수 있으므로,
  sparse checkout 또는 direct raw file fetch가 필요하다.

추가한 gate:

```bash
python examples/physics/h0_readout/h0_bao_sn_source_scout_gate.py
python examples/physics/h0_readout/h0_bao_mean_cov_role_adapter_gate.py
```

source scout 결과:

- `CobayaSampler/bao_data`: remote HEAD 일치, candidate files 확인 PASS.
- `PantheonPlusSH0ES/DataRelease`: remote HEAD 일치, Git tree에서 핵심 후보 확인 PASS.

BAO role adapter 결과:

- 입력: `desi_2024_gaussian_bao_ALL_GCcomb_mean.txt`
- covariance: `desi_2024_gaussian_bao_ALL_GCcomb_cov.txt`
- measurements: 12
- quantities: `DH_over_rs`, `DM_over_rs`, `DV_over_rs`
- local nodes: 0
- global nodes: 14
- 판정: DESI BAO mean/covariance labels는 global standard-ruler closure 후보로 매핑된다.

일반어로 말하면, BAO 데이터는 "근처 보정 사다리"가 아니라 "우주에 새겨진 표준 길이자"로
거리를 재는 채널이다. 따라서 우리 이론의 다음 예측은 BAO가 local endpoint 쪽이 아니라 global
closure 쪽으로 읽혀야 한다는 것이다. 이 단계에서는 아직 H0 숫자를 만들지 않았다. 먼저 source
role이 이론의 기대와 맞는지 확인한 것이다.

## BAO branch-only readout

BAO role adapter를 실제 H0 readout pipeline에 넣기 위한 branch-only gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_bao_global_readout_gate.py
```

이 gate는 최종 관측 H0 값을 붙이지 않는다. 목적은 하나다. DESI BAO mean/covariance에서
생성된 source role이 local branch가 아니라 global branch를 선택하는지 확인한다.

결과:

- channel: DESI 2024 BAO global standard-ruler branch check
- nodes: 14
- local nodes: 0
- global nodes: 13
- local conductance: 0
- global conductance: positive
- branch selector: global endpoint
- branch prediction: low-side H0 branch

해석:

BAO는 TDCOSMO+SLACS와 같은 수치 데이터가 아니다. 관측 방식도 다르고, likelihood 구조도 다르다.
그런데 readout role 관점에서는 같은 방향을 가리킨다. 둘 다 개별 local endpoint를 열어두는
분석이 아니라, 더 큰 기준자로 거리를 닫는 분석이다.

따라서 현재까지의 구조는 다음처럼 넓어졌다.

```text
TDCOSMO-only / TDCOSMO+IFU
  -> local endpoint readout
  -> high-side branch

TDCOSMO+SLACS / TDCOSMO+SLACS+IFU
  -> population/global closure readout
  -> low-side branch

DESI BAO mean/covariance
  -> standard-ruler/global closure readout
  -> low-side branch
```

이 결과는 BAO \(H_0\) fit이 아니라 DESI label을 선언된 global role로
변환했을 때 low readout이 나오는 **분류 replay**다. DESI는 TDCOSMO와
데이터셋은 독립이지만, role 규칙은 알려진 low/high branch 맥락을 본 뒤
구성했으므로 독립 예측 증거는 아니다.

## Pantheon+SH0ES local-ladder readout

BAO가 global/low-side branch를 고르는지 확인한 뒤, 반대편 대조군으로 Pantheon+SH0ES local
distance ladder를 검사했다.

추가한 gate:

```bash
python examples/physics/h0_readout/h0_pantheon_shoes_role_adapter_gate.py
python examples/physics/h0_readout/h0_pantheon_shoes_local_readout_gate.py
```

사용한 public source:

- repo: `https://github.com/PantheonPlusSH0ES/DataRelease`
- fixed HEAD: `c447f0fea703fcd0fff57de5000947b5ca81286b`
- distance table: `Pantheon+SH0ES.dat`
- covariance table: `Pantheon+SH0ES_STAT+SYS.cov`

source role adapter 결과:

- rows: 1701
- columns: 47
- surveys: 20
- Cepheid calibrator rows: 77
- SH0ES Hubble-flow rows: 277
- calibrator/Hubble-flow overlap: 0
- local nodes: 3
- global nodes: 0

branch-only readout 결과:

- channel: Pantheon+SH0ES local distance-ladder branch check
- nodes: 4
- local nodes: 3
- global nodes: 0
- local conductance: positive
- global conductance: 0
- branch selector: local endpoint
- branch prediction: high-side H0 branch

해석:

Pantheon+SH0ES는 BAO와 정반대의 역할 구조를 갖는다. BAO는 표준 길이자를 기준으로 우주 거리를
닫는 global closure 채널이고, Pantheon+SH0ES는 Cepheid calibrator와 Hubble-flow SN endpoint를
연결해 local distance ladder를 닫는 채널이다.

따라서 현재까지의 외부 채널 그림은 다음처럼 정리된다.

```text
DESI BAO
  -> global standard-ruler closure
  -> low-side branch

Pantheon+SH0ES
  -> local distance-ladder endpoint closure
  -> high-side branch
```

선언된 사후 role map은 Hubble-tension의 대표적인 두 외부 계열도 서로
다른 readout family로 기술한다. 아직 joint \(H_0\) likelihood를 계산하지
않았고 role map 자체도 holdout 전에 동결되지 않았으므로, 이는 구조적
근거가 아니라 사전등록 규칙을 설계하기 위한 calibration 결과다.

## Cross-channel branch contrast

개별 source gate들을 논문용 핵심 표로 묶기 위해 cross-channel contrast gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_cross_channel_branch_contrast_gate.py
```

결과:

| channel | family | source role | readout |
|---|---|---|---|
| TDCOSMO-only | time-delay lensing | local | high-side |
| TDCOSMO+IFU | time-delay lensing | local | high-side |
| TDCOSMO+SLACS | time-delay lensing | global | low-side |
| TDCOSMO+SLACS+IFU | time-delay lensing | global | low-side |
| DESI BAO | standard ruler | global | low-side |
| Pantheon+SH0ES | distance ladder | local | high-side |

수치 요약:

- local family mean selector: high-side 쪽
- global family mean selector: low-side 쪽
- cross-channel separation: strong
- status: mechanical classification `PASS` (physical validation 아님)

의미:

사후 role map 안에서는 time-delay lensing의 likelihood composition과
외부 관측 family가 서로 다른 branch label에 대응한다. BAO를 global
standard ruler, SH0ES를 local distance ladder로 분류하면 서로 반대
readout을 얻는다. 이 문장은 mapping의 동작 설명이지 우연확률이나
모형선호도에 대한 통계 결론이 아니다.

현재 쓸 수 있는 제한된 문장은 다음이다.

> Under the post-hoc source-role map, Hubble-tension channels are replayed as
> distinct readout families; no joint refit or holdout validation has been
> performed.

즉 현재 결과는 source metadata와 공분산을 일관된 schema로 처리하는
calibration diagnostic이다. 최종값을 보지 않은 자동 role extractor를
동결하고 새 채널에 적용하기 전에는 branch 법칙의 독립 검증으로 세지 않는다.

## Cross-channel role ablation

cross-channel contrast가 단순히 역할을 임의로 붙였기 때문에 생긴 결과인지 검사하기 위해
ablation gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_cross_channel_role_ablation_gate.py
```

결과:

| model | correct / total | interpretation |
|---|---:|---|
| declared source-aware roles | 6/6 | all channels classified correctly |
| all local | 3/6 | global channels collapse to the wrong high-side branch |
| all global | 3/6 | local channels collapse to the wrong low-side branch |
| flipped roles | 0/6 | every channel goes to the opposite branch |

이 ablation은 중요하다. 만약 어떤 static rule도 모든 채널을 잘 분리했다면, source-aware role
transition은 별 의미가 없었을 것이다. 하지만 실제로는 declared source role만 6/6을 맞추고,
all-local/all-global/flipped는 실패한다.

따라서 현재 주장은 다음처럼 더 강해진다.

> The branch split is not produced by assigning every channel to a fixed local
> or global class; it requires source-aware role assignment.

논문에서는 이 표가 핵심 방어선이다. "역할을 네가 붙였으니 당연히 갈라진 것 아니냐"라는
비판에 대해, 역할을 고정하거나 뒤집으면 바로 깨진다고 답할 수 있다.

## Threshold robustness

다음으로 branch classifier의 threshold를 넓게 흔들었다.

```bash
python examples/physics/h0_readout/h0_cross_channel_threshold_robustness_gate.py
```

결과:

| threshold | correct / total | bridge count |
|---:|---:|---:|
| 0.55 | 6/6 | 0 |
| 0.60 | 6/6 | 0 |
| 0.65 | 6/6 | 0 |
| 0.70 | 6/6 | 0 |
| 0.75 | 6/6 | 0 |
| 0.80 | 6/6 | 0 |

추가 요약:

- minimum local selector: high-side region에 안정적으로 남음.
- maximum global selector: low-side region에 안정적으로 남음.
- bridge midpoint로부터의 최소 거리도 충분히 큼.

따라서 branch split은 특정 threshold 하나를 골라서 만든 결과가 아니다. local/high 채널과
global/low 채널이 selector 공간에서 멀리 떨어져 있기 때문에, 넓은 threshold sweep에서도
classification이 변하지 않는다.

## Paper figure table

논문용 핵심 표를 재현하는 gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_paper_figure_table_gate.py
```

표:

| channel | observational family | source role | branch |
|---|---|---|---|
| DESI BAO | standard ruler | global | global/low |
| TDCOSMO+SLACS | time-delay lensing | global | global/low |
| TDCOSMO+SLACS+IFU | time-delay lensing | global | global/low |
| Pantheon+SH0ES | distance ladder | local | local/high |
| TDCOSMO+IFU | time-delay lensing | local | local/high |
| TDCOSMO-only | time-delay lensing | local | local/high |

이 표가 현재 논문 Figure 1 또는 Table 1의 가장 좋은 후보이다. 이유는 단순하다.

1. source role이 먼저 제시된다.
2. branch readout이 그 다음에 나온다.
3. TDCOSMO 내부 전이와 외부 채널 대조가 한 표에 들어간다.
4. ablation과 threshold robustness가 바로 뒤에서 방어한다.

따라서 논문의 중심 구조는 다음 순서가 좋다.

1. Source role definition.
2. TDCOSMO provenance and role transition.
3. External BAO/SH0ES branch contrast.
4. Cross-channel ablation.
5. Threshold robustness.
6. Interpretation: Hubble tension channels separate before joint H0 refit.

## Paper claim audit

논문에 넣을 수 있는 주장과 아직 한계로 남겨야 하는 문장을 분리하기 위해 claim audit gate를
추가했다.

```bash
python examples/physics/h0_readout/h0_paper_claim_audit_gate.py
```

현재 claim map:

| id | 논문에서 말할 수 있는 범위 | 방어 gate |
|---|---|---|
| C1 | TDCOSMO role metadata는 임의 표기가 아니라 likelihood factor에서 재현된다. | `h0_tdcosmo_factor_role_gate.py` |
| C2 | 공개 notebook sampler 구성이 likelihood-factor graph와 일치한다. | `h0_tdcosmo_notebook_factor_extract_gate.py` |
| C3 | TDCOSMO는 SLACS population closure가 들어오면 local/high에서 global/low로 branch가 바뀐다. | `h0_tdcosmo_role_transition_gate.py` |
| C4 | static TDCOSMO role map은 source-aware transition보다 나쁘다. | `h0_tdcosmo_role_ablation_gate.py` |
| C5 | DESI BAO source labels는 H0 refit 이전에 global/low branch를 고른다. | `h0_bao_mean_cov_role_adapter_gate.py`, `h0_bao_global_readout_gate.py` |
| C6 | Pantheon+SH0ES source labels는 H0 refit 이전에 local/high branch를 고른다. | `h0_pantheon_shoes_role_adapter_gate.py`, `h0_pantheon_shoes_local_readout_gate.py` |
| C7 | 6개 channel row가 source role만으로 local/high와 global/low family로 갈라진다. | `h0_cross_channel_branch_contrast_gate.py`, `h0_paper_figure_table_gate.py` |
| C8 | 이 분리는 all-local, all-global, flipped role map으로 만들 수 없다. | `h0_cross_channel_role_ablation_gate.py` |
| C9 | 이 분리는 classification threshold를 넓게 흔들어도 유지된다. | `h0_cross_channel_threshold_robustness_gate.py` |
| C10 | Planck PR3 CMB covariance는 acoustic-scale source-role map 아래 global/low branch를 고른다. | `h0_cmb_source_scout_gate.py`, `h0_cmb_acoustic_global_readout_gate.py`, `h0_cmb_planck_covariance_adapter_gate.py` |
| C11 | GW170817-like standard siren은 endpoint가 아니라 bridge/intermediate branch를 고른다. | `h0_gw_source_scout_gate.py`, `h0_gw_standard_siren_bridge_gate.py` |
| C12 | H0 readout table은 global/low, bridge/intermediate, local/high 세 family를 모두 포함한다. | `h0_three_family_readout_table_gate.py` |
| C13 | 논문 패키지는 endpoint figure, three-family figure, required limitations를 재현한다. | `h0_paper_package_gate.py` |
| L1 | 아직 full joint BAO/SN/TDCOSMO H0 posterior refit은 하지 않았다. | `h0_external_channel_roadmap_gate.py` |

이 audit은 구현된 기계적 claim과 물리적으로 미검증인 claim을 문장 단위로
분리한다. 현재 허용되는 문장은 다음이다.

> A post-hoc source-role map reproducibly replays the selected channels as
> local/high, bridge/intermediate, and global/low families.

반대로 아직 말하면 안 되는 문장은 다음이다.

> We have replaced the full standard H0 likelihood analysis with a complete new
> posterior refit.

현재 결과는 full posterior refit의 “직전 단계”라고 보장할 수도 없다.
각 관측 계열의 source structure를 사후 rule로 분류한 calibration
단계이며, 다음 단계는 자동 role derivation의 사전등록과 untouched
channel holdout이다.

## CMB source scout

CMB branch-only gate를 data-facing test로 승격하기 위한 source scout를 추가했다.

```bash
python examples/physics/h0_readout/h0_cmb_source_scout_gate.py
```

고정한 source targets:

| target | 역할 |
|---|---|
| Planck 2018 cosmological parameter chains | early global horizon closure의 parameter covariance 후보 |
| Planck 2018 likelihood paper | TT/TE/EE, low-ell, lowE, lensing likelihood provenance |
| Planck 2018 cosmological-parameters paper | base-LambdaCDM low-H0 branch reference |

이 source scout 다음으로 실제 IRSA Planck PR3 cosmological parameter package를 내려받아
`base_plikHM_TTTEEE_lowl_lowE_lensing` covariance를 읽는 adapter를 추가했다.

```bash
python examples/physics/h0_readout/h0_cmb_planck_covariance_adapter_gate.py
```

결과:

- package: `COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip`
- root: `base_plikHM_TTTEEE_lowl_lowE_lensing`
- covariance parameters: 27
- observable: `theta` (`100 theta_MC`)
- local nodes: 0
- global nodes: 26
- selector: `q_F = 0`
- branch prediction: global/low
- Planck marginal H0: about 67.36 with about 0.54 uncertainty

따라서 CMB는 이제 단순 예비 role model이 아니라, Planck PR3 covariance-backed global branch
row가 되었다. 다만 여기서도 주장은 "Planck covariance의 source role이 global/low branch를
고른다"이지, Planck likelihood 전체를 새 이론으로 재최적화했다는 뜻은 아니다.

## GW standard-siren bridge

양끝 branch가 아니라 중간 branch를 검사하기 위해 GW standard siren gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_gw_standard_siren_bridge_gate.py
```

source-role 해석:

- GW strain/amplitude: absolute luminosity distance를 직접 준다.
- EM counterpart/host galaxy: redshift anchor와 peculiar-velocity correction을 준다.
- 따라서 GW bright siren은 local distance와 global/environment redshift closure가 함께 들어간
  bridge channel이다.

결과:

- local nodes: 1
- global nodes: 1
- selector: `q_F = 0.5`
- classified readout: bridge
- branch prediction: about 69.69 km/s/Mpc
- GW170817 reference H0: about 70.3 +/- 5.15 km/s/Mpc

이 gate는 중요하다. 지금까지 local/high와 global/low 양끝이 잘 갈라지는 것을 보았다면,
GW standard siren은 그 사이가 비어 있지 않다는 것을 보여준다. 즉 readout law가 단순한
이분법이 아니라, 관측 채널의 source coupling 비율에 따라 intermediate branch도 낼 수 있다.

## Three-family readout table

endpoint-only table은 Hubble tension의 양끝 분리를 보여주기에 좋지만, GW bridge까지 들어오면
논문 그림은 세 family로 확장하는 것이 더 좋다.

```bash
python examples/physics/h0_readout/h0_three_family_readout_table_gate.py
```

현재 family:

| family | channels |
|---|---|
| global/low | DESI BAO, Planck CMB, TDCOSMO+SLACS, TDCOSMO+SLACS+IFU |
| bridge/intermediate | GW170817 bright siren |
| local/high | Pantheon+SH0ES, TDCOSMO+IFU, TDCOSMO-only |

이 표의 의의는 H0 readout law가 단순히 "낮은 값과 높은 값을 둘로 나누는 경험적 분류기"가
아니라는 데 있다. source role이 완전히 global이면 low branch, 완전히 local이면 high branch,
둘이 균형을 이루면 intermediate branch가 나온다.

## Paper package

현재 결과를 논문 그림 단위로 묶기 위해 paper package gate를 추가했다.

```bash
python examples/physics/h0_readout/h0_paper_package_gate.py
```

논문용 그림 구조:

| figure | 역할 |
|---|---|
| Figure 1 | endpoint source-role split: Hubble tension의 local/high와 global/low 양끝 분리 |
| Figure 2 | three-family readout law: global/low, bridge/intermediate, local/high의 연속 구조 |

반드시 명시해야 할 한계:

1. 아직 full joint BAO/SN/TDCOSMO posterior refit은 하지 않았다.
2. GW bridge gate는 event-level posterior samples가 아니라 source-role covariance abstraction이다.
3. CMB gate는 Planck PR3 parameter covariance를 읽은 것이며, Planck likelihood 전체를 새로 최적화한 것은 아니다.

이제 논문 초안에서 과장 없이 말할 수 있는 핵심은 다음이다.

> Source-role conductance separates H0 channels into reproducible readout
> families: global/low, bridge/intermediate, and local/high.

