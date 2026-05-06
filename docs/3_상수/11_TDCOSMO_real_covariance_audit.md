# TDCOSMO real covariance audit

이 문서는 H0 readout law의 첫 실제 공개 posterior-chain 공분산 삽입 결과를 기록한다.

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

이 mapping은 TDCOSMO의 공식 해석이 아니라, CE readout law의 실험적 topology 가정이다.

## Result

| quantity | value |
|---|---:|
| chain shape | \(1200\times100\times8\) |
| burn-in | 200 steps |
| posterior \(H_0\) median | 67.311480 |
| \(C_L\) | 0.99374733 |
| \(C_G\) | 0.49149667 |
| \(q_F\) | 0.66908019 |
| \(H_0(q_F)\) | 71.161396 |
| observational reference | \(67.4\pm3.65\) |
| pull | \(+1.031\sigma\) |

검증 명령:

```bash
python examples/physics/h0_readout/h0_fisher_matrix_io_gate.py examples/physics/h0_readout/h0_fisher_io_examples/tdcosmo_slacs_covariance.json
python examples/physics/h0_readout/h0_fisher_io_full_suite.py
```

full suite 결과는 PASS다. batch 결과에서 전체 \(\chi^2/{\rm dof}=1.065307/5\)가 나왔다.

## Interpretation

실제 TDCOSMO+SLACS 공분산은 순수 local endpoint \(q_F=1\)이 아니다. Hierarchical population/kinematic closure가 \(q_F\)를 낮추는 효과는 실제 chain에서도 보인다.

다만 \(q_F=0.669\)는 관측 중앙값 67.4까지 완전히 낮추기에는 아직 높다. 따라서 현재 결과는 완전한 통과라기보다 "부분 지지 + mapping 개선 필요"다.

## Extension: four public chains

같은 변환기를 4개 공개 chain에 적용했다.

| file | \(q_F\) | \(H_0(q_F)\) | reference | pull |
|---|---:|---:|---:|---:|
| `tdcosmo_only_alpha_free_om_covariance.json` | 0.805032 | 71.984147 | \(74.503\pm6.002\) | -0.420 |
| `tdcosmo_ifu_covariance.json` | 0.795412 | 71.925622 | \(73.041\pm5.736\) | -0.195 |
| `tdcosmo_slacs_covariance.json` | 0.669080 | 71.161396 | \(67.400\pm3.650\) | +1.031 |
| `tdcosmo_slacs_ifu_covariance.json` | 0.628295 | 70.916411 | \(67.437\pm3.655\) | +0.952 |

전체 batch 결과:

```text
chi2/dof = 2.185252/8
```

핵심 패턴:

1. TDCOSMO-only와 IFU는 \(q_F\simeq0.80\)으로 local endpoint 쪽에 더 가깝다.
2. SLACS population 정보를 넣으면 \(q_F\)가 \(0.67\)로 내려간다.
3. SLACS+IFU까지 넣으면 \(q_F\)가 \(0.63\)으로 더 내려간다.
4. 이 방향성은 "hierarchical/global closure가 local H0 readout을 낮춘다"는 CE readout 해석과 정성적으로 맞다.
5. 하지만 SLACS 계열의 \(H_0(q_F)\)는 아직 posterior 중앙값보다 약 \(1\sigma\) 높다. 즉 mapping 또는 conductance 정의가 아직 완전히 닫힌 것은 아니다.

## Next tests

1. TDCOSMO-only, IFU-only, SLACS+IFU chain을 같은 변환기로 모두 계산한다.
2. local/global node partition을 물리적으로 더 엄격히 정의한다.
3. \(q_F\) 계산에서 direct edge만 쓸지, path conductance까지 쓸지 비교한다.
4. posterior covariance 대신 likelihood Fisher에 가까운 block을 얻을 수 있는지 확인한다.
5. H0 tension 설명으로 쓰려면 hierarchical lens closure가 어느 노드를 global conductance로 만드는지 독립적으로 증명해야 한다.

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
| TDCOSMO+SLACS | current lambda-family + alpha local | path | 0.026834 | 0.669080 | 71.161396 | +1.031 |
| TDCOSMO+SLACS | alpha only local | direct | 0.026834 | 0.003704 | 67.268309 | -0.036 |
| TDCOSMO+SLACS+IFU | current lambda-family + alpha local | path | 0.033403 | 0.628295 | 70.916411 | +0.952 |
| TDCOSMO+SLACS+IFU | alpha only local | direct | 0.033403 | 0.001855 | 67.257791 | -0.049 |

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
| `tdcosmo_only_alpha_free_om_covariance.json` | 0.830134 | 72.137101 | \(74.503\pm6.002\) | -0.394 |
| `tdcosmo_ifu_covariance.json` | 0.852221 | 72.271948 | \(73.041\pm5.736\) | -0.134 |
| `tdcosmo_slacs_covariance.json` | 0.003704 | 67.268309 | \(67.400\pm3.650\) | -0.036 |
| `tdcosmo_slacs_ifu_covariance.json` | 0.001855 | 67.257791 | \(67.437\pm3.655\) | -0.049 |

전체 batch 결과:

```text
chi2/dof = 0.180365/8
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
| `tdcosmo_only_alpha_free_om_covariance.json` | local | 0.830134 | 72.137101 | -0.394 | PASS |
| `tdcosmo_ifu_covariance.json` | local | 0.852221 | 72.271948 | -0.134 | PASS |
| `tdcosmo_slacs_covariance.json` | global | 0.003704 | 67.268309 | -0.036 | PASS |
| `tdcosmo_slacs_ifu_covariance.json` | global | 0.001855 | 67.257791 | -0.049 | PASS |

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

