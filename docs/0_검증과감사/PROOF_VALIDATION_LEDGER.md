# CE 증명 보강 및 검증 Ledger

이 문서는 CE 문서군을 처음부터 검증할 때의 진행 순서를 고정한다. 목적은 모든 주장을 한 번에 `Exact`로 올리는 것이 아니라, 각 단계가 어느 등급에서 닫히는지와 어떤 코드/반증 조건으로 검증되는지를 분리하는 것이다.

## 0. 등급 규칙

| 등급 | 의미 | 승격 조건 |
|---|---|---|
| `Exact` | 정의, 순수 수학, 기능방정식으로 닫힘 | 추가 물리 식별 없이 증명 가능 |
| `Selection` | 유일성, 분기 선택, 정규화 선택 | 대안 가지가 명시적으로 배제됨 |
| `Bridge` | 수학 구조를 물리 관측량에 연결 | 독립 관측 또는 표준모형 매칭 필요 |
| `Phenomenology` | 보정, 동결 시점, 유효 모형 | 데이터/시뮬레이션 gate 필요 |
| `Open` | 현재 재개방 | scorecard에 넣지 않음 |
| `Open test` | 다음 실험/데이터로 검증 예정 | 사전등록 판정 필요 |

## 1. 생존 함수와 부트스트랩

### 1.1 생존 함수

**명제:** 독립 접힘 구간에 대해 생존 함수가 곱성을 만족하고, 연속성/측정가능성/단조성 중 하나와 정규화가 주어지면

$$
S(D_1+D_2)=S(D_1)S(D_2),\quad S(0)=1
$$

의 물리 가지는

$$
S(D)=e^{-\lambda D}
$$

이다. 단위 접힘 규약을 $\lambda=1$로 잡으면 $S(D)=e^{-D}$다.

**현재 판정:** `Exact/Selection`.

**주의:** 곱성 자체가 경로적분 측도에서 자동으로 나오는 것은 아니다. 독립 접힘 구간을 정의하는 단계는 물리적 모형 조건이며, 이 조건이 깨지면 $S(D)=e^{-D}$는 코어 정의가 아니라 유효 근사로 내려간다.

### 1.2 부트스트랩 고정점

**명제:** $D>1$에서 물리 가지 $x\in[0,1/D]$에 대해

$$
x=e^{-(1-x)D}
$$

는 유일한 수축 고정점을 가진다.

**닫힌 해:**

$$
x=-\frac{W_0(-De^{-D})}{D}.
$$

**현재 판정:** `Selection`.

**코드 검증:** `reality_stone/python/reality_stone/clarus/bootstrap_solver.py`

검증 결과:

| 항목 | 값 |
|---|---:|
| $D_{\text{eff}}$ | 3.17776 |
| $x=\varepsilon^2$ | 0.0486466333 |
| residual | $2.08\times10^{-17}$ |
| Newton vs bracketed 차이 | $1.25\times10^{-13}$ |

실행:

```powershell
python reality_stone\python\reality_stone\clarus\bootstrap_solver.py
uv run --extra dev python -m pytest tests\test_bootstrap_solver.py -q
```

### 1.3 바리온 대응

**명제 후보:** 생존분율 $P_{\text{survive}}$를 관측 가능한 선택 측도 $I(P)$로 읽고, 연속성/정규화/독립 곱성을 부과하면 $I(P)=P^c$이며 약억압 한계의 1차항 일치로 $c=1$이 고정된다.

**현재 판정:** 함수형 고정은 `Selection`, 현재 우주의 $\Omega_b$와 동일시하는 마지막 단계는 `Bridge`.

**반증 조건:** CMB-S4 등에서 $\Omega_b$가 $0.0486466333$과 3 sigma 이상 안정적으로 불일치하면 A3b 브리지를 하향 또는 폐기한다.

## 2. 차원 선택

**명제:** 장세기 $F$는 2-form이고, Hodge 별표가 $*F$를 다시 벡터/1-form으로 보내려면 $d-2=1$, 즉 $d=3$이어야 한다.

**현재 판정:** 공간 차원 $d=3$ 선택은 `Selection`.

**주의:** $d=3$에서 색 기본표현 차원 $N_c=3$을 상속한다는 단계는 외부 회전군과 내부 색 게이지군의 동역학적 동일시가 필요하므로 `Bridge`로 유지한다.

## 3. 전자약-강력 연결

**명제 후보:**

$$
\sin^2\theta_W=4\alpha_s^{4/3}.
$$

**현재 판정:** 강하게 제약된 `Bridge`.

**검증 코드:** `examples/physics/alpha_s_closure_gate.py`

**남은 간극:** Yang-Mills 작용의 차원 비분리와 결합상수 곱적 분배의 동일시. 비섭동 상계는 간극을 줄이지만, 엄밀한 `Exact` 승격은 아니다.

## 4. 이번 패스에서 고친 검증 도구

| 파일 | 변경 |
|---|---|
| `reality_stone/python/reality_stone/clarus/bootstrap_solver.py` | Newton 야코비안 부호 수정, SciPy 없는 bracketed fallback 추가, 콘솔 출력 ASCII 안정화 |
| `tests/run_validation.py` | standalone bootstrap 야코비안 부호 수정 |
| `tests/test_bootstrap_solver.py` | Newton/bracketed 일치, residual, 야코비안 수치미분 회귀 테스트 추가 |
| `docs/2_경로적분과_응용/10_공리_정당화.md` | 삭제된 `check_unification.py` 참조를 현재 `alpha_s_closure_gate.py`로 갱신 |
| `docs/경로적분.md` | 동일 재현 경로 갱신 |

## 5. 이번 패스 검증 결과

```text
python reality_stone\python\reality_stone\clarus\bootstrap_solver.py
PASS: eps^2 = 0.0486466333
Residual = 2.08e-17
Newton vs bracketed = 1.25e-13

uv run --extra dev python -m pytest tests\test_bootstrap_solver.py tests\test_dimensionless.py tests\test_layer_a.py tests\test_bridge_gates.py -q
50 passed, 2 warnings

python examples\physics\alpha_s_closure_gate.py
alpha_s(M_Z) = 0.1173  (observed 0.1179, -0.49%)
sin2thetaW  = 0.2297  (observed 0.23122, -0.64%)
Verdict: alpha_s and sin2thetaW reproduced to <1% with zero free dimensionless parameters.

python tests\run_validation.py
Bootstrap solver: PASS
Constants scorecard: 85.7% pass rate
Dimensional analysis: PASS
Overall: VALIDATED
Remaining warning: Omega_Lambda WARN (+2.75 sigma)
```

경고는 PyTorch sparse CSR beta/invariant 안내이며, 이번 수학 검증 실패는 아니다.

## 6. 다음 순서

1. `3_상수/3_부트스트랩.md`의 관측 비교값과 `tests/scorecard.py`, `tests/run_validation.py`의 관측 기준을 통일한다.
2. $D_{\text{eff}}=3+\delta$ 유도는 `Bridge`로 유지하되, 힉스 포탈 유일성 및 Z 채널 가중치의 대안 배제 조건을 더 명확히 쓴다.
3. `A_s`는 raw 총응답 실패와 projected residual-drive readout을 분리하고, 조건부 `Phenomenology`로 둔다.
4. $|V_{cb}|$는 LO 실패를 유지하되, NLO electroweak projector bridge로 조건부 `Phenomenology`에 올린다.

## 7. CKM 첫 판정: $|V_{cb}|$

`tests/run_validation.py` 기준:

| 항목 | 값 |
|---|---:|
| CE LO 후보 $\alpha_s^{3/2}$ | 0.040478 |
| strict 평균 기준 | 0.041530 |
| sigma | 0.000160 |
| offset | 6.58 sigma |

판정: `Open`. `docs/3_상수/4_입자물리.md`와 `docs/상수.md`에서 통과 항목이 아니라 재검토 항목으로 하향했다.

남은 작업:

1. inclusive/exclusive/PDG-average 기준을 분리한다.
2. $\alpha_s^{3/2}$가 LO 주항이라면 필요한 상대 보정은 약 $+2.6\%$임을 명시한다.
3. CKM unitarity, HQE, form-factor 보정 중 어느 것이 CE 내부에서 파라미터 없이 나오는지 검토한다.

## 8. 전체 증명 상태표

전체 문서군의 항목별 판정은 `PROOF_STATUS_MATRIX.md`에 고정한다.

현재 결론:

| 묶음 | 판정 |
|---|---|
| 생존 함수와 자기재귀 고정점 | 조건부 `Exact/Selection` |
| solver와 무차원성 검산 | 코드 기준 `Exact` |
| $d=3$ 선택 | `Selection` |
| $\Omega_b$, $\sin^2\theta_W$, Higgs, PMNS | `Bridge` |
| $\Omega_\Lambda$, $\Omega_{DM}$, $w_0$, $|V_{us}|$, $n_s$ | `Phenomenology` |
| $|V_{cb}|$ | `Phenomenology` |
| Clarus boson, 공학/뇌/AGI 응용 gate | `Open test` 또는 `Phenomenology` |

추가 검산:

```text
python tests\scorecard.py
Total entries: 23
Scored bridge/phenom rows: 11
PASS: 9
WARN: 2
FAIL: 0
EXACT/reference rows: 9
OPEN rows: 2
OPEN TEST rows: 1

WARN:
- Omega_b h^2: -2.28 sigma
- Omega_Lambda: +2.75 sigma

OPEN/excluded:
- Clarus field pole bridge
```

주의: `tests/run_validation.py`는 `Open` 항목도 일부 strict scorecard에 강제로 넣어 실패를 드러내는 감사 도구이고, `tests/scorecard.py`는 현재 문서 판정에 맞춰 `Open`을 채점에서 제외하는 장부 도구다. 따라서 두 도구의 숫자는 서로 다른 질문에 대한 답이다.

## 9. 증명 완료 재시도 결과

`examples/physics/proof_completion_attempt.py`를 추가해 `Open` 항목의 다음 후보를 기계적으로 검산했다.

```text
|V_cb| LO                    obstruction       0.040477665  sigma -6.58
|V_cb| NLO candidate         conditional_pass  0.041622824  sigma +0.58
|V_us| tree                  obstruction       0.23122207   sigma +9.84
|V_us| one-loop candidate    conditional_pass  0.2269636    sigma +0.97
n_s transition-count         conditional_pass  0.96503475   sigma +0.03
A_s raw                      obstruction       7.84e-09     sigma +1686.59
A_s readout candidate        conditional_pass  2.104e-09    sigma -0.47
```

판정 변경:

| 항목 | 이전 | 이후 | 이유 |
|---|---|---|---|
| $|V_{us}|$ | `Open` | `Phenomenology` | tree 식은 실패하지만 $1/(1+\alpha_s/2\pi)$ 보정이 strict 기준 1 sigma 안에 들어온다. |
| $n_s$ | `Open` | `Phenomenology` | $1-2/(dD_{\rm eff}12/2)$가 수치적으로 닫힌다. 단, 전이수 12와 inflationary readout은 가정이다. |
| $|V_{cb}|$ | `Open` | `Phenomenology` | LO 실패는 유지하고, $\delta/(2\pi)$ electroweak projector bridge가 strict 기준 +0.58 sigma로 통과한다. |
| $A_s$ | `Open` | `Phenomenology` | 총응답 raw는 reject, projected residual-drive readout은 strict 기준 -0.53 sigma로 통과한다. |

검증:

```text
uv run --extra dev python -m pytest tests\test_proof_completion_attempt.py -q
5 passed

python tests\scorecard.py
Scored bridge/phenom rows: 13
PASS: 11
WARN: 2
FAIL: 0
OPEN: 0
OPEN TEST: 1
Scored pass rate: 84.6%
```

## 10. $A_s$ readout gate

`examples/physics/primordial_spectrum_readout_gate.py`를 추가해 $A_s$의 raw 실패와 readout 후보를 분리했다.

핵심 판정:

| 읽기 | $A_s\times10^9$ | sigma | 판정 |
|---|---:|---:|---|
| total fixed-point response $|dx/dD|$ | 7.83532 | +1685.21 | reject |
| local residual drive $x(1-x)$ | 5.60008 | +1027.79 | reject |
| phase projected drive $(2/\pi)x(1-x)$ | 2.26963 | +48.24 | candidate |
| integer geometry projected drive $(2/\pi)\sigma^{3/4}x(1-x)$ | 2.10604 | +0.13 | pass |
| effective geometry projected drive $(2/\pi)\sigma^{D_{\rm eff}/(D_{\rm eff}+1)}x(1-x)$ | 2.10381 | -0.53 | pass |

해석: CMB 밀도요동은 고정점이 새 위치로 재조정되는 총응답이 아니라, 자기재귀 잔차 방정식의 국소 구동항을 관측하는 것으로 읽는다. 이 선택은 `Exact`가 아니라 `Phenomenology/Bridge`이며, raw 실패 반례를 유지한다.

## 11. $|V_{cb}|$ NLO projector gate

`examples/physics/ckm_vcb_nlo_gate.py`를 추가해 LO 실패와 NLO projector 후보를 분리했다.

| 읽기 | 값 | sigma | 판정 |
|---|---:|---:|---|
| LO QCD tunneling $\alpha_s^{3/2}$ | 0.040477665 | -6.58 | reject |
| NLO electroweak projector $\alpha_s^{3/2}(1+\delta/2\pi)$ | 0.041622824 | +0.58 | pass |
| half phase $1+\delta/\pi$ | 0.042767983 | +7.74 | control reject |
| quarter phase $1+\delta/(4\pi)$ | 0.041050244 | -3.00 | control reject |
| QCD phase $1+\alpha_s/(2\pi)$ | 0.041237138 | -1.83 | control outside 1 sigma |

판정: `Phenomenology`. LO 식은 그대로 실패로 보존하고, 관측 CKM 원소에는 charged-current electroweak projector가 한 폐위상 루프 평균으로 붙는다고 읽는다. 이는 `Exact`가 아니라 no-free-parameter NLO bridge다.

## 12. Clarus field pole/correlation gate

`examples/physics/clarus_boson_search_gate.py`를 추가해 미관측 예측의 판정 조건을 사전등록했다. 여기서 "Clarus boson"은 코어에서 독립 입자를 먼저 가정한 것이 아니라, 클라루스장 2점 상관함수의 pole을 입자언어로 읽은 bridge다.

등록값:

| 항목 | 값 |
|---|---:|
| $\delta$ | 0.17775842 |
| coupling matching 후보 $\lambda_{HP}=\delta^2$ | 0.03159806 |
| $m_\phi=m_p\delta^2$ | 29.64757 MeV |
| 3-sigma 질량창 | 28.388--30.908 MeV |
| Compton length | 6.65575 fm |

판정 규칙:

| 실험 결과 | gate 판정 |
|---|---|
| 질량창 안의 pole-compatible 5 sigma 신호 | `pole_confirmed` |
| 질량창 안의 5 sigma 미만 excess | `pole_candidate` |
| 질량창 전체와 bridge coupling benchmark 동시 배제 | `bridge_rejected` |
| 질량창만 부분 배제하거나 coupling benchmark 미도달 | `bridge_constrained` |
| 그 외 | `open_test` |

X17류 17 MeV 신호는 질량창 밖이므로 Clarus pole hit로 세지 않는다. 이
gate는 발견 주장이 아니라 실험 판정표다. 여기의 \(29.65\) MeV light-pole
후보와 \(m_\Phi\simeq v\sqrt{\lambda_{HP}}=43.77\) GeV 포탈 질량
benchmark는 서로 다른 bridge 가정이며 같은 pole로 동시에 읽지 않는다.
`bridge_rejected`가 나오더라도 반증되는 것은 국소 스칼라/포탈 readout이지,
경로적분 수렴 구조로서의 클라루스장 자체가 아니다.

## 13. A1/Q0 공변 작용 루프

### 13.1 재개방된 명제

기존의 “스칼라 action의 보통 Hessian이므로 공변 텐서이고,
\(T_{\mu\nu}\)에 대응한다”는 사슬을 철회했다. 비선형 field
reparameterization 아래 보통 Hessian에는 action gradient에 비례하는
추가항이 생긴다. 경로 평균도 이 변환 실패를 자동으로 없애지 않는다.

현재 판정:

| 명제 | 판정 |
|---|---|
| 보통 functional Hessian은 일반 field redefinition 아래 tensor다 | 반례 때문에 채택 불가 |
| connection을 포함한 field-space Hessian은 공변화 후보가 된다 | `Conditional` |
| A1 Hessian kernel이 국소 \(\mathcal K_{\mu\nu}\)다 | `Open` |
| \(\mathcal K_{\mu\nu}=T_{\mu\nu}\) | `Open` |
| 총 stress tensor 보존 | diffeomorphism invariant \(\Gamma_{\rm ren}\), anomaly 부재, EOM과 경계조건 아래 `Conditional` |
| CE+SM Q0 작용 완료 | `Open` |

상세 증명 의무는
`docs/0_검증과감사/A1_Q0_COVARIANT_ACTION_LOOP.md`의
`Q0.0`–`Q0.8`에 고정한다.

### 13.2 실행 게이트

```powershell
python examples/physics/a1_q0_action_gate.py
python -m pytest -q tests/test_a1_q0_action_bridge.py
```

현재 국소 결과:

```text
tensor pullback Hessian       20
ordinary Hessian              32
non-tensor extra term         12
covariant Hessian             20
h-phi cross Hessian           0
legacy portal invisible BR    0.771656
supplied upper limit           0.11
legacy benchmark allowed      False
covariant action complete     False
stress tensor derived         False
spectral density derived      False
```

\(Z_2,\ v_\Phi=0\) 포탈에서 cross-Hessian 0은 상호작용 부재를 뜻하지
않는다. \(h\Phi^2\), \(h^2\Phi^2\) vertex가 남는다. 그 정규화를 그대로
사용한 레거시
\(\lambda_{\rm HP}=0.0316,\ m_\Phi=43.77\,\mathrm{GeV}\) benchmark는
\(\Gamma_{\rm inv}\simeq13.75\,\mathrm{MeV}\),
\(\mathrm{BR}_{\rm inv}\simeq0.772\)로, 문서가 공급한 상한 \(0.11\)을
통과하지 못한다. 이는 선택적 포탈 benchmark의 반증이며, 독립 on-shell
scalar를 두지 않는 코어 분기 자체의 반증은 아니다.

## 14. 뇌 프로그래밍 언어 역공학 구조 gate

`docs/6_뇌/10_신경프로그래밍언어_역공학.md`에 fixed neuron opcode,
population latent IR, 재사용 instruction과 language-like composition을
서로 다른 가설로 분리했다. 첫 실행물은 실제 신경 데이터 분석이 아니라,
알려진 finite-state stochastic transducer를 회수하는 synthetic 방법
대조군이다.

```powershell
python examples/brain/neural_language_gate.py
python -m pytest tests/test_neural_language_gate.py -q
```

고정 manifest 실행 결과:

| 항목 | 값 | 판정 |
|---|---:|---|
| held-out context boundary accuracy | 0.921875 | pass |
| held-out vs leave-one-out context TV | 0.043852 | pass |
| nuisance predictive gain | 0.000000 | pass |
| early-to-late reuse | 0.920750 | pass |
| held-out-tuple `A→B` composition | 0.859677 | pass |
| same-target shuffled-transition control | 0.250806 | reject |
| held-out tuple lookup | 0.220161 | reject |
| seen-tuple lookup memorization | 0.848670 | positive control pass |
| self-feedback open-loop / severed | 1.000000 / 0.000000 | pass |
| cross-assembly feedback open-loop / severed | 1.000000 / 0.207500 | pass |

종합 상태는 `SYNTHETIC_ORACLE_LABELED_METHOD_CONTROL_PASS`다. 이것은
선언된 경계·상태·primitive label을 알고 있는 생성 모형에서 분석 구현이
같은 held-out target의 operator 합성과 tuple lookup을 구별하고,
open-loop feedback가 edge severing보다 낫다는 뜻뿐이다. assembly 경계를
spike-like 관측에서 역추정한 결과도 아니고, base case와 call/return을
가진 언어적 재귀, `NL4` 호출 제어, `NL5` 선택적 인과 교란/rescue,
`NL6` 동물·세션 밖 복제를 통과한 결과도 아니다.

다음 세 플래그는 strict manifest와 report에서 항상 `False`이고, `True`로
입력하면 loader가 거부한다.

```text
full_brain_language_identified = False
neural_clarus_assembly_validated = False
causal_instruction_set_validated = False
```

다음 병목은 공개 다과제 영장류 자료에서 primitive 정의 자료와 시험 자료를
분리하고, 보지 않은 조합에서 frozen primitive composition을 lookup,
동일 차원 latent, monolithic recurrent model, movement/arousal 대조군과
비교하는 일이다. 이 실제 자료 loop가 실패하면 language 주장을 올리지 않고
연속 population dynamics 가설로 돌아간다.

## 15. 실제 processed-data neural-code 역추적 gate

합성 방법 대조군 다음으로 Tafazoli et al.의 공식 processed 자료와 공개
MATLAB 코드를 checksum 기준으로 역추적했다. 이 loop는 하나의 전역
`maybe` 대신 주장별 `YES / NO / TEST_UNAVAILABLE`를 출력한다.

### 15.1 과제 코드 골격

세 기록 과제는 다음 두-slot 후보로 정확히 분해된다.

```text
S1 = READ_SHAPE → ROUTE_AXIS_1
C1 = READ_COLOR → ROUTE_AXIS_1
C2 = READ_COLOR → ROUTE_AXIS_2
S2 = READ_SHAPE → ROUTE_AXIS_2  [predicted, not recorded]
```

따라서 task-design factorization과 missing-cell prediction은 `YES`다.
그러나 `S2`가 기록되지 않아 이것을 neural compositional grammar의
통과로 세지 않는다.

### 15.2 classifier snapshot 구조 감사

`PFC_ClassifierData.mat`의 403열은 한 번에 기록한 population이 아니라
서로 다른 날짜를 붙인 pseudopopulation이다. `TrainStimInds`의 동일
signature로 27개 기록 session을 복원했다.

```text
session neuron counts:
21,17,14,8,12,7,9,7,2,21,30,27,19,29,23,25,25,15,9,9,15,6,14,8,6,12,13

Chico: 97 neurons
Silas: 306 neurons
time: 81 bins, -0.25..0.55 s, 10 ms step
counting window: 100 ms
adjacent-window overlap: 90%
saved classifier snapshots: 1
```

250개 classifier 행은 독립 동물이나 trial이 아니라 재표집 반복이다.
저장 코드는 첫 CV fold 필드를 반복마다 덮어써 마지막 snapshot 하나만
남긴다. 둘째 차원은 train/test 분리 전 rule 전체 평균을 뺀 transductive
전처리가 있어 primary discovery에서 제외했다. 그러므로 403-wide
PCA/HMM은 금지하고 session-local 분석만 허용한다.

### 15.3 공식 processed 결과의 허용 범위

cross-task decoder artifact는 다음과 같이 재현됐다.

| 곡선 | raw peak | post-event mean |
|---|---:|---:|
| color C1→C2 | 0.7182 | 0.5973 |
| color C2→C1 | 0.6621 | 0.5830 |
| response C1→S1 | 0.9597 | 0.7958 |
| response S1→C1 | 0.9581 | 0.7947 |

이는 population reuse의 후보 증거이지 blind operator discovery가 아니다.
angle artifact의 직접 shared-colour 비교는 C1↔C2 한 항목이며,
100–300ms 평균 76.81도, lower-tail 0.000999다. 해당 MAT에는
`ResponseLoc` axis가 없으므로 나머지 color–shape 항목을 shared-response
증거로 재해석하지 않는다.

공식 코드의 `TransferEntropyAnalysis` 경로는 transfer entropy가 아니라
smoothed decoder score의 Pearson correlation이다. 양수 대각선 투영에도
유의성 mask를 쓰지 않는다. 따라서 causal information flow 플래그는
계속 `False`다.

### 15.4 session-local stationary operator 반례

27개 session의 첫째·셋째 차원 train trial만 사용했다. label, test set,
`AllFactors`와 둘째 차원은 보지 않았다. 고정 6-fold whole-trial CV에서
train-only Anscombe 변환, 표준화와 PCA를 수행하고, 겹치지 않는 100ms
간격의 선형 operator를 맞췄다.

중앙값 결과:

| 경로 | source-grand-mean 대비 설명력 | 정방향의 역방향 대비 이득 | trial-shuffle 대비 이득 |
|---|---:|---:|---:|
| D1 within | 0.79% | 0.59% | 2.07% |
| D3 within | 1.68% | 0.34% | 2.59% |
| D1→D3 frozen | 1.22% | 0.19% | 2.87% |
| D3→D1 frozen | 2.56% | 0.97% | 3.18% |

event-locked 시간 평균을 train fold에서 제거하면 정방향의 역방향 대비
이득은 -0.26%에서 +0.13% 사이로 사실상 사라졌다. frozen transfer는
target 좌표에 operator를 다시 맞춘 oracle 대조보다 열세였고, 27개 중
D1→D3는 4개, D3→D1은 3개 session에서만 이겼다. rank cap
1/2/3/5에서도 정방향 이득은 1% 미만이었다.

100ms 창을 10ms씩 옮긴 anchor를 반복 가중하는 영향을 없애기 위해
시작점 자체도 100ms 간격으로 제한한 민감도 분석을 추가했다. 이때 네
경로의 방향성 이득은 `+0.66%, -0.02%, -0.12%, +0.46%`였고 frozen
transfer는 두 방향 모두 target-refit보다 열세여서 판정은 변하지 않았다.

따라서 현재 판정은 다음처럼 분리한다.

```text
YES              session_local_short_memory
NO               shared_stationary_directed_operator
NO               state_dependent_switching_operator
NO               latent_common_successor_proxy
NO               frontend_to_common_callee_observational_candidate
NO               state_parent_rank1_proxy
NO               task_inheritance_tree_identified
TEST_UNAVAILABLE common_callee_or_task_inheritance_exists_or_is_absent
NO               brain_programming_language_identified
```

약한 `YES`는 같은 trial의 100ms 자기기억이 successor-shuffle보다
2–3% 낫다는 뜻이다. shared instruction이나 방향성 계산 규칙의 `YES`가
아니다.

### 15.5 상태 전환·공통 callee·상태 계층 대조

stationary operator가 여러 epoch를 평균내 실패했을 가능성을 실제로
시험했다. 상태 수 \(S=2,3\), 기억 깊이 \(P=1,2,3\)을 분리하고,
현재·과거만 받는 train-frozen gate를 사용했다. 판정 비교 \(S=P\)에서
상태별 map과 `VAR(order=S)`의 동역학 parameter 수를 맞췄고, gate와
모형 선택 비용은 held-out Gaussian codelength/BIC proxy에 더했다.
strict 또는 prequential MDL로 부르지 않는다.

54개 `session×dimension` 중앙값:

| \(S=P\) | switching−VAR 설명력 | switching code 이득 | state-parent+rank1 code 이득 | hub−time 설명력 | hub code 이득 |
|---:|---:|---:|---:|---:|---:|
| 2 | −1.41% | −0.4712 bit/scalar | −0.5084 bit/scalar | −0.33% | +0.2284 bit/scalar |
| 3 | −1.80% | −1.1854 bit/scalar | −1.0538 bit/scalar | −0.73% | −0.0138 bit/scalar |

\(S=P=2\)의 event-time mean 제거 민감도에서도 switching−VAR 설명력은
`−1.20%`, code 이득은 `−0.4725 bit/scalar`, hub−time 설명력은
`−1.52%`였다. 일부 hub 모형은 caller-specific 모형보다 훨씬 짧았지만
time-only보다 예측 오차가 컸고 정방향이 역방향보다 나빴다. 따라서
caller 정보를 버릴 수 있다는 압축 결과를 공통 callee로 승격하지 않는다.

frontend→callee 후보는 within-dimension hub와 같은 상태 수의 D1↔D3
frozen transfer가 동시에 모든 대조를 통과해야 한다. 이 결합 gate도
`NO`다. 상태-parent+rank1은 task inheritance가 아니라 상태 수준
low-rank proxy이며, 이것조차 matched VAR보다 짧지 않았다. 실제 과제
상속은 D1·D3를 paired trial처럼 취급하지 않고 정당화한 공통 과제좌표가
필요하므로 현재 artifact에서는 `identified=NO`,
`exists_or_absent=TEST_UNAVAILABLE`로 분리한다.

이 결과는 matched stationary VAR가 시험한 두 discrete proxy보다 나은
상대 승자였음을 뜻한다. 비선형 연속 동역학 전체를 배제하거나 확정한
결과는 아니므로 `continuous_dynamics_ruled_out=NO`를 유지한다.

### 15.6 실행물과 판정

```powershell
uv run --isolated --with scipy python `
  examples/brain/neural_code_reverse_engineering.py
uv run --isolated --with scipy python `
  examples/brain/tafazoli_session_operator_probe.py
uv run --isolated --with scipy python `
  examples/brain/tafazoli_call_graph_probe.py
python -m pytest tests/test_neural_code_reverse_engineering.py -q
python -m pytest tests/test_tafazoli_session_operator_probe.py -q
python -m pytest tests/test_tafazoli_call_graph_probe.py -q
```

실제 claim gate 상태:

```text
YES              task_design_two_slot_code_skeleton_reconstructed
YES              session_local_operator_pilot_possible
NO               shared_population_transition_primitive_identified
NO               state_dependent_switching_operator_identified
NO               frontend_to_common_callee_observational_candidate_supported
NO               task_inheritance_tree_identified
NO               fixed_neuron_opcode_identified
NO               continuous_dynamics_ruled_out
NO               brain_programming_language_identified
TEST_UNAVAILABLE common_callee_architecture_exists_or_is_absent
TEST_UNAVAILABLE task_inheritance_architecture_exists_or_is_absent
TEST_UNAVAILABLE fixed_neuron_opcode_refuted
TEST_UNAVAILABLE unseen_composition_validated
TEST_UNAVAILABLE causal_instruction_set_validated
TEST_UNAVAILABLE brain_programming_language_exists
```

이 판정은 `benchmarks/neural_code_reverse_engineering_v1.json`에서 잠근다.
현재 상태명은
`TASK_CODE_SKELETON_RECONSTRUCTED_NEURAL_LANGUAGE_NOT_IDENTIFIED`다.

### 15.7 session-local 확산·잡음 프록시 판정

공식 classifier MAT의 MD5를 검증한 뒤, 복원된 27개 physical session에서
D1·D3를 서로 대응하는 trial로 취급하지 않고 각각 분석했다. raw와
train-fold event-time mean 제거 민감도를 합쳐
`27 sessions × 2 dimensions × 2 preprocessing = 108`개
session-level unit을 얻었다. 공분산은 outer-training trial 내부 OOF
residual로만 맞췄고, outer-test target은 공분산이나 state gate에
사용하지 않았다.

```powershell
uv run --isolated --with scipy python `
  examples/brain/tafazoli_diffusion_probe.py
python -m pytest tests/test_tafazoli_diffusion_probe.py -q
```

공분산 ladder 실행은 `YES`다. 그러나 current-state scale이
full covariance, event-time scale, quadratic-mean drift를 모두 이겨야
한다는 사전등록 gate에서 state-dependent noise proxy는 `NO`였다.

| 판정량 | raw | event-time mean 제거 |
|---|---:|---:|
| state scale의 full covariance 대비 code 이득 (bit/scalar) | -0.2521 | -0.2544 |
| 세 대조군 동시 승리 unit 비율 | 0 | 0 |
| 200 ms semigroup median excess (bit/scalar) | 0.01935 | 0.01720 |
| 200 ms semigroup 통과 비율 | 0.519 | 0.648 |
| 300 ms semigroup median excess (bit/scalar) | 0.06677 | 0.05927 |
| 300 ms semigroup 통과 비율 | 0.0185 | 0.0185 |

공통 anchor와 parameter penalty로 비교한 Markov order는 108/108 unit에서
order 1이 선택됐다. order 1의 code 이득은 order 2 대비
`+0.3177 bit/scalar`, order 3 대비 `+0.6411 bit/scalar`였다. 시험한
family 안에서는 **국소 affine + isotropic Gaussian proxy가 상대
승자**라는 뜻이다. 비-Gaussian family와 절대 적합도는 비교하지 않았으므로
Gaussian innovation law 자체는 식별하지 않았다. 하나의 stationary OU
법칙이 300 ms까지 성립한다는 주장도 지지하지 않는다.

forward/reverse 비교는 primary gate가 아닌 기술적 분류다. 108개 unit 중
reverse lower-code가 73개, forward lower-code가 35개였으며, 이를
generative reverse process나 인과 방향의 증거로 세지 않는다.

```text
YES              session_local_covariance_ladder_completed
YES              model_relative_local_affine_isotropic_proxy_winner
NO               gaussian_innovation_law_identified
NO               state_dependent_noise_proxy_survived_controls
NO               biological_diffusion_identified
NO               generative_reverse_process_identified
NO               score_function_identified
NO               causal_diffusion_mechanism_identified
NO               spatial_graph_diffusion_identified
TEST_UNAVAILABLE biological_diffusion_exists_or_is_absent
```

따라서 다음 식별 난제는 non-autonomous/input-driven 또는 hidden-state SDE와
generic regression/noise를 구분하는 일이다. 이를 위해서는 raw simultaneous
trials, 더 조밀하면서 서로 독립인 시간창, multi-area 기록, 선택적
perturbation이 필요하다. 현재 processed snapshot만으로 생물학적 확산의
보편적 존재나 부재를 판정하지 않는다.
