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
| $\lambda_{HP}=\delta^2$ | 0.03159806 |
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

X17류 17 MeV 신호는 질량창 밖이므로 Clarus pole hit로 세지 않는다. 이 gate는 발견 주장이 아니라 실험 판정표다. `bridge_rejected`가 나오더라도 반증되는 것은 국소 스칼라/포탈 readout이지, 경로적분 수렴 구조로서의 클라루스장 자체가 아니다.

## 13. 2026-07-30 우주론·양자론 루프 감사

### 13.1 DESI BAO 고정모델

`examples/physics/ce_residual_forward_model.py`에 p-value, 판정, covariance
기여도와 parameter provenance를 추가했다.

| 모드 | \(\chi^2\) | dof | \(p\) | 판정 |
|---|---:|---:|---:|---|
| CE density + 외부 \(H_0=67.4,r_d=147.09\) 고정 | 37.100260857 | 13 | \(3.9957\times10^{-4}\) | `REJECT` |
| 동일 shape + \(H_0r_d\) scale 1개 진단 fit | 12.608346862 | 12 | 0.398138 | `PASS/Diagnostic` |

복잡도 패널티는 고정모델
\({\rm AIC}={\rm BIC}=37.1003\), scale fit
\({\rm AIC}=14.6083,\ {\rm BIC}=15.1733\)이다.

analytic scale은

$$
q_*
=
\frac{y^\top C^{-1}d}{y^\top C^{-1}y}
=
0.986476933470
$$

이다. 이는 \(r_d=149.106375\,{\rm Mpc}\) 또는
\(H_0=68.323949\,{\rm km\,s^{-1}Mpc^{-1}}\)와 등가다. 같은 DESI 데이터에
맞춘 값이므로 CE 예측으로 세지 않는다. 현재 판정은 “고정 패키지 실패의 주된
원인이 외부 \(H_0r_d\) scale”이라는 원인 분해다.

### 13.2 표준 QM reference gate

`reality_stone.clarus.quantum`에 다음 표준 기준선을 추가했다.

- \(U=\exp(-iH\Delta t/\hbar)\) 일반 Hermitian Hamiltonian 진화
- density matrix와 Born probability/sampling
- GKSL/Lindblad RHS와 positivity를 보존하는 small-system step

이 기준선은 CE 고유 효과가 아니다. 기존 scalar phase step은 \(H=EI\)의
전역위상이라 population·상대위상·간섭을 만들지 못한다는 반례를 회귀로
고정했다.

동시에

$$
\frac{\delta^2S}{\delta\gamma^2}=R
$$

은 Hessian kernel/operator와 Ricci scalar의 type이 다르므로 현 형태로는
승격 불가라고 판정했다. 독립장 \(\phi\ne R\) 또는 constrained curvature
readout 중 하나를 선택해야 한다. Born 분지식은 이미 \(|c_k|^2\)를 입력하므로
Born prior 보존 조건이지 Born rule의 완전 유도가 아니다.

검증:

```text
tests/test_ce_residual_forward_model.py: 21 passed
tests/test_quantum.py: 19 passed
```

상세 수식과 kill rule은
`docs/0_검증과감사/우주론_양자론_루프_감사.md`에 고정한다.

### 13.3 우선순위 보강 실행

외부 \(r_d\)를 대체할 첫 계산 경로로

$$
z_d^{\rm EH}=1020.020419907,
\qquad
r_d^{\rm EH\ hybrid}=151.318753028\ {\rm Mpc}
$$

를 얻었다. 입력은 CE \(\Omega_b,\Omega_m\), 외부 \(H_0,T_{\rm CMB}\),
표준모형 가정 \(N_{\rm eff}=3.044\)와 Eisenstein--Hu fit이다. DESI DR2
13점 full covariance에 넣으면

$$
\chi^2=40.468225544,\qquad
p=1.16176098\times10^{-4},
$$

로 `REJECT`이며 외부 \(r_d=147.09\,{\rm Mpc}\)보다 악화된다. 수치 적분은
물질--복사 닫힌형과 약 \(1.4\times10^{-10}\) 상대오차로 맞으므로 실패 원인은
적분기가 아니라 empirical \(z_d\)와 미구현 precision recombination이다.
DESI는 runtime 입력이 아니지만 이 경로는 DR2 확인 후 추가됐으므로 DR2를
holdout으로 세지 않는다.

EH fit 다음 단계로 외부 \(x_e(z)\) history adapter도 구현했다. convention은

$$
\frac{d\tau_{\rm drag}}{dz}
=
\frac{c\sigma_Tn_e(z)}
{H(z)(1+z)R(z)},
\qquad
R=\frac{3\rho_b}{4\rho_\gamma},
\qquad
\tau_{\rm drag}(z_d)=1
$$

이다. raw SHA-256, solver/version/backend, cosmology, \(Y_p\), 단위,
column/delimiter/order, \(z=0\), 단조성과 crossing 해상도를 강제한다. 합성
history에서 \(z_d=1059.25,\ r_d=147.649757605\,{\rm Mpc}\)를 회수했지만,
실제 CLASS/CAMB/HyRec history가 로컬에 없으므로 이 수치를 precision 결과로
세지 않는다. 닫힌 등급은 외부 solver output의 `Tooling/Bridge` adapter다.

양자 쪽은 독립장 branch A \(\phi\ne R\)를 선택했다.

- Hessian/Jacobi scalar는
  \(\Phi_{\rm eff}=\langle\eta,\mathcal J\eta\rangle/\langle\eta,\eta\rangle\)
  로 읽고 독립장과 동일시하지 않는다.
- \(29.64757\,{\rm MeV}\)는
  \(f=7.1687505314\times10^{21}\,{\rm Hz}\),
  \(\omega=4.5042588010\times10^{22}\,{\rm s^{-1}}\)로 변환되지만 이것은
  pole의 존재 증거가 아니다.
- \(H_{\rm int}/\hbar=gA\otimes O\)의 reduced frequency convention에서는
  \(\gamma=g^2J_{\rm red}\)이고, SI Hamiltonian과 무차원 \(O\)의 raw
  correlator에서는 \(\gamma=(g_J/\hbar)^2G\)다. 코드는 두 변환을 분리하며
  action·bath·결합이나 그 사이 단위 mapping을 유도하지 않는다.
- \(G(\omega)=\int dt\,e^{+i\omega t}\langle O(t)O(0)\rangle\) convention,
  \(G(-\omega)=e^{-\hbar\omega/k_BT}G(+\omega)\) 단일 Hermitian scalar KMS
  gate와 strict-default PSD Kossakowski jump decomposition을 reference
  tooling으로 구현했다. 명시적 PSD 투영·mode cutoff는 residual을 보고한다.
- 2준위 Gibbs 고정점과 cross-term dissipator 동등성은 통과했지만, CE
  action에서 실제 \(G_\phi,J_\phi\)를 유도하는 과정, CPTP instrument와
  no-signalling은 `Open`이다.

현재 focused 회귀는

```text
cosmology forward + drag adapter: 30 passed
quantum baseline + KMS/Kossakowski: 50 passed
future-holdout manifest validator: 21 passed
combined focused gate: 101 passed
full repository: 670 passed, 13 skipped
```

다. 이 증가는 CE의 참일 확률이 아니라 type·단위·수치·generic physicality
게이트의 구현 범위 증가다.

같은 checkout의 상수 장부는

```text
Scored bridge/phenom rows: 13
PASS: 12
CAUTION: 1
Scored pass rate: 92.3%
OPEN TEST: 1
```

이다. 이 92.3%는 상수별 snapshot score이며 DESI full-cov 적합도나 양자
실험 게이트와 다른 질문이다.

### 13.4 미래 holdout preregistration

`experiments/preregistration/`에 우주론·양자론 v1 manifest와 validator를
추가했다. 두 manifest는 canonical JSON self-hash와 계산 코드 artifact
hash를 별도로 고정한다.

| domain | manifest SHA-256 | 현재 상태 |
|---|---|---|
| cosmology | `0f79d9fb27abc7326e3bd136768f0a2b560f720b2db42c1079d9d82c3efe7692` | `VALID / unassigned / NOT_READY` |
| quantum | `4bd3d9777c47465dd419012bbf2622fb0d5c91a312003010dede50cb1c4e853a` | `VALID / unassigned / NOT_READY` |

DESI DR2는 freeze 전에 잔차와 scale 진단에 사용했으므로
`exploratory_calibration`, `holdout_eligible=false`로 고정했다. 과거 ARC
94.6%도 raw artifact가 없는 이미 알려진 기록이므로 future holdout으로
재라벨하지 않는다. 우주론은 full covariance와 holdout fit 0개를 강제하며
\(p<0.0027\)인 고정 후보를 사후 scale 보정 없이 기각한다. 양자론은 CP/Choi,
Born, no-signalling, \(J(\omega)\), GKSL/Kossakowski PSD, KMS와 predictive
density gate를 모두 동결했다.

기본 validator의 성공은 동결 파일의 구조·hash 무결성만 뜻한다.
`--require-assigned-holdout`은 두 manifest 모두 exit code 1로 실패한다.
실제 평가에는 데이터를 보기 전 새 ID와 revision \(\ge2\), v1 supersession,
release metadata와 `holdout_data` artifact SHA-256이 필요하다. 양자 쪽은
그 전에 아직 열려 있는 \(g,A,\mathcal O_\phi\), bath, \(J(\omega)\), instrument
및 단위 mapping도 새 revision에서 고정해야 한다.
