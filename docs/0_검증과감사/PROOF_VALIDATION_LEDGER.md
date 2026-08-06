# CE 증명 보강 및 검증 Ledger

이 문서는 CE 문서군을 처음부터 검증할 때의 진행 순서를 고정한다. 최신
실제 수선은
[전체 정합성 완성 루프](FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md)를
따른다. 상태 변경 자체를 수정으로 세지 않고, 각 단계에 정확한 대체식,
코드/반례 조건, 입력 provenance와 관측 gate가 모두 있을 때만 완료로
기록한다.

> 2026-08-06 수치 계약: 이 ledger의 과거 실행 블록은 당시 fixture를
> 보존한 감사 로그다. \(29.64757\,\mathrm{MeV}\),
> \(43.7677\,\mathrm{GeV}\), \(\mathrm{BR}_{\rm inv}<0.11\) 등이 로그에
> 남아 있어도 현행 acceptance 값이 아니다. 현재 중앙값과 gate는
> `CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`의
> \(29.6991596\,\mathrm{MeV}\), \(43.8056765\,\mathrm{GeV}\),
> \(\mathrm{BR}_{\rm inv}<0.107\)을 따른다.
>
> **상태-history 격리:** `6 이후의 “판정 변경”, `PASS`, scorecard,
> \(A_s\) readout 및 CKM projector 서술은 당시 후보를 탐색한
> **역사적(historical) 실행 기록**이며 현행(current) acceptance가 아니다.
> 현행 계약에서는 \(A_s=2.10\times10^{-9}\)가 외부 정규화 입력이고
> \(n_s,r\)만 그 입력과 \(N_*\)에 조건부인 출력이다. CKM/PMNS는 하나의
> unitary texture를 공동으로 유도·holdout 검증하지 못했으므로
> `Bridge/Open`이다. 이 문서의 과거 `6 이후 상태표로
> `PROOF_STATUS_MATRIX.md`의 현행 판정을 덮어쓰지 않는다.

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
| $D_N$ | 3.177912999513294 |
| $x$ | 0.04863825851598632 |
| multiplier $D_Nx$ | 0.1545681540116411 |
| Lambert-$W$/bracketed/inverse | numeric gate tolerance 안에서 일치 |

실행:

```powershell
python reality_stone\python\reality_stone\clarus\bootstrap_solver.py
uv run --extra dev python -m pytest tests\test_bootstrap_solver.py -q
```

### 1.3 바리온 대응

**완성 모형:** 단순 생존확률이 아니라 renormalized stress의 에너지 가중
측도에서 $x=\langle E_b\rangle/\langle E_{\rm tot}\rangle$를 정의한다.
곱적 사상의 일반해는 $I(P)=P^c$이고 $c=1$에는 접선 정규화가 추가로
필요하다. B2 projector와 평탄 관측 초곡면 아래 $\Omega_b=x$다.

**현재 판정:** B2 작용·측도 모형 안에서 exact; 자연의 projector 선택은
공동 cosmology likelihood로 검증한다.

**반증 조건:** 입력과 겹치지 않는 공동 likelihood에서
$\Omega_b=0.0486382585$를 포함한 동일 parameter vector가 기각되면 해당
B2+dark benchmark를 교체한다. 현재 고정 cosmology package는 이미 full-cov
gate에서 reject됨을 별도 기록한다.

## 2. 차원 선택

**명제:** 장세기 $F$는 2-form이고, Hodge 별표가 $*F$를 다시 벡터/1-form으로 보내려면 $d-2=1$, 즉 $d=3$이어야 한다.

**현재 판정:** 공간 차원 $d=3$ 선택은 `Selection`.

**주의:** $d=3$에서 색 기본표현 차원 $N_c=3$을 상속한다는 단계는 외부 회전군과 내부 색 게이지군의 동역학적 동일시가 필요하므로 `Bridge`로 유지한다.

## 3. 전자약-강력 연결

**명제 후보:**

$$
s_A^2:=4\alpha_s^{4/3}.
$$

**현재 판정:** registered matching ansatz. \(s_A^2\)를 물리적 weak-angle
scheme에 보내는 map 전에는 정밀 score를 주지 않는다.

**현행 검증 코드:** `docs/0_검증과감사/verify_numeric_consistency.py`.
`examples/physics/alpha_s_closure_gate.py`는 낮은 Track-B 근을 숨기고
zero-parameter verdict를 출력하는 legacy/noncanonical diagnostic이라
acceptance에서 제외한다.

**남은 간극:** Yang-Mills 작용의 차원 비분리와 결합상수 곱적 분배의 동일시. 비섭동 상계는 간극을 줄이지만, 엄밀한 `Exact` 승격은 아니다.

## 4. 이번 패스에서 고친 검증 도구

| 파일 | 변경 |
|---|---|
| `reality_stone/python/reality_stone/clarus/bootstrap_solver.py` | Newton 야코비안 부호 수정, SciPy 없는 bracketed fallback 추가, 콘솔 출력 ASCII 안정화 |
| `tests/run_validation.py` | standalone bootstrap 야코비안 부호 수정 |
| `tests/test_bootstrap_solver.py` | Newton/bracketed 일치, residual, 야코비안 수치미분 회귀 테스트 추가 |
| `docs/2_경로적분과_응용/10_공리_정당화.md` | 당시 삭제된 `check_unification.py` 참조를 legacy `alpha_s_closure_gate.py`로 갱신했던 기록; 현행 acceptance는 numeric manifest verifier |
| `docs/경로적분.md` | 현행 Track A/두-root Track B verifier로 재교정 |

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

위 블록은 당시 실행 기록이며 마지막 electroweak verdict와 `Overall:
VALIDATED`는 현행 판정으로 **무효**다. 외부 \(\alpha_{em}(M_Z)\) 입력,
Track-B branch selection과 낮은 양의 근을 누락했고, legacy scorecard가 서로
다른 snapshot을 섞었기 때문이다. 최신 acceptance는 numeric manifest와
전체 consistency gate의 출력만 사용한다.

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
| $\Omega_\Lambda$, $\Omega_{DM}$, $w_0$, $\lvert V_{us}\rvert$, $n_s$ | `Phenomenology` |
| $\lvert V_{cb}\rvert$ | `Phenomenology` |
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
- Clarus inverse-correlation / particle-language bridge
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
| $\lvert V_{us}\rvert$ | `Open` | `Phenomenology` | tree 식은 실패하지만 $1/(1+\alpha_s/2\pi)$ 보정이 strict 기준 1 sigma 안에 들어온다. |
| $n_s$ | `Open` | `Phenomenology` | $1-2/(dD_{\rm eff}12/2)$가 수치적으로 닫힌다. 단, 전이수 12와 inflationary readout은 가정이다. |
| $\lvert V_{cb}\rvert$ | `Open` | `Phenomenology` | LO 실패는 유지하고, $\delta/(2\pi)$ electroweak projector bridge가 strict 기준 +0.58 sigma로 통과한다. |
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
| total fixed-point response $\lvert dx/dD\rvert$ | 7.83532 | +1685.21 | reject |
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

## 12. Clarus inverse-correlation / particle-language gate

`examples/physics/clarus_boson_search_gate.py`는 구 \(\delta\)와 질량창을
hard-code한 **legacy/noncanonical runner**다. 아래 구 창은 당시 탐색 규칙의
재현 기록일 뿐이며 현행 acceptance에 사용하지 않는다. 또한 이 질량은 실제
CE connected 2점함수에서 추출한 pole이 아니라 \(m_p\delta_N^2\)로 등록한
inverse-correlation scale ansatz다. 따라서 향후 질량창 hit도 pole residue,
spectral positivity 또는 CE field identity를 저절로 증명하지 않는다.

현행 canonical manifest의 중앙값은 다음과 같다.

| 항목 | 현행 값 | 지위 |
|---|---:|---|
| $s_A^2$ | 0.2315097758 | Track A registered neutral output |
| $\delta_N=s_A^2(1-s_A^2)$ | 0.1779129995 | neutral-projector index |
| coupling matching 후보 $\lambda_{HP}=\delta_N^2$ | 0.0316530354 | 별도 portal matching ansatz |
| $m_\xi=m_p\delta_N^2$ | 29.6991596 MeV | inverse-correlation central scale |
| reduced Compton length $\hbar c/m_\xi$ | 6.6441941 fm | 중앙값의 단위 변환 |
| $\mu_\Phi=0$ portal mass $v_{\rm EW}\sqrt{\lambda_{HP}}$ | 43.8056765 GeV | light scale과 다른 rejected benchmark |
| 현행 질량 불확실성·3-sigma 창 | 미정 | 입력 covariance와 pole/readout likelihood 필요 |

legacy runner snapshot은
\[
(\delta,\lambda_{HP},m_\xi)
=(0.17775842,0.03159806,29.64757\,\mathrm{MeV})
\]
였고 reduced Compton length는 \(6.65575\,\mathrm{fm}\),
구 3-sigma 창 \(28.388\)--\(30.908\,\mathrm{MeV}\)였다. 이 창을 현
중앙값 주위로 단순 평행이동하거나 현행 3-sigma 구간으로 재명명하지 않는다.

현재는 uncertainty contract가 없으므로 runner의
`experimental_bridge_signal`, `experimental_bridge_candidate`,
`bridge_rejected`, `bridge_constrained` 출력을 최신 판정으로 인용하지 않는다.
먼저 \(m_\xi\) 입력 covariance, coupling benchmark, detector acceptance와
look-elsewhere 처리를 사전등록해야 한다. 그 뒤에도 pole-compatible 신호는
particle-language bridge 후보일 뿐 이론 pole certificate가 아니다.

X17류 17 MeV 신호는 canonical 중앙 scale과도 분리한다. 현행
\(m_\xi=29.6991596\,\mathrm{MeV}\) inverse-correlation 후보와
\(m_\Phi=43.8056765\,\mathrm{GeV}\) canonical 포탈 benchmark는 서로 다른
bridge 가정이며 같은 pole로 동시에 읽지 않는다. 향후 적법한
`bridge_rejected` 판정이 나오더라도 우선 반증되는 것은 국소 스칼라/포탈
readout이지 경로적분 수렴 구조로서의 클라루스장 자체가 아니다.

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

위 실행 블록은 legacy runner의 재현 로그다. 현행 canonical override는
\(\lambda_{\rm HP}=0.0316530354\),
\(m_\Phi=43.8056765\,\mathrm{GeV}\),
\(\Gamma_{\rm inv}=13.790042\,\mathrm{MeV}\),
\(\mathrm{BR}_{\rm inv}=0.77082222\)이며, PDG 2026이 열거한 ATLAS direct
상한 \(0.107\)에서도 판정은 `REJECT`다.

### 13.3 Q0.4–Q0.5 singlet pole·vertex 통제 인증서

선택적 \(Z_2\) singlet portal action의 singlet block을 실제 미분해

\[
K_F(p)=p^2-(m_0^2+\lambda_{HP}v^2)+i0,
\qquad
\operatorname{Res}(G_F/i)=+1
\]

과 \(h\Phi^2,h^2\Phi^2,\chi^2\Phi^2\) vertex를 조건부로 닫았다. 동시에

\[
\Delta S=C\eta^3/3!+D\eta^4/4!
\]

가 배경 gradient/Hessian을 보존하면서 higher vertex를 임의 변경하므로 A1
Hessian만으로 production vertex를 복원할 수 없음을 반례로 고정했다.

29.64757 MeV를 같은 portal field의 tree pole로 역산한 결과는 다음과 같다.

```text
portal-dominated pole                  43.767675473 GeV
required bare mass squared            -1915.608537374 GeV^2
target squared / portal mass shift    4.588505357e-7
portal / target mass ratio            1476.265187
minimum lambda_phi for EW tree vacuum 0.00772
selected EW/Z2 tree vacuum global     True (lambda_phi=0.1 control)
BR(h -> phi phi)                      0.825312044
supplied BR limit                     0.11
maximum |lambda_HP| from that limit   0.005110743
physical Clarus pole derived          False
full LSZ passed                       False
physical SM production rate derived  False
```

따라서 tree pole은 bare mass를 목표에서 역산하면 `constructible`이지만 예측이
아니다. \(m_0^2\ge0\)이면 정확한 하한
\(m_{\rm pole}\ge v\sqrt{\lambda_{HP}}\) 때문에 light same-field pole은
반증된다. 세부 식과 countermodel은
`CE_TWO_POINT_AND_VERTEX_LOOP.md`에 기록했다.

```powershell
uv --cache-dir .uv-cache run python examples/physics/ce_two_point_vertex_gate.py
uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_ce_two_point_vertex_certificate.py -q
```

### 13.4 action provenance · renormalized pole · scalar one-loop 통제

저장소의 action 후보를 다시 감사한 결과 physical renormalized
\(\Gamma_{CE,R}^{(2)}\)의 입력으로 채택 가능한 complete CE action은 없었다.
Q0 선택적 portal은 가장 엄격하지만 고전·bare·tree Abelian control이고
counterterm, renormalization, CE/중력과 full SM sector를 명시적으로 제외한다.
형식적 \(\sigma\) action은 \((-+++ )\) 아래 kinetic/EOM 부호와 무차원장
차원이 맞지 않고 \(\sigma f(R)\) metric variation도 빠져 있어 입력에서
격리했다.

Q0 tree action definition은 다음 digest에 묶었다.

```text
c6e1f448c388900d3a70f997d2c133580f1a87a0682e6fe2309fe58bf21ed233
```

별도 fail-closed 인증서는 action/counterterm/background hash와 gauge/scale
kernel replica에서 root, derivative/residue, imaginary part, first-cut gap과
dispersion을 재계산한다. 현재 CE 결과는 다음과 같다.

```text
maximum stage                         REGISTERED_SCALE
first blocker                         renormalized action manifest is absent
physical LSZ particle derived         False
CE field identity derived             False
```

선택적 portal의 \(h,\phi\) two-real-scalar finite one-loop control은

```text
finite scalar sum                     4.672293184 GeV^2
finite sum / target mass squared      5315.594954
Sigma'(m_phi^2)                       4.886763794e-5
linearized residue                    0.999951135
first h-phi cut                       125.279647570 GeV
finite sum at mu/2,mu,2mu             -1.804844, 4.672293, 11.149430 GeV^2
renormalized pole predicted           False
```

이다. 결합의 loop parameter는 작지만 additive light-mass retuning이 필요한
진단이다. raw finite 합의 scale 의존성은 counterterm/running input이 빠졌다는
표지이며 물리 uncertainty나 pole shift가 아니다. 전체 식과 재현 명령은
`CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md`에 기록했다.

```powershell
uv --cache-dir .uv-cache run python examples/physics/ce_renormalized_pole_gate.py
uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_renormalized_pole_certificate.py `
  tests/test_portal_one_loop_control.py -q
```

### 13.5 CE Euclidean connected correlator · spectral 비유일성 통제

저장소 전체에서 CE action/operator identity에 묶인 원시 paired
\(O_n(t),O_n(0)\) ensemble, chain/configuration ID, Euclidean lattice
spacing·volume·boundary condition과 covariance artifact를 감사했다. 실제 CE
자료는 없었다. `quantum.py`의 bath Fourier transform, TDCOSMO covariance,
RAGTruth ensemble, 생물학 형광·spike 원자료는 각각 convention과 대상이 달라
CE Källén–Lehmann 자료로 재사용하지 않았다.

새 fail-closed scaffold는 원시 paired 표본에서

\[
\widehat C_t=\frac1{N-1}\sum_n
(O_n(t)-\bar O_t)(O_n(0)-\bar O_0)
\]

를 직접 계산하고 delete-one jackknife로 full estimator covariance를 만든다.
진공·비주기적 Hermitian bosonic Laplace control에서 equal-grid 유한차분,
log-convexity와 truncated Hankel 조건을 검사하고, covariance-aware exponential
window fit으로 screening control을 분리한다. finite-temperature wraparound,
contact term, fermionic/tensor kernel은 이 scope 밖이다.

유한 시간 \(N_t\)개와 더 많은 energy bin으로 만든
\(K_{ij}=e^{-E_jt_i}\)에 총 weight 보존 행을 붙인 augmented SVD에서
null vector \(v\ne0\)를 택해

\[
\rho_\pm=\rho_0\pm\epsilon v\ge0,
\qquad K\rho_+=K\rho_-,
\qquad \sum_j(\rho_+)_j=\sum_j(\rho_-)_j
\]

인 서로 다른 두 spectrum을 실제로 구성한다. 따라서 correlator 표본과 알려진
정규화를 함께 고정해도 유한 표본에서 pointwise
spectrum 유일성은 알고리즘 종류와 무관하게 성립하지 않는다. positivity
통과는 오직 `FINITE TWO-POINT POSITIVITY NOT REJECTED`, 안정한 exponential
window는 `EUCLIDEAN_SCREENING_CONTROL`까지 허용한다.

현재 CE 실행 판정은 다음과 같다.

```text
maximum stage                         REGISTERED_SCALE
first blocker                         raw paired O(t), O(0) ensemble is absent
actual CE connected correlator        absent
unique spectral density               False
Minkowski pole / LSZ                   False / False
CE field identity                     False
```

합성 single-exponential control은 covariance-aware screening mass
\(29.647570000\,\mathrm{MeV}\), augmented nullity 8,
correlator-pair 상대잔차 \(1.77\times10^{-16}\), 총 weight 잔차 0을
재현했다. 이 경우에도 unique spectrum, Minkowski pole, LSZ와 CE identity는
모두 False다.

2026-08-04 검증은 전용 21개와 관련 통합 92개 테스트를 통과했다. 작업
트리에서 이미 삭제된 fixture를 직접 요구하는 테스트 파일 5개만 제외한 전체
회귀는 1350 passed, 13 skipped, 0 failed였다. Ruff check/format과 full
certificate JSON 직렬화도 통과했다.

```powershell
uv --cache-dir .uv-cache run python examples/physics/ce_euclidean_correlator_gate.py
uv --cache-dir .uv-cache run --extra dev python -m pytest tests/test_euclidean_correlator_certificate.py -q
```

전체 수식, 범위와 실제 데이터 계약은
`CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md`에 기록했다.

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

## 16. 2026-07-30 우주론·양자론 루프 감사

### 16.1 DESI BAO 고정모델

`examples/physics/ce_residual_forward_model.py`에 p-value, 판정, covariance
기여도와 parameter provenance를 추가했다.

| 모드 | \(\chi^2\) | dof | \(p\) | 판정 |
|---|---:|---:|---:|---|
| canonical CE density + 외부 \(H_0=67.4,r_d=147.09\) 고정 | 40.201450858 | 13 | \(1.28283168\times10^{-4}\) | `REJECT` |
| 동일 shape + \(H_0r_d\) scale 1개 진단 fit | 12.206911338 | 12 | 0.429208739 | `PASS/Diagnostic` |

복잡도 패널티는 고정모델
\({\rm AIC}={\rm BIC}=40.2015\), scale fit
\({\rm AIC}=14.2069,\ {\rm BIC}=14.7719\)이다.

analytic scale은

$$
q_*
=
\frac{y^\top C^{-1}d}{y^\top C^{-1}y}
=
0.985555780095
$$

이다. 이는 \(r_d=149.245738\,{\rm Mpc}\) 또는
\(H_0=68.387809\,{\rm km\,s^{-1}Mpc^{-1}}\)와 등가다. 같은 DESI 데이터에
맞춘 값이므로 CE 예측으로 세지 않는다. 현재 판정은 “고정 패키지 실패의 주된
원인이 외부 \(H_0r_d\) scale”이라는 원인 분해다.

### 16.2 표준 QM reference gate

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

### 16.3 우선순위 보강 실행

외부 \(r_d\)를 대체할 첫 계산 경로로

$$
z_d^{\rm EH}=1019.907163886,
\qquad
r_d^{\rm EH\ hybrid}=151.505227530\ {\rm Mpc}
$$

를 얻었다. 입력은 CE \(\Omega_b,\Omega_m\), 외부 \(H_0,T_{\rm CMB}\),
표준모형 가정 \(N_{\rm eff}=3.044\)와 Eisenstein--Hu fit이다. DESI DR2
13점 full covariance에 넣으면

$$
\chi^2=41.194553577,\qquad
p=8.86018138\times10^{-5},
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
- \(29.6991596174\,{\rm MeV}\)는
  \(f=7.1812248454\times10^{21}\,{\rm Hz}\),
  \(\omega=4.5120966464\times10^{22}\,{\rm s^{-1}}\)로 변환되지만 이것은
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

이 절을 작성했을 당시 historical focused 회귀는

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

### 16.4 미래 holdout preregistration

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

## 17. 2026-08-04 클라루스 공명 물질생성 루프

### 17.1 목표 스케일 정정

1 m 목의 현재 정본은 \(b'(r_0)=-1/3\) full tensor다.

\[
a=4.0535640043\times10^{-18}\ {\rm m},\qquad
\lambda_*=2a=8.1071280086\times10^{-18}\ {\rm m},
\]

\[
E_*=\frac{hc}{\lambda_*}
=152.932330938\ {\rm GeV}.
\]

\(a\)는 ideal Casimir density matching으로 고정되지만 \(\lambda_*=2a\)는
평행판 최저 normal-mode 선택이다. spherical throat의 eigenmode나
single-mode Casimir stress가 유도된 것은 아니므로 152.932 GeV는 형식적
ideal-planar scale로만 기록한다.

과거 \(b'(r_0)=-1\) control의 \(169.247445587\) GeV는 별도 legacy
control로 격리했다. 29.64757 MeV CE pole-mass 후보와 현재 형식 scale의 비는
5158.342857이며 최근접 5158배와도 약 10.16 MeV 어긋난다. 이 비는 고차
vertex의 존재 증명이 아니며, pole mass도 propagating mode의 에너지 상한이
아니다.

### 17.2 정확히 닫힌 부분

고전 pump를

\[
\Phi(x)=\sum_i A_i\cos(K_i\cdot x+\theta_i)
\]

로 공급했을 때 \(\Phi^2\)의 \(2K_i\), \(K_i+K_j\), \(K_i-K_j\), DC 선을
복소 phasor로 합산하는 항등식을 구현했다. 따라서 같은 spectral line에
기여하는 항의 위상 상쇄도 보존한다. 예를 들어

\[
\Phi=A\cos t-\frac A2\cos 3t
\]

에서는 \(2\omega\) 선이 정확히 소거된다.
Fourier-key grouping과 cancellation tolerance가 모두 0일 때만 exact flag를
유지한다. line-shape model이 없으므로 pump linewidth도 보편적인 합 규칙으로
quadratic line에 전파하지 않는다.

현재 구현은 \(p_\perp=0\)인 1+1D collinear sector다. 이 sector에서 동일
daughter pair의 정확한 운동학 gate는

\[
Q^0>0,\qquad Q^2\ge4m_\chi^2
\]

다. 같은 방향 massless pump는 에너지가 커도 \(Q^2=0\)이라 massive pair를
만들지 못한다. 반대 방향 pump의 합은 timelike channel을 열 수 있다.
strict threshold에는 tolerance를 더하지 않으며 tolerance-only channel은
ambiguous로 남는다. null massless channel에는 COM frame도 만들지 않는다.
full 3+1 momentum spectrum은 아직 open이다.

현재 간격에 맞춘 standing wave의 pump 파장은 \(2a\), pump quantum은
약 152.932 GeV다. 29.64757 MeV pole mass를 가정하면 그 제곱장의 timelike
pair line과 \(2E_*\) 사이에 5747.5 eV detuning이 남는다. 따라서 실제
linewidth 없이 정확한 factor-2라 부르지 않는다. 정적 grating 주기는
\(a\)다. formal scale, pump, pair-total, daughter mass를
같은 값으로 취급하지 않는다.

### 17.3 수치적으로 닫힌 toy gate

끝점에서 정확히 꺼지는 smooth finite pulse에 대해 복소 mode equation을
RK4로 풀고 in/out Bogoliubov occupation을 계산했다. 이 solver는
\(\Phi^2\)의 DC shift를 제외한 mean-subtracted generic mass modulation
control이다. 현재 고정 control은

\[
n_N=1.181479166,\quad
n_{2N}=1.181479214,\quad
n_{4N}=1.181479217,
\]

\[
\left|f\dot f^*-f^*\dot f-i\right|
=3.29\times10^{-12}
\]

를 준다. occupation과 Wronskian 외에 one-cycle monodromy의
\(|\operatorname{Tr}M|>2\), \(\det M\simeq1\)을 N, 2N, 4N으로 수렴시킨다.
no-drive, \(g=0\), sudden-switching-only, unresolved tachyon lower bound,
co-propagating false positive 외에 근접 mode beat 삭제, 선택 mode의
off-resonant pulse leakage, leading-band false positive, 서로 다른 \(Q,m,p\)
provenance 결합, 문자열 bool, downstream self-certification을
성공으로 승격하지 않는 반례도 고정했다.

이 결과의 최고 지위는

\[
\boxed{\text{CONDITIONAL ASYMPTOTIC DAUGHTER EXCITATION}}
\]

이다. CE 작용에서 physical pole, positive residue, nonlinear vertex,
pump work와 depletion이 유도되지 않았으므로 실제 Clarus particle
production 증명은 아니다.

### 17.4 음의 응력과의 분리

canonical daughter의 dephased particle stress는

\[
\langle:T_{kk}:\rangle
=\int\frac{d^3p}{(2\pi)^3\omega_p}
(k\cdot p)^2n_p\ge0
\]

다. 따라서 occupation 증가를 throat의 음의 source로 해석하지 않는다.
Casimir 응력에는 단일 real-frequency 반사율이 아니라 imaginary-frequency,
횡운동량, TE/TM 전 응답과 renormalization이 필요하다. active driven
boundary에는 equilibrium Lifshitz shortcut도 허용하지 않는다.

현재 stage ledger는 다음에서 멈춘다.

| gate | 판정 |
|---|---|
| formal ideal-planar scale calibration | PASS |
| phase-aware squared-field spectrum | EXACT CONDITIONAL |
| invariant pair kinematics | EXACT CONDITIONAL |
| same-\(Q,m,p\) smooth finite-pulse daughter excitation | CONDITIONAL PASS |
| physical Clarus pole and action vertex | OPEN |
| finite particle spectrum and energy ledger | NOT REACHED |
| persistent boundary matter phase | NOT REACHED |
| causal full boundary response | NOT REACHED |
| renormalized negative net stress | NOT REACHED |
| stable backreacted throat | NOT REACHED |

새 구현은 casimir_carrier_target.py와 clarus_resonant_matter.py, 재현 예제는
clarus_resonant_matter_gate.py에 둔다. 조건부 toy maximum은
`CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION`이지만 CE 물리 maximum은
`TARGET_SCALE_CALIBRATION_ONLY`다. global certificate까지 묶은 focused
6-file 회귀는 91개가 통과했다. 이 숫자는 조건부 모형의
수치·반례 일관성을 뜻하며 실제 물질이나 wormhole의 실험적 정확도를 뜻하지
않는다.

## 18. 2026-08-04 cutoff-independent global throat certificate

기존 기하와 ADM-matched redshift 보강을 cutoff 판정에서 분리한
`global_throat_exact_certificate()`를 추가했다. 기존 ansatz는 throat,
flare-out, horizon-free, 각 end \(M_{\rm ADM}/r_0=1/3\)과 Bianchi 항등식을
exact로 만족한다. Killing energy를 1로 고정한 양쪽 radial affine ANEC는

\[
\mathcal A_{\rm old}=-2.49755541727731
\]

로 유한·음수지만 coordinate/proper volume-NEC는
\(-(2/3)\ln X\)로 발산하므로 localized finite source가 아니다.

\[
\Phi_{\rm match}
=\frac12\ln\left(1-\frac{2}{3x}\right)+\frac32e^{1-x}
\]

보강은 같은 throat tensor, lapse 제곱 하한 \(1/3\), 각 end ADM \(1/3\)을
보존하면서 stress tail을 지수감쇠시킨다. 수치 적분은

\[
\mathcal A_{\rm match}=-2.29272813381626,
\]

\[
\mathcal V_{\rm coord}/\text{end}=-4.21893534547003,\qquad
\mathcal V_{\rm proper}/\text{end}=-6.09178724755025
\]

로 모두 유한·음수다. 그러나 throat의 \(K/F=7/12>0\)만으로 healthy scalar를
선언할 수 없다. 명시적 유리점

\[
x=\frac{37}{32},\qquad \frac KF=-1.83054671559
\]

이 전역 양성 조건을 바로 반박한다. 따라서 최종 판정은
`EXACT GEOMETRY + FINITE-TAIL CONTROL / HEALTHY GLOBAL SCALAR REFUTED`다.
Bianchi exact도 \(T=G/\kappa\) 역정의의 결과이지 독립 CE matter EOM 증명은
아니다.

아래 수치는 `19` probe pilot 추가 전인 `18` 종료 시점 snapshot이다. 전체
저장소에서는 사용자가 삭제 상태로 둔 artifact/benchmark를 직접 읽는
4개 테스트 파일을 제외하고 `1304 passed, 13 skipped`다. 제외하지 않은 전체
실행은 `1342 passed, 13 skipped, 15 failed, 9 errors`였으며, 실패와 오류는 모두
`test_local_memory_verifier.py`, `test_neural_tree_algorithm_census.py`,
`test_origin_life_branching_verifier.py`, `test_q0_manifest_gate.py`에 한정됐다.
각 원인은 삭제된 artifact/benchmark의 부재이며, 관련 45개 tracked deletion은 이
루프에서 복구하거나 수정하지 않았다.

## 19. Probe-selective dressing / public-response kernel pilot

추가 구현:

- `reality_stone/python/reality_stone/clarus/probe_scaffold_pilot.py`
- `tests/test_probe_scaffold_pilot.py`
- `examples/physics/probe_scaffold_pilot_gate.py`
- `docs/0_검증과감사/PROBE_SELECTIVE_DRESSING_AND_PUBLIC_SCAFFOLD_LOOP.md`

private branch는 단순 on/off 대신

\[
I_i=(\mu_{i,1,1}-\mu_{i,1,0})-(\mu_{i,0,1}-\mu_{i,0,0}),
\qquad S_{AB}=I_A-I_B
\]

의 pump×controller factorial contrast를 쓴다. reference의 비유의성만으로 통과하지
않고 전체 신뢰구간이 equivalence bound 안에 들어가야 한다. phase resultant는
finite-sample bias를 줄인 \(R_{\rm bc}\)로 변환하며, noise level 하나를 held-out으로
남겨 training \(S_{AB}(R_{\rm bc})\)가 그 응답을 예측해야 한다. 여기서 held-out
\(R_{\rm bc}\)는 실제 측정값이므로 response만 조건부 예측하며 noise→\(R(D)\)
동역학을 예측한 것은 아니다. held-out 지정도 외부 hash/timestamp가 없는 선언
metadata다. 현재
effective sample size는 Kish weight 값으로만 계산하므로 시계열 autocorrelation과
Adler dynamics는 아직 `OPEN`이다.
모든 raw mean interval은 최소 4개 관측과 \(df=3\) Student-\(t\) floor를
사용하고, held-out 선형 회귀는 최소 4 training level·interpolation·\(df=2\)
floor를 강제한다. 95% 미만 confidence 요청은 거부한다.
각 level의 raw phase/8개 response stream과 sweep-wide 단일 sign·threshold·confidence
config를 audit에 보존하고 point summary를 재계산한다.

public branch는 최소 세 calibrated probe에서

\[
d_p=c_pK_{\rm post}+\epsilon_p
\]

를 검사한다. 두 training probe의 normalized response로 scalar kernel을 맞추고
held-out으로 지정했다고 선언한 probe를 예측한다. residual pump의 최대 response,
apparatus-memory
상한, pre-pump null, dwell/time ordering과 nuisance monitor를 동시에 veto로 둔다.
raw response 크기만 같은 반례와 잘못된 calibration 반례는 실패한다. training
probe에는 `UNMEASURED_WORST_CASE_CORRELATION`을 적용해 pooled 표준오차 floor와
모든 probe 차이의 additive worst-case 표준오차를 사용했다. probe covariance
자체는 아직 측정하지 않았고
fixed-before-pump calibration·blinding·별도 readout chain도 선언 metadata다.
probe raw pre/post/sham과 residual/nuisance raw monitor stream도 보존·재검산한다.
nuisance에는 별도 nuisance→kernel gain 상한을 적용해 kernel 설명량에서 차감한다.

에너지 gate는 pump/controller/probe/transfer/reservoir-release 입력과
candidate/radiation/thermal-mechanical/reservoir-storage/recovered-work 출력을 고정한
10열 signed ledger다. 전체 covariance \(C\)로

\[
\epsilon_E=s^Te,\qquad \sigma_E^2=s^TCs
\]

를 계산하고 residual과 uncertainty의 candidate-energy 대비 상한을 모두 요구한다.
따라서 giant covariance 또는 giant absolute tolerance로 보존을 가장하는 반례는
실패한다. raw 10채널 trial tuple과 10×10 covariance를 audit에 보존하고
mean/minimum/trial scatter 및 declared sigma를 report validation에서 재계산하므로
sampling sigma·음수 channel minimum·covariance sigma summary만 축소하는 변조도
실패한다. sampling/calibration error의 미측정 상관은 triangle SE로 처리하고,
모든 declaration/pass flag는 문자열 truthiness가 아닌 strict bool만 허용한다.

합성 control의 결과는

```text
private  CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING
public   CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE
public scaffold candidate  False
physical scaffold          False
new matter                 False
```

다. 두 branch는 독립이며 public kernel 후보도 blinded sample transfer와
frequency-dependent causal response가 없으므로 scaffold로 승격하지 않는다. 새
suite는 `21 passed`, 기존 관련 회귀까지 묶은 7-file suite는 `112 passed`다.
physical/new-matter/boundary/stress/wormhole claim-lock
변조도 반례로 고정했다.

`§19`까지 포함한 최신 전체 회귀는 사용자가 삭제 상태로 둔 리소스를 읽는 위
4개 파일을 제외하면 `1337 passed, 13 skipped`다. 제외하지 않은 원본 전체 실행은
`1375 passed, 13 skipped, 15 failed, 9 errors`이며, 실패·오류 파일과 원인은
위 `§18` snapshot과 동일하다. 관련 45개 tracked deletion은 복구하거나 수정하지
않았다.

## 20. 2026-08-05 외부 원자료 field-to-matter 재분석

추가 구현:

- `benchmarks/external_field_to_matter_v1.json`
- `reality_stone/python/reality_stone/clarus/external_field_to_matter.py`
- `tests/test_external_field_to_matter.py`
- `examples/physics/external_field_to_matter_reanalysis.py`
- `docs/0_검증과감사/EXTERNAL_FIELD_TO_MATTER_REANALYSIS.md`

이 루프는 합성 detector를 만들지 않고 세 독립 공개자료를 감사했다. Vezzoli et
al.의 figure source에서

\[
\beta_2\Delta\omega^2+\frac{\beta_4}{12}\Delta\omega^4=mK
\]

를 다시 풀어 \(954.3127/1173.0601\) nm를 얻었고, 공개 CAR 최대 cell
\(953.9375/1172.888889\) nm의 inverse-wavelength energy residual은
\(2.8233\times10^{-4}\)였다. \(g^{(2)}(0)=0.38095\pm0.06\), CAR=0
control \(g^{(2)}(0)=1.00\pm0.04\)도 source table에서 재계산했다. 다만 자료는
처리된 figure source이며 event-level TDC count가 아니고, lab-frame 식은
quasi-phase-matched SFWM과 동일하므로 literal moving-mirror DCE나 Clarus
고유 증거로 승격하지 않는다.

CMS HEPData의 7개 \(p_{T,ee}\) bin을 직접 적분해

\[
\sigma_{\rm fid}=263.3930128\ \mu\mathrm b
\]

를 얻어 논문값 \(263.5\pm1.8_{\rm stat}\pm17.8_{\rm syst}\ \mu\mathrm b\)를
재현했다. 통계오차 적분은 \(1.7238\ \mu\mathrm b\)이고, systematic 공분산
미공개를 반영한 uncorrelated/fully-correlated 두 극한은
\(16.4958\)--\(19.3189\ \mu\mathrm b\)라서 게재값을 포함한다. 이는 quasi-real
photon fusion에 의한 보통 \(e^+e^-\) 생성을 지지하지만 event four-vector가
공개된 것은 아니며 자유 on-shell photon 두 개의 고립 충돌로 과장하지 않는다.

1T-TaS2 `10.nxs`는 \(256\times344\times45\), 두 펌프 합산
\(0.60\ \mathrm{mJ/cm^2}\), 보고 임계 \(0.50\ \mathrm{mJ/cm^2}\), 25 fs 간격을
직접 확인했다. 보고 \(2.2\pm0.1\) THz는 \(454.5\pm20.7\) fs와 주기당 18.2
sample이다. 그러나 본문/원자료의 고정 delay가 35/25 ps, 대응표/NeXus의 온도가
160/20 K로 충돌하고 fit code·bound와 acquisition-level lifetime curve가 없어
exact frequency/lifetime refit은 `False`로 잠갔다. 이는 기존 Ta/S 원자·전자의
준안정 전자·구조 상 재배열이지 새 입자·원자 생성이 아니다.

가장 중요한 bridge 반례는 에너지와 provenance다. 관측 optical pair는
\(2.3567938\) eV이고 \(e^+e^-\) 정지질량 문턱은 \(1.0219979\) MeV이므로

\[
\frac{2m_ec^2}{E_{s+i}}=4.33639\times10^5.
\]

두 결과는 다른 장치·run·입력에서 얻었고 중간 전달 기록이 없다. 따라서
`pump_to_nonclassical_photon_pairs_supported=True`와
`electromagnetic_field_to_ordinary_massive_pairs_supported=True`는 유지하지만
`same_apparatus_end_to_end_chain_proven`, `clarus_field_observed`,
`clarus_coupling_measured`, `new_material_species_created`,
`pump_off_persistent_structure_proven`,
`free_energy_or_vacuum_energy_extraction_proven`은 모두 `False`다.

snapshot은 SHA-256
`8c78bccb0e8d197bcea80c229e068baa72c1c7a2b920c6b8ddd7f473efc0997f`로
고정했다. focused suite 결과는 최종 검증 명령 결과를 따른다.

## 21. 2026-08-05 핵융합 공명 루프 엔지니어링

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_resonance_loop.py`
- `examples/physics/fusion_resonance_loop_gate.py`
- `tests/test_fusion_resonance_loop.py`
- `docs/0_검증과감사/FUSION_RESONANCE_LOOP_ENGINEERING.md`

핵심 반례는 폭의 크기가 아니라 전달 사차원운동량이다. 정적 핵간 교환은
\(q^0=0\), 따라서 \(q^2=-|\mathbf q|^2<0\)이고 양의 질량 scalar pole
\(q^2=m_\Phi^2>0\)에 도달하지 못한다. 그러므로 진공 line의
\(Q=m_\Phi/\Gamma_\Phi\)를 정적 Yukawa potential에 곱하는 bridge는 `REJECT`다.
별도 시간의존 background는 source amplitude, 공간 mode, pump work와
backreaction을 푼 뒤 D--T scattering amplitude에 연결해야 한다.

레거시 산술을 독립 재계산한 결과는 다음과 같다.

```text
vacuum Q                       3.092181401e15
angular frequency rad/s        4.504324130e22
cyclic frequency Hz            7.168854507e21
cyclic linewidth Hz            2.318380967e6
collision sigma ansatz m^2     6.099961571e-34
collision width ansatz MeV     2.384950180e-15
plasma Q under ansatz           2.476237489e15
scalar one-loop delta a_e       2.132519487e-19
```

반사실 \(Q\alpha_\Phi\) Yukawa WKB는 \(Q=10^9\)에서
\(\gamma_0=2.813647838\), \(\gamma_Q=0.958673329\),
\(\Sigma=40.851723790\)을 준다. 핵반경에서 attraction과 Coulomb 항이 같아지는
값은 \(6.0392\times10^7\)이지만 전체 바깥 hump가 20 keV 이하가 되는 analytic
임계는 \(6.2974\times10^{10}\), 형식적 대역폭은
\(1.5880\times10^{-11}\)이다.
이 수치 통과는 `COUNTERFACTUAL_Q_TIMES_YUKAWA_WKB_CONTROL_ONLY`이며 physical
barrier reduction으로 승격하지 않는다.

정본 Z2 분기에서는 단일 scalar--핵자 결합이 0이라 기존 힘이 `CLOSED_OFF`다.
thermal D--T reactivity, Lawson condition, NIF capsule gain과 점화에너지는 모두
`NOT_REACHED/False`로 잠갔다. 다음 허용 루프는 Z2 쌍 vertex, 명시적 Z2 파괴의
비공명 spacelike amplitude, 또는 source-normalized 시간주기 background 가운데
하나에서 표준모형 대비 부호·크기가 고정된 D--T amplitude 잔차를 먼저 유도하는
것이다.

## 22. 2026-08-05 핵융합 전분기 루프 엔지니어링

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_full_loop.py`
- `examples/physics/fusion_full_loop_gate.py`
- `tests/test_fusion_full_loop.py`
- `docs/0_검증과감사/FUSION_FULL_LOOP_ENGINEERING.md`

1차 감사가 남긴 세 물리 분기와 두 후속 공학 분기를 모두 fail-closed gate로
연결했다. 정본 Z2 분기는 h-Phi-Phi tree vertex를 재현하지만 단일 source가 없고,
두-scalar cut 문턱은 59.29514 MeV다. 공급 portal benchmark는 Higgs invisible
BR 0.825312로 공급 상한 0.11을 실패하며 renormalized pair D--T amplitude는 없다.

명시적 Z2 파괴의 레거시 mixing 0.04344는 공급 상한 0.0038의 11.4316배이고
rate-like 제곱비는 130.681이다. 상한에서 핵반경 정적 힘은 Coulomb의
1.2671e-10이며 timelike Q를 곱하지 않는다. coherent 분기는 1% 핵자질량 변조에
필요한 prescribed free-field energy density 3.2656e38 J/m^3, 수명별 보충 power
density 4.7569e45 W/m^3를 음의 대조군으로 고정했으나 source·pump·backreaction·
Floquet D--T scattering은 모두 미도달이다.

표준 기준선은 10 keV에서 Bosch--Hale <sigma v>=1.136165471e-16 cm^3/s와
0차원 n tau=3.000523249e14 cm^-3 s를 재현했다. 후보 sigma(E)가 없으므로 수정
reactivity/Lawson은 False다. NIF 2.05 MJ 기준을 반사실 WKB 40.8517로 나눈
50.1815 kJ는 계산되지만 `rejected_linear_rescale_energy_kj`로만 저장하며 capsule
radiation hydrodynamics와 점화 예측은 False로 잠갔다. 최대 지지 단계는
`STANDARD_DT_BASELINE_PLUS_SOURCE_ENERGY_NEGATIVE_CONTROLS`다.

## 23. 2026-08-05 핵융합 식-수정 반복 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_equation_iteration_loop.py`
- `examples/physics/fusion_equation_iteration_gate.py`
- `tests/test_fusion_equation_iteration_loop.py`
- `docs/0_검증과감사/FUSION_EQUATION_ITERATION_LOOP.md`

레거시 Q-times-Yukawa 식을 제거하고 두 action-traceable EFT 식을 실제
Bosch--Hale cross section과 Maxwellian kernel까지 전파했다. 명시적 Z2 파괴
확장에서는 g_PhiNN=sin(theta) f_N m_N/v의 단일 exchange를 사용했고, 정본
Z2에서는 Higgs를 적분한 C_N=2 lambda_HP f_N m_N/m_h^2 접촉항의 두-scalar
퍼텐셜 (V=-C_N^2 m K1(2mr)/(32 pi^3 r^2))을 사용했다. 수정 nuclear S-factor는
유도되지 않았으므로 표준 S(E)를 고정하고 외부 장벽 WKB 비만 곱하는 조건부
bridge로 명시했다.

deuteron/triton coherent point-nucleus scalar charge product A_D A_T=6을 포함했다.
이는 유한 핵 form factor를 1로 둔 낙관적 상한이다. 10 keV에서 공급 mixing 상한
0.0038의 열반응률 증가율은 6.1813e-10, invisible-width 상한
lambda_HP=0.005110743의 두-scalar 증가율은 3.8335e-18다.
질량 0, unit mixing은 모든 양의 질량과 |sin(theta)|<=1 Higgs-비례 단일-scalar
분기의 점별 상한이며 증가율 4.0194e-4로 선언한 1% 목표를 실패한다. 두-scalar
질량 0 상한도 2.0218e-17이다. turning point에는 sin-squared 변수변환을 적용했고
공급 mixing 증가율은 coarse/default/fine 격자에서 6.181088/6.181294/6.181505e-10으로
수렴했다.

1%를 수학적으로 달성하는 질량 0 직접 핵자 결합은 g_N=0.00569352이고 Higgs
mixing 환산값은 4.97583이다. 현재 선택 작용에 직접 Phi Nbar N 연산자가 없고
unit mixing 범위도 넘으므로 `MATHEMATICAL_TARGET_ONLY_NEW_DIRECT_OPERATOR_REQUIRED`
로 잠갔다. 계산 사슬의 최대 지지 단계는
`CONDITIONAL_STATIC_POTENTIAL_TO_THERMAL_REACTIVITY_CHAIN`이며 물리적 CE 핵융합
upgrade와 NIF 점화 예측은 False다.

## 24. 2026-08-05 핵융합 잔여분기 최종 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_remaining_branches_loop.py`
- `examples/physics/fusion_remaining_branches_gate.py`
- `tests/test_fusion_remaining_branches_loop.py`
- `docs/0_검증과감사/FUSION_REMAINING_BRANCHES_LOOP.md`

1% 열반응률 목표의 직접 핵자 결합을 등록 질량 29.64757 MeV에서 다시 풀어
g_N=0.0174265를 얻었다. massless 하한 0.00569352보다 크며 Higgs-mixing 등가값은
약 15.23이다. 질량비례 quark completion으로 환산한 scale은 16.153 GeV로 scalar
질량보다 545배 높지만 electroweak vev의 0.0657뿐이다. 핵포화밀도 uniform
mean-field 에너지 진단은 0.2124 MeV/nucleon이다. 저에너지 결합은 perturbative이나
선택 portal 작용, gauge-invariant UV completion, NN scattering·nuclear binding·
rare-decay joint likelihood가 없어 physical gate는 False다.

published dynamically assisted electromagnetic control의 1e15--1e16 V/m를 독립
장부화했다. 에너지밀도는 4.427e18--4.427e20 J/m3이고, D--T 상대유효전하로
1 keV·1e16 V/m quiver는 68.98 fm다. 반면 29.64757 MeV CE 주파수에서 같은
field의 quiver는 7.85e-8 fm이며 3.24 fm에는 4.13e23 V/m가 필요하다. 이는
electromagnetic 비교일 뿐 scalar source가 아니다. CE 1% mass-modulation
free-field density는 published max EM density의 7.38e17배이고 source·pump·
backreaction·Floquet scattering은 모두 미도달이다.

허용 정적 gain 6.181e-10과 Higgs-비례 모델계열 상한 4.019e-4를 Lawson/power
경계까지 전파했다. 후자를 NIF 2.05 MJ에 선형 적용한 823.65 J는 명시적으로
rejected upper bound이며 radiation hydrodynamic prediction이 아니다. 선언된
모든 정적·시간의존 분기를 감사했지만 물리적 1% reactivity gain과 reactor/ICF
upgrade는 False, 최대 지지 단계는 `MODEL_CLASS_NO_GO_PLUS_SOURCE_ENERGY_CONTROLS`다.

## 25. 2026-08-05 직접 핵자 연산자 저에너지 핵산란 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_direct_scattering_loop.py`
- `examples/physics/fusion_direct_scattering_gate.py`
- `tests/test_fusion_direct_scattering_loop.py`
- `docs/0_검증과감사/FUSION_DIRECT_SCATTERING_LOOP.md`

D--T coherent scalar charge product 6을 포함한 등록질량 1% 해 g_N=0.0174265를
저에너지 핵관측량으로 내렸다. 자유 Born Yukawa scattering-length 이동은
-0.00509035 fm로 np triplet/singlet 분석의 보고 오차 0.0015/0.0043 fm 대비
3.39/1.18배다. normalized Hulthén deuteron에서 1차 expectation은 -2.07696 keV,
binding fraction 9.34e-4다.

두 값은 fixed/free controls라서 strong interaction을 함께 재적합한 exclusion이
아니다. distorted-wave Born, NN phase shifts, deuteron/triton/helium few-body fit이
모두 False이므로 physical direct-operator gate도 False다. 최대 지지 단계는
`FREE_BORN_AND_HULTHEN_TENSION_CONTROL_FULL_NUCLEAR_REFIT_REQUIRED`다.

## 26. 2026-08-05 핵융합 Floquet/source 식-수정 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_floquet_source_loop.py`
- `examples/physics/fusion_floquet_source_gate.py`
- `tests/test_fusion_floquet_source_loop.py`
- `docs/0_검증과감사/FUSION_FLOQUET_SOURCE_LOOP.md`

Lindsey et al.의 Floquet--Volkov generalized-Bessel sideband를 Bosch--Hale
D--T 단면적에 적용한 뒤 10 keV Maxwellian과 등방 각도분포를 적분했다.
0.3 keV photon에서 \(10^{16}\,\mathrm{V/m}\)는 반응률을 4.223237599% 높이고,
정확히 1%에 필요한 장은
\(4.861597077\times10^{15}\,\mathrm{V/m}\)이다. sideband 확률합,
무구동 복원, Bosch--Hale fit-domain 질량, 에너지·각도·위상 격자 수렴을 모두
통과했다. 다만 Lindsey et al.의 thermal benchmark는 1 keV이고 CN/FV 대조는
0.1--10 keV 충돌에너지다. 10 keV plasma gain의 95.8%는 10 keV보다 높은
에너지에서 누적되므로 이 점은 수치적으로 수렴한 **FV 식 외삽**이지 공개
validation pass가 아니다. 1 keV benchmark의 1% 임계장 8.680352e14 V/m는
별도 published-support gate를 통과한다.

임계장의 에너지밀도는 \(1.04635\times10^{20}\,\mathrm{J/m^3}\)이다.
10 fs, 반지름
10 nm의 선언된 평면파 pulse는 0.09855 J이며, 총 D/T ion density
\(10^{31}\,\mathrm{m^{-3}}\) microvolume에서 추가 fusion/pulse 비는
\(7.65\times10^{-9}\)이다. 따라서 source 숫자는 닫혔지만 net-energy reactor
upgrade는 아니다.

29.64757 MeV CE scalar는 0.3 keV QED mode와 동일하지 않다. exact-Z2에서 두
on-shell scalar mode의 차주파수 beat는 kinematically 가능하고 그 reduced
wavelength 1479.5 fm는 Gamow turning radius보다 길다. 그러나 허용 portal
계수로 1% kinetic-phase toy를 맞추려면 30.24396% mass modulation,
1.7570e6 MeV mode amplitude와 5.65824e40 J/m³가
필요하다. DC 및 59.3 MeV sum-frequency 성분도 피할 수 없고 scalar-specific
finite-pulse Crank--Nicolson 해가 없다. 최종 단계는
`QED_FV_10KEV_FORMULA_EXTRAPOLATION_CE_SCALAR_SOURCE_NO_GO`이며 CE scalar 1%와
reactor/ICF upgrade는 계속 `False`다.

## 27. 2026-08-05 flavor-aligned 직접 scalar 후보 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_flavor_aligned_loop.py`
- `examples/physics/fusion_flavor_aligned_gate.py`
- `tests/test_fusion_flavor_aligned_loop.py`
- `docs/0_검증과감사/FUSION_FLAVOR_ALIGNED_LOOP.md`

등록질량 1% 해 \(g_N=0.0174265\)의 D/T charge product를 flavor-aligned
\((u,d,s)\) scalar에 정확히 맞췄다.
\(g_p=0.154\,\mathrm{GeV}/f_\phi\),
\(g_n=0.158\,\mathrm{GeV}/f_\phi\) matching은
\(f_\phi=8.9710\,\mathrm{GeV}\),
\(g_p=0.0171664\), \(g_n=0.0176123\)을 준다. 5 TeV VLQ 예시의 실제 새
Yukawa 최대값은 0.299이고 left mixing은 0.0104 이하라 표시 coupling의
perturbativity는 통과한다. 큰 kappa-phi=557은 단일 Lagrangian coupling이
아니라 유효 계수비다. full SMEFT--WET RG matching과 29.65 MeV 질량의
radiative stability는 아직 통과하지 않았다.

Broggini et al.의 equal-coupling neutron 상한을 29.64757 MeV로 외삽하면
0.0175796이고 flavor-matched Pb 유효 coupling은 0.0175242라 중앙 여유는
0.3159%다. 그러나 원 signal 범위는 약 6.05 MeV 이하이고 representative
(q^2/m^2=0.198) correction scale이 여유보다 훨씬 크며 mass-specific
differential likelihood가 없다. Delaunay et al. invisible uds 중앙곡선은
후보보다 6.64배 위지만 논문이 partial-NLO correction 최대 10배를 명시한다.
prompt invisible decay 예시는 존재하나 cosmology/SN/direct-detection joint
gate가 없다. 따라서 최종 분류는
`CLOSEST_CONDITIONAL_CANDIDATE_NOT_CONSTRAINT_CLEARED`이며 물리 branch는
계속 `False`다.

## 28. 2026-08-05 핵융합 직접 연산자 대체분기 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_operator_alternatives_loop.py`
- `examples/physics/fusion_operator_alternatives_gate.py`
- `tests/test_fusion_operator_alternatives_loop.py`
- `docs/0_검증과감사/FUSION_OPERATOR_ALTERNATIVES_LOOP.md`

pure trace/gluon 방향은 D/T 1%에
\(\lvert K_\Theta\rvert v/f=5.48\)이 필요해 digitized
one-parameter rare-decay bound \(1.38\times10^{-3}\)을 3968배 넘는다.
Protophobic endpoint는 Pb neutron bound를 1.336배 넘고, neutron-phobic
endpoint는 kaon combinations를 \(9.47\times10^3\),
\(2.75\times10^4\)배
넘는다. Pb charge cancellation은 D/T charge product를 음수로 바꾸므로
attraction blind spot이 아니다.

massless disformal \(r^{-7}\) potential을
WKB--Bosch--Hale--Maxwellian chain에 전파하면 1%에
\(M=180.705\,\mathrm{MeV}\)가 필요하다.
\(M=200\,\mathrm{MeV}\)에서는 gain 0.004396,
\(810\,\mathrm{MeV}\)에서 \(6.02\times10^{-8}\),
\(1.2\,\mathrm{TeV}\)에서 \(2.60\times10^{-33}\)이다.
등록 scalar mass를 넣으면 더 작아지므로 이는
낙관적 상한이다. 200/810 MeV 분광·stellar 숫자는 massless scalar 참고값이라
29.65 MeV mass-specific 배제로 쓰지 않았다. 적용 가능한 light-mediator collider
bound 약 1.2 TeV만으로도 필요한 180.705 MeV scale은 닫힌다. 모든 대체분기의
physical gate는 False이고 최대 단계는
`ALTERNATIVE_OPERATOR_MODEL_CLASS_NO_GO`다.

## 29. 2026-08-05 flavor-aligned 후보 여유 강건성 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_flavor_margin_robustness_loop.py`
- `examples/physics/fusion_flavor_margin_robustness_gate.py`
- `tests/test_fusion_flavor_margin_robustness_loop.py`
- `docs/0_검증과감사/FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md`

정규화 Gaussian D/T one-body density로 29.64757 MeV Yukawa potential을 접고
10 keV WKB--Bosch--Hale Maxwell 평균의 1% 지점을 다시 풀었다. 필요한 charge
product는 점핵 대비 0.97902--0.98579, coupling은 0.98946--0.99287이다.
Gaussian·3차원 exponential·균일구와 두 반지름 세트의 선형응답 외피까지 넓히면
가장 유리한 coupling 비는 0.987857이다. 이 정도 finite-size 효과는 후보를 크게
이동시키지 않으며 ab-initio one/two-body scalar current와 covariance는 아직 없다.

25 keV neutron--Pb에서 30--150도 운동량 전달은 3.531--13.178 MeV다. 유한
propagator와 Pb form factor의 국소 response는 0.925--1.109로 중앙 통과 임계
1.00633을 가로지른다. 자유 normalization을 둔 각분포 p-wave projection은
0.9096--0.9225로 상한을 완화하지만, 10 eV--10 keV 총단면적의 (k^2) 계수
projection은 1.0498--1.0797로 상한을 강화한다. 원 covariance, strong phase,
distorted-wave 및 finite-density provenance가 없어 어느 쪽도 hard likelihood로
승격하지 않았다. sigma2 proxy는 181×241과 1001×1001 격자를 비교했고 최대 상대
이동 3.82e-5가 선언한 1e-4 수치 허용치 안임을 확인했다.

rare-kaon 중앙곡선의 NLO tightening 임계는 점핵 6.6371, 가장 유리한 D/T proxy
6.7187로 논문이 인정한 최대 10배 이론오차보다 작다. 최신 NA62 2016--2022 결과의
전 질량 BR 개선 1--3배는 coupling bound에 \(1/\sqrt I\)로 전달했다. 추가로 JHEP
Figure 2-a PDF 벡터 축과 후보를 감싸는 선분을 코드에 고정해 29.64757 MeV에서

\[
\mathcal B_{2016\text{--}22}^{90\%}=2.4763\times10^{-11},\qquad
\mathcal B_{2016\text{--}18}^{90\%}=3.2968\times10^{-11}
\]

을 보간했다. 중앙 개선 proxy 1.33137은 coupling bound multiplier 0.866665와 점핵
NLO 임계 5.75212를 준다. old/new BR 각각의 독립 5% readout box까지 전파하면
개선은 1.20457--1.47151, 점핵 NLO 임계는 5.47136--6.04729다. 이는 재현 가능한
그림 판독이지만 tabulated CLs bin, acceptance 또는 full uds weak-ChPT recast가
아니므로 `exact_candidate_mass_observed_limit_entered=False`로 유지했다. 최종
`margin_robustness_gate_pass`와 `physical_ce_fusion_branch_accepted`는 모두
`False`다.

## 30. 2026-08-05 핵융합 spin/operator 전수 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_spin_operator_loop.py`
- `examples/physics/fusion_spin_operator_gate.py`
- `tests/test_fusion_spin_operator_loop.py`
- `docs/0_검증과감사/FUSION_SPIN_OPERATOR_LOOP.md`

deuteron spin 1과 triton spin 1/2에서
\(O=\Sigma_D\cdot\Sigma_T\)의 quartet/doublet 고유값 \(+1,-2\)를 사용했다.
raw 비편극 trace는 0이지만 저에너지 D–T의 \(J^\pi=3/2^+\) quartet projector를
통과한 trace는 \(2/3\)이므로 spin-dependent 장거리 힘의 1차항을 기계적으로
0으로 버리지 않았다.

기존 scalar의 1% 필요 charge product \(1.822097176\times10^{-3}\)와 같은 장거리
세기를 맞추면 pseudoscalar는
\(|g_{PD}g_{PT}|=131.0561\), axial-vector는
\(g_{AD}g_{AT}=2.73315\times10^{-3}\)이다. Pseudoscalar는
\(g_P^2/(4\pi)=10.43\)으로 one-boson perturbation 범위를 벗어난다. Axial은 수치상
섭동적이지만 보편 quark K proxy와 약 \(5.9\times10^4\)의 간격이 있고 비보편
flavor/UV completion은 공급되지 않았다.

Vector에서는 attraction을 유지하는 minimax 해
\((g_p,g_n)=(0.12073,-0.09055)\)와 \(^{208}\)Pb의 \(q=0\) charge를 지우는
\((0.13153,-0.08560)\) 해를 찾았다. 후자는 D–T에는 정확히 attractive이지만
finite-\(q\) Pb form factor, π/K likelihood와 anomaly-free gauge completion이 없다.
Spin-2의 보편 등가값 \(c/\Lambda=0.0227767\,\mathrm{GeV^{-1}}\)은 BaBar proxy보다
visible/invisible 각각 759/114배 크며, nucleon-only 비보편에는 그 proxy를 직접
적용하지 않고 보존된 UV completion 부재로 닫았다. Analytic on-shell derivative
node는 같은 Yukawa pole residue도 제거해 contact 연산자로 줄어든다. 정확한
NCSMC/R-matrix와 29.64757 MeV 질량별 likelihood가 없어 모든 physical gate는
`False`, 최대 단계는 `OPERATOR_LEVEL_MATCHES_ONLY_FAIL_CLOSED`다.

## 31. 2026-08-05 D–T 스핀 편극 표준모형 대조 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_spin_polarization_control_loop.py`
- `examples/physics/fusion_spin_polarization_control_gate.py`
- `tests/test_fusion_spin_polarization_control_loop.py`
- `docs/0_검증과감사/FUSION_SPIN_POLARIZATION_CONTROL_LOOP.md`

새 입자 없이 이상적인 \(J^\pi=3/2^+\), \(s\)-wave quartet만 반응한다고 두면
\(A_J=1+P_DP_T/2\)다. 이 상수배율은 Bosch–Hale Maxwellian 평균과 교환되므로
10 keV에서 \(P_DP_T=0.02\)인 **조건부 projector 대조군**의 비는 1.01이다.
실제 \(C_{zz}(E)\)는 직접 측정되지 않았고 energy-dependent 편극 단면적을 새로
적분하지 않았으므로 물리적 반응률 pass로 세지 않는다. 설계 대조점
\(P_D=0.50,P_T=0.10\)은 주입 시 2.5% 여유를 가지며,
반응 시점에 각 종이 같은 비율로 감소하면 63.25% 이상을 보존해야 1%가 남는다.

현장 열평형 편극은 spin-1/spin-1/2 Brillouin 분포를 직접 역산했다. 10 keV에서
필요한 균일장은 \(3.50136\times10^{10}\) T, 에너지밀도는
\(4.87790\times10^{26}\,\mathrm{J/m^3}\)이고 전자 QED 임계장의 7.93배라 탈락한다.
사전 편극 source 장부는 임의 처리량 대신 fusion power 500 MW와 single-pass burn
1%를 선언해 \(1.77316\times10^{22}\) D–T pair/s와
\(1.48075\times10^{-4}\,\mathrm{kg/s}\)를 요구한다. 공개된
\(10^{18}\) D atoms/s는 달성값이 아니라 이상적 편극 deuterium neutral-beam
개발 목표이고 pair rate가 아니므로, per-species D 처리율과의 \(1.77\times10^4\)배
비교만 기록한다. 기준 기대 융합에너지는 pair당 176 keV다. 선형 1% 증분은
1.76 keV/pair = 5 MW지만, 고정 노출의 hazard
\(\tau=-\ln(1-f)\)를 1.01배 하여 \(f'=1-(1-f)^{1.01}\)로 바꾼 보수적 장부는
\(\Delta f=9.94933\times10^{-5}\), 1.75108 keV/pair = 4.97467 MW다.
전기변환효율 40%의 fixed-exposure wall-plug break-even은 700.433 eV/pair =
1.98987 MW, 10배 공학 여유 기준은 70.0433 eV/pair = 0.198987 MW다. 편극 tritium
source, 극저온·마이크로파·재순환의 실측
pair당 비용, 실제 burn-weighted D–T plasma retention이 없으므로 source/pump gate는
`False`다. 이 branch는 CE가 아닌 표준모형 조건부 대조군이며 최대 단계는
`STANDARD_MODEL_IDEAL_QUARTET_ONE_PERCENT_CONTROL_SOURCE_THROUGHPUT_AND_RETENTION_FAIL_CLOSED`다.

최종 물리 gate는 retention/pump aggregate만 신뢰하지 않는다. 실제 burn-weighted
편극 곱의 수치가 source 곱 이하이면서 0.02 이상인지, 측정한 wall-plug eV/pair가
700.433 eV/pair **미만**인지까지 직접 다시 비교하고 D/T source, plasma retention,
cryo/microwave,
tritium handling, recycle/repolarization과 net-positive leaf를 하나씩 논리곱한다.
tiny gain은 cancellation-safe `expm1/log1p` 식으로 계산하며, Bosch–Hale 및 열평형
fusion-domain 입력은 0.2--100 keV 밖에서 거부한다.
reaction·source·retention·pump의 목표가 모두 같은 1% 이상인지도 최종 함수가
교차검사해, 다른 목표에서 계산한 느슨한 pump ceiling을 섞을 수 없게 했다.
또한 저장된 파생 ceiling을 신뢰하지 않고 temperature·polarization·fusion power·burn
fraction·efficiency에서 Bosch–Hale baseline, pair rate, fixed-exposure increment와
wall-plug ceiling을 canonical 식으로 재계산한다.

## 32. 2026-08-05 D–T scalar-current 핵응답 루프

추가 구현:

- `reality_stone/python/reality_stone/clarus/fusion_scalar_current_loop.py`
- `examples/physics/fusion_scalar_current_gate.py`
- `tests/test_fusion_scalar_current_loop.py`
- `docs/0_검증과감사/FUSION_SCALAR_CURRENT_LOOP.md`

기존 flavor-aligned 후보의 0.154/0.158은 dimensionless fraction이 아니라 각각
GeV 단위 p/n sigma numerator임을 코드 명칭에서 바로잡았다. 현대 isoscalar
입력은 조건부로 p=n=72.3 MeV라고 두고, 고정 scale에서 p/n 평균의 제곱 대신
실제 D/T charge 조합을 계산한다. product 비는 0.21388257, 중심값만 맞춘 scale
진단은 4.148854 GeV다.
이는 covariance와 UV·Pb·kaon likelihood를 다시 fit한 새 후보가 아니다.

Körber Helm과 기존 Gaussian의 D/T product는 등록된
\(q=(0,10,20,29.64757,40)\) MeV 5점에서 최대 표본 잔차
\(8.99074\times10^{-5}\)다. \(q=i m_\phi\) 잔차
\(-4.94119\times10^{-5}\)는 exterior-residue 해석 진단일 뿐 full folded
실공간 response나 측정점이 아니다. \(q=40\) MeV scalar-radius 양 끝점의 중심
coupling 진단은 +1.2336--+1.2973%지만 strange-slope와 공동 covariance를 포함한
완전한 불확실성 띠가 아니다.

2026 lattice D/He3 sigma-term을 T proxy로 쓰면 공통 coupling 진단은
+1.10567% ± 1.48046%다. 실제 T 값과 D/T covariance가 없으므로 likelihood가
아니다. Andreoli의 D two-body fraction 0.7--3.0%를
\((1-\Delta^{(2b)})^{-1/2}-1\)로 exact amplitude 변환하고 현대 light weight를
곱한 범위는 +0.21267--+0.92756%다. T 부호, 동일 regulator의 current/potential,
fit된 short-range contact와 공동 covariance는 여전히 없다.

최종 scalar-current certification은 p/n·sigma covariance, normalization likelihood,
ab-initio density, full scalar form factor, 실제 T, uncertainty propagation,
regulator 일치, two-body likelihood/covariance와 3.24--50 fm 실공간 응답 leaf를
직접 재검사한다. 이 인증 뒤에도 upstream UV/action 및 Pb·kaon·dark-sector 제약을
다시 논리곱한다. 현재 모든 단계에서 `scalar_current_certification_pass=False`,
`physical_ce_fusion_branch_accepted=False`다.
