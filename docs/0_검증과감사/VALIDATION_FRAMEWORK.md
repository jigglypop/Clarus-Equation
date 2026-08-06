# CE 이론 검증 프레임워크 (Clarus Equation Proof Requirements)

**최종 업데이트:** 2026-08-06
**목표:** 이전 분석(I-XIII)의 모든 "미증명 부분", "순환논리", "검증 부재" 항목을 체계적으로 정리하고 각각의 증명 경로를 명확히 함

> **2026-08-06 실행 상태 교정:** Co1 bootstrap solver와 Co2 scorecard는
> 구현되어 회귀 테스트가 통과한다. 그러나 구현 통과는 물리 증명이 아니다.
> 수식 반례와 직접 교정은
> [전체 정합성 완성 루프](FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md)를
> 우선한다. 아래의 과거 시간 추정·성공률은 프로젝트 계획 기록이지
> 증명 점수가 아니다.

---

## 개요

CE 이론의 검증은 7개 계층으로 구분된다:

1. **TIER 1: CRITICAL BLOCKERS** - 실패한 공리·bridge가 의존하는 하위 주장만 중지 (Axiom Layer)
2. **TIER 2: BRIDGE INTEGRITY** - Bridge layer 약화 → 응용층 흔들림
3. **TIER 3: COSMOLOGY PREDICTIONS** - 검증 가능하나 부분 실패 용납 가능
4. **TIER 4: PARTICLE PHYSICS VALIDATIONS** - 고정밀도, 검증 가능
5. **TIER 5: BRAIN & NEUROSCIENCE** - 거의 모두 미검증
6. **TIER 6: ENGINEERING APPLICATIONS** - 기술 공극 극심
7. **TIER 7: CODE & COMPUTATIONAL** - 구현상 미완성

---

## 현재 코어 모델선택 게이트 읽기

코어의 수치 검산과 물리 모형 선택을 한 판정으로 합치지 않기 위해, 다음 게이트와 사전등록 manifest를 정본 실행 경로로 둔다.

- 실행 게이트: `examples/physics/core_model_selection_gate.py`
- 후보·관측 역할 manifest: `benchmarks/core_claims_v1.json`
- 구현: `reality_stone/python/reality_stone/clarus/core_model_selection.py`
- 회귀 테스트: `tests/test_core_model_selection.py`

```powershell
python examples/physics/core_model_selection_gate.py
python -m pytest tests/test_core_model_selection.py tests/test_bootstrap_solver.py -q
```

이 게이트는 일반화된 스칼라 고정점

$$
x=S\!\left(D_{\mathrm{eff}}K(x)\right)
$$

에서 생존 법칙 $S$ 3개, 피드백 닫힘 $K$ 3개, $D_{\mathrm{eff}}$ 사상 3개를 조합한 **27개 후보**를 모두 순회한다. 설정 구간의 모든 수치근을 스캔하고 각 근의 residual과 국소 안정반경을 별도로 기록한다.

판정 필드는 다음처럼 읽는다.

| 필드 | 묻는 질문 | 현재 의미 |
|---|---|---|
| `algebraic_status` | 식과 수치 구현이 정규화, 경계조건, root residual, 안정성 계산 등의 계약을 만족하는가 | 현재 manifest의 27개 대조 후보는 `PASS`. 이는 수학·수치 계약 통과이지 물리 이론의 실험적 확증이 아니다 |
| `selection_status` | 입력·참조행을 제외한 독립 관측으로 후보군을 선택할 수 있는가 | 현재 `UNDERIDENTIFIED`. 독립 selection 관측이 1개뿐이므로 27개 구조 가운데 승자를 정하거나 CE를 검증했다고 말할 수 없다 |

`UNDERIDENTIFIED`는 실패를 숨기는 완곡어가 아니라 **현재 데이터 설계로는 모형 식별이 불가능하다는 중지 판정**이다. 각 후보의 $\chi^2$와 표준화 residual은 진단값으로 남기되, 독립 selection 관측이 최소 2개가 되기 전에는 `COMPETITIVE` 또는 `DISFAVORED`로 승격하지 않는다.

또한 이 비교의 범위는 manifest와 결과에

```text
recursion_scope = scalar_equal_row_sum
```

으로 고정된다. 즉 결합행렬 $A$의 행합이 같은 경우 나타나는 **스칼라 불변 부문**만 비교하며, 일반 벡터 $A$ 전체의 재귀를 검증하지 않는다. 행합이 다르거나 비등방적인 다공간 재귀는 `reality_stone/python/reality_stone/clarus/multispace_bootstrap.py`와 별도 게이트에서 다룬다.

```powershell
python examples/physics/multispace_recursion_gate.py
python -m pytest tests/test_multispace_bootstrap.py -q
```

따라서 스칼라 모델선택 게이트의 `algebraic_status=PASS`를 벡터 다공간 게이트의 통과로 전용해서는 안 된다.

### A1에서 Q0 공변 작용으로 가는 국소 게이트

보통 action Hessian을 곧바로 공변 텐서나 stress tensor로 읽는 비약을
막기 위해 다음 국소 게이트를 둔다.

- 실행 게이트: `examples/physics/a1_q0_action_gate.py`
- 구현: `reality_stone/python/reality_stone/clarus/a1_q0_action_bridge.py`
- 회귀 테스트: `tests/test_a1_q0_action_bridge.py`
- 전체 통과 조건:
  `A1_Q0_COVARIANT_ACTION_LOOP.md`의 `Q0.0`–`Q0.8`

```powershell
python examples/physics/a1_q0_action_gate.py
python -m pytest tests/test_a1_q0_action_bridge.py -q
```

이 게이트는 비선형 장 좌표변환에서 보통 Hessian에 기울기 비례 추가항이
생기고 connection 보정이 이를 제거한다는 국소 반례를 재현한다. 또한
\(Z_2,\ v_\Phi=0\) 포탈에서 \(h\)-\(\Phi\) cross-Hessian은 0이지만
\(h\Phi^2\), \(h^2\Phi^2\) vertex가 남는다는 대수도 검산한다. 같은
정규화의 현행 \(\mu_\Phi=0\),
\(\lambda_{HP}=\delta_N^2\) benchmark는
\(m_\Phi=43.8056765\,\mathrm{GeV}\),
\(\mathrm{BR}_{\rm inv}=0.770822\)를 준다. PDG 2026이 열거한 ATLAS
direct Run-2 상한 \(\mathrm{BR}_{\rm inv}<0.107\) (95% CL)에 대해
이 branch는 `REJECT`다.

통과해도 `covariant_action_complete=False`,
`stress_tensor_derived=False`, `spectral_density_derived=False`다.
전체 CE+SM action, field-space geometry, gauge/ghost, measure,
renormalization, metric variation과 물리 pole은 아직 `OPEN`이다.

### Q0.0–Q0.3 최소 manifest와 gauge/ghost 통제 게이트

앞 절의 국소 반례를 실제 작용 convention과 배경에 연결하기 위해, 깨진
\(U(1)\) Abelian-Higgs 모형에 \(Z_2\)-odd 실수 싱글릿을 붙인 최소 통제
절단을 둔다. 이는 전체 전약력 또는 CE+SM 작용이 아니라, 부호·tadpole·
gauge/ghost 항등식을 재현하는 풍동 모형이다.

- 정식 범위 문서:
  `Q0_0_Q0_3_MINIMAL_MANIFEST.md`
- 고정 입력 manifest:
  `benchmarks/q0_minimal_abelian_higgs_v1.json`
- 실행 게이트:
  `examples/physics/q0_manifest_gate.py`
- 구현:
  `reality_stone/python/reality_stone/clarus/q0_manifest_gate.py`
- 회귀 테스트:
  `tests/test_q0_manifest_gate.py`

```powershell
python examples/physics/q0_manifest_gate.py
python -m pytest tests/test_q0_manifest_gate.py -q
```

게이트는 다음 네 결과를 독립적으로 기록한다.

1. `control_q0_0_pass`: 장, 고전 bare tree-level 작용, 부호, 고정
   Minkowski 배경, 경계조건과 제외 sector가 manifest에 모두 있는가.
2. `control_q0_1_pass`: \((h,\chi,\phi)\)의 평탄한 Cartesian scalar
   metric과 한 방향 비선형 좌표 대조군에서 공변 Hessian이 복원되는가.
3. `control_q0_2_pass`: Higgs tadpole이 0이고 \(Z_2\) 배경
   \(\phi=0\)에서 singlet tadpole도 0인가.
4. `control_q0_3_pass`: \(R_\xi\) 항이 \(A^\mu\partial_\mu\chi\)
   혼합을 상쇄하고 FP ghost와 Goldstone의 게이지 의존 질량 장부가
   일치하는가.

틀린 Higgs 배경, gauge-fixing 부호, ghost 질량과 scope 위장 manifest는
같은 테스트에서 실패해야 한다. 네 통제 표시가 모두 참이어도
`full_q0_0_complete`–`full_q0_3_complete`, `full_q0_pass`,
`full_ce_sm_complete`, `stress_tensor_derived`,
`spectral_density_derived`는 모두 `False`다. 빠진 전체
\(SU(3)\times SU(2)\times U(1)\) sector, fermion/Yukawa, graded
field-space, 동적 중력, 전역 gauge orbit, regulator/counterterm와
BRST/Slavnov–Taylor 복구를 통제 모형의 성공으로 대신할 수 없다.

### 양자 jump에서 분지과정으로 가는 구조 게이트

복소 양자진폭이나 Hessian의 비대각 원소를 곧바로 비음수 Poisson
offspring로 읽지 않기 위해 다음 조건부 게이트를 둔다.

- 실행 게이트: `examples/physics/quantum_jump_bridge_gate.py`
- 구현: `reality_stone/python/reality_stone/clarus/quantum_jump_bridge.py`
- 회귀 테스트: `tests/test_quantum_jump_bridge.py`

```powershell
python examples/physics/quantum_jump_bridge_gate.py
python -m pytest tests/test_quantum_jump_bridge.py -q
```

이 게이트는 이미 주어진 Kossakowski 행렬, Hamiltonian, jump operator,
sector projector에 대해 양의 준정부호, population–coherence 폐쇄,
constant hazard, 행-출발 방향을 검사한다. 구조상 통과해도
`ce_sm_derivation_complete=False`와
`poisson_branching_derived=False`를 유지한다. 즉 이 결과는 CE+SM
작용에서 jump operator를 유도했다는 판정도, jump 전이율 \(W\)가 독립
offspring 행렬 \(A\)라는 판정도 아니다. 전체 조건과 실패 대안은
`CORE_STRENGTHENING_LOOP.md`의 Q-loop를 따른다.

---

## TIER 1: CRITICAL BLOCKERS

### Axiom Layer: 핵심 공리 검증 (5개)

**상황:** 이 계층의 항목 중 하나라도 실패하면 해당 계층의 주장 강도가 내려간다. 코어 수학, bridge readout, 현상론/응용층을 분리해 판정해야 한다.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 방법 | 의존성 | 실패시 결과 |
|----|------|------|--------|-----------|--------|--------|---------|---------|-----------|
| **C1** | inverse-correlation scale에서 물리적 pole로 가는 입자언어 bridge | `0_검증과감사/CE_TWO_POINT_AND_VERTEX_LOOP.md`, `0_검증과감사/CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md` | $m_\xi=m_p\delta_N^2=29.6991596\,\mathrm{MeV}$는 정확한 scale identity이고 physical pole·residue·field identity는 `OPEN` | 고정한 $\Gamma_{\rm ren}^{(2)}$의 영점과 양의 residue를 구하고, 같은 장·결합으로 production/decay 및 독립 탐색 likelihood를 닫음 | 4 | - | **재규격화 correlator + 실험** | 입자언어 Bridge layer | **pole 해석만 폐기하고 scale identity는 보존** |
| **C2** | 연결 국소성: $S(D_1+D_2)=S(D_1)S(D_2)$ | `0_검증과감사/CORE_STRENGTHENING_LOOP.md` 1.2절 | 독립 구간·가법 깊이·memory 부재를 명시한 반증 가능한 모형 전제; 함수적분에서 자동으로 나오는 정리는 아님 | 같은 CE+SM 미시 dynamics에서 composition law를 유도하거나 $\Delta_{\rm comp}=0$을 독립 자료로 검정 | 5 | - | **미시 유도 + composition gate** | 지수 생존 부문 | **memory kernel을 포함한 대안 생존모형으로 교체** |
| **C3** | 양의 생존 character의 지수형 유일성 | `0_검증과감사/CORE_STRENGTHENING_LOOP.md` 1.3절 | $S(0)=1$, $0<S\le1$, 곱셈성 아래 $S(D)=e^{-\kappa D}$는 `Exact conditional`; $\kappa=1$은 optical-depth 단위 규약 | 정리의 가정·정의역·비자명성을 proof checker와 문서에서 동일하게 유지 | 1 | 완료 | **해석 증명 + 반례 회귀** | C2의 연결 국소성 | **가정이 깨지면 비지수 대안으로 분기** |
| **C4** | $\Lambda^2V^*\simeq V^*$ closure class의 $d=3$ 해 | `1_강의/A_연역적_유도.md` 4절 | $\binom d2=d$의 양의 비자명 해 $d=3$은 `Exact conditional`; 이 closure class를 자연에 채택하는 단계는 `Selection/Open` | metric·orientation·closure class를 고정하고 anomaly-free SM 표현 및 대안 차원 evidence를 공동 비교 | 2 | 정리 완료 | **기하 대수 + 모형선택** | 차원 Selection layer | **closure class의 물리 채택을 교체** |
| **C5** | $\alpha_{\text{total}}=1/(2\pi)$ 경계조건 | `3_상수/1_격자기본량.md`, `상수.md` | 원의 $2\pi$ 주기는 기하 입력이고 $n=1$은 최소 양의 winding `Selection H2`; scheme 불변 필연성 주장이 아님 | 같은 scale·scheme에서 $n$ 및 대안 경계조건을 사전등록하고 RG·threshold likelihood로 비교 | 4 | - | **Gauge/RG 모형선택** | 결합상수 Track B | **H2 경계조건만 교체·배제** |

**평가:**
- C1의 현행 중앙 scale은 $29.6991596\,\mathrm{MeV}$다. 탐색 신호만으로는 field identity가 정해지지 않으므로 correlator·pole·residue gate를 먼저 통과해야 한다.
- C3와 C4의 수학 부분은 명시된 가정 안에서 닫혔다. C2의 연결 국소성과 C4 closure class의 물리 채택은 별도 모형선택 문제다.
- C5는 필연성 주장이 아니라 사전등록 가능한 경계조건이다. Track A의 입력 $\alpha_s$와 Track B의 입력 $\alpha_{em}$을 독립 경로로 유지한다.

---

### Bridge Layer Core: 검증 필수 (4개)

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **B1** | $s_A^2:=4\alpha_s^{4/3}$ registered matching | `상수.md` Layer 1--2 | Track A 산술과 neutral projector 모형 구현; 물리적 $s_W^2$ scheme map은 Open | RG·threshold·scheme map 뒤 독립 weak/QCD fit 및 UV matching | 4 | - | **Gauge/RG likelihood** | 입자물리 | 실패 시 CE mixing boundary 폐기 |
| **B2** | energy-weighted $x\leftrightarrow\Omega_b$ | `0_검증과감사/BRIDGE_B2_DERIVATION.md` | 공변 stress readout, projector, unitary 실현과 epoch 전달식 구현 | 같은 action의 Boltzmann/perturbation holdout | 5 | - | **QFT+cosmology** | $\Omega_b$ | full likelihood에서 이미 고정 benchmark reject |
| **B3** | Kernel $K(x)=1-x$ | `3_상수/3_부트스트랩.md`, `2_경로적분과_응용/12_전이구간.md` | 완전 이분할·scalar sufficiency·mixture affinity 아래 유일성 정리와 저분율 branch 안정성 구현 | 미시 collision/genealogy에서 같은 충분통계와 kernel 재현 | 3 | - | **kinetic matching** | 고정점 | 대안 memory kernel과 evidence 비교 |
| **B4** | $m_\xi=m_p\delta_N^2$ inverse-correlation scale | `경로적분.md` 8절, `0_검증과감사/CE_TWO_POINT_AND_VERTEX_LOOP.md` | 입력 $m_p$와 $\delta_N$에서 $29.6991596\,\mathrm{MeV}$를 재현하는 exact arithmetic; 물리적 pole 예측은 `OPEN` | 기준 scale 선택의 action-level 유도와 $\Gamma_{\rm ren}^{(2)}$ pole·residue·field identity 검증 | 3 | - | **provenance + two-point gate** | C1 | **scale bridge와 particle pole을 분리 유지** |

---

## TIER 2: BRIDGE INTEGRITY (추가 7개)

Bridge layer 일부 약화는 허용하나, 강도 저하 → 응용층 신뢰도 하락.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **B5** | exact-$Z_2$ Higgs portal benchmark | `경로적분.md` 8.2절, `0_검증과감사/CE_TWO_POINT_AND_VERTEX_LOOP.md` | $\mu_\Phi=0$, $\lambda_{HP}=\delta_N^2$에서 $m_\Phi=43.8056765\,\mathrm{GeV}$, $\mathrm{BR}_{\rm inv}=0.770822$; PDG 2026 direct 상한 $0.107$에 대해 `REJECT` | coupling·질량·채널을 작용에서 새로 고정하고 동일 width/likelihood gate 재실행 | 4 | - | **BSM width + direct limit** | portal branch | **이 canonical benchmark를 폐기하고 새 branch를 사전등록** |
| **B6** | 스케일 승격 입력 $v_{\text{EW}}=246.21965\,\mathrm{GeV}$ | `상수.md` Layer 8 | 차원 있는 외부 기준 입력이며 CE 무차원 코어의 출력으로 세지 않음 | $v_{\text{EW}}$를 독립 scale dynamics에서 유도하거나 입력 provenance를 계속 고정 | 3 | - | **scale provenance** | 모든 차원량 | **무차원 비와 절대 scale을 분리 유지** |
| **B7** | 무차원 코어 $\{e,\pi,i,1,0\}$ 완전성 | `axium.md` 1.2a | "직관적이나 수학적 완전성 미증명" | 이 5개 상수가 모든 무차원 물리를 생성하는가? | 5 | 120 | **정수론/해석학** | 모든 이론 구조 | **추가 상수 필요할 수 있음** |
| **B8** | 부트스트랩 branch 구조 | `3_상수/3_부트스트랩.md`, `2_경로적분과_응용/14_자기재귀성_대칭.md` | $D\le1$ 한 해, $D>1$ 두 해와 저분율 수축구간 증명 완료 | stochastic branch-selection dynamics | 4 | - | **해석·수치 gate** | 모든 cosmology | branch 초기조건 공개 |
| **B9** | $R$ dark 분할 | `상수.md` Layer 3 | $R=\alpha_sD(1+x\delta_N)=0.37823870$으로 식·수치 통일; full-cov 고정 benchmark `Rejected` | 공변 dark stress와 growth를 같은 action에서 계산 | 3 | - | **DESI/CMB/growth** | 암흑섹터 | 새 action의 evidence 필요 |
| **B10** | Dimensionless unit checker 완성 | `reality_stone/python/reality_stone/clarus/dimensionless.py` | 등록 수식의 단위 검사는 구현; 자유형 Markdown 전수 의미 검사는 계속 확장 | action·EOM·width·관측량까지 typed dimension gate에 등록 | 2 | 진행 | **정적 검사 + 회귀** | 모든 식 검증 | **등록되지 않은 식을 수동 감사 대상으로 유지** |
| **B11** | 순환성 제거: 공리→관측 체인 투명성 | 문서군 전체 | "일부 식에서 관측값 사후 피팅 의심" | 각 상수별로 "입력인지 출력인지" 명확화 | 4 | 80 | **문서 감사** | 모든 주장 신뢰성 | **논리 순환 발견 시** |

---

## TIER 3: COSMOLOGY PREDICTIONS (5개)

우주론 예측은 부분 실패 용납 가능 (동적 모델 진화 가능).

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **U1** | 고정-background DESI DR2 모형 gate | `3_상수/7_우주론.md`, `3_상수/9_우주론_수식_의미와_후보.md` | 외부 $r_d=147.09\,\mathrm{Mpc}$에서 $\chi^2=40.20145$, dof 13, $p=1.2828\times10^{-4}$; EH hybrid $r_d=151.50523\,\mathrm{Mpc}$에서 $\chi^2=41.19455$, $p=8.8602\times10^{-5}$로 모두 `REJECT` | 같은 parameter vector의 CMB·BAO·SN·growth와 동적 dark-energy perturbation을 공동 적합 | 2 | - | **DESI DR2 covariance + 미래 holdout** | density bridge | **고정 background를 새 공변 동역학 branch로 교체** |
| **U2** | $H_0$ readout selector $q\in\{0,1\}$ | `3_상수/9_우주론_수식_의미와_후보.md` 5절 | 산술 readout은 $66.8027457$와 $72.7023707\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$를 재현하나 ansatz와 selector는 검증되지 않음 | $q$와 horizon-defect map의 물리적 생성 원리를 고정하고 독립 distance-ladder/GW holdout 평가 | 4 | - | **독립 $H_0$ likelihood** | horizon readout | **readout ansatz만 폐기** |
| **U3** | finite-$\xi$ inflation normalization | `2_경로적분과_응용/05_인플레이션.md` | $A_s=2.10\times10^{-9}$와 $N_*=57.1999$는 입력; $\xi=0.4904868$에서 $n_s=0.9661711385$, $r=0.0043456103$, $\lambda_4=1.3434991\times10^{-10}$을 재현 | RG·isocurvature·reheating과 $A_s$ 입력 covariance를 포함한 공동 likelihood | 5 | - | **inflation likelihood** | $n_s,r$ | **benchmark parameter 영역을 재선택·배제** |
| **U4** | $N_*=57.1999$ | `2_경로적분과_응용/05_인플레이션.md` | reheating benchmark input으로 장부 고정 | exact-$Z_2$ annihilation/preheating에서 재계산 | 3 | - | **Boltzmann/lattice** | primordial GW | $N_*$ posterior 전파 |
| **U5** | dark dynamics | `3_상수/7_우주론.md` | 공변 확장식과 고정-background kill test 분리; 고정 패키지는 reject | background+perturbation 공동 scalar-tensor fit | 4 | - | **CMB+BAO+SN+growth** | $w(z)$ | 고정 분할을 대안 action으로 교체 |

---

## TIER 4: PARTICLE PHYSICS VALIDATIONS (5개)

고정밀 likelihood로 검증할 수 있다. 등록 산술의 재현은 독립 관측 성공으로 세지 않는다.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **P1** | 뮤온 $g-2$ scalar/Wilson diagnostic | `경로적분.md` 9절 | 올바른 CP-even kernel에서 unit Wilson 계수는 $248.5639\times10^{-11}$, 같은 결합의 finite-mass 값은 $162.5520\times10^{-11}$이며 현행 WP25 gate에 `REJECT` | UV Wilson matching·operator mixing·공통 covariance를 고정해 계수를 사전등록 | 4 | - | **EFT matching + precision likelihood** | muon 부문 | **해당 operator/coupling branch를 재구성** |
| **P2** | CKM 전체 unitary matrix construction | `상수.md` Layer 4 | 원소별 $\alpha_s$ 멱법칙은 독립 예측으로 사용하지 않음; 하나의 up/down Yukawa 또는 Wilson matching 입력에서 전체 $V_{\rm CKM}$을 공동 산출하는 경로가 `OPEN` | 동일 scale·scheme의 $Y_u,Y_d$를 대각화해 3각·CP phase·Jarlskog·unitarity를 holdout과 공동 검증 | 4 | - | **flavour global likelihood** | quark Wilson/matching 입력 | **원소별 numerology를 폐기하고 행렬 branch 재구성** |
| **P3** | PMNS 전체 unitary matrix construction | `상수.md` Layer 5 | 원소별 $\sin^2\theta_{13}$ 식은 정본 검증식이 아님; charged-lepton 및 neutrino mass/Wilson 입력의 공동 대각화가 `OPEN` | Dirac/Majorana 선택, Weinberg·seesaw matching, RG를 고정해 3각·phase·mass splittings·unitarity를 공동 검증 | 4 | - | **oscillation global likelihood** | lepton Wilson/matching 입력 | **원소별 numerology를 폐기하고 질량행렬 branch 재구성** |
| **P4** | Higgs pole-mass matching benchmark | `상수.md` Layer 4 | 최신 $M_H=125.11\,\mathrm{GeV}$는 비교 target; $M_Z(1+\alpha_sD_N)=125.3824\,\mathrm{GeV}$의 수치 근접은 matching benchmark이며 독립 pole 예측이 아님 | 같은 action의 tadpole·self-energy·counterterm·RG·threshold로 pole equation과 residue를 닫고 독립 입력을 분리 | 4 | - | **EWSB pole/RG matching** | 기본 scale 입력 | **근접 산술을 예측으로 세지 않음** |
| **P5** | proton-radius numerology | `경로적분.md` 11절 | $\Delta r_p^2\propto\delta_N^2/m_\xi^2$ target은 mediator coupling·form factor·loop matching이 없어 현행 정본에서 `DISCARDED` | 재도입하려면 작용, lepton·proton vertex, Wilson coefficient, form factor와 독립 muonic/electronic likelihood를 처음부터 고정 | - | 폐기 | **EFT form-factor gate** | 없음 | **폐기 상태 유지; 다른 성과에 포함 금지** |

---

## TIER 5: BRAIN & NEUROSCIENCE CLAIMS (6개)

거의 모두 미검증. 생물학적 검증 작업량 극심.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **Br1** | 전역 뇌 방정식 42 매개변수 | `6_뇌/00_읽기지도.md` | "Synthetic validation만, 자기순환" | 공개 데이터 (Allen, HCP, OpenNeuro) 독립 피팅 | 3 | 160 | **fMRI, ephys** | 모든 뇌 예측 | **방정식 재구성** |
| **Br2** | $\rho_B = 0.489$ BYPASS 상수 | `6_뇌/05_실험근거.md` 3절 | "구조 대응 주장, 측정 없음" | 뇌 imaging에서 background connectivity ratio 실제 측정 | 2 | 80 | **Graph 분석** (connectome) | 뇌 역학 | **BYPASS 부재** |
| **Br3** | Laplacian coupling $\Delta_G p_r$ | `6_뇌/04_그래프결합과이완.md` | "Graph Laplacian, 기하 metric 미지정" | Brain geometric metric 정의 (유클리드? 측지선? functional?) | 3 | 100 | **dMRI + graph** | 방정식 타당 | **Coupling 부실** |
| **Br4** | 수면 부트스트랩 $T_{\text{WAKE}} = 0.315$ | `6_뇌/07_수면과복구.md` | "이론, animal trace 검증 없음" | 쥐/초파리 매일 부트스트랩 여부 (NREM 중 재설정?) | 3 | 120 | **Electrophysiology** | continual learning | **메커니즘 불명** |
| **Br5** | STDP 학습 상한 $\eta_{\max}$ | `6_뇌/08_시냅스가소성.md` | "Timing window 맞으나 복잡 task 증명 없음" | STDP-only로 실제 학습 가능? (다른 plasticity?) | 3 | 100 | **Behavioral exp** (STDP manipulation) | 학습 기초 | **생물학적 부실** |
| **Br6** | Consciousness $\mathcal{C} := C_1 \land C_2 \land C_3$ | `7_AGI/7_Consciousness.md` | "철학 정의, 경험적 검증 불가" | 객관적 의식 정의/측정 (IIT Φ? C3?) | 5 | ∞ | **인식론적 불가능** | 의식 주장 | **부분 삭제** |

---

## TIER 6: ENGINEERING APPLICATIONS (5개)

기술 공극 극심. 대부분 불가능성 증명 필요.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **E1** | 핵융합 공명 source/linewidth gate | `4_공학적_활용/01_핵융합_설계.md` 2.4절 | $29.6991596\,\mathrm{MeV}$는 inverse-correlation target일 뿐 관측된 입자선이 아니며, 요구 linewidth와 production rate도 아직 입력·시뮬레이션 단계 | C1의 pole·residue를 먼저 닫고 source transfer function, 총 에너지 예산, achievable linewidth를 같은 benchmark에서 검증 | 5 | - | **spectral R&D + end-to-end simulation** | C1 + fusion source | **물리적 source가 없으면 이 공명 경로만 중지** |
| **E2** | 초전도 보손 flux $10^{40}$ vs $10^{-20}$ | `4_공학적_활용/05_초전도체_설계.md` 5.7.4절 | "60 자리 부족 (!)" | γ→clarus 변환 cross-section 측정/계산 상한 | 4 | 200 | **입자 실험** | 상온 초전도 | **재검토** |
| **E3** | 초전도 coherence 시간 1.2 ps | `4_공학적_활용/05_초전도체_설계.md` 5.7.2절 | "UCP: <100 fs, 물리 모순" | 실제 금속 dephasing 측정 또는 quantum dot 대안 | 2 | 40 | **응축 물질** | 공명 축적 | **불가능** |
| **E4** | QEC 개선 3.16% | `4_공학적_활용/02_양자오류보정.md` 3절 | "Simulation만, 실제 hardware 미검증" | IBM/Google 50+ qubit 시스템 테스트 | 2 | 160 | **양자 hardware** | QEC claim | **기여 없음** |
| **E5** | 기상 모델 "무차원 잔차장" | `4_공학적_활용/08_기상_그래프잔차_연구.md` | "개념만, 구현 전무" | NAS (Clarus regularization 도움 여부) | 3 | 200 | **Training runs** | 기상 예측 | **응용 제거** |

---

## TIER 7: CODE & COMPUTATIONAL VALIDATION (5개)

Co1·Co2는 구현·회귀 통과했고, Co3–Co5의 실자료·응용·전수 단위 gate가 남아 있다.

| ID | 문제 | 위치 | 현재상태 | 필요한 증명 | 난이도 | 시간(h) | 검증 | 의존성 | 실패시 |
|----|------|------|--------|-----------|--------|--------|-----|---------|--------|
| **Co1** | Bootstrap solver | `reality_stone/python/reality_stone/clarus/bootstrap_solver.py` 및 tests | **구현·회귀 통과**; 낮은 가지 선택과 $x=1$ 가지를 분리 | 변수치환·branch 문서와 solver 계약 동기화 | 1 | 완료 | **Unit test** | 모든 cosmology | 회귀 실패 시 downstream 중지 |
| **Co2** | Scorecard.py | `tests/scorecard.py` | **구현됨**; 입력/reference equality와 독립 prediction을 구분해야 함 | provenance·공동 covariance·holdout scorer 추가 | 2 | 진행 | **Regression + statistical gate** | 상수 검증 | reference match를 prediction으로 세지 않음 |
| **Co3** | Brain dynamics fitting | `reality_stone/clarus/` | "42-param fitting, overfitting 위험" | Regularization + cross-validation (real fMRI) | 3 | 200 | **Allen Brain data** | 뇌 부문 | **무효** |
| **Co4** | AGI 3×3+1 vs transformer | `examples/` | "개념문서만, 구현 없음" | 동일 param: CE sparse vs dense 벤치마크 | 2 | 80 | **MLPerf** | AGI claim | **무효** |
| **Co5** | Dimensionless checker | `clarus/dimensionless.py` | "미완성, 모든 식 미검증" | 모든 공식 차원 자동 검증 | 2 | 40 | **프로그래밍** | 모든 식 | **오류 가능** |

---

## 우선순위 & 실행 계획

### Phase 1: IMMEDIATE (즉시 착수, 2주)

**현재:** 핵심 solver와 scorecard는 구현됐고, 전 문서 정적·차원 검사 범위를
넓히는 단계다.

- **Co1** Bootstrap solver → 완료, branch/variable regression 유지
- **Co2** Scorecard.py → 구현 완료, prediction-only 공동 likelihood 보강
- **B10** Dimensionless checker → 등록식 밖 Markdown 수식까지 확장

**검증:** 각 코드 실행 후 unit test 통과 확인

---

### Phase 2: HIGH IMPACT (물리 bridge와 likelihood)

**왜:** 재현 가능한 산술을 물리적 pole·stress·관측 likelihood로 승격하는
단계가 현재의 핵심 병목이다.

- **C1** $29.6991596\,\mathrm{MeV}$ inverse-correlation scale의 two-point gate
  - 먼저 $\Gamma_{\rm ren}^{(2)}$, pole, residue, field identity를 같은 action에서 계산한다.
  - 그 뒤에만 NA62, PADME, LDMX 등 독립 탐색의 production/decay likelihood와 비교한다.

- **U1** 고정-background `REJECT` 이후의 동적 우주론 branch
  - 현재 DESI DR2 covariance snapshot과 두 $r_d$ package의 실패를 regression으로 고정한다.
  - 새 action은 background와 perturbation을 공동 계산하고 미래 untouched data를 holdout으로 둔다.

- **B2** energy-weighted $x\leftrightarrow\Omega_b$ bridge
  - 같은 공변 action에서 baryon projector, total stress, epoch transfer를 유도한다.
  - CMB·BAO·SN·growth 공동 likelihood를 통과하기 전에는 density 산술을 관측 성공으로 세지 않는다.

---

### Phase 3: PROOF EFFORTS (조건부 정리와 물리 선택 분리)

- **C2–C5**
  - C3의 지수형 정리와 C4의 조합론적 $d=3$ 정리는 가정·정의역 회귀를 유지한다.
  - C2의 연결 국소성과 C4 closure class의 자연 채택은 미시 유도/모형선택으로 검증한다.
  - C5의 $n=1$은 selection으로 사전등록하고 RG·threshold likelihood에서 대안과 비교한다.

- **B1, B3, B4, B5, B6**
  - B1은 $s_A^2$와 물리적 $s_W^2$ 사이의 scheme map을 닫는다.
  - B3은 conditional kernel theorem과 미시 genealogy를 연결한다.
  - B4는 scale identity와 physical pole을 분리하고, B5의 rejected portal benchmark는 새 action으로만 교체한다.
  - B6의 $v_{\rm EW}$는 유도 전까지 외부 scale 입력으로 유지한다.

- **U3** finite-$\xi$ inflation
  - $A_s$와 $N_*$를 입력 장부에 고정한 채 $n_s$, $r$, $\lambda_4$ forward map을 회귀한다.
  - RG·isocurvature·reheating 공동 likelihood가 다음 독립 물리 gate다.

---

### Phase 4: VALIDATION AT SCALE (대규모 검증, 6-12개월)

- **Br1-Br6** 뇌과학 부문 (160+시간)
  - 병렬: 신경생물학자 협력
  - 데이터: Allen, HCP, OpenNeuro

- **P1-P5** 입자물리 고정밀 (215시간)
  - 계획: FCC-ee, 다음세대 collider

- **E1-E5** 공학 응용 (target: 불가능성 증명)

---

### Phase 5: CODE COMPLETION & TESTING (6-12개월)

- **Co3-Co5** 구현 완성
- **전체 문서** 순환성 감사 (B11, 80시간)

---

## 증명 상태 요약

42개 항목을 한 개의 proof fraction으로 합치면 조건부 정리, 코드 구현,
입력 재현, 독립 관측 검증이 뒤섞인다. 현행 근거 장부는 다음처럼 서로 다른
판정축을 보존한다.

| 계층 | 항목 수 | 닫힌 수학·구현 근거 | 독립 물리·자료 판정 | 남은 핵심 gate |
|---|---:|---|---|---|
| **Axiom C1–C5** | 5 | C3 지수형 정리와 C4 $d=3$ closure 정리: `2 conditional theorems` | C1 pole `OPEN`; C2 locality와 C4/C5 물리 selection `OPEN` | 미시 composition, pole/residue, RG 모형선택 |
| **Bridge Core B1–B4** | 4 | B1·B4 등록 산술 재현, B3 conditional kernel 정리 | 독립 물리 bridge `0 PASS`; B2 고정 우주론 downstream은 `REJECT` | weak-angle scheme map, 공변 stress, two-point function |
| **Bridge+ B5–B11** | 7 | B8 branch 해석·수치 gate와 B10 등록식 검사가 구현됨 | B5 canonical portal `REJECT`; 나머지 미시/자료 bridge `OPEN` | 새 portal action, scale dynamics, 순환성·전수 단위 감사 |
| **Cosmology U1–U5** | 5 | density·$H_0$·finite-$\xi$ forward arithmetic 재현 | 독립 empirical `0 PASS`; 두 고정-background package 모두 `REJECT` | 동적 background+perturbation 공동 likelihood |
| **Particle P1–P5** | 5 | P1 loop kernel과 수치 diagnostic 재현 | 독립 empirical `0 PASS`; P1 `REJECT`, P2–P4 `OPEN`, P5 `DISCARDED` | Wilson/RG matching과 global likelihood |
| **Brain Br1–Br6** | 6 | 문서·synthetic construction만 존재 | 실자료 gate `0/6` | 사전등록, 독립 dataset, out-of-sample 평가 |
| **Engineering E1–E5** | 5 | 요구식·일부 simulation만 존재 | end-to-end 실험 gate `0/5` | source·loss·hardware를 포함한 전체 예산 |
| **Code Co1–Co5** | 5 | Co1 bootstrap과 Co2 scorecard `2/5 implemented` | 구현 통과는 물리 검증으로 세지 않음 | Co3–Co5 실자료·응용·전수 단위 gate |

따라서 현재 정당한 집계는 “Axiom conditional theorem 2개(C3·C4),
Bridge Core conditional theorem 1개(B3), Bridge+ branch 정리·수치 gate
1개(B8), 핵심 코드 2개 구현, 독립 물리 PASS 0개”다. 명시적 `REJECT`와
`OPEN`은 각 행에 보존한다. 자세한
근거는 `PROOF_STATUS_MATRIX.md`와
`FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md`의 gate를 따른다.

---

## Gate 실패의 영향 범위

실패는 의존 그래프를 따라 전파하되, 실패한 bridge와 무관한 정의·조건부
정리·산술 identity까지 함께 폐기하지 않는다.

| Gate | 현재 또는 가능한 실패 | 중지되는 주장 | 보존되는 근거 | 다음 branch |
|---|---|---|---|---|
| C1/B4 | pole·residue 부재 또는 독립 탐색 배제 | $29.6991596\,\mathrm{MeV}$ particle identity와 이를 쓰는 E1 source | $m_p\delta_N^2$ scale arithmetic | 다른 field/scale 또는 비입자 readout |
| C2 | $\Delta_{\rm comp}\ne0$ 또는 memory 검출 | memoryless exponential survival의 물리 적용 | C3의 조건부 수학 정리 | nonlocal/state-augmented kernel |
| C4 | closure class가 자연에서 선택되지 않음 | $d=3$의 물리적 유일성 | $\binom d2=d$의 조건부 해 | 대안 closure와 evidence 비교 |
| C5/B1 | RG·threshold likelihood에서 CE 경계조건 배제 | H2와 물리적 weak-angle 식별 | Track A/B의 입력별 산술 | 새 boundary condition/scheme map |
| B2/U1 | 공변 stress bridge 또는 full likelihood 실패 | 고정 density의 우주론 모형 해석 | bootstrap 수학과 density vector 계산 | 동적 공변 action; 현재 고정 package는 이미 `REJECT` |
| B5 | direct Higgs-invisible gate 실패 | $\mu_\Phi=0$, $\lambda_{HP}=\delta_N^2$ canonical portal | exact-$Z_2$ pole·width 계산 | coupling 감소, channel closure 또는 새 portal branch |
| P1–P5 | Wilson/global likelihood 실패 | 해당 flavour·$g-2$·Higgs 관측 예측 | unitary construction 요구와 matching 산술 | 공통 mass/Wilson matrix 재구성 |
| Br/E | 독립 자료·hardware gate 실패 | 해당 응용 주장 | 물리 코어의 독립 항목 | 응용 범위만 중지 |

---

## 검증 기준 (Acceptance Criteria)

### Axiom 검증 (C1-C5)

| ID | Pass | Fail/중지 |
|---|---|---|
| C1 | 동일 action의 $\Gamma_{\rm ren}^{(2)}$에 실수 pole, 양의 residue, 안정한 field identity가 있고 사전등록한 독립 탐색 likelihood를 통과 | pole/residue 부재, field mismatch 또는 독립 배제 |
| C2 | 미시 유도 또는 독립 composition 자료에서 사전등록 허용오차 안의 $\Delta_{\rm comp}=0$ | 유의한 composition residual 또는 memory state 필요 |
| C3 | $S(0)=1$, 양의 range, 곱셈성에서 $S=e^{-\kappa D}$ proof와 반례 회귀를 재현 | proof premise 누락 또는 구현이 정리와 불일치 |
| C4 | closure class 안의 조합론 proof와 자연의 closure 선택 likelihood를 각각 통과 | 수학 반례 또는 물리 selection 실패를 구분 기록 |
| C5 | scale·scheme·입력·대안 $n$을 사전등록한 RG likelihood에서 H2가 선택됨 | H2가 대안보다 지지되지 않음 |

### Bridge 검증 (B1-B6)

| ID | Pass | Fail/중지 |
|---|---|---|
| B1 | $s_A^2$에서 특정 physical $s_W^2$ 정의까지 RG·threshold·scheme map과 독립 EW likelihood가 닫힘 | scheme 혼용 또는 입력 재사용 |
| B2 | 동일 공변 action의 stress/projector에서 density를 산출하고 CMB·BAO·SN·growth 공동 holdout 통과 | component별 사후 보정 또는 full covariance reject |
| B3 | conditional theorem의 premise와 같은 충분통계/kernel이 미시 genealogy에서 재현 | 추가 memory/state가 필요 |
| B4 | inverse-correlation scale과 renormalized pole·residue·field identity를 모두 재현 | scale arithmetic만 맞고 pole이 없음 |
| B5 | 새 사전등록 portal이 width와 direct $\mathrm{BR}_{\rm inv}<0.107$ 및 관련 likelihood를 통과 | 현재 canonical branch처럼 limit 초과 |
| B6 | 절대 scale의 독립 dynamics 또는 명시적 외부-input provenance가 유지됨 | 입력을 무차원 코어의 예측으로 중복 계상 |

### Cosmology 검증 (U1-U5)

| Criterion | Pass | Fail/중지 |
|---|---|---|
| **Likelihood** | 사전등록 parameter vector, full covariance, 정규화·nuisance를 포함하고 holdout threshold 통과 | 단일 marginal 일치만 제시하거나 공동 likelihood reject |
| **Dynamics** | background와 perturbation을 같은 action/EOM에서 계산 | density 숫자만 있고 $w(z)$·growth dynamics 부재 |
| **Provenance** | $A_s$, $N_*$, $H_0$, $r_d$ 등 입력을 출력과 분리 | calibration 입력을 예측 성공으로 재계상 |

### Code 검증 (Co1-Co5)

| Criterion | Pass | Fail/중지 |
|---|---|---|
| **Unit/regression** | branch, residual, 단위, 경계조건에 대한 명시적 테스트 통과 | 테스트 실패 또는 중요한 분기 미등록 |
| **Integration** | manifest→solver→likelihood→verdict가 한 provenance chain으로 재현 | 중간 수동 숫자 삽입 또는 입력/출력 혼용 |
| **Reproducibility** | 버전·입력 hash·seed·환경과 수치 허용오차 고정 | 결과를 재생할 provenance 부재 |

---

## 문서 간 의존성 맵

```
axium.md (Axiom layer)
├── C1, C2, C3, C4, C5 (Experimental + Math proofs)
└─────────────────────┐
                      ↓
경로적분.md (Path integral framework)
├── A1-A4 (Operational axioms)
├── B1-B6 (Bridge rules)
└─────────────────────┐
                      ↓
상수.md (Constants derivation)
├── L1-L8 (Layers 1-8)
├── U1-U5 (Cosmology)
├── P1-P5 (Particle physics)
└─────────────────────┐
                      ↓
응용 섹션 (Applications)
├── 6_뇌 (Brain: Br1-Br6)
├── 4_공학적_활용 (Engineering: E1-E5)
└── 7_AGI (AGI: 뇌 의존)

Code layer (parallel)
├── reality_stone/ → Co1, Co2, Co3
└── tests/ → Co4, Co5
```

**의존 경로:** C2→C3는 생존법칙 경로, B2→U1/U5는 우주론 경로,
C1/B4→E1은 입자 source 경로다. B5 portal과 P2/P3 flavour는 각각 별도
action/Wilson 경로이며 한 경로의 실패를 다른 경로의 증명으로 전용하지 않는다.

---

## 최종 평가

### 현재 상태 (2026-08-06)

| Layer | 재현된 근거 | 현재 물리 판정 | 완료 조건 |
|---|---|---|---|
| **Axiom C1–C5** | C3·C4 conditional proof 2개 | C1, C2의 미시 origin, C4/C5 selection `OPEN` | pole/residue, composition, RG selection gate |
| **Bridge Core B1–B4** | B1/B4 산술, B3 conditional theorem | 독립 bridge `0 PASS`; B2 downstream `REJECT` | scheme map, stress/EOM, two-point gate |
| **Cosmology U1–U5** | density, $H_0$, inflation forward map | fixed-background 두 package `REJECT`; 동적 branch `OPEN` | background+perturbation 공동 holdout |
| **Particle P1–P5** | P1 kernel과 matching 산술 | P1 `REJECT`, P2–P4 `OPEN`, P5 `DISCARDED` | 공통 Wilson/RG/global likelihood |
| **Brain Br1–Br6** | synthetic/document construction | 독립 자료 검증 `0/6` | 사전등록 out-of-sample data gate |
| **Engineering E1–E5** | 요구식·일부 simulation | end-to-end 검증 `0/5` | source·loss·hardware 통합 gate |
| **Code Co1–Co5** | Co1·Co2 `2/5 implemented` | 물리 검증과 별개 | Co3–Co5 및 provenance chain 완성 |

근거가 없는 confidence 백분율은 삭제한다. 수학 proof count, 구현 count,
관측 PASS/REJECT는 서로 다른 분모를 가지므로 합산하지 않는다.

### 출판 준비도

- **조건부 수학·재현성 노트:** C3/C4의 가정, proof, 반례 회귀와 실행
  provenance를 동결하면 독립 산출물로 준비할 수 있다.
- **우주론·portal null-result 보고:** 현재 `REJECT`를 숨기지 않고 covariance,
  code, manifest, 환경을 보존하면 재현성/배제 보고의 재료가 된다.
- **통합 물리 예측 논문:** C1 pole, B1 scheme map, B2 공동 likelihood,
  P2/P3 Wilson construction이 닫히기 전에는 publication-ready로 판정하지 않는다.
- **뇌·공학 응용 논문:** 독립 실자료 또는 end-to-end hardware gate 전에는
  응용 검증 논문으로 판정하지 않는다.

### 위험도

근거 없는 성공·실패 확률은 부여하지 않는다.

| Risk | 현재 evidence | 영향 | 닫는 방법 |
|---|---|---|---|
| C1 particle bridge | physical pole 미확인 | E1과 particle-language branch | renormalized two-point + 독립 search likelihood |
| B2/U1 cosmology | 두 고정-background package `REJECT` | 고정 density의 관측 모형 해석 | 새 공변 동적 action + untouched holdout |
| B5 portal | $\mathrm{BR}_{\rm inv}=0.770822>0.107$ | canonical portal benchmark | 새 coupling/mass/channel 사전등록 |
| P2/P3 flavour | 원소별 식 폐기, matrix construction `OPEN` | CKM/PMNS 예측 | 공통 Yukawa/Wilson matching + global fit |
| Brain/Engineering | 독립 data/hardware gate 없음 | 해당 응용층 | out-of-sample 또는 end-to-end 검증 |

---

## 다음 단계 (Next Steps)

### 이 문서 이후

1. **문서·수치 gate 유지**
   - canonical manifest에서 bootstrap, density, $H_0$, portal, scalar loop,
     finite-$\xi$ inflation을 재계산한다.
   - 입력/reference/fit/holdout 표기와 Markdown delimiter·표·링크를 전수 검사한다.

2. **물리 bridge 닫기**
   - C1/B4: two-point pole·residue·field identity를 먼저 계산하고 독립 탐색으로 이동한다.
   - B1: $s_A^2$의 RG·threshold·physical weak-angle scheme map을 구현한다.
   - B2/U1: 동적 공변 우주론 action과 공동 covariance likelihood를 구성한다.

3. **입자·응용 재구성**
   - P2/P3: 하나의 mass/Wilson 입력에서 unitary CKM/PMNS 전체를 산출한다.
   - B5/P1: rejected portal·$g-2$ branch를 새 사전등록 action으로만 교체한다.
   - E1은 C1 통과 후 source·linewidth·energy-budget gate를 실행한다.

---

**문서 버전:** 2.0 (2026-08-06 canonical evidence-ledger revision)

**다음 검토 조건:** canonical manifest 또는 독립 likelihood/pole/Wilson gate가 변경될 때
