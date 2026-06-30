# 디퓨전 오케스트레이션 사양 (Diffusion Orchestration)

> 관련: `7_AGI/12_Equation.md`(canonical 5계층), `7_AGI/17_AgentLoop.md`(자기참조재귀/F절), `6_뇌/04_그래프결합과이완.md`(graph Laplacian Δ_G), `clarus-agent-guard/server/scheduler.py`(현행 셀-워크 오케스트레이션), `experiments/RESULTS_recursion.md`(ClarusCell 고정점 실증)
>
> 이 문서는 "DAGlet 생성과 자기폐루프의 다음 입력조건을 substrate 그래프 위 디퓨전으로 결정한다"는 오케스트레이션 아이디어를 **형식화**한다. 현재 코드/문서에 이 메커니즘은 구현되어 있지 않다. 본 문서의 모든 운영 주장은 `Open` 또는 `Hypothesis`이며, F-게이트(`12_Equation.md` 0.0절 F1–F4)를 우회하는 형태로 읽지 않는다.

---

## 0. 지위 선언 (먼저 고정)

| 구성요소 | 지위 | 근거 |
|---|---|---|
| substrate 그래프 위 $\Delta_G$ 디퓨전 (수학 연산) | `Exact` | 그래프 라플라시안은 정의로 닫힘 (`6_뇌/04`) |
| 디퓨전 정상상태 = 활성 우선순위 readout | `Selection` | 정규화/분기 선택 명시 시 닫힘 |
| 디퓨전 readout → DAGlet 생성 순서 | `Open` | 현행 `scheduler.py`는 셀 순차 워크. 디퓨전 대체는 미구현·미검증 |
| 디퓨전 정상상태 = 자기폐루프 고정점 (F절) | `Hypothesis` | `17_AgentLoop` 수축 사상과의 동일시는 별도 증명 필요 |
| 성능/효율 이득 (셀 워크 대비) | `Open test` | 벤치 없음 |

이 문서는 "디퓨전이 오케스트레이션을 한다"를 **증명하지 않는다.** 무엇을 정의하면 그 주장이 검증 가능해지는지를 형식화한다.

## 1. 기호 사전 (이 문서 한정)

| 기호 | 의미 | 차원/형식 | 지위 |
|---|---|---|---|
| $G=(V,E)$ | cell substrate 그래프 (가능한 모든 cell과 연결) | 이산 구조 | 정의 (guard `server/cells/`) |
| $L_G = D - A$ | combinatorial graph Laplacian, $A$ 인접, $D$ 차수 | 무차원 | `Exact` |
| $\Delta_G$ | 정규화 라플라시안 $D^{-1/2}L_G D^{-1/2}$ | 무차원 | `Exact` |
| $\phi \in \mathbb R^{|V|}$ | substrate 위 활성 잠재장 (cell별 1성분) | 무차원 (정규화) | 정의 |
| $s \in \mathbb R^{|V|}$ | field source — 이벤트/관찰의 cell별 주입 | 무차원 | 정의 |
| $\tau$ | 디퓨전 가상시간 (오케스트레이션 축, 물리시간 아님) | 무차원 | 보조좌표 |
| $\pi$ | 활성 우선순위 readout (DAGlet 노드 선택 측도) | 무차원 확률 | `Selection` |

주의: $\phi$는 `12_Equation.md`의 kernel dynamics 상태 $a_i$와 **자동으로 같지 않다.** $\phi$는 *오케스트레이션 평면*(어떤 cell을 언제 펼칠지)의 장이고, $a_i$는 *셀 내부 동역학*의 활성이다. 동일시하려면 별도 식별 단계가 필요하다(F-게이트 F1 메커니즘 결손 참조).

## 2. 코어 식 (디퓨전 흐름)

substrate $G$ 위에서 활성장 $\phi$는 source-driven 디퓨전으로 진화한다:

$$
\frac{\partial \phi}{\partial \tau} = -\Delta_G\,\phi + s - \lambda\,\phi
$$

- $-\Delta_G\phi$: substrate 결합을 따라 활성을 이웃 cell로 평탄·확산 (`6_뇌/04`의 graph relaxation과 동형).
- $s$: 이번 이벤트의 외생 주입 (salience cell이 만든 초기 관심).
- $-\lambda\phi$: 누설(leak) — 무한 누적 방지, 정상상태 존재 보장.

**정상상태** ($\partial_\tau\phi=0$):

$$
(\Delta_G + \lambda I)\,\phi^\star = s \quad\Longrightarrow\quad \phi^\star = (\Delta_G+\lambda I)^{-1} s
$$

$\lambda>0$이면 $\Delta_G+\lambda I \succ 0$이므로 $\phi^\star$는 **유일**하다. 이것은 substrate 위 source $s$의 정규화된 확산 확산(graph Green's function)이다. — 여기까지 `Exact`.

**무차원 게이트:** $\Delta_G, \lambda, \phi, s$ 모두 무차원이어야 한다($\Delta_G$는 정규화 라플라시안, $s$는 정규화 주입). `참조/무차원_감사_수학.md` 규칙 통과.

## 3. Readout: 디퓨전 → DAGlet 생성

정상상태 $\phi^\star$에서 cell 활성 우선순위를 정규화로 읽는다 (`Selection`):

$$
\pi_v = \frac{\exp(\phi^\star_v/T)}{\sum_{u}\exp(\phi^\star_u/T)}
$$

오케스트레이션 규칙 (현행 `scheduler.py` 순차 워크를 대체하는 후보):

1. DAGlet 시작 노드 = $\arg\max_v \pi_v$ (보통 salience).
2. 다음 노드 = 현재 노드의 substrate 이웃 중 $\pi$ 최대, 단 **이미 펼친 노드 제외**(acyclic 유지 — DAGlet 불변식).
3. `external_action` 라벨 edge는 디퓨전 우선순위와 무관하게 **반드시 policy 노드를 통과**한다(guard 그래프 불변식, `trace/audit.py`). 디퓨전은 *제안*만 하고 *집행 게이트를 우회하지 않는다*.

→ 규칙 1–2는 `Open`(미구현). 규칙 3은 기존 불변식의 재확인이며 **디퓨전이 이를 약화시키지 않음**이 본 사양의 안전 조건이다.

## 4. 자기폐루프와의 연결 (Hypothesis)

`17_AgentLoop.md` F절의 자기참조재귀는 $S_{t+1}=\mathcal T(S_t;e_t)$의 Banach 고정점이다. 본 사양의 가설:

> **(H1)** 디퓨전 정상상태 $\phi^\star_t$가 자기폐루프의 *다음 입력조건* $e_{t+1}$의 일부를 결정한다: $e_{t+1} = g(\phi^\star_t,\,o_t)$.

이때 두 고정점이 정합하려면 (H1)이 $\mathcal T$의 수축성을 깨지 않아야 한다. 충분조건 후보:

$$
\|g(\phi^\star,\cdot)-g(\phi^{\star\prime},\cdot)\| \le L_g\|\phi^\star-\phi^{\star\prime}\|,\quad L_g\cdot\|(\Delta_G+\lambda I)^{-1}\| < 1-\rho
$$

여기서 $\rho$는 `17_AgentLoop`의 기존 수축률. — 이 부등식이 성립하면 디퓨전-결합 폐루프도 유일 고정점을 갖는다. **증명 아님, 충분조건 형식화일 뿐.** (F2 비보존 바이패스 규칙: "수렴 보장" 표현 금지 — 항상 이 충분조건 + 주기적 복원으로 한정.)

실증 경고: `experiments/RESULTS_recursion.md`에서 weight-tied 재귀는 tol 하에서 **고정점에 수렴하지 않았다.** 따라서 (H1)의 고정점 수렴은 현재 **미입증**이며, 디퓨전 도입이 이를 개선하는지는 `Open test`다.

## 5. 측정 게이트 (무엇을 재면 격상되는가)

| 주장 | 닫히기 위한 측정 | 격상 |
|---|---|---|
| 디퓨전 readout이 셀 워크보다 나은 DAGlet | 동일 이벤트셋에서 route accuracy / latency 비교 (guard `bench.run` 확장) | `Open` → `Bridge` |

**측정 결과 (2026-07, `bench/diffusion_route_ab.py`)**: 이벤트 조건화 source $s$(field 플래그별 cell 주입) + `diffusion_route(π)`로 100문항 라우팅을 측정한 결과 — **walk 96/100 = diff 96/100, delta +0.0%, missed-verify 0(안전 바닥 유지)**. 디퓨전 라우터의 오분류 4건은 워크와 **동일한** 4건(상류 salience 플래그 오발화에서 기인, 라우터가 아님). 결론: 디퓨전이 strict-priority 워크의 라우팅을 **원리적 장-확산 메커니즘으로 정확히 복원**(parity)하나 **능가하지는 않음**. 따라서 `Open → Bridge` 격상 조건(우월성)은 **미충족**, 메커니즘 성숙도만 "상수 prior → 이벤트 조건화 parity"로 올림. 구현: `server/diffusion_scheduler.py`(순수 stdlib), 안전: `bench/test_diffusion.py` breaches=0 / route-parity 5-5, 회귀: `bench.all` ALL HELD.
| (H1) 고정점 정합 | closed-loop variant에서 $\hat\rho_t$(수축률 추정)·$I_c$ 보고 (`17_AgentLoop` F.-1.4 지표) | `Hypothesis` → `Bridge` |
| 디퓨전이 집행 불변식 비약화 | `bench.audit_check` breaches=0 유지 | 안전조건 (필수) |
| 성능 이득 | 셀워크 baseline 대비 ASR/false-allow 무회귀 + route↑ | `Open test` → `Phenomenology` |

## 6. 이 문서가 직접 하지 않는 일

- 디퓨전 오케스트레이션이 셀 워크보다 우수함을 **주장하지 않는다**(벤치 없음).
- $\phi$(오케스트레이션 장)와 $a_i$(셀 동역학)를 **동일시하지 않는다**.
- 자기폐루프 고정점 수렴을 **보장하지 않는다**(실증은 현재 미수렴).
- 집행 게이트(capability/policy)를 디퓨전으로 **대체하지 않는다** — 디퓨전은 제안층, 집행은 detection-free 구조층으로 분리 유지.

## 7. 최소 구현 경로 (참고, 미착수)

1. `server/scheduler.py`에 `DiffusionScheduler` 후보 추가 — substrate 인접행렬에서 $\Delta_G$ 구성, $(\Delta_G+\lambda I)^{-1}s$ 풀어 $\pi$ readout, 기존 워크와 A/B.
2. 안전: 규칙 3(action→policy 강제)을 디퓨전 경로에서도 `trace/audit.py`로 검사.
3. 평가: `bench.run`/`bench.audit_check`로 route·breach 비교. 격상은 5절 게이트 충족 시에만.
