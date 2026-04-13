# 용어 통일 가이드

> 이 문서는 리포지토리 전역에서 사용하는 변수명, 모듈명, 개념명의 정본이다.
> 새 코드나 문서를 쓸 때 반드시 이 표를 따른다.

---

## 1. 코어 상태 변수

| canonical 이름 | 코드 이름 | 의미 | 범위 |
|---|---|---|---|
| $a_i$ | `activation` | 국소 활성도 | $(-1, 1)$ |
| $r_i$ | `refractory` | 억제/불응 축적 | $\ge 0$ |
| $m_i$ | `memory_trace` | 국소 기억 흔적 (해마 구현 전 임시 캐시) | $\mathbb{R}$ |
| $b_i$ | `bitfield` | 히스테리시스 비트 | $\{0, 1\}$ |

### 1.1 memory 관련 용어 정리

| 기존 이름 (폐기) | canonical 이름 | 이유 |
|---|---|---|
| `memory` (셀 내부 EMA) | `memory_trace` | 해마와 혼동 방지. 해마 구현 전 임시 캐시 |
| `mem_dim` | `trace_dim` | memory_trace의 차원 |
| `rho_mem` | `trace_decay` | memory_trace의 감쇠율 |
| `w_mem` | `trace_inject` | memory_trace의 주입 가중치 |

현재 `runtime.py`에서는 이미 `memory_trace`로 명명되어 있어 일치한다. 향후 코드에서도 이 명칭을 유지한다.

---

## 2. 결합/구조

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $W_{ij}$ | `weight` | 셀 간 결합 가중치 |
| $W_{ij}(g)$ | Riemannian coupling | 리만 측지선 거리 기반 결합 |
| $\chi_{ij}$ | sparse mask | 결합 존재 여부 마스크 |
| $d_g(i,j)$ | geodesic distance | 리만 다양체 위 측지선 거리 |
| $\sigma$ | coupling radius | 결합 커널 폭 |

---

## 3. 모드/전역 상태

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $M_t$ | `mode` (`RuntimeMode`) | 전역 작동 모드 |
| WAKE / NREM / REM | `RuntimeMode.WAKE` / `.NREM` / `.REM` | 세 전역 모드 |
| $Q_t$ | body-loop control vector | sleep pressure, arousal, autonomic 등 |
| $p_{\text{sleep}}$ | `sleep_pressure` | 수면 압력 |
| arousal | `arousal` | 외부 자극 하중 |
| $B_t$ | energy budget | 모드별 동시 활성 상한 |

---

## 4. 모듈 생애주기

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| ACTIVE | `ModuleLifecycle.ACTIVE` | 현재 연산 참여 |
| IDLE | `ModuleLifecycle.IDLE` | 즉시 깨울 수 있는 대기 |
| DORMANT | `ModuleLifecycle.DORMANT` | 장기 휴면 |
| SLEEPING | `ModuleLifecycle.SLEEPING` | 내부 정리/압축 중 |

---

## 5. 해마/기억

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $H_t$ | `HippocampusMemory` | 해마 상태 (K, V, P) |
| $K_t$ | `_keys` | 기억 인덱스 (cue) |
| $V_t$ | `_values` | 저장된 에피소드 임베딩 |
| $P_t$ | `_priority` | 재생 우선순위 |
| encode | `hippocampus.encode()` | 활성 패턴 저장 |
| recall | `hippocampus.recall()` | 단서 기반 회상 |
| replay | `hippocampus.replay()` | 우선순위 기반 재생 |

---

## 6. 자기참조 루프 (agent loop)

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $z_t$ | relaxed state | relax/수렴 후 상태 |
| $a_t$ | action | 행동 선택 |
| $o_t$ | observation | 실행 결과 관측 |
| $c_t$ | critique | 자기 평가 |
| $S_t$ | agent state | 전체 에이전트 상태 |

### 6.1 행동 타입

```
ACTION_SET = ["THINK", "PLAN", "CRITIC", "REVISE", "SLEEP", "FINALIZE"]
```

---

## 7. 억제/suppression

| canonical 이름 | 코드 이름 | 비고 |
|---|---|---|
| suppression | `suppression` | `suppresson`은 오타. 전부 `suppression`으로 통일 |
| refractory | `refractory` | 불응기 (활성 후 억제) |
| inhibition | -- | refractory의 상위 개념 |

---

## 8. 백엔드

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| backend | `CEBackend` (Protocol) | 연산 백엔드 추상 계약 |
| TorchBackend | `TorchBackend` | PyTorch reference 구현 |
| RustBackend | `RustBackend` | Rust 최적화 구현 |
| `load_backend()` | -- | 단일 진입점에서 백엔드 선택 |
| `RelaxResult` | -- | relax 연산의 표준 반환형 |

**원칙**: `engine.py`, `runtime.py`, agent loop 등 상위 로직에서 Rust/CUDA import를 직접 하지 않는다. 오직 `load_backend()` 한 곳에서만.

---

## 9. 스냅샷/지속성

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| cold checkpoint | `$\mathcal{C}$` | 전체 구조 + 장기 기억 |
| warm snapshot | `BrainRuntimeSnapshot` | 현재 동적 상태 |
| live journal | `$\mathcal{J}$` | 실시간 이벤트 로그 |

---

## 10. 수식 기호 요약

| 기호 | 의미 | 문서 위치 |
|---|---|---|
| $s_i^t$ | 셀 $i$의 $t$ 시점 상태 | Layer A |
| $I_i^t$ | 셀 $i$의 총 입력 | Layer A |
| $W_{ij}(g)$ | 리만 결합 가중치 | Layer B |
| $M_t$ | 전역 모드 | Layer C |
| $\Theta^{(M)}$ | 모드별 파라미터 집합 | Layer C |
| $H_t$ | 해마 상태 | Layer D |
| $G_t$ | 전역 상태 요약 | Layer E |
| $\Pi$ | 모드 전환 함수 | Layer C |
| $\mathcal{E}$ | 해마 인코딩 연산자 | Layer D |
| $\mathcal{R}$ | 해마 회상 연산자 | Layer D |
| $\mathcal{S}$ | 자아 요약 함수 | Layer E |
| $\gamma_a, \kappa_a$ | 활성 감쇠/이득 | Layer A |
| $\gamma_r, \kappa_r$ | 억제 감쇠/이득 | Layer A |
| $\lambda_r, \lambda_m, \lambda_H$ | 억제/기억/재생 주입 계수 | Layer A |
| $\tau_i^+, \tau_i^-$ | 히스테리시스 상하 임계 | Layer A |
| $\eta_i^t$ | 확률적 잡음 | Layer A |
| $B_t$ | 에너지 예산 | Layer C |
| $Q_t$ | body-loop 제어벡터 | Layer C |
| $\Psi_{\text{global}}$ | 전역 뇌파 관측량 | Layer B 출력 |

---

## 11. 폐기 / 사용 금지 명칭

| 폐기 이름 | 대체 이름 | 이유 |
|---|---|---|
| `suppresson` | `suppression` | 오타 |
| `memory` (셀 내부 EMA) | `memory_trace` | 해마와 혼동 |
| `mem_dim` | `trace_dim` | 위와 동일 |
| `rho_mem` | `trace_decay` | 위와 동일 |
| `w_mem` | `trace_inject` | 위와 동일 |
| `phi_global` (brain core state) | backend observable | brain runtime core가 아님 |
| `pi_global` (brain core state) | backend observable | brain runtime core가 아님 |
| `BrainState(r,k,phi,pi,...)` | `BrainRuntimeSnapshot(activation, refractory, memory_trace, bitfield, ...)` | 물리장 시뮬레이터 상태에 묶이지 않게 |
