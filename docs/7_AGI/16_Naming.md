# 용어 통일 가이드

이 문서는 runtime과 관련 문서에서 쓸 canonical name, alias, 폐기명을 정하는 명명 계약이다. 독자는 코드 식별자·API·수식 기호의 차이를 아는 독자를 전제로 하며, 표의 용어는 구현과 문서 migration을 위한 규칙이지 물리·생물 대상의 동일성 주장이 아니다.

코어 상태에서 결합·모드·기억·자기참조·backend·snapshot을 차례로 읽고, 마지막에 금지명과 비유 경계를 확인한다. canonical 이름은 새 코드·API·수식 설명의 기준이고 alias는 호환 migration 범위에서만 허용하며, 폐기명은 새 주장이나 새 인터페이스에 재도입하지 않는다.

> 이 문서는 리포지토리 전역에서 사용하는 변수명, 모듈명, 개념명의 정본이다.
> 새 코드나 문서를 쓸 때 반드시 이 표를 따른다.

---

## 1. 코어 상태 변수

코어 상태 변수 표는 runtime state producer와 consumer가 공유하는 canonical identifier·shape·범위를 고정한다. 코드 이름과 수식 기호가 다를 때에는 mapping을 명시하며, alias가 shape·serialization 계약을 바꾸지 않게 한다.

| canonical 이름 | 코드 이름 | 의미 | 범위 |
|---|---|---|---|
| $a_i$ | `activation` | 국소 활성도 | $(-1, 1)$ |
| $r_i$ | `refractory` | 억제/불응 축적 | $\ge 0$ |
| $m_i$ | `memory_trace` | 국소 기억 흔적 (해마 구현 전 임시 캐시) | $\mathbb{R}$ |
| $b_i$ | `bitfield` | 히스테리시스 비트 | $\{0, 1\}$ |

### 1.1 memory 관련 용어 정리

memory 용어는 short-lived trace, persistent store, replay output을 구분하기 위한 migration 규칙이다. 생물학적 기억 비유와 API 이름을 섞지 않으며, deprecated alias는 기존 artifact 읽기 범위에서만 관리한다.

| 기존 이름 (폐기) | canonical 이름 | 이유 |
|---|---|---|
| `memory` (셀 내부 EMA) | `memory_trace` | 해마와 혼동 방지. 해마 구현 전 임시 캐시 |
| `mem_dim` | `trace_dim` | memory_trace의 차원 |
| `rho_mem` | `trace_decay` | memory_trace의 감쇠율 |
| `w_mem` | `trace_inject` | memory_trace의 주입 가중치 |

현재 `runtime.py`에서는 이미 `memory_trace`로 명명되어 있어 일치한다. 향후 코드에서도 이 명칭을 유지한다.

---

## 2. 결합/구조

결합·구조 이름은 matrix, adjacency, field tensor의 producer·consumer와 수식 기호를 연결한다. 물리 geometry 비유는 구현 shape·정규화·backend API의 canonical 이름을 대체하지 않는다.

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $W_{ij}$ | `weight` | 셀 간 결합 가중치 |
| $W_{ij}(g)$ | Riemannian coupling | 리만 측지선 거리 기반 결합 |
| $\chi_{ij}$ | sparse mask | 결합 존재 여부 마스크 |
| $d_g(i,j)$ | geodesic distance | 리만 다양체 위 측지선 거리 |
| $\sigma$ | coupling radius | 결합 커널 폭 |

---

## 3. 모드/전역 상태

모드·전역 상태 이름은 tick 기반 control label과 summary readout의 API 경계를 정한다. 수면·각성·자아 같은 비유어는 canonical code identifier가 아니며, alias가 mode transition contract를 바꾸지 않는다.

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

생애주기 용어는 생성·활성·휴면·제거와 rollback의 state transition을 명시한다. 이름 migration은 serialization version과 fixture를 함께 갱신해야 하며, 기존 alias의 제거는 artifact compatibility를 고려한다.

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| ACTIVE | `ModuleLifecycle.ACTIVE` | 현재 연산 참여 |
| IDLE | `ModuleLifecycle.IDLE` | 즉시 깨울 수 있는 대기 |
| DORMANT | `ModuleLifecycle.DORMANT` | 장기 휴면 |
| SLEEPING | `ModuleLifecycle.SLEEPING` | 내부 정리/압축 중 |

---

## 5. 해마/기억

해마·기억 명명은 write·read·replay API를 구분하는 구현 contract다. 뇌 해마 비유는 기능 설명에 한정되며, canonical name이 생물학적 기제·기억 효능을 주장하지 않는다.

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

자기참조 이름은 self-state producer와 monitor·action consumer의 인터페이스를 표시한다. 자아·의식 비유는 계산 proxy를 넘지 않으며, API 이름만으로 주관 경험·도덕적 지위를 환원하지 않는다.

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| $z_t$ | relaxed state | relax/수렴 후 상태 |
| $a_t$ | action | 행동 선택 |
| $o_t$ | observation | 실행 결과 관측 |
| $c_t$ | critique | 자기 평가 |
| $S_t$ | agent state | 전체 에이전트 상태 |

### 6.1 행동 타입

행동 타입은 runtime이 선택·기록·검증할 수 있는 유한 출력 label의 canonical 집합이다. 아래 코드는 API 입력·출력 contract이며, 이름 자체가 인지 기능이나 생물학적 행동을 증명하는 것은 아니다.

```
ACTION_SET = ["THINK", "PLAN", "CRITIC", "REVISE", "SLEEP", "FINALIZE"]
```

---

## 7. 억제/suppression

억제 이름은 residual·threshold·intervention tensor의 역할을 구분한다. 환각·병리 비유는 canonical implementation name과 분리하며, alias가 false-positive·negative metric의 뜻을 바꾸지 않게 한다.

| canonical 이름 | 코드 이름 | 비고 |
|---|---|---|
| suppression | `suppression` | `suppresson`은 오타. 전부 `suppression`으로 통일 |
| refractory | `refractory` | 불응기 (활성 후 억제) |
| inhibition | -- | refractory의 상위 개념 |

---

## 8. 백엔드

backend 이름은 reference implementation, kernel, protocol, artifact의 책임 경계를 정한다. 언어별 alias는 parity·serialization migration 범위에서만 쓰며, backend pass는 성능·과학적 참의 이름이 아니다.

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

snapshot·persistence 이름은 save, load, version, rollback artifact의 API를 구분한다. deprecated 경로는 이전 artifact 복구에만 남기고 새 build·새 수식 설명에서는 canonical 명칭을 사용한다.

| canonical 이름 | 코드 이름 | 의미 |
|---|---|---|
| cold checkpoint | `$\mathcal{C}$` | 전체 구조 + 장기 기억 |
| warm snapshot | `BrainRuntimeSnapshot` | 현재 동적 상태 |
| live journal | `$\mathcal{J}$` | 실시간 이벤트 로그 |

---

## 10. 수식 기호 요약

기호 표는 수식의 정의역·shape·단위와 코드 identifier의 대응을 재확인한다. 같은 글자가 물리 유비와 tensor를 동시에 뜻하지 않도록, 구현 문맥에서는 canonical API mapping을 우선한다.

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

폐기명 표는 migration 경계와 재도입 금지를 명시한다. 금지명은 legacy 문서·artifact를 읽는 설명에만 제한적으로 나타나며, 새 코드·API·수식·주장에는 canonical 이름으로 치환한다.

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

---

## 12. 생물학과 런타임의 경계 명명

이 절은 생물·물리 비유어와 runtime 구현명을 분리하는 최종 규칙이다. 비유는 함수 역할을 설명하지만 canonical name·API·수식의 정의역이나 과학적 지위를 바꾸지 않으며, 혼동은 문서와 코드 review에서 failure로 처리한다.

| canonical 이름 | 뜻 | 현재 지위 |
|---|---|---|
| biological Clarus cell | 막·대사·수리로 자기동일성을 유지하는 실제 세포 | 생명 문서의 개념 |
| neural Clarus assembly | 여러 뉴런에 걸친 국소 재귀 계산 단위 후보 | 실데이터에서 찾아야 하는 `Open` 객체 |
| Clarus instruction | assembly 수준에서 문맥을 건너 재사용되는 상태 전이 후보 | `Open` |
| runtime `ClarusCell` | `runtime.py`의 소프트웨어 상태 단위 | 구현 명세 |

`runtime ClarusCell ≠ biological Clarus cell ≠ neural Clarus assembly`를
기본 규칙으로 삼는다. 신경 집단을 말할 때 unqualified `cell`을 쓰지
않으며, synthetic gate 통과를 생물학적 assembly 확인으로 번역하지 않는다.
