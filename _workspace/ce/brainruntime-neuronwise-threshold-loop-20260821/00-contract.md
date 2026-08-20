# Research contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-discrete-hybrid-property-loop-20260821`

Revision: 2 — pre-implementation audits restricted scalar-broadcast equivalence
to the admissible hysteresis domain, acknowledged deterministic `asdict` schema
growth, froze mixed-vector validation plus a nontrivial delay snapshot, and
replaced initialization caching with use-time resolution so mutable legacy
scalar config semantics remain intact.

## 1. 목적과 주장 상한

A7-H는 signed heterogeneous circuit strength $W_{ij}$를 actual runtime과 맞췄지만
bit lower/upper와 active eligibility threshold는 global scalar뿐임을 확인했다. A8-T는
기존 scalar API를 그대로 보존하면서 optional neuronwise threshold vectors를 실제
`BrainRuntime`에 추가한다.

성공 상한은

```text
NEURONWISE_THRESHOLD_RUNTIME_PASS / SCALAR_BACKWARD_COMPATIBLE /
RUST_VECTOR_BIT_UNSUPPORTED_FAIL_CLOSED / EMPIRICAL_UNTESTED
```

이다. 이는 측정된 생물학적 역치, cortical folding, 학습, AGI 성능 또는 Rust vector-bit
지원을 뜻하지 않는다. A8-D delay repair도 이 run에 섞지 않는다.

## 2. source와 predecessor freeze

| frozen file | SHA-256 | role |
|---|---|---|
| predecessor `31-validation.md` | `3da81fba48d92497d42f151e41e9362a285a81354169674ecbdc374d3441e67` | A7 hybrid property receipt |
| predecessor `40-final-report.md` | `0928792665b79de2bf40bfa98d308aee10625c50a4cdc326c0bdad567c6ab9e` | threshold implementation boundary |
| `runtime.py` | `4d73dce1ad79dd51e4bcf757b7a97f0302c43add0478071fbff9197a831f901a` | scalar runtime baseline |
| `tests/test_runtime_contracts.py` | `b365f061fb74353a724988d6fc266c286e38296107007e1a7746dca46bfffefd` | scalar no-delay backend baseline |
| Rust `lib.rs` | `8970a3e76f374ea9dc63ff0b798b5c796d3638f840e9551dbcd928ee02ca041f` | scalar binding boundary |
| Rust `kernel.rs` | `0b26f3e99b5208181402898805d966b7564b25c4fbe567033e2da75c5a0d68c2` | scalar bit threshold boundary |

허용된 구현 파일은 `runtime.py`와 새 focused threshold test 하나뿐이다. empirical
asset은 열지 않는다. interpreter와 dependency receipt는 A7의 Python `3.11.9`,
Torch `2.12.1+cpu`, NumPy `2.4.6`, `reality_stone.__version__=0.2.10`을 다시 기록한다.

## 3. additive config API

기존 scalar fields는 이름과 의미를 바꾸지 않는다.

```text
active_threshold: float
bit_lower_threshold: float
bit_upper_threshold: float
```

다음 optional fields만 더한다.

```text
neuronwise_active_threshold: tuple[float, ...] | None = None
neuronwise_bit_lower_threshold: tuple[float, ...] | None = None
neuronwise_bit_upper_threshold: tuple[float, ...] | None = None
```

입력 sequence는 `__post_init__`에서 immutable `tuple[float,...]`로 정규화한다.
string과 scalar처럼 sequence가 아닌 값은 `TypeError` 또는 `ValueError`로 거부한다.
각 provided vector는 길이가 정확히 `dim`이고 모든 성분이 finite여야 한다. effective
vectors는

$$
\boldsymbol\vartheta=
\begin{cases}
(\vartheta_1,\ldots,\vartheta_q),&\text{neuronwise active provided},\\
\vartheta\mathbf1,&\text{otherwise},
\end{cases}
\tag{T1}
$$

$$
\boldsymbol\theta^\pm=
\begin{cases}
(\theta_1^\pm,\ldots,\theta_q^\pm),&\text{corresponding vector provided},\\
\theta^\pm\mathbf1,&\text{otherwise}.
\end{cases}
\tag{T2}
$$

적어도 하나의 neuronwise bit vector가 provided이면 scalar broadcast counterpart까지
포함한 두 effective bit vectors의 모든 성분이 finite이고, 모든 $i$에서
$\theta_i^-<\theta_i^+$임을 요구한다. vector가 전혀 없는 legacy scalar-only
config에는 새 finite/ordering rejection을 소급하지 않는다.

## 4. runtime 식

actual selection은 A7-H의 budget $K_M$을 유지하되 eligibility만 성분별로 바꾼다.

$$
E=\{i:s_i\ge\vartheta_i\},\qquad
K'=\min(K_M,|E|),\qquad
q_i^+=\mathbf1\{i\in\operatorname{TopK}_{K'}(s_j:j\in E)\}.
\tag{T3}
$$

bit hysteresis는

$$
b_i^+=
\begin{cases}
1,&a_i^+\ge\theta_i^+,\\
0,&a_i^+\le\theta_i^-,\\
b_i,&\theta_i^-<a_i^+<\theta_i^+.
\end{cases}
\tag{T4}
$$

이다. `BrainRuntimeConfig`는 mutable이므로 threshold tensor를 initialization 때 고정
cache하지 않는다. `_select_active`, `_step_torch`, `_use_rust`가 호출될 때마다 현재
config에서 effective vectors를 검증하고 device float tensors로 resolve한다.
scalar-only config는 현재 scalar가 broadcast된 tensor를 얻는다. snapshot은 config를
이미 deepcopy하므로 별도 tensor state 없이 restore 뒤 같은 use-time resolution을 쓴다.
다만 dataclass `asdict(config)` schema에는 vector가 `None`이어도 세 optional key가
새로 들어가므로 기존 config digest는 결정론적으로 바뀐다. 이는 scalar dynamics의
backward compatibility와 구분해 receipt에 기록한다.

$W\in\mathbb R^{q\times q}$는 이번 변경과 독립이며 기존처럼 heterogeneous signed
strength를 유지한다. threshold vector를 $W$나 salience에 흡수하지 않는다.

## 5. Rust fail-closed boundary

현재 Rust ABI와 kernel은 bit lower/upper를 scalar `f32`로만 받는다. 따라서
neuronwise bit lower 또는 upper가 하나라도 supplied이면:

- `backend="auto"`는 CPU에서도 Torch cell path를 선택해야 한다;
- explicit `backend="rust"` constructor는 `ValueError`로 즉시 거부해야 한다;
- scalar-only path는 기존 Rust behavior와 parity test를 그대로 유지해야 한다.

neuronwise **active** threshold만 supplied된 경우는 Rust cell state와 무관하고 outer
Python `step()`이 active selection을 다시 계산하므로 no-delay Rust를 허용한다. Rust가
내부적으로 hard-code한 `.22` active-count는 outer count가 덮어쓰므로 receipt-only다.
이 제한을 Rust vector support라고 부르지 않는다.

## 6. frozen focused tests

새 test file 하나에서 다음을 고정한다.

### T-A — validation and canonicalization

- list/tuple inputs become tuples;
- scalar/string non-sequence, wrong length, NaN, Inf, and effective
  $\theta_i^-\ge\theta_i^+$ reject;
- lower-vector/upper-scalar와 lower-scalar/upper-vector를 각각 시험하고, scalar
  counterpart의 NaN/Inf 및 한 coordinate의 equality/inversion도 reject;
- legacy scalar-only config construction remains accepted.
- runtime construction 뒤 scalar `active_threshold`를 바꾸면 다음 selection이 즉시
  바뀌어야 하며 stale cache가 없어야 한다.

### T-B — exact scalar broadcast compatibility

admissible $\theta^-<\theta^+$인 동일 scalar config와 그 scalar를 `dim`회 반복한
explicit vectors를 forced Torch,
zero noise, same state/input에서 한 step 실행한다. activation, refractory, memory,
adaptation, STP, bit, salience, lifecycle와 active count는 bit-exact 또는 float
`atol=0, rtol=0`이어야 한다. `RuntimeStep`이 salience를 반환하지 않으므로 두
post-state에 동일 input/replay를 넣은 `_compute_salience(...)` 값을 직접 비교한다.

### T-C — heterogeneous bit and eligibility witness

`dim=3`, zero $W$, old activation `.2`, zero drive이면 WAKE activation은 `.164`다.

```text
initial bit = [0,1,1]
theta+ = [.15,.22,.30]
theta- = [.10,.17,.20]
```

따라서 post bit는 정확히 `[1,0,0]`이어야 한다. 별도 salience
`[.30,.40,.50]`, active thresholds `[.35,.35,.55]`, budget 2에서는 only index 1이
eligible이므로 mask `[0,1,0]`이어야 한다.

### T-D — snapshot continuation

tuple-normalized config를 snapshot/restore한 forced-Torch runtime은 vector fields를
정확히 보존하고 같은 다음 input에서 모든 continuous state, bit, lifecycle, delay
buffer/counter와 `RuntimeStep` scalars가 일치해야 한다.
`asdict(config)`가 세 optional key를 포함하고 scalar-only config digest도 predecessor와
달라진다는 schema receipt를 함께 남긴다.
snapshot 직전 delay buffer는
`[[.55,-.24,.33],[-.12,.44,.26]]`, 누적 `_delay_idx=3`으로 고정해 nonzero
buffer와 nonzero counter continuation을 실제로 확인한다.

### T-E — backend boundary

- vector-bit `auto`가 `_use_rust()==False`이고 forced Torch와 same result;
- vector-bit explicit Rust constructor가 `ValueError`;
- scalar config로 runtime을 만든 뒤 vector-bit tuple을 mutation한 경우에도 auto는
  다음 use에서 Torch로 fallback하고 explicit Rust는 cell 실행 전에 `ValueError`;
- active-vector-only, scalar-bit, `axon_delay=False`는 Rust가 available할 때 Torch/Rust
  full-step final active mask/count와 continuous state가 기존 tolerance `1e-5`에서 일치;
- 기존 scalar Rust parity node도 그대로 통과.

## 7. 판정과 재귀 규칙

T-A--T-E와 기존 scalar parity node가 모두 통과하면 위 claim ceiling으로 PASS다.

- formula/API mismatch: `P0_FORMULA`, test values/tolerances를 바꾸지 않고 한 번만 수정;
- source/config wiring error: `P2_APPARATUS`, 실패를 보존하고 최소 수정;
- vector-bit가 Rust로 조용히 흘러가면 `STOP_BACKEND_SEMANTICS`;
- scalar broadcast가 exact하지 않으면 `STOP_BACKWARD_COMPATIBILITY`;
- threshold나 test vector를 결과에 맞춰 조정하는 것은 금지;
- 생물학/AGI/anatomy claim은 성공 후에도 `EMPIRICAL_UNTESTED`다.
