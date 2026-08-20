# Research contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brain-circuit-manifold-property-loop-20260821`

Revision: 3 — pre-implementation math audit first removed an invalid
continuous-time saltation route, then fixed the ring clock, the actual global
selection map, the derivative arithmetic/state-reset protocol, and the
reachable clip-face domain before any witness was executed. A final transcription
check separated the unbounded runtime counter $\kappa$ from its modulo-$L$ slot
$k$; this changed no fixture, call, or tolerance.

## 1. 목적과 주장 상한

이 run은 A6의 smooth delayed map을 실제 `BrainRuntime`의 한 tick에 더 가깝게
확장한다. 다음을 서로 분리해 시험한다.

1. discrete mode가 고정되고 모든 clamp margin이 strict한 branch의 ordinary
   Jacobian,
2. clip face의 one-sided/Bouligand directional derivative,
3. bitㆍTopKㆍlifecycle처럼 discrete state가 바뀌는 event receipt,
4. Torch ring-delay와 Rust no-delay backend semantics.

성공 상한은

```text
DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED /
HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED / EMPIRICAL_UNTESTED
```

이다. actual brain, AGI 성능, 학습, cortical folding 또는 state-space Riemann
metric을 검증하지 않는다.

## 2. predecessor와 source freeze

| frozen file | SHA-256 | role |
|---|---|---|
| predecessor `00-contract.md` | `6322e6045b87f4fc7cd5d2bee29f1ceb800808de9dc1d2b5fcc0604926434fb4` | A6 smooth-domain boundary |
| predecessor `11-math.md` | `e82d15dc9b7b009023b9a61c9f02c567a38a44b84f400678233a1b816abdaea4` | delayed tangent definitions |
| predecessor `20-audit.md` | `3eddcd5830c4eb2d31ec073c05264cd51bd5da2d8ea8827becbe196401f08dd8` | property audit |
| predecessor `31-validation.md` | `7dbc77f3a844ccd773301f2bec135d80ba1b61fed131417741e08148393a0654` | A6 8/8 receipt |
| predecessor `40-final-report.md` | `d35491b525a6d6b11e16ef46a87f5a2ca85fb65bcb48d5835cc0b260fbf1fd27` | next-route boundary |
| `runtime.py` | `4d73dce1ad79dd51e4bcf757b7a97f0302c43add0478071fbff9197a831f901a` | authoritative Python/Torch order |
| `kernel.rs` | `0b26f3e99b5208181402898805d966b7564b25c4fbe567033e2da75c5a0d68c2` | Rust cell kernel |
| `test_runtime_contracts.py` | `b365f061fb74353a724988d6fc266c286e38296107007e1a7746dca46bfffefd` | existing no-delay parity boundary |

실행 환경은 `.codex/hooks/python.cmd`가 고른 Python `3.11.9`, NumPy `2.4.6`,
Torch `2.12.1+cpu`, `reality_stone 0.2.10`, Windows `10.0.26200`으로 고정하고
receipt에 실제 executable/version/source hash를 쓴다. empirical asset은 열지 않는다.

## 3. 왜 saltation을 폐기하는가

continuous-flow saltation matrix

$$
S=DR+\frac{(f^+-DRf^-)n^\top}{n^\top f^-}
$$

는 guard를 통과하는 event time, 양쪽 vector field와 transversality를 요구한다.
`BrainRuntime.step`은 tick boundary에서 clamp, hysteresis, TopK와 lifecycle guard를
평가하는 이산 map이다. event time을 root-find하지 않으므로 위 식의 대상이 아니다.
따라서 saltation을 적용하거나 straight-through gradient를 넣으면 `P0_FORMULA`다.

## 4. A7-H discrete augmented state

continuous state를

$$
z=(a,r,m,w,u,x,d^{(0)},\ldots,d^{(L-1)})\in\mathbb R^{(6+L)q}
\tag{H1}
$$

로 둔다. 각각 activation, refractory, memory trace, adaptation, STP facilitation,
STP resource와 delay ring slot이다. discrete state는

$$
\sigma=(b,q^{\rm prev},\ell,n,\kappa,M,\text{backend})
\tag{H2}
$$

이다. $b$는 hysteretic bit, $q^{\rm prev}$는 이전 lifecycle에서 얻은 active mask,
$\ell$은 lifecycle label, $n$은 inactivity counter, $\kappa\in\mathbb N_0$는
runtime의 누적 delay counter, $M$은 mode다. 실제 read/write slot은
$k=\kappa\bmod L$이다.

한 tick의 derivative 시험에서는 $W$, external $e$, goal, replay, mode, sparse order,
noise seed를 고정한다. STDP, hippocampal encode/recall 변화, auto-mode와 F1 feedback은
끈다. 그렇지 않으면 그 state와 guard를 추가로 증강해야 한다.

## 5. 고정 branch 식

$Q=\operatorname{Diag}(q^{\rm prev})$이고 actual Torch update order를 따라 old $u$를
$x$ update에 쓴다.

$$
\tilde u=\operatorname{clip}_{[0,1]}
\left[u-\alpha u+U(1-u)\odot q^{\rm prev}\right],
$$

$$
\tilde x=\operatorname{clip}_{[0,1]}
\left[x+\beta(1-x)-u\odot x\odot q^{\rm prev}\right].
\tag{H3}
$$

delay가 켜지면 current ring slot을 먼저 읽고 old activation을 그 slot에
**대입**한 뒤 index를 증가시킨다. 정확히, 호출 직전 상태를 아래첨자 $t$로 쓰면

$$
y_t=d_t^{(k_t)},\qquad k_t=\kappa_t\bmod L,\qquad
d_{t+1}^{(j)}=
\begin{cases}
a_t,&j=k_t,\\
d_t^{(j)},&j\ne k_t,
\end{cases}
\qquad
\kappa_{t+1}=\kappa_t+1 .
\tag{H4a}
$$

따라서 delay가 켜진 recurrent input은 $a_{t+1}$가 아니라 읽기 전 slot 값
$y_t$를 사용한다. delay가 꺼지면 $y_t=a_t$로 둔다.

$$
\rho_t=W\left(q_t^{\rm prev}\odot\tilde u_t\odot\tilde x_t\odot y_t\right),
\tag{H4}
$$

$$
h=\rho+g_e e+g_g g+g_RR_M-r_sr-c_ww+\eta,
$$

$$
a^+=\operatorname{clip}_{[-1,1]}
\left[(1-\gamma_a)a+\kappa_a\tanh h\right],
$$

$$
r^+=(1-\gamma_r)r+\kappa_r(a^+)^{\odot2},
\qquad
m^+=(1-\gamma_m)m+\gamma_m a^+,
$$

$$
w^+=\operatorname{clip}_{[0,C_w]}
\left[(1-\gamma_w)w+\gamma_w(a^+)^{\odot2}\right].
\tag{H5}
$$

식 (H3)--(H5)의 actual WAKE constants는

```text
alpha=.0015, beta=.008, U=.5, gamma_a=.18, kappa_a=.82,
gamma_r=.12, kappa_r=.24, gamma_m=.01, gamma_w=.005,
c_w=.12, C_w=2, g_e=.45, g_g=.20, replay_mix=.08,
refractory_scale=.35
```

다.

bit와 salience는

$$
b_i^+=
\begin{cases}
1,&a_i^+\ge\theta_i^+,\\
0,&a_i^+\le\theta_i^-,\\
b_i,&\text{otherwise},
\end{cases}
$$

$$
s_i=|a_i^+|+.35|e_i|+.25|R_i|+.20|g_i|-.15r_i^+.
\tag{H6}
$$

actual outer selection은 heterogeneous threshold 식이 아니라, global
$\vartheta=\texttt{active_threshold}$와 mode별 scalar budget $K_M$을 쓰는

$$
E=\{i:s_i\ge\vartheta\},\qquad
K'=\min(K_M,|E|),\qquad
q_i^+=\mathbf 1\!\left\{i\in\operatorname{TopK}_{K'}
\bigl(s_j:j\in E\bigr)\right\}
\tag{H6a}
$$

이다. 이번 WAKE fixture에서는 `active_ratio=2/3`, `dim=3`,
`f1_self_measure=False`이므로 $K_M=2$다. exact tie의 index 선택 규약은
Torch와 Rust 양쪽에서 정본으로 고정되어 있지 않으므로 tie는 미분·parity
정의역에서 제외하고 receipt-only adverse case로 둔다.

수학 envelope는 heterogeneous $\theta_i^\pm,\vartheta_i$를 허용할 수 있지만 현재
`BrainRuntimeConfig`는 세 threshold를 각각 하나의 global scalar로만 가진다.
따라서 actual runtime이 neuron-specific threshold를 구현했다고 주장하지 않는다.
반면 $W_{ij}$는 heterogeneous signed strength다. 이번 fixture는 `dale_law=False`라
Dale orientation을 시험하지 않는다.

## 6. derivative object

strict interior와 fixed $\sigma$에서는

$$
A_\sigma=D_zF_\sigma(z)
\tag{H7}
$$

를 ordinary branch Jacobian으로 쓴다. 반드시 $u,x,r,w,d$ block과
$q^{\rm prev}$ gate를 포함한다. activation-only A6 Jacobian은 runtime Jacobian이
아니다.

clip $c(y)=\operatorname{clip}_{[l,u]}(y)$의 올바른 one-sided derivative는

$$
c'(y;v)=
\begin{cases}
v,&l<y<u,\\
\max(v,0),&y=l,\\
\min(v,0),&y=u,\\
0,&y<l\text{ or }y>u.
\end{cases}
\tag{H8}
$$

이다. full map은 이 규칙을 forward directional chain rule로 전파한다. clip face를
가로지르는 central difference는 gate가 아니다.

$b$, TopK 또는 lifecycle가 perturbation으로 바뀌면 encoded discrete component의
difference quotient는 $O(1/\epsilon)$이므로 finite Euclidean derivative가 없다.
그 경우 status는 `UNDEFINED_DISCRETE_EVENT`이고 pre/post guard receipt만 기록한다.

arbitrary linear chart는 componentwise tanh, axis-aligned clip, heterogeneous
threshold, absolute value와 TopK를 보존하지 않는다. A7-H의 유일한 covariance gate는
모든 neuron-indexed state, every delay slot, $W$, inputs, thresholds와 labels를 함께
옮기는 neuron permutation이다. exact TopK tie는 이 gate에서 제외한다.

## 7. frozen fixture와 tests

주 fixture는 $q=3,L=2$, permutation $P=(2,0,1)$과 다음 값을 쓴다.

```text
W = [[ .32,-.18, .07], [ .11, .27,-.21], [-.15, .09, .24]]
a = [ .22,-.31, .17]    r = [.06,.11,.04]
m = [-.08,.05,.12]      w = [.09,.13,.07]
u = [.41,.58,.36]       x = [.83,.71,.92]
d0 = [.55,-.24,.33]     d1 = [-.12,.44,.26]
q_prev = [1,0,1]        b = [0,1,0]       kappa=0 (slot k=0)
lifecycle = [ACTIVE,DORMANT,ACTIVE]
external = [.14,-.09,.21]
goal = [.08,.03,-.05]   replay = [-.04,.12,.06]
```

noise는 0, mode는 WAKE, STDP/F1/hippocampal encoding은 off다.
`active_ratio=2/3`, `max_axon_delay=2`, `dale_law=False`를 명시적으로
고정한다. 또한 `active_threshold=.22`, `force_all_active_selection=False`이고,
위 lifecycle code가 실제 `active_mask()`를 통해 `q_prev=[1,0,1]`을 만든다.
중앙차분은 `h=2^-17`, one-sided sequence는
`2^-12,2^-15,2^-18`로 동결한다.

H-B/H-C/H-F의 수학 미분·permutation 계산은 독립 NumPy mirror의 `float64`로
한다. 각 $+$/$-$ perturbation은 같은 immutable base state에서 새 복사본으로
평가하며, 앞선 호출이 바꾼 ring buffer/index를 재사용하지 않는다. H-A와 H-G의
runtime 비교만 Torch/Rust `float32`로 하며, 각 arm은 동일 snapshot을 독립
restore한다. 따라서 `h=2^-17`의 float64 미분 오차와 float32 backend 오차를
섞어 하나의 tolerance로 판정하지 않는다.

### H-A — exact Torch branch mirror

독립 NumPy 식 (H3)--(H6)과 `BrainRuntime._step_torch`의 one-tick continuous state,
bit, recurrent, ring contents/index를 비교한다. float32 경계에서 maximum absolute
error `<=2e-6`, bit/ring index exact를 요구한다.

### H-B — interior branch derivative

독립 float64 mirror에서 all continuous columns의 analytic $A_\sigma$와 central FD normalized Frobenius error
`<=2e-6`를 요구한다. delayed slot, $u,x,r,w$ perturbation을 반드시 포함한다.
주 fixture의 모든 raw clip margin은 `>=1e-4`여야 하며, 아니면
`INTERIOR_BRANCH_NOT_CERTIFIED`다.

### H-C — clamp faces

actual invariant domain에서 도달 가능한 face만 runtime-map 증거로 센다:

- $u$ raw lower face: $u=0,q^{\rm prev}=0$,
- $x$ raw upper/lower face: 각각 $x=1,q^{\rm prev}=0$ 및
  $x=1,u=1,q^{\rm prev}=1$,
- activation raw face: $c\in\{-1,1\}$에 대해
  $h_i^*(c)=\operatorname{artanh}([c-(1-\gamma_a)a_i]/\kappa_a)$를 풀어
  external coordinate를 고정,
- adaptation raw lower face: $w=0,a^+=0$.

$u$ upper와 adaptation upper face는 admissible $u,x\in[0,1]$, $w\in[0,2]$에서
actual constants로 정확히 도달하지 않으므로 full-runtime 증거로 세지 않는다.
대신 식 (H8)의 scalar primitive upper-face 규칙만 별도 domain-extension
diagnostic으로 시험한다. 각 actual face에서 full forward directional derivative와
세 one-sided FD의 최종 error가 `<=3e-6`이어야 하며 central FD는 보고하지 않는다.

### H-D — discrete guards

bit upper/lower crossing은 H-C의 $h_i^*(c)$ 구성에서
$c\in\{\theta^-,\theta^+\}$로 바꾸고 external 방향 $\pm e_i$를 쓴다.
selection은 actual operator에 직접
$s=(\vartheta-\delta,\vartheta,\vartheta+2\delta)$,
$\delta=2^{-8}$, $K_M=2$를 넣어 eligibility 양쪽 mask를 기록한다. kth boundary는
$s=(.70,.50+\epsilon,.50-\epsilon)$의 두 방향을 비교한다. exact
$s=(.70,.50,.50)$ tie는 반환 index를 receipt에는 남기되 backend identity parity와
미분 판정에서는 `TIE_POLICY_UNSPECIFIED`로 fail-close한다. lifecycle idle guard는
inactive count 0, inactive mask에서 $s_i=.08\pm\epsilon$을 직접 시험한다. 어느
discrete label/mask든 바뀌면 derivative를 반환하지 않는다.

### H-E — ring delay와 lifecycle lag

Torch에서 `max_axon_delay=L=2`, zero ring, $k_0=0$, nonzero $a_0$, 고정 ACTIVE
lifecycle로 시작한다. zero-based 호출 표는 정확히 다음과 같아야 한다.

| call $t$ | read | write | post index | recurrent norm |
|---:|---|---|---:|---:|
| 0 | $d_0^{(0)}=0$ | $d_1^{(0)}=a_0$ | 1 | `<=1e-7` |
| 1 | $d_1^{(1)}=0$ | $d_2^{(1)}=a_1$ | 2 | `<=1e-7` |
| 2 | $d_2^{(0)}=a_0$ | $d_3^{(0)}=a_2$ | 3 | `>=1e-4` |

즉 첫 nonzero arrival은 “$L$번째 tick”이라는 모호한 표현이 아니라 세 번째
호출, zero-based $t=L$에서 나타난다. index와 overwritten slot을 매 호출 기록한다.
post-selection lifecycle mask는 다음 tick의 STP/recurrent source를 바꾸며 같은 tick의
이미 계산된 activation을 다시 쓰지 않아야 한다. counterfactual masks를 같은
post-cell continuous snapshot에 적용한 두 arm은 같은-tick activation error
`<=1e-7`, 다음-tick recurrent difference norm `>=1e-4`를 만족해야 한다.

### H-F — permutation

no-tie interior의 float64 mirror에서 continuous output과 branch Jacobian의 neuron
permutation residual을 `<=1e-10`으로 둔다. 별도 actual Torch float32 arm은
continuous residual `<=2e-6`과 bit/mask exact만 요구한다. 모든 vector/threshold,
delay slot과 $W\mapsto PWP^\top$을 함께 옮기며 `dale_law=False`다. arbitrary dense
chart는 시험하지 않는다. 주 fixture의 salience pairwise gap과 `.22` eligibility
margin은 각각 `>=1e-4`여야 한다.

### H-G — backend parity boundary

- `axon_delay=False`, no-tie exact state: Torch/Rust continuous state absolute error
  `<=1e-5`, bit exact.
- `axon_delay=True`, nonzero ring slot distinct from current activation: **expected
  mismatch**. Torch index는 1로 증가하고 slot을 overwrite하지만 Rust index는 0,
  buffer는 그대로여야 한다. activation error `>=1e-4`여야
  `DELAY_PARITY_FAIL_EXPECTED`가 성립한다.

implementation receipt는 import 직후 실제 `_HAS_RUST_KERNEL`, extension module path와
backend 선택 결과를 기록한다. unavailable이면 no-delay parity는
`BLOCKED_KERNEL_UNAVAILABLE`이고 delay source-signature audit만 보고한다. 이번
freeze에서 관찰한 값은 `_HAS_RUST_KERNEL=True`지만 witness가 이를 다시 확인하지
않으면 H-G는 판정하지 않는다.

## 8. 판정과 재귀 규칙

H-A--H-F가 통과하고 H-G의 no-delay PASS/delay expected-FAIL이 함께 재현될 때
`DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED`다.

- branch equation/index mismatch: `P0_FORMULA`; 수식만 한 번 고치고 같은 fixture,
  step, tolerance와 guards를 재실행한다.
- actual Torch mirror mismatch caused by wrong code snapshot/config: `P2_APPARATUS`;
  source/config receipt만 고친다.
- delay parity mismatch는 사전 등록한 blocker이며 threshold tuning으로 없애지 않는다.
  Rust에 ring buffer/index와 snapshot semantics를 구현한 별도 code change 뒤에만 같은
  gate를 다시 연다.
- discrete crossing에 작은 derivative를 억지로 반환하거나 continuous saltation을
  넣으면 즉시 `STOP_FORMULA`다.
- one formula revision 뒤 재실패하면 이 branch를 `STOP_HYBRID_SPEC`로 닫는다.

성공해도 actual runtime의 learning/memory/mode 전체 Jacobian, heterogeneous threshold,
Dale biology, AGI 또는 anatomy는 검증되지 않는다.
