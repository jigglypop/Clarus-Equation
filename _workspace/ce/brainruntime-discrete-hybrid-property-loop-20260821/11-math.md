# Mathematics

Status: COMPLETE

## 1. Preflight verdict

초기 A7-H route의 “hard threshold에는 saltation matrix”라는 문장은 `REVISE`다.
BrainRuntime은 연속 flow가 guard를 통과하는 시간이 아니라 각 tick 끝에서 guard를
평가하는 discrete hybrid map이다. 따라서 continuous saltation의 denominator
`n^T f^-`, reset timing과 vector fields가 정의되지 않는다. 올바른 객체는 다음 셋이다.

1. fixed discrete mode의 branch Jacobian,
2. continuous clip face의 one-sided/Bouligand directional derivative,
3. bitㆍTopKㆍlifecycle transition의 discrete event receipt.

## 2. Exact state and ordering

continuous state는 activation $a$, refractory $r$, memory $m$, adaptation $w$, STP
$u,x$와 $L$개 delay slot을 모두 포함한다. 이전 tick의 lifecycle가 만든 mask
$q^{prev}$는 current tick의 STP와 recurrent source를 gate한다. 현재 activation은
ring slot을 읽은 뒤 그 slot에 쓰며, 새 activation $a^+$가 아니라 old $a$가 저장된다.
주 fixture의 $q^{prev}=(1,0,1)$은 독립 입력이 아니라 실제 lifecycle
`(ACTIVE,DORMANT,ACTIVE)`의 `active_mask()` 산출이다.
호출 직전 상태를 $t$라 하면

$$
y_t=d_t^{(k_t)},\qquad k_t=\kappa_t\bmod L,\qquad
d_{t+1}^{(k_t)}=a_t,\qquad \kappa_{t+1}=\kappa_t+1
$$

이고 나머지 slot은 그대로다. $\kappa$는 실제 runtime처럼 누적되고 slot만
modulo로 정한다. 그러므로 zero ring과 $L=2$에서 첫 두 호출은
zero를 읽고, old $a_0$의 첫 도착은 세 번째 호출인 zero-based $t=2$다.

Torch 순서는

```text
previous lifecycle mask -> STP u/x -> delayed presynaptic output -> W @ output
-> drive -> activation -> refractory/memory/adaptation -> bit -> salience
-> outer TopK -> STDP -> lifecycle -> next tick
```

다. 이 순서를 바꾸면 같은 기호를 써도 다른 map이다.

## 3. Interior branch Jacobian

clip이 모두 strict interior이고 discrete state가 고정됐다고 하자. old $u$를 쓰는
STP blocks는

$$
D_u\tilde u=\operatorname{Diag}(1-\alpha-Uq^{prev}),
$$

$$
D_x\tilde x=\operatorname{Diag}(1-\beta-u\odot q^{prev}),
\qquad
D_u\tilde x=-\operatorname{Diag}(x\odot q^{prev}).
$$

delay read $d^{(k)}$에 대해 recurrent tangent는 product rule로

$$
\delta\rho=W\left[q^{prev}\odot
(\delta\tilde u\odot\tilde x\odot d^{(k)}
+\tilde u\odot\delta\tilde x\odot d^{(k)}
+\tilde u\odot\tilde x\odot\delta d^{(k)})\right].
$$

이어

$$
\delta a^+=\operatorname{Diag}(\kappa_a\operatorname{sech}^2h)\delta h
+\operatorname{Diag}(1-\gamma_a)\delta a
$$

이고

$$
\delta r^+=(1-\gamma_r)\delta r
+2\kappa_r\operatorname{Diag}(a^+)\delta a^+,
$$

$$
\delta m^+=(1-\gamma_m)\delta m+\gamma_m\delta a^+,
\qquad
\delta w^+=(1-\gamma_w)\delta w
+2\gamma_w\operatorname{Diag}(a^+)\delta a^+.
$$

drive에는 `-refractory_scale delta r - adaptation_coupling delta w`도 들어간다.
ring derivative는 current slot을 old `delta a`로 overwrite하고 다른 slot을
보존한다. 이것을 모두 결합한 $A_sigma$가 fixed-branch Jacobian이다.

이 미분 시험은 immutable base state를 매 perturbation마다 복사하는 float64 NumPy
mirror에서 수행한다. 실제 Torch/Rust state는 한 호출마다 ring을 mutate하므로
snapshot/restore 없이 중앙차분을 만들면 서로 다른 map을 빼게 된다. float32 runtime은
mirror 식 자체와 backend parity를 확인하는 별도 gate다.

## 4. Clip face와 discrete event

clip face에서는 direction $v$의 부호가 다음 branch를 정하므로 하나의 선형
Jacobian이 일반적으로 존재하지 않는다. lower face에서는 inward positive direction만,
upper face에서는 inward negative direction만 통과한다. full directional derivative는
STP부터 adaptation까지 이 선택을 순서대로 전파해야 한다.

actual invariant domain에서는 $u$ upper와 adaptation upper raw face가 현재 상수로
도달 불가능하다. 따라서 runtime-map face gate는 $u$ lower, $x$ lower/upper,
activation raw $-1/+1$, adaptation lower만 센다. 도달 불가능한 두 upper face는 scalar
clip primitive의 domain-extension 진단일 뿐 runtime 증거가 아니다.

반면 bit 또는 mask가 바뀌면 encoded output difference는 유한한 jump다. 예를 들어
TopK tie 양쪽에서 mask가 서로 다르면

$$
\frac{\|q(z+\epsilon v)-q(z)\|}{\epsilon}\to\infty.
$$

따라서 `UNDEFINED_DISCRETE_EVENT`를 기록하며 작은 행렬을 반환하지 않는다.

actual selection은 neuronwise threshold가 아니라 하나의 global
$\vartheta=\texttt{active_threshold}$와 scalar budget $K_M$을 쓴다. 즉

$$
E=\{i:s_i\ge\vartheta\},\quad K'=\min(K_M,|E|),\quad
q_i^+=\mathbf1\{i\in\operatorname{TopK}_{K'}(s_j:j\in E)\}.
$$

이번 fixture는 WAKE, `active_ratio=2/3`, `dim=3`, F1 off이므로 $K_M=2$다.
eligibility scalar는 `active_threshold=.22`, selection override는 false로 고정한다.
heterogeneous active threshold는 수학 envelope일 뿐 actual runtime 식으로 시험하지 않는다.

## 5. Symmetry boundary

arbitrary dense rechart는 componentwise nonlinearity와 threshold axes를 바꾸므로 A6의
coordinate covariance를 그대로 요구할 수 없다. neuron permutation $P$만 모든
state block, delay slot, input, threshold, label과 $W->PWP^T$에 동시에 적용했을 때
map이 equivariant하다. TopK exact tie는 tie policy가 canonical하지 않아 제외한다.
float64 mirror의 residual은 $10^{-10}$, actual float32 Torch arm은 $2\times10^{-6}$과
exact bit/mask로 서로 분리해 판정하며 `dale_law=False`를 고정한다.

## 6. Backend counterexample

Torch delay branch는 `runtime.py`에서 ring slot을 읽고 old activation을 write하며
index를 증가시킨다. Rust `brain_step` signature에는 delay buffer/index가 없고
`u*x*a*q_prev`를 바로 recurrent source로 쓴다. 따라서 nonzero ring slot이 current
activation과 다르면 두 backend는 같은 map일 수 없다.

기존 parity test가 통과한 이유는 `axon_delay=False`를 명시했기 때문이다. 올바른
판정은 no-delay/no-tie parity를 좁게 유지하고, delay-on parity는 예상된 blocker로
fail-close하는 것이다.

## 7. Threshold and biology boundary

수학 envelope는 neuronwise threshold vector를 쓸 수 있지만 current
`BrainRuntimeConfig`의 bit lower/upper와 active threshold는 global scalars다. 따라서
edge strength heterogeneity는 구현됐지만 threshold heterogeneity는 actual runtime에서
아직 구현되지 않았다. 또한 이번 run은 `dale_law=False`라 sender/receiver sign
orientation을 판단하지 않는다.

## 8. Formal status

- fixed-branch map과 derivative: **[정의 + 조건부 산출]**.
- clip directional rule: **[조건부 산출]**.
- discrete event derivative nonexistence: **[반례로 확인 가능한 경계]**.
- delayed Torch/Rust parity: **[구현 차단]**.
- actual neural mechanism, learning, AGI, anatomy: **[미완성]**.

Revision 2 구현 전 판정은 `READY_FOR_REAUDIT`이다. saltation 부모 문장은 폐기했고,
ring clockㆍglobal selectionㆍdtype/state resetㆍreachable-face 정의를 witness 실행 전에
고정했다. 독립 재감사 PASS 뒤에만 같은 frozen fixture를 시험한다.
