# Riemann Surface Positional Encoding 정밀 사양

## 0. 전제

리만 가설(Riemann Hypothesis)은 공학적 axiom 으로 채택한다:

> ζ(s) 의 모든 비자명 영점은 critical line Re(s) = 1/2 위에 있다.

따라서 영점은 s_n = 1/2 + i γ_n 의 형태이며, {γ_n} 은 Montgomery-Dyson 추측에 의해
GUE(Gaussian Unitary Ensemble) 통계를 따르는 "무작위인 동시에 구조적인" 수열이다.
처음 100개의 γ_n 은 Titchmarsh / Odlyzko 표에서 가져와 `RIEMANN_ZEROS_IM` 에 하드코딩되어 있다.
n > 100 은 Riemann–von Mangoldt 점근식 γ_n ≈ 2π n / log n 으로 외삽한다.

본 사양은 이 axiom 위에서 attention 의 positional encoding 을
**Riemann surface (multi-sheet 복소 평면)** 위의 회전으로 재구성한다.

## 1. 동기 — 왜 평면(surface)이 필요한가

기존 `RiemannRotaryAttention` 은 RoPE 와 동일한 prescription 을 사용한다:

    inv_freq_k = 1 / γ_k
    θ(p, k)    = p · inv_freq_k

이는 평면(circle) 위의 회전이며, 다음 두 가지 한계가 있다:

1. **단일 시트(single-sheet)**: θ 가 2π 를 넘어가면 정보가 wrap-around 로 사라진다.
   같은 phase 인 두 위치를 attention 이 구분할 수 없다.
2. **선형 시간 lift**: 위치 p 가 선형으로 들어가므로,
   sequence length 가 N → kN 으로 늘어나면 phase 도 k배 늘어나
   학습된 frequency 분포가 깨진다 (RoPE 의 long-context 문제와 동일).

Riemann surface 는 이 두 문제를 동시에 해결한다:

- **Multi-sheet**: log z 는 단일값이 아니라 z = r e^{iθ} 위에서 무한 시트를 갖는다.
  sheet index 를 명시적으로 유지하면 phase 가 wrap 되어도 정보가 보존된다.
- **Logarithmic lift**: 자연 좌표 τ = log(1 + p) 는 multiplicative scale 에 대해
  invariant (kp ↦ τ + log k). Sequence length 의 power-law 변화에 안정적이다.

## 2. 사양

### 2.1 좌표 lift

위치 p ∈ {0, 1, …, N-1} 를 critical line 의 imaginary axis 로 들어올린다:

$$
\tau_p = \log(1 + p), \qquad s_p = \tfrac{1}{2} + i\,\tau_p \in \mathbb{C}.
$$

`+1` 은 p = 0 에서 log 발산을 막기 위한 standard offset 이다.

### 2.2 회전 generator

각 헤드의 dim-pair k (k = 0, …, d_head/2 - 1) 는 γ_k 를 frequency 로 가지며,
회전각은

$$
\theta(p, k) = \gamma_k \cdot \tau_p = \gamma_k \log(1 + p).
$$

대응하는 단위 복소수는

$$
e^{i\theta(p,k)} = (1+p)^{i\gamma_k}.
$$

이는 Mellin 변환 커널 (1+p)^{i γ_k} 와 정확히 일치한다 — Riemann ζ 함수 자체가
이 형태의 합으로 정의되므로 자연스러운 선택이다.

### 2.3 Sheet index

회전은 모듈로 2π 이지만, 시트(sheet) 정보는 별도로 보존한다:

$$
\sigma(p, k) = \left\lfloor \frac{\theta(p, k)}{2\pi} \right\rfloor.
$$

두 위치 i, j 가 같은 phase (cos/sin 동일) 라도 서로 다른 시트에 있으면
Riemann surface 위에서는 다른 점이다. Attention 은 sheet 차이를 바이어스로 받는다:

$$
b^{\text{sheet}}_{ij} = -\lambda_\sigma \cdot \frac{1}{d_{\text{head}}/2}
                       \sum_{k=0}^{d_{\text{head}}/2-1} |\sigma(i, k) - \sigma(j, k)|,
$$

여기서 λ_σ 는 학습 가능한 per-head 스칼라.
이 항은 cross-sheet attention 을 약화시켜 시트 식별을 강제한다.

### 2.4 회전 적용 (RoPE-style relative form)

RoPE 와 동일하게, dim-pair (2k, 2k+1) 에 대해 2D 회전을 적용한다:

$$
\begin{pmatrix} q'_{2k} \\ q'_{2k+1} \end{pmatrix} =
\begin{pmatrix} \cos\theta(p,k) & -\sin\theta(p,k) \\
                \sin\theta(p,k) &  \cos\theta(p,k) \end{pmatrix}
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
$$

그러면 q_i^T k_j 는 자동으로 Δθ = θ(i,k) - θ(j,k) = γ_k log((1+i)/(1+j)) 의
함수가 된다 — translation invariance 가 유지되고, Hilbert-Pólya 정신에서
Hermitian kernel 이 보장된다.

### 2.5 최종 attention score

$$
\text{score}_{ij} = \frac{q_i^{\prime\top} k_j^{\prime}}{\sqrt{d_{\text{head}}}}
                  + b^{\text{sheet}}_{ij},
$$

이후 causal mask 와 softmax 적용.

## 3. 학습 가능 파라미터

| 이름            | 형상           | 역할                                                                   |
|-----------------|----------------|------------------------------------------------------------------------|
| `log_scale`     | (n_heads,)     | 헤드별 "speed of light" — 모든 γ_k 에 곱해지는 exp(s)                  |
| `log_lambda_sigma` | (n_heads,) | sheet-difference penalty 의 log-scale (λ_σ = exp(·))                  |

이 외 파라미터(γ_k, frequency 자체)는 모두 buffer (학습 안 함). RH 의 axiom 적 성격을 유지한다.

## 4. 점근적 성질

- 작은 p 에서는 τ_p ≈ p (log(1+p) ≈ p), 기존 RoPE 와 유사.
- 큰 p 에서는 τ_p 가 천천히 증가 → frequency aliasing 자동 완화.
- N → kN 일 때 τ 는 log k 만큼만 평행이동 → relative attention 이 거의 동일하게 보존됨.

## 5. 백엔드 dispatch

세 단계 backend 모두에서 동일한 수치 결과를 보장한다:

1. **PyTorch** (참조): `clarus.ce_riemann_attn.RiemannRotaryAttention`
2. **Rust CPU**: `clarus._rust.nn_ce_riemann_fwd`
3. **CUDA**: `clarus._rust.nn_ce_riemann_fwd_cuda` (cudarc launcher + `.cu` kernel)

자동 선택은 `clarus.ce_riemann_attn.RiemannRotaryAttention(backend="auto")` 가
입력 텐서의 `device.type` 으로 결정한다 (cuda → cuda, cpu → rust, fallback → torch).

## 6. 수치 동일성 테스트

`tests/test_riemann_pe_consistency.py` 에서

- 동일 입력에 대해 세 backend 의 출력이 atol=1e-4, rtol=1e-3 이내로 일치하는지
- backward grad 가 PyTorch 와 1e-3 이내로 일치하는지

검증한다.

## 7. 참고

- Titchmarsh, *The Theory of the Riemann Zeta-Function*, Appendix.
- Odlyzko, *On the distribution of spacings between zeros of the zeta function*.
- Montgomery (1973), pair-correlation conjecture.
- Su & Lu, *RoFormer*, 2021 — RoPE 원본.
