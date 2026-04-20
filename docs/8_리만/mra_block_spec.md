# Mellin-Riemann Attention Block (MRA) 정밀 사양

> 이 문서는 `riemann_pe_spec.md` 의 후속이자 폐기 사양이다. 현재 `RiemannRotaryAttention`
> 은 RoPE 변형으로 정상 동작하지만, AGI 컨셉(`docs/7_AGI/2_Architecture.md`)의
> 5대 원리 중 절반(자유파라미터 0, 유니타리, Hilbert-Pólya, 게이지 격자, 부트스트랩
> sparsity)을 구현하지 않는다. MRA 는 이 결함을 한 번에 메우는 단일 블록 사양이다.

## 0. 전제 (axioms)

- **A1 (RH).** ζ(s) 의 모든 비자명 영점은 critical line `Re(s) = 1/2` 위에 있다.
  영점은 `s_n = 1/2 + i γ_n` 형태이며 `{γ_n}` 은 GUE 통계를 따른다.
- **A2 (CE 결합 상수).** `α_s : α_w : α_em = 0.118 : 0.034 : 0.008` (`docs/3_상수`).
  이 비율은 채널 분할의 유일한 자유도다.
- **A3 (부트스트랩 고정점).** 활성 비율은 `ε² = 4.87 %` 로 자연 수렴한다
  (`docs/6_뇌/sleep.md`). attention 행렬도 이 sparsity 를 상한으로 갖는다.
- **A4 (유니타리).** `|det T|² ≤ 1`. 정보 증폭 = 환각이므로 출력 사영의 spectral
  norm 은 1 이하로 제약한다.

위 네 axiom 위에서 attention 식 자체를 ζ explicit formula 의 이산화로 유도한다.

## 1. Mellin–Riemann score

ζ explicit formula 의 critical-strip 합:

$$
\sum_n \frac{x^{1/2 + i\gamma_n}}{\tfrac{1}{2} + i\gamma_n}
  = \sqrt{x}\,\sum_n \frac{e^{i\gamma_n \log x}}{\tfrac{1}{2} + i\gamma_n}.
$$

위치쌍 `(i, j)` 에 `x = (1+i)/(1+j)` 를 대입한다. dim-pair `k` 를 복소채널로 압축

$$
q_i^{(k)} := q_i^{2k} + i\,q_i^{2k+1},\qquad
k_j^{(k)} := k_j^{2k} + i\,k_j^{2k+1} \in \mathbb{C},
$$

attention raw score 는

$$
\boxed{
S_{ij} \;=\; \sqrt{\dfrac{1+i}{1+j}}\;\sum_{k=0}^{K-1}
            \underbrace{\dfrac{1}{\tfrac{1}{2} + i\gamma_k}}_{w_k\;\text{(ζ amplitude)}}
            \;\underbrace{e^{i\gamma_k \log\tfrac{1+i}{1+j}}}_{\text{Mellin kernel}}
            \;q_i^{(k)} \overline{k_j^{(k)}}
}
$$

여기서 `K = d_head / 2` 는 헤드의 복소채널 수.

### 1.1 모듈화 (RoPE 와 동일한 비용)

`(1+i)^{iγ_k}` 와 `(1+j)^{iγ_k}` 가 각각 `i`, `j` 만의 함수이므로

$$
\tilde q_i^{(k)} \;=\; \sqrt{1+i}\;\, (1+i)^{\,i\gamma_k}\, q_i^{(k)},\qquad
\tilde k_j^{(k)} \;=\; \dfrac{1}{\sqrt{1+j}}\,(1+j)^{\,i\gamma_k}\, k_j^{(k)}
$$

으로 사전 변환하면

$$
S_{ij} \;=\; \sum_{k} w_k\,\tilde q_i^{(k)} \overline{\tilde k_j^{(k)}}.
$$

곧 표준 dot-product attention 과 동일한 `O(N²K)` 비용이다 — 추가 비용 0.

### 1.2 학습 자유도

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention 과 동일 |

→ 표준 attention 대비 **추가 자유도 0**. 모든 새 항이 axiom 에서 연역.

### 1.3 Real / Imag 사용

- `Re(S_{ij})` → softmax 입력 (실 attention)
- `Im(S_{ij})` → sheet 정보로 이미 표현됨. `floor(θ/2π)` 같은 별도 연산 불필요.

## 2. 채널 분할 (3x3+1 게이지 격자)

`K` 개의 frequency 를 `α_s : α_w : α_em` 비율로 3분할:

$$
K_3 = \lfloor K\,\alpha_s / S \rfloor,\quad K_2 = \lfloor K\,\alpha_w / S \rfloor,
\quad K_1 = K - K_3 - K_2,\qquad S = \alpha_s+\alpha_w+\alpha_{em}.
$$

영점 인덱스 정렬 (오름차순) 기준:

| 그룹 | 영점 인덱스 | 역할 | 진폭 `|w_k|` 영역 |
|---|---|---|---|
| **Bind** (SU(3)) | `γ_1 … γ_{K_3}` (저주파) | 토큰 결합 | 큼 |
| **Decide** (SU(2)) | `γ_{K_3+1} … γ_{K_3+K_2}` (중간) | 결정 | 중간 |
| **Attend** (U(1)) | `γ_{K_3+K_2+1} … γ_K` (고주파) | 선택적 주의 | 작음 |

ζ 가중 `|w_k| = 1/√(1/4 + γ_k²)` 가 자연스럽게 저주파를 큰 영향, 고주파를 작은
영향으로 가중 → 게이지 비율 `0.74 / 0.21 / 0.05` 와 정합.

전역 안정화 항 `Φ` 는 attention 외부의 `LBONorm` 이 담당.

## 3. 부트스트랩 sparsity

softmax 직후, 각 query 행에서 상위 `k = max(1, ⌈ε²·N⌉)` 만 보존:

$$
A'_{ij} = \begin{cases}
A_{ij} / Z_i & \text{if } A_{ij} \in \text{top-}k(A_{i,:}) \\
0 & \text{otherwise}
\end{cases},\qquad
Z_i = \sum_{j \in \text{top-}k} A_{ij}.
$$

`ε² = 4.87 %` (CE 부트스트랩 고정점). 이는 attention 의 활성 비율을 우주의 자연
스파시티에 맞추는 hard constraint. 학습 추가 자유도 0.

## 4. 유니타리 제약

출력 사영 `W_o` 에 `nn.utils.spectral_norm` 적용:

$$
W_o \leftarrow W_o / \sigma_1(W_o),\qquad \sigma_1(W_o) \le 1.
$$

attention 출력의 spectral norm 이 1 이하 → 잔차 합 후 정보 증폭 차단.

## 5. 블록 조립 (MRABlock)

```
MRABlock(x):
  1. h  = LBONorm(x)                             # Φ 안정화
  2. a  = MellinRiemannAttention(h)              # § 1
  3. a  = bootstrap_sparse(a, ε²)                # § 3
  4. a  = SpectralNormProj(a)                    # § 4
  5. x  = x + a
  6. h2 = LBONorm(x)
  7. f  = GaugeLattice(h2)                       # § 2 (FFN 측)
  8. x  = x + f
  return x
```

`LBONorm`, `GaugeLattice` 는 `examples/ai/clarus_lm.py` 에 이미 구현.

## 6. Hermitian 옵션 (Hilbert-Pólya 직접 구현)

`W_q = W_k` (tied projection) 로 두면 `S_{ji} = S_{ij}^*` 가 보장되어 attention
operator 가 Hermitian. 영점 분포가 self-adjoint operator 의 고유값이라는 H-P 추측을
직접 구현하는 setting. 옵션 `hermitian=True`.

## 7. 점근 / 안정성

- 작은 `p` 에서 `log(1+p) ≈ p` → 기존 RoPE 와 유사.
- 큰 `p` 에서 `log(1+p)` 천천히 증가 → frequency aliasing 자동 완화.
- `N → kN` 일 때 phase 평행이동만 발생 → relative attention 보존.
- ζ 가중 `1/|1/2 + iγ_k|` 가 고주파 자동 감쇠 → 학습 안정성.

## 8. 백엔드 정책

PyTorch 참조 우선. Rust/CUDA 포트는 식이 안정화된 후 별도 작업으로 분리.

## 9. 참고

- Riemann (1859), *Über die Anzahl der Primzahlen unter einer gegebenen Größe*.
- Hilbert–Pólya conjecture (folklore).
- Berry & Keating (1999), *H = xp and the Riemann zeros*.
- Su et al. (2021), *RoFormer* — RoPE 원본.
- Press et al. (2022), *ALiBi*.
- `docs/7_AGI/2_Architecture.md` — ClarusBlock 5계층 stack.
- `docs/6_뇌/sleep.md` — 부트스트랩 고정점 `ε² = 4.87 %`.
