# Mellin-Riemann Attention Block (MRA) 정밀 사양

이 문서는 Mellin--Riemann Attention block의 입력 tensor, score, 채널 분할, 제약과 backend contract를 정의한다. 독자는 attention·복소수·정규화의 기본을 아는 독자를 전제로 하며, ζ explicit formula와 Hilbert--Pólya 비유는 구현 score의 동기이지 리만가설의 증명·수치 검증이 아니다.

전제와 score 정의 뒤에 모듈화·자유도·채널·sparsity·unitary·assembly·Hermitian·backend 순으로 읽는다. shape·precision·fixture·정규화가 정의역이며, 안정성·등가성·성능은 baseline·ablation·반례가 없으면 명세 또는 미완성이다.


> 이 문서는 `riemann_pe_spec.md`의 후속이자 폐기 사양이다. 현재 `RiemannRotaryAttention`
> 은 RoPE 변형으로 정상 동작하지만, AGI 컨셉(`paper/7_AGI/2_Architecture.md`)의
> 5대 원리 중 절반(자유파라미터 0, 유니타리, Hilbert-Pólya, 게이지 격자, 부트스트랩
> sparsity)을 구현하지 않는다. MRA는 이 결함을 한 번에 메우는 단일 블록 사양이다.

## 0. 전제 (axioms)

전제는 input/output shape, complex convention, precision, normalization을 고정한다. 이 선택은 구현 axiom이며, 수학적 정리·관측 결과·리만가설 관련 주장을 자동으로 제공하지 않는다.

- **A1 (주파수 입력).** 유한한 실수 목록 `γ_n`을 사용한다. 모든 제타 영점의
  critical-line 위치(RH)와 간격의 GUE 통계는 별도 가설이며 아래 유한합에는 필요하지 않다.
- **A2 (CE 결합 상수).** `α_s : α_w : α_em = 0.118 : 0.034 : 0.008` (`paper/3_상수`).
  이 숫자는 채널 분할에 채택한 설계 입력이며, 표준 상호작용의 유도나 유일한 비율이라는 정리는 아니다.
- **A3 (sparsity 입력).** `ε² = 4.87 %`를 후보 retention 비율로 택한다.
  보편적인 자연 수렴이나 모든 attention 행렬의 상한은 이 사양에서 증명되지 않았다.
- **A4 (출력 사영 수축).** 출력 사영의 spectral norm을 1 이하로 제약하는 설계다.
  `|det T|² ≤ 1`만으로는 수축도 unitary도 보장되지 않는다.
  norm과 환각 사이의 동등성은 이 수학에서 유도되지 않는다.

이 입력을 사용해 아래 attention score를 정의한다. 네 조건만으로 score가
유일하게 유도되지는 않는다. 정리와 반례는 [수학 판정](math_claims_audit.md)을 따른다.

## 1. Mellin–Riemann score

score는 query/key와 position input을 받아 attention logit을 출력하는 정의역 제한 연산자다. scale·branch·precision이 가정이며, score의 형태가 length generalization 또는 영점 구조의 empirical evidence를 보장하지 않는다.

제타 모드에서 동기를 얻어 유한 Mellin형 합 `x^{-s}`를 정의한다. 이 항의
factorization은 정확하지만 모든 영점·다른 항·합 규약을 포함한 explicit formula
자체와 동일하지 않다. 부호 규약은 `mra_paper.md` §3을 따른다.

$$
\sum_n \frac{x^{-(1/2 + i\gamma_n)}}{\tfrac{1}{2} + i\gamma_n}
  = \frac{1}{\sqrt{x}}\,\sum_n \frac{e^{-i\gamma_n \log x}}{\tfrac{1}{2} + i\gamma_n}.
$$

위치쌍 `(i, j)`에 `x = (1+i)/(1+j)`를 대입한다. dim-pair `k`는 복소채널로 압축한다.

$$
q_i^{(k)} := q_i^{2k} + i\,q_i^{2k+1},\qquad
k_j^{(k)} := k_j^{2k} + i\,k_j^{2k+1} \in \mathbb{C},
$$

attention raw score는 다음과 같다.

$$
\boxed{
S_{ij} \;=\; \sqrt{\dfrac{1+j}{1+i}}\;\sum_{k=0}^{K-1}
            \underbrace{\dfrac{1}{\tfrac{1}{2} + i\gamma_k}}_{w_k\;\text{(ζ amplitude)}}
            \;\underbrace{e^{-i\gamma_k \log\tfrac{1+i}{1+j}}}_{\text{Mellin kernel}}
            \;q_i^{(k)} \overline{k_j^{(k)}}
}
$$

여기서 `K = d_head / 2`는 헤드의 복소채널 수다.

### 1.1 모듈화와 계산 차수

모듈화는 기존 RoPE interface에서 MRA score를 호출할 수 있게 하는 구현 contract다. 동일 비용은 지정 shape·backend·precision의 추정 또는 측정이며, 다른 kernel·sequence OOD에서 성립하지 않을 수 있다.

`(1+i)^{-iγ_k}`와 `(1+j)^{-iγ_k}`가 각각 `i`, `j`만의 함수이므로

$$
\tilde q_i^{(k)} \;=\; \dfrac{1}{\sqrt{1+i}}\;(1+i)^{-i\gamma_k}\, q_i^{(k)},\qquad
\tilde k_j^{(k)} \;=\; \sqrt{1+j}\;(1+j)^{-i\gamma_k}\, k_j^{(k)}
$$

으로 사전 변환하면

$$
S_{ij} \;=\; \sum_{k} w_k\,\tilde q_i^{(k)} \overline{\tilde k_j^{(k)}}.
$$

주요 합의 점근 차수는 표준 dot-product attention과 같은 `O(N²K)`다.
주파수 변환·복소 가중치·실수부 계산의 상수 비용이 있어 실제 비용 동일성은 따르지 않는다.

### 1.2 학습 자유도

학습 자유도는 optimizer가 조절할 parameter와 고정 수식 항을 구분한다. free parameter의 존재는 ζ 수학의 추정·리만가설 검증이 아니며, ablation·seed·baseline에서만 효능을 판정한다.

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention과 동일 |

이 설정의 새 계수는 학습하지 않는 고정 입력이다. 고정된 설계 선택은 남으며,
그 값이나 score의 유일성이 네 axiom에서 연역됐다는 뜻은 아니다.

### 1.3 Real / Imag 사용

실수·허수 사용은 complex tensor를 real-valued backend로 표현하는 shape·precision 규약이다. 수치 안정성·gradient parity는 fixture와 tolerance에서 확인하며 물리적 복소 상태를 뜻하지 않는다.

- `Re(S_{ij})` → softmax 입력 (실 attention)
- `Im(S_{ij})` → sheet 정보로 이미 표현됨. `floor(θ/2π)` 같은 별도 연산 불필요.

## 2. 채널 분할 (3x3+1 게이지 격자)

채널 분할은 hidden dimension을 지정 비율의 subspace로 나누는 implementation choice다. 게이지 비유는 block 역할 설명에 한정되며, 비율의 수학적·물리적 필연성이나 성능 우위는 ablation이 필요하다.

`K`개의 frequency를 `α_s : α_w : α_em` 비율로 3분할한다.

$$
K_3 = \lfloor K\,\alpha_s / S \rfloor,\quad K_2 = \lfloor K\,\alpha_w / S \rfloor,
\quad K_1 = K - K_3 - K_2,\qquad S = \alpha_s+\alpha_w+\alpha_{em}.
$$

영점 인덱스는 오름차순 정렬을 기준으로 삼는다.

| 그룹 | 영점 인덱스 | 역할 | 진폭 `|w_k|` 영역 |
|---|---|---|---|
| **Bind** (SU(3)) | `γ_1 … γ_{K_3}` (저주파) | 토큰 결합 | 큼 |
| **Decide** (SU(2)) | `γ_{K_3+1} … γ_{K_3+K_2}` (중간) | 결정 | 중간 |
| **Attend** (U(1)) | `γ_{K_3+K_2+1} … γ_K` (고주파) | 선택적 주의 | 작음 |

ζ 가중 `|w_k| = 1/√(1/4 + γ_k²)`는 저주파에 큰 영향, 고주파에 작은 영향을 자연스럽게 부여한다.
이는 게이지 비율 `0.74 / 0.21 / 0.05`와 정합한다.

전역 안정화 항 `Φ`는 attention 외부의 `LBONorm`이 담당한다.

## 3. 부트스트랩 sparsity

sparsity는 activation 또는 score mask의 shape·분모·threshold를 정하는 정책이다. 희소율은 무차원 설정이며, latency·memory·PPL 이득은 dense baseline과 input-length OOD에서 실패할 수 있다.

softmax 직후, 각 query 행에서 상위 `k = max(1, ⌈ε²·N⌉)`만 보존한다.

$$
A'_{ij} = \begin{cases}
A_{ij} / Z_i & \text{if } A_{ij} \in \text{top-}k(A_{i,:}) \\
0 & \text{otherwise}
\end{cases},\qquad
Z_i = \sum_{j \in \text{top-}k} A_{ij}.
$$

`ε² = 4.87 %`는 CE 부트스트랩 고정점이다. 이는 attention의 활성 비율을 우주의 자연
스파시티에 맞추는 hard constraint이며, 추가 학습 자유도는 없다.

## 4. 출력 사영의 norm 제약

출력 사영의 norm 제약은 해당 선형 사상의 수축만 제한한다. Unitary성과
구별하며 전체 attention·학습·리만가설 관련 해석의 충분조건은 아니다.

출력 사영 `W_o`에 `nn.utils.spectral_norm`을 적용한다.

$$
W_o \leftarrow W_o / \sigma_1(W_o),\qquad \sigma_1(W_o) \le 1.
$$

이는 사영 행렬 자체의 norm 조건이다. Attention 가중치, 입력 의존성,
다른 층과 residual 합까지 포함한 전체 map의 수축은 보장하지 않는다.
Norm 1인 항등 map을 residual로 더해도 전체 map은 2배가 되는 반례가 있다.

## 5. 블록 조립 (MRABlock)

MRABlock은 input hidden tensor에서 normalized attention output으로 가는 producer/consumer 조립 계약이다. residual·shape·precision·backend ordering이 다르면 parity가 깨질 수 있으며, 구현 성공은 benchmark 성능과 별개다.

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

`LBONorm`, `GaugeLattice`는 `legacy examples/ai/clarus_lm.py` (removed)에 이미 구현되어 있다.

## 6. Hermitian 옵션의 추가 조건

Hermitian 옵션은 특정 matrix symmetry를 강제하는 optional implementation branch다. Hilbert--Pólya와의 연결은 동기 또는 구조 비유이며, 옵션 통과·수치 대칭이 리만가설의 증거가 아니다.

`W_q = W_k`만으로는 위 score의 Hermitian 조건이 성립하지 않는다.
한 채널의 대각성분에 복소수 `1/(1/2+iγ)`가 남고, 실수부만 취해도
위치 amplitude와 sine 항 때문에 비대칭일 수 있다. `hermitian=True`라는
이름만으로 조건을 충족했다고 판단하지 않는다. 비음수 실수 가중 Gram 구성,
별도 대칭화와 mask·softmax 이후의 검사가 필요하다. 이 유한 행렬 조건은
Hilbert–Pólya 추측의 직접 구현이나 증명이 아니다.

## 7. 점근 / 안정성

점근·안정성은 정의한 sequence limit, norm, precision에서의 조건부 수학 또는 수치 성질이다. finite fixture·overflow·branch cut·OOD length가 범위를 벗어나면 반례·미완성으로 남는다.

- 작은 `p`에서 `log(1+p) ≈ p`이므로 기존 RoPE와 유사하다.
- 큰 `p`에서 인접 phase 차이가 감소한다. Alias와 학습 안정성은 추가 검증 대상이다.
- 정확한 공통 로그 이동은 `p → k(1+p)-1`이다. 일반적인 위치 배율에는 오차가 있다.
- 가중치 크기 `1/|1/2 + iγ_k|`는 큰 `|γ_k|`에서 감소한다.
  이 사실만으로 전체 학습 과정의 안정성을 증명하지 않는다.

## 8. 백엔드 정책

backend 정책은 CPU/GPU/complex dispatch의 input/output·precision·fallback 책임을 정한다. path 존재는 parity·성능·수학 지위의 증거가 아니며, unsupported shape와 tolerance failure는 rollback 조건이다.

PyTorch 참조 우선. Rust/CUDA 포트는 식이 안정화된 후 별도 작업으로 분리.

## 9. 참고

참고문헌은 수식 동기·구현 방법·외부 배경의 source role을 밝힌다. 인용은 이 명세의 fixture·precision·test evidence나 리만가설 결론을 대체하지 않는다.

- Riemann (1859), *Über die Anzahl der Primzahlen unter einer gegebenen Größe*.
- Hilbert–Pólya conjecture (folklore).
- Berry & Keating (1999), *H = xp and the Riemann zeros*.
- Su et al. (2021), *RoFormer* — RoPE 원본.
- Press et al. (2022), *ALiBi*.
- `paper/7_AGI/2_Architecture.md` — ClarusBlock 5계층 stack.
- `paper/6_뇌/07_수면과복구.md` — 부트스트랩 고정점 `ε² = 4.87 %`.
