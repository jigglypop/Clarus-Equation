# LLM Context Chunk

---
## File: `docs/7_AGI/19_OOD_Generalization.md`

```markdown
# 19. OOD Generalization 의 구조적 결정성: Length Extrapolation 사례

> 관련: `1_AGI.md`(5대 원리), `2_Architecture.md`(구조 규약), `13_Verification.md`(검증 방법론), `8_리만/mra_paper.md`(실증 데이터)
>
> 이 장은 Clarus 본 thesis "구조에서 의미가 나온다" 를 OOD generalization 영역에 처음으로 정량 적용한 결과이다. 현재까지의 증거는 length axis 한 가지에 한정되며, 이는 가설 (`hypothesis`) 에서 단일-축 검증 (`bridge`) 으로 격상된 단계로 분류한다.

---

## 0. 한 줄 결론

> Transformer 의 length-OOD generalization 은 두 tier 로 명확히 갈린다 — **rotation 위에 강한 distance attenuation (linear / multiplicative) 이 있거나, 회전 자체가 없으면 외삽한다. RoPE 회전을 그대로 두고 attenuation 을 약하게 얹거나 빼면 32× 에서 +29 ∼ +55 % 로 무너진다.**

32× length extrapolation (train N=64, eval N=2048) 에서, 9 개의 표준/신규 변종을 두 tier 로 분류 가능하다:

* **Tier 1 (외삽 OK, ≤ +10 %)**: ALiBi (`rope_alibi`, −6 %), Euler-CE 의 e-decay (`euler_no_pi`, −6 %), 풀 Euler-CE (−6 %), xPos (+4 %), NoPE (+7 %).
* **Tier 2 (외삽 BAD, +29 ∼ +55 %)**: RoPE 단독 (+47 %), MRA (+41 %), MRA-bias (+40 %), Euler 의 π-rotation 단독 (+55 %).

이는 AGI 의 핵심 어려움인 "학습 분포 밖에서 모델이 작동하는가" 가 데이터 양이 아니라 **inductive bias 의 형태** 에 의해 결정된다는 thesis 의 첫 직접 증거이다.

---

## 1. AGI 의 본질로서의 OOD Generalization

`1_AGI.md` 의 5대 원리는 모두 어떤 의미에서 OOD generalization 의 구조적 조건을 다룬다:

| 원리 | OOD axis | 메커니즘 |
|---|---|---|
| 부트스트랩 (3장) | task transfer | 수면-각성 순환의 catastrophic forgetting 방지 |
| 유니타리 (2장 § 4) | error propagation | 정보 증폭 차단으로 OOD 영역의 환각 억제 |
| 희소성 (5장) | computational regime | $\varepsilon^2$ 활성 비율로 conservative inference |
| 곡률 정규화 (6장) | latent geometry | 잠재 공간의 spurious peak 억제 |
| STDP (4장) | continual learning | 국소 학습으로 분포 변화 적응 |
| 자기참조재귀 (17장) | long-horizon task / agent drift | 자기비평, 기억, 잔류장을 다음 상태로 접어 오류를 닫힌 루프에서 수정 |

각 원리는 "학습 분포 밖에서 무엇이 작동하는가" 를 구조 선택으로 풀려는 시도이다.

본 장은 가장 측정 가능한 OOD axis — **시퀀스 길이** — 에서 이 thesis 를 정량 검증한다.

여기서 자기참조재귀는 별도 원리라기보다 다른 원리들을 장기 작업으로 묶는 닫힘 조건이다. length extrapolation 실험에서는 distance attenuation 하나만으로도 OOD 구조가 드러나지만, agentic OOD에서는 모델이 자기 출력, 실패, 수정 내역을 다음 state에 다시 넣어야 한다. 즉 장문 추론, tool-use, multi-turn planning에서는 \(S_t \to R(S_t) \to C_t \to S_{t+1}\) 재귀가 없으면 error propagation을 유니타리 제약으로 줄여도 자기수정은 일어나지 않는다.

이를 수학적으로 쓰면 OOD 입력열 \(e_{0:T}\)에 대한 closed-loop 안정성 문제다.

$$
S_{t+1}=\mathcal T_{\theta,e_t}(S_t),
\qquad
e_t\sim \mathcal D_{\rm out}.
$$

훈련 분포 \(\mathcal D_{\rm train}\)에서만 작은 손실을 갖는 것은 충분하지 않다. OOD 일반화에는 다음 조건이 필요하다.

$$
\mathbb E_{e_t\sim\mathcal D_{\rm out}}
\left[
\log
\frac{\|S_{t+1}-S_t\|}
{\|S_t-S_{t-1}\|+\epsilon}
\right]
<0.
$$

즉 평균 로그 수축률이 음수여야 한다. 이 조건은 단일 token PPL보다 agent drift를 더 직접적으로 본다. length extrapolation에서 distance attenuation이 하는 역할은 위치 상태의 \(\hat\rho_t\)를 낮추는 것이고, agentic OOD에서 자기참조재귀가 하는 역할은 critique/memory/residual state의 \(\hat\rho_t\)를 낮추는 것이다.

### 1.1 왜 length 인가

OOD axis 들은 측정 난이도에 큰 차이가 있다:

| axis | 통제 가능성 | 측정 가능성 |
|---|---|---|
| **length** | ★★★ (정확한 N 비율) | ★★★ (PPL @ N) |
| in-context demos | ★★ (분포 조작) | ★★ (task accuracy) |
| 의미 (semantic) | ★ (코퍼스 분리 어려움) | ★ (질적 평가) |
| 모달 (modal) | 비교 불가 | -- |
| task transfer | ★ (zero-shot 정의 모호) | ★★ (benchmark) |

Length 는 train N=64, eval N=2048 같이 정확한 32× OOD 상황을 만들 수 있고, PPL 로 한 자릿수 정확도로 측정된다. **OOD 효과를 다른 변수와 분리할 수 있는 거의 유일한 axis**.

따라서 length 에서 발견되는 구조적 원리는 OOD generalization 일반의 first-order 후보로 다룰 가치가 있다.

---

## 2. 실증 — 9 변종 32× ablation

`docs/8_리만/mra_paper.md` § 7.7 의 raw 결과를 AGI 관점에서 재해석한다.

### 2.1 비교 대상 (모두 ≈ 30 K params, train block = 64, char-level docs corpus)

| 변종 | 거리 처리 메커니즘 | 부류 |
|---|---|---|
| `nope` | 없음 (causal mask 만) | 베이스 (no PE) |
| `std_rope` | π-rotation (RoPE) | 회전 |
| `xpos` | RoPE + 채널별 multiplicative decay | 회전 + 곱셈 감쇠 |
| `mra` | RoPE + ζ amplitude weighting | 회전 + 진폭 |
| `mra_bias` | RoPE + log-additive distance bias | 회전 + log additive |
| `rope_alibi` | RoPE + linear additive bias (ALiBi) | 회전 + linear additive |
| `euler_no_decay` | π-rotation + block-aware base | 회전 |
| `euler_no_pi` | linear additive bias 만 | linear additive |
| `euler_ce_k1` | π-rotation + linear additive | 회전 + linear additive |

### 2.2 32× extrapolation 상대 degradation

(전체 표는 `mra_paper.md` § 7.7 참조)

| 변종 | 부류 | **32× degrad** | tier |
|---|---|---|---|
| `rope_alibi` | RoPE + linear additive (ALiBi) | **−6.2 %** | 1 |
| `euler_no_pi` | linear additive only | **−5.5 %** | 1 |
| `euler_ce_k1` | RoPE + linear additive + π-rotation | **−6.2 %** | 1 |
| `xpos` | RoPE + multiplicative decay | **+4.1 %** | 1 |
| `nope` | no positional encoding | **+7.0 %** | 1 |
| `mra_bias` | RoPE + log additive | +40.3 % | 2 |
| `mra` | RoPE + ζ amplitude | +40.8 % | 2 |
| `std_rope` | RoPE 단독 | +47.2 % | 2 |
| `euler_no_decay` | π-rotation + block-aware base | +54.7 % | 2 |

### 2.3 결정적 패턴: Tier 1과 Tier 2

```
Tier 1  (외삽 OK, ≤ +10 %):
  ALiBi (linear additive)   −6.2 %  ← 가장 강함
  Euler e-decay (= ALiBi)   −5.5 %
  xPos (multiplicative)     +4.1 %
  NoPE (no rotation)        +7.0 %

   ════════════════════════ ↕ 30 ∼ 50 %p cliff ═══════════════════

Tier 2  (외삽 BAD, +29 ∼ +55 %):
  log additive              +40.3 %
  RoPE + ζ amplitude        +40.8 %
  RoPE 단독                 +47.2 %
  π-rotation + base 만      +54.7 %
```

**Tier 1 과 Tier 2 는 30 ∼ 50 %p 의 절벽** 으로 갈린다 (random 결과 아님). 두 tier 의 구별 기준:

| Tier | 구조적 조건 |
|---|---|
| **1 (외삽 OK)** | 강한 distance attenuation 존재 (linear/multiplicative) **OR** rotation 부재 |
| **2 (외삽 BAD)** | rotation 존재 + attenuation 부재 또는 너무 약함 (log) |

핵심은 **NoPE가 RoPE보다 7배 잘 외삽한다**는 점이다(+7.0% 대 +47.2%). 즉 RoPE의 회전 자체가 OOD의 직접 원인이며, 이를 보정하는 강한 distance attenuation이 없으면 회전을 아예 빼는 편이 낫다(Kazemnejad et al. 2023의 finding 재현).

---

## 3. 왜 두 tier 가 갈리는가 — 직관

### 3.1 RoPE 회전이 OOD 의 직접 원인

NoPE가 RoPE보다 7배 잘 외삽한다는 사실(NoPE +7%, RoPE +47%)은 회전 메커니즘 자체가 OOD 위험 요소임을 보여준다. RoPE의 phase `θ_k(p) = p · 10000^{−2k/d}`는 `p`가 학습 분포 [0, 64] 안에서는 학습된 attention 분포를 만들지만, `p ∈ [64, 2048]` 영역에서는 학습 분포 밖의 phase를 만든다. 즉 학습된 attention head가 본 적 없는 회전 상태에서 평가된다.

특히 RoPE 의 wrap-around 특성 (`cos(2π+θ) = cos θ`) 때문에 멀리 떨어진 토큰 간 attention 이 가까운 토큰 간 attention 과 *같은 phase* 로 wrap 되어 관계가 깨진다.

### 3.2 Distance attenuation 이 회전 효과를 dominate

Tier 1 의 핵심 메커니즘은 회전과 무관한 별도의 distance signal:

| 형태 | 거리 의존 | 외삽 안정 |
|---|---|---|
| Linear additive `−m d` | strong, monotonic | **−6 %** (best) |
| Multiplicative `ζ^d` | strong, monotonic | **+4 %** |
| Log additive `−log d` | weak (sub-linear) | +40 % (fail) |

거리에 monotonic 하고 unbounded 한 attenuation 이 있으면 RoPE 회전의 wrap-around 효과를 dominate 하여 attention 이 자연스럽게 가까운 토큰에 집중. Linear 가 가장 강하고 multiplicative 도 작동, log 는 너무 천천히 감쇠해서 효과 없음.

### 3.3 1-jet 충분성 (linear 가 가장 강한 이유)

모든 smooth 함수 `f(d)` 는 점 d₀ 근처에서 1-jet (constant + slope) 으로 근사 가능: `f(d) ≈ f(d₀) + f'(d₀)(d − d₀)`. Linear additive bias 는 학습 분포 안에서 `f'(d)` 의 점추정만 학습하면 되고, 그 추정이 외삽 영역에서도 유효 (선형 함수의 1-jet 은 함수 전체와 동일).

Multiplicative `ζ^d` 도 log-space 에서 linear (`log ζ · d`) 이므로 비슷한 이유로 외삽 안정. Logarithmic 은 비선형 변환이 들어가서 학습 분포 밖에서 형태가 달라짐 → 약함.

이 직관은 가설이며, 형식 증명은 후속 작업.

---

## 4. Clarus 본 thesis 와의 정합

### 4.1 "구조에서 의미가 나온다"

Clarus AGI 의 핵심 명제 (`1_AGI.md` § 0): 우주는 빅뱅에서 한 번의 부트스트랩으로 고정점에 도달했고, 부트스트랩은 **구조적 자기조직화** 의 결과이다. 즉 의미와 구조는 분리 가능한 양이 아니다.

본 발견은 이 명제의 ML 영역 사례:

> 같은 학습 데이터, 같은 모델 크기, 같은 옵티마이저, 같은 손실. 단지 **거리 처리 함수의 형태** 만 다르다. 그런데 외삽 능력은 한 형태에서만 발현. → 외삽은 데이터의 함수가 아니라 **구조의 함수**.

이는 scaling-only 접근 (데이터/파라미터 증량으로 모든 것 해결) 에 대한 강한 반증. 같은 학습 데이터에서 어떤 구조는 외삽하고 어떤 구조는 못 한다 → 데이터로 환원 불가.

### 4.2 자유 파라미터 0 원칙과의 관계

CE 본 thesis 는 자유 파라미터 0 (모든 비율이 axiom 에서 연역). 본 발견은 약한 형태:

> Linear additive bias 의 *slope* 한 개는 학습. 그러나 *형태* 자체 (linear 가 외삽한다는 사실) 는 axiom — 학습으로 발견되지 않으며, 구조 선택의 결과.

즉 ML 학습이 풀 수 있는 부분 (slope) 과 풀 수 없는 부분 (form) 의 경계가 보인다. AGI 시스템은 form-level 선택을 학습 외부에서 (axiom 으로) 가져와야 한다.

---

## 5. 일반화 가설 — 다른 OOD axis

본 발견을 다른 OOD axis 로 일반화한 가설:

### H1 (semantic transfer)

코퍼스 A 학습 → 코퍼스 B 평가 시, **B 의 distance/structure 함수 형태** 가 A 의 학습된 inductive bias 와 정합하는 정도가 transfer accuracy 를 결정한다. Slope-level 차이는 fine-tune 으로 흡수, form-level 차이는 catastrophic.

### H2 (in-context length)

n-shot in-context learning 에서 n 외삽 능력은 토큰 attention 의 distance 함수 형태에 의해 결정된다 (n 도 일종의 sequence position 이므로).

### H3 (compositional generalization)

수학 문제의 추론 깊이 외삽 (train: 5-step, eval: 20-step) 도 step-distance 의 처리 형태에 의해 결정. CoT 의 효과는 효율적으로 step-distance 를 linear 로 표현한 결과.

각 가설은 본 length 결과와 같은 패턴 — "form-level 구조가 OOD 를 결정한다" — 를 다른 axis 로 옮긴 것. 별도 검증 필요.

---

## 5.5. 5 Euler 상수 → 2 비트 head-type taxonomy

본 발견에는 추가적인 구조적 환원이 따른다. Euler 5상수 `{e, π, i, 1, 0}` 의 attention 에서의 작용은:

| 상수 | 작용 | 환원 |
|---|---|---|
| `π`, `i` | rotation generator (`e^{iπt}` 결합) | **axis 1**: rotation |
| `e` | exponential decay base | **axis 2**: decay |
| `1`, `0` | on/off gate values | 각 axis 의 1 비트 |

→ **2 axis × 2 gate value = 2² = 4 head-types**, 2-bit string `(pi, e)` 으로 인코딩:

| (pi, e) | 헤드 타입 | 문헌 분석 |
|---|---|---|
| (0, 0) | NoPE [Kazemnejad 2023] | Tier 1 |
| (0, 1) | ALiBi [Press 2022] | Tier 1 (best) |
| (1, 0) | RoPE [Su 2021] | **Tier 2** (외삽 fail) |
| (1, 1) | xPos [Sun 2023] / Euler-CE | Tier 1 |

**§ 2.2 의 모든 9 변종이 이 4 가지 중 하나로 정확히 매핑**되며, 4 가지 중 단 한 가지 (10 = pure rotation) 만 Tier 2 (외삽 catastrophic). 즉 effective head-type capacity = log₂ 3 ≈ 1.58 비트.

`reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEMinimal` 구현 + 16 개 테스트로 정확성 검증. 2-bit minimal 변종이 canonical PE (NoPE, RoPE, ALiBi, xPos) 를 수치적으로 reproduce (`min_alibi` ≈ `rope_alibi`, `min_xpos` ≈ `euler_ce_k1`, `min_rope` = `std_rope` 정확 일치).

이는 Clarus 본 thesis ("자유 파라미터 0 에 가깝게") 의 또 다른 사례 — **5 차원 continuous bit_logits 가 사실상 2 비트 axiom 으로 환원**되며, 학습은 axis 선택을 풀 필요 없이 axiom 으로 받고 continuous parameter (xi, slope) 만 학습하면 된다.

---

## 6. AGI 아키텍처 설계 권고

본 발견이 직접적으로 시사하는 설계 원칙:

### R1. Rotation 과 Distance attenuation 의 조합 강제

Sequence position, time step, recursion depth, attention distance 등 모든 "거리" 양은 다음 중 하나로 인코딩해야 한다.

* **Tier 1A**: rotation + 강한 attenuation (linear ALiBi 권장 / multiplicative xPos 차선)
* **Tier 1B**: rotation 자체 부재 (NoPE)

회전을 그대로 두고 attenuation 을 빼거나 약하게 (logarithmic) 얹으면 OOD 에서 catastrophic. RoPE 의 인기에 끌려 회전을 무비판적으로 채택하면 long-context 에서 무너진다.

### R2. Form-level 선택은 axiom

ML 학습은 slope 만 풀 수 있다. Form (linear / log / mult / rotation) 은 설계 시점 선택. Form 후보가 여러 개일 때는 OOD 검증이 결정.

### R3. OOD axis 별 구조 검증 의무

새 아키텍처 제안 시 in-distribution PPL 만으로 평가하지 말고, 적어도 한 가지 측정 가능한 OOD axis (length 가 가장 깔끔) 에서 16 ∼ 32× 외삽 검증을 명시. 본 작업의 9 개 변종 비교가 reference framework.

---

## 7. 한계와 후속

### 한계

1. **단일 axis (length)**: H1/H2/H3 미검증.
2. **소규모**: ~30 K params, 600 K char corpus. 1B+ 모델에서 같은 패턴 유지되는지 미확인.
3. **단일 도메인**: Korean+English markdown. 다른 modality 미평가.
4. **이론 부재**: § 3 의 직관 (1-jet, dimensional analysis) 은 가설이며 형식 증명 없음.

### 후속

1. **Scaling**: d_model ∈ {64, 128, 256, 512} × n_layers ∈ {2, 4, 8, 12} 에서 32× 외삽 패턴 유지되는지.
2. **다른 OOD axis**: H1 (코퍼스 transfer), H2 (in-context length), H3 (CoT depth) 각각 검증.
3. **이론**: linear additive 가 외삽하는 형식적 이유 (PAC 학습 이론, Rademacher complexity 분석).
4. **GaugeLattice + linear bias**: 본 발견을 `2_Architecture.md` 의 ClarusBlock 에 통합. 현재 ClarusAttention 은 RoPE 사용 중인데, ALiBi 또는 linear additive variant 로 교체 검토.

---

## 8. 데이터 출처

- 9 변종 × 3 seed × 1500 step × 32× extrapolation 측정: `examples/ai/results/euler_extrap_long.json`, `extrap_full.json` (확장)
- 9 변종 코드: `examples/ai/bench_recursive_euler.py` (`RoPEAttnBlock`, `NoPEAttnBlock`, `XPosAttnBlock`, `RoPEAlibiAttnBlock`, etc.)
- ExtrapLM 및 train/eval 파이프라인: `examples/ai/bench_mra_extrap.py`
- 분석 표 + verdict: `docs/8_리만/mra_paper.md` § 7.7

## 9. References

- Press et al. (2022), *Train Short, Test Long: Attention with Linear Biases (ALiBi)*, ICLR.
- Sun et al. (2023), *A Length-Extrapolatable Transformer (xPos)*, ACL.
- Su et al. (2021), *RoFormer: Rotary Position Embedding*, arXiv:2104.09864.
- Kazemnejad et al. (2023), *The Impact of Positional Encoding on Length Generalization in Transformers*, NeurIPS.
- (Internal) `docs/7_AGI/1_AGI.md`, `2_Architecture.md`, `8_Roadmap.md`.
- (Internal) `docs/8_리만/mra_paper.md` — full ablation data.
```
---
## File: `docs/7_AGI/9_LLM.md`

```markdown
# CE-LLM 실전 구축 가이드

> 관련: 2-6장(이론), `legacy examples/ai/clarus_lm.py` (removed)(처음부터 학습), `examples/ai/ce_gpt2.py`(기존 모델 이식), `legacy examples/ai/train_clarus.py` (removed)(학습 스크립트)
>
> 이 장은 CE-AGI 원리를 적용한 LLM을 실제로 만드는 세 가지 경로를 다룬다. 이론이 아니라 코드와 명령어 중심.

---

## 0. 핵심 원리: 자기참조재귀

CE-LLM의 중심은 특정 attention 변형이나 곡률 regularizer 하나가 아니다. 핵심은 모델이 만든 내부 상태를 다시 자기 입력으로 접어 넣는 **자기참조재귀**다.

일반 Transformer의 한 토큰 추론은

$$
h_{t+1}=F_\theta(h_t, x_t)
$$

처럼 고정 가중치가 입력을 한 번 통과시키는 구조에 가깝다. CE식으로 바꾸면 최소형은

$$
z_t = R(S_t), \qquad
c_{t+1}=C(z_t,a_t,o_t,m_t), \qquad
S_{t+1}=\mathcal U(S_t,z_t,c_{t+1},m_{t+1},\phi_{t+1})
$$

이다. 즉 모델은 출력 직전의 hidden state만 쓰는 것이 아니라, 자기비평 \(c_t\), 기억 \(m_t\), 잔류장 \(\phi_t\), 모드 \(M_t\)를 다시 다음 추론의 조건으로 넣는다. 이것이 `17_AgentLoop.md`의 Layer F이고, LLM 응용에서 가장 먼저 보존해야 할 구조다.

실전 구현에서 자기참조재귀는 세 단계로 나뉜다.

| 단계 | 구현 형태 | 바로 가능한 응용 | 지위 |
|---|---|---|---|
| 내부 재귀 | 한 block 또는 hidden state를 \(k\)회 이완해 고정점 근처로 보냄 | recursive block, long-context 안정화 | 구현/벤치 가능 |
| 비평 재귀 | 출력 후보를 자기비평 점수 \(c_t\)로 평가하고 다음 decoding에 반영 | hallucination suppressor, verifier, reranker | 프로토타입 가능 |
| 기억 재귀 | 행동/관찰/비평을 memory에 쓰고 다음 step의 context로 회수 | agent memory, sleep replay, RAG state | 구현 가능 |

따라서 CE-LLM 이식의 우선순위는 다음 순서다.

1. **한 번 더 생각하는 내부 이완**: \(h \to F(h) \to F(F(h))\) 구조를 넣어 hidden state를 자기 고정점으로 보낸다.
2. **자기비평을 다음 입력으로 접기**: 답변 후 평가가 로그로만 남지 않고, 다음 token 또는 다음 turn의 state에 들어가야 한다.
3. **잔류장을 버리지 않기**: softmax에서 탈락한 후보, 높은 곡률 구간, 불확실성 신호를 \(\phi\)로 보존한다.
4. **수면/리플레이로 재정렬**: online 추론 중 쌓인 \(m,c,\phi\)를 offline replay에서 다시 압축한다.

이 기준으로 보면 Euler-CE attention, LBONorm, spectral norm, MRA, curvature penalty는 모두 보조 장치다. 이 장치들이 의미를 가지려면 결국 \(S_t \to R(S_t) \to C \to \mathcal U(S_{t+1})\) 루프 안에서 자기 상태를 갱신해야 한다.

### 0.1 수학적 최소형

상태공간을

$$
\mathcal S
=
\mathcal G
\times \mathcal M
\times \mathcal C
\times \mathcal H
\times \Phi
$$

로 둔다. 각각 전역 요약, 기억, 자기비평, 이력, 잔류장이다. 한 step의 CE-LLM은 외부 입력 \(x_t\)와 관찰 \(o_t\)에 대해 다음 자기 사상으로 정의된다.

$$
\mathcal T_{\theta,x_t,o_t}:\mathcal S\to\mathcal S,
\qquad
S_{t+1}=\mathcal T_{\theta,x_t,o_t}(S_t).
$$

이를 구성요소로 풀면

$$
\begin{aligned}
z_t &= R_\theta(S_t,x_t),\\
a_t &= \pi_\theta(z_t,S_t),\\
\hat{o}_t &= P_\theta(z_t,a_t),\\
c_{t+1}
&=
C_\theta(z_t,a_t,o_t,m_t)
=
\begin{bmatrix}
d_{\rm pred}(\hat{o}_t,o_t)\\
d_{\rm cons}(z_t,\mathcal R(H_t,c_t))\\
\kappa(z_t)\\
\Delta_{\rm nov}(o_t)
\end{bmatrix},\\
m_{t+1}&=\mathcal M(m_t,z_t,a_t,o_t,c_{t+1}),\\
\phi_{t+1}
&=(1-\alpha_\phi)\phi_t
+\alpha_\phi\,\Pi_{\rm res}(z_t,a_t,c_{t+1}),\\
S_{t+1}&=(G_{t+1},m_{t+1},c_{t+1},h_{t+1},\phi_{t+1}).
\end{aligned}
$$

여기서 중요한 항은 \(c_{t+1}\)이다. 자기비평이 단순 로그나 평가 리포트로 끝나면 루프는 열려 있다. CE식 응용에서는 \(c_{t+1}\)가 다음 step의 에너지에 들어가야 한다.

$$
E_{t+1}(z)
=
E_{\rm task}(z;x_{t+1})
+\lambda_m E_{\rm mem}(z;m_{t+1})
+\lambda_c E_{\rm crit}(z;c_{t+1})
+\lambda_\phi E_{\rm res}(z;\phi_{t+1}).
$$

따라서 CE-LLM의 최소 판정 조건은 다음이다.

$$
\boxed{
\frac{\partial S_{t+1}}{\partial c_{t+1}}\ne0,
\qquad
\frac{\partial E_{t+1}}{\partial c_{t+1}}\ne0
}
$$

이 두 조건이 없으면 모델은 자기비평을 생성할 수는 있어도, 자기비평으로 자기 동역학을 바꾸지는 않는다.

### 0.2 안정성 조건

고정된 task regime에서 이상적인 자기참조재귀는 어떤 attractor \(S^\star\) 근처로 수축해야 한다.

$$
S^\star=\mathcal T(S^\star).
$$

충분조건은 가중 norm \(\|\cdot\|_Q\)에 대해

$$
\|\mathcal T(S)-\mathcal T(S')\|_Q
\le \rho\|S-S'\|_Q,
\qquad 0\le\rho<1
$$

이다. 실제 LLM은 외부 입력과 tool observation 때문에 완전 수축이 아니라 입력-상태 안정성(ISS)으로 읽는다.

$$
\|S_t-S_t^\star\|_Q
\le
\rho^t\|S_0-S_0^\star\|_Q
+
\sum_{\tau=0}^{t-1}\rho^{t-1-\tau}
\left(
\beta_x\|x_\tau-x_\tau^\star\|
+\beta_o\|o_\tau-o_\tau^\star\|
\right).
$$

AI 응용에서 측정할 값은 \(S^\star\) 자체가 아니라 수축률과 잔류 반경이다.

$$
\hat\rho_t
=
\frac{\|S_{t+1}-S_t\|_Q}{\|S_t-S_{t-1}\|_Q+\epsilon},
\qquad
r_{\rm res}
=
\limsup_{t\to\infty}\|S_t-S_t^\star\|_Q.
$$

좋은 CE-LLM 보강은 accuracy만 올리는 모듈이 아니라, \(\hat\rho_t\)를 낮추거나 \(r_{\rm res}\)를 줄이는 모듈이다.

### 0.3 계층화 가능성

여러 CE-LLM 모듈을 쌓을 때는 각 모듈을 하나의 작은 자기 사상으로 본다.

$$
S_{i,t+1}^{(\ell)}
=
\mathcal T_i^{(\ell)}
\left(
S_{i,t}^{(\ell)},\;
u_{i,t}^{(\ell)}
\right).
$$

하위 모듈들의 상태 요약이 상위 상태가 되고,

$$
S_t^{(\ell+1)}
=
A_\ell(S_{1,t}^{(\ell)},\dots,S_{n,t}^{(\ell)}),
$$

상위 critic/goal은 다시 하위 입력으로 내려간다.

$$
u_{i,t}^{(\ell)}
=
B_i^{(\ell)}(S_t^{(\ell+1)}).
$$

이 구조가 안정하려면 `17_AgentLoop.md` F.-1.5의 gain matrix \(G\)가

$$
\rho(G)<1
$$

을 만족해야 한다. 실전적으로는 모듈을 추가할 때마다 세 값을 기록한다.

| 값 | 의미 | 너무 크면 생기는 문제 |
|---|---|---|
| \(\rho_\ell\) | 해당 모듈 자체의 수축률 | 내부 사고가 수렴하지 않음 |
| \(g_\uparrow\) | 하위 요약이 상위 상태를 흔드는 gain | 작은 오류가 global state로 증폭 |
| \(g_\downarrow\) | 상위 critic/goal이 하위 모듈을 흔드는 gain | top-down 명령이 하위 루프를 파괴 |

자기유사하게 같은 모듈을 반복 배치하는 경우 충분조건은

$$
\rho_0+2\sqrt{g_\uparrow g_\downarrow}<1.
$$

따라서 프랙탈형 CE-LLM은 "재귀 모듈을 많이 쌓는 것"이 아니라, 상향 요약과 하향 피드백의 곱을 small-gain 영역 안에 유지하는 설계다.

---

## 1. 세 가지 구축 경로

| 경로 | 설명 | 난이도 | 소요 | 결과물 |
|---|---|---|---|---|
| **A. 처음부터 학습** | ClarusLM을 스크래치로 학습 | 낮음 | GPU 수시간 | 소형 CE-LLM |
| **B. 기존 모델 이식** | GPT-2/Llama 등에 CE 모듈 이식 | 중간 | GPU 수시간 | CE-강화 LLM |
| **C. 대규모 사전학습** | CE 아키텍처로 대규모 학습 | 높음 | 클러스터 수일 | 실용급 CE-LLM |

---

## 2. 경로 A: 처음부터 학습 (ClarusLM)

### 2.1 기존 코드 구조

`legacy examples/ai/clarus_lm.py` (removed)가 CE-LLM의 완전한 모델 정의를 포함한다.

```
ClarusLM
  ├── tok_emb (Embedding)
  ├── pos_emb (Embedding)
  ├── blocks[] (ClarusBlock x N)
  │     ├── norm1 (LBONorm)          // LayerNorm + LBO 확산
  │     ├── attn (ClarusAttention)    // MHA + spectral norm
  │     ├── norm2 (LBONorm)
  │     └── ffn (GaugeLattice)        // 3x3+1 게이지 격자
  │           ├── su3 (SU(3) binding, 74.1%)
  │           ├── su2 (SU(2) decision, 21.1%)
  │           ├── u1 (U(1) attention, 4.9%)
  │           └── phi (LBONorm, smoothing)
  ├── norm (LBONorm)
  └── head (Linear, weight tied)
```

CE 수정 4가지가 모두 내장되어 있다:

1. **LBONorm**: `F.layer_norm` + 저랭크 LBO 확산 (`V^T V`)
2. **GaugeLattice**: 채널 비율 `alpha_s : alpha_w : alpha_em`으로 자동 분할
3. **Spectral Norm**: `nn.utils.spectral_norm(proj)` -- 유니타리 제약
4. **곡률 손실**: `loss = ce + lambda_curv * curv`

### 2.2 학습 실행

**데이터 준비:**

텍스트 파일 하나면 된다. 한국어, 영어, 코드, 수학 -- 무엇이든 가능.

```bash
# 예: 위키피디아 덤프, 논문 텍스트, 코드 파일 등
cat *.txt > train_data.txt
```

**학습 명령:**

```bash
cd examples/ai
python train_clarus.py \
    --data train_data.txt \
    --dim 256 \
    --n_layers 6 \
    --n_heads 8 \
    --seq_len 256 \
    --batch_size 32 \
    --lr 3e-4 \
    --steps 5000 \
    --lambda_curv 0.01 \
    --device cuda
```

**출력 예시:**

```
ClarusLM  4.23M params
  vocab=95  dim=256  layers=6  heads=8
  train=1234567  val=65000 chars
  device=cuda  lambda_curv=0.01

3x3+1 lattice:
  SU(3) binding:   189 dims (74.1%)
  SU(2) decision:   54 dims (21.1%)
  U(1)  attention:  13 dims (4.9%)
  Phi   smoothing: LBO (rank=32)

step     1 | loss 4.5432 | val 4.5123 | curv 0.012345 | ...
step   200 | loss 2.3456 | val 2.4567 | curv 0.003456 | ...
...
```

### 2.3 규모별 설정

| 규모 | dim | layers | heads | 파라미터 | GPU 메모리 | 학습 시간 |
|---|---|---|---|---|---|---|
| Micro | 128 | 4 | 4 | ~1M | < 1GB | 수분 |
| Small | 256 | 6 | 8 | ~4M | < 2GB | 수십분 |
| Medium | 512 | 12 | 8 | ~30M | ~4GB | 수시간 |
| Large | 768 | 12 | 12 | ~85M | ~8GB | 반일 |
| XL | 1024 | 24 | 16 | ~350M | ~24GB | 수일 |

### 2.4 캐릭터 레벨 → 서브워드 토크나이저

기존 `train_clarus.py`는 캐릭터 레벨 토크나이저를 사용한다. 실용급으로 올리려면 서브워드 토크나이저가 필요하다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # BPE, 50257 vocab

model = ClarusLM(
    vocab_size=tokenizer.vocab_size,  # 50257
    dim=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=1024,
    lambda_curv=0.01,
)
```

학습 루프는 `train_clarus.py`의 구조를 그대로 사용하되, `CharTokenizer`를 `AutoTokenizer`로 교체한다.

---

## 3. 경로 B: 기존 모델 이식 (CE-GPT2)

### 3.1 2단계 이식 전략

`examples/ai/ce_gpt2.py`가 GPT-2에 CE를 이식하는 완전한 코드다.

**Phase 1 -- 비파괴 이식 (성능 보존):**

- `LayerNorm` $\to$ `LBONorm` (h=0 초기화, scale/bias 복사 $\to$ 원본과 동일 출발)
- `c_proj` $\to$ `spectral_norm` (가중치 보존 + 유니타리 제약)

이 시점에서 모델 출력은 원본 GPT-2와 **완전히 동일**하다. CE 모듈이 추가되었지만 h=0이므로 LBO 확산이 꺼져 있다.

**Phase 2 -- MLP 압축 (선택적):**

- `MLP` $\to$ `GaugeLatticeV2` (cross-channel mixing 포함)
- 증류(distillation)로 초기화: 원본 MLP의 입출력을 모방하도록 학습

### 3.2 실행

```bash
cd examples/ai

# Phase 1만 (안전, 빠름)
python ce_gpt2.py --data train_data.txt --phase 1 --steps 200

# Phase 2 포함 (MLP 교체, 37% 파라미터 절감)
python ce_gpt2.py --data train_data.txt --phase 2 --steps 500
```

### 3.3 다른 모델에 이식

GPT-2 외의 모델(Llama, Mistral, Phi 등)에도 동일한 원리로 이식 가능하다. 핵심은 3가지:

**1) LayerNorm $\to$ LBONorm:**

```python
def transplant_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            dim = module.normalized_shape[0]
            lbo = LBONorm(dim)
            lbo.scale.data = module.weight.data.clone()
            lbo.bias.data = module.bias.data.clone()
            lbo.h.data.fill_(0.0)  # h=0: 원본과 동일 출발
            parent = get_parent(model, name)
            setattr(parent, name.split('.')[-1], lbo)
```

**2) Attention 출력 사영에 Spectral Norm:**

```python
def transplant_spectral(model):
    for block in model.layers:
        block.self_attn.o_proj = nn.utils.spectral_norm(block.self_attn.o_proj)
```

**3) (선택) MLP $\to$ GaugeLatticeV2:**

```python
def transplant_mlp(model, distill_steps=500):
    for block in model.layers:
        old_mlp = block.mlp
        dim = old_mlp.gate_proj.in_features
        new_lattice = GaugeLatticeV2(dim, mult=4, mix_rank=dim//8)
        distill(old_mlp, new_lattice, steps=distill_steps)
        block.mlp = new_lattice
```

### 3.4 미세조정

이식 후 CE 파라미터(LBO의 h, V, 곡률 정규화)를 미세조정한다.

```python
# CE 파라미터만 학습 (나머지 동결)
for name, param in model.named_parameters():
    param.requires_grad = False
    if any(k in name for k in ['LBONorm', 'lbo', 'phi', 'spectral']):
        param.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5, weight_decay=0.01
)
```

---

## 4. 경로 C: 대규모 CE 사전학습

### 4.1 아키텍처 설정

1B 규모 CE-LLM:

```python
model = ClarusLM(
    vocab_size=32000,       # SentencePiece
    dim=2048,
    n_layers=24,
    n_heads=16,
    max_seq_len=2048,
    ffn_mult=4,
    lambda_curv=0.005,
)
# ~1.3B params (표준 Transformer 대비 37% 적음 -> ~800M 유효)
```

### 4.2 수면 학습 순환 (3장 적용)

대규모 학습에서 수면 순환을 적용하는 방법:

```python
def sleep_train(model, dataloader, n_cycles=10, device='cuda'):
    for cycle in range(n_cycles):
        # === 각성(Wake): 표준 학습 ===
        model.train()
        accumulated_grads = {}
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            # 그래디언트 누적만, 업데이트 보류
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = torch.zeros_like(p.grad)
                    accumulated_grads[name] += p.grad.clone()
            model.zero_grad()

        # === NREM: 곡률 기반 선택적 업데이트 ===
        model.eval()
        EPSILON_SQ = 0.0487
        for name, p in model.named_parameters():
            if name in accumulated_grads:
                g = accumulated_grads[name]
                # 상위 4.87%만 통과
                threshold = torch.quantile(g.abs().flatten(), 1.0 - EPSILON_SQ)
                mask = (g.abs() >= threshold).float()
                p.data -= lr * g * mask

        # === REM: 비선택 그래디언트 재탐색 ===
        for name, p in model.named_parameters():
            if name in accumulated_grads:
                g = accumulated_grads[name]
                threshold = torch.quantile(g.abs().flatten(), 1.0 - EPSILON_SQ)
                pruned = g * (g.abs() < threshold).float()
                # 노이즈 주입 + 재평가
                noise = torch.randn_like(pruned) * pruned.std() * 0.1
                candidate = pruned + noise
                # 개선되면 채택 (간소화된 버전)
                p.data -= lr * 0.01 * candidate

        # 수면 압력 리셋
        print(f"Cycle {cycle+1}/{n_cycles} complete")
```

### 4.3 희소 추론 (5장 적용)

학습 후 추론 시 Top-k 활성화 적용:

```python
class SparseGaugeLattice(GaugeLattice):
    """추론 시 4.87% 활성화만 사용."""

    EPSILON_SQ = 0.0487

    def forward(self, x):
        y = super().forward(x)
        if not self.training:
            # 추론 시 Top-k 활성화
            k = max(1, int(self.EPSILON_SQ * y.shape[-1]))
            topk_vals, topk_idx = torch.topk(y.abs(), k, dim=-1)
            mask = torch.zeros_like(y)
            mask.scatter_(-1, topk_idx, 1.0)
            y = y * mask * (y.shape[-1] / k)  # 스케일 보정
        return y
```

### 4.4 환각 억제 추론 (6장 적용)

생성 시 곡률 모니터링 + 개입:

```python
@torch.no_grad()
def generate_with_curvature_check(model, idx, n_tokens,
                                   temperature=0.8, top_k=40,
                                   curv_threshold=0.1, max_retry=3):
    for _ in range(n_tokens):
        x = idx[:, -model.max_seq_len:]
        logits, _ = model(x)
        logits = logits[:, -1] / temperature

        # 곡률 측정
        avg_curv = sum(b.curvature for b in model.blocks) / len(model.blocks)

        retry = 0
        while avg_curv > curv_threshold and retry < max_retry:
            # 곡률 평탄화: 마지막 블록의 hidden state에 LBO 확산 추가 적용
            for block in model.blocks:
                block.norm1.h.data *= 1.5  # 일시적으로 확산 강도 증가
            logits, _ = model(x)
            logits = logits[:, -1] / temperature
            avg_curv = sum(b.curvature for b in model.blocks) / len(model.blocks)
            retry += 1
            for block in model.blocks:
                block.norm1.h.data /= 1.5  # 복원

        # Top-k 샘플링
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        next_token = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, next_token], 1)

    return idx
```

### 4.5 Grounded CE-LLM: 감각 발화 집합 추가

텍스트-only LLM보다 AGI 쪽으로 가려면, 앞단에 모달리티별 sparse encoder를 두는 편이 더 자연스럽다(`7_AGI/12_Equation.md` 6.8-6.9절).

```python
class GroundedCELLM(nn.Module):
    def __init__(self, text_model, vision_encoder, audio_encoder, touch_encoder):
        super().__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.touch_encoder = touch_encoder
        self.epsilon_sq = 0.0487

    def topk_act(self, h):
        k = max(1, int(self.epsilon_sq * h.shape[-1]))
        vals, idx = torch.topk(h.abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, idx, 1.0)
        return h * mask

    def forward(self, text_ids, image=None, audio=None, touch=None):
        h_t = self.topk_act(self.text_model.embed(text_ids))
        h_v = self.topk_act(self.vision_encoder(image)) if image is not None else None
        h_a = self.topk_act(self.audio_encoder(audio)) if audio is not None else None
        h_h = self.topk_act(self.touch_encoder(touch)) if touch is not None else None
        h_joint = bind_xi([x for x in [h_t, h_v, h_a, h_h] if x is not None])
        return self.text_model.decode_from_joint(h_joint)
```

이 경로의 핵심 예측은 두 가지다.

1. 모달별로 먼저 `4-5%` 발화 집합을 만든 뒤 결합하는 편이 early fusion보다 효율적이다.
2. 시각/청각/촉각 grounding이 추가되면, 텍스트-only 모델보다 멀티모달 환각이 줄어들어야 한다.

---

## 5. CE 모듈별 구현 상세

### 5.1 LBONorm 내부 동작

```python
def forward(self, x):
    # 1단계: 표준 LayerNorm (안정성 보장)
    x = F.layer_norm(x, (self.dim,))

    # 2단계: 저랭크 LBO 확산
    #   xW = x V^T V  (x를 V의 열공간으로 사영)
    #   Lx = x - xW   (사영 잔차 = 고곡률 성분)
    xW = F.linear(F.linear(x, self.V), self.V.T)
    Lx = x - xW

    # 3단계: 확산 적용
    #   h > 0이면 고곡률 성분 Lx가 감쇠됨
    #   h = 0이면 LayerNorm과 동일
    h = self.h.abs().clamp(max=0.5)
    self._curvature = (Lx * Lx).mean()  # 곡률 에너지 저장

    return (x - h * Lx) * self.scale + self.bias
```

핵심: `V`가 "평탄한 부분공간"을 학습한다. $x$를 이 부분공간으로 사영한 것이 $xW$이고, 사영 잔차 $Lx$가 "고곡률 성분"이다. $h > 0$이면 이 고곡률 성분이 감쇠된다.

### 5.2 GaugeLattice 채널 분할

```python
# d=768 예시
total = 0.11789 + 0.03352 + 0.00775  # = 0.15916
d3 = round(768 * 0.11789 / 0.15916)  # = 568 (SU(3) binding)
d2 = round(768 * 0.03352 / 0.15916)  # = 162 (SU(2) decision)
d1 = 768 - 568 - 162                  # =  38 (U(1) attention)

# 입력 x를 [x_3 | x_2 | x_1]으로 분할
# 각각 독립적인 MLP를 통과
# 결과를 concat
```

### 5.3 Spectral Norm 적용

```python
# 적용 전: sigma_1(W) 제약 없음 (정보 증폭 가능)
proj = nn.Linear(dim, dim, bias=False)

# 적용 후: sigma_1(W) <= 1 (유니타리 제약)
proj = nn.utils.spectral_norm(nn.Linear(dim, dim, bias=False))
```

PyTorch의 `spectral_norm`은 power iteration으로 최대 특이값을 추정하고, forward 시 자동으로 `W / sigma_1(W)`를 적용한다.

---

## 6. 학습 모니터링: 무엇을 봐야 하는가

### 6.1 핵심 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| `loss` | Cross-entropy 손실 | 단조 감소 |
| `curv` | 평균 곡률 에너지 $\|\Delta_g h\|^2$ | 학습 초반 증가 후 안정화 |
| `val_loss` | 검증 손실 | train과 괴리 없어야 함 |
| `lr` | 학습률 | warmup + cosine decay |
| `active_ratio` | 실제 활성 비율 | `4-5%` 중심, `3-7%` 실용 대역 |
| `bootstrap_resid` | $\|p_n - p^*\|$ 또는 proxy | 수면 루프에서 감소 |
| `hall_corr` | 곡률-오류 상관 | 양의 상관 기대 |
| `ground_align` | 모달 정합도 | grounded 모델에서 증가 기대 |

### 6.2 곡률 에너지의 해석

- **curv 단조 증가**: 모델이 복잡한 패턴을 학습 중 (정상)
- **curv 급등**: 불안정한 영역 진입 (lambda_curv 증가 고려)
- **curv 수렴**: 모델이 안정적 표현 공간을 찾음 (이상적)
- **curv가 0에 수렴**: LBO가 과도하게 평탄화 (lambda_curv 감소 고려)

### 6.3 3x3+1 격자 균형

`model.lattice_summary()`로 격자 구조를 확인:

```
SU(3) binding:   189 dims (74.1%)   # 결합: 지각 요소 통합
SU(2) decision:   54 dims (21.1%)   # 결정: 분기/선택
U(1)  attention:  13 dims (4.9%)    # 주의: 억제적 게이팅
Phi   smoothing: LBO (rank=32)      # 안정화: 전역 평탄화
```

이 비율은 CE 결합 상수에서 고정이므로 조정 불필요.

### 6.4 예측 점검 루프

실전에서는 아래 순서로만 해석해야 한다.

1. **예측 고정**
   - 활성 비율 중심 `4.87%`
   - 수면 루프 잔차 `1회 15.5%`, `2회 2.4%`, `3회 0.37%`
   - 곡률 제약은 hard bound가 아니라 안정화 편향
2. **A/B 측정**
   - Dense와 Sparse
   - Wake-only와 Sleep
   - Text-only와 Grounded
3. **게이트 판정**
   - 최적점이 `4-5%` 근방인가
   - sleep이 wake-only보다 drift를 줄이는가
   - grounded가 text-only보다 모달 불일치를 줄이는가
4. **실패 시 하향**
   - 맞지 않으면 CE 전체를 선언하지 말고, 해당 예측만 `bridge` 또는 `hypothesis`로 내린다

### 6.5 최소 체크리스트

| 항목 | 기대값 | 실패 시 해석 |
|---|---|---|
| `active_ratio` 스위프 | 최적점 `4-5%`, 실용 대역 `3-7%` | 과제 의존성이 더 큼 |
| sleep residual | `2-3`회 순환에서 급감 | 현재 구현의 동역학이 CE 최소 반복식과 다름 |
| curvature vs error | 양의 상관 | P5는 일반 안정화 regularizer에 가까움 |
| grounded vs text-only | grounding 오류 감소 | 결합 순서 또는 encoder 설계 재검토 |

---

## 7. GaugeLatticeV2: 채널 혼합 구현

`ce_gpt2.py`에 이미 구현된 V2 격자:

```python
class GaugeLatticeV2(nn.Module):
    def __init__(self, dim, mult=4, mix_rank=64):
        super().__init__()
        # ... (채널 분할은 V1과 동일)

        # 채널 간 저랭크 혼합 (섭동적)
        self.mix_down = nn.Linear(dim, mix_rank, bias=False)
        self.mix_up = nn.Linear(mix_rank, dim, bias=False)
        nn.init.zeros_(self.mix_up.weight)  # 0 초기화: 시작 시 혼합 없음

    def forward(self, x):
        # 블록 대각 전이
        y = concat(su3(x_3), su2(x_2), u1(x_1))
        # 섭동적 혼합 추가
        y = y + self.mix_up(self.mix_down(y))
        return self.phi(y)
```

`mix_up`을 0으로 초기화하므로, 시작 시 V1과 동일하다. 학습이 진행되면서 필요한 만큼 채널 간 혼합이 자동으로 학습된다.

---

## 8. 실전 팁

### 8.1 lambda_curv 선택

| 모델 규모 | 권장 lambda_curv | 이유 |
|---|---|---|
| Micro (~1M) | 0.01-0.05 | 작은 모델은 곡률 제약 강하게 |
| Small (~10M) | 0.005-0.01 | |
| Medium (~100M) | 0.001-0.005 | |
| Large (~1B+) | 0.0005-0.001 | 큰 모델은 자연 평탄화 경향 |

### 8.2 LBO rank 선택

$$r = \max(4,\; d / 8)$$

이 경험적 규칙이 대부분의 경우 작동한다. $r$이 너무 작으면 확산이 불충분하고, 너무 크면 파라미터 낭비.

### 8.3 Spectral Norm 주의사항

- 학습 초반에 spectral norm이 그래디언트를 불안정하게 만들 수 있다
- 해결: warmup 동안 spectral norm의 power iteration을 1회만 수행 (기본값)
- `nn.utils.spectral_norm(module, n_power_iterations=1)` (기본값이므로 변경 불필요)

### 8.4 메모리 최적화

CE 모듈의 추가 메모리 비용:
- LBONorm: $r \times d$ (V 행렬) -- 표준 LayerNorm 대비 미미
- GaugeLattice: 표준 FFN 대비 37% 감소
- Spectral Norm: 원래 크기와 동일 + $u, v$ 벡터 (미미)

총합: 표준 Transformer 대비 메모리 **감소**.
```
---
## File: `docs/8_리만/mra_block_spec.md`

```markdown
# Mellin-Riemann Attention Block (MRA) 정밀 사양

> 이 문서는 `riemann_pe_spec.md`의 후속이자 폐기 사양이다. 현재 `RiemannRotaryAttention`
> 은 RoPE 변형으로 정상 동작하지만, AGI 컨셉(`docs/7_AGI/2_Architecture.md`)의
> 5대 원리 중 절반(자유파라미터 0, 유니타리, Hilbert-Pólya, 게이지 격자, 부트스트랩
> sparsity)을 구현하지 않는다. MRA는 이 결함을 한 번에 메우는 단일 블록 사양이다.

## 0. 전제 (axioms)

- **A1 (RH).** ζ(s)의 모든 비자명 영점은 critical line `Re(s) = 1/2` 위에 있다.
  영점은 `s_n = 1/2 + i γ_n` 형태이며 `{γ_n}`은 GUE 통계를 따른다.
- **A2 (CE 결합 상수).** `α_s : α_w : α_em = 0.118 : 0.034 : 0.008` (`docs/3_상수`).
  이 비율은 채널 분할의 유일한 자유도다.
- **A3 (부트스트랩 고정점).** 활성 비율은 `ε² = 4.87 %`로 자연 수렴한다
  (`docs/6_뇌/07_수면과복구.md`). attention 행렬도 이 sparsity를 상한으로 갖는다.
- **A4 (유니타리).** `|det T|² ≤ 1`. 정보 증폭 = 환각이므로 출력 사영의 spectral
  norm은 1 이하로 제약한다.

위 네 axiom 위에서 attention 식 자체를 ζ explicit formula의 이산화로 유도한다.

## 1. Mellin–Riemann score

ζ explicit formula의 critical-strip 합은 다음과 같다.

$$
\sum_n \frac{x^{1/2 + i\gamma_n}}{\tfrac{1}{2} + i\gamma_n}
  = \sqrt{x}\,\sum_n \frac{e^{i\gamma_n \log x}}{\tfrac{1}{2} + i\gamma_n}.
$$

위치쌍 `(i, j)`에 `x = (1+i)/(1+j)`를 대입한다. dim-pair `k`는 복소채널로 압축한다.

$$
q_i^{(k)} := q_i^{2k} + i\,q_i^{2k+1},\qquad
k_j^{(k)} := k_j^{2k} + i\,k_j^{2k+1} \in \mathbb{C},
$$

attention raw score는 다음과 같다.

$$
\boxed{
S_{ij} \;=\; \sqrt{\dfrac{1+i}{1+j}}\;\sum_{k=0}^{K-1}
            \underbrace{\dfrac{1}{\tfrac{1}{2} + i\gamma_k}}_{w_k\;\text{(ζ amplitude)}}
            \;\underbrace{e^{i\gamma_k \log\tfrac{1+i}{1+j}}}_{\text{Mellin kernel}}
            \;q_i^{(k)} \overline{k_j^{(k)}}
}
$$

여기서 `K = d_head / 2`는 헤드의 복소채널 수다.

### 1.1 모듈화 (RoPE와 동일한 비용)

`(1+i)^{iγ_k}`와 `(1+j)^{iγ_k}`가 각각 `i`, `j`만의 함수이므로

$$
\tilde q_i^{(k)} \;=\; \sqrt{1+i}\;\, (1+i)^{\,i\gamma_k}\, q_i^{(k)},\qquad
\tilde k_j^{(k)} \;=\; \dfrac{1}{\sqrt{1+j}}\,(1+j)^{\,i\gamma_k}\, k_j^{(k)}
$$

으로 사전 변환하면

$$
S_{ij} \;=\; \sum_{k} w_k\,\tilde q_i^{(k)} \overline{\tilde k_j^{(k)}}.
$$

곧 표준 dot-product attention과 동일한 `O(N²K)` 비용이다. 추가 비용은 없다.

### 1.2 학습 자유도

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention과 동일 |

→ 표준 attention 대비 **추가 자유도 0**이다. 모든 새 항은 axiom에서 연역된다.

### 1.3 Real / Imag 사용

- `Re(S_{ij})` → softmax 입력 (실 attention)
- `Im(S_{ij})` → sheet 정보로 이미 표현됨. `floor(θ/2π)` 같은 별도 연산 불필요.

## 2. 채널 분할 (3x3+1 게이지 격자)

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

## 4. 유니타리 제약

출력 사영 `W_o`에 `nn.utils.spectral_norm`을 적용한다.

$$
W_o \leftarrow W_o / \sigma_1(W_o),\qquad \sigma_1(W_o) \le 1.
$$

attention 출력의 spectral norm이 1 이하가 되어 잔차 합 이후의 정보 증폭을 차단한다.

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

`LBONorm`, `GaugeLattice`는 `legacy examples/ai/clarus_lm.py` (removed)에 이미 구현되어 있다.

## 6. Hermitian 옵션 (Hilbert-Pólya 직접 구현)

`W_q = W_k`(tied projection)로 두면 `S_{ji} = S_{ij}^*`가 보장되어 attention
operator가 Hermitian이 된다. 영점 분포가 self-adjoint operator의 고유값이라는 H-P 추측을
직접 구현하는 setting이며, 옵션은 `hermitian=True`다.

## 7. 점근 / 안정성

- 작은 `p`에서 `log(1+p) ≈ p`이므로 기존 RoPE와 유사하다.
- 큰 `p`에서 `log(1+p)`가 천천히 증가하므로 frequency aliasing이 자동으로 완화된다.
- `N → kN`일 때 phase 평행이동만 발생하므로 relative attention이 보존된다.
- ζ 가중 `1/|1/2 + iγ_k|`가 고주파를 자동으로 감쇠해 학습 안정성을 높인다.

## 8. 백엔드 정책

PyTorch 참조 우선. Rust/CUDA 포트는 식이 안정화된 후 별도 작업으로 분리.

## 9. 참고

- Riemann (1859), *Über die Anzahl der Primzahlen unter einer gegebenen Größe*.
- Hilbert–Pólya conjecture (folklore).
- Berry & Keating (1999), *H = xp and the Riemann zeros*.
- Su et al. (2021), *RoFormer* — RoPE 원본.
- Press et al. (2022), *ALiBi*.
- `docs/7_AGI/2_Architecture.md` — ClarusBlock 5계층 stack.
- `docs/6_뇌/07_수면과복구.md` — 부트스트랩 고정점 `ε² = 4.87 %`.
```
---
## File: `docs/8_리만/mra_paper.md`

```markdown
# Mellin-Riemann Attention: ζ explicit-formula 유도 attention의 ablation 분석과 분산 절감, 그리고 Euler-CE의 length extrapolation 우위 발견

> 본 문서는 `mra_block_spec.md`의 사양과 `tests/test_mra.py`의 검증, 그리고
> `examples/ai/bench_recursive_euler.py --mode mra`와 `bench_mra_extrap.py`의
> ablation 실측을 한 편의 짧은 논문 형태로 정리한다. **정직한 negative-positive
> 혼합 결과**를 담은 internal write-up이다.

## Abstract

Riemann ζ 함수의 explicit formula의 critical-strip 합으로부터 attention
score를 직접 유도한 새로운 attention 변종 **Mellin-Riemann Attention
(MRA)**를 제안하고, 7개 설계 요소를 ablation으로 검증한다. 또한 비교
대상으로 기존 `euler_ce_k1`의 length extrapolation 성능을 처음으로 직접
측정한다.

`docs` Markdown 코퍼스 (400 K chars, vocab 788) 에서 char-level LM,
1500-step × 3-seed (in-distribution) + 1000-step × 3-seed (extrapolation):

* **부정적**: ζ 영점 `γ_k`를 attention frequency로 직접 사용하는 방식
  (Mellin 커널 `θ_k(p) = γ_k log(1+p)`)은 `γ_k/γ_1`의 범위가 ~5×에 불과해
  RoPE의 ~3000× 다중 스케일 해상도를 잃고, **PPL 14.29(RoPE 12.29 대비
  +2.0)**로 명확히 부진했다.
* **부정적**: ζ amplitude weighting (`w_k = 1/(1/2 + iγ_k)`) 단독 효과 미미
  (`mra` 12.46 vs `mra_noamp` 12.27, +0.17 손해).
* **부정적**: 부트스트랩 sparsity(`ε² = 4.87 %`)와 출력 spectral norm 모두
  1500-step 시점 PPL 손해 (+0.33).
* **긍정적 (분산)**: `MRABlock(decay_mode="bias")`가 평균 PPL 12.24(RoPE
  대비 -0.05)에 도달하고 **seed 분산이 RoPE의 절반(0.10 vs 0.21)**으로 줄었다.
* **긍정적 (외삽)**: `mra_bias`의 logarithmic distance bias가 4× length
  extrapolation에서 RoPE보다 **8 %p 안정**했다(degrad +18.5 % vs +26.7 %).
* **새로운 큰 발견 (4× extrap)**: `euler_ce_k1` 이 4× length extrapolation
  에서 **degrad +6.0 % 만 발생** (RoPE +26.7 %, MRA +18.5 % 대비 압도적).
* **결정적 분해 (32× extrap, 9 변종, § 7.7)**: 외삽 능력은 두 tier로 명확히
  갈린다. Tier 1 (외삽 OK, ≤ +10 %): `rope_alibi`, `euler_no_pi`, `euler_ce_k1`,
  `xpos`, `nope`. Tier 2 (외삽 BAD, +29 ∼ +55 %): `mra`, `mra_bias`, `std_rope`,
  `euler_no_decay`. **공통 패턴**: Tier 1 = "강한 distance attenuation 존재
  (linear/multiplicative)" OR "rotation 자체 부재". Tier 2 = "rotation 존재
  + attenuation 부재 또는 너무 약함 (log)".
* **Euler-CE 외삽 = ALiBi 재발견**: `euler_no_pi`(e-decay only) ≈ `euler_ce_k1`
  ≈ `rope_alibi`(32× 모두 -6 % 근방). π-rotation은 외삽에 기여하지 않았다.
* **NoPE > RoPE 7배 차이**(NoPE +7 % vs RoPE +47.2 %): RoPE의 회전 자체가
  OOD의 직접 원인임을 보였다. Kazemnejad et al. (2023)의 finding을 재현했다.

결론은 세 가지다. (a) ζ 영점을 frequency로 쓰는 직설적 시도는 실패했다.
(b) RoPE + ζ amp의 lean MRA는 평균 PPL은 동급이고 분산은 절반이다.
(c) **Length extrapolation의 구조적 결정성**: rotation을 그대로 두려면 강한
distance attenuation(linear ALiBi 또는 multiplicative xPos)이 필수이며,
그렇지 않으면 rotation 자체가 없는 NoPE가 낫다.

---

## 1. Introduction

Transformer의 positional encoding은 attention 메커니즘이 set-permutation
invariant라는 근본 한계를 외부 주입으로 해결하는 inductive bias의 핵심
지점이다. RoPE [1]가 LLaMA / Qwen / Mistral / Gemma / EXAONE 등 거의 모든
현대 LLM의 표준이 된 후, ALiBi [2], YaRN [3], LongRoPE [4] 등 후속 연구는
대체로 경험적 튜닝으로 long-context extrapolation을 공략해왔다.

본 연구의 출발점은 다른 방향이다. **수학적 axiom에서 attention 식을 직접
유도**하면 어떻게 되는가? Hilbert-Pólya 추측 [5]은 Riemann ζ의 비자명
영점 `{ρ_n = 1/2 + iγ_n}`의 허수부가 어떤 self-adjoint operator의 고유값
임을 가정한다. Berry-Keating [6]은 이 operator가 `H = (xp + px)/2`의
양자화일 가능성을 제시했다.

본 연구는 RH(Riemann Hypothesis)를 **engineering axiom**으로 채택하고,
attention score를 ζ의 explicit formula의 critical-strip 합으로부터 유도
한다. 식 자체는 단순하지만, 어느 부분이 실제로 도움이 되고 어느 부분이
도움이 되지 않는지는 **실증으로만 확인 가능**하다. 본 논문의 기여는 이를
ablation으로 분리한 데에 있다.

### 1.1 Contributions

1. ζ explicit formula의 critical-strip 합으로부터 attention score를
   폐쇄형으로 유도 (§ 3).
2. 7가지 설계 knob(frequency 방식, amplitude weighting, decay 형태,
   sparsity, spectral norm, Hermitian)을 노출하는 단일 ablation 표면 (§ 4).
3. **부정적 발견 명시**: 직설적 ζ-frequency, multiplicative decay,
   bootstrap sparsity, spectral-norm 모두 1500-step 시점 LOSS (§ 6).
4. **긍정적 발견**: lean variant(`mra` / `mra_bias`)는 RoPE와 평균 PPL
   동급 + **seed 분산 2-3× 감소** (§ 7).

---

## 2. Background

### 2.1 RoPE

`d_head` 차원의 dim-pair `(2k, 2k+1)`에 대해 회전 행렬

```
R_θ = [[cos θ, -sin θ], [sin θ, cos θ]],   θ_k(p) = p · 10000^{-2k/d}
```

을 query/key에 적용한다. 두 토큰의 attention score가 상대 위치 `i - j`만의
함수가 되도록 만든다. base frequency 10000은 경험적 선택이다.

핵심 강점은 `inv_freq_k = 10000^{-2k/d}`의 **기하 급수적 분포**가 attention
에 폭넓은 다중 스케일 해상도를 부여한다는 점이다 (`d_head = 16` 시 약 3000×
범위).

### 2.2 ALiBi

attention score에 `−|i − j|/m_h`의 거리 감쇠 bias를 직접 더한다(`m_h`는
헤드별 기울기). 본 논문의 `decay_mode="bias"`는 ALiBi를 logarithmic
거리 함수 `½ log((1+i)/(1+j))`로 일반화한 형태에 해당한다.

### 2.3 ζ explicit formula

von Mangoldt:

```
ψ(x) = x − Σ_ρ x^ρ / ρ − log(2π) − ½ log(1 − x^{-2})
```

핵심 합 `Σ_ρ x^ρ / ρ`는 RH 하에서

```
Σ_n x^{1/2 + iγ_n} / (1/2 + iγ_n) = √x · Σ_n e^{iγ_n log x} / (1/2 + iγ_n)
```

으로 정리된다. **Mellin 커널** `e^{iγ_n log x}`와 **ζ 진폭** `1/(1/2 + iγ_n)`
의 곱이 자연스럽게 나타난다.

---

## 3. Method - Mellin-Riemann Attention

### 3.1 단순 score 식

위치쌍 `(i, j)`에 `x = (1+i)/(1+j)`를 대입한다. dim-pair `k`를 복소채널로
압축하면

```
q_i^(k) := q_i^{2k} + i · q_i^{2k+1},     k_j^(k) := k_j^{2k} + i · k_j^{2k+1},
```

attention raw score는 다음과 같다.

```
S_{ij} = √((1+j)/(1+i)) · Σ_k w_k · e^{−iγ_k log((1+i)/(1+j))} · q_i^(k) · conj(k_j^(k))
```

여기서 `w_k = 1/(1/2 + iγ_k)`. 토큰별 attention 입력은 `Re(S_{ij})`.

### 3.2 폐쇄형 분해

`(1+i)^{−iγ_k}`와 `(1+j)^{−iγ_k}`가 각각 `i`, `j`만의 함수이므로 사전 변환

```
q̃_i^(k) = (1/√(1+i)) · e^{−iγ_k log(1+i)} · q_i^(k)
k̃_j^(k) = √(1+j) · e^{−iγ_k log(1+j)} · k_j^(k)
```

후

```
S_{ij} = Σ_k w_k · q̃_i^(k) · conj(k̃_j^(k))
```

이는 표준 dot-product attention과 동일한 `O(N²K)` 계산이다. 실수 분해는 다음과 같다.

```
q̂_re = w_re · q̃_re − w_im · q̃_im
q̂_im = w_re · q̃_im + w_im · q̃_re
Re(S) = q̂_re @ k̃_re^T + q̂_im @ k̃_im^T
```

→ 표준 attention의 **2배 matmul** 비용이다. softmax / V 적용은 동일하다.

### 3.3 학습 자유도

| 양 | 형상 | 자유도 |
|---|---|---|
| `γ_k` | buffer | 0 (RH axiom) |
| `w_k = 1/(1/2 + iγ_k)` | buffer | 0 (RH axiom) |
| `cos_p, sin_p, log_decay` | buffer | 0 (위치-axiom) |
| `W_q, W_k, W_v, W_o` | learnable | 표준 attention 동일 |

표준 multi-head attention 대비 **추가 학습 파라미터 0**.

---

## 4. 설계 Knob - Ablation 표면

`MellinRiemannAttention(d_model, n_heads, block, **knobs)`가 노출하는 7개
knob은 다음과 같다.

| knob | 값 | 기본 | 의미 |
|---|---|---|---|
| `freq_mode` | `"rope"` / `"zeta_log"` | `"rope"` | 위치 주파수 형태 |
| `amp_weight` | `True` / `False` | `True` | ζ 진폭 weighting `w_k` 사용 |
| `decay_mode` | `"none"` / `"bias"` / `"mult"` | `"none"` | critical-line decay 형태 |
| `sparse_eps2` | `0.0` ∼ `0.0487` | `0.0` | top-k retention 비율 |
| `hermitian` | `False` / `True` | `False` | `W_q = W_k` tied + score symmetrize |
| `spectral_norm_o` | `False` / `True` | `False` | `σ₁(W_o) ≤ 1` 강제 |
| `rope_base` | f32 | `10000` | RoPE base freq (freq_mode="rope" 일 때만) |

### 4.1 `freq_mode`

- `"rope"`: `θ_k(p) = p · base^{−k/K}`. RoPE 기하 급수(~3000× 범위).
- `"zeta_log"`: `θ_k(p) = γ_k · log(1+p)`. Mellin 커널이며, ζ 영점이 곧 frequency다.
  `γ_k/γ_1`의 범위가 K=16에서 ~5×에 불과해 다중 스케일 해상도가 손실된다.

### 4.2 `amp_weight`

- `True`: `w_k = 1/(1/2 + iγ_k)`를 score에 곱한다. `Re(w_k)·Re_part −
  Im(w_k)·Im_part` 형태로 standard RoPE의 `cos` 채널과 보조 `sin` 채널이
  혼합된다.
- `False`: `w_k = 1`. attention 식이 표준 RoPE와 일치한다(control).

### 4.3 `decay_mode`

ζ explicit-formula의 `√x` factor(`x = (1+i)/(1+j)`)를 처리한다.

- `"none"`: 무시.
- `"bias"`: log-space additive. `bias_{ij} = ½(log(1+j) − log(1+i))`를 score에 더한다.
  ALiBi의 logarithmic 일반화다.
- `"mult"`: 원식 그대로 `score_{ij} *= √((1+j)/(1+i))`를 적용한다. Per-key temperature
  처럼 작동하므로 의도와 다소 어긋난다.

### 4.4 `hermitian` (causal LM에서 비추천)

`W_q = W_k` tied + score symmetrize `S ← (S + Sᵀ)/2` 구조다. Bidirectional encoder에서는
self-adjoint operator의 이산화(Hilbert-Pólya 직접 구현)이지만, **causal
LM에서는 mask 전 symmetrize가 future leakage를 일으킨다**. 본 작업의 ablation
표에서 이 knob은 제외했다.

---

## 5. Implementation

`reality_stone/python/reality_stone/clarus/ce_mra.py`는 PyTorch 참조 구현(~250 LOC)이다. 현재 backend는 PyTorch only이며,
`RiemannRotaryAttention`의 Rust/CUDA 패턴을 그대로 이식할 수 있다.

테스트 (`tests/test_mra.py`, 11/11 pass):

- forward shape & finiteness
- block end-to-end
- 모든 ablation knob 조합 finite output
- bootstrap sparsity 정확한 retention 수
- 학습 파라미터 수 == 표준 MHA (4·d²) / Hermitian 모드 == 3·d²
- ζ amplitude가 실제로 출력을 바꾸는지 확인(회귀 가드)
- 공리적 buffer들이 학습되지 않는지 확인
- spectral norm이 σ₁ ≤ 1을 강제하는지 확인

---

## 6. Experiments - Setup

| 항목 | 값 |
|---|---|
| 코퍼스 | `docs/` Markdown (400 K chars, vocab = 788) |
| 분할 | 90 % train / 10 % val |
| 토크나이저 | char-level |
| 모델 | 2-layer × 4-head, `d_model = 64` |
| 파라미터 | ≈ 210 K |
| 시퀀스 | block = 64, batch = 8 |
| 옵티마이저 | AdamW, lr = 3e-4, grad clip 1.0 |
| 학습 | 1500 step × 3 seed |
| 평가 | 마지막 12 batch 평균 cross-entropy → exp |
| 하드웨어 | RTX 4060 (CUDA) |

### 6.1 Variants

| 이름 | freq_mode | amp_weight | decay_mode | 기타 |
|---|---|---|---|---|
| `std_rope` | RoPE | — | — | baseline (`RoPEAttnBlock`) |
| `euler_ce_k1` | π-rotation | — | e-decay | 기존 winner [9] |
| **`mra`** | rope | True | none | primary lean MRA |
| `mra_noamp` | rope | False | none | ablate ζ amplitude |
| `mra_zeta` | zeta_log | True | none | ablate frequency |
| `mra_bias` | rope | True | bias | + additive log decay |
| `mra_mult` | rope | True | mult | + multiplicative decay |
| `mra_sparse` | rope | True | none | + ε² = 0.0487 sparsity |
| `mra_sn` | rope | True | none | + spectral_norm(W_o) |

---

## 7. Results

### 7.1 1500-step × 3-seed PPL ablation

| variant | params | PPL | **σ_PPL** | time/seed (s) |
|---|---|---|---|---|
| `std_rope` | 210 K | 12.288 | 0.208 | 13.0 |
| `euler_ce_k1` | 210 K | **11.864** | 0.126 | 16.1 |
| **`mra`** | 210 K | 12.458 | **0.069** | 13.0 |
| `mra_noamp` | 210 K | 12.266 | 0.129 | 12.8 |
| `mra_zeta` | 210 K | **14.286** | 0.332 | 12.8 |
| **`mra_bias`** | 210 K | **12.240** | **0.097** | 13.0 |
| `mra_mult` | 210 K | 12.259 | 0.077 | 13.8 |
| `mra_sparse` | 210 K | 12.621 | 0.232 | 14.1 |
| `mra_sn` | 210 K | 12.621 | 0.172 | 18.7 |

### 7.2 Verdicts (`std_rope`, `euler_ce_k1` 대비)

| 비교 | z | Δ PPL | 판정 |
|---|---|---|---|
| `mra_zeta` vs `std_rope` | +8.84 | +1.999 | **LOSS** (γ-freq 붕괴 확정) |
| `mra` vs `std_rope` | +1.35 | +0.170 | LOSS (mean), 분산 −67 % |
| `mra_noamp` vs `std_rope` | −0.16 | −0.022 | TIE (≈ pure RoPE) |
| `mra_bias` vs `std_rope` | −0.36 | −0.048 | TIE+ |
| `mra_mult` vs `std_rope` | −0.22 | −0.028 | TIE+ |
| `mra_sparse` vs `std_rope` | +1.86 | +0.334 | LOSS |
| `mra_sn` vs `std_rope` | +2.14 | +0.334 | LOSS |
| `euler_ce_k1` vs `std_rope` | (best baseline) | −0.424 | WIN |

### 7.3 부정적 발견

1. **`mra_zeta` (PPL 14.29)**: ζ 영점을 frequency로 직접 사용하면 RoPE의
   다중 스케일 해상도가 깨진다. `γ_k/γ_1`의 범위는 K=16에서 ~5×에 불과하다.
   RoPE의 `10000^{−k/K}`가 갖는 ~3000× 범위와 비교하면 다중 스케일 해상도가
   거의 없다. 이 한계는 `riemann_rope`의 PPL 19.29(별도 측정)의 직접 원인이며,
   ζ 영점을 frequency로 직접 사용하는 모든 변종에서 재현된다.
2. **ζ amplitude 단독 효과 미미**: `mra` (12.46) − `mra_noamp` (12.27) =
   +0.17 PPL **손해**(z = +1.35). amp_weight가 표준 RoPE attention에서
   별다른 정보 이득을 주지 못한다.
3. **부트스트랩 sparsity / spectral norm**: 1500 step 충분 수렴 시점에서는 모두
   PPL 손해(+0.33)를 보였다. 짧은 학습(200 step)에서는 도움이지만 long-run에서는 over-
   regularization.

### 7.4 긍정적 발견 - seed 분산 절감

mean PPL은 거의 동등하지만 seed-to-seed 분산은 명확히 감소한다.

```
σ_PPL across 3 seeds (lower = more reproducible):
  std_rope     : 0.208         baseline
  mra          : 0.069         3.0×  more stable
  mra_mult     : 0.077         2.7×
  mra_bias     : 0.097         2.1×
  mra_noamp    : 0.129         1.6×
  euler_ce_k1  : 0.126         1.7×  (다른 메커니즘)
```

`mra` 계열(RoPE freq + amp/decay 변종)은 모두 **σ_PPL이 RoPE의 1/3 ∼ 1/2** 수준이다.
`mra_noamp`도 분산이 작은 것으로 보아, 분산 감소는 ζ amplitude 단독 효과가
아니라 RoPE의 `qkv` fused linear 대신 **`W_q, W_k` 분리 + RoPE-style
회전**을 조합한 데서 나오는 것으로 추정된다(추가 조사 필요).

### 7.5 가장 좋은 lean 변종

`decay_mode="bias"` (`mra_bias`):
- mean PPL 12.240: **RoPE보다 0.05 낮음**(z = −0.36, statistically TIE
  with mean preference)
- σ_PPL 0.097: RoPE 분산의 47 %
- 시간 비용은 RoPE와 동일

이는 ALiBi의 logarithmic 일반화로 해석할 수 있다.
```
bias_{ij} = ½ (log(1+j) − log(1+i))
```
선형 거리 `−|i−j|`가 아닌 logarithmic 거리다. critical line `Re(s) = 1/2`의
지수 `½`가 직접 도출된다.

### 7.6 Length extrapolation (4×까지)

train block = 64, eval block ∈ {64, 96, 128, 192, 256} (4× 외삽까지),
1000 step × 3 seed:

| variant | N=64 | N=96 | N=128 | N=192 | N=256 | **degrad @ 4×** |
|---|---|---|---|---|---|---|
| `std_rope`    | 14.99 | 16.89 | 14.20 | 17.24 | 18.98 | **+26.7 %** |
| `euler_ce_k1` | 14.75 | 16.76 | 13.90 | 15.51 | 15.63 | **+6.0 %**  ← BEST |
| `mra`         | 15.42 | 17.46 | 14.76 | 17.14 | 18.36 | +19.0 % |
| `mra_bias`    | 14.96 | 17.06 | 14.39 | 16.62 | 17.72 | **+18.5 %** |
| `mra_zeta`    | 17.39 | 19.76 | 16.93 | 18.67 | 19.16 | +10.2 % |

발견:

1. **`euler_ce_k1`의 extrapolation이 압도적**이다. 4× 외삽에서 PPL은 +6 %만
   증가했다. 다른 모든 변종(RoPE 포함)이 +18 ~ +27 %인 것과 명확히 다른 영역이다.
   Euler-CE의 `e^{-|i-j|/ξ}` decay가 정확한 long-context 인덕티브 바이어스를
   제공하는 것으로 해석된다.
2. **`mra_bias`의 logarithmic decay는 RoPE보다 8 %p 더 부드럽게 degrade한다**
   (+18.5 % vs +26.7 %). lean MRA의 자그마한 long-context 우위다. 다만
   `euler_ce_k1` 만큼은 아님.
3. **`mra_zeta`는 짧은 base에서는 부진해도 외삽은 비교적 잘한다**(+10.2 %).
   `log(1+p)` lift의 진짜 가치는 long-context에서 발현된다. 짧은 학습 + 긴
   외삽 시나리오에 한정해서는 검토 가치 있음.
4. **`std_rope`의 빠른 붕괴**(+26.7 %): 알려진 RoPE periodicity 한계를 재현했다.

이 결과는 logarithmic distance bias가 ALiBi의 일반화로 의미 있는 후속 연구
방향임을 시사한다.

### 7.7.1 Scaling: d_model {64, 128, 256}에서 패턴 유지

§ 7.7의 결과를 모델 크기 scaling에 대해 검증한다. 학습 설정은 동일하다
(train block = 64, 1500 step). seeds는 d=64에서 3, d=128/256에서 메모리상 2다.

#### 32× extrapolation 상대 degradation, d_model 별

| variant | d=64 | **d=128** | d=256 | 부류 |
|---|---|---|---|---|
| `rope_alibi` | −6.2 % | **−3.0 %** | −5.1 % | Tier 1 |
| `euler_ce_k1` | −6.2 % | **−3.0 %** | −5.3 % | Tier 1 |
| `euler_no_pi` | −5.5 % | −4.0 % | −5.1 % | Tier 1 |
| `xpos` | +4.1 % | +7.4 % | +0.8 % | Tier 1 |
| `nope` | +7.0 % | +15.6 % | +10.5 % | Tier 1 |
| `mra_bias` | +40.3 % | +59.0 % | +42.8 % | Tier 2 |
| `mra` | +40.8 % | +65.9 % | +43.7 % | Tier 2 |
| `std_rope` | +47.2 % | **+73.5 %** | +48.3 % | Tier 2 |
| `euler_no_decay` | +54.7 % | **+103.6 %** | +54.2 % | Tier 2 |

#### 발견

1. **Tier 1/Tier 2 분리는 모든 d_model에서 견고**하다. d=64, 128, 256의 세
   가지 모델 크기 모두 ALiBi 계열 (`rope_alibi`, `euler_ce_k1`, `euler_no_pi`)
   은 안정적이고(−3 ∼ −6 %), rotation-only 계열(`std_rope`, `euler_no_decay`,
   `mra`, `mra_bias`)은 catastrophic하다(+42 ∼ +104 %). 본 작업의 핵심 가설
   "rotation + 강한 distance attenuation이 외삽의 충분조건"은 모델 크기에
   invariant하다.
2. **d_model = 128에서 cliff가 가장 깊다**. rotation-only 변종이 +73 ∼
   +104 %로 가장 큰 degradation을 보였다. 가능한 해석은 중간 사이즈가 학습 분포에
   가장 tightly fit하여 OOD 영역과의 격차가 가장 크다는 것이다. d=256은 1500 step에서
   underfit으로 보인다.
3. **ALiBi의 안정성은 모델 크기에 무관**하다. d=64 −6.2 %, d=128 −3.0 %,
   d=256 −5.1 %다. Slope 학습이 단순하므로 sample efficiency가 좋아 모든
   사이즈에서 robust하다. **이는 ALiBi가 production-scale 모델에서도 유효한
   외삽 메커니즘일 강한 후보임을 시사**.
4. **NoPE의 OOD 안정성은 모델이 커질수록 약화**된다. d=64 +7 %, d=128 +15.6 %,
   d=256 +10.5 %다. 모델이 implicit position을 더 강하게 학습할수록 OOD 영역에서의
   generalization이 약해진다. NoPE는 baseline으로는 의미 있지만 production에는 부적합하다.

### 7.7.2 Length extrapolation (32×까지): Euler-CE 분해

위 § 7.6의 발견(Euler-CE의 외삽 우위)의 진짜 원인을 격리한다. EulerCE는
두 개의 분리 가능한 구조를 결합하므로 각각을 frozen으로 끄고 비교한다.

| 변종 | 구성 | 학습 자유도 |
|---|---|---|
| `std_rope` | RoPE | baseline |
| `rope_alibi` | RoPE + ALiBi linear decay (per-head learnable slope) | + n_heads |
| `mra` / `mra_bias` | RoPE + ζ amplitude / + log decay | + n_heads (`mra_bias`) |
| `euler_no_decay` | EulerCE with `e_gate` frozen → π-rotation 만 | (gate 1개 frozen) |
| `euler_no_pi` | EulerCE with `pi_gate` frozen → e-decay 만 | (gate 1개 frozen) |
| `euler_ce_k1` | full Euler-CE | original |

train block = 64, eval block ∈ {64, 128, 256, 512, 1024, 2048} (최대 32×).
모든 모델은 train_block으로 init하고, eval 직전 `extend_to(N_eval)`로 buffer만
grow한다. EulerCE의 block-aware base `π^D_eff · N`도 train 시점 N=64로
계산.

1500 step × 3 seed × batch=4(메모리 한계)로 측정했다. 절대 PPL은 짧은 학습 + 작은
batch 때문에 § 7.6보다 높지만, **상대 degradation은 비교 가능**하다.

#### Relative degradation (PPL(N)/PPL(64) − 1, 낮을수록 좋음)

| variant | ×2 | ×4 | ×8 | ×16 | **×32** | tier |
|---|---|---|---|---|---|---|
| **`rope_alibi`** | −13.0 % | −3.3 % | −9.2 % | −7.6 % | **−6.2 %** | **1** |
| **`euler_no_pi`** (e only) | −12.0 % | −2.4 % | −7.9 % | −6.5 % | **−5.5 %** | **1** |
| **`euler_ce_k1`** | −13.1 % | −3.4 % | −9.5 % | −7.9 % | **−6.2 %** | **1** |
| **`xpos`** | −12.3 % | −2.0 % | −5.8 % | −1.2 % | **+4.1 %** | **1** |
| **`nope`** | −7.4 % | +3.2 % | +1.8 % | +6.0 % | **+7.0 %** | **1** |
| `mra_bias` | −8.8 % | +10.0 % | +13.9 % | +28.6 % | +40.3 % | 2 |
| `mra` | −9.2 % | +10.4 % | +14.1 % | +29.7 % | +40.8 % | 2 |
| `std_rope` | −10.0 % | +12.6 % | +19.9 % | +37.7 % | **+47.2 %** | 2 |
| `euler_no_decay` (π only) | −8.2 % | +14.6 % | +24.3 % | +44.4 % | **+54.7 %** | 2 |

#### 결정적 분해: Tier 1과 Tier 2

| Tier | 기준 | 메커니즘 |
|---|---|---|
| **1 (외삽 OK, ≤ +10 %)** | rope_alibi, euler_no_pi, euler_ce_k1, xpos, nope | distance attenuation **존재** OR rotation **부재** |
| **2 (외삽 BAD, +29 ∼ +55 %)** | mra, mra_bias, std_rope, euler_no_decay | rotation **존재** + 강한 distance attenuation **부재** |

이 분리는 다섯 개의 깨끗한 결론을 만든다.

1. **Euler-CE의 외삽 우위는 100 % e-decay(linear additive distance bias) 단독 효과**다.
   `euler_no_pi`(e-decay only)는 풀 `euler_ce_k1`과 거의 같다(32×에서 −5.5 % vs −6.2 %).
   π-rotation을 빼도 외삽 성능은 유지된다.
2. **π-rotation 단독은 외삽에 무력**하다. `euler_no_decay`(π only)는 32×에서
   +54.7 %로 RoPE보다 오히려 더 나쁘다. block-aware base `π^D_eff · N`도
   long-context 안정성에는 무관하다(in-distribution에서는 도움일 수 있음).
3. **`rope_alibi` ≈ Euler-CE**다. 32× 외삽은 −6.2 % vs −6.2 %다. EulerCE의
   e-decay는 ALiBi(Press 2022)의 per-head learnable slope 변종으로 정확히
   환원된다. **EulerCE의 외삽 우위는 ALiBi의 재발견**이다.
4. **xPos도 외삽 OK**다(+4.1 %). multiplicative magnitude decay도 linear
   additive 정도는 아니지만 명확히 작동한다. distance attenuation의 *형태*보다
   존재 자체가 더 결정적.
5. **NoPE가 RoPE보다 외삽을 잘한다**(+7.0 % vs +47.2 %, ×7 차이). 회전 자체가
   OOD의 직접 원인임을 본 setup에서 재현했다(Kazemnejad et al. 2023).

종합 결론은 **OOD 외삽 = "강한 distance attenuation 존재" 또는 "rotation 부재"의 OR 조건**이다.
RoPE의 회전을 그대로 두고 attenuation만 빼거나 약하게 얹는 것(`mra`, `mra_bias`, `std_rope`)이 가장 위험하다.
logarithmic decay가 linear보다 약한 이유는 long-distance에서 너무 천천히 감쇠해 RoPE 회전의
wrap-around 효과를 dominate하지 못하기 때문이다.

### 7.7.3 2-bit Minimal Euler-CE: operational 환원

§ 7.7의 ablation은 강력한 시사를 만든다. Euler 5상수 `{e, π, i, 1, 0}`의
attention적 의미는 두 axis와 두 게이트로 환원된다.

| 상수 | 작용 | 환원 |
|---|---|---|
| `π`, `i` | rotation generator (`e^{iπt}` 결합) | **axis 1**: rotation |
| `e` | exponential decay base | **axis 2**: decay |
| `1`, `0` | on/off gate values | 1 비트 each |

→ **2 functionally distinct axes × 2 gate values = 2² = 4 head-types**, 각각
2-bit string `(pi, e)`으로 인코딩할 수 있다.

| (pi, e) | 비트 | 헤드 타입 | 문헌 매핑 |
|---|---|---|---|
| (0, 0) | `00` | identity | NoPE [Kazemnejad 2023] |
| (0, 1) | `01` | decay only | ALiBi [Press 2022] |
| (1, 0) | `10` | rotation only | RoPE [Su 2021] |
| (1, 1) | `11` | rotation + decay | xPos [Sun 2023] / Euler-CE |

**§ 7.7 결과의 perfect 매핑**: 4 head-types 중 3개(00, 01, 11)가 Tier 1이고,
단 한 가지(10)만 Tier 2다. 즉 **5 Euler 상수 = 2비트 head-type taxonomy**이며,
**4가지 중 3가지가 작동**한다(= log₂ 3 ≈ 1.58비트의 effective capacity).

#### 구현: `EulerCEMinimal`

`reality_stone/python/reality_stone/clarus/ce_euler.py::EulerCEMinimal`가 본 환원의 정밀 구현이다. 각 헤드는 2비트
spec `head_types ∈ {0, 1, 2, 3}^H`로 head-type을 axiom으로 선택한다. 학습
파라미터는 `xi_h`(decay 헤드만 의미 있음) 1개 + `W_q,k,v,o`(표준 MHA)다.
5차원 `bit_logits`와 sigmoid 게이트 학습은 모두 제거했다.

#### 자유도 비교

| 양 | 원본 EulerCE | EulerCEMinimal |
|---|---|---|
| Discrete head-type | 5-dim continuous (`bit_logits`) | 2-bit axiom (`head_types`) |
| Rotation gate | learnable sigmoid (`pi_gate_logit`) | bit (axiom) |
| Decay gate | learnable sigmoid (`e_gate_logit`) | bit (axiom) |
| Decay length | learnable (`log_xi`) | learnable (`log_xi`) |
| 헤드별 학습 자유도 | 5 + 3 = **8** | 1 (xi only) |

→ **헤드별 학습 자유도가 8 → 1로 감소**한다(head-type 선택을 axiom으로 이동).
이는 Clarus 본 thesis("자유 파라미터 0에 가깝게")와 직접 정렬된다.

#### 검증 — `tests/test_euler_minimal.py` (16 tests)

* `head_types_from_spec`가 6가지 spec 형태(int, str, list, "mix", "all",
  invalid) 모두 정확히 처리.
* `head_types="nope"` → no-PE와 수치적으로 일치(allclose atol=1e-5).
* `head_types="alibi"` → 거리에 따른 attention attenuation 확인.
* `head_types="rope"` → 모든 헤드 e_bit=0, decay 항 0.
* `head_types="mix"` / `"all"` → 정확한 비트 분해.
* `extend_to(N)` 후 forward 정상, 학습 파라미터 변하지 않음.
* Autograd: rotation-only 헤드의 `log_xi` grad는 정확히 0(decay gate off 시
  grad path가 끊김).

이로써 EulerCEMinimal이 본 환원의 정확하고 정밀한 구현임을 확인했다.

#### 실증: minimal 변종이 canonical baseline을 정확히 reproduce

`bench_mra_extrap.py --variants minimal` (1500 step × 3 seed × d_model=64) 결과:

| 2-bit variant | head_types | x32 degrad | canonical 매핑 | x32 (canonical) |
|---|---|---|---|---|
| `min_nope` | uniform 00 | **+7.0 %** | `nope` | +7.0 % (정확 일치) |
| `min_rope` | uniform 10 | **+47.2 %** | `std_rope` | +47.2 % (정확 일치) |
| `min_alibi` | uniform 01 | **−5.7 %** | `rope_alibi` | −6.2 % (≈ 일치) |
| `min_xpos` | uniform 11 | **−5.8 %** | `euler_ce_k1` | −6.2 % (≈ 일치) |
| `min_mix` | half 01, half 11 | −5.5 % | (신규) | — |
| `min_all` | one of each 00/01/10/11 | −0.6 % | (신규) | — |

**검증 결과**: 5 Euler 상수의 2-bit reduction이 정확히 작동한다. 4 canonical
positional encoding(NoPE, RoPE, ALiBi, xPos)이 모두 `EulerCEMinimal`의
단일 head-type spec으로 정확히(≤ 0.5 %p) reproduce된다.

**부수 발견**:

* **`min_mix`(half ALiBi + half xPos) ≈ 단독 best와 동등**하다(−5.5 % vs −5.7
  ∼ −5.8 %). 두 tier-1 type을 섞어도 외삽 우위는 없으므로 실용적으로 head-type
  diversity는 도움이 되지 않는다.
* **`min_all`(4 type 골고루)은 best보다 약간 worse**다(−0.6 %). rotation-only와
  nope 헤드가 평균 attention capacity를 깎는다. **실용적 권고**: 모든 헤드를
  단일 tier-1 type(`alibi` 또는 `xpos`)으로 통일하는 것이 최적이다.

이는 본 작업의 가장 강한 결정적 결론이다. **현존하는 모든 main-stream PE는
사실상 같은 2-bit taxonomy의 인스턴스이며, 4가지 중 단 한 가지(10 = pure
rotation)만 외삽 catastrophic**이다.

---

## 8. ML 이론 함의: 정직한 평가

### 8.1 무엇이 작동했나

* **Logarithmic distance bias**(ζ explicit-formula의 `√x`로부터 유도)는
  ALiBi의 합리적 일반화다.
* **Seed 분산 감소**(RoPE 대비 50-70 % 감소)는 reproducibility 관점에서
  실용적 가치가 있다. 큰 모델 / 비싼 학습에서 한 자릿수 작은 분산은 평균
  0.05 PPL 우위보다 훨씬 큰 비용 절감.

### 8.2 무엇이 작동하지 않았나

* **ζ 영점을 frequency로 직접 사용(`zeta_log`)**: 이론적으로 매력적이지만
  실측에서 다중 스케일 해상도 손실로 인한 명확한 LOSS. 이 결과는 "Riemann
  영점이 LM positional structure를 직접 부여한다"는 strong한 가설이 적어도
  소규모에서는 성립하지 않음을 시사한다.
* **ζ amplitude weighting 단독**: 식적으로는 RoPE의 `cos` 채널에 `sin` 채널을
  섞는 효과지만, 실측 영향은 거의 없거나 약간 부정적이다.
* **부트스트랩 sparsity, spectral norm**: 작은 모델 / 짧은 학습에서는 도움이지만
  충분 수렴 시점에서는 over-regularization으로 작동한다. CE 부트스트랩 비율이
  LM attention에 직접 transfer되지 않는다는 시사다.

### 8.3 어떤 가설이 다음 단계에서 검증 가능한가

* **Length extrapolation**: 짧은 context 학습 후 긴 context에서의 안정성.
  `mra_bias`의 logarithmic decay는 RoPE의 wrap-around 한계를 넘어설 가능성
  이 있다. 별도 측정 필요.
* **대형 모델 분산**: 분산 감소가 모델 크기와 함께 어떻게 변하는지.
* **Attention 스펙트럼**: 학습된 MRA의 attention 행렬 고유값 분포가 GUE
  통계 [13]를 따르는지 직접 측정.

---

## 9. Related Work

- **RoPE** [1]: 회전 기반 상대위치, 본 연구의 직접적 baseline.
- **ALiBi** [2]: 거리 감쇠 bias. `mra_bias`의 logarithmic 변종이 이를 일반화한다.
- **xPos** [10]: RoPE + exponential decay. ad-hoc 결합.
- **YaRN / LongRoPE** [3, 4]: 경험적 frequency rescaling.
- **Geometric Deep Learning** [11]: 기하학적 inductive bias 일반.
- **Equivariant Transformers** [12]: 대칭성 내장.

본 연구는 ζ 영점이라는 **수론** 객체에서 식을 직접 유도하려 시도했다는 점에서 새롭다.
결과는 mixed다. ζ amplitude와 logarithmic decay는 작은 효과를 보였고,
ζ frequency는 negative 효과를 보였다.

---

## 10. Limitations & Future Work

### 10.1 한계

1. **소규모 실험**: 210 K 파라미터, 400 K char 코퍼스다. 본격 결론을 위해서는
   1 M+ 파라미터 × 1 M+ 토큰 × 5+ seed가 필요하다.
2. **Char-level only**: BPE / sentencepiece 토큰화에서의 거동 미검증.
3. **Length extrapolation 미정밀 측정**: 별도 실험 필요.
4. **Native 백엔드 부재**: PyTorch-only, wall-clock 비교 손해. Rust/CUDA 포팅
   가능.
5. **Causal LM 가정**: bidirectional encoder에서는 Hermitian knob의 본격
   효과 검증 가능하나 본 작업 범위 밖.

### 10.2 후속

1. Length extrapolation: train N=64, eval N={128, 256, 512}에서
   `mra_bias`의 logarithmic decay가 RoPE의 periodic wrap보다 실제로 잘
   버티는지.
2. Scale law: `d_model ∈ {128, 256, 512}` × `n_layers ∈ {4, 8, 12}`에서
   분산 감소 효과의 안정성.
3. Attention spectrum: 학습된 attention의 spectral statistics 측정.
4. Bidirectional MRA: `hermitian=True` 모드를 BERT-style 인코더에서 검증.
5. GaugeLattice FFN 결합: AGI 5대 원리 중 채널 분할까지 묶은 full ClarusBlock.

---

## 11. Conclusion

본 작업의 핵심 결론은 세 가지다.

1. **ζ 영점을 attention frequency로 직접 사용하는 시도는 실패**했다. γ_k의
   범위가 RoPE의 다중 스케일 해상도를 따라가지 못한다. 본격 검증 후 폐기.
2. **분산 감소**: lean MRA 변종들이 RoPE의 1/3 ~ 1/2 σ_PPL을 달성했다. 평균
   PPL 동급 + reproducibility 우위.
3. **Length extrapolation의 구조적 결정성**: § 7.7의 9 변종 32× 분해가
   보여주듯, OOD 외삽 능력은 두 tier로 명확히 갈린다. Tier 1(외삽 OK)은
   (a) 강한 distance attenuation이 존재하거나(linear ALiBi, multiplicative xPos)
   (b) rotation이 아예 없는(NoPE) 경우다. Tier 2(외삽 BAD)는 rotation이 존재하면서
   attenuation이 없거나 약한(log) 경우다. EulerCE의 외삽 우위는 ALiBi의 재발견이며,
   **NoPE가 RoPE보다 7배 잘 외삽**한다는 사실은 RoPE의 회전 자체가 OOD의 원인임을 보여준다.

본 작업의 가장 큰 가치는 ζ-attention 가설의 negative findings를 명시하고,
Euler-CE 외삽 우위의 진짜 메커니즘(e-decay = ALiBi)을 32× extrapolation
까지 분해한 데에 있다.

---

## References

[1] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.

[2] O. Press et al., "Train Short, Test Long: Attention with Linear Biases (ALiBi)," ICLR 2022.

[3] B. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023.

[4] Y. Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.

[5] D. Hilbert, G. Pólya, conjecture (folklore, ca. 1910s).

[6] M. V. Berry, J. P. Keating, "H = xp and the Riemann zeros," 1999.

[7] (Internal) `docs/6_뇌/07_수면과복구.md` — CE 부트스트랩 고정점 `ε² = 4.87 %`.

[8] (Internal) `docs/7_AGI/2_Architecture.md` § 4 — 유니타리 제약.

[9] (Internal) `reality_stone/python/reality_stone/clarus/ce_euler.py` — Euler-CE attention.

[10] Y. Sun et al., "A Length-Extrapolatable Transformer (xPos)," ACL 2023.

[11] M. M. Bronstein et al., "Geometric Deep Learning," 2021.

[12] V. G. Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML 2021.

[13] H. L. Montgomery, "The pair correlation of zeros of the zeta function," 1973.

---

## Appendix A. 200-step transient (참고용)

소규모 짧은 학습(200 step × 2 seed)에서는 분산이 더 dramatic하게 보였으나
(σ_PPL: RoPE 1.22, mra 0.16, mra_h 0.06; 7-20× 감소), 1500-step 충분 수렴 시점
에는 σ가 모두 0.07-0.21 범위로 평준화된다. **소규모 실험의 분산 격차는
early-stopping artifact이며 영구적 안정성 우위가 아니다**. 1500-step 결과
(§ 7)만이 신뢰할 수 있는 결론이다.

이 발견 자체가 ML 평가에서 reproducibility의 함정을 보여주는 사례다. 짧은
학습에서 본 "결정적 차이"는 충분히 학습하면 사라질 수 있다.
```
---
## File: `docs/8_리만/riemann_pe_spec.md`

```markdown
# Riemann Surface Positional Encoding 정밀 사양

## 0. 전제

리만 가설(Riemann Hypothesis)은 공학적 axiom으로 채택한다.

> ζ(s)의 모든 비자명 영점은 critical line Re(s) = 1/2 위에 있다.

따라서 영점은 s_n = 1/2 + i γ_n의 형태이며, {γ_n}은 Montgomery-Dyson 추측에 의해
GUE(Gaussian Unitary Ensemble) 통계를 따르는 "무작위인 동시에 구조적인" 수열이다.
처음 100개의 γ_n은 Titchmarsh / Odlyzko 표에서 가져와 `RIEMANN_ZEROS_IM`에 하드코딩한다.
n > 100은 Riemann-von Mangoldt 점근식 γ_n ≈ 2π n / log n으로 외삽한다.

본 사양은 이 axiom 위에서 attention의 positional encoding을
**Riemann surface (multi-sheet 복소 평면)** 위의 회전으로 재구성한다.

## 1. 동기 — 왜 평면(surface)이 필요한가

기존 `RiemannRotaryAttention`은 RoPE와 동일한 prescription을 사용한다.

    inv_freq_k = 1 / γ_k
    θ(p, k)    = p · inv_freq_k

이는 평면(circle) 위의 회전이며, 다음 두 가지 한계가 있다:

1. **단일 시트(single-sheet)**: θ가 2π를 넘어가면 정보가 wrap-around로 사라진다.
   같은 phase인 두 위치를 attention이 구분할 수 없다.
2. **선형 시간 lift**: 위치 p가 선형으로 들어간다.
   따라서 sequence length가 N → kN으로 늘어나면 phase도 k배 늘어나
   학습된 frequency 분포가 깨진다(RoPE의 long-context 문제와 동일).

Riemann surface는 이 두 문제를 동시에 해결한다.

- **Multi-sheet**: log z는 단일값이 아니라 z = r e^{iθ} 위에서 무한 시트를 갖는다.
  sheet index를 명시적으로 유지하면 phase가 wrap되어도 정보가 보존된다.
- **Logarithmic lift**: 자연 좌표 τ = log(1 + p)는 multiplicative scale에 대해
  invariant하다(kp ↦ τ + log k). Sequence length의 power-law 변화에도 안정적이다.

## 2. 사양

### 2.1 좌표 lift

위치 p ∈ {0, 1, …, N-1}를 critical line의 imaginary axis로 들어올린다.

$$
\tau_p = \log(1 + p), \qquad s_p = \tfrac{1}{2} + i\,\tau_p \in \mathbb{C}.
$$

`+1`은 p = 0에서 log 발산을 막기 위한 standard offset이다.

### 2.2 회전 generator

각 헤드의 dim-pair k(k = 0, …, d_head/2 - 1)는 γ_k를 frequency로 가지며,
회전각은

$$
\theta(p, k) = \gamma_k \cdot \tau_p = \gamma_k \log(1 + p).
$$

대응하는 단위 복소수는

$$
e^{i\theta(p,k)} = (1+p)^{i\gamma_k}.
$$

이는 Mellin 변환 커널 (1+p)^{i γ_k}와 정확히 일치한다. Riemann ζ 함수 자체가
이 형태의 합으로 정의되므로 자연스러운 선택이다.

### 2.3 Sheet index

회전은 모듈로 2π이지만, 시트(sheet) 정보는 별도로 보존한다.

$$
\sigma(p, k) = \left\lfloor \frac{\theta(p, k)}{2\pi} \right\rfloor.
$$

두 위치 i, j가 같은 phase(cos/sin 동일)라도 서로 다른 시트에 있으면
Riemann surface 위에서는 다른 점이다. Attention은 sheet 차이를 바이어스로 받는다.

$$
b^{\text{sheet}}_{ij} = -\lambda_\sigma \cdot \frac{1}{d_{\text{head}}/2}
                       \sum_{k=0}^{d_{\text{head}}/2-1} |\sigma(i, k) - \sigma(j, k)|,
$$

여기서 λ_σ는 학습 가능한 per-head 스칼라다.
이 항은 cross-sheet attention을 약화시켜 시트 식별을 강제한다.

### 2.4 회전 적용 (RoPE-style relative form)

RoPE와 동일하게, dim-pair (2k, 2k+1)에 대해 2D 회전을 적용한다.

$$
\begin{pmatrix} q'_{2k} \\ q'_{2k+1} \end{pmatrix} =
\begin{pmatrix} \cos\theta(p,k) & -\sin\theta(p,k) \\
                \sin\theta(p,k) &  \cos\theta(p,k) \end{pmatrix}
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
$$

그러면 q_i^T k_j는 자동으로 Δθ = θ(i,k) - θ(j,k) = γ_k log((1+i)/(1+j))의
함수가 된다. 이로써 translation invariance가 유지되고, Hilbert-Pólya 관점에서
Hermitian kernel이 보장된다.

### 2.5 최종 attention score

$$
\text{score}_{ij} = \frac{q_i^{\prime\top} k_j^{\prime}}{\sqrt{d_{\text{head}}}}
                  + b^{\text{sheet}}_{ij},
$$

이후 causal mask와 softmax를 적용한다.

## 3. 학습 가능 파라미터

| 이름            | 형상           | 역할                                                                   |
|-----------------|----------------|------------------------------------------------------------------------|
| `log_scale`     | (n_heads,)     | 헤드별 "speed of light": 모든 γ_k에 곱해지는 exp(s)                   |
| `log_lambda_sigma` | (n_heads,) | sheet-difference penalty의 log-scale (λ_σ = exp(·))                  |

이 외 파라미터(γ_k, frequency 자체)는 모두 buffer로 두고 학습하지 않는다. 이렇게 RH의 axiom적 성격을 유지한다.

## 4. 점근적 성질

- 작은 p에서는 τ_p ≈ p(log(1+p) ≈ p)이므로 기존 RoPE와 유사하다.
- 큰 p에서는 τ_p가 천천히 증가하므로 frequency aliasing이 자동으로 완화된다.
- N → kN일 때 τ는 log k만큼만 평행이동하므로 relative attention이 거의 동일하게 보존된다.

## 5. 백엔드 dispatch

세 단계 backend 모두에서 동일한 수치 결과를 보장한다.

1. **PyTorch** (참조): `reality_stone.clarus.ce_riemann_attn.RiemannRotaryAttention`
2. **Rust CPU**: `reality_stone.clarus._rust.nn_ce_riemann_fwd`
3. **CUDA**: `reality_stone.clarus._rust.nn_ce_riemann_fwd_cuda` (cudarc launcher + `.cu` kernel)

자동 선택은 `reality_stone.clarus.ce_riemann_attn.RiemannRotaryAttention(backend="auto")`가
입력 텐서의 `device.type`으로 결정한다(cuda → cuda, cpu → rust, fallback → torch).

## 6. 수치 동일성 테스트

`tests/test_riemann_pe_consistency.py`에서

- 동일 입력에 대해 세 backend의 출력이 atol=1e-4, rtol=1e-3 이내로 일치하는지
- backward grad가 PyTorch와 1e-3 이내로 일치하는지

검증한다.

## 7. 참고

- Titchmarsh, *The Theory of the Riemann Zeta-Function*, Appendix.
- Odlyzko, *On the distribution of spacings between zeros of the zeta function*.
- Montgomery (1973), pair-correlation conjecture.
- Su & Lu, *RoFormer*, 2021 — RoPE 원본.
```
---
## File: `pyproject.toml`

```toml
[project]
name = "reality_stone"
version = "0.2.10"
description = "Unified Reality Stone library with the Clarus runtime"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "tokenizers>=0.15,<1",
    "torch>=2.0",
    "tqdm>=4.66,<5",
    "transformers>=4.40,<6",
]

[project.optional-dependencies]
science = [
    "datasets>=2.18,<5",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "pandas>=2.0",
    "seaborn>=0.13",
    "joblib>=1.3",
]
quantum = [
    "qiskit>=1.0",
    "qiskit-aer>=0.14",
    "qiskit-ibm-runtime>=0.20",
]
neuro = ["nibabel>=5.4"]
dev = ["pytest>=7.0", "ruff>=0.4"]
all = [
    "reality_stone[science,quantum,neuro,dev]",
]

[build-system]
requires = ["maturin>=1.5,<2"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "reality_stone/python/reality_stone/clarus/core/Cargo.toml"
python-source = "reality_stone/python"
module-name = "reality_stone.clarus._rust"
features = ["python"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests", "reality_stone/tests"]
markers = [
    "cuda: tests that require CUDA kernels",
]
```
---
## File: `README.md`

```markdown
# Reality Stone

This repository is now centered on `reality_stone`.

The former top-level `clarus` package has been moved into the Reality Stone
Python package as:

```python
import reality_stone.clarus
```

## Layout

```text
reality_stone/
  python/reality_stone/          Reality Stone Python API
  python/reality_stone/clarus/   Clarus runtime, CE ops, agents, and bridges
  src/                           Reality Stone Rust/PyO3 backend
  tests/                         Reality Stone regression tests
  examples/unified_clarus_demo.py

tests/
  test_unified_example.py        smoke test for the single example
```

## Run The Example

```powershell
.\.venv\Scripts\python.exe -B reality_stone\examples\unified_clarus_demo.py
```

The example exercises the unified path:

- imports `reality_stone`
- uses `reality_stone.clarus.runtime.BrainRuntime`
- runs `MetricAttention`
- runs the unified Riemannian fallback bridge

## Quick Checks

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
```

## Native Build Note

The root `pyproject.toml` is the unified checkout entrypoint and targets the
optional Clarus core extension at `reality_stone.clarus._rust`. The vendored
Reality Stone native extension remains available from `reality_stone/pyproject.toml`
as `reality_stone._rust`; Python fallbacks keep the unified package importable
when either native extension is absent.
```
---
## File: `reality_stone/Cargo.toml`

```toml
[package]
name = "reality_stone"
version = "0.2.10"
edition = "2021"
authors = ["jigglypop <donghwanyeom@gmail.com>"]
description = "High-performance hyperbolic neural networks library powered by Rust"
license = "MIT"
readme = "README.md"
repository = "https://github.com/jigglypop/reality_stone"
documentation = "https://github.com/jigglypop/reality_stone"
homepage = "https://github.com/jigglypop/reality_stone"
keywords = [
    "hyperbolic",
    "neural-networks",
    "deep-learning",
    "manifold-learning",
    "riemannian",
]
categories = ["science", "mathematics", "algorithms"]
exclude = [
    "tests/**",
    "target/**",
]

[lib]
name = "_rust"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.21.2", features = ["extension-module"] }
ndarray = { version = "0.15", features = ["rayon", "approx"] }
numpy = "0.21.0"
rayon = "1.8.0"
indicatif = "0.17"
num-traits = "0.2"
num-integer = "0.1"
num-rational = "0.4"
simba = "0.8"
approx = "0.4.0"
anyhow = "1.0"
thiserror = "1.0"
ndarray-rand = "0.15.0"
rand = "0.8.5"
libc = "0.2"
rustfft = "6.4.0"
faer = "0.19"

[build-dependencies]
cc = "1.0"
glob = "0.3"
bindgen = { version = "0.69", optional = true }

[features]
default = []
cuda = ["bindgen"]
abi3-py310 = ["pyo3/abi3-py310"]
```
---
## File: `reality_stone/pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "reality_stone"
version = "0.2.10"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.21,<3",
    "torch>=2.0",
    "tqdm>=4.66,<5",
    "transformers>=4.40,<6",
]
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "reality_stone._rust"
python-source = "python"

[tool.pytest.ini_options]
markers = [
    "cuda: tests that require CUDA kernels",
]
```
---
## File: `reality_stone/python/reality_stone/__init__.py`

```python
__version__ = "0.2.10"

import torch
import sys
from pathlib import Path

_has_rust_ext = False
_has_cuda = False

try:
    from . import _rust  # type: ignore
    _has_rust_ext = not bool(getattr(_rust, "IS_FALLBACK", False))
except Exception:
    _rust = None  # type: ignore
    try:
        lib_path = Path(__file__).parent.resolve()
        local_ext = list(lib_path.glob('_rust*.so')) or list(lib_path.glob('_rust*.pyd'))
        if local_ext:
            if str(lib_path) not in sys.path:
                sys.path.insert(0, str(lib_path))
            from . import _rust as _rust_local  # type: ignore
            _rust = _rust_local  # type: ignore
            _has_rust_ext = True
    except Exception:
        _rust = None  # type: ignore

if _has_rust_ext and torch.cuda.is_available():
    required_cuda_symbols = [
        'mobius_add_cuda',
        'mobius_scalar_cuda',
        'poincare_ball_layer_cuda',
        'poincare_ball_layer_backward_cuda',
        'poincare_distance_cuda',
        'lorentz_layer_forward_cuda',
        'lorentz_ball_layer_backward_cuda',
        'lorentz_distance_cuda',
        'klein_layer_forward_cuda',
        'klein_ball_layer_backward_cuda',
        'klein_distance_cuda',
    ]
    _has_cuda = all(hasattr(_rust, name) for name in required_cuda_symbols)  # type: ignore
else:
    _has_cuda = False

from .core.mobius import MobiusAdd, MobiusScalarMul

from .layers.poincare import (
    PoincareBallLayer,
    poincare_add,
    poincare_scalar_mul,
    poincare_distance,
    poincare_to_lorentz,
    poincare_to_klein,
    project_to_ball,
    HyperbolicLinear,
    GeodesicLinear,
)
from .layers.lorentz import (
    LorentzLayer,
    lorentz_add,
    lorentz_scalar_mul,
    lorentz_distance,
    lorentz_inner,
    lorentz_to_poincare,
    lorentz_to_klein,
    euclidean_to_lorentz,
)
from .layers.klein import (
    KleinLayer,
    klein_add,
    klein_scalar_mul,
    klein_distance,
    klein_to_poincare,
    klein_to_lorentz,
)
from .layers.spline import SplineLinear
from .layers.metric_attention import MetricAttention, SPDMetric
from .layers.rsulf_cuda import RSULFLayerCUDA, RSULFWrapperCUDA, RSULFLMHeadCUDA

try:
    if _has_rust_ext:
        from ._rust import metrikey  # type: ignore
    else:
        from . import metrikey  # type: ignore
except Exception:
    try:
        from . import metrikey  # type: ignore
    except Exception:
        metrikey = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import geodesic as _geodesic  # type: ignore
        geodesic_topk_attention = _geodesic.geodesic_topk_attention
        batched_cholesky = _geodesic.batched_cholesky_cuda
    else:
        geodesic_topk_attention = None  # type: ignore
        batched_cholesky = None  # type: ignore
except Exception:
    geodesic_topk_attention = None  # type: ignore
    batched_cholesky = None  # type: ignore

try:
    from .conversion import convert_to_full_riemannian, convert_to_hyperbolic
except Exception:
    convert_to_full_riemannian = None  # type: ignore
    convert_to_hyperbolic = None  # type: ignore
from .losses import HyperbolicSupConLoss, BellmanConsistencyLoss, laplacian_same_label, poincare_kinetic_energy

from . import optim
from . import layers

try:
    from . import data
except ImportError:
    data = None  # type: ignore

try:
    from . import models
except ImportError:
    models = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import PyUnifiedRiemannianLayer as UnifiedRiemannianLayer  # type: ignore
        from ._rust import compute_metric, geodesic_distance, geodesic_interpolate  # type: ignore
    else:
        from ._fallback import (
            TorchUnifiedRiemannianLayer as UnifiedRiemannianLayer,
            euclidean_metric_np as compute_metric,
            geodesic_distance_np as geodesic_distance,
            geodesic_interpolate_np as geodesic_interpolate,
        )
except Exception:
    from ._fallback import (
        TorchUnifiedRiemannianLayer as UnifiedRiemannianLayer,
        euclidean_metric_np as compute_metric,
        geodesic_distance_np as geodesic_distance,
        geodesic_interpolate_np as geodesic_interpolate,
    )

try:
    if _rust is not None:
        from ._rust import PyRiemannianDiffusion  # type: ignore
    else:
        PyRiemannianDiffusion = None  # type: ignore
except Exception:
    PyRiemannianDiffusion = None  # type: ignore

try:
    if _rust is not None:
        from ._rust import PyRSULFLayer as RSULFLayer  # type: ignore
        from ._rust import fold_metric_svd, fold_ffn, build_causal_laplacian  # type: ignore
        from ._rust import verify_metric_consistency, fold_metric_optimized, nystrom_metric  # type: ignore
        from ._rust import bellman_geodesic_forward, bellman_geodesic_backward  # type: ignore
        from ._rust import extract_metric_cuda  # type: ignore
    else:
        RSULFLayer = None  # type: ignore
        fold_metric_svd = None  # type: ignore
        fold_ffn = None  # type: ignore
        build_causal_laplacian = None  # type: ignore
        verify_metric_consistency = None  # type: ignore
        fold_metric_optimized = None  # type: ignore
        nystrom_metric = None  # type: ignore
        bellman_geodesic_forward = None  # type: ignore
        bellman_geodesic_backward = None  # type: ignore
        extract_metric_cuda = None  # type: ignore
except Exception:
    RSULFLayer = None  # type: ignore
    fold_metric_svd = None  # type: ignore
    fold_ffn = None  # type: ignore
    build_causal_laplacian = None  # type: ignore
    verify_metric_consistency = None  # type: ignore
    fold_metric_optimized = None  # type: ignore
    nystrom_metric = None  # type: ignore
    bellman_geodesic_forward = None  # type: ignore
    bellman_geodesic_backward = None  # type: ignore
    extract_metric_cuda = None  # type: ignore


def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float = None, t: float = 0.5, kappas: torch.Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> torch.Tensor:
    return PoincareBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)


def klein_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return KleinLayer.apply(u, v, c, t)


def lorentz_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return LorentzLayer.apply(u, v, c, t)


__all__ = [
    '__version__',
    '_has_rust_ext',
    '_has_cuda',
    'MobiusAdd',
    'MobiusScalarMul',
    'poincare_add',
    'poincare_scalar_mul',
    'poincare_distance',
    'poincare_ball_layer',
    'PoincareBallLayer',
    'poincare_to_lorentz',
    'poincare_to_klein',
    'project_to_ball',
    'HyperbolicLinear',
    'GeodesicLinear',
    'lorentz_add',
    'lorentz_scalar_mul',
    'lorentz_distance',
    'lorentz_inner',
    'lorentz_to_poincare',
    'lorentz_to_klein',
    'euclidean_to_lorentz',
    'lorentz_layer',
    'LorentzLayer',
    'klein_add',
    'klein_scalar_mul',
    'klein_distance',
    'klein_to_poincare',
    'klein_to_lorentz',
    'klein_layer',
    'KleinLayer',
    'SplineLinear',
    'MetricAttention',
    'SPDMetric',
    'convert_to_full_riemannian',
    'convert_to_hyperbolic',
    'HyperbolicSupConLoss',
    'BellmanConsistencyLoss',
    'laplacian_same_label',
    'poincare_kinetic_energy',
    'optim',
    'layers',
    'data',
    'models',
    'metrikey',
    'geodesic_topk_attention',
    'batched_cholesky',
    'UnifiedRiemannianLayer',
    'compute_metric',
    'geodesic_distance',
    'geodesic_interpolate',
    'PyRiemannianDiffusion',
    'RSULFLayer',
    'fold_metric_svd',
    'fold_ffn',
    'build_causal_laplacian',
    'verify_metric_consistency',
    'fold_metric_optimized',
    'nystrom_metric',
    'bellman_geodesic_forward',
    'bellman_geodesic_backward',
    'extract_metric_cuda',
    'RSULFLayerCUDA',
    'RSULFWrapperCUDA',
    'RSULFLMHeadCUDA',
]
```
---
## File: `reality_stone/python/reality_stone/_fallback.py`

```python
from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import torch


EPS = 1e-7


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def dynamic_curvature(kappa: float, c_min: float, c_max: float) -> float:
    return float(c_min) + (float(c_max) - float(c_min)) * sigmoid(float(kappa))


def mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c_t = torch.as_tensor(float(c), dtype=x.dtype, device=x.device)
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = 1.0 + 2.0 * c_t * xy + c_t * c_t * x2 * y2
    return num / den.clamp_min(EPS)


def mobius_scalar_torch(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    r = float(r)
    c = float(c)
    if abs(r) < EPS:
        return torch.zeros_like(x)
    if abs(c) < EPS:
        return x * r
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    if c > 0.0:
        sqrt_c = math.sqrt(c)
        arg = (sqrt_c * norm).clamp(max=1.0 - EPS)
        scale = torch.tanh(r * torch.atanh(arg)) / (sqrt_c * norm)
        return scale * x
    # For signed/experimental curvature schedules, keep a stable Euclidean limit.
    return x * r


def lorentz_inner_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c = float(c)
    if abs(c) < EPS:
        return torch.linalg.norm(x - y, dim=-1)
    z = c * lorentz_inner_torch(x, y)
    valid = z >= 1.0
    hyper = torch.acosh(z.clamp_min(1.0)) / math.sqrt(abs(c))
    euclid = torch.linalg.norm(x - y, dim=-1)
    return torch.where(valid, hyper, euclid)


def klein_distance_torch(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c = float(c)
    if abs(c) < EPS:
        return torch.linalg.norm(x - y, dim=-1)
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    xy = (x * y).sum(dim=-1)
    den = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(EPS).sqrt()
    arg = ((1.0 - c * xy) / den).clamp_min(1.0)
    return torch.acosh(arg) / math.sqrt(abs(c))


def euclidean_metric_np(x: np.ndarray, metric_type: str, curvature: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if metric_type == "poincare" and abs(curvature) > EPS:
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        lam = 4.0 / np.maximum((1.0 - curvature * x2) ** 2, EPS)
        return np.repeat(lam, x.shape[-1], axis=-1).astype(np.float32)
    return np.ones_like(x, dtype=np.float32)


def geodesic_distance_np(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str = "diagonal",
    curvature: float = 0.0,
) -> np.ndarray:
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if metric_type == "lorentz":
        return lorentz_distance_torch(x_t, y_t, curvature).numpy().astype(np.float32)
    if metric_type == "klein":
        return klein_distance_torch(x_t, y_t, curvature).numpy().astype(np.float32)
    if metric_type == "poincare" and abs(float(curvature)) > EPS:
        x2 = (x_t * x_t).sum(dim=-1)
        y2 = (y_t * y_t).sum(dim=-1)
        diff2 = ((x_t - y_t) * (x_t - y_t)).sum(dim=-1).clamp_min(0.0)
        den = ((1.0 - curvature * x2) * (1.0 - curvature * y2)).clamp_min(EPS)
        arg = 1.0 + 2.0 * curvature * diff2 / den
        return (torch.acosh(arg.clamp_min(1.0)) / math.sqrt(abs(float(curvature)))).numpy().astype(np.float32)
    return np.linalg.norm(np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32), axis=-1).astype(np.float32)


def geodesic_interpolate_np(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str = "diagonal",
    curvature: float = 0.0,
    t: float = 0.5,
) -> np.ndarray:
    del metric_type, curvature
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    t = float(t)
    return ((1.0 - t) * x + t * y).astype(np.float32)


class TorchUnifiedRiemannianLayer:
    def __init__(
        self,
        metric_type: str = "diagonal",
        curvature: float = 0.0,
        input_dim: int = 1,
        enable_bellman: bool = False,
    ) -> None:
        self.metric_type = str(metric_type)
        self.curvature = float(curvature)
        self.input_dim = int(input_dim)
        self.enable_bellman = bool(enable_bellman)

    def forward(self, x, target=None):
        x_arr = np.asarray(x, dtype=np.float32)
        if target is None:
            out = x_arr.copy()
            velocity = np.zeros_like(x_arr)
        else:
            target_arr = np.asarray(target, dtype=np.float32)
            out = geodesic_interpolate_np(x_arr, target_arr, self.metric_type, self.curvature, 0.5)
            velocity = out - x_arr
        energy = self.compute_energy(x_arr, velocity, out, np.zeros(x_arr.shape[0], dtype=np.float32)) if self.enable_bellman else None
        return out, energy

    def backward(self, grad_output, x):
        del x
        return np.asarray(grad_output, dtype=np.float32)

    def geodesic_path(self, x, y, num_steps: int = 10):
        steps = max(2, int(num_steps))
        return [
            geodesic_interpolate_np(x, y, self.metric_type, self.curvature, i / float(steps - 1))
            for i in range(steps)
        ]

    def compute_energy(self, x, v, x_next, reward):
        x = np.asarray(x, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        x_next = np.asarray(x_next, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        kinetic = 0.5 * np.sum(v * v, axis=-1)
        potential = 0.5 * np.sum(x_next * x_next, axis=-1) - reward
        lagrangian = kinetic - potential
        bellman_residual = np.linalg.norm(x_next - x, axis=-1)
        return {
            "kinetic": kinetic.astype(np.float32),
            "potential": potential.astype(np.float32),
            "lagrangian": lagrangian.astype(np.float32),
            "bellman_residual": bellman_residual.astype(np.float32),
        }

    def flow_step(self, x, num_steps: int = 1, learning_rate: float = 0.01):
        out = np.asarray(x, dtype=np.float32).copy()
        lr = float(learning_rate)
        for _ in range(max(1, int(num_steps))):
            out = out - lr * out
        return out.astype(np.float32)


def _hash_seed(parts: Iterable[object]) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def deterministic_spd(key: str, dim: int, min_lambda: float, max_lambda: float, mass: float = 1.0) -> np.ndarray:
    dim = int(dim)
    rng = np.random.default_rng(_hash_seed((key, dim, min_lambda, max_lambda, mass)))
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)).astype(np.float32))
    vals = rng.uniform(float(min_lambda), float(max_lambda), size=dim).astype(np.float32)
    vals = vals * max(float(mass), EPS)
    return (q @ np.diag(vals) @ q.T).astype(np.float32)
```
---
## File: `reality_stone/python/reality_stone/_rust.py`

```python
from __future__ import annotations

import numpy as np

from ._fallback import (
    TorchUnifiedRiemannianLayer as PyUnifiedRiemannianLayer,
    euclidean_metric_np as compute_metric,
    geodesic_distance_np as geodesic_distance,
    geodesic_interpolate_np as geodesic_interpolate,
)


IS_FALLBACK = True


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def _curvature_from_kappa(kappa, c_min: float, c_max: float) -> float:
    k = float(kappa)
    sig = 1.0 / (1.0 + np.exp(-k))
    return float(c_min + (c_max - c_min) * sig)


def _project_ball(x, c: float, eps: float = 1e-6):
    x = _as_f32(x)
    c = max(float(c), eps)
    radius = 1.0 / np.sqrt(c)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = radius - eps
    scale = np.where(norm > max_norm, max_norm / np.maximum(norm, eps), 1.0)
    return (x * scale).astype(np.float32)


def poincare_ball_layer_cpu(u, v, c: float, t: float):
    out = (1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)
    return _project_ball(out, c)


def poincare_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def poincare_ball_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_ball_layer_cpu(u, v, abs(c_val)), c_val


def poincare_ball_layer_layerwise_backward_cpu(
    grad,
    u,
    v,
    kappa,
    layer_idx: int,
    c_min: float,
    c_max: float,
    t: float,
):
    del kappa, layer_idx, c_min, c_max
    grad_u, grad_v = poincare_ball_layer_backward_cpu(grad, u, v, 1.0, t)
    return grad_u, grad_v, 0.0


def poincare_to_lorentz_cpu(x, c: float):
    x = _project_ball(x, c)
    c = max(float(c), 1e-6)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    den = np.maximum(1.0 - c * x2, 1e-7)
    time = (1.0 + c * x2) / (np.sqrt(c) * den)
    space = 2.0 * x / den
    return np.concatenate([time, space], axis=-1).astype(np.float32)


def poincare_to_klein_cpu(x, c: float):
    x = _project_ball(x, c)
    c = float(c)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    return (2.0 * x / np.maximum(1.0 + c * x2, 1e-7)).astype(np.float32)


def lorentz_inner(u, v):
    u = _as_f32(u)
    v = _as_f32(v)
    return (u[..., 0] * v[..., 0] - np.sum(u[..., 1:] * v[..., 1:], axis=-1)).astype(np.float32)


def lorentz_distance(u, v, c: float):
    c = max(float(c), 1e-6)
    z = np.maximum(c * lorentz_inner(u, v), 1.0)
    return (np.arccosh(z) / np.sqrt(c)).astype(np.float32)


def lorentz_layer_forward(u, v, c: float, t: float):
    del c
    return ((1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)).astype(np.float32)


def lorentz_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def lorentz_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return lorentz_layer_forward(u, v, c_val, t), c_val


def lorentz_add(u, v, c: float):
    del c
    return (_as_f32(u) + _as_f32(v)).astype(np.float32)


def lorentz_scalar(x, r: float, c: float):
    del c
    return (_as_f32(x) * float(r)).astype(np.float32)


def lorentz_to_poincare(x, c: float):
    x = _as_f32(x)
    denom = np.maximum(x[..., :1] + np.sqrt(1.0 / max(float(c), 1e-6)), 1e-7)
    return (x[..., 1:] / denom).astype(np.float32)


def lorentz_to_klein(x, c: float):
    del c
    x = _as_f32(x)
    return (x[..., 1:] / np.maximum(x[..., :1], 1e-7)).astype(np.float32)


def klein_layer_forward(u, v, c: float, t: float):
    out = (1.0 - float(t)) * _as_f32(u) + float(t) * _as_f32(v)
    return _project_ball(out, c)


def klein_ball_layer_backward_cpu(grad, u, v, c: float, t: float):
    del u, v, c
    grad = _as_f32(grad)
    return (grad * (1.0 - float(t))).astype(np.float32), (grad * float(t)).astype(np.float32)


def klein_layer_layerwise_cpu(u, v, kappa, layer_idx: int, c_min: float, c_max: float, t: float):
    del layer_idx
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return klein_layer_forward(u, v, abs(c_val), t), c_val


def klein_add(u, v, c: float):
    return _project_ball(_as_f32(u) + _as_f32(v), c)


def klein_scalar(x, r: float, c: float):
    return _project_ball(_as_f32(x) * float(r), c)


def klein_distance(x, y, c: float):
    x = _project_ball(x, c)
    y = _project_ball(y, c)
    c = max(float(c), 1e-6)
    x2 = np.sum(x * x, axis=-1)
    y2 = np.sum(y * y, axis=-1)
    xy = np.sum(x * y, axis=-1)
    den = np.maximum((1.0 - c * x2) * (1.0 - c * y2), 1e-7)
    arg = np.maximum((1.0 - c * xy) / np.sqrt(den), 1.0)
    return (np.arccosh(arg) / np.sqrt(c)).astype(np.float32)


def klein_to_poincare(x, c: float):
    x = _project_ball(x, c)
    den = 1.0 + np.sqrt(np.maximum(1.0 - float(c) * np.sum(x * x, axis=-1, keepdims=True), 0.0))
    return (x / np.maximum(den, 1e-7)).astype(np.float32)


def klein_to_lorentz(x, c: float):
    x = _project_ball(x, c)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - float(c) * np.sum(x * x, axis=-1, keepdims=True), 1e-7))
    return np.concatenate([gamma, gamma * x], axis=-1).astype(np.float32)


def from_poincare_dynamic_cpu(x, kappa, c_min: float, c_max: float):
    c_val = _curvature_from_kappa(kappa, c_min, c_max)
    return poincare_to_lorentz_cpu(x, abs(c_val)), c_val


def from_poincare_dynamic_backward_cpu(grad, x, kappa, c_min: float, c_max: float):
    del x, kappa, c_min, c_max
    return _as_f32(grad), 0.0


def _svd_basis(wq_list, target_rank: int):
    mats = [np.asarray(w, dtype=np.float32) for w in wq_list]
    if not mats:
        return np.eye(target_rank, dtype=np.float32), target_rank
    cat = np.concatenate(mats, axis=0)
    _, _, vt = np.linalg.svd(cat, full_matrices=False)
    rank = min(int(target_rank), vt.shape[0])
    return vt[:rank].T.astype(np.float32), rank


def extract_global_basis(wq_list, wk_list, target_rank: int):
    del wk_list
    u, rank = _svd_basis(wq_list, target_rank)
    return {"u": u, "rank": rank}


def build_causal_laplacian(seq_len: int, window: int = 1):
    seq_len = int(seq_len)
    window = max(1, int(window))
    a = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        lo = max(0, i - window)
        for j in range(lo, i):
            a[i, j] = 1.0 / (1.0 + abs(i - j))
    d = np.diag(a.sum(axis=1))
    return (d - a).astype(np.float32)


def verify_metric_consistency(wq, wk, r: int):
    del r
    wq = np.asarray(wq, dtype=np.float32)
    wk = np.asarray(wk, dtype=np.float32)
    denom = max(float(np.linalg.norm(wq) * np.linalg.norm(wk)), 1e-6)
    score = float(abs(np.sum(wq * wk)) / denom)
    return {"fold_accuracy": max(0.0, min(1.0, score)), "is_valid": True}


def fold_metric_svd(wq, wk, r: int):
    basis, rank = _svd_basis([wq, wk], r)
    return {"u": basis, "rank": rank}


def fold_metric_optimized(wq, wk, r: int):
    return fold_metric_svd(wq, wk, r)


def nystrom_metric(wq, wk, r: int):
    return fold_metric_svd(wq, wk, r)


def fold_ffn(w, r: int):
    w = np.asarray(w, dtype=np.float32)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    rank = min(int(r), len(s))
    return {
        "u": u[:, :rank].astype(np.float32),
        "s": s[:rank].astype(np.float32),
        "v": vt[:rank, :].astype(np.float32),
    }


def bellman_geodesic_forward(x, *args, **kwargs):
    del args, kwargs
    return np.asarray(x, dtype=np.float32)


def bellman_geodesic_backward(grad, *args, **kwargs):
    del args, kwargs
    return np.asarray(grad, dtype=np.float32)


def extract_metric_cuda(w, calib, target_dim: int, num_steps: int, curvature: float, lr: float):
    del calib, num_steps, curvature, lr
    w = np.asarray(w, dtype=np.float32)
    dim = int(target_dim)
    return np.eye(dim, dtype=np.float32) * max(float(np.var(w)), 1e-6)


class PyHyperMetric:
    def __init__(self, u_global, v_global, w1, b1, w2, b2):
        self.u_global = np.asarray(u_global, dtype=np.float32)
        self.v_global = np.asarray(v_global, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)

    def generate_core(self, layer_emb):
        x = np.asarray(layer_emb, dtype=np.float32)
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        out = h @ self.w2 + self.b2
        r = int(round(out.size ** 0.5))
        return out.reshape(r, r).astype(np.float32)

    def project_forward(self, x, layer_emb):
        x = np.asarray(x, dtype=np.float32)
        core = self.generate_core(layer_emb)
        return (x @ self.u_global @ core @ self.v_global.T).astype(np.float32)


class PySymplecticLayer:
    def __init__(self, layer_idx, layer_emb, hyper_metric, dt=0.01):
        self.layer_idx = int(layer_idx)
        self.layer_emb = np.asarray(layer_emb, dtype=np.float32)
        self.hyper_metric = hyper_metric
        self.dt = float(dt)

    def step(self, q, p, kick):
        q = np.asarray(q, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
        kick = np.asarray(kick, dtype=np.float32)
        p_next = p + self.dt * kick
        q_next = q + self.dt * p_next
        return q_next.astype(np.float32), p_next.astype(np.float32)


class PyRSULFLayer:
    def __init__(self, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        self.wq = np.asarray(wq, dtype=np.float32)
        self.wk = np.asarray(wk, dtype=np.float32)
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.d_model = int(d_model)
        self.r = int(r)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.seq_len = int(seq_len)
        self.window = int(window)
        diag = np.ones(self.d_model, dtype=np.float32)
        self.g_diag = diag
        self.g_inv = diag
        self.curvature = 0.0

    @classmethod
    def new_fast(cls, wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window, calibration_samples=1024):
        del calibration_samples
        return cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)

    @classmethod
    def new_with_metric(cls, wq, wk, w1, w2, g_diag, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        obj = cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)
        obj.g_diag = np.asarray(g_diag, dtype=np.float32)
        obj.g_inv = 1.0 / np.maximum(obj.g_diag, 1e-6)
        return obj

    @classmethod
    def new_with_basis(cls, wq, wk, w1, w2, u, rank, d_model, r, eta, alpha, beta, gamma, seq_len, window):
        del u, rank
        return cls(wq, wk, w1, w2, d_model, r, eta, alpha, beta, gamma, seq_len, window)

    def forward(self, x, v=None):
        x = np.asarray(x, dtype=np.float32)
        if v is None:
            v = np.zeros_like(x, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32)
        y = x + self.eta * np.tanh(x)
        v_next = self.gamma * v + (y - x)
        return y.astype(np.float32), v_next.astype(np.float32)

    def export_components(self):
        d = self.d_model
        hidden = self.w1.shape[0] if self.w1.ndim == 2 else d
        r1 = min(self.r, self.w1.shape[0], self.w1.shape[1]) if self.w1.ndim == 2 else min(self.r, d)
        r2 = min(self.r, self.w2.shape[0], self.w2.shape[1]) if self.w2.ndim == 2 else min(self.r, d)
        return {
            "d_model": d,
            "r": self.r,
            "eta": self.eta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "seq_len": self.seq_len,
            "window": self.window,
            "g_diag": self.g_diag,
            "g_inv": self.g_inv,
            "g_sym": np.diag(self.g_diag).astype(np.float32),
            "ffn_u1": np.zeros((hidden, r1), dtype=np.float32),
            "ffn_s1": np.ones(r1, dtype=np.float32),
            "ffn_v1": np.zeros((d, r1), dtype=np.float32),
            "ffn_u2": np.zeros((d, r2), dtype=np.float32),
            "ffn_s2": np.ones(r2, dtype=np.float32),
            "ffn_v2": np.zeros((hidden, r2), dtype=np.float32),
            "curvature": self.curvature,
        }

    def param_count(self):
        original = int(self.wq.size + self.wk.size + self.w1.size + self.w2.size)
        compressed = max(1, int(original / 2))
        return compressed, original, float(original / compressed)


class PyGeodesicMemory:
    pass


class SplineCache:
    pass


class PyRiemannianDiffusion:
    def __init__(self, dim: int, alpha: float, dt: float):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.dt = float(dt)

    def step(self, h, flow):
        return np.asarray(h, dtype=np.float32) + self.dt * np.asarray(flow, dtype=np.float32)
```
---
## File: `reality_stone/python/reality_stone/api/__init__.py`

```python
from .pipeline import pipeline, HierarchicalLLM
from .inference import TextGenerator, TextEditor
from .qa import QuestionAnswerer
from .indexing import DocumentIndexer

try:
    from .. import _rust as _native
except Exception:
    _native = None

poincare = getattr(_native, "poincare", None) if _native else None
lorentz = getattr(_native, "lorentz", None) if _native else None
klein = getattr(_native, "klein", None) if _native else None
metrikey = getattr(_native, "metrikey", None) if _native else None
spline = getattr(_native, "spline", None) if _native else None
geodesic = getattr(_native, "geodesic", None) if _native else None

__all__ = [
    "pipeline",
    "HierarchicalLLM",
    "TextGenerator",
    "TextEditor",
    "QuestionAnswerer",
    "DocumentIndexer",
    "poincare",
    "lorentz",
    "klein",
    "metrikey",
    "spline",
    "geodesic",
]
```
---
## File: `reality_stone/python/reality_stone/api/indexing.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class DocumentIndexer:
    model: object | None = None
    chunk_size: int = 512
    documents: List[str] = field(default_factory=list)

    def add(self, document: str) -> int:
        self.documents.append(str(document))
        return len(self.documents) - 1

    def extend(self, documents: Iterable[str]) -> list[int]:
        return [self.add(doc) for doc in documents]

    def __call__(self, documents):
        if isinstance(documents, str):
            return self.add(documents)
        return self.extend(documents)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        terms = set(str(query).lower().split())
        scored = []
        for idx, doc in enumerate(self.documents):
            words = set(doc.lower().split())
            score = len(terms & words) / max(len(terms), 1)
            scored.append({"id": idx, "text": doc, "score": float(score)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]
```
---
## File: `reality_stone/python/reality_stone/api/inference.py`

```python
from typing import Optional, Dict, Any, List, Union


class TextGenerator:
    
    def __init__(
        self,
        model,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ):
        self.model = model
        self.defaults = {
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_dict: bool = False,
        **kwargs
    ):
        if isinstance(text, list):
            return self.generate_batch(text, return_dict=return_dict, **kwargs)
        
        params = {**self.defaults, **kwargs}
        result = self.model.generate(
            text=text,
            enable_structural_edit=False,
            return_dict=True,
            **params
        )
        return result if return_dict else result["generated_text"]
    
    def generate_batch(
        self,
        texts: List[str],
        return_dict: bool = False,
        **kwargs
    ):
        return [self(text, return_dict=return_dict, **kwargs) for text in texts]


class TextEditor:
    
    def __init__(
        self,
        model,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        enable_structural_edit: bool = True,
        **kwargs
    ):
        self.model = model
        self.defaults = {
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
            "enable_structural_edit": enable_structural_edit,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_topics: bool = True,
        **kwargs
    ):
        if isinstance(text, list):
            return self.edit_batch(text, return_topics=return_topics, **kwargs)
        
        params = {**self.defaults, **kwargs}
        result = self.model.generate(
            text=text,
            return_dict=True,
            **params
        )
        
        output = {
            "original": result["original_text"],
            "edited": result["generated_text"],
        }
        
        if return_topics:
            output["topics"] = result["topics"]
        
        return output
    
    def edit_batch(
        self,
        texts: List[str],
        **kwargs
    ):
        return [self(text, **kwargs) for text in texts]
```
---
## File: `reality_stone/python/reality_stone/api/pipeline.py`

```python
from typing import Optional, Union, Dict, Any
from pathlib import Path
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
)


class HierarchicalLLM:
    
    def __init__(
        self,
        model: HierarchicalSentenceTopicLLM,
        config: HierarchicalLLMConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[HierarchicalLLMConfig] = None,
        device: Optional[str] = None,
    ) -> "HierarchicalLLM":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if config is None:
            if "config" in checkpoint:
                config_dict = dict(checkpoint["config"])
                if "use_pretrained_embeddings" not in config_dict:
                    config_dict["use_pretrained_embeddings"] = False
                config = HierarchicalLLMConfig(**config_dict)
            else:
                raise ValueError("Config not found in checkpoint")
        
        model = HierarchicalSentenceTopicLLM(config)
        
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys or incompatible.missing_keys:
            pass  # non-strict load for compatibility
        
        return cls(model, config, device)
    
    @classmethod
    def from_config(
        cls,
        config: Union[HierarchicalLLMConfig, Dict[str, Any]],
        device: Optional[str] = None,
    ) -> "HierarchicalLLM":
        if isinstance(config, dict):
            config = HierarchicalLLMConfig(**config)
        
        model = HierarchicalSentenceTopicLLM(config)
        return cls(model, config, device)
    
    def generate(
        self,
        text: str,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        enable_structural_edit: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        from reality_stone.models.hierarchical_sentence_topic_llm import infer_hierarchical_llm_on_text
        
        self.model.config.enable_structural_edit = enable_structural_edit
        
        params = {
            "model": self.model,
            "text": text,
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
        }
        params.update(kwargs)
        
        with torch.no_grad():
            result = infer_hierarchical_llm_on_text(**params)
        
        return result if return_dict else result["generated_text"]
    
    def __call__(self, text: str, **kwargs):
        return self.generate(text, **kwargs)
    
    def save_pretrained(self, save_path: Union[str, Path]):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "d_head": self.config.d_head,
                "num_topics": self.config.num_topics,
                "num_heads_topic": self.config.num_heads_topic,
                "n_layer_decoder": self.config.n_layer_decoder,
                "n_head_decoder": self.config.n_head_decoder,
                "c_poincare": self.config.c_poincare,
                "c_lorentz": self.config.c_lorentz,
                "use_pretrained_embeddings": self.config.use_pretrained_embeddings,
                "enable_variable_suppression": self.config.enable_variable_suppression,
            }
        }
        
        torch.save(checkpoint, save_path / "model.pt")


def pipeline(
    task: str = "text-generation",
    model: Optional[Union[str, Path, HierarchicalLLM]] = None,
    config: Optional[Union[HierarchicalLLMConfig, Dict]] = None,
    device: Optional[str] = None,
    **kwargs
):
    if isinstance(model, HierarchicalLLM):
        llm = model
    elif isinstance(model, (str, Path)):
        llm = HierarchicalLLM.from_pretrained(model, config, device)
    elif config is not None:
        llm = HierarchicalLLM.from_config(config, device)
    else:
        raise ValueError("Either model or config must be provided")

    # task dispatcher
    from .inference import TextGenerator, TextEditor
    from .qa import QuestionAnswerer
    from .indexing import DocumentIndexer

    task_map = {
        "text-generation": TextGenerator,
        "text-editing": TextEditor,
        "question-answering": QuestionAnswerer,
        "document-indexing": DocumentIndexer,
    }
    cls = task_map.get(task)
    if cls is None:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Available: {', '.join(task_map.keys())}"
        )
    return cls(llm, **kwargs)
```
---
## File: `reality_stone/python/reality_stone/api/qa.py`

```python
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    answer_question_from_corpus,
    answer_question_with_llm,
)


class QuestionAnswerer:
    
    def __init__(
        self,
        model,
        corpus: Optional[str] = None,
        top_k: Optional[int] = None,
        use_llm: bool = False,
        max_paragraphs: Optional[int] = None,
        **kwargs
    ):
        self.model = model.model
        self.defaults = {
            "corpus": corpus,
            "top_k": top_k or 3,
            "use_llm": use_llm,
            "max_paragraphs": max_paragraphs or 1000,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        question: Union[str, List[str]],
        context: Optional[str] = None,
        **kwargs
    ):
        if isinstance(question, list):
            return self.batch(question, context=context, **kwargs)
        
        params = {**self.defaults, **kwargs}
        corpus_path = context if context else params["corpus"]
        
        if not corpus_path:
            raise ValueError("corpus or context must be provided")
        
        if params["use_llm"]:
            result = answer_question_with_llm(
                model=self.model,
                question=question,
                data_path=corpus_path,
                top_k=params["top_k"],
                max_paragraphs=params["max_paragraphs"],
                max_new_tokens=params.get("max_new_tokens", 32),
            )
            return {
                "question": result["question"],
                "answer": result["answer"],
                "support": result["support"],
            }
        else:
            result = answer_question_from_corpus(
                model=self.model,
                question=question,
                data_path=corpus_path,
                top_k=params["top_k"],
                max_paragraphs=params["max_paragraphs"],
            )
            return {
                "question": result["question"],
                "answers": result["answers"],
                "support": result["support"],
            }
    
    def batch(
        self,
        questions: List[str],
        context: Optional[str] = None,
        **kwargs
    ):
        return [self(q, context=context, **kwargs) for q in questions]
```
---
## File: `reality_stone/python/reality_stone/clarus/__init__.py`

```python
"""Clarus runtime integrated under :mod:`reality_stone`.

Core modules stay import-safe so optional Rust/CUDA kernels and higher-level
runtime pieces can be used independently from the unified Reality Stone package.
"""

__version__ = "1.2.0"

topk_sparse = None
topk_sparse_batch = None
nn_topk_silu_fwd = None
nn_topk_silu_bwd = None
nn_lbo_fused_fwd = None
nn_power_iter = None
nn_gauge_lattice_fwd = None

auto_device = None
safe_print = None
normalize_vector = None
resolve_device = None
AD = PORTAL = BYPASS = T_WAKE = None
ACTIVE_RATIO = STRUCT_RATIO = BACKGROUND_RATIO = None

BrainRuntime = None
BrainRuntimeConfig = None
BrainRuntimeSnapshot = None
HippocampusMemory = None
ModuleLifecycle = None
RuntimeMode = None
RuntimeStep = None
RealityStoneStatus = None
has_reality_stone = None
reality_stone_status = None
RuntimeAgent = None
RuntimeAgentConfig = None
RuntimeAgentStep = None
RuntimeTextAgent = None
RuntimeTextAgentTurn = None
TextEnvironment = None
TextEnvironmentStep = None

try:
    from .device import auto_device  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from .constants import (  # type: ignore[no-redef]
        AD, PORTAL, BYPASS, T_WAKE,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    )
except ImportError:
    pass

try:
    from .utils import safe_print, normalize_vector, resolve_device  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from . import _rust as _rust_mod

    topk_sparse = _rust_mod.topk_sparse
    topk_sparse_batch = _rust_mod.topk_sparse_batch
    nn_topk_silu_fwd = _rust_mod.nn_topk_silu_fwd
    nn_topk_silu_bwd = _rust_mod.nn_topk_silu_bwd
    nn_lbo_fused_fwd = _rust_mod.nn_lbo_fused_fwd
    nn_power_iter = _rust_mod.nn_power_iter
    nn_gauge_lattice_fwd = _rust_mod.nn_gauge_lattice_fwd
except ImportError:
    pass

try:
    from .ce_ops import (
        has_rust as ce_has_rust,
        has_cuda as ce_has_cuda,
        ce_backend,
        pack_sparse as ce_pack_sparse,
        build_metric_basis as ce_build_metric_basis,
        codebook_pull as ce_codebook_pull,
        relax as ce_relax,
        relax_packed as ce_relax_packed,
    )
except ImportError:
    pass

try:
    from .runtime import (  # type: ignore[no-redef]
        BrainRuntime,
        BrainRuntimeConfig,
        BrainRuntimeSnapshot,
        HippocampusMemory,
        ModuleLifecycle,
        RuntimeMode,
        RuntimeStep,
    )
except ImportError:
    pass

try:
    from .reality import (  # type: ignore[no-redef]
        RealityStoneStatus,
        has_reality_stone,
        status as reality_stone_status,
    )
except ImportError:
    pass

try:
    from .agent import (  # type: ignore[no-redef]
        RuntimeAgent,
        RuntimeAgentConfig,
        RuntimeAgentStep,
        RuntimeTextAgent,
        RuntimeTextAgentTurn,
        TextEnvironment,
        TextEnvironmentStep,
    )
except ImportError:
    pass

__all__ = [
    "topk_sparse",
    "topk_sparse_batch",
    "nn_topk_silu_fwd",
    "nn_topk_silu_bwd",
    "nn_lbo_fused_fwd",
    "nn_power_iter",
    "nn_gauge_lattice_fwd",
    "BrainRuntime",
    "BrainRuntimeConfig",
    "BrainRuntimeSnapshot",
    "HippocampusMemory",
    "ModuleLifecycle",
    "RuntimeMode",
    "RuntimeStep",
    "RealityStoneStatus",
    "has_reality_stone",
    "reality_stone_status",
    "RuntimeAgent",
    "RuntimeAgentConfig",
    "RuntimeAgentStep",
    "RuntimeTextAgent",
    "RuntimeTextAgentTurn",
    "TextEnvironment",
    "TextEnvironmentStep",
    "auto_device",
    "safe_print",
    "normalize_vector",
    "resolve_device",
    "AD", "PORTAL", "BYPASS", "T_WAKE",
    "ACTIVE_RATIO", "STRUCT_RATIO", "BACKGROUND_RATIO",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/agent.py`

```python
"""Agent loop: Critic, Action, Bootstrap, Consciousness, Working Memory.

Implements 17_AgentLoop.md F.4, F.7, F.9-F.10, F.17, F.20.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .constants import (
        BOOTSTRAP_CONTRACTION, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV,
        CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD, META_MAX_DEPTH,
        WM_CAPACITY, CEREBELLUM_ALPHA, CEREBELLUM_ETA, NORM_EPS,
    )
    from .runtime import BrainRuntime, RuntimeMode, RuntimeStep
except ImportError:
    from reality_stone.clarus.constants import (
        BOOTSTRAP_CONTRACTION, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV,
        CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD, META_MAX_DEPTH,
        WM_CAPACITY, CEREBELLUM_ALPHA, CEREBELLUM_ETA, NORM_EPS,
    )
    from reality_stone.clarus.runtime import BrainRuntime, RuntimeMode, RuntimeStep


# ---------------------------------------------------------------------------
# F.4  Critic
# ---------------------------------------------------------------------------
@dataclass
class CriticResult:
    c_pred: float
    c_cons: float
    c_nov: float
    score: float


def compute_critic(
    observation: torch.Tensor,
    prediction: torch.Tensor,
    relaxed_state: torch.Tensor,
    recalled_state: torch.Tensor,
    obs_prior: torch.Tensor | None = None,
    w_pred: float = CRITIC_W_PRED,
    w_cons: float = CRITIC_W_CONS,
    w_nov: float = CRITIC_W_NOV,
) -> CriticResult:
    """c_{t+1} = w_p||c_pred|| + w_c||c_cons|| + w_n||c_nov|| (F.4)."""
    c_pred = float((observation - prediction).norm().item())
    c_cons = float((relaxed_state - recalled_state).norm().item())
    if obs_prior is not None and obs_prior.numel() == observation.numel():
        p_obs = torch.softmax(observation.float(), dim=-1).clamp(min=1e-8)
        p_prior = torch.softmax(obs_prior.float(), dim=-1).clamp(min=1e-8)
        c_nov = float(torch.sum(p_obs * (p_obs.log() - p_prior.log())).item())
        c_nov = max(c_nov, 0.0)
    else:
        c_nov = 0.0
    score = w_pred * c_pred + w_cons * c_cons + w_nov * c_nov
    return CriticResult(c_pred=c_pred, c_cons=c_cons, c_nov=c_nov, score=score)


# ---------------------------------------------------------------------------
# F.7  Action selection
# ---------------------------------------------------------------------------
def select_action_discrete(
    z_out: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> int:
    """a_t = argmax sim(enc(a), z_out) (F.7 discrete)."""
    z_norm = z_out / z_out.norm().clamp(min=NORM_EPS)
    a_norm = action_embeddings / action_embeddings.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
    similarity = a_norm @ z_norm
    return int(similarity.argmax().item())


def select_action_continuous(
    z_out: torch.Tensor,
    w_act: torch.Tensor,
    b_act: torch.Tensor,
) -> torch.Tensor:
    """a_t = W_act z_out + b_act (F.7 continuous)."""
    return w_act @ z_out + b_act


# ---------------------------------------------------------------------------
# F.9-F.10  Bootstrap operator
# ---------------------------------------------------------------------------
def bootstrap_operator(
    state: torch.Tensor,
    target: torch.Tensor | None = None,
    rho: float = BOOTSTRAP_CONTRACTION,
) -> torch.Tensor:
    """B: X -> p* + rho*(X - p*) (F.9)."""
    if target is None:
        target = torch.zeros_like(state)
    return target + rho * (state - target)


def agent_step(
    x_t: torch.Tensor,
    relax_delta: torch.Tensor,
    obs_delta: torch.Tensor,
    critic_delta: torch.Tensor,
    suppress_delta: torch.Tensor,
    lambda_r: float = 0.3,
    lambda_o: float = 0.2,
    lambda_c: float = 0.2,
    lambda_s: float = 0.1,
    rho: float = BOOTSTRAP_CONTRACTION,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """X_{t+1} = B[X_t + lambda_R*R + lambda_O*dO + lambda_C*C - lambda_S*S] (F.9)."""
    inner = (
        x_t
        + lambda_r * relax_delta
        + lambda_o * obs_delta
        + lambda_c * critic_delta
        - lambda_s * suppress_delta
    )
    return bootstrap_operator(inner, target=target, rho=rho)


# ---------------------------------------------------------------------------
# F.17  Consciousness / Metacognition
# ---------------------------------------------------------------------------
class ConsciousnessMonitor:
    """C3 self-consistency monitor and consciousness depth tracker."""

    def __init__(
        self,
        tau: float = CONSCIOUSNESS_TAU,
        c_d: float = CONSCIOUSNESS_CD,
        max_depth: int = META_MAX_DEPTH,
        rho: float = BOOTSTRAP_CONTRACTION,
    ) -> None:
        self.tau = tau
        self.c_d = c_d
        self.max_depth = max_depth
        self.rho = rho
        self._deviation_history: deque[float] = deque(maxlen=int(tau))

    def record_deviation(self, active_frac: float, target: float = ACTIVE_RATIO) -> None:
        p_current = torch.tensor([active_frac, 1.0 - active_frac])
        p_target = torch.tensor([target, 1.0 - target])
        dev = float((p_current - p_target).norm().item())
        self._deviation_history.append(dev)

    def d_tau(self) -> float:
        """d_tau(t) = (1/tau) * integral of ||p(s) - p*|| ds (F.17.2)."""
        if not self._deviation_history:
            return 0.0
        return sum(self._deviation_history) / max(len(self._deviation_history), 1)

    def consciousness_depth(self) -> float:
        """depth = exp(-c_d * d_tau) (F.17.2). Range [0, 1]."""
        return math.exp(-self.c_d * self.d_tau())

    def metacognition_step(self, deviation: float) -> list[float]:
        """Recursive self-evaluation: d_{n+1} <= rho * d_n (F.17.3)."""
        steps = [deviation]
        d = deviation
        for _ in range(self.max_depth - 1):
            d = self.rho * d
            steps.append(d)
        return steps


# ---------------------------------------------------------------------------
# F.20  Working Memory + Cerebellum
# ---------------------------------------------------------------------------
class WorkingMemory:
    """Finite FIFO buffer |h_t| <= T_h (F.20)."""

    def __init__(self, capacity: int = WM_CAPACITY) -> None:
        self.capacity = capacity
        self._buffer: deque[tuple[Any, Any]] = deque(maxlen=capacity)

    def append(self, action: Any, observation: Any) -> None:
        self._buffer.append((action, observation))

    def contents(self) -> list[tuple[Any, Any]]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class CerebellumPredictor:
    """Forward model: o_hat_{t+1} = o_hat_t + alpha*(o_t - o_hat_t) (F.20)."""

    def __init__(self, dim: int, alpha: float = CEREBELLUM_ALPHA, eta: float = CEREBELLUM_ETA) -> None:
        self.alpha = alpha
        self.eta = eta
        self.prediction = torch.zeros(dim)

    def predict(self) -> torch.Tensor:
        return self.prediction.clone()

    def update(self, observation: torch.Tensor) -> torch.Tensor:
        error = observation.float() - self.prediction
        self.prediction = self.prediction + self.alpha * error
        correction = -self.eta * error
        return correction


@dataclass
class RuntimeAgentConfig:
    """Minimal Layer-F control settings around BrainRuntime."""

    action_count: int = 4
    working_memory_capacity: int = WM_CAPACITY
    lambda_relax: float = 0.3
    lambda_observation: float = 0.2
    lambda_critic: float = 0.2
    lambda_suppress: float = 0.1
    rho: float = BOOTSTRAP_CONTRACTION

    def __post_init__(self) -> None:
        self.action_count = max(1, int(self.action_count))
        self.working_memory_capacity = max(1, int(self.working_memory_capacity))
        self.lambda_relax = float(self.lambda_relax)
        self.lambda_observation = float(self.lambda_observation)
        self.lambda_critic = float(self.lambda_critic)
        self.lambda_suppress = float(self.lambda_suppress)
        self.rho = min(max(float(self.rho), 0.0), 1.0)


@dataclass
class RuntimeAgentStep:
    """Observable result of one closed-loop Layer-F tick."""

    runtime_step: RuntimeStep
    action_index: int
    critic: CriticResult
    active_ratio: float
    consciousness_depth: float
    working_memory_size: int
    goal_norm: float


def default_action_embeddings(action_count: int, dim: int) -> torch.Tensor:
    """Deterministic action codebook used when no task-specific actions exist."""
    action_count = max(1, int(action_count))
    dim = max(1, int(dim))
    embeddings = torch.zeros(action_count, dim)
    for idx in range(action_count):
        embeddings[idx, idx % dim] = 1.0
        embeddings[idx] += (idx + 1) / (10.0 * action_count * dim)
    return embeddings


class RuntimeAgent:
    """Closed Layer-F loop over BrainRuntime.

    This is the smallest executable AGI core in the repo: runtime relaxation,
    action selection, environment observation, critic, working memory, forward
    model, consciousness monitor, and goal feedback happen in one tick.
    """

    def __init__(
        self,
        runtime: BrainRuntime,
        *,
        action_embeddings: torch.Tensor | None = None,
        config: RuntimeAgentConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or RuntimeAgentConfig()
        dim = self.runtime.config.dim
        if action_embeddings is None:
            action_embeddings = default_action_embeddings(self.config.action_count, dim)
        if action_embeddings.ndim != 2 or action_embeddings.shape[1] != dim:
            raise ValueError("action_embeddings must have shape (n_actions, runtime_dim)")
        self.action_embeddings = action_embeddings.detach().float().to(self.runtime.device)
        self.working_memory = WorkingMemory(capacity=self.config.working_memory_capacity)
        self.cerebellum = CerebellumPredictor(dim=dim)
        self.consciousness = ConsciousnessMonitor(rho=self.config.rho)
        self.self_state = torch.zeros(dim, device=self.runtime.device)

    def step(
        self,
        *,
        external_input: torch.Tensor | None = None,
        observation: torch.Tensor | None = None,
        obs_prior: torch.Tensor | None = None,
        force_mode: RuntimeMode | None = None,
    ) -> RuntimeAgentStep:
        runtime_step = self.runtime.step(
            external_input=external_input,
            force_mode=force_mode,
        )
        relaxed = self.runtime.activation.detach()
        action_index = select_action_discrete(relaxed, self.action_embeddings)

        observation_t = relaxed if observation is None else observation.detach().float().to(
            self.runtime.device
        ).view(self.runtime.config.dim)
        prediction = self.cerebellum.predict().to(self.runtime.device)
        recalled = self.runtime.hippocampus.recall(relaxed).to(self.runtime.device)
        prior_t = None if obs_prior is None else obs_prior.detach().float().to(self.runtime.device)
        critic = compute_critic(observation_t, prediction, relaxed, recalled, obs_prior=prior_t)

        correction = self.cerebellum.update(observation_t.detach().cpu()).to(self.runtime.device)
        self.working_memory.append(action_index, observation_t.detach().cpu())

        active_ratio = runtime_step.active_modules / max(self.runtime.config.dim, 1)
        self.consciousness.record_deviation(active_ratio)

        critic_delta = F.normalize(observation_t - prediction, dim=0) * critic.score
        self.self_state = agent_step(
            self.self_state,
            relax_delta=relaxed - self.self_state,
            obs_delta=observation_t - relaxed,
            critic_delta=critic_delta,
            suppress_delta=self.runtime.refractory + correction,
            lambda_r=self.config.lambda_relax,
            lambda_o=self.config.lambda_observation,
            lambda_c=self.config.lambda_critic,
            lambda_s=self.config.lambda_suppress,
            rho=self.config.rho,
        )
        self.runtime.set_goal(self.self_state)

        return RuntimeAgentStep(
            runtime_step=runtime_step,
            action_index=action_index,
            critic=critic,
            active_ratio=active_ratio,
            consciousness_depth=self.consciousness.consciousness_depth(),
            working_memory_size=len(self.working_memory),
            goal_norm=float(self.runtime.goal.norm().item()),
        )


@dataclass
class TextEnvironmentStep:
    """Result of applying one discrete agent action to a text environment."""

    prompt: str
    action_label: str
    response: str
    observation: torch.Tensor
    done: bool = False
    info: dict[str, float | int | str] = field(default_factory=dict)


class TextEnvironment:
    """Deterministic text-to-vector environment for RuntimeAgent episodes."""

    def __init__(
        self,
        dim: int,
        *,
        actions: list[str] | None = None,
        history_limit: int = 8,
    ) -> None:
        self.dim = max(1, int(dim))
        self.actions = actions or ["answer", "clarify", "recall", "reflect"]
        if not self.actions:
            raise ValueError("TextEnvironment requires at least one action")
        self.history: deque[tuple[str, str]] = deque(maxlen=max(1, int(history_limit)))
        self.current_prompt = ""

    def encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        terms = [part for part in str(text).replace("\n", " ").split(" ") if part]
        if not terms:
            return vec
        for position, term in enumerate(terms):
            digest = hashlib.blake2b(f"{position}:{term}".encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign / math.sqrt(position + 1.0)
        norm = vec.norm()
        if norm.item() < NORM_EPS:
            return vec
        return vec / norm

    def action_embeddings(self) -> torch.Tensor:
        return torch.stack([self.encode(action) for action in self.actions], dim=0)

    def reset(self, prompt: str) -> torch.Tensor:
        self.current_prompt = str(prompt)
        return self.encode(self.current_prompt)

    def step(self, action_index: int, state: torch.Tensor | None = None) -> TextEnvironmentStep:
        idx = int(action_index) % len(self.actions)
        action = self.actions[idx]
        response = self._render_response(action, state)
        self.history.append((action, response))
        observation_text = f"{self.current_prompt}\naction: {action}\nresponse: {response}"
        observation = self.encode(observation_text)
        return TextEnvironmentStep(
            prompt=self.current_prompt,
            action_label=action,
            response=response,
            observation=observation,
            done=False,
            info={"history_size": len(self.history)},
        )

    def _render_response(self, action: str, state: torch.Tensor | None) -> str:
        state_norm = 0.0 if state is None else float(state.detach().float().norm().item())
        if action == "clarify":
            return f"clarify: {self.current_prompt}"
        if action == "recall":
            last = "none" if not self.history else self.history[-1][1]
            return f"recall: {last}"
        if action == "reflect":
            return f"reflect: state_norm={state_norm:.4f}"
        return f"answer: {self.current_prompt}"


@dataclass
class RuntimeTextAgentTurn:
    """One text-environment turn containing both agent and environment outputs."""

    agent_step: RuntimeAgentStep
    env_step: TextEnvironmentStep


class RuntimeTextAgent:
    """RuntimeAgent wired to a deterministic text environment."""

    def __init__(
        self,
        runtime: BrainRuntime,
        *,
        environment: TextEnvironment | None = None,
        config: RuntimeAgentConfig | None = None,
    ) -> None:
        env = environment or TextEnvironment(runtime.config.dim)
        agent_config = config or RuntimeAgentConfig(action_count=len(env.actions))
        self.environment = env
        self.agent = RuntimeAgent(
            runtime,
            action_embeddings=env.action_embeddings(),
            config=agent_config,
        )

    def ask(
        self,
        prompt: str,
        *,
        ticks: int = 1,
        force_mode: RuntimeMode = RuntimeMode.WAKE,
    ) -> RuntimeTextAgentTurn:
        observation = self.environment.reset(prompt)
        prior = None
        last_turn = None
        for _ in range(max(1, int(ticks))):
            agent_out = self.agent.step(
                external_input=observation,
                observation=observation,
                obs_prior=prior,
                force_mode=force_mode,
            )
            env_out = self.environment.step(agent_out.action_index, self.agent.self_state)
            prior = observation
            observation = env_out.observation
            last_turn = RuntimeTextAgentTurn(agent_step=agent_out, env_step=env_out)
        if last_turn is None:
            raise RuntimeError("RuntimeTextAgent did not execute a turn")
        return last_turn
```
---
## File: `reality_stone/python/reality_stone/clarus/bitfield.py`

```python
"""Bitfield runtime: 4-bit quantized CE engine per 12_Equation.md 1.6-1.7.

Euler identity e^{i*pi}+1=0 maps to 4 operations:
  0 -> CLEAR (prune, reset)
  1 -> IDENTITY (keep, normalize)
  e -> DECAY (EMA, sleep pressure via shift-add)
  pi -> RADIUS (connection mask, neighbor rule)
  i -> MODE (2-bit mode register: off/wake/nrem/rem)

State representation:
  Control: bitfield O(N) bits -- active mask, mode, freeze
  State:   4-bit fixed-point O(N) bytes -- activation, trace
  Weight:  4-bit sparse O(N*K) -- CSR with 4-bit values
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

try:
    from .constants import (
        ACTIVE_RATIO, MEMORY_TRACE_DECAY, ADAPTATION_COUPLING,
        ADAPTATION_DECAY, ADAPTATION_CLAMP, NOISE_SIGMA,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        SPARSITY_RADIUS, NORM_EPS,
    )
except ImportError:
    from reality_stone.clarus.constants import (
        ACTIVE_RATIO, MEMORY_TRACE_DECAY, ADAPTATION_COUPLING,
        ADAPTATION_DECAY, ADAPTATION_CLAMP, NOISE_SIGMA,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        SPARSITY_RADIUS, NORM_EPS,
    )


def quantize_4bit(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize float tensor to 4-bit signed [-7, 7] with scale/zero."""
    x_min = x.min().item()
    x_max = x.max().item()
    span = max(x_max - x_min, 1e-8)
    scale = span / 15.0
    zero = x_min
    q = ((x - zero) / scale).round().clamp(0, 15).to(torch.uint8)
    return q, scale, zero


def dequantize_4bit(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    return q.float() * scale + zero


def quantize_8bit(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize to 8-bit unsigned [0, 255]."""
    x_min = x.min().item()
    x_max = x.max().item()
    span = max(x_max - x_min, 1e-8)
    scale = span / 255.0
    q = ((x - x_min) / scale).round().clamp(0, 255).to(torch.uint8)
    return q, scale, x_min


def dequantize_8bit(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    return q.float() * scale + zero


@dataclass
class BitfieldLayout:
    """Memory layout per 12_Equation.md 1.7 (N=dim basis)."""
    dim: int
    avg_neighbors: int = 130

    @property
    def active_mask_bytes(self) -> int:
        return math.ceil(self.dim / 8)

    @property
    def freeze_mask_bytes(self) -> int:
        return math.ceil(self.dim / 8)

    @property
    def mode_bytes(self) -> int:
        return 1

    @property
    def weight_bytes(self) -> int:
        nnz = self.dim * self.avg_neighbors
        return nnz // 2  # 4-bit packed

    @property
    def csr_index_bytes(self) -> int:
        return self.dim * self.avg_neighbors * 2 + (self.dim + 1) * 4

    @property
    def state_bytes(self) -> int:
        return self.dim  # 8-bit per cell

    @property
    def phi_bytes(self) -> int:
        return self.dim  # 8-bit

    @property
    def trace_bytes(self) -> int:
        nnz = self.dim * self.avg_neighbors
        return nnz // 2  # 4-bit

    @property
    def total_engine_bytes(self) -> int:
        return (self.active_mask_bytes + self.freeze_mask_bytes +
                self.mode_bytes + self.weight_bytes + self.csr_index_bytes +
                self.state_bytes + self.phi_bytes + self.trace_bytes + 6)

    def summary(self) -> dict[str, str]:
        return {
            "active_mask": f"{self.active_mask_bytes} B",
            "freeze_mask": f"{self.freeze_mask_bytes} B",
            "mode": f"{self.mode_bytes} B",
            "weights_4bit": f"{self.weight_bytes / 1024:.1f} KB",
            "csr_index": f"{self.csr_index_bytes / 1024:.1f} KB",
            "state_8bit": f"{self.state_bytes / 1024:.1f} KB",
            "phi_8bit": f"{self.phi_bytes / 1024:.1f} KB",
            "trace_4bit": f"{self.trace_bytes / 1024:.1f} KB",
            "total_engine": f"{self.total_engine_bytes / 1024:.1f} KB",
        }


class BitfieldRuntime:
    """Quantized CE runtime operating on 4/8-bit state.

    All arithmetic stays in integer domain where possible.
    Dequantize only for tanh and final output.
    """

    def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:
        self.dim = weight.shape[0]
        self.active_ratio = active_ratio

        w_sparse = weight.clone()
        w_sparse[weight.abs() < 1e-4] = 0
        self.w_q, self.w_scale, self.w_zero = quantize_4bit(w_sparse)
        self.w_mask = w_sparse != 0

        self.activation_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.act_scale = 2.0 / 255.0
        self.act_zero = -1.0

        self.refractory_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.ref_scale = 2.0 / 255.0
        self.ref_zero = 0.0

        self.trace_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.trace_scale = 2.0 / 255.0
        self.trace_zero = -1.0

        self.active_mask = torch.zeros(self.dim, dtype=torch.bool)
        self.mode = 0b01  # wake

        self._float_activation = torch.zeros(self.dim)
        self._float_refractory = torch.zeros(self.dim)
        self._float_trace = torch.zeros(self.dim)

    def step(self, external: torch.Tensor | None = None) -> dict[str, float]:
        """One tick: sparse active-only computation, minimal dequantize."""
        if external is None:
            external = torch.zeros(self.dim)

        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        n_active = active_idx.numel()

        if n_active == 0:
            recurrent = torch.zeros(self.dim)
        else:
            active_vals = self._float_activation[active_idx]
            w_cols = self.w_q[:, active_idx].float() * self.w_scale + self.w_zero
            w_cols = w_cols * self.w_mask[:, active_idx].float()
            recurrent = w_cols @ active_vals

        drive = recurrent + 0.45 * external - 0.35 * self._float_refractory - ADAPTATION_COUPLING * self._float_trace

        gamma_a = 0.18 if self.mode == 0b01 else 0.34
        kappa_a = 0.82 if self.mode == 0b01 else 0.52

        new_act = ((1.0 - gamma_a) * self._float_activation + kappa_a * torch.tanh(drive)).clamp(-1, 1)
        new_ref = (1.0 - 0.12) * self._float_refractory + 0.24 * new_act.square()
        new_trace = (1.0 - MEMORY_TRACE_DECAY) * self._float_trace + MEMORY_TRACE_DECAY * new_act

        self._float_activation = new_act
        self._float_refractory = new_ref
        self._float_trace = new_trace

        k = max(1, int(self.active_ratio * self.dim))
        topk_idx = torch.topk(new_act.abs(), k).indices
        self.active_mask.zero_()
        self.active_mask[topk_idx] = True

        energy = -0.5 * torch.dot(new_act[active_idx], recurrent[active_idx]).item() if n_active > 0 else 0.0
        active_count = int(self.active_mask.sum().item())

        return {
            "energy": energy,
            "active": active_count,
            "active_ratio": active_count / self.dim,
            "act_norm": float(new_act.norm().item()),
        }

    def get_activation(self) -> torch.Tensor:
        return self._float_activation

    def memory_bytes(self) -> int:
        """Actual quantized memory usage."""
        w_bytes = self.w_q.numel()  # uint8, but 4-bit packed would halve
        act_bytes = self.activation_q.numel()
        ref_bytes = self.refractory_q.numel()
        trace_bytes = self.trace_q.numel()
        mask_bytes = math.ceil(self.dim / 8)
        return w_bytes // 2 + act_bytes + ref_bytes + trace_bytes + mask_bytes + 1


class Float32Runtime:
    """Reference float32 runtime for comparison."""

    def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:
        self.dim = weight.shape[0]
        self.active_ratio = active_ratio
        self.weight = weight.float()
        self.activation = torch.zeros(self.dim)
        self.refractory = torch.zeros(self.dim)
        self.trace = torch.zeros(self.dim)
        self.active_mask = torch.zeros(self.dim, dtype=torch.bool)

    def step(self, external: torch.Tensor | None = None) -> dict[str, float]:
        if external is None:
            external = torch.zeros(self.dim)

        recurrent = self.weight @ (self.activation * self.active_mask.float())
        drive = recurrent + 0.45 * external - 0.35 * self.refractory - ADAPTATION_COUPLING * self.trace

        new_act = ((1.0 - 0.18) * self.activation + 0.82 * torch.tanh(drive)).clamp(-1, 1)
        new_ref = (1.0 - 0.12) * self.refractory + 0.24 * new_act.square()
        new_trace = (1.0 - MEMORY_TRACE_DECAY) * self.trace + MEMORY_TRACE_DECAY * new_act

        self.activation = new_act
        self.refractory = new_ref
        self.trace = new_trace

        k = max(1, int(self.active_ratio * self.dim))
        topk_idx = torch.topk(new_act.abs(), k).indices
        self.active_mask.zero_()
        self.active_mask[topk_idx] = True

        energy = -0.5 * torch.dot(new_act, recurrent).item()
        active_count = int(self.active_mask.sum().item())

        return {
            "energy": energy,
            "active": active_count,
            "active_ratio": active_count / self.dim,
            "act_norm": float(new_act.norm().item()),
        }

    def get_activation(self) -> torch.Tensor:
        return self.activation

    def memory_bytes(self) -> int:
        return (self.weight.numel() + self.activation.numel() * 3) * 4


def benchmark(dim: int = 768, steps: int = 200, seed: int = 42) -> dict:
    """Compare bitfield vs float32 runtime on identical inputs."""
    torch.manual_seed(seed)
    w = torch.randn(dim, dim) * 0.01
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    w[w.abs() < 0.005] = 0  # sparsify

    bf = BitfieldRuntime(w)
    fp = Float32Runtime(w)

    inputs = [torch.randn(dim) * 0.3 for _ in range(steps)]

    t0 = time.perf_counter()
    bf_results = []
    for ext in inputs:
        bf_results.append(bf.step(ext))
    bf_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fp_results = []
    for ext in inputs:
        fp_results.append(fp.step(ext))
    fp_time = time.perf_counter() - t0

    act_bf = bf.get_activation()
    act_fp = fp.get_activation()
    cos_sim = torch.cosine_similarity(act_bf.unsqueeze(0), act_fp.unsqueeze(0)).item()
    mse = ((act_bf - act_fp) ** 2).mean().item()
    max_err = (act_bf - act_fp).abs().max().item()

    e_bf = [r["energy"] for r in bf_results]
    e_fp = [r["energy"] for r in fp_results]
    energy_corr = torch.stack([torch.tensor(e_bf), torch.tensor(e_fp)]).corrcoef()[0, 1].item()

    bf_mem = bf.memory_bytes()
    fp_mem = fp.memory_bytes()
    layout = BitfieldLayout(dim)

    return {
        "dim": dim,
        "steps": steps,
        "bitfield_time_ms": bf_time * 1000,
        "float32_time_ms": fp_time * 1000,
        "speedup": fp_time / max(bf_time, 1e-9),
        "bitfield_step_us": bf_time / steps * 1e6,
        "float32_step_us": fp_time / steps * 1e6,
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_absolute_error": max_err,
        "energy_correlation": energy_corr,
        "bitfield_memory_KB": bf_mem / 1024,
        "float32_memory_KB": fp_mem / 1024,
        "memory_ratio": bf_mem / max(fp_mem, 1),
        "theoretical_engine_KB": layout.total_engine_bytes / 1024,
        "layout": layout.summary(),
    }
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_euler.py`

```python
"""Euler-bitfield attention — {e, π, i, 1, 0} as minimal dimensionless grammar.

CE's docs/상수.md treats the Euler-identity constants {e, π, i, 1, 0}
as the minimum vocabulary that generates dimensionless cores. We apply
the same principle to positional/rotational attention encoding:

  - 5 "Euler basis" frequencies: B = {1, π, e, π·e, π/e}
  - Each attention head picks a BITFIELD b ∈ {0,1}^5 selecting the
    subset of frequencies it uses. The head's positional phase is

        theta_head(pos, k) = pos · sum_{j: b_j=1} B_j · 2^{-k / d_head}

    i.e. a sum of Euler-basis log-spaced frequencies.
  - Q and K are rotated by theta_head before scoring (RoPE-style).

The bitfield b is a learnable real-valued vector passed through a
sigmoid (soft bitfield) so gradients flow. At inference it thresholds
to {0, 1} for pure Euler-combinations.

Why this could help:
  1. Multi-base (e-based rotations don't align with π-based ones on any
     finite period) — unique long-range attention signatures.
  2. Compression: 5 bits per head vs O(d) free parameters.
  3. Theory grounding: CE's minimal grammar axiom.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EULER_BASIS = (1.0, math.pi, math.e, math.pi * math.e, math.pi / math.e)
EULER_BASIS_NAMES = ("1", "pi", "e", "pi*e", "pi/e")


# ---------------------------------------------------------------------------
# CE first-principle constants for rotary base.
# ---------------------------------------------------------------------------
# Φ-relaxation coupling from CE physics layer (alpha_s at M_Z).
ALPHA_S = 0.11789
# CE effective dimension D_eff = 3 + δ where δ = sin²θ_W·(1-sin²θ_W) is the
# electroweak residual induced by Φ-relaxation. Numerically D_eff ≈ 3.178.
_SIN2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
D_EFF = 3.0 + _SIN2 * (1.0 - _SIN2)


def ce_rotary_base(block: int, layer_idx: int = 0, n_layers: int = 1,
                   depth_aware: bool = False) -> float:
    """CE-faithful rotary base for π-phase encoding.

    base = π^(D_eff · depth_factor) · block

    Two first-principle factors:
      * π^D_eff : CE dimensional volume (replaces RoPE's empirical 10⁴).
      * × block : causal-cone scaling — keeps the slowest rotary mode
                  near-DC inside the context window for any block size.
                  This is the "step커지면 그만큼" correction: if the
                  sequence length grows N×, the base also grows N×, so
                  the longest period stays a fixed fraction of the window.
      * depth_factor (when depth_aware=True): per-layer RG-running of the
        effective dimension. depth_factor_ℓ = 1 + ℓ/(L-1) ∈ [1, 2].
        Compensates the cumulative phase added by stacking L layers.
    """
    if depth_aware and n_layers > 1:
        depth_factor = 1.0 + layer_idx / (n_layers - 1)
    else:
        depth_factor = 1.0
    return (math.pi ** (D_EFF * depth_factor)) * float(block)


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to adjacent dim pairs: [x0, x1] -> [x0·c - x1·s, x0·s + x1·c]."""
    # x: (..., n, d_head) with d_head even
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# ---------------------------------------------------------------------------
# Shared SDPA helpers — used by EulerRotaryAttention, EulerCEAttention,
# and EulerCEMinimal so the FlashAttention-style Q-tiling + cached causal
# bias live in a single place.
# ---------------------------------------------------------------------------
Q_CHUNK_DEFAULT: int = 256
Q_CHUNK_THRESHOLD: int = 1024


def _causal_softmax_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain causal SDPA (no additive bias). FlashAttention on CUDA, tiled
    math path on CPU — either way the `(B, H, N, N)` scores tensor never
    materializes."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _chunked_decay_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    xi_per_head: torch.Tensor,
    pos: torch.Tensor,
    n: int,
    *,
    e_gate: Optional[torch.Tensor] = None,
    q_chunk: int = Q_CHUNK_DEFAULT,
    q_chunk_threshold: int = Q_CHUNK_THRESHOLD,
) -> torch.Tensor:
    """SDPA with per-head ALiBi-style decay + causal mask, Q-tiled.

    Mask:  `m[h, i, j] = -|i-j| · e_gate[h] / xi[h]`  for `j ≤ i`,
                         `-inf`                       otherwise.

    Tiling cuts peak mask memory from `O(H·N·N)` to `O(H·Q_CHUNK·N)` —
    at N=4096, H=16, that is 32 MB instead of 1 GB. The per-row softmax
    is independent over the K axis, so the tiling is exact.
    """
    h = q.shape[1]
    rate_dev = q.device
    rate_dtype = q.dtype
    if e_gate is not None:
        bias_rate = (e_gate.to(rate_dtype) / xi_per_head.to(rate_dtype)).view(h, 1, 1)
    else:
        bias_rate = (1.0 / xi_per_head.to(rate_dtype)).view(h, 1, 1)
    cols = pos[:n].to(rate_dtype).view(1, 1, n)
    zero_s = torch.zeros((), dtype=rate_dtype, device=rate_dev)
    neg_inf_s = torch.full((), float("-inf"), dtype=rate_dtype, device=rate_dev)

    if n < q_chunk_threshold:
        rows = pos[:n].to(rate_dtype).view(1, n, 1)
        d_full = (rows - cols).abs()
        causal = torch.where(rows >= cols, zero_s, neg_inf_s)
        mask = (-d_full * bias_rate + causal).unsqueeze(0)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    out = torch.empty_like(q)
    for qs in range(0, n, q_chunk):
        qe = min(qs + q_chunk, n)
        rows = pos[qs:qe].to(rate_dtype).view(1, qe - qs, 1)
        d_chunk = (rows - cols).abs()
        causal_chunk = torch.where(rows >= cols, zero_s, neg_inf_s)
        mask_chunk = (-d_chunk * bias_rate + causal_chunk).unsqueeze(0)
        out[:, :, qs:qe] = F.scaled_dot_product_attention(
            q[:, :, qs:qe], k, v, attn_mask=mask_chunk
        )
    return out


class EulerRotaryAttention(nn.Module):
    """Multi-head attention with Euler-bitfield rotary positional encoding.

    Args:
        d_model, n_heads, block: as usual
        softmax_bitfield: if True, soft bitfield via sigmoid(logits).
            Otherwise hard 0/1 bitfield from init template.
        init_bits: initial bitfield per head. Shape (n_heads, 5).
            Default: each head uses all 5 bases.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        softmax_bitfield: bool = True,
        init_bits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even for rotary"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        basis = torch.tensor(EULER_BASIS, dtype=torch.float32)  # (5,)
        self.register_buffer("euler_basis", basis)

        if init_bits is None:
            init_bits = torch.zeros(n_heads, 5)  # start at 0 -> sigmoid 0.5
        if softmax_bitfield:
            self.bit_logits = nn.Parameter(init_bits)
        else:
            self.register_buffer("bit_logits", init_bits)

        # log-space-frequency exponents: k = 0, 2, 4, ..., d_head-2  (per pair)
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", 2.0 ** (-k))  # (d_head/2,)

        # positions
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

    def bitfield(self) -> torch.Tensor:
        """(n_heads, 5) soft bitfield in [0, 1]."""
        return torch.sigmoid(self.bit_logits)

    def head_freq_scalars(self) -> torch.Tensor:
        """Per-head frequency scalar (weighted sum over Euler basis)."""
        b = self.bitfield()  # (n_heads, 5)
        return torch.matmul(b, self.euler_basis)  # (n_heads,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (b, n, h, d_head)
        q = q.transpose(1, 2)  # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build head-specific rotation angles: theta[h, n, k/2] = pos[n] * freq[h] * inv_freq[k]
        freqs = self.head_freq_scalars()  # (h,)
        # theta = pos[n] * freq[h] * inv_freq[k]
        theta = self.pos[:n].view(1, 1, n, 1) * freqs.view(1, self.n_heads, 1, 1) \
                * self.inv_freq.view(1, 1, 1, -1)  # (1, h, n, d_head/2)
        cos = theta.cos()
        sin = theta.sin()

        q = _rotate_pairs(q, cos, sin)
        k = _rotate_pairs(k, cos, sin)

        # SDPA with causal mask — no `(B, H, N, N)` scores tensor, no
        # explicit softmax. Output is bit-identical to the prior manual
        # path within float precision.
        out = _causal_softmax_sdpa(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerAttnBlock(nn.Module):
    """Transformer block using Euler rotary attention."""

    def __init__(self, d_model: int, n_heads: int, block: int,
                 softmax_bitfield: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerRotaryAttention(d_model, n_heads, block,
                                         softmax_bitfield=softmax_bitfield)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


__all__ = [
    "EULER_BASIS",
    "EULER_BASIS_NAMES",
    "EulerRotaryAttention",
    "EulerAttnBlock",
    "EulerCEAttention",
    "EulerCEBlock",
    "EulerCEMinimal",
    "EulerCEMinimalBlock",
    "RecursiveEulerCEBlock",
    "fixed_point_loss",
    "head_types_from_spec",
]


# ---------------------------------------------------------------------------
# EulerCEAttention — theory-correct assignment of {e, π, i, 1, 0}
# ---------------------------------------------------------------------------
#
# Per docs/경로적분.md (lines 51-67):
#   e   -> survival / decay:        S(D) = e^{-D}
#   π   -> periodic normalization:  α_total = 1 / (2π)
#   i   -> path-integral phase:     Z = ∫ Dφ e^{iS/ℏ}
#   1   -> normalized complete state
#   0   -> zero / branch selection
#
# The attention kernel is therefore
#
#   A_ij  =  softmax_j ( Q_i · R_π(i-j) · K_j / √d )  ·  e^{-|i-j|/ξ_e}
#                       └─── π-phase rotary ───┘        └── e decay ──┘
#
#     R_π(Δ): RoPE-style rotation with fundamental period π (not 10^4).
#     ξ_e:    learnable correlation length (decay base e).
#
# The "bitfield" selects which of {π-phase, e-decay, 1-bypass} are active
# per head. A head with bit=0 for π uses identity rotation (no RoPE);
# bit=0 for e means no distance decay. This exposes the 5-constant
# minimum vocabulary as an interpretable head-type switch.


class EulerCEAttention(nn.Module):
    """Theory-correct Euler attention: π-phase + e-decay + {1,0} gates.

    Args:
        d_model, n_heads, block: standard
        xi_init: initial correlation length for the e-decay term (in
            positions). Larger = weaker decay. Default block/2.
        learnable_gates: if True, the two gates (pi_gate, e_gate) are
            per-head learnable sigmoids; if False they are frozen at 1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        xi_init: Optional[float] = None,
        learnable_gates: bool = True,
        layer_idx: int = 0,
        n_layers: int = 1,
        depth_aware_freq: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("tril",
                             torch.tril(torch.ones(block, block, dtype=torch.bool)))

        # π-phase rotary base — see ce_rotary_base() for derivation.
        base = ce_rotary_base(block, layer_idx=layer_idx,
                              n_layers=n_layers,
                              depth_aware=depth_aware_freq)
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("pi_inv_freq", base ** (-k))
        self.register_buffer("pos", torch.arange(block, dtype=torch.float32))

        # e-decay: log xi so xi > 0 strictly. Initialize xi = block/8 so the
        # decay actually bites in [0, block]: e^{-block/xi} = e^{-8} ≈ 3e-4.
        # block/2 made the decay numerically negligible.
        if xi_init is None:
            xi_init = block / 8.0
        self.log_xi = nn.Parameter(torch.full((n_heads,),
                                              math.log(xi_init), dtype=torch.float32))

        # Per-head gates. sigmoid(1.0) ≈ 0.73 starts mild but learnable.
        if learnable_gates:
            self.pi_gate_logit = nn.Parameter(torch.full((n_heads,), 1.0))
            self.e_gate_logit = nn.Parameter(torch.full((n_heads,), 1.0))
        else:
            self.register_buffer("pi_gate_logit", torch.full((n_heads,), 1e4))
            self.register_buffer("e_gate_logit", torch.full((n_heads,), 1e4))

        # Precompute |i-j| distance matrix (non-negative, upper-tri set by mask)
        d_mat = (torch.arange(block).unsqueeze(1) - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional / distance buffers for length-extrapolation eval.
        Learnable parameters (log_xi, gates, qkv, o) are unchanged. The
        rotary base (pi_inv_freq) is intentionally kept at its training-time
        value — that is the *block-aware* design point of EulerCE."""
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        dev = self.pos.device
        self.pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        d = (torch.arange(new_block).unsqueeze(1)
             - torch.arange(new_block).unsqueeze(0)).abs().float().to(dev)
        self.d_mat = d

    def _rotate(self, x, cos, sin):
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        pi_g = torch.sigmoid(self.pi_gate_logit)      # (h,)
        e_g = torch.sigmoid(self.e_gate_logit)        # (h,)

        # π-phase rotation (per-head amplitude scaled by pi_g).
        theta = self.pos[:n].view(1, 1, n, 1) * self.pi_inv_freq.view(1, 1, 1, -1)
        theta = theta * pi_g.view(1, self.n_heads, 1, 1)
        cos = theta.cos(); sin = theta.sin()
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        # SDPA with per-head ALiBi decay, Q-tiled (FlashAttention-style).
        # Algebraically identical to `softmax(QK/√d - e_g·|i-j|/ξ + causal)·V`
        # but never materializes the `(B, H, N, N)` scores tensor.
        xi = torch.exp(self.log_xi)                   # (h,)
        out = _chunked_decay_sdpa(
            q_rot, k_rot, v, xi, self.pos, n, e_gate=e_g
        )
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)


class EulerCEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block: int,
                 learnable_gates: bool = True,
                 layer_idx: int = 0, n_layers: int = 1,
                 depth_aware_freq: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEAttention(d_model, n_heads, block,
                                     learnable_gates=learnable_gates,
                                     layer_idx=layer_idx, n_layers=n_layers,
                                     depth_aware_freq=depth_aware_freq)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# RecursiveEulerCEBlock — self-referential fixed-point iteration (ClarusCell)
# ---------------------------------------------------------------------------
#
# CE bootstrap equation:   epsilon^2 = exp[-(1 - epsilon^2) * D_eff]
# This is a fixed-point equation x* = F(x*). A CE-faithful transformer
# block should therefore be allowed to apply itself repeatedly to its
# own output until convergence, rather than being a one-shot function.
#
# Two semantics offered:
#
# 1. FIXED DEPTH RECURSION (``max_iters=k``, ``tol=None``):
#    h_0 = x;  h_{t+1} = F(h_t);  out = h_k
#    "Universal Transformer" style, weights shared across depth.
#
# 2. WHILE-LOOP RECURSION (``tol>0``):
#    halt when ||h_{t+1} - h_t|| / ||h_t|| < tol  OR  t == max_iters.
#    The halting depth is recorded in ``.last_depths`` for analysis.
#    Non-differentiable halt; backprop flows through the final path.
#
# The optional self-consistency loss
#
#    L_fp = || F(F(h*)) - F(h*) ||^2
#
# pulls the output h* = F(x) toward being a true fixed point.


class RecursiveEulerCEBlock(nn.Module):
    """Self-referential transformer block — ClarusCell as while-loop.

    Args:
        d_model, n_heads, block: standard
        max_iters: maximum number of self-applications (>=1)
        tol: if not None, halt when relative change is below this
             threshold. If None, always run ``max_iters`` iterations.
        learnable_gates: forwarded to EulerCEAttention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        max_iters: int = 1,
        tol: Optional[float] = None,
        learnable_gates: bool = True,
        layer_idx: int = 0,
        n_layers: int = 1,
        depth_aware_freq: bool = False,
        depth_aware_iters: bool = False,
    ) -> None:
        super().__init__()
        self.core = EulerCEBlock(d_model, n_heads, block,
                                 learnable_gates=learnable_gates,
                                 layer_idx=layer_idx, n_layers=n_layers,
                                 depth_aware_freq=depth_aware_freq)
        # depth_aware_iters: deeper layers get more self-iterations to
        # compensate accumulated representational complexity. Schedule:
        #   iters_ℓ = max_iters + ℓ
        # so a 4-layer stack with max_iters=1 yields {1,2,3,4} effective.
        if depth_aware_iters:
            self.max_iters = max_iters + layer_idx
        else:
            self.max_iters = max_iters
        self.tol = tol
        self.last_depths: Optional[torch.Tensor] = None

    def _step(self, h: torch.Tensor) -> torch.Tensor:
        return self.core(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        depths = torch.full((x.shape[0],), self.max_iters,
                            dtype=torch.long, device=x.device)

        if self.tol is None:
            # Fixed-depth recursion (differentiable through all steps).
            for _ in range(self.max_iters):
                h = self._step(h)
            self.last_depths = depths
            return h

        # While-loop recursion with halting on tolerance.
        for t in range(self.max_iters):
            h_new = self._step(h)
            # per-example relative change (detached — halt decision is not diff)
            with torch.no_grad():
                num = (h_new - h).flatten(1).norm(dim=-1)
                den = h.flatten(1).norm(dim=-1).clamp_min(1e-8)
                rel = num / den
                # mark examples that have newly halted
                halted = rel < self.tol
                not_yet = depths == self.max_iters
                just_halted = halted & not_yet
                depths = torch.where(just_halted,
                                     torch.full_like(depths, t + 1),
                                     depths)
                if halted.all():
                    h = h_new
                    break
            h = h_new
        self.last_depths = depths
        return h


def fixed_point_loss(block: RecursiveEulerCEBlock, h: torch.Tensor,
                     scale: float = 1.0) -> torch.Tensor:
    """||F(F(h)) - F(h)||^2 averaged over batch and positions.

    Pulls h toward being a fixed point of ``block.core``. Use as a
    regularizer added to cross-entropy.
    """
    with torch.no_grad():
        fh = block.core(h)
    ffh = block.core(fh)
    diff = (ffh - fh).flatten(1)
    return scale * diff.pow(2).mean()


# ---------------------------------------------------------------------------
# EulerCEMinimal — 2-bit head-type taxonomy
# ---------------------------------------------------------------------------
#
# Operational reduction of {e, π, i, 1, 0}:
#   {π, i}  → rotation generator   (always paired as e^{iπt} = (cos, sin))
#   {e}     → exponential decay
#   {0, 1}  → on/off gate values   (1 bit each, by definition)
#
# → 2 functionally distinct axes (rotation, decay) × 2 gate values
#   = 2² = 4 head-types, each encoded by a 2-bit string (pi_bit, e_bit):
#
#       (pi, e)   head-type       canonical literature analogue
#       --------  --------------  -------------------------------
#       (0, 0)    identity        NoPE      (Kazemnejad 2023)
#       (0, 1)    decay only      ALiBi     (Press 2022)
#       (1, 0)    rotation only   RoPE      (Su 2021)
#       (1, 1)    rotation+decay  xPos      (Sun 2023) / EulerCE
#
# Per-head continuous parameters (only meaningful when bit is on):
#       pi_base : rotary base (RoPE-style geometric, defaults to 10000)
#       xi_h    : decay length (per-head learnable, default block/8)
#
# Empirical finding (`docs/8_리만/mra_paper.md` § 7.7, length extrap. ablation):
#   Head-type (1, 0) — pure rotation — is the only Tier 2 (catastrophic
#   length-OOD). The other three head-types are Tier 1 (extrapolate).
#   So among the 2² = 4 types, 3 are operationally useful → log₂ 3 ≈ 1.58
#   bits is the effective head-type capacity.


_HEAD_TYPE_NAMES = ("nope", "alibi", "rope", "xpos")  # indexed by 2*pi + e


def head_types_from_spec(spec, n_heads: int) -> torch.Tensor:
    """Convert a head-type spec into a (n_heads,) int tensor in {0,1,2,3}.

    Acceptable spec forms:
      * int in [0, 3]           — uniform (all heads same type)
      * list/tuple of length n  — per-head type values
      * str  in {"nope", "alibi", "rope", "xpos"} — uniform name
      * str  "mix" — alternating alibi / xpos
      * str  "all" — round-robin {nope, alibi, rope, xpos}
    """
    name_to_idx = {n: i for i, n in enumerate(_HEAD_TYPE_NAMES)}
    if isinstance(spec, str):
        if spec == "mix":
            ts = [(1 if h % 2 == 0 else 3) for h in range(n_heads)]
            return torch.tensor(ts, dtype=torch.long)
        if spec == "all":
            ts = [h % 4 for h in range(n_heads)]
            return torch.tensor(ts, dtype=torch.long)
        if spec in name_to_idx:
            return torch.full((n_heads,), name_to_idx[spec], dtype=torch.long)
        raise ValueError(f"unknown head-type spec: {spec!r}")
    if isinstance(spec, int):
        if not 0 <= spec <= 3:
            raise ValueError(f"head-type int must be in [0, 3], got {spec}")
        return torch.full((n_heads,), spec, dtype=torch.long)
    spec = torch.as_tensor(spec, dtype=torch.long)
    if spec.shape != (n_heads,):
        raise ValueError(f"head-type tensor must have shape ({n_heads},), got {tuple(spec.shape)}")
    if (spec < 0).any() or (spec > 3).any():
        raise ValueError("head-type values must be in [0, 3]")
    return spec


class EulerCEMinimal(nn.Module):
    """2-bit minimal Euler-CE attention.

    Each head commits to one of four operational types via a 2-bit
    spec (pi_bit, e_bit). The continuous parameters `xi_h` (decay
    length, per-head) and the rotary base are the only learnable
    positional state — head-type itself is an axiomatic design choice,
    not learned.

    Args:
        d_model, n_heads, block: standard.
        head_types: spec for per-head types. See `head_types_from_spec`.
            Default "alibi" (all decay-only, the strongest single tier-1
            choice from the length-extrap ablation).
        rope_base: base for the RoPE-style geometric frequencies used by
            heads with pi_bit = 1. Defaults to 10000 (RoFormer).
        xi_init: initial decay length for heads with e_bit = 1.
            Defaults to block/8 (the EulerCE original).
        learnable_xi: if False, freeze xi at its init value.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        head_types: object = "alibi",
        rope_base: float = 10000.0,
        xi_init: Optional[float] = None,
        learnable_xi: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even (got {self.d_head})")
        self.block = block
        self.rope_base = float(rope_base)

        # 2-bit head-type assignment (axiom — buffer, not learned).
        # High bit = rotation (π), low bit = decay (e). Extracted with
        # bitwise shift/mask — semantically the 2-bit field the name
        # implies, and cheaper than the old `// 2` / `% 2` form.
        types = head_types_from_spec(head_types, n_heads)            # (h,)
        pi_bits = ((types >> 1) & 1).float()
        e_bits = (types & 1).float()
        self.register_buffer("head_types", types)
        self.register_buffer("pi_bits", pi_bits)
        self.register_buffer("e_bits", e_bits)
        # Fast-path detector: when all heads share the same type we can
        # bypass the per-head gating and dispatch to PyTorch SDPA
        # (FlashAttention / Memory-Efficient backend).
        self._uniform_type = int(types[0].item()) if (types == types[0]).all() else -1

        # Packed bitmask form of the head-type assignment. For n_heads
        # ≤ 64 this collapses the (H,) float buffers (pi_bits, e_bits =
        # 8·H bytes) to two Python ints (≤ 16 bytes total). Python's
        # arbitrary-precision int transparently handles larger widths.
        pi_mask_int, e_mask_int = 0, 0
        for h, t in enumerate(types.tolist()):
            pi_mask_int |= ((t >> 1) & 1) << h
            e_mask_int |= (t & 1) << h
        self._pi_mask = pi_mask_int
        self._e_mask = e_mask_int

        # Pre-bucket heads by 2-bit type so the mixed path dispatches
        # SDPA once per present bucket. This eliminates:
        #   * `cos`/`sin` materialization on rotation-off heads,
        #   * `decay_bias` materialization on decay-off heads,
        #   * the `(b, H, n, n)` scores tensor (SDPA tiles internally),
        #   * explicit `softmax` (FlashAttention / mem-efficient path).
        present: list[int] = []
        bucket_heads: list[torch.Tensor] = []
        for t in range(4):
            idx = (types == t).nonzero(as_tuple=True)[0].long().contiguous()
            if idx.numel() > 0:
                present.append(t)
                self.register_buffer(f"_bucket_{t}_idx", idx, persistent=False)
                bucket_heads.append(idx)
        self._present_buckets: tuple[int, ...] = tuple(present)
        concat_idx = (
            torch.cat(bucket_heads, dim=0)
            if bucket_heads
            else torch.arange(n_heads, dtype=torch.long)
        )
        inv_perm = torch.empty(n_heads, dtype=torch.long)
        inv_perm[concat_idx] = torch.arange(n_heads, dtype=torch.long)
        self.register_buffer("_bucket_inv_perm", inv_perm.contiguous(), persistent=False)
        self._bucket_is_identity: bool = bool(
            (concat_idx == torch.arange(n_heads, dtype=torch.long)).all().item()
        )

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Position / distance buffers.
        self.register_buffer(
            "tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))
        self.register_buffer(
            "pos", torch.arange(block, dtype=torch.float32))
        d_mat = (torch.arange(block).unsqueeze(1)
                 - torch.arange(block).unsqueeze(0)).abs().float()
        self.register_buffer("d_mat", d_mat)

        # RoPE-style frequencies for heads with rotation bit on.
        k = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        self.register_buffer("inv_freq", self.rope_base ** (-k))   # (d_head/2,)

        # Per-head decay length.
        if xi_init is None:
            xi_init = block / 8.0
        if xi_init <= 0.0:
            raise ValueError("xi_init must be positive")
        log_xi = torch.full((n_heads,), math.log(xi_init), dtype=torch.float32)
        if learnable_xi:
            self.log_xi = nn.Parameter(log_xi)
        else:
            self.register_buffer("log_xi", log_xi)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional / distance buffers for length-extrap eval.
        Learnable parameters (qkv, o, log_xi) are unchanged."""
        cur = self.pos.shape[0]
        if new_block <= cur:
            return
        dev = self.pos.device
        self.pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        self.d_mat = (torch.arange(new_block).unsqueeze(1)
                      - torch.arange(new_block).unsqueeze(0)).abs().float().to(dev)

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out

    # ------------------------------------------------------------------
    # Forward dispatch:
    #   uniform head-type → SDPA fast path (Flash / Memory-Efficient)
    #   mixed head-type   → per-head-gated reference path (slower but generic)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        H = self.n_heads
        qkv = self.qkv(x).view(b, n, 3, H, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)             # (b, H, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self._uniform_type >= 0:
            out = self._forward_uniform(q, k, v, n, H)
        else:
            out = self._forward_mixed(q, k, v, n, H)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o(out)

    # ------------------------------------------------------------------
    def _forward_uniform(self, q, k, v, n, H):
        """Fast path when all heads share the same 2-bit type.

        Dispatches to torch.nn.functional.scaled_dot_product_attention,
        which selects FlashAttention (no attn_mask) or Memory-Efficient
        Attention (with attn_mask) automatically.
        """
        ht = self._uniform_type
        rotate = (ht & 0b10) != 0
        decay = (ht & 0b01) != 0

        if rotate:
            theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
            cos = theta.cos()
            sin = theta.sin()
            q = self._rotate(q, cos, sin)
            k = self._rotate(k, cos, sin)

        if not decay:
            return _causal_softmax_sdpa(q, k, v)

        # Decay path: Q-tiled to keep attn_mask peak at O(H·Q_CHUNK·N).
        xi = torch.exp(self.log_xi)                                  # (H,)
        return _chunked_decay_sdpa(q, k, v, xi, self.pos, n)

    # ------------------------------------------------------------------
    def _forward_mixed(self, q, k, v, n, H):
        """Bucketed zero-waste path for mixed head-types.

        Heads are grouped by 2-bit type at init. Each present bucket
        dispatches to `scaled_dot_product_attention` once with exactly
        the PE it needs (rotation for `pi_bit=1`, additive distance
        mask for `e_bit=1`). Compared to the prior scalar-gated path
        this drops `cos`/`sin` on rotation-off heads, the `(1, H, n, n)`
        decay bias on decay-off heads, the full `(b, H, n, n)` scores
        tensor, and the explicit softmax — SDPA tiles internally and
        picks FlashAttention when the mask allows it. Output is placed
        directly into the final head slot via `index_copy_`, avoiding
        a `cat` + permutation round-trip.
        """
        b = q.shape[0]
        d_head = q.shape[-1]
        out = torch.empty(b, H, n, d_head, dtype=q.dtype, device=q.device)

        theta_cos: Optional[torch.Tensor] = None
        theta_sin: Optional[torch.Tensor] = None
        causal_view: Optional[torch.Tensor] = None

        for t in self._present_buckets:
            idx = getattr(self, f"_bucket_{t}_idx")
            h_t = idx.numel()
            q_t = q.index_select(1, idx)
            k_t = k.index_select(1, idx)
            v_t = v.index_select(1, idx)

            if (t >> 1) & 1:  # rotation bit
                if theta_cos is None:
                    theta = self.pos[:n].view(1, 1, n, 1) * self.inv_freq.view(1, 1, 1, -1)
                    theta_cos = theta.cos()
                    theta_sin = theta.sin()
                q_t = self._rotate(q_t, theta_cos, theta_sin)
                k_t = self._rotate(k_t, theta_cos, theta_sin)

            if t & 1:  # decay bit
                xi = torch.exp(self.log_xi.index_select(0, idx))        # (h_t,)
                out_t = _chunked_decay_sdpa(q_t, k_t, v_t, xi, self.pos, n)
            else:
                out_t = _causal_softmax_sdpa(q_t, k_t, v_t)

            out.index_copy_(1, idx, out_t)

        return out



class EulerCEMinimalBlock(nn.Module):
    """Pre-LN block wrapping `EulerCEMinimal` + standard 4× FFN."""

    def __init__(self, d_model: int, n_heads: int, block: int,
                 head_types: object = "alibi",
                 rope_base: float = 10000.0,
                 xi_init: Optional[float] = None,
                 learnable_xi: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EulerCEMinimal(
            d_model, n_heads, block,
            head_types=head_types,
            rope_base=rope_base,
            xi_init=xi_init,
            learnable_xi=learnable_xi,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_ffn.py`

```python
"""CE-theory FFN variants.

Baseline:      Linear(d, 4d) -> GELU -> Linear(4d, d)
SwiGLU:        Linear(d, 4d) + Linear(d, 4d) gated by SiLU (known-good)
EulerDecayFFN: GELU * e-decay gate (e role from CE theory)
EulerPhaseFFN: GELU * sin(π · h) periodic modulation (π role)
EulerFullFFN:  combines π-phase and e-decay simultaneously

Each module exposes the same forward signature (x -> x) and can
replace the standard FFN in any transformer block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdFFN(nn.Module):
    """Baseline: Linear -> GELU -> Linear."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU_FFN(nn.Module):
    """SwiGLU — Llama / PaLM standard. Known to beat GELU-FFN."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        # keep param count comparable to StdFFN with mult=4:
        # SwiGLU has 2 up-projections so we use mult' = mult*2/3 rounded
        hidden = int(mult * 2 / 3 * d)
        # round to multiple of 8
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class EulerDecayFFN(nn.Module):
    """GELU modulated by e-decay gate: survival S = e^{-|h|/xi}.

    h   = W_up x
    s   = exp(-|h| / xi)       # e-decay, learnable log_xi
    a   = GELU(h) * s
    y   = W_down a

    Rationale: CE's survival function S(D) = e^{-D} used as an adaptive
    saturation. Large activations get softly clipped by e-decay rather
    than growing without bound (as in GELU).
    """

    def __init__(self, d: int, mult: int = 4, xi_init: float = 3.0):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi_init)))

    def forward(self, x):
        h = self.up(x)
        xi = torch.exp(self.log_xi)
        survival = torch.exp(-h.abs() / xi)
        return self.down(F.gelu(h) * survival)


class EulerPhaseFFN(nn.Module):
    """GELU with π-periodic modulation: a = GELU(h) * (1 + eta · cos(π h / tau)).

    Provides a periodic (phase) dimension to the pointwise nonlinearity
    without spoiling the GELU monotonic region near 0.
    """

    def __init__(self, d: int, mult: int = 4, tau_init: float = 2.0,
                 eta_init: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.eta = nn.Parameter(torch.tensor(eta_init))

    def forward(self, x):
        h = self.up(x)
        tau = torch.exp(self.log_tau)
        phase = torch.cos(math.pi * h / tau)
        return self.down(F.gelu(h) * (1.0 + self.eta * phase))


class EulerFullFFN(nn.Module):
    """Full Euler FFN: π-phase × e-decay on top of GELU.

    a = GELU(h) * (1 + eta · cos(π h / tau)) * exp(-|h|/xi)
    """

    def __init__(self, d: int, mult: int = 4,
                 xi_init: float = 3.0,
                 tau_init: float = 2.0,
                 eta_init: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.log_xi = nn.Parameter(torch.tensor(math.log(xi_init)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.eta = nn.Parameter(torch.tensor(eta_init))

    def forward(self, x):
        h = self.up(x)
        xi = torch.exp(self.log_xi)
        tau = torch.exp(self.log_tau)
        survival = torch.exp(-h.abs() / xi)
        phase = torch.cos(math.pi * h / tau)
        return self.down(F.gelu(h) * (1.0 + self.eta * phase) * survival)


def make_ffn(kind: str, d: int, mult: int = 4) -> nn.Module:
    if kind == "std":
        return StdFFN(d, mult)
    if kind == "swiglu":
        return SwiGLU_FFN(d, mult)
    if kind == "euler_decay":
        return EulerDecayFFN(d, mult)
    if kind == "euler_phase":
        return EulerPhaseFFN(d, mult)
    if kind == "euler_full":
        return EulerFullFFN(d, mult)
    if kind == "zeta":
        from .ce_zeta import ZetaFFN
        return ZetaFFN(d, mult)
    raise ValueError(f"unknown ffn kind: {kind!r}")


__all__ = [
    "StdFFN", "SwiGLU_FFN", "EulerDecayFFN", "EulerPhaseFFN", "EulerFullFFN",
    "make_ffn",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_laplacian.py`

```python
"""Dual-graph attention via row-stochastic Laplacian kernels.

Restructured from the earlier residual-Laplacian form (which had
asymmetry bugs and a dead-gradient alpha=0 path). The block is now an
attention head whose kernel is built from explicit graph adjacency:

    A_lang_ij  = cosine(P_lang h_i, P_lang h_j)_+    (symmetric)
    A_grav_ij  = exp(-||P_grav h_i - P_grav h_j||^2 / 2sigma^2)

Both graphs are SYMMETRIC (no mask-induced asymmetry in A), so the
normalized Laplacian has eigenvalues in [0, 2] as expected. Causal
constraints are enforced ONLY on the row-normalized transition
matrices P_lang_rw, P_grav_rw (D^-1 A with upper-tri zeroed and rows
renormalized) — this is mathematically equivalent to restricting the
random walk to past neighbors.

Output per head:
    y_i = sum_j [omega_lang * P_lang_rw + omega_grav * P_grav_rw]_ij
            * V(h)_j

which is a convex mixture of two row-stochastic kernels acting on V,
i.e. a concrete instantiation of the compendium 6.B.1 attention
kernel family with interpretable graph-theoretic weights.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import T_WAKE


def _cosine_adjacency(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """A_ij = max(0, cos(z_i, z_j)), diagonal zeroed. Symmetric."""
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    zn = z / norm
    A = torch.matmul(zn, zn.transpose(-1, -2)).clamp_min(0.0)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A * (1.0 - eye)


def _rbf_adjacency(z: torch.Tensor, sigma) -> torch.Tensor:
    """A_ij = exp(-||z_i - z_j||^2 / 2 sigma^2), diagonal zeroed. Symmetric.

    ``sigma`` may be a float or a torch.Tensor scalar (for learnable sigma).
    """
    sq = (z * z).sum(dim=-1, keepdim=True)
    d2 = (sq + sq.transpose(-1, -2) - 2.0 * torch.matmul(z, z.transpose(-1, -2))).clamp_min(0.0)
    if isinstance(sigma, torch.Tensor):
        A = torch.exp(-d2 / (2.0 * sigma * sigma))
    else:
        A = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A * (1.0 - eye)


def _row_stochastic_causal(A: torch.Tensor, causal_mask: Optional[torch.Tensor],
                           eps: float = 1e-8) -> torch.Tensor:
    """Convert symmetric adjacency into a causal row-stochastic transition.

    Order of ops matters:
      1. A is symmetric here (both i->j and j->i).
      2. Apply causal mask: (i,j) with j > i dropped.
      3. Re-normalize rows so each row sums to 1 (random-walk kernel).
    """
    if causal_mask is not None:
        A = A * causal_mask.to(A.dtype)
    deg = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / deg


def _sym_normalized_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L = I - D^{-1/2} A D^{-1/2} for SYMMETRIC A. Eigenvalues in [0, 2]."""
    deg = A.sum(dim=-1).clamp_min(eps)
    inv_sqrt = deg.pow(-0.5)
    A_norm = A * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return eye - A_norm


class DualLaplacianBlock(nn.Module):
    """Dual-graph attention head with optional learnable gate and sigma.

    Produces attention output as a convex mix of two row-stochastic
    causal random-walk kernels (cosine on P_lang h, RBF on P_grav h).

    Args:
        learnable_gate: if True, omega_lang is a free parameter
            (initialized from the Borbely mode), passed through sigmoid
            so the mix stays convex. Otherwise frozen at mode_gate(mode).
        learnable_sigma: if True, sigma_grav is trainable in log-space
            (init from the provided scalar).
    """

    def __init__(
        self,
        d_model: int,
        d_lang: Optional[int] = None,
        d_grav: Optional[int] = None,
        sigma_grav: float = 1.0,
        mode: str = "wake",
        learnable_gate: bool = False,
        learnable_sigma: bool = False,
    ) -> None:
        super().__init__()
        d_lang = d_lang or d_model
        d_grav = d_grav or d_model
        self.P_lang = nn.Linear(d_model, d_lang, bias=False)
        self.P_grav = nn.Linear(d_model, d_grav, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)
        self.mode = mode
        self.learnable_gate = learnable_gate
        self.learnable_sigma = learnable_sigma

        # initial gate from Borbely constant for the declared mode
        if mode == "wake":
            init_w_lang = 1.0 - T_WAKE
        elif mode == "nrem":
            init_w_lang = T_WAKE
        else:
            init_w_lang = 0.5

        # store logit so sigmoid(logit) == init_w_lang
        p = max(min(init_w_lang, 0.9999), 0.0001)
        logit = math.log(p / (1.0 - p))
        if learnable_gate:
            self.gate_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        else:
            self.register_buffer("gate_logit", torch.tensor(logit, dtype=torch.float32))

        log_sigma = math.log(max(sigma_grav, 1e-4))
        if learnable_sigma:
            self.log_sigma_grav = nn.Parameter(torch.tensor(log_sigma, dtype=torch.float32))
        else:
            self.register_buffer("log_sigma_grav",
                                 torch.tensor(log_sigma, dtype=torch.float32))

    def current_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_l = torch.sigmoid(self.gate_logit)
        return w_l, 1.0 - w_l

    def current_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_grav)

    def forward(self, h: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_l = self.P_lang(h)
        z_g = self.P_grav(h)
        v = self.V(h)

        A_l = _cosine_adjacency(z_l)
        A_g = _rbf_adjacency(z_g, sigma=self.current_sigma())

        K_l = _row_stochastic_causal(A_l, causal_mask)
        K_g = _row_stochastic_causal(A_g, causal_mask)

        w_l, w_g = self.current_gate()
        K = w_l * K_l + w_g * K_g  # still row-stochastic
        return self.O(torch.matmul(K, v))


def graph_spectrum(
    adjacency_fn,
    h: torch.Tensor,
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Compute eigenvalues of the symmetric normalized Laplacian of a
    graph whose adjacency is ``adjacency_fn(h)``. Returns sorted real
    eigenvalues (only valid when adjacency is symmetric)."""
    A = adjacency_fn(h)
    if not symmetric:
        A = 0.5 * (A + A.transpose(-1, -2))
    L = _sym_normalized_laplacian(A)
    L = 0.5 * (L + L.transpose(-1, -2))  # numeric cleanup
    return torch.linalg.eigvalsh(L).sort().values


__all__ = [
    "DualLaplacianBlock",
    "graph_spectrum",
    "_cosine_adjacency",
    "_rbf_adjacency",
    "_sym_normalized_laplacian",
    "_row_stochastic_causal",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/ce_mra.py`

```python
"""Mellin-Riemann Attention Block (MRA) — refactored with ablation modes.

Spec: docs/8_리만/mra_block_spec.md

Design surface area (all optional, default == lean MRA):

    MellinRiemannAttention(
        freq_mode       = "rope"  | "zeta_log",     # position frequencies
        amp_weight      = True    | False,          # ζ amplitude weighting w_k
        decay_mode      = "none"  | "bias" | "mult",# critical-line decay form
        sparse_eps2     = 0.0     | 0.0487          # bootstrap top-k retention
        hermitian       = False   | True            # bidirectional only
        spectral_norm_o = False   | True            # σ₁(W_o) ≤ 1
    )

The only component that is **genuinely novel** in MRA is the ζ amplitude
weighting `w_k = 1/(1/2 + iγ_k)`. The other knobs exist for ablation and
to reach parity with the preceding `riemann_rope` formulation for
regression testing.

Notes on the earlier (now opt-in) components:

* `freq_mode="zeta_log"` uses `θ(p,k) = γ_k log(1+p)` (Mellin-kernel).
  γ_k/γ_1 spans only ~5×, which collapses RoPE's multi-scale resolution
  (10000^{-2k/d} spans ~3000×). Kept for ablation; not recommended.
* `decay_mode="mult"` multiplies scores by √((1+j)/(1+i)). This is a
  per-key temperature, not an additive log-bias; usually under-performs
  `"bias"` which applies ½(log(1+j)-log(1+i)) as an ALiBi-style add.
* `hermitian=True` symmetrises scores before the causal mask, which
  leaks future information into past scores on a causal LM. For
  bidirectional encoders it realises Hilbert–Pólya self-adjointness
  directly; for causal LM it is **unsafe**.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ce_riemann_attn import riemann_zeros


# Bootstrap fixed point from docs/6_뇌/sleep.md.
BOOTSTRAP_EPS2 = 0.0487


# ---------------------------------------------------------------------------
# Bootstrap sparsity
# ---------------------------------------------------------------------------


def bootstrap_sparse(
    attn: torch.Tensor,
    eps2: float = BOOTSTRAP_EPS2,
    min_keep: int = 1,
) -> torch.Tensor:
    """Top-k retention per row to enforce activation ratio ≈ ε² · N."""
    n = attn.shape[-1]
    k = max(min_keep, math.ceil(eps2 * n))
    if k >= n:
        return attn
    _, idx = attn.topk(k, dim=-1)
    mask = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
    masked = attn * mask
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)


# ---------------------------------------------------------------------------
# Mellin-Riemann Attention
# ---------------------------------------------------------------------------


class MellinRiemannAttention(nn.Module):
    """Self-attention with ζ-amplitude weighting.

    The score is computed as

        Re(S_{ij}) = Σ_k Re(w_k · q̃_i^{(k)} · conj(k̃_j^{(k)}))

    where ``w_k = 1/(1/2 + iγ_k)`` is the ζ explicit-formula amplitude
    (buffer, axiomatic from the Riemann hypothesis), and ``q̃, k̃`` are
    RoPE-rotated in complex form. See module docstring for ablation
    knobs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        freq_mode: str = "rope",
        amp_weight: bool = True,
        decay_mode: str = "none",
        sparse_eps2: float = 0.0,
        hermitian: bool = False,
        spectral_norm_o: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if freq_mode not in ("rope", "zeta_log"):
            raise ValueError(f"freq_mode must be 'rope' or 'zeta_log', got {freq_mode!r}")
        if decay_mode not in ("none", "bias", "mult"):
            raise ValueError(
                f"decay_mode must be 'none', 'bias', or 'mult', got {decay_mode!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {self.d_head}")
        self.K = self.d_head // 2
        self.block = block
        self.freq_mode = freq_mode
        self.amp_weight = amp_weight
        self.decay_mode = decay_mode
        self.hermitian = hermitian
        self.sparse_eps2 = float(sparse_eps2)

        # --- Frequencies ------------------------------------------------
        gamma = riemann_zeros(self.K)              # raw γ_k (buffer)
        gamma = gamma / gamma[0]                   # normalise: γ_0 -> 1
        self.register_buffer("gamma", gamma)

        if amp_weight:
            # w_k = 1/(1/2 + iγ_k) = (1/2 - iγ_k)/(1/4 + γ_k²)
            denom = 0.25 + gamma * gamma
            self.register_buffer("w_re", 0.5 / denom)
            self.register_buffer("w_im", -gamma / denom)
        else:
            self.register_buffer("w_re", torch.ones(self.K))
            self.register_buffer("w_im", torch.zeros(self.K))

        # --- Position phases --------------------------------------------
        pos = torch.arange(block, dtype=torch.float32)
        if freq_mode == "rope":
            # RoPE geometric frequencies (~3000× span) for multi-scale
            # resolution. γ_k is used only for amplitude, not frequency.
            k_idx = torch.arange(self.K, dtype=torch.float32)
            inv_freq = rope_base ** (-k_idx / self.K)     # (K,)
            phase = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        else:  # "zeta_log"
            # Mellin-kernel phase θ(p,k) = γ_k log(1+p). Narrow (~5×) span.
            tau = torch.log1p(pos)
            phase = tau.unsqueeze(1) * gamma.unsqueeze(0)
        self.register_buffer("cos_p", phase.cos())    # (N, K)
        self.register_buffer("sin_p", phase.sin())

        # --- Decay factors ---------------------------------------------
        if decay_mode in ("bias", "mult"):
            log1p = torch.log1p(pos)
            # 0.5 * (log(1+j) - log(1+i))  is a pre-built (N, N) matrix.
            decay_mat = 0.5 * (log1p.view(1, -1) - log1p.view(-1, 1))
            self.register_buffer("log_decay", decay_mat)   # (N, N)
        else:
            self.register_buffer("log_decay", torch.zeros(block, block))

        # --- Causal mask -------------------------------------------------
        self.register_buffer(
            "tril", torch.tril(torch.ones(block, block, dtype=torch.bool)))

        # --- Projections -------------------------------------------------
        if hermitian:
            self.qk = nn.Linear(d_model, d_model, bias=False)
            self.q = self.qk
            self.k = self.qk
        else:
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        o_lin = nn.Linear(d_model, d_model, bias=False)
        if spectral_norm_o:
            self.o = nn.utils.parametrizations.spectral_norm(
                o_lin, n_power_iterations=5)
        else:
            self.o = o_lin

        # Cache constructor args for extend_to (recompute deterministic buffers).
        self._rope_base = float(rope_base)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extend_to(self, new_block: int) -> None:
        """Grow positional buffers for length-extrapolation eval.
        γ_k, w_k stay at training values; cos_p / sin_p / log_decay / tril
        are recomputed for `new_block` using the same rule as __init__."""
        cur = self.cos_p.shape[0]
        if new_block <= cur:
            return
        dev = self.cos_p.device
        pos = torch.arange(new_block, dtype=torch.float32, device=dev)
        if self.freq_mode == "rope":
            k_idx = torch.arange(self.K, dtype=torch.float32, device=dev)
            inv_freq = self._rope_base ** (-k_idx / self.K)
            phase = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        else:
            tau = torch.log1p(pos)
            phase = tau.unsqueeze(1) * self.gamma.to(dev).unsqueeze(0)
        self.cos_p = phase.cos()
        self.sin_p = phase.sin()
        self.tril = torch.tril(
            torch.ones(new_block, new_block, dtype=torch.bool, device=dev))
        if self.decay_mode in ("bias", "mult"):
            log1p = torch.log1p(pos)
            self.log_decay = 0.5 * (log1p.view(1, -1) - log1p.view(-1, 1))
        else:
            self.log_decay = torch.zeros(new_block, new_block, device=dev)
        self.block = new_block

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh, K = self.n_heads, self.d_head, self.K

        q = self.q(x).view(B, N, H, dh)
        k = self.k(x).view(B, N, H, dh)
        v = self.v(x).view(B, N, H, dh)

        q_re = q[..., 0::2]
        q_im = q[..., 1::2]
        k_re = k[..., 0::2]
        k_im = k[..., 1::2]

        cos_b = self.cos_p[:N].view(1, N, 1, K)
        sin_b = self.sin_p[:N].view(1, N, 1, K)

        # RoPE-style rotation: (cos - i sin)(q_re + i q_im)
        qt_re = cos_b * q_re + sin_b * q_im
        qt_im = cos_b * q_im - sin_b * q_re
        kt_re = cos_b * k_re + sin_b * k_im
        kt_im = cos_b * k_im - sin_b * k_re

        # Apply ζ amplitude w_k (or identity if amp_weight=False).
        w_re = self.w_re.view(1, 1, 1, K)
        w_im = self.w_im.view(1, 1, 1, K)
        qhat_re = w_re * qt_re - w_im * qt_im
        qhat_im = w_re * qt_im + w_im * qt_re

        # Two real matmuls → Re(Σ_k w_k q̃_i · conj(k̃_j)).
        qhat_re = qhat_re.transpose(1, 2)
        qhat_im = qhat_im.transpose(1, 2)
        kt_re_t = kt_re.transpose(1, 2)
        kt_im_t = kt_im.transpose(1, 2)

        scores = (
            qhat_re @ kt_re_t.transpose(-1, -2)
            + qhat_im @ kt_im_t.transpose(-1, -2)
        )
        scores = scores / math.sqrt(dh)

        # Critical-line decay.
        if self.decay_mode == "bias":
            scores = scores + self.log_decay[:N, :N].view(1, 1, N, N)
        elif self.decay_mode == "mult":
            scores = scores * torch.exp(self.log_decay[:N, :N]).view(1, 1, N, N)

        # Self-adjoint projection (bidirectional only — will contaminate
        # causal scores with future-direction info).
        if self.hermitian:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        scores = scores.masked_fill(~self.tril[:N, :N], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.sparse_eps2 > 0.0:
            attn = bootstrap_sparse(attn, self.sparse_eps2)

        v = v.transpose(1, 2)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.o(out)


# ---------------------------------------------------------------------------
# MRABlock
# ---------------------------------------------------------------------------


class MRABlock(nn.Module):
    """Pre-LN block wrapping `MellinRiemannAttention` + pluggable FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block: int,
        freq_mode: str = "rope",
        amp_weight: bool = True,
        decay_mode: str = "none",
        sparse_eps2: float = 0.0,
        hermitian: bool = False,
        spectral_norm_o: bool = False,
        ffn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MellinRiemannAttention(
            d_model, n_heads, block,
            freq_mode=freq_mode,
            amp_weight=amp_weight,
            decay_mode=decay_mode,
            sparse_eps2=sparse_eps2,
            hermitian=hermitian,
            spectral_norm_o=spectral_norm_o,
        )
        if ffn is None:
            ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


__all__ = [
    "BOOTSTRAP_EPS2",
    "bootstrap_sparse",
    "MellinRiemannAttention",
    "MRABlock",
]
```
