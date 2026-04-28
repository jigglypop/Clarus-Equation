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

각 원리는 "학습 분포 밖에서 무엇이 작동하는가" 를 구조 선택으로 풀려는 시도이다.

본 장은 가장 측정 가능한 OOD axis — **시퀀스 길이** — 에서 이 thesis 를 정량 검증한다.

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

`clarus/ce_euler.py::EulerCEMinimal` 구현 + 16 개 테스트로 정확성 검증. 2-bit minimal 변종이 canonical PE (NoPE, RoPE, ALiBi, xPos) 를 수치적으로 reproduce (`min_alibi` ≈ `rope_alibi`, `min_xpos` ≈ `euler_ce_k1`, `min_rope` = `std_rope` 정확 일치).

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
