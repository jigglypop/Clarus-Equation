# 19. OOD Generalization 의 구조적 결정성: Length Extrapolation 사례

> 관련: `1_AGI.md`(5대 원리), `2_Architecture.md`(구조 규약), `13_Verification.md`(검증 방법론), `8_리만/mra_paper.md`(역사적 비교)
>
> 과거 9변종×3seed 표의 JSON과 생성 script는 현재 checkout에 없다. 따라서
> 그 표와 `bridge` 승격은 철회한다. 현재 재현 가능한 근거는
> `../experiments/ood_length_repro.py`의 4 head-type tiny-model toy뿐이며,
> length 단일 축의 `hypothesis/toy evidence`로 제한한다.

---

## 0. 한 줄 결론

> 현재 tiny char-LM toy에서는 32× 길이에서 ALiBi와 xPos가 안정적이고,
> NoPE와 RoPE가 악화됐다. 이는 **명시적 강한 distance attenuation이 이 toy의
> length-OOD에 도움을 준다**는 방향성 근거이지, 구조가 OOD 일반화를 결정한다는
> 보편 정리나 scaling-only 접근의 반증이 아니다.

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

본 장은 통제가 쉬운 OOD axis인 **시퀀스 길이**에서 이 thesis의 한 toy
반증 가능성을 시험한다.

여기서 자기참조재귀는 다른 원리들을 장기 작업으로 묶는 후보 닫힘 조건이다.
현재 length toy는 positional attenuation만 비교하며 agentic OOD를 시험하지
않는다. 장문 추론, tool-use, multi-turn planning에서
\(S_t\to R(S_t)\to C_t\to S_{t+1}\) 재귀가 주는 추가 효과는 open-loop
baseline과 별도로 측정해야 한다.

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

평균 로그 수축률의 음수성은 추가 정상성·적분가능성·불변영역 가정 아래의
후보 안정성 지표다. 단일 token PPL보다 agent drift를 직접 보지만, length
extrapolation 결과만으로 attenuation이 전체 agent state의 $\hat\rho_t$를
낮춘다고 결론내릴 수는 없다.

### 1.1 왜 length 인가

OOD axis 들은 측정 난이도에 큰 차이가 있다:

| axis | 통제 가능성 | 측정 가능성 |
|---|---|---|
| **length** | ★★★ (정확한 N 비율) | ★★★ (PPL @ N) |
| in-context demos | ★★ (분포 조작) | ★★ (task accuracy) |
| 의미 (semantic) | ★ (코퍼스 분리 어려움) | ★ (질적 평가) |
| 모달 (modal) | 비교 불가 | -- |
| task transfer | ★ (zero-shot 정의 모호) | ★★ (benchmark) |

Length는 train $N=64$, eval $N=2048$처럼 정확한 배율을 만들 수 있고 PPL로
측정하기 쉽다. 다만 corpus, absolute position distribution, optimization,
evaluation batch가 함께 변할 수 있으므로 OOD 효과가 자동으로 분리되는 것은 아니다.

따라서 length 에서 발견되는 구조적 원리는 OOD generalization 일반의 first-order 후보로 다룰 가치가 있다.

---

## 2. 재현 가능한 tiny-model toy와 증거 장부

### 2.1 현재 실행 가능한 protocol

`../experiments/ood_length_repro.py`는 같은 tiny char-level LM에서
`nope`, `alibi`, `rope`, `xpos` 네 head type을 학습한다. 현재 결과 장부의
설정은 약 127 K parameter, train block 64, repository Python corpus,
500 steps, seed 0--2, eval length 64/256/1024/2048이다. degradation은 각
seed의 eval@64 PPL 대비 상대 변화다.

| head | 32× degradation mean ± sample std | 현재 toy 판정 |
|---|---:|---|
| ALiBi | $-9.1\%\pm5.7\%$ | 안정 방향 |
| xPos | $-8.8\%\pm5.4\%$ | 안정 방향 |
| NoPE | $+31.7\%\pm10.2\%$ | 악화; 과거 Tier-1 주장 반증 |
| RoPE | $+505\%\pm41.4\%$ | 큰 악화 |

이 표는 동일 script로 재생성할 수 있는 toy evidence다. NoPE와 RoPE의
degradation 차이는 $505-31.7=473.3$ percentage points다. 비율을 쓰면 약
$15.9$배지만, 분모가 0에 가까울 수 있는 degradation ratio는 불안정하므로
"몇 배 우수"보다 percentage-point 차이와 seed 분산을 보고한다.

### 2.2 철회된 증거 장부

| 과거 주장 | 필요한 artifact | 현재 상태 |
|---|---|---|
| 9 변종 × 3 seed × 1500 step | `euler_extrap_long.json`, `extrap_full.json`과 생성 log | missing; 수치표 non-evidence |
| MRA/MRA-bias의 32× tier | `bench_mra_extrap.py`, checkpoint/config/corpus hash | missing; 미판정 |
| 9변종 구조가 OOD를 결정 | 다중 seed·scale·corpus·task 결과와 CI | 미검증 |

`EulerCEMinimal` 구현 및 unit test는 head 연산의 코드 정확성을 검사할 수
있지만, OOD 성능 결과를 대신하지 않는다.

### 2.3 주장 승격 acceptance protocol

전체 9변종 결론을 복원하려면 동일 tokenizer/corpus split SHA256, parameter
count $\pm0.1\%$, train tokens, optimizer schedule을 고정한다. 최소 10 seed와
두 개 이상의 corpus에서 length $\{64,128,256,512,1024,2048\}$를 평가하고,
각 run의 config/commit/environment hash, PPL, raw log를 JSONL로 보존한다.
사전등록한 mixed-effects 또는 paired bootstrap 분석에서 attenuation class의
효과 CI가 0을 배제하고 unseen corpus에서도 재현될 때만 `bridge`로 올린다.
현재 3-seed single-corpus tiny result는 그 문턱을 통과하지 않는다.

---

## 3. 가능한 메커니즘 — 아직 가설

### 3.1 RoPE 회전이 OOD 의 직접 원인

현재 toy에서 NoPE도 $+31.7\%$ 악화됐고 RoPE는 $+505\%$ 악화됐다.
따라서 "회전 부재면 외삽한다"는 과거 주장은 반증됐다. RoPE의 phase
`θ_k(p)=p·10000^{-2k/d}`가 train 범위 밖에서 보지 못한 상태를 만든다는 것은
위험 메커니즘 후보지만, 두 architecture의 차이가 rotation 하나뿐이라는
인과 개입은 아니므로 회전 자체가 원인이라고 확정하지 않는다.

특히 RoPE 의 wrap-around 특성 (`cos(2π+θ) = cos θ`) 때문에 멀리 떨어진 토큰 간 attention 이 가까운 토큰 간 attention 과 *같은 phase* 로 wrap 되어 관계가 깨진다.

### 3.2 Distance attenuation 가설

현재 toy에서 안정 방향이었던 두 변형은 명시적 강한 distance signal을 갖는다.

| 형태 | 거리 의존 | 외삽 안정 |
|---|---|---|
| Linear additive `−m d` | strong, monotonic | ALiBi $-9.1\%\pm5.7\%$ |
| Multiplicative `ζ^d`, $0<ζ<1$ | log-space linear attenuation | xPos $-8.8\%\pm5.4\%$ |
| Log additive `−c\log d` | sub-linear | 현재 artifact 없음 |

강한 attenuation을 additive logit $b(d)$에 대해
$\liminf_{d\to\infty}-b(d)/d>0$으로 정의하면 linear bias와
multiplicative $ζ^d$는 포함되고 $-c\log d$는 제외된다. 이 기준은 2-bit
taxonomy의 모순을 피하는 **사전등록 분류 규칙**이지, 아직 충분조건 정리가
아니다. MRA의 amplitude가 이 기준을 만족하는지도 구현식으로 따로 판정해야 한다.

### 3.3 1-jet 충분성 (linear 가 가장 강한 이유)

모든 smooth 함수의 1-jet은 $d_0$ **근방**에서만 근사다. 이 국소 Taylor
사실은 $64\to2048$ 외삽을 정당화하지 않는다. Linear bias가 전 구간에서
같은 slope를 유지하고 multiplicative $ζ^d$가 log-space에서 linear라는 것은
후보 설명이지만, softmax normalization과 학습된 Q/K를 포함한 일반화 정리는
아니다. 따라서 이 절은 형식 증명이 아니라 후속 분석 가설이다.

---

## 4. Clarus 본 thesis 와의 정합

### 4.1 "구조에서 의미가 나온다"

Clarus AGI 의 핵심 명제 (`1_AGI.md` § 0): 우주는 빅뱅에서 한 번의 부트스트랩으로 고정점에 도달했고, 부트스트랩은 **구조적 자기조직화** 의 결과이다. 즉 의미와 구조는 분리 가능한 양이 아니다.

현재 toy가 허용하는 약한 ML 가설은 다음뿐이다.

> 같은 작은 실험 설정에서 positional head type을 바꾸면 length-OOD PPL이
> 달라질 수 있다. 구조는 데이터·규모·최적화와 함께 결과에 영향을 주는 요인이다.

3 seed, 단일 corpus, 약 127 K parameter의 single-axis 결과는 scaling-only
접근을 강하게 반증하지 못한다. 규모와 데이터의 상호작용을 시험하지 않았기
때문이다. 현재 범위는 head-type 효과의 방향성 toy evidence다.

### 4.2 자유 파라미터 0 원칙과의 관계

CE 본 thesis 는 자유 파라미터 0 (모든 비율이 axiom 에서 연역). 본 발견은 약한 형태:

> Linear additive bias의 *slope*와 *형태*를 설계/학습 변수로 분리해 ablation할
> 수 있다. 어느 형태가 외삽하는지는 axiom이 아니라 외부검증 대상이다.

현재 실험으로 ML이 form을 학습할 수 없다는 경계는 나오지 않는다. mixture,
architecture search, meta-learning baseline과 비교하기 전에는 form-level 선택을
학습 외부 axiom으로 고정할 근거가 없다.

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
| `e` | exponential decay base | **axis 2 후보**: strong distance attenuation |
| `1`, `0` | on/off gate values | 각 axis 의 1 비트 |

→ **2 axis × 2 gate value = 2² = 4 head-types**, 2-bit string `(pi, e)` 으로 인코딩:

강한 attenuation bit는 단순히 감쇠 항이 "존재"하는지가 아니라, additive
logit $b(d)$에 대해 $\liminf_{d\to\infty}-b(d)/d>0$인지로 정의한다.

| (pi, e) | 대표 헤드 타입 | 현재 tiny toy |
|---|---|---|
| (0, 0) | NoPE [Kazemnejad 2023] | 악화 ($+31.7\%$) |
| (0, 1) | ALiBi [Press 2022] | 안정 방향 |
| (1, 0) | RoPE [Su 2021] | 큰 악화 |
| (1, 1) | xPos [Sun 2023] / Euler-CE 후보 | 안정 방향 |

이 규칙에서 log-additive $-c\log d$와 distance에 대해 선형 이상으로 줄지
않는 MRA amplitude는 strong bit가 0이다. 따라서 과거 MRA/MRA-bias를
"attenuation 있음"으로 쓰면서 Tier 2에 둔 모순은 제거된다. 단, 해당 구현과
artifact가 없으므로 이 둘의 실제 분류·성능은 미판정이다. "모든 9개가 정확히
매핑되고 pure rotation만 실패"한다는 결론도 철회한다. 현재 toy에서는
NoPE와 RoPE 둘 다 악화됐다.

4개 상태의 고정길이 binary code는 **2 bit**가 필요하다. $\log_2 3\simeq1.58$
bit는 세 상태를 균등하게 샘플할 때의 Shannon entropy 또는 이상적 평균
code length 하한이지 고정 2-bit 레지스터 길이가 아니다.

`EulerCEMinimal`과 unit tests는 이 4개 연산의 코드 경로를 검사한다. unit
equivalence는 OOD 성능 증거가 아니며, 5차원 continuous parameterization이
정보이론적으로 2 bit에 충분하다는 증명도 아니다. 2-bit 표는 구현 taxonomy다.

---

## 6. AGI 아키텍처 설계 권고

본 발견이 직접적으로 시사하는 설계 원칙:

### R1. Distance attenuation을 독립 ablation

현재 length toy가 지지하는 권고는 positional head에서 강한 attenuation을
독립 arm으로 시험하라는 것이다.

* rotation + linear attenuation (ALiBi)
* rotation + multiplicative attenuation (xPos)
* rotation only와 NoPE 대조군

현재 toy에서 NoPE도 악화됐으므로 "회전을 빼면 안전"이라는 규칙은 쓰지
않는다. Sequence position 결과를 time step, recursion depth 등 모든 거리로
전이하는 것도 별도 실험 전에는 금지한다.

### R2. Form-level 선택도 검증 대상

Form(linear/log/multiplicative/rotation)을 axiom으로 고정하지 않고, 같은 예산의
architecture-search 또는 mixture baseline과 함께 OOD holdout에서 선택한다.

### R3. OOD axis 별 구조 검증 의무

새 아키텍처 제안 시 in-distribution PPL만 보고하지 말고, 사전등록한 OOD
axis와 배율에서 seed 분산·CI를 보고한다. 2.3절 protocol이 현재 reference이며,
과거 9변종 결과표는 reference evidence가 아니다.

---

## 7. 한계와 후속

### 한계

1. **단일 axis (length)**: H1/H2/H3 미검증.
2. **소규모**: 약 127 K params, 3 seed, 500 step. seed CI와 1B+ scaling 미확인.
3. **단일 도메인**: repository Python character corpus. 자연어·다른 modality 미평가.
4. **부분 변종**: 현재 재현은 4 head type뿐이고 MRA/MRA-bias 등 5개 과거 arm은 미재현.
5. **이론 부재**: §3의 attenuation 직관은 가설이며 형식 증명 없음.

### 후속

1. **Scaling**: d_model ∈ {64, 128, 256, 512} × n_layers ∈ {2, 4, 8, 12} 에서 32× 외삽 패턴 유지되는지.
2. **다른 OOD axis**: H1 (코퍼스 transfer), H2 (in-context length), H3 (CoT depth) 각각 검증.
3. **이론**: linear additive 가 외삽하는 형식적 이유 (PAC 학습 이론, Rademacher complexity 분석).
4. **GaugeLattice + linear bias**: 본 발견을 `2_Architecture.md` 의 ClarusBlock 에 통합. 현재 ClarusAttention 은 RoPE 사용 중인데, ALiBi 또는 linear additive variant 로 교체 검토.

---

## 8. 데이터 출처와 재현 상태

- 현재 실행 script: `../experiments/ood_length_repro.py`
- 현재 결과 장부: `../experiments/RESULTS_ood_length.md`
- head 구현: `../reality_stone/python/reality_stone/clarus/ce_euler.py`
- unit tests: `../tests/test_euler_minimal.py` (연산 검증; OOD evidence 아님)
- 과거 `euler_extrap_long.json`, `extrap_full.json`,
  `bench_recursive_euler.py`, `bench_mra_extrap.py`: 현재 checkout에서 missing
- `8_리만/mra_paper.md` §7.7: 과거 기록의 위치이며 raw artifact 대체물이 아님

## 9. References

- Press et al. (2022), *Train Short, Test Long: Attention with Linear Biases (ALiBi)*, ICLR.
- Sun et al. (2023), *A Length-Extrapolatable Transformer (xPos)*, ACL.
- Su et al. (2021), *RoFormer: Rotary Position Embedding*, arXiv:2104.09864.
- Kazemnejad et al. (2023), *The Impact of Positional Encoding on Length Generalization in Transformers*, NeurIPS.
- (Internal) `docs/7_AGI/1_AGI.md`, `2_Architecture.md`, `8_Roadmap.md`.
- (Internal) `docs/8_리만/mra_paper.md` — full ablation data.
