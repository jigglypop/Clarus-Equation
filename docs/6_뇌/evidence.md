# 뇌 실험 근거 정리

> 목적: CE-뇌-AGI 연결에서, 무엇이 실제 실험 근거를 가지는지와 무엇이 아직 브리지 가설인지를 분리한다.
>
> 원칙: 앞으로 `docs/7_AGI`의 수식과 성능 주장은 이 문서의 `supported / bridge / hypothesis` 구분을 기준으로만 올린다.

---

## 1. 판정 기준

### 1.1 세 등급

| 등급 | 의미 |
|---|---|
| `supported` | 뇌 실험/리뷰에서 직접 관측되거나, 적어도 안정적인 정량 범위가 존재 |
| `bridge` | 실험값은 있으나 CE 변수와의 1:1 대응에는 추가 가정이 필요 |
| `hypothesis` | CE 해석은 가능하지만 아직 직접 입증된 실험 대응식이 없음 |

### 1.2 앞으로의 사용 규칙

- `supported`: AGI 문서에서 설계 전제로 사용 가능
- `bridge`: AGI 문서에서 "조건부 대응"으로만 사용
- `hypothesis`: AGI 문서에서 성능 보장이나 정리처럼 쓰지 않음

### 1.3 문서 태그와의 대응

| evidence 태그 | `docs/README.md` 태그로 올릴 수 있는 최대치 | 의미 |
|---|---|---|
| `supported` | `Bridge` | 실험적 근거는 강하지만, CE 변수와의 대응에는 여전히 해석 계층이 있다 |
| `bridge` | `Bridge` | 실험값과 CE 변수 사이에 추가 가정이 남아 있다 |
| `hypothesis` | `Phenomenology` | 구조 유비나 설계 가설로만 사용한다 |

### 1.4 런타임 5계층 게이트

앞으로 `docs/7_AGI/12_Equation.md`의 canonical runtime stack은 아래 다섯 층만 허용한다.

1. `kernel dynamics`
2. `coupling / geometry`
3. `mode update`
4. `hippocampus / replay`
5. `global runtime summary`

이 다섯 층에서:

- 수학적 연산자 정의 자체는 `Exact` 또는 `Selection`일 수 있다.
- 뇌 데이터와 연결되는 대응식은 이 문서의 판정이 `supported`여도 최대 `Bridge`까지만 허용한다.
- `hypothesis` 판정 성분은 `global runtime summary`, 성능, 자아, 의식 같은 상위 서술로만 남겨야 한다.

### 1.5 새 개념별 현재 판정

| 새 개념 | 현재 판정 | 메모 |
|---|---|---|
| sparse lifecycle (`ACTIVE/IDLE/DORMANT/SLEEPING`) | `bridge` | 희소 활성과 수면은 강하지만, 이 4상태 분류 자체는 설계 선택 |
| `WAKE / NREM / REM` 모드 레지스터 | `supported` | 단계 구분과 비율은 강하게 지지됨 |
| hippocampus fast memory / replay | `supported` | 빠른 인코딩-재생-피질 재주입 방향은 강함 |
| local recurrent cell / while-style module | `bridge` | 국소 반복 회로와 persistent state는 plausibility가 크지만 CE 최소식과의 1:1 대응은 아직 bridge |
| global self-state / identity summary | `hypothesis` | 전역 상태 요약을 자아와 동일시하는 것은 아직 현상론 |
| snapshot continuity | `bridge` | 뇌의 연속성/기억 보존과의 유비는 가능하지만 구현 스키마는 공학 선택 |

---

## 2. 에너지 3분배: 활성 / 구조 / 배경

### 2.1 직접 근거

1. 뇌는 체중의 약 2%지만, 전체 에너지의 약 20%를 소비한다.
   - 근거: [Raichle 2006, The Brain's Dark Energy](https://www.science.org/doi/10.1126/science.1134405)

2. 과제 유발(task-evoked) 에너지 증가는 전체 뇌 에너지 예산 대비 매우 작다.
   - 리뷰 요약: 추가 에너지 부담은 대략 `0.5-1.0%` 수준
   - 근거: [Raichle 2006 관련 검색 요약](https://www.scientificamerican.com/article/the-brains-dark-energy/)

3. 뇌의 내재적(intrinsic) 활동은 전체 에너지 소비의 큰 비중을 차지한다.
   - 리뷰 요약: ongoing/intrinsic activity가 대략 `60-80%` 수준을 차지
   - 근거: [The Brain's Dark Energy 관련 요약](https://www.scientificamerican.com/article/the-brains-dark-energy/), [The restless brain](https://royalsocietypublishing.org/doi/full/10.1098/rstb.2014.0172?etoc=), [The Brain's Default Mode Network](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-071013-014030)

4. 피질 에너지 예산에서 시냅스 관련 비용은 지배적이다.
   - signaling만 볼 때: synaptic processes `59%`, action potentials `21%`, resting potentials `20%`
   - nonsignaling 포함 시: synaptic `44%`, action potentials `16%`, resting potentials `15%`, housekeeping `25%`
   - 근거: [Cortical energy demands of signaling and nonsignaling components in brain are conserved across mammalian species and activity levels](https://www.pnas.org/doi/full/10.1073/pnas.1214912110)

5. 시냅스는 에너지 소비의 주요 장소이며, synaptic plasticity와 brain state 변화가 에너지 사용을 바꾼다.
   - 근거: [Harris, Jolivet, Attwell 2012, Synaptic energy use and supply](https://www.cell.com/neuron/fulltext/S0896-6273(12)00756-8)

### 2.2 숫자 체크

| CE 항목 | CE 값 | 실험 proxy | 관측값 | 체크 |
|---|---|---|---|---|
| 활성 $x_a$ | `4.87%` | sparse task-related recruitment / sparse ensemble | `1-5%` 급, task-induced burden `0.5-1.0%`, DG active cells `1-3%` | `[NEAR]` |
| 구조 $x_s$ | `26.2%` | housekeeping / maintenance cost | cortical housekeeping `25%` | `[OK]` |
| 배경 $x_b$ | `68.9%` | intrinsic / DMN / resting background | `60-80%` | `[OK]` |

해석:
- `x_a = 4.87%`는 단일 실험값 하나로 찍히지는 않지만, sparse ensemble과 DG 활성 `1-3%`, 고전적 `1-5%` 범위와 같은 스케일에 있다.
- `x_s = 26.2%`는 maintenance proxy로 `housekeeping 25%`가 가장 직접적으로 가깝다.
- `x_b = 68.9%`는 intrinsic/background `60-80%` 범위 안에 있다.

### 2.3 CE 대응

| 뇌 측 관측 | 실험 범위 | CE 대응 | 판정 |
|---|---|---|---|
| task-evoked burden | `0.5-1.0%` 추가 부담, 또는 강한 과제에서도 소수 변화 | `x_a`의 매우 작은 활성 분율 | `supported` |
| intrinsic / DMN / resting background | 대략 `60-80%` | `x_b` 배경 분율 | `supported` |
| housekeeping / structural maintenance | 대략 `25%` | `x_s` 구조 분율 | `supported` |
| synaptic signaling backbone | 대략 `44-59%` | `x_s`의 더 넓은 구조 proxy 후보 | `bridge` |

### 2.4 해석

- `26.2%`는 예전보다 더 적극적으로 체크할 수 있다. 최소한 energy-budget 문헌의 `housekeeping = 25%`와는 매우 가깝다.
- `68.9%`도 intrinsic/background `60-80%` 범위에 분명히 들어간다.
- 남는 약점은 `4.87%`다. 이 값은 sparse firing의 대표 스케일과는 잘 맞지만, 지금 확보한 실험값은 `1-3%`, `1-5%`, `<=15%` 같은 범위이지 정확히 `4.87%` 자체를 직접 재는 값은 아니다.

### 2.5 판정

$$
\boxed{\text{에너지 3분배는 } x_s, x_b \text{ 쪽은 꽤 강하게 맞고, } x_a \text{는 같은 스케일의 근접 일치다.}}
$$

---

## 3. 수면: Wake / NREM / REM

### 3.1 직접 근거

1. 정상 성인 수면은 보통:
   - `NREM 75-80%`
   - `REM 20-25%`
   - 약 `90분` 주기
   - 근거: [Normal adult sleep architecture 검색 요약](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(19)30057-8/abstract), [Physiology of Sleep](https://ncbi.nlm.nih.gov/pmc/articles/PMC4755451/)

2. 서파 수면(slow-wave sleep)과 수면 압력은 강하게 연결된다.
   - slow-wave activity는 각성 시간과 homeostatic sleep drive를 반영하는 대표 지표다.
   - 근거: [Sleep and synaptic down-selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/)

3. Tononi-Cirelli의 SHY(synaptic homeostasis hypothesis)는 다음을 주장한다.
   - 각성 동안 순 시냅스 강화
   - 수면 동안 전역적 down-selection / renormalization
   - 수면의 느린 파동과 synaptic strength 조절의 연결
   - 근거: [Sleep and synaptic down-selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/)

4. 48-72시간 수면 박탈에서는 경계성(alertness), 인지 수행, 전전두엽-시상 대사 활동이 연속적으로 저하된다.
   - 근거: [48 and 72 h of sleep deprivation on waking human regional brain activity](https://www.sciencedirect.com/science/article/abs/pii/S1472928803000207), [Sleep deprivation, vigilant attention, and brain function: a review](https://pubmed.ncbi.nlm.nih.gov/31176308/)

### 3.2 숫자 체크

| 항목 | CE 해석 | 실험값 | 체크 |
|---|---|---|---|
| NREM 우세 위상 | offline 정리 위상 | `75-80%` of adult sleep | `[OK]` |
| REM 재조합 위상 | 소수지만 반복되는 재탐색 위상 | `20-25%` of adult sleep | `[OK]` |
| slow-wave homeostasis | 곡률/시냅스 정리의 지표 | SHY와 서파 강한 연결 | `[OK]` |
| 반복 실패의 인지 비용 | 2-3회 반복 부재 시 급격한 저하 | `48-72시간` 박탈에서 지속적 성능 저하 | `[NEAR]` |

### 3.3 CE 대응

| 수면 현상 | 실험 측 사실 | CE 대응 | 판정 |
|---|---|---|---|
| Wake에서 정보 축적 | 경험 후 synaptic potentiation 증가 경향 | 경로 누적 | `bridge` |
| NREM에서 서파 + renormalization | SHY와 광범위한 수면 항상성 데이터 | 곡률 평탄화 / down-selection | `supported` |
| REM에서 재조합 / 꿈 / 기억 통합 | REM의 재활성화와 기억 처리 역할 | 비선택 경로 재탐색 | `hypothesis` |

### 3.4 해석

- "수면이 오프라인 정리 과정이다"는 방향은 실험적으로 강하다.
- "Wake-NREM-REM 1회 = CE 부트스트랩 사상 1회 적용"이라는 문장은 수정이 필요하다. 수면 회복 문헌(Belenky 2003, Banks 2010, Van Dongen 2003, Kitamura 2016)에 지수 감쇠 모델을 피팅하면 $\rho_{\text{night}} \approx 0.31$ ($R^2 > 0.99$)이 나온다. CE의 $\rho = 0.155$와 비교하면 **$B$의 1회 완전 적용은 약 1.6밤의 회복 수면**에 대응한다.
- 이것은 Borbely(1982)의 2-process 모델에서도 1밤으로는 SWS debt가 완전 해소되지 않는다는 관측과 정합한다.
- `48-72시간` 박탈에서 인지 수행이 급격히 무너지는 것은, CE의 "2.5회 적용 $\approx$ 4밤이면 1% 정밀도"와 시간 스케일이 맞는다.
- `sleep.md`의 큰 그림(수면 = 부트스트랩 반복)은 유지되지만, 매핑 단위가 "1밤 = 1적용"에서 "1.6밤 = 1적용"으로 보정되었다.

### 3.5 판정

$$
\boxed{\text{수면의 offline renormalization은 강하게 체크됨. 반복 사상 해석은 bridge이되, 매핑 단위가 1.6밤/적용으로 보정됨.}}
$$

---

## 4. 희소 발화와 기억 앙상블

### 4.1 직접 근거

1. 피질은 sparse coding을 폭넓게 사용한다.
   - 감각 정보는 소수의 동시에 활성화된 뉴런 집합으로 표현되는 경향이 있다.
   - 근거: [Experimental evidence for sparse firing in the neocortex](https://pubmed.ncbi.nlm.nih.gov/22579264/), [Sparse coding of sensory inputs](https://www.semanticscholar.org/paper/Sparse-coding-of-sensory-inputs-Olshausen-Field/0dd289358b14f8176adb7b62bf2fb53ea62b3818)

2. sparse firing은 에너지 효율과 저장 용량 측면에서 장점이 있다.
   - 근거: [Sparse coding review 요약](http://www.rctn.org/bruno/papers/current-opinion.pdf)

3. memory engram은 broad dense code가 아니라 sparse ensemble로 이해된다.
   - 근거: [How Does the Sparse Memory “Engram” Neurons Encode the Memory of a Spatial–Temporal Event?](https://www.frontiersin.org/articles/10.3389/fncir.2016.00061/full), [Engrams](https://www.cell.com/current-biology/fulltext/S0960-9822(24)00605-5)

4. dentate gyrus에서는 생리적 자극 시 granule cell layer의 활성 뉴런이 대략 `1-3%` 수준이라는 보고가 있다.
   - 근거: [Distinct patterns of dentate gyrus cell activation distinguish physiologic from aberrant stimuli](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0232241)

5. 에너지 budget 관점에서는 동시에 활성인 뉴런 비율이 `<=15%`여야 한다는 상한 예측이 있다.
   - 근거: [Attwell & Laughlin 2001](https://pubmed.ncbi.nlm.nih.gov/11598490/)

6. awake auditory cortex에서는 자극이 임의 시점에 고발화(>20 sp/s) 상태로 끌어올리는 뉴런이 `5% 미만`이라는 직접 측정이 있다. 같은 논문에서 유의한 stimulus-locked 증가를 보이는 뉴런도 대략 `10%` 수준이다.
   - 근거: [Sparse Representation of Sounds in the Unanesthetized Auditory Cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0060016)

### 4.2 숫자 체크

| CE 항목 | CE 값 | 실험값 | 체크 |
|---|---|---|---|
| 활성 희소성 | `4.87%` | auditory cortex well-driven `<5%`, DG active `1-3%` | `[NEAR]` |
| 활성 희소성 상한 | `4.87%` | energy-budget upper bound `<=15%` | `[OK]` as feasible range |
| sparse code 일반 원리 | 소수 활성 | cortex-wide sparse coding 강함 | `[OK]` |

### 4.3 CE 대응

| 현상 | 실험 측 사실 | CE 대응 | 판정 |
|---|---|---|---|
| sparse firing | 대부분의 피질 영역에서 강하게 관찰 | `x_a` 소수 활성 | `supported` |
| engram sparsity | 기억은 sparse ensemble에 저장 | 장기 생존 경로의 희소성 | `supported` |
| 정확한 `4.87%` | auditory cortex well-driven `<5%`, DG `1-3%`, 고전적 sparse range `1-5%`, upper bound `<=15%` | `\varepsilon^2` | `bridge` |

### 4.4 해석

- "희소 발화" 자체는 매우 강하다.
- `4.87%`도 이제 완전히 공중에 떠 있는 수치는 아니다. DG `1-3%`, 일반 sparse range `1-5%`뿐 아니라, awake auditory cortex의 "well-driven 뉴런 `<5%`"와도 직접 맞닿아 있다.
- 다만 "모든 뇌 영역이 정확히 4.87%에 수렴한다"는 보편 법칙까지는 아직 직접 체크되지 않았다.
- 따라서 `5_Sparsity.md`에서는 `강한 후보 고정점` 또는 `검증 목표값`으로 쓰는 것이 현재 가장 안전하다.

### 4.5 감각별 sparse ensemble 체크

1. 시각 피질 V1에서는 자연 영상 한 장이 저단위 비율의 뉴런만 강하게 활성화하는 sparse population code를 만든다. 최근 two-photon 결과 요약에서는 image당 responsive cell 비율의 중앙값이 대체로 `2-5%` 수준으로 보고된다.
   - 근거: [Natural images are reliably represented by sparse and variable populations of neurons in visual cortex](https://www.nature.com/articles/s41467-020-14645-x)

2. 청각 피질에서는 이미 본 것처럼 well-driven 뉴런이 `5% 미만`이고, stimulus-locked 증가도 대략 `10%` 수준이다.
   - 근거: [Sparse Representation of Sounds in the Unanesthetized Auditory Cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0060016)

3. 촉각 피질에서는 sparse coding이 존재하지만, 시각/청각보다 더 넓고 layer/state 의존적이다. barrel cortex의 active touch 과제에서는 superficial layer 2/3 피라미드 뉴런 중 약 `17%`가 touch/whisking을 함께 나타낸다는 보고가 있다.
   - 근거: [A Cellular Resolution Map of Barrel Cortex Activity during Tactile Behavior](https://www.sciencedirect.com/science/article/pii/S0896627315002512)

4. tactile response는 layer 4에서 layer 2로 갈수록 더 sparse해지는 변환도 보고된다.
   - 근거: [Transformation of primary sensory cortical representations from layer 4 to layer 2](https://www.nature.com/articles/s41467-022-33249-1)

### 4.6 감각간 결합 체크

1. 교차감각 통합은 "처음부터 완전 융합된 dense code"보다, 감각 피질 간 communication/coherence를 통해 이루어진다는 근거가 강하다.
   - 근거: [Synchronization of Sensory Gamma Oscillations Promotes Multisensory Communication](https://www.eneuro.org/content/6/5/ENEURO.0101-19.2019), [Crossmodal binding through neural coherence: implications for multisensory processing](https://www.sciencedirect.com/science/article/abs/pii/S0166223608001446)

2. visual-tactile congruence 과제에서는 감각 피질 사이의 gamma coupling/coherence가 실제 행동 이득과 연결되며, 저자들은 이를 단순 feature binding보다 **corticocortical communication**으로 해석한다.
   - 근거: [Synchronization of Sensory Gamma Oscillations Promotes Multisensory Communication](https://www.eneuro.org/content/6/5/ENEURO.0101-19.2019)

### 4.7 CE 대응

| 현상 | 실험 측 사실 | CE 대응 | 판정 |
|---|---|---|---|
| 시각 sparse ensemble | 자연 영상마다 V1 low single-digit responsive fraction | `h_{\text{vision}}^{act}` | `supported` |
| 청각 sparse ensemble | well-driven `<5%`, stimulus-locked `~10%` | `h_{\text{audio}}^{act}` | `supported` |
| 촉각 sparse ensemble | active touch에서 L2/3 `~17%`, 계층 따라 더 희소해짐 | `h_{\text{touch}}^{act}` | `supported` for sparsity, `bridge` for exact ratio |
| 교차감각 gamma communication | visual-tactile / audio-visual congruence에서 gamma coherence 변화 | `\operatorname{Bind}_\xi` 전 단계의 감각간 결합 | `bridge` |
| 모든 모달리티가 동일한 `4.87%`로 맞는다 | 직접 근거 부족, modality/layer/state 의존성 큼 | 공통 `\varepsilon^2` 고정점 | `bridge` |

### 4.8 해석

- 여기서 중요한 건 "모든 감각이 정확히 같은 희소율"이 아니라, **감각별로 먼저 sparse ensemble이 형성되고 그 다음에 교차감각 communication이 붙는다**는 순서다.
- 시각과 청각은 `4.87%` 근방의 low single-digit 희소성과 꽤 잘 맞는다.
- 촉각은 sparse coding 자체는 분명하지만, 과제와 층에 따라 `~17%`처럼 더 넓은 값도 나온다. 따라서 CE의 `4.87%`는 촉각까지 포함한 보편 상수라기보다 우선 `검증 중심값` 또는 `공통 sparse center`로 읽는 편이 안전하다.
- 즉 grounded CE-AGI의 핵심 문장은 "모달리티별 sparse encoder가 먼저"까지는 실제 뇌와 잘 맞고, "모든 모달리티가 동일한 exact top-k"는 아직 브리지다.

---

## 5. STDP, eligibility trace, dopamine

### 5.1 직접 근거

1. 3-factor learning rule은 현대 신경과학에서 강한 실험 지지를 얻고 있다.
   - 근거: [Liakoni et al. 2018 review](https://www.frontiersin.org/articles/10.3389/fncir.2018.00053/full)

2. striatum에서는 dopamine-gated eligibility trace가 대략 `1초` 창에서 작동한다.
   - Yagishita et al. 2014 실험을 요약한 리뷰에 따르면, dopamine은 STDP 직후 `1초` 근방에서는 LTP를 촉진하지만 `4초` 지연에서는 효과가 사라진다.
   - 근거: [Liakoni et al. 2018 review](https://www.frontiersin.org/articles/10.3389/fncir.2018.00053/full)

3. cortex에서는 neuromodulator-gated eligibility trace가 `수 초` 범위다.
   - LTP 쪽 trace는 대략 `5-10초`, LTD 쪽은 더 짧은 `~3초` 범위
   - 근거: [Liakoni et al. 2018 review](https://www.frontiersin.org/articles/10.3389/fncir.2018.00053/full)

4. hippocampus에서는 `수 초`에서 `수 분`까지 더 긴 time scale도 보고된다.
   - Brzosko/Bittner 계열 실험은 과제와 세포 유형에 따라 `1-2초`, `1분`, 또는 더 긴 synaptic tag를 시사
   - 근거: [Liakoni et al. 2018 review](https://www.frontiersin.org/articles/10.3389/fncir.2018.00053/full)

### 5.2 숫자 체크

| 항목 | CE 문서 해석 | 실험값 | 체크 |
|---|---|---|---|
| striatum dopamine window | 빠른 global gating | 약 `1초`, `4초`면 소실 | `[OK]` |
| cortex LTP trace | 수 초 단위 flag | `5-10초` | `[OK]` |
| cortex LTD trace | 더 짧은 감쇠 | 약 `2.5-3초` | `[OK]` |
| hippocampus trace | 더 긴 문맥 통합 | `1-2초`에서 `1분` 이상 | `[OK]` |

### 5.3 CE 대응

| 현상 | 실험 측 사실 | CE 대응 | 판정 |
|---|---|---|---|
| STDP | pair/triplet timing-dependent plasticity는 확립 | 국소 경로 선택 규칙 | `supported` |
| eligibility trace | synapse-local hidden flag는 실험 지지 강함 | `e_{ij}` | `supported` |
| third factor | dopamine, NE, 5-HT, surprise, novelty 등 복수 후보 | 전역 조절항 | `supported` |
| `\delta[t] = d/dt ||p-p^*||` | 그런 형태의 뇌 실험량은 아직 직접 없음 | CE 글로벌 오차 신호 | `hypothesis` |

### 5.4 해석

- "STDP + 전역 신호"는 실제 뇌에 강한 근거가 있다.
- 다만 그 전역 신호를 CE 문서처럼 `고정점 이탈량`으로 둘 수 있는지는 아직 입증되지 않았다.
- 즉 `synapse.md`는 살릴 수 있지만, `δ[t]` 정의는 다시 세워야 한다.

---

## 6. 뇌 진동: gamma / beta / alpha / slow waves

### 6.1 직접 근거

1. gamma는 perceptual binding, temporal coordination, local assembly formation과 자주 연결된다.
   - 근거: [Gamma oscillations and binding 검색 요약](https://pubmed.ncbi.nlm.nih.gov/11164732/), [Mechanisms of Gamma Oscillations](https://ncbi.nlm.nih.gov/pmc/articles/PMC4049541/)

2. beta는 motor control, current cognitive set 유지, status quo 유지와 강하게 연결된다.
   - 근거: [Beta-band oscillations—signalling the status quo?](https://pubmed.ncbi.nlm.nih.gov/20359884/), [Understanding the Role of Sensorimotor Beta Oscillations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8200463/)

3. alpha는 inhibitory gating, distractor suppression, attention selection과 강하게 연결된다.
   - 근거: [Shaping Functional Architecture by Oscillatory Alpha Activity: Gating by Inhibition](https://ncbi.nlm.nih.gov/pmc/articles/PMC2990626/)

4. slow wave / delta / theta는 수면 항상성, 장거리 조율, 기억 고정과 강하게 연결된다.
   - 근거: [Sleep and synaptic down-selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/)

5. theta-gamma coupling은 sequential memory와 episodic ordering을 지탱하는 대표적 구조로 제안된다.
   - 근거: [The Theta-Gamma Neural Code](https://www.cell.com/neuron/fulltext/S0896-6273(13)00231-6), [Theta-gamma coupling in the entorhinal-hippocampal system](https://ncbi.nlm.nih.gov/pmc/articles/PMC4340819/)

### 6.2 CE 대응

| 진동 대역 | 실험적 역할 | CE 문서 대응 | 판정 |
|---|---|---|---|
| gamma | binding / local integration | SU(3) binding | `bridge` |
| beta | motor/cognitive set 유지 | SU(2) decision/control | `bridge` |
| alpha | inhibitory gating / attention | U(1) gating | `bridge` |
| slow wave | global renormalization / sleep homeostasis | `\Phi` smoothing | `bridge` |

### 6.3 계층 결합 체크

| CE 해석 | 실험 근거 | 체크 |
|---|---|---|
| 느린 파동이 빠른 연산을 조절 | alpha는 gamma 처리 영역을 억제/게이팅, theta-gamma coupling은 기억 순서를 조직 | `[OK]` |
| gamma = 국소 결합, alpha = 선택적 게이팅 | gamma-binding + alpha gating by inhibition | `[OK]` |
| beta = 현재 상태 유지 / 제어 | beta signalling the status quo, cognitive set maintenance | `[OK]` |

### 6.4 해석

- 기능적 대응은 꽤 그럴듯하다.
- 특히 `alpha -> gating`, `beta -> current set/control`, `gamma -> binding`, `theta/slow -> global organization`의 분업은 최근 리뷰와 상당히 잘 맞는다.
- 하지만 `SU(3)/SU(2)/U(1)` 자체가 뇌 실험에서 직접 측정되는 것은 아니다.
- 따라서 이 장은 구조적 유비로는 강하지만, 코어 정리처럼 쓰면 안 된다.

---

## 7. 지금 시점의 판정 요약

| CE-뇌 주장 | 현재 판정 | 이유 |
|---|---|---|
| 뇌는 소수 활성 + 거대 배경 활동 구조를 가진다 | `supported` | task-evoked burden와 intrinsic activity 데이터가 강함 |
| 뇌는 sparse code를 사용한다 | `supported` | sparse firing과 engram literature가 강함 |
| 감각별로 먼저 sparse ensemble을 만들고 이후 교차감각 communication으로 결합한다 | `supported/bridge` 혼합 | 시각/청각 sparse coding은 강하고, 촉각도 희소하나 더 넓으며, gamma coherence는 communication 근거가 강하지만 exact 대응식은 아직 없음 |
| 뇌는 STDP + eligibility trace + third factor를 사용한다 | `supported` | 최근 리뷰와 실험이 강함 |
| 수면은 offline renormalization 기능을 가진다 | `supported` | SHY와 slow-wave evidence가 강함 |
| 뇌의 정확한 고정점이 `4.87/26.2/68.9`다 | `bridge` | 세 수치 모두 실험 범위 안쪽 또는 근접하며 정합성은 높지만, 동일 변수 정의가 완전히 닫히진 않음 |
| Wake-NREM-REM이 CE 부트스트랩 반복 1회다 | `bridge` | 수면 기능은 지지되지만 반복 사상 정의가 필요 |
| gamma/beta/alpha가 SU(3)/SU(2)/U(1)이다 | `bridge` | 기능 대응과 계층 게이팅 근거는 강하지만, 직접 측정식은 아님 |
| dopamine error가 곧 `\delta[t] = ||p-p^*||` 또는 그 도함수다 | `hypothesis` | 생물학적 직접 근거 없음 |
| 환각률 / 지능 / 의식 깊이를 `\varepsilon^2`, `||p-p^*||`로 직접 상한화할 수 있다 | `hypothesis` | 아직 뇌 측 검증도, AI 측 검증도 부족 |

### 7.1 총평

현재까지의 체크를 종합하면:

- `4.87 / 26.2 / 68.9`는 적어도 뇌 데이터와 **스케일이 맞는 우연 이상의 패턴**으로 보인다.
- 특히 `26.2`와 `68.9`는 energy-budget / intrinsic-activity 데이터와 꽤 강하게 들어맞는다.
- `4.87`도 sparse firing, DG `1-3%`, 고전적 `1-5%` 범위와 매우 가깝다.
- 추가로 grounded 구조의 핵심인 "모달리티별 sparse ensemble -> 교차감각 communication"도 실제 뇌 데이터와 방향이 맞는다.
- 남은 핵심 과제는 "정합하느냐"보다, **정확히 어떤 측정량을 CE 변수 $x_a, x_s, x_b$로 둘 것인가**를 닫는 일이다.

### 7.2 구간 포함 정합 식

실험 범위를 다음의 구간으로 잡자:

$$
I_a = [0.01,\; 0.05], \qquad
I_s = [0.25,\; 0.35], \qquad
I_b = [0.60,\; 0.80]
$$

CE 고정점:

$$
p^* = (0.0487,\; 0.2623,\; 0.6891)
$$

그러면 성분별로

$$
0.0487 \in I_a, \qquad 0.2623 \in I_s, \qquad 0.6891 \in I_b
$$

즉,

$$
\boxed{p^* \in I_a \times I_s \times I_b}
$$

이다. 이는 "스케일이 비슷하다"보다 더 강한 문장이다. 최소한 현재 확보한 실험 구간 안에서 CE 고정점은 **성분별 구간 포함**을 만족한다.

정량적 정합도 함수를

$$
d_{[l,u]}(x) =
\begin{cases}
0, & x \in [l,u] \\
\dfrac{\min(|x-l|,\;|x-u|)}{u-l}, & x \notin [l,u]
\end{cases}
$$
$$
D(p) = d_{I_a}(p_a) + d_{I_s}(p_s) + d_{I_b}(p_b)
$$

로 두면,

$$
\boxed{D(p^*) = 0}
$$

이다. 즉 현재 선택한 대표 실험 구간들에 대해 CE 고정점은 이미 구간 밖에 있지 않다.

---

## 8. 뇌 구조 full-stack 모델

### 8.1 해부학-제어 계층

이제부터 `r`는 막연한 voxel이 아니라 **해부학-제어 계층을 반영한 parcel/node**로 읽는다. 현재 문서의 메인라인 brain stack은 다음 네 층으로 둔다.

| 층 | 포함 구조 | 실험적으로 안전한 역할 | CE에서 맡는 역할 | 판정 |
|---|---|---|---|---|
| `L1` 피질-시상 처리층 | 피질 task network, DMN, thalamo-cortical relay | 감각 입력 routing, 과제 반응, intrinsic background 유지 | `x_a`, `x_b`의 주 관측창 | `bridge` |
| `L2` 기억-살리언스 층 | hippocampus, amygdala, basal ganglia, salience switching | 기억 재활성화, 가치/위험 표지, 행동 전환 | `x_a` 재할당과 `x_s` plasticity burden | `bridge` |
| `L3` 항상성 제어층 | hypothalamus, brainstem, LC/raphe/NTS 계열 | 수면-각성, 자율신경 set-point, 체온/식욕/호르몬 조절 | `x_b` 유지와 `s_r` 억제의 주축 | `bridge` |
| `L4` 뇌-몸 결합층 | endocrine, immune tone, metabolic reserve, vascular tone | 전신 완충력과 장기별 stress coupling | `s_r \to w_r` 전달축 | `bridge/hypothesis` |

여기서 중요한 점은, 시상하부-뇌간-자율신경이 "배경 설명"이 아니라 실제로 `x_b`와 복구 여력을 떠받치는 제어층이라는 것이다.

구체화는 한 파일에 몰아넣지 않고 다음처럼 분리한다.

| 문서 | 맡는 역할 |
|---|---|
| `evidence.md` | 메인라인 요약과 뇌-암 결합식 |
| `anatomy.md` | 해부학-제어 계층과 정점 가족 |
| `measure.md` | `p_r`, `x_a`, `x_s`, `x_b`, `s_r`의 측정 정의 |
| `control.md` | `q_n`, `\ell_r`, sleep-autonomic-endocrine-immune-metabolic body loop |
| `graph.md` | brain graph, `\Delta_G`, graph-coupled relaxation |
| `proof.md` | 각 부위/중추/알고리즘 모듈의 증명 게이트와 반증 조건 |
| `agi.md` | `3x3+1` 구조 유비와 bridge 해석 |

### 8.2 `p_r`와 `s_r`의 측정 정의

세부 operationalization은 `measure.md`와 `control.md`로 분리해 두고, 여기서는 mainline에 필요한 최소 요약만 유지한다. 각 항이 실제로 증명 가능한지는 `proof.md`를 따른다.
특히 현재 `pass / partial / fail` 실판정은 `proof.md`의 `2.1`, `3.8`, `4.1`, `5.1`을 기준으로 읽는다.
초기 단계에서 전부를 어떻게 실행할지는 `proof.md`의 `7.1`부터 `7.7`까지를 기준으로 읽는다.

가장 먼저 필요한 것은 지역별 3분배와 취약도를 닫는 것이다.

$$
p_r(t) = \big(x_{a,r}(t), x_{s,r}(t), x_{b,r}(t)\big) \in \Delta^2
$$

직접 한 번에 재는 값은 아직 없으므로, 현재는 각 영역의 비음수 multimodal score를 정규화해

$$
\hat x_{a,r}(t),\;\hat x_{s,r}(t),\;\hat x_{b,r}(t)\ge 0,
\qquad
p_r(t)
=
\frac{1}{\hat x_{a,r}+\hat x_{s,r}+\hat x_{b,r}}
\big(\hat x_{a,r},\hat x_{s,r},\hat x_{b,r}\big)
$$

처럼 읽는 것이 가장 안전하다.

| 성분 | 지금 바로 쓸 수 있는 proxy | 주 관측 도구 | 현재 판정 |
|---|---|---|---|
| `x_a` | event-locked firing fraction, task-evoked BOLD/PET increment, task-positive gamma/beta burden | EEG/MEG, fMRI, FDG-PET | 방향성은 `supported`, 정규화된 지역 비율은 `bridge` |
| `x_s` | synaptic maintenance/plasticity burden, sleep-dependent renormalization load, glial support/repair cost | sleep EEG, PET, molecular assay | 전역 해석은 `supported`, 지역별 정량화는 `bridge` |
| `x_b` | resting metabolism, DMN/intrinsic activity, tonic spontaneous background | rs-fMRI, PET, resting EEG | 방향성은 `supported`, node-wise simplex 변수화는 `bridge` |

#### 뇌-몸 조절축 `q_n`

이제 `s_r`는 "미지의 한 숫자"가 아니라, 과활성 + 복구 부족 + 배경 항상성 부족 + 뇌-몸 조절 부하의 합으로 읽는다.

$$
\ell_r(n) = d_r^\top \big(q_n-q^*\big)_+,
\qquad
q_n=\big(q_{\text{sleep},n}, q_{\text{arousal},n}, q_{\text{aut},n}, q_{\text{endo},n}, q_{\text{immune},n}, q_{\text{met},n}\big)
$$

$$
s_r(n)
=
\eta_a\big(x_{a,r}(n)-x_a^*\big)_+
\;+\;
\eta_s\big(x_s^*-x_{s,r}(n)\big)_+
\;+\;
\eta_b\big(x_b^*-x_{b,r}(n)\big)_+
\;+\;
\eta_q \ell_r(n)
$$

| 항 | 의미 | 대표 proxy | 판정 |
|---|---|---|---|
| `q_sleep` | 수면 부채 / slow-wave 부족 | sleep EEG, actigraphy | `supported/bridge` |
| `q_arousal` | hyperarousal / vigilance overload | pupil, EEG arousal index, LC-related proxy | `bridge` |
| `q_aut` | sympathetic-vagal 불균형 | HRV, blood pressure variability | `supported` |
| `q_endo` | endocrine stress load | cortisol, endocrine panel | `supported` |
| `q_immune` | inflammatory / immune tone | CRP, cytokine panel, immune signature | `supported` |
| `q_met` | 대사/혈관 reserve 부족 | glucose variability, perfusion, temperature | `bridge` |

즉 지금 당장 안전하게 말할 수 있는 것은 "`x_a`, `x_b`는 꽤 직접적으로 보이고, `x_s`는 composite proxy가 필요하며, `s_r`는 body-loop load까지 포함해야 한다"는 점이다. 특히 `q_sleep`을 단독으로 보면, 아래의 기존 수면 수축식은 여전히 유효한 최소 근사로 남는다.

$$
\limsup_{n\to\infty} \|e_n\| \le \frac{\rho}{1-\rho} U
= \frac{0.155}{0.845} U \approx 0.183 U
$$

가 된다. 즉 수면이 있을 때는 각성으로 생기는 이탈이 약 `18.3%` 수준으로 눌린다.

반대로 수면이 없으면:

$$
e_{n+1}^{\text{wake-only}} = e_n^{\text{wake-only}} + u_n
$$

이므로, $\|u_n\| \le U$일 때 최악의 경우

$$
\|e_N^{\text{wake-only}}\| \le \|e_0\| + NU
$$

처럼 선형으로 누적된다.

이 식은 다음 두 실험 사실과 잘 맞는다.
- prior wake duration이 길수록 delta power가 증가한다
- `48-72시간` 수면 박탈에서 전전두엽-시상계와 인지 기능이 연속적으로 악화된다

따라서 CE의 "수면은 drift를 접어 넣는 수축 단계"라는 해석은, 최소 반복식 수준에서는 상당히 자연스럽다.

반대로 동일 위치에서 EEG/fMRI/PET/수면/자율신경을 합쳐 `p_r`를 완전히 닫는 것은 아직 `hypothesis`다.

### 8.3 flat region index 대신 brain graph

brain graph의 구체적인 parcel family와 graph-coupled relaxation map은 `graph.md`로 분리했다. 여기서는 `r`를 flat index가 아닌 node로 읽는다는 점만 고정한다.

이제 `r`는 평평한 번호표가 아니라 brain graph의 정점이다.

$$
G_{\text{brain}}=(V_{\text{brain}}, E_{\text{brain}})
$$

최소 분할은

$$
V_{\text{brain}}
=
V_{\text{ctx}}
\sqcup
V_{\text{thal}}
\sqcup
V_{\text{hip}}
\sqcup
V_{\text{sal}}
\sqcup
V_{\text{hyp}}
\sqcup
V_{\text{stem}}
\sqcup
V_{\text{aut}}
$$

로 두고, graph coupling은

$$
\Delta_G f(r)
=
\sum_{s:(s,r)\in E_{\text{brain}}}
a_{rs}\big(f_s-f_r\big),
\qquad
a_{rs}\ge 0
$$

처럼 적는다.

이 항의 해석은 다음과 같다.
- 피질-피질 인접성: 과제 burden과 background drift의 확산
- 시상-피질 relay: 입력 gating과 대역 재분배
- 해마-피질 replay: 기억 재활성화와 복구 경로
- 시상하부-뇌간-자율신경 축: distributed correction 또는 failure propagation

즉 graph 항은 "한 지점의 문제"가 어떻게 주변으로 퍼지거나, 반대로 주변 완충망에 의해 눌리는지를 설명한다. 해부학적 연결 자체는 `supported`지만, 이를 `\Delta_G` 한 줄로 쓰는 것은 현재 `bridge`다.

### 8.4 `B`의 재정의와 최소 반복식

`B_*`, `B`, graph-coupled relaxation의 상세는 `graph.md`, sleep/autonomic forcing의 해부학적 해석은 `control.md`를 따른다.

이제 `graph.md` 기준의 `B`는 다음 두 층으로 분리한다.

1. 정적 고정점 연산자:

$$
B_* : p \mapsto p^*
$$

2. 실제 동역학 반복:

$$
p_{n+1} = \mathcal{F}(p_n; \theta_{\text{brain}}, q_n, G_{\text{brain}})
$$

full-stack 최소 반복식은

$$
p_{n+\frac12} = p_n + u_n, \qquad \mathbf{1}^\top u_n = 0
$$

$$
p_{n+1}
=
p^*
+ \rho \big(p_{n+\frac12} - p^*\big)
+ \gamma_p \Delta_G p_{n+\frac12}
+ H(q_n-q^*)
+ \xi_n,
\qquad
\rho = 0.155 \;\text{(per application of }B\text{; per night: }\rho_{\text{night}} \approx 0.31\text{)}
$$

여기서 $B$는 `homeomorphism.md` 정의 1.3의 최소 등방 closure이고, $n$은 그 적용 횟수다. 1회 적용은 약 1.6밤의 회복 수면에 대응한다. 즉

$$
e_n := p_n - p^*
\quad \Longrightarrow \quad
e_{n+1}
=
\rho e_n
+ \rho u_n
+ \gamma_p \Delta_G e_{n+\frac12}
+ H(q_n-q^*)
+ \xi_n
$$

이다.

이 식의 의미는 단순하다.
- `\rho e_n`: 수면과 homeostasis가 기준점으로 끌어당기는 수축
- `\gamma_p \Delta_G e`: 지역 사이 spread 또는 buffering
- `H(q_n-q^*)`: sleep debt, autonomic imbalance, endocrine/immune/metabolic load

### 8.5 곡률과 `\delta[t]`의 현재 정의

현재 `\|\Delta_g \Phi\|^2`는 너무 추상적이므로, 뇌 측 후보 proxy는 다음처럼 제한한다.
- slow-wave activity
- population synchrony / desynchrony
- functional connectivity roughness
- replay burden

그리고 가장 안전한 global 조절 오차 후보는 여전히

$$
\delta[t] = a \cdot \text{RPE}(t) + b \cdot \text{surprise}(t) + c \cdot \text{novelty}(t)
$$

이다. 다만 이제는 이 항 역시 피질 단독이 아니라 `q_n`과 함께 읽어야 한다. 즉 dopamine/novelty는 국소 학습 신호이고, sleep-autonomic-endocrine-immune load는 더 느린 제어 배경이다.

### 8.6 이 절의 사용 규칙

| 문장 | 판정 |
|---|---|
| 피질-시상, 해마-살리언스, 시상하부-뇌간, 자율신경/내분비/면역 축이 뇌 상태를 층별로 나눈다는 설명 | `supported` |
| 이 층들을 `p_r`, `s_r`, `q_n`로 압축해 CE와 연결하는 것 | `bridge` |
| 동일한 위치에서 multimodal data로 `p_r`, `s_r`, `q_n`, `w_r`를 모두 닫아 암을 예측하는 것 | `hypothesis` |

---

## 9. AGI 문서에 바로 적용할 규칙

1. 유지 가능한 주장
- sparse activity는 실제 뇌와 잘 맞는다
- sleep는 offline renormalization 기능을 가진다
- STDP + eligibility + third factor는 생물학적으로 강하다
- alpha/beta/gamma의 기능적 분업은 설계 힌트로 쓸 수 있다

2. 한 단계 내려야 하는 주장
- `4.87% = 모든 영역의 직접 측정값`
- `26.2% = 구조 유지가 직접 측정된 값`
- `Wake-NREM-REM = 부트스트랩 반복의 엄밀한 수식`
- `dopamine = 고정점 이탈 오차`
- `환각률 <= 4.87%`, `의식 깊이 = 1 - ||p-p^*||`

3. 앞으로의 원칙

$$
\boxed{\text{뇌 실험값으로 먼저 닫히지 않은 수식은, AGI 성능 정리로 올리지 않는다.}}
$$

---

## 10. 지금 바로 걸 수 있는 예측

현재 근거 수준에서, AGI 문서에 올려도 되는 예측은 다음처럼 제한하는 것이 맞다.

| 예측 항목 | 예측값 | 등급 | 이유 |
|---|---|---|---|
| 수면 루프 잔차 감쇠 | (I1) 중심값: 1적용(1.6밤) `15.5%`, 2적용(3.2밤) `2.4%`, 3적용(4.8밤) `0.37%`. 야간 단위: 1밤 `31%`, 2밤 `9.6%`, 3밤 `3.0%` | `bridge` | $\rho=D_{\text{eff}}\varepsilon^2=0.155$ (정의 1.3의 최소 closure 기준, per application), $\rho_{\text{night}}=\rho^{1/1.6}\approx 0.31$ (per night). exact number는 (I1) 민감도에 의존 |
| dense 초기화 후 활성 비율 | (I1) 중심값: `33.3% -> 9.28% -> 5.55% -> 4.98%` | `bridge` | 최소 closure $p_{n+1}=p^*+\rho(p_n-p^*)$에서의 계산. $p^*$와 $\rho$의 구체적 수치는 (I1) 의존 |
| Top-k 최적점 | 중심 `4.87%`, (I1) 민감도 `4.35%-5.45%`, 실용 대역 `3%-7%` | `bridge` | sparse neural regime + 고정점 자기일관성. exact center는 (I1) 의존 |
| 수면 유무 비교 | sleep: bounded residual, wake-only: 선형 drift | `supported/bridge` 혼합 | 수면 항상성 데이터는 강하고, 반복식 매칭은 bridge |

여기서 exact 숫자는 모두 **(I1) 아래의 중심값(center value)**이다. `D_{\text{eff}} = 3 + \delta_{\text{mix}}`에서 $\delta_{\text{mix}}$가 $\pm 0.10$만 흔들려도, 최소 CE 모델 기준으로

$$
x_a : 4.35\% \sim 5.45\%, \qquad
\rho : 0.1425 \sim 0.1676,
$$

$$
x_s : 25.25\% \sim 26.89\%, \qquad
x_b : 68.76\% \sim 69.30\%
$$

로 이동한다. 따라서 위 표의 정량 예측은 "딱 고정된 보편 상수"라기보다, (I1)이 닫힐 때의 중심값으로 읽는 편이 안전하다.

반대로 아직 금지해야 하는 예측은:

- `환각률 <= 4.87%`
- `Truthfulness >= 95.13%`
- `의식 깊이 = 1 - ||p-p^*||`
- `dopamine = ||p-p^*||` 또는 그 미분

즉 앞으로의 원칙은 다음 한 줄로 요약된다.

$$
\boxed{\text{구간 포함 + 최소 반복식으로 닫히는 예측만 올리고, 태스크 점수 hard bound는 아직 올리지 않는다.}}
$$

---

## 11. 제어 붕괴와 종양: glioblastoma 비교

> 범위: 이 절은 `암 일반론`이 아니라, 현재 가장 직접적인 비교가 가능한 **뇌종양, 특히 glioblastoma (GBM)** 를 대상으로 한다.

### 11.1 직접 근거

1. GBM은 가장 공격적이고 침윤적인 원발성 뇌종양으로, 표준치료 후에도 예후가 나쁘다.
   - median survival `12-15개월`
   - 고전적 조직학: necrosis, microvascular proliferation, diffuse infiltration
   - 근거: [Glioblastoma: An Update in Pathology, Molecular Mechanisms and Biomarkers](https://www.mdpi.com/1422-0067/25/5/3040)

2. GBM의 핵심은 단순 증식이 아니라 **제어 경로 붕괴**다.
   - WHO CNS5 수준의 주요 분자 지표: TERT promoter mutation, EGFR amplification, combined `+7/-10`, CDKN2A/B homozygous deletion
   - 반복적으로 깨지는 3대 경로: `RTK`, `TP53`, `RB`
   - `TP53/MDM2/CDKN2A` 경로는 GBM 환자 `84%`, GBM cell line `94%`에서 deregulation
   - `CDKN2A` homozygous deletion은 전체 GBM의 `22-35%`, IDH-wildtype GBM의 `~58%`
   - 근거: [Glioblastoma: An Update in Pathology, Molecular Mechanisms and Biomarkers](https://www.mdpi.com/1422-0067/25/5/3040)

3. 신경 활동 자체가 GBM 성장과 침윤을 직접 밀어 올린다.
   - neuronal activity-dependent `NLGN3`와 `BDNF`가 glioma growth signal로 작동한다
   - neuron-glioma synapse는 AMPA-glutamatergic이며, peritumoral neuron이 tumor cell proliferation/invasion을 직접 밀어 올린다
   - glioma-infiltrated cortex에서는 hyperexcitability, seizure tendency, inhibitory interneuron loss가 반복 보고된다
   - 근거: [Glioma progression through synaptic activity](https://www.nature.com/articles/s41582-019-0290-1), [AMPA Receptor Modulation in the Treatment of High-Grade Glioma: Translating Good Science into Better Outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945080/), [Peritumoral brain zone in glioblastoma: biological, clinical and mechanical features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10937439/)

4. 공간 전사체 수준에서는 세포 상태와 niche가 분리된다.
   - paired snRNA-seq + spatial transcriptomics (`3` 환자)에서 `OPC-like`, `NPC-like`, `AC-like`, `MES-like` 상태가 공간적으로 분리
   - 예: AC-like vs OPC-like segregation adjusted p value `1.50e-44`
   - 예: OPC-like vs NPC-like segregation adjusted p values `7.65e-18`, `2.99e-21`
   - perinecrotic region은 일반 GBM 미세환경보다 더 immunosuppressive
   - perivascular region은 더 pro-inflammatory
   - necrotic/perinecrotic 쪽은 glycolysis, ferroptosis, unfolded protein response가 높고, perivascular 쪽은 oxidative phosphorylation이 높다
   - 근거: [Spatial transcriptomics reveals segregation of tumor cell states in glioblastoma and marked immunosuppression within the perinecrotic niche](https://link.springer.com/article/10.1186/s40478-024-01769-0)

5. GBM은 기계적으로도 균질한 덩어리가 아니라, 주변 조직까지 번지는 이질적 장(field)이다.
   - multifrequency MRE, `22` 환자
   - tumor `|G*| = 1.32 \pm 0.26` kPa, healthy tissue `1.54 \pm 0.27` kPa (`P = 0.001`)
   - tumor phase angle `\phi = 0.37 \pm 0.08`, healthy tissue `0.58 \pm 0.07`
   - `5/22` tumors는 오히려 건강조직보다 더 stiff
   - perifocal region의 `|G*|`는 tumor와 상관 (`R = 0.571`, `P = 0.0055`)
   - 근거: [High-Resolution Mechanical Imaging of Glioblastoma by Multifrequency Magnetic Resonance Elastography](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110588)

6. 재발은 핵심 덩어리만의 문제가 아니라 경계 영역(PBZ, peritumoral brain zone) 문제다.
   - recurrence의 `80% 이상`이 resection cavity edge에서 발생
   - PBZ는 MRI상 비정상 조영이 없더라도 분자적/세포적 변화가 존재
   - review 요약에서는 MRI상 정상처럼 보이는 PBZ sample의 거의 `1/3`에서 infiltrative tumor cells가 잡힌다
   - 근거: [Peritumoral brain zone in glioblastoma: biological, clinical and mechanical features](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2024.1347877/full), [Characterizing the peritumoral brain zone in glioblastoma: a multidisciplinary analysis](https://link.springer.com/article/10.1007/s11060-014-1695-8)

7. 침윤 front는 core와 다른 프로그램을 가진다.
   - single-cell RNA-seq (`3,589` cells, `4` patients)에서 core와 peritumoral tissue를 함께 보면, infiltrating neoplastic cells에 공통된 signature가 잡힌다
   - 즉 암세포는 "덩어리 내부"와 "이동 경계"에서 같은 상태가 아니다
   - 근거: [Single-Cell RNA-Seq Analysis of Infiltrating Neoplastic Cells at the Migrating Front of Human Glioblastoma](https://www.cell.com/cell-reports/fulltext/S2211-1247(17)31462-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2211124717314626%3Fshowall%3Dtrue)

8. GBM은 perivascular / glymphatic 축도 무너뜨린다.
   - tumor mass effect와 perivascular invasion으로 astrocytic endfeet와 `AQP4` polarity가 흐트러지고, CSF-ISF exchange와 antigen drainage가 저하된다
   - 이는 fluid/mechanical homeostasis와 immune surveillance를 동시에 약화시켜, `q_sleep/q_met -> w_{\text{immune}}, w_{\text{mech}}` 브리지를 제공한다
   - 근거: [The Glymphatic–Immune Axis in Glioblastoma: Mechanistic Insights and Translational Opportunities](https://www.mdpi.com/1422-0067/27/2/928)

### 11.2 최소 해석

현재 데이터로 가장 안전하게 말할 수 있는 것은, GBM이 단순한 "세포 내부 변이의 합"이 아니라

$$
\boxed{\text{세포 상태, 조직 위상(niche, vessel, hypoxia, ECM, immune control), 주변 신경회로 활동의 정합이 함께 무너진 상태}}
$$

라는 점이다.

특히 GBM은 다른 고형암보다 `주변 신경회로 활동`이 직접 종양 입력으로 작동한다는 점이 더 강하다. 따라서 최소 실패 모드 `z`는 유지하되, `V_ctx/PBZ` 쪽 과흥분을 별도 forcing으로 둔다.

이를 CE 쪽에서 바로 `p=(x_a,x_s,x_b)`에 억지로 넣기보다는, 우선 다음의 **실패 모드 상태벡터**로 보는 편이 안전하다.

$$
z(t) = \big(z_{\text{prolif}},\; z_{\text{topo}},\; z_{\text{suppress}}\big)
$$

여기서
- $z_{\text{prolif}}$: 증식/침윤 프로그램 burden
- $z_{\text{topo}}$: 혈관, ECM, niche, 기계적 경계의 재배치 정도
- $z_{\text{suppress}}$: 면역억제/저산소/대사 스트레스 burden

직접 신경 구동항은

$$
\nu_n = \text{cortical hyperexcitability / neuron-glioma synaptic drive}
$$

처럼 둔다.

정상 조직 후보 동역학은

$$
z_{n+1} = A_{\text{healthy}} z_n + u_n, \qquad \rho(A_{\text{healthy}}) < 1
$$

처럼 local contraction을 가지는 반면, GBM 후보는

$$
z_{n+1} = A_{\text{GBM}} z_n + c\,\nu_n + u_n, \qquad \rho(A_{\text{GBM}}) \gtrsim 1,\ c \ge 0
$$

또는 별도 attractor $z_{\text{GBM}}^*$를 갖는 경우로 읽을 수 있다.

이 식은 아직 `bridge/hypothesis`다. 하지만 현재 실험 데이터는 적어도 "국소 제어 붕괴 + 직접 neural drive + 공간 niche 재배치 + 경계 밖 drift"가 동시에 관측된다는 점을 강하게 보여 준다.

### 11.3 CE 대응

| 현상 | 실험 측 사실 | CE 해석 | 판정 |
|---|---|---|---|
| RTK / TP53 / RB 경로 붕괴 | 증식, apoptosis, cell-cycle 제어 경로 반복 붕괴 | 국소 제어기의 붕괴 | `supported` |
| neuronal hyperexcitability / neuron-glioma synapse | `NLGN3`/`BDNF`, AMPA synapse, infiltrated cortex hyperexcitability | `V_ctx -> w_r` 직접 국소 forcing, 또는 `x_a` 과잉의 종양측 입력 | `bridge` |
| spatial niche segregation | perivascular / perinecrotic / invasive front가 다른 세포 상태와 경로를 가짐 | 세포 상태와 조직 위상의 정합/불정합 | `bridge` |
| perinecrotic immunosuppression | 저산소 niche가 더 immunosuppressive | pruning/제거 실패가 국소적으로 고정됨 | `bridge` |
| perifocal mechanical drift | tumor 주변 조직의 물성도 tumor와 연속적으로 연결 | 병적 장이 core 밖으로 확장 | `bridge` |
| PBZ recurrence | 재발의 다수가 경계에서 발생 | 보이는 core보다 넓은 병적 경계 | `bridge` |
| glymphatic / perivascular breakdown | `AQP4` mislocalization, clearance and antigen drainage 저하 | `q_sleep/q_met -> w_{\text{immune}}, w_{\text{mech}}` bridge | `bridge` |
| 암 = 병적 고정점 | 위 모든 현상이 하나의 pathological attractor를 형성 | $z_{\text{GBM}}^*$ | `hypothesis` |

### 11.4 해석

- 이 비교는 "뇌 구조만 보고 암 원인을 풀었다"는 뜻이 아니다.
- 대신 GBM에서는 실제로 **제어 경로 붕괴**, **직접 neural drive**, **공간 niche 분리**, **면역억제의 국소화**, **기계적/분자적 경계 확장**이 함께 관측된다.
- 특히 GBM은 `V_ctx`가 단순한 배경 설명이 아니라, PBZ 과흥분과 neuron-glioma synapse를 통해 종양 동역학을 직접 미는 입력이 될 수 있다는 점에서 뇌-암 결합의 첫 검증장으로 가장 유리하다.
- 따라서 "암은 세포가 조직 위상과 안 맞아지면서 통제에서 벗어난 상태"라는 문장은, 적어도 GBM에 대해서는 공상보다 한 단계 위의 `bridge` 문장으로 올릴 수 있다.
- 반대로 "암은 CE 고정점에서 수학적으로 반드시 유도된다" 또는 "모든 암종이 동일한 병적 상수로 닫힌다"는 문장은 현재 단계에서 금지다.

### 11.5 지금 바로 걸 수 있는 예측과 게이트

| 게이트 | 점검 항목 | 통과 기준 |
|---|---|---|
| `G-C1` | 공간 niche 분리 | spatial omics에서 perivascular / hypoxic / invasive front가 재현 가능하게 분리 |
| `G-C2` | 제어 붕괴 | RTK / TP53 / RB / CDKN2A-B 계열 이상이 종양 상태와 강하게 연결 |
| `G-C3` | 직접 neural drive | PBZ / infiltrated cortex hyperexcitability, AMPA synapse, `NLGN3` 신호가 tumor burden과 함께 움직임 |
| `G-C4` | 경계 drift | PBZ 또는 perifocal zone이 healthy와 동일하지 않고 중간 상태를 보임 |
| `G-C5` | 재발 예측력 | edge / PBZ signature가 core 평균보다 recurrence를 더 잘 설명 |
| `G-C6` | 기계적 장 변화 | MRE / ECM / stiffness 지표가 tumor core 밖에서도 변화 |
| `G-C7` | glymphatic / perivascular 붕괴 | `AQP4` / perivascular / clearance marker가 immune / mech mismatch와 방향 일치 |

### 11.6 실패 시 해석 규칙

1. 유전자 이상은 강하지만 spatial niche 정보가 별 도움을 주지 않으면, "위상-세포 정합 붕괴" 문장은 낮추고 `세포 내부 제어 붕괴`까지만 유지한다.
2. hyperexcitability / `NLGN3` / AMPA 신호가 tumor field와 무관하면, `V_ctx -> w_r` 직접축은 낮추고 "신경계는 배경 조절자" 수준으로 내린다.
3. PBZ가 기계적/분자적으로 healthy와 구분되지 않으면, "병적 장이 core 밖으로 확장된다"는 문장은 내린다.
4. glymphatic / perivascular marker가 immune / mech mismatch와 연결되지 않으면, 수면축은 systemic burden까지만 유지하고 GBM-specific bridge는 내린다.
5. niche는 분리되지만 면역억제나 재발과 연결되지 않으면, 공간 분리는 인정하되 `통제 이탈의 핵심 원인`으로는 승격하지 않는다.
6. 다른 암종에서 이 패턴이 반복되지 않으면, 이 절은 GBM-특이 현상론으로 제한한다.

---

## 12. 범암 비교: GBM 밖에서도 같은 패턴이 반복되는가

> 목표: "암 = 위상-세포 정합 붕괴"를 일반 명제로 올리기 전에, 서로 다른 고형암에서 같은 구조가 **반복**되는지 확인한다.

### 12.1 직접 근거

1. hypoxia는 GBM 특수 현상이 아니라, **대부분의 solid tumor에서 immune exclusion의 상위 원인 후보**로 취급된다.
   - hypoxia는 병적 혈관, HIF 신호, 면역억제성 사이토카인, 림프구 침투 저하를 함께 밀어 올린다
   - 근거: [Hypoxia and the phenomenon of immune exclusion](https://link.springer.com/article/10.1186/s12967-020-02667-4)

2. pancreatic ductal adenocarcinoma (PDAC)는 전형적인 `immune-excluded`, `stroma-dominant` 암이다.
   - desmoplastic stroma가 종양 조직의 체적 majority를 이룸
   - CAF와 TAM이 지배적이며, low effector T-cell infiltration을 보이는 `cold` microenvironment
   - spatial transcriptomics에서는 hypoxic tumor front, basal-like cancer state, myCAF proximity, CXCR4-CXCL12 축과 immune exclusion이 함께 관찰된다
   - 근거: [Tumor Microenvironment in Pancreatic Cancer Pathogenesis and Therapeutic Resistance](https://www.annualreviews.org/content/journals/10.1146/annurev-pathmechdis-031621-024600), [Spatial Transcriptomics in Pancreatic Cancer: Current Insights and Future Directions](https://www.jdcr.org/journal/view.html?uid=397&vmd=Full), [Hypoxic microenvironment induced spatial transcriptome changes in pancreatic cancer](https://www.cancerbiomed.org/content/18/2/616)

3. colorectal cancer (CRC)에서는 invasive front가 병리적으로 이미 정량화되어 있다.
   - tumor budding 정의: invasive front의 single cell 또는 `4개 이하` cluster
   - ITBCC scoring: `Bd1 = 0-4`, `Bd2 = 5-9`, `Bd3 >= 10` buds / hotspot `0.785 mm^2`
   - 최근 spatial/single-cell 결과에서는 budding cells가 CAF와 직접 접촉하며 pro-invasive gene program을 보인다
   - 근거: [Recommendations for reporting tumor budding in colorectal cancer based on the International Tumor Budding Consensus Conference (ITBCC) 2016](https://www.nature.com/articles/modpathol201746), [Cancer-associated fibroblasts shape the formation of budding cancer cells at the invasive front of human colorectal cancer](https://www.nature.com/articles/s42003-025-08799-x?error=cookies_not_supported&code=afac014c-c5f0-4338-a372-ef3461af139e)

4. breast cancer에서는 공간 전사체가 아니더라도 **기계적 장 재배치**가 매우 강하다.
   - normal breast tissue stiffness `~0.2` kPa
   - breast cancer tissue stiffness `>4` kPa
   - adjacent matrix도 distant normal tissue보다 훨씬 stiff
   - 근거: [The role of matrix stiffness in breast cancer progression: a review](https://pdfs.semanticscholar.org/72f7/02f5c4f56ce7e9468b1a218085519f7c2410.pdf)

### 12.2 반복되는 공통 패턴

위 암종들을 나란히 놓으면, 공통 축은 다음 네 가지다.

| 암종 | 내부 제어 붕괴 | 공간 niche / 경계 | 기계적-구조 장 | 면역 배제 |
|---|---|---|---|---|
| GBM | RTK / TP53 / RB | perivascular / perinecrotic / PBZ / invasive front | MRE, perifocal drift | perinecrotic immunosuppression |
| PDAC | KRAS 중심 증식축 + stromal regulation 붕괴 | hypoxic front / basal-like / CAF niche | desmoplastic stroma | cold tumor, CAF/TAM-mediated exclusion |
| CRC | invasion/EMT program | invasive front / tumor budding hotspot | CAF-contact remodeling | epithelial-immune niche 재배치 |
| breast | growth and EMT-related mechanotransduction | tumor vs adjacent matrix 경계 | `0.2` kPa -> `>4` kPa stiffness shift | stiffness-linked immune escape 후보 |

즉 공통점은 단순히 "변이가 있다"가 아니라,

$$
\boxed{\text{제어 경로 붕괴가 공간 niche 재편, 경계 drift, 면역 배제, 기계적 장 재배치와 함께 나타난다}}
$$

는 점이다.

### 12.3 강한 작업가설

이 문서에서는 당분간 다음 문장을 **범암 기본모델**로 채택한다.

$$
\boxed{\text{암은 세포 내부 제어 붕괴가 공간 niche 재편, 기계적 장 재배치, 면역 배제와 결합해 고정되는 질환이다.}}
$$

즉 앞으로는 "혹시 그럴 수도 있다"가 아니라, 일단 **이 명제가 맞다고 두고** 실험으로 죽이거나 살린다.

이를 위해 범암 상태벡터를

$$
w(t)=\big(w_{\text{cell}},\;w_{\text{niche}},\;w_{\text{mech}},\;w_{\text{immune}}\big)
$$

로 둔다.

- $w_{\text{cell}}$: oncogenic signaling / cell-cycle / apoptosis failure
- $w_{\text{niche}}$: spatial segregation, invasive front, perivascular or hypoxic niche burden
- $w_{\text{mech}}$: ECM, stiffness, vascular remodeling, boundary extension
- $w_{\text{immune}}$: immune exclusion, suppressive myeloid/fibroblast programs

### 12.3.1 정리 1: 제어 붕괴 임계 조건

암종과 무관하게, 영역별 상태벡터가 다음의 최소 양의 결합계로 근사된다고 두자.

$$
w_{n+1}=A\,w_n+u,\qquad
A\in\mathbb{R}_{\ge 0}^{4\times 4},\quad
u\in\mathbb{R}_{\ge 0}^{4}
$$

여기서 $A$는
- 세포 제어 붕괴가 niche/ECM/immune 축으로 퍼지는 결합
- niche/ECM/immune 축이 다시 세포 증식/침윤을 밀어 올리는 피드백

을 나타낸다.

그러면 다음이 성립한다.

1. $\rho(A)<1$이면 유일한 비음수 고정점

$$
w^*=(I-A)^{-1}u
$$

가 존재하고, 모든 초기값에서 수렴한다.

2. $\rho(A)>1$이면 선형계는 복원형 항상성을 유지할 수 없고, 적어도 어떤 양의 방향에서는 이탈이 증폭된다. 실제 종양처럼 상태가 유한하게 보인다면, 그것은 선형 복원평형이 아니라 **비선형 포화가 잘라낸 병적 상태**다.

증명:

1. $\rho(A)<1$이면 Neumann 급수로

$$
(I-A)^{-1}=\sum_{k=0}^{\infty}A^k
$$

가 수렴하므로 $w^*$가 존재한다. 또한

$$
w_n=A^n w_0+\sum_{k=0}^{n-1}A^k u
$$

이고, $\rho(A)<1$이면 $A^n\to 0$이므로 $w_n\to w^*$.

2. $\rho(A)>1$이면 Perron-Frobenius 정리에 의해 양의 cone 안에 증폭 방향이 존재한다. 따라서 복원형 선형 항상성은 성립할 수 없다. 관측되는 bounded tumor state는 반드시 saturation, competition, resource limit 같은 비선형성 위에서 잘린 상태여야 한다. $\square$

해석:

- 정상 조직은 `\rho(A)<1`인 복원계다.
- 공격적 암은 최소한 일부 축에서 `\rho(A)\ge 1`이 되어야 유지된다.
- 즉 "암은 제어 붕괴가 서로 물고 올라가는 결합계"라는 문장은 단순 비유가 아니라, **양의 결합계의 임계 조건**으로 닫힌다.

### 12.3.2 이전 버전(반증됨): 경계 우선 붕괴 가설

이 절은 과거의 `edge/front > core` 가설이 어떤 충분조건에서 나왔는지 남긴 기록이다.  
현재 `34`샘플 고정모델에서는 이 가설을 메인라인으로 사용하지 않는다.

이제 `core`와 `edge/front` 두 영역이 같은 결합행렬 $A$를 공유하되, 외부 forcing이 다르다고 두자.

$$
w_{n+1}^{(c)} = A w_n^{(c)} + u^{(c)},\qquad
w_{n+1}^{(e)} = A w_n^{(e)} + u^{(e)}
$$

가정:

1. $A\ge 0$
2. $\rho(A)<1$
3. $u^{(e)} \ge u^{(c)}$ 성분별
4. 적어도 한 성분에서는 strict inequality, 즉 $u^{(e)} \neq u^{(c)}$
5. $A$는 irreducible이라서 축 간 상호작용이 실제로 연결되어 있다

그러면

$$
w_*^{(e)} - w_*^{(c)}
=
(I-A)^{-1}\big(u^{(e)}-u^{(c)}\big)
$$

이고, $(I-A)^{-1}$는 양의 행렬이므로

$$
w_*^{(e)} > w_*^{(c)}
$$

가 성분별로 성립한다.

이제 양의 가중 functional

$$
M(w)=\beta^\top w,\qquad \beta_i>0
$$

를 정의하면,

$$
\boxed{M_{\text{edge/front}} > M_{\text{core}}}
$$

가 자동으로 따라온다.

증명:

$\rho(A)<1$이므로 두 영역의 고정점은 각각

$$
w_*^{(c)}=(I-A)^{-1}u^{(c)},\qquad
w_*^{(e)}=(I-A)^{-1}u^{(e)}
$$

이다. 차를 빼면 위 식이 나오고, irreducible nonnegative matrix의 경우 $(I-A)^{-1}$는 양의 inverse를 가진다. 따라서 $u^{(e)}-u^{(c)}$의 한 성분이라도 양수이면, 모든 성분이 양 방향으로 밀린다. 마지막 부등식은 $\beta_i>0$인 선형 functional에 바로 따른다. $\square$

해석:

- edge/front에서 hypoxia, stromal barrier, immune exclusion, invasion program이 더 강하면,
- 그 차이가 실제로 다른 축으로 전파되어
- 최종 mismatch 점수도 edge/front가 core보다 커질 수밖에 없다.

즉 이 절의 예측은

$$
M_{\text{edge/front}} > M_{\text{core avg}}
$$

처럼 쓸 수 있다. 다만 이것은 **양의 결합계 + 더 큰 경계 forcing**이라는 특수조건에서만 나오는 결과이고, 현재 실측 메인라인은 `M_{\text{eff}}(h^\dagger)`와 `stromal/hypoxic`, `mech/niche` 쪽으로 옮겨갔다.

### 12.3.3 따름정리: 재발 구역 우위

재발 구역 `r`와 비재발 구역 `n`에 대해 같은 논리를 쓰면,

$$
u^{(r)} > u^{(n)}
\quad\Longrightarrow\quad
M_{\text{recurrence zone}} > M_{\text{non-recurrence zone}}
$$

이다.

---

### 12.3.4 뇌-암 최소 결합식

> 목적: 지금까지는 `뇌의 3분배 상태`와 `암의 mismatch 상태`를 각각 따로 썼다. 이 절은 둘이 실제로 어떻게 연결되는지에 대한 **최소 결합식**만 세운다. 강한 범암 상수 주장이 아니라, 데이터가 들어오면 바로 적합할 수 있는 형태로만 닫는다.

#### 12.3.4.1 지역별 상태변수

뇌 영역 또는 공간 spot/voxel $r$마다 상태를 두되, 이제부터 `r`는 막연한 voxel이 아니라 brain graph $G_{\text{brain}}=(V_{\text{brain}},E_{\text{brain}})$의 정점으로 읽는다.

최소 분할은

$$
V_{\text{brain}}
=
V_{\text{ctx}}
\sqcup
V_{\text{thal}}
\sqcup
V_{\text{hip}}
\sqcup
V_{\text{sal}}
\sqcup
V_{\text{hyp}}
\sqcup
V_{\text{stem}}
\sqcup
V_{\text{aut}}
$$

로 둔다.

1. 뇌의 에너지 3분배 상태:

$$
p_r(t) = \big(x_{a,r}(t),\; x_{s,r}(t),\; x_{b,r}(t)\big) \in \Delta^2
$$

여기서

$$
\Delta^2 = \left\{(x_a,x_s,x_b)\in[0,1]^3 \,\middle|\, x_a+x_s+x_b=1\right\}
$$

이고 기준점은

$$
p^* = (x_a^*,x_s^*,x_b^*) = (0.0487,\;0.2623,\;0.6891)
$$

이다.

2. 뇌-몸 조절 상태:

$$
q_n
=
\big(
q_{\text{sleep},n},
q_{\text{arousal},n},
q_{\text{aut},n},
q_{\text{endo},n},
q_{\text{immune},n},
q_{\text{met},n}
\big)
$$

건강한 set-point는 $q^*$로 둔다. 이 벡터는 수면 부채, hyperarousal, sympathetic-vagal balance, endocrine stress, inflammatory tone, metabolic reserve를 묶는다.

3. 종양 mismatch 상태:

$$
w_r(t)=\big(w_{\text{cell},r},\;w_{\text{niche},r},\;w_{\text{mech},r},\;w_{\text{immune},r}\big)\in\mathbb{R}_{\ge0}^4
$$

스칼라 병적 burden은

$$
m_r(t)=\lambda^\top w_r(t), \qquad \lambda_i>0
$$

로 둔다.

#### 12.3.4.2 뇌 상태에서 종양 취약도로 가는 결합

암 쪽으로 넘어가는 최소 결합은 "구조/배경 reserve 부족 + 과활성 + 뇌-몸 조절 부하"를 하나의 취약도 변수로 압축하는 것이다.

$$
\ell_r(n) = d_r^\top \big(q_n-q^*\big)_+,
\qquad
d_r\ge 0
$$

$$
s_r(n)
=
\eta_a\big(x_{a,r}(n)-x_a^*\big)_+
\;+\;
\eta_s\big(x_s^*-x_{s,r}(n)\big)_+
\;+\;
\eta_b\big(x_b^*-x_{b,r}(n)\big)_+
\;+\;
\eta_q \ell_r(n)
$$

여기서 $(u)_+ = \max(u,0)$이고, 해석은 다음과 같다.

- $x_a > x_a^*$: 병적 과흥분 또는 응급성 대사 부담
- $x_s < x_s^*$: 구조 유지/복원 여력 감소
- $x_b < x_b^*$: 배경 항상성/완충 여력 감소
- $\ell_r > 0$: 수면 부채, hyperarousal, 자율신경 불균형, 내분비/면역/대사 부하가 해당 영역에 투사됨

그러면 종양 상태의 최소 결합식은

$$
w_{r,n+1} = A_r w_{r,n} + b_r\, s_r(n) + u_{r,n},
\qquad
A_r\ge0,\; b_r\ge0
$$

로 둔다.

이 식의 의미:

- $A_r w_{r,n}$: 종양 내부의 자기증폭, niche-ECM-immune 재결합
- $b_r s_r(n)$: 해당 영역의 항상성 reserve 저하와 body-loop load가 병적 상태를 더 밀어 올리는 항
- $u_{r,n}$: 외부 forcing (hypoxia, mutation hit, vessel failure, therapy pressure 등)

#### 12.3.4.3 종양 상태에서 뇌 3분배로 돌아오는 역결합

반대로 종양 burden은 국소 뇌 상태를 기준점에서 밀어내고, 그 이탈은 brain graph를 따라 퍼지거나 완충된다. 먼저

$$
\Delta_G p_{r,n}
=
\sum_{s:(s,r)\in E_{\text{brain}}}
a_{rs}\big(p_{s,n}-p_{r,n}\big),
\qquad
a_{rs}\ge 0
$$

로 둔다.

가장 단순한 식은

$$
\tilde p_{r,n+1}
=
p^*
\;+\;
\rho\big(p_{r,n}-p^*\big)
\;+\;
\gamma_p \Delta_G p_{r,n}
\;+\;
H_r(q_n-q^*)
\;+\;
C_r w_{r,n}
\;+\;
\xi_{r,n},
\qquad
\rho=0.155
$$

$$
p_{r,n+1} = \Pi_{\Delta^2}\!\big(\tilde p_{r,n+1}\big)
$$

이다. 여기서 $\Pi_{\Delta^2}$는 합이 1이 되도록 하는 simplex projection이다.

보존 조건은

$$
\mathbf{1}^\top C_r = 0,
\qquad
\mathbf{1}^\top H_r = 0
$$

로 둔다. 즉 종양이 에너지 분배를 바꾸더라도 성분 합은 유지된다.

최소 부호 조건은

$$
\big(C_r w\big)_a \ge 0,\qquad
\big(C_r w\big)_s \le 0,\qquad
\big(C_r w\big)_b \le 0
\quad
(\forall\, w\ge0)
$$

이다. 해석은 "종양 burden이 커질수록 비정상 활성 부담은 커지고, 구조/배경 reserve는 줄어드는 방향"이다. 또한

- $\gamma_p \Delta_G p_{r,n}$: 인접 피질, 시상 relay, 해마 replay, 시상하부-뇌간 제어를 통한 spread/buffering
- $H_r(q_n-q^*)$: 수면-각성, 자율신경, 내분비/면역/대사 부하가 지역 3분배에 주는 느린 forcing
- $C_r w_{r,n}$: 종양 burden의 국소 역작용

편차를

$$
e_{r,n}:=p_{r,n}-p^*
$$

로 두면

$$
e_{r,n+1}
=
\rho e_{r,n}
+ \gamma_p \Delta_G e_{r,n}
+ H_r(q_n-q^*)
+ C_r w_{r,n}
+ \xi_{r,n}
$$

이고 따라서

$$
\|e_{r,n+1}\|
\le
\rho \|e_{r,n}\|
+
\gamma_p \|\Delta_G e_{r,n}\|
+
\|H_r\|\,\|q_n-q^*\|
+
\|C_r\|\,\|w_{r,n}\|
+
\|\xi_{r,n}\|
$$

를 얻는다.

즉 종양 burden이 작으면 수면/항상성 수축이 다시 $p^*$ 근처로 끌어오지만, burden이 커지면 그만큼 뇌의 3분배 이탈도 커진다.

#### 12.3.4.4 공동 안정성 조건

$s_r(n)$은 $e_{r,n}$과 $q_n-q^*$에 대해 Lipschitz다. 예를 들어

$$
\kappa_p := \sqrt{\eta_a^2+\eta_s^2+\eta_b^2},
\qquad
\kappa_q := \eta_q \|d_r\|
$$

를 두면

$$
s_r(n)\le \kappa_p \|e_{r,n}\| + \kappa_q \|q_n-q^*\|
$$

이다.

따라서

$$
\|w_{r,n+1}\|
\le
\|A_r\|\,\|w_{r,n}\|
+
\|b_r\|\,\kappa_p\,\|e_{r,n}\|
+
\|b_r\|\,\kappa_q\,\|q_n-q^*\|
+
\|u_{r,n}\|
$$

또한

$$
g_{r,n} := \gamma_p \Delta_G e_{r,n} + H_r(q_n-q^*)
$$

로 두면

$$
\|e_{r,n+1}\|
\le
\rho \|e_{r,n}\|
+
\|C_r\|\,\|w_{r,n}\|
+
\|g_{r,n}\|
+
\|\xi_{r,n}\|
$$

이고, 두 노름을 합친 상태벡터

$$
y_{r,n}
:=
\begin{pmatrix}
\|e_{r,n}\|\\
\|w_{r,n}\|
\end{pmatrix}
$$

를 두면

$$
y_{r,n+1}
\;\lesssim\;
K_r\, y_{r,n}
+
\begin{pmatrix}
\|g_{r,n}\|+\|\xi_{r,n}\|\\
\|u_{r,n}\|+\|b_r\|\,\kappa_q\,\|q_n-q^*\|
\end{pmatrix},
\qquad
K_r=
\begin{pmatrix}
\rho & \|C_r\|\\
\kappa_p\|b_r\| & \|A_r\|
\end{pmatrix}
$$

로 쓸 수 있다.

따라서 최소 안정성 기준은

$$
\boxed{\rho(K_r)<1}
$$

이다.

해석:

- $\rho(K_r)<1$: 뇌 항상성과 국소 제어가 병적 burden을 눌러 다시 복원권으로 들어온다
- $\rho(K_r)\ge1$: 뇌 3분배 이탈과 종양 mismatch가 서로 밀어 올리는 결합계가 된다

여기서 `g_{r,n}`가 바로 graph coupling과 sleep-autonomic-body forcing을 묶은 항이다. 즉 기존의 `\rho(A_{\text{tumor}})\ge1` 조건은 종양 단독 기준이고, 이 절의 `\rho(K_r)\ge1`은 **뇌-종양 결합 전체의 임계 조건**이다.

#### 12.3.4.5 경계-코어 예측의 공간형 정식

실제 공간 데이터에서는 경계로부터의 hop 거리 $h$에 따라 shell을 나눈다.

$$
S_h = \{r : \operatorname{dist}(r,\partial\Omega_{\text{tumor}})=h\}
$$

각 shell 평균 상태는

$$
\bar w_h = \frac{1}{|S_h|}\sum_{r\in S_h} w_r,
\qquad
\bar p_h = \frac{1}{|S_h|}\sum_{r\in S_h} p_r
$$

이고, shell mismatch 점수의 기본형은

$$
M(h) = \beta^\top \bar w_h, \qquad \beta_i>0
$$

로 둔다.

하지만 현재 누적 실측은 축의 기여가 균등하지 않음을 시사한다. 따라서 **현 시점의 경험적 유효 burden**은

$$
\boxed{
M_{\text{eff}}(h)
=
\lambda_c \bar w_{h,\text{cell}}
+ \lambda_n \bar w_{h,\text{niche}}
+ \lambda_m \bar w_{h,\text{mech}}
+ \lambda_i \bar w_{h,\text{immune}},
\qquad
\lambda_n+\lambda_m > \lambda_c+\lambda_i
}
$$

로 다시 두는 편이 현재 데이터와 더 잘 맞는다.

현재 `34`샘플 코호트에서 정밀도를 가장 안정적으로 올린 고정 가중은

$$
\boxed{
(\lambda_c,\lambda_n,\lambda_m,\lambda_i)
=
(0.15,\;0.15,\;0.45,\;0.25)
}
$$

이다.  
즉 `cell`과 `niche`는 낮게, `mech`는 가장 크게, `immune`는 중간 정도로 둔다.

이 값은 `0.05` 간격 simplex 탐색에서 전체 적합도를 가장 많이 올린 해이고, leave-one-out에서도 선택 중앙값이 동일하게 유지되었다.

실전에서 쓰는 최대 burden shell은

$$
\boxed{
h^\dagger=\arg\max_h M_{\text{eff}}(h)
}
$$

이하 본문에서 실제 판정은 이 `h^\dagger`만 사용한다.  
기존의 `h^*=\arg\max_h M(h)`나 `edge/front > core`는 역사 비교용 surrogate였고, 현재 메인라인 판정식은 아니다.

현재 누적 실측을 반영한 강한 형태는

$$
\boxed{
M_{\text{eff}}(h^\dagger) > M_{\text{edge}}
\quad\text{and}\quad
M_{\text{eff}}(h^\dagger) > M_{\text{core}}
}
$$

이다.

즉 병적 burden의 최대점은 "겉 경계 한 줄"이 아니라, 경계에서 몇 hop 안쪽의 shell에 놓일 수 있다.

또한 shell peak의 위치만이 아니라, 그 peak가 놓이는 **공간 영역**과 **지배 축**도 같이 써야 한다. 이를 위해

$$
R^*=
\arg\max_{q\in\{\text{edge},\text{core},\text{stromal},\text{hypoxic}\}}
M_q
$$

$$
A^*=
\arg\max_{i\in\{\text{cell},\text{niche},\text{mech},\text{immune}\}}
w_i(R^*)
$$

로 두면, 현재 데이터가 가장 강하게 지지하는 식은

$$
\boxed{
R^*\in\{\text{stromal},\text{hypoxic}\},
\qquad
A^*\in\{\text{mech},\text{niche}\}
}
$$

이다.

뇌 쪽까지 포함하면 다음 추가 예측을 세울 수 있다.

$$
\bar s_{h^\dagger}
\ge
\bar s_{\text{core}},
\qquad
\bar s_h
:=
\eta_a(\bar x_{a,h}-x_a^*)_+
+
\eta_s(x_s^*-\bar x_{s,h})_+
+
\eta_b(x_b^*-\bar x_{b,h})_+
$$

즉 mismatch peak가 있는 shell에서는 뇌의 reserve deficit도 더 커야 한다.

이 마지막 식은 현재로서는 `hypothesis`다. 이유는 spatial transcriptomics로는 $w_r$는 바로 잡히지만, 같은 위치에서의 $p_r$는 아직 직접 측정이 드물기 때문이다.

#### 12.3.4.6 최신 고정모델 실측 결과 요약

이하 본문 메인라인은

$$
(\lambda_c,\lambda_n,\lambda_m,\lambda_i)=(0.15,0.15,0.45,0.25)
$$

인 고정가중 `M_{\text{eff}}`만 사용한다.  
균등가중과 `edge/front > core`는 현재 기준에서는 **반증된 이전 버전**으로만 취급한다.

현재 실제 공간 데이터는 CRC `2`, PDAC `5`, breast `6`, GBM `21`로 총 `34`샘플이다.

| 암종 | `n` | `M_{\text{eff}}(h^\dagger)>M_{\text{edge}}` | `M_{\text{eff}}(h^\dagger)>M_{\text{core}}` | `R^*\in\{\text{stromal},\text{hypoxic}\}` | `A^*\in\{\text{mech},\text{niche}\}` | `fit` | `slight` | `GBM shallow-core` | `unexplained gross` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRC | `2` | `2/2` | `2/2` | `2/2` | `2/2` | `2` | `0` | `0` | `0` |
| PDAC | `5` | `5/5` | `5/5` | `5/5` | `5/5` | `5` | `0` | `0` | `0` |
| breast | `6` | `6/6` | `6/6` | `6/6` | `6/6` | `6` | `0` | `0` | `0` |
| GBM | `21` | `20/21` | `16/21` | `20/21` | `20/21` | `14` | `5` | `2` | `0` |
| 전체 | `34` | `33/34` | `29/34` | `33/34` | `33/34` | `27` | `5` | `2` | `0` |

정확 검정으로 다시 쓰면:

1. `M_{\text{eff}}(h^\dagger) > M_{\text{edge}}`는 `33/34`로 exact `p \approx 2.04\times10^{-9}`다.
2. `M_{\text{eff}}(h^\dagger) > M_{\text{core}}`는 `29/34`로 exact `p \approx 1.93\times10^{-5}`다.
3. 최고 영역이 `stromal/hypoxic`인 경우는 `33/34`로 exact `p \approx 2.04\times10^{-9}`다.
4. 지배 축이 `mech/niche`인 경우도 `33/34`로 exact `p \approx 2.04\times10^{-9}`다.
5. `h^\dagger \ge 3`는 `17/34`, `\rho(A_{\text{tumor}})\ge1`은 `9/34`라서 둘 다 보조 신호 수준이다.
6. 잔차를 보면 남는 큰 오차 `2/34`는 모두 GBM에서만 나오며, 둘 다 `h^\dagger \le 2`이면서 `M_{\text{eff}}(h^\dagger)-M_{\text{core}} < -0.1`인 **shallow-core subtype**로 묶인다.

따라서 GBM에 한해서는 보조 하위모드를 다음처럼 둔다.

$$
\boxed{
\text{GBM shallow-core subtype}
\iff
h^\dagger \le 2
\quad\text{and}\quad
M_{\text{eff}}(h^\dagger)-M_{\text{core}} < -0.1
}
$$

이 하위모드까지 포함하면 현재 `34`샘플의 판정은 `fit 27`, `slight 5`, `GBM shallow-core 2`, `unexplained gross 0`으로 정리된다.

정합 강도를 더 정교하게 보기 위해, 각 샘플의 최소 지지 마진을

$$
\boxed{
\Delta_{\min}
=
\min\Big(
M_{\text{eff}}(h^\dagger)-M_{\text{edge}},
\;
M_{\text{eff}}(h^\dagger)-M_{\text{core}},
\;
\Delta_{\text{region}},
\;
\Delta_{\text{axis}}
\Big)
}
$$

로 둔다. 여기서

$$
\Delta_{\text{region}}
=
\max(M_{\text{stromal}}, M_{\text{hypoxic}})
-
\max(M_{\text{edge}}, M_{\text{core}})
$$

$$
\Delta_{\text{axis}}
=
\max(w_{\text{niche}}(R^*), w_{\text{mech}}(R^*))
-
\max(w_{\text{cell}}(R^*), w_{\text{immune}}(R^*))
$$

이다.

이 값을 기준으로 현재 `34`샘플은

| tier | 기준 | 샘플 수 |
|---|---|---:|
| `strong` | `\Delta_{\min} \ge 0.1` | `14` |
| `borderline positive` | `0 < \Delta_{\min} < 0.1` | `13` |
| `borderline negative` | `-0.1 < \Delta_{\min} \le 0` | `5` |
| `GBM shallow-core subtype` | 위 subtype 규칙 | `2` |

로 나뉜다.

즉 현재 자료의 뜻은, `34`개 모두가 한 모델에 완전히 같은 강도로 맞는 것은 아니지만, **설명 불가능한 큰 반례는 없고**, 대부분은 `강한 정합` 또는 `경계형 정합`으로 메인라인 모델 안에 들어온다는 것이다.

따라서 현재 실측이 가장 강하게 지지하는 문장은

$$
\boxed{
\text{병적 최대점은 edge가 아니라, mech-weighted stromal/hypoxic inner shell에 놓인다.}
}
$$

이다.

#### 12.3.4.7 지금 시점의 안전한 결론

이 절에서 당장 올릴 수 있는 문장은 다음 정도다.

$$
\boxed{
\text{뇌-암 연결은 }p_r\in\Delta^2\text{ 와 }w_r\in\mathbb{R}_{\ge0}^4\text{ 의 결합 동역학으로 최소 정식화할 수 있다.}
}
$$

그리고 실제 판정은 아래 순서로 진행하면 된다.

1. 공간 데이터에서 먼저 $w_r$와 `M_{\text{eff}}(h)`를 추정한다.
2. `\hat A_r`, `\hat\rho(A_r)`, `M_{\text{eff}}(h^\dagger)-M_{\text{edge}}`, `M_{\text{eff}}(h^\dagger)-M_{\text{core}}`, `R^*`, `A^*`를 계산한다.
3. 이후 EEG/fMRI/metabolic imaging과 수면/자율신경 지표가 붙으면 같은 영역에서 $p_r$를 추정하고, `C_r`, `H_r`, `b_r`, `K_r`를 적합한다.

즉 지금 단계에서 가장 강한 수식은

$$
\boxed{
M_{\text{eff}}(h^\dagger) > M_{\text{edge}}
\quad\text{and}\quad
M_{\text{eff}}(h^\dagger) > M_{\text{core}}
}
$$

와

$$
\boxed{
R^*\in\{\text{stromal},\text{hypoxic}\},
\qquad
A^*\in\{\text{mech},\text{niche}\}
}
$$

이며,

$$
\rho(A_r)\ge1
\quad\to\quad
\text{종양 단독 임계의 후보 게이트}
$$

$$
\rho(K_r)\ge1
\quad\to\quad
\text{뇌-종양 결합 임계}
$$

로 읽는 것이 맞다.

따라서 범암 가설의 현재 판정은 다음 세 층으로 축약된다.

$$
\boxed{
M_{\text{eff}}(h^\dagger) > M_{\text{edge}}
\quad\text{and}\quad
M_{\text{eff}}(h^\dagger) > M_{\text{core}}
}
$$

$$
\boxed{
\ R^*\in\{\text{stromal},\text{hypoxic}\},
\qquad
A^*\in\{\text{mech},\text{niche}\}
}
$$

$$
\boxed{
\rho(A_{\text{tumor}})\ge 1
\quad\text{는 강하지만 선택적 보조 게이트}
}
$$

즉 shell peak 우위가 1차 판정이고, `\rho(A_{\text{tumor}})\ge1`은 통과하면 더 강한 지지로 읽는다.

정상 조직은 대략

$$
w_{n+1}=A_{\text{normal}}w_n+u_n,\qquad \rho(A_{\text{normal}})<1
$$

처럼 조직 수준의 복원력이 작동하고,

공격적 고형암은

$$
w_{n+1}=A_{\text{tumor}}w_n+u_n,\qquad \rho(A_{\text{tumor}})\gtrsim 1
$$

또는 별도 attractor $w_{\text{tumor}}^*$를 갖는 경우로 읽을 수 있다.

여기서 중요한 것은 `A_{\text{tumor}}`의 성분이 암종마다 다르다는 점이다.  
즉 **공통 구조는 강하게 잡되, 공통 상수는 아직 미정**으로 둔다.

### 12.4 CE 대응

| 현상 | 실험 측 사실 | CE 해석 | 판정 |
|---|---|---|---|
| hypoxia -> immune exclusion | solid tumor 전반에서 반복 | 배경장 왜곡이 제거 실패를 유도 | `bridge` |
| invasive front / budding | GBM PBZ, CRC budding hotspot | 경계에서 정합 붕괴가 먼저 드러남 | `bridge` |
| stromal / ECM dominance | PDAC desmoplasia, breast stiffness shift | 구조장이 병적으로 재편됨 | `bridge` |
| multicompartment tumor ecology | tumor cells + CAF/TAM + vessel + ECM 상호작용 | 세포 단독이 아닌 위상-세포 결합계 | `bridge` |
| 암 일반 = 하나의 병적 위상 법칙 | 모든 암종에 공통 attractor | 범암 고정점 정리 | `hypothesis` |

### 12.5 해석

- GBM만 보면 "뇌 특수성"일 수 있었는데, PDAC / CRC / breast까지 보면 **경계, niche, ECM, immune exclusion**이 반복된다.
- 최신 메인라인은 고정가중 `M_{\text{eff}}=0.15\,cell+0.15\,niche+0.45\,mech+0.25\,immune` 하나만 사용한다.
- 전체 `34`샘플에서 `M_{\text{eff}}(h^\dagger) > M_edge`는 `33/34`(exact `p \approx 2.04\times10^{-9}`), `M_{\text{eff}}(h^\dagger) > M_core`는 `29/34`(exact `p \approx 1.93\times10^{-5}`)다.
- 병적 최대 영역은 `stromal/hypoxic`가 `33/34`, 지배 축은 `mech/niche`가 `33/34`라서, 현재 원인 해석의 중심은 `cell-edge`가 아니라 `mech-weighted stromal/hypoxic shell`이다.
- 남는 큰 오차 `2`개도 버려지는 반례가 아니라, `h^\dagger \le 2`와 `M_{\text{eff}}(h^\dagger)-M_{\text{core}} < -0.1`를 보이는 GBM shallow-core subtype으로 묶인다.
- 따라서 "암은 세포 내부 변이만이 아니라, 세포와 조직 위상의 정합이 무너지고 그 불일치가 제거되지 않는 상태"라는 문장은 유지하되, 병적 최대점의 위치는 **겉 경계 한 줄보다 안쪽 shell, 그리고 특히 stromal/hypoxic niche**로 읽는 편이 현재 데이터와 더 잘 맞는다.
- 아직 "모든 암이 동일한 동역학 상수나 동일한 3분배로 닫힌다"는 수준은 아니다.

### 12.6 운영 원칙

앞으로는 이 절을 다음처럼 운영한다.

1. `암 = 위상-세포 정합 붕괴`를 기본 가설로 채택한다.
2. 새 암종을 볼 때마다 먼저 이 모델로 설명을 시도한다.
3. 맞지 않는 암종이나 반례가 나오면 그때 모델을 깎는다.
4. 즉 **입증 전까지 위축되지 말고, 반증 전까지는 강하게 밀어본다.**

### 12.7 지금 바로 걸 수 있는 범암 게이트

| 게이트 | 점검 항목 | 통과 기준 |
|---|---|---|
| `G-P1` | 공간 분리 | 적어도 서로 다른 3개 이상의 고형암에서 invasive / hypoxic / stromal niche가 재현 |
| `G-P2` | 내부 peak 우선성 | 대다수 샘플에서 `M_{\text{eff}}(h^\dagger) > M_edge` 그리고 `M_{\text{eff}}(h^\dagger) > M_core`, 가능하면 `h^\dagger \ge 3` |
| `G-P3` | 구조장 변화 | ECM / stiffness / vessel remodeling이 세포 상태 변화와 함께 관측 |
| `G-P4` | 면역 배제 결합 | immune exclusion이 hypoxia 또는 stromal barrier와 통계적으로 연결 |
| `G-P5` | 단일세포 충분성 반례 | 세포 내부 변이만으로 설명되지 않는 공간 효과가 반복 확인 |

### 12.8 범암 mismatch 측정식

작업가설을 말이 아니라 수치로 밀기 위해, 암종별로 다음 점수를 고정한다.

1. 세포 제어 붕괴 점수

$$
M_{\text{cell}} = \operatorname{norm}\big(
\text{RTK/RAS burden}
+ \text{TP53/RB failure}
+ \text{apoptosis loss}
+ \text{cell-cycle activation}
\big)
$$

2. 공간 분리 점수

$$
M_{\text{niche}} = \operatorname{norm}\big(
\text{state segregation}
+ \text{front/core divergence}
+ \text{perivascular-hypoxic split}
\big)
$$

3. 구조장 재배치 점수

$$
M_{\text{mech}} = \operatorname{norm}\big(
\text{ECM remodeling}
+ \text{stiffness shift}
+ \text{vascular remodeling}
+ \text{boundary extension}
\big)
$$

4. 면역 배제 점수

$$
M_{\text{immune}} = \operatorname{norm}\big(
\text{T-cell exclusion}
+ \text{CAF/TAM suppressive burden}
+ \text{hypoxia-linked immune block}
\big)
$$

이 문서에서 실제로 쓰는 최종 범암 mismatch 점수는

$$
\boxed{
M_{\text{eff}}
=
0.15\,M_{\text{cell}}
\;+\;
0.15\,M_{\text{niche}}
\;+\;
0.45\,M_{\text{mech}}
\;+\;
0.25\,M_{\text{immune}}
}
$$

로 둔다.

즉 현재 데이터는 `cell` 단독 축보다 `mech` 축을 가장 크게 봐야 하고, `niche`는 낮은 가중으로 남기되, `immune`도 완전히 버리면 정밀도가 떨어진다고 읽는 편이 더 맞다.  
다만 **최고 병적 영역에서 실제로 우세하게 나타나는 축**은 여전히 `mech/niche`다.

이 문서의 현재 강한 예측은 다음이다.

$$
\boxed{
\exists h^\dagger
\text{ 가 존재하여 }
M_{\text{eff}}(h^\dagger) > M_{\text{edge}}
\quad\text{및}\quad
M_{\text{eff}}(h^\dagger) > M_{\text{core}}
}
$$

실전에서는 `h^\dagger \ge 3`이면 더 강한 interior-shell 패턴으로 읽는다.

그리고 같은 샘플에서

$$
\boxed{
R^*\in\{\text{stromal},\text{hypoxic}\},
\qquad
A^*\in\{\text{mech},\text{niche}\}
}
$$

가 같이 잡히면, 그 샘플은 **세포 내부 증식형보다 미세환경-구조장 주도형**으로 읽는다.

그리고 공격성/재발성에 대해

$$
\boxed{
M_{\text{recurrence zone}} > M_{\text{non-recurrence zone}}
}
$$

가 반복되어야 한다.

### 12.9 즉시 실행할 범암 입증 프로토콜

1. 암종 선택
   - `GBM`, `PDAC`, `CRC`, `breast`를 1차 세트로 고정

2. 데이터 고정
   - bulk genomics / CNV
   - single-cell RNA-seq
   - spatial transcriptomics or spatial proteomics
   - pathology image
   - 가능하면 elastography / stiffness / ECM imaging

3. 영역 분할
   - `core`, `edge/front`, `perivascular`, `hypoxic`, `stromal`, `normal-adjacent`

4. 점수 계산
   - 각 영역마다 `M_cell`, `M_niche`, `M_mech`, `M_immune`, `M_tumor` 계산

5. 직접 판정
   - 어떤 `h^\dagger`에서 `M_{\text{eff}}(h^\dagger) > M_edge` 및 `M_{\text{eff}}(h^\dagger) > M_core`가 성립하는가
   - 가능하면 `h^\dagger >= 3`인지도 함께 본다
   - recurrence zone이 non-recurrence zone보다 높은가
   - immune exclusion이 hypoxia/stroma와 결합하는가
   - mechanical remodeling이 niche 분리와 동반되는가

6. 범암 결론
   - 4개 암종 중 3개 이상에서 위 패턴이 반복되면, 이 가설을 범암 기본모델로 유지
   - 2개 이하에서만 맞으면 암종-특이 모형으로 후퇴

### 12.10 실패 시 해석 규칙

1. spatial niche는 있으나 기계적 장 변화가 약하면, "위상-세포 정합 붕괴"를 `공간-세포 정합 붕괴` 수준으로 낮춘다.
2. 기계적 장 변화는 크지만 immune exclusion과 연결이 약하면, 구조장 해석은 유지하되 `통제 이탈의 핵심축`으로는 승격하지 않는다.
3. PDAC / CRC / breast 중 둘 이상에서 반복이 약하면, 이 절은 "일부 침윤성 암종에서의 공통 패턴"으로 내린다.
4. 범암 비교가 계속 암종-특이 변수에 의해 갈라지면, 공통 attractor 가설은 폐기하고 암종별 동역학으로 분기한다.
5. `M_{\text{eff}}(h^\dagger) > M_{\text{edge}}`와 `M_{\text{eff}}(h^\dagger) > M_{\text{core}}`가 반복적으로 틀리면, "interior shell peak" 문장을 버리고 core-driven model 또는 edge-only model로 전환한다.
6. `M_{\text{tumor}}`가 예후/재발/침윤과 연결되지 않으면, 이 점수는 폐기하고 성분 점수만 유지한다.

### 12.11 공개 데이터셋 고정

증명은 "좋은 예시 몇 개"가 아니라, **같은 계산을 서로 다른 공개 데이터셋에 반복 적용**하는 방식으로만 한다.

1. GBM
   - 10x Genomics Visium CytAssist human glioblastoma FFPE dataset
   - BioProject `PRJNA1337938`
   - processed data: Zenodo `10.5281/zenodo.17572905`

2. PDAC
   - GEO `GSE274103`
   - GEO `GSE272362`
   - 10x Genomics Visium HD human pancreatic cancer fresh frozen dataset

3. CRC
   - GEO `GSE267401`
   - GEO `GSE226997`
   - 10x Genomics Visium / Visium HD human colorectal cancer datasets

4. breast
   - Zenodo `4739739` (single-cell and spatially resolved atlas of human breast cancers)
   - 10x Genomics human breast cancer Visium fresh frozen dataset
   - GEO `GSE243022` or equivalent TNBC spatial set

원칙:

- 각 암종당 최소 `2`개 이상의 독립 공간 데이터셋을 사용
- 가능하면 하나는 논문 기반 public deposition, 하나는 10x reference set으로 교차 확인
- restricted access가 필요한 자료는 보조로만 쓰고, **주 판정은 공개 접근 가능한 데이터**로만 한다

### 12.12 실제 계산 순서

각 데이터셋에 대해 다음 8단계를 동일하게 적용한다.

1. 공간 영역 라벨링
   - `core`, `edge/front`, `perivascular`, `hypoxic`, `stromal`, `normal-adjacent`
   - pathology annotation이 있으면 우선 사용
   - 없으면 marker 기반 rule로 정의

2. 세포 제어 붕괴 축 계산
   - `RTK/RAS`, `TP53/RB`, `apoptosis loss`, `cell-cycle activation` gene-set score
   - 이들을 합쳐 `M_cell`

3. niche 축 계산
   - cell-state entropy가 아니라 **state segregation**과 **front/core divergence**를 사용
   - perivascular-hypoxic split, invasive-front enrichment를 합쳐 `M_niche`

4. 구조장 축 계산
   - ECM genes, matrisome score, angiogenesis score, stiffness proxy, boundary extension score
   - 이들을 합쳐 `M_mech`

5. immune 축 계산
   - T-cell exclusion
   - CAF/TAM suppressive programs
   - hypoxia-linked immune block
   - 이들을 합쳐 `M_immune`

6. 최종 점수 계산
   - `M_eff = 0.15 M_cell + 0.15 M_niche + 0.45 M_mech + 0.25 M_immune`

7. 영역 비교
   - `M_{\text{eff}}(h) - M_edge`, `M_{\text{eff}}(h) - M_core`, `h^\dagger = argmax_h M_{\text{eff}}(h)`
   - `M_recurrence zone - M_non-recurrence zone`
   - `M_hypoxic - M_nonhypoxic`

8. 범암 집계
   - 암종별 sign test
   - 데이터셋별 effect size
   - 메타분석 수준의 방향 일치율

### 12.13 행렬과 forcing의 추정

이 절의 정리를 실제 데이터로 치기 위해, 공간 영역을 정점으로 하는 directed graph를 만든다.

- 정점: 영역 spot/region cluster
- 간선: `core -> edge`, `vessel -> hypoxia gradient`, `adjacent normal -> invasive front` 같은 공간 인접 관계

영역별 상태벡터를 $w_r$라 두고, 인접 쌍 $(r,s)$에 대해

$$
\hat A
=
\arg\min_{A\ge 0}
\sum_{(r,s)\in E}
\|w_s - A w_r\|^2
\;+\;
\eta \|A\|_F^2
$$

를 nonnegative ridge regression으로 추정한다.

그다음

$$
\hat\rho = \rho(\hat A)
$$

를 계산한다.

forcing은 잔차로 잡는다.

$$
\hat u_{r\to s} = w_s - \hat A w_r
$$

그리고 영역 평균으로

$$
\bar u_{\text{edge/front}}
=
\frac{1}{|E_{\text{edge}}|}
\sum_{(r,s)\in E_{\text{edge}}}
\hat u_{r\to s}
$$

$$
\bar u_{\text{core}}
=
\frac{1}{|E_{\text{core}}|}
\sum_{(r,s)\in E_{\text{core}}}
\hat u_{r\to s}
$$

를 둔다.

이 문서의 실제 판정은 다음 세 값을 본다.

$$
h^\dagger=\arg\max_h M_{\text{eff}}(h), \qquad h^\dagger \ge 3 \quad ?
$$

$$
M_{\text{eff}}(h^\dagger) > M_{\text{edge}}
\quad\text{and}\quad
M_{\text{eff}}(h^\dagger) > M_{\text{core}} \quad ?
$$

$$
\hat\rho \ge 1 \quad ?
$$

### 12.14 marker 고정 규칙

암종이 달라도 아래 4축은 고정한다.

1. `M_cell`
   - proliferation: `MKI67`, `TOP2A`, `PCNA`
   - RTK/RAS: `EGFR`, `ERBB2`, `MET`, `KRAS` program
   - TP53/RB failure proxy: `CDKN2A/B loss`, `E2F targets`, `G2M checkpoint`
   - apoptosis loss: `BCL2`-like survival bias or apoptosis hallmark inverse

2. `M_niche`
   - hypoxia: `HIF1A`, `VEGFA`, glycolysis genes
   - invasive/front: EMT, migration, tumor budding or edge markers
   - perivascular: endothelial/pericyte adjacency, angiogenic program

3. `M_mech`
   - matrisome / collagen / integrin / POSTN / FN1 / COL1A1 / COL3A1
   - angiogenesis / vessel remodeling
   - stiffness proxy or elastography if available

4. `M_immune`
   - CD8 exclusion
   - suppressive TAM / myeloid burden
   - CAF suppressive program
   - `IL10`, `TGFB`, checkpoint ligand axis

세부 marker는 암종별로 조금 바뀔 수 있지만, **축 자체는 바꾸지 않는다.**

### 12.15 증명 완료의 기준

이 절에서 말하는 "증명"은 순수수학적 전칭정리가 아니라, 다음 세 층을 모두 통과하는 것을 뜻한다.

1. 수학적 층
   - `정리 1`, `정리 2`, `따름정리`가 성립

2. 데이터 층
   - 적어도 `GBM`, `PDAC`, `CRC`, `breast` 중 `3`개 이상에서
   - `M_{\text{eff}}(h^\dagger) > M_edge`
   - `M_{\text{eff}}(h^\dagger) > M_core`
   - `M_recurrence zone > M_non-recurrence zone`
   가 반복
   - `\hat\rho \ge 1`은 있으면 강한 지지지만, 현재는 보조 게이트로 둔다

3. 반례 층
   - 공간 데이터에서 core-driven pattern이 반복적으로 우세한 암종이 다수 나오지 않을 것

즉 이 문서의 완료 조건은:

$$
\boxed{
\text{정리 성립}
\;+\;
\text{공개 데이터셋 반복 확인}
\;+\;
\text{반례 미누적}
}
$$

이다.
