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
- 하지만 "Wake-NREM-REM 1회가 CE 부트스트랩 사상 1회 적용"이라는 문장은 아직 직접 실험식이 아니다.
- 그래도 `48-72시간` 박탈에서 전전두엽-시상계와 인지 수행이 급격히 무너지는 것은, CE 문서가 말해온 "2-3회 반복의 중요성"과 시간 스케일 면에서 꽤 잘 맞는다.
- 즉 `sleep.md`의 큰 그림은 유지할 수 있지만, 수학적 반복법으로 바로 쓰기에는 중간 동역학이 더 필요하다.

### 3.5 판정

$$
\boxed{\text{수면의 offline renormalization은 강하게 체크되며, 반복 사상 해석만 아직 브리지다.}}
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

## 8. 다음에 풀 수식 문제

### 8.1 `p = (x_a, x_s, x_b)`의 측정 정의

가장 먼저 필요한 것은:

$$
p_{\text{brain}}(t) = (x_a(t), x_s(t), x_b(t))
$$

를 뇌 데이터에서 어떻게 측정할지 정하는 것이다.

후보 proxy:
- `x_a`: task-evoked metabolic increment, event-locked firing fraction
- `x_s`: synaptic signaling + maintenance + plasticity-related cost
- `x_b`: intrinsic / resting / DMN / baseline spontaneous activity

### 8.2 `B`의 재정의

이제 `homeomorphism.md`에서는 `B`를 다음 두 층으로 분리한다.

1. 정적 고정점 연산자:

$$
B_* : p \mapsto p^*
$$

2. 실제 동역학 반복:

$$
p_{n+1} = \mathcal{F}(p_n; \theta_{\text{brain}})
$$

여기서만 야코비안과 수렴률을 말할 수 있다.

#### 최소 관측-정합 반복식

현재 실험 사실과 CE 수축률을 동시에 만족하는 가장 단순한 반복식은 다음과 같다.

1. 각성으로 인한 이탈:

$$
p_{n+\frac12} = p_n + u_n, \qquad \mathbf{1}^\top u_n = 0
$$

여기서 $u_n$은 각성 중 누적되는 task load / synaptic load / background drift다.

2. 수면으로 인한 수축:

$$
p_{n+1} = p^* + \rho \big(p_{n+\frac12} - p^*\big) + \xi_n, \qquad \rho = 0.155
$$

즉,

$$
e_n := p_n - p^* \quad \Longrightarrow \quad e_{n+1} = \rho e_n + \rho u_n + \xi_n
$$

이다.

만약 sleep noise를 무시하고 $\|u_n\| \le U$라면,

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

### 8.3 곡률의 뇌 측 proxy

현재 `\|\Delta_g \Phi\|^2`는 너무 추상적이다. 후보 proxy는:
- slow-wave activity
- population synchrony / desynchrony
- functional connectivity roughness
- replay burden

### 8.4 `\delta[t]`의 생물학적 정의

가장 안전한 후보는:

$$
\delta[t] = a \cdot \text{RPE}(t) + b \cdot \text{surprise}(t) + c \cdot \text{novelty}(t)
$$

이고, CE 고정점과의 연결은 그 다음 단계에서 검토하는 것이 맞다.

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
| 수면 루프 잔차 감쇠 | 1회 `15.5%`, 2회 `2.4%`, 3회 `0.37%` | `bridge` | 최소 반복식 $e_{n+1}=\rho e_n+\rho u_n+\xi_n$, $\rho=0.155$ |
| dense 초기화 후 활성 비율 | `33.3% -> 9.28% -> 5.55% -> 4.98%` | `bridge` | $p_{n+1}=p^*+\rho(p_n-p^*)$의 직접 계산 |
| Top-k 최적점 | 중심 `4.87%`, 실용 대역 `3%-7%` | `bridge` | sparse neural regime + 고정점 자기일관성 |
| 수면 유무 비교 | sleep: bounded residual, wake-only: 선형 drift | `supported/bridge` 혼합 | 수면 항상성 데이터는 강하고, 반복식 매칭은 bridge |

반대로 아직 금지해야 하는 예측은:

- `환각률 <= 4.87%`
- `Truthfulness >= 95.13%`
- `의식 깊이 = 1 - ||p-p^*||`
- `dopamine = ||p-p^*||` 또는 그 미분

즉 앞으로의 원칙은 다음 한 줄로 요약된다.

$$
\boxed{\text{구간 포함 + 최소 반복식으로 닫히는 예측만 올리고, 태스크 점수 hard bound는 아직 올리지 않는다.}}
$$
