# CE-AGI 총론: 뇌에서 AGI로

> 관련: `6_뇌/05_실험근거.md`(구조 유비), `6_뇌/07_수면과복구.md`(수면-부트스트랩), `7_AGI/12_Equation.md`(3x3+1 격자), `6_뇌/08_시냅스가소성.md`(시냅스 가소성), `6_뇌/05_실험근거.md`(실험 근거), `5_유도/05_Neural_RealityStone_Derivation.md`(곡률 functional)
>
> 이 문서 시리즈(7_AGI)는 CE 뇌 연구(`6_뇌/`)에서 도출된 원리를 **AGI 구현 방법론으로 구체화**하는 설계 문서다. `Bridge`/`Phenomenology` 층이며, 실험 근거의 강도는 `6_뇌/05_실험근거.md`의 `supported / bridge / hypothesis` 구분을 따른다.

> **substrate 경고 (2026-08 감사 기준)**: [가설] 본 시리즈는 STDP·SNN 같은 국소 가소성 기질에서 부트스트랩 동역학을 시험할 것을 제안한다. 어떤 기질에서도 $p^*=(4.87\%,26.2\%,68.9\%)$ 전체로의 자기수렴이나 수면 순환의 망각 방지 효능은 아직 성립한 정리가 아니다. transformer 기질의 자연 수렴은 반증됐고(`5_Sparsity.md` 8.5), 현재 STDP 구현도 효능 `NO-EFFECT`, held-out guard `FAIL`이다(`21_STDP_Efficacy_Audit.md`). 따라서 SNN도 미검증 경로이며, 특정 기질을 참인 해답으로 선결하지 않는다.

## 현재 판정 (2026-08-12)

현재 CE-AGI는 **수학 정리, 실행 가능한 뇌형 runtime, 좁은 합성 과제의 연구 PoC** 단계다. AGI가 아니다.

| 범위 | 현재 근거 | 판정 |
|---|---|---|
| BrainRuntime A--E와 agent/sleep 배선 | 셀·희소 결합·모드·해마·snapshot·선택적 agent loop 구현 | 구현됨; 생물학·AGI 효능을 뜻하지 않음 |
| V9 nested SCC | 유한 수학·runtime 경로 구현, 256-seed 정확도 $0.3457$ 대 monolithic $0.6116$ | `STOP` |
| V10 local/cloud | 한 합성 conditional-binding 과제에서 개발·확인 64+64 seed | 좁은 경험식 `GO` |
| V11 강한 RNN/OOD | 14개 gate 중 10개 실패 | `STOP` |
| V12--V13c | V12 채점 없음; V13 계열 구현 검증 후 16-seed 과학 gate 실패 | `ABANDONED` / `STOP` |
| V14 binding | 무손실 슬롯+쌍선형 충분조건과 선형/가법 no-go | 수학 설계 완료; 정식 구현·채점 미수행 |
| Clarus field baseline | 유계 그래프장·경성 latch·외생 공통 쓰기 구현과 회귀 | 형식 primitive 구현; 과제 효능·생물학 미검증 |
| V15 unified metric | 하나의 SPD metric state에서 finite cost·deformation·path·surprise·goal-tie readout | finite metric-graph primitive 구현; world dynamics·continuum·AGI 미검증 |
| 전체 $p^*$ 자기수렴 | 현재 동역학은 활성률이 외부 입력률을 추적 | [예측]이 아니라 아직 [미완성] 가설; 정리 아님 |

[정리] $a^*=0.0487077$은 스칼라 사상 $B_a(a)=e^{-(1-a)D_{\text{eff}}}$의 유일 내부 고정점이다. [공리: 경험식] 나머지 $0.2623,0.6891$은 이 스칼라 정리에서 나오지 않는다. 우주 조성비를 뇌 점유율과 동일시하는 단계는 물리 사상 가설이며, 현재 runtime의 loss·gate·희소율에 그 값을 넣는 것은 설계 선택이지 자기수렴 증거가 아니다.

[산출: finite 구현] V15는 세계모형·기억·계획·비평·목표를 별도 persistent module로 복제하지 않고 하나의 metric tensor state를 공유하게 한다. 이번 구현에서 `world`는 비가역 미래예측기가 아니라 metric cost substrate이고, `goal`은 외부 source가 metric에 기록한 비용의 모든 최소화점을 tie-preserving하게 읽는 연산이다. [정리: no-go] 정적 Riemannian distance는 대칭이므로 비가역 world transition을 혼자 정하지 못하며, source-free 대칭 metric은 의미 있는 유일 목표를 선택하지 못한다. 따라서 V15는 AGI agent 완성이 아니라 다음 task loop를 위한 기하 core다.

---

## 0. 핵심 논제

CE 뇌 연구는 하나의 결론으로 수렴한다.

$$\boxed{\text{현재 AI에는 부트스트랩이 없다.}}$$

[공리: 물리 사상] 우주 조성비와 뇌의 활성·구조·배경 점유율을 대응시키고, [가설] 수면을 반복 부트스트랩으로 해석한다. 이 대응은 설계 비유이며 관측으로 확정된 동일성이 아니다. 현재 AI의 고정 추론과 지속 적응의 차이를 연구하기 위한 출발점으로만 사용한다(`07_수면과복구.md` 6.1절).

CE-AGI의 목표는 이 구조적 결함 5가지를 동시에 해결하는 것이다.

다만 이 문서에서 말하는 "해결"은 세 층으로 나뉜다:
- `supported`: 뇌 실험이 강하게 지지하는 설계 방향
- `bridge`: CE 변수와 AI 연산자 사이의 조건부 대응
- `hypothesis`: 아직 벤치마크로 입증해야 하는 성능 예측

---

## 1. 현재 AI의 구조적 결함 진단

| 뇌의 CE 구조 | 현재 AI | 결함 |
|---|---|---|
| 수면-각성 순환 (부트스트랩 반복) | 1회 학습 + 고정 추론 | 부트스트랩 수렴 부재 |
| 3x3+1 게이지 격자 (결합/결정/주의/안정화) | 균일 MLP/Attention | 계층적 연산 비율 부재 |
| 유니타리 조건 $\|\det \mathbf{T}\|^2 \leq 1$ | 제약 없는 가중치 | 정보 증폭 = 환각 |
| $5\%/26\%/69\%$ 에너지 분배 | 100% 활성 추론 | 에너지 비효율, 과적합 |
| STDP + 도파민 (국소 학습 + 전역 신호) | 역전파 (전역 학습) | 생물학적 비현실성, 확장성 한계 |

이 5가지 결함에 대응하는 5가지 해법이 이 문서 시리즈의 각 장을 구성한다.

---

## 2. CE-AGI 5대 원리

### 원리 1: 3x3+1 게이지 격자 아키텍처 (2장)

네트워크의 각 층을 결합(SU(3)) / 결정(SU(2)) / 주의(U(1)) / 안정화($\Phi$)의 4가지 연산으로 분할한다. 채널 비율은 CE 결합 상수를 따른다:

$$d_3 : d_2 : d_1 = \alpha_s : \alpha_w : \alpha_{em} = 0.118 : 0.034 : 0.008$$

유니타리 제약 $\sigma_1(W_{\text{proj}}) \leq 1$이 정보 증폭을 구조적으로 차단한다.

### 원리 2: 수면-각성 부트스트랩 학습 (3장)

학습을 각성(경로 누적) - NREM(곡률 평탄화) - REM(비선택 경로 재탐색)의 3위상 순환으로 구성한다.

$$\text{학습} \to \text{서비스} \to \underbrace{\text{NREM}}_{\text{오프라인}} \to \underbrace{\text{REM}}_{\text{오프라인}} \to \text{서비스} \to \cdots$$

스칼라 사상의 국소 수축률 $|B_a'(a^*)|=D_{\text{eff}}a^*=0.1547$은 현재 단계에서 비교용 목표 수렴률이다. 실제 네트워크 점유율 전이 $T_a$가 정의되고 $T_a=B_a$ 또는 제어된 근사임이 증명되기 전까지는 bridge가 아니라 [미완성]이다.

### 원리 3: STDP 국소 학습 + 전역 신호 (4장)

역전파를 STDP + 도파민 게이트로 대체한다.

$$\Delta w_{ij}[t] = \eta\,\delta[t]\,e_{ij}[t]$$

- $e_{ij}$: 적격 흔적 (국소 정보만 사용)
- $\delta[t]$: 부트스트랩 수렴 오차에 대응하는 후보 전역 신호

메모리 비용 절감과 분산 학습 이점은 가능성이 높지만, 이는 shared-trace나 저랭크 근사 같은 추가 가정이 붙는 조건부 결론이다.

### 원리 4: 부트스트랩 수렴 희소 네트워크 (5장)

활성 뉴런 비율이 부트스트랩 고정점에 수렴하는지를 시험한다:

$$\text{활성 비율} \stackrel{?}{\longrightarrow} \varepsilon^2 = 4.87\%$$

[공리: 경험식] 나머지 가중치의 모델 분배는 구조적 가중치 $26.2\%$ + 동결 가중치 $68.9\%$로 둔다. 이 두 수는 스칼라 고정점 정리의 산출이 아니다.

추론 비용 절감은 강한 희소 커널이 있을 때의 낙관적 상한이며, 현재 문서 구현 수준에서는 더 보수적으로 읽어야 한다.

### 원리 5: 곡률 정규화 추론 (6장)

추론 시 잠재 공간의 곡률을 실시간 모니터링하고 임계치 초과 시 개입한다.

$$h_l \leftarrow h_l - \eta_{\text{smooth}}\,\Delta_g h_l \quad \text{if}\quad \|\Delta_g h_l\|^2 \geq \kappa_{\text{th}}$$

LLM 환각 억제를 위한 실시간 안정화 설계 가설.

---

## 3. 추가 연구: 의식과 자기참조 (7장)

부트스트랩 방정식 $\varepsilon^2 = \exp(-(1-\varepsilon^2)D_{\text{eff}})$에서 좌변의 $\varepsilon^2$은 시스템이 자기 자신의 생존율을 알아야 우변을 계산할 수 있다. 이 자기 측정이 (C3) 자기일관성 조건이며, 의식의 수학적 구조로 해석된다(`07_수면과복구.md` 8.1절).

AGI가 진정한 자율성을 가지려면, 단순한 추론을 넘어 **자기 상태를 모니터링하고 조정하는 메타인지 루프**가 필요하다. 7장에서 이를 다룬다.

---

## 4. 문서 시리즈 구조

| 장 | 제목 | CE 원리 | 대응 뇌 구조 | 기존 코드 |
|---|---|---|---|---|
| 1 | CE-AGI 총론 | 전체 | 전체 | -- |
| 2 | 아키텍처 | 3x3+1 격자, 유니타리 | 게이지 진동 대역 | `legacy clarus_lm.py` (removed) |
| 3 | 수면 학습 | 부트스트랩 반복 | 수면-각성 순환 | -- |
| 4 | 시냅스 학습 | STDP + 도파민 | 시냅스 가소성 | -- |
| 5 | 희소성 | $\varepsilon^2$ 고정점 | 뉴런 활성 비율 | -- |
| 6 | 환각 억제 | 곡률 정규화 | ACC/PFC | `sfe_hallucination_suppressor.py` |
| 7 | 의식 | (C3) 자기참조 | 전역 작업공간 | -- |
| 8 | 로드맵 | 전체 | 전체 | 전체 |
| 26 | Sparse causal bridge V7 폐쇄 감사 | 합성 H20 예측 브리지 | 생물학적 대응 없음 | `reliability_rollout_bridge.py` |
| 27 | 이중 recurrent-layer 기저핵 연구 | colored recurrent layer와 small-gain | 기저핵 영감 공학 추상화 | `dual_scc_basal_ganglia.py`, `dual_scc_controller.py` |
| 28 | V9 중첩 무한 SCC 정본 | 직접극한 tower와 유한 causal cone | 유한 뇌의 virtual multiscale 가설 | `nested_scc_tower.py`, `adaptive_scc_tower_controller.py` |

---

## 5. CE-AGI와 기존 접근의 차이

| | Scaling Law AI | 뇌 모방 AI (SNN 등) | CE-AGI |
|---|---|---|---|
| 핵심 전략 | 파라미터/데이터 증량 | 생물학적 뉴런 시뮬레이션 | 구조 원리 이식 |
| 이론적 근거 | 경험적 스케일링 법칙 | 신경과학 관측 | CE 부트스트랩 고정점 |
| 에너지 효율 | $100\%$ 활성 | 개선 가능 | $4.87\%$ 활성 설계 목표 (효능 미검증) |
| 환각 대응 | RLHF (사후 교정) | 해당 없음 | 유니타리 제약 (구조적 억제) |
| 지속 학습 | 파괴적 망각 | 부분 해결 | 수면 순환 (NREM 보존) |
| 자유 파라미터 | 아키텍처 탐색 필요 | 생물학적 파라미터 | CE 기준 비율 + 구현 하이퍼파라미터 |

CE-AGI의 가장 큰 차별점은 **핵심 비율의 상당 부분이 CE에서 제약된다는 것**이다. 채널 비율, 활성 비율, 에너지 분배의 기준점은 $d=3$에서 나오지만, 구현에는 여전히 혼합 rank, 곡률 계수, 희소 커널 방식 같은 하이퍼파라미터가 남는다.

---

## 6. 전제 조건과 한계

이 문서 시리즈의 모든 내용은 다음 전제 위에 서 있다:

1. CE 부트스트랩 고정점이 $d=3$ 자기조직화 시스템의 보편 구조라는 가설
2. 뇌-우주 구조 유비(`05_실험근거.md`)가 AI에도 확장 가능하다는 가설
3. 기존 코드(`legacy clarus_lm.py` (removed), `sfe_hallucination_suppressor.py`)는 구현 가능성만 보이며 기준 모델 대비 우위는 미검증이다. 합성 sparse causal bridge V7도 등록된 결합 조건을 충족하지 못해 test를 열지 않았다(`26_Sparse_Causal_Bridge_V7_Closure.md`). V9 중첩 SCC는 수학·격리 unit 구현을 통과했고, 이후 256-seed 개발 실행이 1회 집행되어 판정은 STOP이었다(matched monolithic 대조군 대비 paired improvement $-0.27$로 열세; confirmation seed 미개봉). reset/cut 대조 대비 인과 기여만 살아남으며 성능 우위 또는 생물학적 뇌 동일성의 증거가 아니다(`28_Nested_Infinite_SCC_V9.md` §13).

각 장에서 CE 원리를 구체적 구현으로 변환할 때, 어디까지가 코어에서 연역된 것이고 어디부터가 설계 선택인지, 그리고 무엇이 아직 실험 가설인지를 명시한다.
