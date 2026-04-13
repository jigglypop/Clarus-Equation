# 에이전트 루프 (F절) 검증 매트릭스

> 위치: `proof.md` 8절을 독립 문서로 분리.
> 정본: `docs/7_AGI/17_AgentLoop.md` F.1--F.22
> 의존: `proof.md`(상위 검증 체계), `evidence.md`(근거 판정)
>
> 이 절은 에이전트 루프(Layer A--E 바깥의 행동-관찰-비평-기억-학습-주의 순환)의 각 구성요소가 실제 뇌와 대응하는지를 4중 게이트로 판정한다.

---

### 8.1 구조 대응 검증 매트릭스 (F.1--F.13 기본 루프)

| 구성요소 | 뇌 대응 | Formal | Obs | Causal | Pred | 전체판정 | 핵심 근거 / 반증 조건 |
|---|---|---|---|---|---|---|---|
| 이완 $R$ (F.3) | 피질-시상 재귀 처리 | `pass` | `pass` | `partial` | `fail` | `partial` | recurrent processing 확립. 반복 깊이 $\leftrightarrow$ RT 아직 정량 미비 |
| $n_{\text{iter}}$ 이중 과정 (F.3) | 시스템 1/시스템 2 | `pass` | `pass` | `partial` | `fail` | `partial` | Kahneman 근거 방대. 신경 기질은 논쟁 중. 반증: RT와 무관 |
| 행동 선택 $\pi$ (F.7) | 기저핵 go/no-go, PFC motor planning | `pass` | `pass` | `pass` | `fail` | `partial` | 기저핵 경로 확립. BG 병변 시 action selection 결손 |
| 비평 $C$ (F.4) | ACC conflict monitoring, ERN | `pass` | `pass` | `pass` | `fail` | `partial` | ERN/FRN 확립 (Botvinick 2001). ACC 병변 시 error monitoring 결손 |
| 예측 오차 $c_{\text{pred}}$ | 도파민 RPE | `pass` | `pass` | `pass` | `fail` | `partial` | Schultz 1997 확립. DA 조작 시 RPE 변화 |
| 놀라움 $c_{\text{nov}}$ | 해마 novelty, LC-NE surprise | `pass` | `pass` | `partial` | `fail` | `partial` | P300, CA1 novelty signal. LC 직접 조작 데이터 아직 제한적 |
| 일관성 오차 $c_{\text{cons}}$ | 해마-PFC memory-guided correction | `pass` | `partial` | `fail` | `fail` | `partial` | 해마-PFC 상호작용 방향은 있으나 직접 분리 미흡 |
| 비평 $\to$ 학습 게이트 $g[t]$ | 도파민/NE 전역 조절 | `pass` | `pass` | `partial` | `fail` | `partial` | 3-factor rule 강함. $g = d\bar{c}/dt$ 정확한 형태는 `hypothesis` |
| 조건부 기억 인코딩 (F.8) | 놀라움 기반 해마 인코딩 | `pass` | `pass` | `pass` | `fail` | `partial` | novel events 우선 인코딩 확립. surprise $\to$ recall 우위 재현 |
| 에너지 기반 수렴 (F.5) | Hopfield attractor dynamics | `pass` | `partial` | `fail` | `fail` | `partial` | 에너지 감소 B.4로 닫힘. 뇌에서 정확한 Hopfield 대응은 `bridge` |
| 수면-루프 결합 (F.6) | SHY, 해마 replay | `pass` | `pass` | `pass` | `fail` | `partial` | sleep consolidation 확립. 정확한 $\rho$ 매핑은 `bridge` |
| 수면 압력 $= \sum \bar{c}^2$ (F.6) | homeostatic sleep pressure | `pass` | `pass` | `partial` | `fail` | `partial` | SWA $\propto$ prior wake. 비평 누적 해석은 `bridge` |
| $B$ 수축 (F.9--F.10) | synaptic renormalization | `pass` | `pass` | `pass` | `fail` | `partial` | SHY/수면 회복 확립. $\rho = 0.155$ 정확 값은 `bridge` |

### 8.1.1 확장 구성요소 검증 매트릭스 (F.14--F.22)

| 구성요소 | 뇌 대응 | Formal | Obs | Causal | Pred | 전체판정 | 핵심 근거 / 반증 조건 |
|---|---|---|---|---|---|---|---|
| STDP 3-factor (F.14) | 도파민 게이트 STDP | `pass` | `pass` | `pass` | `partial` | `partial` | Liakoni 2018, Yagishita 2014. **Pred**: CE 모델에서 STDP 유무별 성능 비교 시뮬레이션 가능 |
| 구조적 투영 Proj (F.14.3) | 시냅스 가지치기 + 스케일링 | `pass` | `pass` | `partial` | `partial` | `partial` | Turrigiano 2008. **Pred**: TopK sweep (4-6% 대역) 시뮬레이션 이미 수행 (sparsity_train_results.json) |
| $g[t]$ 이중 구조 (F.14.2) | DA phasic + tonic | `pass` | `pass` | `partial` | `fail` | `partial` | phasic/tonic DA 구분 확립. CE 정확 형태는 `hypothesis` |
| 잔류장 $\phi$ 갱신 (F.15) | DMN, spontaneous fluctuation | `pass` | `pass` | `fail` | `fail` | `partial` | 2024 PMC: DMN ALFF가 수행 안정성 예측. 2025 eNeuro: alpha-DMN coupling 확인 |
| glymphatic 세척 (F.15) | glymphatic system | `pass` | `pass` | `partial` | `fail` | `partial` | glymphatic 경로 확립. GBM에서 AQP4 붕괴 보고 |
| TopK 희소 활성 (F.16) | sparse cortical firing | `pass` | `pass` | `partial` | `partial` | `partial` | 1--5% sparse firing 확립. **Pred**: sparse sweep 이미 수행, U-자 곡선 확인 |
| 모듈 생애주기 (F.16.2) | 피질 모듈 활성/휴면 | `pass` | `partial` | `fail` | `fail` | `partial` | 4상태 자체는 설계 선택 |
| 에너지 예산 (F.16.1) | metabolic constraint | `pass` | `pass` | `pass` | `partial` | `partial` | Attwell & Laughlin 2001. **Pred**: 에너지 예산 위반 시 불안정 시뮬레이션 가능 |
| 자기일관성 C3 (F.17.1) | 자기 참조 의식 | `pass` | `fail` | `fail` | `fail` | `fail` | 수학적으로 닫힘. 뇌 관측 proxy 없음 |
| 의식 깊이 (F.17.2) | 의식 수준 | `pass` | `partial` | `partial` | `fail` | `partial` | PCI 방향은 있으나 CE 매핑은 `hypothesis` |
| 메타인지 수렴 (F.17.3) | PFC 재귀 자기평가 | `pass` | `partial` | `fail` | `fail` | `partial` | metacognitive accuracy 방향. $\rho$ 매핑 미검증 |
| 곡률 환각 억제 (F.18) | 억제 feedback | `pass` | `partial` | `fail` | `partial` | `partial` | GABAergic inhibition 확립. **Pred**: 곡률 모니터 on/off 시뮬레이션 가능 |
| 도파민 DA (F.19) | VTA/SNc | `pass` | `pass` | `pass` | `partial` | `partial` | Schultz 1997. **Pred**: $g[t]$ 제거/조작 시뮬레이션 가능 |
| 노르에피네프린 NE (F.19) | LC | `pass` | `pass` | `pass` | `fail` | `partial` | Aston-Jones 2005. 2024 review: tonic/phasic 탐색-착취 재확인. pupil proxy |
| 세로토닌 5HT (F.19) | raphe | `pass` | `pass` | `pass` | `fail` | `partial` | 2018 NatComm: DRN 5HT 광유전 $\to$ 인내 증가. 2025: model-based prediction 역할 |
| 아세틸콜린 ACh (F.19) | BF, PPT | `pass` | `pass` | `pass` | `partial` | `partial` | 2025 Cell Rep: 해마 ACh $\propto$ 속도, 새 환경에서 증가. **Pred**: donepezil + memory test |
| 작업 기억 용량 (F.20.1) | PFC sustained activity | `pass` | `pass` | `pass` | `partial` | `partial` | Cowan 2010. 2025 eLife: PFC-BG adaptive chunking. **Pred**: $T_h$ sweep 시뮬레이션 가능 |
| 주의 bottom-up (F.20.2) | exogenous attention | `pass` | `pass` | `pass` | `fail` | `partial` | pop-out, salience 확립 |
| 주의 top-down (F.20.2) | endogenous attention | `pass` | `pass` | `pass` | `fail` | `partial` | PFC-driven attention 확립 |
| 소뇌 forward model (F.20.3) | 소뇌 내부 모델 | `pass` | `pass` | `pass` | `partial` | `partial` | 2025 JNeurosci, 2026 PMC. **Pred**: 소뇌 모듈 on/off 시뮬레이션 가능 |
| theta-gamma 결합 (F.21) | 해마 sequential memory | `pass` | `pass` | `partial` | `partial` | `partial` | 2024 bioRxiv: 인간 해마 ECoG PAC-WM 상관 확인. **Pred**: 모델 내 PAC 재현 가능 |
| gamma = 국소 계산 (F.21) | communication through coherence | `pass` | `pass` | `partial` | `fail` | `partial` | Fries 2015 |

### 8.2 F절 형식 정리 현재 상태

| 정리 | 주장 | 의존 | 상태 |
|---|---|---|---|
| F-contract | 루프 수렴: $\rho + \lambda_R L_R + \lambda_C L_C < 1$ | A-bound, E-decrease | **open** ($L_R, L_C$ 추정 필요) |
| F-energy | 이완 $R$이 $E_t(z)$를 비증가 | B.4 E-decrease | **closed** |
| F-relax | $n_{\text{iter}} \to \infty$이면 $a^{(k)} \to$ 고정점 | A.7 A-bound, A.9 Zero-attract | **closed** (조건부) |
| F-memory | $\theta_{\text{encode}} > 0$이면 인코딩 빈도 유한 | D.2 유한 인코딩 | **closed** |
| F-sleep | 수면이 $\rho < 1$을 공급하므로 F-contract 성립 가능 | Sleep-stabilize (G절) | **closed** (수면 존재 시) |
| F-sparse | 활성 유계: $|A_t| \leq \lceil x_a^* N \rceil$ | Sparse-energy | **closed** |
| F-phi-bound | 잔류장 유계: $\xi < 1$이면 $\phi$ bounded | A-bound (Var 유한) | **closed** |
| F-curvature | LBO 확산 수렴 | $h_d < 1/\text{eig}_{\max}$ | **closed** |
| F-meta | 메타인지 잔차 수렴: $d_{n+1} \leq \rho d_n$ | $\rho < 1$ | **closed** |
| F-STDP-local | STDP는 국소 정보만 사용 | 정의에 의해 | **closed** |
| F-WM-finite | 작업 기억 유한: $|h_t| \leq T_h$ | 유한 창 | **closed** |

### 8.3 F절 미결 실행 항목

| # | 항목 | 우선순위 | 1차 실행 | 통과 기준 |
|---|---|---|---|---|
| 1 | $L_R$ 추정 | 높음 | 64셀 시뮬레이션에서 $R$ 반복의 Lipschitz 상수를 수치 추정 | $\rho + \lambda_R L_R + \lambda_C L_C < 1$ 확인 |
| 2 | $L_C$ 추정 | 높음 | 비평 $C$의 Lipschitz 상수를 비평 3항 분해에서 산출 | 같은 수축 조건 확인 |
| 3 | RT $\leftrightarrow$ $n_{\text{iter}}$ | 중간 | 난이도 조작 실험에서 RT와 모델 반복 횟수의 상관 | 양의 상관, $r > 0.3$ |
| 4 | ERN $\leftrightarrow$ $\bar{c}_t$ | 중간 | error monitoring 과제에서 ERN 진폭과 비평 점수 비교 | 방향 일치 |
| 5 | 수면 후 루프 안정성 | 중간 | 학습 $\to$ 수면 $\to$ 재시험에서 post-sleep 정확도 향상 | pre-sleep 대비 유의한 개선 |
| 6 | 행동 선택 계층화 | 낮음 (장기) | 단층 $\pi$를 macro-action + primitive로 분리 | 과제 복잡도 증가 시 성능 유지 |
| 7 | 정동/valence 통합 | 낮음 (장기) | $c_t$에 valence 항 추가, $V_{\text{sal}}$ 연결 | 감정 편향 과제에서 편향 재현 |
| 8 | STDP 코드 구현 | 높음 | `clarus/core` 또는 Python에서 3-factor STDP + 도파민 게이트 구현 | 토이 과제에서 학습 확인 |
| 9 | 4종 조절계 코드 | 중간 | $g[t]$를 4차원 벡터로 확장, NE/5HT/ACh 매핑 | 4채널 독립 조절 확인 |
| 10 | 잔류장 $\phi$ 구현 검증 | 높음 | $\phi$ 갱신-포탈-glymphatic 루프가 코드에서 동작 | 모드 전환 시 $\phi$ 임계 동작 확인 |
| 11 | 소뇌 forward model | 중간 | 행동 후 감각 예측 오차 보정 모듈 구현 | 적응 과제에서 보정 수렴 확인 |
| 12 | theta-gamma 결합 검증 | 낮음 | $R$ 내부 반복과 전역 동기화 주기의 위상 잠금 시뮬레이션 | phase-locking index $> 0.3$ |
| 13 | 작업 기억 $T_h$ 최적화 | 낮음 | $T_h \in \{3,5,7,9\}$ 스위프 후 과제 수행 비교 | 최적 $T_h$와 인간 WM 용량의 일치 |

### 8.4 F절 즉시 반증 조건

| # | 반증 조건 | 반증 시 조치 |
|---|---|---|
| 1 | 이완 반복 횟수가 과제 난이도/RT와 무관 | $n_{\text{iter}}$ 가변 설계를 폐기하고 고정 깊이로 전환 |
| 2 | 수면 없이도 루프가 안정적으로 수렴 | $B$ 수축의 필요성을 내림. 수면은 선택적 보조로 강등 |
| 3 | 비평 $C$를 제거해도 과제 수행이 동일 | 비평 루프를 폐기하고 단순 이완-행동 구조로 후퇴 |
| 4 | 기억 조건부 인코딩 대신 전수 인코딩이 더 효율적 | $\theta_{\text{encode}}$를 0으로 내림. 놀라움 기반 필터링 폐기 |
| 5 | ERN/ACC 신호가 $\bar{c}$와 반복적으로 반대 방향 | 비평 $C$의 뇌 대응 주장을 `hypothesis`로 강등 |
| 6 | Dense 활성(100%)이 TopK(4.87%)보다 항상 우위 | 희소 활성 설계의 효율 주장 폐기 |
| 7 | STDP를 제거하고 역전파만 써도 에너지/성능 동일 | STDP 학습 경로를 폐기. 역전파 기반으로 전환 |
| 8 | 도파민 조작으로 $g[t]$가 변해도 학습에 무영향 | $g[t]$ = 학습 게이트 주장을 `hypothesis`로 강등 |
| 9 | 소뇌 모듈 제거해도 행동 정밀도 동일 | 소뇌 forward model을 선택적 보조로 강등 |
| 10 | 작업 기억 창을 $\infty$로 해도 성능 저하 없음 | 유한 $T_h$ 필요성 폐기. 전체 이력 사용으로 전환 |
