# 뇌 방정식의 진화적 역추적 계획

## 질문

인간 뇌는 갑자기 생긴 장치가 아니라, 원시 신경계와 고대 생물의 감각-운동 회로 위에 층이 쌓인 진화체다.  
따라서 현재의 전역 뇌 방정식은 다음 질문을 받아야 한다.

> 인간의 \(U^{(d)}\), \(\Delta_G\), \(H\), \(F_{\mathrm{syn}}\) 항은 어느 진화 단계부터 보존되는가?

이 질문은 상당히 좋다. 이유는 현재 우리가 event-level로 찾은 특수 연산자들이 모두 진화적으로 오래된 축이기 때문이다.

| 현재 찾은 축 | 진화적으로 더 오래된 대응 |
|---|---|
| 시각/청각 | 광감지, 진동/기계감각, 방향탐지 |
| 통증/촉각 | 회피, 손상감지, 체성감각 |
| 얼굴정서/사회시각 | 동종 개체 인식, 위협/접근 판단 |
| 인지조절 | 행동 선택, 억제, 방향 전환 |
| 주의/경계 | 각성, 탐색, 반응 준비 |
| 작업기억 | 지연 반응, 짧은 시간의 상태 유지 |

즉 인간 특수영역은 완전히 새로 생겼다기보다, 오래된 감각-운동-각성 연산자 위에 고차 피질 계층이 붙은 형태일 가능성이 크다.

## 식의 진화적 형태

현재 전역식:

$$
P_{n+1}
=
\Pi_{\mathcal S}
\left[
(1-\rho_B)P^*
+\rho_B P_n
+\gamma\Delta_G P_n
+\sum_d a_{d,n}U^{(d)}(z_n)
+H(Q_n-Q^*)
+F_{\mathrm{syn},n}
+F_{\mathrm{slow},n}
\right]
$$

종 \(s\)를 넣으면 다음처럼 쓸 수 있다.

$$
P^{(s)}_{n+1}
=
\Pi_{\mathcal S_s}
\left[
(1-\rho_s)P^{*(s)}
+\rho_s P^{(s)}_n
+\gamma_s\Delta_{G_s}P^{(s)}_n
+\sum_{d\in\mathcal D_s}a^{(s)}_{d,n}U^{(s,d)}(z_n)
+H_s(Q_n-Q_s^*)
+F^{(s)}_{\mathrm{syn},n}
\right]
$$

진화적 보존성은 같은 domain \(d\)에 대해 다음이 작게 유지되는지로 정의한다.

$$
\mathcal C(d;s_1,s_2)
=
\left\|
\Phi_{s_1\to s_2}
U^{(s_1,d)}
-
U^{(s_2,d)}
\right\|^2
$$

여기서 \(\Phi_{s_1\to s_2}\)는 서로 다른 종의 회로/region을 공통 기능축으로 사상하는 mapping이다.

## 가장 먼저 볼 종 사다리

| 단계 | 생물/자료 | 뇌 방정식에서 볼 것 |
|---|---|---|
| 1 | `C. elegans` | 감각-중간-운동 3층 회로, 회피/탐색 |
| 2 | Drosophila | 시각, 후각, 행동선택, 학습 회로 |
| 3 | larval zebrafish | 전뇌/중뇌/후뇌, 전신 감각-운동, 각성 |
| 4 | mouse | mammalian cortex/thalamus/basal ganglia, cell type atlas |
| 5 | primate/human | 고차 피질, 언어/사회/작업기억 확장 |

핵심은 크기가 아니라 계층이다.

$$
\mathrm{sensory}
\rightarrow
\mathrm{integrative/interneuron}
\rightarrow
\mathrm{motor/autonomic}
$$

이 3층 구조가 유지되는지 본다. 이것은 현재 \(p_r=(x_a,x_s,x_b)\) 3성분과도 잘 맞는다.

## 공개 자료 후보

| 자료 | 바로 가능한 검증 |
|---|---|
| OpenWorm / C. elegans connectome | 감각-중간-운동 block Laplacian, 회피/탐색 domain 분리 |
| FlyWire/Codex Drosophila connectome | 시각/후각/운동선택 회로 motif, hub/relay 구조 |
| Fish1 zebrafish connectome | 광감각, 운동, 전뇌-후뇌 연결, 흥분/억제 표지 |
| Allen Brain Atlas / Cell Types / Mouse Connectivity | 포유류 region/cell type mapping, mouse-human 대응 |
| OpenNeuro human datasets | 인간 task-domain event/BOLD gate |

## 검증 1: 원시 3층 회로가 전역식의 최소형인가

가장 작은 형태는 다음이다.

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+\gamma\Delta_G P_n
+U_{\mathrm{sensory}}
+U_{\mathrm{homeostasis}}
\right]
$$

`C. elegans`에서는 region 대신 neuron class 또는 circuit module을 쓴다.

$$
V
=
\{
\mathrm{sensory},
\mathrm{interneuron},
\mathrm{motor}
\}
$$

성공 조건:

$$
\mathcal L_{\mathrm{block\ graph}}
<
\mathcal L_{\mathrm{flat}}
$$

즉 감각-중간-운동 block graph가 무작위/flat graph보다 실제 연결을 잘 설명해야 한다.

## 검증 2: 오래된 domain은 인간 특수영역의 뿌리인가

domain을 다음처럼 대응시킨다.

| 원시 domain | 인간 domain |
|---|---|
| light/phototaxis | visual |
| mechanosensation | tactile/pain |
| chemosensation | smell/reward/avoidance |
| escape/avoidance | salience/threat |
| locomotor selection | cognitive control/action selection |
| arousal/sleep-like state | vigilance/sleepiness |

검증식:

$$
\mathcal L_{\mathrm{matched\ evolutionary\ domain}}
<
\mathcal L_{\mathrm{wrong\ evolutionary\ domain}}
$$

이것이 통과하면, 인간 \(U^{(d)}\)가 고대 domain의 확장이라는 주장이 강해진다.

## 검증 3: 진화하면서 추가된 것은 새 식인가, 계수 지도인가

가설은 두 개다.

### A. 식 보존, 계수/그래프 확장

$$
P_{n+1}
=
\Pi[
\rho P_n+\gamma\Delta_G P_n+\sum_d U^{(d)}+...
]
$$

이 형태는 종을 넘어 유지되고, 달라지는 것은 \(G_s,\rho_s,\gamma_s,U^{(s,d)}\)다.

### B. 포유류/인간에서 새 항 추가

예를 들어 작업기억, 사회인지, 언어 같은 것은 추가 memory/workspace 항이 필요할 수 있다.

$$
P_{n+1}
=
\Pi[
\cdots
+W_{\mathrm{workspace},n}
+M_{\mathrm{episodic},n}
]
$$

검증 기준:

$$
\mathcal L_{\mathrm{base\ ancient}}
\quad\text{vs}\quad
\mathcal L_{\mathrm{base+new\ mammalian}}
$$

인간/포유류에서만 새 항이 holdout을 크게 낮추면, 그 항은 진화적으로 후기 추가항이다.

## 예상 함의

1. 전역 안정식은 매우 오래된 신경계 원리일 가능성이 있다.
2. 인간의 고차 기능은 완전히 새 식이 아니라, 고대 감각-운동-각성 연산자의 재조합일 수 있다.
3. \(U^{(d)}\)는 인간 실험실 과제명이 아니라, 진화적으로 보존된 행동-생존 domain으로 재정의해야 한다.
4. 3성분 \(x_a,x_s,x_b\)는 고등피질 전용이 아니라, 원시 회로에서도 active/structural/background 상태로 해석 가능하다.
5. 진짜 새로 생긴 것은 전역식 자체가 아니라 \(G_s\)의 깊이, recurrent loop, memory/workspace 항일 가능성이 크다.

## 바로 다음 실행 단위

가장 현실적인 첫 검증은 `C. elegans`다.

작게 닫을 수 있는 질문:

$$
\boxed{
\mathcal L_{\mathrm{sensory/interneuron/motor\ block}}
<
\mathcal L_{\mathrm{flat/random}}
}
$$

이게 통과하면, 뇌 방정식의 그래프 항 \(\Delta_G\)가 인간 뇌 이전의 원시 신경계에서도 같은 문법을 갖는다는 1차 증거가 된다.

다음 파일 후보:

- `c_elegans_connectome_gate.py`
- `brain_evolutionary_trace_report.md`

주의할 점:

이 단계는 인간 의식이나 고차인지 검증이 아니다.  
목표는 더 근본적이다. **전역식의 최소 문법이 원시 신경계에도 있는지**를 보는 것이다.

## 1차 실행 결과

`examples/physics/c_elegans_connectome_gate.py`로 OpenWorm/ConnectomeToolbox의 Witvliet adult dataset 8을 실제로 읽어 계산했다.

사용한 자료:

- 원본: `witvliet_2020_8.xlsx`
- 출처: OpenWorm ConnectomeToolbox / Witvliet et al. adult connectome
- 사용 구조: BrainMap-A의 L1/L2/L3 module assignment
- 사용 연결: module에 속한 neuron 사이 chemical/electrical synapse weight

결과:

| 항목 | 값 |
|---|---:|
| modules | 12 |
| used edges | 2102 |
| used synapses | 7222.0 |
| flat loss | 1810764.638889 |
| L1/L2/L3 block loss | 1302631.125000 |
| block / flat | 0.719382 |
| block / random mean | 0.805302 |
| permutation p | 0.025397 |
| gate | pass |

따라서 1차 결론은 다음이다.

$$
\boxed{
\mathcal L_{\mathrm{L1/L2/L3\ block}}
<
\mathcal L_{\mathrm{flat}}
\quad\text{and}\quad
p_{\mathrm{perm}}<0.05
}
$$

이 결과는 강한 최종 결론은 아니지만, 원시 신경계에도 감각-input, 중간-relay, premotor/integrative 층화가 connectome 구조에서 우연 이상으로 보인다는 1차 신호다.

진화적 함의:

- 인간에서 찾은 domain별 \(U^{(d)}\) 분리는 완전히 새 원리가 아니라 원시 회로의 감각-중간-운동 분해 위에 얹힌 확장일 수 있다.
- 그래프 항 \(\Delta_G\)는 고등피질 전용이 아니라, C. elegans 수준에서도 구조적 설명력을 갖는 후보다.
- 단순 feedforward chain은 아니다. C. elegans 회로는 recurrent/residual 성격이 강하므로, 검증 기준은 forward dominance가 아니라 layer-block reconstruction이다.

## 1차 robustness 확장

같은 검증을 weighted/binary, chemical/electrical로 나눠 다시 실행했다.

| matrix | block / flat | permutation p | pass |
|---|---:|---:|---|
| all weighted | 0.719382 | 0.026795 | pass |
| chemical weighted | 0.717370 | 0.026595 | pass |
| electrical weighted | 0.830206 | 0.089982 | fail |
| all binary | 0.914439 | 0.491102 | fail |
| chemical binary | 0.902778 | 0.365927 | fail |
| electrical binary | 0.819574 | 0.129774 | fail |

이 결과는 매우 중요하다.

단순히 “어떤 neuron끼리 연결되어 있다”는 binary topology만으로는 L1/L2/L3 층화가 강하게 나오지 않는다. 통과한 것은 weighted chemical synapse다. 즉 원시 신경계의 계층 구조는 edge existence가 아니라 **연결 강도 분포**에 실려 있다.

따라서 진화적 해석은 다음처럼 더 정밀해진다.

$$
\Delta_G
\quad\text{must use weighted chemical structure, not only binary adjacency.}
$$

전역 뇌 방정식의 그래프 항도 실제 검증에서는 binary connectome이 아니라 weighted effective/structural connectivity를 써야 한다는 함의가 생긴다.

## 2차 실행: C. elegans 발달 단계 역추적

성체 dataset 8만으로는 “나중에 생긴 구조”일 가능성을 배제하기 어렵다. 그래서 Witvliet dataset 1-8 전체를 같은 게이트로 돌렸다.

실행:

```bash
python examples\physics\c_elegans_developmental_connectome_gate.py --permutations 2000
```

결과:

| stage | synapses | chemical block/flat | permutation p | pass |
|---:|---:|---:|---:|---|
| 1 | 1235.0 | 0.756479 | 0.057971 | fail |
| 2 | 1791.0 | 0.737261 | 0.043978 | pass |
| 3 | 1957.0 | 0.734236 | 0.041979 | pass |
| 4 | 2697.0 | 0.714496 | 0.029485 | pass |
| 5 | 3958.0 | 0.728187 | 0.030485 | pass |
| 6 | 4113.0 | 0.717524 | 0.029985 | pass |
| 7 | 6624.0 | 0.717431 | 0.028986 | pass |
| 8 | 7222.0 | 0.717370 | 0.028986 | pass |

요약:

| 항목 | 값 |
|---|---:|
| passed stages | 7 / 8 |
| Spearman stage vs chemical block/flat | -0.761905 |
| Spearman stage vs synapses | 1.000000 |
| Spearman stage vs lambda max | 1.000000 |

해석:

- stage 1은 p=0.057971로 거의 경계선이다.
- stage 2부터 stage 8까지는 weighted chemical L1/L2/L3 구조가 모두 통과한다.
- stage가 올라갈수록 synapse 수와 Laplacian 최대 고유값은 단조 증가한다.
- chemical block/flat은 대체로 낮아진다. 낮을수록 block 구조 설명력이 강하므로, 발달이 진행되며 층화 구조가 안정화된다는 신호다.

따라서 더 강한 결론은 다음이다.

$$
\boxed{
\text{C. elegans의 weighted chemical layer structure는 성체에서 갑자기 생긴 것이 아니라 발달 초기에 이미 나타난다.}
}
$$

이것은 인간 뇌 방정식에 다음 함의를 준다.

전역 그래프 항 \(\Delta_G\)는 고차 피질의 후기 산물이 아니라, 신경계 발생 초기부터 등장하는 weighted chemical connectivity의 안정화 문법일 가능성이 있다.

## 3차 실행: C. elegans 다음 단계, Drosophila larva

원시 3층 회로 다음에 무엇이 생기는지 보기 위해 Drosophila larva brain connectome을 분석했다.

자료:

- `fly_larva` Netzschleuder CSV
- 원 논문: Winding et al., *The connectome of an insect brain*, Science 2023
- 노드: 2956
- edge: 116922
- synapse count: 352611

검증 질문:

$$
\mathrm{primitive\ 3\ class}
\quad\text{vs}\quad
\mathrm{primitive + mushroom\ body\ memory/action\ loop}
$$

결과:

| 항목 | 값 |
|---|---:|
| memory node fraction | 0.129229 |
| primitive block/flat | 0.921003 |
| primitive permutation p | 0.106579 |
| extended block/flat | 0.824371 |
| extended permutation p | 0.299340 |
| extended improvement over primitive | 0.104920 |
| memory-loop touched fraction | 0.325455 |
| strict gate | fail |

정확한 해석:

- mushroom body를 따로 분리하면 primitive 3-class 모델보다 block loss가 약 10.5% 감소한다.
- 그러나 extended label permutation p=0.299340으로, 엄격한 random-label 기준은 통과하지 못했다.
- 따라서 이것은 “검증 완료”가 아니라 “다음 단계 후보 발견”이다.

그럼에도 중요한 관측:

| 지표 | 의미 |
|---|---|
| memory node fraction 12.9% | 신경계의 상당 부분이 memory/action-selection 계열로 분화 |
| memory internal synapse fraction 18.4% | mushroom body 내부 loop가 강함 |
| memory-loop touched fraction 32.5% | memory 계열이 projection/lateral/action 회로와 넓게 얽힘 |
| sensory->projection / sensory->action ≈ 6.4 | 직접 반사보다 relay/projection 경로가 강함 |

따라서 C. elegans 다음 단계에서 보이는 후보는 다음이다.

$$
\boxed{
\text{primitive weighted chemical control}
\rightarrow
\text{projection relay + mushroom-body memory/action-selection loop}
}
$$

이것은 지능으로 가는 첫 추가항 후보와 연결된다.

$$
P_{n+1}
=
\Pi[
\cdots
+F_{\mathrm{syn}}
+M_{\mathrm{memory/action}}
]
$$

단, 현재 자료만으로는 \(M_{\mathrm{memory/action}}\) 항을 최종 확정하지 않는다. 다음에는 Drosophila mushroom body만 분리한 더 세밀한 connectome 또는 adult hemibrain/FlyWire에서 같은 부등식을 다시 봐야 한다.

### 예상식 반례 점검

우리 예상은 “C. elegans 다음에 mushroom-body memory/action loop가 추가된다”였다. 이 예상은 일부 지표에서는 맞지만, 엄격 검증에서는 아직 부족하다.

competing model을 비교했다.

| model | block/flat | permutation p | 해석 |
|---|---:|---:|---|
| primitive 3-class | 0.921003 | 0.111296 | 원시 감각-relay-action만으로는 약함 |
| extended memory | 0.824371 | 0.301233 | 손실은 줄지만 permutation 실패 |
| action split | 0.815008 | 0.135288 | memory보다 순수 손실은 더 낮음 |
| cell type | 0.000000 | 1.000000 | 포화모델, 과적합 상한선 |

비포화 모델 중 순수 손실 최저는 `action_split`이고, BIC-like 벌점까지 주면 `all_one`이 최저다. 따라서 현재 Drosophila larva 자료만으로는 “다음 단계가 memory loop 하나다”라고 확정하면 안 된다.

수정된 결론:

$$
\boxed{
\text{C. elegans 다음 단계에서는 memory loop 후보와 action/descending 분화가 함께 나타난다.}
}
$$

즉 다음 항은 하나가 아닐 수 있다.

$$
P_{n+1}
=
\Pi[
\cdots
+M_{\mathrm{memory}}
+A_{\mathrm{descending/action}}
]
$$

현재 가장 안전한 표현은 다음이다.

> Drosophila larva에서는 원시 weighted chemical control 위에 memory/action-selection 관련 내부 loop가 강하게 보이지만, block model 기준으로는 action/descending 분화도 같은 수준 이상의 설명력을 가진다. 따라서 우리의 예상식은 “memory 단독 추가”가 아니라 “memory + action-selection loop의 공동 분화”로 수정해야 한다.

## 4차 실행: 최초 신경계와 2스텝 모두 반례 상정

두 단계 모두에 대해 예상식이 틀렸을 가능성을 명시적으로 점검했다.

### 4.1 최초 신경계: C. elegans 반례

예상식:

$$
\mathrm{weighted\ chemical\ L1/L2/L3}
$$

반례 후보:

- L1/L2/L3가 아니라 module family가 설명한 것일 수 있다.
- 특정 축 하나, 예를 들어 taxis/avoidance/lateral만 설명한 것일 수 있다.
- 단순히 더 세밀한 module label을 쓰면 다 설명되는 포화모델일 수 있다.

결과:

| model | block/flat | p | BIC-like | 해석 |
|---|---:|---:|---:|---|
| all one | 1.000000 | 1.000000 | 1364.250 | baseline |
| layer L1/L2/L3 | 0.719382 | 0.026795 | 1356.580 | 경제성 기준 최저 |
| module family | 0.704610 | 0.266347 | 1433.110 | 손실은 더 낮지만 파라미터가 많고 p 실패 |
| lateral vs other | 0.960934 | 0.435713 | 1373.421 | 약함 |
| avoidance vs other | 0.933393 | 0.248550 | 1369.234 | 약함 |
| taxis vs other | 0.915743 | 0.180764 | 1366.485 | 약함 |
| module | 0.000000 | 1.000000 | saturated | 포화모델 |

판정:

$$
\boxed{
\text{C. elegans 최초 신경계에서는 L1/L2/L3가 가장 경제적인 coarse 설명으로 살아남는다.}
}
$$

단, 순수 손실만 보면 `module_family`가 더 낮다. 따라서 L1/L2/L3는 유일한 설명이 아니라 **최소 coarse 문법**으로 보는 것이 정확하다.

### 4.2 2스텝: Drosophila larva 반례

예상식:

$$
\mathrm{primitive}
\rightarrow
\mathrm{memory/action\ loop}
$$

반례 후보:

- memory가 아니라 action/descending 분화가 핵심일 수 있다.
- sensory modality 분화가 핵심일 수 있다.
- cell type 전체 분화가 핵심일 수 있다.
- coarse model 자체가 너무 약해서 all-one과 큰 차이가 없을 수 있다.

결과:

| model | block/flat | p | BIC-like | 해석 |
|---|---:|---:|---:|---|
| all one | 1.000000 | 1.000000 | 5034.912 | BIC-like 최저, coarse 벌점 때문 |
| primitive | 0.921003 | 0.111296 | 5054.495 | 약함 |
| extended memory | 0.824371 | 0.301233 | 5174.662 | 개선되나 p 실패 |
| action split | 0.815008 | 0.135288 | 5107.373 | 비포화 순수 손실 최저 |
| sensory modality | 0.824371 | 0.301233 | 5174.662 | memory와 같은 label 수/성능 |
| cell type | 0.000000 | 1.000000 | saturated | 포화모델 |

판정:

$$
\boxed{
\text{Drosophila larva에서는 memory 단독 추가 가설이 충분히 강하지 않다.}
}
$$

더 안전한 수정식:

$$
\boxed{
\mathrm{primitive\ weighted\ chemical\ control}
\rightarrow
\mathrm{cell\ type\ diversification}
\rightarrow
\mathrm{memory/action\ selection\ loop}
}
$$

또는 방정식 항으로 쓰면:

$$
P_{n+1}
=
\Pi[
\cdots
+D_{\mathrm{celltype}}
+M_{\mathrm{memory}}
+A_{\mathrm{descending/action}}
]
$$

현재 2스텝의 진짜 함의는 “memory가 생겼다” 하나가 아니라, **세포형 분화, action 출력 분화, memory loop가 동시에 나타나며 이 중 무엇이 가장 근본인지는 더 세밀한 adult fly/FlyWire 자료에서 다시 검증해야 한다**는 것이다.

## 수정된 전체 가설

처음 가설:

$$
\mathrm{primitive}
\rightarrow
\mathrm{memory}
\rightarrow
\mathrm{intelligence}
$$

수정된 가설:

$$
\boxed{
\mathrm{weighted\ chemical\ control}
\rightarrow
\mathrm{celltype/action/memory\ differentiation}
\rightarrow
\mathrm{stable\ recurrent\ workspace}
}
$$

즉 양적 증가보다 먼저 오는 것은 **기능 분화**다. memory는 그 중 하나이며, action-selection과 분리해서 생각하면 안 된다.

## 5차 실행: 최초 신경계-행동 proxy

인간 행동은 아직 무리지만, C. elegans에서는 connectome만으로 자극-domain이 행동-output domain으로 보존되는지 1차 proxy를 만들 수 있다.

검증식:

$$
\mathrm{Flow}(L1_d\to L3_d)
>
\mathrm{Flow}(L1_d\to L3_{d'\ne d})
$$

여기서 flow는 row-normalized weighted chemical graph의 direct path와 two-step path를 평균한 값이다.

실행:

```bash
python examples\physics\c_elegans_stimulus_behavior_gate.py --permutations 5000
```

결과:

| matrix | matched/wrong | p | pass |
|---|---:|---:|---|
| chemical weighted | 3.431872 | 0.034393 | pass |
| all weighted | 3.357210 | 0.034393 | pass |
| all binary | 1.113045 | 0.034393 | fail |

효과크기 기준을 \(matched/wrong>1.5\)로 두었기 때문에 binary graph는 실패한다.

이 결과의 의미:

$$
\boxed{
\text{C. elegans에서는 weighted chemical graph가 자극 domain을 같은 output domain으로 보존하는 구조 경로를 갖는다.}
}
$$

즉 최초 신경계 분석은 단순 구조 계층을 넘어서, 행동 proxy까지 어느 정도 가능하다.

단, 이것은 실제 행동 기록이 아니라 connectome proxy다. 최종 행동 방정식은 실제 자극-행동 trial 자료와 함께 검증해야 한다.

## 6차 실행: C. elegans 자극-행동 channel 발달 확장

adult dataset 8에서 통과한 자극-domain to output-domain 구조 proxy를 Witvliet dataset 1-8 전체로 확장했다.

실행:

```bash
python examples\physics\c_elegans_developmental_stimulus_behavior_gate.py --permutations 5000
```

결과:

| stage | chemical matched/wrong | p | chemical pass | binary matched/wrong | binary pass |
|---:|---:|---:|---|---:|---|
| 1 | 3.186427 | 0.034393 | pass | 1.446780 | fail |
| 2 | 2.600517 | 0.034393 | pass | 1.264549 | fail |
| 3 | 3.302136 | 0.034393 | pass | 1.380026 | fail |
| 4 | 3.464723 | 0.034393 | pass | 1.328685 | fail |
| 5 | 3.297920 | 0.034393 | pass | 1.208942 | fail |
| 6 | 2.995919 | 0.034393 | pass | 1.193745 | fail |
| 7 | 3.428513 | 0.034393 | pass | 1.040436 | fail |
| 8 | 3.431872 | 0.034393 | pass | 1.113045 | fail |

요약:

| 항목 | 값 |
|---|---:|
| chemical weighted pass | 8 / 8 |
| binary pass | 0 / 8 |
| mean chemical matched/wrong | 3.213504 |
| Spearman stage vs matched/wrong | 0.476190 |

함의:

$$
\boxed{
\text{C. elegans의 stimulus-output domain channel은 발달 초기부터 안정적이며, binary가 아니라 weighted chemical graph에 실려 있다.}
}
$$

이 결과는 최초 신경계에서 “행동으로 이어지는 domain routing”이 매우 이른 단계의 핵심 기능일 수 있음을 시사한다.
