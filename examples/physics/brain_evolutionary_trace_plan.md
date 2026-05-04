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
