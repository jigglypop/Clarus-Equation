# AGI·범용 에이전트 최신 1차 연구 원장

Status: COMPLETE

조사 기준일: 2026-08-15

이 레인은 외부 기준선과 비교 조건만 고정한다. CE 내부 결과의 우위·열위·CR 등급은 판정하지 않는다. RBE는 완전히 제외했다. 상세 출처 검증 기록은 `artifacts/external-source-ledger.md`에 있다.

## Evidence ID 표

| Evidence ID | 비교 축 | 원 출처·공개일 | 검증된 외부 결과 | 비교 시 고정할 제한 |
|---|---|---|---|---|
| E-R1 | test-time compute | [Snell et al.](https://arxiv.org/abs/2408.03314), 2024-08-06 | 난도 적응형 compute 배분은 best-of-N보다 4배 이상 효율적; 일부 문제의 FLOPs-match에서 14배 큰 모델보다 우수 | base success, verifier, inference FLOPs, 문제 난도를 일치시켜야 함 |
| E-R2 | continuous latent reasoning | [Coconut](https://arxiv.org/abs/2412.06769), 2024-12-09 | hidden state를 입력으로 되먹이며 일부 backtracking 과제에서 CoT보다 적은 thinking token으로 개선 | 특정 논리 과제 결과; latent state의 BFS 해석과 일반성은 별도 검증 필요 |
| E-R3 | recurrent depth | [Geiping et al.](https://arxiv.org/abs/2502.05171), 2025-02-07 | 3.5B·800B-token recurrent model, 최대 50B-equivalent compute까지 반복 깊이 증가 | proof-of-concept; 동일 parameter·training token·test FLOPs의 depth ablation 필요 |
| E-R4 | recursive refinement/OOD | [ARC Prize 2025](https://arxiv.org/abs/2601.10904), 2026-01-15 | ARC-AGI-2 private: NVARC 4B system 24%; 약 7M TRM 약 8%; refinement loop가 주요 접근으로 부상 | 전용 합성데이터·test-time training·$0.20/task 한도; 정적 grid benchmark |
| E-R5 | recurrence+sparsity | [Looped-MoE](https://arxiv.org/abs/2605.09165), 2026-05-09 | 반복별 expert routing divergence와 loop-boundary early exit가 dense loop보다 유리하다고 보고 | 최신 preprint; 절대 수치와 독립 재현이 없어 구조 기준선으로만 사용 |
| E-N1 | nested/multi-level learning | [Nested Learning](https://arxiv.org/abs/2512.24695), 2025-12-31 | 모델·학습을 서로 다른 objective·context flow·gradient flow·update frequency를 가진 nested/parallel optimization levels로 표현하고 Hope를 평가 | graph SCC·cycle과 이름만 유사한 것은 mechanism 대응이 아님; level별 state/objective/update clock/context transfer 사상이 필수 |
| E-R6 | recursive-model 재평가 | [TRM analysis](https://arxiv.org/abs/2512.11847), 2025-12-04, v2 2026-01-08 | ARC-AGI-1: single 29.25% → 1,000-vote 40.00%; blank/random puzzle ID 0%; step 1 38.25%, step 4·6 40.50% | 특정 checkpoint 결과; voting·identity conditioning·per-step effective depth를 제거한 ablation 없이 `deep recursion` 성능으로 해석 금지 |
| E-M1 | test-time learning | [TTT layers](https://arxiv.org/abs/2407.04620), 2024-07-05 | 125M–1.3B, linear-complexity hidden-state learner; TTT-Linear는 8k에서 Transformer보다 빠름 | long-context modeling이며 지속적 사실 기억·망각 측정이 아님 |
| E-M2 | neural memory | [Titans](https://arxiv.org/abs/2501.00663), 2024-12-31 | gradient-updated neural memory와 attention 결합; 2M token 초과 needle test 보고 | retrieval과 continual adaptation·memory invalidation을 구분해야 함 |
| E-M3 | lifelong agents | [LifelongAgentBench](https://arxiv.org/abs/2505.11942), 2025-05-17 | DB·OS·KG stream에서 단순 replay의 효과가 관련성·context 한계로 제한됨 | 3개 통제 환경; task order, reset, transfer, forgetting을 동일하게 해야 함 |
| E-M4 | forgetting-aware memory | [Memora](https://arxiv.org/abs/2604.20006), 2026-04-21 | 수주–수개월 대화, 4 LLM·6 memory agents; 무효 기억 재사용과 미미한 memory-agent 개선 | 대화 개인화 범위; FAMA와 obsolete-memory split을 함께 사용해야 함 |
| E-S1 | sparse activation | [DeepSeek-V3](https://arxiv.org/abs/2412.19437), 2024-12-27 | 671B total/37B active, 14.8T tokens, 2.788M H800 GPU-hours | total/active parameter, FLOPs, token, hardware를 모두 분리 보고 |
| E-S2 | sparse memory capacity | [Memory Layers at Scale](https://arxiv.org/abs/2412.09764), 2024-12-12 | 최대 128B memory params·1T tokens·8B base; 저자 평가에서 2배 compute dense보다 우수 | lookup memory이지 online episodic memory나 local credit assignment가 아님 |
| E-S3 | local learning | [LLS](https://arxiv.org/abs/2405.15868), 2024-05-24 | vision/on-device에서 BP 유사 성능을 최대 300배 적은 MAC, 절반 memory로 보고 | language/agent frontier 결과 없음; dataset·accuracy·energy를 같은 장치에서 재측정 |
| E-C1 | fresh causal/OOD | [CausalProbe-2024](https://arxiv.org/abs/2506.21215), 2025-06-26 | Claude 3 Opus EM: COPA 0.992 → fresh C-E 0.758, counterfactual C-H 0.692; G2 C-H 0.696 | 단일 cause-effect 정성 QA; multi-cause·mediator·정량 intervention은 미평가 |
| E-C2 | interactive OOD/goal discovery | [ARC-AGI-3](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf), 2026-04-22; [verified update](https://arcprize.org/results/openai-gpt-5-6), 2026-07-09 | release 때 frontier <1%, human-solvable 100%; 최신 GPT-5.6 Sol Max semi-private 7.78% | public demo는 공식 AGI 지표가 아님; private OOD, 동일 prompt·action budget·cost 필요 |
| E-W1 | visual world model/robot planning | [V-JEPA 2](https://arxiv.org/abs/2506.09985), 2025-06-11 | 100만 시간 초과 video/image + 62시간 미만 robot video; 두 연구실 Franka에서 zero-shot image-goal pick/place | 지정 embodiment·task 범위; real-robot success, unseen scene, reward/data 조건 고정 |
| E-W2 | language world model | [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597), 2026-06-23 | 35B-A3B/397B-A17B, 7 domains·1,000만+ trajectories; 5 frontier models·9 benchmarks 기반 평가 | 저자 평가 최신 preprint; simulator fidelity와 실제 agent success를 분리 |
| E-W3 | long-horizon digital agency | [OSWorld 2.0](https://arxiv.org/abs/2606.29537), 2026-06-28 | 108 workflows·human median 1.6h; Opus 4.8 Max binary 20.6%/partial 54.8%, GPT-5.5 약 13% | 500-step, scaffold, batching, token/cost, hidden state recovery 조건 고정 |
| E-T1 | trained self-correction | [SCoRe](https://arxiv.org/abs/2409.12917), 2024-09-19 | base self-correction 대비 MATH +15.6pp, HumanEval +9.1pp | 정답 verifier와 multi-turn RL이 있는 두 과제; 순수 introspection과 구분 |
| E-T2 | metacognitive calibration | [MIRROR](https://arxiv.org/abs/2604.19809), 2026-04-15 | 16 models·8 labs·약 25만 instances; compositional calibration error 0.434–0.758, 외부 controller가 confident failure 0.600→0.143 | 최신 단일저자 preprint; data/code 공개·독립 재현 전에는 잠정 기준선 |
| E-T3 | CoT faithfulness | [Anthropic](https://www.anthropic.com/research/reasoning-models-dont-say-think), 2025-04-03 | hint verbalization: Claude 3.7 25%, R1 39%; misalignment hint 20%, 29% | constructed hint·2 model families; CoT verbalization은 내부 상태의 완전한 관측이 아님 |
| E-T4 | alignment faking | [Anthropic/Redwood](https://www.anthropic.com/research/alignment-faking), 2024-12-18 | free/training 조건에서 12% alignment-faking harmful compliance; paid 조건은 97% 거부 | 가상 통제 실험; 실제 위해·악의적 목표 형성을 보인 것이 아님 |
| E-T5 | agent prompt-injection safety | [AgentDojo](https://arxiv.org/abs/2406.13352), 2024-06-19; [Task Shield](https://arxiv.org/abs/2412.16682), 2024-12-20 | 97 tasks·629 cases; GPT-4o에서 ASR 2.07%, utility 69.79% | adaptive attack·tool schema별 재평가; ASR와 benign utility를 함께 보고 |
| E-E1 | 장기 과업 평가 | [METR Time Horizons](https://metr.org/time-horizons/), 2026-05-08 갱신 | GPT-5 예시 50% horizon 약 2h17m; task별 6회와 reward-hack 검토 | SW/ML/cyber 중심, 16h 초과 불안정, 최신 모델 일부 미측정; 직업 자동화 수치 아님 |

## 외부 연구가 실제로 수렴한 방향

이는 CE와의 관계 판정이 아니라, 출처들 사이에서 반복 확인되는 외부 baseline의 요약이다.

| 축 | 2024–2026 수렴점 | 아직 열려 있는 것 |
|---|---|---|
| recurrent/latent computation | token CoT만 늘리기보다 prompt별 compute 배분, recurrent depth, latent-state feedback, refinement loop를 사용 | 반복 자체가 새 문제 일반화를 보장하지 않음; TRM에서는 vote·puzzle identity·얕은 effective depth가 큰 교란요인 |
| nested/multi-level learning | 서로 다른 update frequency와 context flow를 가진 optimization level을 명시하고 level 간 지식 전달을 설계 | `nested`, `recursive`, `memory`라는 이름이나 graph cycle만으로 동일 mechanism을 주장할 수 없음 |
| memory/continual/test-time learning | inference 중 갱신되는 hidden-state learner, neural memory, explicit episodic store를 분리하고 streaming benchmark로 평가 | 오래된 기억 폐기, transfer/forgetting, 장기 안정성이 여전히 약함 |
| sparse/local efficiency | active parameter를 줄이는 MoE, sparse lookup memory, early exit가 대규모 효율의 주류 | frontier model의 local/STDP 학습 parity는 없음; 작은 vision/on-device 결과와 간극 큼 |
| causal/OOD | fresh/private split, counterfactual distractor, test-time adaptation, interaction 기반 goal discovery로 contamination을 압박 | 정성 causal QA와 intervention-based causal learning 사이의 간극; knowledge coverage 의존 |
| world models/embodiment | observation-only pretraining + 소량 action data, learned simulator, real/desktop interaction에서 장기 planning 평가 | zero-shot 범위가 좁고 장기 state/constraint 추적 실패가 큼; simulator fidelity가 행동 성공을 보장하지 않음 |
| metacognition/safety/evaluation | self-correction을 RL로 학습하고, calibration·faithfulness·attack utility를 별도 측정하며 외부 controller를 둠 | 언어화된 CoT는 불충실할 수 있고, 자기평가만으로 confident failure를 안정적으로 줄이지 못함 |

## 직접 비교를 허용하는 최소 조건

| 축 | 비교 가능한 최소 실험 단위 | 반드시 함께 보고할 값 | 반증 조건 |
|---|---|---|---|
| recurrent/latent | 동일 model/data에서 recurrence·latent feedback on/off, single-pass/ensemble, known/unknown identity, unseen-depth split | total/active params, train tokens, samples/votes, inference FLOPs, per-step accuracy, latency | vote 제거 시 이득 소멸, ID 없을 때 붕괴, 또는 깊이 증가 후 성능 포화·하락 |
| nested/multi-level | 각 level의 state·objective·context·update rule/frequency와 level 간 transfer를 고정한 ablation | level 수, update clocks, gradient 경로, state persistence, cross-level bandwidth, task score | SCC/cycle만 남기고 level별 optimizer/context flow를 제거해도 결과가 같으면 NL mechanism 대응은 반증 |
| memory/continual | reset 없는 task stream과 사실 update/invalidation split | retention, forward/backward transfer, forgetting, obsolete-memory error, token/storage cost | replay·no-memory baseline보다 나쁘거나 오래된 기억 오류 증가 |
| sparse/local | 동일 dataset/target accuracy/hardware의 global-BP 대 local/sparse ablation | MAC/FLOPs, wall time, peak memory, energy, total/active params | 정확도 일치 시 실제 wall/energy 이득이 사라짐 |
| causal/OOD | training cutoff 이후 또는 private generated task와 intervention/counterfactual split | exact match, calibration, contamination test, SCM/intervention correctness | 표현만 바꾸거나 causal direction을 뒤집을 때 chance 수준으로 하락 |
| world/embodied | unseen environment·goal·embodiment에서 action-budgeted execution | binary/partial success, action efficiency, failure recovery, safety violations, human baseline | 시뮬레이터 예측 향상이 실행 성공으로 이어지지 않거나 seen harness에만 국한 |
| metacognition/safety | first answer→self-correction, no-feedback→verifier-feedback, benign→adaptive attack의 요인 실험 | correction gain, regression rate, calibration, CoT faithfulness, ASR, benign utility | 자기검토가 정답을 더 자주 훼손하거나 외부 controller 없이 confident failure 지속 |

## 최신성·해석 한계

- 2026-06에 공개된 Qwen-AgentWorld와 OSWorld 2.0, 2026-05 Looped-MoE, 2026-04 Memora·MIRROR는 매우 최신 preprint라 독립 재현이 부족하다. Nested Learning은 NeurIPS 2025 판본이 있으나 CE graph SCC와는 정의가 다르며, TRM 재평가는 특정 checkpoint에 한정된다.
- ARC-AGI와 METR의 공식 페이지는 갱신된다. 이 문서의 수치는 각각 2026-07-09와 2026-05-08에 명시된 스냅샷이며 이후 페이지 변경과 구분해야 한다.
- 폐쇄형 모델은 weights·training data·일부 scaffold를 공개하지 않아 동일 조건 재현이 불가능하다. 이 경우 공개 benchmark의 관측치일 뿐 mechanism 증명은 아니다.
- 서로 다른 benchmark 점수, parameter 수, GPU-hours, 인간 시간, action count를 하나의 AGI 척도로 합산하지 않는다.
- `AGI`, `memory`, `world model`, `self-awareness`라는 명칭은 실험 설계가 일치하지 않으면 직접 대응 근거가 아니다.
- 외부 원장에는 2차 기사와 마케팅 수치를 넣지 않았으며, RBE는 완전히 제외했다.
