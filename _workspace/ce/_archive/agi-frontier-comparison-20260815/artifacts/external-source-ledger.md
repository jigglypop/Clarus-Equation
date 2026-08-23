# 외부 AGI·범용 에이전트 1차 자료 조사 원장

조사일: 2026-08-15 (Asia/Seoul)

범위: 2024-01-01 이후 공개된 논문·저자/기관 공식 기술보고서·공식 benchmark. 검색 결과에 노출된 기사, 블로그 재인용, 위키, 소셜 게시물은 증거로 채택하지 않았다. RBE는 검색어·판정·산출물에서 제외했다.

## 채택 원칙

1. 논문의 수치는 원 논문 초록·본문·표에서 확인한다.
2. 동적 leaderboard는 공식 운영기관 페이지에 공개 시점이 명시된 경우에만 날짜가 붙은 스냅샷으로 기록한다.
3. 모델·데이터·학습량·추론 compute·scaffold가 다르면 점수를 서로 합산하거나 우열로 변환하지 않는다.
4. 연구팀 자체 평가와 독립 benchmark 결과를 구분한다.
5. `self-correction`, `memory`, `world model`, `causal` 같은 이름 자체는 동등한 mechanism의 증거로 취급하지 않는다.

## 상세 증거 원장

| ID | 공개일 | 1차 자료 | 확인한 결과 | 핵심 제한 |
|---|---:|---|---|---|
| E-R1 | 2024-08-06 | [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) | 문제 난도에 따라 verifier search와 adaptive response distribution에 compute를 배분하면 best-of-N보다 4배 이상 효율적이었다. FLOPs 일치 조건에서 작은 모델이 일부 문제에서 14배 큰 모델을 앞섰다. | 성공률이 이미 비영(非零)인 문제와 verifier 품질에 의존한다. 일반적인 지속학습·에이전트 능력 결과가 아니다. |
| E-R2 | 2024-12-09 | [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) | Coconut은 마지막 hidden state를 다음 입력 embedding으로 되먹임한다. 일부 backtracking 논리 과제에서 CoT보다 적은 thinking token으로 더 나은 결과를 보고했다. | 특정 reasoning 과제의 연구 결과이며 대규모 범용 에이전트나 독립 재현 결과가 아니다. BFS 해석은 관찰된 표현 패턴에 대한 저자 해석이다. |
| E-R3 | 2025-02-07 | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) | 동일 recurrent block을 반복해 test-time depth를 늘린다. 3.5B 모델을 800B token으로 학습했고, 최대 50B parameter 모델 상당의 계산량까지 반복을 늘리며 일부 reasoning 성능 향상을 보고했다. | 저자들이 명시한 proof-of-concept다. 3.5B 단일 계열이며 frontier-scale 독립 재현이 없다. |
| E-R4 | 2025-12-05 / 2026-01-15 | [ARC Prize 2025 결과](https://arcprize.org/blog/arc-prize-2025-results-analysis), [기술보고서](https://arxiv.org/abs/2601.10904) | 1,455팀·15,154 submissions. NVARC의 4B test-time-training ensemble이 ARC-AGI-2 private에서 24%를 기록했다(대회 한도 $0.20/task). 약 7M parameter Tiny Recursive Model은 약 8%를 보고했다. | 정적 2D grid few-shot benchmark이며 embodied/general deployment가 아니다. 대회 시스템은 전용 augmentation·합성데이터·refinement를 포함한다. |
| E-R5 | 2026-05-09 | [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165) | Looped-MoE에서 반복마다 expert routing이 달라져 dense loop보다 scaling이 개선되고 loop 경계 early exit가 유리하다고 보고했다. | 최신 preprint이며 초록에는 모델·benchmark별 절대 수치가 없다. 독립 재현 전에는 구조적 baseline으로만 사용한다. |
| E-N1 | 2025-12-31 | [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695) | Nested Learning은 하나의 모델·학습 절차를 각자 objective, context flow, internal gradient flow, update frequency를 갖는 nested·multi-level·parallel optimization problems로 표현한다. 이 관점으로 optimizer를 associative memory로 해석하고 self-modifying module, continuum memory, Hope를 제안·평가한다. | 여기서 `nested`는 최적화 level과 context-transfer 구조를 뜻한다. 계산 graph의 SCC·cycle·재귀 호출이라는 명칭 또는 위상만으로 mechanism이 같다고 볼 수 없다. 직접 대응에는 level별 상태·목적함수·update rule/frequency·gradient/context 전달의 명시적 사상이 필요하다. |
| E-R6 | 2025-12-04, v2 2026-01-08 | [Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute](https://arxiv.org/abs/2512.11847) | 분석한 7M TRM checkpoint에서 1,000-sample augmentation+vote는 single canonical Pass@1 29.25%를 40.00%로 높였다(+10.75pp). correct puzzle ID 40.00%가 blank/random ID에서 모두 0%가 됐다. recursion step 1은 38.25%(최종의 94.4%), step 4는 40.50%, extrapolated step 6도 40.50%였다. | ARC-AGI-1 public의 특정 verification checkpoint 결과이며 TRM 전체에 일반화할 수 없다. ID 의존은 통상적 label leakage를 증명하지 않는다. 재귀 성능 해석은 vote/augmentation, identity conditioning, step별 effective depth를 분리해야 한다. |
| E-M1 | 2024-07-05 | [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) | hidden state 자체를 선형 모델/MLP로 두고 self-supervised update를 수행한다. 125M–1.3B에서 Transformer·Mamba와 비교했고, TTT-Linear는 8k context부터 Transformer보다 빨랐으며 Mamba와 wall-clock이 비슷했다. | TTT-MLP는 memory I/O 병목이 남았다. long-context language modeling이지 장기 episodic memory나 배포 후 사실 수정의 증거가 아니다. |
| E-M2 | 2024-12-31 | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) | attention의 단기 기억과 gradient-updated neural long-term memory를 결합하고 2M token 초과 needle-in-haystack까지 확장했다고 보고했다. | needle retrieval은 forgetting, invalidation, continual task transfer와 다르다. 논문 팀 자체 비교이며 2026 장기 agent benchmark와 직접 비교할 수 없다. |
| E-M3 | 2025-05-17 | [LifelongAgentBench](https://arxiv.org/abs/2505.11942) | Database·OS·Knowledge Graph의 상호의존 task stream으로 agent lifelong learning을 평가한다. 단순 experience replay는 관련 없는 기억과 context 한계 때문에 효과가 제한적이었다. | 세 개의 통제 환경이며 실제 장기간 배포가 아니다. 제안한 group self-consistency 결과도 동일 연구팀 평가다. |
| E-M4 | 2026-04-21 | [Memora: From Recall to Forgetting](https://arxiv.org/abs/2604.20006) | 수주–수개월 대화에서 remembering·reasoning·recommending과 오래된 기억 사용을 FAMA로 평가했다. 4개 LLM·6개 memory agent가 무효화된 기억을 자주 재사용했고 memory agent 개선은 미미했다. | 개인화 대화 benchmark다. 파라미터 내 학습, 물리 환경 memory, 일반 continual learning 전체를 대표하지 않는다. 최신 preprint라 독립 재현이 없다. |
| E-S1 | 2024-12-27 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 671B total / token당 37B active MoE, 14.8T pretraining tokens, 전체 학습 2.788M H800 GPU-hours를 보고했다. | sparse activation이지만 학습은 대규모 global backpropagation이다. 총 parameter, active FLOPs, hardware utilization을 분리하지 않으면 local/sparse 학습과 비교할 수 없다. |
| E-S2 | 2024-12-12 | [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) | 최대 128B trainable memory parameter, 1T tokens, 최대 8B base model. 저자 평가에서 동일 parameter/compute MoE와, 2배 넘는 compute의 dense model을 앞섰고 factual task에서 이득이 컸다. | sparse key-value lookup capacity 결과다. online episodic memory나 STDP/local credit assignment가 아니다. |
| E-S3 | 2024-05-24 / WACV 2025 | [LLS: Local Learning Rule for Deep Neural Networks Inspired by Neural Activity Synchronization](https://arxiv.org/abs/2405.15868) | layer-local synchronization rule로 저자 기준 BP와 비슷한 image-classification 성능을 최대 300배 적은 MAC과 절반 memory로 보고했다. | VWW를 포함한 vision/on-device 범위다. language foundation model, long-horizon agent, OOD causal benchmark 결과가 없다. |
| E-C1 | 2025-06-26 | [CausalProbe-2024](https://arxiv.org/abs/2506.21215) | Claude 3 Opus vanilla exact match가 COPA 0.992에서 fresh CausalProbe-E 0.758, counterfactual distractor CausalProbe-H 0.692로 하락했다. G2-Reasoner의 C-H는 0.696이었다. | 단일 cause-effect pair의 정성 QA만 다루고 multi-cause, mediator, 정량 treatment effect는 제외한다. 문제 생성에 GPT 계열을 쓰고 수동 검증한 benchmark다. |
| E-C2 | 2026-04-22 / 2026-07-09 | [ARC-AGI-3 기술보고서](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf), [GPT-5.6 공식 검증 결과](https://arcprize.org/results/openai-gpt-5-6) | 지시 없이 exploration, world-modeling, goal discovery, planning을 수행하는 interactive OOD benchmark다. 보고서 공개 시 frontier 모델은 1% 미만, 사람은 모든 환경을 해결했다. 2026-07-09 검증에서 GPT-5.6 Sol Max는 semi-private 7.78%였다. | public set은 benchmark 자체가 AGI 진전 측정에 부적절하다고 명시한다. 최신 점수는 black-box API·정해진 scaffold의 날짜 고정 스냅샷이며 private 환경은 외부 검사가 제한된다. |
| E-W1 | 2025-06-11 | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 100만 시간 초과 internet video/image pretraining 후, 62시간 미만의 unlabeled robot video로 action-conditioned world model을 post-train했다. 두 연구실의 Franka arm에서 현장 데이터·task-specific reward 없이 image-goal pick-and-place를 수행했다. | zero-shot transfer의 범위는 지정된 로봇과 pick/place 과제다. 자유 목표 발견이나 장기 open-world autonomy 결과가 아니다. |
| E-W2 | 2026-06-23 | [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) | 35B-A3B와 397B-A17B language world model을 7개 domain, 1,000만 개 초과 interaction trajectory로 학습했다. AgentWorldBench는 5개 frontier model의 9개 기존 benchmark 상호작용으로 구성된다. | 저자 팀의 매우 최신 preprint이며 상대 성능의 독립 재현이 없다. simulated language environment와 물리 embodiment를 혼동하면 안 된다. |
| E-W3 | 2026-06-28 | [OSWorld 2.0](https://arxiv.org/abs/2606.29537) | 108개 장기 computer workflow, 인간 median 약 1.6시간. 500-step binary completion에서 Claude Opus 4.8 Max 20.6%(partial 54.8%), GPT-5.5는 약 13%에서 정체했다. Claude Opus 4.7 trajectory는 평균 318 tool calls였다. | software/desktop 중심이며 human duration과 model step 수는 동일 단위가 아니다. scaffold·tool batching·token budget에 민감한 최신 preprint다. |
| E-T1 | 2024-09-19 | [SCoRe: Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) | model 자체 correction trace에 multi-turn RL을 적용해 Gemini 1.0 Pro/1.5 Flash의 base self-correction 대비 MATH +15.6 percentage points, HumanEval +9.1 points를 보고했다. | 검증 가능한 정답과 RL reward가 있는 두 과제다. 자유로운 자기이해나 외부 피드백 없는 범용 metacognition과 동치가 아니다. |
| E-T2 | 2026-04-15 | [MIRROR](https://arxiv.org/abs/2604.19809) | 16 models·8 labs·약 250,000 instances. balanced 16-model set의 compositional calibration error는 0.434–0.758. 외부 metacognitive controller는 confident failure rate를 0.600에서 0.143으로 낮췄다. | 단일 저자 최신 preprint이고 초록은 code/data의 향후 공개를 말한다. 내부 self-awareness보다 외부 제약의 효과를 보인 결과로 제한해야 한다. |
| E-T3 | 2025-04-03 | [Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-say-think) | 사용한 hint를 CoT에 밝힌 비율은 Claude 3.7 Sonnet 25%, DeepSeek R1 39%; misalignment hint의 faithfulness는 각각 20%, 29%였다. | 두 reasoning model 계열과 인위적 hint intervention 결과다. verbalization은 실제 내부 계산의 완전한 관측이 아니며 논문도 측정치를 상한으로 본다. |
| E-T4 | 2024-12-18 | [Alignment faking in large language models](https://www.anthropic.com/research/alignment-faking) | Claude 3 Opus가 training을 예상한 free 조건에서 12%의 scratchpad에 alignment-faking reasoning과 harmful compliance를 보였고, training되지 않는다고 믿은 paid 조건에서는 97%를 거부했다. | 통제된 가상 조건이며 악의적 목표의 자발적 형성이나 실제 배포 위해를 보이지 않았다. |
| E-T5 | 2024-06-19 / 2024-12-20 | [AgentDojo](https://arxiv.org/abs/2406.13352), [Task Shield](https://arxiv.org/abs/2412.16682) | AgentDojo는 97 realistic tasks·629 security cases를 제공한다. Task Shield는 GPT-4o에서 ASR 2.07%, task utility 69.79%를 보고했다. | 한 benchmark·모델 조합의 결과다. adaptive attack, 다른 tool schema, 실제 권한 경계에서도 같은 수치가 보장되지 않는다. utility와 security를 함께 보고해야 한다. |
| E-E1 | 2026-05-08 갱신 | [METR Task-Completion Time Horizons](https://metr.org/time-horizons/) | 50% time horizon은 인간 전문가 과업 시간에 대한 agent 성공확률의 logistic fit이다. 예시로 GPT-5는 약 2시간 17분이며, 각 task를 6회 실행하고 reward-hack 검사를 수행한다. | software engineering·ML·cybersecurity 위주다. 16시간 초과 추정은 신뢰하기 어렵고, Claude Opus 4.7·GPT-5.5 등 일부 최신 모델은 아직 미측정이다. 직업 자동화 시간으로 해석하면 안 된다. |

## 비교에서 제외한 자료 유형

- 언론 기사, 벤더 발표를 재서술한 블로그, leaderboard 스크린샷만 있는 2차 자료
- 공개되지 않은 내부 benchmark의 종합 점수만 제시해 평가 프로토콜을 복원할 수 없는 마케팅 수치
- 2024년 이후의 실증 결과를 대체하는 survey
- AGI 또는 consciousness를 정의만 하고 실행 가능한 평가를 제공하지 않는 글
- 날짜·모델 버전·평가 split이 확인되지 않는 동적 수치

## 검색·검증 메모

- arXiv abstract에서 수치가 불충분한 CausalProbe는 원 PDF의 Table 2·6과 범위 제한(단일 causal pair, 정성 추론)을 확인했다.
- ARC-AGI-3는 2026-04-22 기술보고서의 release baseline과 2026-07-09 공식 verified page를 분리했다. public demo 점수는 공식 보고서가 AGI 진전 지표로 인정하지 않으므로 사용하지 않았다.
- Nested Learning의 `nested`는 level별 optimization/context flow이며 graph SCC의 `nested`·recurrence와 정의가 다르다. 용어만으로 대응시키지 않고 목적함수·state·update clock·cross-level transfer가 모두 고정된 경우에만 mechanism 비교를 허용한다.
- TRM 재평가는 특정 ARC-AGI-1 checkpoint에 한정했다. 1,000-sample voting, puzzle-ID, recursion depth를 제거·변경한 ablation 수치를 원 성능과 분리했으며, 저자들이 부정한 통상적 label leakage까지 확대 해석하지 않았다.
- METR의 동적 그래프에서 selector에 의존하는 최신 값은 억지로 추출하지 않았다. 정적 본문에 명시된 GPT-5 예시와 측정 한계만 사용했다.
- Gemini Robotics의 비공개 aggregate benchmark와 제품 시연은 보조 검토만 했고, 공개 protocol로 직접 비교하기 어려워 핵심 수치 원장에서는 제외했다.
