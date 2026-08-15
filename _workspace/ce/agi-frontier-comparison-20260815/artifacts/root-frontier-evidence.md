# Root frontier evidence notes

Snapshot/access: 2026-08-15. 아래는 최종 비교용 1차 자료 원장이다. 수치는 서로 다른 benchmark 사이에서 직접 우열 비교하지 않는다.

| ID | 1차 자료 | 확인한 결과 | 제한 |
|---|---|---|---|
| F-REC-1 | Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (2025), https://arxiv.org/abs/2502.05171 | shared recurrent block을 test time에 더 반복하는 3.5B model, 800B training tokens; 일부 reasoning benchmark에서 추가 latent depth로 향상 | preprint; 대규모 사전학습과 benchmark 성능이 핵심이며 recurrence 자체가 원인이라는 충분조건은 아님 |
| F-REC-2 | Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* (2025), https://arxiv.org/abs/2510.04871 | 7M TRM, reported ARC-AGI-1 45%, ARC-AGI-2 8% | 후속 분석 https://arxiv.org/abs/2512.11847 은 1000-sample voting·puzzle identity 의존과 얕은 effective recursion을 지적 |
| F-MEM-1 | Sun et al., *Learning to (Learn at Test Time)* (2024), https://arxiv.org/abs/2407.04620 | hidden state 자체를 linear/MLP learner로 갱신; 125M–1.3B에서 Transformer/Mamba와 비교해 match/exceed 보고 | long-context sequence modeling; 영구 continual knowledge와 동일하지 않음 |
| F-MEM-2 | Behrouz et al., *Titans* (2025), https://arxiv.org/abs/2501.00663 | test-time neural long-term memory + attention; language/common-sense/genomics/time series, >2M needle context | benchmark·architecture 보고이지 인간 기억 동일성 아님 |
| F-MEM-3 | Behrouz et al., *Nested Learning* (2025), https://arxiv.org/abs/2512.24695 | nested optimization/context flow, self-modifying learner, continuum memory, Hope module의 LM/knowledge/continual/long-context 결과 | preprint; CE의 graph SCC와 이름만 같아 mechanism 대응이 필요 |
| F-MEM-4 | Kim et al., OAKS (2026), https://arxiv.org/abs/2603.07392 | streaming knowledge benchmark; 14 models와 agentic memory가 state tracking·distraction에 취약, agentic memory가 naive RAG보다 낮은 aggregate accuracy도 보고 | 합성/novel knowledge stream; 모든 continual learning을 대표하지 않음 |
| F-SPARSE-1 | DeepSeek-V3 technical report (2024), https://arxiv.org/abs/2412.19437 | 671B total, token당 37B activated(약 5.51%), learned MoE routing과 load balancing | CE 4.87%와 수치가 가깝지만 expert parameter activation과 CE neuron/weight partition은 다른 양 |
| F-SNN-1 | Pes et al., *Traces Propagation* (2025), https://arxiv.org/abs/2509.13053 | eligibility trace + local contrastive loss의 forward-only rule; NMNIST/SHD에서 local baselines 상회, DVS·speech 확장 | label-derived local targets 사용; 완전한 unsupervised biological learning 아님 |
| F-ARC-1 | ARC Prize verified leaderboard, https://arcprize.org/leaderboard | 2026-08 snapshot에서 verified frontier entry는 ARC-AGI-2 72.1%, ARC-AGI-3 1.5%까지 표시 | 모델·비용·scaffold 의존; competition private final과 구분 |
| F-ARC-2 | ARC-AGI-3 technical report (2026), https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf | novel interactive worlds에서 explore, infer goals, world model, plan; 2026-03 frontier systems <1%, humans 100% environments | benchmark는 abstract turn-based worlds이며 실제 로봇 전부를 대표하지 않음 |
| F-WM-1 | Bruce et al., *Genie* (2024), https://arxiv.org/abs/2402.15391; DeepMind Genie 3 (2025), https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ | Genie 11B unsupervised interactive environment; Genie 3는 720p/24fps, 몇 분 consistency의 real-time interactive worlds를 기관이 보고 | Genie 3는 institutional report이며 공개 재현/독립 benchmark가 제한됨 |
| F-EMB-1 | Gemini Robotics technical report (2025), https://arxiv.org/abs/2503.20020 | VLA가 unseen objects/environments, open-vocabulary instruction, new embodiment와 long-horizon manipulation을 시험 | 대규모 비공개 model/data/hardware; CE와 compute-matched 비교 불가 |
| F-AGENT-1 | METR task-completion time horizons, https://metr.org/time-horizons/ | software/ML/cyber tasks에서 human-duration 대비 50%/80% horizon을 측정; 장기 trend는 약 7개월 doubling으로 보고, 16h 이상은 현재 suite에서 불안정 경고 | 잘 정의된 기술 task 중심이며 경제 전반·자율 시간과 동일하지 않음 |
| F-META-1 | Huang et al., ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html | 외부 feedback 없는 intrinsic reasoning self-correction은 종종 향상하지 못하고 악화 | 2023-era model/prompt 범위; 모든 feedback-based correction을 부정하지 않음 |
| F-META-2 | Wang, MIRROR (2026), https://arxiv.org/abs/2604.19809 | 16 models/약 250k instances에서 compositional self-prediction 오류, external metacognitive control이 confident failure를 0.600→0.143으로 낮췄다고 보고 | 최신 단일-author preprint; code/data 공개 약속의 실제 상태를 별도 확인해야 함 |

## 비교에 쓰는 해석 규칙

1. CE의 fixed 4.87%와 MoE 5.51%는 구조적으로 다른 분모이므로 일치 증거가 아니다.
2. CE의 SCC 중첩은 graph/state construction이고, Nested Learning은 nested optimization이다. 이름 유사성은 CR0이며 update objective와 benchmark가 연결돼야 CR1 이상이다.
3. ARC-AGI-3가 요구하는 exploration·goal inference·online world-model update는 CE L3–L8 finite construction이나 V10 conditional-binding보다 넓다.
4. frontier recurrence의 긍정 결과와 TRM 후속 반론을 함께 사용한다. 반복 깊이 자체를 reasoning으로 동일시하지 않는다.
