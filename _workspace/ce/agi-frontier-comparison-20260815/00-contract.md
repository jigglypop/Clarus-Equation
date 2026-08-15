# CE-AGI 최신 연구 비교 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/repository-code-analysis-20260815

## 질문

2026-08-15 기준 AGI·범용 에이전트 연구의 최신 1차 결과와 현재 Clarus-Equation 저장소의 AGI 연구를 비교한다. 단순한 용어 유사성이 아니라 구조, 실행 가능한 구현, 내부 검증, 외부 benchmark, 규모와 재현성의 단계로 나눠 다음을 판정한다.

1. 현재 CE-AGI가 실제로 구현·검증한 결과는 무엇인가.
2. 최신 연구가 수렴하는 방향 중 CE가 구조적으로 따라가는 부분은 무엇인가.
3. 개념만 비슷하고 empirical parity가 없는 부분은 무엇인가.
4. CE에 독자적이지만 아직 외부 증거가 없는 가설은 무엇인가.
5. 가장 적은 자유도와 비용으로 비교 가능성을 높이는 다음 실험은 무엇인가.

## 범위

- 내부 정본: `docs/7_AGI/` 전체, 관련 `docs/6_뇌/`, `reality_stone/python/reality_stone/clarus/`, `examples/agi/`, `tests/`, `experiments/`, 최신 비-RBE 선행 감사 결과
- 외부 기준: 2024-01-01 이후 공개된 논문·기술보고서·공식 benchmark/evaluation을 우선하고, 필요한 기반 연구만 이전 연도로 확장
- 비교 축: recurrent/latent computation, memory·continual/test-time learning, sparse/local learning·efficiency, causal/OOD generalization, world models·embodied agents, metacognition·safety/evaluation
- 명시적 제외: `RBE/` 전체, 마케팅 기사·2차 요약·기억에 의존한 최신 수치, 비공개 주장, consciousness의 현상학적 실재 판정
- 스냅샷: 내부 작업 트리와 외부 공개 자료 모두 2026-08-15 기준

## 비교 등급

외부 연구와의 관계는 다음 5단계로만 표기한다.

| 등급 | 뜻 |
|---|---|
| CR0 | 용어 또는 직관만 유사하며 대응 mechanism이 고정되지 않음 |
| CR1 | 구조적 가설과 측정량이 명시됨 |
| CR2 | 실행 가능한 구현과 내부 controlled test가 있음 |
| CR3 | 공개 benchmark, 강한 baseline, ablation, held-out 평가로 직접 비교 가능 |
| CR4 | 독립 재현 또는 동급 규모에서 frontier 경쟁력이 확인됨 |

`따라간다`는 CR1 이상의 구조 정렬을 뜻하고, `경쟁한다`는 CR3 이상에서만 사용한다. benchmark 이름이 같지 않거나 데이터·compute·모델 규모가 다르면 성능 수치를 직접 비교하지 않는다.

## 활성 주장

- **C1 [미완성]**: CE의 recurrent SCC, local/cloud kernel, episodic/local memory, delayed credit, finite-host L3-L8이 최신 recurrent computation·memory 연구와 구조적으로 정렬되는가.
- **C2 [미완성]**: sparse/STDP/sleep/metric regularization이 최신 효율·test-time learning·continual learning 결과와 mechanism 및 evidence 수준에서 정렬되는가.
- **C3 [미완성]**: V7-V13 causal/OOD route가 최신 world-model·embodied-agent·reasoning benchmark의 외부 일반화 수준을 따라가는가.
- **C4 [가설]**: nested SCC와 self-recursive/metacognitive loop가 frontier의 recurrent depth·latent reasoning·self-correction보다 검증 가능한 독자 예측을 제공하는가.
- **C5 [산출 목표]**: 각 축의 CR 등급, 보존 가능한 좁은 결과, 격차와 우선 실험을 근거에서 산출할 수 있는가.

## 증거 규칙

- 외부 최신 사실은 논문 본문, 저자/기관 공식 기술보고서, benchmark 공식 페이지로 확인하고 URL·공개일·접근일을 남긴다.
- 내부 결과는 파일:1-based 줄, 실행 명령, artifact/manifest 상태를 근거로 한다.
- 문서의 `closed`, `confirmation`, `supported` 표기는 독립 benchmark 성능과 동일시하지 않는다.
- synthetic toy, same-distribution confirmation, held-out test, external benchmark, embodied deployment를 분리한다.
- 수치 일치와 구조 유사성은 원인·우위의 증거가 아니다.
- 선행 감사에서 확정한 V17 broad-API 반례와 backend delay mismatch는 재유도하지 않고 영향만 인용한다.

## 허용 오차와 판정

- 외부 benchmark 수치는 원 출처의 reported uncertainty/official scoring을 그대로 사용한다.
- 내부 수치는 manifest가 없거나 현재 작업 트리로 재현되지 않으면 `문서 기록`, 실행되면 `재현`, 외부 비교 조건이 맞으면 `비교 가능`으로 구분한다.
- 모델·데이터·compute가 다른 성능 차이는 우열 수치로 합산하지 않는다.
- 완전한 반례 또는 외부 benchmark claim 오표기는 P0, 비교 불가능한 과장·핵심 evidence gap은 P1, 표기·최신성 문제는 P2로 분류한다.

## 산출물

- 최신 1차 연구 원장과 연구 방향 요약
- 내부 CE-AGI 결과·형식 지위·재현 상태 원장
- 축별 CR0-CR4 비교표
- 앞선 부분, 같은 방향, 뒤처진 부분, 독자 가설의 분리
- 외부 benchmark로 연결되는 우선순위 실험 3개 이상과 반증 조건
