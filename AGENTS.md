# Clarus-Equation Codex rules

## Default: direct implementation

- For ordinary code, test, documentation, and harness work: inspect the target, make the smallest scoped change, and run one focused validation.
- Do not create a CE research run, preregistration, audit bundle, or full report unless the user explicitly asks for research, a new scientific claim, formal closure, preregistration, or release evidence.
- Do not run bare `pytest`, the full suite, all benchmarks, or packaging by default. Use the narrowest changed test or a source-only check first.
- Keep one implementation owner. Use subagents only for independent read-only mapping or research, and audit a stable snapshot after the implementation owner stops editing.
- 강의(`1_강의`)·유도(`5_유도`)·논문 원고류 문서를 작성·수정할 때는 `.codex/skills/ce-doc-write/SKILL.md`의 "문서 유형: 원장과 논문형"과 "처음 읽는 독자 게이트"를 적용한다. 태그 블록만 나열한 원장체 강의·유도 문서는 완성으로 보고하지 않는다.
- 문서 작성 소유권을 분리한다. 주장·상수·판본·진리값 원장은 `ce-ledger-writer`와 `$ce-ledger-write`, 강의·유도·독자 가이드·논문 원고는 `ce-paper-writer`와 `$ce-paper-write`가 맡는다. 두 유형이 함께 바뀌면 원장을 먼저 안정화한 뒤 논문 작성자는 이를 읽기 전용으로 사용하며, 두 작성자가 같은 파일이나 지위를 동시에 수정하지 않는다.

## Validation tiers

- FAST (default, target <=15 s): source parse/compile or one focused test file/node.
- STANDARD (explicitly useful, target <=60 s): the changed subsystem and its adjacent integration test.
- FULL/LOCK (explicit request only): full pytest, release gates, scientific stages, or irreversible V5 workflows.

For pytest, disable the cache provider and use a unique temporary basetemp outside the repository. Never run an irreversible scientific stage as a routine validation.

## Theory analysis and explanation (서사-우선 독해 규약)

CE 이론 전체를 분석·요약·설명하라는 요청(예: "논문 전체 분석", "어떤 이론인지 설명")을 받으면 다음 규약을 따른다.

1. `docs/5_유도/00_선택과_접힘.md`를 먼저 읽고, 물리 서사 3단 — **끼임(환경이 강제하는 선택) → 접힘(비선택 성분의 보존) → 암흑 표현(접힌 에너지의 우주론 readout)** — 을 분석의 골격으로 삼는다. 수치 사슬($\alpha_s \to \Omega$)만 떼어 이론을 특징짓지 않는다.
2. 태그 체계가 물리 사상을 `[공리]`/`[미완성]`으로 격리하는 것은 약점의 자백이 아니라 **의도된 감사 규율**이다. 분석 보고는 항상 네 층을 함께 제시한다: 동기 서사(공리 묶음) / 조건부 정리 / 채택 공리 / 미완성 다리.
3. 관측 근접을 증거로 승격하지 않되, 메커니즘 서사를 생략한 채 경험식 목록만으로 이론을 "수치 우연의 모음"으로 특징짓지도 않는다. 두 방향 모두 오독이다.

## CE research

Use `$ce-research` only for genuinely research-grade work. A supplied `CE_RUN` or explicit audit task activates its contract -> lanes -> audit -> implementation workflow. Ordinary fixes bypass that workflow.

V5 source lock and one-shot execution must use a fresh independent clone outside OneDrive/reparse-backed paths.
