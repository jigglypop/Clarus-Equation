# C. elegans connectome graph replay MVP: final report

Status: COMPLETE

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 실험, 계산, 감사를 새로 수행하지
않았으며, 기존 stage 파일(`20-audit.md`, `30-implementation.md`,
`31-validation.md`)의 판정을 요약만 한다. **이 보고서는 새 증거를 만들지 않았다.**

## 계약 질문

동결된 공개 adult-hermaphrodite *C. elegans* 구조 connectome을 source byte에서
canonical graph로 결정론적으로 replay하고, transport·schema·graph 무결성·공개 규모
요약 통계를 독립 검증할 수 있는가 (C1--C7).

## 감사 판정 (20-audit.md 그대로)

- `Gate: PASS` — 단, 이 게이트는 동결된 구현 envelope만 인가하며 구현 주장이 이미
  통과했다는 증거가 아니다 (감사 명문).
- 열린 P0 없음. `P0-MATH-01`은 R1 선택(파싱된 관측 컨테이너의 ordinal 보존
  permutation만 C5 정리 정의역)으로 닫혔고, 더 넓은 raw-byte permutation 해석은
  반례와 함께 삭제되었다.
- 저장소 라이선스 부재는 실제 제한이다: 이 run은 재배포 권리를 주장할 수 없고
  원본 CSV와 전체 derived artifact는 run-local로 유지해야 한다.

## 구현·검증 결과 (30/31 그대로)

- 30-implementation: COMPLETE — 감사 envelope 내에서 표준 라이브러리 replay
  라이브러리, offline CLI, immutable manifest, 합성 fixture를 구현. 이 단계는
  전체 source CSV를 읽지 않았고 전체 derived artifact를 생산하지 않았다.
- 31-validation (Status: IN_PROGRESS 그대로): focused fixture test `1 passed in
  3.18s`. 이것은 fixture 검증뿐이며 full-connectome 재현이 아니다 (파일 명문).
  독립 full frozen-byte replay는 상위 검토 단계로 남겨졌고 실행되지 않았다.

## 미완 항목 (BLOCKED)

- C6 (전체 release replay = 등록된 정확 count vector): **미실행**. 2026-08-23 기준
  `artifacts/`에는 원본 `herm_full_edgelist.csv`와 취득 노트만 있고 전체 canonical
  artifact `c_elegans_connectome_replay.full.json`은 존재하지 않는다. 계약 수용
  기준 5에 따라 fixture-only pass는 full-connectome 재현으로 기술할 수 없으므로
  C6은 BLOCKED다.
  - 재개 조건: 등록 manifest
    `experiments/preregistration/c_elegans_connectome_replay_v1.json`의 raw
    SHA-256 검증을 통과한 동결 byte에 대해 문서화된 CLI
    (`examples/brain/c_elegans_connectome_replay.py`)를 offline으로 1회 실행하고,
    정확 정수 count 비교와 canonical digest를 기록한다.
- C7 경계는 유지된다: 산출물은 구조 그래프 artifact이며 기능적 뇌 시뮬레이션,
  dynamics, 학습, 인간 connectome 주장이 아니다.

## 원장 인용

`_workspace/ce/brain-algorithm-route-ledger.md`에는 이 run을 인용하는 행이 없다
(2026-08-23 확인). 이 run은 brain-algorithm 후보가 아니라 구조 replay MVP다.
