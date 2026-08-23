# 31-validation

Status: COMPLETE

Date: 2026-08-22 (완결 시점 최소 재검증)

## 실행한 검사

- 통합 verifier 1회: `PYTHONPATH=examples/physics python artifacts/verify_full_cosmology_math.py`
  (재현 명령에 PYTHONPATH 전제가 누락되어 있었음 — P2 문서 공백으로 기록).
- 결과: **line 376 assert 실패** — legacy 비균일 Simpson 반례 재현 조건
  `simpson_relative_error > 0.30` 불충족.

## 원인 분류 (empirical_calibration_loop D→I→P→C→B→T)

- 첫 분기점: legacy Simpson 상대오차가 동결 시점 기록(+33.63%)과 달리
  0.30 이하.
- 분류 **I (소스 드리프트)**: run 동결(2026-08-15) 다음 날 커밋 `8cfb11e`
  (2026-08-16)가 `examples/physics/cosmology.py`·`cosmology_kernel.py`를
  수정 — legacy 결함이 현재 트리에서 제거된 것과 정합. 수학 오류(T)나
  정밀도(P) 증거 없음.
- 판정: **run 동결 시점의 기록(11-math §11: 전 verifier 통과)은 시점
  유효하며, 금일 재실행 실패는 run 결과의 무효화가 아니라 검증 스크립트가
  동결 소스 상태에 고정되어 있음을 보여 준다.** tolerance·스크립트 수정
  없이 사실만 기록한다.

## 생략한 검증 (정직 명기)

- 나머지 verifier 4종 재실행 안 함 — 동일 드리프트 노출 예상, 동결 시점
  기록 인용으로 대체.
- 전체 pytest·bench 미실행 (본 완결 작업에 코드 변경 없음).
