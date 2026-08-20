# 실행 경로

Status: FROZEN

1. 원래 causal-parent 식의 receipt gate를 먼저 기록한다. 예상 판정이 아니라 필드 존재 여부 판정이다.
2. 다섯 GCaMP와 한 GFP의 calibration/construction/validation/test 역할을 audit JSON에서 재현한다.
3. A1 post-fit routing을 validation에서 실행한다.
4. A1이 계약 gate를 실패하면 영수증을 남기고 A2 anisotropic ridge를 구현한다.
5. A2 코드를 source-only/focused synthetic witness로 검증한 뒤 test를 정확히 한 번 연다.
6. 결과와 무관하게 원래 causal claim은 blocked로 유지한다.

STOP 조건은 test 재개봉, 결과 기반 threshold/cycle/parameter 변경, `acorr` 구조 사용, recording 간 row 결합, 또는 GFP를 primary 신호로 섞는 경우다.

