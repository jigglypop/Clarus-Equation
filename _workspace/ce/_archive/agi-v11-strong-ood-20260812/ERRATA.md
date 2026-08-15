# ERRATA — agi-v11-strong-ood-20260812

날짜: 2026-08-12 (사후 감사에서 발견, 원본 파일은 해시 보존을 위해 미수정)

1. `artifacts/post-run-audit.md`는 "Ten of fourteen primary gates failed"라고 기술하나,
   `artifacts/ood-development-result.json`의 `result.gates` 실제 엔트리는 **13개**다
   (4 strong_noninferiority + 4 compute_matched_superiority + 4 accuracy + 1 integrity).
   실패 게이트 수 10개는 정확하다. "fourteen"은 `overall` 판정을 게이트로 중복 계수한
   장부 오류이며, 판정(STOP)에는 영향이 없다.

2. 이 errata는 원본 결과·등록 JSON과 문서를 변경하지 않는다.
