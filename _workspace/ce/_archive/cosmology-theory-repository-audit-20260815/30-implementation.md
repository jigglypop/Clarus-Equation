Status: SKIPPED (read-only audit; no product source change was requested)

# 30 — 구현 범위

`20-audit.md`의 형식 감사는 `Gate: PASS`로 닫혔다. 이 PASS는 반례가
있는 부모 주장을 활성 결론에서 제외했다는 뜻이며, 소스에 남은 잘못된
지위 문자열·수치 경로·exit policy를 고쳤다는 뜻은 아니다.

이번 요청은 저장소 전체의 우주론 이론 검증이므로 제품 코드, 테스트,
정본 문서는 수정하지 않았다. 기존 dirty worktree와 사용자의 변경을
그대로 보존했고, 이 run 아래 감사 문서와 독립 검산 artifact만 추가했다.

후속 구현이 승인될 경우 범위는 다음으로 제한해야 한다.

1. 밀도분율의 `ce_prediction` 지위를 공리·경험식으로 정정한다.
2. 복사·바리온을 실제로 사용하는 계산 전까지 Hubble toy 수치 closure를
   격리하고 회귀 반례를 추가한다.
3. log-grid cumulative 적분을 비균일 격자 적분으로 교체하고 수렴 시험을
   추가한다.
4. covariance의 finite·SPD/domain 검사를 fail-closed로 만들고,
   `REJECT`·`CAUTION`을 report-only 계약 또는 명시적 exit policy에 연결한다.
5. 잘못된 DESI 요약값과 식별 불가능한 hybrid baseline을 versioned
   likelihood·chain provenance로 교체한다.

구현하지 않은 항목을 이 파일에서 완료로 표시하지 않는다.
