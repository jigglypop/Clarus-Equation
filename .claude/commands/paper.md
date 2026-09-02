---
description: 연구 원장의 L3+ 항목을 논문 정본(paper/)에 반영 (외부 루프, 원장→원고 두 단계)
---

paper-writer 서브에이전트를 Agent 도구로 띄워 다음을 수행하라. 오케스트레이터는 직접 쓰지 않는다.

대상: $ARGUMENTS (분야/논문 또는 장 경로)

1. `ledger/index.md`와 `ledger/entries/*.yaml`에서 `status: resolved` 질문의 L3 이상 항목만 고른다. L2 이하는 목록으로만 보고하고 인용하지 않는다.
2. 1단계(원장): 해당 `paper/검증_원장/*.md`와 `paper/진전_원장.md` §3–§5에 지위 태그와 각주 `[E-YYYYMMDD-NNN]`으로 옮긴다. 편집을 멈춘다.
3. 2단계(원고): 안정화된 원장을 읽기 전용으로 삼아 대상 장을 제자리 갱신한다. 장 머리 세 줄 상자, 결과 우선, 영어 용어 병기(ko-academic-prose). 충돌은 고치지 말고 보고한다.
4. 탈고 전 `.claude\hooks\python.cmd links`와 `lint`를 실행하고 결과를 적는다.
5. 마지막 메시지: 인용한 항목 id, 제외한 항목 id와 등급, 갱신한 파일, 링크·lint 결과, 진전 원장 §2와의 연결 한 줄.
