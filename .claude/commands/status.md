---
description: 연구 현황 보고 (진전 원장 §2·§6·§7 + ledger 요약 + git)
---

현재 CE 연구 현황을 보고하라. 서브에이전트 없이 직접 한다.

1. `paper/진전_원장.md` §2(표적·완료 조건·kill 조건·갱신일)·§6(트랙)·§7(최근 3행)을 읽는다.
2. `.claude\hooks\python.cmd python .claude\hooks\lib\ledger.py summary`로 active 질문·추측 카드·최근 항목·escalated를 보고, `ledger.py ladder <Q>`로 사다리 진행(닫힌 단/전체)·다음 단·`force_pivot`·재발견 횟수를 본다.
3. `git status --short`와 `git log -5 --format='%h %ad %s' --date=short`로 미커밋 변경과 최근 작업을 확인한다.
4. 기계 상태와 이론 지위를 분리해 보고하고, §2 표적에 대한 다음 최소 작업을 한 줄로 제안한다(카드가 없으면 그 작업은 "예측식·예산식 카드 작성"이다). 진전 원장 갱신일이 마지막 커밋보다 오래됐으면 첫 줄에 적는다.
