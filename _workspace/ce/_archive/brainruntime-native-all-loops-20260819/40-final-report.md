# BrainRuntime-native Loops 6--10 final report

Status: ABANDONED

## 대체 경로 (supersession) — 2026-08-23 사후 봉인

이 절은 2026-08-23에 추가된 사후 봉인이다. 실험을 새로 수행하지 않았고 후속 run의
기록만 인용한다. **이 보고서는 새 증거를 만들지 않았다.**

이 run은 light 후속 run
`_workspace/ce/_archive/brainruntime-native-all-loops-p1-20260819`(Status:
COMPLETE)로 대체되었다. p1은 PREDECESSOR로 이 run을 명시하고, stable-snapshot
감사가 지적한 세 가지 결함(applied causal weight invariant, Loop 8의 실제
`TemporalAuditedMemory` 선택, Loop 7 supplied-context precedence)만 교정한 뒤
Loops 6--10을 실제 Torch `BrainRuntime`에서 완주했다: Loop 6 8/8 GO, Loop 7 8/8
GO, Loop 8 Route A 0/8 STOP / Route B 8/8 GO, Loop 9 Route A 0/8 STOP / Route B
8/8 GO (shuffled control 0/8), Loop 10 8/8 GO (p1 `40-final-report.md` 그대로).
이 run의 Route A/B development artifact는 p1에서 불변 증거로 승계되었다.
confirmation seeds `98101..98132`는 p1에서도 사용자 인가 전까지 미개봉으로
남았다. 후속 연구 계약(`brain-mechanism-alternative-routes-20260819`)은 p1을
PREDECESSOR로 인용한다.

재개 조건: 이 디렉토리에서의 재개는 없다. 후속 작업은 p1(및 그 이후 원장 경로)을
PREDECESSOR로 하는 새 계약으로만 진행한다.

## 원래 기록 (그대로 보존)

The implementation and development experiment are complete. Confirmation remains unopened and
therefore the research run cannot yet be marked complete. Current development status is Loop 6
GO, Loop 7 GO, Loop 8 STOP, Loop 9 STOP, and Loop 10 GO. Route B also leaves Loops 8 and 9 at
STOP. See `31-validation.md` and the JSON artifacts for exact evidence.
