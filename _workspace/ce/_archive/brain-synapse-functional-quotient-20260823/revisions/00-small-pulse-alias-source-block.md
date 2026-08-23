# Revision witness — small pulse-grid source block

Date: 2026-08-23

Outcome contact: train JSON structural equality only; no magnitude fit/score; confirmation untouched

최초 초안은 `stp_all_stimuli.pulse_amplitudes`를 pulse별 좌표로 읽어 36D input과
36D disjoint-protocol target을 제안했다. 독립 source lane과 Gate auditor가 결과값을
열기 전에 producer의 Python alias bug를 확인했다.

```python
collect_pulse_amps = [[]] * 12
```

이 한 줄 때문에 12 slot은 같은 list를 공유한다. 모든 pulse가 하나의 list에
누적되고 같은 aggregate가 12번 저장된다. train-only 구조 equality 검사에서 모든
protocol record가 이 결함을 재현했다.

따라서 최초 grid, FPCA dimension menu, RBF fit과 ELPD 계획은 실행되지 않았고 active
contract에서 제거했다. 보존되는 것은 source-level 반례와 medium event-row requirement다.
