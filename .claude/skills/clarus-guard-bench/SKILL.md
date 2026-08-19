---
name: clarus-guard-bench
description: clarus-agent-guard 제품의 벤치 7종을 한 번에 돌려 스코어보드와 CI exit code를 낸다. guard 코드(capability/executor/cells/firewall)를 고친 뒤, 또는 "guard 벤치 돌려줘 / ASR 어떻게 나오나 / 회귀 있나" 류 요청에 사용. 정직한 산업 지표(false-allow/ASR/false-block)만 보고한다.
---

# clarus-agent-guard 벤치 실행

`clarus-agent-guard/`는 별도 상용 가드 제품. 벤치는 산업 지표로만 평가한다: **false-allow / ASR(attack success rate) / false-block**. AGI 지표·과장 금지.

## 실행

```powershell
cd clarus-agent-guard
python -m bench.all
```

`bench.all`이 7종을 한 번에 돌려 스코어보드 + CI exit code(0=통과)를 낸다. 개별 게이트:

- 능력 집행: `python -m bench.test_capability`
- 인터셉션: `python -m bench.test_interception`
- 주입 공격: `python -m bench.inject_run`
- 하드셋: `python -m bench.hard_run`
- 방화벽: `python -m bench.firewall_run`
- AgentDojo: `python -m bench.agentdojo_suite` (공식: `agentdojo_official.py`)
- 감사: `python -m bench.audit_check`

## 절차

1. 무엇을 고쳤는지 확인(`git status`, `git diff --stat clarus-agent-guard/`).
2. `python -m bench.all`을 돌린다. exit code와 스코어보드를 캡처.
3. 회귀(이전 통과 → 현재 실패)를 최상단에 강조.
4. **집행 경로 불변식 확인**: 변경이 detection-free 집행(`server/capability.py`, `server/executor.py`)을 탐지 의존으로 바꾸지 않았는지 본다. under-allow(breach)는 capability 0 고정이 정답.

## 보고

```
변경: <요약>   bench.all exit code: <0/n>

| 벤치 | 지표 | 값 | 통과 |
|---|---|---|---|
| inject | ASR | ... | ... |
| capability | false-allow | ... | ... |
| firewall | false-block | ... | ... |
| ... | | | |

회귀: <없음 / 항목>
결론: <한 줄, 정직하게>
```

## 금지
- 벤치를 돌리지 않고 통과 주장
- 과장된 성능/AGI 문구
- false-allow를 숨기거나 ASR를 유리하게 반올림
