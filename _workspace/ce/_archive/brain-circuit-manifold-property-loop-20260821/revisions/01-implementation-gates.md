# Revision 1 — implementation gates

Status: COMPLETE

Role: implementation

## Trigger

최초 동결 실행은 `PROPERTY_PASS`였으나 사후 감사 Gate는 `REVISE`였다. 수식 P0가
아니라 이미 기록 중인 두 domain condition이 Boolean PASS에 연결되지 않은 P1이다.

## Preserved initial evidence

| artifact | SHA-256 |
|---|---|
| `artifacts/a6_property_witness.initial.py` | `79d0ef045a0bbad460ae77aa07735bb7b75d431a7fe58457259205327274eda5` |
| `artifacts/a6_property_result.initial.json` | `3120bc215328f65633dd2fbdc14564bc0d5edb122fd535761942b5fd1665f4c5` |

## Allowed corrections

- `jacobian_rank==q`를 seed PASS에 강제하고 `PASSIVE_FULL_RANK` 또는
  `PASSIVE_RANK_UNCERTIFIED`를 기록한다.
- `||C_Gamma||_infinity<=0.48`와 path 전체 `max|W|<=0.47`를 seed PASS에
  강제한다.
- nonfinite diagnostic을 문자열로 직렬화해 strict JSON receipt를 만든다.
- source/interpreter/dependency/contract hashes를 receipt에 추가한다.
- misleading zero-response diagnostic을 실제 direct-edge-only partial derivative로
  교체한다.

## Forbidden changes

seed, dimension, delay, horizon, fixture ranges, equations, finite-difference step,
tolerances, adverse thresholds와 claim ceiling은 그대로다. 수식 revision count는 0이다.
