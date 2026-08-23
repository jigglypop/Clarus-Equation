# 31-validation — BA-SRM2

Status: `SCHEMA_HARNESS_UNIT_PASS / REALDATA_EXECUTION_PENDING`

response estimator는 아직 구현ㆍ실행하지 않았다. schema-only auditor의 focused test는

```text
.codex\hooks\python.cmd pytest _workspace\ce\brain-synapse-functional-quotient-20260823\artifacts\test_audit_medium_schema.py -q
10 passed
```

이고 BA-SRM2 physical reference/kernel core의 dimensionless test는

```text
.codex\hooks\python.cmd pytest tests\test_dimensionless.py::test_ba_srm2_event_history_target_and_kernel_are_dimensionless -q
1 passed
```

이다. 이는 harness와 차원 정합의 기계 검증일 뿐 real-data PASS나 생물학적 결과가
아니다. medium 다운로드, SHA-256, full integrity와 실제 schema audit가 남아 있다.
