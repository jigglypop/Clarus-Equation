---
name: ce-validate
description: CE의 고정점·상수 원장·무차원성·브리지·회귀 테스트를 실행해 수치 무결성을 보고하며, 그 기계 결과를 이론의 수학적 지위와 분리하는 검증 하네스.
---

# CE 검증 하네스 실행

이 스킬은 계산과 구현의 회귀를 찾는다. 결과는 정리의 증명이나 공리의 정당화가 아니며, canonical Markdown의 형식적 지위는 `ce-closure-gate`가 정한다.

## 기준 명령

저장소 루트의 PowerShell에서 실행한다.

```powershell
python reality_stone\python\reality_stone\clarus\bootstrap_solver.py
python tests\scorecard.py
python tests\run_validation.py
python examples\physics\proof_completion_attempt.py
uv run --extra dev python -m pytest tests\test_bootstrap_solver.py tests\test_dimensionless.py tests\test_layer_a.py tests\test_bridge_gates.py -q
```

필요한 경우 범위를 확장한다.

- 무차원: `python -m pytest tests\test_dimensionless.py -q`
- bridge 계산: `python -m pytest tests\test_bridge_gates.py tests\test_ckm_vcb_nlo_gate.py -q`
- 우주론 비율: `python -m pytest tests\test_cosmology_ratio_audit.py -q`
- 전체 회귀: `uv run --extra dev python -m pytest -q`

## 절차

1. `git status --short`와 `git diff --stat`으로 기존 변경을 확인한다.
2. 수정 영역의 작은 테스트부터 실행하고 정본 하네스와 전체 회귀로 넓힌다.
3. 각 명령의 원래 상태 문자열, 잔차, 표본 수, 제외 수와 환경 오류를 그대로 기록한다.
4. 이전 기준과 달라진 수치를 최상단에 보고한다.
5. 외부 fixture 부재와 계산 불일치를 구분한다.

## 해석 규칙

- 무차원 검사와 수치 일치는 구현 무결성의 증거일 뿐 물리적 참의 증명이 아니다.
- scorecard는 관측 스냅샷과의 수치 비교다. 형식적 출처는 별도로 `정의/정리/공리/산출/경험식/미완성/예측`으로 기록한다.
- 외부 입력과 비채점 항목을 분모에 넣지 않는다. `alpha_s`는 외부 입력이다.
- 부호 있는 sigma 잔차를 보존하고 공분산·상류 입력 오차의 포함 여부를 명시한다.
- canonical 문서에는 하네스의 상태 문자열을 판정 배지처럼 복사하지 않는다.
- 완전한 반례가 잠근 폐기 부모 주장은 문서에 되살리지 않고 회귀 테스트에서만 감시한다.

## 보고 형식

다음을 간결히 보고한다.

- 변경 영역과 실행한 명령
- bootstrap 잔차
- scorecard의 전체/채점/외부입력 수와 주의가 필요한 최대 잔차
- 무차원 검사와 관련 pytest의 성공·실패 개수
- 기존 기준 대비 회귀
- 환경 또는 fixture 때문에 실행되지 않은 항목

도구가 출력한 상태 문자열은 원문 그대로 인용할 수 있으나, 이를 수학적 지위로 해석하지 않는다.
