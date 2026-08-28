---
name: ce-validate
description: CE 런타임의 무차원성·브리지·계층·회귀 테스트를 실행해 수치 무결성을 보고하며, 그 기계 결과를 이론의 수학적 지위와 분리하는 검증 하네스.
---

# CE 검증 하네스 실행

이 스킬은 계산과 구현의 회귀를 찾는다. 결과는 정리의 증명이나 공리의 정당화가 아니며, canonical Markdown의 형식적 지위는 `ce-closure-gate`가 정한다.

## 범위 선택

아래 목록을 일괄 실행하지 않는다. 요청·변경 파일에 직접 대응하는 **한 항목만** 먼저 고른다.

- 무차원: `python -m pytest tests\test_dimensionless.py -q`
- bridge 계산: `python -m pytest tests\test_bridge_gates.py -q`
- 런타임 계층: `python -m pytest tests\test_layer_a.py -q` (변경 계층에 맞춰 a–e 선택)
- 런타임 상수·수렴: `python -m pytest tests\test_convergence.py -q`
- 전체 회귀: 사용자가 `전체`, `full`, `release`, `CI 재현`을 명시한 경우에만 `python -m pytest -q`

명령 예시의 `python`은 프로젝트의 기존 interpreter를 뜻한다. 실제 pytest 실행에는 `-B`,
`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, 실행별 고유 `--basetemp`를 추가한다.
현재 interpreter가 필요한 dependency를 이미 제공하면 `uv run`으로 다시 감싸지 않는다.
Windows agent 실행에서는 먼저 `.codex/hooks/python.cmd doctor`를 확인하고, pytest는 같은 래퍼의 `pytest` 모드로 실행한다. 차단된 `.venv`나 대화형 uv 프롬프트를 기다리거나 우회하지 않는다.

`reality_stone` 제거 후 저장소 전체의 무기록 Python 구문 검사는 `.codex/hooks/python.cmd source`를 사용한다. 이 모드는 `.codex`, `tests`, `examples`, `experiments`, `docs`의 `.py`를 메모리에서 AST 파싱하며 pyc나 cache를 만들지 않는다. focused pytest가 제거된 모듈을 요구하면 환경 실패로 기록하고 독립 실행 가능한 가장 가까운 테스트를 선택한다.

`reality_stone` 제거 후 저장소 전체의 무기록 Python 구문 검사는 `.codex/hooks/python.cmd source`를 사용한다. 이 모드는 `.codex`, `tests`, `examples`, `experiments`, `docs`의 `.py`를 메모리에서 AST 파싱하며 pyc나 cache를 만들지 않는다. focused pytest가 제거된 모듈을 요구하면 환경 실패로 기록하고 독립 실행 가능한 가장 가까운 테스트를 선택한다.

## 절차

1. `git status --short`와 `git diff --stat`으로 기존 변경을 확인한다.
2. 수정 영역의 가장 작은 테스트 하나를 실행한다. green이면 검증을 종료한다.
3. 작은 검사가 실패했거나 공용 API·의존성 경계를 바꾼 경우에만 관련 회귀 한 단계로 넓힌다. 전체 회귀는 사용자 명시 요청 없이는 금지한다.
4. 각 명령의 원래 상태 문자열, 잔차, 표본 수, 제외 수와 환경 오류를 그대로 기록한다.
5. 이전 기준과 달라진 수치를 최상단에 보고한다.
6. 외부 fixture 부재와 계산 불일치를 구분한다.

## cache·임시물 규율

- Python bytecode와 pytest/Ruff cache 생성을 끈다: Python `-B`, `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`, Ruff `--no-cache`.
- pytest basetemp는 실행마다 새 소유 경로를 만들고 `finally`에서 삭제한다. 고정 `.pytest_tmp_*`를 재사용하지 않는다.
- 문법 검사는 exact source를 메모리에서 `compile()`한다. `compileall`과 `py_compile`로 pyc를 남기지 않는다.
- cache를 지우기 위해 `.tmp`, `.venv`, `_workspace`, 사용자 artifact를 광역 삭제하지 않는다. 이번 실행이 만든 exact path만 제거한다.
- 동일 byte에서 이미 green인 명령은 다시 실행하지 않는다. 다른 레인의 재현 가능한 green 로그가 있으면 인용한다.

## 해석 규칙

- 뇌/AGI 검증은 `.codex/harnesses/real_brain_equation_discovery_loop.md`의 생물 baseline 재현, measurement-model parity, 사전 고정 held-out residual, 대안 모델, falsifier, matched control과 CE 추가항 ablation을 우선 보고한다. post-hoc refit은 검증 PASS가 아니라 calibration artifact이며 새 판본 계약과 독립 confirmation을 요구한다.
- 무차원 검사와 수치 일치는 구현 무결성의 증거일 뿐 물리적 참의 증명이 아니다.
- 부호 있는 sigma 잔차를 보존하고 공분산·상류 입력 오차의 포함 여부를 명시한다.
- canonical 문서에는 하네스의 상태 문자열을 판정 배지처럼 복사하지 않는다.
- 완전한 반례가 잠근 폐기 부모 주장은 문서에 되살리지 않고 회귀 테스트에서만 감시한다.

## 보고 형식

다음을 간결히 보고한다.

- 변경 영역과 실행한 명령
- 무차원 검사와 관련 pytest의 성공·실패 개수
- 기존 기준 대비 회귀
- 환경 또는 fixture 때문에 실행되지 않은 항목
- 실행하지 않은 전체 회귀와 그 이유
- 종료 후 하네스 소유 temp/cache residue 여부

도구가 출력한 상태 문자열은 원문 그대로 인용할 수 있으나, 이를 수학적 지위로 해석하지 않는다.
