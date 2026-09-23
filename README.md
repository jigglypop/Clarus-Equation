# Clarus Equation

미발표 물리 연구 가설과 최소 계산·검증 코드다. 조건부 수학 결과, 물리 가정,
관측 비교를 구분한다. 테스트 통과는 자연법칙의 증명이 아니다.

이론의 정본은 [통합 논문](paper/CE_통합_논문.md), 읽는 순서는
[논문 안내](paper/README.md), 성공 기준은
[연구 목표 계약](paper/검증_원장/연구_목표_계약.md)에 둔다.

## 저장소 구성

| 경로 | 책임 |
|---|---|
| [examples/physics](examples/physics/README.md) | 필수 테스트가 사용하는 계산 모듈 12개 |
| [tests](tests/README.md) | 회귀·계약 검사 파일 12개 |
| `test_support/` | 공용 경로와 Markdown 수식 검사 |
| [experiments/preregistration](experiments/preregistration/README.md) | 동결 계약 4개와 검증기 |
| `benchmarks/cosmology/` | 계산에 필요한 관측 요약과 Pantheon 입력 |
| [paper](paper/README.md) | 원고·조건부 증명·반례·과거 기록 |
| `.codex/`, `AGENTS.md` | 연구 작업 규약 |

`verify/`, `ledger/`, `derivations/`, `_workspace/`, `scripts/`, `artifacts/`는
삭제했다. 유지할 구현은 `examples/physics/`로 통합했다. 새 계산은 기존 모듈에
넣고, 독립 책임이 생길 때만 모듈을 추가한다. 결과·로그·원장을 매 실행마다 복제하지 않는다.
논문에 남은 과거 파일 경로는 현재 재실행 가능하다는 뜻이 아니다.

루트의 `CE_observation_prediction_bundle_2026-09-09.zip`은 제공받은 원본 자료다.
현재 실행 코드의 의존성이 아니며 원문 출처로 보존한다.

## 실행

설치 가능한 패키지가 아닌 저장소이며, 루트에서 실행한다.

```powershell
python -m pip install -r requirements-harness.txt
python -B -m pytest -p no:cacheprovider tests -q
```

직접 의존성은 NumPy·SciPy·pytest 세 가지다. 검증에 사용한 버전은
`requirements-harness-pinned.txt`에 기록한다. 수식 표기 검사·정리는 다음 명령을 사용한다.

```powershell
python -B -m test_support.markdown_math paper
python -B -m test_support.markdown_math paper --write
```
