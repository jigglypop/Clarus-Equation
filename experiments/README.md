# 실험 자료 안내

`experiments/`는 실행 구현이 아니라 실험 계약과 탐색 기록을 보관한다.

| 경로 | 역할 | 상태 |
|---|---|---|
| `preregistration/` | 관측 전 동결한 holdout·실험 계약과 검증기 | 활성 계약과 과거 판본이 함께 있으므로 하위 README를 따른다. |
| `notes/` | EulerCE 등 독립 탐색의 설명과 결과 요약 | CE 통합 논문의 정본 근거가 아닌 연구노트 |

실행 가능한 물리 구현은 `examples/` 또는 `verify/`, pytest 회귀는 `tests/`, 생성 결과는 `verify/*.json` 또는 `artifacts/`에 둔다. 새 Python 구현을 `experiments/` 최상위에 추가하지 않는다.
