# 사전등록 계약

우주론·양자 holdout의 v1·v2 계약 네 개와 `validate_holdout_manifest.py`를 유지한다.
`rendering_predictions_v1.json`은 43장 렌더링 식 체계의 예측 10개를 2026-09-23에 동결한 계약이며,
`tests/test_rendering_registry.py`가 자기 해시·레지스트리 해시·예측 재현을 검사한다.
계약의 원문과 해시는 바꾸지 않았다. v1은 v2의 선행 판본 검증에 필요하다.

`tests/test_holdout_preregistration.py`가 동결·해시·자료 역할·재피팅 금지를 검사한다.
실제 holdout 배정과 입력 확보는 별도 조건이며, 문법 통과가 평가 준비 완료는 아니다.
삭제된 과거 파일을 참조하는 계약은 실행 가능하다고 표시하지 않는다.

```powershell
python -B -m experiments.preregistration.validate_holdout_manifest --no-verify-artifacts
python -B -m pytest -p no:cacheprovider tests/test_holdout_preregistration.py -q
```

별도 도메인의 미사용 계약과 탐색 노트는 제거했다. 새 계약은 현재 검증기와 연결되는
독립 실험에만 추가하며, 이미 동결한 계약을 결과에 맞춰 수정하지 않는다.
