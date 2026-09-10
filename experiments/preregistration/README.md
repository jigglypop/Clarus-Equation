# 사전등록 계약

사전등록 JSON은 결과 확인 전에 고정한 비교 규칙과 holdout 상태를 기록한다. 파일을 단순 결과 산출물로 취급해 덮어쓰지 않는다.

## 현재 직접 검증되는 계약

- `cosmology_future_holdout_v1.json`, `cosmology_future_holdout_v2.json`
- `quantum_future_holdout_v1.json`, `quantum_future_holdout_v2.json`
- `validate_holdout_manifest.py`

`tests/test_holdout_preregistration.py`가 위 네 판본과 검증기를 직접 검사한다. 기본 CLI는 최신 v2 두 판본을 검사한다.

## 보존 중인 미분류 계약

`agi_world_memory_*`, `episodic_ltm_*`, `sparse_causal_bridge_*`, `c_elegans_*`, `causal_recurrent_geometry_*`는 현재 코드에서 직접 소비하는 경로가 확인되지 않았지만, 사전등록의 역사적 무결성을 위해 삭제하거나 덮어쓰지 않는다. 별도 출처·해시 원장을 만든 뒤 보관 경로로 이동한다.

검증 예:

```powershell
python -B experiments/preregistration/validate_holdout_manifest.py --no-verify-artifacts
python -B -m pytest -p no:cacheprovider tests/test_holdout_preregistration.py -q
```
