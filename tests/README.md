# 최소 필수 테스트

`tests/`에는 연구 목표와 저장소 계약을 직접 지키는 pytest 파일 12개만 둔다. 세부 구현별
회귀를 쌓는 대신 대표 경계에서 상태 보존, 중력 극한, 공동 오차, 무재피팅을 검사한다.
수치 일치는 구현의 회귀 증거이며 물리 법칙의 증명은 아니다.

| 책임 | 필수 파일 |
|---|---|
| 저장소·논문·사전등록 계약 | `test_repository_harness.py`, `test_canonical_document_policy.py`, `test_holdout_preregistration.py` |
| 공동 잔차·우주론·암흑부문 | `test_dimension_joint_candidate.py`, `test_ce_residual_forward_model.py`, `test_cosmology.py`, `test_kinetic_dark_sector_gate.py` |
| 비관측 양자 상태·비동기 틱 | `test_contextual_obstruction.py`, `test_finite_ctp_diagonal_source_obstruction.py`, `test_time_homogeneous_pointer_qca.py` |
| 이산기하·일반상대론 | `test_zerod_plebanski_closure.py`, `test_regge_tent_transfer.py` |

`test_repository_harness.py`의 `ESSENTIAL_TESTS`가 이 목록을 정확히 고정한다. 삭제한 세부
테스트는 활성 실행 계약이 아니며 필요하면 Git 이력에서 복구한다. 구현과 역사적 결과 원장은
삭제하지 않았으므로 과거 문서에 나온 테스트 파일명은 당시의 검산 영수증으로만 읽는다.

저장소 경로는 `test_support.paths`의 의미 상수를 사용한다. 전체 집합에는 NumPy와 SciPy가
필요하며 의존성은 `requirements-harness.txt`, 재현 스냅샷은
`requirements-harness-pinned.txt`에 있다.

```powershell
python -B -m pytest -p no:cacheprovider tests/test_repository_harness.py -q
python -B -m pytest -p no:cacheprovider tests/test_holdout_preregistration.py -q
python -B -m pytest -p no:cacheprovider tests -q
```

새 테스트를 추가하거나 기존 대표 파일을 교체할 때는 [tests 전용 규약](AGENTS.md)을 따른다.

