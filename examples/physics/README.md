# 계산 모듈

필수 회귀에서 직접 또는 간접으로 사용하는 구현만 유지한다. 동일 계산을 `verify/`나
원고 폴더에 복제하지 않는다. 검사와 실행 범위는 [tests](../../tests/README.md)를 따른다.

| 분야 | 모듈 | 책임 |
|---|---|---|
| 인과 | `causal/contextual_obstruction.py` | 문맥성과 공통 상태의 조건부 장애 |
| 암흑부문 | `darksector/cosmology.py` | 관측 입력·FLRW·성장률 |
| 암흑부문 | `darksector/ce_residual_forward_model.py` | 공분산 잔차·BAO·전방 계산 |
| 암흑부문 | `darksector/kinetic_dark_sector_gate.py` | 배경 진화·Pantheon·안정성 |
| 암흑부문 | `darksector/dimension_joint_candidate.py` | 연속 차원 후보·공동 잔차 산술 |
| 중력 | `gravity/causal_face_simplicity.py` | 유한 인과 면의 단순성 |
| 중력 | `gravity/zerod_plebanski_closure.py` | 유한 Plebanski 구성 |
| 중력 | `gravity/regge_tent_transfer.py` | 레게 작용·정준 전달·합성 |
| 기록 | `record/finite_ctp_diagonal_source_obstruction.py` | 비관측 환경·CTP 원천 |
| 기록 | `record/instrument_record_kernel.py` | 기록 커널·에너지 장부 |
| 기록 | `record/dimensional_filter.py` | 각도에 따른 1·2·3성분 투영과 비선택 상태 보존 |
| 기록 | `record/record_dust_bridge.py` | 기록 커널의 공통 보조 구현 |
| 기록 | `record/time_homogeneous_pointer_qca.py` | 비동기 포인터 기록 |
| 렌더링 | `rendering/ce_rendering_registry.py` | α_s 하나로 묶은 양자·거시 행의 공동 채점, 렌더링 규칙·bit 장부 (43장, 사전 등록 v1 잠금) |
| 렌더링 | `rendering/ce_rendering_derivations.py` | 음향 각도 θ* 검사, θ*-교정 판본 III, 부스트 반례 (43장 §43.11, v2 잠금) |
| 렌더링 | `rendering/ce_rendering_planck_readout.py` | 플랑크 단위 판독 규칙 R-Pl, 판본 IV, 유클리드 회전 C2 (43장 §43.12, v3 잠금) |

`regge_tent_transfer.json`은 해당 코드의 SHA-256과 적용범위를 검사하는 고정 fixture다.
새 실행 결과를 기존 검증 영수증에 자동으로 덮어쓰지 않는다.
