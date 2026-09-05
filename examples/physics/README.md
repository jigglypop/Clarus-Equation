# examples/physics 색인

유한 증인(witness) 스크립트를 섹터 디렉터리로 묶는다. 여기 있는 계산 결과는 원장(`ledger/`) 항목 없이는 증거가 아니다. 색인은 `python .claude/hooks/lib/examples_index.py write`가 docstring 첫 줄에서 생성한다. 규칙(섹터 밖 모듈 금지·한국어 docstring·시험 존재)은 `tests/test_harness.py`가 검사한다.

모듈 수: 25. 시험 열은 `tests/test_<모듈>.py`가 있으면 ✓, 다른 시험이 import만 하면 ○.

## gravity — 중력·스핀폼 (simplicity·Plebanski·Urbantke; Q-0008 중력 브리지) (5)

| 모듈 | 시험 | 요약 |
|---|---|---|
| `gravity/causal_face_simplicity.py` | ✓ | 인과 합성 면(composition face)과 유한 플레바인스키(Plebanski) 단순성(simplicity) 감사. |
| `gravity/curved_plebanski_hinge.py` | ✓ | 닫힌 라벨 경첩(hinge) 위의 정확한 유한 비평탄 플레바인스키(Plebanski) 증인. |
| `gravity/gaussian_fold.py` | ✓ | 양의 가우스 측도의 접힘 제약과 유한 분해능 갱신을 계산한다. |
| `gravity/urbantke_shape_matching_rg.py` | ✓ | 플랑크(Planck) 렌더링 브리지를 위한 유한 우르반트케(Urbantke)/모양 맞춤(shape-matching) 감사. |
| `gravity/zerod_plebanski_closure.py` | ✓ | CE 0차원(0D)→플레바인스키(Plebanski) 사슬의 구성적 유한 닫힘과 그 보조 모듈 세 개를 한 파일에 모은다. |

## causal — 인과 구조·격자·문맥성 장애 (5)

| 모듈 | 시험 | 요약 |
|---|---|---|
| `causal/autonomous_covariant_single_collision.py` | ✓ | 명시적 시계 비용을 갖춘 자율 공변(autonomous covariant) 단일 충돌 브리지를 감사한다. |
| `causal/causal_domino.py` | ✓ | 유한 깊이 인과 양자 도미노(causal quantum domino)와 그 반복·격자 확장의 유한 인증서를 모은다. |
| `causal/causal_light_geometry.py` | ✓ | 인과 순서(causal order)에서 빛 기하와 계량(metric)이 어디까지 복원되는지 유한 반례와 장난감 재구성으로 감사한다. |
| `causal/contextual_obstruction.py` | ✓ | CHSH 상자에 대한 국소 공통 씨앗(seed) 장애와 전역 단면(global section) 장애의 유한 인증서를 모은다. |
| `causal/receipt_source.py` | ✓ | 가시 확률의 피셔 기하, 숨은 원천의 인수분해·영수증 계수(rank), 스칼라 영수증의 원천 허용에 대한 유한 인증서를 모은다. |

## record — 기록·선택·측정 (극장·영수증·장벽·관측) (8)

| 모듈 | 시험 | 요약 |
|---|---|---|
| `record/barrier_instrument.py` | ✓ | 유한 장벽 계측기 묶음: 기저 누설(E16)·이중 우물 스펙트럼 도약(E17)·단일 에너지 두 포트 산란 계측기를 한 모듈에 담는다. |
| `record/coarse_observation.py` | ✓ | 거친 관측(coarse observation) 섹터의 유한 증명서 네 종을 한 모듈에 모은다. |
| `record/covariant_energy_bridge.py` | ✓ | 세계관(worldtube) 에너지 다리: 조건부 영수증 정합, 두 스칼라 교환 전류, 유한 물질 격자 비용을 한 모듈에 둔다. |
| `record/finite_ctp_diagonal_source_obstruction.py` | ✓ | 관측되지 않은 양자 환경의 영향을 유한 닫힌 시간 경로(closed time path, CTP)로 증명하는 증인 모음이다. |
| `record/instrument_record_kernel.py` | ✓ | 에너지 분해 양자 기구(quantum instrument) 기록 커널과 유한 보른(Born) 선택 커널을 구성한다. |
| `record/record_dust_bridge.py` | ✓ | 표시된 인과 기록에서 먼지 응력 텐서로 가는 구성적 브리지와 단일 시계 작용 허용 판정. |
| `record/theater_opening.py` | ✓ | 극장 개장 비유의 조건부 양자 급랭(quantum quench) 모형과 틱 접힘(tick-fold) 규칙 스캔을 한 모듈에 담는다. |
| `record/time_homogeneous_pointer_qca.py` | ✓ | 유한 지평·시간 균질(time-homogeneous) 포인터 전파 감사 모듈이다. |

## darksector — 암흑 부문·우주론 (kinetic dark sector·FLRW·잔차 전방모형) (7)

| 모듈 | 시험 | 요약 |
|---|---|---|
| `darksector/ce_residual_forward_model.py` | ✓ | CE 잔차 우주론의 최소 전방 모델(forward model)이다. |
| `darksector/cosmology.py` | ✓ | 평탄 FLRW 배경 커널, CE 밀도비 감사, 성장률 수치 계산을 한 모듈에 모은다. |
| `darksector/kinetic_dark_sector_adiabatic_stress.py` | ✓ | 일정 질량 FLRW 스칼라에 대한 4차 단열(adiabatic) 응력 빼기다. |
| `darksector/kinetic_dark_sector_backreaction.py` | ✓ | 동적 시계에 대한 정확한 무차원 FLRW 워드(Ward)/되먹임(backreaction) 닫힘이다. |
| `darksector/kinetic_dark_sector_gate.py` | ✓ | 자기비동일성 운동 암흑부문의 배경 게이트, Pantheon 초신성 형상 게이트, FLRW 모드 진화를 한 모듈에 모은다. |
| `darksector/kinetic_dark_sector_perturbation.py` | ✓ | 운동 암흑부문의 단일 시계 섭동·EFT 차단 게이트와 급냉 스펙트럼-현재 존재비 조건부 다리를 한 모듈에 모은다. |
| `darksector/kinetic_dark_sector_quench.py` | ✓ | SNKC 극장 개막 급냉(quench) 생산 배경 게이트, 해석적 자외선 꼬리 상계, 봉인 홀드아웃 평가를 한 모듈에 모은다. |
