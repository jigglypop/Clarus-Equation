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
| 렌더링 | `rendering/ce_rendering_bao_ruler.py` | CE 음향 눈금 r_d, BAO 눈금 예측, 판본 V (43장 §43.13, v4 잠금) |
| 렌더링 | `rendering/ce_rendering_cycle.py` | 우주 순환의 반지름·주기·온도, 접선–현 기울기 (43장 §43.14, v5 잠금) |
| 렌더링 | `rendering/ce_rendering_spiral.py` | 돌면서 퍼지는 나선 긴장, 허블 비 √(1+Ω_Λ/4) (43장 §43.15, v5 잠금) |
| 렌더링 | `rendering/ce_rendering_complex_scale.py` | 복소 척도인자 A=a e^{iφ}, FLRW 극한·나이테 불변·기록 위상 차 (43장 §43.16, v6 잠금) |
| 렌더링 | `rendering/ce_rendering_growth.py` | 무적합 S8, 두 원(틀 회전과 순환 위상)의 구분 (43장 §43.17, v6 잠금) |
| 렌더링 | `rendering/ce_rendering_event_scale.py` | 2차원 렌더링 사건 척도 판별(M_Z) (43장 §43.18, v7 잠금) |
| 렌더링 | `rendering/ce_rendering_neutrino.py` | 중성미자 질량 행, 43행 공동 채점 (43장 §43.19, v7 잠금) |
| 렌더링 | `rendering/ce_rendering_gauge.py` | 렌더링 단계 V3⊕V2에서 표준모형 게이지군·한 세대·이상 상쇄, 렌더링 사상의 CPTP·무신호 (43장 §43.20) |
| 렌더링 | `rendering/ce_rendering_generations.py` | 우로보로스 3-순환 세대: CKM 선도 차수 단위행렬, TBM = 푸리에×π/4, 중력파 속도 (43장 §43.21) |
| 렌더링 | `rendering/ce_rendering_ewsb.py` | 약 통로(Higgs)의 확정: Q = T3+Y 보존, W·Z 질량, 전하 양자화 (43장 §43.22) |
| 렌더링 | `rendering/ce_rendering_inflation.py` | 급팽창 N_gauge = 12를 단계 보존 부분대수 차원으로 계산, N_e = 18D (43장 §43.23) |
| 렌더링 | `rendering/ce_rendering_nu_ledger.py` | 중성미자 장부: 이른 우주 ω_c에서 CE ω_ν를 뺌, CMB–BAO +3.07→+2.65σ, 판본 V 0.950 (43장 §43.24, v8) |
| 렌더링 | `rendering/ce_rendering_w_branch.py` | [반례] 저장소 w0·wa 식 W1: 고정 h에서 θ* +57σ, 기각; 살아남을 꼴은 −1 둘레 진동뿐 (43장 §43.25) |
| 렌더링 | `rendering/ce_rendering_vacuum_tilt.py` | [가설] W2 진공 판독 1+ξ²cos(H_Λt/2): 판본 V 0.909, CMB–BAO +2.17σ, w(0)=−0.989 (43장 §43.26, v9) |
| 렌더링 | `rendering/ce_rendering_vacuum_harmonic.py` | W2 = 복소 척도 위상의 최저 실수 조화(k=1 → ν=1/2), 부호 1 bit, 비앙키 틈 기록 (43장 §43.27) |
| 렌더링 | `rendering/ce_rendering_mimetic_vacuum.py` | W3 모방 시계 작용: 진공→먼지 확정, 비앙키 성립, 부호=에너지 양수; 판본 V 0.920, 43행 혼재 → 경쟁 판본 (43장 §43.28) |
| 렌더링 | `rendering/ce_rendering_light_limit.py` | c = 렌더링 한계: O1 = 빛원뿔 π/4의 이등분, 모든 신호 ≤ c, 렌더링 지평선의 온도·엔트로피 (43장 §43.29) |
| 렌더링 | `rendering/ce_rendering_probability_weight.py` | 양자중력 = 확률의 무게: 가지 무게는 초광속 신호(기각), 확률 무게는 무신호; BMV 얽힘 0(P23), Ω_k = 0(P24) (43장 §43.30, v10) |
| 렌더링 | `rendering/ce_rendering_record_update.py` | 기록 갱신 R1: 빛원뿔 안 기록만 원천 갱신, Page–Geilker 정합, 사건 척도 자발 붕괴 기각 (43장 §43.31) |
| 렌더링 | `rendering/ce_rendering_gradient.py` | CMB–BAO 그라데이션: C3 모양 G1–G3 기각, 가설 G1m(k=Ω_m)으로 −0.22σ, 판본 V 0.834 (43장 §43.32, v11) |
| 렌더링 | `rendering/ce_rendering_closure.py` | 마감: 기록 완성 = 란다우어(중복도 2는 GHZ로 기각), 지평선 정수 통로 기각, 가설 4.1 bit 장부 (43장 §43.33) |
| 렌더링 | `rendering/ce_rendering_reverse_derivations.py` | 가설 역유도: k=Ω_m = 확률 무게 평균, W2 부호 = 널 에너지, U1 = O1 이등분 직각삼각형, R-Pl = 4×⑤ (43장 §43.34) |
| 렌더링 | `rendering/ce_rendering_axiom_proofs.py` | 공리 증명 검산: QG1(HJW+아핀 ⇒ Tr ρT), C6(비르팅거), L(이그나토프스키+허수 빠르기), 이등분(minimax) (43장 §43.35) |
| 렌더링 | `rendering/ce_rendering_open_predictions.py` | 정리 E(나이테 = 무게 평균 1차), 미증명 항목의 예측 P26 사이렌 H0·P27 κ_λ·P28 M_H/M_Z (43장 §43.36, v12) |
| 렌더링 | `rendering/ce_rendering_sn_holdout.py` | Pantheon 40구간 초신성 보류 시험: L0·W2·W3 모두 최적 ΛCDM 대비 Δχ² < 0.5 (43장 §43.37) |
| 렌더링 | `rendering/ce_rendering_theta_nu.py` | θ*를 정확한 FD 중성미자로 통일 계산, 경로 차이 해소·판정 불변 (43장 §43.38) |
| 렌더링 | `rendering/ce_rendering_w3_growth.py` | W3 먼지 전달(δQ=0) 성장 보정, W2–W3 판가름: W3 경쟁 판본 유지 (43장 §43.39) |
| 렌더링 | `rendering/ce_rendering_data_sensitivity.py` | 자료 판본 민감도(NuFIT SK·렌즈·TRGB 16조합): NuFIT 판본이 지배적 (43장 §43.40) |

`regge_tent_transfer.json`은 해당 코드의 SHA-256과 적용범위를 검사하는 고정 fixture다.
새 실행 결과를 기존 검증 영수증에 자동으로 덮어쓰지 않는다.
