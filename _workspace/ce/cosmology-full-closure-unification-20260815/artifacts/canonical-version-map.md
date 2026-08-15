# CE 우주론 정본·판본 통합 지도

Status: COMPLETE

기준일: 2026-08-15  
기준 commit: `5414336ae2ff20197efe3bf8a92ec5183ad079aa`  
작업 성격: U1/U7 read-only inventory. 이 파일 외의 정본·제품·테스트 파일은 수정·삭제·이동하지 않았다.

## 1. 결론

현재 저장소에는 서로 다른 네 종류의 값이 같은 이름으로 겹쳐 있다.

1. `alpha_s -> s_W^2 -> delta -> D_eff -> q_ext`의 **정밀 계산 사슬**
2. LO, 3-layer, NLO처럼 가정을 더한 **모형별 후기우주 분할값**
3. `0.0487/0.2623/0.6891`처럼 제품 동작을 고정한 **반올림 호환 기본값**
4. Planck/ACT/SPT/DESI/SH0ES 등의 **관측 스냅샷**

따라서 “하나로 통합”은 모든 숫자를 하나로 덮어쓰는 일이 아니다. 하나의 형식화된
registry가 각 값의 `role`, `formula`, `precision`, `model_id`, `source_id`,
`formal_status`를 관리하게 만들고, 기존 이름은 compatibility alias로 보존하는 것이
안전한 통합이다.

핵심 의미 충돌도 같은 방식으로 해소해야 한다. `q_ext`는 소멸확률이고
`s_branch=1-q_ext`가 생존확률이다. `q_ext -> Omega_b`는 정본 일부에서 채택한
모형 공리이지만 고정점 정리 자체는 아니다. 직접 동일시를 보편적 유도로 주장한
경로는 반례를 통과하지 못했으나, 목표값과 역사적 모형을 삭제할 이유는 없다.
`LEGACY_DIRECT_READOUT_V1`이라는 이름 붙은 공리적 경로로 보존하고, 새
current/freeze-out 경로를 별도 claim ID로 전개하면 된다.

## 2. 조사 범위와 판정 규칙

조사한 활성 정본은 `docs/axium.md`, `docs/경로적분.md`, `docs/상수.md`,
`docs/README.md`, `docs/1_강의/`, `docs/2_경로적분과_응용/`, `docs/3_상수/`,
`docs/참조/`, `docs/9_등호이전/`이다. 구현 범위는
`reality_stone/python/reality_stone/clarus/`, `examples/physics/`, `tests/`이고,
선행 우주론 실행은 다음 두 run을 우선 대조했다.

- `_workspace/ce/cosmology-density-bridge-derivation-20260815/`
- `_workspace/ce/cosmology-theory-repository-audit-20260815/`

아래 표의 짧은 파일명은 다음 repo-relative 경로를 뜻한다. 따라서 모든
`파일명:줄` anchor는 이 표를 통해 단일 파일로 해석된다.

| 짧은 이름 | 정확한 repo-relative 경로 |
|---|---|
| `constants.py` | `reality_stone/python/reality_stone/clarus/constants.py` |
| `bootstrap_solver.py` | `reality_stone/python/reality_stone/clarus/bootstrap_solver.py` |
| `core_axioms.py` | `reality_stone/python/reality_stone/clarus/core_axioms.py` |
| `agent.py`, `runtime.py`, `stdp.py`, `__init__.py` | 각각 `reality_stone/python/reality_stone/clarus/<name>` |
| `cosmology.py` | `examples/physics/cosmology.py` |
| `cosmology_ratio_audit.py` | `examples/physics/cosmology_ratio_audit.py` |
| `cosmology_discrimination_gates.py` | `examples/physics/cosmology_discrimination_gates.py` |
| `ce_residual_forward_model.py` | `examples/physics/ce_residual_forward_model.py` |
| `hubble_tension.py` | `examples/physics/hubble_tension.py` |
| `cosmological_constant_holographic_gate.py` | `examples/physics/cosmological_constant_holographic_gate.py` |
| `primordial_spectrum_readout_gate.py` | `examples/physics/primordial_spectrum_readout_gate.py` |
| `validate_manuscript.py` | `docs/2_경로적분과_응용/validate_manuscript.py` |
| density run `20-audit.md`, `40-final-report.md` | `_workspace/ce/cosmology-density-bridge-derivation-20260815/<name>` |
| repository audit `20-audit.md`, `40-final-report.md` | `_workspace/ce/cosmology-theory-repository-audit-20260815/<name>` |

형식 지위는 `[정의]`, `[정리]`, `[공리]`, `[산출]`, `[경험식]`, `[미완성]`,
`[예측]`을 구분한다. 수치가 기계적으로 재현된다는 사실은 그 수치를 자연에
대응시키는 물리 사상을 증명하지 않는다. 아래 `정밀`은 식을 full precision으로
평가한 값, `표시`는 문서용 반올림, `호환`은 기존 API/회귀 고정값, `관측`은 특정
자료·모형의 외부 스냅샷을 뜻한다.

## 3. 핵심 수치 원장

| ID / 기호 | 값과 의미 | 현재 근거와 형식 지위 | 실제 소비자와 충돌 | 제안 one-source-of-truth |
|---|---|---|---|---|
| `CORE.alpha_s_mz` | `0.11789`; CE 계산의 외부 강결합 입력 | `README.md:43-48`은 외부 입력 `[공리]`; `docs/2_경로적분과_응용/03_자유매개변수.md:44-54`의 사슬 시작. 관측에서 무입력 유도된 값이 아니다. | 제품/문서 전체에 exact token이 31파일 43줄. 코드에는 `constants.py:116`, `cosmology.py:320`, `cosmology_discrimination_gates.py:35`, `validate_manuscript.py:486`, `hubble_tension.py:82` 등 별도 literal이 있다. scale/scheme이 생략된 소비자가 섞인다. | `cosmology_registry.py::CORE_EXACT_V1.alpha_s`, provenance에 `scale=M_Z`, scheme, source ID 필수 |
| `CORE.sin2_theta_w` | `4 alpha_s^(4/3) = 0.23122206826075514` | 관계는 `[경험식]`: `docs/axium.md:261-268`, `docs/상수.md:103-106`; 수치는 `[산출]`: `docs/2_경로적분과_응용/03_자유매개변수.md:52`. | 문서가 정밀값과 `0.2312221` 표시값(`docs/2_경로적분과_응용/04_해결한_난제.md:119`)을 함께 사용한다. | registry에서 식으로만 계산; 문서는 `display_digits=7/10`만 지정 |
| `CORE.delta_exact` | `s_W^2(1-s_W^2) = 0.17775842340997383` | `delta`는 `[정의]`: `docs/axium.md:249-251`; `D`에 더하는 해석은 `[공리]`: `docs/axium.md:304-321`; 표시값은 `docs/2_경로적분과_응용/03_자유매개변수.md:53`. | `bootstrap_solver.py:26`은 별도 호환값 `0.17776`을 사용한다. 두 값을 같은 `DELTA`로 취급하면 q의 마지막 자릿수가 달라진다. | `CORE_EXACT_V1.delta`와 `LEGACY_DELTA_5DP_V1.delta`를 서로 다른 항목으로 보존 |
| `CORE.d_eff_exact` | `3+delta = 3.1777584234099736` | 균일 행합 toy family 아래 `[공리]+[산출]`: `docs/axium.md:304-321`, `docs/경로적분.md:353-371`; 표시값 `docs/2_경로적분과_응용/03_자유매개변수.md:54,194`. | `bootstrap_solver.py:27`은 `3.17776`; README의 q도 이 5자리 D를 따른다. | `CORE_EXACT_V1.d_eff`는 계산 property; `LEGACY_D_EFF_5DP_V1` 별도 |
| `CORE.q_ext_exact` | 최소 소멸근 `0.048646719644028225` | Lambert-W/Poisson 최소해 `[정리]`: `docs/axium.md:274-300`, `docs/경로적분.md:338-347`, `docs/상수.md:129-137`; full-chain 표시는 `docs/2_경로적분과_응용/03_자유매개변수.md:194-203`. | 정확 토큰 `0.048646719...`은 활성 조사 범위에서 1파일 1줄뿐이다. parameterized 정리 구현은 `core_axioms.py:233-344`; 새 density audit은 exact D를 사용한다. 제품 runtime은 이 full precision 값을 직접 소비하지 않는다. | `CORE_EXACT_V1.q_ext`, 안정적인 low-root solver와 residual certificate를 함께 저장 |
| `LEGACY.q_ext_5dp` | `0.0486466333`; `delta=0.17776`에서 나온 회귀값 | 수학식은 같지만 입력이 반올림된 `[산출/호환]`: `README.md:72-83,97`, `bootstrap_solver.py:26-27`. `benchmarks/core_claims_v1.json:122-128`도 `delta_legacy`라고 명시한다. | `tests/test_bootstrap_solver.py:15,30`, `tests/test_core_axioms.py:128`, `tests/test_core_model_selection.py:93`가 이 값을 잠근다. exact-chain 값으로 무표시 교체하면 회귀가 깨진다. | `LEGACY_DELTA_5DP_V1.q_ext`; alias와 characterization test 유지 |
| `DISPLAY.q_ext_5sf` | `0.04865`; 설명·scorecard용 표시값 | 정밀 q의 5유효숫자 표시. `tests/scorecard.py:175`, `tests/run_validation.py:37,148`에서 관측 비교/검산에 사용된다. | `bootstrap_solver.py` docstring은 이를 “baryon survival rate”로 부르지만 실제 수학은 extinction root다. | `format_q_ext(5)`; 숫자 literal을 독립 상수로 승격하지 않음 |
| `RUNTIME.active_ratio` | `0.0487`; task-active/legacy baryon target | `constants.py:26`; 물리적으로는 반올림 경계값, 런타임에서는 operational target이다. `[호환 기본값]`, 우주론 예측 아님. | `agent.py:19-20,141-192`, `runtime.py:32-48,603,645-691`, `stdp.py:20-26,84-134`, `__init__.py:317-319,630-632`; `cosmology_ratio_audit.py:16-41`가 AST로 읽고 residual이 이를 다시 import한다. | `LEGACY_ROUNDED_RUNTIME_V1.active_ratio`; `ACTIVE_RATIO` alias 유지 |
| `CORE.s_branch` | `1-q_ext = 0.9513532803559718`; 생존확률 | `[정의]`: `docs/참조/정합성_검증.md:27-30`, `docs/경로적분.md:346-347`, exact 표시 `docs/2_경로적분과_응용/03_자유매개변수.md:200-201`. | `q_ext`와 의미가 뒤집힌 설명은 `bootstrap_solver.py:4-7`에 있다. `epsilon^2`가 어느 쪽인지 명시하지 않은 소비자는 위험하다. | `CORE_EXACT_V1.survival`; `SURVIVAL_BRANCH`는 오직 `1-Q_EXT`로 정의. `EPSILON_SQUARED_LEGACY`는 q alias로만 유지 |
| `CORE.contraction_exact` | `D_eff*q_ext = 0.15458752312007412` | 주어진 D와 q의 `[산출]`; 안정성 조건 `Dq<1`은 `docs/axium.md:294-295`. | `constants.py:29`의 `0.155`는 런타임 호환 표시값이다. 이를 exact theorem value로 읽으면 안 된다. | `CORE_EXACT_V1.contraction`; `BOOTSTRAP_CONTRACTION`은 `LEGACY_ROUNDED_RUNTIME_V1.contraction_display` alias |
| `DENSITY.lo_v1` | `R_LO=alpha_s D=0.3746259405358018`; `(Omega_b,Omega_DM,Omega_L)=(0.048646719644028225,0.25927170943410105,0.6920815709218708)`; `Omega_m=0.3079184290781293` | `R_LO`와 후기우주 분할은 `[공리]`, 그 아래 숫자는 `[산출]`: `docs/2_경로적분과_응용/03_자유매개변수.md:241-280,328-330`; 요약 `04_해결한_난제.md:130-150`. | `cosmology.py:344-366`의 기본 `--model bootstrap`이 이 경로다. rounded runtime triplet 및 3-layer 결과와 다르다. | `DENSITY_LO_V1`, formula와 exact core version을 참조 |
| `DENSITY.three_layer_manuscript_v1` | supplied coupling sum `1.0147344271`; `(Omega_L,Omega_DM,Omega_m)=(0.6890735470,0.2622797333,0.3109264530)` | supplied EW coupling을 더한 `[경험식/산출]`: `docs/2_경로적분과_응용/validate_manuscript.py:831-850`; 코드도 “selection is Phenomenology”라고 출력한다. | standalone active test가 없고, `tests/test_canonical_document_policy.py:57-64`는 refuted 3-layer parent를 활성 markdown에서 금지한다. 역사적 executable witness는 삭제할 필요가 없다. | `DENSITY_THREE_LAYER_MANUSCRIPT_V1`, `historical=True`, `active_parent=False` |
| `DENSITY.three_layer_approx_v1` | `ratio_sum=1.015`; `(Omega_L,Omega_DM,Omega_m)=(0.6890727868885254,0.26228049346744653,0.31092721311147475)` | 근사 coupling sum을 공급한 `[경험식/산출]`: `cosmology_discrimination_gates.py:53-70`. | manuscript variant와 7번째 자리부터 다르지만 둘 다 `0.6891/0.2623`으로 반올림된다. 이름 없이 합치면 provenance가 사라진다. | `DENSITY_THREE_LAYER_APPROX_V1`, `ratio_sum_source="rounded 1.015"` |
| `DENSITY.nlo_candidate_v1` | `R_NLO=R_LO+R_LO^2/(4pi)=0.3857942085766924`; `(Omega_L,Omega_DM,Omega_m)=(0.6865040093745797,0.26484927098139216,0.3134959906254204)` | discrimination alternative `[경험식/후보]`: `cosmology_discrimination_gates.py:62-70`. | 기본값으로 선택되지 않았고 likelihood 판정도 없다. | `DENSITY_NLO_CANDIDATE_V1`, explicit opt-in only |
| `RUNTIME.rounded_triplet` | `(0.0487,0.2623,0.6891)`, 원시 합 `1.0001` | `constants.py:26-29`의 제품 호환값. 3-layer 계열을 4자리 표시한 것과 같아 보이지만 provenance가 명시돼 있지 않다. | runtime target, cosmology boundary, “CE prediction” 세 역할로 재사용된다. `ce_residual_forward_model.py:47-55`가 기본 경계값으로 받고 `:89-102`에서 background용으로 정규화한다. | `LEGACY_ROUNDED_RUNTIME_V1`, `raw_sum=1.0001`, `normalization_policy` 명시 |
| `BACKGROUND.normalized_legacy_v1` | `(Omega_m_hat,Omega_L_hat)=(0.31096890310968905,0.6890310968903111)` | rounded triplet을 합으로 나눈 조건부 `[산출]`: `ce_residual_forward_model.py:89-102`. | raw table의 `Omega_L=0.6891`과 background 계산의 `0.689031...`이 같은 출력처럼 보일 수 있다. | `CE_RESIDUAL_FLAT_LCDM_GR_V1`에 raw와 normalized를 모두 기록 |

## 4. 주장·기호 의미 대응표

| Claim ID | 주장 / 정확한 위치 | 현재 지위와 수치 provenance | 충돌 및 활성 소비자 | 통합 판정 |
|---|---|---|---|---|
| `C-Q-01` | `q_ext`는 Poisson 분지과정의 최소 소멸 고정점: `docs/axium.md:274-300`, `docs/1_강의/A_연역적_유도.md:141-163`, `docs/참조/정합성_검증.md:27-30` | `[정리]`; D가 공급되면 유일하게 계산 | `core_axioms.py:233-344`, `examples/physics/core_axiom_loop.py:12-61`, `tests/test_core_axioms.py`; legacy solver도 같은 식이나 반올림 D 사용 | 활성 정본 유지. q를 survival이라고 부르는 설명만 compatibility note로 교정 대상 |
| `C-Q-02` | `s_branch:=1-q_ext`: `docs/경로적분.md:346-347`, `docs/3_상수/7_우주론.md:36-41` | `[정의]`, exact chain에서 `0.95135328036` | `q_ext`, `epsilon^2`, active ratio와 혼용 위험 | q/s 별도 symbol과 serialization key 강제 |
| `C-R-01` | multiplicative readout `R(xy)=R(x)R(y)` 및 `R(x)=x^c`; CE에서 `c=1`: `docs/axium.md:338-349`, `docs/1_강의/A_연역적_유도.md:165-178` | 함수형 부분 `[정리]`; `c=1`은 `[공리]` | 이 정리는 어떤 물리 observable을 R의 치역으로 택할지 정하지 않는다. | 수학 정리와 cosmology route를 별도 claim ID로 유지 |
| `C-B-LEGACY-01` | direct `q_ext -> Omega_b`: `docs/axium.md:351-365,477`, `docs/경로적분.md:391-423,600,624`, `docs/상수.md:139-140`, `docs/3_상수/3_부트스트랩.md:54-76`, `docs/3_상수/7_우주론.md:43-63` | 정본에는 명시적 `[공리]`; 숫자는 q를 대입한 모형 경계값 | 강의는 별도 물리 사상이 현재 `[미완성]`이라고 한다: `docs/1_강의/A_연역적_유도.md:178-180`, `B_귀납적_유도.md:100-108`, `C_다섯_상수.md:98-99`. 선행 반례는 q만으로 Omega를 정할 수 없음을 보였다: density run `20-audit.md:57`. | 삭제·무표시 강등 금지. `LEGACY_DIRECT_READOUT_V1`이라는 **모형 공리**로 보존하고 “core theorem 아님”을 모든 참조에 병기 |
| `C-B-COMP-01` | matter 내부 조성비 `f_b^(m)=q` | equal-energy two-dust와 boundary constraint 아래 조건부 `[산출]`: density run `40-final-report.md:282-302` | 정확한 관계는 `Omega_b=q Omega_m`; `f_b^(m)=q => Omega_b=q`는 거짓: density run `20-audit.md:67-69` | `COMPOSITION_TWO_DUST_V1`로 독립 보존. direct readout의 대체 증명으로 오용 금지 |
| `C-B-CURRENT-01` | local branching label current -> conserved baryon current -> freeze-out normalization | 긍정 연구경로 `[미완성]`: density run `40-final-report.md:359-420`; 구현 판정 `:428-429`는 model axiom/incomplete/no prediction | action, charge normalization, covariant freeze-out surface, Omega_m dynamics가 아직 없다. | 다음 활성 유도 경로. target `Omega_b`는 유지하되 route status만 fail-closed |
| `C-FLAT-01` | `Omega_b+Omega_DM+Omega_Lambda=1`: `docs/경로적분.md:416-423`, `docs/3_상수/7_우주론.md:54-63` | 평탄 후기우주 모형을 채택한 `[공리/조건]` | 원시 rounded triplet은 1.0001이고 residual은 이를 정규화한다. radiation/curvature가 포함되는 epoch에는 그대로 쓸 수 없다. | config에 `epoch=t0`, `flat=True`, `components`와 normalization policy 명시 |
| `C-DARK-LO-01` | `R_LO=alpha_s D`와 complement의 DM/Lambda 분할 | `[공리]`; 수치는 `[산출]`: `03_자유매개변수.md:258-280,330` | 3-layer, NLO, rounded runtime과 경쟁. 관측 우도에 의해 선택된 바 없음 | LO를 유일 default로 위장하지 않고 named active toy config로 유지 |
| `C-DARK-3L-01` | three-layer partition | `[경험식/역사적 산출]` | exact manuscript와 rounded approximation 두 판본, active markdown parent 금지, standalone test 부재 | 두 판본 모두 registry에 보존하고 historical flag 및 공통 반올림 표시를 기록 |
| `C-DARK-TARGET-01` | `0.0487/0.2623/0.6891`을 자연의 예측이라고 주장 | 저장소 코드 일부는 `ce_prediction`으로 라벨: `ce_residual_forward_model.py:825-855` | 선행 repository audit은 Omega_b가 supplied boundary이고 DM/DE도 phenomenology라고 판정: repository audit `20-audit.md:37-42` | 값은 보존. 출력 role을 `legacy_supplied_boundary`/`phenomenology`로 분리; prediction 승격에는 blind joint likelihood 필요 |

## 5. LO·3-layer·default 구성 분리

현재 `default`라는 단어는 세 맥락을 덮는다. 이를 아래처럼 한 registry 안에서
이름 붙여야 한다.

| 제안 config ID | 내용 | 기본 사용 정책 | 현재 구현 anchor |
|---|---|---|---|
| `CE_CORE_EXACT_V1` | full-precision alpha/delta/D/q/s/contraction | 모든 새 수학 계산의 기본 | `docs/2_경로적분과_응용/03_자유매개변수.md:44-54,194-203`; `core_axioms.py:233-344` |
| `CE_DENSITY_LO_V1` | exact q + `R_LO=alpha_s D` | 명시적 toy-model 선택 | `cosmology.py:344-366`; `03_자유매개변수.md:258-280` |
| `CE_DENSITY_3L_MANUSCRIPT_V1` | coupling sum `1.0147344271` | historical witness, active scientific default 아님 | `validate_manuscript.py:831-850` |
| `CE_DENSITY_3L_APPROX_V1` | supplied `ratio_sum=1.015` | discrimination-only | `cosmology_discrimination_gates.py:53-70` |
| `CE_DENSITY_NLO_CANDIDATE_V1` | `R_LO+R_LO^2/(4pi)` | candidate-only | `cosmology_discrimination_gates.py:62-70` |
| `LEGACY_ROUNDED_RUNTIME_V1` | `0.0487/0.2623/0.6891`, contraction `0.155` | API 호환 default만 유지; 출력마다 `legacy_compat` 표시 | `constants.py:26-29` |
| `CE_RESIDUAL_FLAT_LCDM_GR_V1` | legacy boundary를 정규화한 flat w0-wa/GR background/growth | 현재 가장 완성된 조건부 forward path | `ce_residual_forward_model.py:45-118,1466-1526` |

새 API에는 `scientific_default=None`이 안전하다. 기존 API의 `default`는 당장
없애지 않고 `LEGACY_ROUNDED_RUNTIME_V1`을 가리키게 하되, provenance를 반드시
출력한다. 즉 호환 기본값과 이론적으로 선택된 기본값을 같은 것으로 선언하지 않는다.

## 6. background·H0·growth 경로 지도

| 경로 | 정확한 위치와 입력 | 형식 지위 | 충돌 / 소비자 | 제안 기준 위치 |
|---|---|---|---|---|
| legacy LO background | `cosmology.py:116-133,317-366`; default `H0=67.4`, `sigma8=0.811`, `--model bootstrap`, 별도 `calibrate` | 공급된 경계값 아래 `[산출]`; SFE `mu` 선택은 별도 모형 | residual이 `cosmology.py:25`의 수치 적분 helper만 import한다. standalone background/growth CLI의 직접 test는 약하다. | 계산 helper는 공용화하되 config는 `CE_DENSITY_LO_V1`로 명시 |
| residual background/distances | `ce_residual_forward_model.py:1-7,45-118`; raw triplet, `H0=67.4`, `rd=147.09`, `w0=-1`, `wa=0` | supplied boundary와 flat w0-wa/GR 아래 조건부 `[산출]`; 문서 `docs/9_등호이전/05o_CE_잔차_우주론_전방모형.md:5-73` | `cosmology_ratio_audit.py:165-174`는 background/growth가 없다고 아직 `False`를 반환해 stale하다. residual test suite는 존재한다. | `CE_RESIDUAL_FLAT_LCDM_GR_V1`; 현재 canonical computational path |
| residual growth/S8 | `ce_residual_forward_model.py:1466-1526`; `sigma8_0=0.811`, GR normalization | `[산출]`; CE 고유 중력 예측이 아니라 외부 sigma8와 GR을 사용 | `cosmology.py`의 기본/선택 가능한 SFE `mu`와 조용히 달라질 수 있다. | gravity model ID와 sigma8 source를 config 필수필드로 |
| phase-area H0 readout | `docs/2_경로적분과_응용/04_해결한_난제.md:29-51`; `validate_manuscript.py:871-887`에서 `67.2472445605` | readout 식 대입은 `[산출]`, 우주론적 해석은 `[경험식]`: 문서 `:47,178-179` | `67.4` 외부 baseline과 다른 역할이다. 같은 `H0_default`로 합치면 안 된다. | `H0_PHASE_AREA_TOY_V1`, `conditional_output` |
| Hubble-tension toy | `hubble_tension.py:1-69,81-107,290-349`; external `H0_true=73.04` at `:358`, target tension을 사용하는 scan `:481-511` | `[미완성/target-aware toy]`; `docs/2_경로적분과_응용/04_해결한_난제.md:141-150` | test/import consumer가 없고 `om_b_h2` argument는 계산 본문에서 실질적으로 사용되지 않는다. | `H0_TENSION_LEGACY_TOY_V1`, 기본 경로에서 제외하되 파일 보존 |
| missing H0 covariance path | `docs/3_상수/10_H0_readout_law_audit.md:33-49`, `11_TDCOSMO_real_covariance_audit.md:1-59`, `12_H0_source_role_readout_paper_draft.md:1-47` | 세 문서 모두 `[미완성]`; `examples/physics/h0_readout/` 및 원자료 bundle 부재 | 과거 결과를 현재 재현 가능한 경로처럼 참조하면 안 된다. | source manifest/covariance bundle이 생길 때까지 explicit `unavailable` registry entry |

`H0=67.4`, `H0_true=73.04`, `H0_phase=67.247...`는 서로 충돌하는 세
예측값이 아니다. 각각 외부 baseline, local input, 조건부 toy output이다. 이 역할을
type으로 분리하는 것이 통합이며, 평균을 내거나 하나를 지우는 것은 통합이 아니다.

## 7. 관측 스냅샷과 U7 provenance

| 관측 항목 | 저장소 위치 | 현재 provenance 상태 | 통합 조치 |
|---|---|---|---|
| Planck/ACT/SPT density compressed sets | `cosmology_ratio_audit.py:109-138`: `(omega_b h2,omega_c h2,H0)` 네 tuple | 이름·설명만 있고 release, likelihood/model, URL, checksum, covariance가 없다. `[관측 스냅샷]`이며 theory constant가 아니다. | 관측 manifest로 이동 참조; 현재 literal은 compatibility copy로 검증 |
| scorecard density values | `tests/scorecard.py:115-205`: alpha/PDG, `0.02237+/-0.00015`, `0.11933+/-0.00091`, `0.6847+/-0.0073`; CE 측은 `0.04865*(0.674)^2`, `0.2623*(0.674)^2`, `0.6891` | pointwise snapshot; full covariance와 동일 dataset joint fit가 아니다. 계산값은 각각 `0.0221005274`, `0.1191565948`. | regression scorecard로 보존하되 “blind prediction” 라벨 금지; manifest source ID 연결 |
| primordial `A_s,n_s` | `docs/3_상수/9_우주론_수식_의미와_후보.md:61-89`, `docs/2_경로적분과_응용/12_유도결손_보완.md:183-245`, `tests/scorecard.py:272-293`, `primordial_spectrum_readout_gate.py:28-30` | CE projector `A_s=2.104e-9`는 `[경험식]`; Starobinsky 결과는 action/Ne 아래 `[산출]`; 관측값은 snapshot | projector, inflation model output, observed normalization을 서로 다른 manifest role로 분리 |
| embedded DESI DR2 BAO | `ce_residual_forward_model.py:1077-1128`; 설명 문서 `docs/9_등호이전/05o_CE_잔차_우주론_전방모형.md:187-200` | data vector/covariance가 코드에 있으나 원자료 URI·release asset checksum이 완전한 machine manifest로 잠기지 않았다. | `observations_v1.json`에 source URI, SHA256, covariance layout, model, units를 고정하고 코드는 manifest를 읽음 |
| DESI+CMB Omega-Lambda 비교 | predecessor repository audit `20-audit.md:55`, `40-final-report.md:135` | 저장소의 `0.693+/-0.005` hybrid provenance가 검증되지 않았고, audit가 확인한 referenced alternative는 `0.6973+/-0.0036`이었다. | hybrid row는 삭제하지 말고 `invalid_provenance=True`; 공식 source가 잠길 때까지 판정에서 제외 |

제안 관측 원장은 `benchmarks/cosmology/observations_v1.json`이다. 각 entry에 최소한
다음 필드를 둔다.

```text
dataset_id, release, analysis_model, likelihood_id, source_uri,
retrieved_at, asset_sha256, parameter_names, means, units,
covariance_asset, covariance_sha256, train_or_holdout, notes
```

관측 원장과 theory registry는 분리한다. 동일 파일에 넣으면 관측 업데이트가 core
상수의 버전까지 바꾸고, target leakage를 추적하기 어려워진다.

## 8. 중복과 stale consumer 정량화

검색 기준은 `README.md`, `docs/`, `reality_stone/python/reality_stone/clarus/`,
`examples/physics/`, `tests/`의 `*.md`, `*.py`이며 `.tmp`는 제외했다. line 수는
2026-08-15 snapshot의 `rg -n -P` 매치 줄 수다.

| literal / symbol | 파일 수 | 매치 줄 수 | 해석 |
|---|---:|---:|---|
| exact token `0.11789` | 31 | 43 | 외부 입력이 문서·검증기·제품에 반복됨 |
| `q_{\rm ext}` 또는 `q_ext` | 18 | 90 | 정리, readout, target claim이 한 symbol에 모임 |
| old exact token `0.0486466333` | 5 | 7 | 5자리 delta 기반 legacy chain |
| new exact prefix `0.048646719...` | 1 | 1 | full-precision chain이 아직 확산되지 않음 |
| exact token `0.04865` | 9 | 25 | display/scorecard/legacy observational comparison 혼재 |
| exact token `0.0487` | 16 | 50 | runtime target, density boundary, 문서 표가 혼재 |
| exact token `0.2623` | 15 | 27 | runtime structural target와 dark-matter density 혼재 |
| exact token `0.6891` | 19 | 31 | runtime background, Omega-Lambda, holographic input 혼재 |

생산 코드만 보면 `0.11789` 직접 literal은 12파일 13줄, `0.0487`은 3파일
5줄, `0.2623`은 4파일 5줄, `0.6891`은 6파일 7줄이다. literal이 없어도
`constants.py`를 통해 간접 소비하는 코드가 있으므로 이것은 하한이다.

우선 stale consumer는 다음과 같다.

1. `bootstrap_solver.py:26-27`과 3개 회귀 테스트가 legacy D/q를 “현재 exact”와
   구별하지 않는다. 값은 유지하되 이름을 바꾼 alias가 필요하다.
2. `README.md:72-83,97`은 `0.0486466333`을 정밀 사슬의 유일 값처럼 표시한다.
3. `cosmology_ratio_audit.py:165-174`는 background/growth가 없다고 보고하지만
   `ce_residual_forward_model.py:106-118,1466-1526`에는 둘 다 있다.
4. 3-layer는 `validate_manuscript.py:831-850`과
   `cosmology_discrimination_gates.py:53-70` 두 판본인데, standalone regression과
   active/historical selector가 없다.
5. `cosmological_constant_holographic_gate.py:28`은 `Omega_L=0.6891`을 hardcode하고
   제거된 3-layer 설명에 의존한다; direct test consumer가 없다.
6. H0 toy island인 `hubble_tension.py`와 세 H0 문서는 test/import 연결이 없고,
   문서가 요구하는 `examples/physics/h0_readout/`도 없다.
7. `cosmology_ratio_audit.py:109-138`의 네 관측 baseline과 scorecard 관측값은
   machine-readable source/covariance manifest가 없다.

## 9. 제안 one-source-of-truth 구조

### 9.1 기계 원장

제안 파일은
`reality_stone/python/reality_stone/clarus/cosmology_registry.py` 하나다. 이 파일이
다음 typed entry를 가진다.

- `CoreChain`: 입력 alpha, scheme/scale, 식 버전, exact delta/D/q/s, solver residual
- `DensityConfig`: `model_id`, core version, 분할식, supplied quantities, raw/normalized values
- `RouteClaim`: target claim과 route claim, 형식 지위, counterexample, predecessor artifact
- `CompatibilityAlias`: 기존 symbol, 가리키는 entry, 반올림 정책, removal 없음
- `ObservationRef`: 관측 manifest ID만 보관하고 관측값 자체는 보관하지 않음

활성 수학 정본은 `CE_CORE_EXACT_V1`; 제품 호환 기본값은
`LEGACY_ROUNDED_RUNTIME_V1`; 조건부 background/growth 기준은
`CE_RESIDUAL_FLAT_LCDM_GR_V1`로 분리한다. density physics의
`scientific_default`는 blind model selection 전에는 비워 둔다.

### 9.2 문서 원장

제안 문서 기준점은 `docs/3_상수/00_우주론_원장.md`다. 기존
`docs/axium.md`, 강의, 2장, 3장 문서를 이동하거나 삭제하지 않고, 각 수치표 옆에
`model_id`와 이 원장 링크를 붙인다. 과거 설명은 “Historical route” 블록으로
남기며, 원장의 status matrix가 활성 여부를 결정한다.

### 9.3 관측 원장

`benchmarks/cosmology/observations_v1.json`을 유일한 관측 SOT로 두고, scorecard,
ratio audit, residual model은 source ID로만 참조한다. 자료 갱신은 새 manifest
version으로 추가하며 기존 snapshot을 덮어쓰지 않는다.

## 10. 비파괴 통합 순서

1. **baseline 동결**: 현재 commit, `git status`, legacy q/triplet/LO/3-layer/H0
   출력을 characterization fixture로 기록한다. 선행 run artifact는 immutable
   evidence로 참조한다.
2. **registry만 추가**: 아직 소비자를 전환하지 않고 exact chain, named density
   configs, route IDs, provenance schema를 추가한다. exact/legacy 계산을 모두 test한다.
3. **호환 alias 추가**: 기존 API는 다음처럼 그대로 작동하게 한다.

   ```text
   ACTIVE_RATIO = LEGACY_ROUNDED_RUNTIME_V1.active_ratio
   STRUCT_RATIO = LEGACY_ROUNDED_RUNTIME_V1.struct_ratio
   BACKGROUND_RATIO = LEGACY_ROUNDED_RUNTIME_V1.background_ratio
   BOOTSTRAP_CONTRACTION = LEGACY_ROUNDED_RUNTIME_V1.contraction_display
   EPSILON_SQUARED_LEGACY = LEGACY_DELTA_5DP_V1.q_ext
   ```

   `EPSILON_SQUARED_LEGACY`를 survival alias로 만들지 않는다.
4. **solver 이중 경로**: `BootstrapSolver`의 현재 `DELTA=0.17776`은 legacy mode로
   보존한다. 새 `q_ext_exact(CORE_EXACT_V1)` 경로를 추가하고 두 결과가 왜 다른지
   metadata와 test로 고정한다.
5. **claim 분할**: `q theorem`, `legacy direct readout axiom`, `two-dust composition`,
   `current/freeze-out research route`, `Omega_b target`을 서로 다른 claim ID로 만든다.
   반례가 맞은 route만 inactive로 표시하고 target·값·역사판을 삭제하지 않는다.
6. **consumer 단계 전환**: ratio audit -> residual -> cosmology CLI -> runtime 순으로
   한 파일씩 registry를 읽게 한다. 매 단계에서 old/new 출력 동등성 fixture를
   통과시키고 되돌릴 수 있게 alias를 유지한다.
7. **background 통합**: residual의 flat w0-wa/GR 경로를 현재 computational
   reference로 지정한다. legacy `cosmology.py`와 SFE growth는 named alternative로
   남긴다. H0/rd/sigma8는 CE prediction이 아니라 supplied input임을 출력한다.
8. **H0 경로 보존·격리**: `67.247...` phase output과 `73.04` toy input을 별도
   config로 등록한다. missing `h0_readout` bundle은 `unavailable`로 fail-closed하고
   파일/문서를 삭제하지 않는다.
9. **관측 manifest 추가**: 기존 literal을 당장 지우지 않고 manifest 값과 byte-for-
   byte/parity check를 먼저 건다. source URI, release, covariance hash, blind split이
   없는 row는 score 산출에서 제외하되 historical display는 유지한다.
10. **문서 참조 전환**: 기존 모든 우주론 표에 `core_version`, `density_model_id`,
    `observation_id`, `formal_status`를 추가한다. 중복 수치를 없애는 대신 registry와
    일치하는지 CI가 검사하게 한다.
11. **closure gate**: 다음을 fail-closed로 검사한다: q/s 의미 반전, exact와 legacy
    q 무표시 혼용, raw triplet 합과 normalized background 혼용, model ID 없는
    density 출력, source/covariance 없는 관측 점수, target-aware H0/primordial
    결과의 prediction 라벨.
12. **deprecation은 마지막**: 최소 두 번의 호환 릴리스와 전체 consumer migration
    뒤에도 alias를 바로 삭제하지 않는다. 먼저 warning과 replacement ID만 제공한다.

이 순서는 기존 성공 경로를 내리지 않으면서도, 새 exact chain과 새 물리 route를
추가할 수 있다. rollback은 각 consumer가 alias import로 돌아가는 한 파일 단위
변경으로 가능하다.

## 11. dirty-worktree 경계

이 inventory 시작 시 worktree는 이미 광범위하게 dirty였다. 특히 조사 범위와
가까운 다음 항목은 이 run의 변경으로 간주하지 않았다.

- `tests/test_canonical_document_policy.py`는 수정 상태였다. diff는 삭제된
  `ce-paper-writer` 참조와 Gate 문구 예외를 다루며, 여기서 인용한
  `REMOVED_PARENT_PATTERNS`의 3-layer 정책 자체를 이 inventory가 만든 것이 아니다.
- `reality_stone/python/reality_stone/clarus/__init__.py`,
  `dimensionless_checker.py`, `unified_metric.py` 등은 선행 사용자 변경 상태였다.
- `_workspace/ce/cosmology-density-bridge-derivation-20260815/`,
  `_workspace/ce/cosmology-theory-repository-audit-20260815/`,
  `examples/physics/density_bridge_variational_audit.py`,
  `tests/test_density_bridge_variational_audit.py`는 선행/untracked 자료였다.
- 현재 full-closure run의 다른 artifact와 verifier는 병렬 연구자의 산출물이며 이
  inventory의 수정으로 귀속하지 않는다.

이 파일은 위 상태를 증거로 읽었을 뿐 product/canonical 파일의 dirty diff를
정리하거나 덮어쓰지 않았다.

## 12. 최종 U1/U7 판정

- **U1 inventory: COMPLETE.** exact chain, legacy rounded chain, LO, 두 3-layer
  판본, NLO, runtime default를 분리했고 소비자·충돌·통합 위치를 지정했다.
- **q/s 의미 판정: COMPLETE.** q는 extinction, s는 survival이다. legacy epsilon
  이름은 q compatibility alias로만 허용한다.
- **density route 통합 설계: COMPLETE.** direct-readout 역사판과 target은 보존하고,
  반례가 맞은 보편 유도만 route 단위로 비활성화한다. current/freeze-out 경로가
  다음 활성 연구경로다.
- **background/H0/growth inventory: COMPLETE.** residual GR path, legacy LO/SFE,
  phase H0, target-aware H0 toy, missing covariance path를 분리했다.
- **U7 provenance inventory: COMPLETE.** 관측 snapshot을 theory constants와
  분리하고 source/covariance/blind manifest의 정확한 SOT와 migration 순서를
  제시했다.
- **실제 canonical migration: NOT EXECUTED BY DESIGN.** 본 subtask의 권한은
  read-only inventory와 이 artifact 작성까지이며, 위 순서는 후속 구현 gate다.
