# CE 우주론 형식 지위 감사

Status: COMPLETE

Gate: PASS

## 0. Gate의 정확한 뜻

이 `PASS`는 **현재 CE가 검증된 우주론이라는 뜻이 아니다.** 이번 run의
활성 결론에서 완전 반례·잘못된 provenance가 있는 부모 주장을 모두
제외하고, 아래의 좁은 정리와 조건부 산출만 남겼다는 형식 감사 판정이다.
현재 checkout에서 새로 `[예측]`으로 승격되는 우주론 항목은 **0개**다.

읽기 전용 계약 때문에 제품 소스는 고치지 않았다. 따라서 소스에 남아 있는
`ce_prediction`, `zero free parameters`, report-only 명령의 exit 0 같은
표현은 이 run의 결론을 번복하지 못하는 **격리된 구현·표기 부채**다. 그
표현을 근거로 출판·release하거나 우주론 검증을 주장하는 것은 계속
금지한다. 즉 감사 산출물의 gate는 통과했지만, **이론 승격 gate와 저장소
release gate는 통과하지 않았다.**

감사 입력 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`는
모두 `Status: COMPLETE`였고, 코드 inventory와 실행 ledger도 함께 대조했다.

## 1. C1--C6 최소 주장 원장

`현재 지위`는 소스가 직접 쓰거나 실행 경로가 암시하는 지위다. `실제 지위`는
정의역·독립 전제·관측 절차를 회계한 뒤의 판정이다. `P0-종결`은 소스를
고쳤다는 뜻이 아니라 **반례가 있는 부모 주장을 이번 run의 활성 결론에서
삭제해 열린 P0로 남기지 않았다**는 뜻이다.

| Claim ID | 파일:줄 | 최소 주장 | 현재 지위 | 실제 지위 | 심각도·처분 | 보존 가능한 좁은 결론 |
|---|---|---|---|---|---|---|
| C1.1 | `reality_stone/python/reality_stone/clarus/core_axioms.py:294-365`; `11-math.md:61-103` | $D>1$에서 $q=e^{-D(1-q)}$의 비자명 최소근이 $(0,1/D)$에 유일하고 $q=1$은 다른 근이다. | 정본 정리 | **[정리]** | PASS | 두 가지의 존재·유일성, 작은 가지의 $Dq<1$ 안정도, Lambert-$W$ 표현을 보존한다. |
| C1.2 | `reality_stone/python/reality_stone/clarus/bootstrap_solver.py:23-27,43-54,130-243`; `11-math.md:105-135` | 코드의 $D=3.17776$에서 저근 $q=0.048646633337\ldots$를 푼다. | 수치 검증/물리 readout | **[산출]** | solver 자체 PASS; 원장 정밀도 P1 | **반올림된 그 $D$**에 대한 저근과 잔차만 보존한다. |
| C1.3 | `bootstrap_solver.py:4-7`; `docs/9_등호이전/05n_CE_cosmology_modern_audit.md:55` | 작은 고정점을 오늘의 $\Omega_b$와 동일시한다. | 유도된 baryon readout처럼 표기 | **[공리: 물리 사상]**; 그 사상의 유도는 [미완성] | **P0-종결: 예측/정리 지위 삭제** | 명시적 경계조건으로 $q=\Omega_b(t_0)$를 채택한 조건부 계산만 가능하다. |
| C1.4 | `reality_stone/python/reality_stone/clarus/constants.py:26`; `11-math.md:123-139` | `ACTIVE_RATIO=0.0487`이 exact fixed-point output이다. | 상수 원장 값 | **[경험식: 반올림 snapshot]** | P1 | exact root가 아니라 표시 정밀도의 상수임을 밝힐 때만 보존한다. |
| C2.1 | `examples/physics/ce_residual_forward_model.py:825-855`; `tests/test_ce_residual_forward_model.py:494-504` | $\Omega_b$가 `ce_prediction`이다. | [예측] 문자열 | **[공리: 물리 사상]** | **P0-종결: `ce_prediction` 결론 제외** | supplied/assumed density boundary로만 사용한다. |
| C2.2 | `ce_residual_forward_model.py:845-855`; `tests/scorecard.py:173-205` | $\Omega_{\rm DM},\Omega_\Lambda$ 분할이 고정점에서 유도된 `ce_prediction`이다. | [예측]과 Phenomenology가 충돌 | **[경험식]** | **P0-종결: 유도·예측 지위 삭제** | 분할 ansatz를 이름 붙인 branch의 조건부 입력만 남긴다. |
| C2.3 | `examples/physics/cosmology.py:344-366`; `ce_residual_forward_model.py:43-98`; `11-math.md:162-198` | 하나의 고유한 CE 배경이 있다. | 실행물마다 기본 CE 배경 | **[미완성: branch 선택]** | **P0-종결: 단일 배경 부모 주장 삭제** | LO $(0.307918,0.692082)$와 normalized constants $(0.310969,0.689031)$를 서로 다른 조건부 branch로만 보존한다. |
| C3.1 | `ce_residual_forward_model.py:46-70,926-955`; `docs/5_유도/04_Dark_Energy_Derivation.md:196-203` | 평탄 FLRW, CPL, GR, 외부 $H_0,r_d,\sigma_{8,0}$를 채택한다. | model/default | **[공리]** | PASS, 단 domain 검사는 P1 | 가정과 양의 scale을 명시한 branch를 보존한다. |
| C3.2 | `ce_residual_forward_model.py:926-990,1131-1165`; `11-math.md:201-264` | $E(a)$, 거리, dust+$\Lambda$ 나이 계산이 정의에서 따른다. | forward prediction | **[산출]** | PASS | supplied background에서의 거리·나이 산출과 analytic limit 일치를 보존한다. |
| C3.3 | `ce_residual_forward_model.py:1-5,947-990`; `11-math.md:216-264` | $\mu=1$의 표준 선형 GR 성장 계산이 정의에서 따른다. | forward prediction | **[산출]** | PASS | 주어진 background와 normalization 아래의 GR 성장만 보존한다. |
| C3.4 | `ce_residual_forward_model.py:55-58`; `docs/5_유도/04_Dark_Energy_Derivation.md:151,191` | $\mu\ne1$ 결합이 CE에서 유도된다. | configurable coupling | **[경험식]** | P1 | 별도 coupling ansatz로 표시한 수치 실험만 허용한다. |
| C3.5 | `examples/physics/cosmology.py:20-54,136-146,368-373`; `11-math.md:266-285` | log-grid의 cumulative $S(a)$와 의존 growth가 현재 구현으로 검증됐다. | [산출] | **삭제** | **P0-종결: branch 전체 제외** | 기본 ratio branch와 residual model의 별도 적분은 이 반례와 독립이므로 보존한다. |
| C3.6 | `reality_stone/python/reality_stone/clarus/dimensionless.py:240-252`; `dimensionless_checker.py:482-525`; `11-math.md:470-493` | 우주론 식 전체가 dimensionless gate로 검증됐다. | 등록 checker PASS | **[산출: 수동 감사]**, 자동 전체 보장은 [미완성] | P1 | 고정점/CPL/BAO/growth의 명시적으로 검사한 인자만 무차원이라고 말한다. |
| C3.7 | `ce_residual_forward_model.py:51-58,1525-1527`; `11-math.md:458-469` | $S_8$은 CE 독립 예측이다. | forward output | **[산출: 외부 $\sigma_8$ 재사용]** | P1; raw/normalized 차이는 P2 | 선택한 $\sigma_{8,0}$와 $\Omega_m$에서 계산한 파생량으로만 보존한다. |
| C4.1 | `examples/physics/primordial_spectrum_readout_gate.py:28-64,124-140`; `11-math.md:375-404` | raw 및 각 projector의 $A_s$ 산술이 맞다. | candidate/readout | **[산출]** | arithmetic PASS; 물리 bridge P1 | 다섯 식의 수치값과 raw route가 관측에서 크게 벗어난다는 계산만 보존한다. |
| C4.2 | `primordial_spectrum_readout_gate.py:91-121,149-159`; `docs/3_상수/7_우주론.md:232-233,318` | 가장 가까운 projected $A_s$가 사전 예측이다. | candidate/Bridge이지만 근접값 강조 | **[경험식]** | P1; 예측 지위 제외 | target-aware 다중후보 readout으로만 보존한다. |
| C4.3 | `examples/physics/hubble_tension.py:7-18,259-320,324-347`; `11-math.md:287-319` | 현재 toy가 baryon-aware acoustic matching으로 $\Delta H_0=+5.5595$를 산출한다. | Open/Phenomenology readout | **삭제** | **P0-종결: 수치 closure 제외** | 수정·사전등록 전의 toy flow ODE라는 연구 아이디어만 남긴다. |
| C4.4 | `examples/physics/cosmological_constant_holographic_gate.py:7-13,55-58`; `11-math.md:321-345` | 같은 Planck-mass convention에서 de Sitter entropy와 Friedmann 식의 관계가 성립한다. | absolute-scale gate의 일부 | **[정리]** | PASS | $\rho_\Lambda=\Omega_\Lambda(3/8)M_P^4/S_{\rm dS}$ 항등식만 보존한다. |
| C4.5 | `cosmological_constant_holographic_gate.py:10-16,68-85`; `docs/3_상수/7_우주론.md:175,287-290` | phase-area law가 무입력으로 유도됐다. | zero-free bridge | **[공리]**; action/covariant 유도는 [미완성] | P1 | 모든 integer·부호·계수를 공개한 단일 entropy ansatz로만 보존한다. |
| C4.6 | `cosmological_constant_holographic_gate.py:68-103`; `docs/2_경로적분과_응용/validate_manuscript.py:437-475` | $H_0$와 $\rho_\Lambda^{1/4}$가 서로 독립인 zero-parameter 예측 두 개다. | code-only 강한 결론 | **삭제** | **P0-종결: 독립 절대척도 부모 주장 제외** | phase-area 공리를 넣었을 때 하나의 $H_0$ scale을 환산한 조건부 산출만 남긴다. |
| C4.7 | `cosmological_constant_holographic_gate.py:89-99`; `11-math.md:365-374` | 출력한 122.59가 완전한 hierarchy exponent다. | [산출] | **[산출]**이지만 현재 표기는 불완전 | P1 | correction과 prefactor를 포함한 122.94481만 완전한 환산값으로 보존한다. |
| C5.1 | `examples/physics/cosmology_ratio_audit.py:109-174`; `10-sources.md:21-38` | 네 압축 baseline이 4% 이내이므로 관측 지지가 있다. | coverage PASS | **[경험식]** | P1; hybrid 행은 P0-종결 | uncertainty/covariance 없는 중심값 diagnostic이라는 사실만 보존한다. |
| C5.2 | `tests/scorecard.py:195-205`; `10-sources.md:18-22` | DESI DR2+CMB의 $\Omega_\Lambda=0.693\pm0.005$이고 offset은 $-0.78\sigma$다. | scored observational reference | **삭제: provenance 불성립** | **P0-종결: 해당 행을 증거에서 제외** | 공식 flat $\Lambda$CDM $\Omega_\Lambda=0.6973\pm0.0036$를 같은 모형에서 참고하면 offset은 $-2.28\sigma$다. |
| C5.3 | `ce_residual_forward_model.py:1077-1126,1324-1445`; `10-sources.md:89-120` | 내장 DESI DR2 13점과 full covariance에서 고정 모형의 fit을 계산한다. | validation gate | **[경험식: 이미 본 자료의 진단]** | 수치 PASS, provenance P1 | fixture가 SPD일 때 $\chi^2=37.1003/13$, $p=3.996\times10^{-4}$라는 exploratory rejection만 보존한다. scale-fit은 CE 예측이 아니다. |
| C5.4 | `ce_residual_forward_model.py:1168-1193,1352-1377`; `11-math.md:408-425` | parser가 받아들인 모든 행렬에 $\chi^2$ 해석이 가능하다. | covariance 산출 | **[산출]**, 단 SPD가 독립 전제 | P1 | SPD가 따로 보장된 frozen covariance의 $r^TC^{-1}r$만 보존한다. |
| C5.5 | `experiments/preregistration/cosmology_future_holdout_v2.json:187-234`; `validate_holdout_manifest.py:357-492,928-940` | future holdout으로 예측 검증이 끝났다. | frozen manifest | **[미완성]** | P1; 평가 주장 제외 | manifest 문법 VALID, holdout `unassigned`, 평가 `NOT_READY`만 보존한다. |
| C5.6 | `examples/physics/cosmology_discrimination_gates.py:191-198`; `10-sources.md:29-33` | 두 $\Omega_{\rm DM}/\Omega_\Lambda$ 값과 sigma가 동결된 관측 gate다. | current-status scenario | **[미완성]** | P1 | joint chain·모형·중성미자 convention이 pin되기 전 `UNVERIFIED`로 둔다. |
| C5.7 | `ce_residual_forward_model.py:51-58,1525-1527`; `cosmological_constant_holographic_gate.py:79-85`; `10-sources.md:45-52` | $S_8$ 및 $\rho_\Lambda^{1/4}$ 근접은 독립 교차검증이다. | observational comparison | **[산출: 외부량 재표현]** | P1 | 외부 $\sigma_8$ 또는 $(H_0,\Omega_\Lambda)$에서의 파생 readout으로만 보존한다. |
| C6.1 | `artifacts/validation-command-ledger.md:14-45`; `11-math.md:530-568` | focused tests가 등록 계산과 schema를 재현한다. | validation PASS | **[산출]** | PASS | 91개 math-lane test와 85개 focused test의 구현 회귀 통과만 보존한다. |
| C6.2 | `ce_residual_forward_model.py:1644-1708`; `tests/run_validation.py:342-364`; `tests/scorecard.py:372-399,501-511` | process exit 0이 과학적 PASS를 뜻한다. | CLI가 암시할 수 있음 | **삭제** | **P0-종결: report-only로 해석 고정** | exit 0은 프로그램 실행 성공뿐이며 출력의 `REJECT/CAUTION/FAIL`이 과학 판정이다. |
| C6.3 | `artifacts/validation-command-ledger.md:78-90` | 저장소 전체 회귀가 깨끗하다. | 일반 test status | **[산출]** | P1 release block | 현재 결과 `49 failed, 2500 passed, 14 skipped, 41 errors`; 대부분 범위 밖이지만 clean-repo 주장은 금지한다. |
| C6.4 | `docs/2_경로적분과_응용/validate_manuscript.py:1177,1203-1264`; `artifacts/validation-command-ledger.md:25-32` | 문서 validator PASS가 CE 물리 closure를 증명한다. | document/arithmetic gate | **[산출]** | PASS only after narrowing | 47/47 구현 검사는 보존하되 validator 자체가 보고한 active `[예측]` 0, CE-specific physical closure 0을 함께 유지한다. |

## 2. P0 반례 폐쇄 원장

다음 P0는 소스 수정으로 해결한 것이 아니라, 반례가 배제한 부모 결론을
활성 주장 집합에서 제거해 폐쇄했다. 이 목록의 항목은 후속 최종 보고서에서
생존 정리의 전제로 재사용할 수 없다.

| Closure ID | 해당 Claim | 완전 반례·결정적 결손 | 삭제·축소 범위 | 상태 |
|---|---|---|---|---|
| CL-P0-01 | C1.3, C2.1 | 고정점 방정식에는 시간, critical density, frame 또는 관측 사상이 없다. 정본도 사상을 `[미완성]`으로 둔다. | $q\to\Omega_b$의 정리·예측 지위를 삭제; 명시적 물리 사상 공리만 허용. | CLOSED-EXCLUDED |
| CL-P0-02 | C2.2, C2.3 | DM/DE 분할은 방정식에 없고 live LO와 normalized-constants 배경의 $\Omega_m$이 약 0.99% 다르다. | 유도된 세 밀도분율과 단일 고유 CE 배경을 삭제; branch별 조건부 산출만 허용. | CLOSED-EXCLUDED |
| CL-P0-03 | C3.5 | uniform-grid Simpson을 logspace에 넣어 $a\simeq0.1$에서 약 +33.9% 오차가 난다. | cumulative $S(a)$와 그 의존 growth 전체를 삭제. | CLOSED-EXCLUDED |
| CL-P0-04 | C4.3 | radiation 배경에서 $-12\Omega_r$가 빠졌고, `om_b_h2=0.001`과 0.1이 동일한 $\theta_\star$를 낸다. | 현재 $\Delta H_0=+5.5595$와 baryon-aware CMB matching 결론을 삭제. | CLOSED-EXCLUDED |
| CL-P0-05 | C4.6 | $H_0$, $S_{\rm dS}$, $\rho_\Lambda$는 같은 Friedmann–entropy 항등식의 한 scale이며 phase-area, $d,N_g,\alpha_s,\Omega_\Lambda$가 추가된다. | zero-input·독립 두 예측·우주상수 문제 해소 결론을 삭제; 항등식과 조건부 one-scale 산출만 보존. | CLOSED-EXCLUDED |
| CL-P0-06 | C5.1, C5.2 | 공식 DESI DR2 flat $\Lambda$CDM 값은 $\Omega_\Lambda=0.6973\pm0.0036$이며 저장소 값은 식별되지 않는다. `Planck_ACT_SPT_combined`도 단일 posterior가 아니다. | 두 잘못된 baseline을 관측 증거에서 삭제; versioned likelihood만 허용. | CLOSED-EXCLUDED |
| CL-P0-07 | C6.2 | 고정 DESI 모형은 `REJECT`, 전체 validator는 `CAUTION`인데 관련 CLI는 exit 0이다. | exit 0을 과학적 PASS로 읽는 부모 결론을 삭제; 모든 명령을 report-only로 격리. | CLOSED-EXCLUDED |

열린 P0 수: **0**. 소스에 남은 문자열·exit policy는 위 결론을 다시
활성화하지 않는 조건으로 구현 부채다. 그 조건을 어기면 즉시 gate는
`REVISE`로 되돌아간다.

## 3. 숨은 공리와 미완성 bridge

다음은 고정점 정리에서 따라오지 않는 독립 전제다.

1. $D_{\rm eff}$를 만드는 $\alpha_s$, electroweak 조합과 반올림 정밀도.
2. $q=\Omega_b(t_0)$라는 시간·frame·critical-density 물리 사상.
3. LO, three-layer, normalized-constants 중 분할 branch 선택.
4. 평탄성, GR, CPL 형태 및 $w_0,w_a$.
5. 외부 $H_0,r_d,\sigma_{8,0},T_{\rm CMB},N_{\rm eff}$와 재결합 history.
6. $\mu(a)$ 또는 $\epsilon(a)$ flow coupling law와 초기조건.
7. primordial source·projector·geometry exponent·$N_e$·reheating map.
8. phase-area law, $d=3$, $N_g=12$, correction 부호와 Planck-mass convention.
9. likelihood release, covariance, nuisance, dof, 후보 집합과 kill rule.

미완성 bridge는 (i) 밀도 사상의 action/관측 유도, (ii) dark-sector
covariant action과 perturbation, (iii) Mukhanov–Sasaki normalization 및
reheating, (iv) radiation·baryon-aware Einstein–Boltzmann $H_0$ 계산,
(v) 독립 future holdout 배정, (vi) TDCOSMO·joint-chain covariance ingest다.
`12-routes.md`의 A--F를 검토했으며, 현재 즉시 완결 가능한 경로는 A의
정확한 고정점 코어 축소뿐이다. B--F는 모두 새 공리나 아직 없는 자료·작용이
필요하므로 차단 대신 `[미완성]`으로 유지한다.

## 4. 살아 있는 활성 결론

이번 run의 최종 문서가 사용할 수 있는 결론은 다음 네 묶음뿐이다.

1. **[정리]** $D>1$ 고정점의 두 양의 가지와 비자명 최소근의 유일성·안정도.
2. **[산출]** 선택한 $D$에서의 수치근. 단 exact 원장과 반올림 원장을
   구분한다.
3. **[산출]** 명시적으로 공급한 평탄 FLRW/CPL/GR 경계조건에서의
   background·거리·나이·기본 성장. cumulative $S(a)$ branch는 제외한다.
4. **[정리]/[산출]** 같은 Planck convention의 Friedmann–de Sitter entropy
   항등식과, SPD가 별도로 보장된 frozen covariance의 이차형식.

관측에 대해서는 내장 DESI DR2 고정모형이 exploratory 자료에서
`p=3.996e-4`로 기각된다는 진단과, future holdout이 아직 `NOT_READY`라는
사실만 남는다. 둘 다 CE 고유 물리 사상의 확인이나 사전 예측 성공이 아니다.

## 5. 집계와 gate 근거

- 검사한 최소 비자명 주장: **32개**
- 완전 증명되어 보존한 정리: **2개**
- 조건부·수치 산출로만 보존: **12개**
- 명시 공리로 강등·고정: **4개**
- 경험식으로 제한: **6개**
- 미완성 유지: **3개**
- 완전 삭제한 부모 주장: **5개**
- 추가로 정리/예측 지위를 제거해 공리·경험식으로 축소한 부모 형태: **5개**
- 현재 `[예측]`: **0개**
- 열린 P0: **0개**

P1은 provenance, SPD/domain gate, 원장 정밀도, 자동 무차원 coverage와
action-level bridge의 결손으로 남아 있다. 이들은 현재 살아 있는 좁은
정리의 참·거짓을 뒤집지 않지만, 관측 예측 또는 이론 승격을 막는다. P2는
$S_8$ raw/normalized convention, `survival rate` 용어, branch label 정리다.

따라서 형식 감사 자체는 `PASS`다. 반례 있는 부모를 다시 활성화하거나
`[예측]`으로 승격하면 이 판정은 무효다. 우주론 이론으로의 승격 재개 조건은
`12-routes.md`의 kill test를 통과하는 action/bridge, versioned likelihood와
독립 holdout, 그리고 소스의 지위·exit policy 정정이다.

## 6. 종료 체크리스트

- [x] 모든 C1--C6 비자명 주장에 Claim ID와 실제 지위를 부여했다.
- [x] 완전 반례가 있는 부모 주장에 삭제 범위와 좁은 생존 결론을 적었다.
- [x] 열린 P0를 0으로 만들기 위해 소스 수정이 아니라 활성 결론 제외를
  명시했고, 이를 이론 검증 PASS와 구분했다.
- [x] 숨은 공리, 미완성 bridge, P0/P1/P2와 재개 조건을 기록했다.
- [x] 제품 소스를 수정하지 않았다.
- [x] `Status:`와 `Gate:` 프로토콜 줄을 넣었다.
