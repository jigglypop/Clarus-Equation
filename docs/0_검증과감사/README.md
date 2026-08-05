# 0. 검증과 감사

이 폴더는 CE 문서군의 주장 강도, 미해결 항목, 검산 상태를 관리하는 정본 카테고리다. 여기 있는 문서는 새 물리식을 추가하기보다, 기존 문서가 무엇을 `Exact`, `Selection`, `Bridge`, `Phenomenology`, `Open`, `Open test`로 주장할 수 있는지 정리한다.

## 살린 문서

| 문서 | 역할 | 판정 |
|---|---|---|
| [PROOF_STATUS_MATRIX.md](PROOF_STATUS_MATRIX.md) | 전체 증명 등급표 | 정본 |
| [PROOF_VALIDATION_LEDGER.md](PROOF_VALIDATION_LEDGER.md) | 실제 검산 이력과 판정 변경 기록 | 정본 |
| [VALIDATION_FRAMEWORK.md](VALIDATION_FRAMEWORK.md) | 남은 증명 요구사항과 tier별 blocker | 참고 정본 |
| [MATHEMATICAL_PHYSICS_ISSUES.md](MATHEMATICAL_PHYSICS_ISSUES.md) | 수학/물리 문제점 감사 목록 | 참고 정본 |
| [BRIDGE_B2_DERIVATION.md](BRIDGE_B2_DERIVATION.md) | \(P_{\text{survive}}\leftrightarrow\Omega_b\) bridge 보강 기록 | bridge 보조 |
| [CORE_STRENGTHENING_LOOP.md](CORE_STRENGTHENING_LOOP.md) | 코어 반례군을 최소 원리·정리·코드 gate로 줄이는 반복 보강 정본 | working canonical |
| [A1_Q0_COVARIANT_ACTION_LOOP.md](A1_Q0_COVARIANT_ACTION_LOOP.md) | 보통 Hessian 반례, 공변 장공간 Hessian, metric variation과 Q0 통과 조건 | working canonical |
| [CE_TWO_POINT_AND_VERTEX_LOOP.md](CE_TWO_POINT_AND_VERTEX_LOOP.md) | 역상관 scale의 pole 비식별 반례, 선택적 포탈 Q0.4–Q0.5 tree pole·vertex와 29.65 MeV/43.77 GeV 양립성 정리 | control certificate |
| [CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md](CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md) | action provenance, renormalized pole fail-closed gate와 선택적 portal scalar one-loop 방사 안정성 감사 | control certificate |
| [CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md](CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md) | 실제 CE ensemble 부재 감사, connected/jackknife·유한 positivity gate와 spectral 비유일성 구성 반례 | control certificate |
| [CLARUS_RESONANT_MATTER_LOOP.md](CLARUS_RESONANT_MATTER_LOOP.md) | 위상 중첩·invariant pair·Floquet/Bogoliubov toy와 cutoff-independent global throat source-tail 감사 | working canonical |
| [EXTERNAL_FIELD_TO_MATTER_REANALYSIS.md](EXTERNAL_FIELD_TO_MATTER_REANALYSIS.md) | 광학 analogue DCE·CMS 광자융합·1T-TaS2 공개 원자료의 수치 재현과 Clarus bridge 실패 감사 | external-data certificate |
| [FUSION_RESONANCE_LOOP_ENGINEERING.md](FUSION_RESONANCE_LOOP_ENGINEERING.md) | 정적 spacelike 교환/pole 공명 분리, 레거시 WKB 교정과 점화 claim-lock | control certificate |
| [FUSION_FULL_LOOP_ENGINEERING.md](FUSION_FULL_LOOP_ENGINEERING.md) | Z2 쌍·명시적 파괴·coherent 구동·열반응률/Lawson·NIF 전분기 fail-closed 감사 | control certificate |
| [FUSION_EQUATION_ITERATION_LOOP.md](FUSION_EQUATION_ITERATION_LOOP.md) | 허용 단일/두-scalar 퍼텐셜을 WKB·Bosch--Hale·Lawson까지 전파하고 1% 목표의 모델계열 상한을 인증 | control certificate |
| [FUSION_REMAINING_BRANCHES_LOOP.md](FUSION_REMAINING_BRANCHES_LOOP.md) | 직접 핵자 연산자 UV/EFT·핵물질 gate, 시간의존 강전기장 대조군과 reactor/ICF 경계를 최종 통합 | control certificate |
| [FUSION_DIRECT_SCATTERING_LOOP.md](FUSION_DIRECT_SCATTERING_LOOP.md) | 1% 직접 결합의 free Born np 산란길이와 Hulthén deuteron 이동, full nuclear refit blocker | control certificate |
| [FUSION_FLOQUET_SOURCE_LOOP.md](FUSION_FLOQUET_SOURCE_LOOP.md) | QED Floquet--Volkov 10 keV 식 외삽과 1 keV 공개 benchmark, 입사 pump 장부, CE exact-Z2 beat source-energy no-go 분리 | control certificate |
| [FUSION_FLAVOR_ALIGNED_LOOP.md](FUSION_FLAVOR_ALIGNED_LOOP.md) | 1% 직접 해의 flavor-aligned uds/VLQ UV 매칭과 neutron--Pb·rare-kaon·invisible 제약 경계 | conditional-candidate certificate |
| [FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md](FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md) | D/T one-body finite-size, Pb 각분포/총단면적 recast, kaon NLO·최신 NA62 질량점 보간의 여유 강건성 | fail-closed robustness certificate |
| [FUSION_OPERATOR_ALTERNATIVES_LOOP.md](FUSION_OPERATOR_ALTERNATIVES_LOOP.md) | trace/gluon, p/n-phobic, Pb blind spot, disformal massless upper의 제약 전파 | model-class no-go certificate |
| [FUSION_SPIN_OPERATOR_LOOP.md](FUSION_SPIN_OPERATOR_LOOP.md) | quartet projector 아래 pseudoscalar·axial·vector·spin-2·derivative-node 장거리 해와 질량별 제약 blocker | operator-level fail-closed certificate |
| [FUSION_SPIN_POLARIZATION_CONTROL_LOOP.md](FUSION_SPIN_POLARIZATION_CONTROL_LOOP.md) | 표준모형 D–T 편극의 이상적 quartet 1% 조건부 대조군, 10 keV 열평형 자기장 no-go와 source·pump·retention 장부 | non-CE conditional-control certificate |
| [FUSION_SCALAR_CURRENT_LOOP.md](FUSION_SCALAR_CURRENT_LOOP.md) | flavor-aligned 후보의 핵자 정규화, D/T Helm 형상, scalar radius, sigma-term proxy와 chiral two-body current를 공동 likelihood 요구까지 감사 | scalar-current fail-closed certificate |
| [PROBE_SELECTIVE_DRESSING_AND_PUBLIC_SCAFFOLD_LOOP.md](PROBE_SELECTIVE_DRESSING_AND_PUBLIC_SCAFFOLD_LOOP.md) | pump×controller 선택성, phase/noise held-out, pump-off 3-probe common-kernel과 signed energy ledger; public scaffold/new matter claim 잠금 | working canonical |
| [CAUSAL_MASK_AND_SPACELIKE_MARGINAL_LOOP.md](CAUSAL_MASK_AND_SPACELIKE_MARGINAL_LOOP.md) | \((g,R,S,\tau)\) 결합 crossed-holdout, block/provenance·동시구간 gate와 local-QFT spacelike no-signalling 감사 | control certificate |
| [MULTIMODE_GLOBAL_THROAT_LOOP.md](MULTIMODE_GLOBAL_THROAT_LOOP.md) | cutoff-independent global throat, finite-tail redshift 보강과 explicit scalar \(K/F<0\) 반례 | working canonical |
| [Q0_0_Q0_3_MINIMAL_MANIFEST.md](Q0_0_Q0_3_MINIMAL_MANIFEST.md) | 깨진 \(U(1)\)+\(Z_2\) 싱글릿 통제 절단에서 범위·장공간·배경·게이지/ghost를 한 convention으로 고정 | control manifest |
| [우주론_양자론_루프_감사.md](우주론_양자론_루프_감사.md) | DESI 잔차·\(H_0r_d\) 원인 분해와 표준 QM/\(\Phi\) 형식 감사 | 실행 감사 |
| [문서_전체_완성도_감사.md](문서_전체_완성도_감사.md) | docs 전체 완성도와 보강 순서 | 운영 감사 |
| [미해결_난제_목록.md](미해결_난제_목록.md) | 인류/CE 미해결 난제 카탈로그 | open-problem index |

## 죽인 문서

다음 문서는 정식 카테고리 문서가 아니라 임시 산출물, 생성 덤프, stale roadmap이므로 제거한다.

| 문서/폴더 | 이유 |
|---|---|
| `TEMP.md` | 비공식 대화/임시 사고 로그 |
| `llm_extract/` | working tree에서 생성한 LLM 컨텍스트 덤프 |
| `ANALYSIS_SUMMARY.txt` | stale validation 요약, 정본 문서로 흡수됨 |
| `IMPLEMENTATION_ROADMAP.md` | 오래된 구현 로드맵, 현재 코드/테스트 상태와 불일치 |
| `MASTER_ACTION_PLAN.md` | 루트 임시 계획 문서, 카테고리 문서로 대체 |
| `PROJECT_AUDIT_FINAL_REPORT.md` | 과거 단발 감사 산출물, 현재 ledger/status matrix와 중복 |
| `ALTERNATIVE_PROOFS_AND_DATA_UPDATE.md` | 과거 보강 초안, 현재 proof ledger와 audit 문서로 흡수 |
| `README_VALIDATION.md` | 삭제된 요약/로드맵 파일을 가리키던 stale index |

## 읽는 순서

1. [코어 독자 가이드](../코어_독자_가이드.md): 수식 전에 환기구 비유,
   다공간 재귀, 증명과 bridge의 경계를 잡는다.
2. [A1/Q0 공변 작용 루프](A1_Q0_COVARIANT_ACTION_LOOP.md): Hessian,
   stress tensor와 보존법칙 사이의 열린 조건을 먼저 본다.
3. [Q0.0–Q0.3 최소 manifest](Q0_0_Q0_3_MINIMAL_MANIFEST.md):
   전체 CE+SM과 통제 절단의 통과 표시를 왜 분리하는지 본다.
4. [CE two-point–vertex 루프](CE_TWO_POINT_AND_VERTEX_LOOP.md): 역상관
   scale, tree pole, physical pole과 vertex의 층을 분리한다.
5. [CE renormalized pole–one-loop 루프](CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md):
   실제 action provenance, pole holdout과 방사 안정성의 첫 blocker를 본다.
6. [CE Euclidean correlator–spectral 루프](CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md):
   실제 표본, 유한 positivity, screening energy와 실시간 pole의 경계를 본다.
7. [CORE_STRENGTHENING_LOOP.md](CORE_STRENGTHENING_LOOP.md): 코어를 어떤 반례와 gate로 강화하는지 본다.
8. [EXTERNAL_FIELD_TO_MATTER_REANALYSIS.md](EXTERNAL_FIELD_TO_MATTER_REANALYSIS.md): 실제 공개 데이터가 어디까지 재현되고 Clarus bridge가 어디서 끊기는지 본다.
9. [FUSION_RESONANCE_LOOP_ENGINEERING.md](FUSION_RESONANCE_LOOP_ENGINEERING.md): 핵융합 공명 사슬이 static exchange gate에서 왜 끊기는지 본다.
10. [FUSION_FULL_LOOP_ENGINEERING.md](FUSION_FULL_LOOP_ENGINEERING.md): 살아남은 Z2 쌍·파괴·구동·열반응률·ICF 분기를 전부 닫아 본다.
11. [FUSION_EQUATION_ITERATION_LOOP.md](FUSION_EQUATION_ITERATION_LOOP.md): 허용 식을 열반응률까지 전파하고 어느 모델 변경부터 새 물리인지 확인한다.
12. [FUSION_REMAINING_BRANCHES_LOOP.md](FUSION_REMAINING_BRANCHES_LOOP.md): 직접 연산자와 시간의존 source의 마지막 물리 gate를 확인한다.
13. [FUSION_DIRECT_SCATTERING_LOOP.md](FUSION_DIRECT_SCATTERING_LOOP.md): 직접 결합이 저에너지 핵산란 정밀도에서 무시 가능한지 확인한다.
14. [FUSION_FLOQUET_SOURCE_LOOP.md](FUSION_FLOQUET_SOURCE_LOOP.md): QED 시간의존 식 외삽·공개 benchmark와 CE scalar source 분기를 단위·에너지 장부까지 분리한다.
15. [FUSION_FLAVOR_ALIGNED_LOOP.md](FUSION_FLAVOR_ALIGNED_LOOP.md): 가장 가까운 직접 scalar 후보가 어느 실험 여유에서 멈추는지 확인한다.
16. [FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md](FUSION_FLAVOR_MARGIN_ROBUSTNESS_LOOP.md): finite-size와 constraint-theory 오차를 넣어 중앙 후보의 작은 여유가 강건한지 확인한다.
17. [FUSION_OPERATOR_ALTERNATIVES_LOOP.md](FUSION_OPERATOR_ALTERNATIVES_LOOP.md): 흔한 coupling-basis 우회로가 실제 제약을 피하는지 확인한다.
18. [FUSION_SPIN_OPERATOR_LOOP.md](FUSION_SPIN_OPERATOR_LOOP.md): 스핀·비표준 Lorentz 연산자가 scalar 제약을 우회하는지 확인한다.
19. [FUSION_SPIN_POLARIZATION_CONTROL_LOOP.md](FUSION_SPIN_POLARIZATION_CONTROL_LOOP.md): 새 입자 없는 편극 D–T의 이상적 quartet 1% 계산과 실제 energy-dependent Maxwellian·source 장부 사이의 미검증 경계를 확인한다.
20. [FUSION_SCALAR_CURRENT_LOOP.md](FUSION_SCALAR_CURRENT_LOOP.md): 가장 가까운 scalar 후보의 one-/two-body D/T 핵응답과 최신 sigma-term 정보가 1% 여유를 실제로 인증하는지 확인한다.
21. [PROOF_STATUS_MATRIX.md](PROOF_STATUS_MATRIX.md): 지금 무엇이 닫혔고 무엇이 열려 있는지 본다.
22. [PROOF_VALIDATION_LEDGER.md](PROOF_VALIDATION_LEDGER.md): 왜 그 판정이 되었는지 검산 이력을 본다.
23. [미해결_난제_목록.md](미해결_난제_목록.md): CE가 아직 못 닫은 외부/내부 난제를 확인한다.
24. [VALIDATION_FRAMEWORK.md](VALIDATION_FRAMEWORK.md): 남은 blocker와 필요한 증명/실험을 본다.
25. [우주론_양자론_루프_감사.md](우주론_양자론_루프_감사.md): 우주론·양자론의 최신 실행 잔차와 최소 보강을 본다.
26. [문서_전체_완성도_감사.md](문서_전체_완성도_감사.md): 문서군 정리 우선순위를 본다.

## 운영 규칙

- 새 주장이 나오면 먼저 `PROOF_VALIDATION_LEDGER.md`에 검산 근거를 남긴 뒤 status matrix를 올린다.
- 수학적으로 닫힌 항목과 관측 readout bridge는 반드시 분리한다.
- 임시 분석, 긴 대화 덤프, 생성된 LLM context는 루트나 `docs/`에 남기지 않는다.
- 검산 코드 없이 "해결"로 부르는 문서는 이 폴더의 audit 대상이다.
