# CE 전체 증명 상태표

이 문서는 `docs/` 전체를 처음부터 검증할 때의 판정표다. 목표는 모든 항목을 억지로 `Exact`로 올리는 것이 아니라, 증명이 닫힌 층과 아직 브리지/현상론/재개방인 층을 분리하는 것이다.

## 판정 규칙

| 등급 | 의미 | 채점 방식 |
|---|---|---|
| `Exact` | 정의, 항등식, 정리, 기능방정식으로 닫힘 | 관측값 없이 검산 가능 |
| `Selection` | 분기 선택 또는 정규화 선택까지 명시하면 닫힘 | 대안 배제 조건 필요 |
| `Bridge` | 수학 구조를 물리 관측량에 연결 | 독립 매핑 또는 실험 gate 필요 |
| `Phenomenology` | 유효모형, 보정, 스케일 선택 포함 | 데이터/시뮬레이션 gate 필요 |
| `Open` | 현재 검증 통과로 세지 않음 | scorecard에서 제외 |
| `Open test` | 아직 관측되지 않은 예측 | 사전등록된 실험 gate 필요 |

## 코어 수학

| 항목 | 현재 판정 | 검증 결과 | 남은 조건 |
|---|---|---|---|
| 생존 함수 $S(D)=e^{-D}$ | `Exact/Selection` | Cauchy형 곱셈 조건, 연속성/측정가능성, $S(0)=1$에서 $S(D)=e^{-\lambda D}$; 단위 정규화로 $\lambda=1$ | 곱셈 조건 자체는 물리 가정이다 |
| 자기재귀 고정점 $x=e^{-(1-x)D}$ | `Selection` | $D>1$, $x\in[0,1/D]$에서 닫힌 Lambert-W 해와 수치 고정점 검산 | 물리량 대응은 별도 |
| Newton/bracketed solver | `Exact` for code residual | residual $2.08e-17$, Newton vs bracketed $1.25e-13$ | 없음 |
| Hodge 기반 $d=3$ 선택 | `Selection` | 2-form의 쌍대가 다시 1-form/벡터가 되려면 $d-2=1$ | $N_c=d$ 동일시는 `Bridge` |
| 무차원성 검사 | `Exact` for listed formulas | 7/7 dimensionally consistent | 차원 일치는 참/거짓 판정일 뿐 물리 진실을 보장하지 않음 |
| A1 보통 Hessian $\to$ 공변 $\mathcal K$ | `Open` | 비선형 좌표 반례에서 tensor pullback 20, 보통 Hessian 32, connection 보정 뒤 20을 재현 | 전체 field-space metric/connection, bi-local-to-local projection, gauge quotient와 measure |
| Hessian $\to T_{\mu\nu}$ 및 보존 | `Open/Conditional` | stress tensor는 $\Gamma_{\rm ren}$의 metric variation이고 on-shell Noether identity 아래 보존됨 | A1 kernel과 metric variation의 matching, anomaly·경계항·EOM |
| CE+SM Q0 공변 작용 | `Open` | 포탈 국소 미분과 반례 gate만 통과; 완료 플래그 3개는 모두 `False` | `Q0.0`–`Q0.8` 전체 |

## 상수와 관측량

| 항목 | 현재 판정 | 로컬 검증 | 결론 |
|---|---|---|---|
| $\alpha_s(M_Z)$ | `Selection` | `scorecard.py`: PASS | 입력/스케일 선택 성격이 강함 |
| $\sin^2\theta_W=4\alpha_s^{4/3}$ | `Bridge` | closure gate: scale-consistent input에서 <1%; `scorecard.py`: PASS | 좋은 브리지 후보, 아직 `Exact` 아님 |
| $\Omega_b$ | `Bridge` | dimensionless strict 기준은 PASS; 최신 scorecard의 \(\Omega_bh^2\)는 -1.80 sigma `CAUTION` | 관측 기준을 통일해야 함 |
| $\Omega_\Lambda$ | `Phenomenology` | 최신 scorecard는 +0.60 sigma `PASS`; 더 작은 오차를 쓰는 strict validator는 +2.75 sigma `WARN` | baseline/오차 snapshot을 통일해야 함 |
| $\Omega_{DM}$ | `Phenomenology` | PASS | 유효모형 판정 |
| Higgs mass | `Bridge` | PASS | 공식 기반 독립 유도 감사 필요 |
| $|V_{cb}|$ | `Phenomenology` | LO는 strict 기준 FAIL(+6.58 sigma); NLO electroweak projector $\alpha_s^{3/2}(1+\delta/2\pi)$는 +0.58 sigma | 1-loop projector bridge로 조건부 통과 |
| $|V_{us}|$ | `Phenomenology` | tree $\sin^2\theta_W$는 실패; $\sin^2\theta_W/(1+\alpha_s/2\pi)$는 +0.97 sigma | 1-loop 보정 브리지로 조건부 통과 |
| PMNS $\sin^2\theta_{13}$ | `Bridge` | PASS | 브리지 후보 |
| $A_s$ | `Phenomenology` | raw 총응답은 실패; projected residual-drive readout은 -0.53 sigma | 관측 readout bridge로 조건부 통과, `Exact` 아님 |
| $n_s$ | `Phenomenology` | $1-2/(dD_{\rm eff}12/2)=0.96503$, +0.03 sigma | 전이수 12와 inflationary readout을 받아야 닫힘 |
| $w_0$ | `Phenomenology` | 상수 scorecard의 Planck-family 기준은 PASS; CE density+외부 \(H_0r_d\)+\(w=-1\) 고정 DESI BAO 패키지는 `REJECT` | 단일 상수 일치와 full-cov model fit을 분리 |
| Clarus field pole bridge | `Open test` | $m_\phi=29.64757$ MeV, 3-sigma window $28.388$--$30.908$ MeV | 클라루스장 2점 상관함수 pole / 입자언어 bridge gate 사전등록 완료 |
| 레거시 $Z_2$ 포탈 $(\lambda_{\rm HP}=0.0316,\ m_\Phi=43.77\,{\rm GeV})$ | `Phenomenology` benchmark rejected | 같은 라그랑지안 정규화에서 $\Gamma_{\rm inv}=13.75$ MeV, $\mathrm{BR}_{\rm inv}=0.772$; 공급된 상한 0.11 실패 | coupling을 낮추거나 채널을 닫고 새 benchmark로 재검증; CE 동일시는 별도 |

## 우주론·양자론 실행 게이트

| 항목 | 현재 판정 | 최신 실행 결과 | 남은 조건 |
|---|---|---|---|
| CE density boundary + 외부 \(H_0,r_d\) + \(\Lambda\)CDM BAO | `Phenomenology/REJECT` | DESI DR2 full-cov \(\chi^2=37.1003\), dof 13, \(p=3.996\times10^{-4}\) | 독립 \(H_0r_d\) calibration과 holdout |
| CE density + 외부 \(H_0,T_{\rm CMB}\) + 표준 \(N_{\rm eff}\) + EH hybrid \(r_d\) | `Selection/Approximation/REJECT` | \(z_d=1020.0204,\ r_d=151.3188\,{\rm Mpc}\); DESI DR2 \(\chi^2=40.4682\), dof 13, \(p=1.162\times10^{-4}\) | precision recombination/Boltzmann benchmark; DR2는 untouched holdout 아님 |
| hashed external \(x_e(z)\) drag adapter | `Tooling/Bridge` | CLASS convention \(\tau_{\rm drag}=1\), provenance/hash/grid/cosmology gate; 합성 \(z_d=1059.25,\ r_d=147.6498\,{\rm Mpc}\) 회귀 통과 | 실제 CLASS/HyRec·CAMB export와 solver 자체 \(z_d,r_d\) 교차검증 |
| \(H_0r_d\) scale ablation | `Diagnostic fit` | \(q_*=0.986477\), \(\chi^2=12.6083\), dof 12, \(p=0.398\) | 같은 DESI로 맞춘 값이므로 CE 예측으로 승격 금지 |
| 일반 unitary·density matrix·Born sampling·GKSL | `Tooling/Exact under model` | norm, trace, Hermiticity, positivity, sampling, dephasing 회귀 통과 | CE 고유 증거가 아니라 reference baseline |
| \(\delta^2S/\delta\gamma^2=R\) | `Bridge blocked by type gate` | Hessian kernel/operator와 Ricci scalar의 type 불일치 | 독립장 또는 curvature-assigned readout 중 하나 선택 |
| 독립장 \(\phi\ne R\) + Hessian readout + correlator/KMS/Kossakowski | `Tooling/Bridge` | 분리된 action/EOM, scalar-Hermitian KMS, strict-PSD jump gate와 Gibbs 고정점; SI/reduced rate 규약 분리 | action에서 실제 \(G_\phi,J_\phi\) 유도, scalar→matrix index·단위 mapping, 실제 결합과 장치 데이터 |
| Born measurement bridge | `Open as foundation` | \(|c_k|^2\) prior의 보존은 가능하나 유도·single-shot dynamics 아님 | apparatus instrument, CPTP, no-signalling, holdout |
| ARC 94.6%, \(p_{\rm th}=e^{-2}\) | `Recorded result/Open test` | 현행 코드·seed·raw artifact·QEC 설정 없음 | artifact 복원과 code/noise/decoder별 threshold gate |
| cosmology·quantum future-holdout v1 | `Preregistered/NOT_READY` | canonical self-hash와 code artifact hash, zero-fit 정책, tolerance·hard-kill 규칙 검증; 둘 다 `VALID/unassigned/NOT_READY` | 모델 준비 후 미열람 미래 release를 새 revision에 배정; repo trust anchor의 외부 서명·시각인증 |

## 응용 문서 판정

| 영역 | 현재 판정 | 이유 |
|---|---|---|
| 공학적 활용 | `Phenomenology/Open test` | 수식 구조는 제안 가능하지만 물성/소자/초전도체 설계는 외부 실험 gate 없이는 증명 아님 |
| 뇌/생명 문서 | `Phenomenology/Open test` | 데이터셋 gate와 식별성 검사가 핵심이며 코어 정리의 직접 귀결이 아님 |
| 뇌 프로그래밍 언어 / neural Clarus assembly | `Open test` | 과제 설계의 `READ_*→ROUTE_*` 골격, 공유 표현 frontend/backend 후보와 빠진 `S2` 예측은 `YES`; 공식 processed cross-decoding과 27개 session 경계도 재현했다. 그러나 stationary·state-switching·잠재 common-successor·D1↔D3 frozen transfer·state-parent+rank1 proxy는 모두 `NO`다. 생물학적 callee·과제 상속·고정-neuron opcode·언어 식별도 `NO`, 그 구조의 보편적 존재·부재, unseen composition과 인과 검사는 `TEST_UNAVAILABLE`다. |
| Tafazoli 확산/잡음 프록시 | `Open test` | 공식 파일 MD5와 27개 session을 검증하고 session×D1/D3×raw/demean 108개 단위를 완주했다. covariance ladder 완료는 `YES`지만 state-dependent noise proxy는 `NO`: state-vs-full은 raw -0.2521, demean -0.2544 bits/scalar이고 joint win은 0이다. Markov order 1은 108/108이며 order 2·3 대비 이점은 각각 +0.3177, +0.6411 bits/scalar다. 200 ms semigroup은 경계선(raw median excess 0.01935, pass 0.519; demean 0.01720, pass 0.648), 300 ms는 실패(raw 0.06677, 0.0185; demean 0.05927, 0.0185)했다. reverse 73/forward 35는 기술적 분류일 뿐 gate가 아니다. 시험한 family 중 local affine+isotropic Gaussian proxy가 상대 승자지만 비-Gaussian family와 절대 적합도를 검사하지 않아 Gaussianity는 식별하지 않았다. 생물학적 확산의 보편적 존재·부재는 `TEST_UNAVAILABLE`; 생물학적 확산, 생성적 reverse, score, 인과 기전, 공간 그래프 식별은 모두 `NO`다. |
| AGI/런타임 문서 | `Open test` | 구현 명세와 벤치마크 영역이다. 물리 상수 증명과 분리해야 함 |
| 리만/attention 문서 | `Phenomenology/Open test` | 구조적 아이디어와 엔지니어링 성능 검증의 문제 |

## 이번 검증 명령

```powershell
python reality_stone\python\reality_stone\clarus\bootstrap_solver.py
python tests\scorecard.py
python tests\run_validation.py
python examples\physics\proof_completion_attempt.py
python examples\physics\ce_residual_forward_model.py --bao-dataset desi-dr2-all
python -m experiments.preregistration.validate_holdout_manifest
uv run --extra dev python -m pytest tests\test_bootstrap_solver.py tests\test_dimensionless.py tests\test_layer_a.py tests\test_bridge_gates.py -q
uv run --extra dev python -m pytest tests\test_ce_residual_forward_model.py tests\test_recombination_drag_adapter.py tests\test_quantum.py tests\test_quantum_kms.py tests\test_holdout_preregistration.py -q
```

## 현재 결론

1. 코어 고정점 수학과 solver 검산은 닫혔다.
2. $S(D)=e^{-D}$는 조건부 정리로는 닫히지만, 곱셈/독립 조건을 물리에 부여하는 순간 `Selection/Bridge`가 된다.
3. $\sin^2\theta_W$ 브리지는 수치적으로 강하지만, 아직 물리적으로 `Exact`가 아니다.
4. $|V_{us}|$와 $n_s$는 조건부 `Phenomenology`로 올릴 수 있다.
5. $A_s$는 projected residual-drive gate로 `Phenomenology`까지 올렸다.
6. $|V_{cb}|$는 LO 실패를 유지하되, NLO electroweak projector bridge로 `Phenomenology`까지 올렸다.
7. A1의 보통 Hessian을 공변 spacetime tensor나 stress tensor로 읽는
   단계와 CE+SM Q0 작용은 `Open`이다.
8. 외부 \(r_d\)와 EH hybrid DESI BAO 패키지는 모두 `REJECT`이며,
   \(H_0r_d\) scale 진단의 `PASS`는 CE 예측으로 세지 않는다.
   drag-history adapter는 구현됐지만 실제 precision solver export 검증
   전까지 `Tooling/Bridge`다.
9. 표준 QM과 서로 분리된 action, scalar KMS, PSD Kossakowski reference
   gate는 닫혔지만, 이를 CE action의 실제 스펙트럼과 하나의 단위·index
   사슬로 연결하고 Born·ARC/QEC·장치 데이터에 보내는 단계는 여전히
   `Bridge/Open`이다.
10. future-holdout v1은 사후 조정을 막는 구조적 preregistration만
    통과했다. 실제 데이터는 둘 다 미배정이므로 실증 점수는 0건이며
    `PASS`로 승격하지 않는다.
11. 뇌 과제의 `READ_SHAPE/READ_COLOR → ROUTE_AXIS_1/2` 후보 골격은
   복원했지만 이것은 task design의 factorization이다. 공식 snapshot의
   session-local stationary operator뿐 아니라 같은 parameter budget의
   \(K=2/3\) switching, 잠재 common successor, D1↔D3 frozen transfer와
   state-parent+rank1 proxy도 대조군을 이기지 못했다. 이 snapshot 안의
   shared execution instruction 판정은 `NO`다. 다음 `Open test`는 전뇌
   공통 opcode가 아니라 다영역 `local DSL + narrow communication
   interface`, 최소 충분 회로, 실제 unseen composition과 인과 개입이다.
12. 같은 snapshot의 diffusion probe는 covariance ladder와 1차 Markov
    적합성을 확인했지만 state-dependent noise proxy와 300 ms semigroup을
    통과하지 못했다. reverse 우세 방향은 기술 통계일 뿐이며, 이 결과로
    생물학적 확산·생성적 reverse·score·인과 기전·공간 그래프를 식별하지
    않는다.
13. 전체 CE 문서의 정직한 상태는 “일부 코어는 증명됨, 핵심 작용
    bridge는 열려 있음, 다수 물리 출력은 브리지/현상론, 미관측 예측은
    Open test”이다.

## 증명 완료 경계

현재 관측 scorecard의 채점 행에는 `Open` 항목이 없지만, 이것이 전체
증명의 폐쇄를 뜻하지 않는다. A1/Q0는 관측 scorecard 밖의 선행 구조
의무이며 `Open`이다.

| 항목 | 닫히기 위한 필요조건 |
|---|---|
| A1/Q0 공변 작용 bridge | `A1_Q0_COVARIANT_ACTION_LOOP.md`의 `Q0.0`–`Q0.8`: field-space 공변성, background, gauge/ghost, operator/vertex, stress, renormalization과 재현 gate |
| Clarus field pole bridge | 장 상관함수 pole gate는 등록됐다. 미관측 예측이므로 실제 실험 hit 또는 bridge exclusion 전까지 scorecard 통과/실패로 세지 않는다. 배제되더라도 코어 클라루스장 자체가 아니라 입자언어 readout/coupling bridge가 내려간다. |
| 뇌 프로그래밍 언어 | `NL0`–`NL6` 가운데 실제 자료의 경계·상태·재사용·보지 않은 합성·호출·선택적 인과 개입·동물/세션 밖 복제가 필요하다. oracle-labelled synthetic method pass는 생물학적 통과로 세지 않는다. |
| Phenomenology 항목 전체 | `Exact` 승격에는 각 bridge readout의 독립 유도 또는 외부 독립 데이터 gate가 필요하다. |

$A_s$는 총 고정점 응답 $dx/dD$가 아니라 잔차 방정식의 국소 구동력 $\partial_D r=x(1-x)$를 반주기 위상과 유효 기하 깊이로 투영하는 readout으로 조건부 채택했다. $|V_{cb}|$는 LO QCD tunneling 실패를 유지하고, $Z_{cb}^{(1)}=1+\delta/(2\pi)$를 1-loop electroweak projector bridge로 채택했다. Clarus boson은 독립 입자를 먼저 가정한 것이 아니라 클라루스장 상관함수의 pole을 입자언어로 읽은 bridge이며, `examples/physics/clarus_boson_search_gate.py`로 그 pole/readout 조건을 사전등록했다. 따라서 현재 가장 강한 판정은 `부분 증명 + 열린 A1/Q0 작용 bridge + 조건부 현상론 + 사전등록된 미관측 Open test`이다.
