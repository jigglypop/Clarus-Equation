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
| 시간여행/CTC no-go | `Refuted/Conditional` | $N\ge0$, $|\det T|\le1$만의 no-go는 주기적 시간 반례로 기각; 전역 시간함수 A1 아래 no-CTC 정리는 증명 | A1은 현재 가정이며 완전한 CE+중력 작용에서의 유도는 `Open` |
| 완전역사 유한 readout | `Exact/Finite` | 모든 역사·시간표본의 이중 유한합으로 유일하게 정의; 실행 gate 통과 | 연속 경로공간 적분은 측도·적분가능성 필요 |
| 조건부 미래 가능성 shift | `Exact/Finite` | $q(u)=e^uq/(1-q+e^uq)$, $0<q<1$이면 $q'(u)>0$; log-domain 구현으로 빈 target, $u=1000$도 NaN 없이 prior 보존 | 물리적 제어 $u$와 CE 작용의 대응은 `Bridge/Open` |
| zero-support 역사 생성 | `Refuted` | 유한 양의 재가중은 support를 보존하므로 prior 질량 0인 과거·CTC를 생성하지 못함 | 후보공간 확장 또는 역사간 전이 kernel 필요 |
| 이미 벌어진 과거의 미래개입 불변성 | `Exact/Finite` | 정규화 kernel에서 $\sum_f p(h)K_u(f\mid h)=p(h)$; 서로 다른 미래 제어에 대해 실행 residual 0 | 연속 장론·양자중력 개입으로의 승격은 별도 |
| 미래 증거의 과거 역추론 | `Exact/Finite` | Bayes posterior는 과거 믿음을 바꾸지만 support를 확장하지 않음; zero-prior 과거는 계속 0 | posterior 변화는 물리적 과거 변경이 아님 |
| $d=0$ 선재 우주 | `Open` | $d(d-3)=0$의 근 $0,3$만 산술적으로 증명; 시간순서·전이진폭·초기조건은 나오지 않음 | 차원 동역학, 양자중력 측도, 관측 흔적 필요 |
| 공간접힘 shortcut 기하 | `Exact/Kinematic` | $L\gg\ell$, $0<\beta<1$에서 $v_{\rm app}/c=\beta L/\ell\gg1$, 국소 고유시간 양수; 실행 gate 통과 | 실제 입구·목 생성은 별도 물리 문제 |
| 정규화 Alcubierre wall | `Numerical/Negative control` | smooth profile, $u=(r-R)/\Delta$ 적분, Eulerian WEC 위반과 $E\propto-v^2$ 계산; 10m/1m, $v=c$에서 $-3.3727\times10^{44}$ J | 직접 null projection, material action, backreaction, 전체 안정성 |
| 초광속 warp 축 지평선 | `Finite/Kinematic` | $f(r_h)=1-1/\beta$를 직접 풀어 $\beta=2$에서 $r_h=10.000000004$m; $\beta\le1$에는 root 없음 | 가속·형성·정지 과정과 horizon 제어 |
| CE 통과 가능 웜홀 물질원 | `Open/Negative control` | 1 m control throat NEC 요구량은 정적 CE 셀보다 $2.85\times10^{16}$배 큼; 기존 SI 변환 오류 정정 | renormalized $T_{\mu\nu}^{\rm CE,res}$, 보존, quantum inequality, backreaction, 안정성 |
| 입력 없는 0차원 관찰자의 위치 식별 | `Refuted` | 0차원 단일 상태의 내부 위치 정보는 0 bit; $N>1$개 후보의 식별에는 외부에서 최소 $\log_2N$ bit가 필요 | 3차원 또는 경로공간 readout이 필요 |
| 전체 이력 readout의 0차원 목표 선택 | `Exact/Finite` | $J(x)=\sum_{h,t}p(h)w(t)C(h,t,x)$ 뒤 $\arg\min_xJ(x)$ 또는 Gibbs 분포로 목표 선택; 실행 gate 통과 | 위치 정보는 0차원에 내재하지 않고 경계 입력으로 수입됨 |
| 모든 시간을 포함한 목표 선택의 자기일관 해가 항상 유일 | `Refuted` | $x_*\in\arg\min_xJ(x;\gamma_{x_*})$가 필요하며 유한 반례에서 고정점은 0개·여러 개·1개 모두 가능 | 실제 계의 존재·유일성 조건을 별도로 증명해야 함 |
| 0차원 목표 라벨의 원격 공간접힘 작동 | `Open` | 목표 선택과 목적지 응력 텐서·두 번째 입구 생성은 서로 다른 단계 | 국소 작동 사상과 인과적 정보 경로가 필요 |
| broadcast 응력장의 목적지 국소화 | `Refuted` | 동일한 열을 갖는 rank-1 응답은 목표 명령에 따라 위치를 구별하지 못함 | 위치별 독립 actuator가 필요 |
| 적응적 원격 목표 명령의 즉시 전달 | `Refuted/Local causality` | $N$개 목표에는 $\log_2N$ bit가 필요하고 거리 $L$의 전달시간은 $t\ge L/c$; 1광년은 최소 1년 | 비국소 작용을 추가하면 no-signalling·보존 재증명 필요 |
| CE 셀의 목 반지름 밀도–coherence 창 | `Open/Negative control` | $b'=-1$ 제어모형에서 $r_{\min}=1.6883\times10^8$ m, $r_{\max}=6.65\times10^{-15}$ m로 교집합 없음 | 장거리 coherent 다중셀 $T_{\mu\nu}$ 유도 필요 |
| 사전 설치 입구망의 최소시간 경로 | `Exact/Finite` | 양의 유한 edge 통과시간에 대해 최단경로 계산; 1광년 외부거리·10m 목·$0.1c$에서 $3.3356\times10^{-7}$ s | 입구 제작·유지 물리는 `Open` |
| 유한 입구망의 임의 연속좌표 도착 | `Refuted` | 도착점은 설치 mouth의 유한 집합이며 허용오차 밖의 좌표는 덮지 못함 | 연속 배치 또는 이동 가능한 입구 필요 |
| 입구망 좌표시간 negative-cycle gate | `Exact/Finite` | directed cycle의 총 $\Delta t<0$을 Bellman--Ford 반례로 검출 | 통과는 연속 시공간의 전역 시간함수 증명이 아님 |
| mouth clock-offset 동기화 | `Exact/Finite` | $w_{ij}+s_j-s_i\ge\epsilon$을 difference constraints로 풀고 실행 gate 통과 | drift·jitter보다 큰 $\epsilon$을 운용 입력으로 정해야 함 |
| 음의 시간 cycle의 clock relabel 제거 | `Refuted` | cycle에서 offset이 telescoping되므로 $\sum_Cw'_{ij}=\sum_Cw_{ij}$ | 실패 cycle의 edge 차단 필요 |
| 엄격한 graph 시간함수 | `Exact/Finite` | 모든 cycle total 양수 또는 DAG이면 strict 해 존재; margin 0의 임의 witness 거짓 음성을 최소 cycle total과 양의 재해결로 교정 | 연속 시공간의 전역 시간함수는 별도 `Open` |
| 실시간 chronology interlock | `Exact/Finite` | $\underline w=\hat w-u-r\Delta t$에서 edge를 순차 enable하고 각 단계 difference-constraint 재검사; 위험 cycle edge 차단 | greedy 정책은 최소 차단·최대 처리량을 보장하지 않음 |
| mouth 센서 fault 처리 | `Exact/Code` | NaN 포함 프레임에서 모든 edge를 disable하는 fail-closed gate 통과 | 독립 하드웨어 차단 latency 검증 필요 |
| 동적 graph interlock의 물리적 chronology 보호 | `Open/Conditional` | 입력된 uncertainty·drift bound 안에서는 enabled graph 안전 | bound의 실험적 보장과 연속 시공간 causal curve 검증 필요 |
| CE resonance-$Q$ 웜홀 응력 후보 | `Open/Ansatz` | $T_{kk}\propto Q^p$, $\xi\propto Q$ 가정 시 1m 목은 $p=1$에서 $Q=2.85\times10^{16}$, $p=2$에서 coherence 병목 $Q=1.50\times10^{14}$ | $p$, 부호, 포화와 renormalized $T_{\mu\nu}$를 CE 작용에서 유도해야 함 |
| CE 최소결합 canonical scalar의 고전 목 지지 | `Refuted` | null contraction $T_{kk}=(k\cdot\partial\sigma)^2\ge0$ | 비최소 결합·양자상태는 별도 후보 |
| phantom scalar 목 지지 | `Refuted as stable source` | kinetic sign 반전은 $T_{kk}<0$이나 최소 ghost-free gate 실패 | ghost-free UV completion 없이는 채택 불가 |
| 외부 양자장 웜홀 대조군의 CE 승격 | `Open` | GJW와 MMP는 각자의 장·경계조건에서 구조를 닫지만 CE 유도가 아님 | CE field content, state, boundary coupling과 backreaction mapping 필요 |
| $\xi(Q)$에서 웜홀 stress exponent $p$ 식별 | `Refuted` | 같은 $\xi(Q)=Q\xi_0$에 대해 $Z(Q)=Q^{d+p}Z_0$로 임의의 프록시 exponent $p$를 구성 | pole residue·spectral density·renormalization 필요 |
| CE resonance의 물리적 pole과 residue | `Open` | 문서가 pole 형식을 ansatz로 제시하나 Q0 gate는 spectral density와 LSZ를 미유도 판정 | $\Gamma_Q^{(2)}$, positivity, pole/cut, $Z_Q$ 계산 필요 |
| CE 공간접힘 가능성 단계 | `W1/Kinematic` | W0 선택·제어와 W1 주어진 기하 shortcut은 통과; W2 renormalized 음의 $T_{\mu\nu}$ 미도달 | W2 stress source가 전체 사슬의 현재 병목 |
| CE $\xi R\Phi^2$ 비최소장의 국소 null source | `Conditional pass/W2a` | 선택한 Jordan-frame convention에서 $N_{kk}=(\Phi')^2-\xi(\Phi^2)''$; $\xi=0.49$와 $F=1-\xi\Phi^2>0$인 음의 국소 반례 실행 | CE Q0 부호·정규화, EOM과 전체 perturbation 안정성 필요 |
| 국소화 비최소 CE profile의 단순 ANEC 우회 | `Refuted in boundary-zero control` | endpoint jump 0이면 $\int N_{kk}=\int(\Phi')^2\ge0$ | Casimir 경계·곡률 진공상태·가변 $F$의 완전 계산 필요 |
| 이상적 Casimir plate의 1m 목 scale | `Negative control` | normal null stress 역산에서 $a=3.66\times10^{-18}$ m, CE $\xi$의 $5.5\times10^{-4}$ | 실제 구면 경계, 물질, 전체 보존 stress와 backreaction 미해결 |
| 비최소 $1/F$만의 CE 밀도격차 증폭 | `Singular-limit control` | $2.85\times10^{16}$배에는 $F=3.51\times10^{-17}$, 중력 pole 상대거리 $1.75\times10^{-17}$ 필요 | regular effective gravity/strong-coupling gate 실패 가능성 큼 |
| 이상적 Casimir의 zero-redshift throat tensor matching | `Negative control` | $\rho,p_r$ matching은 $b'=-1/3$이나 tangential pressure residual $\Delta p_t=C/3$ | 비최소 scalar 또는 redshift gradient의 보조 stress 필요 |
| 6.65fm CE massive vacuum polarization의 1m backreaction | `Refuted as order-one macroscopic source` | large-mass 차원 estimate $\rho_{vac}/\rho_{req}=2.90\times10^{-97}$, 필요 multiplicity $3.44\times10^{96}$ | massless/collective sector가 없다면 단독 후보 탈락 |
| CE-native 거시적 음의 source 최상위 후보 | `Frontier A/Open` | 일반-redshift Casimir 국소 series와 비최소 CE+Casimir completion이 생존 | CE boundary $T_{\mu\nu}$, 전역 ODE와 perturbation 안정성 필요 |
| 이상적 Casimir 일반-redshift 국소 throat series | `Exact/Finite control` | $b'_0=-1/3$, $r_0\varphi'_0=-1/2$에서 $\rho,p_r,p_t$ 전부 matching; $r_0a'_0/a_0=1/2$에서 보존식 matching | 전역 ODE, CE renormalized boundary stress와 안정성 필요 |
| CE 공간접힘 source frontier | `W2b-local/Control` | 일반-redshift Casimir control은 국소 전체 tensor·보존 통과 | CE 물리 전체는 boundary realization 미유도로 W2 미완성 |
| 고정 Casimir EoS의 전역 공명-envelope 연장 | `Refuted` | 유한 redshift는 tail $n=8/3$, 유한 energy/ADM mass는 $n>3$을 요구해 양립 불가 | 공간적으로 변하는 anisotropy 또는 junction/completion 필요 |
| 1m 목 Casimir 특정 파장 | `Finite control` | $a=3.66\times10^{-18}$m에서 $\lambda=2a=7.33\times10^{-18}$m, $hf\simeq169$GeV | 실제 CE boundary mode와 negative stress 유도 필요 |
| CE 29.65MeV light pole의 high-$Q$ 해결 | `Refuted as linear resonance` | 요구 carrier는 약 5708배 높고 $Q$는 선형계 carrier frequency를 바꾸지 않음 | 고차 harmonic/독립 mode와 multi-mode pressure engineering은 `Open` |

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

## 다중 모드 전역 throat 추가 판정 (2026-08-04)

| 항목 | 현재 판정 | 검증 결과 | 남은 bridge |
|---|---|---|---|
| 가변 비등방 전역 throat target | `Exact/Finite control` | 목의 정확한 Casimir tensor, 유한 lapse, 응력 보존, 점근 평탄성, 유한 ADM 질량 $M/r_0=1/3$ | 실제 물질장 구현 |
| 유한 공간 모드 표현 | `Numerical control` | $1\le r/r_0\le10$에서 32모드 최대 정규화 오차 $1.66\times10^{-7}$ | 구현 가능한 공진기 spectrum과의 동일시 |
| CE 공명 음의 응력 | `Open` | 필요한 target tensor는 특정됨 | 재규격화된 CE $\langle T_{\mu\nu}\rangle$, carrier-envelope 결합, 양자부등식, 안정성 |

## 다중 공명 현실화 추가 판정 (2026-08-04)

| 항목 | 현재 판정 | 정량 결과 |
|---|---|---|
| 1 m exact-target Casimir boundary | `Deferred/Physical fail` | $a=4.05$ am, $hf=152.93$ GeV; 물리적 반사경 모델 없음 |
| 양쪽 areal-coordinate/Misner--Sharp energy drop | `Exact for target` | proper energy가 아님; $4.03\times10^{43}$ J, 질량환산 지구 75.2개 |
| 양쪽 static-slice proper matter energy | `Numerical control` | $I=0.631466179317807$, $7.642\times10^{43}$ J, 질량환산 지구 142.38개 |
| 핵 크기 경계 해상도 trade-off | `Negative control` | $r_0=42.9$ km가 필요하고 coordinate/proper 질량환산은 각각 태양 9.69/18.36개 |
| 다중모드에 의한 총량 감소 | `Refuted for fixed geometry` | 모드 분할은 같은 Einstein tensor의 적분 에너지를 바꾸지 않음 |
| 동적 squeezed-state 정적 source | `Refuted/Control` | 1 m 광통과시간/QI 지속시간 규모비 $2.70\times10^{17}$ |
| CE 비물질 경계 또는 수정중력 | `Open/Decisive` | 현재 shortcut 현실화에 필요한 새 물리 bridge |

## 현실화 경로 funnel 추가 판정 (2026-08-04)

10개 경로를 작용·음의 응력·backreaction·shortcut·안정성·공학 bridge로 감사했다.
현재 6/6 통과 경로는 없다. CE 내부에서는 기존 \(\xi R\Phi^2\) 비최소결합 scalar의
전역 inverse reconstruction과 quadratic perturbation spectrum이 다음 결정적 gate다.
beyond-Horndeski는 더 많은 구조 gate를 만족하지만 현재 CE 작용 밖이며 안정성 검사가
불완전하다.

## 비최소결합 scalar 역산 추가 판정 (2026-08-04)

기존 지수형 전역 target은 목에서 건강한 scalar 총 kinetic을
\(-17F/12\)로 요구하므로, \(F>0\)인 단일·다중 canonical scalar source로는 반증됐다.
다만 Casimir 목의 0차·1차 자료를 유지한 2차 co-design은
\(3b''_0+8r_0^2\Phi''_0+12\le0\)에서 양의 kinetic을 허용한다.
\(b''_0=-5, r_0^2\Phi''_0=0\)은 `Local conditional pass`이며 전역 연장은 `Open`이다.

후속 전역 polynomial-exponential co-design 탐색은 국소 양의 kinetic 후보에서도
\(r\simeq1.62r_0\)에 음의 kinetic 영역을 남겼다. 양수·유한 \(F\)와 ghost-free field-space
metric을 유지하는 scalar-tensor 전역 no-go와 합쳐, standalone CE \(\xi R\Phi^2\)
단일·다중 scalar source는 `Refuted globally`로 갱신한다.

## Beyond-Horndeski 현실화 추가 판정 (2026-08-04)

Beyond-Horndeski가 기존 no-go를 회피하고 명시적 점근평탄 배경을 갖는 control model은
존재한다. 그러나 공변 작용·완전 안정성·GR weak-field asymptotics·견고한 \(c_T=1\)·공학
bridge가 한 모델에서 닫힌 사례는 확인되지 않았다. 또한 CE의 단독
\(\alpha_2\|\nabla^2\phi\|^2\)는 최고미분 Hessian이 \(2\alpha_2\ne0\)인 비퇴화 항이므로
DHOST completion으로 읽을 수 없다. 판정은 `External theory frontier / reality open`이다.

## Charged-fermion 위상 경계 추가 판정 (2026-08-04)

MMP 외부 control은 magnetic flux의 charged massless fermion zero mode로 음의 Casimir
backreaction을 명시적으로 만들지만 long wormhole이라 ambient shortcut이 아니다. 1 m에서
유효 무질량 한계는 \(1.97\times10^{-7}\) eV이고 전자는 이보다 \(2.59\times10^{12}\)배
무겁다. 현재 CE에는 이보다 가벼운 charged fermion이나 quantized magnetic flux action이
없으므로 사람 크기 CE mapping은 `Deferred/field content absent`다.

## 순수 위상 경계 응력 추가 판정 (2026-08-04)

상수계수 4D Euler/Gauss–Bonnet, gravitational Pontryagin, gauge
\(\theta F\wedge F\), 3D boundary Chern–Simons를 metric variation gate로 감사했다.
이들은 순수 위상항으로서 국소 bulk \(T_{\mu\nu}\)를 공급하지 않으므로 standalone throat
source는 `Refuted`다. 위상은 flux와 edge sector를 선택할 수 있지만, 실제 음의 응력에는
별도의 동역학적 edge QFT가 필요하다. 현재 CE edge action은 `Not specified`다.

## 박막/결함 QFT 추가 판정 (2026-08-04)

대칭 Schwarzschild cut-and-paste 목의 Israel 접합조건을 SI 단위로 구현했다.
1 m 평탄 외부 껍질은 `-1.926e43 J/m^2`, 총량으로 음의 450.95 지구질량을
요구한다. 지평선 근접 극한에서는 음의 에너지가 `sqrt(f)`로 줄지만 횡압이
`1/sqrt(f)`로 발산하므로 비용 전체를 제거하지 못한다. 또한 등방 2+1차원
scale-free edge QFT는 `p=sigma/2<0`인 반면 접합조건은 모든 허용 `f`에서
`sigma<0`, `p>0`를 요구한다. 따라서 conformal/Casimir edge 단독 후보는
`Refuted by EoS sign`이다. massive·비등방·상호작용 defect QFT 전체는
`Open`, CE의 명시적 defect action은 `Absent`다.

후속 방사형 안정성 계산은
`a^2 V''=2 eta(1-3f)-(1+3f^2)/(2f)`를 준다. 전 Schwarzschild 허용
구간에서 `0<=eta=c_s^2<=1`과 `V''>0`은 교차하지 않는다. 따라서 인과적
gradient-stable barotropic defect fluid completion은 `Refuted`이며,
비barotropic·비등방 elastic defect만 다음 후보로 남는다.

최소 등방 elastic membrane의 전단응답은 구면 `l=0`에서 정확히 사라져
radial no-overlap을 고치지 못한다. `p=-sigma`인 negative-tension
Nambu--Goto 막은 `f=1/3`에서 배경 응력은 맞지만 `a^2V''=-2`이고 transverse
bending kinetic의 부호가 뒤집혀 ghost다. 두 최소 elastic completion은
`Refuted`; 내부 자유도/mode mixing/nonlocality를 가진 비최소 defect만 `Open`이다.

smooth free-field QEI control을 1 m 박막에 적용하면 한 species의 음의 층은
`d<=2.50e-24 m`, sampling time `8.35e-33 s`, UV energy `>=7.89e16 eV`가
필요하다. 이는 boundary Casimir 자체에 대한 엄밀 no-go가 아니라 규모 gate다.
경계 loophole에는 반사체/결함의 전체 stress와 backreaction이 필요하며 CE에는
그 미시 completion이 없다.

2022 beyond-Horndeski 재감사에서 한 covariant 모델이 odd/even parity의
고에너지 ghost 및 radial/angular gradient gate를 모두 통과함을 반영했다.
따라서 이 경로는 `high-energy stable existence control`까지 승격한다. 다만
slow tachyon, 먼 거리 GR 복귀, physical-frame luminal GW, CE 유도, throat
scale/strong-coupling bridge는 같은 모델에서 열려 있어 전체 현실성은 `Open`이다.

후속 `arXiv:2404.06297v2`는 tachyon과 superluminal mode까지 포함한 일반
정적 구면대칭 안정성 조건을 유도했다. 하지만 2022 explicit wormhole에 이를
적용한 통과 결과는 없고, 그 예시의 최종 Lagrangian coefficient functions도
해석식으로 제공되지 않아 slow-spectrum 독립 재현은 현재 불가능하다. 판정은
`complete criteria derived / passing explicit wormhole still open`이다.

thin-shell 내부 공명은 안정한 passive mode를 적분소거하면
`K_eff=K_rr-B C^-1 B^T`가 되어 radial tachyon을 개선하지 못한다. 따라서
수동 다중공명 mixing 단독 안정화는 `Refuted`; 별도 직접 radial stiffness나
driven/Floquet feedback만 `Open`이다.

Floquet radial control은 exact monodromy 적분에서
`Gamma/Omega=0.05, epsilon=0.1`이 `|tr M|<2`, `det M=1`을 만족해
`Finite control`로 승격했다. 다만 static negative stress를 공급하지 않고
drive loss 후 tachyon이 복귀하므로 물리 source/realization 판정은 변하지 않는다.

Floquet-to-Israel 역산은 1 m, `f=1/3` control에 `0.954 GHz`, pressure
stiffness `6.68e44 N/m^2`, peak reactive power bound `5.03e43 W`, drive-loss
e-fold `3.34 ns`를 요구한다. 명시적 stress actuator가 없어 engineering gate는
`Fail/Absent`이고 negative-energy source 문제도 그대로다.

negative-tension brane에 local `alpha K^2` rigidity를 추가한 propagator는
`1/(Tz+alpha z^2)=(1/T)/z-(1/T)/(z+T/alpha)`다. `T<0`에서 IR bending
pole의 residue가 계속 음수이므로 minimal rigid-brane completion은 `Refuted`다.

induced-gravity defect는 ghost-free coupling 영역 자체는 존재하지만 CE의
localized EH coefficient, modified junction equation, bulk boundary condition,
KK/brane-bending spectrum이 없다. `External open frontier / CE mapping absent`로
판정하며 thin-shell 경로 안에서만 집계한다.
