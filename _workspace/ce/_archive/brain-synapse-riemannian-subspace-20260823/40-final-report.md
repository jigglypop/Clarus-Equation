# 실제 시냅스 요인과 반응-풀백 기하의 엄격 검정

Status: COMPLETE

기계 상태: `STOP / DIAGNOSTIC_ONLY / RANK_UNIDENTIFIED / DEVELOPMENT_STOP_CONFIRMATION_UNTOUCHED / BIO_EVIDENCE_L1_DIAGNOSTIC`

<!-- PROVENANCE: 본문은 병행 세션(2026-08-23 01:51)이 완결 집필. 오케스트레이터가 2026-08-23 10:59에 누락된 Status 줄과 기계 상태 줄만 추가 (내용 무변경, 소유 세션 9시간 정체 후 인계 규약에 따름). -->

## 초록

이 보고서는 하나의 시냅스 연결을 단일 가중치로 압축하지 않고, Allen-SynPhys의 실제 paired patch-clamp 요약값에서 식별 가능한 네 요인과 네 반응 요약을 분리한 뒤, 그 반응 지도가 국소 리만 계량을 유도하는지 검정한다. 핵심 결과는 부정적이지만 유익하다. 조건부 SPD 정리는 살아남고 측정 스키마도 명확해졌으나, 실제 mouse V1 흥분성 및 억제성 층화 모두에서 안정적인 4차원 랭크와 예측 우위의 동결 게이트를 통과하지 못했다. 따라서 이 실행은 `STOP / DIAGNOSTIC_ONLY`이며, 확인 자료에는 접촉하지 않았다(`confirmation_contact=false`).

## 문제와 측정 대상

시냅스 생리를 기하학으로 묻는 일에서 첫 위험은 서로 다른 물리량을 하나의 $W_{ij}$로 부르는 일이다. 연결 존재, 전달물질 부호와 세포형, 첫 반응의 크기, 지연, 상승·감쇠 시간, 단기 가소성, 막 성질, 접촉 수와 형태, 방출·양자 과정, 수용체 과정, 장기 가소성 및 구조 변화는 서로 다른 측정 층에 속한다. 이 보고서의 기여는 이들을 한 식으로 환원하는 데 있지 않다. 실제 자료가 관측한 층과 관측하지 않은 층을 분리한 typed factor registry를 만들고, 그중 서로 다른 원천 표에서 안정적으로 결합되는 항만으로 엄격한 검정 문제를 구성한 데 있다.

입력은 source-locked strict chart

$$
z=\left(
\log\frac{|r_1|}{r_{\rm ref,\chi}},
\log\frac{L_{\rm soma}}{1\,{\rm m}},
\log\frac{R_{\rm in,post}}{1\,\Omega},
\log\frac{\tau_{m,\rm post}}{1\,{\rm s}}
\right)
$$

이다. 여기서 $r_1$은 휴지 상태 첫 PSP 크기, $L_{\rm soma}$는 쌍의 soma 간 거리, $R_{\rm in,post}$와 $\tau_{m,\rm post}$는 postsynaptic input resistance 및 membrane time constant이다. 로그 안의 모든 비는 무차원이다. 출력은 50 Hz 자극에서 pulse 2, pulse 6--8의 중앙값, 250 ms 회복 뒤 pulse 9--12의 중앙값, pulse 5--8의 noise-corrected log variability로 만든 네 스칼라 요약 $y$이다. 흥분성에서는 부호 $+1$, 억제성에서는 $-1$을 써서 전류클램프 PSP의 방향을 맞추고 각 진폭은 $r_{\rm ref,\chi}$로 나누었다. 이 분리는 첫 반응의 resting-state pipeline과 뒤 pulse의 dynamics pipeline이 같지 않다는 사실을 존중한다. latency·rise·decay는 유용한 반응 요약이지만 이 target과 같은 pulse-response pool을 공유하므로 별도의 shared-summary 진단으로만 남겼다.

자료는 Allen-SynPhys r2.1 small SQLite이며, SHA-256은 `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53`, SQLite integrity는 `ok`이다. 엄격 complete case는 979 pair와 512 slice group이었고, mouse V1의 주 분석은 흥분성 246 pair/160 slice와 억제성 343 pair/199 slice로 나뉘었다. 각 층화는 train/development/confirmation으로 slice 단위 분할되었으며 development는 각각 39 pair/27 slice 및 59 pair/37 slice였다. small tier에는 event-level pulse row가 없으므로 pulse 별 독립 결합은 검증하지 못했다. 이 한계는 `PIPELINE_SEPARATED / ROW_LEVEL_UNVERIFIED`로 남겼고, 결측을 보간하거나 event identity를 재구성하지 않았다.

## 조건부 기하 정리

훈련 자료에서만 이차 반응 지도 $y=\mathcal H_2(z)+\epsilon$, $\epsilon\sim\mathcal N(0,R_\chi)$를 적합하고, 야코비안 $J=\partial_z\mathcal H_2$를 계산했다. 잔차 공분산 $R_\chi$가 양의 정부호이고 $J$가 네 방향 모두를 보존할 때,

$$
g_{\rm resp}(z)=J(z)^{\mathsf T}R_\chi^{-1}J(z)
$$

는 strict chart의 리만 계량이다. 실제로 임의의 벡터 $v$에 대하여 $v^{\mathsf T}g_{\rm resp}v=(Jv)^{\mathsf T}R_\chi^{-1}(Jv)\ge0$이고, 영이 되는 방향이 없을 필요충분조건이 $\operatorname{rank}J=4$이기 때문이다. 이는 반응 공간의 Mahalanobis 계량을 입력 공간으로 당겨온 조건부 수학 정리이며, 생물학적 기전의 증거나 모든 시냅스에 대한 주장도 아니다.

비교의 공정성을 위해 train covariance로 정한 $g_{\rm ref}$와 그 whitened 표현을 기준으로 삼았고, 고정 계량은 선형 좌표변환 뒤의 Euclidean 모델과 동등하다는 점을 반영했다. 그러므로 가변 계량은 reference Euclidean, diagonal response metric, train-mean constant full response metric뿐 아니라 직접 이차 반응지도와도 비교해야 한다. held-out 점은 train graph에만 접속했고 test-test 인접성은 만들지 않았다. 좌표계 변화 검사는 노드와 인접성을 transport한 채 64개의 affine rechart에서 line element, generalized spectrum, prediction을 확인했다.

## 관측된 수와 실패한 게이트

다음 표는 development 자료의 물리 단위 요약이다. 각 칸은 중앙값 [제1사분위수, 제3사분위수]이며, ex는 흥분성, in은 억제성 층화다.

| 측정량 | ex (n=39) | in (n=59) |
|---|---:|---:|
| 휴지 PSP 절댓값 (mV) | 0.2649492741022319 [0.11910830401208829, 0.6563004515962682] | 0.38162392285451113 [0.21643211858070044, 0.8977459821309901] |
| soma 거리 (µm) | 83.20907900284583 [49.04150993114358, 124.93540755709802] | 85.91803263745214 [56.23651987208214, 121.05608724375307] |
| postsynaptic input resistance (MΩ) | 142.3056274652481 [63.50898742675781, 244.2941889166832] | 114.46765065193176 [87.7932570874691, 170.82812637090683] |
| postsynaptic membrane $\tau$ (ms) | 14.38060149119262 [6.973766384540802, 27.363410799677602] | 10.787885904125382 [7.08988427477853, 23.00630997548947] |
| STP initial, 50 Hz (mV) | 0.1487067464020879 [0.03215145738697685, 0.38394079032769707] | -0.3419798509755416 [-0.571322134590009, -0.10132667246573501] |
| STP induction, 50 Hz (mV) | 0.087306980266082 [0.03658638153776153, 0.26915117081603634] | -0.18473446370695096 [-0.29141454886019247, -0.08913338901157336] |
| STP recovery, 250 ms (mV) | 0.12579066872827538 [0.03758354125121441, 0.32066070858345685] | -0.17986903829265408 [-0.35378910591674, -0.09338074046232389] |
| STP induced-state variability | 0.7397869527735427 [-0.09071157729096216, 1.1766221384280184] | -0.35348605013185 [-0.7063874082533463, 0.3920616875668404] |

두 층화 모두 명목상 모든 support에서 full rank였고, 1,000 bootstrap에서도 nominal full-rank fraction은 1이었다. 그러나 이 사실은 안정적인 계량을 뜻하지 않는다. 가장 약한 고유방향까지 포함한 bootstrap support-minimum ratio의 2.5% 하한은 ex에서 $1.08724397257579\times10^{-10}$, in에서 $2.40460726237342\times10^{-11}$였으며, 미리 정한 $10^{-4}$ 문턱보다 각각 약 $9.2\times10^5$배와 $4.2\times10^6$배 작았다. 따라서 두 층화의 rank status는 모두 `RANK_UNIDENTIFIED`다. ex의 중위 ratio는 $0.0022325338855740417$, in의 중위 ratio는 $0.0004269163601862352$였지만, 국소 약방향의 불안정을 덮지 못한다.

좌표계 검사는 ex에서 PASS였다. generalized-spectrum 상대오차 최대값은 $4.5747214408807795\times10^{-10}$로 허용치 $10^{-8}$ 아래였다. in에서는 `orthogonal-83103`에서 $1.3451971937706788\times10^{-8}$가 되어 같은 허용치를 넘었으므로 FAIL이다. line-element 및 prediction 오차가 각각 $1.3768655889612\times10^{-14}$와 $2.33840724561674\times10^{-15}$로 작더라도, 동결 계약은 세 검사를 모두 요구한다. 작은 실패를 사후에 PASS로 바꾸지 않은 이유가 여기에 있다.

예측도 가변 반응 계량을 구제하지 못했다. ex의 최선 matched control은 direct quadratic였고, 가변 metric의 비교값은 $\Delta\mathrm{ELPD}=-18.485394967480673$, $2SE=86.94860714306063$, 양의 slice 비율 $0.7777777778$ (27 slice)였다. in의 최선 control은 metric_diagonal이었고, $\Delta\mathrm{ELPD}=-4.856882928028329$, $2SE=9.600172804408121$, 양의 slice 비율 $0.1891891892$ (37 slice)였다. 즉 어느 경우도 가변 기하가 직접 반응 회귀나 일치한 상수·대각 계량보다 미리 요구된 예측 이득을 보이지 않았다. rank 실패와 gauge 실패 및 예측 실패가 겹쳤으므로 confirmation target은 읽지 않았고, 상태는 `DEVELOPMENT_STOP_CONFIRMATION_UNTOUCHED`다.

## 감사가 한 일과 남은 주장 경계

이 결과는 방정식을 부주의하게 만들었음을 뜻하지 않는다. 사전 감사는 오히려 그러한 위험을 막는 반증 장치였다. 좌표변환 아래 불변이 아닌 비교량, 동일 pulse pool에서 input과 target을 겹쳐 쓰는 누수, 불명확한 랭크 판정, held-out graph의 재구성 문제를 outcome을 보기 전에 제거했다. 이후 독립 구현 감사가 bootstrap의 support 5% 분위수 대신 각 bootstrap draw의 support minimum을 써야 한다는 P1 문제를 발견했다. 통계량을

$$
T_b=\min_{z\in\mathcal S}
\frac{\lambda_{\min}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
{\lambda_{\max}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
$$

로 고친 것은 사후의 bootstrap-min correction이지만, 같은 seed에서 문턱을 더 엄격하게 만들기만 했고 outcome을 통과시키는 방향의 수정은 아니었다. 수정 뒤에도 STOP 판정과 confirmation 미접촉은 변하지 않았다.

이 실행이 보존하는 것은 관측 요인 정의, source-locked 4차원 측정모형, 그리고 명시한 조건 아래의 SPD 정리다. 반대로 관측하지 않은 양을 관측했다고 바꾸어 부를 수 없다. 휴지 PSP 크기는 conductance가 아니며 holding potential, reversal potential, access/series resistance와 파형 QC가 없는 한 수용체별 conductance로 환산할 수 없다. 접촉 수, 방출 부위 수, release probability 및 quantal size를 곱한 $Npq$도 pair별로 식별되지 않았다. AMPA/NMDA/GABA 수용체 조성, reversal potential, gating도 latent다. spike-to-onset latency는 별도 요약일 뿐 축삭 경로 길이도 아니고, 방향성 있는 지연은 대칭 리만 거리도 아니다.

마찬가지로 STDP나 다른 장기 가소성의 eligibility trace, neuromodulator, homeostatic state는 이 단면 자료의 fitting state가 아니다. 접촉의 생성·소멸, spine survival, 형태 변화와 장기 turnover도 같은 연결을 시간에 따라 추적한 자료가 없으므로 관측되지 않았다. soma 거리는 형태학적 contact count, PSD/ASI 또는 세부 형태를 대신하지 않으며, 다른 데이터셋의 형태 요약을 이 pair/event frame에 무단 결합할 수 없다. 그러므로 본 결과는 생물학적 기전의 확증도 아니고, 기억이나 일반 지능에 관한 승격도 아니다.

## Remaining Problems

다음 측정 계약은 먼저 event-level pulse identity와 waveform QC를 포함하여, 동일 pair/event frame에 clamp mode, holding/reversal potential, access resistance와 수용체·conductance 추정의 전제를 고정해야 한다. $N,p,q$와 receptor 항을 쓰려면 각 항의 직접 또는 식별 가능한 측정과 불확실성 모형이 필요하다. delay를 쓰려면 방향성 및 order를 별도 quasi-metric 계약에서 raw-delay와 order null에 견줘야 하며, 형태·contact와 longitudinal turnover를 쓰려면 동일 시냅스의 공동 identity와 반복 관측을 확보해야 한다. eligibility, STDP, homeostasis는 자극 이력과 장기 추적을 읽는 새 계약 없이는 추가할 수 없다.

마지막으로 고정 4차원은 뇌의 차원에 관한 선언이 아니라, 이 자료에서 식별 가능하고 누수를 피할 수 있도록 고른 엄격한 sieve였다. 사용자가 지시한 후속 경로는 BA-SRM2 functional quotient이며, 이는 고차원 또는 무한차원 표현을 quotient 수준에서 새 측정 계약으로 다루어야 한다. 본 보고서는 BA-SRM2의 결과를 제공하거나 예단하지 않는다.
