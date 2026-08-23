# 12-routes — BA-SRM1 대안식과 선택 경로

Status: COMPLETE

Claim mapping: R0/R1/R3--R5는 `BA-SRM1-C5`, 선택 R2는
`BA-SRM1-C2`--`BA-SRM1-C4`를 시험한다.

## R0 — 단일 연결세기 $W_{ij}$

식:

$$
h_i(t)=\sum_j W_{ij}a_j(t-d_{ij}).
$$

판정: `REJECTED_AS_MEASUREMENT_COLLAPSE`.

이 식은 회로 계산의 축약형으로는 쓸 수 있지만 Allen paired recording의 response
amplitude, delay, kinetics, STP, membrane integration을 한 숫자로 합친다. PSP
amplitude를 conductance·release probability·contact 수와 동일시하게 되므로 이번
실제 자료의 measurement model로는 선택하지 않는다.

## R1 — shared-response 4D chart

식:

$$
z_{\rm shared}=
\left(
\log|r_1|/r_{\rm ref},
\log d/t_{\rm ref},
\log\tau_r/t_{\rm ref},
\log\tau_d/t_{\rm ref}
\right).
$$

판정: `DIAGNOSTIC_ONLY`.

장점은 synaptic response shape 자체를 가장 직접적으로 표현한다는 점이다. 그러나
latency·rise·decay가 late-pulse target과 같은 pulse-response pool을 사용한다.
held-out slice split만으로 이 within-pair measurement overlap을 제거할 수 없다.
따라서 수치 분포와 상관 진단은 가능하지만 passive-pullback의 주 경로가 아니다.

추가 자유도: 없음. 주 경로보다 우세해도 독립 기하 증거로 승격하지 않는다.

## R2 — strict factor-to-response pullback (선택)

입력:

$$
z=\left(
\log\frac{|r_1|}{r_{\rm ref,\chi}},
\log\frac{L_{\rm soma}}{1\,\mathrm m},
\log\frac{R_{\rm in,post}}{1\,\Omega},
\log\frac{\tau_{m,\rm post}}{1\,\mathrm s}
\right).
$$

target:

$$
y=\left(
s_\chi a_2/r_{\rm ref,\chi},
s_\chi a_{6:8}/r_{\rm ref,\chi},
s_\chi a_{9:12}^{250\rm ms}/r_{\rm ref,\chi},
v_{5:8}
\right).
$$

$a_2$, $a_{6:8}$, $a_{9:12}^{250\rm ms}$와 $v_{5:8}$는 pulse 구간별
scalar summary 한 개씩이므로 $y\in\mathbb R^4$다.

response map과 metric:

$$
y=\mathcal H_2(z)+\epsilon,\qquad
\epsilon\sim\mathcal N(0,R_\chi),
$$

$$
g_{\rm resp}(z)=J(z)^TR_\chi^{-1}J(z),qquad
J=\partial_z\mathcal H_2.
$$

선택 이유:

1. 네 입력은 실제 DB의 서로 구분된 synapse/pair/post-cell 측정이다.
2. 모든 입력과 출력이 기준 단위로 무차원화된다.
3. target 네 개라 4D full rank의 필요조건은 충족한다.
4. actual strict complete support가 mouse V1 ex 246 pair/160 slice, in 343
   pair/199 slice로 계약 최소치를 넘는다.
5. 휴지막 진폭과 late-pulse 위치는 source pipeline에서 분리된다.

제한:

- soma distance는 axon path/delay가 아니다.
- membrane properties는 post-cell covariate이지 pair-specific strength가 아니다.
- 4D full rank는 데이터가 보장하지 않으며 Gate O2에서 기각될 수 있다.
- small DB에는 raw event row가 없어 pulse 분리는 pipeline-level receipt다.

추가 자유도는 quadratic ridge alpha 9개, graph $k$ 3개, bandwidth 5개이며
전부 train inner-fold에서만 선택한다. direct quadratic, reference Euclidean,
diagonal response metric, constant full response metric과 같은 budget으로 비교한다.

## R3 — directed latency quasi-metric

식:

$$
D(i\to j)=\min_{P:i\to j}\sum_{(a,b)\in P}d_{ab}.
$$

판정: `SEPARATE_BA-CG1 / NOT_OPENED_HERE`.

$d_{ij}\ne d_{ji}$일 수 있어 이는 Riemannian distance가 아니다. raw delay matrix,
order shuffle, sample null보다 추가 예측을 주어야 하는 BA-CG1 falsifier를 그대로
보존한다. BA-SRM1의 soma-distance 좌표를 이 식의 edge delay로 바꾸지 않는다.

## R4 — full/NWB waveform conductance route

후속 식:

$$
I_{\rm syn}(t)=\sum_r\bar g_r s_r(t)(V(t)-E_r).
$$

판정: `DEFERRED_NEW_CONTRACT`.

medium DB는 event-level fit summaries 약 11 GB, full DB는 short waveforms 약
268 GB이며 NWB는 완전 원파형을 준다. holding potential, reversal potential,
series/access resistance와 waveform QC를 잠그지 않고 $\bar g$를 역산하지 않는다.

## R5 — morphology/contact bridge

후속 상태:

$$
q_{\rm morph}=(N_{\rm contact},\mathrm{PSD/ASI},\mathrm{spine\ survival},\ldots).
$$

판정: `DEFERRED_JOINT_FRAME_REQUIRED`.

MICrONS/de Vivo는 형태 자료를 보완하지만 현재 Allen pair와 같은 synapse/event
identity frame이 아니다. ASI를 $W$, conductance 또는 $Npq$로 대체하지 않는다.

## 최종 경로

R2 하나만 실제 L1/L2 검증 후보로 선택한다. R1은 leakage adverse diagnostic,
R3은 directed-order 별도 가설, R4/R5는 새 원자료·measurement contract가 필요한
후속이다. R2가 rank·gauge·held-out control gate를 통과하지 못하면 다른 route로
갈아타지 않고 BA-SRM1을 STOP 또는 diagnostic으로 종료한다.
