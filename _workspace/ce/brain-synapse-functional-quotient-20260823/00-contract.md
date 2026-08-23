# 00-contract — BA-SRM2 시냅스 함수공간과 관측가능 몫기하

Status: COMPLETE

Run state: `ACQUIRING_MEDIUM_INPUT / PREACCESS_PREREG_FROZEN / SMALL_PULSE_GRID_REJECTED / CONFIRMATION_UNTOUCHED`

Candidate: `BA-SRM2`

Evidence target after unblock: `L2 functional statistics → conditional L3 held-out prediction`

## 1. 목표와 predecessor

BA-SRM2는 실제 시냅스 상태의 차원을 4로 고정하지 않는다. BA-SRM1의 4차원
chart는 small DB에서 누수 없이 식별한 strict sieve였고, 실제 뇌의 본질적 차원에
관한 주장이 아니었다. BA-SRM1은 rank와 예측 gate에서 STOP됐으며 그 결과와
confirmation 봉인을 그대로 보존한다.

이번 후보는 자극 이력에 따른 시냅스 반응을 함수로 보고, 데이터가 식별하는
관측가능 몫공간의 차원만 train에서 선택한다. 그러나 local small DB의 겉보기
12-pulse field가 producer alias bug로 무너졌으므로 결과값 접근과 구현을 시작하지
않는다.

## 2. Claim map

| Claim ID | 지위 | 내용 |
|---|---|---|
| `BA-SRM2-C1` | [정의] | 시냅스 상태를 causal-history/protocol response 함수공간으로 정의한다. |
| `BA-SRM2-C2` | [정리] | 유한 관측 pullback은 무한차원 전체에서 퇴화하며 관측가능 몫공간에서만 내적이 된다. |
| `BA-SRM2-C3` | [산출: source no-go] | small DB의 `pulse_amplitudes[1..12]`는 같은 aggregate의 반복이라 pulse-resolved 좌표가 아니다. |
| `BA-SRM2-C4` | [미완성/예측 후보] | medium event rows에서 train-only sieve 차원과 quotient rank를 선택하는 시험은 입력 잠금 뒤 새 계약이 필요하다. |
| `BA-SRM2-C5` | [미완성] | waveform, conductance, release, STDP, homeostasis와 morphology bridge는 식별되지 않는다. |

## 3. BIO_STARTING_MECHANISM

시냅스 반응은 단일 상수 $W_{ij}$가 아니라 receptor gating, 막 적분과 spike
history에 의존한다. conductance 수준의 기준식은

$$
C_i\dot V_i
=-g_{L,i}(V_i-E_{L,i})
-\sum_{j,r}A_{ij}\bar g_{ij}^{(r)}s_{ij}^{(r)}(t)(V_i-E_r)
+I_i^{\rm ext}(t)+\eta_i(t)
$$

이다. short-term plasticity의 한 기준 모형은

$$
\tau_F\dot u=U-u,\qquad \tau_D\dot x=1-x,
$$

$$
u^+=u^-+U(1-u^-),\qquad a_k=u^+x^-,\qquad
x^+=x^-(1-u^+)
$$

이다. 이 식은 생물학적 출발 구조이며 Allen summary에서 $\bar g,U,x,\tau_D,
\tau_F$를 직접 측정했다는 뜻이 아니다.

## 4. 함수공간 가설

정규화된 causal history의 이상적 상태공간을

$$
\mathcal H
=\mathbb R^p\oplus
L^2([-T,0],\mathbb R^r;w(s)\,ds)
$$

로 정의한다. $h_t(s)$에는 $s\le0$인 과거 전압, 전류, spike/event channel과
source-locked cell/protocol covariate만 들어간다. 미래 response와 target pulse는
입력에 넣지 않는다.

$M:\mathcal H\to\mathcal Y$가 Fréchet 미분 가능하고 $DM_x$가 bounded라고
가정한다. 관측 noise covariance $C$는 양의 정부호이며 functional output에서는
$DM_xu$가 $C^{-1/2}$의 정의역에 있어야 한다. 그러면

$$
G_x(u,v)
=\left\langle C^{-1/2}DM_xu,C^{-1/2}DM_xv\right\rangle_{\mathcal Y}
$$

를 response pullback으로 정의한다. 이는 반응 분포의 대칭적 국소
구별가능성이며 directed temporal propagation이나 delay distance가 아니다.

## 5. 유한 관측 no-go와 몫공간

실제 관측연산자 $O_m:\mathcal Y\to\mathbb R^m$에 대해

$$
J_{m,x}=D(O_mM)_x,\qquad
G_x^{(m)}=J_{m,x}^TR_m^{-1}J_{m,x}
$$

이고

$$
\operatorname{rank}G_x^{(m)}\le m.
$$

$\mathcal H$가 무한차원이고 $m<\infty$이면 전체 $\mathcal H$에서 data-identified
SPD metric은 불가능하다. 점 $x$에서 식별되는 대상은

$$
T_x\mathcal H/\ker J_{m,x}
$$

인 관측가능 몫공간이다. 이를 manifold bundle로 승격하려면 평가 domain에서 rank가
국소적으로 일정하고 kernel distribution이 닫힌 smooth subbundle이어야 한다.
pointwise rank 하나만으로 전역 quotient manifold를 주장하지 않는다.

$\lambda I$로 full rank를 만드는 것은 analyst prior이지 관측 식별 증거가 아니므로
금지한다.

## 6. SMALL_DB_SOURCE_NO_GO

공식 commit `545a990ee171e6c0d23dd4bba413e1ccbf2f0853`의
`aisynphys/dynamics.py`는

```python
collect_pulse_amps = [[]] * 12
```

로 pulse 저장용 list를 만든다. Python list multiplication 때문에 12개 slot은
하나의 inner list를 공유한다. pulse 1--12의 값은 모두 그 하나의 list에 append되고,
저장 단계에서 동일한 aggregate `(median,std,n)`가 12번 반복된다.

train-only 구조 감사에서 ex 1,324개와 in 2,613개 protocol record 모두 NaN 동치까지
고려하면 12 slot이 동일했다. 따라서 이 필드는 12차원 pulse trajectory가 아니라
rank 1 이하의 중복 summary다. 이를 12D, 36D 또는 48D response field로 펼치는
경로는 source-level 반례로 기각한다. train bucket의 JSON은 12 slot 구조 동일성을
확인할 때만 복호화했고 amplitude 크기를 보고ㆍ적합ㆍ채점하지 않았다. confirmation
blob과 값은 열지 않았다.

독립적으로 계산된 `stp_initial`, `stp_induction`, `stp_recovery`,
`stp_recovery_single`은 protocol summary로 사용할 수 있지만, raw causal history나
pulse-resolved 함수공간을 식별하지 않는다. 이들을 보고 차원을 사후 구성하는
대체 실행은 이번 계약에서 금지한다.

## 7. DATA_PROVENANCE와 missing prerequisite

검증된 local small DB는 다음과 같다.

- file: `synphys_r2.1_small.sqlite`, schema 22;
- bytes: 176,771,072;
- SHA-256: `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53`;
- `pulse_response=0`, `pulse_response_fit=0`, `stim_pulse=0`.

필요한 공식 medium object는

```text
https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_medium.sqlite
```

이며 HEAD receipt는 다음을 준다.

- Content-Length: 11,125,997,568 bytes (10.36 GiB);
- Last-Modified: 2023-01-26 02:25:26 GMT;
- multipart ETag: `d954cbad0d7c7b0002bf3a2879e40e90-1327`.

medium 파일은 아직 local에 없고 SHA-256, SQLite integrity, table counts와 exact
event join support가 `UNVERIFIED`다. 따라서 Gate B를 통과하지 못한다.

## 8. successor contract minimum

medium 다운로드는 사용자가 승인했다. outcome 접근 전 후속 시험은
`revisions/01-medium-event-preaccess-prereg.md`에 고정했으며 receipt가 잠길 때까지
분석은 계속 BLOCKED다. 그 revision은 다음을 고정한다.

1. `StimPulse.pulse_number`, exact induction frequency/recovery delay,
   `PulseResponseFit.dec_fit_reconv_amp`, recording/pair/slice join과 QC;
2. input history와 held-out future/protocol target의 row-level disjointness;
3. pair/slice nesting과 proposal/confirmation split;
4. train-only FPCA/RKHS sieve 차원 menu와 constant-rank quotient rule;
5. missingness가 target 값을 읽지 않는 selection rule;
6. covariance estimator, regularization과 predictive likelihood;
7. direct causal response, Euclidean, constant-full, diagonal, linear, order-shuffle와
   channel-ablation controls;
8. affine rechart에서 kernel/reference metric 자체를 함께 transport하는 gauge test;
9. dimension stability, rank bootstrap, held-out ELPD와 confirmation unlock threshold.

isotropic RBF를 새 affine chart에서 그대로 재계산하면 scale/shear 공변성이 없으므로
허용하지 않는다. reference metric을 kernel distance와 함께 tensor로 transport하거나
orthogonal gauge로 claim을 제한해야 한다.

## 9. FALSIFIER, REVISION_TRIGGER, CLAIM_CEILING

현재 falsifier는 source-level alias 증인이고 이미 small pulse-grid route를 기각했다.
medium receipt와 event support가 없으면 구현ㆍ점수화ㆍconfirmation 접근은 모두
`SKIPPED`다.

후속 결과를 보고 dimension, protocol set, kernel, missingness, threshold 또는 split을
바꾸려면 새 판본과 독립 confirmation이 필요하다. 성공해도 주장할 수 있는 것은
“Allen event observation에서 예측력이 있는 finite observable quotient geometry”뿐이다.
전체 무한차원 뇌 manifold, 물리 시공간, causal routing, conductance, $Npq$,
receptor identity, STDP, eligibility, homeostasis, morphology, 장기 구조변화, 기억
또는 AGI mechanism을 주장하지 않는다. $\Delta F_{\rm CE}=0$이다.
