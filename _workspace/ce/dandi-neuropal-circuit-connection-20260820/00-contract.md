# DANDI NeuroPAL 회로-접속 실데이터 루프 계약

Status: SOURCE_REVISION_1_FROZEN

PREDECESSOR_EVIDENCE:

- `_workspace/ce/cloudcell-heterogeneous-circuit-geometry-20260820/31-validation.md`: A1 post-fit gain 실패, A2 anisotropic ridge confirmation 미통과
- `_workspace/ce/cloudcell-heterogeneous-circuit-geometry-20260820/artifacts/confirmation-result.json`
- `docs/6_뇌/11_리만계량_라우팅_논문.md` 식 (21)--(28)

## 질문

CloudCell A2가 circuit-shuffle을 이기지 못한 원인은 recording-local dense autoregressive weight에서 geometry와 predictor를 동시에 만들었기 때문일 수 있다. 새 후보 A3는 NeuroPAL identity·position과 calcium time series가 함께 있는 독립 worm에서 **회로장을 dense weight multiplier가 아니라 저차원 connection operator**로 사용한다.

이 run은 물리적 뇌 주름이나 시냅스 parent를 주장하지 않는다. DANDI 000541은 optogenetic assignment receipt를 광고하지 않으므로 A3는 처음부터 observational connection prediction으로 제한한다.

## 고정 자료

DANDI 000565의 `<300 MB` 선택은 schema gate에서 정적 NeuroPAL 영상뿐임이 확인되어 endpoint 전에 폐기됐다. 실패 byte와 schema는 삭제하지 않는다.

수정 source는 DANDI 000541, published version `0.241009.1457`, CC-BY-4.0이다. 공식 metadata는 `C. elegans head NeuroPAL and Calcium imaging`, 관련 논문은 약 4분·약 4 Hz라고 명시한다. response를 열기 전에 공식 `contentSize`가 작은 순서의 8개 자산을 고정했다. ID, path, SHA-256, URL은 `artifacts/source-manifest-v2.json`에 있다.

자산의 response array를 열기 전 고정한 순서는 manifest의 size 오름차순이다.

- development/schema+formula: 가장 이른 3 worms
- confirmation: 나머지 5 worms

confirmation 5개는 A3 식, field path, event window, normalization, controls가 고정될 때까지 response value를 집계하지 않는다.

## 필수 schema gate

각 NWB에서 다음 byte-backed object를 영수증으로 기록한다.

1. calcium response matrix와 sample timestamps/rate;
2. ROI row와 같은 행의 NeuroPAL neuron label 또는 confidence;
3. ROI position 또는 centroid와 좌표 단위;
4. trace row와 NeuroPAL label row의 연결 receipt;
5. missing/failed trace retention.

1--4 중 하나가 빠지면 A3 전체를 `BLOCKED_INPUT`으로 중단한다.

## A3 수학 후보

Primary signal은 released `dNMFCalciumImResponseSeries/data`; `SignalRawFluor`는 robustness만 허용한다. row는 `/processing/CalciumActivity/NeuronIDs/labels`와 같은 길이여야 하고 blank label은 결과를 보지 않고 제외한다. NeuroPAL `voxel_mask`의 $(x,y,z)$ index에는 NWB `CalciumImVol/grid_spacing`을 곱하고, 그 좌표를 worm별 median 6-NN 거리로 무차원화한다.

936 sample과 4 Hz인 첫 schema를 기준으로, 각 worm의 chronological 역할을 비율로 고정한다. $K=[0,0.2T)$, $B=[0.2T+8,0.7T-8)$, $T_{test}=[0.7T+8,T-1)$이다. 각 chemical stimulus 구간의 시작 5초 전부터 종료 20초 후까지는 세 역할 모두에서 제외한다. 이 규칙은 다른 worm의 sample count/rate에도 초 단위로 그대로 적용한다.

calibration-only $K$에서 뉴런별 median/MAD와 threshold를 고정한다.

$$
z_i(t)={x_i(t)-\mu_i\over\max(s_i,10^{-6})},
\qquad
q_i(t)=\mathbf 1[z_i(t)\ge2.5]
\min\left(1,{[z_i(t)-2.5]_+\over2.5}\right).
$$

calibration-only $K$에서 학습한 top-3 principal component를 모든 split에서 제거한 residual을 $r(t)$라 한다. 개발 endpoint를 열기 전에 역치가 회로 구성식에서 빠져 있던 불일치를 수정하고, 관측 사건-진폭을

$$
u_i(t)=q_i(t)r_i(t)
$$

로 고정한다. 아래의 모든 $C$ 식에서 $r$ 대신 $u$를 사용한다. 예측 대상은 연속 상태 $z(t+1)$로 유지한다. 한 sample lag covariance를

$$
C_{ij}^{+}=\mathbb E_B[u_i(t+1)u_j(t)],
\qquad
C_{ij}^{-}=\mathbb E_B[u_i(t-1)u_j(t)]
$$

로 정의한다. 이 forward/reverse 차이는 direction-sensitive observational statistic이지 causal parent receipt가 아니다.

shift-null 평균을 뺀 비음수 symmetric conductance와 directed circulation은

$$
c_{ij}=\left[
{C_{ij}^{+}+C_{ji}^{+}\over2}
-\mathbb E_{\tau\in\{17,31,47\}}
{C_{ij}^{(\tau)}+C_{ji}^{(\tau)}\over2}
\right]_+,
$$

$$
\Omega_{ij}={ (C_{ij}^{+}-C_{ij}^{-})-(C_{ji}^{+}-C_{ji}^{-})\over4},
\qquad \Omega^\top=-\Omega
$$

이다. $c$와 $\Omega$는 6-NN 공간 edge에서만 남긴다. $c\ge0$에서 $L_c=D_c-c\succeq0$이다. A3 predictor는 dense 자유 weight를 만들지 않고 세 operator만 사용한다.

$$
\widehat z(t+1)
=b+\beta_0 z(t)-\beta_d L_c z(t)+\beta_\circlearrowright\Omega z(t).
$$

계수 $b_i,\beta_0,\beta_d,\beta_\circlearrowright$는 각 worm의 construction data에서 neuron-time observation을 쌓은 scalar ridge $10^{-2}$로만 적합한다. 즉 neuron별 자유항 외에는 worm마다 세 scalar만 학습한다. 이 식은 symmetric diffusion과 directed circulation이 future neural state에 추가 정보를 주는지를 직접 시험한다.

## 필수 대조

- spatial-only $L_{6NN}$, no circulation;
- $L_c$만 있고 $\Omega=0$;
- edge-label shuffle preserving $c$ multiset and degree bin;
- time/block shift;
- source-target reversal $\Omega\mapsto-\Omega$;
- identity shuffle within confidence bin;
- phase-randomized trace control preserving each neuron spectrum.

## 판정과 수정 루프

primary score는 worm별 held-out Gaussian log-score에서 spatial-only 대비 A3 차이 $\Delta_w$다. confirmation unit은 worm이다. `PASS_A3`는 5/5 $\Delta_w>0$, exact one-sided sign-flip $p<0.05$, A3 mean이 모든 matched control보다 큼, source reversal이 A3를 재현하지 않음을 모두 요구한다.

development gate는 첫 3 worms 중 2개 이상에서 $\Delta_w>0$, mean $\Delta>0$, 그리고 mean이 no-circulation·edge-shuffle·time-shift·reversal·identity-shuffle·phase-randomized 모두보다 큰 것이다. 이를 못 넘으면 confirmation은 열지 않는다.

development 실패 뒤 허용되는 수정은 정확히 하나다. numerical receipt가 $\rho(\beta_0I-\beta_dL_c+\beta_\circlearrowright\Omega)\ge1$을 보이면 operator를 spectral normalization하여 같은 세 coefficient를 다시 적합한다. 그렇지 않은 실패에서 threshold, window, labels, graph, latent dimension, decoder, seed를 바꾸지 않는다. confirmation 실패 뒤 같은 자료에 추가 식을 맞추지 않는다.
