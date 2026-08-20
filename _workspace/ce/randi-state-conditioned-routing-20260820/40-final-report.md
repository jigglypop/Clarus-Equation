# Randi 상태-조건부 유효 라우팅 연구 보고

Status: COMPLETE

Final decision: `SCHEMA_PROBE_COMPLETE / EMPIRICAL_ROUTE_BLOCKED_CONDITIONAL`

## 초록

이 연구는 붙임 메모가 제안한 “같은 자극의 전파가 자극 직전 전뇌 상태에 따라
달라지는가”를 Randi *C. elegans* 자료에서 시험할 수 있는지 먼저 판정했다. 기존
Loop 8의 M1 결합 성공을 재실행하지 않고, 실제 자료의 source identity와 배정 장치를
결과 비열람 조건에서 감사했다. 동결한 1.27-MB NWB 표본은 event, target ROI와
response-side NeuroPAL schema를 갖지만 stimulation source를 정본 identity에 연결하는
명시적 join과 event-level random/manual assignment receipt가 없다. 따라서 schema-only
probe는 재현 가능하게 완료됐으나 response 효과 계산은 금지되며 empirical route는
조건부 차단이다. 다음 연구는 immutable source join과 assignment/control 영수증을
publication-native object에서 먼저 확보해야 한다.

## 연구 질문과 핵심 결과

붙임 메모의 중요한 전환은 고정 weight 하나를 계속 고치는 모델에서, 느린 구조
$W_{\mathrm{slow}}$ 위에 현재 상태가 빠른 유효 경로 $R_t(z_t)$를 고르는 모델로
이동하는 것이다. 이 아이디어의 첫 실제-자료 질문은 다음 식으로 요약된다.

$$
Y^{\mathrm{post}}=f(A,q,X^{\mathrm{pre}}).
$$

**[산출]** 이 식에서 full pre-state가 예측을 개선하는 것만으로는 라우팅이 아니다.
state의 일반적 예측력을 흡수한 additive model보다 source-by-state interaction model이
held-out animal/session에서 좋아야 한다. 이에 따라 핵심 score를 다음처럼 고정했다.

$$
\Delta_{\mathrm{config}}
=\ell(M_{2,\mathrm{int}})-\ell(M_{2,\mathrm{add}}).
$$

**[정리]** $Y^B=h(X^{\mathrm{pre}})+\epsilon$이고 source effect가 0인 반례에서도
일반 full-state model은 baseline과 gain-only model을 이길 수 있다. 따라서
$M_2>M_1>M_0$만으로 상태-조건부 라우팅을 결론 내릴 수 없다. 이 반례 때문에
source-by-state interaction, 실제 randomization stratum, positivity와 optical/history
adverse control을 필수 조건으로 올렸다.

## 실제 자료 입장 결과

**[공리: 외부 입력]** Randi et al.은 대부분의 source를 약 30초마다 현재 영상에서
찾은 뉴런 중 무작위로 골랐고, 일부 dim neuron은 별도 recording에서 수동 표적했다고
보고한다. recording 뒤 NeuroPAL colour·position·size를 이용해 identity를 배정했다.
이는 가설에 알맞은 장치 설명이지만 event별 배정 영수증은 아니다.

**[산출]** DANDI `001075/0.240920.1434`의 결정론적 최소 segmentation asset을 다시
받아 1,273,970 bytes와 SHA-256
`40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e`를 확인했다.
선행 동결 schema에 대한 새 probe는 세 target reference가 모두 identity 열이 없는
`TargetPlaneSegmentation`으로 끝남을 확인했다. event 열은 onset, stop, power,
target, pattern과 site뿐이며 random/manual, stratum, control, failure 또는 condition
field가 없다. 반면 response-side PumpProbe ROI에는 NeuroPAL mapping schema가 있다.

**[산출]** 기계 판정은
`BLOCKED_EXPLICIT_SOURCE_JOIN / BLOCKED_ASSIGNMENT_RECEIPT`다. response `data`, target
coordinate value와 neural effect는 읽지 않았다. 같은 명령의 두 실행은 JSON SHA-256
`26ed26e811d388ac15be61a27293b2dad7f4532c8a4099f9b049ccbe14a9cdc9`로 일치했다.

## 해석

현재 결과는 상태-조건부 라우팅 가설의 반증이 아니다. 실험 장치는 과학적으로
유망하지만, source identity와 배정 층이 outcome과 독립적으로 복원되지 않은 상태에서
response를 읽으면 spatial matching, source subset 또는 control 선택이 결과에 맞춰질
수 있다. 그래서 이번 중단은 연구 정체가 아니라 잘못된 인과 주장을 막는 apparatus
판정이다.

**[미완성]** target pixel/depth와 tracked-neuron centroid를 연결하는 geometric route는
registration frame, 단위, z calibration, radius, ambiguity rule과 tie-break가 동결되지
않아 열지 않았다. 이 규칙을 결과와 독립적으로 제공하는 validation object가 생기면
R2를 재개할 수 있다.

**[미완성]** publication-native `Fconn`/OSF event object가 source identity,
automatic/manual assignment, failure, event order와 dose를 보존할 가능성은 남아 있다.
공식 bytes와 checksum 또는 compact manifest를 확보하면 R3를 재개한다. code semantics나
processed pair mean만으로 event record를 대신하지 않는다.

## 다음 확인 실험

R1 또는 R3가 통과하면 별도 계약에서 다음 순서로 실행한다.

1. target, response window, source/control set, history, latent dimension과 split을 outcome
   열람 전에 고정한다.
2. train-fold pre-state만으로 global gain $g$와 residual configuration $r$을 만든다.
3. $M_0$, $M_1$, $M_{2,\mathrm{add}}$, $M_{2,\mathrm{int}}$를 animal/session 완전
   holdout에서 비교한다.
4. randomization stratum과 history를 보존한 event–state permutation, lag/block-circular
   shift, source-label permutation과 optical/ROI adverse control을 함께 실행한다.
5. $M_{2,\mathrm{int}}$의 이득이 additive state, gain, dose·geometry, session과 carryover
   대조를 모두 넘어설 때만 randomized source-targeting policy의 state effect
   modification을 보고한다.

양성 결과의 상한도 고정된 immobilized *C. elegans* calcium apparatus에서의 정책
효과다. direct synaptic edge, endogenous $do(X)$, 기억 결합, 포유류 AGI, 의식 또는
metric mediation으로 승격하지 않는다.

## 재현

```powershell
.codex\hooks\python.cmd doctor
.codex\hooks\python.cmd python `
  _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\audit_source_join_schema.py `
  --nwb _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\sub-24-segmentation.nwb `
  --schema _workspace\ce\randi-neural-propagation-source-audit-20260820\artifacts\e2syt-exemplar-schema.json `
  --output _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\source-join-schema-audit.json
```

`$ce-research`가 지정한 `C:\Users\dongh\.codex\hooks\run.ps1`는 현재 환경에 없어
자동 stage check는 실행하지 못했다. stage 이름, 독립 출처·수학·감사 레인과 stable
hash gate는 수동으로 유지했다.

## 참고문헌

- Randi, F. et al., “Neural signal propagation atlas of Caenorhabditis
  elegans,” *Nature* 623, 406–414 (2023),
  https://doi.org/10.1038/s41586-023-06683-4, 접근 2026-08-20.
- DANDI Archive, dandiset `001075`, version `0.240920.1434`,
  https://doi.org/10.48324/dandi.001075/0.240920.1434, 접근 2026-08-20.
- Leifer Lab, `pumpprobe`, https://github.com/leiferlab/pumpprobe, 접근 2026-08-20.
- CatalystNeuro, `leifer_lab_to_nwb`,
  https://github.com/catalystneuro/leifer_lab_to_nwb, 접근 2026-08-20.
