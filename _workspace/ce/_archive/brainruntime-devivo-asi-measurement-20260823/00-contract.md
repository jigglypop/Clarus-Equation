# BA-DV1 계약 — de Vivo 실제 ASI 자료의 측정식 고정과 기준선 재현

Status: COMPLETE

CE_RUN: _workspace/ce/brainruntime-devivo-asi-measurement-20260823
Candidate: BA-DV1
Priority: HIGHEST_PRIORITY_FOR_BRAIN_AGI_RESEARCH
Date frozen: 2026-08-23 (Asia/Seoul)
Source revision: 1/2 (`physics-sourcer`; checksum/schema only, before effect computation)
Claim ceiling: L1 descriptive calibration; no causal, intervention, AGI, or CE-specific mechanism claim

## 1. 질문과 성공의 의미

공식 공개 `synapse_data.csv`에서 de Vivo et al. (2017)의 ASI
(axon–spine interface) sleep/wake 비교를 원 논문의 표본 중첩과 측정 정의를
보존해 재현할 수 있는가? 재현 후에만, CE의 “선택된 강한 접촉은 더 보존되고
비선택 접촉은 더 많이 접힌다”는 가설을 나타내는 매끄러운 크기 의존식이
균일한 multiplicative shift 기준선보다 animal-level 교차검증에서 나은지
진단한다.

이 run의 성공은 공개 자료의 측정식과 기준선을 재현하고 두 사전 고정 후보를
정직하게 비교하는 것이다. 어느 후보가 우세해도 실제 뇌가 CE 식을 사용한다는
증거가 아니며, 원 논문과 같은 자료에서 알려진 size dependence를 재서술하는
L1 정합을 넘지 않는다.

## 2. 선행 증거와 금지된 재해석

| 선행 항목 | 판정 | 보존 산출 | 금지 |
|---|---|---|---|
| BA-TS1 | STOP | absolute top-exemption은 순환정상상태를 만들지 못한다 | hard exemption 재도입 |
| BA-V3-1 | BLOCKED | $\lambda(w)>0$만으로 정상성이 보장되지 않는 반례; 실제 측정식 선행 | 검증되지 않은 R1′–R6′ 동시 피팅 |
| de Vivo 2017 | 경험 관측 | sleep군 ASI가 wake군보다 낮고 효과가 size-dependent하다는 SBEM 비교 | ASI를 직접 전기생리 strength로 동일시하거나 개인 시냅스의 paired trajectory로 해석 |

`PREDECESSOR_EVIDENCE`: BA-V3-1의 `10-sources.md`, `11-math.md`,
`20-audit.md`, `40-final-report.md`. 현재 계약 이전 접촉은 논문·공식 데이터셋
  페이지·파일명·크기·공개 MD5뿐이며 최초 동결 시 CSV 내용과 열은 열지
  않았다. 이후 source revision 1에서 checksum과 schema만 열었고 조건별
  효과값·분포·회귀계수는 계산하지 않았다.

## 3. 데이터 provenance와 접촉 규칙

- 논문: de Vivo et al., *Science* 355, 507–510 (2017), DOI
  `10.1126/science.aah5982`.
- 공식 페이지:
  `https://centerforsleepandconsciousness.psychiatry.wisc.edu/data-sets/`.
- 공식 파일:
  `https://centerforsleepandconsciousness.psychiatry.wisc.edu/wp-content/uploads/2021/02/synapse_data.csv`.
- 페이지에 게시된 MD5: `12e0c2e5ea231619df91a3c8d816d246`.
- 2026-08-23 현재 canonical HTTPS 응답: 442,376 bytes, ETag
  `"6c008-5f424506f58bc"`, Last-Modified `2023-02-07T23:14:49Z`.
- 두 독립 client에서 동일한 현재 파일: MD5
  `78e2a39bb8b7bf8f7b2b3e2864285e9d`, SHA-256
  `673f56e6e8208d1eb1980d32cf3d12673f3d3d888f55832d2e787b60bc8cb5ec`.
- 게시 MD5와 현재 canonical 파일은 불일치한다. 이를 숨기거나 게시값을
  교체하지 않으며 상태를
  `VERIFIED_SOURCE_IDENTITY_WITH_STALE_PUBLISHED_MD5`로 제한한다.
- local raw path: `data/external/devivo2017/synapse_data.csv` (gitignored;
  수정 금지).
- raw acquisition 뒤 SHA-256, byte count, retrieval time, URL을
  `artifacts/realdata/download-receipt.json`에 기록한다.
- 행 삭제·열 변환·결측 처리 전후의 수와 이유를 receipt로 남긴다. 원본을
  덮어쓰지 않는다.
- schema-only 접촉 결과는 6,943행, 12마리, 168 dendrite다. ASI 결측
  638행과 head-volume 결측 23행이 있어 각각의 유효 표본은 6,305와
  6,920이다. ASI 분석은 ASI가 있는 6,305행만 쓰고 imputation하지 않는다.
  한 개의 완전 동일 행은 생물 개체 ID 중복 증거가 아니므로 보존한다.

## 4. 실제 뇌 출발식, CE 차이식, 측정식

### 4.1 관측 변수

$A>0$는 원 자료의 ASI 면적이고

$$
z=\log(A/A_{\rm ref})
$$

로 무차원화한다. $A_{\rm ref}$는 전체 자료 median이 아니라 source-locked
단위에서의 고정 기준 `1 square micrometre`로 둔다. 실제 열 단위가 이와
다르면 데이터 접촉 후 값이 아니라 metadata에 의해 등가 기준을 정하고,
그 변경은 source lane에 기록한 뒤 audit 전 계약 revision으로만 허용한다.

### 4.2 생물학적 기준 후보 M0

균일한 multiplicative sleep scaling의 수면 좌표 $s\in[0,1]$ 식은

$$
\frac{dz}{ds}=F_{\rm bio}(z)=-\delta_{\rm bio},
\qquad \delta_{\rm bio}\ge0.
$$

따라서 한 수면 구간 뒤 $A_S=A_W e^{-\delta_{\rm bio}}$이다. 이것은
비교 기준이지 실제 개별 spine의 paired dynamics라는 주장이 아니다.

### 4.3 CE 가설 후보 M1

강한/선택된 접촉의 상대적 보존과 약한/비선택 접촉의 graded folding을
hard threshold 없이 나타내는 단일 구조 차이는

$$
\frac{dz}{ds}
=F_{\rm bio}(z)+\Delta F_{\rm CE}(z)
=-\delta_\infty
-\delta_{\rm fold}\,
\sigma\!\left(\frac{c-z}{\tau}\right),
$$

$$
\sigma(q)=\frac{1}{1+e^{-q}},
\quad \delta_\infty,\delta_{\rm fold}\ge0,
\quad \tau>0.
$$

M0은 $\delta_{\rm fold}=0$인 nested control이다. M1은 BA-V3-1의
$\lambda(w)\to0$ tail을 그대로 쓰지 않으며 $\delta_\infty$를 허용한다.
이번 판본에서 허용되는 구조식은 M0/M1뿐이다. 데이터 잔차를 본 뒤 함수족,
link, threshold 수를 추가하지 않는다.

### 4.4 측정모형 $\mathcal H_d$

자료는 개인 spine의 wake→sleep pair가 아니라 서로 다른 animal의
cross-sectional 표본이다. 따라서 식은 조건별 분포 transport로만 읽는다.
원 논문에 맞춘 기본 관측식은

$$
z_i=\mu+\beta_{g[i]}+\beta_{r[i]}
+\gamma\log(D_i/D_{\rm ref})
+b_{m[i]}+b_{d[i]}+\epsilon_i,
$$

$$
b_m\sim\mathcal N(0,\sigma_m^2),\quad
b_d\sim\mathcal N(0,\sigma_d^2),\quad
\epsilon_i\sim\mathcal N(0,\sigma^2).
$$

$g$는 sleep/spontaneous-wake/enforced-wake, $r$은 region, $D$는 dendrite
diameter, $m$은 mouse, $d$는 dendrite/segment다. CSV schema가 원 논문
변수를 제공하지 않으면 항을 조용히 삭제하지 않고 `UNAVAILABLE`로 표시해
측정모형을 revision gate로 되돌린다.

M0/M1의 분포 비교는 wake 기준의 source-locked 조건부 quantile
$Q_W(p)$를 각각의 flow로 $s=1$까지 보낸 예측 $\widehat Q_S(p)$와
animal-clustered 관측 $Q_S(p)$의 잔차로 한다. 순위 보존은 분석 가정이지
개별 spine 추적 사실이 아니다.

## 5. 분할과 데이터 사용

- 분할 단위는 항상 mouse다. spine/row 단위 무작위 분할은 금지한다.
- 12마리(보고된 3조건×4마리)가 확인되면 고정된 leave-one-mouse-out
  (LOMO) 교차검증을 사용한다. 식별자는 정렬 후 fold를 만들며 seed를 쓰지
  않는다.
- LOMO는 development 재사용이므로 independent confirmation이 아니다.
  이번 자료 전체의 claim ceiling은 descriptive calibration이다.
- 원 논문의 전체 자료 효과 재현과 LOMO predictive comparison을 분리한다.
  전체 자료 결과를 보고 M1 구조나 cutoff를 바꾸지 않는다.

## 6. 관측량과 잔차 규칙

### O1 — 원 논문 효과 재현

$$
D_{S,W}=1-\exp(\beta_S-\beta_W).
$$

자발 각성 기준 보고값은 0.189, 강제 각성 기준은 0.175다. 코드·단위·조건
라벨 sanity gate는 각 값에 대해 절대오차 0.03 이내로 사전 고정한다. 이
tolerance 통과는 파이프라인 재현일 뿐 새 과학 증거가 아니다. 원 논문의
정확한 estimator와 공개 열이 다르면 수치를 강제로 맞추지 않고 discrepancy를
보고한다.

### O2 — 분포 잔차

$p\in\{0.1,0.2,\ldots,0.9\}$에서

$$
r_p=Q_S(p)-\widehat Q_S(p)
$$

를 mouse-cluster bootstrap uncertainty와 함께 기록한다. top 20% 하나만을
별도 tuning target으로 쓰지 않는다.

### O3 — predictive model comparison

mouse holdout별 log predictive density 차이의 합 $\Delta\mathrm{ELPD}$와
fold 간 표준오차를 사용한다. M1 채택 조건은

$$
\Delta\mathrm{ELPD}>2\,\mathrm{SE}
$$

이고, 그렇지 않으면 단순한 M0을 보존한다. M1 파라미터가 경계에 붙거나
profile이 평평하면 `UNIDENTIFIED`로 판정한다.

## 7. falsifier와 matched controls

- F1: 서로 다른 client의 bytes/SHA-256이 불일치하거나 HTTP
  content-length/ETag receipt가 달라짐, 또는 핵심 mouse/condition/ASI/nesting
  열 부재 → 분석 중지, source/schema block. 게시 MD5 불일치는 이미 확인된
  provenance defect로 계속 보고하되, 동일 canonical bytes의 분석 자체를
  단독으로 막지는 않는다.
- F2: O1이 tolerance를 벗어나고 원 논문 estimator 차이로 설명되지 않음 →
  측정모형 실패, M1 평가 금지.
- F3: M1이 M0보다 $2\,\mathrm{SE}$ 이상 개선하지 못함 → CE 구조 채택 금지.
- F4: LOMO의 단일 mouse 제거로 $\delta_{\rm fold}$ 부호 또는 모델 판정이
  뒤집힘 → `UNSTABLE`, 채택 금지.
- F5: $\delta_\infty=0$ 근방으로 가며 큰-$A$ restoring loss를 제거하거나,
  $\tau,c$가 비식별 → BA-V3-1 정상성 반례를 해결했다는 주장 금지.

Matched controls는 M0, condition-label permutation(검정 전용), region·diameter
항을 포함한 동일 중첩 측정모형이다. M1에만 유리한 전처리나 표본 삭제는
허용하지 않는다.

## 8. revision trigger와 판본 규칙

schema/source 불일치만 audit 전에 계약 revision 후보가 될 수 있으며 역할별
한도를 지킨다. 결과를 본 뒤 구조식을 고치지 않는다. M1 실패 뒤 다른 함수족,
homeostatic feedback, wake plasticity를 시험하려면 BA-DV2 같은 새 계약과 새
falsifier가 필요하다. 한 판본에서는 구조식 한 개만 바꾼다.

## 9. 단계별 중지점

1. raw acquisition + checksum + schema receipt.
2. source/math 독립 레인과 audit.
3. audit가 구현을 허용할 때만 O1 기준선 재현.
4. O1이 성립할 때만 O2/O3 M0 대 M1 비교.
5. 결과와 무관하게 equation version, residual, falsifier, 실패 판본을 보존.

확인 자료를 제안 자료로 재사용하지 않으며, 임계값·endpoint·전처리·fold를
결과를 본 뒤 바꾸지 않는다.
