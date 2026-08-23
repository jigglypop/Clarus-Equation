# 10-sources — RT-0 관측 프로토콜 대조 (BA-V3-1)

Status: COMPLETE

ce-physics-sourcer 레인(2026-08-22, 웹 대조)의 전사. BA-TS1 게이트 쌍
모순의 관측 원인을 세분화하고 R2 조작정의를 고정하기 위한 원자료.

## 확립 (게이트 산정 사용 가능)

| 관측 | 수치·정의 | 출처 | 계약 반영 |
|---|---|---|---|
| 성체 V1 spine 안정성 | 1개월 96% 안정 (t0 spine 중 재촬영 잔존, 개체군-분모), 반감기 >13개월 | Grutzendler 2002, *Nature* 420:812 | R1' 목표 0.04 [0.02,0.08] 유지 — 단 분자 구성 P1 (아래) |
| persistent 정의 | 수명 ≥8일 (명시적 임계) | Holtmaat 2005, *Neuron* 45:279 | R2' 임계 8일 |
| 연령별 persistent 분율 | 35%→54%→66%→73% (PND16-25→175-225) | Holtmaat 2005 | R2' 대역: 발달 [0.25,0.45], 성체 [0.60,0.85] |
| 2-클래스 spine 인구 | transient(수일)/persistent 혼합 — Trachtenberg 2002 정성 일치(~50% ≥1개월) | Holtmaat 2005; Trachtenberg 2002 | 모형은 수명 혼합을 낳아야 함 |
| 수면 ASI 하강 | 전체 −18% (vs 자발각성 p=0.001; vs 강제각성 −17.5% p=0.003), 6,920 시냅스/12마리 | de Vivo 2017, *Science* 355:507 | R3a' 0.18 [0.10,0.25] |
| 상위 ~20% "무변화"의 실측 | 점추정 +0.7%~+2.0%, p≈0.99 — **2.7% 감소를 배제하지 못함** | de Vivo 2017 | R3b'는 등식·절대 불변으로 쓰지 않음 (§핵심 판정) |
| 수면과 형성 | 수면박탈 시 학습가지 신규 형성 4.9±0.7% vs 9.3±0.7% (P<0.0001, 7h 창) | Yang 2014, *Science* 344:1173 | R6' 정성 게이트 근거 |
| REM 제거 촉진 | 정성 확립 (경험의존 제거, REM·Ca²⁺ 스파이크 차단 시 감소) | Zhou 2020, *Nat Commun* 11:4819 | 정량 % 미확보 — 게이트 불사용 |

## 핵심 판정 (RT-0)

1. **R2의 (P) 해석 기각 — 논리적 근거**: 임의 생존곡선의 단조성상
   $S(8\text{일})\ge S(30\text{일})$인데 (P)로 읽으면 $0.73<0.96$ —
   Holtmaat 수치는 모집단-스냅샷 생존확률일 수 없다. **R2'는 (N)-계열
   (관측 창 내 전체 spine 인스턴스의 수명 분류)로 고정**하고, 원문
   Experimental Procedures 미열람이므로 "[미완성: 원문 분모 미확인, (N)
   잠정 채택]"을 계약에 명기한다.
2. **R1 분자 구성 미확인 (P1)**: "4%/월"이 제거 전용인지 (제거+형성)인지
   원문 Table 미대조. 목표·대역은 유지하되 P1 미해결 표기, 모형은 두
   구성 모두 보고.
3. **R3b 판별 불가능성**: de Vivo의 상위 분위 "무변화"는 통계적으로
   2.7% 수준 손실과 구별 불가 — BA-TS1의 RT-A 교차 예측은 이 관측으로
   반증도 확증도 되지 않는다. 계약에 명기.

## UNVERIFIED (게이트 불사용)

Grutzendler 원문 imaging 스케줄·"4%/월" 직접 표기 여부, Holtmaat 원문
분모, transient 평균 수명, 2017 Neuron 리뷰 본문(403), de Vivo 정식
검출한계, Yang 원문 표(PDF 추출 실패), Zhou 정량 %.

접근 실패 원자료 목록과 저장소 참조 파일은 sourcer 레인 보고 원문에 기록
(TS1 아카이브 4개 파일 인용). 접근 날짜: 전 항목 2026-08-22.

## 2026-08-23 실제 뇌 식 기반 재감사

새 정본 `.codex/harnesses/real_brain_equation_discovery_loop.md`에 따라 원
방정식의 관측 대상, 분자·분모, 측정모형과 재분석 가능한 실제자료를 다시
감사했다. 아래 판정이 앞의 잠정 게이트 해석보다 우선한다.

| Gate | 1차 출처에서 확인한 측정량 | 현재 상태 | 구현 전 조치 |
|---|---|---|---|
| R1′ | 성체 시각피질 spine의 1개월 안정분율 약 96% | `UNVERIFIED_AS_GATE` | $4\%$를 제거율로 자동 치환하지 않는다. `lost/baseline`, gained와 net change를 분리하고 exact interval을 고정한다. |
| R2′ | persistent spine은 수명 $\ge8$일; 연령대별 비율 35%, 54%, 66%, 73% | `VERIFIED_N_SERIES_WITH_METADATA_REQUIRED` | 해당 imaging session·age cohort의 전체 spine population을 분모로 쓰되 cohort weighting, censoring와 interval을 원 방법절에서 전사한다. |
| R3a′ | sleep군의 ASI는 spontaneous wake 대비 18.9%, enforced wake 대비 17.5% 작음 | `VERIFIED_MORPHOLOGY_COMPARISON` | ASI를 직접 시냅스 강도로 동일시하지 않고 SBEM 관측모형과 hierarchical sample unit을 적는다. |
| R3b′ | largest 20% ASI subgroup은 +0.7%, +2.0%; $p=0.999,0.994$ | `DEFINITION_VERIFIED_GATE_UNVERIFIED` | largest 20%는 rank-based 분석 선택이다. CI/SE가 없는 $(0,0.05]$ 정밀 gate는 제거하고 subgroup effect와 uncertainty를 보고한다. |
| R4′ | 주기 총강도 표류 $<5\%/100$일 | `UNVERIFIED` | 이름 있는 longitudinal dataset과 estimator가 없으므로 현재 gate에서 제거한다. |
| R5′ | 연령별 persistent fraction의 증가 | `UNVERIFIED_AS_RATIO` | 출처는 1.5라는 과잉 peak 비율을 주지 않는다. 명시적 원자료 산출 없이는 현재 gate에서 제거한다. |
| R6′ | post-training 8시간 동안 high-frequency branch 신규 형성: sleep 9.3±0.7%, deprivation 4.9±0.7% | `VERIFIED_NARROW_CONTRAST` | 일반 생존율 비가 아니다. branch class, training, deprivation, 8시간 window를 보존한 formation contrast로 다시 정의한다. |

### 1차 출처와 실제자료

- Grutzendler et al., *Nature* 420, 812–816, DOI
  [10.1038/nature01276](https://doi.org/10.1038/nature01276).
- Holtmaat et al., *Neuron* 45, 279–291, DOI
  [10.1016/j.neuron.2005.01.003](https://doi.org/10.1016/j.neuron.2005.01.003),
  [CSHL repository record](https://repository.cshl.edu/id/eprint/22598/).
- de Vivo et al., *Science* 355, 507–510, DOI
  [10.1126/science.aah5982](https://doi.org/10.1126/science.aah5982),
  [author manuscript](https://escholarship.org/content/qt9sw6509h/qt9sw6509h.pdf).
- Yang et al., *Science* 344, 1173–1178, DOI
  [10.1126/science.1249098](https://doi.org/10.1126/science.1249098),
  [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC4447313/).

de Vivo 연구는 공식 연구실 페이지에서 `synapse_data.csv`를 공개한다:
[UW–Madison data sets](https://centerforsleepandconsciousness.psychiatry.wisc.edu/data-sets/).
페이지가 제시한 파일 크기는 약 450 kB, MD5는
`12e0c2e5ea231619df91a3c8d816d246`이다. 자료는 ASI/HV 산출을 포함하지만
원 SBEM volume은 아니며, 세 집단 각 4마리의 cross-sectional 설계다.
spine를 임의 분할하지 않고 animal을 표본·분할 단위로 써야 한다. 작은
동물 수 때문에 이 자료 하나를 식 제안과 독립 confirmation에 동시에 쓰지
않는다.

## Source-lane 판정

R2′와 R3a′는 측정 metadata를 명시한 기술 비교로 보존할 수 있다. R1′,
R3b′, R6′는 scoring 전에 다시 정의해야 하고 R4′, R5′는 새 원자료가 없으면
삭제한다. 현재 계약의 전 gate 동시 적합은 source lock을 통과하지 못했으므로
구현할 수 없다. 주장 상한은 L1 정합이며 CE 고유 기전이나 개입 동일성을
식별하지 않는다.
