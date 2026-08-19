# CloudCell local input eligibility audit

Status: COMPLETE

이 문서는 로컬 바이트와 공식 재현 코드만 읽은 입력 감사다. 원자료를
수정·추출·다운로드하지 않았고, neuron row를 임의의 source/target 집단으로
나누지 않았다. 아래 `PASS_INPUT`은 경험 결과가 아니라 후속 사전등록 분석에
필요한 필드가 있다는 뜻이다.

## 1. 출처와 로컬 바이트

공식 코드 사본은 `data/external/cloudcell/PredictionCode`에 있다. 이 사본의
`README.md`는 Hallinen et al., *eLife* 2021, “Decoding locomotion from population
neural activity in moving C. elegans”, DOI `10.7554/eLife.66135`의 재현 코드이며
원자료 위치를 `https://osf.io/dpr3h`로 명시한다. 로컬 원자료 묶음은 다음과
같다.

| 로컬 archive | bytes | SHA-256 | tar 내부 root | 기록 수 |
|---|---:|---|---|---:|
| `data/external/cloudcell/AML310_moving.tar.gz` | 348,444,164 | `144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a` | `AKS297.51_moving/` | 4 |
| `data/external/cloudcell/AML32_moving.tar.gz` | 1,218,075,251 | `6b71a6ba1a5d2f1ef3bf9661e845e1e52634bae217fc0c2630a83fca07daed63` | `AML32_moving/` | 7 |
| `data/external/cloudcell/AML18_moving.tar.gz` | 1,409,801,111 | `588d7666f4e8afebad1ab9b8483244a6de0303251d862425522c2b8dd78bbd82` | `AML18_moving/` | 11 |

각 tar의 공식 recording directory에는 `centerline.mat`, `heatData.mat`,
`heatDataMS.mat`, `pointStatsNew.mat`, `positionDataMS.mat`가 있다. 일부 기록에는
`*_old.mat` 또는 `*_preedit.mat`도 있으나 이 감사에서는 정본 입력으로
선택하지 않았다.

### AML310 archive와 AKS297.51 root의 정확한 경계

`AML310_moving.tar.gz`라는 archive 이름과 달리 archive 내부 dataset log와 root는
각각 `AKS297.51_moving/AKS297.51_moving_datasets.txt`와
`AKS297.51_moving/`다. 로컬 추출 root도
`data/external/cloudcell/extracted/AKS297.51_moving`이다. 반면 공식
`PredictionCode/utility/get_all_recordings.py`는 GCaMP 입력 directory를 literal
`AML310_moving`과 `AML32_moving`으로 찾는다. 따라서 현재 로컬 tree에 공식
loader를 그대로 실행하면 AML310 계열 경로를 찾지 못한다.

- archive-name ↔ internal-root 대응 자체: `PASS_PROVENANCE`
- 수정 없는 공식 loader의 현재 로컬 실행: `APPARATUS_PATH_BLOCKED`
- 후속 구현 조건: raw byte를 바꾸거나 이름을 암묵적으로 동일시하지 말고,
  manifest에 이 단일 경로 대응을 명시한 read-only adapter를 사용한다.

## 2. 현재 추출 상태

| 계열 | 현재 추출 directory | 현재 파일 | 판정 |
|---|---|---|---|
| AML310/AKS297.51 | `data/external/cloudcell/extracted/AKS297.51_moving/<recording>_MS` | 네 기록 모두 `centerline.mat`, `heatDataMS.mat`, `pointStatsNew.mat` | neural/volume-aligned behavior 즉시 판독 `PASS`; `positionDataMS.mat`는 tar에만 있음 |
| AML32 | `data/external/cloudcell/extracted/AML32_moving/<recording>_MS` | 일곱 기록 모두 `heatDataMS.mat`만 | neural/embedded behavior `PASS`; 원 50 Hz centerline 재계산은 현재 추출 상태에서 `BLOCKED_EXTRACTED_CENTERLINE` |
| AML18 | `data/external/cloudcell/extracted/AML18_moving/<recording>_MS` | 열한 기록 모두 `heatDataMS.mat`만 | GFP control 판독 `PASS`; 원 50 Hz centerline 재계산은 현재 추출 상태에서 `BLOCKED_EXTRACTED_CENTERLINE` |

AML32와 AML18의 누락 파일은 공식 tar member로 존재한다. 이번 read-only 감사에서
새로 추출하지 않았으므로 “없음”이 아니라 “현재 filesystem에 미추출”이다.

## 3. `heatDataMS.mat` 정본 schema와 시간축

22개 `heatDataMS.mat`는 모두 `MATLAB 5.0 MAT-file`이다. 모든 기록에서 다음
schema가 확인됐다.

| 변수 | shape | 의미/제한 |
|---|---|---|
| `rRaw`, `gRaw` | `N x T` | red/green raw fluorescence |
| `rPhotoCorr`, `gPhotoCorr` | `N x T` | released photobleach-corrected channels |
| `R2`, `G2`, `Ratio2` | `N x T` | released derived fluorescence arrays |
| `acorr` | `N x N` | 모든 기록에서 finite·symmetric인 derived activity correlation; structural connectivity가 아님 |
| `cgIdx`, `cgIdxRev` | `N x 1` | MATLAB 1-based correlation-cluster ordering; anatomical identity나 route label이 아님 |
| `XYZcoord` | `N x 3` | 있을 때 recording-local neuron coordinates; 다수 기록은 변수가 있으나 값 전체가 NaN |
| `hasPointsTime` | `T x 1` | neural volume clock |
| `clTime` | `T_cl x 1` | centerline clock |
| `behavior` | `1 x 1 struct` | `ethogram`, `x_pos`, `y_pos`, `v`, `pc1_2`, `pc_3` |

`ethogram`, `x_pos`, `y_pos`, `v`, `pc_3`는 각각 길이 `T`, `pc1_2`는
`T x 2`다. 즉 embedded behavior는 neural row의 각 volume과 같은 index에 있다.
`hasPointsTime`의 median 간격은 2017–2018 AML32 네 기록에서 약 `0.166 s`,
나머지에서 약 `0.165 s`다. `clTime`의 median 간격은 약 `0.020 s`다. 공식
`PredictionCode/utility/data_handler.py`는 `hasPointsTime`을 `clTime`에
보간한 뒤 가장 가까운 centerline index로 반올림하고, embedded behavior와
neural arrays에는 같은 `idx_data`를 적용한다.

모든 기록에서 `behavior` 길이와 neural `T`가 일치한다. 기계 재감사
`cloudcell-input-audit.json`은 10개 recording의 첫 차분(index 0)에서 정확히 한
번의 중복 timestamp를 찾았고, `BrainScanner20200310_141211`에서는 median 양의
간격의 3배를 넘는 gap 3개(index 83, 121, 135)를 찾았다. 모든 recording은
고정된 선두 12-volume guard 뒤에는 엄격 순증가한다. 따라서 후속 분석은 선두
12 volume을 버리고 gap을 가로지르는 모든 history/future window를 제외해야
한다. 이 고정 guard 아래 같은-recording neural→future behavior output 정렬은
`PASS_TIME_ALIGNMENT_WITH_GUARD`다. 다만 native trial ID, task-block ID,
randomized event timestamp는 없다. `ethogram`의 유한 raw code는 모든 기록에서
`{-1, 1, 2}`였지만, 로컬 코드가 이 세 값을 task context로 정의하지 않으므로
그렇게 승격하지 않는다.

기계 영수증은 이 규칙을 선언만 하지 않고 anchor index로 고정한다. history와
future를 각각 6 volume으로 두고, 선두 12 volume 뒤의 시간축을 60/20/20으로
나눈 다음 각 split 경계 양쪽에 12-volume embargo를 적용한다. 전체
history/future window 안에 `dt <= 0` 또는 `dt > 3 * median_positive_dt`가 있으면
그 anchor를 제거한다. 22/22 recording에서 train/validation/test 허용 anchor가
모두 남았다. `BrainScanner20200310_141211`에서는 gap 때문에 anchor 36개가
제외됐고, 허용 수는 train 829, validation 261, test 273이다. 모든 recording의
정확한 허용 anchor 목록은 `cloudcell-input-audit.json`에 저장했다. 이 mask는
clock-only이므로 후속 분석은 neural/output finite-window 조건도 추가로 적용해야
한다.

## 4. signal class

공식 `PredictionCode/utility/get_all_recordings.py`는 `AML310_moving`과
`AML32_moving`을 `gcamp`로, `AML18_moving`을 `gfp`로 분리해 서로 다른
`*_recordings.dat`를 만든다.

- AML310/AKS297.51 4 + AML32 7: primary neural activity 입력
  `PASS_SIGNAL_CLASS`.
- AML18 11: motion, photobleaching, gain 및 nuisance negative control
  `PASS_CONTROL`; primary neural Fisher metric 또는 neural routing 입력은
  `BLOCKED_SIGNAL_CLASS`.

## 5. GCaMP recording별 적격성

`usable T`는 공식 dataset log에 두 번째 정수가 있으면 공식 loader와 같이
zero-based `index <= cutVolume`을 적용한 길이다. `majority-valid T`와
`units >=75%`는 raw red와 green이 동시에 finite인 비율을 기술한 schema
진단일 뿐, 후속 분석의 unit 선택 threshold가 아니다.

| recording | N x full T | log cut / usable T | majority-valid T | units >=75% | finite XYZ | 공식 보조 정보 |
|---|---:|---:|---:|---:|---:|---|
| `BrainScanner20200130_105254` | 128 x 1654 | 1524 / 1525 | 1341/1525 | 127/128 | 0/128 | exclusion 65–75 s; AVA row 95 |
| `BrainScanner20200130_110803` | 134 x 1641 | 1465 / 1466 | 1447/1466 | 134/134 | 0/134 | AVA rows 32, 15 |
| `BrainScanner20200310_141211` | 116 x 1582 | 1494 / 1495 | 1268/1495 | 108/116 | 116/116 | exclusions 200–210, 240–250 s; AVA rows 71, 42 |
| `BrainScanner20200310_142022` | 97 x 1627 | 1479 / 1480 | 1426/1480 | 97/97 | 97/97 | AVA rows 15, 16 |
| `BrainScanner20170424_105620` | 112 x 4317 | — / 4317 | 4096/4317 | 112/112 | 112/112 | — |
| `BrainScanner20170610_105634` | 107 x 3955 | — / 3955 | 3857/3955 | 107/107 | 107/107 | — |
| `BrainScanner20170613_134800` | 118 x 3922 | — / 3922 | 3758/3922 | 118/118 | 0/118 | — |
| `BrainScanner20180709_100433` | 135 x 4153 | — / 4153 | 4151/4153 | 135/135 | 135/135 | — |
| `BrainScanner20200309_151024` | 121 x 1622 | 1492 / 1493 | 1427/1493 | 121/121 | 0/121 | exclusions 30–40, 125–135 s |
| `BrainScanner20200309_153839` | 131 x 2025 | 1924 / 1925 | 1918/1925 | 131/131 | 0/131 | exclusions 35–45, 160–170 s |
| `BrainScanner20200309_162140` | 134 x 1924 | — / 1924 | 1861/1924 | 134/134 | 0/134 | exclusions 0–10, 300–310 s |

AML310/AKS dataset log의 두 번째 정수는 공식 README가 BFP laser를 켜 AVA를
식별한 volume으로 설명하며 그 뒤를 behavior 분석에서 제외한다. AML32의
두 recording에도 두 번째 정수가 있으나 README는 그 원인을 설명하지 않는다.
따라서 AML32에서는 공식 cut으로만 적용하고 BFP나 다른 사건으로 해석하지
않는다. exclusion interval은 공식 `get_all_recordings.py`에 있는 값이다.

위 11개 기록은 같은-recording output-Fisher
`G^{o\leftarrow A}`에 필요한 neural time series, future behavior output,
clock 및 recording provenance를 가진다는 의미에서 `PASS_G_INPUT`이다. 여기서
허용되는 source population은 manifest에 적힌 해당 recording의 GCaMP rows
전체뿐이다. chart, normalizer, output horizon 및 held-out contiguous block은
결과를 열기 전 후속 계약에서 고정해야 한다.

## 6. GFP control recording

| recording | N x T | finite XYZ | 허용 용도 |
|---|---:|---:|---|
| `BrainScanner20200116_145254` | 114 x 3656 | 0/114 | GFP nuisance control |
| `BrainScanner20200116_152636` | 129 x 3681 | 0/129 | GFP nuisance control |
| `BrainScanner20200204_102136` | 122 x 2531 | 0/122 | GFP nuisance control |
| `BrainScanner20200310_153952` | 116 x 1966 | 0/116 | GFP nuisance control |
| `BrainScanner20200311_100140` | 105 x 2039 | 0/105 | GFP nuisance control |
| `BrainScanner20200929_140030` | 117 x 2188 | 0/117 | GFP nuisance control |
| `BrainScanner20200929_143439` | 137 x 3223 | 0/137 | GFP nuisance control |
| `BrainScanner20210503_122703` | 131 x 2190 | 131/131 | GFP nuisance/spatial control |
| `BrainScanner20210503_135244` | 139 x 2521 | 0/139 | GFP nuisance control |
| `BrainScanner20210503_151831` | 129 x 2144 | 0/129 | GFP nuisance control |
| `BrainScanner20210503_154404` | 126 x 2310 | 0/126 | GFP nuisance control |

## 7. neuron identity와 anatomy 경계

동일한 `heatDataMS.mat`에서 모든 neural 행렬은 같은 `N`과 row ordering을
공유한다. 공식 README는 이 파일들을 neuron-registration pipeline의
산출물로 설명한다. AKS297.51의 현재 추출 `pointStatsNew.mat`에는 volume별
`straightPoints`, `rawPoints`, `pointIdx`, `trackIdx`, `trackWeights` 등이 있어
recording 내부 tracking provenance를 더 직접 확인할 수 있다. 이 근거로
same-record row identity는 `PASS_WITHIN_RECORDING_IDENTITY`다.

그러나 recording 사이 canonical cell ID, atlas neuron name 또는 별도
connectome node ID map은 없다. `XYZcoord`도 recording-local이고 11개 GCaMP
기록 중 다섯 개에서만 전 행이 finite다. 따라서 다음은 모두 차단된다.

- cross-recording same-neuron 결합: `BLOCKED_CROSS_RECORDING_IDENTITY`
- OpenWorm 구조 connectome과 row-by-row 결합: `BLOCKED_CONNECTOME_ALIGNMENT`
- `acorr` 또는 `cgIdx`를 구조 edge나 해부학적 route로 해석:
  `BLOCKED_DERIVED_PROXY`

공식 `figures/fig2/aggregate_AVA.py`는 네 AKS/AML310 기록 모두에 AVA row를
지정한다. 이것은 source-A 후보 provenance의 일부지만,
`BrainScanner20200130_105254`에는 한 row만 지정되고, 어느 기록에도 검증된
target-B neural population이 없다.

## 8. 주장별 최종 판정

| 요구/주장 | 판정 | 근거와 claim ceiling |
|---|---|---|
| archive와 recording provenance | `PASS` | 세 official archive, byte hash, 내부 dataset log와 22 recording directory 확인 |
| MAT schema parse | `PASS` | 22/22 MATLAB v5, 공통 변수와 정확한 shape 확인 |
| same-record neural↔behavior clock | `PASS_WITH_GUARD` | behavior 길이는 T와 일치; 10개 첫 timestamp 중복과 한 recording의 gap 3개는 고정 선두 guard·gap-crossing window 제거 필요 |
| GCaMP primary neural signal | `PASS` | AKS/AML310 4 + AML32 7; GFP 11은 control로 격리 |
| 같은-recording output-Fisher G 입력 | `PASS_INPUT` | 11 GCaMP 기록에서 neural chart와 future behavior likelihood를 분리 추정할 입력 존재 |
| native task context/intervention/trial ID | `BLOCKED` | locomotion 변수와 raw ethogram code만 있고 randomized task context·trial provenance 없음 |
| experimental/task-context `Xi` | `BLOCKED_CONTEXT` | 같은 session에서 검증된 experimental context label 없음 |
| observational locomotor-regime stratification | `DIAGNOSTIC_ONLY` | pre-t 행동 history에서만 정의할 수 있으나 randomized context가 아님 |
| source A provenance | `PARTIAL` | 전체 GCaMP population은 G source로 사용 가능; AVA row는 네 AKS 기록에만 일부 식별 |
| target B provenance | `BLOCKED_TARGET_B` | 검증된 neural target population/area label 없음 |
| same-record `R^{A\to B}` | `BLOCKED_SOURCE_TARGET_DEFINITION` | 임의 row·좌표·cluster partition 없이 A와 B를 동시에 고정할 수 없음 |
| structural/causal routing | `BLOCKED` | 구조 W, perturbation, target-B 및 intervention 없음 |
| physical Riemannian/anatomical metric | `BLOCKED` | XYZ는 일부 recording-local 좌표일 뿐 물리 metric/구조 field가 아님 |

따라서 현재 로컬 입력으로 바로 열 수 있는 가장 강한 경험 경로는 11개 GCaMP
recording 각각에서 미래 locomotion output에 상대적인 조건부 Fisher geometry를
held-out time block으로 추정하고 11개 GFP recording을 nuisance control로 쓰는
것이다. A→B predictive routing, context modulation, structural connectome 또는
causal mediation은 새로운 검증된 label이나 입력이 생기기 전에는 실행하지
않는다.
