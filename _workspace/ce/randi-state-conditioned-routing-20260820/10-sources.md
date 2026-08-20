# 출처 레인: Randi 상태-조건부 라우팅 입력

Status: COMPLETE

접근일: 2026-08-20

## 1차 출처와 동결 객체

**[공리: 외부 입력]** Randi et al., “Neural signal propagation atlas of
Caenorhabditis elegans,” *Nature* 623, 406–414 (2023),
DOI `10.1038/s41586-023-06683-4`는 표적 2광자 자극과 동시 전뇌 calcium
imaging으로 23,433 neuron pair의 신호 전파를 측정했다고 보고한다. 논문 방법은
대부분의 자극 source를 약 30초마다 현재 영상에서 찾은 뉴런 중 무작위로 선택했고,
RID·AFD 같은 일부 dim neuron은 별도 recording에서 수동으로 선택했다고 밝힌다.
NeuroPAL identity는 recording 뒤 colour·position·size를 atlas와 비교해 수동 배정했다.

- 논문: https://www.nature.com/articles/s41586-023-06683-4
- DANDI: https://doi.org/10.48324/dandi.001075/0.240920.1434
- DANDI manifest: https://dandiarchive.s3.amazonaws.com/dandisets/001075/0.240920.1434/assets.yaml
- publication-native OSF: https://doi.org/10.17605/OSF.IO/E2SYT
- 분석 코드: https://github.com/leiferlab/pumpprobe
- NWB 변환 코드: https://github.com/catalystneuro/leifer_lab_to_nwb

**[산출]** 선행 manifest가 고정한 DANDI `001075`, version `0.240920.1434`에는
113 subjects, 223 assets가 있다. segmentation asset은 110개, 합계 893,457,040
bytes다. 이번 run은 결과를 보기 전에 정한 최소 크기 표본 하나만 다시 받았다.

| 필드 | 값 |
|---|---|
| subject | `sub-24` |
| asset UUID | `d076d282-162a-4946-a1c8-68e72b6cce54` |
| bytes | 1,273,970 |
| SHA-256 | `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e` |
| local receipt | `artifacts/sub-24-segmentation.nwb` |

## source identity 경로

**[산출]** 동결 schema inventory에서 event table은 `id`, `power`,
`start_time`, `stop_time`, `stimulus_pattern`, `stimulus_site`, `targets`를 가진다.
세 event의 `targets`는 각각 `OptogeneticStimulusTarget*`를 거쳐
`TargetPlaneSegmentation`의 ROI를 가리킨다. 이 target table의 열은
`depth_in_um`, `id`, `imaging_plane`, `pixel_mask`, `pixel_mask_index`,
`reference_images`뿐이다. NeuroPAL label·identity·confidence 열이나 PumpProbe/
NeuroPAL table reference는 없다.

**[산출]** response 쪽 PumpProbe green/red segmentation에는 `neuropal_ids`가 있고,
그 설명은 PumpProbe ROI와 NeuroPAL ROI의 match를 명시한다. 따라서 response ROI의
정본 identity 경로는 있으나 stimulation target에서 그 identity로 가는 명시적 경로는
없다. 기계 판정은 `BLOCKED_EXPLICIT_SOURCE_JOIN`이다.

**[미완성]** target pixel/depth와 tracked-neuron centroid를 맞추는 방법은 새 geometric
matching algorithm이다. 논문·동결 코드에서 registration frame, 단위, 반경,
ambiguity rule과 tie-break를 결과와 독립적으로 복원하기 전에는 schema fact로 쓸 수
없다.

## 배정과 control

**[공리: 외부 입력]** 논문은 source가 “mostly randomly selected”됐다고 설명하지만, 일부
수동 표적 예외도 함께 보고한다. 이 서술은 event별 배정 영수증이 아니다.

**[산출]** exemplar의 event schema에는 random/manual assignment, stratum, control,
sham, failed stimulation, autoresponse 또는 condition 열이 없다. 따라서 자동 무작위
event와 수동 표적 event를 구분하고 같은 층에서 active-source control의 positivity를
검사할 수 없다. 기계 판정은 `BLOCKED_ASSIGNMENT_RECEIPT`다.

**[미완성]** exemplar 하나에서 열이 없다는 사실은 전체 OSF native object에 필드가
없다는 보편 명제가 아니다. publication-native `Fconn` bytes와 checksum 또는 공식
compact manifest가 확보되면 source identity, event order, failure와 assignment를
다시 감사할 수 있다. code semantics만으로 source bytes를 대신하지 않는다.

## 출처 레인 판정

**[산출]** 현재 판정은 `BLOCKED_SOURCE_JOIN + BLOCKED_ASSIGNMENT`다. response
matrix를 읽거나 효과를 계산하지 않았다. 후속 양성 결과의 주장 상한도 고정된
immobilized *C. elegans* 장치에서 source-targeting policy가 calcium response를
바꾸는지에 한정한다. direct synaptic edge, endogenous routing, 행동, 기억, AGI,
포유류 의식 또는 metric mediation을 뜻하지 않는다.
