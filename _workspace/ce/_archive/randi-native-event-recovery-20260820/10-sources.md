# 출처 레인: publication-native event 입력

Status: COMPLETE

접근일: 2026-08-20

## 1차 출처 고정

**[공리: 외부 입력]** Randi et al., “Neural signal propagation atlas of
Caenorhabditis elegans,” *Nature* 623, 406–414 (2023), DOI
`10.1038/s41586-023-06683-4`는 원자료를 OSF `E2SYT`에, 분석 코드를 Zenodo와
GitHub에 공개했다고 명시한다.

| 객체 | immutable identifier | 공식 파일 receipt |
|---|---|---|
| OSF data | `10.17605/OSF.IO/E2SYT` | provider `osfstorage`, project `e2syt` |
| pumpprobe | `10.5281/zenodo.8312985` | `pumpprobe-1.1.zip`, 1,287,278 bytes, MD5 `40d87e790193d38528b4ba0cecf23e8c` |
| wormdatamodel | `10.5281/zenodo.8247252` | `leiferlab/wormdatamodel-v1.0.1.zip`, 198,410 bytes, MD5 `373c21e216e0a9cb628d99dc5590bf67` |
| wormbrain | `10.5281/zenodo.8247254` | `leiferlab/wormbrain-v1.0.1.zip`, 221,394 bytes, MD5 `a9840e7ee7541dbd743140e8bbd23542` |
| acquisition | `10.5281/zenodo.8247258` | 83,716,342-byte archive로 이번 25-MB object ceiling에서 제외 |

**[산출]** 현재 GitHub default-branch HEAD도 별도 provenance 후보로 동결했다.
publication-pinned Zenodo release가 우선이며 GitHub HEAD는 변환·후속 코드 비교에만 쓴다.

| repository | commit | tree |
|---|---|---|
| `leiferlab/pumpprobe` | `1dbc5e0a2b609d54bc9b1c90c73d4e3bf183d3c7` | `9021fea7bfeca1e208d84c6049b4eb99bc2c2841` |
| `leiferlab/wormdatamodel` | `2ab956199e3931de41a190d2b9985e961df3810c` | `ebdb09bf5d5ec8e6a744739fd0b0f605a5500640` |
| `leiferlab/wormbrain` | `d94f7a1456622f77eebcc181de1d8504618764c4` | `96544419d5a953542e47222fc2bff291f5aab6f1` |
| `catalystneuro/leifer_lab_to_nwb` | `3544c9bb59f90d5630fa1871850d990db9cafc18` | `6b806e352be21d79308f3a069bdf12bc06ee2519` |

## OSF provider tree

**[산출]** 공식 OSF API
`https://api.osf.io/v2/nodes/e2syt/files/osfstorage/`는
`raw_extracted_data` folder ID `671a5286badd54a2128707e3`를 제공한다. child endpoint의
pagination metadata는 678 objects를 보고한다. 동결 구현은 server-default page size로
68개 metadata page를 순회했다(별도의 100 objects/page 조회에서는 마지막 page가 7이다).
첫 metadata page에서 다음 publication-native object family를 확인했다.

- `*_ds_name.txt`: recording/session identifier 후보
- `*_labels.txt`: recording-local neuron index-to-label 사전 후보
- `*_stim_neurons.txt`: event별 stimulated-neuron index 후보
- `*_stim_volume_i.txt`: stimulation volume/index metadata 후보
- `*_t.txt`: recording clock 후보
- `*_gcamp.txt`: fluorescence response 후보로 이번 run에서 제외

각 file object는 provider ID, materialized path, byte size, modified time, stable download
URL과 checksum field를 제공한다. full recursive manifest와 실제 checksum 존재율은
구현 레인에서 동결한다. `provider_checksum_present`, 검증 성공과 local download
SHA-256은 서로 다른 필드로 기록하며 checksum 부재를 성공으로 세지 않는다.

## 가능한 native join

**[미완성]** object 이름과 공개 `pumpprobe` schema를 함께 읽으면 다음 join 후보가
생긴다.

$$
(dataset,event,stim\_neurons[event])
\longrightarrow labels[stim\_neurons[event]].
$$

여기에 event order, stimulation volume index와 clock을 붙이면
`(dataset,event,t_stim,A_index,A_label,u_index)` 후보를 구성할 수 있다. 그러나 실제
bytes의 rowwise 값, index domain, missing code와 cardinality가 일치하는지는 아직
검증하지 않았다.

**[미완성]** 파일 family에는 automatic/manual assignment, selection probability/
stratum, failed assignment, identity confidence, registration transform 또는 geometric
ambiguity receipt를 명시하는 별도 object가 보이지 않는다. `stim_neurons` 내부 sentinel이나
publication code가 이를 보존할 가능성은 구현 레인이 outcome을 읽지 않고 검사한다.

## 출처 판정

**[산출]** OSF acquisition blocker는 해소됐다. 공식 tree와 byte download route가 있다.
최종 지위는 `PASS_OFFICIAL_ACQUISITION / BLOCKED_SOURCE_INDEX_JOIN /
BLOCKED_ASSIGNMENT_RECEIPT`다. 허용 bytes가 제공한 부분 결합은
`PARTIAL_SOURCE_INDEX_SCHEMA / OBSERVATIONAL_SCHEMA_ONLY`로만 보존한다. object 이름이나
code schema만으로 `PASS_SOURCE_JOIN` 또는 `PASS_ASSIGNMENT_RECEIPT`를 선언하지 않는다.

## 공식 링크

- https://doi.org/10.1038/s41586-023-06683-4
- https://doi.org/10.17605/OSF.IO/E2SYT
- https://api.osf.io/v2/nodes/e2syt/files/osfstorage/
- https://doi.org/10.5281/zenodo.8312985
- https://doi.org/10.5281/zenodo.8247252
- https://doi.org/10.5281/zenodo.8247254
- https://github.com/catalystneuro/leifer_lab_to_nwb
