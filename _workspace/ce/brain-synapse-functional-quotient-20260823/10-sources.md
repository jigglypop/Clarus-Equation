# 10-sources — BA-SRM2 실제 시냅스 함수자료

Status: COMPLETE

## S1 — 잠긴 small release

- DB: `data/external/allen-synphys/raw/synphys_r2.1_small.sqlite`
- bytes: 176,771,072
- SHA-256: `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53`
- schema 22, SQLite integrity `ok`
- official code commit:
  `545a990ee171e6c0d23dd4bba413e1ccbf2f0853`

## S2 — 유효한 protocol summary 구조

공식 `aisynphys/dynamics.py`는 current-clamp pulse response를
`(clamp_mode, induction_frequency, recovery_delay)`로 묶는다. protocol마다
`stp_initial`, `stp_induction`, `stp_recovery`, `stp_recovery_single`을 별도 list에서
계산한다. 이 summary는 각각 `(median,std,n)`이고 protocol surface의 유한 관측으로는
쓸 수 있다.

local object는 JSON이며 `aisynphys/database/database.py`의
`JSONObject.process_result_value`가 `json.loads`로 복호화한다. small DB에는
`stp_all_stimuli` nonempty dynamics row가 2,126개이고 protocol record는 pair당
1--14개다. frequency domain은 10, 20, 25, 50, 100, 200 Hz이고 recovery delay는
125, 250, 500 ms, 1, 2, 4 s다.

## S3 — 기각된 pulse slot

동일 producer는 `collect_pulse_amps = [[]] * 12`를 사용한다. 모든 pulse append가
같은 inner list로 들어가므로 `pulse_amplitudes` 12 slot은 서로 다른 pulse가 아니라
같은 전체-pulse aggregate의 반복이다. 이후 copy/reset은 없다.

train-only structural equality 감사에서 ex 1,324개와 in 2,613개 protocol record
모두 12 slot이 동일했다. apparent inequality는 NaN의 비반사성뿐이었고 NaN을
canonicalize하면 예외는 0이다. 따라서 이 필드는 pulse trajectory로 사용할 수 없다.

## S4 — canonical delay duplicate

measured recovery delay가 알려진 delay에서 5 ms 안이면 producer가 canonical delay로
치환한다. 같은 pair의 서로 다른 실제 delay가 같은 key로 중복될 수 있고 actual delay는
JSON에 남지 않는다. summary를 임의 병합하거나 첫 record를 고르는 것은 금지한다.

## S5 — medium/full 경계

small DB의 `pulse_response`, `pulse_response_fit`, `stim_pulse`는 0행이다.
`util/bake_sqlite.py`의 release rule상 medium은 row-level response/fit/stimulus와
recording 관계를 보존하고 waveform array를 제거한다. full은 waveform도 보존한다.

official multipatch importer `aisynphys/pipeline/multipatch/dataset.py`는 stimulus pulse를
`StimPulse(recording=rec_entry, ...)`로 만들지만 `cell` relation은 설정하지 않는다.
반면 `PulseResponse`는 `stim_pulse=all_pulse_entries[pre_dev][pulse_n]`와 ordered
`pair_entry(pre_dev, post_dev)`를 함께 저장한다. 따라서 medium multipatch에서
`stim_pulse.cell_id` NULL은 source pipeline의 허용 상태이고, presynaptic identity는
pre-recording electrode와 `pair.pre_cell` electrode의 일치로 감사해야 한다.

공식 medium URL의 HEAD receipt:

| 필드 | 값 |
|---|---|
| URL | `https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_medium.sqlite` |
| Content-Length | 11,125,997,568 bytes |
| Last-Modified | 2023-01-26 02:25:26 GMT |
| ETag | `d954cbad0d7c7b0002bf3a2879e40e90-1327` |

medium acquisition은 2026-08-23 사용자 승인 뒤 진행 중이다. partial object는 분석
입력이 아니며 완료 전 SHA-256, integrity와 event support는 계속
`UNVERIFIED_INPUT`이다. multipart ETag는 MD5나 SHA-256으로 해석하지 않는다.

## S6 — 직접 관측하지 않은 항

small/medium의 event amplitude만으로도 receptor conductance, $Npq$, quantal size,
release-site count, STDP timing curve, eligibility, neuromodulator, homeostasis,
contact/PSD, spine survival, longitudinal turnover와 axon conduction path는 직접
식별되지 않는다. full waveform도 holding/reversal/access-resistance contract 없이
conductance로 승격하지 않는다.

## S7 — 공식 provenance와 이용 경계

- AWS Open Data Registry: `https://registry.opendata.aws/allen-synphys/`
- official database access documentation:
  `https://aisynphys.readthedocs.io/en/current-release/database_access.html`
- official relational dataset structure:
  `https://github.com/AllenInstitute/aisynphys/blob/current-release/doc/source/dataset_structure.rst`
- official tool/source repository: `https://github.com/AllenInstitute/aisynphys`

dataset은 Allen Institute 이용약관, source tool은 repository의 Allen Institute
Software License를 따른다. medium은 관계형 event metadata와 fit 결과 검증용이며 raw
electrophysiology waveform은 포함하지 않는 경계로 취급한다. 정확한 table/column/FK와
row support는 완료 파일의 schema-only audit 전에는 확정하지 않는다.
