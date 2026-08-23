# BA-SRM3 source lock

Date: 2026-08-23

Status: `SOURCE_LOCKED / TRAIN_SUMMARIES_DISCLOSED / DEV_CONFIRMATION_UNTOUCHED`

## 고정 입력

- DB: `data/external/allen-synphys/raw/synphys_r2.1_medium.sqlite`
- bytes: `11,125,997,568`
- SHA-256: `dbf19786f9e0d0d73c26351dc29d69ef8c10a2e67e32e19ac73034a5624d48c5`
- integrity/schema receipt:
  `../brain-synapse-functional-quotient-20260823/artifacts/medium-schema-receipt.v3.json`
- schema receipt SHA-256:
  `a94c940e1426d9968bcb48ac20343c00b106d944d4620d55ccad0d7106eb9cc0`
- frozen train manifest:
  `../brain-synapse-functional-quotient-20260823/artifacts/train-sequence-manifest.v2.jsonl`
- manifest SHA-256:
  `4ddb4a52294a55b011c5118a02432ca28c057ca5b5ebb63d8d7c945923aa62c2`
- imported support helper SHA-256:
  `d0f521a48c22f532cbdd0ff808d70647da627c585c84af2a4d8addff1b941a0d`

## QC provenance

Pinned local source `data/external/allen-synphys/aisynphys/aisynphys/`에서 다음을 고정한다.

1. `database/schema/dataset.py`: `StimPulse.qc_pass`는 generic stimulus QC field다.
2. `pipeline/multipatch/dataset.py`: multipatch `StimPulse` 생성 시 `qc_pass`를 넣지 않는다.
3. `pipeline/opto/opto_dataset.py`: opto pipeline만 이를 별도로 계산한다. opto provenance를
   multipatch에 옮기지 않는다.
4. `qc.py::pulse_response_qc_pass`: postsynaptic recording, presynaptic spike 존재,
   인접 pulse, noise/artifact, sign별 holding potential로 `ex_qc_pass/in_qc_pass`를 만든다.
5. `pipeline/multipatch/dataset.py`: 위 두 response QC 값을 `PulseResponse`에 기록한다.
6. `dynamics.py::pulse_response_query(qc_pass=True)`와 `stim_sorted_pulse_amp`: synapse type과
   맞는 `PulseResponse` QC만 사용하며 `StimPulse.qc_pass`를 filter하지 않는다.

따라서 BA-SRM3의 sign-matched response QC는 source-defined input이고, BA-SRM2의 strict
stimulus-QC gate는 별도 STOP 결과로 남는다.

## 데이터 접촉 경계

- schema/integrity와 structural metadata: 읽음;
- frozen train 3,000 sequence의 fit/QC: BA-SRM2 support에서 읽음;
- waveform BLOB: 읽지 않음;
- development fit/QC/outcome: 읽지 않음;
- confirmation fit/QC/outcome: 읽지 않음.

이 문서를 쓸 때 알려진 train 집계는 contract에 전부 공개했다. response-QC와 complete-target의
교집합 및 target scale은 BA-SRM3 첫 support 실행 전에는 미지다.
