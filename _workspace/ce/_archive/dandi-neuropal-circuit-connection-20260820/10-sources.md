# 출처 레인

Status: SOURCE_REVISION_1_FROZEN

- superseded: DANDI 000565 `<300 MB` assets. First downloaded file passed SHA but contained static NeuroPAL only, so `BLOCKED_INPUT` for time-series A3.
- replacement: DANDI 000541 version `0.241009.1457`
- official description: `C. elegans head NeuroPAL and Calcium imaging`
- selected assets: smallest 8/21 by official content size
- full asset bytes if all downloaded: 11,600,186,344
- exact IDs, hashes and S3 URLs: `artifacts/source-manifest-v2.json`

000541 선택 규칙은 그 dataset의 response array를 열기 전에 고정했다. 파일 크기는 과학 결과가 아니라 실행 가능성 기준이다. 가장 작은 한 자산은 전부 다운로드해 SHA와 schema를 고정하고, 나머지는 필요한 HDF5 dataset만 range-read한다.
