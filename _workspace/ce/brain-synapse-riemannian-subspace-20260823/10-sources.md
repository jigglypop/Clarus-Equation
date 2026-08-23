# 10-sources — BA-SRM1 실제 시냅스 자료·측정식 감사

Status: COMPLETE

Source gate: `PASS_INPUT_SCHEMA / PIPELINE_SEPARATED / ROW_LEVEL_EVENTS_UNAVAILABLE`

Outcome contact: 없음. 이 단계는 판본·열·단위·결측 수·pipeline 정의만 열었다.

Claim mapping: `BA-SRM1-C1`, `BA-SRM1-C2`, `BA-SRM1-C5`의 source 지위를 감사한다.

## 1. 1차 출처와 판본

- 공식 AWS Open Data registry: <https://registry.opendata.aws/allen-synphys/>
- 공식 접근 코드: <https://github.com/AllenInstitute/aisynphys>
- 공식 database API: <https://aisynphys.readthedocs.io/en/current-release/api_database.html>
- 공식 release manifest:
  <https://raw.githubusercontent.com/AllenInstitute/aisynphys/download_urls/download_urls.json>
- 1차 논문: Campagnola et al., *Science* 375 (2022),
  DOI <https://doi.org/10.1126/science.abj5861>
- 분석 코드 commit: `545a990ee171e6c0d23dd4bba413e1ccbf2f0853`

이 commit의 schema version은 22이며 official manifest가 지원하는 최신 세 tier는
`synphys_r2.1_{small,medium,full}.sqlite`다. 문서의 r1.0 예시 파일명을 current로
간주하지 않았다. 이번 판본은 관계형 summary용 `small`만 사용한다.

## 2. 원자료 receipt

| 항목 | 값 |
|---|---|
| URL | `https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_small.sqlite` |
| bytes | 176,771,072 |
| SHA-256 | `7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53` |
| MD5 | `5254e04f9b6d121f69795d4e38580111` |
| HTTP Last-Modified | 2023-01-26 02:29:36 GMT |
| ETag | `46d336fbe4f8200e08d40e64c3dd5093-22` |
| SQLite integrity | `ok`, 32 tables |
| DB metadata | schema/db version 22, creation date 2021-12-21 |

첫 range-resume 시도는 timeout 뒤 자식 전송과 후속 전송이 잠시 겹쳐 서버 길이보다
327,680 bytes 큰 파일을 만들었다. 이를 `corrupt-resume-overlap`으로 격리하고
분석 입력에서 제외했다. 단일 재다운로드 검증본만 정식 파일명으로 승격했다.

기계 receipt는 `artifacts/realdata/download-receipt.json`, manifest snapshot은
`download-urls.json`, HTTP snapshot은 `http-head.txt`다.

## 3. “시냅스 세기”의 층별 출처 지위

| 인자 | 실제 자료 지위 | 이번 식의 처리 |
|---|---|---|
| 연결 존재 | manual chemical-synapse call | 이산 stratum; smooth 좌표 아님 |
| E/I sign | `synapse.synapse_type` | ex/in 별도 분석 |
| 휴지막 PSP 세기 | current-clamp response fit, volt | strict chart의 `log(abs(amplitude)/reference)` |
| soma 간 거리 | `pair.distance`, metre | strict chart; axon 길이·전파거리로 해석 금지 |
| postsynaptic input resistance | `intrinsic.input_resistance`, SI resistance | strict chart |
| postsynaptic membrane tau | `intrinsic.tau`, second | strict chart |
| latency | spike max-slope부터 response onset, second | shared-pulse diagnostic; directed Riemann distance 아님 |
| rise/decay | averaged PSP fit, second | shared-pulse diagnostic |
| STP late-pulse amplitude | pulse 2, 6--8, recovery 9--12 summary | target vector |
| STP variability | noise-corrected log variability | target vector |
| conductance | 별도 pipeline의 reversal/holding model 추정치 | 직접 관측으로 부르지 않음; v1 제외 |
| $N,p,q$ | pair별 직접 식별 안 됨 | latent; v1 제외 |
| TM $U,\tau_D,\tau_F$ | source summary와 동일하지 않음 | 역추정 금지 |
| eligibility/homeostasis/longitudinal $\Delta W$ | 없음 | v1 제외 |
| PSD/ASI/contact count | Allen paired DB와 joint frame 없음 | MICrONS/de Vivo 후속 계약 |

PSP와 PSC는 각각 volt와 ampere이므로 섞지 않는다. v1 primary는 current-clamp PSP다.
conductance는 voltage/current, reversal potential, holding/access model을 통해 얻는
추정량이지 native direct measurement가 아니다.

## 4. pipeline 기반 측정 분리

공식 `resting_state.py`와 multipatch resting-state pipeline은
`StimPulse.previous_pulse_dt > 8.0 s`인 QC-passed response만 평균해
`synapse.psp_amplitude`를 갱신한다. 반면 `dynamics.py`의 target은 다음이다.

- `pulse_amp_stp_initial_50hz`: 50 Hz pulse 2 진폭 중앙값;
- `pulse_amp_stp_induction_50hz`: pulse 6--8 진폭 중앙값;
- `pulse_amp_stp_recovery_250ms`: 250 ms 회복 뒤 pulse 9--12 중앙값;
- `variability_stp_induced_state_50hz`: pulse 5--8의 noise-corrected log variability.

네 항목은 각각 scalar summary 하나이며 target은 4차원이다.

따라서 휴지막 진폭과 target pulse 위치는 pipeline 정의상 분리된다. 다만 small DB는
`resting_state_fit.ic_pulse_ids` blob을 2,318개 nonempty row에서 보존하면서도
`pulse_response`, `pulse_response_fit`, `stim_pulse` 행은 모두 0개다. row별 ID
교집합은 이 tier에서 재계산할 수 없다. 판정은
`PIPELINE_SEPARATED / ROW_LEVEL_UNVERIFIED`다.

반대로 latency·rise·decay는 `get_pair_avg_fits(..., max_ind_freq=50)`가 pulse train
위치와 무관하게 QC-passed response를 평균해 만든다. late-pulse target과 원자료를
공유하므로 strict chart에서 제외했다. 이것이 기존 amplitude+latency+kinetics
초안의 실제 source-level 누수 교정이다.

## 5. outcome-free schema/support 결과

필수 열은 전부 존재한다. 전체 strict complete case는 978 pair/511 slice다.
primary mouse V1 두 project를 합치고 E/I를 분리하면 다음과 같다.

| stratum | pair | slice | train pair/slice | dev pair/slice | confirmation pair/slice |
|---|---:|---:|---:|---:|---:|
| ex | 246 | 160 | 159 / 103 | 39 / 27 | 48 / 30 |
| in | 343 | 199 | 222 / 125 | 59 / 37 | 62 / 37 |

이는 계약의 최소 80 pair/20 slice 및 split별 10/5/5 slice gate를 통과한다.
donor ID가 공개 schema에 없으므로 slice split은 donor split이 아니며 population
generalization을 지지하지 않는다.

재현 명령:

```powershell
.codex\hooks\python.cmd python _workspace\ce\brain-synapse-riemannian-subspace-20260823\schema_audit.py data\external\allen-synphys\raw\synphys_r2.1_small.sqlite
```

## 6. 접근·라이선스 경계

AWS registry는 public bucket과 `--no-sign-request` 명령을 제시하지만 이 환경의
bucket listing은 `AccessDenied`였다. official manifest가 준 exact HTTPS object는
성공했고 hash와 SQLite 무결성을 검증했다. listing 실패를 데이터 부재로
재해석하지 않는다.

registry의 dataset terms와 repository의 Allen Institute Software License는 서로
다른 대상이다. 원자료는 git에 넣지 않고 local ignored path에 유지하며, 코드와
데이터 재배포 조건을 별도로 기록한다.

## 7. source 결론

실제 데이터는 단일 $W$보다 풍부하지만 모든 생물 인자를 한 번에 식별하지 않는다.
이번에 source-locked된 엄격 부분공간은 휴지막 PSP 세기, soma 거리,
postsynaptic input resistance, membrane tau의 네 좌표뿐이다. latency·kinetics는
공유-summary 진단, STP pulse summary는 target, conductance/$Npq$/장기 가소성은
미관측 또는 model-derived로 남긴다.
