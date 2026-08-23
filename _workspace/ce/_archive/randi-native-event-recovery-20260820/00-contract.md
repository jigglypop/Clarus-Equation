# Randi publication-native event 복원 계약

Status: COMPLETE

Scope: 계약 동결 단계의 완료만 뜻한다. source·assignment 입장과 empirical route의
판정은 `20-audit.md`와 `40-final-report.md`가 맡는다.

PREDECESSOR: `_workspace/ce/randi-state-conditioned-routing-20260820`

## 목적

선행 run에서 차단된 두 입력, 즉 stimulation target에서 정본 NeuroPAL identity로
가는 immutable source join과 event-level automatic/manual assignment·control receipt를
publication-native OSF/`Fconn` 객체 또는 공식 변환 코드의 동결 mapping에서 복원한다.
이번 run은 input recovery다. fluorescence response 값, autoresponse 결과, pair effect,
state encoder, endpoint와 neural routing score를 읽거나 계산하지 않는다.

## PREDECESSOR_EVIDENCE

| 선행 증거 | 상태 | SHA-256 | 보존하는 가장 좁은 주장 | 재시도 금지 조건 |
|---|---|---|---|---|
| route ledger | `EMPIRICAL_ROUTE_BLOCKED_CONDITIONAL` | `e2bdb50eadf9e3ac341606fa5cb69a1657f055926121c20b72075e282580ead6` | R3 native event object가 현재 1순위 재개 경로다 | 합성 source label이나 outcome-tuned spatial match로 대체하지 않음 |
| predecessor routes | `R3 OPEN_CONDITIONAL` | `5ccba338201562475b24873a30ee9659472296ca8320f70a1601b6d4eb1392e2` | 공식 bytes/checksum에서 identity·assignment·failure·order를 복원해야 한다 | code semantics만으로 source receipt를 대신하지 않음 |
| predecessor validation | `BLOCKED_EXPLICIT_SOURCE_JOIN / BLOCKED_ASSIGNMENT_RECEIPT` | `efbca90b4ee7be4cec0976c2224f2cf2781dc60c9efd24520e06d447028a71c3` | DANDI exemplar의 response identity schema는 통과했으나 source·assignment가 없다 | response matrix 열람 금지 |
| predecessor final | `SCHEMA_PROBE_COMPLETE / EMPIRICAL_ROUTE_BLOCKED_CONDITIONAL` | `41592eddfa0c288f8e39df7178eb4cc48dd2c043e201ff20bb5d0e692120aaf5` | 실제 효과 계산 전 두 receipt가 모두 필요하다 | 하나만 복원하고 empirical GO로 승격하지 않음 |

## 결과 전 후보 순위

| 순위 | 후보 | 선택 이유 | 독립 falsifier / STOP |
|---:|---|---|---|
| 1 | R3a OSF `E2SYT` recursive metadata tree | publication-native provider가 object path·size·provider identifier를 보존할 가능성이 가장 크다 | 공식 API tree·stable object URL·byte receipt를 만들지 못하면 `BLOCKED_ACQUISITION` |
| 2 | R3b frozen `pumpprobe`/`wormdatamodel` source | `Fconn`의 source identity, event flags, manual target와 exclusion semantics를 정의할 수 있다 | code field가 실제 공개 object에 존재한다는 byte-level 확인이 없으면 schema candidate로만 유지 |
| 3 | R1b frozen `leifer_lab_to_nwb` conversion mapping | native field가 NWB에서 어디로 소실·변환됐는지 식별할 수 있다 | response outcome이나 geometric fit을 사용해 source label을 생성하면 `APPARATUS_INVALID` |
| 4 | R2 validation object | source coordinate-to-NeuroPAL join이 공식 독립 validation rule로 고정됐는지 검사한다 | frame·unit·transform·radius·ambiguity·tie-break 중 하나라도 없으면 퇴역 |

## metadata-first acquisition 규칙

1. OSF project는 DOI `10.17605/OSF.IO/E2SYT`의 공식 API provider tree만 조회한다.
2. 먼저 모든 provider/file/folder의 path, object ID, size, modified time, download URL와
   provider checksum을 가능한 범위에서 manifest로 동결한다.
3. 결과를 열기 전 eligible object를 이름과 크기만으로 선택한다. 우선순위 token은
   `fconn`, `brains`, `labels`, `ids`, `targets`, `events`, `stim`, `flag`, `manual`,
   `metadata`, `config`, 정확한 suffix `t.txt`다. `signal`, `calcium`, `response`, `fit`, `kernel`, `qvalue`,
   `pvalue`, `deltaf`, `dff` object는 이번 run에서 제외한다.
4. 단일 object 최대 크기는 25 MB다. 후보가 여러 개면 token 우선순위, byte size,
   canonical path 순으로 결정한다. archive라면 directory listing만 먼저 검사한다.
5. 공식 GitHub 저장소는 default branch HEAD commit과 recursive tree SHA를 먼저 고정하고
   source 파일만 받는다. release/tag가 있으면 publication DOI가 지시한 tag를 우선한다.
6. provider checksum이 없으면 내려받은 bytes의 SHA-256과 immutable object URL을 함께
   기록한다. source bytes를 확보하지 못하면 code semantics 이상의 주장을 하지 않는다.

### 총 acquisition bound

- OSF에서 선택하는 non-outcome txt는 최대 600 files, 합계 최대 25,000,000 bytes다.
- 허용 family는 `ds_name.txt`, `labels.txt`, `stim_neurons.txt`,
  `stim_volume_i.txt`, `t.txt`뿐이며 family별 최대 120 files, 15,000,000 bytes다.
- Zenodo는 DOI `10.5281/zenodo.8312985`의 `pumpprobe-1.1.zip` 한 개만 허용하며
  최대 2,000,000 bytes다.
- R1b용 GitHub archive는 이미 동결한 converter commit
  `3544c9bb59f90d5630fa1871850d990db9cafc18`의 ZIP 한 개만 허용하며 최대
  5,000,000 bytes다. 전체 다운로드 상한은 602 files, 32,000,000 bytes다.
- manifest에서 cap을 넘으면 해당 bytes를 받지 않고
  `BLOCKED_ACQUISITION_BOUND`로 중지한다. cap 적용 순서는 exclusion token 적용 뒤
  `(family token priority, byte size, canonical path)`다.
- 각 object에 `provider_checksum_present`, `provider_checksum_verified`,
  `download_sha256`을 별도 기록한다. provider checksum이 없으면 verified를 false로
  두고 `UNVERIFIED_PROVIDER_CHECKSUM`으로 표시한다. local SHA-256 계산은 provider
  checksum 검증을 대신하지 않는다.

## 복원해야 할 event receipt

최소 event row는 다음 필드를 outcome과 독립적으로 제공해야 한다.

$$
(animal,session,event,t_{stim},A_{id},A_{confidence},assignment,
manual,failed,u_{stim},order).
$$

여기서 $A_{id}$는 post-response로 선택하지 않은 canonical source identity다.
`assignment`는 automatic random selection과 manual targeted exception을 구분해야 한다.
`failed`는 downstream response나 autoresponse success로 event를 사후 삭제하지 않도록
원래 assigned event를 보존해야 한다. active control과 positivity는 실제 source set과
assignment stratum에서 후속 계약이 별도로 판정한다.

## 허용 probe와 금지

- 허용: filenames, archive member names, source code, config schema, object keys/dtypes/
  shapes, identity strings, event onset/order, source index/label, assignment/manual/failure
  flags, dose metadata와 checksums.
- 금지: fluorescence arrays, autoresponse amplitude, downstream response, fitted kernel,
  responder classification, p/q-value, effect size, state embedding과 outcome-derived exclusion.
- pickle/NumPy object를 열 때는 임의 code execution을 막는 restricted parser 또는
  `allow_pickle=False`가 가능한 형식만 사용한다. 신뢰할 수 없는 pickle을 load하지 않는다.
- 공식 object가 Python pickle뿐이면 pickle opcode·member metadata를 정적으로 검사하고,
  안전한 별도 변환 경로가 없으면 `BLOCKED_UNSAFE_SERIALIZATION`을 보고한다.

## 입장 판정

- `PASS_SOURCE_JOIN`: immutable object에서 $A_{id}$와 confidence/provenance를 복원한다.
- `PASS_ASSIGNMENT_RECEIPT`: automatic/manual, original assigned events, order와 dose를
  outcome 없이 복원한다.
- `PASS_APPARATUS_INPUT`: 위 두 gate를 모두 통과한다. 이 판정도 effect가 아니다.
- `PASS_OBSERVATIONAL_ONLY`: source identity는 있으나 assignment/positivity가 없다.
- `BLOCKED_ACQUISITION`: official tree/bytes/checksum을 확보하지 못한다.
- `BLOCKED_SOURCE_JOIN`: source identity mapping을 복원하지 못한다.
- `BLOCKED_ASSIGNMENT`: automatic/manual 또는 complete assigned-event receipt가 없다.
- `APPARATUS_INVALID`: outcome을 읽거나 결과 기반 matching/exclusion이 필요하다.

## 다음 단계

`PASS_APPARATUS_INPUT`일 때만 별도 empirical 계약을 열어 $M_0$, $M_1$,
$M_{2,\mathrm{add}}$, $M_{2,\mathrm{int}}$와 adverse controls를 실행한다. 이번 run은
그 모델을 fit하지 않는다. 실패하더라도 R3a, R3b, R1b, R2를 순서대로 모두 소진하고
각 경로의 재개 조건을 남긴 뒤 완료한다.

## Revision 1 — R3 뒤 R1b 소진

R3 구현 결과를 response 비열람 상태에서 확인한 뒤에도 계약에 이미 고정한 R1b와 R2가
남았다. R1b를 생략해 R3 실패를 최종화하지 않도록 위 converter archive의 immutable
commit과 별도 5-MB cap을 명시했다. converter source는 ZIP metadata와 UTF-8 text만
정적으로 읽으며 extraction, import, execution과 dependency 설치를 금지한다. native
field-to-NWB mapping, `-1/-2/-3` sentinel, complementary/manual label, target ROI와
assignment/failure field의 보존·유실만 검사한다.

## 실행기 예외

`$ce-research`가 지정한 `C:\Users\dongh\.codex\hooks\run.ps1`가 이 환경에 없으므로
동일한 8개 stage 이름과 수동 gate 검사를 사용한다.
