# Phase A split contamination ledger

Status: COMPLETE

## 1. 폐기 판정

초기 manifest 후보의 development block `2001`--`2024`는
`ABANDONED_PRE_REGISTRATION_TEST_CONTAMINATION`으로 전부 폐기했다. 초기 focused
test가 아래 범위를 실제로 생성하거나 채점했기 때문이다.

- `2001`--`2008`: `n=4`, `m=2`, `K=3`, train `240`, held-out `96`,
  `sigma=0.05`, ridge `1e-6`, bootstrap `400`인 pilot-style 소형 fixture에서
  R1/R3 NLL과 input-shuffle endpoint를 채점했다. 구현 agent와 root의 독립
  focused 실행을 포함해 같은 fixture가 여러 번 재생됐다.
- `2001`--`2003`: generator replay, learner boundary와 arm accounting test에서도
  생성됐다.
- role digest test는 초기 block의 일부 정수 namespace를 호출했다.

이는 공식 development도 scored confirmation도 아니다. 초기 후보 block으로
manifest 설정인 train `480`, held-out `192`, bootstrap `4000`을 실행한 적은 없고,
one-shot result 또는 confirmation result도 만들지 않았다. 하지만 일부 endpoint를
이미 보았으므로 오염되지 않은 나머지만 재사용하지 않고 초기 block 전체를
폐기했다.

## 2. rotation-2 선택 규칙

새 development block은 결과와 무관한 다음 domain 문자열로 결정했다.

```text
CE-PHASE-A-V1|DEVELOPMENT-ROTATION-2|SEALED-MANIFEST-ONLY|2026-08-16
```

선택 알고리즘은 다음과 같다.

1. 위 UTF-8 bytes의 SHA-256을 계산한다.
2. digest는
   `c308706f75cb59e37f7b0034fa562e6e9eafcdfb8f641be7fd2cafcbd6163108`이다.
3. `start = int(digest[0:7], 16)`으로 둔다.
4. development block은 `start + i`, `i=0,...,23`인 연속 24개 정수다.

raw rotation-2 block은 preregistration manifest에만 기록하며 이 ledger, test,
production module, runner, 명령 기록에는 반복하지 않는다. manifest 삽입 전
repository-wide fixed-string nonuse scan은 occurrence `0`이었고, 삽입 후 manifest를
제외한 같은 scan도 occurrence `0`이었다. focused test는 이후 pilot block
`1001`--`1004`만 사용한다. rotation-2 block의 generator, role namespace,
endpoint와 aggregate는 one-shot 전까지 호출하지 않았다.

## 3. confirmation commitment 경계

split 회전과 함께 confirmation commitment domain도
`CE-PHASE-A-V1-CONFIRMATION-SPLIT-ROTATION-2`로 바꿨다. digest는 결과를 보기
전에 transient secret material에서 생성했지만, raw block의 외부 custody,
custodian 또는 future reveal 증거가 없다. 따라서 이 digest는 실행 가능한 blind
holdout이 아니다.

manifest와 향후 result는 이를 다음처럼 제한한다.

- `reservation_kind: reservation_only`
- `custody_status: custody_unverified`
- `holdout_status: not_executable_holdout`
- `execution_authorized: false`
- `status: reserved_unopened`

새 raw confirmation seed를 복원하거나 생성하는 코드 경로는 없다. 실제
confirmation을 하려면 별도 run에서 custody가 검증된 새 holdout protocol을 먼저
사전등록해야 한다.

## 4. 판정 경계

rotation은 성능이 나쁜 seed를 버리고 더 좋은 seed를 고른 행위가 아니다. 폐기
사유는 focused test가 초기 development 역할을 실제로 소비했다는 protocol 위반이고,
새 block은 endpoint를 계산하기 전에 hash-derived 규칙과 repository nonuse scan으로
고정했다. 초기 소형 결과는 Phase A H1/H2의 증거나 반증으로 보고하지 않는다.
