# CE 반례-구조 피벗 하네스

이 하네스는 CE 연구에서 후보 식의 반례를 연구 목표의 자동 축소로 오인하지 않게
한다. 정확히 틀린 식은 즉시 비활성화하되, 같은 run 안에서 실패 원인을 바꾸는
구조적 경로를 열어 목표를 계속 검증한다. 반례를 무시하거나 보존법칙을 희생해
겉보기 성공을 만드는 규칙이 아니다.

## 1. 두 층을 먼저 분리한다

- **연구 목표**: 설명하거나 유도하려는 물리적 대상과 성공 판정. 한 후보 식보다
  상위에 있으며 `00-contract.md`에 고정한다.
- **후보 식/경로**: 목표를 구현하려는 작용, 상태변수, 경계조건, coarse-graining,
  readout의 한 조합. `12-routes.md`에서 `Route-ID: ...`로 식별한다.

한 후보 식의 완전한 반례는 그 식과 그 식에 의존하는 하위 주장만 거짓으로 만든다.
그 사실만으로 연구 목표나 서로 독립인 후보 클래스까지 거짓이 되지는 않는다.

## 2. 반례 인증서와 음성대조군

구조적 피벗 전에 `artifacts/negative-controls/<route>-<failure>.md`를 만들고 다음을
고정한다.

1. 실패한 식과 최소 가정 집합
2. 반례의 입력 또는 초기·경계조건
3. 처음 실패하는 정확한 계산 줄
4. 실패 유형: 차원, 부호, 보존, 정규화, 초기값 문제, 안정성, 식별 가능성,
   관측 불일치 중 해당 항목
5. 반례가 제거하는 주장과 제거하지 않는 상위 목표
6. 이후 경로가 같은 실패를 되풀이하면 잡아낼 회귀 조건

코어가 빈 인증서나 이름만 바꾼 경로를 거부할 수 있도록 다음 기계 판독 필드를
각각 한 줄로 쓴다. 필드 이름은 명령 규약이므로 그대로 두되 값과 설명은 한국어로
작성한다.

    Objective-ID: <부모 목표 식별자>
    Failure-Equation: <실패식 또는 식 번호>
    Minimal-Assumptions: <반례에 필요한 최소 전제>
    Counterexample: <입력·초기조건·반례 증인>
    First-Failing-Line: <처음 무너지는 계산 줄>
    Failure-Type: <차원|부호|보존|정규화|초기값|안정성|식별성|관측>
    Removed-Claims: <삭제·비활성화할 하위 주장>
    Preserved-Objective: <반례가 제거하지 않은 부모 목표>
    Regression-Test: <재도입을 잡는 검사>

실패식을 이름만 바꾸거나 숨기지 않는다. 원장과 정본에서는 비활성/반례 지위로
남기고 새 식과 구분한다.

## 3. 구조적 피벗 게이트

다음 경로는 실패 원인의 자유도를 실제로 바꿀 때만 새 route다.

1. **작용·상태 경로**: 동역학 변수, 작용 항, 제약식 또는 대칭 구조를 바꾼다.
2. **경계·원천 경로**: Cauchy 자료, 경계 작용, source/reservoir와 총 보존
   bookkeeping을 바꾼다.
3. **미시·거시 경로**: open-system, coarse-graining, influence functional 또는
   미시 모형과 유효 계수의 matching을 바꾼다.
4. **관측·readout 경로**: 같은 숨은 상태를 관측량으로 보내는 사상과 식별 가능성
   조건을 바꾼다.

계수 미세조정, 변수명 교체, 같은 식의 차수만 늘리기, 반례 입력 제외는 구조적
피벗이 아니다. 새 경로는 결과를 보기 전에 `12-routes.md`에 다음을 적는다.

- `Route-ID: <id>`
- 바뀐 구조와 그대로 유지되는 목표 불변량
- 새 자유도와 parameter accounting
- 총 보존식·차원·안정성 조건
- 선행 음성대조군을 통과해야 하는 이유와 새 falsifier

그 뒤 다음 명령으로 같은 run의 route를 전환한다.

새 route 절에는 아래 필드를 모두 한 줄씩 두며, `Objective-ID`는 음성대조군과
같아야 한다. `Structural-Class`는 허용한 네 구조 클래스 중 하나여야 하고,
`Prior-Negative-Control`은 실제 인증서 상대경로와 정확히 같아야 한다.

    Route-ID: <id>
    Objective-ID: <부모 목표 식별자>
    Structural-Class: <action-state|boundary-source|micro-macro|observable-readout>
    Changed-Structure: <실제로 바뀐 자유도·작용·경계·사상>
    Preserved-Objective: <그대로 유지한 목표>
    New-Degrees-of-Freedom: <추가·제거한 자유도>
    Parameter-Accounting: <새 매개변수와 입력/산출 구분>
    Conservation-Law: <총 보존 장부>
    Dimension-Check: <모든 새 항의 차원>
    Stability-Condition: <양성·안정성 조건>
    Prior-Negative-Control: artifacts/negative-controls/<file>.md
    Falsifier: <새 경로의 정확한 반증 조건>

    .codex/hooks/run.cmd pivot <run-dir> <route-id> artifacts/negative-controls/<file>.md

`revise` 한도는 route와 역할의 쌍마다 적용된다. 한도가 찼다는 이유만으로 별도
run을 만들지 않는다.

## 4. 목표 축소와 BLOCKED의 높은 문턱

연구 목표의 축소·기각은 다음 중 하나가 있을 때만 허용한다.

- 허용할 모델 클래스를 결과 전에 명시했고, 그 클래스들을 덮는 objective-level
  no-go 정리 또는 완전한 반례가 있다.
- 목표를 열기 위해 반드시 필요한 가정이 계약·관측·사용자 제약과 모순된다.
- 필요한 정보가 원리적으로 식별 불가능함을 증명했다.

그 전에는 틀린 후보만 제거하고 목표는 `OPEN`으로 둔다. `BLOCKED`를 쓰면 시도한
구조 클래스, 각 기각 근거, 아직 시도하지 못한 클래스와 재개 조건을 함께 적는다.
거짓 식을 살리는 것과 연구 목표를 계속 여는 것은 별개의 일이다.
