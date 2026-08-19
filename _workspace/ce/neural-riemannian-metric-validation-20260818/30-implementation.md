# 신경 계량 후보 전수 구현

Status: COMPLETE

## 구현 범위

`artifacts/e17_candidate_tournament.py`는 원장에 고정된 27개 ID를 타입별로 처리한다. E17에서 입력이 있는 상태 SPD, finite-time deformation, condition-information field, graph metric/quasi-metric, discrete directed action과 Wasserstein 후보는 등록된 grid를 전부 평가한다. $B,R_u,Q_x$, calibrated SDE, identifiable one-form, nonquadratic potential 또는 smooth metric field가 필요한 식은 proxy를 대입하지 않고 `UNTESTABLE_MISSING_INPUT`으로 유지한다.

이번 구현의 일반화 목표는 엄격한 cross-animal metric transport가 아니다. 각 held-out session의 fit block에서 chart, $J,Q$, scalar calibration과 decoder를 맞추고, outer-train animal의 inner block으로 hyperparameter만 선택한 뒤 held-out test에서 선택 tuple 하나를 평가하는 `LOAO_HYPERPARAMETER_GENERALIZATION_WITH_HELDOUT_SESSION_CALIBRATION`이다. E17은 이미 열린 자료이므로 모든 출력은 retrospective discovery다.

## V1 무효화와 V2.2

첫 전수 출력은 보존하되 추론에서 무효화했다. `S7-H,H=1`은 feature와 target이 같은 완전한 항등식이었고, `S9,lambda_G=0`의 rank-one field 하나가 $2.8\times10^{-17}$ 수준의 수치 고유값 때문에 SPD로 잘못 통과했다. 또한 `S8/S9` field가 held-out test probability를 훑었고 freeze가 입력 byte와 tuple 완전성을 실행 전에 강제하지 못했다. 무효 사유와 원본 해시는 `artifacts/e17-candidate-tournament-v1-invalidated.json`에 고정했다.

교정된 V2.2는 다음을 강제한다.

1. `S7-H,H=1`의 88개 session-condition-ridge tuple을 모두 `INELIGIBLE_TAUTOLOGY`로 기록하고 외부 순위에서 제외한다.
2. `S8/S9`는 session fit block에서만 field를 만들며, $r>1$, $\lambda_G=0$을 부동소수 고유값과 무관하게 대수적으로 `INELIGIBLE_SINGULAR`로 판정한다. 두 field는 독립 predictive endpoint가 아닌 fit-only gate다.
3. `artifacts/e17-candidate-tournament-freeze-v2.2.json`의 runner, validator, 원장, fixture와 11개 MAT SHA-256이 맞아야 자료를 읽는다.
4. uncertainty, deformation, decoder/field, graph, action, distribution의 모든 session-condition-horizon cell에서 예상 tuple key와 실제 key를 대조한다. 누락, 추가 또는 중복이 하나라도 있으면 결과 파일을 쓰지 않는다.
5. 결과 파일은 OS exclusive-create mode로만 열며 overwrite option이 없다. 비유한 수는 `null`로 바꾸지 않고 직렬화 전에 실패한다. Python/NumPy/SciPy/platform/BLAS 정보를 결과에 기록한다.
6. raw tuple 시도, outer tuple 선택, 실제 outer 평가를 별도 필드로 기록한다. 따라서 graph 후보처럼 local tuple은 계산됐지만 공통 LOAO tuple이 없는 경우를 승자로 오독할 수 없다.

## 산출물

- 최종 runner: `artifacts/e17_candidate_tournament.py`, SHA-256 `0e7b739d667015bce4b77f52642c636f61a1006b4140a1fec61f776dd692dbc6`.
- 사전 freeze: `artifacts/e17-candidate-tournament-freeze-v2.2.json`, SHA-256 `114ad4b65418d6650a22ab2f7d3961cc69df44dd76ef7655b3d2dba3be59eb06`.
- 최종 결과: `artifacts/e17-candidate-tournament-results-v2.2.json`, SHA-256 `fff9e93c1711341a5a77a5ba4f15996535279fccc687b3bf0fadd9ed7a4b9271`.
- 사후 result lock: `artifacts/e17-candidate-tournament-result-lock-v2.2.json`.
- 표준 라이브러리 validator: `artifacts/validate_e17_candidate_tournament.py`.
- 수학 fixture: `artifacts/math/candidate_math_fixture.py`와 `candidate_math_fixture_output_v2.2.json`.
- write-policy 용어 보정: `artifacts/e17-candidate-tournament-v2.2-wording-clarification.json`.

V2는 Python boolean을 JSON number `1/0`으로 쓴 schema 결함 때문에 교체됐다. V2.1은 boolean을 고쳤지만 exclusive-create 경합, nonfinite-to-null 변환과 제한된 사후 validator를 강화하기 위해 V2.2로 교체됐다. V2.1과 V2.2의 과학적 payload는 byte-parsed object 비교에서 완전히 동일하며 각 이전 파일은 불변 해시와 supersession ledger로 보존했다.

V2.2 freeze의 `atomic exclusive-create`는 경로 예약과 overwrite 방지가 open 시점에 원자적이라는 뜻으로만 읽는다. 임시 파일 후 rename하는 crash-safe publication은 아니며, 중간 crash로 partial file이 남으면 post-run result lock과 validator가 실패한다. 현재 V2.2는 완전한 파일의 hash lock과 validator `PASS`를 모두 받았다.
