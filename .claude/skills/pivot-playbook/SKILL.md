---
name: pivot-playbook
description: 반례가 발견됐을 때, sourcer가 문헌 재발견(identical·special_case)을 보고했을 때, 같은 경로에서 3회 실패했을 때, judge가 verdict pivot을 내렸을 때 다음 경로를 고르기 위해 반드시 참조. 반례는 축소 신호, 재발견은 확장 신호다. prover(후보·추측 모드)와 judge가 쓴다.
---

# 피벗 플레이북 (확장 2 + 축소 4)

방향이 둘이다. **확장**은 더 강한 식을 세우고, **축소**는 같은 식을 좁힌다. 옛 플레이북은 축소만
있어서 네 번 좁히면 문헌에 있는 결과가 남았다. 어느 방향인지는 신호가 정한다.

| 신호 | 방향 | 첫 단계 |
|---|---|---|
| sourcer identical·special_case (재발견) | 확장 | `conjecture` |
| 유도는 닫혔는데 CE 고유 내용이 없음 | 확장 | `generalize` |
| 반례가 특정 영역에서만 | 축소 | `partial` |
| 결론은 수치로 맞는데 증명이 안 됨(L2 고착) | 축소 | `alt_derivation` |
| 반례가 정의역 전체에 흩어짐 | 축소 | `reformulate` |
| 축소 1~3 소진, 반례가 늘 같은 숨은 가정 | 축소 | `weaken` |

`questions.yaml`의 `pivots_tried`에 시도한 단계를 쌓는다. 축소 4단계를 다 썼는데 L3 미달이면
`ledger.py`가 `force_pivot: conjecture`를 붙인다(주차가 아니라 확장). 확장 2 + 축소 4가 모두
소진되고도 L3 미달이면 그때 `parked`다.

## 확장 1. `conjecture` — 새 공리 후보 (예측식·예산식 카드)

좁히지 말고 세운다. 현재 질문이 겨냥한 양(비율·장부·분포)을 내놓는 식을 conjecture-first 카드로
선언하고, 숫자와 kill을 사전등록하고, 증명 사다리를 연다.

- 가는 신호: sourcer가 identical·special_case. 세 attempt가 모두 "문헌의 특수 사례"였다(Q-0002).
- prover(모드: 추측)가 낼 것: `derivations/<Q>/F-NN.formula.md`. 후보의 모습은 "문헌 결과 R을
  포함하되 R이 말하지 않는 숫자 하나를 더 내는 식". 반증조건은 카드의 `kill`.
- 금지: 문헌 결과를 CE 기호로 옮겨 적은 카드. adversary `content` 감사와 sourcer가 잡는다.

## 확장 2. `generalize` — 알려진 결과의 CE 고유 일반화

닫힌 보조정리를 가정 하나를 떼거나 정의역을 넓혀 CE만 예측하는 영역으로 밀어낸다.

- 가는 신호: 유도가 L3인데 sourcer가 special_case, 또는 "새 지위 태그 없음".
- 후보의 모습: 원 claim ⊂ 새 claim, 새 claim이 문헌 정리와 갈라지는 **판별 사례**(입력·기대값)
  하나를 명시. 판별 사례가 없으면 generalize가 아니라 재서술이다.

## 축소 1. `partial` — 부분명제

원래 claim을 특수사례로 좁혀 먼저 세운다(차원 축소, 대칭 가정 추가, 정수 n만, 평탄 배경만).
후보의 모습: "원 claim이되 정의역을 D'⊂D로 제한", 반증조건은 원래 반례가 D' 안에 재현되지 않는지.

## 축소 2. `alt_derivation` — 다른 유도

같은 claim을 다른 도구로(변분→직접 계산, 실공간→푸리에, 귀납→생성함수). `assumptions_added` 비어 있음.

## 축소 3. `reformulate` — 다른 정식화

claim을 동치이거나 더 약한 형태로(등식→부등식, 존재→상계, 정확→점근). 원 claim과의 함의 방향 명시.

## 축소 4. `weaken` — 가정 명시 약화

필요한 가정을 명시 추가하고 그 아래에서만 주장한다. 자유 파라미터 수는 재현할 무차원 비율 수보다
적어야 한다. 이유: 닫힘 예산은 "공리 1개 명시 추가"를 정당한 선택으로 인정하지만 적합은 아니다.

## judge 규칙과의 관계

- 반례 있음 → `pivot`, `pivot_step`은 축소 순서에서 `pivots_tried`에 없는 첫 단계.
- 반례 없음 + sourcer identical·special_case → `pivot`/`conjecture` (질문을 park하지 않는다.
  카드 attempt에서 카드 자체가 재발견이면 `refute`이고 `force_pivot`이 붙는다).
- 반례 없음 + L3 + special_case → `pivot`/`generalize` 또는 사다리 단이면 `promote`+cited.
- 같은 경로 3회 실패(attempt ≥ 3, 반례 없음, L1~L2) → `pivot`/`alt_derivation`.
- `force_pivot`이 있으면 judge는 그 값을 `pivot_step`으로 쓴다. 어길 수 없다.

## 왜 탐색은 fail-open이고 논문은 fail-closed인가

틀린 길을 빨리 걷는 것이 안 걷는 것보다 낫다. 탐색 훅은 실패를 문맥에 넣되 차단하지 않고,
논문은 L3 미만 인용을 막는다. 확장 방향도 같은 이유다. 틀릴 수 있는 식을 빨리 세워야 kill이
빨리 돌고, 살아남은 식만 사다리를 오른다.
