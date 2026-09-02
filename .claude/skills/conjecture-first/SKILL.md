---
name: conjecture-first
description: "attempt를 시작할 때 질문에 추측 카드(derivations/<Q>/F-NN.formula.md)가 없을 때, 예측식·예산식을 세우거나 감사·채택·기각할 때, 사다리 단을 고를 때, questions.yaml에 force_pivot=conjecture가 있을 때 반드시 참조. 식을 먼저 적고 단계로 증명한다. prover(모드: 추측)·adversary(카드 감사)·sourcer(신규성)·judge(adopt/refute)가 쓴다."
---

# 추측 우선 (식을 먼저, 증명은 단계로)

재발견 루프의 원인은 순서였다. 반증이 싼 후보를 고르면 문헌에 있는 보조정리로 수렴하고,
공리는 숨은 가정을 고백할 때만 태어났으며, 진전의 정의에 "새 식"이 없었다. 이 스킬은 순서를
뒤집는다. **식을 먼저 적고(공리 후보), 그 식이 내놓는 숫자나 장부 항등식을 사전등록한 뒤,
증명 사다리를 한 단씩 오른다.** 식이 틀리면 kill 조건이 죽이고, 문헌에 있으면 더 강한 식으로 간다.

## 1. 카드 — `derivations/<Q>/F-NN.formula.md`

프론트매터 필수 키(`ledger.py card-check`가 검사, 저장 즉시 verify 훅 실행):

| 키 | 내용 |
|---|---|
| `question`, `card` | Q-id, `F-01` (재추측은 F-02, 옛 카드는 지우지 않는다) |
| `kind` | `예측식`(숫자를 내놓음) 또는 `예산식`(보존·장부 항등식 `total = Σ parts`) |
| `formula`, `formula_latex` | sympy 문법 한 줄(`lhs - rhs` 형태) + 표시용 LaTeX |
| `symbols`, `dimensions` | derivation-style과 같은 기호 선언 + 차원표(무차원은 1) |
| `free_parameters` | 이름·무엇이 고정하는가. **개수 < 예측 비율 개수**(같으면 적합이지 예측이 아니다) |
| `predicts` (예측식) | `{observable, value, uncertainty, baseline{source,value,error,accessed}, comparison_frozen: true}` ≥1. 숫자를 **지금** 적는다 |
| `budget` (예산식) | `{total, parts[≥2], defined_on, conserved_by}` |
| `recovers` | 기존 극한 복원 ≥1 (`limit`, `known`, `check`=verify 인덱스). 목표 계약 게이트 1 |
| `kill` | 사전등록 반증 조건 ≥2. 각각 계산·관측으로 판정 가능해야 한다 |
| `ladder` | 1~7단 `{step, claim, kind}`; kind ∈ 보조정리·외부기존·수치시험·예측시험 |
| `novelty` | `ce_specific`(문헌에 없는 것 한 문장), `nearest_prior_art`(sourcer가 채움) |
| `verify` | ≥1 (극한 복원·차원 항등식). 산출물 `verify/<Q>/F-NN/hook_result.json` |

본문은 세 절뿐: `## 왜 이 식인가`(3문장, 유도 아님) · `## 사다리`(단마다 무엇을 증명하고 무엇이
죽이는지 한 줄) · `## 죽는 조건`(kill을 계산 절차로). 예시는 `references/card-example.md`.

## 2. 진취성 규율

- **식은 유도 없이 선언한다.** 지위는 `[공리: 후보]`. 정당화는 유도가 아니라 (a) 극한 복원
  (b) 차원 일관 (c) 사전등록 예측·kill (d) 문헌 부재 넷이다. closure-gate의 "유도 가능해
  보이는 공리는 축소 후보" 규칙은 사다리가 닫힐 때까지 카드에 적용하지 않는다.
- **카드마다 숫자 하나 또는 항등식 하나.** 숫자 없는 예측식, 항이 둘 미만인 예산식, 정의의
  재서술은 adversary가 P1 "무내용"으로 돌려보낸다.
- **틀릴 수 있는 만큼 강하게.** 후보가 여럿이면 반증이 싼 것이 아니라 **반증 가능성이 크고
  문헌에 없는 것**을 고른다. 안전한 추측은 재발견이다.
- **결과를 본 뒤 바꾸지 않는다.** 예측 숫자·kill·tolerance는 카드에 고정된다. 바꾸려면 새
  카드(F-02)이고, 옛 카드는 기각으로 남는다.
- **한 attempt = 사다리 한 단.** 외부기존 단은 sourcer 인용으로 닫고(`ladder_cited`) attempt를
  쓰지 않는다. 보조정리 단은 derivation-style 유도, 수치시험·예측시험 단은 `verify/` 스크립트.
- **재발견은 신호다.** 보조정리가 문헌에 있으면 그 단만 `cited`로 닫고 다음 단으로 간다. 카드
  자체가 identical·special_case면 `refute`이고 질문은 살아서 `force_pivot: conjecture`가 된다.
  같은 세션에서 더 강한 카드(F-02)를 한 번 더 세운다.
- **사다리는 7단 이하.** 8단이 필요하면 식이 너무 약하거나 너무 멀다. 질문을 쪼개지 말고 식을 고친다.

## 3. 루프 안의 위치

```
카드 없음(kind≠conjecture 또는 force_pivot=conjecture)
  → prover(모드: 추측) F-NN.formula.md (훅 verify) → adversary(카드 감사 6종)
  → sourcer(신규성 대조, 필수) → judge adopt(카드 등록, 사다리 열림) | refute
카드 있음 → 다음 open 단 하나 → prover(유도) → adversary → [sourcer] → judge
  promote(단 닫힘) | continue | pivot | refute(kill 발동 → 질문 parked, 진전 원장 §4)
모든 단 closed/cited → 질문 resolved, 카드 status 정리 → /paper
채택 전 카드가 adversary 반례(P0)로 죽으면(재발견·kill 발동 아님) 질문은 active+force_pivot:
  conjecture로 살아 같은 세션에서 보정 카드 F-(NN+1)을 세운다 (예: Q-0008 F-01→F-02)
일괄 모드(사용자 지시) → 질문 여러 개에 카드 attempt를 병렬로; adversary 카드 감사는 opus
```

`ledger.py after-attempt`가 사다리 상태를 옮기고, `summary`·`ladder <Q>`가 다음 단을 보여 준다.
재발견 2회 또는 축소 4단계 소진이면 `force_pivot: conjecture`가 자동으로 붙는다.

## 4. adversary 카드 감사 6종

`dimension`(차원표 대조) · `recovers`(각 극한을 verify로 실제 실행) · `dof`(자유 파라미터 < 예측
비율) · `content`(정의 재서술·동어반복인가) · `kill_executable`(각 kill이 실제 계산·관측 절차인가)
· `ladder_complete`(단을 모두 닫으면 식이 정리가 되는가, 빠진 단). 결함 P0(극한 파괴·차원 불일치·
무내용) → refute. P1(사다리 공백·kill 모호) → 같은 세션에서 카드 수정 1회 후 재감사. P2 표기.

## 5. 진전 종류와 지위

카드 adopt = **예측**(진전 원장 §5 한 행, §2 "현재 추측" 갱신). 단 closed = 닫힘. 카드 refute(kill)
= 기각. 카드 refute(재발견) = 정리(진전 아님)와 재추측. 논문 지위: 채택 카드 `[공리: 후보]`, 그
숫자 `[예측: 사전등록]`, 사다리 완주 뒤 `[정리]`(paper-writer가 원장 기준으로 승격). 관측 근접은
여전히 증명이 아니다. 카드가 L3라는 것은 "일관성 검증 통과"이지 참이라는 뜻이 아니다.
