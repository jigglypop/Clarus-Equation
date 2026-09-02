---
name: derivation-style
description: 유도·증명·계산 과정을 쓰거나 고칠 때, derivations/<Q>/attempt-NN.derivation.md 파일을 만들 때, 논문에 실을 유도 절을 쓸 때 반드시 참조. 한 줄에 한 단계, 단계 번호 (Sn), 프론트매터 verify 블록 필수. 문체 규칙은 여기 없다(ko-academic-prose 몫).
---

# 유도 파일 계약

파일 위치: `derivations/<Q-id>/attempt-NN.derivation.md` (NN은 두 자리). 추측 카드
`derivations/<Q-id>/F-NN.formula.md`는 conjecture-first 스킬의 계약을 따르며 같은 훅이 verify
블록을 돌린다(산출물 `verify/<Q>/F-NN/`). 유도 파일은 프론트매터 `ladder_step`으로 어느 단을 닫는지 적는다.
저장 즉시 PostToolUse 훅이 `verify_derivation.py`를 돌리고 결과가 문맥에 들어온다.
이유: verify 블록이 없으면 루프가 한 바퀴 느려진다. 블록이 최우선 산출물이다.

## 프론트매터 (필수 키)

```yaml
---
question: Q-0007
attempt: 4
ladder_step: 2         # 카드 사다리의 몇 단인가 (lemma 질문이면 생략)
claim: "한 문장"
assumptions:
  - "ρ ∈ C^1 이며 ρ(x)=O(|x|^{-n-1})"
symbols:                # 기계검증용 기호 선언. 값: real | positive real | positive integer |
  n: positive integer   #      integer | nonnegative real | nonzero real | complex | function
  x: real
  rho: function
verify:                 # verify_derivation.py가 실행하는 검사 목록
  - type: identity
    lhs: "n*(n+1)/2 + (n+1)"
    rhs: "(n+1)*(n+2)/2"
  - type: numeric
    lhs: "..."
    rhs: "..."
    samples: 20
    tol: 1e-9
---
```

- 표현식은 sympy 문법(`**`, `sqrt`, `exp`, `pi`, `oo`)이다. `^`는 `**`로 자동 치환된다.
- `symbols`에 없는 기호를 쓰면 검사가 fail이다. 이유: 미선언 기호는 오타이거나 숨은 가정이다.
- `function` 기호가 들어간 검사는 수치 표본이 불가하므로 identity/limit(기호)만 의미가 있다.
- 유도가 미완이어도 검증 가능한 중간 항등식을 최소 1개 넣는다.
- 가정을 추가하면 `assumptions`에 반드시 적는다. 이유: judge가 `weaken` 단계를 판단할 근거다.

## 본문

```markdown
## 유도

$$ \int \rho\,\nabla\phi\,dx = -\int \phi\,\nabla\rho\,dx + [\rho\phi]_{\partial} $$  (S1) 부분적분
$$ [\rho\phi]_{\partial} = 0 $$  (S2) 가정 ρ=O(|x|^{-n-1})로 경계항 소거
```

- 한 줄에 한 단계. 한 단계에 두 조작 금지. 이유: 어느 조작이 틀렸는지 기계와
  adversary가 짚을 수 있어야 한다.
- 각 단계는 `$$ ... $$  (Sn) <조작 이름>` 형식. 직전 단계에서 무엇을 했는지 한 구절.
- 표시 수식은 `$$` 블록, 인라인은 `$`. `\[ \]`, `\( \)`는 쓰지 않는다(paper 정책과 통일).
- 미완인 단계는 `(Sn) [미완성: 무엇이 빠졌는지]`로 표시하고 건너뛰지 않는다.

## 검사 유형 상세

`references/checks.md`에 identity/limit/numeric/inequality 작성 예와 결과 해석이 있다.

## 논문에 옮길 때

paper-writer는 L3 이상 항목의 유도만 옮기며, 단계 번호는 유지하되 훅 결과
문자열(pass/fail)은 본문에 쓰지 않는다. 이유: 기계 상태는 이론 지위가 아니다.
