# 추측 카드 예시

이미 닫힌 E40(hard-blockade Fibonacci history의 Parry 비율)을 **카드 형식으로 재서술**한 것이다.
새 주장이 아니며 형식 예시로만 쓴다. 실제 카드는 문헌에 없는 식이어야 한다(§2 진취성 규율).

```markdown
---
question: Q-EX
card: F-01
kind: 예측식
formula: "R - ((1+sqrt(5))/2)**2"
formula_latex: "$$ R=\\phi^{2},\\qquad p_{\\rm m}=\\frac{1}{1+\\phi^{2}},\\qquad \\phi=\\frac{1+\\sqrt5}{2} $$"
symbols:
  n: positive integer
dimensions:
  R: 1
  p_m: 1
  n: 1
free_parameters: []          # 0개 < 예측 비율 2개
predicts:
  - observable: "hard-blockade 무가중 Fibonacci history의 late attractor 비율 R"
    value: 2.6180339887
    uncertainty: 0
    baseline: {source: "E40 유한 부피 수치", value: 2.618, error: 1.0e-3, accessed: 2026-09-02}
    comparison_frozen: true
  - observable: "기록 확률 p_m"
    value: 0.2763932023
    uncertainty: 0
    baseline: {source: "E40", value: 0.2764, error: 1.0e-4, accessed: 2026-09-02}
    comparison_frozen: true
recovers:
  - limit: "blockade 제거(모든 이웃 허용, 균등 가중)"
    known: "R → 1 (단일 비율 없음)"
    check: 0
kill:
  - "N ≥ 2^10 유한 부피에서 R(N)이 φ²로 수렴하지 않음 (|R(N)-φ²| > 1e-3, 씨앗 20260902)"
  - "Parry 측도 대신 다른 최대 엔트로피 측도 선택에서 R이 φ²와 다른 값으로 안정됨"
ladder:
  - {step: 1, claim: "Fibonacci adjacency 행렬의 Perron 고유값이 φ", kind: 외부기존}
  - {step: 2, claim: "Parry 측도 아래 두 기호의 정상 비율은 Perron 고유벡터 성분비", kind: 외부기존}
  - {step: 3, claim: "hard-blockade history 공간이 golden-mean 부분이동과 동형", kind: 보조정리}
  - {step: 4, claim: "late attractor R(N)→φ², 오차 O(φ^{-2N})", kind: 수치시험}
novelty:
  ce_specific: "blockade 규칙만으로 무입력 단일 무차원 비율 φ²가 나오고 그것이 영수증 장부의 기록 확률을 고정한다"
  nearest_prior_art: []
verify:
  - type: identity
    lhs: "((1+sqrt(5))/2)**2"
    rhs: "(1+sqrt(5))/2 + 1"
  - type: numeric
    expr: "1/(1+((1+sqrt(5))/2)**2) - 0.2763932023"
    tol: 1.0e-9
---

## 왜 이 식인가

blockade는 인접 두 기록을 금지하므로 허용 history는 golden-mean 부분이동이다. 무가중이면
최대 엔트로피 측도가 Parry 측도이고 그 비율은 Perron 고유벡터가 준다. 그래서 자유 입력
없이 비율 하나가 떨어진다.

## 사다리

1. 외부기존: Perron–Frobenius, 인용으로 닫는다.
2. 외부기존: Parry 1964, 인용으로 닫는다.
3. 보조정리: 동형 사상을 derivation으로 쓴다. 죽이는 것은 blockade가 3-기호 금지어를 추가로 만드는 경우.
4. 수치시험: `verify/Q-EX/F-01/check_attractor.py`, N=2^6..2^12, tol 1e-3.

## 죽는 조건

kill 1은 4단의 스크립트가 판정한다. kill 2는 측도를 Parry 외로 바꾼 같은 스크립트가 판정한다.
```

## 예산식 카드의 다른 점

`kind: 예산식`이면 `predicts` 대신 `budget`을 쓴다.

```yaml
budget:
  total: "E_total"
  parts: ["E_seen", "E_unseen"]
  defined_on: "제약 수준(H|Ψ⟩⟩=0) 유한 모형, Z_N 군평균 Π"
  conserved_by: "Π가 C와 가환 — verify 0"
```

항등식은 `formula: "E_total - (E_seen + E_unseen)"`처럼 `= 0` 형태로 적고, `recovers`에는
"비선택 성분 없음(E_unseen=0)이면 표준 기댓값 복원" 같은 극한을 둔다.
