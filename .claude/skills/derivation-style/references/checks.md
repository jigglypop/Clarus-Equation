# verify 블록 검사 유형

실행기: `.claude/hooks/lib/verify_derivation.py`. 난수 씨앗 20260902 고정.
결과 파일: `verify/<Q>/attempt-NN/hook_result.json`.

## 기호 선언 (`symbols`)

| 값 | sympy 가정 | 수치 표본 |
|---|---|---|
| `real` (기본) | real=True | U(-3, 3) |
| `positive real` | positive=True | U(0.1, 3) |
| `nonnegative real` | nonnegative=True | U(0, 3) |
| `nonzero real` | real, nonzero | U(-3,3), |x|≥1e-3 |
| `integer` | integer=True | {-6..6} |
| `positive integer` | integer, positive | {1..12} |
| `complex` | (가정 없음) | U(-3,3) 실수만 |
| `function` | sp.Function | 표본 불가 → 수치검사 skipped |

## identity — 항등식 (symbolic + numeric)

```yaml
- type: identity
  lhs: "sin(x)**2 + cos(x)**2"
  rhs: "1"
```
sympy: `simplify(expand(lhs - rhs)) == 0` → symbolic pass/fail. 이어서 표본 20점에서
상대오차 `|a-b|/(1+|b|) ≤ tol`(기본 1e-9) → numeric pass/fail. sympy가 없으면 symbolic은
skipped(reason sympy-not-installed), numeric만 numpy로 평가한다.

## limit — 극한 (symbolic만)

```yaml
- type: limit
  expr: "sin(x)/x"
  var: x
  point: "0"       # "oo", "-oo" 가능
  expected: "1"
  dir: "+"         # 선택. "+" | "-"
```
`sp.limit(expr, var, point, dir)`와 expected의 차를 simplify. sympy 없으면 skipped.

## numeric — 수치 대조 (numeric만)

```yaml
- type: numeric
  lhs: "(x+1)**2"
  rhs: "x**2 + 2*x + 1"
  samples: 20
  tol: 1e-9
```
또는 `expr`만 주면 `|expr| ≤ tol`. 표본에서 무한/NaN은 건너뛴다.

## inequality — 부등식 (numeric만)

```yaml
- type: inequality
  lhs: "x**2 + 1"
  rhs: "2*x"
  relation: ">="   # <=, <, >=, >
  samples: 50
```
모든 표본에서 관계가 성립해야 pass. 위반 표본 최대 5개가 `violations`에 기록된다.

## 집계

- `symbolic`: identity/limit 중 하나라도 fail이면 fail, 하나라도 pass면 pass, 아니면 skipped.
- `numeric`: 수치 평가된 검사 중 하나라도 fail이면 fail, 하나라도 pass면 pass, 아니면 skipped.
- 훅은 항상 exit 0이다. 결과는 additionalContext로 문맥에 들어온다.

## 흔한 실패

- `undeclared symbol in expression`: symbols에 기호 추가.
- `function symbol cannot be sampled`: 함수 기호가 있는 검사는 identity/limit로만 쓴다.
- `numeric eval error`: sympy 없는 환경에서 `Sum`, `Integral`, `diff` 같은 심볼릭 연산은
  평가 불가. 닫힌 형식으로 풀어 쓰거나 sympy 설치 후 재검증.
