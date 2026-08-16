# 물리 독립 AGI Core V0 무차원 감사

Status: COMPLETE

## 판정

V0의 두 exact family와 public record는 무차원 소프트웨어 좌표로만 구성되므로 무차원 게이트를 통과한다. 이 판정은 물리적 타당성, 학습 성능, AGI 또는 의식을 뜻하지 않는다.

| 코어 항목 | 차원 벡터 | 무차원 | 계약 |
|---|---|---|---|
| categorical state `s` | `(0,0,0,0)` | yes | 유한 정수 label |
| categorical action `a` | `(0,0,0,0)` | yes | 유한 정수 또는 canonical symbol |
| tick·horizon | `(0,0,0,0)` | yes | 이산 step count |
| transition `T(s,a)` | `(0,0,0,0)` | yes | categorical state 반환 |
| score·risk·confidence | `(0,0,0,0)` | yes | finite normalized scalar |
| digest·nonce·identifier | 해당 없음 | yes | 물리량이 아닌 canonical byte label |

V0 family는 다음 exact categorical 연산만 사용한다.

$$
T_{\oplus}(s,a)=s\oplus a,
\qquad
T_{\rm set}(s,a)=a,
\qquad S=A=\{0,1\}.
$$

exp, log, 삼각함수, 물리적 norm, 차원 있는 덧셈 또는 차원 있는 고정점은 없다. 따라서 전역 `dimensionless_checker` registry를 변경하지 않는다. focused test는 state/action이 exact finite categorical scalar인지, bool·float·문자열을 정수 slot으로 암묵 변환하지 않는지 검사해야 한다.

후속 실제 환경 adapter가 길이·시간·에너지 같은 물리량을 넣을 경우 core record에 직접 전달하면 안 된다. adapter가 명시적 기준척도 `x_ref`, `t_ref`, `E_ref`를 provenance에 기록하고 `x/x_ref`, `t/t_ref`, `E/E_ref`로 정규화한 뒤 전달해야 하며, 그때 별도 무차원 감사를 다시 연다.
