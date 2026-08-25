# 기회비용 함수 무차원 감사

Status: COMPLETE

| 코어 인자 | 차원 벡터 $(M,L,T,\Theta)$ | 무차원? | 정규화 또는 scale |
|---|---:|---:|---|
| $p_a$, $p_U$, $q_a$, $r_a$ | $(0,0,0,0)$ | yes | probability |
| $\ln p_a$, $\ln(p_a/r_a)$ | $(0,0,0,0)$ | yes | log에는 probability ratio만 입력 |
| $C_I$, $H$, $D$ | $(0,0,0,0)$ | yes | 정보 readout |
| $H/(k_BT)$ | $(0,0,0,0)$ | yes | energy를 $k_BT$로 정규화 |
| $k_BT D$ | $(1,2,-2,0)$ | no: energy | $k_BT$가 energy scale 공급 |
| $\hbar\ln(Z/Z_{\rm ref})$ | $(1,2,-1,0)$ | no: action | ratio는 무차원, $\hbar$가 action 공급 |
| $\hbar C_I/\tau_*$ | $(1,2,-2,0)$ | no: energy | 독립 time scale $\tau_*$ 필요 |
| $\epsilon_*f(C)$ | $(1,-1,-2,0)$ | no: energy density | $f$의 인자 $C$는 무차원 |

핵심 판정은

$$
[C_I]=1,
\qquad
[k_BT C_I]=E,
\qquad
[-\hbar\ln(Z/Z_{\rm ref})]=E\,T
$$

다. 따라서 $-\ln p$와 entropy를 실제 energy라고 부르는 식은 차원 게이트에서
실패한다. $k_BT$, $E_*$, $\hbar/\tau_*$ 또는 $\epsilon_*$를 명시하면 차원은
맞지만 그 scale의 물리적 정당성은 별도다.

차원 상태: 정보 코어 무차원, thermal/action/stress 경로는 명시 scale에서만 정합.
무차원성은 물리적 동일성을 증명하지 않는다.
