# 48. 확장 위상공간과 Maxwell 변분의 복원 — CE-GF2

## 48.1 계산 전 등록

**[새 후보 비교]** 2026-09-10. CE-GF1의 rank 제한을 넘는 두 Abelian 구성을 비교한다.
외부 Lorentzian 계량을 공급하고 $S=-\int\sqrt{-g}F_{\mu\nu}F^{\mu\nu}/(4g_A^2)$를
추가 작용으로 선택한다. 이 Maxwell 형태와 결합상수는 공통 스칼라 작용에서
유도됐다고 가정하지 않는다.

- 후보 A: 네 scalar $r^1,r^2,r^3,r^4$,
  $F=dr^1\wedge dr^2+dr^3\wedge dr^4$.
  비퇴화 영역에서 Maxwell 변분과 동등한지, 퇴화 영역에 여분 해가 있는지 검사한다.
- A의 고정 대조: $r^1=x$, $r^2=y\sin x-z\cos x$, $r^3=r^4=0$.
  대응 자기장 $\mathbf B=(0,\cos x,\sin x)$, $\mathbf E=0$의 Maxwell 잔차와
  scalar 변분 잔차를 별도로 계산한다.
- 후보 B: 여덟 scalar $(q^a,p_a)$, $a=0,1,2,3$,
  $A=p_a dq^a$, $F=dp_a\wedge dq^a$, $\det(\partial_\mu q^a)\ne0$.
  coherent oscillator 네 개의 정규화된 선 프레임으로 이 연결을 구성한다.
- 임의 $A_\mu$의 표현과 모든 Maxwell 변분을 포함하는지를 해석적으로 검사한다.
  진공에서도 변분의 rank가 떨어지는지 확인한다.
- 고정 수치: $q=(0.1,-0.2,0.3,0.4)$, $p=(0.2,0.1,-0.3,0.2)$,
  oscillator cutoff $L=4,8,16$에서 Berry 연결을 검산한다.
  A의 잔차는 $x=0,\pi/6,\pi/2$에서 검사한다. 표준 장 대조는 CE-GF1과 같게 둔다.
- 수치 항등식 허용오차 $10^{-12}$, Berry 연결은 $L=16$에서 $10^{-10}$.
  원래 공통 계량·기하 질량·네 힘의 완성을 이 Abelian 검사로 대신하지 않는다.

위 조건은 계산 전에 등록했다. 다음은 그 유도·검산·판정이다.

## 48.2 네 scalar의 표현력과 변분의 손실

후보 A는 $F\wedge F\ne0$인 장도 표현하므로 CE-GF1의 rank 반례를 넘는다.
그러나 표현 가능한 장과 올바른 운동방정식을 얻는 것은 다른 조건이다.
$\mathcal E^\mu=\nabla_\nu F^{\nu\mu}/g_A^2$라 하면 Maxwell 작용의 변분은
$\delta S=\int\sqrt{-g}\,\mathcal E^\mu\delta A_\mu$다.
네 scalar를 변분하면 exact 1-form을 제외한 퍼텐셜 변분은

$$
\delta A=\delta r^1dr^2-\delta r^2dr^1
+\delta r^3dr^4-\delta r^4dr^3
$$

이므로 얻는 식은

$$
\mathcal E^\mu\partial_\mu r^I=0,\qquad I=1,2,3,4.
$$

$\partial_\mu r^I$가 가역일 때는 $\mathcal E^\mu=0$과 동등하다.
하지만 rank가 떨어지면 모든 Maxwell 식이 따라오지 않는다.
퇴화장에 Euler potential을 사용하는 기존 논의는
[Gralla·Jacobson](https://arxiv.org/abs/1401.6159)과 연결되며, 아래에서는
추가 두 scalar가 있어도 퇴화 영역에서 변분의 손실이 남는 직접 반례를 쓴다.

## 48.3 Maxwell 식을 만족하지 않는 정류 해

등록한 $r^1=x$, $r^2=y\sin x-z\cos x$, $r^3=r^4=0$에서

$$
F=dx\wedge(\sin x\,dy-\cos x\,dz),\qquad
\mathbf B=(0,\cos x,\sin x),\quad\mathbf E=0.
$$

상수 계량·$g_A=1$ 단위에서 Maxwell 잔차는

$$
\mathcal E^\mu=(0,0,\cos x,\sin x)\ne0.
$$

그런데 $\mathcal E\cdot dr^1=0$이고
$\mathcal E\cdot dr^2=\cos x\sin x-\sin x\cos x=0$이다.
나머지 두 미분은 0이므로 네 scalar의 변분식은 모두 만족한다.
Bianchi 식도 $F=dr^1\wedge dr^2$여서 만족한다.
전류가 자기장에 평행한 이 구성은 원천 없는 Maxwell 식의 해가 아니지만
후보 A의 작용에서는 stationary configuration이다. 안정성까지 검사한 것은 아니다.

따라서 후보 A를 퇴화장을 포함한 일반 Maxwell 변분과 동일시하는 판본은 기각한다.
서로 독립적인 네 $dr^I$가 있는 비퇴화 영역의 조건부 동등성은 보존한다.

## 48.4 네 쌍의 coherent 좌표로 임의의 연결을 표현하기

후보 B의 내부 선 프레임은 네 oscillator의 전체 Fock 공간에서

$$
v(q,p)=e^{-\frac i2\sum_ap_aq^a}
\bigotimes_{a=0}^3\left|\frac{q^a+ip_a}{\sqrt2}\right\rangle
$$

로 정의한다. 모든 유한 실수 $q,p$에서 정규화되고 필요한 수 연산자 모멘트가 존재한다.
coherent state 하나의 연결은 $(p\,dq-q\,dp)/2$이며,
위의 전체 위상은 $d(pq)/2$를 추가한다. 따라서 정확히

$$
\boxed{i\langle v|dv\rangle=p_a dq^a,\qquad
F=dA=dp_a\wedge dq^a.}
$$

이는 임의의 연결을 표시할 수 있는 구체적인 정규화 상태다.
시공간에서 $J_\mu{}^a=\partial_\mu q^a$가 가역이면

$$
A_\mu=J_\mu{}^ap_a,\qquad
p_a=(J^{-1})_a{}^\mu A_\mu.
$$

따라서 그 영역의 임의의 매끄러운 Abelian 퍼텐셜 $A_\mu$를 포함한다.
예를 들어 국소 좌표 $q^a=x^a/\ell$를 선택하면 $p_a=\ell A_a$다.
$q,p$는 무차원, $A_\mu$는 길이의 역수 단위이며 $\ell$은 공급한 단위 변환이다.
비가역 $J$, 전역 좌표·비자명 bundle까지 이 한 공식으로 덮는다고 주장하지 않는다.

공통 단위를 써서 $q=x$, $p=(0,0,x,t)$로 놓으면
$A=x\,dy+t\,dz$이며 CE-GF1의 평행 전기·자기장 반례를 정확히 재현한다.
이번 확장에서는 $F\wedge F$가 0일 필요가 없다.

## 48.5 진공을 포함한 Maxwell 변분의 국소 복원

이제 $q,p$를 모두 변분하면

$$
\delta A_\mu=J_\mu{}^a\delta p_a+p_a\partial_\mu\delta q^a.
$$

$p$의 변분에서 $\mathcal E^\mu J_\mu{}^a=0$이 나온다.
$J$가 가역이므로 이는 모든 Maxwell 식 $\mathcal E^\mu=0$을 준다.
$q$의 변분식 $\nabla_\mu(p_a\mathcal E^\mu)=0$은 그때 자동으로 만족한다.
반대로 Maxwell 해는 두 scalar 변분식을 모두 만족한다.
compact 지지의 변분 또는 적절한 경계조건을 전제로 한다.

이 논증은 $p=0$, $F=0$에서도 유지된다. $q=x/\ell$를 고정한 변분으로도
$\delta A_\mu=\delta p_\mu/\ell$가 임의의 1-form을 주므로,
진공 주변 $F^2$가 $p$에 대한 quadratic 운동항을 갖는다.
CE-GF1·후보 A의 진공 변분 퇴화를 피한다.

단, 이 동등성은 **작용이 $A$를 통해서만 $q,p$에 의존할 때**의 결과다.
국소적으로 $(q,p)$를 $(q,A)$로 다시 표시하면 이 작용은 $q$와 무관하다.
$\delta q^a=\eta^a$, $\delta p_a=-(J^{-1})_a{}^\mu p_b\partial_\mu\eta^b$는
$A$를 그대로 유지하고, $\delta p_a=(J^{-1})_a{}^\mu\partial_\mu\lambda$는
$A\mapsto A+d\lambda$를 준다. 따라서 이 작용만을 쓸 때 여덟 scalar를 여덟 개의
독립 물리 입자로 세지 않는다. 표준 Maxwell 작용의 국소 재표현으로 이해한다.

## 48.6 공통 이론에 결합할 때 다시 닫아야 할 부분

이번 Maxwell 형태의 작용과 외부 Lorentzian 계량은 **공급한 입력**이다.
임의의 전자기장과 올바른 변분을 정규화된 선 프레임으로 표현할 수 있음을 보였지만,
그 작용·부호·결합상수가 원래 공통 스칼라 작용에서 발생했다고 증명하지 않았다.

특히 기존 프레임의 양자 운동항처럼 $q,p$에 따로 의존하는 항을 추가하면
위의 추가 중복성과 순수 Maxwell 변분이 깨질 수 있다.
$p=0$이어도 $q=x/\ell$인 프레임은 변하므로 그 정상 미분의 운동항은 일반적으로 0이 아니다.
이를 무시하고 기존 전체 작용이 이미 Maxwell 진공을 복원했다고 할 수 없다.

또한 Abelian 선을 두 oscillator에서 네 oscillator로 바꿨다.
기존 약한·색깔·flavor 인자와 결합한 공통 계량, 기하 질량, 상승 스펙트럼 중복도와
UV 응답은 다시 계산해야 한다. 새 좌표의 양의 Hilbert 계량을 Lorentzian 시공간
계량으로 동일시하지 않는다. 비가환 세력의 독립 동역학과 Einstein 극한도 남는다.

따라서 이 결과는 ‘세 힘의 세 투영’을 보존할 수 있는 더 큰 상태공간의 Abelian
구성 요소 후보이며, 완성된 새 통일장 이론은 아니다.

## 48.7 재현과 판정

[계산 코드](../../verify/ce_extended_maxwell_variation.py)와
[산출·소스 해시](../../verify/ce_extended_maxwell_variation.json)를 사용한다.

```powershell
python -B verify/ce_extended_maxwell_variation.py
```

**[산출]** 후보 A의 세 위치에서 Maxwell 잔차 노름은 1인 반면 scalar 변분 수축은
수치적으로 모두 0이었다. 후보 B의 두 가역 coframe에서 진공 변분 rank는 4였고,
임의로 고정한 퍼텐셜 변분의 복원 오차는 $7.86\times10^{-17}$ 이하였다.
평행 전기·자기장의 반대칭 행렬도 정확히 재현했다.

| oscillator cutoff $L$ | 전체 선 상태 차원 | Berry 연결 최대 오차 | 버린 상태 노름제곱 합 상계 |
|---:|---:|---:|---:|
| 4 | 625 | $7.55\times10^{-7}$ | $1.23\times10^{-7}$ |
| 8 | 6561 | $4.49\times10^{-14}$ | $3.51\times10^{-15}$ |
| 16 | 83521 | $5.56\times10^{-17}$ | $2.99\times10^{-32}$ |

정규화된 전체 텐서곱과 인자별 연결의 차이는 $3.75\times10^{-16}$ 이하였고,
전체 노름 오차는 $6.67\times10^{-16}$ 이하였다.
상태 꼬리 상계를 미분 오차의 상계로 바꾸어 쓰지 않았다. 연결 오차는 별도로 계산했다.
무한 정규화와 변분의 동등성은 위 해석적 식으로 증명하며, 유한 표는 구현 검산이다.

**판정:** 네 scalar의 일반 Maxwell 동등성은 퇴화 영역 반례로 기각한다.
여덟 scalar의 coherent 선 프레임은 가역 coframe 영역에서, 공급한 Maxwell 작용의
모든 장과 변분을 진공까지 포함해 표현하는 조건부 구성으로 보존한다.
공통 작용에서의 유도·추가 양자항·비가환 부문·동적 시공간과 공동 RMSE는 미완성이다.
