# 49. Maxwell 확장과 공통 양자항의 충돌 — CE-GF3

## 49.1 계산 전 등록

**[결합 검사]** 2026-09-10. CE-GF2의 네 쌍 $(q^a,p_a)$와 $A=p_a dq^a$에,
CE-GN4의 균등 light–heavy gap에서 생긴 상대 두 미분 항을 함께 넣는다.
이 검사는 균등 gap의 상대 응답을 사용하며 상승 스펙트럼 판본의 전체 loop와
동일하다고 가정하지 않는다. 외부 Minkowski 계량 $\eta=(-,+,+,+)$와 Maxwell 작용은 유지한다.

- coherent 선의 정상 계량을 사용해 Lorentzian 두 미분 항의 부호와 성분을 정한다.
- 배경 $q^a=x^a/\ell$, $p_a=0$에서 모든 $q,p$의 quadratic 변분을 계산한다.
  추가 항이 있는 이론에서 $q$를 gauge 고정으로 제거했다고 가정하지 않는다.
- 고정값 $a=1,b=5,g_A=1,\ell=1$, $C=C(a,b)>0$.
  비교는 $C/\kappa$가 $1/2,1,2$인 경우, $\kappa=1/(g_A^2\ell^2)$다.
- $k=0.01,0.05,0.1$의 평면파에서 속도 Hessian, 종방향 분산식,
  별도 일차 시간 진화 행렬의 고윳값을 비교한다. 허용오차 $10^{-10}$.
- 같은 $q$에서 $p\mapsto p+\Delta p$가 frame 위상변환인지도 검사한다.
  $|\Delta p|=0,0.1,0.5,1$의 coherent 상태 겹침을 고정한다.
- 게이지 변분이 작용을 바꾸거나 진공에 불안정 모드가 있으면 단순 결합 판본을 기각한다.
  Maxwell 항만의 성공을 전체 작용의 성공으로 승계하지 않는다. 피팅 없음.

위 조건을 계산 전에 등록했다.

**추가 계산 전 조건:** $\ell=1$ 배경은 미분 전개의 느린 배경을 보장하지 않으므로
그 숫자는 잘린 작용의 검사로만 사용한다. 별도 고정 비교 $\ell=100,g_A=0.01$도
등록한다. 이는 같은 $\kappa=1$을 유지하면서 배경 기울기를 낮추는 대조이며,
관측에 맞춘 보정이 아니다. 전체 scalar determinant의 정확한 Lorentzian 해석은
이번 두 미분 안정성 검사에서 완료로 세지 않는다.

## 49.2 두 성공 결과를 합친 작용

CE-GF2의 coherent 선은 정상 계량
$ds_{\rm normal}^2=\tfrac12\sum_a[(dq^a)^2+(dp_a)^2]$를 갖는다.
CE-GN4의 균등 gap 상대 1-loop 결과는 따라서 Euclidean 부호에서

$$
\Gamma_{\sigma,E}=\frac C2\int d^4x\sum_a
[(\partial_\mu q^a)^2+(\partial_\mu p_a)^2],\qquad
C=C(a,b)>0
$$

다. 내부공간이 네 oscillator로 바뀌어도 이 유한 접선의 균등 gap 이차 계수는
같은 두 질량 쌍 계산을 적용할 수 있다. 절대 무한 determinant의 존재를 주장하지 않는다.
Lorentzian 두 미분 작용을 표준 scalar 부호로 이어 쓰는 결합 후보는

$$
\mathcal L=-\frac1{4g_A^2}F_{\mu\nu}F^{\mu\nu}
-\frac C2\eta^{\mu\nu}\sum_a
(\partial_\mu q^a\partial_\nu q^a+\partial_\mu p_a\partial_\nu p_a).
$$

내부 $a$ 합은 양의 Hilbert 계량으로 한다. 이를 Lorentzian 벡터 지수의 수축으로
바꾸면 다른 이론이므로 부호를 임의로 바꾸지 않는다.
Maxwell 항·계량은 공급한 입력, $C$는 선언한 scalar loop의 상대 계수다.

## 49.3 프레임 위상 대칭과 Maxwell 중복성은 다르다

고정 $q$에서 $p\mapsto p+\Delta p$인 두 선 상태는

$$
|\langle v(q,p)|v(q,p+\Delta p)\rangle|^2
=e^{-|\Delta p|^2/2},\qquad
\|P(q,p+\Delta p)-P(q,p)\|_{\rm HS}^2
=2(1-e^{-|\Delta p|^2/2})
$$

를 만족한다. 일반적으로 같은 사영의 위상 차이가 아니다.
CE-GF2에서 Maxwell gauge 변환을 $\Delta p_a=(J^{-1})_a{}^\mu\partial_\mu\lambda$로
표시했을 때, 이는 이제 실제 projector를 바꾸는 변환이다.

예를 들어 Euclidean 국소 coframe $q=x/\ell$와 $p=0$에서
$p_a=\ell\partial_a\lambda$로 바꾸면 $F$는 계속 0이지만,

$$
\Delta\Gamma_{\sigma,E}
=\frac{C\ell^2}2\int d^4x\sum_{\mu,a}
(\partial_\mu\partial_a\lambda)^2
$$

이다. compact 지지의 비자명 $\lambda$에서는 양수다. 그러므로 이 변환은 결합
작용의 중복성이 아니다. $A$만 같은 gauge 궤도라고 하여 상태를 동일시할 수 없다.

반면 $v\mapsto e^{-i\lambda}v$는 $P$를 유지하는 프레임 기저 위상변환이며
그 대칭은 여전히 가능하다. 별도 위상 변수를 더해 형식적인 $U(1)$ 중복성을
표시하는 것과, 결합 작용에서 추가 $p$ 자유도를 제거하는 것은 다른 문제다.
보완 공간까지 변환하는 더 큰 대칭을 도입할 수도 있지만, 그 경우 전체 미분 작용과
연결을 함께 바꿔야 하며 현재 후보에는 그런 법칙을 넣지 않았다.

## 49.4 진공 배경의 속도 Hessian

공급한 평탄 배경에서 $q^a=x^a/\ell$, $p=0$를 취한다.
두 미분 scalar 식에서 이 $q$는 $\Box q^a=0$을 만족한다.
그 배경의 응력이 Einstein 식까지 만족한다는 주장은 하지 않는다.

$q=x/\ell+\delta q$, $p=\delta p$로 전개하면 Maxwell의 일차 퍼텐셜은
$A_\mu=p_\mu/\ell$이다. $p_a\partial_\mu\delta q^a$는 이차여서 Maxwell quadratic
작용에는 들어가지 않고, scalar 항에서도 $q,p$가 분리된다.
따라서 모든 변분을 허용해도 $p$의 quadratic 부문을 독립적으로 검사할 수 있다.

$\kappa=1/(g_A^2\ell^2)$라 하면

$$
\mathcal L_p^{(2)}=
\frac\kappa2\sum_i(\dot p_i-\partial_i p_0)^2
-\frac\kappa4\sum_{i,j}(\partial_i p_j-\partial_jp_i)^2
+\frac C2\sum_{a=0}^3[\dot p_a^2-(\nabla p_a)^2].
$$

속도 Hessian은 $\operatorname{diag}(C,C+\kappa,C+\kappa,C+\kappa)$다.
Maxwell만 있을 때 rank 3이었던 행렬이 $C>0$에서 rank 4가 된다.
$p_0$가 독립적으로 움직이므로 이전 Gauss 제약을 그대로 부과할 수 없다.
추가 $q$ 변분은 이 quadratic 부문과 분리돼 그 손실을 복구하지 않는다.

## 49.5 종방향 분산식과 불안정 영역

$z$ 방향 파수 $k$를 택한다. 두 횡방향 모드는 $\Omega^2=k^2$다.
시간 성분·종방향 성분의 운동방정식은

$$
C\ddot p_0-\kappa\partial_z\dot p_z+(\kappa-C)\partial_z^2p_0=0,
$$

$$
(C+\kappa)\ddot p_z-\kappa\partial_z\dot p_0-C\partial_z^2p_z=0.
$$

평면파를 대입한 2×2 determinant는

$$
C(\Omega^2-k^2)[(C+\kappa)\Omega^2-(C-\kappa)k^2]=0.
$$

따라서 $C>0$에서 추가 두 종방향 모드는
$\Omega^2=k^2$와 $\Omega^2=(C-\kappa)k^2/(C+\kappa)$다.
첫 모드는 Maxwell 관점에서는 순수 gauge 방향일 수 있지만 지금 작용에서는
scalar 운동항이 있으므로 제거할 수 없다.

$0<C<\kappa$이면 두 번째 모드의 주파수제곱이 음수이고
성장률 $\gamma=|k|\sqrt{(\kappa-C)/(\kappa+C)}$를 갖는다.
종방향 Hamiltonian의 공간 기울기 항도 같은 문제를 드러낸다.

$$
\mathcal H_{0z}=
\frac C2\dot p_0^2+\frac{C+\kappa}2\dot p_z^2
+\frac{C-\kappa}2(\partial_zp_0)^2+\frac C2(\partial_zp_z)^2.
$$

속도 Hessian이 양수라는 것만으로 안정성이 보장되지 않는다.
$C>\kappa$에서는 이 두 미분 부문의 해당 기울기 불안정이 사라지지만,
추가 자유도가 남아 순수 Maxwell 이론과 같아지지는 않는다.
$C=\kappa$는 추가 모드의 공간 복원항이 0인 경계이며 gauge 제약을 복구하는 지점이 아니다.
$C=0$은 Hessian 자체가 퇴화하는 별도 Maxwell 극한이다.

## 49.6 산출과 미분 전개의 적용 범위

[계산 코드](../../verify/ce_maxwell_quantum_collision.py)와
[고정 산출·소스 해시](../../verify/ce_maxwell_quantum_collision.json)를 사용한다.

```powershell
python -B verify/ce_maxwell_quantum_collision.py
```

**[산출]** $a=1,b=5$에서 $C=0.006257866129291753$이다.
$\kappa=1$에서 추가 속도제곱은 $-0.9875621024392812$이며,
$k=0.01,0.05,0.1$의 성장률은 각각
0.009937615924, 0.049688079618, 0.099376159235다.
네 $C$ 선택·세 파수의 12개 사례에서 별도 일차 시간 진화 행렬의 고윳값과
해석적 분산식의 차이는 $5.83\times10^{-15}$ 이하였다.

이 수치는 **잘린 quadratic 두 미분 작용**의 결과다.
$\ell=1$만으로는 느린 프레임 배경을 보장하지 않는다.
별도 등록한 $\ell=100,g_A=0.01$은 같은 $\kappa=1$을 가지면서 배경 기울기를
0.01로 낮춘다. 일반적으로 $g_A\ell$을 고정하고 $\ell\sqrt a\to\infty$,
$k/\sqrt a\to0$인 형식적 영역에서는 같은 선도 분산식과 느린 배경을 양립시킬 수 있다.
이 매개변수 선택을 실제 전자기 결합상수의 예측이나 관측 적합으로 세지 않는다.
전체 비국소 양자 작용·고차 미분·중력 되먹임까지 계산했다는 뜻도 아니다.

coherent 겹침의 네 대조도 정규화·폐형식과 $10^{-12}$ 이내로 일치했다.
$|\Delta p|=1$이면 겹침제곱은 0.6065306597, 사영 차이의 HS 노름제곱은
0.7869386806으로, 위상 차이만이라는 해석과 명확히 다르다.

## 49.7 판정과 다음 결합 조건

**기각:** CE-GF2의 Maxwell 작용에 기존 양의 projector 운동항을 단순 합산해도
Maxwell의 중복성·자유도·안정성이 자동으로 유지된다는 판본.
서로 다른 앞선 검산의 통과를 전체 공통 이론의 통과로 더할 수 없다.

다음 후보는 관측 퍼텐셜의 gauge 변화와 전체 상태공간에서의 물리적 동일성을
같은 작용에 구현해야 한다. 추가 항이 $A$와 gauge 불변 조합으로만 닫히는지,
아니면 실제 추가 상태·제약·보완 연결이 필요한지를 계산 전에 명시해야 한다.
$C$를 관측마다 지우거나 맞추어 실패를 숨기지 않는다.
공통 스칼라 작용이 주는 유한 상대 계수와 CE-GF2의 단독 Maxwell 재표현은
각자의 조건부 결과로 보존한다. 비동기 관측 선택·네 힘의 독립 동역학·Einstein 극한과
전체 공동 RMSE는 계속 미완성이다.
