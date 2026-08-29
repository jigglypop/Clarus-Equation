# 25. 공유 영수증의 먼지--상수진공 FLRW

24장에서 따로 전파한 먼지와 상수 진공을 한 우주에 함께 놓으려면, 같은 기록 에너지를 두 번 쓰지 않았음을 먼저 증명해야 한다. 이 장은 영수증 분할을 먼저 고정하고 homogeneous flat FLRW의 혼합 해를 재현한다. 순서는 **영수증 분할 → 연속 방정식 → 팽창 해 → 아직 없는 물리**다.

## 25.1 한 영수증을 두 지갑에 복사할 수 없는 이유

기록 총에너지를 $E_{\rm record}$, 먼지와 진공의 배정량을 각각 $E_m,E_\Lambda$라 하자. QD-M5-M1은 서로소 allocation identifier와

$$
E_{\rm record}=E_m+E_\Lambda,\qquad E_{\rm unassigned}=0
\tag{1}
$$

가 있을 때만 두 channel을 함께 사용한다. 진공 전이는 외부 battery가 아니라 정확히 남은 $E_\Lambda$에서 구성한다. 물리 부피 $V$에서

$$
\rho_\Lambda V=E_{\rm transferred},\qquad
T_{\hat\mu\hat\nu}=\operatorname{diag}(\rho_\Lambda,-\rho_\Lambda,-\rho_\Lambda,-\rho_\Lambda)
\tag{2}
$$

를 검사한다. 작은 수를 $0$으로 취급해 가짜 진공을 통과시키지 않도록 상대-스케일 불변식도 쓴다.

**문과 비유.** 한 장의 식비 영수증을 점심 지출과 저축 이체로 나눌 수는 있다. 그러나 영수증 전체를 점심에 쓴 뒤 같은 전체 금액을 저축했다고 적으면 돈이 생긴 것이 아니라 장부를 두 번 센 것이다. 식 (1)은 물리 이론의 완전한 기원이 아니라 그 회계 오류를 막는 규칙이다.

## 25.2 분할 뒤의 혼합 FLRW 해

$a(t)$는 scale factor, $H=\dot a/a$는 Hubble rate, $x=a/a_*>0$는 무차원 팽창비다. homogeneous comoving pressureless dust, channel 사이 교환 없음 $Q=0$, flat expanding FLRW, global constant covariant vacuum action을 **별도로 채택**한다. 이 네 조건 아래 QD-M5-M2는 양의 초기 밀도 $\rho_{m*},\rho_{\Lambda*}>0$에 대해

$$
\begin{aligned}
\rho_m(x) &= \rho_{m*}x^{-3}, && \text{먼지는 부피와 함께 희석한다},\\
\rho_\Lambda(x) &= \rho_{\Lambda*}, && \text{진공 밀도는 상수다},\\
H^2(x) &= \frac{8\pi G}{3}\bigl(\rho_m(x)+\rho_\Lambda(x)\bigr),\\
p(x) &= -\rho_\Lambda.
\end{aligned}
\tag{3}
$$
여기서 첫 줄은 $\dot\rho_m+3H\rho_m=0$을 적분한 것이고, 둘째 줄은 global action의 입력에서 온다. 따라서 식 (3)은 기록 하나에서 나온 정리가 아니라 명시한 배경 가정 아래의 **조건부 정리**다.

$$
f=\frac{\rho_{m*}}{\rho_{m*}+\rho_{\Lambda*}},\quad 0<f<1,\qquad
H^2(x)=H_*^2\left[f x^{-3}+1-f\right].
\tag{4}
$$

여기서 $H_*^2=8\pi G(\rho_{m*}+\rho_{\Lambda*})/3$다. 등밀도 시점은 $x_{\rm eq}=(f/(1-f))^{1/3}$이고, 가속도 전환은 $\rho_m=2\rho_\Lambda$이므로 $x_{\rm acc}=(f/[2(1-f)])^{1/3}$다. 효과적 상태방정식은

$$
w_{\rm eff}(x)=-\frac{\rho_\Lambda}{\rho_m(x)+\rho_\Lambda}.
\tag{5}
$$

**문과 비유.** 먼지는 같은 양의 잉크를 점점 큰 종이에 펴 바르는 것이라 옅어진다. 상수 진공은 종이가 커질수록 잉크 총량도 함께 늘지만, 그 증가를 압력 $p=-\rho$가 맞추는 경우다. 어느 잉크가 처음 얼마나 있었는지는 이 비유도, 이 계산도 정하지 않는다.

## 25.3 시간 재구성과 열린 다리

$$
\Delta t=\frac{2}{3H_\Lambda}
\left[\operatorname{asinh}\!\left(\sqrt{\frac{\rho_{\Lambda*}}{\rho_{m*}}}\,x^{3/2}\right)
-\operatorname{asinh}\!\sqrt{\frac{\rho_{\Lambda*}}{\rho_{m*}}}\right],
\qquad H_\Lambda^2=\frac{8\pi G\rho_{\Lambda*}}{3}.
\tag{6}
$$

식 (6)은 $0<f<1$의 mixed interior에만 쓴다. 순수 먼지와 진공 끝점은 24장의 별도 branch다. 구현은 [partitioned_dark_sector_flrw.py](../../examples/physics/partitioned_dark_sector_flrw.py)와 [대응 테스트](../../tests/test_partitioned_dark_sector_flrw.py)에 있고 focused 결과는 11 passed다. 이는 구현 재현이지 우주 파라미터의 측정이 아니다.

QD-M5-M3의 경계는 남는다. $f$와 절대 밀도 척도는 입력이고, global vacuum action은 한 slice의 영수증에서 유도되지 않는다. 미시 선택법칙, renormalized covariant $T_{\mu\nu}$, perturbation과 structure growth, CE 고유의 독립 예측도 아직 없다. 다음 [26장](26_실제_Euclidean_Regge_1_to_5_내부_Hessian과_경계_Schur.md)은 실제 4차원 이산 기하의 다른 열린 다리로 간다.
