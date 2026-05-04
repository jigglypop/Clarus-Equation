# 뇌 방정식 특수 연산자 수학 정리

## 1. 전역식

현재 뇌 전역 방정식 후보는 모든 region 상태를 쌓은 \(P_n\)에 대해 다음 꼴이다.

$$
P_{n+1}
=
\Pi_{\mathcal S}
\left[
(1-\rho_B)P^*
+\rho_B P_n
+\gamma\Delta_G P_n
+U_n
+H(Q_n-Q^*)
+F_{\mathrm{syn},n}
+F_{\mathrm{slow},n}
\right]
$$

여기서 \(\Pi_{\mathcal S}\)는 각 region의 3성분 상태를 simplex로 되돌리는 투영이다.

$$
p_{r,n}=(x_{a,r,n},x_{s,r,n},x_{b,r,n}),
\qquad
x_a+x_s+x_b=1,\quad x_i\ge0
$$

3성분은 사용자가 말한 것처럼 3차원 상태좌표다. 현재 해석은 다음이다.

| 성분 | 의미 |
|---|---|
| \(x_a\) | active/task-responsive |
| \(x_s\) | structural/synaptic burden |
| \(x_b\) | background/reserve |

## 2. 특수 영역 입력항 분해

과제 입력 \(U_n\)은 하나의 스칼라가 아니라 domain별 입력항의 합으로 둔다.

$$
U_n
=
\sum_{d\in\mathcal D} a_{d,n} U^{(d)}(z_n)
$$

여기서 \(d\)는 감각 또는 기능 domain이다.

$$
\mathcal D
=
\{
\mathrm{visual},
\mathrm{auditory},
\mathrm{pain},
\mathrm{face/emotion},
\mathrm{cognitive/control},
\mathrm{vigilance},
\mathrm{working\ memory},
\mathrm{arousal}
\}
$$

각 domain 입력은 region-family mask와 event feature의 곱으로 쓸 수 있다.

$$
U^{(d)}(z_n)
=
M^{(d)} B^{(d)} \phi_d(z_n)
$$

- \(M^{(d)}\): 해당 domain이 recruit하는 region mask
- \(B^{(d)}\): event feature를 3성분 상태 변화로 보내는 계수
- \(\phi_d(z_n)\): target rate, pain rating, KSS, response time 같은 관측 feature

즉 전역식은 하나지만, 실제 작동은 domain별 \(M^{(d)},B^{(d)},\phi_d\)가 달라진다.

## 3. 선형화

기준상태 근방에서 \(Y_n=P_n-P^*\)로 두면, 투영이 경계에 걸리지 않는 영역에서는 1차 근사가 가능하다.

$$
Y_{n+1}
\approx
A_G Y_n
+\sum_d a_{d,n} U^{(d)}(z_n)
+H(Q_n-Q^*)
+F_{\mathrm{syn},n}
+F_{\mathrm{slow},n}
$$

여기서

$$
A_G=\rho_B I+\gamma\Delta_G
$$

이고 \(\Delta_G=-L_G\)이면, \(L_G v_k=\lambda_k v_k\)에 대해 mode별 증폭률은

$$
\alpha_k=\rho_B-\gamma\lambda_k
$$

따라서 기본 안정 조건은 다음이다.

$$
\max_k|\rho_B-\gamma\lambda_k|<1
$$

이 조건은 충분조건이다. 실제 데이터에서는 여기에 입력항의 유계성도 같이 봐야 한다.

## 4. matched-vs-wrong 부등식 유도

event-level gate에서 관측 상태를 \(s_i\)라 하고, domain \(d\)의 prototype을

$$
\mu_d
=
\frac{1}{|C_d|}
\sum_{i\in C_d}s_i
$$

로 둔다.

matched 손실:

$$
\mathcal L_{\mathrm{matched}}
=
\sum_i
\|s_i-\mu_{d_i}\|^2
$$

wrong 손실:

$$
\mathcal L_{\mathrm{wrong}}
=
\sum_i
\min_{d\ne d_i}
\|s_i-\mu_d\|^2
$$

generic 손실:

$$
\mathcal L_{\mathrm{generic}}
=
\sum_i
\|s_i-\bar\mu\|^2
$$

여기서 \(\bar\mu\)는 전체 평균 prototype이다.

domain 내부 분산을

$$
\bar W_d
=
\frac{1}{|C_d|}
\sum_{i\in C_d}
\|s_i-\mu_d\|^2
$$

prototype 사이 거리를

$$
B_{d,d'}=\|\mu_d-\mu_{d'}\|^2
$$

라고 하자. 최소 분리도가

$$
\delta^2=\min_{d\ne d'}B_{d,d'}
$$

이고 평균 내부분산이

$$
\bar W=\frac{1}{|\mathcal D|}\sum_d\bar W_d
$$

이면, readiness의 최소 필요조건은

$$
\bar W<\delta^2
$$

이다. 더 강하게, 각 점이 자기 prototype 주변 반지름 \(\epsilon\) 안에 있고 prototype 간 거리가 \(2\epsilon\)보다 크면 nearest-prototype matched 판정은 안정하다.

$$
\|s_i-\mu_{d_i}\|\le\epsilon,
\qquad
\|\mu_d-\mu_{d'}\|>2\epsilon
$$

그러면 삼각부등식으로

$$
\|s_i-\mu_{d'}\|
\ge
\|\mu_{d_i}-\mu_{d'}\|-\|s_i-\mu_{d_i}\|
>
\epsilon
\ge
\|s_i-\mu_{d_i}\|
$$

이므로 matched가 wrong보다 작다.

## 5. holdout이 필요한 이유

위 부등식은 같은 데이터에서 prototype을 만들면 너무 쉽게 통과할 수 있다. 그래서 subject holdout을 둔다.

$$
\mu_d^{(-s)}
=
\frac{1}{|C_d^{(-s)}|}
\sum_{i\in C_d,\ \mathrm{subj}(i)\ne s}
s_i
$$

holdout 손실은

$$
\mathcal L_{\mathrm{matched}}^{(s)}
=
\sum_{i:\mathrm{subj}(i)=s}
\|s_i-\mu_{d_i}^{(-s)}\|^2
$$

로 정의한다. 통과 조건은 모든 평가 가능한 holdout subject에 대해

$$
\mathcal L_{\mathrm{matched}}^{(s)}
<
\min
\left(
\mathcal L_{\mathrm{wrong}}^{(s)},
\mathcal L_{\mathrm{generic}}^{(s)}
\right)
$$

이다.

이 부등식이 의미하는 것은 단순 label 기억이 아니라, 한 subject에서 얻은 domain 구조가 다른 subject에도 유지된다는 것이다.

## 6. 지금 더 유도할 수 있는 것

데이터 없이도 더 밀 수 있는 수학은 세 가지다.

1. **분리 가능성 정리**  
   \(\bar W/\delta^2\)가 작을수록 matched-vs-wrong gate의 안정성이 커진다.

2. **식별성 조건**  
   실제 \(P_n\to P_{n+1}\) 전이에서 계수 \(\Theta\)를 역추정하려면 design matrix가 full rank여야 한다.

   $$
   \Delta P_n
   =
   X_n\Theta+\varepsilon_n,
   \qquad
   \mathrm{rank}(X)=\dim(\Theta)
   $$

3. **그래프 항 필요조건**  
   \(\gamma\Delta_G P_n\)는 flat, shuffled graph, degree-preserving graph보다 holdout 손실이 낮아야 한다.

   $$
   \mathcal L_{\mathrm{graph}}
   <
   \min(
   \mathcal L_{\mathrm{flat}},
   \mathcal L_{\mathrm{shuffled}},
   \mathcal L_{\mathrm{degree}}
   )
   $$

## 7. 현재 닫힌 수학적 결론

현재 부분 데이터 게이트들이 닫은 것은 다음이다.

$$
\boxed{
\bar W<\delta^2
\quad\mathrm{and}\quad
\mathcal L_{\mathrm{matched}}
<
\min(
\mathcal L_{\mathrm{wrong}},
\mathcal L_{\mathrm{generic}}
)
}
$$

즉 특수 연산자 \(U^{(d)}\)를 둘 수 있는 event-level 필요조건은 통과했다.

아직 닫히지 않은 식은 다음이다.

$$
\boxed{
\mathcal L_{\mathrm{full}}(P_{n+1}^{\mathrm{BOLD/EEG}})
<
\mathcal L_{\mathrm{best\;ablation}}
}
$$

이 마지막 부등식이 실제 region-resolved \(p_r\)에서 통과해야 신경 방정식 검증으로 올릴 수 있다.
