# CE-RI1 — 기록 스펙트럼에서 공통 반응과 계수로, 그리고 자연값 선택의 경계

연구일: 2026-09-20
기준 원격 main: `8721ab787a3848bddc6cfaf8f8c3622479d62639`
선행: CE-CR1 공간 기록, CE-RV1 기록별 공동 상태와 미래 값
게시 상태: 로컬 후속 장 및 추가 전용 패치. 원격 main 변경 없음.

## 0. 결론과 계산의 층위

이번에는 기록의 분해능을 임의로 더 높이는 대신, 같은 잔여장에 이미 존재하는 시간·공간 상관을 읽었다. 두 가지 역구성을 실제로 수행했다.

1. 고정 힉스 배경의 4차원 Gaussian 장에서는 요동 스펙트럼의 문턱과 크기가 질량분할·포털 결합을 대수적으로 식별한다. 동일한 스펙트럼에서 소산과 운동량 의존 이차 응답이 정해지고, 위상 간 차이는 원래 CE의 U_ss로 돌아온다.
2. 별도의 CE-RV1 유한모드 공동 양자계에서는 두 기록의 모멘트와 독립적인 가속도로 힉스 국소 계수를 식별하고, 세 번째 기록의 가속도를 다시 조절하지 않고 계산한다.

이것은 **합성 기록에서 이미 지정한 계수를 복원하는 정합성 시험**이다. 실험 기록을 받지 않았으며 실제 자연상수의 값을 예측하거나 관측 오차를 개선하지 않았다. 두 계산의 정규화와 차원이 다르므로 동일한 완성 QFT 데이터 세트라고 합치지 않는다.

자연값 선택도 별도로 검사했다. 단순 Schur 제거의 양의 포털 생성, 스칼라만의 4차원 한 루프 비영 고정점, 위상 퍼텐셜만으로 반지름 선택은 현재 가정에서 성립하지 않았다. 미지의 동역학을 이미 알아냈다고 포장하지 않고, 다음 미시 모형이 만족해야 할 조건으로 남긴다.

새 실행은 **36개 검사**다. 일부는 연산자·부호·정규화 항등식 검사이고, 일부는 독립 적분·행렬·시간발전 대조와 음성대조다. 36개가 36개의 독립적인 관측 증거라는 뜻은 아니다. 선행 논문들의 모든 검사를 재실행했다고 하지 않는다.

## 1. 유지한 출발식과 추가 조건

원래 질량제곱 행렬은

\[
X(u)=(s_0+\kappa u)I+Y(\theta),\qquad
Y=\epsilon(e^{i\theta/3}S+e^{-i\theta/3}S^\dagger),\quad S^3=I.
\]

차원 있는 공통 epsilon을 유지한다. 고정 배경의 최소위상 theta=pi에서는

\[
x_L=s_0+\kappa u-2\epsilon,\qquad
x_H=s_0+\kappa u+\epsilon\quad(\text{중복도 }2).
\]

여기서 x는 질량제곱이며 m=sqrt(x)다. s0>2epsilon>0, kappa>0를 사용한다. kappa=0은 음성대조로 따로 다룬다. 위상이 pi라는 선택은 선행 결과에서 가져온 조건이지 이번에 모든 동적 Y를 다시 해결한 것이 아니다.

장 계산의 조건은 평탄 3+1차원, hbar=c=1, 정준화된 세 복소 자유 매개장, 일정한 힉스 불변량 u, 그 배경에 대한 진공이다. O=sum chi_j^dagger chi_j를 사용하며 두점함수에서는 평균을 뺀 연결 성분을 취한다. 포털에 대한 일반화 원천은 F=-kappa O다.

O는 자유 Gaussian 장의 **이차 연산자**이므로 O 자체가 Gaussian 확률변수라고 가정하지 않는다. 아래 스펙트럼은 자유장 배경에서 정확한 연결 두점함수다. 이를 이용해 전체 영향작용을 잡음·소산의 이차식으로 쓰는 것은 원천 변형에 대한 이차 응답이며, 모든 고차 누적량을 없앤 완전 영향함수가 아니다.

기록 장치는 이 상관의 에너지·운동량 및 절대 정규화를 읽을 수 있어야 한다. 여기서는 특히 공간 운동량 q=0 성분을 사용한다. 단일 국소 검출기의 시간 기록과 q=0의 부피 정규화된 스펙트럼을 같은 것으로 놓지 않는다. 실제 장치 구현과 유한 표본의 문턱 추정 오차는 이번에 계산하지 않았다.

## 2. 기존 잔여 연산자의 스펙트럼

Fourier 규약을

\[
\mathcal P_u^>(\omega,\mathbf q)=\int d^4x\,
e^{i\omega t-i\mathbf q\cdot\mathbf x}\langle F(x)F(0)\rangle_c
\]

로 둔다. 복소장 한 종에 대한 Wick 수축은 두 양의 주파수 전파함수의 곱이다. q=0에서

\[
\mathcal P_{u,j}^>(\omega,0)
=\kappa^2\int\frac{d^3p}{(2\pi)^3}\frac{2\pi}{4E_{p,j}^2}
\delta(\omega-2E_{p,j}),\qquad E_{p,j}=\sqrt{p^2+x_j}.
\]

각도 적분 후 p에 대한 delta 함수를 처리하면

\[
\boxed{\mathcal P_u^>(\omega,0)=\frac{\kappa^2}{8\pi}
\sum_j\sqrt{1-\frac{4x_j}{\omega^2}}\,
\Theta(\omega-2\sqrt{x_j}).}
\]

이 1/(8pi) 정규화는 두 입자 위상공간에서 온다. 모드·복소 성분을 이중으로 세지 않는다. 독립 검산은 delta를 좁은 정규화 Gaussian으로 바꾼 **원래 운동량 적분**으로 수행했다. Wick 전개 및 위상공간은 표준 QFT 도구[R1], 환경 스펙트럼에서 잡음과 소산을 읽는 방법은 [R2]의 일반 틀과 연결된다. CE 고유 부분은 위에 대입한 공통 질량구조다.

진공에서 P^>(omega)=0 for omega<0이고, 첫 문턱 아래에서도 0이다. 따라서 무거운 진공 환경에 저주파 백색 잡음을 임의로 부여할 수 없다. 이는 작은 조화 구동의 선형 응답 또는 엄밀히 문턱 아래에 Fourier 지지를 둔 시험원에 대한 사실이다. 유한시간 스위칭의 고주파 꼬리, 열적 상태, 다중양자 과정이나 고차 응답까지 전부 없다는 뜻은 아니다.

## 3. 기록 문턱에서 s0, epsilon, kappa를 복원

최소위상에서 두 문턱을 T_L=2sqrt(x_L), T_H=2sqrt(x_H)라고 읽는다. 그러면

\[
\boxed{\epsilon=\frac{T_H^2-T_L^2}{12}},\qquad
\boxed{s(u)=\frac{T_L^2+2T_H^2}{12}}.
\]

서로 다른 알려진 원천 u_a,u_b의 기록에서

\[
\boxed{\kappa=\frac{s(u_b)-s(u_a)}{u_b-u_a}},\qquad
\boxed{s_0=s(u_a)-\kappa u_a}.
\]

두 문턱 사이의 한 주파수에서 스펙트럼의 절대 크기를 사용하면 독립적으로

\[
\boxed{\kappa^2=\frac{8\pi\mathcal P_u^>(\omega)}
{\sqrt{1-T_L^2/\omega^2}}},\qquad T_L<\omega<T_H.
\]

잡음 크기만으로는 kappa의 부호를 못 읽지만, 질량 문턱의 원천 기울기는 부호를 정한다. 동일 epsilon 가설의 추가 예측은

\[
\frac{dT_L^2}{du}=\frac{dT_H^2}{du}=4\kappa,\qquad
\frac{d(T_H^2-T_L^2)}{du}=0.
\]

고주파에서 첫 종의 기여를 제거하면 둘째 종의 중복도 2도 독립적으로 확인할 수 있다. 따라서 문턱 두 개를 읽었다고 무조건 CE라고 선언하지 않고, 기울기·중복도·전체 함수형까지 함께 대조한다.

### 합성 기록의 결과

기존 진단값 s0=.5, epsilon=.15, kappa=1을 넣어 만든 기록을 역으로 읽었다. 입력을 숨겨서 실험으로 새로 발견했다는 뜻이 아니다.

| u | T_L | T_H |
|---:|---:|---:|
| .2 | 1.264911064067 | 1.843908891459 |
| .8 | 2.000000000000 | 2.408318915758 |

복원값은 s0=.5000000000000001, epsilon=.15000000000000002, kappa=.9999999999999998이다. 절대 잡음 크기로는 kappa=1, 두 번째 중복도는 2.000000000000002로 회수됐다. 최적화나 관측별 피팅은 하지 않았다. 그러나 이는 **식별**이지 이 숫자들의 **발생 원인**을 설명하는 선택 정리가 아니다.

또한 시간·u·검출 세기를 정규화할 기준이 필요하다. omega->a omega, x->a^2 x에서 잡음의 무차원 모양은 같으므로, 상대적인 모양만으로 GeV 같은 절대 단위를 만들 수 없다. epsilon=0에서도 기록은 존재한다는 선행 반례도 유지한다.

## 4. 같은 기록으로 잡음·소산·운동량 의존 응답을 묶기

P^<(omega)=P^>(-omega), rho=P^>-P^<, N=(P^>+P^<)/2로 정의한다. retarded **상관함수**의 규약을

\[
G_R(t)=-i\Theta(t)\langle[F(t),F(0)]\rangle
\]

로 두면 rho=-2 Im G_R이고, 영온에서는

\[
\boxed{N(\omega)=-\operatorname{sgn}(\omega)\operatorname{Im}G_R(\omega).}
\]

이는 부호 규약이 명시된 요동–소산 관계다. 외부 힘을 Hamiltonian에 더하는 부호에 따라 실제 응답식 앞의 부호는 함께 정해야 한다. 잡음과 감쇠 함수를 따로 자유롭게 고르지 않는다. 표준적인 영향작용·스펙트럼 방법은 [R2] 참조.

포털 kappa^2를 제외한 O의 Kallen–Lehmann 밀도를 t=M^2에 대해

\[
\rho_O(t)=\frac1{16\pi^2}\sum_j\sqrt{1-4x_j/t}\,\Theta(t-4x_j)
\]

로 두면, Euclidean 두점함수의 **차감된** 부분은

\[
\boxed{B_E(Q^2)-B_E(0)=\int_0^\infty dt\,\rho_O(t)
\left[\frac1{t+Q^2}-\frac1t\right].}
\]

각 종의 Feynman 매개변수 계산은

\[
B_{E,j}(Q^2)-B_{E,j}(0)
=-\frac1{16\pi^2}\int_0^1 da\,\ln\left[1+\frac{a(1-a)Q^2}{x_j}\right].
\]

두 적분을 독립적으로 대조한 최대 차이는 3.47e-18이었다. 낮은 운동량의 계수도

\[
-\left.\partial_{Q^2}B_E\right|_0
=\frac1{96\pi^2}\sum_j\frac1{x_j}
\]

로 이어진다. 이로써 주어진 Gaussian 매개장이 만드는 **운동량 의존 이차 응답**은 기록 스펙트럼으로 재구성된다. B_E(0) 자체의 국소 차감상수, 원래 힉스 운동항의 기준값, 다른 고차 국소 EFT 항이 전부 사라진다는 주장은 하지 않는다.

## 5. 원래 CE의 U_ss와 실제로 연결되는 합규칙

동일한 s,epsilon에서 theta와 pi를 비교하면 원래 두 모멘트 sum x, sum x^2는 같다. 따라서 상대 스펙트럼의 적분에서는 공통 국소 모호성이 상쇄된다.

\[
\boxed{\mathcal U_{ss}(s,\epsilon,\theta)
=-\frac1{\pi\kappa^2}\int_0^\infty\frac{d\omega}{\omega}
[\mathcal P_\theta^>(\omega)-\mathcal P_\pi^>(\omega)].}
\]

증명은 U_ss=-Delta B_E(0)와 t=omega^2를 이용한다. 결과는 원전의

\[
\mathcal U_{ss}=\frac1{16\pi^2}\ln\frac{s^3-3s\epsilon^2+2\epsilon^3\cos\theta}
{s^3-3s\epsilon^2-2\epsilon^3}
\]

와 같다. 원전의 게이지 보정 -T(R) U_ss/3 등은 이 동일한 상대 반응으로 돌아간다. 모든 원래 게이지군과 전하·표현이 이번에 생성됐다는 뜻은 아니다.

s=1,epsilon=.15,theta=1.2에서 직접 식은 6.259409149152256e-5, 기록 합규칙은 6.259409149152222e-5였다. 네 진단점의 최대 차이는 1.03e-18이다. 공통 큰 invariant-mass cutoff를 사용한 적분의 해석적 원시함수를 고정밀 계산했고, 수렴 잔차와 부동소수점 비교를 구분했다.

새로 묶은 것은 기록 -> 같은 매개장 스펙트럼 -> 원래 공통 반응이다. 개별 관측값을 맞추기 위한 새 자유함수를 추가하지 않았다.

## 6. 기록에서 초기 준비를 얼마나 알 수 있는가

고정 배경의 정상 Gaussian Gibbs 상태라는 가설 안에서는 q=0, omega>0의 쌍생성 연속부에 대해

\[
P_\beta^>(\omega)=P_0^>(\omega)[1+n_B(\omega/2)]^2,\qquad
P_\beta^<(\omega)=P_0^>(\omega)n_B(\omega/2)^2.
\]

그러므로

\[
\boxed{\beta=\frac1\omega\ln\frac{P_\beta^>(\omega)}{P_\beta^<(\omega)}}.
\]

열적 영주파수 산란항은 위 양의 주파수 쌍생성 식에서 제외했다. 상호작용에 의한 열질량은 이 자유장 진단에 넣지 않았다.

서로 다른 omega에서 동일 beta가 나오면 해당 정상 Gibbs 가설의 일관성을 검사할 수 있다. 합성 beta=2에서는 세 주파수 모두 2를 회수했다. 비열적 점유 반례에서는 각각 3.29456, 2.87168, 2.35741로 서로 달랐다. 기록을 무조건 열평형 상태로 읽는 처방은 통과하지 않는다.

모든 모드의 음의 주파수 쌍소멸 기록이 0이고, 영평균·정상 Gaussian·양의 질량 조건이 유지되면 n_j(k)^2의 비음수 합이 0이므로 점유가 0임을 제한적으로 판정할 수 있다. 그러나 유한 대역 기록에서 모든 초기 상태를 배제할 수는 없으며, 비정상 squeezing, 서로 다른 대칭 부문의 위상과 다점 상관까지 두점 기록 하나로 전부 복원할 수 없다.

## 7. 별도 공동 양자계: 힉스 계수도 두 기록에서 복원

이 절은 4차원 고정배경 장이 아니라 **선행 CE-RV1의 유한모드 signed Higgs proxy**다. 그 원래 Hamiltonian과 Gaussian 기록 연산을 그대로 불러왔다. 두 계산의 단위·정규화를 같은 것으로 합치지 않는다.

원래 정확한 Heisenberg 식은

\[
\ddot{\bar u}_r=\langle p_\varphi^2\rangle_r
-4\lambda(\langle u^2\rangle_r-u_0\bar u_r)
-2\kappa\langle uO\rangle_r.
\]

다음을 기록에서 계산한다.

\[
a_r=\langle u^2\rangle_r,\quad b_r=\bar u_r,\quad
q_r=\langle p_\varphi^2\rangle_r-2\kappa\langle uO\rangle_r-\ddot{\bar u}_r.
\]

그럼 q_r=A a_r+B b_r, A=4lambda, B=-4lambda u0이므로 두 독립 기록에서

\[
\begin{pmatrix}A\\B\end{pmatrix}
=\begin{pmatrix}a_1&b_1\\a_2&b_2\end{pmatrix}^{-1}
\begin{pmatrix}q_1\\q_2\end{pmatrix},\qquad
\boxed{\lambda=A/4,\quad u_0=-B/A}.
\]

이 역식은 원래 O의 기록 r 한 숫자만을 뜻하지 않는다. 동일하게 준비한 계의 반복 계측 또는 그 상태의 완전한 계산을 통해 u, u^2, p_varphi^2, uO와 시간 가속도에 접근해야 한다. 비가환하는 양들을 한 번의 계측으로 동시에 확정했다고 하지 않는다. 이번 합성 시험에서는 기존 공동 양자상태가 이 모멘트들을 공급했다.

행렬식이 비영이어야 한다. 이번 기록 쌍의 조건수는 약 47.9다. 실제 잡음 자료에는 오차 증폭이 있으므로 표의 긴 자릿수를 실험 정밀도로 읽지 않는다.

검사에서는 가속도를 위 모멘트 식으로 만든 뒤 같은 식을 거꾸로 푸는 순환 검산을 피했다. 전체 유한 Hamiltonian의 **직접 이중 교환자**로 가속도를 구했고, 한 기록에서는 독립 시간발전의 이차차분까지 비교했다.

| 기저 차원 | 복원 lambda | 복원 u0 | 쓰지 않은 r=2 가속도 오차 |
|---:|---:|---:|---:|
| 6,860 | 9.999999999875 | .500000000001 | 2.46e-13 |
| 17,496 | 9.999999999999 | .500000000000 | 1.24e-14 |

계수 복원은 r=.5,4만 사용했다. 세 번째 기록 r=2에 다시 계수를 맞추지 않았다. 독립 시간발전 가속도 검산 오차는 2.12e-9다. 여전히 **이미 지정한 Hamiltonian의 계수를 합성 기록으로 식별**한 것이다. 자연이 lambda=10이나 u0=.5를 선택한 이유의 증명이 아니다.

## 8. 전부 선택하기 위해 검사한 첫 경로: 위상 유한성을 반지름까지 확장

원전의 상대 유한성은 theta를 바꿀 때

\[
\operatorname{tr}X=3s,\qquad \operatorname{tr}X^2=3s^2+6\epsilon^2
\]

가 보존된다는 조건에 의존한다. 반면 epsilon과 s 자체를 선택 변수로 바꾸면 이 두 모멘트가 변한다.

4차원 hard cutoff를 쓴 복소 보손 행렬식의 장 의존 발산항은

\[
V_{1,\Lambda}=
\frac{\Lambda^2}{16\pi^2}\operatorname{tr}X
-\frac{\ln(\Lambda^2/\mu_*^2)}{32\pi^2}\operatorname{tr}X^2
+\text{장 독립항과 유한항}.
\]

따라서 같은 s에서 두 epsilon을 비교해도

\[
\frac{d\,\Delta V_1}{d\ln\Lambda}
\longrightarrow-\frac{6(\epsilon_2^2-\epsilon_1^2)}{16\pi^2}
\]

가 남는다. epsilon=.1,.2, s=1을 비교한 로그 기울기의 수치 오차는 4.52e-10이었다. 발산항을 피하려고 원하는 epsilon에 맞춰 임의 cutoff를 선택하지 않았다.

허용되는 국소항은 예를 들어

\[
\delta V=A_0+A_1u+A_2u^2+B\operatorname{tr}Y^2
\]

다. theta 비교에는 없어지지만 u나 epsilon에 대한 기울기에는 남는다. 그 유한 계수를 결정하는 미시 정합 또는 별도 경계조건 없이 상대 위상식만으로 모두를 선택할 수는 없다. 유효퍼텐셜의 국소 재규격화 구조는 표준적인 틀이며[R3], 여기서는 원래 CE의 보존 모멘트가 어떤 방향에서 더 이상 보존되지 않는지 계산했다.

고정된 u=.5의 같은 잔여 잡음 기록을 유지하면서 (lambda,u0)=(10,.5)와 (13,.3)의 국소 힘은 각각 0,-5.2였다. 따라서 **잔여의 고정배경 기록만**으로 힉스 국소 퍼텐셜까지 결정한다는 주장은 반례를 갖는다. 힉스의 동적 기록까지 추가하면 7절처럼 식별은 가능하지만, 계수의 기원을 설명한 것은 아니다.

## 9. 두 번째 선택 경로: 스칼라 RG 흐름이 비영 포털을 선택하는가

단순 최소에너지 조건을 반복하는 대신, 차원을 바꾸어 읽어도 유지되는 비영 고정점이 있는지 검사했다. 양의 정준 운동항을 가진 4차원 **스칼라만의** 진단이며, Higgs의 네 실수 성분 h와 세 복소 잔여장의 여섯 실수 성분 c를 사용한다. 부드러운 질량분할은 자외선의 이 질량독립 한 루프 계수에 영향을 주지 않는다. 게이지·Yukawa·중력·비섭동 항을 포함한 전체 CE의 고정점 정리는 아니다.

\[
V_4=\frac{\lambda_H}{4}(h^2)^2+
\frac{\lambda_C}{4}(c^2)^2+\frac{\kappa}{4}h^2c^2.
\]

한 루프 사차 텐서 수축에서

\[
16\pi^2\beta_{\lambda_H}=24\lambda_H^2+3\kappa^2,
\]
\[
16\pi^2\beta_{\lambda_C}=28\lambda_C^2+2\kappa^2,
\]
\[
16\pi^2\beta_\kappa=\kappa(12\lambda_H+16\lambda_C+4\kappa).
\]

각 성분을 일반 사차 텐서의 세 채널 수축으로 직접 계산해 대조했다. 가장 중요한 부호는 앞 두 식의 제곱합이다. 이 차수에서 beta들이 모두 0인 실수 해는 lambda_H=lambda_C=kappa=0뿐이다. 따라서 **이 순수 스칼라 한 루프 후보는 필요한 비영 포털의 값을 선택하지 못한다**. 다성분 스칼라의 고정점·차원 의존성은 [R4,R5]의 연구 문맥과 연결되지만, 해당 논문이 CE 전체를 배제했다는 뜻은 아니다.

더 중요한 점은 lambda_C=0이라도 kappa!=0이면 beta_lambdaC>0라는 사실이다. 포털을 켜고 잔여 사차항을 모든 척도에서 영으로 고정하는 것은 이 4차원 양자 확장에서 닫힌 처방이 아니다. 이것을 새 자유계수를 피팅할 허가로 쓰지 않는다. 필요한 사차 정합을 기존 미시원리가 제공해야 한다는 다음 조건으로 남긴다. 앞선 유한 진동자 계산은 자외선 연속 QFT가 아니므로 이 문제 때문에 그 수치 결과가 자동으로 무효화되는 것은 아니다.

## 10. 세 번째 선택 경로: 기회비용의 Schur 제거만으로 양의 포털이 나오는가

잔여 제거의 정확한 식 K_eff=A-BC^{-1}B^dagger에서 A,C가 Higgs와 무관하고 C>0, B=varphi L이라고 놓으면

\[
\boxed{\frac{\partial K_{\rm eff}}{\partial(\varphi^2)}=-LC^{-1}L^\dagger\preceq0.}
\]

u=varphi^2/2에 대한 계수는 이 식의 두 배로 여전히 비양수다. 즉 **이 제한된 기전만으로는 이전에 사용하던 양의 kappa 포털을 만들 수 없다**. 양의 kappa 자체가 물리적으로 금지된다는 말이 아니다. A,C의 Higgs 의존, 직접 정준 운동·대각 구조, 다른 양자 기여가 있으면 계산은 달라진다. 그런 항을 실제 공통 작용에서 도출해야 한다.

임의의 양의 C와 비영 L을 사용한 직접 행렬 계산에서도 비양수 고유값을 확인했다. '잔여의 완화 비용'이라는 언어만으로 포털의 부호·크기까지 얻었다고 하는 주장을 이번에 더 좁혔다.

## 11. 초기상태와 절대 진공값에 남는 정확한 비유일성

### 11.1 기록 대칭이 보지 못하는 초기 정보

유한 signed Higgs 대리좌표에서는 parity P:varphi->-varphi가 H,u,O 및 이들 기록 연산과 가환한다. 따라서 그 연산들로 만든 모든 시간 기록의 효과 E_R도 P와 가환한다.

짝수·홀수 상태 |e>,|o>로부터 |psi_+>=(|e>+|o>)/sqrt2, |psi_->=(|e>-|o>)/sqrt2를 만들면 두 상태는 직교하지만

\[
\langle\psi_+|E_R|\psi_+\rangle=
\langle\psi_-|E_R|\psi_-\rangle
\]

가 모든 기록 이력에서 성립한다. 비영 포털과 비가환 환경 운동을 가진 작은 유한 모형으로 서로 다른 세 기록열을 계산했고, 확률밀도 차이는 수치상 0이었다.

이는 현재 기록 대수로 초기 상태의 모든 위상정보를 식별할 수 없다는 결과다. 실제 표준모형에서 gauge로 같은 Higgs 부호를 서로 다른 관측 가능한 세계라고 부르는 증거가 아니다. 물리적인 상태 공간의 대칭 몫과 관측 대수를 먼저 정해야 한다.

### 11.2 절대 진공 에너지

평탄 배경에서 H->H+C I는 전체 시간발전에 공통 위상만 곱한다. 모든 정규화된 기록확률과 조건부 값은 같다. 따라서 이 기록만으로 절대 상수항을 결정할 수 없다. 실제 중력을 포함한 응력·곡률 기록은 추가 정보를 줄 수 있으나 이번 계산에 포함하지 않았다.

## 12. 현재 연구 판정과 다음 접합 조건

| 대상 | 이번에 얻은 것 | 아직 자연값 선택이 아닌 이유 |
|---|---|---|
| s0, epsilon, kappa | 같은 스펙트럼의 문턱·기울기·크기에서 대수 복원 | 합성 데이터, 고정 원천·단위·정규화·준비를 지정 |
| lambda, u0 | 공동 양자 기록 두 개에서 복원, 세 번째 반응 예측 | 기존 Hamiltonian을 생성하는 미시원리 미결정 |
| 조건부 분포와 미래 | CE-RV1 경로 유지, 상관 모멘트 보존 | 한 점 선택·Born 규칙 유도 아님 |
| 초기 열적 준비 | KMS 가설 아래 beta 및 진공 점유 판정 | 일반 초기상태·대칭 위상은 미식별 |
| 위상 | 선행 pi 최소점과 호환 | epsilon의 반지름 및 절대척도는 별도 |
| 잡음·소산·비국소 이차항 | 하나의 스펙트럼과 분산 적분으로 연결 | 국소 차감항과 고차 영향함수는 별도 |
| 진공상수 | 기록 불변 변환을 확인 | 평탄 기록만으로는 읽히지 않음 |

다음 모형은 적어도 양의 포털의 미시 부호·정규화, 분할 반지름의 복사안정성, 국소항의 정합, 물리적인 기록 대수와 초기 경계조건을 같은 구성에서 제공해야 한다. 현재 스칼라 후보에서 탈락한 방식을 다시 숫자만 바꿔 재사용하지 않는다. 이미 있던 게이지·관계 운동구조가 이 조건을 제공하는지는 별도 유도와 검산의 대상이다. 이것이 다음 연구의 목표이지 완료된 결과는 아니다.

## 13. 재현과 출처

실행:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_identification.py --output rerun.json
```

입력값은 코드와 results.json에 모두 기록돼 있다. 원본 CE-RV1 모듈은 inherited/ce_rv1.py에 그대로 보존했다. 전체 과거 40개 검사를 재실행한 것은 아니며, 이번 36개 검사에서 필요한 원래 공동 상태·기록·시간발전을 새로 계산했다. 기록을 바탕으로 원래 물리식의 평균과 상관을 구하는 도구는 [R6]의 표준 계측과 연결된다.

외부 문헌은 도구의 출처이며 CE의 관측적 확증으로 인용하지 않는다.

- R1. D. Tong, Quantum Field Theory, §3 Interacting Fields: https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S3.html
- R2. D. Boyanovsky, Effective Field Theory out of Equilibrium: Brownian quantum fields, arXiv:1503.00156v2 (2015): https://arxiv.org/abs/1503.00156
- R3. S. P. Martin, Two-loop effective potential for a general renormalizable theory and softly broken supersymmetry, arXiv:hep-ph/0111209v2: https://arxiv.org/abs/hep-ph/0111209
- R4. P. Calabrese, A. Pelissetto, E. Vicari, Multicritical phenomena in O(n1)+O(n2)-symmetric theories, arXiv:cond-mat/0209580v2: https://arxiv.org/abs/cond-mat/0209580
- R5. A. Eichhorn, D. Mesterhazy, M. M. Scherer, Multicritical behavior in models with two competing order parameters, arXiv:1306.2952v2: https://arxiv.org/abs/1306.2952
- R6. K. Jacobs, D. A. Steck, A Straightforward Introduction to Continuous Quantum Measurement, arXiv:quant-ph/0611067: https://arxiv.org/abs/quant-ph/0611067

선행의 공통 스펙트럼·반응식은 제공된 CE식.txt 및 이전 첨부 보고서에 의존한다. 그 문헌에서 절대 상수 선택까지 이미 증명됐다고 바꾸지 않았다. 신규 정리·진단과 선행 원문 복사본은 읽기 지도에서 분리한다.
