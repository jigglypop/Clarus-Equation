# CE-RV1 — 기록에서 값으로: 공동 양자상태의 조건화와 이후 운동

연구 기준일: 2026-09-19
기준 원격 main: `8721ab787a3848bddc6cfaf8f8c3622479d62639`
선행: CE-RH2 공동 상태, CE-RS1 기록 조건부 원천, CE-CR1 공간 기록
상태: 명시한 유한모드의 해석적 항등식과 수치 검산. 관측 피팅 없음.

## 0. 이번에 끝낸 계산과 끝내지 않은 주장

기존에는 서로 다른 외부 힉스 배경이 구별되는 환경 상태를 만든다는 것과, 기록이 주어진 상태에서 원천을 계산하는 일반식이 있었다. 이번에는 기존 힉스–매개장 공동 해밀토니언을 실제로 풀고, 그 공동 상태에 유한 분해능의 기록 연산을 적용했다. 각 기록에 대한 힉스 크기의 조건부 분포, 평균, 상관 원천, 초기 가속도 및 이후의 완전한 유니터리 시간발전을 계산했다. 힉스의 시간 이력을 외부 함수로 공급하지 않았다.

가장 직접적인 연결은 다음 두 항등식이다.

\[
\boxed{\partial_r\bar u_r(0)=\frac{\operatorname{Cov}_r(u,O)}{\sigma^2}},
\qquad
\boxed{\partial_r\bar u_r(t)=\frac{\operatorname{Cov}^{\rm sym}_{\rho_r(0)}(u_H(t),O)}{\sigma^2}}.
\]

기록에 따라 장의 예측값이 얼마나 달라지는지, 그리고 그 차이가 미래로 어떻게 전달되는지를 원래 포털의 동일한 상관함수로 결정한다. 기록 대응 함수를 따로 적합하지 않았다. 이 식의 r 미분은 **기록 좌표에 대한 미분**이며 물리 시간의 미분 또는 새로운 붕괴 시간법칙이 아니다.

그러나 여기서 계산한 평균은 확정된 힉스 고유값이나 전약 진공값이 아니다. 조건부 분산은 여전히 비영이다. 계측기의 상호작용과 분해능, 바닥상태 준비, 기존 작용의 상수는 입력이다. 자연상수, Born 규칙, 우주의 단일 결과, 전체 표준모형 또는 연속 QFT를 유도한 것으로 세지 않는다.

## 1. 원래 모형과 입력 장부

원전의 세 채널 스펙트럼과 동일한 포털을 유지한다.

\[
X(u)=(s_0+\kappa u)I_3+Y(\theta),\quad
Y=\epsilon(e^{i\theta/3}S+e^{-i\theta/3}S^\dagger),\quad S^3=I.
\]

여기서 S는 실제 세 채널 순환행렬이다. 계산에는 기존 RH2의 한 실수 힉스 대리좌표 \(\varphi\), \(u=\varphi^2/2\), 세 복소 조화모드 \(\chi_j=(q_{j1}+iq_{j2})/\sqrt2\)를 사용한다.

\[
O=\sum_j\chi_j^\dagger\chi_j=\frac12\sum_{j,\alpha}q_{j\alpha}^2,
\]
\[
\boxed{\mathscr H=\frac{p_\varphi^2}{2}+\lambda(u-u_0)^2+
\sum_{j,\alpha}\left(\frac{p_{j\alpha}^2}{2}+\frac{x_{0j}q_{j\alpha}^2}{2}\right)
+\kappa uO},
\quad x_{0j}=s_0+2\epsilon\cos\frac{\theta+2\pi j}{3}.
\]

입력은 선행과 동일하게 \(s_0=.5,\epsilon=.15,\kappa=1,\lambda=10,u_0=.5,\theta=\pi\)다. 바닥상태 \(\rho_0=|\Psi_0\rangle\langle\Psi_0|\)는 이 해밀토니언의 가장 낮은 고유상태로 계산한다. 바닥상태를 준비한다는 경계조건을 자연적으로 유도한 것은 아니다.

새 계측 입력은 \(\sigma=1\), 비교할 기록 좌표 \(r=.5,2,4\)뿐이다. sigma는 O와 같은 단위의 검출 분해능이다. 이 숫자는 자연상수나 우주적 잡음률이 아니다. 모든 표는 기존 유한모드 정규화의 내부 단위이며, 4차원 장의 수치적 차원 정합 또는 실제 GeV 값과 동일시하지 않는다.

원형 해밀토니언에 chi를 명시적으로 남겼으므로, 같은 chi를 적분해 만든 \(\operatorname{Tr}\ln K\) 또는 CE 상대 퍼텐셜을 다시 더하지 않는다.

## 2. 유한 분해능 기록과 실제 기록 확률

기존 원천 연산자 O의 Gaussian 기록을 다음 Kraus 연산자로 정의한다.

\[
\boxed{M_r=(2\pi\sigma^2)^{-1/4}
\exp[-(r-O)^2/(4\sigma^2)]}.
\]

스펙트럼 정리에 따라 \(\int M_r^\dagger M_r\,dr=I\)다. 따라서

\[
p(r)=\operatorname{Tr}(M_r\rho_0M_r^\dagger),\qquad
\rho_r(0)=\frac{M_r\rho_0M_r^\dagger}{p(r)}
\]

는 정규화된 확률밀도와 조건부 공동 상태를 준다. r은 연속 기록이므로 p(r)은 한 점의 확률이 아니라 밀도다. 구간의 확률은 p(r)을 적분해야 한다.

이 연산은 Gaussian 초기 파동함수 \(g_\sigma(R)\)를 가진 포인터에 대해 \(U_{CR}=\exp(-iO\otimes P_R)\)를 적용하고 포인터 위치를 읽으면 \(M_r=\langle r|U_{CR}|g_\sigma\rangle\)로 구현되는 표준 계측 모형이다. 포인터 결합을 기존 CE 포털에서 자동 유도한 것은 아니다. 계측을 하나의 지정된 연산으로 다루며, 실제 포인터의 전체 자율 해밀토니언이나 스위칭 이력을 푼 것도 아니다. Born 조건화는 사용한 양자역학 규칙이다. 표준 도구의 출처는 R1이다.

힉스 크기의 예측값과 전체 축약상태는

\[
\boxed{\bar u_r(t)=\frac{\operatorname{Tr}\left[u\,e^{-it\mathscr H}M_r\rho_0M_r^\dagger e^{it\mathscr H}\right]}{p(r)}},
\qquad
\rho_{H|r}(t)=\operatorname{Tr}_C\rho_r(t).
\]

단일 평균만 정의한 것이 아니다. 임의의 힉스 크기 구간 B에 대한 확률도 \(\operatorname{Tr}[1_B(u)\rho_r(t)]\)로 정해진다. 이 계산의 폐쇄는 전체 상태를 진화시키는 방식이며, 평균 u 하나만으로 된 닫힌 고전 운동방정식을 얻었다고 하지 않는다.

## 3. 기록–값 기울기의 정확한 증명

\(\partial_rM_r=(O-r)M_r/(2\sigma^2)\)와 정규화를 미분하면

\[
\partial_r\rho_r=\frac{\{O-\langle O\rangle_r,\rho_r\}}{2\sigma^2}.
\]

따라서 임의의 연산자 A에 대해

\[
\partial_r\langle A\rangle_r=\frac{\frac12\langle\{A,O\}\rangle_r-\langle A\rangle_r\langle O\rangle_r}{\sigma^2}.
\]

A=u일 때 u와 O는 서로 다른 부분계에 작용하므로 가환한다. 이에 따라

\[
\boxed{\partial_r\bar u_r=\frac{\langle uO\rangle_r-\bar u_r\langle O\rangle_r}{\sigma^2}}.
\]

같은 계산은 기록 밀도의 score에 대해서도

\[
\boxed{\partial_r\ln p(r)=\frac{\langle O\rangle_r-r}{\sigma^2}},
\qquad
\boxed{\langle O\rangle_r=r+\sigma^2\partial_r\ln p(r)}
\]

를 준다. 따라서 원래 반작용의 상관항을 기록식으로 정확히 다시 쓸 수 있다.

\[
\boxed{\langle uO\rangle_r=
\bar u_r\left[r+\sigma^2\partial_r\ln p(r)\right]
+\sigma^2\partial_r\bar u_r}.
\]

마지막 미분항을 지우는 것은 \(\langle uO\rangle_r\)를 \(\bar u_r\langle O\rangle_r\)로 인수분해하는 것과 같다. 즉 원래 요구했던 잔여의 상관 반작용을 버린다.

시간이 지난 뒤에는 \(u_H(t)=e^{it\mathscr H}ue^{-it\mathscr H}\)로 놓으면 된다.

\[
\boxed{\partial_r\bar u_r(t)=\frac{\frac12\langle\{u_H(t),O\}\rangle_{\rho_r(0)}-
\bar u_r(t)\langle O\rangle_{\rho_r(0)}}{\sigma^2}}.
\]

일반 시간에는 교환자가 0일 필요가 없으므로 symmetrized covariance를 써야 한다. 동일시각의 단순 곱을 미래에도 그대로 사용하는 처방은 채택하지 않았다.

이 항등식은 해당 계측 모형에서의 연산자 미분 결과다. 표준 양자계측 전체에 대한 새로운 최초 발견을 주장하지 않는다. CE에서는 기존 원천 O와 공동 해밀토니언을 대입하여 구체적인 기록 응답을 계산했다.

## 4. 기록을 입력으로 받은 이후 운동

원래 해밀토니언으로 Heisenberg 방정식을 계산하면

\[
\dot u=\tfrac12\{\varphi,p_\varphi\},\qquad
\boxed{\ddot u=p_\varphi^2-4\lambda u(u-u_0)-2\kappa uO}.
\]

따라서 기록 직후의 정확한 조건부 가속도는

\[
\boxed{\left.\frac{d^2\bar u_r(t)}{dt^2}\right|_{t=0}=
\langle p_\varphi^2\rangle_r-4\lambda(\langle u^2\rangle_r-u_0\bar u_r)
-2\kappa\left\{\bar u_r[r+\sigma^2\partial_r\ln p(r)]+\sigma^2\partial_r\bar u_r\right\}}.
\]

모든 항은 계산한 공동 상태의 조건부 모멘트다. 기록만 보고 기존 바닥상태나 작용을 몰라도 값을 정할 수 있다는 뜻은 아니다. 또한 \(\langle u^2\rangle_r\)나 \(\langle p_\varphi^2\rangle_r\)를 \(\bar u_r\)만의 함수로 임의 치환하지 않았다.

실제 시간발전은 \(|\Psi_r(t)\rangle=e^{-it\mathscr H}|\Psi_r(0)\rangle\)를 그대로 계산했다. 따라서 외부 tanh 힉스 이력, 현상론적 감쇠율, 상관을 제거한 평균장 치환을 쓰지 않았다. 유한 투영 행렬에서의 정확한 유니터리 문제이며, 무한 기저로의 수렴은 별도 수치 검사다.

\(\varphi\to-\varphi\) 대칭은 계측 뒤에도 유지된다. 그러므로 \(\langle\varphi\rangle_r=0\)인 반면 \(\langle\varphi^2/2\rangle_r\)는 변한다. 표의 u를 실제 힉스 VEV라고 부르지 않는 이유다.

## 5. 기록 비용: 유한 분해능은 유한한 에너지를 쓴다

기록을 읽지 않은 채 평균한 채널은 O 고유기저에서

\[
\rho_{oo'}\longmapsto e^{-(o-o')^2/(8\sigma^2)}\rho_{oo'}
\]

이며, 연산자 표기로는 \(\mathcal E=\exp[-\operatorname{ad}_O^2/(8\sigma^2)]\)다. 원래 연속 좌표의 유한모드 해밀토니언에서

\[
[O,[O,\mathscr H]]=-2O,\qquad [O,O]=0.
\]

따라서 더 높은 중첩 교환자는 사라지고

\[
\boxed{\mathcal E^*(\mathscr H)=\mathscr H+\frac{O}{4\sigma^2}},
\qquad
\boxed{\Delta\bar E=\frac{\langle O\rangle_0}{4\sigma^2}}.
\]

이는 위 지정된 계측 연산이 HC 계에 전달한 평균 에너지다. 실제 장치의 완전한 에너지 예산을 미시 해밀토니언으로 풀었다는 뜻은 아니다. 각 기록에서의 에너지 변화는 서로 다르며 평균과 혼동하지 않는다. 계측 이후 각 조건부 상태는 다시 같은 고정 해밀토니언으로 움직이므로 자신의 계측 직후 에너지를 보존한다.

\(\sigma>0\)에서 비용은 유한하지만, 이상적으로 \(\sigma\to0\)인 정확한 O 측정의 평균 비용은 발산한다. 이 결과를 진공 에너지의 임의 기준 상수와 혼동하지 않는다. 또한 유한한 개수의 oscillator에 대한 결과를 모든 공간 모드를 무한 정밀도로 읽는 검출기로 승격하지 않는다.

유한 Fock 투영에서는 연속 정준 교환관계가 경계에서 정확하지 않다. 그러므로 코드에서는 유한 행렬의 정확한 dephasing 채널을 계산하고, 별도로 위 식에 접근하는지를 검사했다.

## 6. 실제 공동 상태 및 기록별 수치

바닥상태 계산은 힉스 짝수 부문과 각 복소 oscillator의 각운동량 0 부문을 쓴다. 해밀토니언, O 계측, 초기 바닥상태가 이 부문들을 보존한다. 원래 polynomial을 넓은 공간에서 만든 뒤 투영하여 잘린 위치행렬을 단순 제곱하는 오류를 피했다.

가장 큰 정적 기저는 \(24\times21^3=222,264\)차원이다. 기존 미관측 바닥상태 값은 다시

\[
\langle u\rangle_0=0.240399684809\ldots,\quad
\langle O\rangle_0=1.838928551449\ldots,\quad
\operatorname{Cov}_0(u,O)=-0.025665420022\ldots
\]

로 재현됐다.

| 기록 r | 밀도 p(r) | 조건부 평균 u | 조건부 표준편차 | 초기 가속도 d²u/dt² |
|---:|---:|---:|---:|---:|
| 0.5 | 0.205335 | 0.255624 | 0.258212 | +0.089844 |
| 2.0 | 0.270298 | 0.241310 | 0.249680 | +0.001002 |
| 4.0 | 0.078953 | 0.212676 | 0.230980 | −0.159670 |

**표준편차는 수치계산 오차가 아니라 그 조건부 양자상태에 남는 물리적 분산이다.** 평균을 여러 자릿수로 계산했다는 사실로 확정된 자연값을 얻었다고 하지 않는다. 이 유한 환경과 계측은 정확한 값 하나를 선택하는 데 충분하지 않다.

| r | 조건부 Cov(u,O) = sigma² du/dr | 조건부 에너지 |
|---:|---:|---:|
| .5 | −0.0074273360 | 4.581537 |
| 2 | −0.0118043158 | 4.471758 |
| 4 | −0.0159792467 | 5.154701 |

검사한 세 기록에서 상관은 음수다. 모든 상태·모든 기록에서 이 부호가 고정된다는 일반 정리를 주장하지 않는다. 상관을 버리면 가속도의 차이는 각각 약 .01485, .02361, .03196으로 남는다.

정적 환경 기저를 17에서 21로 늘렸을 때 세 u 평균의 최대 차이는 \(4.75\times10^{-9}\), 조건부 에너지의 최대 차이는 약 \(9.56\times10^{-6}\)였다. 서로 다른 양의 수치 오차를 같은 자릿수로 표현하지 않는다.

## 7. 이후 값을 외부에서 주지 않은 시간발전

시간 계산은 \(N_H=18, N_C=9,13,17\)에서 수행했다. 계산 효율을 위한 힉스 oscillator 기저 주파수는 8을 썼으며, 물리 파라미터가 아니다. 별도로 주파수 3, 더 큰 힉스 기저로 동일 관측값을 확인했다.

가장 큰 시간발전 기저 \(18\times17^3=88,434\)의 결과는 다음과 같다.

| 내부 시간 | r=.5의 u 평균 | r=2의 u 평균 | r=4의 u 평균 | 측정하지 않은 바닥상태 |
|---:|---:|---:|---:|---:|
| 0 | .255624 | .241310 | .212676 | .240400 |
| .25 | .257999 | .241303 | .208271 | .240400 |
| .5 | .261320 | .241145 | .200672 | .240400 |
| .75 | .259288 | .241184 | .200803 | .240400 |
| 1 | .247678 | .241785 | .216910 | .240400 |

여기서 u(t)를 미리 정해 넣은 것은 아니다. 각 기록의 실제 조건부 공동 상태를 초기값으로 삼아 원래 해밀토니언으로 계산했다. 새 힘 함수나 기록별 결합상수를 붙이지 않았다.

수치 검사:

- 기저 13→17에서 표의 전체 시간·기록에 대한 u 평균 최대 차이: \(7.72\times10^{-7}\).
- 힉스 기저 크기와 주파수를 독립 변경한 t=.5, N_C=9의 u 차이: \(1.22\times10^{-14}\).
- 전체 검사에서 norm 최대 오차: \(1.32\times10^{-13}\).
- 계측 이후 에너지 최대 drift: \(7.43\times10^{-13}\).
- 직접 시간발전의 초기 이차미분과 정확한 상관 원천의 차이: \(1.26\times10^{-10}\).
- 작은 독립 행렬의 완전 대각화와 sparse 지수 시간발전의 상태 차이: \(3.74\times10^{-15}\).

시간에 따른 재진동이 있으므로 유한 계가 비가역적으로 하나의 고전적 값에 수렴한다고 하지 않는다. norm·에너지의 작은 수치오차를 무한 기저의 전체 물리 오차와 동일시하지 않는다.

미래 기록 응답도 별도 검산했다. r=2, t=.5에서 기록을 실제로 ±0.0002씩 바꿔 독립 시간발전을 한 미분은

\[
\partial_r\bar u_r(.5)=-0.0168603988919\ldots
\]

이고, 같은 상태의 서로 다른 시각 상관으로 계산하면

\[
\sigma^{-2}\operatorname{Cov}^{\rm sym}(u_H(.5),O)=-0.0168603989020\ldots
\]

이다. 차이는 약 \(1.02\times10^{-11}\)이다.

## 8. 계측 에너지와 비선별 평균의 수치 확인

sigma=1의 해석적 평균 비용은

\[
\Delta\bar E=1.838928551449\ldots/4=0.459732137862\ldots.
\]

| 복소모드 기저 N_C | 유한 행렬에서의 정확한 평균 비용 | 연속 좌표 항등식과의 차이 |
|---:|---:|---:|
| 9 | .459579082747 | 1.53e−4 |
| 13 | .459716536011 | 1.56e−5 |
| 17 | .459729917190 | 2.22e−6 |
| 21 | .459731745008 | 3.93e−7 |

비선별 기록 연산은 C에만 작용하는 trace-preserving channel이다. 따라서 기록 직후

\[
\int p(r)\rho_{H|r}\,dr=\rho_H,\qquad
\int p(r)\bar u_r\,dr=\langle u\rangle_0
\]

다. 독립 적분과 부분추적으로 확인했다. 기록을 읽는 행위만으로 원격의 비선별 힉스 상태를 즉시 바꿨다고 해석하지 않는다. 이후의 변화는 실제 C의 계측 교란과 기존 H–C 상호작용을 통한 시간발전이다.

## 9. 별도 극한: 채널별 기록에서 원래 질량행렬로 돌아가기

이 절은 **고정 c-number 힉스 배경의 Gaussian 진공**에 대한 보조 정리다. 위의 얽힌 공동 바닥상태에 같은 식을 직접 대입하지 않는다. 또한 전체 O의 기록 하나만으로 아래의 채널별 행렬 정보를 모두 얻는다고 하지 않는다.

고정 운동량 k에서 복소장의 공분산 \(C(k)=\langle\chi\chi^\dagger\rangle\)는

\[
C(k)=\tfrac12(k^2I+X)^{-1/2},\qquad
\boxed{X=\tfrac14 C(k)^{-2}-k^2I}
\]

를 만족한다. 따라서 충분한 채널별 상관 기록이 있으면

\[
s=\operatorname{Tr}X/3,\quad Y=X-sI,\quad
\epsilon=\sqrt{\operatorname{Tr}Y^2/6},\quad
\cos\theta=\frac{\operatorname{Tr}Y^3}{6\epsilon^3}
\]

로 원래 공통 스펙트럼을 복원한다. epsilon=0에서는 위상은 정의되지 않는다. 스펙트럼만으로 theta와 −theta는 구별되지 않으며, 방향 있는 순환곱을 읽으려면 정해진 채널 틀의 상관이 추가로 필요하다.

한 복소 진공모드의 세기 \(z=|\chi|^2\)는 \(p(z|u)=2\omega e^{-2\omega z}\)다. 이 분포의 Fisher 정보는 \(\kappa^2/(4\omega^4)\)로 순수 진공 계열의 local QFI와 같다. 3차원 공간에서의 밀도는

\[
\boxed{\mathcal I_u/V=\frac{\kappa^2}{32\pi}\sum_j\frac1{m_j}}
\]

이며 선행 CE-CR1의 겹침에서 구한 정보계량과 일치한다. 즉 이상적 전역 판별이라는 이름만 붙이지 않고, 이 고정 진공 극한에서는 정보를 담는 구체적인 세기 기록을 지정할 수 있다. 유한 대비의 Helstrom 최적 판별을 이 세기 측정이 항상 달성한다는 주장은 아니다. Gaussian 추정의 일반 배경은 R2다.

복원한 스펙트럼은 원전의 \(\mathcal U_{ss},\mathcal U_{sss}\)에 그대로 들어간다. 같은 함수의 직접 고정밀 미분과 determinant/역질량 합을 대조했다. 이것은 기록에서 질량을 추정한 뒤 반응을 예측하는 조건부 연결이며, 자연의 질량상수 자체를 무입력으로 정한 것이 아니다.

음성대조도 남겼다. \(x_1=.7\)의 진공과 \(x_2=1.4\), 점유수 \(n=(\sqrt2-1)/2\)인 상태는 같은 위치 공분산을 가질 수 있다. 준비상태가 알려지지 않은 경우에 진공 역공식을 쓰면 잘못된 질량을 얻는다.

## 10. 이 가지에서 아직 결정되지 않는 것

첫째, 기록은 기존 공동 상태에 대한 조건부 법칙을 제공한다. 같은 C 상태에 서로 다른 H 상태를 곱하면 모든 p(r)은 동일하지만 조건부 힉스 평균은 다르다. 따라서 p(r) 하나만으로 초기 공동 상태나 작용까지 유일하게 알아낸다는 명제는 반례를 갖는다. 이번 수치는 기존 해밀토니언과 바닥상태 준비가 그 공동 상태를 공급하기 때문에 결정됐다.

둘째, 기록에 따른 상관의 변화는 상수 kappa 또는 epsilon의 운동방정식이 아니다. 그 상수들을 동적 장으로 승격하려면 원래 작용에 그 자유도와 경계조건이 있어야 한다. 단순 기록 요구만으로 epsilon>0가 강제되지 않는 CE-CR1의 반례를 뒤집지 않았다. CS1/TH1의 별도 SWAP 상태 준비 모형을 이 포털에 임의로 합치지 않았다.

셋째, 조건부 평균의 계산 정확도와 조건부 분산은 다르다. 이 계산에서 분산은 크고 비영이며, 거시적인 안정된 단일 값의 생성과 전체 공간의 중복 기록까지 보이지 않았다. 반복 QND 조건화가 특정 조건에서 수렴한다는 표준 정리는 R3에 있지만, 이 상호작용하는 유한 HC 모형이 그 모든 전제를 만족한다고 하지 않는다.

넷째, 고정 시공간에서 H에 상수 CI를 더하면 진화에는 전체 위상만 생기고 기록별 예측은 바뀌지 않는다. 코드로 확인했다. 따라서 이 자료만으로 절대 진공에너지를 결정하지 않는다. 이것은 중력까지 포함한 모든 계에서 상수 진공에너지가 관측 불가능하다는 명제가 아니다.

## 11. 검증 범위와 재현

신규 검사 40개 = 정적·Gaussian 극한 30개 + 시간발전 10개다. 선행의 37개 또는 27개 검사를 이번 통과 수에 다시 더하지 않는다. 실제 관측 데이터, 우주론 RMSE, 성공 확률 또는 이론 완성 퍼센트는 계산하지 않았다.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_values.py --section static --output static_results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_values.py --section time --output time_results.json
# 두 단계를 한 실행으로 수행:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_values.py --section all --output results_rerun.json
```

정적 계산은 새로 실행했다. 시간발전의 세 기저 계산은 완료된 체크포인트를 보존했고, 실행 제한으로 중단된 통합 호출 이후 그 동일 수치에 대한 검사를 별도 실행했다. `time_results.json`의 실행시간은 전체 시뮬레이션 누계가 아니라 해당 검증 호출의 시간이다. 독립 대조 시간발전과 미래 상관 미분은 검증 호출에서 실제로 다시 계산했다. `CE_RV_RESUME=1`은 같은 디렉터리의 시간 체크포인트를 명시적으로 재사용하는 복구 옵션이며 기본 실행은 재계산한다.

이번 실행의 저장소 수정·새 푸시는 수행하지 않았다. 기준 원격 main의 읽기 확인과 새 산출물 작성은 구별한다. 추가 전용 Git 패치는 별도 파일로 제공하며 적용 전 기준 브랜치 및 경로 충돌을 확인해야 한다.

## 출처와 새 계산의 구분

P1. 첨부 원전 `CE식.txt`: 공통 세 채널 스펙트럼, epsilon의 차원, 공통 반응 미분 및 남은 입력. 이 문서의 선택된 가정은 원전 전체의 유도 완료를 뜻하지 않는다.

P2. 첨부 `CE_residual_higgs_closure_20260919/REPORT_ko.md`, `verify_closure.py`: 본 유한 공동 해밀토니언, 고정 입력, 바닥상태와 상관. 핵심 바닥상태를 이번에 다시 계산했다.

P3. main `8721ab7`의 `experiments/ce_record_selection_20260919/REPORT_ko.md`: 기록 안의 정확한 원천과 다른 기록을 다시 더하지 않는 원칙.

P4. 첨부 및 main의 `ce_continuum_records_20260919/REPORT_ko.md`: 조건부 진공 겹침·정보계량, 유한 준비 에너지, 기록 조건만으로 epsilon을 선택하지 못한다는 반례.

R1. K. Jacobs, D. A. Steck, *A Straightforward Introduction to Continuous Quantum Measurement*, Contemporary Physics 47, 279–303. https://arxiv.org/abs/quant-ph/0611067 . 일반화 계측·조건화와 교란의 표준 도구.

R2. D. Šafránek, *Estimation of Gaussian quantum states*. https://arxiv.org/abs/1801.00299 . Gaussian QFI의 일반 배경. CE 계수와 수치 검산은 본 계산이다.

R3. M. Bauer, D. Bernard, *Convergence of repeated quantum non-demolition measurements and wave function collapse*, Phys. Rev. A 84, 044103. https://arxiv.org/abs/1106.4953 . 반복 QND 조건화와 수렴의 추가 전제에 관한 비교 원전. Born 규칙을 CE에서 유도한 출처가 아니다.
