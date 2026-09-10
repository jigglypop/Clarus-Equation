# 54. 양의 상태 원뿔과 Lorentz 확률 보존 — CE-LC1

## 54.1 계산 전 등록

**[다른 기하 연결 후보]** 2026-09-10. CE-GR5의 양의 상태 거리·실수 곡률
계량 대신, $2\times2$ Hermitian 행렬의 determinant를 Lorentz 이차형식으로
읽는 후보를 검사한다. 이 대수적 대응은 알려진 구성이며, 상태 매개변수가 실제
시공간 좌표라는 해석과 중력 작용을 이미 유도한 것으로 세지 않는다.

- $X=x^0I+x^i\sigma_i$, $X\ge0$인 원뿔과 $X\mapsto SXS^\dagger$,
  $S\in SL(2,\mathbb C)$를 사용한다. 시공간 부호 비교는 $(-+++)$다.
- $S=e^{-i\varphi\sigma_y/2}e^{\eta\sigma_z/2}$,
  $\eta=0,0.3,0.7,1.2$, $\varphi=0,0.4$를 고정한다.
  determinant·원뿔·Lorentz 행렬을 검사하고, 고정 trace 정규화를 함께 유지하는지
  모든 상태의 연산자 조건과 구체적인 상태로 대조한다.
- 물리적 변환 대조는 $\rho_\pm=(I\pm\sigma_z)/2$와 그 반반 혼합이다.
  normalized filter의 선형성, $M=e^{-\eta/2}S$의 성공 확률과 보완 Kraus
  연산자를 검사한다. 실패 분기를 버려 결정론적 부스트라고 해석하지 않는다.
- 관측자 프레임 대조는 $G=I$, $G'=S^{-\dagger}GS^{-1}$,
  $\rho'=S\rho S^\dagger$, $E'=S^{-\dagger}ES^{-1}$로 고정한다.
  $\rho=(I+0.3\sigma_x+0.2\sigma_y-0.4\sigma_z)/2$,
  $E=(I+\sigma_x)/2$에서 가중 정규화·확률·표준화한 밀도행렬의 스펙트럼을 검사한다.
- 같은 프레임 변화가 CE-AM4 기록에 미치는 영향은 별도 spinor 인자 $\mathbb C^2$를
  사용한다. 초기 $\bigl(|0,q,0_A\rangle+|1,1,0_A\rangle\bigr)/\sqrt2$,
  기록 펄스 면적 $\pi/4$에서 기록 확률과 계+장치 에너지를 검사한다.
  spinor의 2차원을 약한 힘의 내부 2차원과 동일시하지 않는다.
- 시공간 연결 대조는 단일 행렬장 $X(q)=y^A(q)\sigma_A$에서
  $g_{\mu\nu}=\ell^2\eta_{AB}\partial_\mu y^A\partial_\nu y^B$를 구성한다.
  $y=(3+e^\xi\sinh\tau,e^\xi\cosh\tau,u,v)$,
  $(\tau,\xi,u,v)=(0,0,0.1,-0.1),(0.2,0.3,0.1,-0.1),(-0.2,-0.3,0.1,-0.1)$,
  $\ell=1$에서 곡률을 검사한다. 일반 가역 Jacobian의 경우도 해석한다.
- 대수 항등식 허용오차 $10^{-12}$, 곡률의 독립 차분 검사 $10^{-9}$,
  차분 간격 0.001, 0.0005, 피팅 없음. 프레임 변경과 물리적 상태 조작,
  원뿔의 부호와 동적 시공간 유도를 구분한다.

## 54.2 양의 원뿔에 존재하는 Lorentz 이차형식

**[알려진 대수 대응의 적용]** Pauli 행렬로 전개한 Hermitian $X$의 고유값과
행렬식은

$$
\lambda_\pm=x^0\pm|\mathbf x|,\qquad
\det X=(x^0)^2-|\mathbf x|^2,
\qquad X\ge0\ \Longleftrightarrow\ x^0\ge|\mathbf x|
$$

이다. 따라서 양의 행렬 원뿔은 미래 Lorentz 원뿔과 대수적으로 대응한다.
이것은 [Arrighi·Patricot의 원 논문](https://arxiv.org/html/quant-ph/0212135)에서도
사용한 대응이다. 양의 Hilbert–Schmidt 거리와 다른 이차형식인 determinant를
쓰므로 CE-GR5의 양의 상태 거리에 관한 제한과 모순되지 않는다.

$S\in SL(2,\mathbb C)$에서 $X'=SXS^\dagger$는 양성을 보존하고
$\det X'=\det X$다. 네 성분에 작용하는 실수 행렬은

$$
\Lambda^A{}_B(S)=\frac12\operatorname{Tr}(\sigma_A S\sigma_BS^\dagger),
\qquad \Lambda^{\mathsf T}\eta\Lambda=\eta,\qquad
\eta=\operatorname{diag}(-1,1,1,1).
$$

등록한 회전·boost는 proper orthochronous branch에 있다. rank-1 양의 행렬은
이 원뿔의 null 경계에 놓인다. 그러나 상태의 rank를 물리적 입자의 질량이나
빛의 전파로 동일시하지 않는다. $\operatorname{Herm}(2)$를 선택한 사실과 이
원뿔을 물리적 인과구조로 읽는 해석은 추가 입력이다.

## 54.3 고정 정규화에서 boost는 결정론적 상태 사상이 아니다

**[모든 상태의 조건]** 고정된 보통 trace를 사용하는 밀도행렬에
$\rho\mapsto S\rho S^\dagger$가 trace를 보존하려면

$$
\operatorname{Tr}(S\rho S^\dagger)=1\quad\text{모든 }\rho
\quad\Longleftrightarrow\quad S^\dagger S=I
$$

이어야 한다. $SL(2,\mathbb C)$ 중 이 조건을 만족하는 것은 $SU(2)$다.
비영 boost에는 성립하지 않는다. 결과를 trace로 나눈

$$
\mathcal F_S(\rho)=\frac{S\rho S^\dagger}{\operatorname{Tr}(S\rho S^\dagger)}
$$

도 일반적으로 비선형이므로 고정 계의 결정론적 양자 channel과 같지 않다.
$\rho_\pm=(I\pm\sigma_z)/2$, $\rho_m=(\rho_++\rho_-)/2$와
$S=e^{-i\varphi\sigma_y/2}e^{\eta\sigma_z/2}$에서

$$
\frac12\left\|\mathcal F_S(\rho_m)
-\frac{\mathcal F_S(\rho_+)+\mathcal F_S(\rho_-)}2\right\|_1
=\frac{\tanh\eta}{2}
$$

이다. 실제 물리적 filter로 구현할 때에는

$$
M=e^{-\eta/2}S,\qquad
N=\operatorname{diag}(0,\sqrt{1-e^{-2\eta}}),\qquad
M^\dagger M+N^\dagger N=I
$$

를 사용할 수 있다. 최대혼합 상태의 성공 확률은
$p_s=(1+e^{-2\eta})/2$다. 성공 분기에 조건부 정규화를 하면 위 $\mathcal F_S$와
같은 결과를 얻지만, 실패 분기도 전체 상태에 남는다.
등거리 사상 $|\psi\rangle\mapsto|s\rangle M|\psi\rangle+
|f\rangle N|\psi\rangle$는 모든 초기 내적을 보존한다.
조건부 filter와 Lorentz 행렬 대응의 구분은
[Verstraete·Dehaene·De Moor의 원 논문](https://arxiv.org/html/quant-ph/0011111)의
local filtering 맥락과 같다. 대응 자체를 성공 확률 1인 물리적 boost로 바꾸지 않는다.

## 54.4 상태와 측정 기준을 함께 옮기면 확률은 보존된다

**[프레임 표현의 조건부 구성]** 양의 행렬 $G$로 spinor 내적
$\langle\psi,\chi\rangle_G=\psi^\dagger G\chi$를 정의한다.
ensemble의 양의 행렬 $\rho$는 $\operatorname{Tr}(G\rho)=1$로 정규화한다.
측정 효과 $E_r\ge0$에는 $\sum_rE_r=G$를 요구하고
$p_r=\operatorname{Tr}(E_r\rho)$로 둔다. 일반 비직교 기저에서 표준 확률을
표현하는 방식이다.

이제 상태와 쌍대 측정 기준을 함께 변환한다.

$$
\rho'=S\rho S^\dagger,\qquad
G'=S^{-\dagger}GS^{-1},\qquad
E'_r=S^{-\dagger}E_rS^{-1}.
$$

양성과 완비성 $\sum_rE'_r=G'$를 보존하며

$$
\operatorname{Tr}(G'\rho')=1,\qquad
\operatorname{Tr}(E'_r\rho')=\operatorname{Tr}(E_r\rho)
$$

가 된다. 표준 내적으로 옮긴 밀도행렬을
$\widehat\rho=G^{1/2}\rho G^{1/2}$라 하면

$$
\widehat\rho'=U\widehat\rho U^\dagger,\qquad
U=G'^{1/2}SG^{-1/2},\qquad U^\dagger U=I.
$$

따라서 같은 실험의 밀도행렬 스펙트럼·순도와 모든 효과의 확률이 보존된다.
이는 실제 filter를 수행해 새로운 상태를 준비하는 §54.3과 다르다. 일반 상대론의
모든 물리적 boost가 유한 spinor 하나의 결정론적 조작이라는 주장도 아니다.

$G$-직교 사영 $P$는 $P'=SPS^{-1}$로 옮긴다. 보통 Hermitian이 아닐 수 있지만
$P'^\dagger G'=G'P'$이고 $G'P'$는 양의 효과다. 따라서 $P/Q$ 확률을 보존하려면
사영·쌍대 효과·정규화 기준을 함께 옮겨야 한다.
일반 instrument의 연산자도 $L'_r=SL_rS^{-1}$로 옮기면
$\sum_rL_r'^\dagger G'L'_r=G'$가 보존된다.

이번에 $G$는 관측자/기저의 정규화 기준으로 공급했다. 모든 시공간 방향에서
$G$가 공변 상수라는 추가 조건을 부과하지 않았으며, 곡률을 가진 spin 연결이나
실제 관측자의 운동법칙을 이 식만으로 유도하지 않는다.

## 54.5 비관측 성분과 기록을 포함한 32차원 대조

**[같은 기록 모형의 결합 검사]** CE-AM4의 16차원 계·장치에 별도의 spinor
인자 $\mathbb C^2$를 붙인다. 이 인자는 약한 힘 내부 doublet과 다른 지수다.
준비 상태와 기록 pulse는

$$
|\Psi_0\rangle=\frac{|0,q,0_A\rangle+|1,1,0_A\rangle}{\sqrt2},\qquad
U_{\rm rec}=I_2\otimes e^{-i(\pi/4)T}
$$

다. spinor와 관측/비관측 성분을 처음부터 얽히게 하여 product 상태만의 검사가
되지 않도록 했다. 원래 기록 확률은 $1/4$, 계·장치 자유 에너지는 각각 2와 1이다.

전체 frame을 $S\otimes I_{16}$으로 바꾸고 내적을 $G'\otimes I_{16}$으로
같이 바꾸면 기록 확률과 같은 내부 에너지 관측량의 값은 유지된다.
코드는 특정 상태의 값 외에도 모든 상태에 대한 등거리·효과 변환 연산자
항등식을 검산했다. 반면 상태만 옮긴 뒤 보통 trace로 정규화하고 기존 효과를
그대로 읽으면 기록 확률은 $(1+\tanh\eta)/4$가 된다.

| rapidity $\eta$ | 같은 측정 기준을 함께 변환 | 상태만 변환한 뒤 보통 trace 정규화 |
|---:|---:|---:|
| 0 | 0.25 | 0.25 |
| 0.3 | 0.25 | 0.322828153113 |
| 0.7 | 0.25 | 0.401091944279 |
| 1.2 | 0.25 | 0.458413651753 |

이 에너지 대조는 같은 내부 Hamiltonian의 표현 공변성이다. 서로 움직이는
관측자가 측정하는 4-운동량의 시간 성분이 항상 같다는 주장이 아니다.
시공간 운동량·관측자 운동·기록 장치의 자율적 구동은 아직 공급하지 않았다.
CE-AM4의 전체 진폭 보존은 이 frame 변경에서도 유지되지만, 실제 단일 기록
선택이나 기록 안정성의 미완성 부분은 그대로다.

## 54.6 원뿔의 네 성분을 곧바로 좌표로 쓰면 평탄하다

**[직접 시공간 사상의 제한]** 단일 Hermitian 행렬장
$X(q)=y^A(q)\sigma_A$에서 determinant의 이차형식을 미분에 적용하면

$$
ds^2=-\ell^2\det(dX)
=\ell^2\eta_{AB}dy^A dy^B,
\qquad
g_{\mu\nu}=\ell^2\eta_{AB}\partial_\mu y^A\partial_\nu y^B
$$

를 정의할 수 있다. 여기의 $\det(dX)$는 접벡터 $v$에 평가한
$\det(dX(v))$의 대칭 이차형식이며, 1-form들의 wedge 행렬식이 아니다.
$y^A$는 무차원 상태 매개변수, $\ell$은 공급한 길이척도다.
Jacobian이 가역인 열린 영역에서는 $y^A$ 자체를 좌표로 쓸 수 있으므로
계량은 상수 Minkowski 계량의 좌표 pullback이다. 따라서 Riemann 곡률은 0이다.
정규화 trace를 고정해 $y^0$를 상수로 두면 Jacobian rank가 3 이하이고
비퇴화 4차원 계량도 얻지 못한다.

등록한 원뿔 내부 사상

$$
y=(3+e^\xi\sinh\tau,e^\xi\cosh\tau,u,v)
$$

는 각 검산점에서 양의 $X$를 가지며

$$
ds^2=-e^{2\xi}d\tau^2+e^{2\xi}d\xi^2+du^2+dv^2
$$

를 준다. $\Gamma^\tau_{\tau\xi}=\Gamma^\tau_{\xi\tau}=
\Gamma^\xi_{\tau\tau}=\Gamma^\xi_{\xi\xi}=1$이지만 Riemann 곡률은 0이다.
좌표 의존 계량과 비영 Christoffel 기호만으로 중력이 유도되었다고 판단하지 않는다.

위 coframe에 위치 의존 Lorentz frame을 곱해도 계량은 바뀌지 않는다.
동반 연결을 순수 frame 변환으로 구성했다면 그 곡률도 0이다.
따라서 중력에는 이 단일 가역 상태 좌표 사상보다 많은 구조와 동역학이 필요하다.
더 일반적인 coframe·추가 상태 embedding을 고려할 수 있지만, 비적분 coframe
하나를 썼다는 사실만으로 Einstein 곡률과 작용이 생기는 것은 아니다.

## 54.7 검산과 연구 지위

**[계산 검산]** [실행 코드](../../verify/ce_positive_cone_lorentz.py)와
[산출 JSON](../../verify/ce_positive_cone_lorentz.json)은 8개 회전·boost 조합에서
행렬식·Lorentz 이차형식·양성·weighted 효과·사영·밀도행렬 스펙트럼을 검사했다.
normalized filter의 비선형성은 두 순수 상태와 그 혼합으로 독립 대조했고,
성공/실패 Kraus의 완비성과 전체 등거리 사상도 검사했다.

같은 8개 frame에서 얽힌 기록 모형을 확인했으며 최대 대수 항등식 오차는
$4.885\times10^{-15}$다. 세 좌표점에서 Christoffel 기호를 embedding의
이차 미분과 계량 미분으로 독립 계산하고, 두 차분 간격으로 Riemann 곡률을
검사했다. 최대 곡률 잔차는 $1.112\times10^{-13}$로 등록 허용오차 안이다.
코드·의존 파일 해시와 환경 판본을 저장했다. 관측 자료와 피팅은 쓰지 않았다.

**[조건부 진전과 한계]** 양의 양자 상태 원뿔에서 Lorentzian 대수 구조를 읽고,
상태·측정·내적을 함께 변환해 비관측 성분과 기록 확률을 보존하는 표현은 구성할
수 있다. 고정 trace의 물리적 channel로 boost를 동일시하는 판본은 실패한다.
또 단일 행렬장의 가역 네 좌표로 만든 계량은 평탄하므로 일반상대론의 동적
중력을 복원하지 못한다.

이 결과는 공간차원·시간·관측 확정의 자연적 유도나 통일 이론의 완성 선언이 아니다.
같은 상태에 비평탄 coframe과 작용을 연결하는 원리, spinor·내부 힘 표현의 선택,
자율적 비동기 사건과 실제 기록, 전체 UV·공동 관측 검증이 남는다.

관련 선행 문서: [실수 곡률 계량의 한계](53_곡률_계량의_부호와_자기_운동항의_위상화.md),
[색깔 공변 기록](../01_측정과_접힘/09_색깔_공변_기록과_보완_에너지_전달.md),
[공통 물질 표현](46_공통_수송의_물질_표현과_이상_상쇄.md).
