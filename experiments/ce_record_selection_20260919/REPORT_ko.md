# CE-RS1: 기록 조건부 잔여 반작용과 비균일 위상 선택

2026-09-19. 기준 main: `6e473c28099c300045d732719468ba0b8cdbfdf1`.

## 0. 게시 범위와 결과

이 문서는 첨부 CE-RS1 연구의 게시용 정리다. 원본 보고서의 바이트 복사본이 아니라 증명·전제·미완성 항목을 보존한 편집본이다. 독립 검산 코드는 현재 첨부본과 Git blob `783b4d45b91563489cdd7a278cb0c48c0bc8300d`가 일치한다. 이번 게시에서 37개 검사를 재실행했다. 현재 환경의 수치 기록은 `verification_rerun.json`에 분리한다. 이전 대화에 표시한 수치표를 새로운 실행 결과로 대체하여 인용하지 않는다.

확보한 것은 조건부 기록 안의 원천항, 원래 포털에 의한 부분적인 환경 기록, 고정 연결크기 양의 보손 격자의 비균일 위상 비교 정리다. 유일한 결과 발생, 모든 상수의 결정, 실제 우주의 초기상태와 공동 관측 성공을 선언하지 않는다. 테스트 통과는 가정의 물리적 확증이 아니다.

## 1. 원전과 추가 전제

원전 `source/CE_input.txt`의 공통 스펙트럼, 차원 있는 epsilon 고정, 상대 퍼텐셜을 유지한다. X=(s0+kappa u)I+Y, u=H†H이며 Y=epsilon(exp(i theta/3)S+exp(-i theta/3)S†), S는 실제 세 채널 순환행렬이다. s0>2epsilon>0, kappa>=0를 사용한다. 이는 표준모형 Higgs 이중항 자체의 유도가 아니다.

비균일 위상 비교에는 비음수 연결 w_xy의 유한 Laplacian L, 양의 스칼라–힉스 측도, 같은 내부 연결크기와 공간 운동연산자를 추가로 고정한다. 임의의 모든 Y, fermion determinant, 중력의 전체 경로적분을 이 정리에 포함하지 않는다. 포털 기록 시험에는 준비된 Gaussian 상태와 주어진 두 힉스 배경 이력을 쓴다. Born 조건화 규칙을 사용하며 새로 유도하지 않는다.

## 2. 선택한 기록 안의 잔여와 다른 기록을 구별

전체 상태를 rho_HCR, 기록 사영을 Pi_r라 두면

$$p_r=Tr(Pi_r rho),\qquad rho_{HC|r}=Tr_R(Pi_r rho Pi_r)/p_r.$$

기록이 이후 Hamiltonian에 보존되고 고정된 구간에서

$$J_{\varphi,r}=-\kappa Tr(rho_{HC|r}\,\varphi O),\qquad O=\sum_j\chi_j^\dagger\chi_j.$$

기록을 읽지 않은 평균은 sum_r p_r J_r이다. 특정 r에 조건화한 다음 다른 r의 원천을 다시 더하지 않는다. 이것은 공간적으로 결합된 잔여 자유도의 영향을 지우는 주장이 아니다. 실제 기록 간 결합을 Hamiltonian에 더하면 Pi_r와 H가 가환하지 않고 전이·간섭이 가능하다. 37번 검사는 이 양성대조다. 변하는 사영이나 비보존 기록에서는 사영 및 정규화의 시간미분항도 필요하다.

독립적인 trace-preserving 기록 조작은 전체 HC 축약상태를 바꾸지 않는다. 선별 결과를 얻은 조건부 변화와 원격 비선별 신호를 혼동하지 않는다. 코드 25–30번 검사는 이 구분과 조건부 잔여 분산을 확인한다.

## 3. 같은 연산자의 평균은 반작용, 분산은 초기 기록 형성을 정한다

서로 다른 고정 힉스 원천 ua,ub에 대해 H_C(u)=H_C,0+kappa u O와 U_a=exp(-it H_C(ua))를 사용한다. 환경 겹침은

$$\nu_{ab}(t)=Tr[U_a(t)rho_C U_b(t)^\dagger],\qquad rho_{H,ab}(t)=rho_{H,ab}(0)\nu_{ab}(t).$$

시간발전의 2차 전개와 정규화를 적용하면

$$|\nu_{ab}(t)|=1-\frac12\kappa^2(ua-ub)^2 Var_{rho_C}(O)t^2+O(t^3).$$

한편 J_u=-kappa <O>다. 서로 다른 기록 함수와 힘 함수를 맞춘 것이 아니라 같은 O의 평균과 분산이다. 모든 순수 환경 상태에서 |nu|<1은 대안별 환경 상태가 구별된다는 뜻이다. 안정한 거시 기록, 절대적 단일 결과 또는 초기 확률의 유도까지 뜻하지 않는다.

지정한 세 복소 조화모드에서 x0=(s0-2epsilon,s0+epsilon,s0+epsilon), omega_j(u)=sqrt(x0_j+kappa u)를 사용한다. 초기 원천 uref에서 Gaussian 진공을 준비한다. 복소모드 하나는 두 실수모드이므로 그 계수를 일관되게 포함한다. 한 실수모드의 Gaussian 폭은

$$a(t)=\frac{\omega_0\cos\omega t+i\omega\sin\omega t}{\cos\omega t+i(\omega_0/\omega)\sin\omega t}.$$

두 실수 Gaussian 상태 겹침의 제곱은 2 sqrt(Re a Re b)/|a+b*|이다. 두 실수모드로 구성한 복소모드의 겹침 절댓값은 이와 같다. 세 모드의 곱을 Fock 직접 대각화와 대조한다.

s0=.5, epsilon=.15, kappa=1, uref=.5, ua=.2, ub=.8은 진단 입력이다. t=1,2,4의 |nu|는 약 .905824,.881711,.953007이다. 재간섭이 있으므로 유한 환경을 불가역한 기록으로 승격하지 않는다. u=phi²/2인 포털은 phi의 부호를 구별하지 않는다. kappa=0에서는 기록 기전이 사라진다. 초기 힉스 대안들의 대각 확률은 이 제어 진화에서 보존된다.

## 4. 비균일 Euclidean 시공간 위상 비교 정리

유한 격자에서

$$K_\theta=L\otimes I_3+\bigoplus_x[(s0+\kappa u_x)I_3+Y(\theta_x)]$$

로 놓는다. theta_x,u_x는 점마다 달라도 된다. s_x>2epsilon이며 L은 비음수 edge weight의 Laplacian이다. K=D-T_theta로 분해하면 D의 성분은 공간 degree+s_x, T의 절댓값 행합은 degree+2epsilon이다. 따라서 R_theta=D^(-1/2)T_theta D^(-1/2)의 절댓값 행렬은 spectral radius<1이다. 대각 유사변환과 행합 상계로 이를 보인다.

T0=|T_theta|, K0=D-T0를 택한다. 일정한 내부 대각 unitary diag(1,exp(2pi i/3),exp(4pi i/3))를 모든 격자점에서 공통으로 사용하면 K_pi가 K0로 바뀐다. 공간 연결은 이 공통 기저변환에서 보존된다.

정확하게 수렴하는 급수는

$$\log\det K_\theta=\log\det D-\sum_{n\ge1}Tr(R_\theta^n)/n.$$

각 trace는 닫힌 경로들의 합이다. 각 경로의 실수 위상인자는 그 절댓값보다 클 수 없으므로 Tr(R_theta^n)<=Tr(|R_theta|^n)이다. 따라서

$$\boxed{\log\det K_\theta-\log\det K_\pi\ge0.}$$

이는 위상을 상수로 둘 필요가 없는 비교다. 양의 힉스 경로 측도로 점별 부등식을 적분하면 Z[theta_x]<=Z[pi], F[theta_x]>=F[pi]이다. 추가 위상 의존 국소 퍼텐셜, 부호가 다른 측도, 바뀌는 연결크기나 일반 행렬 Y에서는 다시 검사해야 한다. 기존 조건에 독립 항 -cos(theta)를 더하면 최소점 판정이 바뀔 수 있음을 음성대조에 남겼다.

이 정리는 외부로 지정한 위상 배경의 비용 순서다. theta(t,x)의 전체 Lorentzian 양자진화나 인플레이션 초기조건을 푼 것이 아니다. 유한온도의 theta 경로적분이 delta(theta-pi)라는 뜻도 아니다. 33점 양의 위상 측도에서는 모든 점의 가중치가 비영임을 별도 검사했다.

## 5. 국소 힉스 응답과 위상 Hessian

G=K^(-1)일 때 국소 상대 힉스 원천과 위상 Hessian은

$$J_{u_x}=-\kappa\,tr_3(G-G_\pi)_{xx},$$
$$H_{xy}=Tr(G K_{,xy})-Tr(G K_{,x}G K_{,y}).$$

같은 행렬식의 미분이므로 혼합미분이 서로 일치한다. 직접 유한차분, 양의 Hessian 고유값, closed-walk 급수와 직접 logdet를 대조했다. 128개 진단의 최소 delta Gamma는 양수다. 작은 부동소수점 계수의 음수 ~1e-17은 총 부등식이나 엄밀 interval 증명으로 오해하지 않는다. 해석적 급수 꼬리 상계와 부동소수점 오차는 별개다.

## 6. 계수를 자유롭게 최소화한 후보의 실패

별도의 parameter cost가 없는 H(kappa)=H(0)+kappa uO에서 uO>=0이므로 dE0/dkappa=<uO>>=0이다. kappa>=0를 자유롭게 최소화하면 kappa=0 경계를 선호한다. 기록이 필요하다는 조건은 비영 결합의 필요조건일 뿐 kappa=.5,1,2 중 하나를 유일하게 선택하지 않는다.

고정된 양의 힉스 연산자 고유값 h_n에서 theta=pi의 Gaussian 작용은 sum_n[ln(h_n-2epsilon)+2ln(h_n+epsilon)]다. epsilon을 자유롭게 하면

$$\partial_\epsilon\Gamma=-6\epsilon\sum_n[(h_n-2\epsilon)(h_n+\epsilon)]^{-1}<0.$$

단순한 최소화는 유한 간극 내부의 epsilon을 선택하지 않는다. 상대 퍼텐셜은 pi에서 모든 epsilon에 대해 0이라 그 자체로 구분하지 못한다. 이러한 자유화는 원래 고정-moment 조건을 풀어 검사한 별도 후보이며, 입자 상수가 실제로 이 유한 determinant를 최소화한다는 법칙은 아니다.

K->a²K는 상대 determinant를 바꾸지 않는다. 따라서 절대 eV 척도는 이 순서식에서 선택되지 않는다. 유니터리 시간발전도 서로 다른 전체 초기상태의 trace distance를 없애지 않는다. 초기 상태 선택을 이완과 혼동하지 않는다.

## 7. 판정과 다음 증명 의무

물리적으로 결합한 잔여가 기록 내부에서 반작용하고 힉스의 정보를 담을 수 있다는 기전은 통과했다. 힉스=선택된 세계 전체, 클라루스=버려진 다른 결과 전체라는 사상은 사용하지 않았다. 비균일 위상 비용의 최소화는 유일한 관측 결과가 아니다. 모든 입력상수, full SM, 중력, 우주 초기상태와 전체 공동 관측 점수는 미완성이다.

다음 과제는 원래 공간 운동부가 제공하는 물리적 장모드로 기록을 확장하고, 준비·소산·열의 자원을 함께 계산하며, 비영 결합과 유한 간극을 유지할 실제 미시 조건을 찾는 것이다. 환경 수나 잡음률을 관측에 맞춰 선택하지 않는다.

## 8. 재현 및 출처

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_selection.py --output rerun.json`

37개 테스트와 진단은 위 명시된 범위다. 이전 RH2의 39,930차원 바닥상태를 이번에 재실행한 것으로 세지 않는다. 게시한 원전·코드·요약과 원본 ZIP의 해시를 구별하며, 해시만으로 ZIP 원본이 저장소에 있다고 주장하지 않는다.

일반 도구의 원전: Schlosshauer, quant-ph/0312059; Ollivier, Poulin and Zurek, quant-ph/0408125; Zurek, quant-ph/0509174; Güneysu, Keller and Schmidt, arXiv:1301.1304. 이 문헌들은 CE의 선택 원리를 실험적으로 입증한 자료가 아니다.
