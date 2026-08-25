# 무차원 자기측정 시간과 기회비용 결합 연구 계약

Status: COMPLETE

PREDECESSOR:

- `_workspace/ce/quantum-path-opportunity-functional-20260825`
- `_workspace/ce/measurement-wall-dimensionality-20260825`
- `_workspace/ce/measurement-record-one-way-compatibility-20260825`
- `_workspace/ce/zero-dimensional-fold-memory-field-20260825`

## 1. 세 아이디어

1. 비선택 가능성의 기회비용 $C_I$.
2. completed record의 0D와 벽 강도에서 얻는 무차원 measurement time.
3. 한 특정 measurement time을 무수한 약한 자기측정 합성의 동치류로 보는 self-measurement 구조.

## 2. 무차원 measurement time

고정된 complete orthogonal record partition의 dephasing projector를

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r,
\qquad \mathcal D_P^2=\mathcal D_P
$$

로 둔다. partial wall channel과 additive 후보 좌표는

$$
\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P,
\qquad
\theta=-\ln(1-\eta),
\qquad 0\le\eta<1
$$

이다. $\theta$는 초 단위 시간이 아니라 dimensionless measurement depth다.

## 3. 무수한 약한 측정의 합성

고정된 $\mathcal D_P$에 대해

$$
\Phi_{\eta_2}\circ\Phi_{\eta_1}
=\Phi_{\eta_1+\eta_2-\eta_1\eta_2}
$$

와 $\theta_{12}=\theta_1+\theta_2$를 검사한다. 임의 partition
$\theta_*=\sum_{k=1}^N\delta\theta_k$에서

$$
\eta_k=1-e^{-\delta\theta_k}
$$

를 쓰면 finite $N$에서도 합성이 $\Phi_{1-e^{-\theta_*}}$와 정확히 같은지, $N\to\infty$ weak limit가

$$
\Phi_\theta=e^{\theta(\mathcal D_P-I)}
$$

인지 검산한다.

## 4. self-measurement의 operational 정의

“자기가 자신을 본다”를 한 Hilbert factor가 자기 전체 상태를 완전히 복제한다는 뜻으로 두지 않는다. 더 큰 system $U$ 안의 object subsystem $S$, record/controller $R$, 또는 successive time slices를 구분하고, past-adapted recursion

$$
\rho_{n+1|r_n}
=\frac{\mathcal I_{r_n}^{(m_n)}(\rho_n)}
{\operatorname{tr}\mathcal I_{r_n}^{(m_n)}(\rho_n)},
\qquad
m_{n+1}=F(m_n,r_n)
$$

로 정의한다. $\mathcal I^{(m_n)}$는 $m_n$까지의 과거 record로만 정한다. 현재 outcome이나 미래 record에 의존하는 circular rule은 제외한다. exact unknown-state cloning 또는 one-copy full tomography는 주장하지 않는다.

## 5. 기회비용과 measurement depth

time-indexed instrument가 $p_a(\theta)$를 주면

$$
\overline C_I(\theta)
=\sum_a p_a(\theta)[1-p_a(\theta)][-\ln p_a(\theta)]
$$

와

$$
C_{\rm self}(\theta_*)
=\int_0^{\theta_*}e^{-\theta}
\overline C_I(\theta)d\theta
$$

를 사전 고정한다. 이는 $d\eta=e^{-\theta}d\theta$를 쓴 기존 $C_{\rm wall}$의 reparameterization인지 검사한다. finite alphabet size $n$에서

$$
0\le C_{\rm self}(\theta_*)
\le(1-e^{-\theta_*})\ln n
$$

를 증명하거나 반례를 찾는다.

## 6. fold-memory에 대한 후보 deposition

$$
d\mu_{\rm self}(\theta,B)
=e^{-\theta}\sum_a p_a(\theta)[1-p_a(\theta)]
[-\ln p_a(\theta)]\mathbf1_B(F_\theta(a))d\theta
$$

를 양의 무차원 후보 measure로만 검사한다. 이를 persistent $\mu_F$로 보존하는 retention map, 환경장 $\chi$와의 feedback, energy/stress는 별도 공리로 유지한다.

## 7. 반증·범위 조건

1. 고정 $\mathcal D_P$에서도 $\theta$ composition이 additive하지 않는 반례.
2. noncommuting/time-dependent record partitions에서도 단일 scalar $\theta$가 path-independent하다는 부모 주장.
3. non-Markovian recoherence에서도 monotone $\eta$와 $\theta$가 보편적으로 존재한다는 부모 주장.
4. record subsystem 또는 temporal split 없이 한 copy가 자기 unknown state 전체를 완전히 읽고 보존한다는 주장.
5. $N\to\infty$가 무한 information, energy 또는 dark density를 자동 생성한다는 주장.

## 8. 고정 예제와 허용오차

qubit $P_0,P_1$와 $\rho_0=|+\rangle\langle+|$를 사용한다. $\theta_*=1.5$를 $N=1,2,5,100$개의 equal increments로 분할해 exact channel equality를 $10^{-12}$ tolerance로 검사한다. $(p_0,p_1)=(0.8,0.2)$가 일정한 control에서 $C_{\rm self}=(1-e^{-1.5})\overline C_I$인지 확인한다.

## 9. 주장 상한

통과 가능한 최강 결론은 fixed dephasing semigroup에서 $\theta$가 repeated weak self-monitoring의 additive operational coordinate이고, opportunity cost가 그 좌표에서 bounded dimensionless functional이라는 것이다. 서로 commuting하는 여러 generator로의 확장은 각 generator의 가법성, 공통 domain과 경로 독립성을 별도로 입증한 경우에만 포함한다. physical time, consciousness, objective collapse, external $Z$ identity, energy, stress 또는 dark-sector identity는 주장하지 않는다.

## 10. 사용자 확장: 자기비동일성으로 정의한 흐름

연속 parameter에는 “직전 점”이 없으므로 다음 operational 명제로
바꾸어 검증한다. trace distance

$$
D_{\rm tr}(\rho,\sigma)=\frac12\|\rho-\sigma\|_1
$$

를 metric으로 쓰고, absolutely continuous path $\rho_\theta$의 metric
speed와 길이를

$$
v(\theta)=\lim_{h\downarrow0}
\frac{D_{\rm tr}(\rho_{\theta+h},\rho_\theta)}{h},
\qquad
L(\theta_*)=\int_0^{\theta_*}v(\theta)d\theta
$$

로 정의한다. 고정 dephasing semigroup에서 초기 off-diagonal 성분이
0이 아닐 때 모든 finite $\theta$와 $h>0$에 대해
$\rho_{\theta+h}\ne\rho_\theta$인지, $v$, $L$, residual clock을 닫힌식으로
유도한다. 무한 분할에서 각 increment는 0으로 가지만 총길이가 finite한지
검증한다.

완전 반례로는 이미 dephased된 stationary state와, 모든 국소 구간에서
변하지만 한 주기 뒤 원상복귀하는 unitary orbit를 사용한다. 따라서
“계속 다르다”만으로 보편 물리시간, 시간의 화살 또는 비가역성이
증명된다고 주장하지 않는다. 그런 결론에는 fixed reference에 대한
strict Lyapunov/contractivity 조건이 추가로 필요하다.
