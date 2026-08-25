# 무차원 자기측정 깊이의 수학 감사

Status: COMPLETE

## 1. 고정 측정 분할과 부분 벽 채널

완전 직교 기록 분할 $P=\{P_r\}$에 대해

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r,
\qquad
\mathcal D_P^2=\mathcal D_P
$$

로 둔다. $0\leq\eta<1$인 부분 측정 벽은

$$
\Phi_\eta
=(1-\eta)\operatorname{Id}+\eta\mathcal D_P
=\mathcal D_P+(1-\eta)(\operatorname{Id}-\mathcal D_P)
$$

이다. 항등 채널과 dephasing 채널의 convex combination이므로 CPTP이다.

## 2. 정리: 측정깊이의 가법성과 정확한 분할 불변성

**정리.** 같은 $\mathcal D_P$를 사용하는 두 채널의 합성은

$$
\Phi_{\eta_2}\circ\Phi_{\eta_1}
=\Phi_{\eta_1+\eta_2-\eta_1\eta_2},
$$

따라서

$$
1-\eta_{12}=(1-\eta_1)(1-\eta_2)
$$

를 만족한다. 그러므로

$$
\boxed{\theta=-\ln(1-\eta)}
$$

를 정의하면 $\theta_{12}=\theta_1+\theta_2$이다.

**증명.** $Q=\operatorname{Id}-\mathcal D_P$로 두면
$\mathcal D_PQ=Q\mathcal D_P=0$, $Q^2=Q$이다. 따라서

$$
(\mathcal D_P+q_2Q)(\mathcal D_P+q_1Q)
=\mathcal D_P+q_2q_1Q,
\qquad q_k=1-\eta_k.
$$

이는 위 합성식과 로그 가법성을 동시에 준다. $\square$

더 강하게, 임의의 유한 분할

$$
\Pi(\theta_*)=
\left\{(\delta\theta_1,\ldots,\delta\theta_N):
N\geq1,\ \delta\theta_k\geq0,\ 
\sum_{k=1}^N\delta\theta_k=\theta_*\right\}
$$

과 $\eta_k=1-e^{-\delta\theta_k}$에 대해

$$
\prod_{k=1}^N\Phi_{\eta_k}
=\Phi_{1-e^{-\theta_*}}
$$

가 모든 유한 $N$에서 정확히 성립한다. 따라서 하나의 특정
$\theta_*$는 그와 같은 총효과를 내는 모든 약한 측정 분할의
**채널 동치류**를 표지한다. $N\to\infty$는 이 정확한 항등식의
연속 표현일 뿐이며, 실제로 무한개의 독립 관측자나 무한 정보가
생긴다는 뜻은 아니다.

이 동치는 unconditional state channel의 동치이다. 각 분할에서 생기는
outcome history, conditional trajectory 또는 feedback law까지 동일하다는
뜻은 아니다.

연속 semigroup 표기는

$$
\Phi_\theta
=e^{\theta(\mathcal D_P-\operatorname{Id})}
=\mathcal D_P+e^{-\theta}(\operatorname{Id}-\mathcal D_P)
$$

이다. 이때 $\theta$는 물리시간이 아니라 무차원 누적 측정깊이다.
시간률 $\gamma(t)$가 따로 주어진 Markov 모형에서만
$\theta(t)=\int_0^t\gamma(s)ds$라고 연결할 수 있다.

## 3. 자기측정의 비순환 operational 정의

전체 장치를 하나의 폐쇄된 표지 $U$로 부를 수는 있지만, 실제
instrument에는 object $S$와 record/controller $R$, 또는 앞선 시각과
뒤 시각의 분리가 필요하다. 기록 filtration을

$$
\mathcal F_n=\sigma(r_0,\ldots,r_n)
$$

로 두고, $n$번째 instrument 설정 $m_n$은 $\mathcal F_{n-1}$에 대해
measurable하다고 요구한다. 그러면

$$
p(r_n\mid\mathcal F_{n-1})
=\operatorname{tr}\mathcal I_{r_n}^{(m_n)}(\rho_n),
$$

$$
\rho_{n+1\mid r_n}
=\frac{\mathcal I_{r_n}^{(m_n)}(\rho_n)}
{\operatorname{tr}\mathcal I_{r_n}^{(m_n)}(\rho_n)},
\qquad
m_{n+1}=F(m_n,r_n)
$$

가 causal한 자기모니터링 recursion을 이룬다. 현재 결과 $r_n$을
$m_n$의 입력으로 미리 사용하거나 미래 기록을 참조하면 정의가
순환한다. 이 구조는 한 개의 미분할 Hilbert factor가 자기의 임의
unknown state 전체를 완전 복제한다는 주장이 아니므로 no-cloning과
충돌하지 않는다.

## 4. 기회비용의 측정깊이 표현과 유계성

유한 outcome alphabet의 conditional probabilities를
$p_a(\theta)$라고 하고 $0\ln0:=0$으로 둔다. 예측 평균 기회비용은

$$
\overline C_I(\theta)
=\sum_a p_a(\theta)[1-p_a(\theta)][-\ln p_a(\theta)]
$$

이다. 기존 wall 변수와 $d\eta=e^{-\theta}d\theta$의 관계를 쓰면

$$
\boxed{
C_{\rm self}(\theta_*)
=\int_0^{\theta_*}e^{-\theta}\overline C_I(\theta)d\theta
}
$$

를 얻는다. 각 항에 $0\leq1-p_a\leq1$을 적용하면

$$
0\leq\overline C_I(\theta)
\leq-\sum_ap_a(\theta)\ln p_a(\theta)
\leq\ln n.
$$

따라서

$$
\boxed{
0\leq C_{\rm self}(\theta_*)
\leq(1-e^{-\theta_*})\ln n
\leq\ln n
}
$$

이다. 약한 측정 분할의 개수 $N$을 무한히 늘려도 이 functional은
발산하지 않는다. 연속 출력의 raw differential entropy가 sampling,
bandwidth 또는 reference measure에 따라 발산할 수 있다는 별도 문제와
혼동하면 안 된다.

$p=(0.8,0.2)$가 일정하고 $\theta_*=1.5$이면

$$
\overline C_I=0.293213034199730,
\qquad
C_{\rm self}=0.227788362921137.
$$

## 5. 0D 기록으로의 후보 pushforward

outcome $a$를 공간의 point carrier로 보내는 measurable map
$F_\theta$를 별도 공리로 줄 때에만

$$
d\mu_{\rm self}(\theta,B)
=e^{-\theta}\sum_a p_a(\theta)[1-p_a(\theta)]
[-\ln p_a(\theta)]\mathbf1_B(F_\theta(a))d\theta
$$

라는 양의 유한 measure를 정의할 수 있다. 전체 질량은
$C_{\rm self}$와 같고 $\ln n$ 이하이다. 이것은 정보 가중 measure이지
에너지 measure가 아니다. persistent $\mu_F$가 되려면 retention map이,
중력원이 되려면 action, 독립 energy-density scale과 보존법칙이 더 필요하다.

## 6. 정리: 무수한 자기비동일성의 유한 흐름

연속 parameter 집합은 dense하므로 한 점의 “직전 점”은 존재하지 않는다.
따라서 문장을 모든 양의 increment에 대한 비동일성과 metric derivative로
정식화한다. $Q=\operatorname{Id}-\mathcal D_P$와

$$
A=Q\rho_0
$$

를 두면 고정 semigroup 경로는

$$
\rho_\theta=\mathcal D_P\rho_0+e^{-\theta}A
$$

이다. 임의의 finite $\theta\geq0$와 $h>0$에 대해

$$
\rho_{\theta+h}-\rho_\theta
=-e^{-\theta}(1-e^{-h})A.
$$

따라서

$$
\boxed{
A\ne0
\quad\Longleftrightarrow\quad
\rho_{\theta+h}\ne\rho_\theta
\text{ for every finite }\theta\geq0,\ h>0
}
$$

이다. 이 의미에서 자기 자신이 이전의 자신과 같지 않음이 임의로 미세한
분할마다 반복된다.

trace distance를 쓰면 정확히

$$
D_{\rm tr}(\rho_{\theta+h},\rho_\theta)
=\frac12e^{-\theta}(1-e^{-h})\|A\|_1
$$

이고 metric speed와 누적 경로길이는

$$
v(\theta)=\frac12e^{-\theta}\|A\|_1,
$$

$$
\boxed{
L(\theta_*)
=\int_0^{\theta_*}v(\theta)d\theta
=\frac12(1-e^{-\theta_*})\|A\|_1
}
$$

이다. 분할의 mesh를 0으로 보내면 nonzero increment의 수는 제한 없이
늘지만, 합은 이 finite path length로 수렴한다. 따라서 “무수한 변화”는
“무한 총변화”와 동의어가 아니다.

fixed point에 대한 residual은

$$
R(\theta)
=D_{\rm tr}(\rho_\theta,\mathcal D_P\rho_0)
=\frac12e^{-\theta}\|A\|_1
$$

이므로 $A\ne0$일 때

$$
\boxed{
\theta
=\ln\frac{R(0)}{R(\theta)}
=\ln\frac{v(0)}{v(\theta)}
=-\ln\left(1-\frac{L(\theta)}{L(\infty)}\right)
}
$$

로 측정깊이를 자기차이의 감소에서 복원할 수 있다. 이것이 이 모형 안에서
“흐름”에 방향을 주는 strict Lyapunov coordinate이다. 물리시간을 얻으려면
여전히 $\theta(t)=\int\gamma(t)dt$라는 독립 rate/clock bridge가 필요하다.

기회비용과의 조립도 정확하다. $A\ne0$이면

$$
dC_{\rm self}=e^{-\theta}\overline C_I(\theta)d\theta,
\qquad
dL=\frac12e^{-\theta}\|A\|_1d\theta,
$$

따라서

$$
\boxed{
dC_{\rm self}
=\frac{2\overline C_I(\theta)}{\|A\|_1}dL
}
$$

이다. $\overline C_I$가 일정하면
$C_{\rm self}=2\overline C_I L/\|A\|_1$이다. 하지만
$A=0$인 diagonal state에서도 outcome distribution을 이용한
$\overline C_I$는 양수일 수 있다. 그러므로 opportunity accounting과
actual state motion은 일반적으로 같은 양이 아니며, 위 식은
$A\ne0$인 선택한 경로에서만 성립한다.

## 7. 완전 반례 1: 비가환 측정축

qubit Bloch vector에 대해 축 $\mathbf n$의 dephasing은

$$
\mathbf r\longmapsto(\mathbf n\cdot\mathbf r)\mathbf n
$$

이다. $z$축과 $\mathbf m=(x+z)/\sqrt2$축, $\eta_z=0.7$,
$\eta_m=0.2$, 입력 $\rho=(I+M)/2$를 택하면

$$
\left\|
\Phi_{m,0.2}\Phi_{z,0.7}(\rho)
-\Phi_{z,0.7}\Phi_{m,0.2}(\rho)
\right\|_F
=0.0494974746830583>0.
$$

따라서 changing/noncommuting partitions에는 순서와 경로를 지우는
보편 단일 scalar $\theta$가 없다. 이 경우 path-ordered superoperator나
여러 개의 제어 좌표가 필요하다.

## 8. 완전 반례 2: non-Markovian recoherence

coherence multiplier가

$$
\lambda(t)=\cos^2(gt/2),
\qquad
\rho_{01}(t)=\lambda(t)\rho_{01}(0)
$$

인 CPTP dephasing family를 생각하자. $gt=0,\pi,2\pi$에서
$\lambda=1,0,1$이므로 $\eta=1-\lambda$는 $0,1,0$으로 되돌아간다.
$\theta=-\ln\lambda$는 중간점에서 발산한 뒤 다시 0으로 돌아오므로
전 구간의 finite monotone 측정깊이가 될 수 없다. 환경 기억과
recoherence가 있는 일반 과정에는 process tensor 또는 memory kernel가
필요하다.

## 9. 완전 반례 3: 국소 변화만으로는 시간의 화살이 안 나옴

unitary orbit

$$
\rho_t=e^{-i\omega Zt/2}|+\rangle\langle+|e^{i\omega Zt/2}
$$

는 모든 충분히 작은 $h>0$에서 $\rho_{t+h}\ne\rho_t$이고 trace-distance
speed가 $\omega/2$이지만

$$
\rho_{2\pi/\omega}=\rho_0.
$$

한 주기의 path length는 $\pi$인데 endpoint distance는 0이다. 따라서
자기비동일성의 무수한 반복은 흐름의 경로량을 정의하기에는 충분하지만,
보편적인 시간 방향이나 비가역성을 증명하기에는 불충분하다. fixed point에
대한 strict monotone residual 같은 조건이 추가되어야 한다.

또한 $A=0$, 즉 $\rho_0=\mathcal D_P\rho_0$이면 모든 $\theta$에서
$\rho_\theta=\rho_0$이다. 이는 모든 초기상태가 자기차이 흐름을 가진다는
부모 명제의 직접 반례이다.

## 10. 형식 지위

| claim | status | reason |
|---|---|---|
| 고정 $\mathcal D_P$에서 $\theta$ 가법성과 분할 불변성 | [정리] | idempotence로 정확히 증명 |
| past-adapted subsystem/temporal self-monitoring | [정의/조건부 구성] | instrument와 filtration을 명시해야 함 |
| $C_{\rm self}$의 $\ln n$ 상계 | [정리] | finite alphabet에서 증명 |
| $d\mu_{\rm self}$ | [정의/미완성 다리] | pushforward와 retention은 독립 공리 |
| $A\ne0$인 fixed dephasing 경로의 모든-step 비동일성, speed와 length | [정리] | trace distance로 닫힌식 증명 |
| residual에서 $\theta$ 복원 | [조건부 정리] | fixed $\mathcal D_P$, calibrated $\rho_0$, $A\ne0$ 필요 |
| 자기차이 흐름과 opportunity cost의 국소 결합 | [조건부 산출] | $A\ne0$; 두 양의 존재론적 동일시는 아님 |
| 모든 상태가 항상 자기차이 흐름을 가짐 | [삭제: 완전 반례] | $A=0$ stationary state |
| 국소 자기비동일성이 보편 시간의 화살을 증명 | [삭제: 완전 반례] | periodic unitary return |
| 임의의 비가환 측정에 보편 scalar $\theta$ | [삭제: 완전 반례] | order dependence |
| non-Markov 과정에 전역 monotone $\theta$ | [삭제: 완전 반례] | recoherence revival |
| 한 복사본의 완전 자기상태 관측 | [삭제: 기존 no-cloning과 type 오류] | object/record 또는 시간 분리 필요 |
| 무한 분할이 무한 정보·에너지를 자동 생성 | [삭제: 완전 반례] | $C_{\rm self}\leq\ln n$ |
| $C_{\rm self}$ 자체가 에너지·암흑성분 | [미완성] | 차원 있는 scale과 action 부재 |
