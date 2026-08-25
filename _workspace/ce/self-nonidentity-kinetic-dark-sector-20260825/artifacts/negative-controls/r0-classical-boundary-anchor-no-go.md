# R0 음성대조군 — 고전 경계항 하나로 앵커와 양의 전류를 함께 유도할 수 없음

Objective-ID: SNKC-QPATH-ORIGIN-01
Failure-Equation: 정확한 $T|_{\Sigma_*}=0$과 $S_{\Sigma_*}=\int_{\Sigma_*}\sqrt h\,[-\Pi_FT]$에서 $J_i=\Pi_F>0$를 동시에 자연 경계조건으로 얻는 식
Minimal-Assumptions: 시간 차원의 $T$, 무차원 $X$, 국소 고전 벌크 작용 $P(T,X)$, 하나의 초기 초곡면 $\Sigma_*$, 자유로운 경계 변분 또는 정확한 디리클레 조건
Counterexample: 정확한 디리클레 조건은 $\delta T|_{\Sigma_*}=0$을 강제하므로 경계항 변분이 사라지고 $J_i$를 정하지 못한다
First-Failing-Line: $\delta T|_{\Sigma_*}=0\Rightarrow\delta S_{\Sigma_*}=0$이므로 $J_i+B_T=0$을 경계 방정식으로 사용할 수 없다
Failure-Type: 초기값·변분·차원·국소성
Removed-Claims: 고전 국소 경계항 하나가 정확한 $T_i=0$과 $\Pi_{\rm fold}>0$의 값 및 시간 화살을 모두 유도한다는 주장
Preserved-Objective: 비선택 경로의 보존 기록이 양의 초기 전류를 준비하고 동일 벌크장의 암흑물질형·암흑에너지형 성분으로 읽히는 보존적 미시 완성을 찾는 목표
Regression-Test: 정확한 디리클레 조건과 자유 경계 변분을 동시에 사용하거나 무차원 0D 자료만으로 질량차원 4의 $\Pi_F$를 만들면 실패로 판정한다

## 1. 차원

자연단위계에서 $T$는 시간 차원이므로 질량차원은 $[T]=-1$이다. 따라서

$$
[X]=0,\qquad [P_X]=[J_i]=[\Pi_F]=4,\qquad
[d^3x]=-3.
$$

경계 라그랑지안 밀도 $B$의 차원은 3이어야 한다. 선형 경계항은

$$
B(T)=-\Pi_FT,\qquad \Pi_F=\Lambda_\Pi^4
f(\mu_F,C_{\rm self})
$$

처럼 써야 차원이 맞는다. 여기서 $\Lambda_\Pi$는 질량차원 1의 독립 matching
척도다. 무차원 carrier 자료만으로 이 척도를 만들 수 없다.

## 2. 경계 변분

벌크 작용을 변분하면 초기 경계에서

$$
\delta S_{\rm bulk}\big|_{\Sigma_*}
=\int_{\Sigma_*}d^3x\sqrt h\,J_i\,\delta T_i
$$

를 얻는다. $T_i$를 자유롭게 변분하면 $B_T=-\Pi_F$와 함께

$$
J_i+B_T=0
\quad\Longrightarrow\quad
J_i=\Pi_F
$$

가 된다. 그러나 이 계산은 $T_i$를 자유롭게 두었을 때만 성립한다. 정확한
$T_i=0$을 디리클레 자료로 고정하면 $\delta T_i=0$이므로 위 경계 방정식은
나오지 않는다.

경계 승수 $\lambda T_i$를 더해도 두 조건을 동시에 유도하지 못한다. $\lambda$
변분은 $T_i=0$을 주지만 $T_i$ 변분은

$$
J_i-\Pi_F+\lambda=0
$$

을 주므로 자유로운 $\lambda$가 전류를 다시 흡수한다.

## 3. 0차원과 FLRW 균일성

시공간의 한 점에 놓인 엄격한 0차원 원천은 $\delta^{(4)}(x-x_0)$로 표현되며
균일한 FLRW 배경을 깨뜨린다. 전 공간에 같은 $J_i$를 준비하려면 균일한
$\Sigma_*$, 공간 분포 또는 coarse-grained record density가 추가로 필요하다.
따라서 점 하나의 0차원성만으로 homogeneous 초기 전류를 얻었다고 주장할 수
없다.

## 4. 음성대조 판정

이 반례는 기존 벌크 $P(T,X)$의 조건부 보존 정리를 무너뜨리지 않는다. 제거되는
것은 양의 $\Pi_{\rm fold}$를 고전 경계항 하나가 미시적으로 유도한다는 더 강한
주장이다. 다음 경로는 앵커를 정확한 동시 고정값이 아니라 양의 초기 양자상태의
평균과 공분산으로 준비하고, reservoir까지 포함한 총 보존 장부를 제시해야 한다.
