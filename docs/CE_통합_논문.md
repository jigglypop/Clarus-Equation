# Clarus Equation 통합 논문: 선택, 접힘, readout과 조건부 응용의 현재 지형

## 초록

Clarus Equation(CE)은 하나의 숫자식이나 완결된 통일장이 아니다. 이 문서가
정리하는 CE의 중심은 세 단계의 물리 서사다. 환경과의 상호작용이 가능한
성분들 가운데 한 사건을 **끼워** 선택하게 하고, 선택되지 않은 성분을
단순 소거가 아니라 **접힘**으로 보존하며, 그 보존된 성분을 우주론에서
암흑 부문으로 **표현**한다는 그림이다. 이 그림은 동기를 주는
**[공리: 물리 사상]**이다. 반면 그 그림에 인접한 수학에는 완결된 정리,
명시적 EFT, 조건부 수치 산출, 경험식과 아직 비어 있는 bridge가 함께
있다. 특히 현재 R1--R2 경로는 비선택 경로를 독립 에너지라고 선언하지 않고,
무차원 잔류량ㆍ양의 초기 Gaussian 운동량ㆍ유계 Gaussian 저장소를 분리해
조건부 열린계 작용으로 만든다. 이 작용은 인과적 반응, 양의 잡음과 총응력
보존을 시험한다. R2는 그 뒤 단일-clock 관측창에서 물리 곡률섭동과 cutoff를
검사했을 뿐이며, 0차원 기록에서의 초기값 산출과 암흑부문 abundance 예측은 아직
제공하지 않는다.

이 통합 논문은 저장소의 이 다층 구조를 처음 읽는 독자가 한 흐름으로
따르도록 쓴 논문형 정본이다. 우선 형식 지위와 기호를 고정하고, 선택–접힘–
암흑 표현의 사슬을 설명한다. 이어 곱적 함수, Hodge 폐쇄, Poisson 고정점,
공변 $Z_2$ portal EFT, 유한 Euclidean cutoff, flavor와 우주론 branch를
전개한다. 마지막에는 등호이전 수학, 뇌·생명·AGI, 공학과 리만/MRA 응용을
각각 조건부 결과와 검증 상태로 분리한다.

독자는 미적분, 선형대수, 기초 확률론과 표준 장론·우주론 용어를 한 번은
접했다고 가정한다. 처음 읽을 때에는 1–8장을 순서대로 읽고, 관심 분야에
따라 9–13장을 읽은 뒤 14–16장의 재현과 열린 문제로 돌아오는 것이 좋다.
증명 전문과 데이터·판본의 상세는 후속 링크에 남기지만, 이 문서 자체만으로
현재 무엇이 닫혔고 무엇이 닫히지 않았는지는 판단할 수 있게 한다.

## 1. 형식 지위와 독해법

중심 서사를 수치 적합이나 구현 검사와 혼동하지 않기 위해, 먼저 모든
비자명 문장을 일곱 지위로 읽는다. **[정의]**는 대상·기호·정의역을
고정한다. **[정리]**는 명시한 전제에서 빠짐없는 증명이 있는 명제다.
**[공리]**는 모델 선택, 경계조건, 물리 사상 또는 외부 입력이다.
**[산출]**은 앞선 정의·공리·정리의 직접 계산이고, **[경험식]**은 자료,
보정 또는 현상론적 관계다. **[미완성]**은 작용·사상·증명·독립 데이터 중
하나가 비어 있음을 뜻하며, **[예측]**은 입력과 판정 절차를 자료 전에
고정한 경우에만 쓴다.

이 구분의 실천적 의미는 간단하다. 예컨대 $q_{\rm ext}$가 특정 방정식의
최소 소멸확률이라는 것은 **[정리]**일 수 있지만, 그것을 현재 바리온
분율 $\Omega_b$로 읽는 일은 별도 **[공리]**다. 모델 값이 관측값에
가깝다는 사실은 이 사상을 정리로 바꾸지 않는다. 반대로 공리와
미완성이 있다는 사실이 조건부 정리를 무효로 만들지도 않는다. 이 문서는
두 층을 한 등호로 합치지 않는다.

장론 절에서는 자연단위 $c=\hbar=1$, metric 부호 $(-+++)$, reduced
Planck mass $M_{\rm Pl}^{-2}=8\pi G$를 쓴다. 따라서 작용은 무차원이며
지수·로그·확률의 인자는 무차원이어야 한다. 이 규칙은 후술할
Euclidean 가중 $e^{-S_E/\hbar}$와 경로 선택 functional에도 적용된다.
형식 지위의 전체 규약과 개별 증명은 [공리계](axium.md)와
[핵심 정리 증명](검증_원장/참조_핵심_정리_증명.md)에 보존한다.

## 2. 선택과 접힘의 물리 서사

형식 지위를 고정했으므로 이제 CE가 무엇을 설명하려 하는지 말할 수 있다.
첫 단계인 **끼임**은 고립된 양자 하나가 임의로 답을 고른다는 그림을
거부한다. 계와 환경이 상호작용하면 환경은 위상 정보를 흡수하고, 특정
기저에서 간섭항이 억제된다. CE는 이 환경 선택을 이웃 전체가 한 성분을
사건으로 강제하는 과정으로 읽는다. 같은 받침대 위의 메트로놈이 받침대
매개 결합으로 박자를 맞추는 모습은 이 직관에 도움이 되지만, 고전 진동은
복소 진폭·간섭·완전양성 동역학을 포함하지 않는다는 한계가 있다.

둘째 단계인 **접힘**은 선택되지 않은 성분을 ‘없어졌다’고 선언하지
않는다. CE의 수학적 후보는 비선택 경로에 제한된 잔류 측도와 그
pushforward다. 종이를 접으면 안쪽 면이 눈에는 사라져도 종이의 질량은
사라지지 않는다는 비유가 여기서 쓰인다. 그러나 종이의 접힌 방향은
실제 상태공간, 국소 장, 공변 작용을 정의하지 않는다. 따라서 비유가
물리적 보존 법칙을 증명하는 것은 아니며, 잔류장과 분지과정 생존분율의
동일시는 **[미완성]**이다.

셋째 단계인 **암흑 표현**은 접힌 분율을 우주에서 어떻게 읽을 것인가의
질문이다. 선택되어 끝난 분율을 보이는 물질로, 남은 분율을 암흑물질과
암흑에너지의 근원으로 해석하는 것이 CE의 동기다. 다만 확률을 stress
tensor와 에너지 밀도로 바꾸는 사상은 분지과정 정리가 제공하지 않는다.
이 구분을 유지한 가장 상세한 사슬은 [선택과 접힘](5_유도/00_선택과_접힘.md)에
있으며, 본 논문은 그 사슬을 다음 장의 닫힌 수학과 연결한다.

## 3. 핵심 수학: 곱적 구조, Hodge 폐쇄, 고정점

선택 서사를 수학적으로 다룰 때 첫 도구는 독립 사건의 조합을 보존하는
함수다. 연속 함수 $I:(0,1]\to(0,1]$가 $I(xy)=I(x)I(y)$와 $I(1)=1$을
만족하면, $x=e^{-u}$로 치환하여 가법 함수 방정식으로 옮길 수 있다.

$$
I(e^{-u-v})=I(e^{-u})I(e^{-v}).
$$

연속성 아래 로그를 취하면 $-log I(e^{-u})$는 $u$의 선형 함수다.
따라서

$$
I(x)=x^c,\qquad c\geq0.
$$

이는 **[정리]**이다. 선형 readout $c=1$은 CE가 채택할 수 있는
**[공리]**이지, 함수 방정식이 강제하는 유일한 선택은 아니다. 독립
확률의 곱을 보존한다는 가정이 이 결론의 핵심이며, 상관된 사건에는
그대로 적용할 수 없다.

공간 차원의 조건부 도구는 Hodge 사상이다. 방향과 양의 내적을 가진
$d$차원 공간에서 $*: \Lambda^2V^*\to\Lambda^{d-2}V^*$다. 2-form과
1-form이 추가 지표 없이 맞물리려면 성분 수가 같아야 하므로

$$
\binom d2=d
\quad\Longleftrightarrow\quad d(d-3)=0.
$$

비자명한 양의 정수해는 $d=3$이다. 이는 Hodge 폐쇄의 **[정리]**다.
실제 공간 차원이 이 branch라는 해석, 그리고 $N_c=d$, $N_w=d-1$,
$N_{\rm gen}=d$ 같은 표준모형 수의 식별은 추가 **[공리]**다. 숫자
일치만으로 게이지군·표현·hypercharge가 유도되었다고 말할 수 없다.

접힘 분율을 계산하는 닫힌 수학은 Poisson 분지과정이다. 자손 수가 평균
$D$인 독립 Poisson 변수라면 확률생성함수는 $f(z)=e^{D(z-1)}$다. 세대
$n$ 안에 소멸할 확률을 $q_n$이라 하면 독립성 때문에

$$
q_{n+1}=f(q_n)=e^{-D(1-q_n)},\qquad q_0=0.
$$

$q_n$은 증가하고 1로 유계이므로 최소 고정점 $q_{\rm ext}$로 수렴한다.
$D>1$에서 그 비자명 해는

$$
q_{\rm ext}=-\frac1D W_0(-De^{-D}),
\qquad s_{\rm branch}=1-q_{\rm ext}.
$$

여기서 $W_0$는 Lambert $W$ 함수의 주가지다. $q=1$도 고정점이므로
유일한 고정점이 아니라 ‘최소 소멸 고정점’이라고 부른다. 연기 경보기의
신호 계보는 이 수학의 좋은 비유지만, 양의 자손 수가 복소 양자 진폭에서
어떻게 나오는지는 다음 장의 열린 bridge다.

### 3.1 연속 곱적 함수

곱적 readout 정리의 전제는 $I:(0,1]\to(0,1]$의 연속성과
$I(xy)=I(x)I(y)$다. $x=e^{-u}$와 $J(u)=-\log I(e^{-u})$를 두면

$$
J(u+v)=J(u)+J(v).
$$

연속 Cauchy 가법 함수는 $J(u)=cu$이므로
$I(x)=x^c$가 된다. $c\geq0$은 range 조건에서 온다. 독립 사건의 곱이
아닌 상관된 readout에는 이 한 줄 유도를 적용할 수 없다.

### 3.2 외대수 성분수와 Hodge

$d$차원에서 1-form 수는 $d$, 2-form 수는 $\binom d2$다. 두 수를
같게 두면

$$
d=\frac{d(d-1)}2\Longleftrightarrow d(d-3)=0.
$$

양의 비자명 해는 $d=3$이고, 방향·양의 내적에서 Hodge 사상은
$\Lambda^2V^*$를 $\Lambda^1V^*$에 대응시킨다. 이는 수학 정리이며,
실제 공간 또는 표준모형 세대 수로의 식별은 공리다.

### 3.3 Poisson 고정점과 최소성

평균 $D$의 Poisson 자손수 $K$에 대해 생성함수는
$f(z)=\mathbb E z^K=e^{D(z-1)}$다. $q_n$을 $n$세대 안의 소멸확률로
두면 독립 자손 때문에

$$
q_{n+1}=f(q_n),\qquad q_0=0.
$$

$q_n$은 증가하고 1로 유계이므로 극한이 존재하며, 연속성으로
$q=f(q)$의 최소해가 된다. $D\leq1$에는 1만, $D>1$에는 비자명 최소해가
있다. 이는 offspring의 독립성과 비음수성이 깨지면 보장되지 않는다.

### 3.4 균일 축약과 Lambert W

$A=dI+\delta B$, $B\boldsymbol1=\boldsymbol1$이면
$A\boldsymbol1=(d+\delta)\boldsymbol1$이므로 균일 sector의 깊이는
$D_{\rm eff}=d+\delta$다. 고정점에 $-Dq$를 곱하면

$$
(-Dq)e^{-Dq}=-De^{-D},
\qquad
q_{\rm ext}=-D^{-1}W_0(-De^{-D}).
$$

다른 실가지는 $q=1$에 해당한다. 이 축약은 지정 toy family의 산출이지
양자 질량행렬이 $A$라는 결론이 아니다.

### 3.5 차원, 에스컬레이터와 자유매개변수

Hodge의 $d=3$ 결과는 3+1 시공간, 게이지 계층 또는 모든 상수의 값을
자동으로 산출하지 않는다. 공간 branch에서 시간 차원을 추가하는
에스컬레이터는 물리 사상이며, 자유매개변수 감사는 어느 관계가 정리,
외부 입력, 경험식인지 분리한다. 수치 관계의 수가 많아도 독립 parameter
count가 줄었다는 증명은 별도 Jacobian·RG·likelihood 분석을 요구한다.

### 3.6 강한 CP, 중성미자와 바리온 비대칭

현재 portal EFT는 QCD theta angle을 동적으로 제거하는 axion mechanism을
포함하지 않으므로 strong CP 해법을 제공하지 않는다. 또한 정확 $Z_2$
singlet만으로는 표준 Weinberg operator나 Majorana mass의 기원을 완결하지
않으며, neutrino mass에는 추가 field 또는 operator가 필요하다. baryon
asymmetry에는 baryon violation, CP violation, departure from equilibrium,
washout와 열이력이 필요하다. 이들은 가능한 확장 조건이지 CE 코어의
산출이 아니다.

### 3.7 페르미온 질량, 전이와 인과성

Yukawa 존재구성은 질량을 실현하지만 hierarchy를 선택하지 않는다.
게이지 격자 전이 ansatz는 local light-cone, signal speed, 초기값 문제와
gauge constraint를 만족해야 한다. 고정점 control parameter의 전이구간은
수학적 bifurcation 또는 crossover와 물리 phase transition을 구별해야
하며, 임계값을 관측 fitting 뒤에 정하면 예측이 아니다.

### 3.8 위상공간과 자기재귀 대칭

상태공간·게이지궤도·경로공간의 위상은 measure, compactness, boundary,
gauge quotient를 지정할 때만 수학적 대상이 된다. 자기·타공간 재귀의
대칭은 지정한 연산자와 불변부분공간에서 검증할 수 있는 조건부 구조다.
이를 우주의 실제 복제, 인과적 wormhole 또는 무한 에너지 채널로 읽는
부모 주장은 활성 정본에 없으며, 물리 구현은 미완성이다.

## 4. 공변 EFT와 유한 Euclidean 경로적분

Poisson 수학을 물리장과 혼동하지 않기 위해, CE는 독립 스칼라장을 가진
명시적 EFT branch를 따로 둔다. 시공간 좌표의 함수인 실수장 $\phi(x)$,
장배치의 이차 변분 연산자 $\mathcal H=\delta^2S$, 그리고 Ricci scalar
$R(x)$는 서로 다른 대상이다. 특히 Hessian을 물질장이나 곡률로
동일시하지 않는다. 이 구분은 gauge theory에서 gauge fixing, ghost,
measure와 재규격화가 따로 필요하다는 사실과도 연결된다.

**[공리: EFT branch]** CE가 보존하는 작용은 다음 $Z_2$ singlet-portal
모형이다.

$$
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\Lambda_0
-\frac12(\nabla\phi)^2+\mathcal L_{\rm SM}^{\rm kin+gauge+Yuk}
-V(H,\phi)\right],
$$

$$
V(H,\phi)=V_H(H)+\frac12m_\phi^2\phi^2
+\frac{\lambda_\phi}{4}\phi^4
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H.
$$

여기서 $H$는 표준모형 Higgs doublet, $\xi$와 $\lambda$들은 무차원
결합, $m_\phi$는 질량 매개변수다. $\phi\mapsto-\phi$ 대칭과
$M_{\rm Pl}^2-\xi\phi^2>0$, $\lambda_H>0$, $\lambda_\phi>0$,
$\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}$를 적용 정의역으로
둔다. 마지막 조건은 큰 장값에서 quartic potential이 아래로 유계가
되게 하는 **[정리]**의 전제다.

작용을 변분하면 $\phi$의 운동방정식과 총 stress의 on-shell 보존을
얻는다. $H^\dagger H=v^2/2$, $R=\bar R$, $\langle\phi\rangle=0$인
$Z_2$ 보존 branch에서

$$
m_{\phi,\rm eff}^2=m_\phi^2+\xi\bar R+
\frac{\lambda_{H\phi}}2v^2.
$$

$m_{\phi,\rm eff}^2>0$이면 그 진공은 국소 안정하고 선형 Higgs–$\phi$
혼합은 없다. 가장 가벼운 odd 양자가 안정하다는 결과는 가능한
암흑물질 EFT 구조를 제공하지만, relic density나 특정 GeV 질량을
산출하지는 않는다. 가능한 입자와 실제 우주 abundance는 다른 문제다.

Lorentzian 경로적분은 진폭의 진동 적분이고, Euclidean 적분은 특정
조건 아래 양의 측도를 준다. scalar–Higgs bosonic truncation, 고정
Euclidean 배경, 유한 격자와 coercive potential을 **[공리]**로 택하면

$$
Z_N=\int_{\mathbb R^N}e^{-S_{E,N}(z)}\,d^Nz
$$

가 유한하고 다항식 모멘트가 존재한다는 **[정리]**가 성립한다. 이는
유한 cutoff의 정확한 결과다. continuum limit, reflection positivity,
renormalized parameter의 cutoff 독립성 및 Lorentzian 재구성은 여전히
**[미완성]**이다. 바닥이 단단한 유한 차원 그릇에서는 물이 잘 담긴다는
비유는 유한 측도의 직관을 주지만, continuum QFT 전체를 담는다고
말하지 않는다.

고립된 비퇴화 Euclidean 최소점에서는 Laplace 근사가 작용차뿐 아니라
Hessian determinant와 measure prefactor의 비를 포함한다. 따라서
$e^{-1}$은 $\Delta S_E/\hbar=1$과 prefactor 비 1을 함께 택한 특수
산출일 뿐 보편 상수가 아니다. 영모드, 음의 모드, gauge orbit가 있으면
집단좌표·gauge fixing·적분 contour가 추가로 필요하다.

### 4.1 Z2 portal 작용과 차원

portal branch의 독립장 $\phi$와 Hessian, Ricci scalar를 구분한 뒤 작용을
택한다. 4차원에서 $[\phi]=1$, $[R]=2$, $[\lambda_\phi]=[\lambda_{H\phi}]
=[\xi]=0$이므로 각 Lagrangian 항의 차원은 4다. 이 차원 계산은 지수나
확률 인자에 차원 있는 양을 넣지 못하게 하는 경계다.

### 4.2 Quartic 안정성

큰 장값에서는 quartic 부분만 안정성을 정한다. $\lambda_H>0$,
$\lambda_\phi>0$, $\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}$이면
두 변수의 quartic form은 아래로 유계다. 이는 완전제곱 또는 양의
이차형식 판정으로 얻는 **[정리]**이며, loop-level potential과 cutoff
독립성까지 뜻하지 않는다.

### 4.3 운동방정식과 stress 보존

작용을 $\phi$로 변분하면

$$
\Box\phi-(m_\phi^2+\xi R+\lambda_{H\phi}H^\dagger H)\phi
-\lambda_\phi\phi^3=0.
$$

모든 장의 on-shell 방정식과 미분동형사상 불변성을 함께 쓰면
$\nabla^\mu T_{\mu\nu}^{\rm total}=0$이다. portal이 있으면 개별
sector stress는 교환항 때문에 따로 보존되지 않을 수 있다.

### 4.4 유한 Euclidean cutoff 측도

고정 Euclidean 배경, 유한 격자, 양의 kinetic operator와 coercive
potential을 택하면 $S_{E,N}$은 유한차원 연속 함수다. 따라서

$$
Z_N=\int_{\mathbb R^N}e^{-S_{E,N}(z)}\,d^Nz<\infty
$$

이고 다항식 모멘트도 존재한다. 이는 finite cutoff의 정리이며 continuum
limit, reflection positivity, Lorentzian reconstruction은 포함하지 않는다.

### 4.5 Laplace saddle

고립된 비퇴화 최소점 $z_j$와 양의 Hessian $\mathcal H_j$에서 국소
Laplace 전개는

$$
Z_j\simeq \mathcal N_j e^{-S_E(z_j)/\hbar}
\left(\det\frac{\mathcal H_j}{2\pi\hbar}\right)^{-1/2}.
$$

따라서 작용차 외에 determinant와 measure prefactor가 필요하다. 영모드,
음의 모드, gauge orbit는 이 식의 전제를 깨므로 별도 처리가 필요하다.

## 5. 끼임에서 양의 확률과 접힘으로 가는 bridge

EFT와 Euclidean 측도가 각각 일관되더라도, 그것만으로 양자 선택이
Poisson offspring을 만든다는 결론은 나오지 않는다. CE의 핵심 병목은
다음 화살표다.

$$
\text{CE+SM amplitude}
\longrightarrow\text{reduced dynamics}
\longrightarrow\text{positive jump process}
\longrightarrow A_{ij}\geq0.
$$

여기서 $A_{ij}$는 type $i$ 사건 하나가 type $j$ 자손을 만드는 평균
수다. 이 화살표를 닫으려면 환경 상태와 상호작용을 지정하고, 완전양성
reduced dynamics, 양의 Kossakowski 행렬, population sector의 폐쇄,
Markov·독립 증가량 근사, 그리고 genealogy의 식별을 함께 제시해야 한다.
현재 이 묶음은 **[미완성]**이다. 부호 있는 Hessian 또는 질량행렬
성분을 제곱해 $A_{ij}$라고 부르는 지름길은 허용되지 않는다.

접힘을 형식화하는 별도 도구는 등호이전(pre-equality) 수학이다. 유한
후보 $x$에 무차원 cost $E(x)$를 주고 $\beta$를 선택 scale로 두면
Gibbs 재가중은 후보 질량을 낮은 cost 쪽으로 농축한다. compact·noncompact
공간, Gamma 수렴, 조건 자체가 후보가 되는 joint manifest, Markov/Kleisli
커널과 tropical 극한까지가 조건부 **[정리]**로 전개되어 있다. 이 층에서
‘manifest’는 한 후보 또는 조건–값 쌍으로 수렴하는 수학적 현상이지,
그 자체가 물리적 측정 결과라는 선언은 아니다.

경로공간 bridge에서는 비선택 경로집합 위의 잔류 측도와 커널
$K_\phi(x,\gamma)$를 이용해 pushforward 장 후보를 정의할 수 있다.

$$
\phi_\beta(x)=\int_{\Gamma_{\rm ns}}
K_\phi(x,\gamma)\,\nu_{{\rm ns},\beta}(d\gamma).
$$

이 식은 수학적 정의다. $K_\phi$의 물리적 식별, 잔류장의 국소성·공변성,
good-rate 작용과 physical prior, 그리고 잔류 측도와 실제 접힌 에너지의
동일시는 **[미완성]**이다. 종이접기 비유가 여기서 멈춰야 하는 이유도
같다. 접힌 면을 그릴 수 있어도, 어떤 힘이 그 면을 장으로 만드는지는
아직 작용 수준에서 쓰이지 않았다.

### 5.1 Reduced dynamics와 GKSL/Kossakowski

계와 환경의 결합 상태에서 환경을 부분대각합해 얻는 reduced dynamics가
완전양성 trace-preserving map인지 먼저 확인해야 한다. Markovian limit의
GKSL generator에는 Kossakowski 행렬의 양의 준정부호성이 필요하다.
decoherence라는 말만으로 이 조건이나 CE+SM 작용에서의 유도는 얻지 못한다.

### 5.2 Population closure, Markov와 genealogy

양자 밀도행렬의 coherence를 버려 population sector가 닫혀야 양의 jump
rate를 쓸 수 있다. coarse-graining time, 독립 증가량, Markov 근사와
offspring genealogy를 고정한 뒤에만 $A_{ij}\geq0$를 다형 Poisson 평균으로
식별할 수 있다. 이 단계들은 현재 함께 **[미완성]**이다.

### 5.3 비선택 잔류측도와 phi pushforward

선택되지 않은 경로집합 $\Gamma_{\rm ns}$의 raw mass와 conditional shape를
구분한 뒤 kernel $K_\phi$를 택하면

$$
\phi_\beta(x)=\int_{\Gamma_{\rm ns}}K_\phi(x,\gamma)
\nu_{{\rm ns},\beta}(d\gamma)
$$

를 정의할 수 있다. 이는 수학적 pushforward이지 잔류장이 국소·공변
물질장이라는 증명은 아니다.

### 5.4 Euclidean 보조축

양의 자기수반 $A$에 대해

$$
\Phi(\chi)=e^{-\chi A}\Phi(0),\qquad \chi\geq0
$$

는 수축 semigroup이며 고유모드를 $e^{-\chi\omega}$로 억제한다.
$\chi A$는 무차원이어야 한다. 이를 물리적 두 번째 시간이나 접힘 방향으로
읽는 것은 별도 공리이며 인과성·unitarity·초기값 문제가 남는다.

### 5.5 열린 bridge 의무

bridge를 닫으려면 완결된 CE+SM 작용과 진공, 환경 state와 coupling,
CP reduced dynamics, positive Kossakowski matrix, population closure,
Markov/genealogy, physical kernel, species current와 total stress를
공동으로 제시해야 한다. 어느 하나를 수치 근접이나 비유로 대신할 수 없다.

## 6. 확률 코어, 혼합 통계량과 수치 체인

열린 bridge를 분명히 둔 상태에서, 일단 비음수 $A_{ij}$가 주어지면
다형 Poisson 코어는 정확히 풀린다. 모든 행합이 $D$인 경우 균일 벡터
$\boldsymbol1$에 대해 $\boldsymbol q=q_{\rm ext}\boldsymbol1$로
축약되고, 3장의 Lambert $W$ 해가 그대로 적용된다. 일반 다형 모형에서는
최소 고정점 벡터를 쓰며 Perron 임계성과 비균일 전달을 별도로 다룬다.
이것은 확률모형의 정리이지 임의의 QFT가 그 모형이라는 명제가 아니다.

CE의 균일 toy sector는 $B\geq0$, $B\boldsymbol1=\boldsymbol1$인
row-stochastic 전달 연산자를 택해

$$
A=dI+\delta B,\qquad
A\boldsymbol1=(d+\delta)\boldsymbol1,\qquad
D_{\rm eff}=d+\delta
$$

로 정의한다. $I$는 항등행렬, $d$는 Hodge branch에서 읽은 공간 재귀
깊이, $\delta$는 다음 전자약 readout이다. $D_{\rm eff}$가 균일
부분공간에서 정확히 닫힌다는 것은 **[산출]**이다. $\delta\to A$가
자연의 실제 전달법칙이라는 것은 **[공리]**다.

one-Higgs-doublet 표준모형의 중성 질량행렬에서 혼합각
$\tan\theta_W=g'/g$를 정의하면, 지정한 $W^3/B$ 기저의 정규화된
비대각 진폭 제곱은

$$
\delta=\sin^2\theta_W\cos^2\theta_W,\qquad 0\leq\delta\leq\frac14.
$$

범위 부등식은 $s(1-s)$의 초등적 최대값으로 얻는 **[정리]**다.
$\delta$를 끼임의 세기 또는 재귀 증분으로 읽는 것은 CE readout
**[공리]**다. $\sin^2\theta_W=4\alpha_s^{4/3}$은 동일 scheme·scale에서
시험하는 **[경험식]**이며, $\alpha_s$는 외부 입력이다. 따라서 이 수치
체인은 계산 가능한 benchmark이지만 우주론의 제1원리 유도가 아니다.

### 6.1 다형 Poisson과 균일 축약

type $i$가 type $j$ 자손을 독립 Poisson으로 만들고 평균행렬이
$A_{ij}\geq0$일 때 최소 소멸확률은

$$
q_i=\exp\!\left[-\sum_jA_{ij}(1-q_j)\right]
$$

의 최소해다. 공통 행합 $D$이면 균일 벡터가 불변이어서 scalar fixed
point로 축약된다. 이는 비음수·독립 offspring의 **[정리]**다.

### 6.2 전자약 혼합과 유효 깊이

지정한 $W^3/B$ 기저에서

$$
\delta=\sin^2\theta_W\cos^2\theta_W,\qquad
0\leq\delta\leq\frac14
$$

이고, $A=dI+\delta B$, $B\boldsymbol1=\boldsymbol1$을 **[공리]**로
택하면 $D_{\rm eff}=d+\delta$가 **[산출]**된다. 혼합 통계량을
offspring 증분으로 읽는 일은 정리가 아니다.

### 6.3 외부 입력과 고정점 산출

$\alpha_s(M_Z)$는 외부 benchmark이고
$\sin^2\theta_W=4\alpha_s^{4/3}$은 **[경험식]**이다. 같은 scheme과
scale에서 이 입력을 대입하면 $\delta$, $D_{\rm eff}$, 그리고

$$
q_{\rm ext}=-\frac1{D_{\rm eff}}
W_0(-D_{\rm eff}e^{-D_{\rm eff}})
$$

를 계산할 수 있다. 작은 residual은 이 계산을 확인할 뿐 경험식을
물리 유도로 승격하지 않는다.

## 7. flavor, 질량과 입자 현상론의 범위

확률 코어 뒤에 입자 수치를 나열하기보다, 먼저 무엇이 수학적으로
실현 가능한지 분리해야 한다. 임의의 양의 quark 질량과 unitary CKM
행렬이 주어지면 그 자료를 재현하는 Dirac Yukawa 행렬을 구성할 수 있다.
마찬가지로 양의 중성미자 질량과 unitary PMNS 행렬이 주어지면 복소
대칭 Majorana 질량행렬을 구성할 수 있다. 이는 존재구성의 **[정리]**다.
그러나 그 구성은 입력 각도와 질량을 줄이지 않으므로, 왜 자연이 그 값을
선택하는지 예측하지 않는다.

Koide 관계는 세 양의 질량 $m_i$에서 $x_i=\sqrt{m_i}$를 만들고 민주축
$u=(1,1,1)/\sqrt3$를 잡을 때 기하학적으로 읽힌다.

$$
\frac{\sum_i m_i}{(\sum_i\sqrt{m_i})^2}=\frac23
\quad\Longleftrightarrow\quad
\angle(x,u)=45^\circ.
$$

이 동치는 **[정리]**다. 이를 실제 flavor potential의 정지조건으로
만들 symmetry, vacuum alignment, 다른 sector로의 확장과 RG 안정성은
**[미완성]**이다. 같은 이유로 CKM·PMNS·질량의 수치 관계는 경험식 또는
입력 자료로 관리하며, 관측 근접을 mass-generation 동역학의 증명으로
바꾸지 않는다.

저에너지 scalar 현상론도 이 구분을 따른다. $-g_\ell\phi\bar\ell\ell$
같은 Yukawa를 별도 EFT branch로 택하면 렙톤 자기모멘트의 one-loop
적분은 계산할 수 있다. 그러나 정확한 $Z_2$ portal branch와는 추가
$Z_2$ 깨짐 또는 gauge-invariant UV completion 없이는 동시에 놓이지
않는다. proton 산란의 scalar exchange 역시 가능한 조건부 효과일 뿐
고유 전자기 form factor나 반경 편이를 자동으로 정하지 않는다.

### 7.1 CKM/Yukawa 존재구성

양의 quark 질량과 unitary $V_{\rm CKM}$이 주어지면
$D_u=\sqrt2\,\operatorname{diag}(m_u,m_c,m_t)/v$,
$D_d=\sqrt2\,\operatorname{diag}(m_d,m_s,m_b)/v$를 두고
$Y_u=D_u$, $Y_d=V_{\rm CKM}D_d$로 구성할 수 있다. 이는 주어진 입력을
실현하는 **[정리]**이며 질량·각도를 예측하지 않는다.

### 7.2 PMNS/Majorana

양의 중성미자 질량행렬 $D_\nu$와 unitary $U_{\rm PMNS}$가 주어지면

$$
M_\nu=U_{\rm PMNS}^*D_\nu U_{\rm PMNS}^\dagger
$$

는 복소 대칭 Majorana 질량행렬이다. 존재구성은 Weinberg operator 또는
UV completion의 선택 이유를 주지 않으며, flavor symmetry는 **[미완성]**이다.

### 7.3 Koide 기하와 potential 공백

$x_i=\sqrt{m_i}$와 $u=(1,1,1)/\sqrt3$에 대해

$$
\frac{\sum_i m_i}{(\sum_i\sqrt{m_i})^2}=\frac23
\Longleftrightarrow \angle(x,u)=45^\circ
$$

는 **[정리]**다. 실제 질량이 이 cone을 고르는 potential, vacuum alignment,
RG 안정성은 따로 제시되지 않았으므로 **[미완성]**이다.

### 7.4 Scalar 저에너지 현상론

$-g_\ell\phi\bar\ell\ell$ 같은 Yukawa를 별도 EFT branch로 택하면
one-loop $g-2$ 적분을 계산할 수 있다. 이는 exact $Z_2$ portal과 자동
호환되지 않으며 $Z_2$ breaking 또는 UV completion을 요구한다. proton
산란도 $g_\ell g_p$와 form factor를 입력해야 하며 CE 코어는 이 계수를
정하지 않는다.

## 8. 우주론 branch와 암흑 표현의 한계

확률 코어에서 얻은 $q_{\rm ext}$를 우주에 쓰려면 먼저 우주론 방정식의
독립적 branch를 분리해야 한다. 평탄 FLRW에서 최소 결합 canonical scalar
$\phi(t)$는

$$
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,\qquad
\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0.
$$

여기서 $H=\dot a/a$는 Hubble rate, $a(t)$는 scale factor다. $\rho_\phi>0$
이면

$$
w_\phi+1=\frac{\dot\phi^2}{\rho_\phi}\geq0.
$$

따라서 positive-kinetic canonical 단일장에는 $w<-1$ crossing이 없다는
**[정리]**가 성립한다. 상수 potential $V_0$는 $T_{\mu\nu}=-V_0g_{\mu\nu}$,
$w=-1$을 준다. 이 결과는 진공항의 방정식상태를 고정하지만, 작은 절대값과
radiative stability를 설명하지 않는다.

별도 inflation branch에서는 Einstein frame, 정준 단일장, positive
$V_0$, scalar dominance, adiabatic Bunch–Davies 초기상태와 leading
slow-roll을 택해 Starobinsky형 potential을 쓴다. 이 전제 아래
$n_s=1-2/N+\cdots$, $r=12/N^2+\cdots$가 **[산출]**된다. $N$은 reheating에
의존하고 amplitude normalization은 $V_0$를 고정하므로, 이는 포탈
EFT나 다섯 상수에서 나온 보편 예측이 아니다.

암흑 표현의 역사적 경계모형 `C-B-LEGACY-01`은

$$
q_{\rm ext}\longmapsto\Omega_b,
\qquad
\Omega_b=\frac{\rho_b(t_0)}{\rho_{\rm crit}(t_0)}
$$

를 **[공리]**로 둔다. radiation과 curvature를 무시한 평탄 후기 우주에서
$\Omega_b+\Omega_{\rm DM}+\Omega_\Lambda=1$을 함께 택하고, 남은 분율의
내부 비를 별도 공리로 두면 수치가 산출된다. 그러나 species current,
유일한 전이 면, 총 stress 보존, baryon current, 절대 yield와 섭동은
아직 닫히지 않았다. 따라서 관측과의 잔차가 작아도 이 화살표는
**[미완성]** bridge를 포함한다.

### 8.1 FLRW와 canonical no-go

FLRW에서 $H=\dot a/a$이며

$$
H^2+\frac{k}{a^2}=\frac{\rho}{3M_{\rm Pl}^2},
\qquad \dot\rho+3H(\rho+p)=0.
$$

canonical scalar는 $\rho_\phi=\dot\phi^2/2+V$,
$p_\phi=\dot\phi^2/2-V$이므로 $\rho_\phi>0$에서
$w_\phi+1=\dot\phi^2/\rho_\phi\geq0$다. 이는 positive kinetic branch의
정리이며 crossing에는 추가 자유도 또는 modified gravity가 필요하다.

### 8.2 상수 진공과 dust/Lambda

상수항의 변분은 $T_{\mu\nu}^{(0)}=-V_0g_{\mu\nu}$, $w=-1$을 준다.
빠른 quadratic scalar의 dust 평균과 flat dust+Lambda 나이 적분은
각각 adiabatic 및 GR 전제의 조건부 결과다. $V_0$의 절대값과
radiative stability는 미완성이다.

### 8.3 Inflation slow-roll

Einstein frame 정준 단일장, $V_0>0$, scalar dominance, Bunch--Davies
초기상태와 leading slow-roll에서

$$
V(\phi)=V_0(1-e^{-\sqrt{2/3}\phi/M_{\rm Pl}})^2
$$

이면 $n_s=1-2/N+O(\log N/N^2)$, $r=12/N^2+O(\log N/N^3)$다.
$N$은 reheating에, 진폭은 $V_0$에 의존하므로 무입력 예측이 아니다.

### 8.4 q_ext에서 Omega_b로

$q_{\rm ext}\mapsto\Omega_b$는 C-B-LEGACY-01의 경계 공리다. flat
late-time closure와 DM/Lambda 내부비를 추가로 택할 때만 분할 수치가
산출되며, 확률 동일시는 분지 정리의 결론이 아니다.

### 8.5 관측 비교 규율

비교표에는 모델, 기준값, 불확도, 잔차와 snapshot을 함께 둔다. 같은
자료로 맞춘 scale을 독립 예측으로 세지 않으며 관측 근접은 경험식이나
공리를 정리로 승격하지 않는다.

### 8.6 Transient physical bridge

물리적 readout에는 species current, 전이면, total stress 보존, 전이 뒤
baryon current, absolute yield와 섭동의 공동 유도가 필요하다. 이 의무가
닫히지 않았으므로 transient bridge 전체는 미완성이다.

### 8.7 자기비동일성 시계장과 R1 열린계

암흑 표현을 더 밀어붙이려면 ‘선택되지 않은 것이 에너지인가’라는 질문을 먼저 장부의 질문으로 바꿔야 한다. 본 논문에서 제안하는 R1 branch는 기회비용을 $\theta=\Gamma T$의 무차원 누적량 $1-e^{-\theta}$로만 읽는다. 이 양은 자체로 중력을 만들지 않으며, 에너지밀도는 독립 척도 $\rho_\infty$를 넣은 시계장 작용에서만 생긴다. 따라서 한 장의 $P(T,X)$ 작용은 정확히 진공형 $\rho_V$와 작은 $\delta=X/X_*-1>0$에서 물질형인 $\rho_K$를 분해하지만, 영 전류에서는 즉시 $J<0$, $\delta<0$, $c_s^2<0$로 가는 완전 반례를 갖는다.

살아남는 조건부 해는 $\langle T_i\rangle=0$과 $\langle J_i\rangle=\Pi_{\rm F}>0$를 Gaussian 초기상태의 평균으로 준비한다. 이는 0차원 점이 새 시공간 차원이라는 주장이 아니다. 0차원은 측정 사건과 기록의 타입이며, 균일한 FLRW 초곡면으로 보내는 공간 분포ㆍ거친갈기 사상은 별도 미완성이다. 이웃 부트스트랩과 자기측정은 저장소ㆍ기록의 후보 기제일 뿐, 초기 전류의 에너지 공급을 증명하지 않는다. 변화의 반복은 $X$로 읽지만, 미래 화살은 retarded 응답과 초기 상태라는 별도 조건이다.

고전 경계항은 이 준비를 대체하지 못한다. $B=-\Pi_{\rm F}T$는 자유 변분에서 전류를 주지만, 정확한 $T_i=0$에서는 허용 변분이 사라진다. 또한 선형 $gT\phi$ 저장소는 완성제곱 뒤 $-g^2T^2/(2m^2)$를 남겨 하방으로 유계가 아니다. R1은 대신 유계 source $s_A(T)=\mu_A^3F_A(\Gamma T)$를 가진 Gaussian 저장소를 채택한다. 변분하면 시계장ㆍ저장소 방정식이 함께 나오고, 저장소를 적분하면 retarded kernel $D_R$과 양의 noise kernel $N$을 가진 Schwinger--Keldysh 영향함수가 나온다. 상호작용 응력까지 포함한 총 Ward 장부만 보존되며, 축약된 시계장만으로는 보존되지 않는다.

HuㆍPazㆍZhang(1992)은 Gaussian 환경이 비국소 소산과 colored noise를 함께 만든다는 표준 예를 제공한다. JanaㆍLoganayagamㆍRangamani(2020)은 실시간 영향함수를 Schwinger--Keldysh 상관함수로 조직하는 틀을, CrossleyㆍGloriosoㆍLiu(2017)와 GloriosoㆍCrossleyㆍLiu(2017)는 소산 유효장이론의 두 경로ㆍ대칭ㆍ상태 조건을 제공한다. 이는 R1의 형식 도구를 지지할 뿐 CE의 비선택 경로가 실제 저장소라는 동일성이나 $\Pi_{\rm F}$의 값을 증명하지 않는다.

유한 Gaussian artifact는 비인과 성분 0, noise 최소 고윳값 $-8.03\times10^{-16}\simeq0$, Robertson 여유 $0.22$, 에너지 드리프트 $4.73\times10^{-14}$, 교환식 잔차 $1.19\times10^{-17}$을 확인했다. 이는 한정된 표본의 구현 검증일 뿐 R1의 관측 예측이 아니다. 사전 `doctor`가 삭제된 모듈 때문에 실패한 것은 작업트리 환경 문제이며 과학 결과와 분리한다.

R1의 다음 병목은 고정배경 장파장에서 남는 $m_{\rm eff}^2<0$과 $c_s^2\to0$이다. 첫 음수만으로 결합 우주론의 실패를 선언하지는 않지만, metric mixing, bath self-energy, $k^4$ 완성과 full CMB/LSSㆍlensingㆍhalo 계산이 없으므로 안정성ㆍabundanceㆍ수치 예측도 선언하지 않는다. 상세 유도와 반례는 [암흑에너지 유도](5_유도/04_Dark_Energy_Derivation.md#4101-유계-gaussian-저장소의-작용과-변분)에 둔다.

### 8.8 R2 관측창: 물리 mode와 cutoff의 조건부 검사

R2는 R1의 음의 고정배경 질량을 그대로 결론으로 쓰지 않고, 먼저 그것이 어떤
변수의 부호인지 다시 묻는다. $\pi=\delta T$는 시간좌표 선택에 의존하므로,
unitary gauge에서는 사라진다. Einstein 중력과 시계장만 남긴 부분계에서
lapse와 shift 제약을 제거하면 물리 스칼라는 곡률섭동 $\zeta$이며,

$$
S^{(2)}_\zeta=\int a^3Q_s\left[\dot\zeta^2-c_s^2(\nabla\zeta)^2/a^2\right],
\qquad Q_s=\frac{XA}{H^2},\qquad c_s^2=\frac BA
$$

가 된다. 장파장 해 $\zeta=C_1+C_2\int dt/(a^3Q_s)$에는 독립 질량항이 없다.
동결 관측창에서 고정계량 진단의 최대 $|m_{\rm eff}^2|/H^2$는
$3.0455\times10^{-18}$, 보수적 로그 성장상계는 $2.4838\times10^{-17}$이었다.
단일-clock 작용에서는 $\min c_s^2=9.2138\times10^{-19}$,
$\min Q_s/M_{\rm Pl}^2=3.3167\times10^5$였고, pump 기울기 두 개도
$3.93909$, $3.469545$로 양수였다. 따라서 유한 관측창에서 두 번째 장파장
mode의 $\dot\zeta$와 적분함수는 감소한다. 이것은 선택한 단일-clock 부분계의
검사일 뿐, 임의 초기값ㆍ무한 미래ㆍ다유체 우주의 안정성 증명은 아니다.

같은 R2는 작은 음속에서 에너지 cutoff와 파수 cutoff를 분리했다. 정준장 전개의
$\Lambda_3$는 $79.7$--$80.6\,{\rm eV}$, $\Lambda_4$는
$94.8$--$95.8\,{\rm eV}$이고,
$\Lambda_E=\Lambda_3c_s^{7/4}$와 $q_{\rm sc}=\Lambda_3c_s^{3/4}$다.
관측창에서 각각 $1.3336\times10^{-14}\,{\rm eV}$,
$1.3893\times10^{-5}\,{\rm eV}$보다 작아지지 않았으며, Hubble 에너지와
$1\,{\rm Mpc}^{-1}$ 물리 파수보다 각각 $9.2757\times10^{18}$,
$2.1725\times10^{24}$배 위에 있다. $k^4$ crossover가 이 cutoff 전에 오기 위한
$\bar M$의 필요하한은 오늘 $0.225\,{\rm eV}$, 관측창 최악 $7.31\,{\rm eV}$다.
$\bar M\sim80\,{\rm eV}$ 후보는 필요조건을 충족할 수 있지만 그 사실은
higher-derivative completion이나 관측 예측을 만들지 않는다. 미래에는
two-derivative cutoff가 계속 내려가므로, 이 경로는 구체적 후보이며 남은 계산
과제다.

그러므로 R1에서 R2까지 얻은 것은 포화 readout을 가진 조건부 EFT와 관측창
부분계의 안정성ㆍcutoff 검사다. 비선택 양자경로가 실제 암흑부문이라는 동일성,
0차원 기록에서 $\Pi_{\rm F}$를 얻는 사상, 존재량의 분할, bath pole과 전체
coupled ADM, 정확한 $k^4$ degeneracy, Einstein--Boltzmann CMB/LSSㆍlensingㆍhalo는
모두 미완성이다. 이 단계를 ‘암흑에너지의 증명’ 또는 예측으로 승격하지 않는다.

## 9. 유도 응용: 방정식에서 기능형까지

코어 이론의 조건을 다른 분야로 옮길 때에는 공통 작용이 모든 방정식을
산출한다고 가정하지 않는다. 이 장은 각 분야에서 이미 닫힌 표준 결과를
먼저 쓰고, CE가 제안하는 기능형·readout·결합이 어느 지위에 머무는지를
차례로 분리한다.

### 9.1 Navier--Stokes: 에너지 항등식과 잔차 목적함수

유체 적용에서는 난류를 해결했다는 선언보다 경계조건이 에너지 계산에
어떻게 들어가는지가 먼저다. 차원 $d=2$ 또는 $3$의 periodic 영역 또는
매끄러운 bounded 영역에서, 속도 $\mathbf u$, 압력 $p$, 밀도 $\rho>0$,
동점성계수 $\nu>0$, 단위질량당 외력 $\mathbf f$를 둔다. periodic
branch에서는 평균을 0으로, bounded branch에서는 no-slip
$\mathbf u|_{\partial\Omega}=0$을 둔다.

**[정의]** 비압축성 방정식은

$$
\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u
=-\rho^{-1}\nabla p+\nu\Delta\mathbf u+\mathbf f,
\qquad \nabla\cdot\mathbf u=0.
$$

충분히 매끄러운 해에 $\mathbf u$를 내적하여 적분하면, 비선형항은
발산정리와 경계조건으로 0이 된다. 압력항도 divergence-free 조건으로
0이 되고, Laplacian 항은 부분적분으로 음의 gradient norm이 된다.
그러므로

$$
\frac12\frac{d}{dt}\|\mathbf u(t)\|_2^2
+\nu\|\nabla\mathbf u(t)\|_2^2=(\mathbf f,\mathbf u)
$$

가 **[정리]**로 따른다. 이 식은 점성이 운동에너지를 소산하고 외력이
공급한다는 정확한 장부다. 욕조의 물을 저어도 벽과 점성이 없으면 같은
장부가 성립하지 않는다는 점에서, ‘난류 억제’의 일반 비유에는 한계가 있다.

기준 길이 $L$과 속도 $U$로 무차원화하면 Reynolds 수
$\operatorname{Re}=UL/\nu$가 나온다. 이 변수에서 PDE residual과
divergence residual의 제곱을 적분한 $\mathcal J$를 수치 목적함수로
**[공리]**로 둘 수 있다. 경계·초기조건을 만족하는 허용 함수에서
$\mathcal J=0$이면 PDE가 a.e. 성립한다는 것은 **[정리]**다. 그러나
minimizer의 존재·유일성·격자 수렴과 3차원 global regularity는 이
동치에서 따라오지 않는다. CE stress를 넣으려면 별도의
$\tau_{\rm CE}$ 구성방정식, causal relaxation과 energy estimate가
필요하며, 현재는 **[미완성]**이다.

### 9.2 단백질 접힘: 경로 가중과 곡률 억제

유체의 잔차가 방정식 위반을 측정했다면, 단백질 절은 가능한 접힘 경로의
비교 규칙을 제안한다. 상태공간 $\mathcal X$의 점 $x$는 하나의 입체
구조이고, $x(t)$는 고정된 시간구간 $[0,T]$의 경로다. 자유에너지
$E(x)$는 외부 분자모형의 입력이며, 복잡도 $\mathcal C(x)$의 실제
정의는 아직 유일하지 않다.

**[공리: 모델 선택]** 무차원 상태좌표와
$[\beta]=\mathsf E^{-1}\mathsf T^{-1}$를 택해

$$
P_{\rm sel}[x]\propto
\exp\!\left[-\beta\int_0^T
\big(E(x(t))+\alpha_C\mathcal C(x(t))\big)\,dt\right]
$$

로 경로를 가중한다. $[\alpha_C]=\mathsf E\mathsf C^{-1}$이므로 지수는
무차원이다. 이는 열평형 Boltzmann factor의 유도가 아니라, 시간 적분된
경로 비용에 대한 역작용 가중의 선택이다.

매끄러운 경로와 국소 복잡도 근사를 함께 가정하면

$$
\mathcal C(x(t))\approx c_0+c_1\|\dot x(t)\|^2
+c_2\|\ddot x(t)\|^2.
$$

모든 후보가 같은 $T$를 공유할 때에만 $c_0T$는 비교에서 소거된다.
복잡도 유래 항은

$$
\mathcal S_{\rm fold}[\phi]=\int_0^T
\left(a_0\|\dot\phi\|^2+
\lambda_{\rm fold}\|\ddot\phi\|^2\right)\,dt,
\quad
a_0=\beta\alpha_Cc_1,\quad
\lambda_{\rm fold}=\beta\alpha_Cc_2
$$

가 된다. 구불구불한 등산로가 같은 고도차를 더 긴 시간에 오르는 비유는
이 항의 직관을 주지만, 실제 protein landscape의 곡률을 측정했다는
뜻은 아니다. RMSD 개선을 주장하려면 단백질 집합·baseline·자료 분리·seed·
불확도를 사전 고정해야 하며, 이 검증은 **[미완성]**이다.

### 9.3 암흑에너지: FLRW와 scalar branch의 경계

접힘의 우주론 readout을 논하려면 먼저 확률과 무관하게 성립하는 FLRW
배경을 고정해야 한다. scale factor $a(t)$와 Hubble rate $H=\dot a/a$,
total density $\rho$, pressure $p$에 대해 Einstein 방정식은

$$
H^2+\frac{k}{a^2}=\frac{\rho}{3M_{\rm Pl}^2},
\qquad
\dot\rho+3H(\rho+p)=0.
$$

따라서 가속 $\ddot a>0$에는 $\rho+3p<0$가 필요하다. 이는 source의
절대 scale을 정하지 않는 **[정리]**다.

최소 결합 canonical scalar를 **[공리]**로 택하면

$$
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,
\qquad
w_\phi+1=\frac{\dot\phi^2}{\rho_\phi}\geq0.
$$

마지막 부등식은 positive kinetic과 $\rho_\phi>0$ 아래의 **[정리]**다.
그러므로 이 branch는 phantom crossing을 만들지 못한다. 상수 potential은
$w=-1$을 주지만, 작은 진공에너지의 크기와 radiative stability는
설명하지 않는다. scalar--tensor branch로 넘어가려면 $F(\phi)>0$과
Einstein-frame kinetic positivity, matter coupling과 초기조건을 따로
고정해야 한다. 원하는 $H(a)$를 먼저 맞춘다고 하나의 공변 작용이
자동으로 생기지는 않는다.

### 9.4 Master action: 변분, 축약, 소산의 분리

여러 응용에서 쓰인 작용의 공통 부분은 변분 원리다. field를 $\Phi^A$,
coupling을 $\lambda_I$로 쓰고

$$
S[\Phi;\lambda]=\int_M d^4x\sqrt{-g}\,
L(\Phi^A,\nabla_\mu\Phi^A;\lambda_I)
$$

를 정의하면, variation은 bulk Euler--Lagrange 항과 boundary term으로
갈린다. boundary condition 또는 적절한 boundary action을 정한 뒤에만
$\delta S=0$에서 field equation이 따른다. 미분동형사상 불변 완결
작용에서는 Noether identity가 total stress의 on-shell 보존을 준다.

ansatz $\Phi^A=\iota^A(\psi)$를 대입한 reduced action의 stationary
point는 full equation의 tangent projection만 보장한다. 버린 방향의
equation도 0일 때에만 consistent truncation이다. 이는 지도에서 지하철
노선 하나만 보고 도시 전체 교통이 맞는다고 말할 수 없는 것과 같다.
Navier--Stokes나 GKSL처럼 소산적인 식에는 bath, influence functional,
Schwinger--Keldysh doubled field와 positivity/noise가 더 필요하다.
하나의 실수 scalar action이 모든 transport coefficient를 산출한다는
주장은 **[미완성]**이다.

### 9.5 Neural RealityStone: 곡률 기능형의 해석 한계

뇌와 LLM 절은 실제 생물학적 작용을 쓰는 것이 아니라, 상태 변화의
안정성을 비교하는 보조 기능형을 제안한다. 뇌 활성 $x$와 LLM hidden
state $z$가 각각 계량을 가진 상태공간에 놓인다고 두고, 과업 상태
$\phi$의 gradient와 Hessian류 변화를 지표로 읽는다. 실제 계량과 곡률을
측정하는 절차는 아직 정해지지 않았다.

**[공리: 모델 선택]** 각성 상태에는

$$
\mathcal S_{\rm awake}=\int\!\!\int_{V_{\rm task}}
\left(\|\nabla\phi\|^2+
\lambda_{\rm awake}\|\nabla^2\phi\|^2\right)\,dx\,dt
$$

를, NREM에는 2차 항 가중이 큰 유사 functional을 둔다. REM에는 탐색
noise 항을 더하는 해석을 택할 수 있다. 평평한 지형에서 공이 작은
교란에 덜 흔들린다는 비유는 가능하지만, 뇌의 수면 단계가 이 식을
최소화한다는 증거는 아니다.

ACC–오류감시, PFC–정책수정, 해마–장기기억, 수면–오프라인 평탄화의
대응은 설계 가설이다. 조현병·뇌전증·환각을 각각 특정 곡률 항의 고장으로
읽는 일도 임상 자료와 사전 고정 실험이 필요하다. 특히 LLM 환각률을
하나의 상수로 동일시하지 않으며, runtime benchmark의 결과는 생물학
기전의 검증으로 전용하지 않는다.

### 9.6 상수장 블랙홀: 보존되는 Einstein branch

마지막 응용은 scalar--tensor 작용에서 상수장 $\phi=\phi_0$가 실제 해가
되는 조건을 확인한다. $F_0=F(\phi_0)>0$, $V_0=V(\phi_0)$일 때 scalar
방정식은 pointwise로

$$
\frac{M_{\rm Pl}^2}{2}F'_0R(x)-V'_0=0
$$

을 만족해야 한다. 이 조건과 metric 방정식을 함께 쓰면

$$
G_{\mu\nu}+\Lambda_{\rm eff}g_{\mu\nu}
=8\pi G_{\rm eff}T^{(m)}_{\mu\nu},
\quad
G_{\rm eff}=\frac{1}{8\pi M_{\rm Pl}^2F_0},
\quad
\Lambda_{\rm eff}=\frac{V_0}{M_{\rm Pl}^2F_0}
$$

로 정확히 축약된다. metric 방정식만 맞추고 scalar equation을 생략하면
이 결론은 얻을 수 없다.

점근 평탄 진공, 정적 구면대칭과 연결된 외부영역을 더 가정하면
$f(r)=1-2G_{\rm eff}M/r$인 Schwarzschild branch가 나온다. horizon
$r_h=2G_{\rm eff}M$, 온도 $T_H=(8\pi G_{\rm eff}M)^{-1}$, Wald entropy
$S_{\rm W}=A_H/(4G_{\rm eff})$는 이 branch의 조건부 산출이다. 상수장을
일정 비율로 바꾸면 shadow가 바뀐다는 식의 비유는 물리적으로 고정된
$G_{\rm eff}$와의 비교를 놓친다. 비상수 hair, full perturbation,
greybody spectrum, backreaction과 양자중력 종말은 **[미완성]**이다.

## 10. 등호이전 수학과 측정 문제

응용 기능형의 수학적 기반은 등호이전 문서군에서 더 일반적으로 전개된다.
유한 후보공간에서 Gibbs 재가중은 유일 최소 후보로 농축하고, 연속·비콤팩트
공간에서는 coercivity, recovery mass와 support 조건이 그 역할을 한다.
조건 자체를 후보로 올린 joint manifest, 조건 주변화와 Bayes readout,
`PreEq_fin`의 비음수 커널 범주, tropical/min-plus 극한은 각각 정의역을
명시한 수학 결과다. 이 층은 ‘답이 나타나기 전 후보들이 어떻게 정렬되는가’를
다루며, CE 물리의 대체 설명이 아니다.

양자 측정과 Born rule은 특히 엄격히 분리한다. finite branch refinement
아래 특정 prior가 Born 가중을 준다는 조건부 정리는 있으나, 실제 양자
상태·측정장치·환경에서 그 prior가 왜 선택되는지는 추가 물리 사상이다.
따라서 `Born prior`, 잔류 경로, AGI residual channel은 정리·공리·실험
프로토콜의 서로 다른 지위를 유지한다. 이 층의 전체 정의와 no-go는
[등호이전 수학](9_등호이전/README.md)에 있다.

### 10.1 유한 후보와 Gibbs 농축

앞 절의 경로 후보를 더 단순한 유한 집합으로 제한하면 선택의 수학적
내용을 투명하게 볼 수 있다. 후보집합 $X$와 무차원 cost $E:X\to\mathbb R$,
초기확률 $\mu_0$를 고정하면 Gibbs 재가중은

$$
\mu_\beta(x)=\frac{e^{-\beta E(x)}\mu_0(x)}
{\sum_{y\in X}e^{-\beta E(y)}\mu_0(y)}.
$$

유일 최소점이 초기분포에서 양의 질량을 가지면 $\beta\to\infty$에서
$\mu_\beta$는 그 최소점의 Dirac 측도로 수렴한다. 이는 **[정리]**다.
동점 최소점 안에서 어떤 후보를 고를지는 이 정리만으로 정해지지 않는다.

### 10.2 연속, 비콤팩트, Gamma 극한

유한합을 적분으로 바꾸면 minimizer로 도망가지 않게 하는 조건이 필요하다.
compact 공간에서는 연속성과 양의 기준측도가, 비콤팩트 공간에서는
coercivity와 recovery mass가 농축의 전제가 된다. 에너지 자체가 scale에
따라 변할 때에는 국소 균등수렴 또는 Gamma 수렴이 최소값의 안정성을
통제하지만, 최소집합 내부 선택까지 자동으로 주지는 않는다.

### 10.3 조건, Bayes와 범주 도구

값뿐 아니라 조건 $c$도 후보라면 joint weight를 먼저 만들고 주변화해야
한다. 조건부 readout은 joint posterior의 비율이며, 사후 선택을 이미
선택된 값의 원인으로 되돌려 쓰지 않는다. PreEq_fin의 비음수 커널,
Markov/Kleisli 위치와 tropical 극한은 이 합성을 엄밀히 기록하는
**[정의]**과 **[정리]**다. 물리적 측정 사상은 이 범주 결과에 추가로
붙여야 한다.

### 10.4 phi pushforward와 good-rate 경로

유한 후보의 농축을 CE 경로공간에 적용하려면 $W=S_E/\hbar+S_{\rm supp}$
가 good rate function이고, prior가 필요한 support를 실제로 가져야 한다.
이 전제에서 비선택 subprobability와 조건부 shape를 분리하고 커널
$K_\phi$로 pushforward를 정의할 수 있다. 어느 prior와 kernel가 자연의
것인지, Brownian 경로와 Sobolev 경로 중 무엇을 택할지는 **[미완성]**이다.

### 10.5 Born bridge의 정확한 경계

측정 후보를 refinement할 때 branch count prior가 진폭 제곱 prior로
정렬되는 조건부 finite-branch 결과가 있다. 그러나 실제 Born rule은
Hilbert 공간, 측정 interaction, decoherence와 apparatus를 포함하는
물리 이론의 문제다. 따라서 finite refinement 정리를 Born rule의
무조건 유도로 부르지 않는다.

## 11. 뇌, 생명, AGI의 검증형 확장

물리 코어의 선택·잔류·readout 어휘는 뇌와 생명에서 검증 질문을
조직할 뿐 생물학적 동형성을 뜻하지 않는다. 아래 절은 관측 사실, toy
모형, 구현과 실험 판정을 분리한다.

### 11.1 뇌 해부와 관측 정의

뇌 이론은 먼저 무엇을 관측량으로 부를지 고정한다. 세포형·층·영역·
투사는 구조 자료이고, spike·calcium·LFP·행동은 서로 다른 시간해상도의
관측이다. 영역 $i$의 시계열을 $x_i(t)$, 과업 입력을 $u(t)$, 행동
readout을 $y(t)$로 쓰는 것은 **[정의]**이며, 이것이 완전한 신경 상태
좌표계라는 뜻은 아니다.

ACC의 오류 관련 활성과 PFC의 규칙 수정은 문헌 기반 기준 사실이지만,
각 영역이 하나의 CE scalar를 계산한다는 결론은 없다. recording modality
정렬, session·animal 변이, 결측과 intervention을 명시하지 않으면
상관은 회로 인과성을 주지 않는다. 해부 지도만으로 동역학을 추론하는
일은 도로 지도만으로 교통량을 예측하는 것과 같으며, 그 bridge는
**[미완성]**이다.

### 11.2 항상성, 그래프 결합과 이완

관측을 고정한 뒤에는 외란에서의 복구를 시험할 수 있다. 영역을 노드,
유효 연결을 $W_{ij}$로 두는 선형화는

$$
\dot x_i=-\lambda_i(x_i-\bar x_i)+\sum_jW_{ij}x_j+u_i(t)
$$

처럼 쓸 수 있는 **[공리: 모델 선택]**이다. $\bar x_i$는 기준 활성,
$\lambda_i>0$는 국소 이완률이다. 선형화 행렬의 고유값 실수부가
음수이면 그 근방에서 안정하다는 것은 **[정리]**이지만, 비선형 뇌 전체의
보편 안정성은 아니다.

feedback gain과 지연은 항상성의 핵심 조건이다. 온도조절기의 gain을
과도하게 높이면 overshoot가 생기는 비유는 유용하지만 endocrine·immune·
neural 축을 하나의 gain으로 환원하지 않는다. $W$와 $\lambda$는 공개
자료·개입·불확도 규약으로 추정해야 하며, fitted correlation을 directed
causal coupling으로 읽는 일은 **[미완성]**이다.

### 11.3 수면, 가소성, 복구

이완이 깨어 있는 상태의 짧은 복구를 다룬다면 수면은 더 긴 시간척도의
재조직을 다룬다. NREM 재생·동기화와 REM 내부생성은 외부 문헌의 기준
사실로 분리하며, CE functional의 직접 산출로 부르지 않는다.

가소성 toy update는

$$
\Delta w_{ij}=\eta F(t_i^{\rm post}-t_j^{\rm pre})
-\gamma(w_{ij}-w_0)
$$

로 쓸 수 있다. 첫 항은 pre/post timing 의존 변화, 둘째 항은 기준
$w_0$를 향한 항상성 완화의 예다. 이는 특정 세포형의 보편 법칙이
아니다. 수면·replay가 기억 또는 회복을 개선하는지는 stage, task,
intervention, 행동 readout을 사전 고정해 비교해야 한다.

offline replay와 생물학적 sleep은 이름이 비슷해도 동형이 아니다.
STDP 효능의 일부 일반 주장은 감사에서 지지되지 않아 비활성 기본값으로
남아 있다. 수면 기능형의 계수와 실제 circuit variable의 matching은
**[미완성]**이다.

### 11.4 원시생명: 존재, 결합, 연령구조와 경험 gate

원시생명 절은 생명을 한 식으로 정의하지 않고 재현·유전·선택을 가진
최소 모형에서 성립하는 명제를 분리한다. type $i$가 type $j$ 자손을
평균 $A_{ij}$개 만드는 다형 branching process는 수학 대상이고,
실제 prebiotic chemistry와의 식별은 다른 문제다.

연령구조에서는 상태가 type과 age로 함께 색인된다. survival·fertility
kernel과 초기분포가 주어지면 성장률과 extinction threshold를 계산할
수 있다. 결합 유전·선택 정리는 지정 inheritance rule 아래의
**[정리]**이지 자연의 진화 역사를 무입력으로 재구성하는 결과가 아니다.

경험 gate는 원자료, 단위, exclusion rule, baseline과 판정 기준을
요구한다. 이 항목 없이 수치 근접을 evidence로 승격하지 않는다. chemical
network, 환경 변화, 관측 편향을 갖춘 실제 생명 기원으로 가는 사상은
**[미완성]**이다.

### 11.5 AGI architecture와 runtime

생명 모형의 검증 경계를 유지한 채 AGI architecture는 구현 설계로
다룬다. gauge lattice, recurrent layer, local/cloud cell과 agent loop는
채널·메모리·제어를 나누는 코드 구성요소이며, 뇌 영역의 등가물이 아니다.
hidden state $h$에서 후보 action 또는 token 점수 $s_a(h)$를 만들면

$$
p(a\mid h)=\frac{e^{s_a(h)}}{\sum_b e^{s_b(h)}}
$$

로 후보분포를 정의할 수 있다. 이는 선택 규칙의 **[정의]**이지 의식이나
물리적 측정의 유도가 아니다. 타당성은 unit test, failure isolation,
latency·memory budget, safety boundary와 task benchmark에서 판정한다.

CloudCell monad와 recurrent tower는 합성 규칙의 **[공리: 구현 선택]**이다.
사양 파일의 선언만으로 성능·인과성·보안이 보장되지 않으며, 일부 연결
주장은 부정적 감사가 있다. runtime 효과는 사전등록 실험 전까지
**[미완성]**이다.

### 11.6 memory, replay, planning

runtime이 한 step의 후보를 고르면 memory는 시간적으로 떨어진 관측을
어떻게 재사용할지 다룬다. episodic memory는 event–context–outcome,
working memory는 현재 제약, replay buffer는 과거 trajectory로 구분한다.
계층 planning은 상위 goal을 subgoal과 action sequence로 분해하지만,
성공은 미래 보상·안전 제약·도구 실패를 포함해 평가해야 한다.

replay의 효과는 동일 task distribution, context budget, seed, tool
availability에서 no-replay baseline과 비교해야 한다. retrieval oracle,
데이터 누수, prompt 차이를 제거한 ablation이 필요하다. memory가 길수록
무조건 좋다는 명제는 interference, stale context, compute cost 때문에
성립하지 않는다. biological hippocampus 대응과 일반 장기일관성 보장은
**[미완성]**이다.

### 11.7 sparsity, causal bridge와 OOD

sparsity는 활성 channel 수를 줄이는 설계 원리일 수 있으나 transformer
기질에서 특정 sparsity ratio가 자연 수렴한다는 보편 주장은 감사에서
지지되지 않았다. sparse mask, routing threshold, causal bridge는
hyperparameter와 ablation을 가진 **[공리: 모델 선택]**으로 관리한다.

OOD generalization은 학습 분포 밖 길이·구조·도구 조합에서의 성능이다.
in-distribution 평균 점수는 long context나 causal perturbation에서도
유지된다는 뜻이 아니다. edge 선택 효과는 randomized edge, dense baseline,
shuffled control과 비교해야 한다. intervention 또는 강한 식별 가정 없이
causal graph가 실제 원인을 회복했다고 말할 수 없으며, bridge closure와
세계모형 일반화는 **[미완성]**이다.

### 11.8 hallucination, residual gate와 검증

환각은 단일 상수보다 주장·근거·도구 결과의 불일치를 검출하는 절차로
다룬다. residual vector $r$와 양의 weight $W$를 고정하면

$$
R=r^\mathsf TW r
$$

를 제약 위반 score로 사용할 수 있다. $R$이 작다는 것은 선택한 제약과
일치한다는 뜻일 뿐 사실성·완전성·안전성 전체를 보장하지 않는다.

phi 재주입 gate는 $\alpha_\phi=0$과 $\alpha_\phi>0$의 ablation을 자료
전에 고정해 비교한다. accuracy, false allow, false block, latency,
비용과 failure 사례, shuffled-residual control을 함께 기록해야 한다.
hallucination gate나 claim verifier의 성공은 뇌 오류감시 또는 양자
선택 bridge의 증명이 아니다. coverage가 낮은 constraint, adversarial
prompt, source 오류와 reward hacking은 **[미완성]** 과제다.

## 12. 공학적 활용의 조건부 설계

뇌·AGI의 검증 규율은 공학 문서군에서도 유지된다. 아래 각 절은 필요한
외부 입력을 갖춘 조건부 설계식 또는 verifier이며, CE 코어가 재료·bath·
transport 계수를 무입력 산출한다는 주장은 하지 않는다.

### 12.1 핵융합 설계

핵융합 모형은 핵종, 온도·밀도, Coulomb barrier, plasma screening,
confinement time과 transport를 입력으로 요구한다. 두 전하의 유효
potential과 WKB tunnelling exponent를 계산할 수 있어도 reactor gain은
그 적분 하나로 정해지지 않는다. scalar coupling을 추가하려면 독립
EFT 계수와 실험 제약을 함께 둔다. 설계 검증은 반응률, power balance,
불안정성 및 안전 한계를 baseline plasma model과 비교하는 절차이며,
CE의 직접 에너지 생산 예측은 **[미완성]**이다.

### 12.2 양자오류보정

열린 양자계에는 system, bath, coupling, noise spectrum과 recovery
channel이 필요하다. 완전양성 trace-preserving map은

$$
\mathcal E(\rho)=\sum_kK_k\rho K_k^\dagger,\qquad
\sum_kK_k^\dagger K_k=I
$$

로 정의할 수 있지만, 이 식만으로 물리 noise나 threshold가 정해지지
않는다. logical error rate를 physical error, code distance, decoder
budget 아래에서 측정하고 no-recovery baseline과 비교해야 한다. CE
residual을 syndrome으로 동일시하는 사상은 **[미완성]**이다.

### 12.3 진공에너지

상수 potential은 $T_{\mu\nu}=-V_0g_{\mu\nu}$와 $w=-1$을 주지만,
laboratory device가 $V_0$를 조절한다는 결론은 없다. vacuum subtraction,
boundary condition, Casimir geometry와 measurable force를 구별해야 한다.
관측 가능한 힘·압력·열유속을 계산하는 것은 조건부 문제이고, absolute
vacuum energy extraction 또는 gravitational shielding은 현재 근거 없는
**[미완성]** 주장이다.

### 12.4 공학적 이론 한계

초광속 signalling, 순간이동, 무한 에너지 이득, 완전한 error-free
computation은 설계 목표가 아니라 먼저 no-go와 resource accounting을
적용할 대상이다. 인과성에는 light cone과 signal definition, 열역학에는
work reservoir와 entropy flow, 양자에는 no-cloning과 measurement
back-action이 필요하다. 조건부 식을 이런 경계 밖으로 외삽하지 않는다.

### 12.5 초전도체 설계

초전도 모형에는 band structure, pairing interaction, Coulomb repulsion,
phonon 또는 bosonic glue, disorder와 bath가 필요하다. BCS형 gap equation

$$
\Delta_k=-\sum_{k'}V_{kk'}\frac{\Delta_{k'}}{2E_{k'}}
\tanh\!\left(\frac{E_{k'}}{2T}\right)
$$

은 주어진 kernel 아래의 self-consistency 식이다. Floquet drive나
scalar coupling을 넣어도 heating, lifetime, competing order를 검증해야
한다. CE가 임의 재료의 $T_c$를 산출한다는 주장은 **[미완성]**이다.

### 12.6 공학식 총람과 master-action 경계

master action은 Euler--Lagrange equation을 주는 변분 장치다. 유체
점성, Lindblad 소산, BCS collision integral, nuclear transport에는
추가 자유도와 coarse graining이 필요하다. 따라서 총람의 식은 필요한
입력·정의역·출력·검증 benchmark를 함께 읽어야 하며, 서로 다른 분야의
계수를 하나의 CE 상수로 대체하지 않는다.

### 12.7 Euler--Riemann attention

Euler--Riemann attention은 위치·거리·회전 bias를 transformer에
넣는 구현 가설이다. attention score는

$$
\operatorname{softmax}\!\left(\frac{QK^\mathsf T}{\sqrt d}+B\right)V
$$

처럼 baseline과 같은 비용·parameter budget에서 비교한다. 수론 상수나
Euler 구조를 bias에 넣는 일이 language modeling 성능을 보장하지 않으며,
seed·length·ablation과 baseline이 판정의 기준이다.

### 12.8 기상 그래프 잔차

기상 관측망은 station node, edge, 시간해상도, 결측 처리와 forecast
target을 명시해야 한다. graph residual은 예측과 관측의 차이를 공간
연결에 따라 집계하는 목적함수일 뿐 물리 PDE의 대체가 아니다. 독립
forecast baseline, spatial holdout, extreme-event slice와 calibration을
비교해야 하며, data leakage와 station-density bias는 핵심 한계다.

### 12.9 무차원 잔차장 환각억제

서로 단위가 다른 제약은 기준 scale로 나누어야 비교할 수 있다. residual
$\widehat r_i=r_i/\Lambda_i$와 양의 weight를 택하면

$$
\mathcal J=\frac12\sum_iw_i|\widehat r_i|^2
$$

는 무차원 score다. threshold는 false allow·false block·latency를
포함한 held-out benchmark로 정해야 한다. 낮은 residual은 선택한
검사의 통과이지 사실성 전체의 증명은 아니다.

### 12.10 PreEq LLM manifest verifier

PreEq verifier는 후보 답변·제약·근거를 후보분포와 manifest 절차로
기록하는 도구다. 입력 schema, source provenance, reject reason과
replay log가 없으면 재현 가능한 verifier가 아니다. 검증은 known-valid,
known-invalid, ambiguous case와 shuffled-evidence control에서 precision,
recall, abstention을 함께 보고한다. 도구의 판정은 이론 지위의 자동
승격이 아니라 구현 결과다.

### 12.11 claim residual verifier

claim residual verifier는 주장 $c$, 전제 $a$, 관측 또는 계산 $o$ 사이의
불일치를 구조화한다. 허용 범위를 넘는 residual은 claim을 거부·보류할
이유가 되지만, residual 0은 누락된 전제나 잘못된 ontology를 검출하지
못할 수 있다. 따라서 audit trail, adversarial claim, human review와
versioned input을 함께 둔다. 이 verifier는 문서 규율을 돕는 공학 도구이며
자연 법칙의 판정기가 아니다.

## 13. 리만, MRA와 수론 기반 attention의 교훈

수론과 transformer를 잇는 MRA 문서군은 ‘장식적인 수식’과 실제 성능
기제를 분리해야 한다는 CE의 검증 사례다. 아래 절은 positional encoding
사양, attention block, ablation 결과와 해석 한계를 차례로 둔다.

### 13.1 문제 설정과 Riemann positional encoding

긴 context의 causal transformer는 위치 $n$과 상대 거리 $n-m$를 어떤
좌표로 표현할지 선택해야 한다. RoPE류 회전 encoding은 위치별 위상을
곱하지만 매우 긴 길이에서 phase wrapping과 주파수 배치가 일반화를
제한할 수 있다. Riemann surface positional encoding은 복소 좌표,
Mellin형 scale, 또는 zeta 영점 후보를 위치 feature로 넣는 **[공리:
구현 선택]**이다.

입력에는 sequence length, model dimension, head 수, frequency schedule,
causal mask와 normalization 규약이 필요하다. 수론 객체를 feature에
넣는 것은 정의일 뿐 Riemann hypothesis, 양자 chaos 또는 언어 구조의
증명이 아니다. 같은 parameter·token·optimizer budget 아래에서 표준
RoPE, ALiBi, xPos, NoPE와 비교할 때만 성능 주장이 의미를 갖는다.

### 13.2 MRA 수식과 블록

MRA block은 query $Q$, key $K$, value $V$와 위치 bias $B$를 받아

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\!\left(
\frac{QK^\mathsf T}{\sqrt{d_h}}+B\right)V
$$

를 계산한다. $d_h$는 head dimension이다. MRA는 $B$ 또는 회전
frequency·amplitude에 Mellin/Riemann 후보 항을 넣으며, causal mask는
미래 위치를 차단한다. block 사양은 residual connection, normalization,
FFN, dropout과 tensor shape까지 고정해야 재현 가능하다.

logarithmic distance decay 같은 항은 멀리 떨어진 token의 score를
감쇠시키는 직접적 기제다. 반면 zeta zero frequency는 회전 주파수의
후보일 뿐, attention spectrum을 자동으로 개선하지 않는다. 따라서
수식은 각 knob를 독립적으로 끄고 켤 수 있게 구현해야 하며, block의
정확한 kernel과 numerical stability는 사양·단위검사로 확인한다.

### 13.3 실험, ablation과 부정 결과

실험은 character-level language modeling의 perplexity, seed 간 분산,
학습 step, wall-clock, parameter 수와 length extrapolation을 함께
기록한다. ablation은 baseline RoPE에서 amplitude, frequency, decay,
Hermitian option을 하나씩 바꾸고 동일 data split과 optimizer를 유지한다.
짧은 학습의 큰 분산 차이는 early-stopping artifact일 수 있으므로,
충분한 step과 여러 seed가 필요하다.

현재 문서군의 핵심 부정 결과는 zeta 영점을 attention frequency로 직접
쓴 가설이 효과적이지 않았다는 점이다. 반면 일부 lean MRA 변종의
분산 감소나 OOD length 결과는 attenuation 또는 rotation 제거라는
더 단순한 설명과 경쟁한다. 특히 NoPE 또는 강한 거리감쇠가 좋은 결과를
보이면, 이를 수론적 기제의 승리로 부르지 않는다.

### 13.4 해석 한계와 후속 검증

MRA의 관측된 결과는 작은 모델·작은 corpus·제한된 seed의 조건부
benchmark다. BPE 또는 sentencepiece, 대형 모델, native CUDA backend,
long-context distribution shift와 bidirectional encoder에서 동일한
결론이 유지되는지는 아직 알 수 없다. attention 행렬의 고유값 분포가
GUE 통계를 따른다는 해석도 직접 측정 없이는 가설이다.

후속 검증은 train length를 고정한 뒤 더 긴 eval length에서 extrapolation
곡선을 보고, model scale과 seed 수를 늘리고, 모든 ablation의 비용을
공개하는 방식이어야 한다. 수론 기반 feature가 baseline보다 낫다는
주장은 이 절차의 독립 자료에서만 **[예측]** 또는 경험 결과가 될 수
있다. MRA는 CE 물리 코어의 증명이 아니라, 부정 결과를 보존하는
검증형 ML 연구다.

## 14. 재현과 문서 지도의 사용법

이 논문이 제시한 구분은 재현 절차에서도 유지된다. 수학 명제는 증명
원장에서 전제를 확인하고, 수치 코어는 고정점 solver와 무차원 검사를
수행하며, 문서 정책은 태그와 반례 처리를 점검한다. 대표적인 좁은 검사는
다음과 같다.

    py -m pytest tests\test_bootstrap_solver.py -q -p no:cacheprovider
    py -m pytest tests\test_dimensionless.py -q -p no:cacheprovider
    py -m pytest tests\test_canonical_document_policy.py -q -p no:cacheprovider

실행 환경·임시 경로·검사 대상은 저장소의 현재 harness 규약을 따른다.
코드가 식을 높은 정밀도로 푼다는 것은 구현된 정의와 알고리즘을 검증한다.
그 변수에 자연의 물리량이라는 해석을 부여하는 문제는 별도의 작용, 자료,
likelihood와 예측 절차를 요구한다.

상세를 확인할 때에는 이 문서의 흐름을 끊지 않도록 역할별로 돌아가면 된다.
[공리계](axium.md)는 공변 EFT·확률·우주론 branch의 정본 정의를,
[선택과 접힘](5_유도/00_선택과_접힘.md)은 중심 서사를,
[경로적분](경로적분.md)은 물리 모형군을, [상수 원장](상수.md)과
[우주론 원장](검증_원장/상수_우주론_원장.md)은 수치·판본을, [등호이전
수학](9_등호이전/README.md)은 후보 농축과 경로 bridge를 제공한다.
분야별 상세는 강의, 유도, 뇌, AGI, 공학, 리만 문서군에 두되, 이 링크들은
논증의 누락을 감추는 전제가 아니라 증명·실험의 후속 근거다.

## 15. 남은 문제와 연구 우선순위

모든 절의 결과를 합치면 CE의 가장 큰 열린 문제는 여전히 두 개의 bridge다.
첫째는 양자 진폭에서 완전양성 reduced dynamics와 비음수 분지과정으로
가는 길이다. 둘째는 선택·잔류 확률에서 보존되는 species current와
우주론 stress tensor로 가는 길이다. R1은 이 둘째 길에서 유계 Gaussian
저장소의 조건부 작용ㆍ인과ㆍ양성ㆍ총 Ward 장부까지 전진했지만, 초기
$\Pi_{\rm F}$의 0차원 기원과 결합 섭동 안정성은 닫지 못했다. 이 둘이 닫히지 않는 한,
끼임–접힘–암흑 표현은 동기를 주는 물리 서사이며, 일련의 조건부 정리와
공리가 그 주변에 놓인 구조로 읽어야 한다.

그 다음 우선순위는 EFT의 renormalized matching, 진공에너지 절대값,
flavor 선택 potential과 RG 안정성, scalar–tensor 및 perturbation의
공동 likelihood, 실제 데이터의 사전 고정 검증이다. 생명·뇌·AGI와 공학
응용에서는 계량·관측량·baseline·독립 자료·반례 규칙을 먼저 고정해야
한다. MRA처럼 부정적 결과를 보존하는 것이 이 과정의 일부다.

CE를 한 문장으로 요약하면, 이는 수치 일치를 모으는 프로그램이 아니라
선택의 물리 서사와 후보 농축의 수학, 공변 EFT, 조건부 확률·우주론
readout, 그리고 검증형 응용 사이의 경계를 드러내는 연구 프로그램이다.
그 경계를 지키는 한, 닫힌 정리는 재사용할 수 있고 열린 사상은 정확히
어떤 증명·작용·관측을 더 요구하는지 알 수 있다.

## 16. 문서 근거 지도

이 통합 논문은 `docs`의 167개 Markdown을 다음 문서군으로 분류해 읽었다.
강의 3개는 연역·귀납·상수 독해를, 경로적분과 응용 17개는 물리 문제와
형식 경계를, 상수 14개는 입력·경험식·우주론 판본을, 유도 7개는 재현
가능한 조건부 전개를 담당한다. 뇌 30개와 AGI 33개는 검증·감사·runtime
확장을, 공학 11개는 조건부 설계를, 리만 3개는 MRA 사양·실험을,
등호이전 35개는 후보공간·측도·bridge 수학을, 루트와 참조 14개는
공통 공리·증명·기호를 담당한다.

이 분류는 원장·감사·사양을 논문형 본문의 서술 자료로 읽되 그 지위나
수치를 변경하지 않는다는 뜻이다. 상세 외부 참고문헌은 각 분야 문서의
참고문헌에 보존하며, 이 문서의 인용은 저장소 내부의 근거 위치를 가리킨다.
새로운 수치, 독립 예측, 또는 기존 지위의 승격은 이 통합 논문에서
제안하지 않는다.
