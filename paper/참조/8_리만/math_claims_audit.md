# 리만·Mellin 사양의 수학적 판정과 사용 조건

## 1. 계산 전 검산 계약: CE-MATH-R1

저장소의 리만 위치 부호화와 MRA 원고의 수학적 함의를 직접 판정한다.
경험적 언어모델 성능을 새로 재현하거나 리만 가설을 증명하는 작업은 아니다.
일반 명제는 아래 증명·반례로 판단하고, 별도 구현은 그 항등식을 검산한다.

고정 반례는 위치쌍 $(0,1)$의 평행이동·두 종류의 배율 변환,
cycle 좌표 $(1/10,3/5)$의 $1/2$ 이동, 한 채널 $\gamma=1$의 복소
가중치, $T=\operatorname{diag}(2,1/2)$와 두 행이 같은
$(9/10,1/10)$ 확률행렬이다. 양의 커널 검산은 주파수 $(1,2,3)$,
비음수 가중치 $(1/2,1/3,1/6)$와 고정 위치들을 쓴다.
난수 seed 120911, 부동소수 허용치 $10^{-10}$, 관측 입력·피팅은 없다.
정확한 행렬·분수 항등식은 SymPy로 별도 검산한다.

검산 전에 이 절을 해시로 고정한다. 실제 제타 영점을 계산하는 데서
발생하는 수치 오차나 특정 backend의 구현 성능을 위 검산으로 보장하지 않는다.

## 2. 증명과 반례

### R1. 실수 주파수 선택은 리만 가설을 요구하지 않는다

**[정의·정리]** 유한한 실수열 $\gamma_1,\ldots,\gamma_K$와 $x>0$에
대해 $x^{i\gamma_k}=\exp(i\gamma_k\log x)$는 잘 정의되고 절댓값은 1이다.
따라서 해당 2차원 실수 회전은 직교이고 norm을 보존한다.

**증명.** 양의 실수축의 실수 로그를 사용하면 지수의 실수부가 0이다.
$R(\theta)^\mathsf T R(\theta)=I$는 $\cos^2\theta+\sin^2\theta=1$에서
따른다. 이 증명은 $\gamma_k$가 제타 영점인지 사용하지 않는다. $\square$

비자명 영점 전체의 위치에 대한 RH와, 정규화된 영점 간격의 통계에 대한
GUE 주장은 서로 다른 명제다. 유한 주파수 목록을 고정하는 데 둘 다
필요하지 않다. 비자명 영점 전체를 critical line의 목록으로 대체하려면
RH를 별도 가정해야 한다. [Clay의 RH 설명](https://www.claymath.org/millennium/riemann-hypothesis/)은
이를 미해결 문제로 다룬다. 이 문서는 그 지위를 바꾸지 않는다.

### R2. 로그 좌표의 정확한 대칭

**[정리]** $p>-1$, $\tau(p)=\log(1+p)$라 하자.
같은 주파수를 쓰는 회전의 상대 인자는
$\gamma\log((1+p_j)/(1+p_i))$다.
위치 변환 $p\mapsto c(1+p)-1$, $c>0$는 이 차이를 보존한다.
보통의 $p\mapsto p+a$나 $p\mapsto cp$는 일반적으로 보존하지 않는다.

**증명.** $R(\theta_i)^\mathsf T R(\theta_j)=R(\theta_j-\theta_i)$이고
$\tau(c(1+p)-1)=\log c+\tau(p)$다. 반면 $(p_i,p_j)=(0,1)$에서
로그 차이는 $\log2$지만 $(1,2)$로 평행이동하면 $\log(3/2)$,
두 배한 $(0,2)$에서는 $\log3$이다. $\square$

$c\ge1$, $p\ge0$에서 보통 배율의 오차는 정확히

$$
\tau(cp)-\tau(p)-\log c
=\log\frac{p+1/c}{p+1}
$$

이다. $-\log(1-u)\le u/(1-u)$, $0\le u<1$를 쓰면 절댓값은
$(c-1)/(cp+1)$ 이하로 제어된다. $c=1$에서는 0이다.
이는 원점 부근을 제외한 점근 제어이며 모든 위치의 정확한 불변성이 아니다.
토큰 내용 $q_i,k_j$와 mask는 별도 입력이므로 위치 인자의 대칭만으로
전체 attention 함수의 불변성을 주장하지 않는다.

### R3. Sheet index가 대칭과 연속성에 미치는 영향

**[정리·반례]** $\sigma(a)=\lfloor a\rfloor$에 대해
$|\sigma(a)-\sigma(b)|$는 공통 실수 이동에 일반적으로 불변이 아니다.
정수 이동에서는 불변이며, 임의 공통 이동으로 생기는 변화는 1 이하이다.

**증명.** $a\ge b$이면 $\lfloor a\rfloor-\lfloor b\rfloor$는
$\lfloor a-b\rfloor$ 또는 $\lceil a-b\rceil$다. 공통 이동은 $a-b$를
보존하므로 차이가 최대 1이고, 정수 이동은 두 floor에 같은 정수를
더한다. $(a,b)=(1/10,3/5)$의 floor 거리 0은 $1/2$ 공통 이동 뒤
1이 된다. $\square$

따라서 회전 부분의 정확한 shifted-scale 대칭도 sheet penalty까지
포함하면 깨질 수 있다. $\lambda_\sigma\ge0$인 채널 평균 penalty의
변화 절댓값은 $\lambda_\sigma$ 이하이나, 항상 0은 아니다.

또 $a\to1^-$와 $a\to1^+$, $b=1/2$에서 복소 위상은 연속으로
이어지지만 floor 거리는 0에서 1로 뛴다. 이는 명시적으로 도입한
score의 불연속이다. 이를 실제 양자 기록의 판정 경계로 동일시하지 않는다.
양의 실수축의 $\log x$ 자체는 단일값이다. 위상을 sheet와 잔여각으로
분해하는 것은 좌표 표현이며, 물리적 Riemann surface의 발생 증명이 아니다.

### R4. Hermitian 양의 커널을 얻는 충분조건

**[정리]** $w_k\ge0$, $\gamma_k,\tau_i\in\mathbb R$이면

$$
K_{ij}=\sum_k w_k e^{i\gamma_k(\tau_i-\tau_j)}
$$

는 Hermitian positive-semidefinite 행렬이다.

**증명.** $V_{ik}=\sqrt{w_k}e^{i\gamma_k\tau_i}$라 두면 $K=VV^\dagger$다.
따라서 $K^\dagger=K$와 $c^\dagger Kc=\|V^\dagger c\|^2\ge0$이다.
동일한 content feature를 양쪽에 사용하는 Gram 구성도 같은 증명으로
성립한다. 서로 다른 query/key에는 이 증명을 적용할 수 없다. $\square$

실수 주파수만으로 자기수반 score가 보장되지는 않는다. 자유롭게 선택할
수 있는 회전 후 query/key가
$Q=I_2$, $K=\begin{pmatrix}1&1\\0&1\end{pmatrix}$이면
$QK^\mathsf T=\begin{pmatrix}1&0\\1&1\end{pmatrix}$로 비대칭이다.
각 위치의 회전이 가역이므로 어떤 고정 위치에서도 이러한 feature를
주는 회전 전 입력을 선택할 수 있다.

### R5. 기존 MRA의 tied projection만으로는 Hermitian이 아니다

**[반례]** 한 채널에서 $\gamma=1$,
$w=(1/2+i)^{-1}=2/5-4i/5$이고 $q_i=k_i=1$이면
기존 score의 대각성분은 $S_{ii}=w$다. Hermitian 행렬의 대각성분은
실수여야 하므로 tied projection만으로는 충분하지 않다.

실수부를 취해도 일반 대칭은 회복되지 않는다.
$a=\operatorname{Re}w$, $b=-\operatorname{Im}w>0$,
$x_i/x_j=4$, $\gamma\log4=\theta$라 하면

$$
\operatorname{Re}S_{ij}=\frac12(a\cos\theta-b\sin\theta),\qquad
\operatorname{Re}S_{ji}=2(a\cos\theta+b\sin\theta).
$$

두 식은 일반적으로 다르다. 이 수치는 유한 고정 반례로 검산한다.
비대칭 amplitude가 없어도 복소 가중치의 sine 항이 남는다.

**[조건부 대안]** 복소 가중치를 제거하고 먼저 R4의 $K\succeq0$를 만든
경우, $D=\operatorname{diag}(x_i^{-1/2})$로 $S=DKD^{-1}$라 두면
$G=D^{-2}$인 가중 내적에서만 $S^\dagger G=GS$다.

**증명.** 두 변을 대입하면 모두 $D^{-1}KD^{-1}$이다. $\square$

이는 통상적인 내적의 Hermitian 조건이 아니며 기존 복소 가중치 전체에
대한 구제도 아니다. Causal mask나 행별 softmax는 별도로 검사해야 한다.
대칭 실수 score $L=\begin{pmatrix}0&0\\0&\log2\end{pmatrix}$도
행별 softmax 뒤에는 $\begin{pmatrix}1/2&1/2\\1/3&2/3\end{pmatrix}$가 되어
비대칭이다. 유한 대칭화 옵션을 Hilbert–Pólya 추측의 직접 구현으로
부르는 주장은 성립하지 않는다.

### R6. Determinant·수축·unitarity는 서로 다른 조건이다

**[반례와 함의]** 복소 정사각행렬 $T$에서 $T^\dagger T=I$는 unitary 조건,
$T^\dagger T\preceq I$는 Euclidean norm 수축 조건이다.
정사각행렬에서는 수축이 $|\det T|\le1$을 함의하지만 역은 거짓이다.
$T=\operatorname{diag}(2,1/2)$는 determinant 1이면서 norm 2다.

**증명.** 특이값으로 쓰면 수축은 모든 $\sigma_j\le1$,
unitarity는 모든 $\sigma_j=1$, determinant 조건은
$\prod_j\sigma_j\le1$이다. 반례에 직접 대입한다. $\square$

출력 사영 $W_o$만 수축으로 만들어도 전체 attention 또는 residual
map의 수축은 따르지 않는다. 두 행이 $(9/10,1/10)$인 확률행렬은
$\ell^2$ operator norm이 $\sqrt{41}/5>1$이다.
$f(x)=x$는 norm 1인 map이지만 $x+f(x)=2x$다.
따라서 norm 제약에서 환각 방지나 잔차 합의 비증폭을 연역할 수 없다.

### R7. Mellin 커널과 제타 함수의 정확한 범위

**[정의·유도]** $f\in L^1((0,\infty),dx/x)$이면

$$
\widehat f(\gamma)=\int_0^\infty f(x)x^{-i\gamma}\frac{dx}{x}
=\int_{\mathbb R}f(e^u)e^{-i\gamma u}\,du.
$$

$u=\log x$의 변수변환이므로 Mellin의 한 부호 규약이 로그 좌표의
Fourier transform이 된다. 절대적분 가능성이 두 적분의 존재와
변수변환을 보장한다. [DLMF의 Mellin 정의](https://dlmf.nist.gov/1.14#iv)와
연결할 때에는 $dx$와 $dx/x$, $s-1$ 지수와 부호를 함께 변환해야 한다.
점별 phase $x^{i\gamma}$ 하나가 적분 연산자 전체인 것은 아니다.

$\zeta(s)=\sum_{n\ge1}n^{-s}$라는 Dirichlet 급수 정의는
$\operatorname{Re}s>1$의 절대수렴 영역에서 사용한다.
Critical line의 순수 위상 유한합을 이 급수 자체로 부를 수 없다.
다른 영역에는 analytic continuation이 필요하다.
[DLMF 제타 정의](https://dlmf.nist.gov/25.2#i)가 정의역을 명시한다.

기존 MRA의 유한 $x^{-(1/2+i\gamma)}/(1/2+i\gamma)$ 합에는 별도의
query/key와 truncation이 추가됐다. 이를 명시적인 score 정의로
사용할 수 있지만, 원래 explicit formula의 prime 항·trivial zero 항·
모든 영점의 합 규약을 생략하고 수론 항등식과 같다고 주장하지 않는다.
$x^\rho$에서 $x^{-\rho}$로 바꾸는 것도 domain과 다른 항들을 함께
변환해야 하므로 단순한 부호 교체로 전체 공식을 유지하지 못한다.

### R8. 자기수반 연산자의 존재와 RH의 구분

**[정리]** 임의의 실수열 $\gamma_n$에 대해 $\ell^2$ 위의

$$
D c=(\gamma_n c_n)_n,\qquad
\operatorname{Dom}D=\{c\in\ell^2:\sum_n\gamma_n^2|c_n|^2<\infty\}
$$

는 조밀하게 정의된 자기수반 연산자다.

**증명.** 유한 지지 수열이 정의역에 포함돼 조밀하고, 실수 $\gamma_n$
덕분에 $D$는 대칭이다. Adjoint 정의를 각 표준기저 $e_n$에 적용하면
$y\in\operatorname{Dom}D^*$일 때 $(D^*y)_n=\gamma_n y_n$이고 이
수열이 $\ell^2$에 속해야 한다. 이는 정확히 위 정의역 조건이다.
역으로 이 조건을 만족하면 Cauchy–Schwarz로 adjoint pairing이
성립한다. 따라서 $D^*=D$이고 정의역도 같다. $\square$

이는 임의의 실수열에 성립하므로 그 존재만으로 제타 함수의 모든
비자명 영점과의 spectral 동정을 증명하지 못한다.
유한 attention 행렬, 선택된 주파수와 학습 성능으로 이 누락된 동정을
대체할 수 없다.

## 3. 판정과 재사용

확정 가능한 도구는 norm을 보존하는 회전, 정확한 로그 비율,
가정이 명시된 Mellin 변환, 비음수 가중 Gram kernel,
정의역을 명시한 대각 자기수반 연산자다.
기존 사양의 자동 translation invariance·tied projection Hermitian·
determinant로 unitary 보장·출력 사영만으로 전체 비증폭은 기각한다.

계량·곡률의 비유일성은 [측정 12장 B9](../../01_측정과_접힘/12_토글_판정경계와_관측밖_상태의_수학.md)에
별도 증명돼 있다. 이 도구들이 물리적 암흑부문이나 사건 선택을
정한다는 연결은 여기서 얻지 못했다.

## 4. 검산

[별도 계산](../../../verify/ce_mathematical_boundary_audit.py)과
[결과 JSON](../../../verify/ce_mathematical_boundary_audit.json)에
회전 항등식·로그 비율·floor 반례·Gram 양성·MRA factorization·
Hermitian 및 수축 반례를 기록했다. 전체 B·R·E 묶음은 정확 항등식
69개를 확인하고 반례 30항목을 기록한다. 부동소수 최대 절대오차는
$2.5121479338940403\times10^{-15}$로 허용치 $10^{-10}$ 이하다.

이 문서 사전등록 prefix의 SHA256은
b16bae05689ff530e994688baa98b9456389c6a3a1eb482954ed3b1dd969e8e6이다.
일반 정리는 본문 증명으로 판단한다. 실제 제타 영점 목록·backend·
과거 언어모델 성능의 재검증과 proof-assistant 형식 증명은 이 계산에 없다.
