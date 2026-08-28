# 06a. 조건부 유한분지 Born weight 정리

이 문서는 유한 분지 확률 assignment가 명시한 ray 불변성·정규화·대칭·refinement·연속성 공리 아래 제곱진폭 규칙으로 제한되는 조건부 정리를 제시한다. 이 결과는 Born 규칙의 무가정 물리 유도가 아니며 공리와 measurement bridge의 독립 정당화가 남는다.

독자는 06의 instrument 기준선과 유한 확률의 조건부 readout을 먼저 읽는다. 공리 B0–B5, 동일·유리·일반 진폭의 정리, 말하지 않는 것, PreEq 결합과 기계 검산 경계를 순서대로 확인한다.

## 0. 범위

정리의 정의역은 유한 branch와 지정한 probability functional이다. decoherence, infinite outcome, frequency, 실제 apparatus는 결론의 전제가 아니므로 별도로 분리한다.

이 문서는 모든 유한차원 직교분할에 걸친 probability assignment가 강한
refinement covariance와 연속성을 만족할 때 $|c_i|^2$가 강제됨을
증명한다. Gleason 정리의 대체 증명이나 실제 측정장치의 유도는 아니다.

## 1. Probability assignment

다음 공리는 probability assignment의 대칭과 정규화를 고정한다. 공리 중 하나를 Born 결론으로부터 정당화하면 순환이 되므로 각 항은 채택 전제로 읽어야 한다.

**[정의]** 모든 유한차원 Hilbert space의 orthonormal basis
$\mathcal B=\{|i\rangle\}_{i=1}^n$와 unit vector
$$
|\psi\rangle=\sum_i c_i|i\rangle
$$
에 probability vector
$\mu_{\psi,\mathcal B}\in\mathcal P(\{1,\dots,n\})$를 배정한다고 하자.

다음은 이 배정족 전체에 대한 **[공리]**다.

### B0. Ray 불변성

ray 불변성은 전역 위상이 확률을 바꾸지 않는다는 채택 가정이다. 이는 Hilbert ray의 물리적 해석을 증명하지 않는다.

$$
\mu_{e^{i\theta}\psi,\mathcal B}
=
\mu_{\psi,\mathcal B}.
$$

### B1. 정규화

정규화는 가능한 outcome 확률의 합을 1로 고정한다. 이는 측정 instrument가 존재한다는 사실과는 별개다.

$$
\mu_{\psi,\mathcal B}(i)\geq0,
\qquad
\sum_i\mu_{\psi,\mathcal B}(i)=1.
$$

### B2. Nullity

null amplitude outcome의 영확률은 assignment의 경계 조건이다. 실험의 detector inefficiency나 background count를 포함하지 않는다.

$$
c_i=0
\quad\Longrightarrow\quad
\mu_{\psi,\mathcal B}(i)=0.
$$

### B3. Equal-branch symmetry

동일 진폭 branch의 대칭은 라벨 재명명에 대한 불변성 가정이다. 실제 decoherence가 branch를 동등하게 만들었다는 동역학 정리는 아니다.

같은 basis에서
$$
|c_i|=|c_j|
\quad\Longrightarrow\quad
\mu_{\psi,\mathcal B}(i)
=
\mu_{\psi,\mathcal B}(j).
$$

이 공리는 상대위상과 label permutation이 equal-modulus branch의
확률을 가르지 않는다는 강한 대칭 가정이다.

### B4. Refinement covariance

refinement covariance는 branch 분할을 확률 assignment와 양립시키는 공리다. 어떤 refinement가 물리적으로 허용되는지는 별도 measurement model이 필요하다.

양의 정수 $n_i$에 대해 basis isometry를
$$
V|i\rangle
=
\frac1{\sqrt{n_i}}
\sum_{\alpha=1}^{n_i}
e^{i\vartheta_{i\alpha}}|i,\alpha\rangle
$$
로 정의할 수 있다고 하자. 그러면
$$
\mu_{\psi,\mathcal B}(i)
=
\sum_{\alpha=1}^{n_i}
\mu_{V\psi,\mathcal B'}(i,\alpha).
$$

즉 ancillary equal-amplitude refinement 뒤의 coarse probability가
원래 probability와 같다. 이는 단순한 유한 가법성보다 강하며, 서로 다른
Hilbert-space dimension의 배정을 연결한다.

### B5. 상태 연속성

연속성은 유리 제곱진폭에서 일반 진폭으로 넘어가는 정규성 가정이다. discontinuous assignment 가능성을 배제하는 선택임을 명시한다.

고정 basis에서 $\|\psi_k-\psi\|\to0$이면 각 $i$에 대해
$$
\mu_{\psi_k,\mathcal B}(i)
\to
\mu_{\psi,\mathcal B}(i).
$$

## 2. 동일 진폭

동일 진폭 경우는 equal-branch symmetry와 normalization이 직접 작동하는 출발점이다. branch 수가 유한하고 refinement가 허용된다는 가정 밖에서는 같은 계산을 쓰지 못한다.

**[정리]**
$$
|\psi\rangle
=
\frac1{\sqrt N}
\sum_{i=1}^Ne^{i\theta_i}|i\rangle
$$
이면
$$
\mu_{\psi,\mathcal B}(i)=\frac1N.
$$

**증명.** B3으로 모든 branch probability가 같은 $p$이고 B1로
$Np=1$이다. $\square$

## 3. 유리 제곱진폭

유리 제곱진폭은 refinement로 동일 branch를 구성해 계산한다. 이 구성은 공리 B4에 의존하며 physical splitting mechanism을 유도하지 않는다.

**[정리]** $|c_i|^2=n_i/N$이고
$n_i\in\mathbb Z_{\geq0}$, $\sum_i n_i=N$이면
$$
\mu_{\psi,\mathcal B}(i)=|c_i|^2.
$$

**증명.** $n_i=0$인 branch에는 $V|i\rangle=|i,1\rangle$인 한 개
zero-amplitude microbranch를 두고, $n_i>0$인 branch에는 B4의
refinement를 적용하면
$$
V|\psi\rangle
=
\sum_{i:n_i>0}\sum_{\alpha=1}^{n_i}
\frac{c_i}{\sqrt{n_i}}
e^{i\vartheta_{i\alpha}}|i,\alpha\rangle.
$$
모든 nonzero microbranch의 수는 정확히 $N$이고 coefficient
magnitude는
$$
\frac{|c_i|}{\sqrt{n_i}}=\frac1{\sqrt N}.
$$
B2로 zero-amplitude microbranch의 probability는 0이다. B3으로
나머지 $N$개 probability가 모두 같고 B1로 각각 $1/N$이다.
B4로
$$
\mu_{\psi,\mathcal B}(i)
=
\sum_{\alpha=1}^{n_i}\frac1N
=
\frac{n_i}{N}
=
|c_i|^2.
\quad\square
$$

## 4. 일반 제곱진폭

일반 진폭은 유리 근사와 B5 연속성으로 확장된다. 근사 topology와 probability functional의 연속성이 빠지면 결론이 성립하지 않을 수 있다.

**[정리]** B0--B5가 모든 유한 refinement에 대해 성립하면
$$
\mu_{\psi,\mathcal B}(i)=|c_i|^2
$$
이다.

**증명.** Probability simplex의 rational points가 dense하므로
$$
p^{(k)}_i\in\mathbb Q_{\geq0},
\qquad
\sum_i p_i^{(k)}=1,
\qquad
p_i^{(k)}\to|c_i|^2
$$
인 수열을 잡는다. $c_i\ne0$이면 $\theta_i=\arg c_i$로 두고,
$c_i=0$이면 $\theta_i$를 임의로 택해
$$
|\psi_k\rangle
=
\sum_i\sqrt{p_i^{(k)}}e^{i\theta_i}|i\rangle
$$
로 두면 $\|\psi_k-\psi\|\to0$이다. 3절과 B5로
$$
\mu_{\psi,\mathcal B}(i)
=
\lim_k\mu_{\psi_k,\mathcal B}(i)
=
\lim_kp_i^{(k)}
=
|c_i|^2.
\quad\square
$$

## 5. 정리가 말하지 않는 것

정리는 공리계 안의 유일성만 말하고 Born rule의 물리적 기원·decoherence·frequency·collapse를 설명하지 않는다. 이 범위 제한이 순환적 해석을 막는다.

B4는 physical ancilla에서 가능한 모든 refinement가 확률을 보존한다는
강한 **[공리]**다. 다음 항목은 위 증명에 포함되지 않으며
**[미완성]**이다.

- 실제 apparatus interaction이 B4 isometry를 구현하는 조건
- POVM과 무한차원 Hilbert space로의 확장
- contextual assignment를 배제하는 물리 원리
- preparation independence와 반복측정 frequency
- CE path amplitude에서 이 probability assignment로 가는 사상

따라서 이 결과를 “PreEq가 무가정으로 Born rule을 유도한다”고 읽지
않는다.

## 6. PreEq와의 결합

PreEq와 결합하려면 branch prior·energy·instrument가 Born 공리와 독립적으로 주어져야 한다. Born weight를 미리 Gibbs prior에 넣는 구성은 유도가 아니라 재표현이다.

위 공리계를 채택하면 초기 prior는
$$
\mu_0(i)=|c_i|^2
$$
로 고정된다. 그 뒤 무차원 측정 cost를 택한 재가중
$$
\mu_\beta(i)
=
\frac{e^{-\beta\mathcal I_{\rm meas}(i)}|c_i|^2}
{\sum_j e^{-\beta\mathcal I_{\rm meas}(j)}|c_j|^2}
$$
에는 [06_측정문제와Born.md](06_측정문제와Born.md)의 유한 농축 정리가
적용된다.

Born prior와 Gibbs selection은 역할이 다르다. 첫째는 초기 probability
assignment이고, 둘째는 추가 cost 아래의 조건부 재가중이다.

## 7. 기계 검산의 범위

기계 검산은 유한 공리 전개와 산술 항등식을 확인할 수 있다. 코드 통과는 공리의 물리적 참이나 실제 측정 bridge의 완결성을 증명하지 않는다.

유리 branch count와 위상 불변성의 유한 대수는
$$
\texttt{python -m pytest tests/test\_pre\_eq.py -q}
$$
로 회귀검사할 수 있다. 이 계산은 B0--B4를 구현한 예제를 검산할 뿐 B4의
물리적 보편성이나 quantum instrument를 증명하지 않는다.
