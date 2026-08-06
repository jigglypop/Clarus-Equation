# 06. 측정 문제와 Born bridge

## 0. 목표

이 장은 등호 이전 수학을 양자 측정 문제에 연결한다.

핵심 구분:

- PreEq는 후보분포가 조건에 의해 어떻게 manifest 되는지 설명한다.
- Born rule 전체를 자동으로 증명하지는 않는다.
- \(\mu_0(i)=|c_i|^2\)를 왜 써야 하는지는 별도 bridge다.

## 1. 후보공간

양자 상태가

$$
|\psi\rangle=\sum_i c_i|i\rangle
$$

라 하자. 측정 후보공간은

$$
A=\{i\}
$$

이다.

Born prior를 받아오면

$$
\mu_0(i)=|c_i|^2
$$

이다.

이 단계는 현재 `Bridge`다.

## 2. 측정 조건 에너지

측정 장치와의 상호작용은 후보 \(i\)마다 조건 에너지를 만든다.

$$
E_{\mathrm{meas}}:A\to\mathbb R_{\ge0}
$$

PreEq 재가중은

$$
\mu_\beta(i)
=
\frac{e^{-\beta E_{\mathrm{meas}}(i)}|c_i|^2}
{\sum_j e^{-\beta E_{\mathrm{meas}}(j)}|c_j|^2}
$$

이다.

## 3. Manifest

최소 에너지 후보 집합

$$
A_*=\operatorname*{argmin}_{i:|c_i|^2>0}E_{\mathrm{meas}}(i)
$$

으로 \(\mu_\beta\)가 집중한다.

유일 최소 후보 \(k\)가 있으면

$$
\mu_\beta\to\delta_k
$$

이다.

해석:

$$
|\psi\rangle
\quad\leadsto\quad
k
$$

는 외부에서 갑자기 생기는 collapse가 아니라, 후보분포가 측정 조건 에너지 아래에서 manifest 되는 과정으로 읽을 수 있다.

## 4. Born rule과의 정확한 관계

중요한 점은 다음이다.

PreEq가 직접 주는 것은

$$
\mu_0 \mapsto \mu_\beta \mapsto \mu_\infty
$$

이다.

Born rule은

$$
\mu_0(i)=|c_i|^2
$$

를 말한다.

따라서 Born rule 전체를 증명하려면 다음 중 하나가 필요하다.

1. Hilbert space norm에서 확률측도 \(\mu_0\)가 유일하게 \(|c_i|^2\)임을 보이는 정리
2. Gleason류 정리와 PreEq prior의 연결
3. CE 경로적분의 위상/간섭 구조에서 \(|c_i|^2\)가 prior로 내려오는 유도

현재 이 셋은 이 폴더에서 증명하지 않는다.

### 4.1 표준 측정 기준선과 CE가 추가해야 할 것

표준 finite-dimensional 측정 instrument는 Kraus operator \(M_k\)로

$$
p(k)
=
\operatorname{Tr}\!\left(M_k\rho M_k^\dagger\right),
\qquad
\rho_k
=
\frac{M_k\rho M_k^\dagger}{p(k)},
$$

$$
\sum_k M_k^\dagger M_k=I
$$

를 만족한다. 현재 `reality_stone.clarus.quantum`에는 일반 Hermitian
Hamiltonian 진화, density matrix, Born sampling, GKSL/Lindblad 기준선이
구현되어 있다. 이는 CE 증거가 아니라 CE 측정 bridge가 반드시 회복해야 할
baseline이다.

이번 감사에서는 독립 스칼라장 분기 \(\phi\ne R\)만 채택했다. 최소 action
후보는

$$
S_\phi
=
\int d^4x\sqrt{-g}
\left[
-\frac12(\nabla\phi)^2
-\frac12m_\phi^2\phi^2
-\frac12\xi R\phi^2
-V_{\rm int}(\phi)
\right]
$$

이다. 반면 Hessian/Jacobi의 scalar 투영

$$
\Phi_{\rm eff}[\gamma,\eta]
=
\frac{\langle\eta,\mathcal J_\gamma\eta\rangle}
{\langle\eta,\eta\rangle}
$$

은 별도 readout이다. `jacobi_rayleigh_scalar`는 이 투영의 유한차원 형식
게이트를 구현하지만 \(\Phi_{\rm eff}=\phi\)라는 mapping은 만들지 않는다.
`ScalarFieldMassGap`도 \(E=\hbar\omega=hf\)의 단위 변환일 뿐
현행 \(m_{\rm light}=29.6991596\,{\rm MeV}\)의 측정 또는 유도가 아니다.
과거 \(29.64757\,{\rm MeV}\) fixture도 같은 이유로 물리 pole 증거가 아니다.

분지 억압 후보를 확률로 쓰면 정규화된 식은

$$
\widetilde p_k
=
\frac{
p(k)e^{-\Phi_k/\Lambda_\Phi}
}{
\sum_jp(j)e^{-\Phi_j/\Lambda_\Phi}
}
$$

다. \(\Phi_k\)가 모든 분지에서 같으면 표준 \(p(k)\)를 그대로 보존한다. 이
경우 CE 고유 예측은 없다. \(\Phi_k\)가 다르면 Born 확률에서 벗어나므로,
\(\Phi_k\)를 apparatus/action에서 계산하고 수정된 map이 CPTP와 no-signalling을
만족함을 보여야 한다.

PreEq의

$$
\mu_\beta(i)
\propto
e^{-\beta E_{\rm meas}(i)}\mu_0(i)
$$

는 \(\mu_0\)를 조건부로 농축한다. 유일 최소 에너지가 있으면
\(\beta\to\infty\)에서 그 최소점을 결정론적으로 고르므로, 반복 실험에서 Born
빈도로 single-shot outcome을 생성하는 stochastic instrument는 별도 과제다.

## 5. 측정 조건도 후보가 된다

03장의 조건공간을 쓰면 measurement operator도 후보다.

조건공간:

$$
K=\{M_1,M_2,\dots\}
$$

값공간:

$$
A=\{i\}
$$

joint 상태:

$$
\rho(M,i)
$$

joint energy:

$$
E(M,i)
=
E_{\mathrm{outcome}}(M,i)+\lambda E_{\mathrm{apparatus}}(M)
$$

이다.

그러면 측정은 단순히 값 \(i\)만 고르는 일이 아니라, 측정 조건 \(M\)과 결과 \(i\)의 쌍이 manifest 되는 일이다.

이것이 TEMP의 "measurement도 모호함의 한 결"이라는 말을 수학적으로 읽는 방식이다.

## 6. CE 측정 문서와의 연결

`docs/4_공학적_활용/02_양자오류보정.md`는 측정 문제를 CE 접힘으로 읽는다.

이 장의 위치:

| CE 측정 문서 | 이 장 |
|---|---|
| 분지별 경로 가중치 \(W_k\) | 후보 prior와 조건 재가중 |
| 접힘은 붕괴가 아님 | manifest는 농축 극한 |
| 비고전 경로는 소멸하지 않음 | 비선택 잔류 \(\mu_{\mathrm{ns}}\) |
| Born rule은 접힘 역학 결과라는 과거 주장 | 현재는 finite branch prior 보존 후보로 강등, CE 경로적분 유도는 `Open` |

## 7. 다음 작업

Born prior의 finite branch 유도 조건은 [06a_Born_prior_유도.md](06a_Born_prior_유도.md)로 분리했다.

남은 작업:

1. 06a의 branch refinement 공리를 실제 측정장치 모델과 연결
2. CE 경로적분의 분지 가중치 \(W_k\)에서 \(|c_k|^2\)가 내려오는 조건 정리
3. \(E_{\mathrm{meas}}\)와 장치 상호작용/접힘 에너지의 대응 검토
4. \(M_k\), CPTP, no-signalling을 회복하는 장치 map 구현
5. 표준 Born sampling과 CE 수정안의 동일-seed preregistered ablation
6. 독립장 결합 \(H_{\rm int}=gA\otimes\mathcal O_\phi\), bath state와
   \(J_\phi(\omega,T)\)를 고정한 뒤
   \(\gamma_\phi(\omega)=g^2J_\phi(\omega)\ge0\)의 유효 범위를 검증
