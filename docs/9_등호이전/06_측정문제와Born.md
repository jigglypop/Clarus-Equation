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
| Born rule은 접힘 역학 결과 | Born prior 유도는 아직 bridge |

## 7. 다음 작업

Born bridge를 닫으려면 별도 문서가 필요하다.

가능한 제목:

`06a_Born_prior_유도.md`

필요한 항목:

1. 상태공간이 Hilbert space일 때 확률측도 선택 공리
2. 위상 불변성
3. 직교분해 가법성
4. \(\mu(i)=|c_i|^2\) 유일성
5. CE 경로적분 prior와의 대응
