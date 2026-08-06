# 06a. Born Prior 유도 조건

## 0. 목표

06장은 Born prior

$$
\mu_0(i)=|c_i|^2
$$

를 bridge로 두었다. 이 문서는 어떤 공리를 추가하면 finite branch 상황에서 그 prior가 나오는지 정리한다.

주의:

> 이 문서는 전체 Gleason 정리의 완전 증명이 아니다. 여기서는 유한 분지, branch refinement, 연속성 공리 아래에서 $|c_i|^2$가 유일한 prior가 되는 최소 도구만 쓴다.

현재 판정:

| 항목 | 판정 |
|---|---|
| finite equal-branch proof | `Exact` |
| rational amplitude proof | `Exact under refinement axioms` |
| irrational amplitude extension | `Exact under continuity` |
| 물리적 측정장치와의 동일시 | `Bridge` |

## 1. 공리계 B

상태는 유한차원 Hilbert space의 unit vector다.

$$
|\psi\rangle=\sum_i c_i|i\rangle
$$

측정 후보공간은 직교 basis의 index 집합 $A=\{i\}$다.

### B0. 위상 불변성

전체 위상은 prior를 바꾸지 않는다.

$$
|\psi\rangle\sim e^{i\theta}|\psi\rangle
$$

### B1. 정규화

각 후보 prior는 음이 아니고 총합은 1이다.

$$
\mu_\psi(i)\ge0,\qquad \sum_i\mu_\psi(i)=1
$$

### B2. 직교 coarse-graining 가법성

서로 직교한 후보 묶음 $S\subset A$에 대해

$$
\mu_\psi(S)=\sum_{i\in S}\mu_\psi(i)
$$

이다.

### B3. 대칭성

측정 basis의 두 후보가 상태에서 같은 계수 크기를 가지면 prior도 같다.

$$
|c_i|=|c_j|
\quad\Rightarrow\quad
\mu_\psi(i)=\mu_\psi(j)
$$

### B4. Branch refinement 불변성

한 후보 $i$를 $n_i$개의 동일한 미세분지로 나누어도, 그 미세분지들의 prior 합은 원래 후보 $i$의 prior와 같다.

### B5. 연속성

$|c_i|^2$가 연속적으로 변하면 $\mu_\psi(i)$도 연속적으로 변한다.

## 2. 동일 진폭 정리

**정리 2.1**  

$$
|\psi\rangle=\frac1{\sqrt N}\sum_{i=1}^N e^{i\theta_i}|i\rangle
$$

이면

$$
\mu_\psi(i)=\frac1N
$$

이다.

**증명.**

모든 후보의 계수 크기가 같다. B3에 의해 모든 $\mu_\psi(i)$가 같은 값 $p$다. B1에 의해

$$
Np=1
$$

이므로 $p=1/N$이다. $\square$

## 3. 유리 진폭 정리

**정리 3.1**  
$|c_i|^2=n_i/N$이고 $\sum_i n_i=N$인 자연수 $n_i$가 존재하면

$$
\mu_\psi(i)=|c_i|^2
$$

이다.

**증명.**

후보 $i$를 $n_i$개의 미세분지

$$
(i,1),\dots,(i,n_i)
$$

로 refinement한다. 총 미세분지 수는 $N$이다. refinement된 상태를 동일 진폭 상태로 표현하면 각 미세분지는 정리 2.1에 의해 prior $1/N$을 가진다.

B2와 B4에 의해 원래 후보 $i$의 prior는 미세분지 prior의 합이다.

$$
\mu_\psi(i)
=
\sum_{k=1}^{n_i}\frac1N
=
\frac{n_i}{N}
=
|c_i|^2
$$

이다. $\square$

## 4. 일반 진폭

**정리 4.1**  
B0-B5가 성립하면 유한 basis 측정에서

$$
\mu_\psi(i)=|c_i|^2
$$

이다.

**증명.**

임의의 $|c_i|^2$ 는 유리수열 $n_{i,k}/N_k$로 근사할 수 있다. 각 근사 상태에 대해서는 정리 3.1이 성립한다. B5의 연속성으로 극한을 취하면

$$
\mu_\psi(i)=\lim_k\frac{n_{i,k}}{N_k}=|c_i|^2
$$

이다. $\square$

## 5. PreEq와의 결합

Born prior가 닫히면 06장의 초기 모호함은

$$
\mu_0(i)=|c_i|^2
$$

로 고정된다.

그 뒤 측정 조건은 PreEq 재가중이다.

$$
\mu_\beta(i)
=
\frac{e^{-\beta E_{\mathrm{meas}}(i)}|c_i|^2}
{\sum_j e^{-\beta E_{\mathrm{meas}}(j)}|c_j|^2}
$$

따라서 역할이 나뉜다.

| 층 | 담당 |
|---|---|
| Born prior | 측정 전 후보 질량 $\mu_0$ |
| PreEq dynamics | 조건 에너지 아래 manifest 농축 |
| CE bridge | $E_{\mathrm{meas}}$와 접힘/장치 상호작용의 식별 |

## 6. 반증 조건

이 문서의 유도는 다음 중 하나가 깨지면 무너진다.

| 깨지는 공리 | 결과 |
|---|---|
| B2 가법성 실패 | coarse-grained outcome 확률이 합으로 닫히지 않음 |
| B3 대칭성 실패 | 같은 크기 계수 분지가 다른 prior를 가짐 |
| B4 refinement 실패 | 미세분지 표현에 따라 prior가 바뀜 |
| B5 연속성 실패 | 유리 진폭에서 일반 진폭으로 확장 불가 |

## 7. 코드 검증

finite branch 수준의 닫힌 부분은 `reality_stone.clarus.pre_eq`에 구현했다.

| 함수 | 의미 |
|---|---|
| `born_prior(amplitudes)` | \(\lvert c_i\rvert^2\) 정규화 prior |
| `refined_branch_prior(counts)` | 동일 진폭 미세분지 count에서 coarse prior |

회귀검사:

```powershell
python -m pytest tests\test_pre_eq.py -q
```

검증되는 항목:

1. 유리 branch count \((n_i/N)\)와 \(|c_i|^2\) prior가 일치한다.
2. 전체 위상 \(e^{i\theta}\)를 곱해도 prior는 변하지 않는다.

이 코드는 B0-B4의 finite branch 결과만 검산한다. Hilbert space 전체 측정장치 모델이나 Gleason류 일반 정리는 여전히 이 문서 밖의 bridge다.

## 8. 결론

Born rule은 PreEq가 자동으로 만드는 것이 아니다. 그러나 Born prior를 위 공리들로 닫으면, PreEq는 그 prior가 측정 조건 아래 어떻게 하나의 manifest outcome으로 농축되는지를 설명한다.
