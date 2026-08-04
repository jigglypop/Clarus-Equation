# CE Euclidean connected correlator · spectral 루프

기준일: 2026-08-04  
범위: 실제 표본 provenance → connected subtraction → covariance → 유한
Euclidean positivity → screening energy → spectral 비유일성

> 결론: 현재 저장소에는 CE 연산자의 원시 ensemble과 공분산을 갖춘 실제
> connected 2점함수 자료가 없다. 따라서 CE 단계는 계속
> REGISTERED_SCALE이고 물리 pole 증거는 0/100이다. 이번 루프가 닫은 것은
> 데이터가 들어왔을 때의 fail-closed 계산 절차와, 유한 Euclidean 시간
> 표본만으로는 양의 spectral density도 유일하게 복원되지 않는다는 구성적
> 반례다. 합성 양의-spectrum 대조군이 통과해도 최고 단계는
> EUCLIDEAN_SCREENING_CONTROL이며 Minkowski pole·LSZ·CE field identity는
> 계속 False다.

## 1. 질문을 세 층으로 분리한다

1. raw ensemble에서 connected Euclidean 2점함수를 통계적으로 추정할 수 있는가.
2. 유한 표본이 양의 Laplace spectrum에 필요한 조건과 양립하는가.
3. 그 결과가 고립 실시간 pole, 양의 residue와 안정한 CE 입자를 증명하는가.

1과 2의 통제 절차는 구현할 수 있다. 3은 1과 2만으로 따라오지 않는다.

이 gate의 kernel 범위는 진공·Hermitian bosonic scalar operator·비주기적
시간축의 forward connected correlator다. 유한온도 또는 주기적 Euclidean
시간에서는

\[
C_\beta(\tau)=
\int_0^\infty\frac{d\omega}{2\pi}\rho(\omega)
\frac{\cosh[\omega(\beta/2-\tau)]}{\sinh(\beta\omega/2)}
\]

같은 thermal kernel을 써야 한다. wraparound 자료, \(t=0\) contact term,
fermion 및 tensor reflection factor에 단순 exponential gate를 적용하지 않는다.

## 2. 저장소 provenance 감사

2026-08-04 작업 트리에서 다음을 동시에 만족하는 CE 자료는 찾지 못했다.

- CE action과 operator/field identity
- 같은 ensemble에서 짝지어진 원시 \(O_n(t)\), \(O_n(0)\)
- configuration ID와 chain order
- Euclidean time grid, lattice spacing 또는 물리 단위
- volume, temperature와 boundary condition
- connected subtraction 전 원시값
- jackknife 또는 bootstrap을 재현할 표본
- volume, spacing, operator가 다른 holdout

혼동하기 쉬운 기존 항목은 다음과 같이 격리한다.

| 후보 | 실제 역할 | CE spectral 자료가 아닌 이유 |
|---|---|---|
| clarus/quantum.py의 correlator Fourier 변환 | caller가 공급한 bath correlator의 유한창 수치변환 | CE ensemble을 생성하지 않고 양·음의 실시간 grid와 bath convention을 요구함 |
| TDCOSMO covariance 감사 | 우주론 관측 공분산 | CE field/operator의 2점함수가 아님 |
| RAGTruth ensemble detector | NLP benchmark의 ensemble score | 물리 ensemble이나 Euclidean correlator가 아님 |
| 기존 CE tree two-point 인증서 | 선택적 portal action의 해석적 bare kernel | 측정·시뮬레이션된 connected correlator가 아님 |
| 신경·원시생명 원자료 | 형광, spike 또는 분자 실험 | CE Euclidean action/operator와 KL convention이 없음 |

그러므로 29.64757 MeV는 아직 correlator fit 결과가 아니라
\(m_p\delta^2\)에서 등록된 inverse-correlation scale ansatz다.

## 3. connected estimator와 covariance

독립 configuration 또는 사전 block되어 독립인 bin
\(n=1,\ldots,N\)에서 짝지어진
\(X_{nt}=O_n(t)\), \(Y_n=O_n(0)\)를 받는다고 하자. 원시 표본에서

\[
\widehat C(t)=
\frac{1}{N-1}\sum_{n=1}^N
\bigl(X_{nt}-\bar X_t\bigr)
\bigl(Y_n-\bar Y\bigr)
\]

을 직접 계산한다. 이는 iid paired sample의 unbiased sample covariance다.
caller가 만든 connected=True 같은 표지를 신뢰하지 않는다.

connected subtraction은 모든 표본을 사용하므로 각 시각을 독립으로 취급하면
안 된다. configuration 하나씩 뺀 \(\widehat C^{(-n)}(t)\)로

\[
\widehat{\operatorname{Cov}}_J[C_t,C_s]
=\frac{N-1}{N}\sum_n
\left(C_t^{(-n)}-\bar C_t^{(\cdot)}\right)
\left(C_s^{(-n)}-\bar C_s^{(\cdot)}\right)
\]

을 계산한다. derived array의 finiteness, connected subtraction identity와
covariance의 positive semidefiniteness도 stage 전에 재검산한다.

이는 autocorrelation이 있는 Markov chain의 binning 문제를 자동 해결하지
않는다. 실제 chain에서는 integrated autocorrelation time을 측정하고 block
크기를 사전등록해야 한다.

## 4. 양의 Laplace spectrum의 유한 필요조건

등간격 \(t_n=t_0+n\Delta t\)에서

\[
C_n=\int_0^\infty e^{-E(t_0+n\Delta t)}\,d\mu(E),
\qquad d\mu(E)\ge0
\]

라면 \(x=e^{-E\Delta t}\in[0,1]\)이고 \(e^{-Et_0}\)를 측도에 흡수해

\[
C_n=\int_{[0,1]}x^n\,d\nu(x)
\]

인 Hausdorff moment 수열이 된다.

### 4.1 교대 유한차분

forward difference를 \(\Delta C_n=C_{n+1}-C_n\)라 두면

\[
(-1)^k\Delta^k C_n
=\int x^n(1-x)^k\,d\nu(x)\ge0.
\]

따라서 \(C_n\ge0\), \(C_{n+1}\le C_n\), convexity와 사용 가능한 모든
고차 교대 유한차분 부호가 필요하다.

### 4.2 log-convexity와 effective mass

Cauchy–Schwarz로

\[
C_n^2\le C_{n-1}C_{n+1}.
\]

따라서

\[
m_{\rm eff}(n)
=\frac{1}{\Delta t}\log\frac{C_n}{C_{n+1}}
\]

은 양의 spectrum 혼합에서 \(n\)에 따라 비증가한다. 큰 \(t\)의 plateau가
여러 covariance-aware fit window에서 안정하면 가장 낮은 Euclidean
screening energy의 통제 증거가 된다. 이는 유한-volume Minkowski particle
mass 증명이 아니다.

### 4.3 truncated Hausdorff matrix

\(C_0,\ldots,C_{2r+1}\)이 주어진 홀수 최고차수의 경우

\[
[C_{i+j+1}]_{i,j=0}^{r}\succeq0,
\qquad
[C_{i+j}-C_{i+j+1}]_{i,j=0}^{r}\succeq0.
\]

\(C_0,\ldots,C_{2r}\)이 주어진 짝수 최고차수의 경우

\[
[C_{i+j}]_{i,j=0}^{r}\succeq0,
\qquad
[C_{i+j+1}-C_{i+j+2}]_{i,j=0}^{r-1}\succeq0.
\]

코드는 표본 개수 parity에 따라 이 localizing matrix를 바꾼다. parity를
무시하면 모든 단순 부호 검사를 통과하면서도 양의 measure가 존재하지 않는
거짓 양성이 생긴다.

현재 finite-difference와 Hankel 판정은 central estimate의 수치 진단이다.
실제 noisy ensemble의 통계적 기각을 말하려면 각 preblocked
jackknife/bootstrap replica에서 difference와 최소 eigenvalue를 다시 계산하고
다중검정 기준을 사전등록해야 한다.

유한 2점 조건을 통과한 것은 finite necessary diagnostic 또는 truncated
positive measure와의 양립성이다. 일반 interacting QFT의
Osterwalder–Schrader reflection positivity는 모든 positive-time test
functional과 Schwinger 함수 계층에 대한 조건이다. 2점함수 몇 개의 통과를
full reflection positivity 증명으로 부르지 않는다. 양의 spectral
representation과 reflection positivity의 관계는
[Usui](https://arxiv.org/abs/1201.3415)를 대조했다.

## 5. 유한 spectral reconstruction의 구성적 비유일성

\(N_t\)개 시간과 \(N_E\)개 energy bin을 택하면

\[
C_i=\sum_{j=1}^{N_E}K_{ij}\rho_j,
\qquad
K_{ij}=e^{-E_jt_i},
\qquad \rho_j\ge0.
\]

알려진 총 spectral weight까지 보존하기 위해 sampling matrix에 정규화
행을 붙인다.

\[
\widetilde K=
\begin{bmatrix}
1&\cdots&1\\
&K&
\end{bmatrix}.
\]

\(N_E>\operatorname{rank}\widetilde K\)이면
\(\widetilde Kv=0\)인 \(v\ne0\)가 있다. 따라서 \(Kv=0\)인 동시에
\(\sum_jv_j=0\)이다. 성분이 모두 양수인 base \(\rho_0\)와 충분히 작은

\[
0<\epsilon<
\min_{v_j\ne0}\frac{(\rho_0)_j}{|v_j|}
\]

을 택하면

\[
\rho_+=\rho_0+\epsilon v,
\qquad
\rho_-=\rho_0-\epsilon v
\]

는 서로 다르고 둘 다 비음수지만

\[
K\rho_+=K\rho_-=K\rho_0,
\qquad
\sum_j(\rho_+)_j=\sum_j(\rho_-)_j.
\]

다. 인증서는 augmented SVD null vector를 실제 구성하고 \(Kv\), \(\sum_jv_j\)
잔차, 두 spectrum의 비음수성·거리·총 weight와 두 correlator의 차이를 다시
계산한다. 이는 유한 kernel만으로 spectrum 유일성을 주장하는 일반 논리를
반박하는 synthetic counterexample다.
현재 관측 correlator에 맞는 두 spectrum을 찾았다는 뜻은 아니다.

실제 noisy 자료에서는 pointwise spectrum보다 사전등록한 resolution kernel에
대한 smeared spectral quantity의 범위를 구하는 편이 방어 가능하다.
inverse Laplace bounds는
[Lawrence](https://arxiv.org/abs/2408.11766), Euclidean 2점함수의
Backus–Gilbert 접근은
[Harris–Meyer–Robaina](https://arxiv.org/abs/1611.02499)을 대조했다.

## 6. 단계 잠금

    REGISTERED_SCALE
      → CONNECTED_CORRELATOR_CONTROL
      → POSITIVE_SPECTRUM_NECESSARY_CONTROL
      → EUCLIDEAN_SCREENING_CONTROL

여기까지 어느 단계도 다음 값을 True로 만들지 않는다.

    unique pointwise spectral density       False
    full reflection positivity              False
    isolated Minkowski pole                 False
    positive invariant pole residue         False
    infinite-volume stable particle         False
    LSZ asymptotic state                     False
    CE field identity                        False
    physical SM production rate              False

real-time 응답이나 decay rate를 Euclidean 자료에서 복원하는 문제가
ill-posed하다는 점은 mock lattice 자료를 비교한
[Huang–Liang](https://arxiv.org/abs/2309.11114)의 결과와도 일치한다.

## 7. 현재 점수

점수는 물리적 진실의 확률이 아니라 각 gate의 증거 완성도다.

| 항목 | 점수 | 판정 |
|---|---:|---|
| 원시 CE ensemble provenance | 0/100 | 자료 없음 |
| connected subtraction·jackknife 절차 | 95/100 | 독립 또는 preblocked 합성 통제; 실제 chain holdout 없음 |
| 유한 Euclidean positivity 진단 | 90/100 | central-value gate; replica 통계검정과 full OS positivity가 아님 |
| 유한 표본에서 spectrum 유일성 | 0/100 | 두 비음수 spectrum nullspace 반례로 기각 |
| 합성 screening-energy 대조군 | 95/100 | 알려진 spectrum의 방법 통제에 한정 |
| 실제 CE screening energy | 0/100 | 실제 correlator 없음 |
| 실제 CE spectral density | 0/100 | inverse 자료와 regularization holdout 없음 |
| physical CE pole·LSZ | 0/100 | analytic continuation·volume flow·field identity 없음 |

방법 구현 점수와 현재 CE 물리 증거 점수를 평균내지 않는다. 전자는 합성
대조군에서 높지만 후자는 여전히 0이다.

## 8. 실제 자료의 최소 계약

다음 루프를 물리 증거로 올리려면 하나의 machine-readable manifest와 원시
배열 묶음이 최소한 다음을 가져야 한다.

    action_definition_sha256
    operator_definition_sha256
    field_id / quantum_numbers / projection
    ensemble_id / configuration_ids / chain_order
    times / time_unit / lattice_spacing
    volume / boundary_conditions / temperature
    operator_at_t[configuration, time]
    operator_at_zero[configuration]
    binning_or_autocorrelation_protocol
    predeclared_fit_windows
    independent volume/spacing/operator holdouts

현재 PairedEuclideanEnsemble은 time과 원시 두 배열을 받는 계산 scaffold다.
action/operator hash, configuration ID·chain order, spacing·volume·boundary
condition·source hash를 아직 담지 않으므로 provenance certificate가 아니다.
실제 자료를 연결할 때 위 manifest를 먼저 고정하고 배열 hash와 configuration
ID를 결합해야 한다.

우선순위는 다음과 같다.

1. 동일 action hash에서 원시 paired ensemble을 생성하거나 공개 원자료를 연결한다.
2. chain autocorrelation을 측정하고 block 크기를 고정한다.
3. blind fit 전에 covariance rank와 positivity 위반을 본다.
4. 여러 operator의 correlator matrix와 variational holdout을 추가한다.
5. spacing·volume extrapolation 뒤에 screening level을 pole 후보와 비교한다.
6. analytic structure, pole/cut, residue와 LSZ는 독립 gate로 둔다.

## 9. 재현

핵심 구현:

- reality_stone/python/reality_stone/clarus/euclidean_correlator_certificate.py
- tests/test_euclidean_correlator_certificate.py
- examples/physics/ce_euclidean_correlator_gate.py

합성 method control의 실행값:

    current CE stage                         REGISTERED_SCALE
    current CE first blocker                 raw paired O(t), O(0) ensemble is absent
    synthetic stage                          EUCLIDEAN_SCREENING_CONTROL
    synthetic mean screening mass            29.647570000 MeV
    registered mass relative error            0
    normalization-augmented nullity           8
    correlator pair relative residual         1.767588592e-16
    total spectral-weight pair residual       0
    unique spectral density                   False
    Minkowski pole / LSZ / CE identity        False / False / False

2026-08-04 집중 검증:

- 전용 회귀: 21 passed
- 관련 pole/action 통합 회귀: 92 passed
- 전체 회귀: 1350 passed, 13 skipped, 0 failed
- 전체 회귀의 warning 2건은 기존 PyTorch sparse CSR beta/invariant 안내다.
- 작업 트리에서 이미 삭제된 fixture를 직접 요구하는 테스트 5개만 명시적으로
  제외했다: local_memory, origin_life_branching, origin_life_coupled,
  q0_manifest, neural_tree_algorithm_census.
- Ruff check: 통과
- Ruff format check와 full certificate JSON 직렬화: 통과
- 전체 suite용 전용 basetemp는 경로를 검증한 뒤 생성했고, 통과 후 그
  생성 디렉터리만 제거했다.

재현 명령:

    uv --cache-dir .uv-cache run python examples/physics/ce_euclidean_correlator_gate.py

    uv --cache-dir .uv-cache run --extra dev python -m pytest tests/test_euclidean_correlator_certificate.py -q
