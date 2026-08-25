# 0차원 공간 접힘의 단일 환경 기억장 연구 계약

Status: COMPLETE

PREDECESSOR:

- `_workspace/ce/zero-dimensional-overlap-bootstrap-20260825`
- `_workspace/ce/dark-sector-observational-census-derivation-20260825`

MODE: full

## 1. 연구 질문

외부 0차원이 현재 차원에 반복적으로 출력을 주는 기존 모형을 보수하지 않는다.
대신 현재 시공간 $M$ 안에서 공간적으로 0차원인 접힘이 생성될 때 그 흔적이
환경에 남아 하나의 공통 기억장 $\psi$를 만들고, 그 장이 미래의 다른 접힘
생성률을 높이는 자기흥분 구조를 처음부터 구성한다.

핵심 인과 구조는

$$
\text{0D fold deposit}
\longrightarrow \psi\text{ in the environment}
\longrightarrow \text{later 0D fold deposit}
$$

이다. 미래 사건은 과거 사건에 직접 연결되지 않고 동일한 장 $\psi$를 통해서만
연결한다.

## 2. 타입과 정의역

1. $M$은 시간지향을 가진 globally hyperbolic $3+1$차원 시공간이다. 평탄한
   국소 한계와 homogeneous cosmological frame에서 먼저 계산하고, 인과성은
   $J^-(x)$로 표시한다.
2. 공간 절편에서 한 접힘은 점으로 이상화한다. 생성 시각 이후에도 결함 자체가
   남으면 그 spacetime support는 0D가 아니라 worldline이다. strict spacetime-0D
   사건인 경우에는 사건 자체가 아니라 $\psi$의 retarded tail만 남는다고 구분한다.
3. 접힘 기록은 locally finite counting measure
   $N(d^4x)=\sum_a\delta_{x_a}(d^4x)$로 쓴다.
4. $\psi(t,\mathbf x)$는 하나뿐인 환경 기억장이다. 기본 차원은 공간 수밀도
   $[\psi]=L^{-3}$로 고정한다.
5. $K_{\ell,R}(x,y)$는 $y\notin J^-(x)$이면 0인 retarded kernel이다. 점원천의
   coincidence singularity를 피하기 위해 먼저 유한 smearing scale $\ell>0$에서
   정의하고, strict-point limit은 별도 극한으로 감사한다.
6. 조건부 접힘 강도 $\lambda(x\mid\mathcal F_{x^-})$의 차원은
   $T^{-1}L^{-3}$이다.

## 3. 사전 고정한 후보 경로

### R1. 선형 단일 기억장

각 접힘이 장에 크기 $A$의 흔적을 남긴다고 두고

$$
\psi(x)=A\int_{J^-(x)}K_{\ell,R}(x,y)N(d^4y),
$$

$$
\lambda(x)=\lambda_0(x)+\beta\psi(x)
$$

를 검사한다. $A$는 event당 무차원 deposit, $[\beta]=T^{-1}$이다. kernel은

$$
\int_{J^+(y)}K_{\ell,R}(x,y)d^4x=\tau
$$

로 정규화하며 $\tau$는 기억시간이다. 예상 재생산수 후보는

$$
\mathcal R=A\beta\tau
$$

다.

### R2. 포화된 단일 기억장

선형 폭주를 피하는 최소 비선형 후보는

$$
\lambda(\psi)=\lambda_0+
\frac{\beta\psi}{1+\psi/\psi_s}
$$

다. homogeneous closure에서

$$
\dot\psi=-\frac{\psi}{\tau}+A\lambda(\psi)
$$

를 유도하고, 양의 고정점의 존재·유일성·안정성과 stochastic extinction의
차이를 검사한다.

### R3. 닫힌 작용 기반 단일장

별도 점과정이나 reservoir 없이 한 canonical real scalar 자체가 안정한 0D
공간 lump를 만들고 서로를 활성화할 수 있는지

$$
S[\phi]=\int_M d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2-V(\phi)\right]
$$

에서 검사한다. Derrick scaling, point-source self-energy와 보존법칙을 반례
게이트로 사용한다. 실패하면 higher derivative, complex field, gauge/multiplet,
nonlocal 또는 open-system 추가가 정확히 무엇을 바꾸는지 대안으로만 기록한다.

## 4. 고정할 명제와 계산

1. event-deposit과 persistent spatial defect의 차원 구분을 증명한다.
2. R1의 retarded solution, 평균 방정식과 stationary mean을 유도한다.
3. $\mathcal R<1$, $\mathcal R=1$, $\mathcal R>1$에서 선형 모형의 안정성·폭주·
   extinction 지위를 분리한다.
4. R2에서 $\lambda_0=0$일 때 후보 고정점
   $\psi_*=\psi_s(\mathcal R-1)$을 처음부터 유도하고 선형 안정성을 계산한다.
5. mean-field의 양의 고정점이 유한 stochastic realization의 almost-sure
   survival을 뜻하지 않음을 반례 또는 정리로 감사한다.
6. $\ell\to0$ coincidence limit, $K_R(x,x)$와 field energy의 UV 거동을
   계산한다.
7. 한 장만으로 닫힌 Hamiltonian dynamics와 stochastic creation rule을 동시에
   요구할 때 에너지·확률 보존이 가능한지 검사한다.
8. 공간 Fourier--시간 Laplace 선형화에서
   $1-A\beta_{\rm eff}\widetilde K_R(s,\mathbf k)=0$의 pole 조건을 유도한다.
9. 새 식의 exp/log/확률/고정점 인자와 모든 합의 차원을 감사한다.

## 5. 허용 오차와 수치 검사

- analytic identity residual: $10^{-12}$ 이하
- 고정점 residual: $10^{-12}$ 이하
- 유한차분 Jacobian과 해석 Jacobian 상대오차: $10^{-8}$ 이하
- kernel 정규화 residual: $10^{-10}$ 이하
- 수치 예는 단위 선택에 의존하지 않는 $\mathcal R$와 정규화 변수만 사용한다.

## 6. 반증 조건

다음 중 하나가 성립하면 해당 부모 주장을 제거한다.

1. retarded kernel이 미래 support 또는 무한 coincidence 값을 피하지 못한다.
2. 양의 stationary state가 존재해도 선형화 eigenvalue가 양수다.
3. $\mathcal R$가 차원 있는 양이거나 임의 scale 선택에 따라 임계값이 변한다.
4. strict point limit에서 정규화 없이 self-trigger rate 또는 에너지가 발산한다.
5. 한 canonical real scalar의 안정한 정적 0D lump가 Derrick scaling을 통과하지
   못한다.
6. stochastic source를 쓰면서도 별도 reservoir 없이 닫힌 에너지 보존을
   주장해야만 모형이 성립한다.

## 7. 주장 상한

이 run이 닫을 수 있는 최대 주장은 “유한 해상도에서 0D 접힘 deposit을 한 개의
인과적 환경 기억장으로 표현하고, 그 장이 다음 deposit을 자기흥분시키는 최소
유효모형의 존재·임계조건을 유도했다”이다. 양자중력의 미시적 공간 접힘,
표준 양자역학으로부터의 유도, 실제 암흑물질·암흑에너지 동일성, 절대 abundance,
관측 적합은 이 run의 예측이 아니다.

## 8. 경로 선택 규칙

R1--R3를 모두 감사한 뒤 인과성, positivity, UV 유한성, 에너지 장부, 한 장이라는
요구를 가장 적은 추가 공리로 만족하는 경로만 활성 후보로 남긴다. 수치가 기존
$q$ 또는 $\Omega$에 가까운지는 선택 기준으로 사용하지 않는다.

## 9. 의미 교정 장부

초기 R1--R2는 “남는다”를 strict spacetime-0D 사건의 retarded trace가 남는다는
뜻으로 우선 형식화했다. 이후 타입 감사를 통해 사용자의 문장을 더 문자 그대로
만족하는 별도 경로가 필요함을 확인했다.

$$
\text{persistent spatial-0D carrier}
\longrightarrow
\text{one dynamic Volterra field}
\longrightarrow
\text{other carriers' activation}.
$$

이 경로에서 접힘은 각 공간 절편에서 점이지만 시공간에서는 worldline이고,
carrier measure $\mu_F$는 fixed/quenched parameter다. 실제 동역학적 상태는
$\psi$ 하나뿐이다. 이 의미 교정은 관측 데이터나 결과를 본 뒤 계수를 바꾼 것이
아니며, `11-math.md`의 P-series와 `12-routes.md`의 R0으로 별도 표기한다. 기존
event-deposit R1--R2는 “새 접힘이 생성된다”는 다른 의미의 비교 경로로 보존하고,
두 경로의 임계량 $A\rho(W)$와 $A\beta\tau$를 합치지 않는다.

교정 뒤의 주장 상한은 “고정된 공간-0D carrier들이 한 인과적·포화 Volterra
상태장을 통해 서로 활성화되는 최소 유효모형과 그 spectral threshold를
유도했다”까지다. carrier의 생성, 미시 양자 유도, stress-energy, 암흑부문
동일성과 abundance는 여전히 이 상한 밖이다.
