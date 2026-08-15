# 우주론 P0·미완성 bridge의 대안 경로

Status: COMPLETE

이 문서는 [11-math.md](11-math.md)의 P0와 [미완성] 항목을 억지로
승격하지 않고, 서로 다른 구조로 살릴 수 있는 범위를 정리한다.
어느 경로도 현재 checkout의 [예측] 승격을 뜻하지 않는다.

## 0. 공통 규칙

- 각 후보의 새 물리 공리는 최대 하나다.
- 외부 경계조건과 fitted degree of freedom(dof)을 분리해 센다.
- 이미 본 관측 목표를 보고 만든 식은 target-aware로 표시한다.
- 같은 항등식에서 나온 두 표현은 교차예측 두 개로 세지 않는다.
- 후보군을 여러 개 본 경우 가장 좋은 하나만 보고하지 않고
  look-elsewhere 집합 전체를 동결한다.
- kill test를 통과하기 전에는 [미완성] 또는 [경험식]을 유지한다.

## 1. 후보 A — 정확한 고정점 코어로 축소

### 목표와 구조

물리 사상이 없는 가장 좁은 수학 코어만 남긴다.

$$
D>1,\qquad q=e^{-D(1-q)},\qquad
q_{\rm ext}\in(0,1/D).
$$

$q=\Omega_b$ 동일시와 DM/DE 분할을 코어에서 제거한다. 우주론 계산은
사용자가 제공한 $\Omega_i,H_0,r_d,\sigma_{8,0}$에 대한 일반
FLRW/CPL 계산기로만 둔다.

### 단계별 지위

1. 고정점 존재·유일성: [정리].
2. 선택한 $D$의 $q$: [산출].
3. $\Omega_i,H_0,r_d,\sigma_{8,0}$ 입력: [공리] 경계조건.
4. 거리·나이·GR 성장: [산출].
5. 자연이 CE 밀도 사상을 따른다는 주장: 이 경로에서는 삭제.

### dof와 선택 효과

- 새 물리 공리: 0.
- 고정점 theorem의 fitted continuous dof: 0.
- empirical selection dof: 0.
- FLRW 계산기의 외부 입력: 최소
  $\Omega_m,\Omega_\Lambda,H_0,r_d,\sigma_{8,0}$; 이들은 예측 dof가
  아니라 사용자가 정한 경계조건이다.
- target-aware: 아니오. 정리는 관측 목표를 쓰지 않는다.
- look-elsewhere: 고정점 가지 선택은 “최소 안정 가지”라는 수학 조건으로
  고정하면 0. 물리 밀도 branch는 애초에 세지 않는다.

### 교차예측과 kill test

- 교차예측: $q(D)$의 단조성, $Dq<1$, $q<1/D$, analytic
  Lambert-$W$ 표현과 수치 solver의 일치.
- empirical 교차예측: 없음.
- kill test:
  1. $D>1$에서 추가 양의 근을 하나라도 찾음.
  2. 독립 bracket solver와 구현이 고정 tolerance를 넘김.
  3. FLRW 계산기가 동일 외부 입력의 표준 analytic limit를 재현하지 못함.

### salvage 범위

가장 강한 완결성을 가진 1순위다. 우주 성분의 기원이나 $H_0$ tension은
살리지 않지만, 거짓 독립 예측을 만들지 않고 고정점 정리와 조건부
cosmology utility를 완전히 보존한다.

## 2. 후보 B — frozen future-BAO 경계조건 검정

### 목표와 구조

$q\mapsto\Omega_b$와 분할을 유도라고 부르지 않고, 하나의 동결된
boundary ansatz로 취급한다. 이미 존재하는 future holdout manifest를
실제 미개봉 release에 배정한 뒤 full covariance로만 평가한다.

새 공리가 필요하다면 정확히 하나다.

> [공리] normalized constants branch
> $(\Omega_b,\Omega_{\rm DM},\Omega_\Lambda)$를 오늘의 평탄 FLRW
> 경계조건으로 사용한다.

### 단계별 지위

1. 밀도 사상·분할: [공리], 유도 아님.
2. external-$r_d$와 EH-hybrid forward models: [산출].
3. 현재 DESI DR2 결과: [경험식], 이미 본 자료.
4. 미배정 holdout: [미완성].
5. release 배정·hash·covariance 동결 후 결과: 그때만 [예측] 후보.

### dof와 선택 효과

- 새 물리 공리: 1.
- holdout에서 fit하는 continuous dof: 후보당 0이어야 한다.
- 외부 고정 입력: $H_0,r_d,\sigma_{8,0},w_0,w_a$ 및 nuisance 규칙.
  값과 출처를 release 열기 전에 hash로 동결한다.
- discrete model dof: 현재 최소 2
  (external-$r_d$, EH-hybrid). 둘 다 결과를 보고하며 사후 선택하지 않는다.
- target-aware:
  - 현재 DR2에 대해서는 예. 후보와 constants가 관측 후 설계됐다.
  - 실제 unopened future release에 대해서는 assignment 이전에 완전히
    동결할 경우 아니오.
- look-elsewhere: 두 후보와 모든 사전 exploratory branch가 가족이다.
  primary endpoint를 하나만 두고 family-wise correction 또는 둘 모두의
  고정 kill rule을 사용한다.

### 교차예측과 kill test

- 교차예측: 여러 redshift bin의 $D_M/r_d$, $D_H/r_d$, $D_V/r_d$를
  한 background로 동시에 계산한다. growth를 넣으려면
  $f\sigma_8$ endpoint와 covariance도 사전에 별도 동결해야 한다.
- 중복이 아닌 이유: 서로 다른 redshift와 transverse/radial observable은
  같은 scalar identity의 단순 재표현은 아니지만 covariance 상관을
  반드시 포함한다.
- kill test:
  1. 어느 등록 후보든 full-covariance fixed-model
     $p<0.0027$이면 그 후보를 기각.
  2. release 이후 parameter, covariance, bin, candidate choice를 바꾸면
     전체 run을 invalid 처리.
  3. full covariance 또는 raw provenance가 없으면 평가하지 않음.

### salvage 범위

관측 검증으로 가는 가장 짧은 2순위다. 성공하더라도 “고정점에서
성분비가 유도됐다”가 아니라 “사전 동결한 CE boundary ansatz가 새
자료와 살아남았다”까지만 말할 수 있다.

## 3. 후보 C — covariant scalar action에서 DM/DE를 다시 시작

### 목표와 구조

비율 projector를 버리고 action-level stress tensor와 perturbation을
출발점으로 삼는다. 새 물리 공리는 하나다.

> [공리] 명시한 하나의 covariant, $Z_2$-symmetric canonical scalar
> action이 dark sector의 유효 작용이다.

예를 들어 $V_0$와 massive oscillating component/portal을 같은 action에
명시하되, $q$를 밀도분율로 동일시하지 않는다.

### 단계별 지위

1. 선택한 action과 frame: [공리].
2. $T_{\mu\nu}$, $w=-1$ constant limit, perturbation equation: [정리].
3. relic abundance·배경·성장·scattering: [산출].
4. 관측 fit: [경험식].
5. radiative stability와 UV completion: [미완성].

### dof와 선택 효과

- 새 물리 공리: 1.
- 최소 continuous physical dof: 적어도 4
  ($m_\phi$, portal/self coupling 하나, 초기 또는 reheating abundance,
  $V_0$). 실제 action에 항을 더하면 그대로 증가시킨다.
- fitted nuisance: 실험·천체 nuisance를 별도 센다.
- target-aware:
  - covariant scalar 구조 자체는 기존 문서에 있으므로 아니오.
  - 현재 density ratios와 $S_8$을 맞추도록 parameter를 고르면 예.
- look-elsewhere: potential, portal, initial-condition branch의 개수를
  명시하고 evidence 또는 동일 penalty로 비교한다.

### 교차예측과 kill test

- 교차예측:
  1. 동일 parameter로 $H(a)$와 $f\sigma_8$.
  2. relic abundance와 scalar sound speed/free-streaming.
  3. portal이 있으면 laboratory/direct-detection bound.
  4. $V_0$의 loop correction과 vacuum stability.
- kill test:
  1. ghost/gradient instability 또는 superluminal ill-posed branch.
  2. 한 parameter region에서 배경·성장·relic·laboratory bound를
     동시에 만족할 수 없음.
  3. 필요한 $V_0$가 radiative correction 아래 기술적으로 유지되지 않음.

### salvage 범위

장기적으로 가장 물리적인 3순위다. 대가로 최소 4개 continuous dof가
생기며, 현재의 “무매개변수 성분비 예측”은 전혀 보존되지 않는다.
대신 진짜 독립적인 교차검정이 가능해진다.

## 4. 후보 D — Mukhanov–Sasaki 기반 primordial spectrum

### 목표와 구조

다섯 projector 중 관측에 가까운 식을 고르는 방식을 폐기하고, 하나의
inflation action에서 mode equation과 normalization을 계산한다.
새 공리는 하나다.

> [공리] 문서에 명시된 단일 Einstein-frame canonical
> Starobinsky형 potential이 inflationary action이다.

### 단계별 지위

1. action·vacuum prescription: [공리].
2. Mukhanov–Sasaki equation과 power-spectrum formula: [정리].
3. 주어진 $(V_0,N_\star)$의 $(A_s,n_s,r,\alpha_s)$: [산출].
4. reheating map $N_\star(k_\star)$: 완성 전 [미완성].
5. 자료 likelihood: [경험식], blind holdout일 때만 [예측] 후보.

### dof와 선택 효과

- 새 물리 공리: 1.
- 최소 continuous dof: 2
  ($V_0$, $N_\star$ 또는 reheating parameter).
- $A_s$로 $V_0$를 보정하면 예측 dof가 하나 소모되고, $A_s$는
  교차예측에서 제외된다. 남는 주요 dof는 reheating/$N_\star$다.
- target-aware: 현재 projected readout family는 예. 관측 $A_s$를 본 뒤
  최소 다섯 source/projector를 비교했다.
- look-elsewhere: 기존 다섯 projector와 검토한 potential family 전체를
  기록한다. action route의 결과와 projector 결과를 한 후보처럼
  소급 합치지 않는다.

### 교차예측과 kill test

- 교차예측: 같은 action과 reheating history가
  $(n_s,r,\alpha_s)$ 및 scale dependence를 동시에 정한다.
- $A_s$로 $V_0$를 fit했다면 $n_s,r,\alpha_s$만 독립 교차예측이다.
- kill test:
  1. 허용되는 reheating 전 범위에서 joint CMB likelihood를 통과하지 못함.
  2. 필요한 $V_0$와 slow-roll/perturbative control이 양립하지 않음.
  3. 사후 projector 변경이 발생하면 preregistered prediction 무효.

### salvage 범위

현재 0.166$\sigma$ effective-geometry 숫자는 버리지만, primordial
section을 표준 action-level 계산으로 살린다. 정확도는 숫자 근접도가
아니라 action, normalization, reheating bridge의 폐쇄 여부로 판정한다.

## 5. 후보 E — radiation·baryon-aware $H_0$ likelihood

### 목표와 구조

현재 toy 음향각 matching을 폐기하고 radiation, baryon loading,
recombination, drag epoch 및 perturbation을 하나의 forward likelihood에
넣는다. 새 공리는 하나만 허용한다.

> [공리] 명시한 $\epsilon(a)$ flow law가 표준 Einstein–Boltzmann
> 배경/섭동 방정식의 어느 항을 어떻게 수정하는지 covariantly 고정한다.

$\xi=\pi^2/2$, $\Delta\epsilon_0=-\delta/\pi$를 고정값으로 유지하려면
관측을 보고 선택했다는 사실을 그대로 기록한다.

### 단계별 지위

1. 표준 radiation+baryon+recombination sector: [공리] 입력과 [정리].
2. CE flow coupling law: 새 [공리].
3. Boltzmann observables: [산출].
4. 현재 $+5.6$ 근접값: 폐기된 [경험식].
5. frozen joint likelihood 결과: [예측] 후보.

### dof와 선택 효과

- 새 물리 공리: 1.
- flow amplitude와 stiffness를 truly fixed로 두면 continuous fitted dof 0,
  그러나 이미 목표를 본 고정 선택 두 개
  $(\Delta\epsilon_0,\xi)$는 discrete/design dof로 기록한다.
- 둘 중 하나라도 fit하면 fitted continuous dof가 1 또는 2로 증가한다.
- 표준 nuisance: $\omega_bh^2,\omega_ch^2,\tau,A_s,n_s$와 calibration을
  고정 또는 likelihood nuisance로 정확히 센다.
- target-aware: 예. $-\delta/\pi$와 $\pi^2/2$ branch가 알려진
  $H_0$ 차이 근처에서 제시됐다.
- look-elsewhere: closed-form, anchored, mirror 및 검토한 부호/계수
  branch 전체. mirror를 출력 후 제외하는 것도 사전 selection rule이
  아니면 penalty 대상이다.

### 교차예측과 kill test

- 교차예측:
  1. $\theta_\star$뿐 아니라 CMB peak heights/damping/lensing.
  2. BAO $r_d$ ratios와 BBN baryon abundance.
  3. late-time SN/chronometer $H(z)$와 growth.
- kill test:
  1. corrected Ricci와 baryon-aware recombination을 넣은 뒤
     $\Delta H_0$ 효과가 사라짐.
  2. joint CMB+BAO+BBN likelihood가 frozen threshold에서 기각.
  3. $H_0$ 개선이 peak-height/damping/lensing 중 하나를 악화시켜
     전체 likelihood 이득이 없음.

### salvage 범위

이 경로만 현재 Hubble-tension 아이디어를 물리적으로 시험할 수 있다.
현재 스크립트의 숫자를 보존하는 것이 목적이 아니라, 그 숫자가
radiation과 baryon physics를 통과하는지를 반증 가능하게 만드는 경로다.

## 6. 후보 F — phase-area entropy를 단일 새 공리로 동결

### 목표와 구조

holographic gate를 “우주상수 두 번째 예측”으로 세지 않고, entropy가
하나의 절대 Hubble scale을 정한다는 단일 가설로 축소한다.

> [공리] $\log S_{\rm dS}=(\pi^2/2)N_e-\pi\delta\sigma$를
> convention과 모든 integer 선택까지 포함해 사전에 동결한다.

### 단계별 지위

1. Friedmann–entropy 관계: [정리].
2. phase-area law: [공리].
3. 그로부터 환산한 $H_0$: [산출].
4. $\rho_\Lambda$ 환산: 같은 산출의 재표현, 별도 예측 아님.
5. 관측 전 동결되지 않은 현재 비교: [경험식].

### dof와 선택 효과

- 새 물리 공리: 1.
- 식을 완전히 동결하면 fitted continuous dof: 0.
- 외부/구조 입력: $\alpha_s,d=3,N_g=12,\Omega_\Lambda$.
  $\Omega_\Lambda$는 $\rho_\Lambda$ 환산에만 필요하고 $H_0$에는 불필요하다.
- discrete design dof: $d,N_g$, leading coefficient, correction의
  포함 여부와 부호. 후보 수가 문서화되지 않아 현재 유효
  look-elsewhere penalty는 계산 불가다.
- target-aware: 예. 67 km/s/Mpc와 2.24 meV 목표를 이미 본 상태에서
  gate가 작성됐다.

### 교차예측과 kill test

- 독립 교차예측: frozen law가 주는 $H_0$ 하나.
- 독립 교차예측이 아닌 것:
  $\rho_\Lambda=\Omega_\Lambda\rho_{\rm crit}(H_0)$,
  $10^{-122}$ hierarchy, $\rho_\Lambda^{1/4}$는 같은 항등식의
  단위·거듭제곱 재표현이다.
- kill test:
  1. 새 독립 $H_0$ 자료가 사전 선언한 이론+관측 불확실도 밖.
  2. 결과 후 $N_g$, correction, coefficient, $\Omega_\Lambda$를 변경.
  3. phase-area law를 covariant entropy calculation에서 유도할 수 없고
     공리로도 유지하지 않기로 결정.

### salvage 범위

가장 약한 6순위다. 계산은 단순하고 무차원성이 좋지만 target-aware이고
교차예측이 하나뿐이다. 현재의 “우주상수 문제를 독립적으로 해결”이라는
말은 살릴 수 없고, 명시적인 one-scale entropy ansatz만 남는다.

## 7. 후보 비교

| 후보 | 구조 | 새 공리 | fitted continuous dof | target-aware | 독립 교차예측 | 현 지위 |
|---|---|---:|---:|---|---|---|
| A | 정리 코어로 축소 | 0 | 0 | 아니오 | 수학적 관계만 | [정리]/[산출] |
| B | frozen future BAO | 1 | 후보당 0 | DR2 예, future는 동결 시 아니오 | 다중 BAO bin | [미완성] |
| C | covariant scalar action | 1 | 최소 4 | parameter 선택은 예 | 배경·성장·relic·lab | [미완성] |
| D | Mukhanov–Sasaki action | 1 | 최소 2 | 예 | $n_s,r,\alpha_s$ | [미완성] |
| E | Einstein–Boltzmann $H_0$ | 1 | 0–2 + nuisance | 예 | CMB+BAO+BBN+growth | [미완성] |
| F | frozen phase-area entropy | 1 | 0 | 예 | $H_0$ 하나 | [경험식]/[미완성] |

권고 순서는 A → B → C/D/E → F다. A는 즉시 완결 가능한 수학 코어,
B는 가장 가까운 진짜 blind 관측 검정, C/D/E는 action/likelihood를
새로 닫아야 하는 연구 경로다. F는 같은 $H_0$–$\rho_\Lambda$ 항등식을
교차예측으로 중복 계산하지 않는다는 조건에서만 유지할 가치가 있다.

## 8. 즉시 적용할 kill-list

소스 수정 여부와 별개로, 다음 주장은 후속 문서에서 증거로 재사용하면 안 된다.

1. 현재 hubble_tension.py의 $\Delta H_0=+5.6$.
2. holographic gate의 “zero free parameters” 및
   “$\rho_\Lambda$ 독립 재현”.
3. effective-geometry $A_s$의 0.166$\sigma$를 사전 예측으로 부르는 것.
4. constants 밀도 세 값을 ce_prediction으로 부르는 것.
5. exploratory DESI DR2 또는 unassigned holdout을 예측 검증으로 부르는 것.
6. 테스트 PASS를 C2/C4 물리 사상의 증명으로 쓰는 것.

