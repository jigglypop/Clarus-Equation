# 시공간 응답 마스크와 spacelike marginal 루프

작성일: 2026-08-05  
코드: `resonant_spatiotemporal_mask.py`, `spacelike_marginal_gate.py`

## 0. 결론

종교 문헌은 물리 증거로 사용하지 않는다. 이번 루프가 가져온 것은 다음 네 개의
제어 인덱스뿐이다.

\[
(g,R,S,\tau)
=
(\text{고정된 결합 보정},\text{공진 프로필},
\text{물리적 selector와 공간 support},\text{유한 작동시간}).
\]

핵심 돌파는 `observer`를 새 물리변수로 두지 않고, 측정 가능한 probe 내부상태와
국소 결합으로 바꾼 것이다. 이 번역을 하면 가능한 결과는 정확히 세 가지다.

1. 선택된 probe만 변하면 `probe-selective dressing`이다.
2. 미래 광원뿔 안의 독립 probe들이 같은 retarded kernel을 재구성하면
   `public causal response candidate`다.
3. spacelike probe의 공개 분포가 선택에 따라 변하면 먼저 누설·시간동기 오류 또는
   local-QFT 가정의 실패다.

네 번째 경우인 `observer에게만 존재하는 public reality`는 local Lorentz-covariant
QFT의 결정론적 국소 연산으로 표현되지 않는다.

현재 실제 CE 자료가 없으므로 이번 코드가 도달할 수 있는 최대 단계는 합성 control의
`CONDITIONAL_DECLARED_BLOCK_SPATIOTEMPORAL_RESPONSE_MASK`다. 이 이름은 독립성이
검증된 것이 아니라 block ID와 iid/Gaussian 선언에 조건부라는 점을 명시한다.
또한 기존 이름에서 `CAUSAL`을 제거했다.
caller가 지정한 `prearrival_mask`만으로 광원뿔을 증명할 수 없기 때문이다. 인과성은
별도의 randomized spacelike-marginal gate에서만 검사한다. public scaffold, 새 물질,
CE actuator, physical pole, 재규격화 응력은 모두 `False`로 잠근다.

### 0.1 2026-08-05 v3 합성 회귀 결과

고정 seed의 protocol-validation fixture에서만 다음 수치가 나온다.

| 항목 | 결과 | 사전 기준 |
|---|---:|---:|
| local selector TV LCB | 0.896689 | (>0.8) |
| max spacelike TV UCB | 0.003311 | (\le0.02) |
| global amplitude | 1.999270429 | synthetic truth (2.0) |
| Student-(t) Bonferroni multiplier | 3.020753840 | FWER (0.05), 18 comparisons |
| max crossed-heldout residual UCB | 0.006658500 | (\le0.05) |

6개 관련 모듈의 focused suite는 115 tests를 통과했다. 이 숫자는 코드와 protocol의
회귀 결과이지 CE 실험 정확도가 아니다. 실제 CE raw tensor가 0개이므로 new matter,
public scaffold와 causal field에 대한 경험적 정확도는 산정 불가다.

v1을 깨던 certificate boolean 변조, 1-cell 포화 GLS, protected zero-variance,
exact-zero covariance, posthoc minimum-(N), duplicated/permuted block ID와 고분산
exact-mean 반례는 v3에서 모두 실패하도록 회귀 테스트로 고정했다. 여기에 극단
Student-\(t\) tail, 공백 SHA/ID, bool--int·string--Enum type confusion과 exact paired-row
복제 반례도 추가했다. float64 roundoff를 rank로 오인하지 않도록 rank tolerance는
\([10^{-12},10^{-2}]\), condition cap은 \(\le10^{12}\)이면서 rank tolerance의 역수
이하로 강제했다. subnormal tail 역산도 막기 위해
\(10^{-12}\le\alpha_{\rm FWER}\le0.2\)와 cell별 tail \(\ge10^{-15}\)을 요구한다.

## 1. observer-scope no-go

상호작용 영역을 \(\mathcal K\), 그 영역에 국소화된 nonselective channel의 Kraus
연산자를 \(K_a\in\mathcal A(\mathcal K)\)라 하자.

\[
\sum_aK_a^\dagger K_a=1.
\]

\(\mathcal K\)와 spacelike인 영역의 공개 관측량
\(B\in\mathcal A(\mathcal K')\)에는 microcausality로
\([K_a,B]=0\)가 성립한다. 따라서 Heisenberg map은

\[
\mathcal T^*(B)
=\sum_aK_a^\dagger B K_a
=B\sum_aK_a^\dagger K_a
=B
\]

이고 모든 상태 \(\rho\)에 대해

\[
\operatorname{Tr}[B\,\mathcal T(\rho)]
=\operatorname{Tr}[B\rho]
\]

다. 국소 selector \(S=0,1\)에 따라 spacelike 검출기 \(B\)의 결과분포가 달라져

\[
D_{\rm TV}\!\left(P_B(\cdot\mid S=1),P_B(\cdot\mid S=0)\right)>0
\]

이면 \(B\)에서 \(S\) 비트를

\[
P_{\rm success}=\frac{1+D_{\rm TV}}2>\frac12
\]

로 읽을 수 있다. 이는 국소 결정론적 연산이 아니다.

측정 결과 \(a\)에 조건부인 selective state는 예외가 아니다.

\[
\rho_a=\frac{\mathcal I_a(\rho)}{p_a},
\qquad
\sum_a p_a\langle B\rangle_a=\langle B\rangle.
\]

결과 \(a\)를 고전적 인과 채널로 전달하기 전까지 원격 ensemble은 변하지 않는다.

## 2. 살아남는 국소 모델

selector는 사람의 이름이나 의식 label이 아니라 probe의 물리적 내부상태
\(m_S\)여야 한다. 최소 상호작용은 다음처럼 쓸 수 있다.

\[
S_{\rm int}
=-\lambda\int d\tau'\,
f_\tau(\tau')R(z(\tau'))m_S(\tau')\mathcal O_X(z(\tau')).
\]

여기서 \(z(\tau')\)는 probe worldline이고 \(\tau'\)는 좌표시간이 아닌
proper time이다. \(f_\tau\)는 유한 switching, \(R\)은 국소 공간·공진 profile,
\(\mathcal O_X\)는 검증돼야 할 Hermitian local operator다.

\[
\mathcal T_{g,R,S,\tau}(\rho_X)
=\operatorname{Tr}_P\!\left[
U_{\mathcal K}(\rho_X\otimes\sigma_S)U_{\mathcal K}^\dagger
\right].
\]

probe별 결합이 다르면

\[
(G_p^R)^{-1}=(G_{p,0}^R)^{-1}-\Sigma_p^R,
\qquad
\Sigma_p^R\simeq\lambda_p^2D_X^R
\]

가 되어 한 probe만 크게 반응할 수 있다. 이는 probe의 self-energy 차이이며
observer별 public pole이 아니다.

외부 source profile을 손으로 공급하면 field 부분만의 stress는 보존되지 않는다.

\[
\nabla_\mu T_{\rm fields}^{\mu\nu}
=(\nabla^\nu J)\mathcal O_X.
\]

controller, pump, selector와 장치를 포함한 총 작용에서만 on-shell
\(\nabla_\mu T_{\rm total}^{\mu\nu}=0\)를 요구할 수 있다. 기존 scalar energy
ledger의 평균 닫힘은 이 local Ward identity의 증명이 아니다.

## 3. 완전한 단일 파장·공간·시간 마스크는 불가능하다

표준편차로 정의한 Fourier 폭에는

\[
\Delta t\,\Delta\omega\ge\frac12,
\qquad
\Delta x\,\Delta k\ge\frac12
\]

가 성립한다. 따라서 정확한 단일 주파수, compact 공간 support, 정확한 유한
지속시간을 동시에 delta 함수처럼 지정할 수 없다. finite pulse는 sideband를 만들고,
공간 mask의 edge는 여러 \(k\) mode를 만든다.

새 게이트는 이를 실패로 숨기지 않는다. 현재 구현은 전체 calibrated design tensor와
공간 누설·early-window equivalence bound를 먼저 고정하고 held-out 자료에서 검사한다.
실제 실험으로 승격하려면 주파수 좌표, 위치 좌표, clock uncertainty와 lightcone
margin도 별도 geometry artifact에 고정해야 한다.

## 4. Gate A: randomized spacelike marginal

로컬 검출기 \(A\)와 spacelike 검출기 \(B_j\)의 결과를 미리 정한 유한 bin에 넣는다.
selector \(S=0,1\)에 대한 multinomial counts로 empirical TV를 계산한다.

\[
\widehat D_{\rm TV}(P,Q)
=\frac12\sum_k|\hat p_k-\hat q_k|.
\]

각 empirical distribution에는 Weissman \(L_1\) concentration radius를 사용하고,
모든 selector·검출기 분포에 Bonferroni family correction을 적용한다. 두 분포의 TV
오차는 각 \(L_1\) radius 합의 절반으로 전파한다. 표본이 작아 interval이
\([0,1]\) 전체가 되면 통과시키지 않는다.

통과조건은

\[
\operatorname{LCB}D_{\rm TV}(P_A^1,P_A^0)>\Delta_{\min},
\]

\[
\max_j\operatorname{UCB}D_{\rm TV}(P_{B_j}^1,P_{B_j}^0)
\le\delta_{\rm NS}
\]

다. 첫 식은 실제 로컬 효과를 요구하고, 둘째 식은 spacelike marginal의
equivalence-to-zero를 요구한다.

## 5. Gate B: crossed-holdout spatiotemporal mask

paired raw response tensor를

\[
D_{npfxta}
=Y^{\rm matched}_{npfxta}-Y^{\rm sham}_{npfxta}
\]

로 둔다. 축은 trial \(n\), probe \(p\), frequency \(f\), 위치 \(x\), 시간 \(t\),
명령 \(a\)다. 독립 calibration과 manifest에서 동결한 설계 tensor는

\[
M_{pfxta}=g_pR_fS_{xa}T_t(\tau_{xa})
\]

다. 물리적 인과 실험으로 확장할 때 profile은 적어도

\[
T_t(\tau_{xa})=0
\quad\text{for}\quad
t-t_{{\rm cmd},a}<\frac{L_{xa}}{v}-\Delta_{\rm sync}
\]

를 만족해야 한다. 단, 현재 `prearrival_mask`는 이 좌표식에서 코드가 유도한 값이
아니므로 early-time control일 뿐 causal certificate가 아니다.

각 paired row에는 matched/sham 양쪽의 block ID를 붙인다. 두 ID 열은 같은 순서로
일치하고 모두 유일해야 한다. exact paired-difference row 복제도 거부한다. 그러나
새 ID와 미세 jitter를 붙이면 내용 비교만으로 실제 독립 획득을 증명할 수 없다.
반복 측정은 먼저 외부에서 인증된 acquisition/cluster block 단위로 집계해야 한다.
block ID, 최소 표본 수, 전처리 artifact hash, design calibration hash와 모든 통계
threshold는 manifest hash에 포함되지만, hash 문자열 자체는 외부 서명이 아니다.

훈련 cell에서는 global amplitude 하나만 paired covariance GLS로 적합한다.

\[
\hat\alpha
=\frac{M_{\rm tr}^{\mathsf T}\Sigma_{\rm tr}^{-1}D_{\rm tr}}
{M_{\rm tr}^{\mathsf T}\Sigma_{\rm tr}^{-1}M_{\rm tr}}.
\]

여기서는 pseudoinverse를 쓰지 않는다. training covariance가 full-rank SPD가 아니거나
조건수가 사전 한계를 넘으면 바로 실패한다. training cell은 최소 3개, nonzero signal
cell은 최소 2개여야 하며 자유도는 정확히

\[
\nu_{\rm tr}=N_{\rm tr}-1
\]

이다. 따라서 amplitude 하나와 training cell 하나인 포화 모형은 통과할 수 없다.

frequency·space·time·command가 교차된 미사용 cell에는

\[
D_{\rm ho}\stackrel{?}{=}\hat\alpha M_{\rm ho}
\]

를 예측한다. held-out 값은 설계 mask를 곱해 강제로 0으로 만들지 않는다.
pre-arrival와 off-support raw response도 그대로 equivalence bound에 들어간다.

training residual에는

\[
L_{\rm tr}=I-M_{\rm tr}w^{\mathsf T},\qquad
C_{r,{\rm tr}}=L_{\rm tr}C_{\rm tr}L_{\rm tr}^{\mathsf T}
\]

를 사용하고, heldout residual covariance에는 training--heldout cross-covariance까지
포함한다.

\[
C_{r,{\rm ho}}
=C_{hh}+M_hM_h^{\mathsf T}(w^{\mathsf T}C_{tt}w)
-C_{ht}wM_h^{\mathsf T}-M_hw^{\mathsf T}C_{th}.
\]

두 residual covariance 모두 finite·symmetric·PSD, 예상 rank, 양의 variance floor와
condition-number gate를 통과해야 한다. 음수 variance를 0으로 clamp하지 않는다.

pointwise \(1.96\,SE\) 대신 사전 지정한 전체 comparison 수

\[
m=N_{\rm tr}+N_{\rm ho}+N_{\rm pre}+N_{\rm off}+N_{\rm target}
\]

에 Student-\(t\) Bonferroni 임계값

\[
t_{1-\alpha_{\rm FWER}/(2m),\,N_{\rm block}-1}
\]

을 적용한다. training residual도 point estimate가 아니라 simultaneous UCB가
equivalence bound 아래여야 한다. 고정 raw-mean contrast의 Student-\(t\) bound와 달리,
같은 표본 covariance에서 (w\)를 추정하는 feasible-GLS residual 구간은 exact pivot이
아닌 조건부 plug-in 근사다. 실제 자료 승격에는 독립 covariance calibration/split 또는
covariance와 (w\)를 매 반복 재적합하는 preregistered block max-\(T\)가 필요하다.

위치 \(a\)의 held-out localization에는

\[
\operatorname{LCB}(A_{aa})
>
\max_{i\ne a}\operatorname{UCB}(A_{ia})
\]

를 요구한다. marginal별로 \(g,R,S,T\)가 각각 그럴듯해도 frequency×space interaction이
rank 2이면 joint holdout에서 실패한다.

보고서에는 canonical raw paired rows와 설정을 immutable tuple로 보존한다. validator는
그 원자료에서 manifest, covariance, 모든 interval, stage와 blocker를 전부 다시
계산하고 dataclass의 모든 필드를 비교한다. supplied boolean만 바꾼 certificate는
통과하지 않는다.

## 6. 정확히 증명된 비식별성

설계 tensor의 곱만 관측하면

\[
(g,R,S,T)
\mapsto
(c_gg,c_RR,c_SS,(c_gc_Rc_S)^{-1}T)
\]

는 모든 cell에서 같은 \(M\)을 만든다. 따라서 joint mask가 통과해도 각 factor의
절대 normalization과 CE coupling은 자료만으로 식별되지 않는다. 독립 calibration이
필요하며, 코드의 `individual_factors_physically_identified`는 항상 `False`다.

또한 단일 probe response \(D_A\)는 언제나

\[
K=D_A/g_A
\]

인 public-kernel 모형과 \(U_A=D_A\)인 private-response 모형에 동시에 맞는다.
독립 보정된 crossed held-out probe 없이는 두 설명을 분리할 수 없다.

## 7. 판정 사다리

| 관측 | 최대 판정 |
|---|---|
| 로컬 \(A\) 효과 없음 | selector 가설 기각 |
| \(A\)만 반응, spacelike null | `CONDITIONAL_LOCAL_PROBE_INSTRUMENT` |
| pre-lightcone 차이 | 누설·동기화 오류 또는 local-QFT class 실패 |
| 선언된 block 모형 + early-window null + crossed holdout | `CONDITIONAL_DECLARED_BLOCK_SPATIOTEMPORAL_RESPONSE_MASK` |
| 위 결과 + 좌표에서 유도한 광원뿔 mask + spacelike null | public causal response의 추가 후보 |
| pole·spectrum·양자수·입자 inventory까지 독립 확인 | 그 뒤에만 new-matter 검사 시작 |

이번 루프는 마지막 세 단계를 달성하지 않는다.

## 8. 현재 blocker

1. block ID용 API와 artifact hash gate는 생겼지만, 외부 timestamp/signature와 실제
   acquisition/cluster mapping을 가진 tensor가 없다. exact 복제는 막지만 새 ID와
   미세 jitter를 붙인 상관 복제를 독립 획득과 구별할 수 없다.
2. feasible-GLS의 covariance와 weight를 같은 표본에서 추정한다. 실제 인증에는 독립
   covariance split 또는 refit block max-\(T\)가 필요하다.
3. 현재 phase resultant는 실제 시계열 autocorrelation 교정이 없다.
4. probe 간 covariance와 독립 전원·clock·readout 자료가 없다.
5. 현재 public kernel은 scalar이고 측정된 \(D^R(\omega,\mathbf k,x,t)\)가 없다.
6. spatial actuator map은 caller가 공급하며 CE 작용에서 유도되지 않았다.
7. CE는 실제 connected correlator, physical pole, residue, LSZ, vertex가 없다.
8. controller를 포함한 local 4-momentum 및 Ward identity가 없다.

따라서 현재 CE 물리 단계는 계속 `REGISTERED_SCALE`이며, 29.64757 MeV를
observer-dependent pole로 사용할 근거는 없다.

## 9. 재현

```powershell
uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_clarus_resonant_matter.py `
  tests/test_probe_scaffold_pilot.py `
  tests/test_spacelike_marginal_gate.py `
  tests/test_resonant_spatiotemporal_mask.py `
  tests/test_targeted_spatial_actuation.py `
  tests/test_resonance_stress_identifiability.py -q
```

## 참고

- Kitajima, [Local Operations and Completely Positive Maps in Algebraic Quantum Field Theory](https://arxiv.org/abs/1704.01229)
- Fewster and Verch, [Quantum Fields and Local Measurements](https://arxiv.org/abs/1810.06512)
- Fewster, Jubb and Ruep, [Asymptotic Measurement Schemes for Every Observable of a Quantum Field Theory](https://arxiv.org/abs/2304.13356)
