# D–T 핵융합 scalar-current 검증 루프

코드: `reality_stone/python/reality_stone/clarus/fusion_scalar_current_loop.py`  
게이트: `examples/physics/fusion_scalar_current_gate.py`  
테스트: `tests/test_fusion_scalar_current_loop.py`

## 결론

29.64757 MeV flavor-aligned scalar의 D–T 장벽 계산에서 기존 Gaussian one-body
근사는 공개 Helm 중심값과 매우 잘 맞는다. $q=0,10,20,29.64757,40$ MeV의 등록된
5점 격자에서 D와 T form-factor 곱의 최대 표본 상대 차이는
$8.99\times10^{-5}$, $q=i m_\phi$의 exterior-residue 해석 진단에서 차이는
$-4.94\times10^{-5}$다. 이 검산은 Gaussian 중심곡선이 크게 잘못되었다는 가설을
기각하지만 연속구간 최대나 물리 인증을 완성하지는 않는다.

공개 입력을 끝까지 결합하면 다음 세 가지가 남는다.

- 2026 lattice-QCD의 D와 $^3$He sigma-term을 쓰는 진단값은 필요한 공통
  coupling 보정이 $+1.11\%\pm1.48\%$다. 중심값은 기존 $\pm1.2\%$ 비교띠
  안이지만 1표준편차 상단은 $2.59\%$다.
- nucleon scalar-radius 저에너지 전개는 $q=40$ MeV에서 scalar-radius 양 끝점과
  strange-slope 중심을 쓴 coupling 보정 $+1.23\%$--$+1.30\%$를 준다. 다만 이는
  완전한 불확실성 띠나 form-factor likelihood가 아니라 slope 진단값이다.
- 실제 T sigma-term, 동일 regulator에서 계산된 $q$-의존 D/T one-plus-two-body
  current, short-range two-nucleon contact, D/T 공분산, 그리고
  $r=3.24$--50 fm 실공간 응답이 공개 입력에 함께 존재하지 않는다.

따라서 `scalar_current_certification_pass=False`,
`physical_ce_fusion_branch_accepted=False`다. 여기서 $\pm1.2\%$는 앞선 density
morphology proxy의 비교띠이지 통계적 신뢰구간이 아니다.

## 1. 고정한 1차 출처와 역할

각 수치의 출처와 계산 가정은 코드에서 별도 dataclass로 저장한다. 논문 링크가
있다는 사실을 covariance 제공으로 오인하지 않도록 모든 covariance 필드도 따로
기록한다.

| 키 | 고정 버전 | 이 루프에서 쓴 내용 |
|---|---|---|
| `broggini_2025_v2` | [Broggini et al., arXiv:2509.03486v2](https://arxiv.org/abs/2509.03486v2) | flavor alignment, $u,d,s$ nucleon scalar fraction |
| `korber_2017_v1` | [Körber, Nogga, de Vries, arXiv:1704.01150v1](https://arxiv.org/abs/1704.01150v1) | D/T/He3 Helm 계수, one-/two-body scalar response |
| `andreoli_2019_v2` | [Andreoli et al., arXiv:1811.01843v2](https://arxiv.org/abs/1811.01843v2) | QMC two-body cutoff 의존성 |
| `devries_2024_v2` | [de Vries et al., arXiv:2310.11343v2](https://arxiv.org/abs/2310.11343v2) | 고차 regulator 의존성, 미결정 contact |
| `filandri_2024_v2` | [Filandri, Viviani, arXiv:2403.06599v2](https://arxiv.org/abs/2403.06599v2) | 저운동량 D scalar RME order 진단 |
| `chakraborty_2026_v1` | [Chakraborty et al., arXiv:2603.28872v1](https://arxiv.org/abs/2603.28872v1) | D와 He3의 $q=0$ light-quark sigma-term ratio |
| `agadjanov_2024_v2` | [Agadjanov et al., arXiv:2303.08741v2](https://arxiv.org/abs/2303.08741v2) | $\sigma_{\pi N}$, $\sigma_s$ 정규화 |
| `alarcon_weiss_2017_v1` | [Alarcón, Weiss, arXiv:1707.07682v1](https://arxiv.org/abs/1707.07682v1) | light-quark nucleon scalar radius |

## 2. one-nucleon scalar charge

Flavor alignment가

\[
\frac{g_u}{m_u}=\frac{g_d}{m_d}=\frac{g_s}{m_s}=\frac{1}{f_\phi}
\]

이면 source가 사용하는 nucleon coupling은

\[
g_N=\frac{m_N}{f_\phi}\sum_{q=u,d,s}f_{Tq}^{(N)}
\]

이다. Broggini et al.의 입력은

\[
\begin{aligned}
p &: (0.020\pm0.004,\ 0.026\pm0.005,\ 0.118\pm0.062),\\
n &: (0.014\pm0.003,\ 0.036\pm0.008,\ 0.118\pm0.062)
\end{aligned}
\]

이다. 중요한 단위 구분이 있다. 기존 코드의 0.154와 0.158은 위의
dimensionless fraction 합 그 자체가 아니다. 각각
$m_N\sum f_{Tq}^{(N)}$를 계산한 **sigma numerator, 단위 GeV**다. 이 때문에 새
audit 필드도 `candidate_*_sigma_numerator_gev`로 명명했다.

$f_\phi=8.9709918$ GeV에서 기존 후보는 다음과 같다.

| 양 | 값 |
|---|---:|
| proton sigma numerator | 0.154 GeV |
| neutron sigma numerator | 0.158 GeV |
| $g_p$ | 0.0171664409 |
| $g_n$ | 0.0176123225 |
| $Q_D=g_p+g_n$ | 0.0347787634 |
| $Q_T=g_p+2g_n$ | 0.0523910858 |
| $Q_DQ_T$ | 0.00182209718 |

각 flavor 오차를 독립으로 단순 합성하면 fraction-sum 오차는 p에서 0.06233,
n에서 0.06259로 매우 크다. 특히 공통 strange 항이 지배하므로 p/n 상관 없이
독립오차처럼 쓰면 안 된다. 공개 표는 이 후보에 필요한 p/n 공동 covariance를
주지 않는다.

현대 lattice 정규화인

\[
\sigma_{\pi N}=43.7\pm3.6\ {\rm MeV},\qquad
\sigma_s=28.6\pm9.3\ {\rm MeV}
\]

를 쓰면 중심 $u+d+s$ 합은 72.3 MeV다. 기존 isoscalar numerator 156 MeV와의
비는 0.463462다. 이 비의 제곱은 p/n을 평균낸 isoscalar 근사일 뿐이다. 현대
입력도 조건부로 p=n=72.3 MeV인 isoscalar nucleon 값이라고 두고 실제 D/T charge
조합 $(144.6\times216.9)/(312\times470)$을 쓰면 $f_\phi$를 그대로 둘 때
product 비는 0.213883, 동일 product를 맞추기 위해 중심값만 재조정한
scale은 4.14885 GeV다. 이것은 정규화 민감도 진단일 뿐이다.
$\sigma_{\pi N}$--$\sigma_s$ covariance와 재조정된
UV/kaon/neutron likelihood를 다시 계산하지 않았으므로 새로운 fit이나 허용
후보로 승격하지 않는다.

## 3. Helm 대 Gaussian one-body 검산

Körber et al.의 정규화된 Helm 근사는

\[
F_H(q)=\frac{3j_1(qr_n)}{qr_n}\exp\left[-\frac{(qs)^2}{2}\right],
\]

\[
r_n^2=c^2+\frac{7}{3}\pi^2a^2-5s^2,\qquad
c=1.23A^{1/3}-0.60\ {\rm fm}
\]

이다. $q$를 MeV로 입력할 때 코드에서는 모든 $qR$에 $\hbar c$를 나눈다.
Table 4의 D $(a,s)=(0.47,1.09)$ fm와 T $(0.38,0.96)$ fm를 사용했다.
Helm rms는

\[
R_{\rm rms}^2=\frac35r_n^2+3s^2
\]

이므로 D 1.89564 fm, T 1.67993 fm다. 비교한 기존 Gaussian rms는 D 1.975 fm,
T 1.59 fm이고 곱은

\[
F_G^D(q)F_G^T(q)=
\exp\left[-\frac{q^2(R_D^2+R_T^2)}{6(\hbar c)^2}\right]
\]

이다.

| $q$ (MeV) | $F_D^H$ | $F_T^H$ | Helm product | Gaussian product | Helm/Gauss $-1$ |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 0 |
| 10 | 0.998463079 | 0.998792756 | 0.997257690 | 0.997252085 | $5.62\times10^{-6}$ |
| 20 | 0.993866472 | 0.995179762 | 0.989075799 | 0.989053564 | $2.25\times10^{-5}$ |
| 29.64757 | 0.986571378 | 0.989438343 | 0.976151549 | 0.976103332 | $4.94\times10^{-5}$ |
| 40 | 0.975690683 | 0.980857992 | 0.957014005 | 0.956927970 | $8.99\times10^{-5}$ |

Yukawa folding에서 자주 등장하는 해석적 연속 $q=i m_\phi$도 별도로 계산했다.

| 양 | 값 |
|---|---:|
| $F_D^H(i m_\phi)$ | 1.013611401 |
| $F_T^H(i m_\phi)$ | 1.010674383 |
| Helm product | 1.024431077 |
| Gaussian product | 1.024481699 |
| 상대 잔차 | $-4.9412\times10^{-5}$ |

따라서 exterior-residue 중심곡선 진단 tolerance $10^{-4}$는 통과한다. 그러나
$q=i m_\phi$는 측정점이 아니다. 유한 $r$, 특히 핵밀도가 겹치는 장벽 안쪽에서는
$F(i m)e^{-mr}/r$ residue가 full folded response를 대신하지도 않는다. Helm 계수와
one-body scalar-density covariance, two-body current도 빠져 있다. 그래서
`exterior_residue_analytic_diagnostic_pass=True`와
`one_body_shape_certification_pass=False`가 동시에 맞는 판정이다.

## 4. 장벽 창 $r=3.24$--50 fm와 진단 격자 $q=0$--40 MeV

매개자의 Compton 길이는

\[
\lambda_\phi=\frac{\hbar c}{m_\phi}=6.65576\ {\rm fm}
\]

이다. 점핵 Yukawa 지수만 표시하면 다음과 같다.

| $r$ (fm) | $\exp(-m_\phi r/\hbar c)$ |
|---:|---:|
| 3.24 | 0.614592 |
| 5 | 0.471786 |
| 10 | 0.222582 |
| 20 | 0.0495428 |
| 50 | 0.000546326 |

이 루프가 결합 비교용으로 택한 $q_{\max}=40$ MeV 격자가 직접 분해하는 최소
길이는 $\hbar c/q_{\max}=4.933$ fm다. 3.24 fm 안쪽을 같은 기준으로 분해하려면
$q\simeq60.90$ MeV가 필요하다. 이는 **이번 joint diagnostic grid의 한계**이지
Körber 논문의 Helm fit 자체가 40 MeV에서 끝난다는 뜻이 아니다. 따라서 현재
격자는 장벽 바깥쪽 중심곡선 검산에는 유용하지만 전체 3.24--50 fm 실공간
current likelihood를 대신하지 않는다.

## 5. nucleon scalar radius와 form-factor slope

Alarcón–Weiss의 $\sigma(0)\simeq45$ MeV 행에 대응하는 light scalar radius 구간
$\langle r_s^2\rangle=1.34$--1.49 fm²와 Körber et al.이 사용한 strange slope
$\dot\sigma_s=0.3\pm0.2\ {\rm GeV}^{-1}$를 결합했다. Agadjanov 중심값으로 light
weight는

\[
w=\frac{\sigma_{\pi N}}{\sigma_{\pi N}+\sigma_s}=0.604426
\]

이다. per-nucleus 저-$q$ intrinsic amplitude 진단은

\[
\delta(q)=-w\frac{q^2\langle r_s^2\rangle}{6(\hbar c)^2}
-(1-w)\frac{\dot\sigma_s(q/1000)^2}{\sigma_s/1000}
\]

로 계산했다.

| $q$ (MeV) | $\delta$, radius 1.34 fm² | $\delta$, radius 1.49 fm² | strange-slope 1σ |
|---:|---:|---:|---:|
| 10 | $-0.07616\%$ | $-0.08004\%$ | 0.02766% |
| 20 | $-0.30465\%$ | $-0.32017\%$ | 0.11065% |
| 29.64757 | $-0.66944\%$ | $-0.70355\%$ | 0.24315% |
| 40 | $-1.21858\%$ | $-1.28067\%$ | 0.44260% |

D와 T에 같은 intrinsic factor를 곱하고 기존 product를 유지하는 정확한 공통
coupling factor는 $1/(1+\delta)$다. 따라서 strange-slope 중심을 고정하고
scalar-radius 양 끝점만 바꾼 $q=40$ MeV coupling 진단은
$+1.2336\%$--$+1.2973\%$로 1.2% 비교띠를 넘는다. 반면 $q=i m_\phi$에서는
amplitude가 $+0.6694\%$--$+0.7036\%$, 필요한 coupling 보정은
$-0.6650\%$--$-0.6986\%$다.

이 부호 차이는 spacelike slope와 해석적 연속을 섞지 않기 위해 명시했다. 또한
저-$q$ 전개를 완전한 form factor로 승격하지 않았고 light/strange slope 공동
covariance도 없으므로 hard correction으로 사용하지 않는다.

## 6. 2026 D/He3 sigma-term proxy

Chakraborty et al. v1의 Table 3은

\[
\frac{\sigma_D}{\sigma_N}=1.975(6)(41),\qquad
\frac{\sigma_{^3{\rm He}}}{\sigma_N}=2.929(29)(126)
\]

를 준다. 이 논문 자체도 물리점 two-nucleon binding의 확정에는 완전한 유한부피
amplitude 분석이 필요하다고 적는다. 통계와 계통을 단순 quadrature로 합친 뒤
$^3$He를 T의 isospin proxy로 놓으면 light-quark 비가산성은

\[
\delta_D^{ud}=-1.250\%\pm2.072\%,\qquad
\delta_{T,\mathrm{proxy}}^{ud}=-2.367\%\pm4.310\%
\]

이다. strange가 additive라는 중심 proxy 아래에서 $w=0.604426$를 곱하면

\[
\delta_D^{uds}=-0.7555\%\pm1.2523\%,\qquad
\delta_{T,\mathrm{proxy}}^{uds}=-1.4305\%\pm2.6050\%.
\]

product와 1차 오차전파는

\[
\Delta_P=(1+\delta_D)(1+\delta_T)-1
=-2.1752\%\pm2.8648\%,
\]

\[
\Delta_g=(1+\Delta_P)^{-1/2}-1
=+1.1057\%\pm1.4805\%.
\]

즉 코드가 출력하는 `+1.11% +/- 1.48%`는 정확히 이 계산의 반올림이다. 출처와
별개로 저장한 가정은 다음과 같다.

| 가정/결손 | 코드 판정 |
|---|---:|
| He3를 T isospin proxy로 사용 | `True` |
| D와 He3 오차를 독립으로 취급 | `True` |
| $\sigma_{\pi N},\sigma_s$는 중심 dilution에만 사용 | `True` |
| 두 sigma-term 오차와 상관 전파 | `False` |
| $q=0$에서만 평가 | `True` |
| 1차 Gaussian 오차전파 | `True` |
| 실제 T sigma-term 입력 | `False` |
| D/T covariance 입력 | `False` |

따라서 중심 $1.1057\%<1.2\%$라는 비교만으로 통과시킬 수 없다. 1σ 상단은
2.5861%이고, 더 근본적으로 이 분포는 실제 D/T likelihood가 아니다.

## 7. chiral two-body scalar current

공개 계산이 사용하는 대표 NLO pion-exchange 구조는

\[
J_{2b}=-c_{\rm is}\left(\frac{g_A}{2f_\pi}\right)^2m_\pi^2
\,\boldsymbol\tau_1\!\cdot\!\boldsymbol\tau_2
\frac{(\boldsymbol\sigma_1\!\cdot\!\mathbf q_1)
(\boldsymbol\sigma_2\!\cdot\!\mathbf q_2)}
{(q_1^2+m_\pi^2)(q_2^2+m_\pi^2)}
\]

이다. 분자의 두 괄호는 합이 아니라
`(sigma1 dot q1) * (sigma2 dot q2)`라는 곱이다. 수치 ledger는 다음처럼
분리한다.

- Körber et al.: 고차 D squared-response 대표값은 $+1.6\%\pm0.8\%$이고 A=3
  효과는 D보다 대략 5배 작지만 N2LO 상대오차가 약 100%다.
- Andreoli et al.: Table 1의 D two-body fraction $\Delta^{(2b)}$는 cutoff 500 MeV와
  10 GeV에서 각각 약 $+0.7\%$와 $+3.0\%$이고 A=3 부호가 regulator 구간에서
  바뀐다. 진폭 변환 $A/A_{1b}=(1-\Delta^{(2b)})^{-1/2}$에 현대 $uds$ light
  weight를 곱하면 exact D 진단은 약 $+0.213\%$--$+0.928\%$다. 선형화
  $\Delta^{(2b)}/2$는 $+0.212\%$--$+0.907\%$로 별도 저장한다.
- de Vries et al.: 더 높은 order에서도 regulator 의존성이 남고, 새 short-range
  two-nucleon contact의 finite part가 미결정이다. 따라서 cutoff band 자체를
  확률분포로 읽지 않는다.
- Filandri–Viviani: $q=0.05\ {\rm fm}^{-1}=9.866$ MeV의 D scalar RME에서 AV18은
  LO $-14.4$, NLO $-0.218$, N2LO $+0.153$으로 NLO/LO $+1.51\%$, 누적
  $+0.45\%$다. N4LO500은 NLO $-0.0806$, N2LO $+0.103$으로 누적
  $-0.156\%$다. 표의 $q\le0.2\ {\rm fm}^{-1}$는 39.465 MeV까지 닿는다.

이 결과들은 “two-body가 수 퍼센트 이하일 가능성이 높다”는 중심 경향을
지지하지만, 등록 질량에서 T 부호·contact·D/T covariance를 함께 고정하지 않는다.

## 8. hard gate

기존 one-body proxy 비교값은 다음과 같이 provenance를 고정했다.

| 비교량 | coupling 보정 |
|---|---:|
| reference Gaussian | $-0.9887\%$ |
| morphology envelope | $-1.2143\%$--$-0.8308\%$ |
| 이 루프의 절대 비교띠 | $\pm1.2\%$ |

새 문헌을 넣은 뒤에도 다음 네 입력이 모두 없으므로 scalar-current 인증 gate는
논리곱에서 닫힌다.

1. 실제 T $q=0$ sigma-term과 momentum-dependent scalar response,
2. $q=0$--40 MeV D/T 공동 covariance,
3. 동일 regulator와 current-consistent potential에서 fit한 short-range contact,
4. $r=3.24$--50 fm 전체 장벽에 대한 one-plus-two-body 실공간 likelihood.

코드는 최종 `False`를 독립 상수로 주입하지 않는다. 위 네 입력뿐 아니라 p/n 및
sigma-term covariance, normalization likelihood, ab-initio density covariance,
full scalar form factor, proxy 제거, uncertainty propagation, regulator 일치와
two-body covariance의 각 leaf를 최종 함수가 직접 다시 논리곱한다. 하위 aggregate만
`True`로 바꾸는 우회가 실패하는지 각 leaf mutation 회귀 테스트가 고정한다.

물리적 CE branch는 여기에 upstream flavor 후보의 UV/action gate와 Pb·kaon·dark
sector 제약 gate까지 다시 논리곱한다. 따라서 가상의 완전한 D/T 핵응답만으로도
전체 CE branch가 자동으로 열리지 않으며, 두 upstream gate는 현재 모두 `False`다.

필요한 다음 산출물은 이 네 항을 포함하는 registered-mass ab-initio 응답 테이블과
machine-readable covariance다. 그 전에는 Helm 중심검산 통과, 작은 two-body
중심값, $+1.11\%$ proxy 중 어느 것도 물리 branch를 열지 않는다.
