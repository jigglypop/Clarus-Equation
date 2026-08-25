# 10-sources — 0D 접힘 단일 환경장 근거

Status: COMPLETE

최종 접근일: 2026-08-25

## 1. 직접 근거

| 주장 또는 제한 | 1차 출처 | 출처가 확립하는 범위 | 이 연구에 주지 않는 것 |
|---|---|---|---|
| 한 연속장과 공간 kernel의 자기활성 pattern | S.-i. Amari, “Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields,” *Biological Cybernetics* 27 (1977), [DOI 10.1007/BF00337259](https://doi.org/10.1007/BF00337259) | 한 field의 비국소 결합에서 지속 localized excitation과 여러 pattern dynamics가 가능한 구체적 수학 모형 | 그 field가 공간 접힘·양자장·암흑부문이라는 동일성, 우주론적 stress |
| 비음 kernel을 가진 선형 Volterra 방정식 | T. Naito, J. S. Shin, S. Murakami, and P. H. A. Ngoc, “Characterizations of Linear Volterra Integral Equations with Nonnegative Kernels,” *Journal of Mathematical Analysis and Applications* 335 (2007), [DOI 10.1016/j.jmaa.2007.01.070](https://doi.org/10.1016/j.jmaa.2007.01.070) | positivity, resolvent, Perron--Frobenius형 결과와 exponential stability 판정 구조 | 이 문서의 nonlinear saturation·양의 평형이 가정 없이 자동 존재한다는 명제 |
| $\delta$ point interaction의 연산자·renormalization 경계 | K. G. Akbaş, F. Erman, and O. T. Turgut, “On Schrödinger Operators Modified by $\delta$ Interactions,” *Annals of Physics* 458 (2023), [DOI 10.1016/j.aop.2023.169468](https://doi.org/10.1016/j.aop.2023.169468) | point interaction은 적절한 operator/domain 구성으로 다뤄야 하며 일부 경우 renormalization이 필요함 | arbitrary $\delta$ carrier를 local relativistic field source에 그대로 넣어도 finite stress가 된다는 보장 |
| 과거 사건이 미래 조건부 강도를 높이는 선형 자기흥분 과정 | A. G. Hawkes, “Spectra of Some Self-Exciting and Mutually Exciting Point Processes,” *Biometrika* 58 (1971), [DOI 10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet.58.1.83) | 선형 self-exciting point process와 스펙트럼 구조 | 기억장이 물리적 우주장이라는 동일성, 에너지 보존 |
| 자기흥분 과정의 cluster/branching 표현 | A. G. Hawkes and D. Oakes, “A Cluster Process Representation of a Self-Exciting Process,” *Journal of Applied Probability* 11 (1974), [DOI 10.2307/3212693](https://doi.org/10.2307/3212693) | stationary process의 immigration--birth 표현과 subcritical 존재 구조 | supercritical 과정의 유한 stationary 상태 |
| 비선형 Hawkes의 존재·안정성 충분조건 | P. Brémaud and L. Massoulié, “Stability of Nonlinear Hawkes Processes,” *Annals of Probability* 24 (1996), [DOI 10.1214/aop/1065725193](https://doi.org/10.1214/aop/1065725193) | Lipschitz nonlinear intensity와 kernel norm을 이용한 stationary version·수렴의 충분조건 | mean-field의 양의 고정점이 stochastic 영구 생존을 보장한다는 명제 |
| 정준 실수 스칼라의 정적 국소 lump 제한 | G. H. Derrick, “Comments on Nonlinear Wave Equations as Models for Elementary Particles,” *Journal of Mathematical Physics* 5 (1964), [DOI 10.1063/1.1704233](https://doi.org/10.1063/1.1704233) | $3$공간차원 canonical scalar의 finite-energy static lump에 대한 scaling obstruction | 시간 의존, 복소장, gauge, higher-derivative, nonlocal, open-system 구성의 금지 |
| 복소장 국소화 대안 | S. Coleman, “Q-balls,” *Nuclear Physics B* 262 (1985), [DOI 10.1016/0550-3213(85)90286-X](https://doi.org/10.1016/0550-3213(85)90286-X) | global $U(1)$ charge와 적절한 potential을 가진 complex scalar의 non-topological soliton | canonical real scalar 하나의 정적 lump, 자기흥분 사건 법칙 |
| 시간 의존 장수명 국소화 대안 | M. A. Amin, “K-oscillons: Oscillons with Noncanonical Kinetic Terms,” *Physical Review D* 87 (2013), [DOI 10.1103/PhysRevD.87.123505](https://doi.org/10.1103/PhysRevD.87.123505) | 비정준 kinetic term에서 localized oscillatory configuration과 작은 진폭 조건 | 무한 수명, Hawkes source와의 동일성 |
| source와 retarded field | G. F. R. Ellis, “Fields of Moving Multipoles,” *Nature* 205 (1965), [DOI 10.1038/205582a0](https://doi.org/10.1038/205582a0) | source history와 retarded field의 인과적 구성 | source work 또는 stochastic 생성의 에너지 장부 |
| 곡률 시공간 retarded Green 함수 | Y.-Z. Chu and G. D. Starkman, “Retarded Green’s Functions in Perturbed Spacetimes,” [arXiv:0808.0642](https://arxiv.org/abs/0808.0642) | 곡률 시공간에서 초기자료·source의 causal retarded response와 tail | tail의 positivity, 암흑부문 동일성 |

## 2. 출처로부터 허용되는 최소 사상

사용자의 현재 문장을 가장 직접적으로 표현하는 결합은 persistent carrier와 한
동적장이다.

$$
\mu_{F,t}(B)=\int_{\Gamma_{\rm ns}}
w(\gamma)\mathbf1_B(F_t(\gamma))\nu_{\rm ns}(d\gamma).
$$

이 weighted pushforward는 비선택 history data를 carrier measure로 보내는
**별도 물리 사상**이다. 표준 instrument 조건부화나 아래 문헌에서 자동으로
나오는 중력 source가 아니다.

$$
\mu_F(d^3y)=\sum_jw_j\delta_{\mathbf X_j}(d^3y),
$$

$$
\chi(t,\mathbf x)=b(t,\mathbf x)+A
\int_{t_i}^{t}ds\int
K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma(\chi(s,\mathbf y))\mu_F(d^3y).
$$

$\mu_F$는 고정 carrier이고 $\chi=\psi/\psi_s$만 동적 변수다. Amari와
positive Volterra 문헌은 “한 장 + kernel + 포화 readout”의 수학적 가능성과
stability 분석 도구를 제공하지만, 그 장의 우주론적 존재론을 주지는 않는다.

“실행”이 기존 carrier의 재활성화가 아니라 새 접힘 사건의 생성이라는 별도 뜻이면
Hawkes 문헌이 허용하는 결합은 다음 세 단계다.

$$
N(d^4y)=\sum_a\delta_{y_a}(d^4y),
$$

$$
\psi(x)=A\int_{J^-(x)}K_{\ell,R}(x,y)N(d^4y),
$$

$$
\lambda(x\mid\mathcal F_{x^-})=\lambda_0(x)+F(\psi(x)).
$$

$F(\psi)=\beta\psi$는 선형 Hawkes 경로이고,

$$
F(\psi)=\frac{\beta\psi}{1+\psi/\psi_s}
$$

는 별도로 선언한 포화 경로다. 첫 번째 출처 묶음은 이러한 자기흥분 확률과정의
수학을 지지하지만, $\psi$가 공간 접힘의 실제 미시적 장이라는 해석을 지지하지
않는다.

## 3. 출처 경계

1. persistent-carrier 경로의 $A\rho(W)$는 fixed network의 field gain이고,
   Hawkes의 $\mathcal R$은 기대 자손수·stationarity 척도다. 둘은 같은 기호나
   물리량으로 합치면 안 되며 어느 쪽도 에너지 밀도나 우주론적 $\Omega$가 아니다.
2. retarded Green 함수는 인과적 전달을 주지만 positivity, finite self-energy,
   접힘 생성률을 자동으로 주지 않는다.
3. Derrick 반례는 canonical real scalar의 정적 finite-energy lump에 한정한다.
   Q-ball과 oscillon은 그 전제를 바꾸는 대안이지 현재 모형의 도출 결과가 아니다.
4. stochastic deposit에는 source/reservoir와 총 stress-energy 장부가 필요하다.
   위 문헌 어느 것도 무비용 자기생성을 허용하지 않는다.
