# CE 우주 밀도 사상 1단계: 고정점에서 바리온 밀도로

Status: COMPLETE

연구 기준일: 2026-08-15  
선행 run: `cosmology-theory-repository-audit-20260815`

## 초록

이번 단계는 Poisson 소멸확률 $q_{\rm ext}$를 안정한 균일장 값으로 만드는
공변 scalar 작용을 관측 밀도값 없이 구성했다. 이 구성의 변분식, 두 정지
가지, 국소 안정성, stress tensor와 FLRW 방정식은 닫혔다. 그러나 같은
$q_{\rm ext}$를 유지한 채 additive vacuum offset으로 에너지분율을 바꿀 수
있고, 정지 scalar의 상태방정식은 $w=-1$이므로 이것만으로 바리온 먼지를
얻을 수는 없다. 확률과 energy fraction 사이에는 equal-conditional-energy
조건이 정확히 필요하며, two-dust 구성에서 자연스럽게 얻는 양은
$\Omega_b$가 아니라 matter-composition fraction이다. 실행 게이트는 이
경계를 보존해 수학 감사에는 성공하지만 완전한 물리 브리지를 요구하면
의도적으로 실패한다.

## 1. 질문과 결론의 범위

이 보고서가 다루는 질문은 $D>1$인 Poisson 고정점

$$
q_{\rm ext}=\exp[-D(1-q_{\rm ext})]
\tag{1}
$$

을 우주론적 밀도분율에 연결하는 첫 단계다. 식 (1)의 최소 양의 해가
존재하고 유일하다는 선행 정리는 다시 증명하지 않는다. 여기서는 그
무차원 확률을 장론의 정지값, conserved matter composition, 그리고
critical-density fraction으로 옮기는 세 사상을 분리한다.

이번 단계의 정확한 결론은 다음과 같다.

1. `[정리: 존재구성]` 식 (1)을 균일 Euler--Lagrange 정지조건으로 갖는
   dimension-four scalar 작용을 만들 수 있다.
2. `[정리: 국소]` 작은 가지는 선언한 영역에서 안정하고 $x=1$ 가지는
   불안정하다.
3. `[정리: no-go]` 정지 scalar 값만으로 에너지분율이나 baryon dust를
   결정할 수 없다.
4. `[정리: 필요충분]` event 확률과 energy-weighted fraction이 같으려면
   두 event class의 조건부 평균 에너지가 같아야 하고, 그 조건이면
   충분하다.
5. `[산출: 조건부]` equal-energy two-dust 구성은
   $f_b^{(m)}=q_{\rm ext}$를 줄 수 있지만 정확한 임계밀도 관계는
   $\Omega_b=q_{\rm ext}\Omega_m$이다.
6. `[미완성]` conserved current의 미시 작용, total abundance,
   freeze-out과 $\Omega_m$이 아직 고정되지 않았으므로 활성 관측 예측은
   없다.

형식 게이트의 `PASS`는 1--5의 좁은 정리와 no-go가 닫혔다는 뜻이다.
`$q_{\rm ext}=\Omega_b$가 자연에서 유도되었다`는 뜻은 아니다.

## 2. 정의, 공리와 차원

`[정의]` 자연단위 $c=\hbar=1$, metric 부호 $(-+++)$와
$M_{\rm Pl}^{-2}=8\pi G$를 쓴다. $x$는 무차원 실수장이고 정의역은
$0<x\leq1$이다. $D>1$과 potential $v_D$도 무차원이며 $F$와 $M$의
질량차원은 1이다.

`[공리: 외부 입력]` $D$는 이번 작용에서 유도하지 않고 고정점 family의
외부 입력으로 둔다.

`[공리: 모델 선택]` 다음 후보 작용을 택한다.

$$
S_x=\int d^4x\sqrt{-g}\left[
-\frac{F^2}{2}g^{\mu\nu}\partial_\mu x\partial_\nu x
-M^4v_D(x)\right],
\tag{2}
$$

$$
v_D(x)=x\log x-x+D\left(x-\frac{x^2}{2}\right)+C.
\tag{3}
$$

식 (2)--(3)은 고정점 식을 적분해 얻은 명시적 후보이며 유일한 작용이
아니다. $[x]=[D]=[v_D]=0$이므로 로그 인자는 무차원이고,
$F^2(\partial x)^2$와 $M^4v_D$의 질량차원은 모두 4다. 따라서 4차원
작용은 무차원이다.

## 3. 변분 임베딩

이 절은 고정점 식이 실제 장방정식의 정지조건이 될 수 있음을 보인다.
compact-support variation 또는 경계에서 $\delta x=0$을 두면 식 (2)의
변분은

$$
\delta S_x=\int d^4x\sqrt{-g}
\left[F^2\Box x-M^4v_D'(x)\right]\delta x
\tag{4}
$$

이고,

$$
v_D'(x)=\log x+D(1-x).
\tag{5}
$$

`[정리: 존재구성]` 시공간적으로 균일하고 상수인 해에서는 $\Box x=0$이므로

$$
v_D'(x)=0
\iff \log x=-D(1-x)
\iff x=\exp[-D(1-x)].
\tag{6}
$$

따라서 식 (2)는 정확히 식 (1)의 두 정지 가지를 갖는다. 이 결과는
existence construction이다. 임의의 양의 함수 $w(x)$에 대해

$$
\widetilde v'(x)=w(x)[\log x+D(1-x)]
\tag{7}
$$

도 같은 stationary set을 가지므로, 고정점 식은 kinetic term이나
potential의 곡률과 상호작용을 유일하게 정하지 않는다.

## 4. 두 가지와 국소 안정성

potential의 두 번째 미분은

$$
v_D''(x)=\frac1x-D.
\tag{8}
$$

선행 고정점 정리에서 작은 근은 $0<q_{\rm ext}<1/D$이고 다른 근은
$x=1$이다. 따라서

$$
v_D''(q_{\rm ext})=\frac1{q_{\rm ext}}-D>0,
\qquad
v_D''(1)=1-D<0.
\tag{9}
$$

`[정리: 선언 영역]` $q_{\rm ext}$는 $0<x\leq1$에서 유일한 최소이고
$x=1$은 한쪽 최대이자 tachyonic 정지점이다. 외부 입력
$D=3.1777584234099736$을 넣은 조건부 수치는

| 양 | 값 |
|---|---:|
| $q_{\rm ext}$ | 0.04864671964402821 |
| 고정점 잔차 | $-1.39\times10^{-17}$ |
| $v_D''(q_{\rm ext})$ | 17.3786122285 |
| $v_D''(1)$ | -2.17775842341 |

`[미완성]` 이 판정은 국소적이다. 선언한 $0<x\leq1$이 Lorentzian
시간발전에서 자동으로 불변이라는 증명은 없고, 식 (3)을 $x>1$로 단순
연장하면 $-Dx^2/2$ 항 때문에 아래로 무한해진다. 전역 field-space
completion, cutoff와 radiative stability는 별도 과제다.

## 5. 공변 동역학과 stress

metric variation으로 얻는 scalar stress tensor는

$$
T^{(x)}_{\mu\nu}
=F^2\partial_\mu x\partial_\nu x
-g_{\mu\nu}\left[\frac{F^2}{2}(\partial x)^2+M^4v_D(x)\right].
\tag{10}
$$

평탄 FLRW의 균일장에서는

$$
\ddot x+3H\dot x+\frac{M^4}{F^2}v_D'(x)=0,
\tag{11}
$$

$$
\rho_x=\frac{F^2}{2}\dot x^2+M^4v_D(x),
\qquad
p_x=\frac{F^2}{2}\dot x^2-M^4v_D(x).
\tag{12}
$$

`[정리: 국소 고전]` $x=q_{\rm ext}+\delta x$와
$\varphi=F\delta x$를 쓰면 작은 요동의 질량은

$$
m_*^2=\frac{M^4}{F^2}\left(\frac1{q_{\rm ext}}-D\right)>0.
\tag{13}
$$

$F^2>0$에서 ghost가 없고 $c_s^2=1$이며 작은 가지에는 tachyon이 없다.
다만 relaxation 속도는 자유로운 비 $m_*/H$에 의존한다. 그러므로
정지점의 위치가 정해져도 우주가 그곳에 언제 도달하는지는 정해지지
않는다.

## 6. scalar 값에서 density fraction으로 바로 갈 수 없는 이유

이 절은 단순한 부품 누락이 아니라 완전한 반례 두 개를 제시한다.

### 6.1 additive-offset 반례

정지해에서

$$
\rho_x=M^4v_D(q_{\rm ext}),
\qquad p_x=-\rho_x.
\tag{14}
$$

$C$는 식 (5)에 없으므로 정지점과 Hessian을 바꾸지 않지만 식 (14)의
에너지는 바꾼다. 특히

$$
C_0=q_{\rm ext}-\frac D2q_{\rm ext}^2
\tag{15}
$$

이면 $v_D(q_{\rm ext})=0$이다. 같은 $D,q_{\rm ext},F,M$과
$\rho_{\rm other}=M^4$에서 $C=C_0+1/4$로만 옮기면

$$
\Omega_x(C_0)=0,
\qquad
\Omega_x(C_0+1/4)=\frac{1/4}{1+1/4}=0.2.
\tag{16}
$$

`[정리: no-go]` 같은 고정점과 같은 국소 안정성이 서로 다른
energy fraction을 허용하므로 scalar 값만으로 $\Omega_x$를 정할 수 없다.

### 6.2 stress 종류 반례

`[정리: no-go]` 비영 정지 scalar의 식 (14)는 정확히 $w_x=-1$이다.
비상대론적 baryon dust는 $p_b=0$, $w_b=0$이고 conserved timelike current가
필요하다. 따라서 정지 scalar stress 자체를 baryon stress로 읽을 수 없다.
식 (2)의 real scalar에는 baryon number로 읽을 연속 Noether current도
없다.

## 7. 확률에서 energy fraction으로 가는 정확한 정리

`[정의]` event $E$의 확률을 $P(E)=q$라 하고, 양의 energy weight를 $W$라
한다. event class가 차지하는 energy-weighted fraction을

$$
\Omega_E^{(W)}
=\frac{\mathbb E[W\mathbf1_E]}{\mathbb E[W]}
\tag{17}
$$

로 정의한다.

`[정리: 필요충분]` $0<q<1$이고 $0<\mathbb E[W]<\infty$이면

$$
\Omega_E^{(W)}-q
=\frac{\operatorname{Cov}(\mathbf1_E,W)}{\mathbb E[W]}
=\frac{q(1-q)
[\mathbb E(W\mid E)-\mathbb E(W\mid E^c)]}{\mathbb E[W]}.
\tag{18}
$$

따라서

$$
\Omega_E^{(W)}=q
\iff
\mathbb E(W\mid E)=\mathbb E(W\mid E^c).
\tag{19}
$$

독립성은 충분조건이지만 필요조건보다 강하다. 예를 들어 조건부 평균
weight가 2와 1이면 $q=0.0486466333372$에서
$\Omega_E^{(W)}=0.0927798398254$다. 이 정리는 probability-to-energy
bridge에 정확히 어떤 대칭 또는 동역학이 필요한지 보여주지만, 그
대칭을 CE core에서 유도하지는 않는다.

## 8. conserved two-dust에서 얻는 양

`[공리: 조건부 존재구성]` baryon과 partner가 같은 평균 rest energy를
갖는 두 dust sector이고, local label event가 probability $q_{\rm ext}$로
정해지며, freeze-out 뒤 두 current가 각각 보존된다고 하자. 이때
equal-conditional-energy 조건 아래 얻는 자연스러운 조성비는

$$
f_b^{(m)}
:=\frac{\rho_b}{\rho_b+\rho_c}
=q_{\rm ext}.
\tag{20}
$$

`[정리: 항등식]` 임계밀도 분율은

$$
\Omega_b
=f_b^{(m)}\Omega_m
=q_{\rm ext}\Omega_m.
\tag{21}
$$

따라서 식 (20)은 식 (21)에서 $\Omega_m$을 제거하지 않는다.
$\Omega_b=q_{\rm ext}$까지 가려면 $\Omega_m=1$, tuned interaction 또는
별도의 critical-density boundary closure가 추가로 필요하다. 이는
matter composition과 absolute cosmic abundance를 구분하는 핵심이다.

## 9. 보존법칙의 시간의존성 no-go

`[정리]` 고정 질량의 conserved baryon dust는
$\rho_b\propto a^{-3}$이다. 평탄 GR 배경에서
$\Omega_b=\rho_b/\rho_{\rm tot}$와 총 연속방정식을 함께 쓰면

$$
\frac{d\log\Omega_b}{d\log a}=3w_{\rm tot}.
\tag{22}
$$

따라서 열린 시간구간에서 $\Omega_b=q_{\rm ext}$를 상수로 유지하려면
그 구간의 $w_{\rm tot}$가 정확히 0이어야 한다. radiation, matter와
vacuum-energy가 섞여 $w_{\rm tot}$가 변하는 배경에서는 상수 scalar
attractor와 conserved dust fraction을 같은 시간함수로 둘 수 없다.

`[산출: 조건부]` 상수 fraction을 강제로 추적하는 interacting fluid에는
유일하게

$$
Q_b=-3Hq_{\rm ext}p_{\rm tot}
=-3Hw_{\rm tot}\rho_b
\tag{23}
$$

가 필요하다. 고정 baryon mass에서 식 (23)은
$\nabla_\mu J_b^\mu\neq0$을 뜻하므로 $w_{\rm tot}\neq0$에서는 conserved
baryon current 가정과 양립하지 않는다.

## 10. 형식 지위 원장

| 대상 | 최종 지위 | 이유 |
|---|---|---|
| $D$ | `[공리: 외부 입력]` | 이번 작용에서 산출하지 않음 |
| 식 (2)--(3)의 함수형 | `[공리: 모델 선택]` | 같은 roots를 갖는 무한 family가 있음 |
| 정지조건과 고정점의 동치 | `[정리: 존재구성]` | 식 (4)--(6)의 직접 변분 |
| 작은 가지 안정성 | `[정리: 국소]` | 식 (8)--(9), 선언 영역 한정 |
| scalar EOM과 stress | `[산출]` | 선택한 작용의 변분 |
| $q_{\rm ext}$만으로 $\Omega_b$ 결정 | 활성 주장 아님 | 식 (16)의 완전 반례 |
| 정지 scalar를 baryon dust로 동일시 | 활성 주장 아님 | $w=-1$ 대 $w=0$ 반례 |
| weighted-event equality | `[정리: 필요충분]` | 식 (18)--(19) |
| $f_b^{(m)}=q_{\rm ext}$ | `[산출: 조건부 존재구성]` | 독립 A1--A7 전제 필요 |
| $\Omega_b=q_{\rm ext}\Omega_m$ | `[정리: 항등식]` | density 정의에서 직접 따름 |
| $\Omega_b=q_{\rm ext}$ 자연 유도 | `[미완성]` | current normalization, freeze-out, $\Omega_m$ 부재 |
| 새 관측량 | `[예측]` 0개 | blind protocol과 독립 holdout 부재 |

완전 반례가 있는 부모 주장은 활성 결론에서 제외했다. 회귀 테스트에는
그 재도입을 막는 반례만 남겼다.

## 11. 가능한 다음 유도 경로

이번 결과는 시도를 막는 것이 아니라 다음 작용이 무엇을 반드시 해야
하는지 좁힌다. 가장 직접적인 다음 후보는 scalar 값을 baryon density로
정의하는 방식이 아니라, 두 current의 local composition을 동적으로 만드는
방식이다.

`[미완성: 다음 후보]` 공유 four-velocity를 갖는 두 number current를

$$
J_b^\mu=n_bu^\mu,
\qquad
J_c^\mu=n_cu^\mu,
\qquad
y=\frac{n_b}{n_b+n_c}
\tag{24}
$$

로 두고 total number는 보존하되 서로 전환하도록

$$
\nabla_\mu J_b^\mu=\Gamma,
\qquad
\nabla_\mu J_c^\mu=-\Gamma
\tag{25}
$$

를 구성할 수 있다. 예를 들어 후보 반응법칙

$$
\Gamma=-\kappa(n_b+n_c)v_D'(y),
\qquad \kappa>0,\quad [\kappa]=1
\tag{26}
$$

를 **선택하면** 균일 배경에서 $\dot y=-\kappa v_D'(y)$이고 작은 가지가
국소 composition attractor가 된다. 그러나 식 (26)은 아직
`[공리: 후보 반응법칙]`이다. 미시적 Lorentz-covariant action,
detailed balance, entropy production, Standard Model baryon number와의
연결에서 산출되지 않았다.

이 경로가 닫혀도 절대밀도에는 다음 사슬이 더 필요하다.

$$
q_{\rm ext}
\longrightarrow y_b
\longrightarrow Y_b=q_{\rm ext}Y_{\rm tot}
\longrightarrow \rho_{b0}=m_bY_bs_0
\longrightarrow
\Omega_{b0}=\frac{m_bY_bs_0}{3M_{\rm Pl}^2H_0^2}.
\tag{27}
$$

식 (27)에서 total yield $Y_{\rm tot}$, freeze-out hypersurface, entropy
normalization, $m_b$와 background $H_0$는 고정점 위치와 독립인 자료다.
다음 단계의 통과 조건은 이들을 관측 $\Omega_b$에 맞춰 넣는 것이 아니라,
미시 작용과 초기상태에서 산출하고 별도 observable로 죽일 수 있게 만드는
것이다.

구체적으로 다음 run은 아래 여섯 항목을 동시에 요구해야 한다.

1. local species label과 $D$, reaction law를 정하는 microscopic action.
2. 두 current와 총 stress의 공변 보존 및 positive entropy production.
3. equal conditional energy 또는 zero covariance를 보장하는 대칭과 그
   gravity·interaction correction bound.
4. 현재 시각을 보고 고르지 않은 covariant freeze-out criterion.
5. total charge 또는 reheating entropy에서 나오는 absolute normalization.
6. 별도 dark-sector 동역학으로 계산한 $\Omega_m$과 사전등록한 kill test.

## 12. 구현과 검증

감사 승인 범위는 독립 실행 모듈
`examples/physics/density_bridge_variational_audit.py`에 고정했다. 출력은
potential 선택을 `MODEL_AXIOM`, $D$를 `EXTERNAL_INPUT`, 임계밀도 브리지를
`INCOMPLETE`, 관측 예측을 `NONE`으로 기록한다. 기본 모드는 승인된
수학만 검사해 exit 0이지만, `--require-physical-bridge`를 주면 브리지가
미완성이므로 exit 2로 실패한다.

검증 결과는 다음과 같다.

| 검사 | 결과 |
|---|---|
| 신규 전용 pytest | 13 passed |
| 신규+인접 회귀 | 34 passed |
| CE 정본 회귀 묶음 | 58 passed, 2 unrelated warnings |
| 독립 수학 원장 | `ALL DENSITY-BRIDGE MATH CHECKS PASSED` |
| Ruff | `All checks passed!` |
| CE build hook | `OK build` |
| 물리 브리지 필수 CLI | 의도된 exit 2 |

표준 CE scorecard는 이 변경 뒤에도 aggregate `CAUTION`이며, 이는
implementation integrity와 별개다. 수치 테스트의 성공은 식의 구현이
감사한 대수와 일치한다는 증거이지 자연의 baryon abundance를 유도했다는
증거가 아니다.

## 13. 재현

저장소 루트에서 다음을 실행한다.

```powershell
python _workspace\ce\cosmology-density-bridge-derivation-20260815\artifacts\verify_density_bridge_math.py
uv run --extra dev python -m pytest tests\test_density_bridge_variational_audit.py -q
python examples\physics\density_bridge_variational_audit.py
python examples\physics\density_bridge_variational_audit.py --require-physical-bridge
```

마지막 명령의 exit 2가 현재의 올바른 물리 판정이다. 자세한 명령과 원문
출력은 `31-validation.md`에 있다.

## 14. 이론 배경 출처

이번 계약은 관측 중심값을 action에 사용하지 않아 공식 관측 source lane은
`SKIPPED (관측 인용 없음)`으로 닫혔다. current와 relativistic-fluid
후보의 이론 배경은 다음 1차 문헌과 `artifacts/action-route-study.md`에
정리했다.

- J. D. Brown, “Action functionals for relativistic perfect fluids,”
  <https://arxiv.org/abs/gr-qc/9304026>.
- B. F. Schutz, “Perfect Fluids in General Relativity: Velocity Potentials and
  a Variational Principle,” <https://doi.org/10.1103/PhysRevD.2.2762>.
- A. H. Taub, “General Relativistic Variational Principle for Perfect Fluids,”
  <https://doi.org/10.1103/PhysRev.94.1468>.
- A. Pourtsidou, C. Skordis, E. J. Copeland, “Models of dark matter coupled to
  dark energy,” <https://arxiv.org/abs/1307.0458>.
- N. Andersson and G. L. Comer, “Relativistic fluid dynamics: physics for many
  different scales,” <https://arxiv.org/abs/1306.3345>.

## 15. 최종 판정

1단계에서 실제로 닫힌 것은 “고정점 식을 안정한 국소 장론 정지조건으로
구성할 수 있다”와 “확률을 energy fraction으로 바꾸는 정확한 조건이
무엇이다”라는 두 결과다. scalar 값에서 conserved baryon current와
critical-density fraction으로 바로 건너가는 부모 주장은 반례로
제외했다. 따라서 다음 단계는 값을 다시 동일시하는 식을 찾는 것이 아니라,
식 (24)--(27)의 current, reaction, freeze-out과 absolute abundance를
하나의 미시 작용에서 닫는 작업이다.
