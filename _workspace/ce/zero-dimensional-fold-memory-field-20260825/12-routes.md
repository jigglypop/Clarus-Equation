# 12-routes — 단일 환경장 경로 비교

Status: COMPLETE

## 경로 판정표

| 경로 | 한 propagating field | 인과적 자기흥분 | 유한 정상상태 | 닫힌 에너지 | 판정 |
|---|---:|---:|---:|---:|---|
| R0 persistent 0D carrier + Volterra field | 동적장 하나 | 예 | 포화 branch에서 조건부 예 | 아니오, carrier/reservoir 필요 | **활성 중심 후보** |
| R1 선형 retarded Hawkes field | 예 | 예 | $\mathcal R<1$에서만 | 아니오, source 필요 | 좁은 정리로 보존 |
| R2 포화 retarded event-memory field | 상태장 하나 | 예 | deterministic closure에서 예 | 아니오, noise/source/reservoir 필요 | 새 fold 생성의 보조 후보 |
| R3 canonical real scalar static lump | 예 | 사건 법칙 없음 | 안정 lump 없음 | 예 | 부모 주장 제거 |
| R4 permanent additive trace | 예 | 예 | 없음 | 별도 장부 없음 | 제거 |
| R5 complex Q-ball | complex field 하나 | 자동 아님 | 조건부 가능 | action에 포함 가능 | 대안, 미도출 |
| R6 oscillon | real field 하나 | 자동 아님 | 장수명만 가능 | action에 포함 가능 | 대안, 미도출 |
| R7 higher-gradient/nonlocal field | 형식상 예 | 선택에 의존 | 가능성 | 작용에 의존 | 대안, 안정성 미증명 |

## R0. persistent spatial-0D carrier + one Volterra field

비선택 history subprobability $\nu_{\rm ns}$를 persistent carrier로 보내는
weighted pushforward를 별도 물리 사상으로

$$
\mu_{F,t}(B)=\int_{\Gamma_{\rm ns}}
w(\gamma)\mathbf1_B(F_t(\gamma))\nu_{\rm ns}(d\gamma)
$$

라고 채택한다. 이는 표준 양자역학의 자동 source rule이 아니며 total mass도
에너지나 $\Omega$가 아니다. 그 atomic, fixed-comoving 특수화에서 공간 절편마다
점으로 남는 접힘을

$$
\mu_F(d^3y)=\sum_jw_j\delta_{\mathbf X_j}(d^3y)
$$

로 두고, 유일한 동적장을 $\chi=\psi/\psi_s$로 둔다.

$$
\chi(t,\mathbf x)=b(t,\mathbf x)+A
\int_{t_i}^{t}ds\int
K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\frac{\chi(s,\mathbf y)}{1+\chi(s,\mathbf y)}
\mu_F(d^3y).
$$

$\mu_F$는 persistent하지만 동역학 변수가 아닌 quenched carrier다. 따라서
접힘 위치·가중치를 독립적으로 진화시키지 않는 범위에서 실제 상태장은 하나다.
finite-width form factor, nonnegative retarded kernel, finite initial history와
carrier summability를 요구한다.

exponential memory와 유한 carrier에서는 이 식이 정확히

$$
\tau\dot{\boldsymbol\chi}
=-(\boldsymbol\chi-\mathbf b)+AW\sigma(\boldsymbol\chi)
$$

로 환원된다. $W\ge0$, $W\mathbf1=w\mathbf1$이면

$$
\mathcal B=A\rho(W)=Aw,
\qquad
\chi_*=\mathcal B-1\quad(\mathcal B>1)
$$

이고, 전체 Jacobian mode는

$$
\operatorname{spec}J_*
=\left\{\tau^{-1}
[-1+A\sigma'(\chi_*)\lambda_k(W)]\right\}_k
$$

다. $A\sigma'(\chi_*)\rho(W)=1/\mathcal B<1$이므로 양의 uniform branch는
국소 안정하고, 이 부등식은 arbitrary nonnegative propagation delay에서도
우반평면 root를 배제하는 충분조건이다. 다만 seed는 별도로 필요하고,
$\mu_F$의 생성, quantum map, stress tensor와 암흑부문 동일성은 미완성이다.

## R1. 선형 retarded field

$$
\psi=A K_R*N,\qquad
\lambda=\lambda_0+\beta\psi
$$

는 과거 0D deposit이 한 장에 남아 다음 deposit rate를 올리는 가장 직접적인
수학 모형이다. 정확한 평균 Volterra 식과

$$
\mathcal R=A\beta\tau
$$

를 준다는 좁은 정리는 보존한다. 그러나 $\mathcal R<1$에서는 유한 seed가
소멸하고, $\mathcal R>1$에서는 finite stationary mean을 잃는다. 따라서
“선형식만으로 안정하고 영구적인 자기실행장”이라는 부모 주장은 제거한다.

## R2. 포화 retarded memory field

$$
\lambda=\lambda_0+
\frac{\beta\psi}{1+\psi/\psi_s}
$$

와 finite memory를 함께 두면 homogeneous deterministic closure에서
$\mathcal R>1$일 때

$$
\psi_*=\psi_s(\mathcal R-1),
\qquad
J_*=-\frac{\mathcal R-1}{\mathcal R\tau}<0
$$

인 안정한 양의 고정점이 생긴다. 실제 stochastic law는 primitive Poisson
random measure $\Pi$에서

$$
N_\psi(dV_y)=
\int_0^\infty
\mathbf1_{\{z\le\lambda(y\mid\psi_{y^-})\}}
\Pi(dV_y,dz)
$$

로 event record를 만들고 이를 retarded $\psi$ 식에 되넣는다. 따라서 유일한
propagating state field는 $\psi$이며 $N_\psi$는 독립된 두 번째 장이 아니다.
그러나 primitive noise와 source/reservoir가 필요하므로 완전히 자율적인
one-field Hamiltonian theory라고 부르지는 않는다.

다만 다음 여섯 조건은 숨길 수 없다.

1. $\ell>0$인 유한 smearing,
2. $K_{\ell,R}\ge0$인 positivity-preserving causal response,
3. $\operatorname{supp}S_\ell(\cdot,y)\subseteq J^+(y)$인 causal smearing,
4. finite memory 또는 동등한 negative feedback,
5. strictly retarded predictable intensity,
6. source/reservoir를 포함한 total stress bookkeeping.

조건 1--3의 공집합 가능성은

$$
K_{\ell,R}^{\rm w}(t,r)=
\Theta(t-\ell/c_\psi)e^{-(t-\ell/c_\psi)/\tau}
\frac{3\Theta(c_\psi t-r)}{4\pi(c_\psi t)^3},
\qquad 0<c_\psi\le c
$$

가 배제한다. 이 witness는 nonnegative이고 causal cone 안에만 있으며
spacetime 적분이 정확히 $\tau$다. 다만 comoving rest frame의 유효 kernel이지
일반 공변 미시작용의 Green 함수는 아니다.

또한 pathwise jump 식과 deterministic closure를 분리해야 한다. 포화 law는
bounded·Lipschitz이므로 finite volume에서 nonexplosive path를 구성할 수 있지만,
$\mathcal R>1$ branch의 nontrivial global stationary measure와 stochastic
almost-sure survival은 아직 정리가 아니다. 따라서 R2의 지위는
one-state-field open stochastic 유효모형이며 양자중력 정리가 아니다.

## R3. canonical real scalar static lump

정적 scaling은

$$
E(\lambda)=\lambda^{-1}T+\lambda^{-3}U,
$$

$$
E'(1)=0\Longrightarrow U=-T/3,
$$

$$
E''(1)=-2T<0
$$

를 준다. 이는 정의역 안의 완전한 반례다. 따라서 “canonical real scalar
하나가 안정한 정적 spatial-0D fold들을 만든다”는 부모 주장은 활성 후보에서
삭제한다.

## R4. permanent additive trace

각 activation의 additive 흔적이 감쇠하지 않으면 kernel norm $\tau$가 무한대이고
$\mathcal R=A\beta\tau$도 발산한다. intensity만 포화해도 기억장 자체는
선형으로 계속 자란다. finite stationary state를 원하면 relaxation,
annihilation, resource depletion 또는 bounded order-parameter map이 반드시
필요하다. R0에서는 carrier $\mu_F$가 영구히 남고 장의 activation은 relax한다.
따라서 persistent fold와 permanent additive memory를 동일시하지 않고도 사용자의
“남는다”를 문자 그대로 유지할 수 있다.

## R5--R7. 대안

복소장 Q-ball은 conserved $U(1)$ charge와 시간 의존 위상으로 Derrick의 정적
전제를 피할 수 있다. oscillon은 real scalar에서도 장수명 localized state를
줄 수 있지만 일반적으로 metastable하다. higher-gradient 또는 nonlocal
term은 scaling balance를 바꿀 수 있지만 ghost, Lorentz symmetry, 초기값
well-posedness를 새로 감사해야 한다.

이 대안들은 “0D 접힘들이 하나의 환경장을 통해 서로를 실행한다”는 carrier
메커니즘을 자동으로 주지 않는다. 현재 단계에서 R0보다 적은 공리로 질문을
직접 구현하지 못하므로 활성 중심 경로로 선택하지 않는다.

## 최종 경로

사용자 문장의 문자적 활성 후보는

$$
\boxed{
\text{persistent spatial-0D carrier }\mu_F
\to
\text{one causal saturating state field }\psi
\to
\text{other carriers' activation}
}
$$

다. 새 fold 자체가 생성된다는 별도 뜻에서만 R2의 event-deposit 경로를 쓴다.
strict point coupling, permanent additive activation memory, closed canonical scalar-only라는
세 부모 요구는 각각 UV 발산, 정상상태 부재, Derrick/energy 반례 때문에
동시에 유지할 수 없다. “하나의 장”은 현재 단계에서 하나의 propagating
state field를 뜻한다. fixed carrier와 reservoir가 없다는 뜻으로 확장하지 않는다.
