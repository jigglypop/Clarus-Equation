# Source lane — kinetic flow, unified dark sector, and clock constraints

Status: LANE_COMPLETE — R2 sources mapped; full objective open

Access date: 2026-08-25

## 1. Primary-source ledger

| source | primary result used here | what it does not establish |
|---|---|---|
| Garriga and Mukhanov, “Perturbations in k-inflation,” arXiv:hep-th/9904176, Phys. Lett. B 458 (1999) 219, https://arxiv.org/abs/hep-th/9904176 | General $P(\phi,X)$ stress tensor and scalar sound speed $c_s^2=P_X/(P_X+2XP_{XX})$ | No 0D measurement or unselected-path origin |
| Scherrer, “Purely kinetic k-essence as unified dark matter,” arXiv:astro-ph/0402316, Phys. Rev. Lett. 93 (2004) 011301, https://arxiv.org/abs/astro-ph/0402316 | Exact shift-current solution; a kinetic extremum gives constant energy plus an $a^{-3}$ correction with small sound speed | Does not predict the dark-energy scale or prove quantum-branch identity |
| Arkani-Hamed, Cheng, Luty, and Mukohyama, “Ghost Condensation and a Consistent Infrared Modification of Gravity,” arXiv:hep-th/0312099, JHEP 05 (2004) 074, https://arxiv.org/abs/hep-th/0312099 | A nonzero timelike scalar velocity at a kinetic extremum can have $\rho=-p$; the low-energy completion includes a $k^4$ dispersion term | Does not make the quadratic $P(X)$ model UV complete automatically |
| Anisimov and Vikman, “The Classical Stability of the Ghost Condensate,” arXiv:hep-ph/0411089, https://arxiv.org/abs/hep-ph/0411089 | Overshoot and wrong-side evolution can enter an unstable branch | Stability must be checked along the whole trajectory, not only at the endpoint |
| Chamseddine and Mukhanov, “Mimetic Dark Matter,” arXiv:1308.5410, JHEP 11 (2013) 135, https://arxiv.org/abs/1308.5410 | A constrained scalar clock produces an integration-density behaving as pressureless dust | A multiplier construction is not automatically a healthy UV completion |
| Chaichian, Klusoň, Oksanen, and Tureanu, “Mimetic dark matter, ghost instability and a mimetic tensor-vector-scalar gravity,” arXiv:1404.4008, https://arxiv.org/abs/1404.4008 | Positive mimetic dust is a required domain; some configurations are unstable | Positivity cannot be assumed from the sign of initial data alone |
| Dutta et al., “Cosmological dynamics of mimetic gravity,” arXiv:1711.07290, https://arxiv.org/abs/1711.07290 | A mimetic scalar with a potential can conditionally pass through radiation, matter, and dark-energy eras | Phase-space viability is not a derivation from measurement theory |
| Jain and Kovtun, “Schwinger-Keldysh effective field theory for stable and causal relativistic hydrodynamics,” arXiv:2309.00511, https://arxiv.org/abs/2309.00511 | Genuine causal dissipative EFT uses doubled Schwinger--Keldysh variables and fluctuation/dissipation structure | A single real local action cannot be relabeled as an irreversible open-system law |

## 2. Frozen CE dependencies

This run consumes, without re-promoting, four earlier results.

- `../_archive/dimensionless-self-measurement-time-20260825` proves in the fixed dephasing
  model that $\theta=-\ln(1-\eta)$ is additive, that nonstationary
  self-distinguishability has a finite path length, and that the normalized
  accumulated cost is $c=1-e^{-\theta}$ in the constant-integrand special case.
- `../_archive/zero-dimensional-fold-memory-field-20260825` constructs a retarded,
  finite-width, one-environment-field carrier model, but proves that its carrier
  measure and activation are not energy density without an action.
- `../_archive/measurement-record-one-way-compatibility-20260825` proves that a strict
  external singleton, an informative record, and strict no-return signalling
  cannot be the same object. The external 0D boundary, record, and 4D field must
  remain distinct types.
- `../_archive/quantum-neighbor-bootstrap-dark-sector-20260825` proves that local
  facilitation does not supply energy and that nonselected quantum histories do
  not become gravitational sources under standard conditional quantum mechanics.

The new action is therefore a bridge axiom from those operational objects to a
covariant field, not a re-derivation of them.

## 3. Source-supported facts

For

$$
S=\int d^4x\sqrt{-g}\,P(T,X),\qquad
X=-\frac12\nabla_\mu T\nabla^\mu T,
$$

the standard field-theory results are

$$
T_{\mu\nu}=P_X\nabla_\mu T\nabla_\nu T+Pg_{\mu\nu},
\qquad
\rho=2XP_X-P,\qquad p=P,
$$

and

$$
c_s^2=\frac{P_X}{P_X+2XP_{XX}}.
$$

For shift-symmetric $P(X)$, the homogeneous equation integrates to

$$
a^3P_X\dot T=\text{constant}.
$$

A nonzero timelike kinetic extremum $P_X(X_*)=0$ can therefore have
$w=-1$ without $\dot T=0$. Near a regular quadratic extremum, the
positive-side displacement can behave as dust only in a declared small-displacement
domain.

## 4. Evidence boundary

The sources establish that kinetic-condensate and constrained-clock dark sectors are
known EFT structures. They do not establish any of the following:

- a Born-rule or decoherence derivation of $P(T,X)$;
- that unselected alternatives carry an independent positive energy;
- the map from 0D events to $T$, $X_*$, $\kappa$, or the initial current;
- the values of $\rho_\infty$ or $\Gamma$;
- nonlinear halo viability, caustic avoidance, or a full CMB/LSS likelihood.

Accordingly, source agreement licenses the mathematical route but not the proposed
microscopic identity.

## 5. R1 열린계 1차 출처 — 2026-08-26

| 1차 출처 | 표준적으로 확보되는 결과 | 이 run에서 확보되지 않는 것 |
|---|---|---|
| Hu, Paz, Zhang, *Quantum Brownian motion in a general environment: Exact master equation with nonlocal dissipation and colored noise*, Phys. Rev. D 45, 2843 (1992), [DOI](https://doi.org/10.1103/PhysRevD.45.2843) | 선형 결합의 Gaussian 환경을 적분하면 지연 반응과 잡음이 함께 들어간 정확한 비국소 영향함수를 얻는다. 초기 상관과 환경 스펙트럼이 동역학을 바꾼다. | CE의 0차원 기록이 해당 환경이라는 동일성, $\Pi_F$의 값 |
| Jana, Loganayagam, Rangamani, *Open quantum systems and Schwinger-Keldysh holograms*, JHEP 07 (2020) 242, [arXiv:2004.02888](https://arxiv.org/abs/2004.02888) | 환경을 적분한 실시간 영향함수는 Schwinger--Keldysh 상관함수로 정해지며 저주파 조건에서 확률적 기술로 바꿀 수 있다. | 임의 경계항이 실제 환경에서 유도됐다는 결론 |
| Crossley, Glorioso, Liu, *Effective field theory of dissipative fluids*, JHEP 09 (2017) 095, [arXiv:1511.03646](https://arxiv.org/abs/1511.03646) | 공변 열린 EFT에는 두 CTP 사본, 계량 변분, 보존량 대칭과 상태 조건이 필요하다. | 단순한 FLRW 배경 치환만으로 완성되는 총 Ward 항등식 |
| Glorioso, Crossley, Liu, *Effective field theory of dissipative fluids (II)*, JHEP 09 (2017) 096, [arXiv:1701.07817](https://arxiv.org/abs/1701.07817) | 열 또는 국소평형을 추가로 가정할 때 dynamical KMS가 소산·요동 관계를 제약한다. | 일반 비평형 0차원 기록에 KMS/FDR를 자동 적용하는 권한 |

표준 결과가 허용하는 주장 상한은 “명시한 Gaussian 환경을 적분하면 인과적
반응핵과 양의 잡음핵을 함께 구성할 수 있다”까지다. 초기 Gaussian 평균은
변위로 준비할 수 있지만 공분산 전체가

$$
\mathbf V+\frac{i\hbar}{2}\mathbf\Omega\succeq0
$$

를 만족해야 한다. 비열적 초기상태에는 KMS를 강제하지 않는다. 초기
system--environment 상관을 버린 factorized 상태는 초기 jolt를 만들 수 있으므로,
0차원 기록이 이미 환경과 얽혀 있다는 해석을 채택하려면 correlated initial
state의 경계항을 별도로 계산해야 한다.

## 6. R2 섭동·고차 EFT 1차 출처 — 2026-08-26

| 1차 출처 | 표준 결과 | R2에서 허용되는 사용 |
|---|---|---|
| Garriga, Mukhanov, *Perturbations in k-inflation*, [arXiv:hep-th/9904176](https://arxiv.org/abs/hep-th/9904176) | $A=P_X+2XP_{XX}$가 시간 kinetic, $B=P_X$가 공간 gradient를 정하고 $c_s^2=B/A$다. | 배경 적합과 별도로 $A>0$, $B\ge0$, 질량·metric mixing을 검사한다. |
| Arkani-Hamed et al., *Ghost Condensation and a Consistent Infrared Modification of Gravity*, [arXiv:hep-th/0312099](https://arxiv.org/abs/hep-th/0312099) | kinetic extremum 근방에 $k^4$ 분산을 주는 저에너지 completion이 가능하지만 중력과 섞인 물리적 scalar mode가 남는다. | $c_s^2\to0$의 UV gradient 보정 후보로만 사용한다. $k=0$ 질량이나 우주론 전체 안정성의 자동 해결로 쓰지 않는다. |
| Cheung et al., *The Effective Field Theory of Inflation*, [arXiv:0709.0293](https://arxiv.org/abs/0709.0293) | unitary gauge에서 $g^{00}$와 extrinsic curvature 연산자로 시간배경 scalar EFT를 조직한다. | 나이브한 $(\Box T)^2$ 대신 lapse의 고차 시간미분을 만들지 않는 ADM 연산자를 쓴다. 계수와 cutoff는 별도 입력이다. |
| Gubitosi, Piazza, Vernizzi, *The Effective Field Theory of Dark Energy*, [arXiv:1210.0201](https://arxiv.org/abs/1210.0201) | dark-energy 배경의 scalar--metric kinetic mixing과 ghost·gradient 조건은 배경 Friedmann 해와 독립적으로 계산해야 한다. | lapse·shift를 제거한 reduced quadratic matrix를 R2 최종 gate로 요구한다. |
| Crisostomi et al., *Degenerate higher order scalar-tensor theories*, [arXiv:1810.12070](https://arxiv.org/abs/1810.12070) | 공변 고차 연산자의 추가 자유도 ghost를 없애려면 degeneracy 조건이 필요하다. | covariant $k^4$ completion을 쓸 경우 자유도 수와 DHOST/degeneracy를 함께 고정한다. |

이 문헌은 섭동 계산 도구와 안정성 조건만 제공한다. 0차원 측정, 비선택
경로와 저장소의 동일성, 암흑부문 존재량을 지지하는 경험적 근거로 사용하지
않는다.

R2의 작은-$c_s$ power counting에서는 EFT 에너지와 물리 파수를 구분한다.
정준화된 cubic이 주는 에너지 cutoff가
$\Lambda_E\sim\Lambda_3c_s^{7/4}$이면 같은 on-shell mode의 물리 파수
cutoff는 $q_{\rm sc}=\Lambda_E/c_s$다. 따라서 ghost-condensate crossover
$q_\times$는 $\Lambda_E$가 아니라 $q_{\rm sc}$와 비교한다. 이 구분은 위
문헌의 분산관계와 EFT 스케일링을 현재 작용에 적용한 산출이며, 새로운
경험적 입력이 아니다.
