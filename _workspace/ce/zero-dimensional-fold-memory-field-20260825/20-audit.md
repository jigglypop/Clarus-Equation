# 20-audit — persistent spatial-0D carrier와 단일 환경장 형식 감사

Status: COMPLETE

Gate: PASS

이 판정은 최신 carrier-centered `11-math.md`와 `12-routes.md`의 안정된 snapshot에만 적용한다. 이전 감사문이 event-deposit 경로를 중심처럼 기록했던 P1 문서 불일치는 아래 지위표로 교체하여 해소했다. 수학적 P0 반례는 남지 않았지만, quantum retention, stress와 dark-sector bridge는 미완성이다.

## 1. 활성 중심 경로의 지위

| ID | 명제 | 형식 지위 | 판정 근거 |
|---|---|---|---|
| `FOLD-CARRIER-01` | 각 공간 절편의 carrier support는 0D이고, 지속 carrier의 시공간 support는 worldline이다 | [정의] | 공간 support와 시공간 history의 차원 분리 |
| `FOLD-PUSH-01` | $\mu_{F,t}=(F_t)_*(w\nu_{\rm ns})$로 비선택 history를 carrier measure에 보낸다 | [공리: 물리 사상] | weighted pushforward는 정의되나 $F_t,w$의 물리 선택은 미도출 |
| `FOLD-CARRIER-FIXED-01` | 활성 최소모형의 $\mu_F=\sum_jw_j\delta_{\mathbf X_j}$는 fixed/quenched carrier 배경이다 | [공리: 모형 선택] | carrier를 독립 상태로 진화시키지 않음 |
| `FOLD-VOLTERRA-01` | 유일한 동적 상태 $\chi=\psi/\psi_s$는 finite-width, nonnegative, retarded kernel을 통해 carrier activation을 매개한다 | [공리: 모형 선택] | one-state-field의 정확한 범위 |
| `FOLD-NETWORK-01` | finite carrier와 exponential memory에서 Volterra 식은 유한 network ODE로 정확히 환원된다 | [정리: 조건부] | convolution auxiliary variable의 정확한 미분 |
| `FOLD-BOOTSTRAP-01` | $W\ge0$에서 $\mathcal B=A\rho(W)$가 zero-branch threshold를 정한다 | [정리: 조건부] | Perron mode 선형화 |
| `FOLD-FIXED-01` | constant-row-sum network와 $\mathcal B>1$에서 $\chi_*=\mathcal B-1$이고 국소 안정이다 | [정리: 조건부] | full Jacobian spectrum과 $A\sigma'(\chi_*)\rho(W)=1/\mathcal B<1$ |
| `FOLD-DELAY-01` | $A\sigma'(\bar\chi)\rho(W)<1$은 임의의 비음 propagation delay에 대한 우반평면 root 부재의 충분조건이다 | [정리: 조건부] | characteristic equation의 modulus bound |
| `FOLD-SEED-01` | $\mathbf b=0$이고 initial history가 0이면 $\boldsymbol\chi\equiv0$이다 | [정리/제약] | threshold는 seed 생성 법칙이 아님 |
| `FOLD-EVENT-ALT-01` | $N_\psi=N[\psi,\Pi]$인 Hawkes/event 모형은 새 fold가 생성되는 경우의 보조 경로다 | [공리: 대안 모형] | persistent carrier 중심 경로와 상태·gain 분리 |
| `FOLD-STRESS-01` | carrier/source/reservoir를 포함한 covariant total stress | [미완성] | $\chi$의 microscopic action과 carrier stress 부재 |
| `FOLD-DARK-01` | $\chi$ 또는 $\mu_F$의 암흑물질·암흑에너지 동일성 | [미완성] | abundance, perturbations, Einstein--Boltzmann observables 부재 |

## 2. 활성 최소식

비선택 history에서 carrier로 가는 후보 map은

$$
\mu_{F,t}(B)=\int_{\Gamma_{\rm ns}}w(\gamma)
\mathbf1_B(F_t(\gamma))\nu_{\rm ns}(d\gamma)
$$

이다. 이 weighted pushforward는 수학적으로 잘 정의될 수 있지만, 표준 양자역학이 실제 공간 carrier를 만든다는 정리는 아니다. atomic fixed-comoving 특수화에서

$$
\mu_F(d^3y)=\sum_{j=1}^n w_j\delta_{\mathbf X_j}(d^3y),
\qquad w_j\ge0
$$

를 고정하고 유일한 동적 환경 상태를 $\chi=\psi/\psi_s$로 둔다. 활성 식은

$$
\chi(t,\mathbf x)=b(t,\mathbf x)+A\int_{t_i}^{t}ds
\int_{\Sigma_s}K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma(\chi(s,\mathbf y))\mu_F(d^3y),
$$

$$
\sigma(\chi)=\frac{\chi}{1+\chi}.
$$

$K^F_{\ell,R}$에는 finite width $\ell>0$, nonnegativity, retarded support, local boundedness와 필요한 summability를 요구한다. strict delta는 carrier support의 표기일 뿐 무규제 local coupling의 허가가 아니다.

## 3. 정확한 유한망 환원과 안정성

exponential memory $h_\tau(u)=\tau^{-1}e^{-u/\tau}\Theta(u)$와 finite carrier에서 위 Volterra 식은 상수 seed에 대해 정확히

$$
\tau\dot{\boldsymbol\chi}
=-(\boldsymbol\chi-\mathbf b)+AW\sigma(\boldsymbol\chi)
$$

로 환원된다. $W\ge0$, $W\mathbf1=w\mathbf1$, $\mathcal B=Aw=A\rho(W)>1$이면

$$
\chi_*=\mathcal B-1,
$$

$$
\operatorname{spec}J_*
=\left\{\frac{-1+A\sigma'(\chi_*)\lambda_k(W)}{\tau}\right\}_k,
\qquad
A\sigma'(\chi_*)\rho(W)=\frac1{\mathcal B}<1.
$$

따라서 모든 Jacobian mode는 국소 안정이다. propagation delay를 보존한 characteristic equation

$$
(1+\tau z)v_i=A\sigma'(\bar\chi)
\sum_jW_{ij}e^{-zd_{ij}}v_j
$$

에서도 $A\sigma'(\bar\chi)\rho(W)<1$이면 $\operatorname{Re}z\ge0$ root가 없다는 충분조건이 성립한다. 이 결과는 row-sum 조건 없는 임의 network의 positive fixed-point 존재나 전역 안정성을 주장하지 않는다.

## 4. 유지되는 반례와 no-go

1. strict point coupling은 finite-width regulator 없이는 local quadratic quantity와 self-energy가 발산한다.
2. 감쇠 없는 양의 additive field memory는 finite stationary state를 만들지 못한다. 이는 고정 carrier $\mu_F$가 남고 activation $\chi$가 relax하는 현재 모형을 금지하지 않는다.
3. canonical real scalar 하나의 정적 finite-energy 0D lump는 Derrick scaling에서 안정하지 않다.
4. prescribed source가 있으면 field-only stress는 $\nabla_\mu T^\mu{}_{\nu}=-J\nabla_\nu\phi$로 독립 보존되지 않는다.
5. $\mathcal B>1$이어도 zero seed와 zero history에서는 zero solution이 유지된다.

따라서 strict point, permanent additive activation, closed canonical scalar-only, threshold-as-spontaneous-seed라는 부모 주장들은 활성 문서에서 보존하지 않는다.

## 5. event 생성 보조 경로의 격리

$N_\psi=N[\psi,\Pi]$와 Hawkes gain $\mathcal R=A\beta\tau$는 새 fold event가 계속 생성되는 대안 경로에만 속한다. 중심 carrier gain $\mathcal B=A\rho(W)$와 합치지 않는다. event 경로의 finite-volume nonexplosion 및 deterministic closure 결과는 좁은 조건부 정리로 보존하지만, persistent carrier의 생성 법칙이나 dark bridge로 사용하지 않는다.

## 6. 남은 미완성

1. $F_t,w$를 정하는 instrument/decoherent-history 기반 quantum retention map.
2. atomic image, covariance, no-double-counting 및 carrier 생성/활성화 구분.
3. $\chi$의 microscopic action 또는 CP/open-system completion.
4. carrier, environment와 reservoir를 합친 covariant stress conservation.
5. dark-sector absolute abundance, perturbations, lensing 및 Einstein--Boltzmann observables.

## 7. 우선순위 판정

- P0: 없음.
- P1: 이전 event-centered 감사문의 snapshot 불일치를 이 문서에서 해소함.
- P2: 없음.

통과한 것은 persistent carrier + one causal saturating state field의 조건부 수학이다. quantum origin과 dark identity는 통과 범위 밖이다.
