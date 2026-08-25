# 자기측정 잔여량 암흑에너지 형식 감사

Status: COMPLETE
Gate: PASS

## 감사 범위

게이트는 다음 조건부 경로에만 적용한다.

$$
\text{operational retention 공리}
\longrightarrow
\text{local scalar initial data}
\longrightarrow
\text{canonical action}
\longrightarrow
\text{FLRW/background}
\longrightarrow
\text{smooth-DE growth와 profiled BAO shape}.
$$

microscopic 0D quantum creation, 암흑에너지 절대 scale, radiative
naturalness, full Einstein--Boltzmann observable 또는 dark-energy identity는
감사 범위 밖이며 각각 `[미완성]` 또는 `[외부 보정]`으로 남는다.

## P0

주장 상한 안의 P0는 없다.

literal 누적비용 경로

$$
V_L=\rho_*(1-e^{-\Theta})
$$

는 source-free rest data에서 $\ddot\Theta<0$인 완전 반례가 있으므로
positive-canonical 자율 realization이라는 부모 주장에서는 제거됐다. 이는
채택한 잔여량 경로 $V=\rho_*e^{-\Theta}$의 반례가 아니다.

## 해결한 P1

1. $m_{\rm eff}^2>0$은 $m_{\rm eff}^2\ge0$으로 고쳤고, strict positivity는
   $\lambda>0$에만 한정했다.
2. $\lambda=0$은 유한 $f$ 작용의 모수점이 아니라 $f\to\infty$인
   $\Lambda$CDM limit control로 분리했다.
3. inflation review를 late-time minimally coupled quintessence perturbation과
   single-field dark-energy EFT의 1차 출처로 교체했다.

## 구현 단계의 필수 falsifier

- $E^2(1-q^2/6)>0$, 실제 shooting bracket의 부호변화와 $E(0)=1$.
- Friedmann constraint 및 matter/radiation/scalar conservation residual.
- grid refinement에서 $E(z)$, BAO shape, $D(z)$의 상대오차.
- $\lambda=0$ limit control의 $\Lambda$CDM 배경 재현.
- 고정점 residual 및 Jacobian eigenvalue.
- pinned DESI DR2 13-vector hash/order/covariance, $\widehat s>0$,
  profile stationary condition, $\chi^2$, AIC, BIC.

## 검산된 수학 코어

명시한 action에서 stress tensor와 Klein--Gordon 식, FLRW 밀도·압력,
Friedmann·Raychaudhuri 식은 일관된다. scalar fixed point에서

$$
w_\Theta=-1+\frac{\lambda^2}{3},\qquad
\Theta'=\lambda^2
$$

이며 acceleration은 $\lambda^2<2$, matter/radiation을 포함한 미래 안정성은
$\lambda^2<3$이다. 정준 branch는 positive kinetic coefficient와 $c_s^2=1$을
갖는다.

## Gate A 판정

수정된 안정 스냅샷에는 P0와 P1이 남아 있지 않으므로 조건부 SMQ canonical
quintessence 구현 범위를 승인한다. 위 falsifier들은 post-Gate 구현 acceptance
test이며 하나라도 실패하면 해당 수치·관측 산출을 승격하지 않는다. Gate가
열려도 operational-to-microscopic bridge와 절대 에너지 scale은
`[미완성]`/`[외부 보정]`으로 유지한다.
