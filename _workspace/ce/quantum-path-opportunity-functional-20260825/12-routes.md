# 12-routes — 기회비용 경로 비교

Status: COMPLETE

## 경로 판정표

| 경로 | 정의됨 | energy 차원 | 중력 stress | 판정 |
|---|---:|---:|---:|---|
| R1 비선택 weighted surprisal $C_I$ | 유한 coarse-graining에서 예 | 아니오 | 아니오 | **활성 정보 readout** |
| R1a aggregate surprisal $-\ln p_U$ | $p_U>0$에서 예 | 아니오 | 아니오 | 중심 총량에서 제외 |
| R1b conditional entropy $H(q)$ | $p_U>0$에서 예 | 아니오 | 아니오 | 내부 다양성 보조량 |
| R1c relative entropy $D(q\|r)$ | $q\ll r$에서 예 | 아니오 | 아니오 | reference 명시 시 비교량 |
| R2 $k_BT D(\rho_U\|\gamma_T)$ | thermal setup에서 예 | 예 | 자동 아님 | **조건부 free-energy 경로** |
| R3 energy regret | $E_a$ 지정 시 예 | 예 | 자동 아님 | 효용/반사실적 보조량 |
| R4 influence action | microscopic action 지정 시 예 | action | metric variation 필요 | 물리적 기억의 유력 후속 경로 |
| R5 covariant opportunity EFT | $\epsilon_*$와 action 채택 시 예 | 예 | 예 | 물리 사상 공리, 미도출 |
| 직접 probability $\to$ dark energy | 아니오 | 차원 불일치 | 없음 | 부모 주장 제거 |

## R1 선택

비선택 총량은

$$
C_I(o)=-\sum_{a\ne o}p_a\ln p_a
=p_U[-\ln p_U+H(q)]
$$

로 고정한다. 이 값은 nonnegative이고 $p_U\to0$에서 0으로 간다. 기존 carrier
pushforward의 weight에 곱해 “기회비용으로 장식된 carrier measure”를 만들 수
있다. 그러나 instrument/coarse-graining dependent인 무차원 bookkeeping이다.

$-\ln p_U$는 비선택 집합 자체의 surprisal이지만 대안이 사라질수록 발산하므로
기회비용 총량으로 채택하지 않는다. $H(q)$는 내부 다양성을 보지만 singleton
비선택 집합에서는 0이므로 $p_U$를 대체하지 못한다. $D(q\|r)$는 사전 고정한
reference에 대한 비교에는 유용하지만 reference-free invariant가 아니다.

## R2 조건부 승격

비선택 조건부 상태 $\rho_U$, Hamiltonian, bath $T$와 Gibbs state를 고정하면

$$
E_{\rm opp}^{\rm cond}
=k_BT D(\rho_U\|\gamma_T)
$$

는 nonequilibrium free-energy excess다. 이 경로가 “에너지 없는 에너지”라는
직관에 가장 가까운 물리적 번역이다. 다만 실제 energy 차원은 $k_BT$가 공급하고,
extractable work의 의미는 preparation과 allowed operations에 의존한다.

## R4--R5 중력 경계

환경을 적분한 Feynman--Vernon influence action은 비선택·환경 자유도의 효과를
기억, dissipation과 noise로 남길 수 있다. 이 경로는 단순 outcome entropy보다
미시 동역학에 가깝지만, 일반적으로 복소·비국소이고 양의 local energy가 아니다.

중력 source로 승격하려면

$$
S_{\rm opp}=-\int\sqrt{-g}\,
V_{\rm opp}(C,\chi,\nabla\chi;\epsilon_*)d^4x,
$$

$$
T_{\mu\nu}^{\rm opp}
=-\frac2{\sqrt{-g}}\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}
$$

를 별도 채택해야 한다. 이때도 $\epsilon_*$, $C$의 dynamics와 reservoir stress는
PREDECESSOR의 0D carrier 식에서 나오지 않는다.

## 최종 경로

현재 활성식은

$$
\boxed{
\nu_{\rm ns}
\longrightarrow
\mu_C\ \text{(dimensionless information shadow price)}
}
$$

다. thermal setup이 독립적으로 고정되면

$$
\mu_C
\longrightarrow
k_BT D(\rho_U\|\gamma_T)
$$

를 조건부로 시험할 수 있다. 그 다음의

$$
k_BT D
\longrightarrow
S_{\rm opp}
\longrightarrow
T_{\mu\nu}^{\rm opp}
$$

는 아직 물리 사상과 미시 유도가 필요한 미완성 bridge다.
