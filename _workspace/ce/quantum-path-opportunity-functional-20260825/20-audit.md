# 20-audit — 비선택 경로 기회비용 형식 감사

Status: COMPLETE

Gate: PASS

## 1. 형식 판정

| 항목 | 지위 | 판정 |
|---|---|---|
| $C_I(o)=-\sum_{a\ne o}p_a\ln p_a$ | [정의] | 유한 coarse-graining에서 일관적 |
| $C_I=p_U[-\ln p_U+H(q)]$ | [정리] | 직접 대입으로 성립 |
| $p_U\to0$에서 $C_I\to0$ | [정리] | 유한 outcome에서 성립 |
| $k_BT D(\rho_U\|\gamma_T)$ | [정리: 조건부] | $H,T$, Gibbs reference와 protocol 필요 |
| $C_E$, $C_E^+$ | [정의: 반사실적 효용] | energy/value와 양의 부분 선택 필요 |
| influence action | [정리: 조건부 구성] | 미시 action과 regulator 필요 |
| $C\to S_{\rm opp}\to T_{\mu\nu}$ | [공리: 물리 사상] | $\epsilon_*$와 covariant action 미도출 |
| dark identity/abundance | [미완성] | 현재 식에서 나오지 않음 |

## 2. 제거한 부모 주장

다음 부모 주장은 활성 결론에서 제거한다.

1. probability, entropy 또는 relative entropy만으로 actual energy 차원이 생긴다.
2. 비선택 outcome weight가 선택 branch의 Einstein source에 자동 가산된다.
3. scalar 기회비용 하나가 pressure와 전체 stress tensor를 유일하게 정한다.
4. $-\hbar\ln Z$가 시간·온도 scale 없이 energy다.
5. Landauer bound가 모든 measurement 또는 미실현 경로에 저장된 energy다.
6. continuum path entropy가 regulator, reference와 coarse-graining 없이 고유하다.

## 3. 수학·차원 감사

두 outcome $p=(0.8,0.2)$에서

$$
H(p)=0.5004024235381879,
$$

$$
-\ln p_U=1.6094379124341003,
$$

$$
C_I=-0.2\ln0.2=0.3218875824868201
$$

이고 singleton conditional entropy는 0이다. full delta와 uniform reference의
KL은 $\ln2$이며 reference 변경에 따라 값이 바뀐다. $E_1-E_0=\Delta$를
외부에서 줄 때만 expected energy regret가 $0.2\Delta$가 된다.

$C_I,H,D$는 무차원이고, $k_BT D$는 energy, $-\hbar\ln(Z/Z_{\rm ref})$는
action이다. 후자를 energy로 바꾸려면 inverse-time 또는 thermal scale이
필요하다. 수치 검증기 14개 항목은 허용오차 $10^{-12}$에서 모두 성립했다.

## 4. 중력·보존 경계

$$
T_{\mu\nu}^{\rm opp}
=-\frac2{\sqrt{-g}}\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}
$$

를 정의하려면 $S_{\rm opp}$가 먼저 있어야 한다. $C(x)$를 외부 scalar로 두고
$V=\epsilon_*f(C)$만 가정하면 $T_{\mu\nu}=-Vg_{\mu\nu}$지만, $C$가 변할 때
$\nabla^\mu T_{\mu\nu}=-\partial_\nu V$이므로 field/apparatus/reservoir까지 포함한
full action 없이는 보존되지 않는다.

## 5. 잔여 결함

- P0: 0
- P1: 0
- P2: quantum instrument의 CP trace-nonincreasing 및 합 channel의
  trace-preserving 조건을 명시하면 타입이 더 완결된다.
- P2: continuum $\mu_C$의 실제 coarse-graining/reference 구성은 미완성이다.
- P2: $C\to T_{\mu\nu}$ bridge의 $\epsilon_*$, action과 reservoir가 미도출이다.
- P3: 정본 통합 시 개별 Claim ID가 필요하다.

잔여 항목은 활성 주장을 정보 shadow-price와 조건부 thermal/EFT 경로로 제한하면
Gate를 차단하지 않는다.
