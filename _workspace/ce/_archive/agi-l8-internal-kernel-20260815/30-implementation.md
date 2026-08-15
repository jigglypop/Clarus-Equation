# 30-implementation — L8 내부 커널 (등록 쌍 $S$)

Status: COMPLETE

이 문서는 인가된 G-CODE만 기록한다. $L8$-$E1$--$E3$와 $L8$-$H1$은 구성 검사로 잠근다. 「비트값 $K$면 충분하다」를 산 주장으로 쓰지 않는다. 닫힘·유도됨·제1원리·자율·BrainRuntime·셋째 큐브·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §10. `universe_life_kernel.py` 위 작은 형 헬퍼. 새 서브시스템 클래스 없음. 셋째 큐브 없음. BrainRuntime 없음.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 동결 `HostTuple`, `registered_host_pair`, `internal_kernel` |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 신규 공개 이름 lazy export |
| `tests/test_l8_internal_kernel.py` | $E1$--$E3$, $H1$ 구성 잠금 |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. 호스트 $H=(t,E,Z^{\mathrm{S}},Z^{\mathrm{A}},\sigma,I)$. 등록 $S$:

$$
H_{\star}=(0,e^{(2)},P_{\star},P_{\star},1,1),
\qquad
H_{\circ}=(0,e^{(2)},P_{\circ},P_{\circ},1,1).
$$

$P_{\star}$, $P_{\circ}$는 기존 L6 점. $e^{(2)}=(0,1)$.
2. $W=I$, $I=1$이면 $u^{\mathrm{A}}=1$, $u^{\mathrm{S}}=0$. $t\mapsto t+1$. $E$와 비트는 한 스텝 고정. wash 없음.
3. 등록 $K$는 형 $H\to H$인 한 스텝 $\Phi$. `internal_kernel`은 새 맵이 아니다.
4. 센서 $u=0$: 한 스텝 소멸 $m'=0$. 부호 $1-\lambda(1-49/99)=-26/99<0$.
5. 작용 $u=1$: $(m',b')$는 기존 L6 잠금 분수.
6. $L8$-$E1$: $S$에서 $K(H)=\Phi(H)$. $K$는 $H$와 같은 칸을 낸다.
7. $L8$-$E2$: $o^{\mathrm{A}}$는 둘 다 $1$. 작용 $(m',b')$는 L6 분수로 갈림.
8. $L8$-$E3$: $S$에서 $K$와 $o^{\mathrm{A}}$는 다음 호스트 공간으로의 맵으로 같지 않다.
9. $L8$-$H1$: 비트값 맵은 $\Phi(H)$와 종류가 다르다. 형·공역 검사.

$T=32$를 돌리지 않는다. 한 점의 정확한 $F^{32}$ Fraction 궤적은 쓰지 않는다.

## 2. 구현하지 않은 것 / 주장하지 않은 것

- V15–V18b, `delayed_linear_credit`, `covariant_metric_flow`, `unified_metric`, `runtime`, BrainRuntime import 없음.
- `docs/7_AGI/` 편집 없음.
- 새 서브시스템 클래스 없음. 셋째 큐브 없음.
- 「비트값 $K$면 충분하다」를 산 주장으로 쓰지 않음.
- 자율 $A$, BrainRuntime, AGI, `AGI GO` 선포 없음.
- `drive` 기본값 $1$ 유지. 기존 $L0$--$L7$ 호출은 그대로.

## 3. 불변식

- 정본 5계층을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트. $F_0$는 기존 모듈에 남김.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음 (stdlib만). 테스트 import는 파일 상단. $U_0$ 기하는 `tests/test_l3_ne2_open_set.py`.
- `drive` 기본값 $1$. 기존 $L0$--$L7$ 호출은 그대로.

## 4. 검증 명령

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py tests/test_l7_region_loop.py tests/test_l8_internal_kernel.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
