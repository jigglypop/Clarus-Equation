# 30-implementation — L7 영역 루프 (등록 세 에포크)

Status: COMPLETE

이 문서는 인가된 G-CODE만 기록한다. $L7$-$E1$--$E3$와 $L7$-$H1$은 구성 검사로 잠근다. 「셋째 큐브가 필요하다」를 산 주장으로 쓰지 않는다. 닫힘·유도됨·제1원리·자율·mouse / CCF·$L8$·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §10. `universe_life_kernel.py` 위 작은 헬퍼. 셋째 큐브 클래스 없음. 둘째 이름 칸 필드 없음.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 과제 (L7.1), $\gamma$ 게이트 (L7.3), $\sigma\leftarrow o^{\mathrm{A}}$ |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 신규 공개 이름 lazy export |
| `tests/test_l7_region_loop.py` | $E1$--$E3$, $H1$ 구성 잠금 |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. 등록 시작 $(1/2,49/99,3/4)\in U_0\times\{3/4\}$. $\kappa=1/4$, $T=32$. $W=I$. 본체 $(S,A)=(L,R)$.
2. 과제 $\phi^{(1)}=(e^{(1)},e^{(2)},e^{(2)})$, $\phi^{(2)}=(e^{(1)},e^{(1)},e^{(2)})$.
3. $\alpha$ 뒤 $\sigma=o^{\mathrm{S}}$. $\beta$는 L5 게이트 $u^{\mathrm{A}}=\sigma\,u_I(e^{\beta})$. 그 다음 $I=o^{\mathrm{A}}(\beta)$.
4. 루프 $\gamma$ 게이트:

$$
u^{\mathrm{A}}=I\,u_I(e^{\gamma}).
$$

`loop_gate_drives`는 `role_split_drives`와 같은 곱. $\sigma$는 유지.
5. feedforward는 $\sigma$를 얼리고 $I$를 무시. $\gamma$는 $u^{\mathrm{A}}=\sigma\,u_I(e^{\gamma})$.
6. 덮어쓰기 $\sigma\leftarrow o^{\mathrm{A}}(\beta)$는 같은 이름 칸. $\gamma$ 게이트는 이름 $I$와 같다.
7. $L7$-$E1$: 루프 판독 $\phi^{(1)}$에서 $1$, $\phi^{(2)}$에서 $0$. $u=1$은 선행 $U_0$ 헐 인용. $u=0$은 닫힌 $B_{\mathrm{c}}$ 한 스텝 소멸, $1-\lambda(1-b)\le-53/297$.
8. $L7$-$E2$: frozen-$\sigma$ 판독이 두 과제에서 같다. 공통 값은 채점하지 않음.
9. $L7$-$E3$: $\{\phi^{(1)},\phi^{(2)}\}$ 위 연산자로 루프와 feedforward는 다르다.
10. $L7$-$H1$: 덮어쓰기 판독은 루프와 같다. $1$ 대 $0$. 새 몸이 아님.

한 점의 정확한 $F^{32}$ Fraction 궤적은 쓰지 않는다 ($q$-맵 3차, 비트 폭발).

## 2. 구현하지 않은 것 / 주장하지 않은 것

- V15–V18b, `delayed_linear_credit`, `covariant_metric_flow`, `unified_metric`, `runtime`, BrainRuntime import 없음.
- `docs/7_AGI/` 편집 없음.
- 셋째 `HybridState` 영역 없음. 「셋째 큐브가 필요하다」를 산 주장으로 쓰지 않음.
- 자율 $A$, mouse / CCF, $L8$, AGI 선포 없음.
- `drive` 기본값 $1$ 유지. 기존 $L0$--$L6$ 호출은 그대로.

## 3. 불변식

- 정본 5계층을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트. $F_0$는 기존 모듈에 남김.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음 (stdlib만). 테스트 import는 파일 상단. 선행 헐은 `tests/test_l3_ne2_open_set.py`.
- `drive` 기본값 $1$. 기존 $L0$--$L6$ 호출은 그대로.

## 4. 검증 명령

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py tests/test_l7_region_loop.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
