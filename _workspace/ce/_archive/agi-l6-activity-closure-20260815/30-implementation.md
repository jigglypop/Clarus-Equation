# 30-implementation — L6 활동 한 스텝 (등록 쌍)

Status: COMPLETE

이 문서는 인가된 G-CODE만 기록한다. $L6$-$E1$--$E3$는 구성 검사로 잠근다. $L6$-$H1$을 새 헐 정리로 올리지 않는다. 「활동이 $U_0$ 점유에 필요하다」, 재귀 $u_t=\sigma m_t$를 산 주장으로 쓰지 않는다. 닫힘·유도됨·제1원리·자율·zebrafish·$L7$·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §10. `universe_life_kernel.py` 확장. $F_0$ 재복사 없음. 새 서브시스템 클래스 없음.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 등록 쌍 (L6.1), 잠금 분수 $(m',b')$, `registered_activity_pair`, `activity_readout` |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 신규 공개 이름 lazy export |
| `tests/test_l6_activity_closure.py` | $E1$--$E3$ 구성 잠금. $H1$은 $U_0$ 소속 인용만 |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. 등록 쌍 $P_{\star}=(1/2,49/99,3/4)$, $P_{\circ}=(7/15,49/99,3/4)$. 둘 다 $U_0\times\{3/4\}$. $\sigma=1$, $u=1$, $\kappa=1/4$.
2. 한 스텝 판독은 $(m',b')$. $q'$는 버린다. 맵은 선행 `source_hybrid_step`.
3. $L6$-$E1$: $F(P_{\star})=(7187/12672,491/990)$, $F(P_{\circ})=(16891/29700,133/270)$. $\Delta m=-1487/950400\neq 0$, $\Delta b=1/297\neq 0$.
4. $L6$-$E2$: 비트 예측기는 $\sigma$만 본다. 이 쌍에서 $\sigma$의 상은 $\{1\}$. 한 값으로 두 참 다음 상태를 동시에 맞추지 못함.
5. $L6$-$E3$: 유한 $\{P_{\star},P_{\circ}\}$ 위 $\{P\mapsto(m',b')\}$와 $\{\sigma\mapsto$ 한 쌍$\}$은 다름.
6. $L6$-$H1$은 인용. 두 점이 $U_0$에 있어 선행 $O$-$E1$이 적용된다. $T=32$ Fraction 궤적 없음. 새 헐 없음.

한 점의 정확한 $F^{32}$ Fraction 궤적은 쓰지 않는다.

## 2. 구현하지 않은 것 / 주장하지 않은 것

- V15–V18b, `delayed_linear_credit`, `covariant_metric_flow`, `unified_metric`, `runtime`, BrainRuntime import 없음.
- `docs/7_AGI/` 편집 없음.
- $L6$-$H1$을 새 헐 정리로 잠그지 않음.
- 「활동이 $U_0$ 점유에 필요하다」를 산 주장으로 쓰지 않음.
- 재귀 $u_t=\sigma m_t$를 승자 정리로 쓰지 않음.
- 자율 $A$, zebrafish, $L7$, AGI 선포 없음.
- `drive` 기본값 $1$ 유지. 기존 $L0$--$L5$ 호출은 그대로.

## 3. 불변식

- 정본 5계층을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트. $F_0$는 기존 모듈에 남김.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음 (stdlib만). 테스트 import는 파일 상단. $U_0$ 기하는 `tests/test_l3_ne2_open_set.py`.
- `drive` 기본값 $1$. 기존 $L0$--$L5$ 호출은 그대로.

## 4. 검증 명령

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
