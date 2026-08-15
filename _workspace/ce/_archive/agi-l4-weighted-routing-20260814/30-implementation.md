# 30-implementation — L4 두 채널 가중 라우팅

Status: COMPLETE

이 문서는 인가된 G-CODE만 기록한다. $L4$-$E1$--$E3$는 구성 검사로 잠근다. $L4$-$H1$ 전칭과 「가중이어야 한다」를 코드에 복사하지 않는다. 닫힘·유도됨·제1원리·자율·C. elegans·$L5$·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §10. `universe_life_kernel.py` 확장. $F_0$ 재복사 없음.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 선택 `drive`$\in[0,1]$ (기본 $1$). $u=WE$. 두 복사 `RoutedTwoCopy`. $R_0$ 점유 비트 |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 신규 공개 이름 lazy export |
| `tests/test_l4_weighted_routing.py` | $E1$--$E3$ 구성 잠금. $I$와 스왑의 killing test |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. $u=1$은 기존 성장 괄호. $u=0$은 성장항만 제거. $q$-맵은 드라이브와 비결합.
2. 닫힌 $B_{\mathrm{c}}$에서 $1-\lambda(1-b)\le-53/297<0$. $u=0$이면 한 스텝 $\widetilde m=0$.
3. $W=I$: $e^{(1)}\mapsto(1,0)$, $e^{(2)}\mapsto(0,1)$. 점유 쌍 $(1,0)$ 대 $(0,1)$. $u=1$ 점유는 선행 $U_0$ 헐 인용. $u=0$은 한 스텝 소멸.
4. $A_{\mathbf 1}$: 두 플럭스 모두 $u=(1/2,1/2)$. 같은 초기값이면 점유 쌍이 같다. 공통 값은 채점하지 않음.
5. $\{e^{(1)},e^{(2)}\}$ 위 점유 연산자로 $I$와 $A_{\mathbf 1}$은 다르다.
6. Killing test: $I$와 스왑 행렬이 둘 다 가른다. 전칭이 아님.

등록 내부점: 중심 $(1/2,49/99)\in U_0$, $q_0=3/4$, $\kappa=1/4$, $T=32$.
한 점의 정확한 $F^{32}$ Fraction 궤적은 쓰지 않는다 ($q$-맵 3차, 비트 폭발).

## 2. 구현하지 않은 것 / 주장하지 않은 것

- V15–V18b, `delayed_linear_credit`, `covariant_metric_flow`, `unified_metric`, `runtime`, BrainRuntime import 없음.
- `docs/7_AGI/` 편집 없음.
- $L4$-$H1$ 전칭, 「가중이어야 한다」, 그래프 전칭을 주석·테스트에 정리로 쓰지 않음.
- 자율 $A$, C. elegans, $L5$, AGI 선포 없음.
- `UniverseKernel`의 $E$를 드라이브로 바꾸지 않음. 기존 $L0$ 규약 유지.

## 3. 불변식

- 정본 5계층을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트. $F_0$는 기존 모듈에 남김.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음 (stdlib만). 테스트 import는 파일 상단. 선행 헐은 `tests/test_l3_ne2_open_set.py`.
- `drive` 기본값 $1$. 기존 $L0$--$L3$ 호출은 그대로.

## 4. 검증 명령

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
