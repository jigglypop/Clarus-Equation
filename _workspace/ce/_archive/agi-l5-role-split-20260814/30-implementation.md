# 30-implementation — L5 역할 분리 (wash + $\sigma$)

Status: COMPLETE

이 문서는 인가된 G-CODE만 기록한다. $L5$-$E1$--$E3$는 구성 검사로 잠근다. $L5$-$H1$을 정리로 올리지 않는다. 「이름 비트만이 충분통계」, 이중저장, $q$-기억을 산 주장으로 쓰지 않는다. 닫힘·유도됨·제1원리·자율·Drosophila / C. elegans·$L6$·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §10. `universe_life_kernel.py` 확장. $F_0$ 재복사 없음.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | wash, $\sigma$ 게이트 (L5.2)--(L5.3), `WashedRoleSplit`. $W=I$, 본체 $(S,A)=(L,R)$ |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 신규 공개 이름 lazy export |
| `tests/test_l5_role_split.py` | $E1$--$E3$ 구성 잠금. $H1$은 unfinished 표기만 |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. 등록 시작 $(1/2,49/99,3/4)\in U_0\times\{3/4\}$. $\kappa=1/4$, $T=32$.
2. wash는 두 복사를 같은 시작으로 되돌린다. 이름 $\sigma$는 정육면체 상태가 아니라 wash 밖 비트로 남는다.
3. $\sigma=1$이면 $\beta$ 구동은 보통 $L4$. $\sigma=0$이면 $u^{\mathrm{A}}=0$. 센서는 항상 $u_I(e^{\beta})$.
4. $L5$-$E1$: wash+$\sigma$ 판독 $o^{\mathrm{A}}$는 $\tau^{(1)}$에서 $1$, $\tau^{(2)}$에서 $0$. $u=1$은 선행 $U_0$ 헐 인용. $u=0$은 닫힌 $B_{\mathrm{c}}$ 한 스텝 소멸, $1-\lambda(1-b)\le-53/297$.
5. $L5$-$E2$: no-store wash는 같은 시작·같은 $u^{\mathrm{A}}=u_I(e^{(2)})$. 판독이 같다. 공통 값은 채점하지 않음.
6. $L5$-$E3$: $\{\tau^{(1)},\tau^{(2)}\}$ 위 연산자로 역할 분리와 no-store는 다르다.
7. $L5$-$H1$은 unfinished. $\tau^{(1)}$의 $m=0$ 흡수는 적고, $\tau^{(2)}$ 둘째 창은 $U_0$ 시작이 아니라 채점하지 않음.

한 점의 정확한 $F^{32}$ Fraction 궤적은 쓰지 않는다 ($q$-맵 3차, 비트 폭발).

## 2. 구현하지 않은 것 / 주장하지 않은 것

- V15–V18b, `delayed_linear_credit`, `covariant_metric_flow`, `unified_metric`, `runtime`, BrainRuntime import 없음.
- `docs/7_AGI/` 편집 없음.
- $L5$-$H1$을 통과 정리로 잠그지 않음.
- 「이름 비트만이 충분통계」, 이중저장 곱, $q$-기억을 승자 정리로 쓰지 않음.
- 자율 $A$, Drosophila / C. elegans, $L6$, AGI 선포 없음.
- `drive` 기본값 $1$ 유지. 기존 $L0$--$L4$ 호출은 그대로.

## 3. 불변식

- 정본 5계층을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트. $F_0$는 기존 모듈에 남김.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음 (stdlib만). 테스트 import는 파일 상단. 선행 헐은 `tests/test_l3_ne2_open_set.py`.
- `drive` 기본값 $1$. 기존 $L0$--$L4$ 호출은 그대로.

## 4. 검증 명령

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
