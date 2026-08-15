# 30-implementation — $N$-$E2$ 열린 집합 점유 갈림

Status: COMPLETE

이 문서는 기존 커널 옆의 기계 잠금만 기록한다. 커널 사상을 바꾸지 않았다. $R_0$ 전체 갈림이나 횟수 갈림을 정리로 올리지 않는다. 닫힘·유도됨·자율·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §11. 테스트 또는 기존 커널 옆의 좁은 helper. 새 $\kappa$ 채널 금지.

| 경로 | 역할 |
|---|---|
| `tests/test_l3_ne2_open_set.py` | 사전등록 $U_0=\operatorname{int}(B_c)$ 기하, 1보 분기, $T=32$ 바깥 헐 부등식, 횟수 $32$ 대 $32$ |
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 변경 없음 |
| `docs/7_AGI/` | 편집 없음 |

잠금 (구성 검사. 정리 대체 증명이 아님):

1. 사전등록 기하: $U_0=\operatorname{int}(B_c)$, $B_c\subset R_0$, 중심 $(1/2,49/99)$. 동심 $U_1$은 $B_c$의 선형 $1/3$.
2. 닫힌 $\overline{U_0}=B_c$의 1보 $\widetilde m$ 범위는 $q=1/4$에서 $[1098217/1425600,\,319086769/355658688]$, 하한 $-3/4=29017/1425600>0$. $q=3/4$에서도 하한이 $3/4$ 위. 두 표지 모두 1보 분열.
3. $T=32$ 바깥 헐, $q=1/4$: $m\le 48924156634417547/125000000000000000<2/5$. $R_0$과 서로소.
4. $T=32$ 바깥 헐, $q=3/4$: $2/5\le m\le 3/5$, $4/9\le b\le 6/11$. $R_0$ 안에 들어 있다.
5. 같은 헐에서 분열 횟수는 $32$ 대 $32$. 횟수 갈림은 거짓으로 잠근다. 점유 갈림은 $U_0$ 위 구성 검사다. $R_0$ 전체 갈림이 아니고 횟수 정리가 아니다.

구간 helper는 테스트 파일에 복사했다 (분모 $10^{18}$ 바깥 반올림). 생산 모듈은 바꾸지 않았다. `growth_at_label`만 커널에서 읽어 $r(1/4)=63/16$, $r(3/4)=81/16$을 맞춘다.

## 2. 구현하지 않은 것 / 주장하지 않은 것

- 커널 맵·새 $\kappa$ 채널 없음.
- V15–V18b, `runtime` import 없음.
- `docs/7_AGI/` 편집 없음.
- $R_0\setminus\overline{U_0}$ 갈림을 정리로 쓰지 않음.
- 횟수 갈림을 정리로 쓰지 않음. $O$-$E2$ 횟수 형식은 거짓.
- 자율 $A$, L4--L8, AGI 선포 없음.

## 3. 불변식

- 정본 5계층(runtime kernel/coupling/mode/hippocampus/global)을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트 모듈. 호스트 모듈 자체는 변경 없음.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음. inline import 없음. 테스트 import는 파일 상단.
- 새 $\kappa$ 채널 없음.
- $R_0$ 전체 갈림·횟수 정리를 코드 주석에 복사하지 않음.

## 4. 검증 명령

```
python -m pytest tests/test_l3_ne2_open_set.py tests/test_l3_nonlinear_las.py tests/test_universe_life_kernel.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
