# 30-implementation — L3 boxed map, nonlinear LAS and $T=32$

Status: COMPLETE

이 문서는 기존 커널 옆의 기계 잠금만 기록한다. 커널 사상을 바꾸지 않았다. N-E2를 정리로 올리지 않는다. 닫힘·유도됨·AGI를 쓰지 않는다. 기계 통과는 정리 지위가 아니다.

## 1. 구현한 것

인가 범위: `20-audit.md` §11. 테스트 또는 기존 커널 옆의 좁은 helper. 새 $\kappa$ 채널 금지.

| 경로 | 역할 |
|---|---|
| `tests/test_l3_nonlinear_las.py` | $Z_\pm$, N-E3 면적 $1/10$, $Q_-$ 행합, 사전등록 $5\times5$ 구성 검사 |
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 변경 없음 |
| `docs/7_AGI/` | 편집 없음 |

잠금:

1. $Z_-=(7/18,7/16,1/4)$는 기존 `source_hybrid_step`·`UniverseKernel`에서 $F_{1/4}$의 고정점. $\widetilde m=7/9>3/4$. $\kappa=0$에서는 고정이 아님.
2. $Z_+$는 $\mathbb Q(\sqrt{18601})$ 좌표 $(49+\sqrt{18601})/324$, $(\sqrt{18601}-51)/160$, $3/4$. 테스트 파일의 최소 `Qs` helper로 $F_{1/4}(Z_+)=Z_+$와 $b=2m/(1+2m)$를 잠근다. 커널 사상은 그대로다.
3. N-E3: $q=1/2$에서 $r=r_0=9/2$, 한 스텝 소멸 면적 $1/10\ge 1/20$. 기존 `source_one_step_extinction_area`.
4. $Q_-$ 상자 $(\nu,w,u)=(1/200,1,2)$의 행합 $\|DF\|_w=16861/18000<1$, $\widetilde m$ 하한 $216807469/288000000$. $I_r$ 전칭 LAS가 아니다.
5. 사전등록 $R_0$의 $5\times5$ 격자는 구성 검사다. 점유 25/25는 증인 표본이다. 열린 집합 정리가 아니다. N-E2는 미완성.

## 2. 구현하지 않은 것 / 주장하지 않은 것

- 커널 맵·새 $\kappa$ 채널·$I_r$ 밖 값 없음.
- V15–V18b, `runtime` import 없음.
- `docs/7_AGI/` 편집 없음.
- N-E2를 정리로 승격하지 않음. L3 결합을 닫지 않음.
- 자율 $A$, L4--L8, AGI 선포 없음.

## 3. 불변식

- 정본 5계층(runtime kernel/coupling/mode/hippocampus/global)을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- $m_i=w_i=0$, $u_j x_j\to 1$ 환원 대상이 아닌 별도 호스트 모듈. 호스트 모듈 자체는 변경 없음.
- F1–F4 우회 없음.
- 저장소 로컬 import 없음. inline import 없음. 테스트 import는 파일 상단.
- 죽이는 시험 $\{1/2,1\}$을 상자 원소로 넣지 않음.
- N-E2 격자를 정리처럼 주장하지 않음.

## 4. 검증 명령

```
python -m pytest tests/test_l3_nonlinear_las.py tests/test_universe_life_kernel.py -q
```

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다. N-E2는 미완성이다.
