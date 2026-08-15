# 30-implementation — universe-simulator life-climb

Status: COMPLETE

이 문서는 L0 호스트와 상자 P-H1 구성만 기록한다. P-E1을 닫지 않는다. 닫힘·유도됨·AGI·최초 생명을 쓰지 않는다.

## 1. 구현한 것

인가 범위: `20-audit.md` §9. G-HOST (P-C3 구성)와 상자 P-H1 대수에 대한 G-COUPLE 코딩.

| 경로 | 역할 |
|---|---|
| `reality_stone/python/reality_stone/clarus/universe_life_kernel.py` | 저장소 로컬 import 없는 L0 모듈. stdlib만. `F_0` 로컬 복사. |
| `tests/test_universe_life_kernel.py` | G-HOST / G-COUPLE 기계 검사 |
| `reality_stone/python/reality_stone/clarus/__init__.py` | 기존 lazy optional export와 같은 방식으로 공개 이름 추가 |

상태 `U = (t, E, subsystems, phi)`. 기본값 `ν=0`, `E_★=1`, `φ=0`. 서브시스템은 `step(E)`를 받고 `E`를 쓰지 않는다. 커널은 같은 `E`를 모든 서브시스템에 주고 reward / label / marker 인자를 받지 않는다. `E≡1`이면 커널 궤적은 로컬 `F_0`과 유리수 항등.

명목 유리 모수: `r=9/2`, `λ=5/2`, `ρ=1/5`, `δ=1/10`, `s=1/2`, `μ=3/32`, `η=1`, `θ_D=3/4`, `K=1`.

상자 구성 `F_κ`: `r(q)=r_0(1+κ(2q-1))`, `r_0=9/2`, `κ ∈ I_r ∪ {0}`, `I_r=(0, 86/315)`. 기본 `κ=0`. `{1/2, 1}`은 거절한다.

## 2. 구현하지 않은 것 / 주장하지 않은 것

- P-E1 네 bullet을 정리로 닫지 않음. 비선형 LAS, `T=32` 점유, 자손수 부호 의존은 코드에 없음.
- P-H2 `ρ(q)`, `λ(q)`, 두 딸 생존, 문턱 `θ_D(q)` 없음.
- `origin_life_existence.py`를 import하지 않음. `F_0` 대수만 로컬 복사.
- V15–V18b, `runtime`, `unified_metric`, `covariant_metric_flow`, `delayed_linear_credit`, `nested_scc_tower` import 없음.
- `docs/7_AGI/` 편집 없음. dimensionless checker에 식 추가 없음.
- occupancy 삼분율, `C_strict`, BrainRuntime 배선 없음.

## 3. 불변식

- 저장소 로컬 import 없음. inline import 없음.
- 커널 5계층(runtime kernel/coupling/mode/hippocampus/global)을 건드리지 않음. canonical 상태 차원 승격 없음. STDP 없음.
- `m_i=w_i=0`, `u_j x_j→1` 환원 대상이 아닌 별도 호스트 모듈.
- `κ=0`에서 `q`는 `(m,b)` 갱신에 들어가지 않음.
- `κ=1/4`에서 `q`가 분열 전 질량에 들어감.
- `κ ∉ I_r ∪ {0}` 거절. 죽이는 시험 `{1/2, 1}`을 `I_r` 원소로 넣지 않음.
- `q=1/2`에서 `r`은 명목. 출처 소멸 쐐기 면적 `1/10 ≥ 1/20`.

## 4. 검증 명령

`python -m pytest tests/test_universe_life_kernel.py -q`

원문 출력은 `31-validation.md`. 기계 통과는 정리 지위가 아니다.
