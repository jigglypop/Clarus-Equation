# 학습 메트릭 곡률 Rust/CUDA 동등성 하네스

Status: `SPECIFIED / IMPLEMENTATION_INCOMPLETE`

Date: 2026-08-22

이 문서는 BA-TR11/12에서 사용한 곡률 진단식을 Rust CPU와 CUDA에서 빠르게
교차 검증하기 위한 구현 계약이다. 이미 동결된 BA-TR11/12 결과나
source-freeze를 소급 변경하지 않는다. 현재 증거는 Python/Torch `float64`
참조 구현과 그 focused test뿐이며, Rust/CUDA 구현을 검증했다는 뜻이 아니다.

## 1. 동일하게 구현해야 하는 식

입력은 학습된 행렬 $B\in\mathbb R^{4\times4}$, 외부에서 한 번 고정해 전달한
평면 $P\in\mathbb R^{4\times2}$, 상태 $u\in\mathbb R^2$다. 각 backend는
다음을 동일하게 계산한다.

$$
A=BP,\qquad z=Au,\qquad F(u)=\tanh z,
$$

$$
J=\operatorname{diag}(1-\tanh^2 z)A,\qquad g=J^\top J,
$$

$$
F_{ab}=-2\tanh z\,(1-\tanh^2z)\,A_{:a}A_{:b},
$$

$$
\Pi_\perp=I-Jg^{-1}J^\top,\qquad
\mathrm{II}_{ab}=\Pi_\perp F_{ab},
$$

$$
K=\frac{
\langle\mathrm{II}_{00},\mathrm{II}_{11}\rangle
-\langle\mathrm{II}_{01},\mathrm{II}_{01}\rangle
}{\det g}.
$$

TR12의 경로 진단까지 포함할 때는 같은 점별 $g,K$로 고정된 midpoint
quadrature의 길이 $L$, 곡률 비용 $C_K$, strain 비용과 held-out distortion을
계산한다. quadrature 점, 반지름, 방향, 순서와 누산 정밀도는 Python 기준과
동일해야 한다.

## 2. backend와 정밀도 계약

1. Python/Torch `float64`는 비교 oracle이다.
2. Rust CPU는 별도의 `f64` analytic kernel이어야 한다.
3. CUDA는 Rust가 노출하는 실제 CUDA `f64` kernel이어야 한다. Python/Torch로
   되돌아간 결과를 CUDA 결과로 기록하면 안 된다.
4. CUDA `f32`는 속도 진단에만 쓸 수 있다. 곡률의 비영성, backend parity,
   route 순위 또는 과학 판정을 통과시킬 수 없다.
5. SVD의 부호·기저 convention을 비교하지 않도록 $P$는 Python에서 한 번
   고정해 모든 backend에 같은 byte 값으로 전달한다.

작은 곡률 값에서 $I-J(J^\top J)^{-1}J^\top$의 상쇄가 크므로 `f64`가
필수다. inverse를 명시적으로 만들지 말고 선형계 solve를 사용한다. ridge나
자동 jitter로 퇴화를 숨기지 않는다.

## 3. 필수 출력

하네스는 최종 $K$만 비교하지 않는다. 각 fixture마다 다음을 직렬화한다.

- 실제 backend 식별자, device, dtype과 native-kernel 호출 횟수
- $A,F,J,g,F_{00},F_{01},F_{11}$
- $\det g$, metric eigenvalues, condition number, $\Pi_\perp$와
  $\mathrm{II}_{00},\mathrm{II}_{01},\mathrm{II}_{11}$
- `CURVATURE_DEFINED` 또는 `CURVATURE_UNDEFINED_DEGENERATE`
- 정의된 경우 $K$; TR12에서는 $L,C_K$, strain과 distortion

중간값을 출력해야 최종 $K$의 우연한 일치와 동일한 구현 오류를 구분할 수
있다.

## 4. focused acceptance matrix

| Gate | 고정 fixture | 필수 결과 |
|---|---|---|
| Python↔Rust CPU | BA-TR10의 동결 $B$, 동일 $P,u$ | 상태가 같고 모든 중간값과 $K$가 허용오차 안에서 일치 |
| Rust CPU↔CUDA | 위 fixture를 한 batch로 실행 | 실제 CUDA `f64` 호출 receipt와 수치 동등성 |
| 선형 평탄성 | full-rank $F(x)=Bx$ | `LINEAR_PULLBACK_FLAT`, $K=0$ |
| 비선형 평탄 반례 | invertible $A\in\mathbb R^{2\times2}$, $F=\tanh(Au)$ | 모든 고정점에서 $|K|\le10^{-10}$ |
| immersed 비영 곡률 | well-conditioned $A\in\mathbb R^{4\times2}$ | `CURVATURE_DEFINED`, Python/Rust/CUDA 일치 |
| 정확한 퇴화 | rank-one $A$와 all-ones $B$ | ridge 없이 세 backend 모두 `CURVATURE_UNDEFINED_DEGENERATE` |
| 경계 판정 | rank/condition cutoff의 양쪽 fixture | 세 backend의 status가 정확히 같음 |
| 독립 내재 곡률 | 별도 코드로 $g(u\pm he_i)$를 미분해 Christoffel→Riemann $K$ 계산 | analytic Gauss $K$와 수렴 오차 안에서 일치 |
| 등거리 null | output signed permutation, 대응 source rechart | $g,K$ 불변 |
| 비등거리 witness | componentwise `tanh` 앞 일반 hidden rotation | 원점 metric은 같고 비영 상태의 $K$ 차이는 재현 |
| TR12 적분 | 고정 ray·radius·midpoint | $L,C_K$, strain, distortion 동등성 |

최소 허용오차는 다음처럼 고정한다. 모든 비교는 absolute-only가 아니라 아래
mixed bound를 사용한다.

| 비교 | 중간 행렬/벡터 | $K$ 또는 $C_K$ |
|---|---:|---:|
| Python `float64` ↔ Rust CPU `f64` | `atol=1e-11`, `rtol=1e-11` | `atol=1e-9`, `rtol=1e-8` |
| Rust CPU `f64` ↔ CUDA `f64` | `atol=1e-9`, `rtol=1e-9` | `atol=1e-7`, `rtol=1e-6` |
| Gauss ↔ 독립 Riemann finite difference | 해당 없음 | `atol=1e-5`, `rtol=1e-3`, 두 $h$에서 수렴 확인 |

condition number가 독립 finite-difference fixture의 사전 고정 범위를 벗어나면
그 fixture는 수치 증거로 쓰지 않고 fail-closed한다. 개발 결과를 본 뒤
tolerance, $h$, fixture 또는 cutoff를 바꾸지 않는다.

## 5. 빠른 실행 순서

정확성 gate를 먼저 통과시킨 뒤에만 성능을 잰다.

1. Rust CPU `f64`의 flat/degenerate/한 개 immersed fixture를 실행한다.
2. 같은 fixture로 Python↔Rust 중간값 parity를 확인한다.
3. CUDA가 있으면 모든 점을 하나의 batch로 올려 kernel을 한 번 실행하고
   Rust CPU↔CUDA parity를 확인한다. sample마다 device synchronize하지 않는다.
4. 위 세 단계가 통과한 뒤 BA-TR10/11/12의 전체 동결 fixture batch를 연다.
5. 마지막에 warm-up과 반복 횟수를 고정한 CPU/CUDA wall-clock을 별도로
   기록한다. 속도는 수학적 PASS를 대신하지 않는다.

예정 focused test 경로는
`tests/test_runtime_curvature_backend_parity.py`다. 이 테스트는 backend가 없을
때 녹색 PASS를 만들면 안 된다. 상태는 다음 중 하나여야 한다.

- `RUST_NOT_IMPLEMENTED`
- `CUDA_NOT_AVAILABLE`
- `CUDA_NOT_RUN`
- `BACKEND_PARITY_FAIL`
- `BACKEND_PARITY_PASS`

Rust 또는 CUDA가 빠진 상태는 전체 native validation의 PASS가 아니다.
fallback이 발생하거나 native 호출 receipt가 0이면 즉시
`BACKEND_PARITY_FAIL`이다. source-freeze에는 Python oracle, Rust kernel,
binding, CUDA source, fixture, test와 실행 환경의 compiler/CUDA/device 정보를
모두 포함한다.

## 6. 성능 하네스

정확성 gate(§4–§5)를 통과한 backend에만 적용한다. 속도 수치는 어떤
경우에도 수학적 통과를 대신하지 않는다.

- warm-up 횟수, 반복 횟수, batch 크기를 사전에 고정하고 CPU/CUDA
  wall-clock과 fixture당 처리량을 compiler/CUDA/device 정보와 함께
  artifacts에 기록한다.
- CUDA는 전체 fixture를 단일 batch로 올리고 per-sample device
  synchronize를 금지한다 (§5와 동일).
- `BACKEND_PARITY_PASS` 이후 무거운 fixture sweep(BA-TR 전체 batch, route
  quadrature 스캔)은 Rust CPU 또는 CUDA `f64`로 실행할 수 있다. 단,
  세션마다 무작위 fixture 표본 3개 이상을 Python oracle과 재대조하고
  결과를 기록한다. 표본 재대조 실패는 즉시 `BACKEND_PARITY_FAIL`로
  강등한다.
- 직전 기록 대비 2배 이상 느려지면 성능 회귀로 P2 기록한다. 회귀 진단은
  `empirical_calibration_loop.md` §7의 경량 트랙을 따른다.
- CUDA `f32`는 여기서도 속도 진단 전용이다 (§2 조항 4 유지).

## 7. parity 실패 시 진단 순서

`BACKEND_PARITY_FAIL`이 나오면 같은 명령을 반복하지 말고
`empirical_calibration_loop.md`의 루프를 따른다. §3의 중간값 직렬화가
첫 분기 연산 식별의 근거다. tolerance·fixture·cutoff 완화는 교정이
아니며, convention 차이로 판정된 경우 고정 지점을 이 문서 §2에 추가한다.

## 8. 현재 판정

BA-TR11의 Python 식은 analytic Gauss-curvature reference로 유지한다. 현재
Rust의 `DynamicCurvature`나 Poincare/Lorentz/Klein의 상수 곡률 반환은 이
하네스의 구현으로 인정하지 않는다. Rust/CUDA native 식과 위 acceptance
test가 추가되기 전까지 상태는 `IMPLEMENTATION_INCOMPLETE`다.
