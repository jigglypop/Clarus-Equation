# Beyond-Horndeski 현실화 루프

## 범위

건강한 scalar-tensor 전역 no-go를 피하기 위해 beyond-Horndeski/DHOST 확장을 검토한다.
이는 현재 CE 작용에서 유도된 결과가 아니라 명시적 이론 확장이다.

## 동일 모델 원칙

| 계열 | 실제로 닫힌 부분 | 열린 부분 |
|---|---|---|
| 2018 spherical EFT | ghost/gradient no-go 회피 구조 | 미시 공변 작용과 완전 UV completion |
| 2018 명시적 공변 예 | radial non-spherical even과 odd sector의 일부 안정 조건 | 구면 even, angular even, slow tachyon; weak gravity가 GR과 다름 |
| 2021 disformal/Lovelock 계열 | regular하고 asymptotically flat한 진공 wormhole 배경 | 동일 모델의 완전 spectrum과 현실 matter frame |

한 논문의 배경해와 다른 논문의 안정 조건을 합쳐 성공한 단일 모델로 세지 않는다.
현재 동일 모델에서 모든 gate를 닫은 후보는 없다.

## DHOST 퇴화 gate

CE의 \(\alpha_2\|\nabla^2\phi\|^2\)를 시공간 물리항으로 직접 읽고 최고미분 부분을

\[
L_{\rm high}=\alpha_2\ddot q^{,2}
\]

로 축약하면

\[
\frac{\partial^2L_{\rm high}}{\partial\ddot q^2}=2\alpha_2.
\]

\(\alpha_2\ne0\)에서는 비퇴화이므로 Ostrogradsky mode를 제거하는 constraint가 없다.
DHOST가 되려면 \(\phi_{\mu\nu}\)의 전체 operator basis, metric-scalar kinetic mixing,
그리고 계수 사이의 퇴화관계를 함께 지정해야 한다. 따라서 현재 CE의 단독 2차미분
제곱항은 DHOST completion이 아니다.

## 관측 gate

GW170817/GRB170817A의 보수적 hard gate를

\[
|c_T/c-1|\le5\times10^{-16}
\]

로 둔다. 국소 throat 확장이 우주론적 bound에 즉시 배제된다고 단정할 수는 없지만,
matter가 결합하는 Jordan frame과 background 변화에도 견고한 \(c_T=1\) 관계를 동일
작용에서 증명해야 한다.

## 필요한 CE 확장 manifest

1. 전체 quadratic DHOST basis와 coefficient functions
2. 정확한 degeneracy matrix와 rank
3. matter가 최소결합하는 physical metric
4. throat background와 양쪽 GR asymptotics
5. odd, radial/angular even, \(l=0\), slow-tachyon 전체 spectrum
6. background 변화에 견고한 luminal tensor cone
7. strong-coupling cutoff가 throat curvature보다 높다는 증명
8. CE 상수에서 coefficient functions를 얻는 독립 유도

## 판정

| 명제 | 판정 |
|---|---|
| beyond-Horndeski가 Horndeski no-go를 피할 수 있음 | `DEMONSTRATED IN CONTROL MODELS` |
| 명시적 점근평탄 wormhole 배경 존재 | `DEMONSTRATED` |
| 완전 안정·GR asymptotics·luminal GW의 단일 모델 | `OPEN` |
| CE 단독 \(\alpha_2\|\nabla^2\phi\|^2\)가 DHOST | `REFUTED` |
| CE에서 최소 DHOST 확장 유도 | `OPEN` |
| 현실화 | `NOT ESTABLISHED` |

## 근거

- [Stable wormholes in scalar-tensor theories](https://arxiv.org/abs/1811.05481)
- [More about stable wormholes in beyond Horndeski theory](https://arxiv.org/abs/1812.07022)
- [Traversable wormholes in beyond Horndeski theories](https://arxiv.org/abs/2111.09857)
- [DHOST classification and degeneracy](https://arxiv.org/abs/1608.08135)
- [GW170817-compatible scalar-tensor relations](https://arxiv.org/abs/1710.05877)

## 재현

```powershell
uv run pytest tests/test_beyond_horndeski_reality.py -q
uv run python examples/physics/beyond_horndeski_reality_gate.py
```

## 2022 high-energy stability 갱신 (2026-08-04 재감사)

`arXiv:2212.05969`는 이전 2018 예시보다 강하다. 명시적 covariant
beyond-Horndeski Lagrangian을 재구성하고, odd/even parity 모두에서 ghost와
radial/angular gradient instability가 없는 구체적 wormhole을 제시한다.
따라서 `angular even 미검증`을 전체 연구 frontier의 현재 상태로 쓰는 것은
더 이상 정확하지 않다.

동시에 논문이 스스로 남긴 경계도 명확하다.

- high-energy perturbation만 완결하며 slow tachyon mass sector는 미검증이다.
- 기하는 양쪽 Minkowski형으로 점근 평탄하지만 먼 거리의 중력은 GR로
  돌아가지 않는다.
- physical matter frame에서 관측적으로 견고한 `c_T=c`를 증명하지 않는다.
- throat parameter `tau`를 실제 장치·에너지·CE 상수와 연결하지 않는다.
- CE 작용으로부터 해당 `F, G4, F4` 함수들을 유도하지 않는다.

갱신 판정은 `high-energy stable existence control = demonstrated`,
`complete stability and realization = open`이다. 이는 후보를 반증하지 않지만
곧바로 현실 장치가 된다는 뜻도 아니다.

근거: [In hot pursuit of a stable wormhole in beyond Horndeski theory](https://arxiv.org/abs/2212.05969)

## 2024/2025 complete-criteria 후속 루프

`arXiv:2404.06297v2`는 임의의 정적 구면 대칭 beyond-Horndeski 배경에 대해
ghost, gradient instability, tachyon, superluminal mode를 모두 배제하기 위한
완전한 조건을 유도한다. 이는 slow sector를 검사할 **방법**이 생겼다는 중요한
진전이다.

그러나 다음 두 명제는 구분해야 한다.

1. complete stability criteria가 존재한다: `Yes`.
2. 그 criteria를 만족하는 명시적 wormhole이 제시됐다: `Not demonstrated`.

2022 wormhole 논문은 최종 `f_i(pi), g_4i(pi)`를 긴 해석식으로 제공하지 않고
그래프로 제시한다고 명시한다. 따라서 공개된 본문만으로 그 동일 모델의 전체
저주파 mass operator를 재구축해 eigenvalue를 독립 검산할 수 없다.

갱신 판정:

| 항목 | 판정 |
|---|---|
| tachyon 포함 일반 안정성 조건 | `Derived` |
| 2022 explicit wormhole의 tachyon 조건 통과 | `Not evaluated` |
| 동일 모델의 독립 수치 재현 | `Blocked by unspecified analytic coefficients` |
| 완전히 안정한 명시적 wormhole 존재 | `Open` |

근거: [Complete stability for spherically symmetric backgrounds in beyond Horndeski theory](https://arxiv.org/abs/2404.06297)
