# CE 순수 위상 경계 응력 루프

## 질문

물질판이나 새로운 입자 없이 위상 경계 자체가 wormhole에 필요한 음의
\(T_{\mu\nu}\)를 공급할 수 있는가?

## Metric variation gate

중력원은 Hilbert stress tensor

\[
T_{\mu\nu}=-\frac{2}{\sqrt{-g}}
\frac{\delta S}{\delta g^{\mu\nu}}
\]

로 정의된다. 작용이 metric에 의존하지 않거나 4차원에서 상수계수 topological invariant면
bulk metric variation은 0 또는 순수 경계항이다. 따라서 국소 bulk stress가 없다.

| 순수 항 | 국소 bulk \(T_{\mu\nu}\) | 음의 null stress |
|---|---:|---:|
| 4D Euler/Gauss–Bonnet, 상수계수 | 0 | 없음 |
| 4D gravitational Pontryagin, 상수계수 | 0 | 없음 |
| gauge \(\theta F\wedge F\), 상수 \(\theta\) | 0 | 없음 |
| 3D boundary Chern–Simons | bulk 0 | edge 이론 없이는 없음 |

그러므로 순수 topological term은 throat를 직접 지탱할 수 없다.

## 위상이 할 수 있는 일

위상은 다음을 선택할 수 있다.

- quantized flux sector
- 허용 boundary condition
- Wilson/'t Hooft line sector
- boundary anomaly와 edge mode의 존재
- zero-mode degeneracy

하지만 edge mode가 실제 에너지와 압력을 가지려면 induced metric에 결합하는 동역학적
edge action이 필요하다. Chern–Simons bulk에서 유도되는 chiral edge boson/WZW 계열이
대표적이다. 이 stress의 부호와 크기는 pure topology가 아니라 edge Hamiltonian, 상태,
경계조건과 재규격화에서 결정된다.

따라서

```text
topology selects edge modes
    != topology supplies negative stress
```

이다. 동역학적 edge fermion을 추가하면 앞선 charged-fermion/Casimir 경로로 돌아가며
1 m 유효 무질량 한계와 flux 비용을 다시 통과해야 한다.

## 가변 계수 우회

\(f(\phi)\mathcal G_{GB}\), \(\theta(\phi)R\wedge R\)처럼 계수를 장으로 만들면
metric variation이 0이 아닐 수 있다. 그러나 이는 더 이상 순수 위상 source가 아니다.
새 scalar 동역학과 고차미분 결합을 가진 수정중력이며, DHOST 퇴화·GW 속도·전체 안정성
gate로 되돌아간다.

## Topological censorship

순수 위상항의 bulk stress가 0이고 나머지 물질이 NEC를 만족한다면, globally hyperbolic하고
asymptotically flat한 조건 아래 topological censorship을 피할 음의 에너지가 없다.
따라서 관측 가능한 shortcut을 topology만으로 열 수 없다.

## 판정

| 명제 | 판정 |
|---|---|
| 순수 위상항의 국소 bulk stress | `ZERO` |
| 순수 위상항으로 음의 throat source 생성 | `REFUTED` |
| 위상으로 flux/edge sector 선택 | `POSSIBLE` |
| CE 동역학적 edge action | `NOT SPECIFIED` |
| edge \(\langle T_{\mu\nu}\rangle_{ren}<0\) | `NOT DERIVED` |
| 순수 CE 위상 경계 순간이동 | `REFUTED` |
| topology + 새로운 edge QFT | `OPEN, BUT NEW FIELD CONTENT` |

## 근거

- [Friedman–Schleich–Witt, Topological Censorship](https://arxiv.org/abs/gr-qc/9305017)
- [Metric dependence of Chern–Simons edge states](https://arxiv.org/abs/2110.13203)
- [4D Einstein–Gauss–Bonnet 정의 문제](https://arxiv.org/abs/2004.03390)

## 재현

```powershell
uv run pytest tests/test_topological_boundary_stress.py -q
uv run python examples/physics/topological_boundary_stress_gate.py
```
