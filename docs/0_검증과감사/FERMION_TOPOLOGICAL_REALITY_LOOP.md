# Charged-fermion 위상 경계 현실화 루프

## 외부 control model

Maldacena–Milekhin–Popov(MMP) 구성은 Einstein–Maxwell 이론과 magnetic flux,
charged massless fermion을 사용한다. 자기 flux가 lowest Landau level의 유효 1+1차원
fermion zero mode를 만들고, 이들의 Casimir-like 음의 에너지가 backreaction을 지탱한다.
따라서 음의 응력의 부호가 단순 ansatz가 아니라 양자장 메커니즘에서 나온다는 점은 강하다.

그러나 이 해는 ambient space의 인과율을 보존하는 **long wormhole**이다. throat 통과가
외부 경로보다 빠른 순간이동 shortcut은 아니다.

## 사람 크기에서의 fermion 질량 gate

길이 \(L\)인 channel에서 fermion이 Casimir zero mode로 사실상 무질량이려면 최소한

\[
m_fc^2\ll\frac{\hbar c}{L}
\]

이어야 한다. \(L=1\) m이면

\[
\frac{\hbar c}{L}=1.9733\times10^{-7}\ {\rm eV}.
\]

전자의 질량에너지는 \(5.1099895\times10^5\) eV이므로 비는

\[
\frac{m_ec^2}{\hbar c/L}=2.5896\times10^{12}.
\]

표준모형의 더 가벼운 중성미자는 전기적으로 중성이므로 이 magnetic charged-zero-mode
역할을 그대로 맡지 못한다. 따라서 MMP의 Standard Model embedding이 가능한 미시적
고에너지 영역과 사람 크기의 macroscopic mouth를 동일시할 수 없다.

## 다중모드와 flux 수

자기 flux 정수 \(q\)는 zero-mode degeneracy를 키워 음의 Casimir 에너지를 증가시킨다.
하지만 이는 resonator의 quality factor처럼 비용 없는 증폭이 아니다. 최근 QEI 감사의
parametric control은 긴 throat에서 대략

\[
q\gtrsim\frac{\ell}{r_e}\gg1
\]

을 요구한다. 예를 들어 \(\ell/r_e=1000\)이면 적어도 1000개의 flux zero mode가 필요하다.
정확한 integer flux는 gauge bundle과 magnetic charge를 포함한 작용에서 나와야 한다.
계산 gate도 이 이산성을 보존한다. Python 정수와 `numbers.Integral`을 구현하는 NumPy 정수
스칼라는 허용하지만, `bool`과 실수형 및 `NaN`/`Inf`는 정수 변환 전에 거부한다. 따라서
예를 들어 `999.9`가 `999`로 조용히 절삭되어 물리적 flux 수로 오인되는 경로는 없다.

## CE mapping

현재 CE 문서에는 Standard Model sector가 포함되지만 다음은 없다.

1. 사람 크기에서 질량이 \(\ll10^{-7}\) eV인 새로운 charged fermion
2. CE 고유 compact gauge group과 quantized magnetic flux action
3. near-extremal magnetic mouth의 생성·분리·유지 과정
4. 해당 모드의 재규격화 Casimir tensor와 CE metric variation
5. long wormhole을 ambient shortcut으로 바꾸면서 인과율/QEI를 지키는 메커니즘

따라서 외부 control의 성공을 CE 성공으로 이전할 수 없다.

## 판정

| 명제 | 판정 |
|---|---|
| charged massless fermion의 음의 Casimir backreaction | `PASS IN MMP CONTROL` |
| ghost-free 알려진 QFT 메커니즘 | `YES, EXTERNAL CONTROL` |
| 1 m Standard Model charged mode | `REFUTED` |
| flux 다중모드를 free resonance gain으로 사용 | `REFUTED` |
| MMP 해가 ambient shortcut | `NO: LONG WORMHOLE` |
| CE charged fermion/flux sector | `NOT SPECIFIED` |
| 사람 크기 CE 순간이동 현실화 | `FAIL/OPEN NEW FIELD CONTENT` |

## 근거

- [Maldacena–Milekhin–Popov, Traversable wormholes in four dimensions](https://arxiv.org/abs/1807.04726)
- [Kontou, Wormhole restrictions from quantum energy inequalities](https://arxiv.org/abs/2405.05963)

## 재현

```powershell
uv run pytest tests/test_fermion_topological_reality.py -q
uv run python examples/physics/fermion_topological_reality_gate.py
```
