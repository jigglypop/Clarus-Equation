# 공간접힘 현실성 종합 원장

날짜: 2026-08-04

## 결론

검사한 10개 물리 경로 중 완전한 6/6 현실성 gate를 통과한 것은 없다.
클래스 수준의 치명적 veto를 아직 받지 않은 경로는 두 개다.

1. beyond-Horndeski wormhole: `4/6`, high-energy stable existence control
2. thin-shell cut-and-paste: `3/6`, exact geometry but microscopic source open

이는 성공확률을 뜻하지 않는다. 첫 경로는 수정중력 이론의 완전성과 관측
정합성이, 둘째 경로는 실제 결함 물질의 존재가 아직 증명되지 않았다.

## 동일 gate 집계

| 경로 | gate | 판정 | 결정적 이유 |
|---|---:|---|---|
| beyond-Horndeski | 4/6 | `ACTIVE` | 고에너지 안정 배경 존재; slow tachyon·GR limit·scale 미완성 |
| thin shell | 3/6 | `ACTIVE` | 접합 기하 exact; 알려진 최소 source들은 불안정/ghost |
| wrong-sign/phantom | 4/6 | `VETO` | propagating ghost |
| charged-fermion long wormhole | 3/6 | `VETO` | ambient shortcut이 아니며 CE field content 부재 |
| AdS double-trace | 3/6 | `VETO` | 잘못된 asymptotics와 비국소 coupling |
| static material Casimir | 3/6 | `VETO` | 1 m에서 am/subnuclear boundary와 거대 에너지 |
| dynamic Casimir pulse | 3/6 | `VETO` | 정적 중력원이 아니며 QI duration fail |
| CE nonminimal scalar | 2/6 | `VETO` | healthy global reconstruction no-go |
| massive vacuum polarization | 1/6 | `VETO` | Compton 억제로 macroscopic source 불가 |
| pure topological boundary | 1/6 | `VETO` | bulk Hilbert stress가 0 |

## thin-shell 하위 후보 소거 원장

| 하위 후보 | 결과 |
|---|---|
| isotropic scale-free/CFT edge | `Refuted`: Israel EoS와 pressure sign 불일치 |
| causal barotropic fluid | `Refuted`: `0<=c_s^2<=1`과 radial stability 무교차 |
| minimal elastic membrane | `Refuted`: shear가 `l=0` radial mode에 결합하지 않음 |
| negative-tension Nambu--Goto | `Refuted`: `f=1/3`에서 radial unstable + bending ghost |
| one-species smooth quantum layer | `Physical-scale fail`: `d<=2.50e-24 m`, `UV>=7.90e16 eV` |
| nonlocal/internal-mode defect | `Open`: 명시적 action 없음 |

수동적인 안정 내부모드 mixing은 Schur complement
`K_eff=K_rr-B C^-1 B^T` 때문에 음의 radial eigenvalue를 더 낮춘다. 따라서
`passive multimode resonance`는 안정화 수단으로 소거된다. 직접 양의 radial
stiffness를 생성하는 새 작용 또는 driven/feedback 제어만 남으며, 후자는 정적
해가 아닌 시간의존 안정성 문제다.

driven 후보는 exact monodromy 계산에서 `Gamma/Omega=0.05`,
`epsilon=0.1`일 때 안정화됨을 확인했다. 따라서 Floquet 제어 자체는
`Demonstrated control`이다. 그러나 음의 junction stress를 공급하지 않고
drive loss 시 불안정성이 복귀하므로 realization gate 수는 올리지 않는다.

Floquet 계수를 Israel pressure로 역산하면 1 m control에 `0.954 GHz`,
`6.68e44 N/m^2`의 pressure stiffness, `5.03e43 W`의 피크 반응성 기계출력
상한과 `3.34 ns`의 drive-loss e-fold가 나온다. 물리 actuator action이 없으므로
`control PASS / engineering FAIL`이다.

negative-tension 막의 local `K^2` rigidity 보강도 pole decomposition에서
massless bending residue `1/T<0`를 보존하고 추가 pole에 반대 residue를 만든다.
따라서 최소 rigid-brane cure는 `Refuted`다.

induced-gravity defect는 ghost-free parameter class가 존재해 반증하지 않는다.
그러나 2+1 localized EH 단독에는 국소 graviton이 없고 bulk mixing 전체가
필요하다. CE에는 localized coefficient, modified junction solution, KK/bending
spectrum이 없어 `external open frontier`로만 남는다.

## beyond-Horndeski 생존 조건

2022 covariant 예시는 odd/even parity의 high-energy ghost 및 radial/angular
gradient gate를 한 모델에서 통과한다. 남은 필수조건은 다음과 같다.

1. 전체 저주파 mass operator에서 tachyon 고유값이 없을 것
2. 두 점근영역에서 실제 matter metric이 GR로 복귀할 것
3. 배경 변화에도 physical tensor cone이 관측 bound 안에서 luminal일 것
4. strong-coupling cutoff가 throat curvature 및 운용 주파수보다 높을 것
5. CE action으로부터 `F, G4, F4`를 독립적으로 유도할 것
6. throat scale, 생성 과정, 입구 고정 및 payload backreaction을 닫을 것

하나라도 실패하면 현재 유일한 수정중력 생존 경로도 veto된다.

2024/2025 후속 연구는 tachyon과 superluminality까지 포함한 일반 stability
criteria를 유도했다. 그러나 이 criteria를 통과한 명시적 wormhole을 제시하지는
않았다. 2022 예시의 최종 coefficient functions도 해석식/기계판독 데이터로
제공되지 않아 동일 모델의 slow spectrum 독립 재현은 현재 막혀 있다. 따라서
`criteria derived`를 `candidate passed`로 합치지 않는다.

## 현재 가능한 말의 최대치

- 수학적 공간 지름길 기하: 가능
- 알려진 외부 이론에서 고에너지 안정 wormhole control: 가능
- Clarus가 그 이론을 유도함: 미증명
- 현실 물질로 1 m 입구 제작: 근거 없음
- 실험실 순간이동 장치: 현재 불가능

전체 판정은 `THEORETICAL POSSIBILITY REMAINS / PHYSICAL REALIZATION NOT
ESTABLISHED`이다.

## 재현

```powershell
$env:PYTHONPATH='.;reality_stone/python'
uv run python examples/physics/realization_pathway_funnel_gate.py
uv run pytest tests/test_realization_pathway_funnel.py -q
```
