# 공간접힘 현실화 전체 경로 funnel

## 목적

현실화 후보를 아이디어 수가 아니라 다음 여섯 hard gate로 비교한다.

1. 명시적 공변 작용
2. 음의 재규격화 응력 또는 유효 NEC 위반의 유도
3. 자기일관 backreaction
4. 외부 공간보다 빠른 실제 shortcut
5. 완전한 선형 안정성
6. 실험·공학 scale bridge

Gate 개수는 확률이 아니다. ghost, 잘못된 점근구조, shortcut 부재처럼 목표와 양립하지
않는 항목은 별도 hard veto로 처리한다.

## 10개 물리 계열 감사

| 경로 | gate | veto | 판정 |
|---|---:|---|---|
| beyond-Horndeski wormhole | 4/6 | 없음 | 외부 이론 frontier, 안정성 불완전 |
| thin-shell cut-and-paste | 3/6 | 없음 | 기하는 있으나 exotic shell의 물리 source 없음 |
| CE \(\xi R\Phi^2\) 비최소결합 scalar | 2/6 | 없음 | CE 내부 최우선, Q0·전역해 open |
| CE 비물질 위상 경계 | 1/6 | 없음 | 결정적 가설이나 작용 자체가 없음 |
| MMP charged-fermion long wormhole | 3/6 | shortcut 아님 | 알려진 4D 물리 모델 |
| AdS double-trace wormhole | 3/6 | 점근구조 불일치 | 강한 이론 control |
| 물질판 정적 Casimir | 3/6 | subnuclear 경계 실패 | 현재 target에서 중단 |
| 동적 Casimir 다중모드 | 3/6 | 정적 source 아님 | 실험 가능하지만 목표와 다름 |
| CE heavy-field vacuum polarization | 1/6 | 거시 scale 반증 | 1 m source에서 중단 |
| phantom wrong-sign field | 4/6 | ghost | 폐기 |

## 현실화 우선순위

### CE 이론을 유지할 때

최우선은 기존 라그랑지안에 이미 적힌

\[
\mathcal L\supset \xi R\Phi^2+rac12(\partial\Phi)^2-V(\Phi)
\]

경로다. 물질판 없이 scalar의 여러 radial eigenmode를 중첩할 수 있으므로 다중공명 허용과
양립한다. 그러나 현재 문서의 \(F=1+\alpha_sD_{\rm eff}\) 평균장 닫힘은 상수 유효중력으로
읽히며, 그 자체로 wormhole source를 만들지 않는다. 필요한 것은 위치 의존 \(\Phi(r)\),
경계조건, potential, coupled metric EOM과 quadratic perturbation operator다.

다음 계산은 성공값 fitting이 아니라 inverse reconstruction이다.

\[
\{b(r),\Phi_{\rm redshift}(r)\}
\xrightarrow{\text{Einstein+scalar EOM}}
\{\Phi(r),V(\Phi),\xi\}
\xrightarrow{\delta^2S}
\text{ghost/gradient spectrum}.
\]

단일값 함수 \(V(\Phi)\)가 존재하지 않거나, \(F(\Phi)>0\)를 유지하지 못하거나, quadratic
operator에 음의 norm/gradient mode가 생기면 이 CE 내부 경로는 반증된다.

### 이론 확장을 허용할 때

beyond-Horndeski가 현재 가장 많은 구조 gate를 만족하지만 CE의 현재 작용 밖이다. 기존
연구도 모든 parity-even 각방향 mode와 느린 tachyon을 포함한 완전 안정성을 닫지 못했다.
따라서 이를 바로 현실적 장치라고 부를 수 없다.

## 문헌 경계

- [Maldacena–Milekhin–Popov 4D long wormhole](https://arxiv.org/abs/1807.04726)
- [Gao–Jafferis–Wall double-trace traversability](https://arxiv.org/abs/1608.05687)
- [Beyond-Horndeski wormhole stability—불완전 범위 명시](https://arxiv.org/abs/1812.07022)
- [Wormhole quantum-energy restrictions](https://arxiv.org/abs/2405.05963)

## 판정

현 시점에 6/6을 통과한 경로는 없다. CE를 유지하는 현실화 루프의 다음 단일 목표는
`비최소결합 scalar 전역 inverse reconstruction + 전체 radial perturbation spectrum`이다.

## 재현

```powershell
uv run pytest tests/test_realization_pathway_funnel.py -q
uv run python examples/physics/realization_pathway_funnel_gate.py
```
