# 클라루스장 음의 동력원 후보 탐색 루프

## 1. 탐색 범위

클라루스장을 음의 동력원 후보로 유지하되, “장 자체가 음수”라고 가정하지
않고 어떤 상태·결합·경계조건에서 null source가 음수가 되는지 분해했다.

탐색 후보는 다음 일곱 개다.

1. 비최소 클라루스장 + Casimir 경계
2. 비최소 클라루스장의 곡률 진공편극
3. CE+SM charged fermion magnetic Casimir 사상
4. CE 두 경계/double-trace 상태
5. resonance-$Q$ 단독
6. beyond-Horndeski 확장
7. phantom 클라루스장

## 2. 발견: CE 비최소 결합은 최소결합 no-go를 우회한다

CE 문서는

\[
V_{\rm eff}(\Phi,R)=V(\Phi)+\xi R\Phi^2,
\qquad \xi\simeq0.49
\]

를 사용한다. 부호와 정규화를

\[
S\supset\int\sqrt{-g}\,
\frac12(1-\xi\Phi^2)R
\]

인 Planck-normalized Jordan-frame control로 고정하면, affine null 방향의
재배열된 Einstein 식 numerator는

\[
N_{kk}
=(\Phi')^2-\xi(\Phi^2)''
=(1-2\xi)(\Phi')^2-2\xi\Phi\Phi''
\]

이고 유효 중력계수는

\[
F(\Phi)=1-\xi\Phi^2
\]

이다. 따라서 $F>0$을 유지하면서도 $N_{kk}<0$인 국소영역이 존재한다.
실행 대조값 $\xi=0.49$, $\Phi=0.5$, $\Phi'=0.1$, $\Phi''=1$에서는

\[
N_{kk}=-0.4898,
\qquad F=0.8775>0.
\]

canonical kinetic sign을 뒤집지 않고 국소 effective NEC 위반 gate가
통과한다. 이전 “canonical scalar 탈락” 판정은 **최소결합 채널에만**
유효하며, CE의 비최소결합 채널에는 적용되지 않는다.

이 결과는 선택한 Jordan-frame 부호·정규화 아래의 국소 필요조건이다.
CE Q0 전체 작용, 운동방정식과 perturbation operator가 미완성이므로 전역
해나 안정성 증명은 아니다.

## 3. 평균 null gate: 경계 또는 곡률상태가 필요하다

위 numerator를 완전 affine null curve에서 적분하면

\[
\int N_{kk}\,d\lambda
=\int(\Phi')^2d\lambda
-\xi\left[(\Phi^2)'\right]_{-\infty}^{+\infty}.
\]

국소화된 profile이 양끝에서 같은 진공으로 돌아가 boundary jump가 0이면
첫 항만 남아 비음수다. 따라서 국소 음의 pocket만으로 averaged gate를
통과하지 못한다.

이 때문에 현재 CE-native 최상위 후보는 다음 두 하이브리드다.

- Casimir 경계가 endpoint/boundary contribution을 제공하는 경우
- 곡률 배경의 진공편극이 state-dependent renormalized stress를 제공하는 경우

[Casimir+scalar 보존 연구](https://arxiv.org/abs/2312.16736)는 Casimir
밀도만 주면 압력 성분과 보존 때문에 보조장이 필요할 수 있음을 보이고,
특정 fixed-plate+potential 조합을 예외 후보로 분석한다. 이는 CE에서도
스칼라 밀도 하나가 아니라 전체 압력과 보존식을 풀어야 한다는 대조다.

[비등각 스칼라 진공편극 long-throat 연구](https://arxiv.org/abs/1809.06202)는
곡률배경에서 renormalized stress를 계산하고 semiclassical Einstein 해를
찾는 외부 control이다. CE의 $\xi\simeq0.49$를 이 계산에 넣는 mapping은
아직 수행되지 않았다.

## 4. Casimir 크기 역산

이상적인 전자기 평행판의 에너지 밀도는

\[
|\rho|=\frac{\pi^2\hbar c}{720a^4},
\qquad |\rho+p_\perp|=4|\rho|
\]

이다. 1m control throat의 $9.63\times10^{42}\,\mathrm{J/m^3}$ null
요구량을 맞추는 plate separation은

\[
a\simeq3.66\times10^{-18}\,\mathrm m.
\]

이는 CE 상관길이 $6.65\times10^{-15}$m의 약 $5.5\times10^{-4}$다.
이상적 평행판 공식의 역산일 뿐, 그 크기의 경계를 1m 구면 목 전체에
배치할 수 있다는 뜻은 아니다. [Casimir wormhole 해 연구](https://arxiv.org/abs/2406.03588)도
경계기하와 equation of state를 명시해 해를 구성하며, 단일 에너지 밀도
수치를 곧바로 장치로 승격하지 않는다.

## 5. 비최소 중력계수 증폭의 극한

정적 CE 밀도 격차 $A=2.85\times10^{16}$를 오직
$1/F=1/(1-\xi\Phi^2)$로 증폭한다고 가정하면

\[
F_{\rm req}=A^{-1}=3.51\times10^{-17}.
\]

$\xi=0.49$의 중력 pole은 $\Phi_c=1/\sqrt\xi\simeq1.4286$이고,
필요 field는 상대거리

\[
\frac{\Phi_c-\Phi}{\Phi_c}simeq1.75\times10^{-17}
\]

안쪽이다. 대수적으로 격차를 닫지만 effective Planck factor가 거의 0인
강결합/특이 극한이므로 정규적인 해결책으로 판정하지 않는다.

## 6. 확장 후보 funnel

| 순위층 | 후보 | 현재 판정 | 결정적 다음 계산 |
|---|---|---|---|
| `FRONTIER_A` | Casimir 경계 + 일반 redshift | 후속 국소 tensor·보존 통과 | 전역 ODE와 boundary heat kernel |
| `FRONTIER_A` | 비최소 CE + Casimir completion | 국소 부호 통과, 전체 stress 열림 | coupled scalar-boundary ODE |
| `FRONTIER_B` | CE+SM fermion magnetic Casimir | MMP 외부모형 W3, CE mapping 없음 | magnetic topology와 fermion zero mode |
| `FRONTIER_B` | CE double-trace 경계 | GJW AdS에서 통과, CE coupling 없음 | CE 두 경계 상호작용 유도 |
| `DEFERRED` | resonance-$Q$ 단독 | stress 부호·residue 비식별 | spectral residue 선행 |
| `EXTERNAL` | beyond-Horndeski | 일부 안정성 gate 통과, CE action에 없음 | 확장 정당화와 전체 모드 안정성 |
| `REJECTED` | phantom CE | NEC 통과, ghost 실패 | ghost-free completion 없이는 중단 |

[Maldacena--Milekhin--Popov](https://arxiv.org/abs/1807.04726)는 charged
massless fermion의 magnetic/Casimir-like 에너지로 4차원 해를 구성한다.
CE+SM에 charged fermion이 있다는 것만으로 이 topology와 zero mode가
자동 생성되지는 않지만, CE 내부장만 고집하지 않는다면 가장 강한 W3
대조군이다.

[beyond-Horndeski 연구](https://arxiv.org/abs/1812.07022)는 기존 Horndeski
no-go를 일부 우회하는 예를 주지만 안정성 분석이 불완전하며 현재 CE
작용에 해당 고차미분 sector가 없다. 따라서 우선순위는 낮다.

## 7. 현재 최대 가능성

단계를 더 세분하면 다음과 같다.

| 단계 | 상태 |
|---|---|
| W0 선택·제어 | 통과 |
| W1 주어진 기하 shortcut | 통과 |
| W2a CE-native 국소 음의 null source 후보 | **조건부 통과** |
| W2b 평균된 renormalized 전체 $T_{\mu\nu}$ | 열림 |
| W2c 보존·quantum inequality | 열림 |
| W3 self-consistent 안정 전역 해 | CE 미도달 |
| W4 제작·유지 | 미도달 |
| W5 장치 | 미도달 |

따라서 전체 사슬은 아직 W1에 묶이지만, 후보 frontier는 W2a까지 전진했다.
가장 가치 있는 다음 루프는 `비최소 CE + Casimir/곡률 진공편극`의
renormalized $T_{\mu\nu}$를 계산하는 것이다.

후속 `CLARUS_BACKREACTION_CANDIDATE_LOOP.md`에서 두 `FRONTIER_A`를
backreaction scale로 비교했다. 이상적 Casimir는 zero-redshift throat에서
tangential pressure $C/3$이 부족하고, 6.65fm massive vacuum polarization은
1m에서 $2.90\times10^{-97}$에 불과했다. 이후 일반 redshift를 허용한
Casimir 국소 throat series가 별도 `FRONTIER_A`로 승격됐다.

## 8. 실행

```powershell
uv run --extra dev python -m pytest tests/test_clarus_negative_source_search.py -q
uv run python examples/physics/clarus_negative_source_search_gate.py
```
