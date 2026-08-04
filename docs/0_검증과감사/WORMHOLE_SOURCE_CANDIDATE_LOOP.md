# 공간접힘 물질원 후보군 1차 루프

## 1. 공통 승격 게이트

후보가 단순한 음의 숫자에서 물리적 웜홀 물질원으로 승격하려면 다음을
순서대로 통과해야 한다.

| Gate | 요구사항 |
|---|---|
| S0 | 적절한 null 방향에서 $T_{\mu\nu}k^\mu k^\nu<0$ 또는 필요한 ANEC 위반 |
| S1 | 상태·경계조건·재규격화 규약을 포함한 전체 $\langle T_{\mu\nu}\rangle_{\rm ren}$ |
| S2 | 운동방정식 위에서 $\nabla_\mu T^{\mu\nu}=0$과 Ward/anomaly 감사 |
| S3 | quantum inequality·ANEC·지속시간과 공간 평균 제약 |
| S4 | $G_{\mu\nu}-(8\pi G/c^4)\langle T_{\mu\nu}\rangle_{\rm ren}=0$의 backreaction 해 |
| S5 | 선형·비선형 안정성, 입구 생성과 유지 에너지 |
| S6 | 위 구조가 외부 대조군이 아니라 CE 작용에서 유도됨 |

한 gate의 실패를 뒤쪽 gate의 가정으로 덮지 않는다.

## 2. CE 정적 Casimir 셀

문서의 $0.62\,\mathrm{keV}/294\,\mathrm{fm^3}$를 사용하면 음의 밀도
크기는 $3.3787\times10^{26}\,\mathrm{J/m^3}$이다. 1m Morris--Thorne
제어목에 필요한 null projection보다 $2.85\times10^{16}$배 작고,
$6.65$fm 상관길이도 목을 덮지 못한다.

따라서 이 후보는 S0의 부호를 가정하더라도 국소 크기·공간 범위에서
탈락하며 S1의 전체 renormalized tensor도 아직 없다.

## 3. CE resonance-$Q$ 후보

검사용 ansatz를

\[
|T_{kk}(Q)|=|T_{kk}(1)|Q^p,
\qquad
\xi(Q)=\xi_0Q
\]

로 두었다. 이는 CE 유도가 아니라 필요한 크기를 역산하기 위한 가정이다.

| 가정 | 밀도 조건의 $Q$ | coherence 조건의 $Q$ | 결합 요구량 |
|---|---:|---:|---:|
| $p=1$ | $2.85\times10^{16}$ | $1.50\times10^{14}$ | $2.85\times10^{16}$ |
| $p=2$ | $1.69\times10^8$ | $1.50\times10^{14}$ | $1.50\times10^{14}$ |

두 ansatz 모두 선택한 법칙 아래에서는 숫자 조건을 통과한다. 그러나 CE
문서의 $\xi_{\rm eff}=Q\xi$가 곧 $T_{kk}\propto Q^p$를 뜻하지 않으며,
$p$의 값·부호·포화·가열·수명 어느 것도 유도되지 않았다. 따라서 이
후보의 현재 첫 gate는 `S0/S1 OPEN`이다.

## 4. CE canonical scalar와 phantom 대조군

canonical scalar의 null projection은 potential 항이 null contraction에서
사라져

\[
T_{\mu\nu}k^\mu k^\nu=(k^\mu\partial_\mu\sigma)^2\ge0
\]

이다. 따라서 고전 canonical scalar 하나로 Morris--Thorne 목의 NEC
위반을 만드는 채널은 S0에서 반증된다.

kinetic sign을 뒤집은 phantom scalar는
$T_{kk}=-(k\cdot\partial\sigma)^2<0$를 만들지만 wrong-sign kinetic
term이므로 최소 ghost-free 안정성 gate S5를 실패한다. “NEC 위반”만으로
후보가 살아남지 않는 직접 대조다.

## 5. 양자 음의 에너지 상태

양자장에서는 국소 renormalized energy가 음수가 되는 상태가 가능하지만,
지속적이고 거시적인 웜홀 응력원이라는 결론은 따라오지 않는다.
[Fewster--Roman](https://arxiv.org/abs/gr-qc/0209036)은 4차원 Minkowski의
null geodesic 위 가중 null 평균에는 일반적인 하한이 없다는 구체적 상태를
구성하면서도, 분석한 상태의 ANEC와 timelike worldline quantum inequality를
구분한다. 따라서 국소 음의 pulse는 S0의 일부 증거일 뿐 S3--S5는 열린다.

[Ford--Roman의 웜홀 제약](https://arxiv.org/abs/gr-qc/9510071)은 거시적
정적 웜홀의 음의 에너지 크기와 지속시간을 강하게 제한한다. CE 후보도
동일하게 sampling 함수와 상태를 지정한 quantum-inequality 계산이 필요하다.

## 6. 구조가 닫힌 외부 대조군

- [Gao--Jafferis--Wall](https://arxiv.org/abs/1608.05687)은 두 AdS 경계의
  double-trace 결합에서 음의 평균 null energy와 backreaction을 계산해
  Einstein--Rosen bridge를 통과 가능하게 한다. 외부 인과성 위반 장치는
  아니며 CE의 평탄공간 임의 목적지 해도 아니다.
- [Maldacena--Milekhin--Popov](https://arxiv.org/abs/1807.04726)는 4차원
  Einstein--Maxwell과 charged massless fermion의 Casimir-like 에너지로
  구체적 해를 준다. 이 대조군이 보여주는 핵심은 장 내용, 경계조건,
  $T_{\mu\nu}$와 backreaction 해를 함께 닫아야 한다는 점이다.

두 대조군은 “웜홀이 수학적으로 절대 불가능”이라는 명제를 반박하지만,
CE 후보의 S6를 통과시키지는 않는다.

## 7. 1차 판정

| 후보 | 첫 실패 또는 열린 gate | 판정 |
|---|---|---|
| CE 정적 Casimir 셀 | 밀도·coherence 및 S1 | `FAIL / OPEN` |
| CE resonance-$Q$ | $T_{kk}(Q)$ 법칙과 S1 | `OPEN` |
| CE 최소결합 canonical scalar 채널 | S0 | `REFUTED` |
| phantom scalar | S5 ghost | `REFUTED AS STABLE SOURCE` |
| 양자 음의 에너지 상태 | S3 지속 평균, S4 backreaction | `OPEN CONTROL` |
| GJW AdS double trace | S6 CE mapping | `PASS IN OTHER MODEL` |
| MMP 4D fermion Casimir | S6 CE field mapping | `PASS IN OTHER MODEL` |

1차 screening 시점에는 resonance-$Q$를 먼저 검사했다. 반증 질문은
“CE의 명시적 2점함수/유효작용에서 $Q$에 따른 renormalized $T_{kk}$의
부호와 scaling exponent $p$를 계산할 수 있는가?”였다.

후속 `RESONANCE_STRESS_IDENTIFIABILITY_LOOP.md`의 반례로 $\xi(Q)$만으로
$p$를 식별할 수 없음을 증명했다. residue, spectral density와 metric
variation이 없는 현재 resonance 후보는 `W1 / KINEMATIC ONLY`에 머문다.

후속 `CLARUS_NEGATIVE_SOURCE_SEARCH_LOOP.md`에서 CE 문서의
$\xi R\Phi^2$ 비최소 결합을 분리해 재검사했다. 최소결합 no-go와 달리
canonical kinetic sign을 유지한 국소 effective NEC 위반 영역이 존재한다.
따라서 우선순위는 resonance-$Q$ 단독에서 비최소 CE + Casimir/곡률
진공편극 하이브리드로 이동했다.

## 8. 실행

```powershell
uv run --extra dev python -m pytest tests/test_wormhole_source_candidates.py -q
uv run python examples/physics/wormhole_source_candidate_gate.py
```
