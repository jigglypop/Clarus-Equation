# 시간여행 인과성 루프

## 판정할 명제

`11_게이지_격자와_인과성.md`의 국소 조건

\[
N\ge 0,\qquad |\det T|^2\le 1
\]

만으로 모든 닫힌 시간꼴 곡선(CTC)을 배제할 수 있는지를 판정한다.

## 루프 1A — 국소 no-go의 반례

평탄한 로런츠 계량

\[
ds^2=-dt^2+dx^2+dy^2+dz^2
\]

에서 시간을 (t\sim t+T)로 동일시한 몫공간 (S^1_T\times\mathbb R^3)을
생각한다. 공간좌표를 고정하고 (t)만 증가시키면 각 국소 구간은
future-directed timelike이고 (N=1>0)이다. CE 문서의 전이식을 그대로
적용해도 각 스텝에서

\[
|\det T|^2=e^{-\alpha_{\rm total}\Delta\tau}\le1
\]

이다. 그러나 (t)가 한 주기 (T) 증가하면 몫공간에서는 출발 사건으로
돌아온다. 고유시간은 (T>0)만큼 증가했지만 곡선은 닫힌다.

따라서 다음 함의는 거짓이다.

\[
N\ge0\ \wedge\ |\det T|\le1
\not\Rightarrow \text{no CTC}.
\]

이 반례는 주기적 시간이 실제 우주라는 주장이 아니다. 국소 부등식만으로
전역 위상을 통제할 수 없다는 논리적 반례다.

## 루프 1B — CE A1 아래의 불가능 정리

`참조/epsilon_제1원리_유도.md` 12.1절과 18.1절은 전역 시간함수

\[
t:M\to\mathbb R
\]

가 존재하고 모든 허용 causal curve에서 (t\circ\gamma)가 단조 증가한다고
가정한다.

**정리.** 이 가정 아래에는 future-directed CTC가 없다.

**증명.** CTC (\gamma:[0,1]\to M)가 존재한다고 가정한다. 닫힘 때문에
(\gamma(0)=\gamma(1))이고, 함수의 단일값성 때문에

\[
t(\gamma(1))-t(\gamma(0))=0
\]

이다. 그러나 (t\circ\gamma)가 future-directed causal curve에서 엄격히
증가하면 같은 차이는 양수여야 한다. 모순이다. \(\square\)

## 루프 1 판정

| 대상 | 판정 | 의미 |
|---|---|---|
| determinant/lapse no-go | `REFUTED` | 국소 조건은 CTC 부재의 충분조건이 아님 |
| 전역 시간함수 A1 | `PROVED/CONDITIONAL` | A1을 받아들이면 CTC는 정의상 배제됨 |
| CE가 A1을 동역학적으로 유도했는가 | `OPEN` | 현재 문서는 전역쌍곡성을 가정함 |
| CE 시간여행 장치 | `NOT ESTABLISHED` | 반례는 장치나 물리적 생성 메커니즘이 아님 |

현재 가장 강한 결론은 “CE가 시간여행을 동역학적으로 금지했다고 증명한
것이 아니라, 전역쌍곡성과 전역 시간함수를 시공간 공리로 선택해
배제했다”이다.

## 루프 2 — A1의 동역학적 지위

저장소 전체에서 전역 시간함수와 전역쌍곡성의 기원을 다시 검색했다.
해당 조건은 `참조/epsilon_제1원리_유도.md` 12.1절에서 “가정한다”고
도입되고, 18.1절에서 시공간 공리 A1로 재기록된다. 완전한 CE 작용의
해공간에서 이 조건을 유도하거나, CTC 위상을 동역학적으로 배제하는
정리는 발견되지 않았다.

기존 Q0 구조 게이트도 이 경계를 확인한다.

```text
full Q0 pass          False
full CE+SM complete   False
gravity_and_ce_sector excluded
stress tensor derived False
```

`tests/test_q0_manifest_gate.py`와 `tests/test_a1_q0_action_bridge.py`의 52개
회귀 테스트는 통과했지만, 이는 제한된 Abelian-Higgs + Z2 singlet control
slice의 구조 검증이다. 중력 섹터와 재규격화 stress tensor가 제외되어
있으므로, 현재 실행계는 topology change나 chronology protection을 계산할
수 없다.

### 루프 2 판정

| 질문 | 판정 |
|---|---|
| A1이 CTC를 배제하는가 | `YES / PROVED` |
| CE 동역학이 A1을 유도했는가 | `OPEN` |
| CE 동역학이 CTC를 허용한다고 보였는가 | `OPEN` |
| 현재 코드로 웜홀/CTC 생성 여부를 판정할 수 있는가 | `NO / MISSING BRIDGE` |

따라서 “CE에서 시간여행은 증명됐다”와 “CE에서 시간여행은 동역학적으로
반증됐다”는 둘 다 아직 허용되지 않는다. 확정된 것은 국소 determinant
no-go의 반증과, A1을 조건으로 한 no-CTC 정리뿐이다.

## 실행 게이트

```powershell
uv run --extra dev python -m pytest tests/test_time_travel_causality.py -q
uv run python examples/physics/time_travel_causality_gate.py
```
