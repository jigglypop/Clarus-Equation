# 비최소결합 scalar 전역 역산 루프

## 질문

전역 다중모드 target을 CE 라그랑지안에 이미 있는 \(F(\phi)R\), 특히
\(\xi R\phi^2\), 그리고 ghost-free scalar들로 실제 생성할 수 있는가?

## 장방정식 결합

양의 field-space metric을 가진 scalar-tensor 계열의 metric equation을

\[
F G_{ab}=\kappa T^{(\phi)}_{ab}
+\nabla_a\nabla_bF-g_{ab}\Box F
\]

로 둔다. 고유 radial 거리 \(s=\ell/r_0\)에서 \(tt+\theta\theta\) 식은
기하로부터 \(F\)의 기울기를 고정하고, \(tt+rr\) 식은

\[
\frac{\kappa G_{IJ}\phi_s^I\phi_s^J}{F}
= (G_{\hat t\hat t}+G_{\hat r\hat r})
-\frac{F_{ss}}F+\Phi_s\frac{F_s}F
\]

를 준다.

## 기존 지수형 target 반증

기존 target

\[
b/r_0=\frac23+\frac13e^{-(x-1)},\qquad
\Phi=\frac12e^{-(x-1)}
\]

에서는 목에서 정확히

\[
(\ln F)_x=\frac18,\qquad
\frac{F_{ss}}F=\frac1{12},\qquad
G_{\hat t\hat t}+G_{\hat r\hat r}=-\frac43
\]

이므로

\[
\boxed{
\frac{\kappa G_{IJ}\phi_s^I\phi_s^J}{F}=-\frac{17}{12}}
\]

이다. \(F>0\)와 양의 definite \(G_{IJ}\) 아래에서는 좌변이 음수가 될 수 없다.
따라서 기존 지수형 전역 target은 단일 scalar뿐 아니라 건강한 canonical 다중 scalar
공명으로도 생성할 수 없다. potential을 정하기 전에 이미 kinetic sign에서 반증된다.

## 2차 기하 co-design

이 결과는 비최소결합 scalar 계열 전체의 no-go는 아니다. 목의 Casimir 값

\[
b'_0=-\frac13,\qquad r_0\Phi'_0=-\frac12
\]

를 유지하고

\[
b/r_0=1-\frac z3+\frac\gamma2z^2+\cdots,
\qquad
\Phi=\Phi_0-\frac z2+\frac v2z^2+\cdots
\]

로 두면

\[
(\ln F)_x=\frac{3\gamma+8v-4}{8},
\]

\[
\frac{\kappa G_{IJ}\phi_s^I\phi_s^J}{F}
=-\frac{3\gamma+8v+12}{12}.
\]

따라서 건강한 국소 kinetic의 필요충분 부호 조건은

\[
\boxed{3\gamma+8v+12\le0}
\]

이다. \(\gamma=-5,v=0\)은 Casimir throat 값을 그대로 유지하면서 kinetic/F를
\(+1/4\)로 만든다.

## 현재 판정

| 명제 | 판정 |
|---|---|
| 기존 지수형 target + 건강한 nonminimal scalar | `REFUTED` |
| canonical scalar 모드 수 증가로 기존 target 복구 | `REFUTED` |
| 2차 기하 co-design \(\gamma=-5,v=0\) | `LOCAL CONDITIONAL PASS` |
| co-design의 점근평탄 전역 연장 | `OPEN` |
| \(F=1-\xi\phi^2\), 단일값 \(V(\phi)\) 재구성 | `OPEN` |
| 전체 radial/angular perturbation 안정성 | `OPEN` |

다음 루프는 \(3\gamma+8v+12\le0\) 영역에서 전역 \(b(r),\Phi(r)\)를 구성하고,
\(F>0\), kinetic \(\ge0\), 점근평탄성, 단일값 potential을 동시에 검사하는 것이다.

## 재현

```powershell
uv run pytest tests/test_nonminimal_global_reconstruction.py -q
uv run python examples/physics/nonminimal_global_reconstruction_gate.py
```
