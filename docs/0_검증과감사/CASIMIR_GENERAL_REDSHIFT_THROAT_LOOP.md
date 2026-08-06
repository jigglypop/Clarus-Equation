# Casimir 일반 redshift 국소 throat-series 루프

## 1. zero-redshift 잔차의 재검사

앞 루프에서 이상적 Casimir tensor는 zero-redshift throat의 $\rho,p_r$를
맞추지만 $p_t$가 $C/3$ 부족했다. 이번에는 유한한 일반 redshift
$\varphi(r)$를 허용해 이 잔차가 구조적 불가능인지 검사했다.

## 2. 일반 throat 극한

Morris--Thorne metric을

\[
ds^2=-e^{2\varphi(r)}dt^2
+\frac{dr^2}{1-b(r)/r}+r^2d\Omega^2
\]

로 두고

\[
C=\frac{c^4}{8\pi G r_0^2},
\qquad u=r_0\varphi'(r_0)
\]

라 하자. $b(r_0)=r_0$이고 $\varphi'(r_0)$가 유한하면 throat 극한은

\[
\frac{(\rho,p_r,p_t)_{\rm geom}}C
=\left(b'_0,-1,\frac{1-b'_0}{2}(1+u)\right)
\]

이다.

normal이 radial인 이상적 전자기 Casimir stress를

\[
\frac{(\rho,p_r,p_t)_{\rm Casimir}}C
=\left(-\frac13,-1,+\frac13\right)
\]

로 맞추면 세 성분의 유일한 throat 조건은

\[
\boxed{b'_0=-\frac13,\qquad r_0\varphi'_0=-\frac12}
\]

이다. $b'_0<1$이므로 flare-out을 만족하고 redshift slope도 유한하다.
따라서 zero-redshift에서 남았던 tangential residual은 일반 redshift로
정확히 닫힌다.

## 3. 보존식과 경계 profile

정적 구대칭 anisotropic stress의 보존식은

\[
p'_r=-(\rho+p_r)\varphi'+\frac2r(p_t-p_r)
\]

이다. 위 throat 값에서는

\[
p'_r(r_0)=\frac{2C}{r_0}.
\]

Casimir 크기가 plate separation $a(r)$에 대해 $U\propto a^{-4}$이고
$p_r=-3U$이면 같은 derivative를 얻는 조건은

\[
\boxed{r_0\frac{a'_0}{a_0}=+\frac12}
\]

이다. 실행 gate에서 Einstein tensor 세 성분, flare-out, 유한 redshift와
anisotropic conservation이 동시에 통과했다.

## 4. 증명 범위

증명된 것은 이상적 Casimir equation of state를 source로 준 국소 throat
Taylor data의 존재다.

```text
b(r0)=r0
b'(r0)=-1/3
r0 phi'(r0)=-1/2
r0 a'(r0)/a(r0)=+1/2
```

아직 증명되지 않은 것은 다음이다.

- 이 Taylor data가 두 개의 asymptotically regular 영역으로 전역 연장됨
- redshift가 어디에서도 horizon을 만들지 않음
- radially varying Casimir boundary의 실제 기하와 재규격화 tensor
- CE 클라루스장이 이상적 전자기 Casimir pressure ratio를 가짐
- quantum inequality, boundary backreaction과 선형 perturbation 안정성
- 필요한 $a_0\simeq3.66\times10^{-18}$m 경계의 물리적 실현

따라서 이것은 `W2b-local/control pass`이지 CE 물리적 W2 완성은 아니다.

## 5. 후보 판정 갱신

| 후보 | 판정 |
|---|---|
| Casimir + zero redshift | tensor mismatch negative control |
| Casimir + 일반 redshift + radial boundary | `W2b-local/control pass` |
| 비최소 CE + Casimir | global completion 예비후보 |
| CE massive vacuum polarization | 거시적 단독 탈락 |

다음 루프는 위 Taylor data를 초기조건으로 일반 ODE를 적분해

\[
b(r)<r,\qquad e^{2\varphi(r)}>0,\qquad b(r)/r\to0
\]

를 유지하는 전역 branch가 존재하는지 검사하는 것이다.

후속 `CASIMIR_GLOBAL_AND_WAVELENGTH_RESONANCE_LOOP.md`에서 고정 Casimir
압력비의 전역 asymptotic 조건을 풀었다. 유한 redshift는
\(\rho\sim r^{-8/3}\)을, 유한 질량은 \(n>3\)을 요구해 amplitude-only envelope는
전역 no-go다. 특정 파장 역산은 $7.33\times10^{-18}$m, 약 169GeV로 CE
light pole보다 5708배 높은 mode를 요구했다. 이는 $b'=-1$인 legacy null
control의 역사값이다. 현재 $b'=-1/3$ full tensor는
$a=4.0536\times10^{-18}$m를 정하고, 추가 ideal-planar $\lambda=2a$ 선택의
형식 scale은 152.932GeV다. 어느 값도 spherical boundary eigenmode 또는
단일 Casimir carrier의 유도가 아니다.

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_clarus_backreaction_candidates.py -q
uv run python examples/physics/clarus_backreaction_candidate_gate.py
```
