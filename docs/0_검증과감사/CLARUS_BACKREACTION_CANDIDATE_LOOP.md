# 클라루스 음의 동력원 backreaction 후보 루프

## 1. 목표

앞 루프의 `FRONTIER_A` 두 후보를 실제 웜홀 source 성분과 중력 크기에
대입했다.

1. 비최소 CE + Casimir 경계의 성분별 Einstein tensor matching
2. 비최소 CE massive vacuum polarization의 1m backreaction 크기

## 2. 이상적 Casimir tensor의 성분 matching

zero-redshift Morris--Thorne throat에서

\[
C=\frac{c^4}{8\pi G r_0^2}
\]

로 정규화하면 기하가 요구하는 throat stress는

\[
(\rho,p_r,p_t)_{\rm geom}
=\left(b',-1,\frac{1-b'}2\right)C
\]

이다. 판의 normal을 radial 방향으로 둔 이상적 전자기 Casimir tensor는

\[
(\rho,p_r,p_t)_{\rm Casimir}=(-u,-3u,+u)
\]

형태다.

$\rho$와 $p_r$를 동시에 맞추면

\[
b'=-\frac13,
\qquad u=\frac C3.
\]

이때

\[
(\rho,p_r,p_t)_{\rm geom}
=\left(-\frac13,-1,\frac23\right)C,
\]

\[
(\rho,p_r,p_t)_{\rm Casimir}
=\left(-\frac13,-1,\frac13\right)C.
\]

따라서 tangential residual은

\[
\Delta p_t=+\frac13C
\]

다. 이상적 Casimir tensor 단독은 이 zero-redshift control geometry의 전체
성분과 일치하지 않는다. 비최소 scalar stress 또는 nonconstant redshift가
부족한 압력을 공급해야 한다.

이 판정은 zero-redshift·평행판 방향성 control에 한정된다. 다른 redshift,
구면 Casimir 경계 또는 equation of state가 해를 만들 가능성을 보편적으로
반증하지 않는다. 실제 Casimir 웜홀 연구도 경계기하와 압력 관계를 별도로
선택한다.

## 3. CE massive vacuum polarization 크기

상관길이 $\xi$인 massive field의 large-mass DeWitt--Schwinger 항은 차원상

\[
\rho_{\rm vac}\sim
C_D\frac{\hbar c\,\xi^2}{r_0^6}
\]

이고, throat curvature source scale은

\[
\rho_{\rm req}\sim\frac{c^4}{8\pi G r_0^2}.
\]

$C_D=1$, $r_0=1$m, $\xi=6.65\times10^{-15}$m를 사용하면

\[
\frac{r_0}{\xi}=1.5038\times10^{14}
\]

로 large-mass control은 충분히 깊지만

\[
\frac{\rho_{\rm vac}}{\rho_{\rm req}}
=2.90\times10^{-97}.
\]

order-one coefficient의 같은 field가

\[
N\sim3.44\times10^{96}
\]

개 필요할 규모다. 정확한 tensor coefficient나 부호는 아직 계산하지
않았지만, order-one 또는 통상적인 loop 계수 변화로 97자릿수 격차를
메울 수 없다. 따라서 CE의 29.65MeV/6.65fm heavy pole을 이용한 **거시적
1m vacuum-polarization 단독 후보는 negative control에서 탈락**한다.

이 결론은 massless/near-zero mode, 거대한 species 수, nonperturbative
collective state에는 적용되지 않는다. 그러나 그런 sector는 현재 CE
spectral density에 없다.

## 4. 후보 funnel 갱신

| 후보 | 새 판정 |
|---|---|
| 비최소 CE + Casimir 경계 | `FRONTIER_A`, 보조 tangential pressure 필요 |
| 비최소 CE massive vacuum polarization | `DEFERRED_MACRO`, $2.90\times10^{-97}$ |
| CE+SM fermion magnetic Casimir | `FRONTIER_B`, CE topology mapping 필요 |
| CE double-trace | `FRONTIER_B`, CE boundary interaction 필요 |
| resonance-$Q$ | `DEFERRED`, stress 미식별 |
| beyond-Horndeski | `EXTERNAL_EXTENSION` |
| phantom | `REJECTED` |

## 5. 현재 최대 가능성

CE-native 후보 중 W2b로 가장 가까운 경로는 하나다.

\[
\boxed{
\text{비최소 }\xi R\Phi^2
+\text{ Casimir 경계}
+\text{ anisotropic pressure matching}
}
\]

다음 결정식은 일반 redshift $\varphi(r)$를 포함한

\[
G_{\mu\nu}[b,\varphi]
=\frac{8\pi G}{c^4}
\left(langle T_{\mu\nu}^{\rm Casimir}\rangle_{\rm ren}
+T_{\mu\nu}^{\Phi,\xi}\right)
\]

의 throat series와 보존식이다. 여기서 $\Delta p_t=C/3$을 regular scalar
profile이 공급하면서 $F=1-\xi\Phi^2>0$이고 ghost/gradient eigenvalue가
양수인지 검사해야 한다.

후속 `CASIMIR_GENERAL_REDSHIFT_THROAT_LOOP.md`에서 일반 redshift를 먼저
허용하자 이상적 Casimir tensor 단독의 국소 throat series가
$b'_0=-1/3$, $r_0\varphi'_0=-1/2$로 닫혔다. 보존식은 radially varying
boundary에 $r_0a'_0/a_0=1/2$를 요구한다. 따라서 비최소 scalar는 국소
잔차의 필수항이 아니라 전역 연장 실패 시의 completion 후보로 재분류됐다.

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_clarus_backreaction_candidates.py -q
uv run python examples/physics/clarus_backreaction_candidate_gate.py
```
