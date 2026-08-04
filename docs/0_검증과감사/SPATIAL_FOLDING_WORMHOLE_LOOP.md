# 공간 접힘 순간이동: 통과 가능 웜홀 루프

## 1. 연구 명제

“공간을 접어 이동한다”를 두 입구의 외부거리 (L)보다 내부 고유거리
(\ell)이 훨씬 짧은 통과 가능 웜홀로 정의한다. 이는
(\Delta t=0,\Delta x>0) 점프가 아니라 유한한 내부 경로를 국소적으로
광속 이하로 통과하는 방식이다.

## 2. 기하학적 단축 정리

내부 속도를 (v=\beta c), (0<\beta<1)라 하면

\[
t_{\rm throat}=\frac{\ell}{\beta c}>0,
\qquad
\tau_{\rm traveler}=t_{\rm throat}\sqrt{1-\beta^2}>0.
\]

외부 관측자가 (L/t_{\rm throat})로 정의한 겉보기 속도는

\[
\frac{v_{\rm app}}c=\beta\frac{L}{\ell}
\]

이므로 (L\gg\ell)이면 (c)를 크게 넘을 수 있다. 여행자의 국소속도와
고유시간은 정상이다. 따라서 기하가 이미 주어졌다는 조건 아래
“국소 FTL 없이 외부 경로보다 빠른 이동”은 성립한다. 정확히 순간적인
이동은 아니다.

## 3. 목의 NEC 요구량

zero-redshift Morris-Thorne control metric에서 목 (r_0)는
(b(r_0)=r_0)이고 flare-out은 (b'(r_0)<1)이다. 목의 radial null
projection은

\[
\rho+p_r=
\frac{c^4}{8\pi G r_0^2}\left[b'(r_0)-1\right]<0.
\]

단순 control profile (b(r)=r_0^2/r)에서는 (b'(r_0)=-1)이므로

\[
|\rho+p_r|=\frac{c^4}{4\pi G r_0^2}.
\]

(r_0=1\,\mathrm m)이면 약 (9.63\times10^{42}\,\mathrm{J/m^3})의
음의 null projection이 필요하다.

## 4. CE 카시미르 셀의 단위 감사

기존 문서가 명시한 셀 추정

\[
\frac{0.62\,\mathrm{keV}}{294\,\mathrm{fm^3}}
\]

을 SI로 변환하면

\[
|u_{\rm CE}|=3.3787\times10^{26}\,\mathrm{J/m^3},
\qquad
|u_{\rm CE}|/c^2=3.7593\times10^9\,\mathrm{kg/m^3}.
\]

기존 (3.4\times10^{29}\,\mathrm{kg/m^3}) 표기는 약
(9.0\times10^{19})배 큰 단위변환 오류다.

1 m control throat와 비교한 국소 밀도 격차는

\[
\frac{|\rho+p_r|}{|u_{\rm CE}|}
\approx2.85\times10^{16}.
\]

또한 (\xi=6.65\,\mathrm{fm}) 상관길이를 1 m까지 늘리는 데 필요한
형식적 길이비는

\[
Q_{\rm coherence}\ge\frac{1\,\mathrm m}{6.65\times10^{-15}\,\mathrm m}
\approx1.50\times10^{14}.
\]

길이 coherence가 늘어난다는 것과 renormalized negative stress가 같은
비율로 증폭된다는 것은 다른 명제다. 현재 CE 공명 문서는 전자를 웜홀
stress tensor로 연결하는 식을 유도하지 않았다.

## 5. 판정

| 명제 | 판정 |
|---|---|
| 주어진 웜홀이 외부거리보다 빠른 국소 인과 이동을 제공 | `PROVED / KINEMATIC` |
| 정확한 0초 순간이동 | `REFUTED` |
| 통과 가능 목이 NEC 위반을 요구 | `PROVED / CONTROL METRIC` |
| 기존 CE 카시미르 SI 변환 | `REFUTED / CORRECTED` |
| 정적 CE 셀이 1 m 목의 국소 NEC 크기를 충족 | `NO`, 약 (2.85\times10^{16})배 부족 |
| CE 공명이 필요한 stress tensor와 공간 coherence를 생성 | `OPEN` |
| CE 물리적 웜홀 해 | `NOT ESTABLISHED` |

## 6. 다음 필수 브리지

다음 식을 실제로 닫아야 한다.

\[
G_{\mu\nu}[g_{\rm WH}]
=\frac{8\pi G}{c^4}
\langle T_{\mu\nu}^{\rm CE,res}\rangle_{\rm ren}.
\]

필수 항목은 공명 상태의 renormalized stress tensor, 보존
(\nabla_\mu T^{\mu\nu}=0), quantum inequality, backreaction, 목 안정성,
입구 생성·분리 에너지다. 현재 Q0에서 gravity/CE와 renormalized stress
sector가 열려 있으므로 이 루프의 최종 지위는 `KINEMATIC PASS / PHYSICAL
BRIDGE OPEN`이다.

## 7. 기존 이론과의 대조

- Ford–Roman의 quantum inequality 분석은 거시적 정적 웜홀에서 음의
  에너지의 크기와 지속시간을 동시에 제한하여, Planck 크기 또는 목보다
  극도로 얇은 음의 에너지 층을 요구하는 문제를 제시한다:
  <https://arxiv.org/abs/gr-qc/9510071>
- Gao–Jafferis–Wall은 AdS/BTZ의 두 경계를 결합해 음의 평균 null energy로
  다리를 통과 가능하게 만들지만, 외부 인과성 위반에는 사용할 수 없다고
  명시한다: <https://arxiv.org/abs/1608.05687>
- Maldacena–Milekhin–Popov는 4차원 Einstein–Maxwell 이론과 질량 없는
  charged fermion의 Casimir-like 음의 에너지로 구체적 해를 제시한다.
  이는 음의 에너지라는 말만이 아니라 완전한 장 내용과 Einstein 방정식의
  해가 필요하다는 좋은 control이다: <https://arxiv.org/abs/1807.04726>

따라서 CE의 다음 연구 목표는 공명 (Q)를 임의로 목 크기에 맞추는 것이
아니라, 위 control들과 같은 수준으로 장 내용에서 renormalized stress
tensor와 전역 해를 함께 유도하는 것이다.

후속 `WORMHOLE_SOURCE_CANDIDATE_LOOP.md`에서 정적 Casimir, resonance-$Q$,
canonical/phantom scalar, 양자 음의 에너지 상태와 외부 웜홀 대조군을 같은
S0--S6 gate로 1차 screening했다.

## 실행

```powershell
uv run --extra dev python -m pytest tests/test_spatial_folding.py -q
uv run python examples/physics/spatial_folding_gate.py
```
