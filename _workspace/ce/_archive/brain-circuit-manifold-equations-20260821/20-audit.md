# 상태 감사

Status: COMPLETE

Gate: PASS

안정 스냅샷: `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`

## 감사 결론

수동적 flow-pullback과 능동적 reachability-energy를 분리한 Revision 1은 수학·타입·무차원성 게이트를 통과했다. 남은 P0/P1은 없다. 이는 empirical brain claim의 승격이 아니라 조건부 수학 구조의 완결 판정이다.

## 수정 이력

초기 초안은 Gramian pseudoinverse를 넓은 의미의 Riemann/sub-Riemannian metric처럼 표현했고, activity-dependent $p$의 도함수를 누락할 수 있었다. 수학 감사에서 이를 반례로 확인한 뒤 다음과 같이 좁혔다.

1. $g_{\rm pass}=J_T^\top G_TJ_T$를 passive pullback으로 분리하고 $\operatorname{rank}J_T=q$를 필수 조건으로 두었다.
2. $E_T^*(v)=v^\top\mathcal W_c^\dagger v$를 reachable subspace의 finite-horizon endpoint minimum-energy로 제한했다.
3. 기본 tangent branch에는 $C^1$, circuit-response branch에는 $C^2$와 $\varepsilon$-미분가능성을 명시했다.
4. $\dot h$, $\dot a$, $\dot A$, $\dot{\mathcal B}$, $\dot\Phi$, $\dot H$를 모두 적어 total circuit response를 재현 가능하게 만들었다.
5. $b_i$와 $\theta_i$는 $b_i-\theta_i$로만 나타나므로 독립 calibration 없이는 식별되지 않는다고 명시했다.

## 형식 지위

| 항목 | 지위 | 전제 또는 입력 | 판정 |
|---|---|---|---|
| A6.1 bounded delayed neural dynamics | [정의] | normalized state, signed $W$, frozen integer delays | PASS |
| A6.2--A6.3 tangent system | [조건부 산출] | $C^1$, frozen/exogenous $p,B$ | PASS |
| A6.4--A6.7 passive pullback | [조건부 정리] | fixed history/input/chart, $G_T\succ0$, full-rank $J_T$ | PASS |
| A6.7a--A6.7c circuit response | [조건부 산출] | $C^2$, differentiable reference path, fixed non-$W$ coefficients | PASS |
| A6.8--A6.10a reachability energy | [조건부 정리] | fixed LTV system, $R\succ0$, reachable terminal direction | PASS |
| A6.11 eligibility plasticity | [공리: 모델 선택] | modulator, clock, projection/event semantics | OPEN, central theorem에 미사용 |
| A6.12 anatomical metric | [정의] | measured embedding $X(\sigma,t)$ | PASS as definition |
| A6-P와 anatomical folding의 bridge | [미완성] | longitudinal anatomy, material/growth law, observation map | BLOCKED_INPUT |

## 반례 감사

- activity-dependent $p(a)$를 frozen coefficient로 미분하는 부모 주장은 완전 반례로 폐기했다.
- rank-deficient $J_T^\top G_TJ_T$를 Riemann metric이라고 부르는 부모 주장은 폐기했다.
- one-horizon Gramian inverse를 일반적인 sub-Riemannian metric이라고 부르는 부모 주장은 폐기했다.
- $B=0$, saturation, nonnormal transient, actuator/cost dependence, hard threshold를 경계 증인으로 보존했다.
- A3--A5의 threshold·clip·RMS·ridge·horizon retune은 predecessor 계약대로 retired다.

## 무차원 감사

활성도·drive·input·cost를 고정 reference scale로 무차원화했다. $s_{\rm pass}$, generalized eigenvalue $\Lambda$, determinant ratio, energy ratio만 log/square-root kernel에 들어간다. raw voltage·current·seconds를 넣는 물리 모델은 먼저 별도 기준척도로 정규화해야 한다.

집중 checker 결과는 `tests/test_dimensionless.py`의 A6 등록을 포함해 `17 passed`다. 이 결과는 차원 정합만 보이며 생물학적 타당성을 증명하지 않는다.

## 실행 권한 경계

이번 gate는 synthetic math witness와 정본 서술 갱신까지만 허용한다. 실제 response asset, confirmation cohort, physical cortical-fold fit은 열지 않는다.
