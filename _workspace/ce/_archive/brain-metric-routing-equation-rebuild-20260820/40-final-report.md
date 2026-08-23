# 실제 신경 계량–라우팅 식 재정비 최종 보고

Status: COMPLETE

## 결과

기존의 $g=C^{-1}$ 중심 서술을 실제 뇌의 primary metric에서 내리고, 다음 두 성분을 분리한 관측식을 채택했다.

$$
\boxed{
\mathcal B_{j,c}^{A\to B}(z)
=\left(
G_{j,c}^{o\leftarrow A}(z),
R_{j,c}^{A\to B}
\right)
}
$$

첫 성분은 미래 행동·과업 출력 분포의 조건부 Fisher pullback이다. 둘째 성분은 source history가 target future의 held-out log score를 개선한 양이다. 첫 성분은 output-relative geometry이고, 둘째는 lagged conditional predictive transfer다.

## 새로 닫힌 정리

좌표변환 $z'=\phi(z)$에서

$$
G'=J_\phi^{-T}GJ_\phi^{-1},
\qquad dz'^TG'dz'=dz^TGdz
$$

가 성립한다. 선요소, generalized spectrum, AIRM, likelihood-ratio score는 적절한 비교 조건에서 무차원이다. Gaussian likelihood의 정확한 Fisher는 mean 항뿐 아니라 state-dependent covariance 항과 nuisance-history 평균을 포함한다.

## 새로 닫힌 no-go

- $C^{-1}$은 일반 nonlinear chart의 local tensor가 아니다.
- 같은 $G$에서 다른 $R$, 같은 $R$에서 다른 $G$가 가능하다.
- hidden common input만으로 $A\to B$ edge 없이도 $R>0$가 가능하다.
- 관측 joint distribution만으로 $W\to G\to x$ mediation을 식별할 수 없다.
- 한 점 또는 상수 SPD로 curvature를 추정할 수 없다.

따라서 metric과 routing을 하나의 scalar로 합치거나 인과 화살표로 잇지 않는다. 현재 정직한 context 보고값은

$$
\Xi_j=\left(
d_{\rm AI}(\bar G_{j,c_0},\bar G_{j,c_1}),
\Delta R^{A\to B}_{j,\rm ctx}
\right)
$$

이라는 ordered pair다.

## 다음 단계

이제 canonical brain paper를 이 식에 맞춰 갱신한다. 그 뒤 로컬 실제 시계열의 same-session neural/behavior clock, unit provenance, context와 held-out block 적격성만 먼저 판정한다. 적격할 때만 별도 empirical run에서 $G$와 $R$을 독립 추정한다. 적격하지 않으면 새 synthetic seed를 만들지 않고 `BLOCKED_INPUT`으로 멈춘다.
