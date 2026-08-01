# Weighted directed graph dynamics loop

> 상태: `synthetic PASS / AML310 exploratory FAIL 0/4 / AML32 untouched`

## 1. 검사한 중첩식

국소 기준은 다음으로 고정했다.

$$
G_0:\quad
\widehat x_{i,t+h}
=
f_i(x_{i,t},x_{i,t-1},x_{i,t-2}).
$$

대칭·비음수 correlation graph의 diffusion message를 추가한 모델은

$$
G_1:\quad
\widehat x_{i,t+h}
=
f_i(\mathrm{local})
+b_i^\top
\sum_j a_{ij}
\left(
x_{j,t:t-2}-x_{i,t:t-2}
\right)
$$

이고, one-step local innovation으로 train-only signed edge를 고른 directed
effective graph는 다음 두 형태로 검사했다.

$$
G_{2a}:\quad
\widehat x_{i,t+h}
=
f_i(\mathrm{local})
+c_i^\top\sum_j w_{ij}x_{j,t:t-2},
$$

$$
G_{2b}:\quad
\widehat x_{i,t+h}
=
f_i(\mathrm{local})
+\sum_{j\in N_i}\beta_{ij}x_{j,t}.
$$

마지막 $G_3$은 train-only population PC1 median으로 두 regime을 정하고
regime마다 $G_{2b}$의 graph를 별도로 학습했다.

## 2. 누수·반례 게이트

- chronological `60/20/20` split과 5-frame embargo
- 결측치 대치, 표준화, adjacency, ridge 선택은 test를 보지 않음
- graph는 항상 $h=1$에서 학습하고 $h=6,30$에도 그대로 재사용
- $PAP^\top$ node permutation 19개로 weight와 degree 분포를 보존한 null
- target neuron이 아니라 recording/animal을 반복 단위로 사용
- activity row에 neuron class label이 없으므로 anatomical connectome과의
  동일시는 금지

합성 directed-ring, independent-AR, switching-ring 대조군과 test-block
변조 검사는 모두 통과했다. 구현은
`reality_stone.clarus.graph_dynamics`, 회귀는
`tests/test_graph_dynamics.py`에 있다.

## 3. AML310 탐색 결과

표의 값은 recording별 median
$\Delta R^2=R^2_{\rm directed}-R^2_{\rm local}$이다.

| model | horizon | 110803 | 105254 | 141211 | 142022 | pass |
|---|---:|---:|---:|---:|---:|---:|
| aggregate $G_{2a}$ | 1 | -0.000012 | -0.000040 | -0.000135 | -0.000060 | 0/4 |
| aggregate $G_{2a}$ | 6 | -0.0061 | -0.0230 | -0.0858 | -0.0445 | 0/4 |
| aggregate $G_{2a}$ | 30 | -0.0143 | -0.0326 | -0.1010 | -0.0430 | 0/4 |
| sparse VAR $G_{2b}$ | 1 | -0.00002 | -0.00004 | -0.00013 | -0.00009 | 0/4 |
| sparse VAR $G_{2b}$ | 6 | -0.0079 | -0.0351 | -0.1502 | -0.0540 | 0/4 |
| sparse VAR $G_{2b}$ | 30 | -0.0408 | -0.0643 | -0.1576 | -0.0896 | 0/4 |
| two-regime $G_3$ | 1 | -0.00002 | -0.00007 | -0.00017 | -0.00013 | 0/4 |
| two-regime $G_3$ | 6 | -0.0150 | -0.0660 | -0.0666 | -0.0973 | 0/4 |
| two-regime $G_3$ | 30 | -0.0247 | -0.0824 | -0.1786 | -0.1476 | 0/4 |

$G_3$의 positive-target fraction은 최대 `0.433`이고 recording-level
rewired-null의 최저값도 $p=0.40$이었다. 따라서 양의 효과가 작은데 검정력이
모자란 상황이 아니라, 복잡한 graph일수록 local 기준보다 일관되게 나빠졌다.

## 4. 판정과 다음 루프

$$
\boxed{
\text{이 네 moving recording에서는 weighted directed graph residual을 기각한다.}
}
$$

이 판정은 anonymous activity에 대한 predictive effective graph를 기각한
것이다. anatomical connectome 전체나 다른 개입 자료의 signal-propagation
graph를 반증하지 않는다.

공식 OSF의 `AML32_moving.tar.gz` 7-recording 패널은 SHA-256
`6b71a6ba1a5d2f1ef3bf9661e845e1e52634bae217fc0c2630a83fca07daed63`
으로 검증했지만, 실패한 graph를 confirmatory panel에 반복 적용하지 않고
untouched로 유지한다.

다음 후보는 개별 neuron graph가 아니라 지연 임베딩된 population-state
manifold 위의 diffusion/analog transition이다.
