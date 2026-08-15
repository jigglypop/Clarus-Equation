# Phase A 공유 입력·문맥별 전이 식별 수학 검산

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-connectome-geometric-memory-20260816/11-math.md`

## 1. 범위와 결론

이 레인은 계약 `PA-N1`, `PA-T1`, `PA-T2`, `PA-D1`, `PA-D2`, `PA-I1`--`PA-I3`, `PA-H1`, `PA-H2`, `PA-X1`을 고정된 known-identity chart의 유한차원 이산시간 선형계 범위에서 독립 검산한다. 제품 코드나 앞선 SCC·metric 구현은 수입하지 않았다.

핵심 결론은 다음과 같다.

1. 문맥별 $A_z$와 공유 $B$를 한꺼번에 유일 식별하는 필요충분 finite-design 조건은 문맥 indicator를 포함한 stacked design의 row rank가 $Kn+m$인 것이다.
2. 각 문맥의 상태행렬이 따로 full rank인 것만으로는 부족하다. 상태 block을 제거한 뒤에도 입력이 $m$개 독립 방향을 남겨야 한다.
3. rank가 부족하면 관측을 정확히 똑같이 만드는 서로 다른 $(A_z,B)$가 존재하므로 exact certificate는 반드시 거부해야 한다.
4. unknown mix에서는 관측좌표의 예측계수만 식별되고 latent edge support는 similarity orbit 안에서 바뀐다.
5. `PA-H1`과 `PA-H2`는 정리가 아니라 development 비교다. 문맥 전이가 실제로 같거나 $B=0$이면 strict 개선·shuffle 악화는 사라질 수 있다.

## 2. stacked design과 공유 $B$의 식별 조건

문맥 $z\in\{0,\ldots,K-1\}$에 속한 표본을 열로 모아 $X_z\in\mathbb R^{n\times N_z}$, $U_z\in\mathbb R^{m\times N_z}$, $Y_z\in\mathbb R^{n\times N_z}$라 하자. 전체 상태 block과 입력을

$$
D=\operatorname{blockdiag}(X_0,\ldots,X_{K-1}),
\qquad
U=[U_0\ \cdots\ U_{K-1}],
$$

$$
\Phi=
\begin{bmatrix}
D\\ U
\end{bmatrix}
\in\mathbb R^{(Kn+m)\times N},
\qquad
W=[A_0\ \cdots\ A_{K-1}\ B]
$$

로 두면 학습식은 $Y=W\Phi$다. 여기서 $D$의 각 block은 자기 문맥의 열에만 놓인다.

**PA-T1 조건부 정리.** noiseless known-identity 계에서 $(A_0,\ldots,A_{K-1},B)$는 finite design으로부터 유일하게 식별될 필요충분조건이

$$
\operatorname{rank}\Phi=Kn+m
\tag{1}
$$

인 것이다. 이때

$$
W=Y\Phi^T(\Phi\Phi^T)^{-1}.
\tag{2}
$$

**증명.** 식 (1)이 성립하면 $\Phi\Phi^T$가 가역이므로 식 (2)가 나온다. 반대로 rank가 $Kn+m$보다 작으면 $h^T\Phi=0$인 $h\ne0$이 있다. 임의의 $a\ne0\in\mathbb R^n$에 대해 $\Delta W=ah^T$로 두면 $\Delta W\ne0$이지만 $(W+\Delta W)\Phi=W\Phi$다. 따라서 유일하지 않다.

식 (1)은 다음 두 조건과 동치다.

1. 모든 문맥에서 $\operatorname{rank}X_z=n$;
2. $P_D=D^T(DD^T)^{-1}D$라 할 때

$$
U(I-P_D)U^T\succ0.
\tag{3}
$$

식 (3)은 각 문맥의 상태로 선형 설명되지 않는 입력 variation이 합쳐서 $m$차원을 모두 span해야 한다는 뜻이다. 따라서 단순히 각 $X_z$가 full rank라고 보고 `shared B identified`라고 판정하면 안 된다.

noise가 있고 $\mathbb E[\epsilon_t\mid x_t,u_t,z_t]=0$이면 population second moment $\mathbb E[\phi_t\phi_t^T]\succ0$ 아래 conditional mean 계수는 유일하다. finite noisy 표본의 OLS가 exact truth와 일치한다는 뜻은 아니며, exact $10^{-10}$ gate는 noiseless fixture에만 적용한다.

## 3. rank-deficient 반례와 certificate

각 문맥에서 $U_z=R_zX_z$라 하자. 각 $X_z$가 full row rank여도 입력은 문맥별 상태에 완전히 포함된다. 임의의 $D_B\ne0\in\mathbb R^{n\times m}$에 대해

$$
A'_z=A_z-D_BR_z,
\qquad
B'=B+D_B
\tag{4}
$$

로 두면 모든 학습 열에서

$$
A'_zX_z+B'U_z=A_zX_z+BU_z.
$$

따라서 `PA-T2`는 [정리: 조건부 no-go]로 닫힌다. 구현 certificate는 수치 rank만 boolean으로 보지 말고 적어도 다음을 기록해야 한다.

- 선언한 tolerance와 singular values;
- required rank $Kn+m$와 observed rank;
- 문맥별 $\operatorname{rank}X_z$;
- 식 (3)의 최소 eigenvalue 또는 residualized-input singular value;
- rank 부족이면 coefficient/support certificate `false`.

ridge는 rank 부족을 수치적으로 풀어 한 해를 고를 수 있지만 데이터를 통해 유일 식별하게 만들지는 않는다. 그러므로 ridge inverse의 존재를 exact identification certificate로 쓰면 P0 오류다.

## 4. 관측 동치와 unknown-mix refusal

`PA-N1`의 선행 no-go는 그대로 유효하다. $x_{t+1}=A_zx_t+Bu_t$, $y_t=Cx_t$에서 미지의 가역 $H$에 대해

$$
x'_t=Hx_t,
\quad
A'_z=HA_zH^{-1},
\quad
B'=HB,
\quad
C'=CH^{-1}
\tag{5}
$$

로 두면 모든 허용 입력에서 관측열은 같다. $A_z$와 $A'_z$의 off-diagonal support는 다를 수 있다. 알려진 input label만으로는 이 gauge가 사라지지 않는다. latent intervention direction까지 고정 chart에서 알려져 있다는 추가 anchor가 있어야 한다.

따라서 `PA-I3`의 exact-edge certificate는 다음 conjunction일 때만 true가 될 수 있다.

$$
\texttt{known\_identity}
\land \texttt{declared\_linear\_class}
\land \texttt{full\_rank}
\land \texttt{finite\_valid\_inputs}.
\tag{6}
$$

`known_mask`와 `unknown_mix`에서는 식 (6)을 무조건 false로 두고 관측 subspace 또는 관측좌표 prediction만 채점한다. unknown mix에서 $CA_zC^{-1}$를 정확히 복원해도 그것을 latent $A_z$의 edge recovery라고 부르면 안 된다.

## 5. Gaussian NLL의 scale 선택

state-vector test sample이 $N_*$개이고 공통 isotropic scale이 $\sigma_{\rm eval}>0$이면 모델 $M$의 total score를

$$
\operatorname{NLL}_M
=\frac{N_*n}{2}\log(2\pi\sigma_{\rm eval}^2)
+\frac{1}{2\sigma_{\rm eval}^2}
\sum_i\lVert x_{i,+}-\widehat x_{i,+}^{(M)}\rVert_2^2
\tag{7}
$$

로 둔다. V1 primary에는 manifest에 생성 전에 고정된 **generator truth $\sigma$를 scorer-only 공통 scale로 사용**할 것을 권고한다. 이 값은 estimator API에 전달하지 않는다. 그러면

$$
\Delta_s
=\operatorname{NLL}_{\rm pooled}-\operatorname{NLL}_{\rm shared}
=\frac{\operatorname{SSE}_{\rm pooled}-\operatorname{SSE}_{\rm shared}}
{2\sigma^2}
\tag{8}
$$

이고 model별 scale 추정이 평균예측 비교를 오염시키지 않는다.

실자료로 확장할 때의 사전 지정 대안은 training 내부 calibration subset에서 하나의 공통 $\widehat\sigma_{\rm eval}$를 추정하고 두 모델에 같이 고정하는 것이다. test residual로 scale을 추정하거나 test마다 재조정하는 것은 금지한다. model별 train MLE scale은 predictive-density calibration까지 함께 비교하는 별도 secondary로만 허용하며 primary와 섞지 않는다. noiseless exact-recovery fixture는 $\sigma=0$ NLL을 만들지 않고 coefficient error로만 채점한다.

## 6. 자유도와 ridge 회계

intercept가 없고 모든 계수를 dense로 세면 nominal coefficient dof는 다음과 같다.

| 모형 | nominal dof |
|---|---:|
| pooled $A$, shared $B$ | $n(n+m)$ |
| context $A_z$, shared $B$ | $n(Kn+m)$ |
| context $A_z$, context $B_z$ | $nK(n+m)$ |
| bilinear $A(z)=A_0+\sum_{j=1}^q h_j(z)M_j$, shared $B$ | $n((q+1)n+m)$ |

따라서 계약 candidate의 pooled 대비 추가 dof는 $(K-1)n^2$이고 fully separated 대비 절약 dof는 $(K-1)nm$다. $q=K-1$인 full categorical contrast의 bilinear 모형은 shared-$B$ candidate의 재매개화일 뿐 새로운 독립 baseline이 아니다.

ridge $\lambda>0$을 쓰면 위 수는 parameter count다. effective dof는 해당 design $\Phi$에 대해

$$
\operatorname{dof}_{\rm eff}
=n\operatorname{tr}\!\left[
\Phi^T(\Phi\Phi^T+\lambda I)^{-1}\Phi
\right]
\tag{9}
$$

를 함께 보고한다. 후보마다 다른 $\lambda$를 endpoint를 보고 고르는 행위는 추가 선택 자유도다. V1은 동일한 사전 고정 $\lambda$를 사용하거나 development 내부 nested selection 횟수와 grid를 모두 ledger에 남겨야 한다.

## 7. seed와 leakage의 정확한 의미

`PA-I1`에서 독립성은 graph·trajectory·intervention이 인과적으로 무관하다는 뜻이 아니다. graph는 당연히 trajectory 생성에 사용된다. 요구되는 것은 randomness source의 domain separation이다.

권고 manifest key는 `(experiment_version, master_seed, role, graph_index, replicate_index)`이며 role을 `graph`, `trajectory`, `intervention`으로 분리한다. 각 seed는 이 tuple의 안정적인 hash 또는 `SeedSequence.spawn`으로 생성하고 다른 role의 RNG state를 소비하지 않는다. 동일 tuple은 byte-identical replay를 만들고, 한 role의 index 변경은 다른 role seed를 바꾸지 않아야 한다. 서로 다른 정수 seed만 사용했다는 사실은 수학적 통계 독립성 증명이 아니며, 독립 PRNG stream이라는 모델 가정 아래의 재현성 계약이다.

`PA-I2`를 위한 API 경계는 다음과 같다.

- fit 입력: training의 $x_t,u_t,z_t,x_{t+1}$와 사전 고정 hyperparameter만;
- scorer-only: truth $A_z,B$, test $x_{t+1}$, manifest truth $\sigma$;
- fit 반환 뒤에만 scorer가 coefficient error와 NLL을 계산;
- graph/trajectory/intervention seed와 confirmation receipt는 estimator object에 넣지 않음.

## 8. 주장별 최종 판정

| Claim ID | 수학 지위 | P 등급 | 판정 |
|---|---|---|---|
| `PA-N1` | [정리: no-go] | 없음 | 식 (5)와 exact fixture로 서로 다른 latent support의 동일 관측열이 성립한다. |
| `PA-T1` | [정리: 조건부] | 없음 | noiseless known-identity에서 식 (1)이 필요충분이며 식 (2)로 exact recovery한다. |
| `PA-T2` | [정리: 조건부 no-go] | 없음 | rank 부족이면 식 (4) 또는 null direction으로 다른 계수가 같은 출력을 만든다. certificate 거부가 필수다. |
| `PA-D1` | [정의] | 없음 | anatomy, latent causal support, 관측좌표 predictive transition을 별도 typed field로 두는 계약이 일관된다. |
| `PA-D2` | [정의] | 없음 | 선언한 $x,u,A_z,B,\epsilon,\sigma$와 standardized residual은 모두 무차원이다. 이는 식별 증명이 아니다. |
| `PA-I1` | [미완성: 구현] | P1 | domain-separated deterministic seed 규약은 닫혔으나 실제 generator의 isolation/replay test가 필요하다. |
| `PA-I2` | [미완성: 구현] | P1 | fit/scorer 경계는 닫혔으나 실제 함수 signature와 mutation test가 필요하다. |
| `PA-I3` | [미완성: 구현] | P1 | 식 (6)의 fail-closed 규칙은 옳다. known-mask/unknown-mix 실행 fixture가 필요하다. |
| `PA-H1` | [미완성: 경험 비교] | P1 | strict 우위 정리는 없다. $A_z$가 같으면 pooled와 동률일 수 있으므로 development 결과 그대로 판정한다. |
| `PA-H2` | [미완성: integrity 비교] | P1 | $B=0$ 또는 무신호 입력이면 shuffle 악화가 보장되지 않는다. nonzero intervention signal에서의 development kill test다. |
| `PA-X1` | 활성 제외 | 없음 | 이 선형 benchmark에서 SCC·기억·생물학·의식·AGI 결론은 나오지 않는다. |

현재 P1은 다음 구현 단계가 닫아야 할 명시적 gate이며 수학 레인의 누락된 증명은 아니다. 열린 수학 P0는 0개다.

## 9. 재현 검산

독립 fixture는 제품 코드를 import하지 않는다.

```powershell
.venv\Scripts\python.exe _workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/artifacts/verify_phase_a_math.py
```

검산 항목은 full-rank shared-$B$ exact recovery, 각 $X_z$가 full rank여도 생기는 rank-deficient 반례, similarity support no-go, nominal dof 차이, 공통-scale NLL 항등식, seed namespace replay다. 원문 출력은 `artifacts/verify_phase_a_math.log`에 보존한다.
