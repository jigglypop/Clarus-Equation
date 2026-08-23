# M4-R 수학 검산

Status: COMPLETE

## 형식과 방향

**[정리]** 계약의 cue/value-start trace와 terminal error는

$$
p_{i,c/v}^{(\lambda)},\delta_{i,c/v}\in\mathbb R^d,
\qquad
E_{i,c/self}^{(\lambda)}=\delta_{i,c/v}(p_{i,c/v}^{(\lambda)})^\top
\in\mathbb R^{d\times d}
$$

이므로 row-post/column-pre recurrent matrix와 shape가 같다. $h=0,\ldots,H-1$의 delayed
presynaptic trace가 terminal $h=H$ postsynaptic error와 결합되므로 time index도 닫힌다.

value $\bar v_i$의 unit normalization은 presentation에서 한 번만 수행한다. 그 뒤 trace와
error의 각 좌표는 해당 pre/post neuron activation만 사용한다. $D_\lambda$의 Frobenius
normalization과 projection/install bound는 local trace가 아니라 후보 설치 apparatus다.

## 무차원 감사

| 코어 인자 | 차원 벡터 | 무차원 | 정규화 |
|---|---|---|---|
| activation, $\bar v_i$, $p$, $\delta$, $E$, $D$, $W$ | $(0,0,0,0)$ | yes | simulator coordinate |
| $\cos_\epsilon$ 인자 | $(0,0,0,0)$ | yes | Euclidean norm ratio |
| $\lVert W_k-W\rVert_F^2/5^2$ | $(0,0,0,0)$ | yes | install bound 5 |
| $-(J_k-J_{min})/0.1$ | $(0,0,0,0)$ | yes | dimensionless temperature 0.1 |
| $F_t$, elementwise floor $10^{-6}$ | $(0,0,0,0)$ | yes | same coordinate variance |

차원 상태는 무차원이다. 이 판정은 알고리즘의 타당성이나 생물학적 정당성을 뜻하지 않는다.

## 반례와 수정

**[정리: no-go]** pooled M1 eligibility에 양의 scale만 달리하면 identity를 제외한 후보는
동일 방향이다. 따라서 후보 선택이 T1의 factor-frequency collapse를 구조적으로 고칠 수
없다. 이 M4-0 부모 주장은 활성 후보에서 제거했다.

**[정리: leakage no-go]** 관측 `(0,0),(0,1),(1,0)` target의 affine 조합은 factor direct-sum
codebook에서 held-out `(1,1)` target과 같다. 이를 candidate score에 넣으면 held-out answer로
weight를 선택한다. 개정 계약은 이 항을 완전히 삭제했다.

**[산출]** M4-R은 closed-form $W^*$, SVD 또는 least-squares matrix를 계산해 설치하지 않는다.
그러나 $\delta=\bar v-x_H$를 쓰므로 target-error supervised rule이다. 허용 지위는
`experience-supervised candidate selection`이며 answer-blind plasticity가 아니다.

## 식별 가능성

**[미완성]** Loop 9 composition 성공은 수식만으로 보장되지 않는다. 각 epoch에서

$$
\operatorname{rank}_{10^{-6}}[
\operatorname{vec}(D_{0.50}),
\operatorname{vec}(D_{0.80}),
\operatorname{vec}(D_{0.95})]
$$

와 pairwise cosine을 기록한다. rank가 1이면 M4-R은 그 seed에서 M1-equivalent direction이며
성공으로 세지 않는다. held-out exclusion, formula-discovery/development-validation/confirmation
분리와 matched controls가 경험 판정을 맡는다.

Gate: PASS

## Revision 2 수학 판정

**[산출]** basic discovery는 residual write capability와 candidate-selection necessity를
분리했다. Loop 8 절대 과제는 8/8이지만 fixed no-selection도 8/8이므로 selection 주장은
반례를 만났다. Loop 9 절대 과제는 4/8이며 no-selection이 learned와 동률 또는 우세했다.

**[산출]** max-scale 75% trigger가 다섯 seed-task arm에서 성립해 등록된 fold 시험을 열 수
있다. 거의 선형인 projection에서 fold는 $D/\sqrt F$를 통해 entrywise support를 평탄화할
수 있지만 argmin selection의 필요성을 보장하지 않는다. 따라서 Revision 2는 saturation
hypothesis의 단일 검정이며, no-selection과 fold-ablation보다 0.20 이상 좋아야만 산출로
보존한다. 실패 뒤 추가 hyperparameter 수정은 허용되지 않는다.
