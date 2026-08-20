# 자기선택 recurrent deformation 계약

Status: COMPLETE

Mode: full

PREDECESSOR: `_workspace/ce/brain-memory-contrastive-predictive-routes-20260819`

## 질문

외부에서 정답 recurrent matrix를 계산해 설치하지 않고, 경험한 cue/value trajectory가 만든
terminal error와 local trace만으로 후보 deformation을 생성한 뒤 native internal rollout으로
하나를 선택하는 동일 규칙이 Loop 8 zero-store binding과 Loop 9 held-out composition을 모두
통과하는가?

## PREDECESSOR_EVIDENCE

| 증거 | 지위 | SHA-256 | 보존하는 좁은 주장 | 재시도 금지 |
|---|---|---|---|---|
| route ledger | `ACTIVE` | `38f8534d367eaa70a665fceeee24adeeb7e5009d7e91e9886b4a273ea7263962` | 새 후보는 M1의 Loop 8 성공과 T1의 Loop 9 실패를 함께 설명해야 한다 | threshold·seed·decoder만 바꾼 M1/T1 재실행 금지 |
| predecessor routes | `COMPLETE` | `e61fdfc2daa4e17ef63a99c982d4ed8b48900c3e7c326838d1949dd5ecd62475` | target-shuffle·no-write·time/sign controls가 필요하다 | M2/M3 이름만 바꾼 재실행 금지 |
| predecessor validation | `T1 STOP 11/16` | `1b20fbb8ef43feed9c044637ff4326dbd884edee23fa4df34b8abcaaa5952f99` | M1 binding은 불균형 factor schedule에서 stable composition을 보장하지 않는다 | M1 파라미터 재조정 금지 |
| predecessor final | `ALL SUCCESSOR CLAIMS STOP` | `7dfd0fdd939f7b2dc2d5d79c2e852e1d833ba5539df75ee312ae18d4f7597663` | teacher-forced write capability와 credit-rule 식별은 다르다 | shuffled transition과 동률인 규칙 승격 금지 |
| M0/M1 confirmation | `M0,M1 32/32` | `536590a9d38669c5c7fc7485b388f7c4af2d413e213de1cfa68613c287c8f8bb` | supervised rank-4는 capacity ceiling, M1은 Loop 8 ceiling이다 | M0를 학습 알고리즘으로 부르지 않음 |
| T1 audited result | `STOP 11/16` | `1c1914b952ead084a21a88a35abca983314dccb013e57c333d1d1075436841fa` | 실패 5개가 모두 빈도 높은 `(1,0)`으로 오분류됐다 | 같은 schedule의 사후 threshold 보정 금지 |

## 정의역과 경계

**[정의]** 경험 집합은 cue/value 쌍 $\mathcal E=\{(c_i,v_i)\}_{i=1}^n$이다. $W$는
실제 `BrainRuntime` recurrent matrix다. $R_h(W,z)$는 $z$를 한 번 입력한 뒤 외부 입력 0으로
$h$ tick 진행한 native activation이다. cue trajectory는 $x^c_{i,h}=R_h(W,c_i)$, value-start
persistence trajectory는 $x^v_{i,h}=R_h(W,v_i)$다. 모든 최종 평가는 temporal과
hippocampal store를 물리적으로 비운 sealed snapshot에서 한다.

이 규칙은 experience-supervised candidate selection이다. 각 post neuron은 경험한 value와
현재 rollout의 차이를 사용하므로 answer-blind local plasticity라고 주장하지 않는다. 다만
M0/Route B처럼 정답 outer product, SVD 또는 least-squares weight를 직접 계산해 설치하지 않는다.

## M4-R: rollout-residual candidate deformation

**[정의]** 안정화 cosine은

$$
\cos_\epsilon(a,b)=\frac{a^\top b}{(\lVert a\rVert_2+\epsilon)(\lVert b\rVert_2+\epsilon)},
\qquad \epsilon=10^{-8}
$$

이다. activation과 weight는 simulator 내부 무차원 좌표이며 cosine, norm ratio, score도
무차원이다.

각 경험에 대해 현재 frozen $W$에서 cue rollout과 value-start persistence rollout을 만든다.
지연 후보 $\lambda\in\{0.50,0.80,0.95\}$의 local presynaptic trace와 terminal post error를

$$
p_{i,c}^{(\lambda)}=\sum_{h=0}^{H-1}\lambda^{H-1-h}x^c_{i,h},
\qquad
p_{i,v}^{(\lambda)}=\sum_{h=0}^{H-1}\lambda^{H-1-h}x^v_{i,h},
$$

$$
\delta_{i,c}=\bar v_i-x^c_{i,H},
\qquad
\delta_{i,v}=\bar v_i-x^v_{i,H},
\qquad
\bar v_i=\frac{v_i}{\lVert v_i\rVert_2+\epsilon}
$$

로 둔다. $\bar v_i$는 presentation에서 한 번만 계산한다. 각 neuron은 이후 자신의
activation 성분과 value 성분만 사용한다. row-post/column-pre eligibility 후보는

$$
E_{i,c}^{(\lambda)}=\delta_{i,c}\left(p_{i,c}^{(\lambda)}\right)^\top,
\qquad
E_{i,self}^{(\lambda)}=\delta_{i,v}\left(p_{i,v}^{(\lambda)}\right)^\top,
$$

$$
D_\lambda=\frac1n\sum_i\left(E_{i,c}^{(\lambda)}+0.65E_{i,self}^{(\lambda)}\right).
$$

이는 terminal error와 과거 local activation의 세 인자 곱이며, target identity·factor label·
held-out row를 읽지 않는다. 한 epoch 안에서는 어떤 후보도 actual $W$를 바꾸지 않는다.

고정 scale $s\in\{0.50,1,2\}$와 identity 후보를 사용한다.

$$
W_{\lambda,s}=\Pi\!\left(W+0.8s
\frac{D_\lambda}{\lVert D_\lambda\rVert_F+\epsilon}\right).
$$

$\Pi$는 기존 structural projection, diagonal zero, 5.0 Frobenius install bound다.
$\lVert D_\lambda\rVert_F\le\epsilon$이면 해당 후보는 identity와 같다.

후보는 staged experience만으로 평가한다.

$$
J_{\lambda,s}=\frac{1}{2n}\sum_i\left[
1-\cos_\epsilon(R_H(W_{\lambda,s},c_i),v_i)
+1-\cos_\epsilon(R_H(W_{\lambda,s},v_i),v_i)
\right]
+0.02\frac{\lVert W_{\lambda,s}-W\rVert_F^2}{5^2}
+0.10I_{unstable}.
$$

$I_{unstable}=1$은 cue/value-start rollout 중 하나라도 nonfinite이거나, reset 직후 첫
입력으로 얻은 각 $x_0$를 기준으로 어느 tick에서든 activation norm이
$2\max(1,\lVert x_0\rVert_2)$를 넘을 때이고, 아니면 0이다. 최소 score 후보를 선택하며
동점은 작은 $s$, 작은 $\lambda$ 순서로 푼다. 선택된 실제 projected delta 하나만 설치한다.

## held-out 누수 금지

후보 생성과 score는 실제 staging된 경험만 읽는다. 관측된 세 factor 쌍의 affine 조합으로
`(1,1)` cue나 target을 만들지 않는다. held-out cue, target, decoder score는 모든 epoch
update와 후보 선택이 끝나고 $W$가 freeze된 뒤 endpoint에서 정확히 한 번만 읽는다.

## 탈락 후보 fold의 사전 정의

M4-R이 formula-discovery split에서 최대 scale을 epoch의 75% 이상 선택하거나 instability를
한 번이라도 만들 때만 Revision 2/2로 fold를 활성화할 수 있다. 실제 후보 변화
$k=(\lambda,s)$와 $A_k=(W_k-W)/0.8$를 사용한 stable softmax

$$
q_k=\frac{\exp[-(J_k-J_{min})/0.1]}{\sum_l\exp[-(J_l-J_{min})/0.1]},
\qquad q_k^{\neg}=\frac{q_k}{1-q_{k^*}}\quad(k\ne k^*)
$$

를 사용해

$$
F_t=0.9F_{t-1}+0.1\sum_{k\ne k^*}q_k^{\neg}(A_k-A_{k^*})^{\odot2}
$$

를 보존한다. 다음 epoch의 각 $D_\lambda$는

$$
P_\lambda=\frac{D_\lambda}{\sqrt{\operatorname{max}_{elementwise}(F_t,10^{-6})}},
\qquad D_\lambda\leftarrow\frac{P_\lambda}{\lVert P_\lambda\rVert_F+\epsilon}
$$

로 precondition한다. fold ablation은 후보·rollout·score를 그대로 두고 $F_t=0$만 강제한다.
trigger가 없으면 fold를 결과 탐색용으로 열지 않는다.

## 폐기한 식

**[정리: no-go]** pooled M1 eligibility의 scale만 바꾸는 M4-0는 모든 nonzero 후보가 같은
방향이므로 T1의 factor-frequency collapse를 구조적으로 분해하지 못한다. 또한 경험 target의
affine 조합을 candidate score에 넣는 식은 held-out `(1,1)` 정답 누수다. 두 식은 구현 후보에서
삭제하고 회귀 증인만 `artifacts/revision-log.md`에 남긴다.

## matched controls

- `identity`: 후보 평가 횟수는 같지만 actual write를 0으로 둔다.
- `no_selection`: 같은 candidate bank를 만들고 항상 $(\lambda,s)=(0.80,1)$을 선택한다.
- `target_shuffled`: cue/RNG/candidate enumeration은 같고 $v_i$를 residual, value-start
  input/target, score에서 모두 같은 고정 cyclic permutation으로 바꾼다.
- `trace_shuffled`: terminal error와 다른 경험의 presynaptic trace를 고정 cyclic permutation한다.
- `sign_flipped`: 선택된 delta의 부호만 뒤집는다.
- `fold_ablation`: Revision 2가 열릴 때만 $F_t=0$으로 둔다.

모든 control은 epoch, experience, rollout, candidate 수와 projection/install 호출 수가 같다.
각 epoch는 $[\operatorname{vec}(D_{0.50}),\operatorname{vec}(D_{0.80}),
\operatorname{vec}(D_{0.95})]$의 numerical rank와 pairwise cosine을 기록한다. tolerance
$10^{-6}$에서 rank 1이면 해당 seed는 `M1_EQUIVALENT_DIRECTION`이며 M4-R 성공으로 세지 않는다.

## 개발·수정·확인 계약

- formula-discovery seeds는 `97401..97408`이다. 식 수정은 여기에서만 허용한다.
- development-validation seeds는 `97409..97416`이며 formula freeze 전에는 열지 않는다.
- confirmation seeds는 `99401..99432`이며 development-validation 통과 전에는 열지 않는다.
- M4-R 기본식이 discovery에서 두 loop 중 하나라도 실패하면 남은 Revision 2/2는 위 fold
  trigger가 실제로 성립한 경우에만 쓴다. trigger가 없으면 M4-R은 STOP이다.
- 각 circuit/seed에서 task endpoint와 각 실행 control 대비 advantage를 먼저 계산한다.
  `min_control_advantage`는 learned score에서 실행된 control score의 최댓값을 뺀 값이다.
  한 seed는 아래 모든 필수 gate와 모든 실행 control을 동시에 통과해야 pass다. 결측 receipt,
  nonfinite 값, zero/nonzero ambiguity는 해당 seed fail이다.
- validation은 8개 중 최소 7개 seed가 Loop 8과 Loop 9를 모두 통과해야 한다. confirmation은
  32개 중 최소 26개 seed가 둘 다 통과해야 한다. pooled 평균으로 seed 실패를 덮지 않는다.
- Loop 8 gate: clean $\ge0.80$, corrupt $\ge0.65$, deleted/unknown abstention $\ge0.95$,
  attractor gain $\ge0.05$, 실행한 모든 matched control에 대한
  `min_control_advantage` $\ge0.20$, 실제 weight 변화와
  finite/dense-sparse/snapshot/store-cutoff 통과.
- Loop 9 gate: held-out `(1,1)` accuracy $\ge0.70$, identity/no-selection/target-shuffle/
  trace-shuffle/sign-flipped 및 fold 활성 시 fold-ablation 모두에 대한
  `min_control_advantage` $\ge0.20$, held-out absence와 store-cutoff 통과.
- factor-codebook SHA-256, collectors/update/score에서 `(1,1)` row가 0개였다는 receipt,
  decoder threshold/calibration이 held-out를 읽지 않았다는 receipt를 seed별로 남긴다.
- replay epochs 12, rollout horizon 6, cue gain 5, abstain threshold 0.20, candidate bank와
  tie-break는 두 loop에서 동일하다. Loop 9 전용 변경은 `APPARATUS_INVALID`다.

## 주장 상한

통과하면 **[산출]** `CONFIRMED_SIMULATOR_EXPERIENCE_SELECTED_DEFORMATION`만 허용한다.
실제 뇌 plasticity, prospective branching, rejected residue, consciousness 또는 AGI 완성을
뜻하지 않는다.

## Revision 2/2 — fold activation

Status: FROZEN BEFORE FOLD RUN

M4-R basic discovery artifact SHA-256은
`254c022cce943b25fee4a74da369ed4f41c1f9d302705d1569d8e06493cd66e7`이고 source SHA-256은
`77fb40e1f93779c77d12f64f5263b8a13dd4a30f9ed15c7107265b316c5e9327`이다. Loop 8 basic
task는 8/8이지만 모든 seed의 `min_control_advantage=0`, Loop 9 basic task는 4/8이고
advantage는 0 또는 -1이었다. 따라서 M4-R basic은 STOP이다.

사전 trigger는 instability가 아니라 selected max scale 포화로 성립했다. Loop 8 seed
97401, 97406, 97407과 Loop 9 seed 97402, 97407에서 $s=2$가 12 epoch 중 9회 선택됐다.
이에 따라 위 M4-R fold 식을 Revision 2/2로 활성화한다. 다른 learning rate, scale,
temperature, epoch, threshold 또는 decoder 변경은 금지한다.

fold arm의 $k^*=\arg\min J$는 actual install과 $F$ 중심에 모두 사용한다. `no_selection`은
9개 후보를 모두 score하지만 $k_{ref}=(0.80,1)$을 actual install과 $F$ 중심에 모두 쓰며,
$q^\neg$도 $k_{ref}$를 제외해 다시 정규화한다. `fold_ablation`은 normal과 같은 후보,
score, $k^*$와 install을 사용하고 shadow $F$를 계산·기록하지만 preconditioning에서만
$F=0$을 강제한다. `identity`도 모든 score와 no-op install receipt를 남긴다.

같은 formula-discovery seeds `97401..97408`에서 fold를 한 번 실행한다. 두 loop를 동시에
통과하는 seed가 7/8 미만이거나 어떤 seed든 fold-ablation/no-selection과의
`min_control_advantage`가 0.20 미만이면 terminal STOP이며 validation을 열지 않는다.
통과한 경우에만 formula를 freeze하고 untouched validation `97409..97416`을 연다. 추가
수정은 없다.
