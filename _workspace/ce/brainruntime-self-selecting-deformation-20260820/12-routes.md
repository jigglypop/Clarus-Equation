# M4-R 경로와 반례 지도

Status: COMPLETE

| 순위 | 경로 | 입력 | 진행 조건 | decisive STOP |
|---:|---|---|---|---|
| 1 | M4-R basic | discovery `97401..97408` | 두 loop와 candidate-rank audit | 어느 loop든 실패하고 fold trigger 없음 |
| 2 | M4-R fold | 같은 discovery split | max-scale가 epoch 75% 이상 또는 instability 1회 이상 | trigger 부재, 또는 $F=0$ ablation과 동률 |
| 3 | frozen validation | `97409..97416` | 7/8 이상이 두 loop 동시 GO | formula/config 변경 또는 7/8 미만 |
| 4 | confirmation | sealed `99401..99432` | 26/32 이상이 두 loop 동시 GO | validation 전 seed 접근 또는 26/32 미만 |

## Discovery revision status

- M4-R basic: `STOP`. Loop 8 basic 8/8이나 selection advantage 0/8; Loop 9 basic 4/8.
- M4-R fold: `OPEN_AUTHORIZED_REVISION_2`. max-scale 75% trigger가 사전 조건대로 성립했다.
- fold discovery가 joint 7/8과 모든 control advantage를 통과하지 못하면 terminal `STOP`.
- validation과 confirmation은 계속 봉인한다.

## 구현 순서

1. M4-R을 frozen M1/T1과 격리된 module에 구현한다.
2. Loop 8과 factorized Loop 9가 같은 config와 candidate bank를 호출하게 한다.
3. discovery에서 basic 식과 모든 unconditional control을 실행한다.
4. fold trigger를 machine receipt로 판정한다. trigger가 없으면 Revision 2를 열지 않는다.
5. 식을 freeze한 뒤에만 validation, 통과 뒤에만 confirmation을 연다.

## matched controls

- `identity`: 모든 후보를 평가하지만 write를 설치하지 않는다.
- `no_selection`: bank와 score를 모두 계산한 뒤 argmin을 버리고 $(0.80,1)$을 설치한다.
- `target_shuffled`: $v_i$ permutation을 residual, self input/target, score에 일관되게 쓴다.
- `trace_shuffled`: terminal error와 다른 경험의 trace를 cyclic permutation한다.
- `sign_flipped`: 선택된 actual delta의 부호만 반전한다.
- `fold_ablation`: fold가 활성화될 때만 candidate apparatus를 유지하고 $F_t=0$으로 둔다.

각 seed의 `min_control_advantage`는 learned endpoint에서 그 seed에 실행된 모든 matched-control
endpoint의 최댓값을 뺀 값이다. sign-flipped도 항상 gating이며 fold가 열리면 fold-ablation도
gating이다. 결측·비유한 값이나 ambiguous write receipt는 seed fail이고 pooled 평균으로
대체하지 않는다.

Revision 2에서 normal은 argmin $k^*$를 install과 $F$ 중심에 함께 쓴다. no-selection은
$k_{ref}=(0.80,1)$을 두 위치에 함께 쓰고, fold-ablation은 shadow $F$만 기록한 채
preconditioning $F$를 0으로 둔다. 이 factorial separation을 어기면 apparatus invalid다.

## 무효화 조건

held-out `(1,1)` cue, target 또는 decoder를 $W$ freeze 전에 읽거나, Loop 9에서만 learning
rate·trace decay·scale bank·epoch·threshold를 바꾸면 `APPARATUS_INVALID`다. delayed direction
rank가 tolerance $10^{-6}$에서 1인 seed는 `M1_EQUIVALENT_DIRECTION`이다. target-shuffle 또는
no-selection이 learned arm과 동률이면 candidate selection mechanism은 STOP이다.

## 주장 상한

통과해도 합성 runtime의 experience-supervised candidate deformation만 성립한다. 실제 뇌의
후보 생성, 탈락 residue, offline consolidation 또는 AGI 일반 학습 규칙은 별도 경로다.
