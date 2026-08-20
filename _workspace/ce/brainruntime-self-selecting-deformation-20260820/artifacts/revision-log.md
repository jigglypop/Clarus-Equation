# Revision log

## Revision 1/2 — contract/math

수학 검산에서 두 P0를 발견해 구현 전에 수정했다.

1. 관측 `(0,0),(0,1),(1,0)`의 affine target 조합은 정확히 held-out `(1,1)` target이므로
   candidate score에서 삭제했다. 후보 선택은 staged experience만 읽는다.
2. scale만 다른 후보의 normalized direction 분산은 0이므로, fold를 projected actual
   candidate delta 차이의 무차원 분산으로 바꿨다. zero direction은 no-update로 정의했다.
3. pooled eligibility에 scale만 달리한 M4-0는 후보 방향이 하나뿐이라 T1 반례를 고칠 수
   없었다. 이를 구현 전에 퇴역시키고, terminal rollout error와 delayed local trace가 만드는
   비공선 후보 bank M4-R로 식을 교체했다.
4. target을 읽는 selector는 answer-blind가 아니므로 주장 지위를
   `experience-supervised candidate selection`으로 낮췄다.
5. 식 선택용 `97401..97408`과 untouched development-validation `97409..97416`을 분리했다.

이 수정은 아직 결과를 열기 전에 이루어졌다.

## Revision 2/2 — rejected-candidate fold

Corrected basic discovery에서 Loop 8 basic task는 8/8이었으나 no-selection과 동률이었고,
Loop 9 basic task는 4/8이었다. 사전 max-scale trigger가 Loop 8 seed 97401/97406/97407과
Loop 9 seed 97402/97407에서 성립해 actual candidate-delta variance fold를 활성화했다.
이 revision은 entrywise saturation hypothesis만 시험한다. 실패 뒤 추가 식·수치 수정은 없다.
