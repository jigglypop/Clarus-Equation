# 수학·식별 레인: publication-native event recovery

Status: COMPLETE

## 입장 정의

**[정의]** `PASS_APPARATUS_INPUT`은 효과나 인과 결론이 아니다. outcome을 읽지 않은
상태에서 immutable official bytes에 근거한 source-to-canonical-identity join과 원래
배정된 모든 event의 assignment receipt를 동시에 복원했다는 입력 판정이다.

**[정의]** unique key `(animal,session,event)`에 대한 최소 receipt는 다음이다.

$$
(t_{stim},A_{id},A_{confidence/provenance},assignment,manual,
failed,u_{stim},order).
$$

$A_{id}$는 response, kernel fit, responder 판정, autoresponse, downstream fluorescence
또는 outcome-tuned spatial fit으로 만들 수 없다. `failed`는 사후 response로 고른
control이 아니라 원래 assigned-event population의 결측·실패 표지여야 한다.

## 선행 반례

**[정리]** pre/post difference는 no-light effect가 아니다.

증명. light effect가 0이어도 $Y(t)=\beta t+\epsilon(t)$이면 다음이 성립한다.

$$
\mathbb E[Y(+\Delta)-Y(-\Delta)]=2\beta\Delta.
$$

$\beta\ne0$이면 차이는 0이 아니다. drift, calcium convolution, adaptation과 carryover도
같은 형태의 대체 설명을 만든다. □

**[정리]** autoresponse-negative event를 control로 선택하면 post-treatment selection이
될 수 있다. latent excitability $L$이 source autoresponse와 target response에 함께
영향을 주는 모형에서는 autoresponse로 conditioning한 뒤 source-target contrast가
원래 assignment effect와 달라진다. 따라서 failure receipt는 outcome과 독립적이어야
한다. □

**[정리]** frozen code의 field semantics는 actual source receipt를 대신하지 못한다.
코드는 가능한 field와 해석을 정하지만 publication object의 존재, rowwise value,
missing/failure completeness와 실제 assignment policy를 함의하지 않는다. 따라서 code
probe만으로 apparatus input을 통과할 수 없다. □

## 단위와 domain gate

**[정의]** $t_{stim}$, duration과 order의 session clock을 분리한다. power [W], duration
[s], pulse train/duty cycle, wavelength [nm]와 spot geometry를 각각 보존하며 `power`
또는 `stim_volume_i` 하나를 dose로 재명명하지 않는다.

**[미완성]** geometric R2는 pixel/µm/z frame, registration transform, calibration,
matching radius와 ambiguity/tie-break를 모두 고정해야 한다. categorical neuron identity는
coordinate proximity 자체로 증명되지 않는다. confidence/provenance는 무차원 metadata다.

## 판정 gate

- `PASS_SOURCE_INDEX_JOIN`: actual bytes에서 event source index가 local label로 안전하게
  결합되고 cardinality·missing code가 일치한다.
- `PASS_SOURCE_JOIN`: 위 label이 canonical $A_{id}$이고 confidence/provenance까지
  outcome-blind하게 연결된다.
- `PASS_ASSIGNMENT_RECEIPT`: 모든 original assigned event에 automatic/manual,
  exception, failed, order, timing과 dose metadata가 있다.
- `PASS_APPARATUS_INPUT`: `PASS_SOURCE_JOIN`과 `PASS_ASSIGNMENT_RECEIPT`를 모두 통과한다.
- `PASS_OBSERVATIONAL_ONLY`: source label은 있으나 assignment policy, positivity 또는
  complete missingness가 없다.

**[미완성]** apparatus pass 뒤에도 causal active-source effect에는 actual randomization
strata, within-stratum positivity, fixed controls, carryover rule과 treatment-independent
missingness가 추가로 필요하다. direct edge, endogenous $do(A)$, no-light effect 또는
state-conditioned causal routing은 receipt만으로 입증되지 않는다.

## 안전한 parser 경계

metadata/event-only JSON·CSV·TSV, primitive numeric/text와 `allow_pickle=False`인 NumPy
입력만 직접 해석한다. object dtype는 거부한다. archive는 member metadata를 먼저 읽고
encrypted member, symlink/path traversal, 과도한 member count·decompressed size와 미승인
member를 거부한다. pickle은 `pickletools`로 bounded opcode만 정적 검사하며 load하지
않는다.

