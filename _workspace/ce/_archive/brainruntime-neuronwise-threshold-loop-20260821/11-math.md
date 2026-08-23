# Mathematics

Status: COMPLETE

## 1. Type separation

The recurrent matrix and threshold vectors have different roles:

$$
W\in\mathbb R^{q\times q},\qquad
\boldsymbol\vartheta,\boldsymbol\theta^-,\boldsymbol\theta^+\in\mathbb R^q.
$$

$W_{ij}$ is a signed directed circuit strength from sender $j$ to receiver $i$.
$\vartheta_i$ gates eligibility after salience is computed, while
$\theta_i^\pm$ define the bit hysteresis of neuron $i$. Multiplying or folding these
quantities together would change the model and is not identifiable from the current tests.

All runtime state variables and thresholds are normalized dimensionless tick quantities, so
the comparisons in (T3)--(T4) are dimensionally valid. This does not turn the normalized
numbers into physiological voltages.

## 2. Scalar broadcast theorem for this implementation

Let an admissible scalar configuration satisfy $\theta^-<\theta^+$, and define repeated vectors

$$
\boldsymbol\vartheta=\vartheta\mathbf1,\quad
\boldsymbol\theta^-=\theta^-\mathbf1,\quad
\boldsymbol\theta^+=\theta^+\mathbf1.
$$

For every fixed salience $s$, activation $a^+$, bit state $b$, budget and deterministic
no-tie TopK order, the componentwise predicates

$$
s_i\ge\vartheta_i\iff s_i\ge\vartheta,
$$

$$
a_i^+\ge\theta_i^+\iff a_i^+\ge\theta^+,
\qquad
a_i^+\le\theta_i^-\iff a_i^+\le\theta^-
$$

are identical. Consequently scalar and repeated-vector selection/bit outputs are identical.
Because threshold tensors do not enter the continuous cell equation, all continuous states are
also identical on the same backend and input. T-B tests the full runtime composition rather
than treating this algebra alone as implementation proof.

This statement deliberately excludes legacy scalar configurations with
$\theta^-\ge\theta^+$. The old Torch assignment evaluates the upper write and then the lower
write, so an overlap ends at bit 0, while the new vector path rejects the ambiguity. Preserving
construction of that legacy scalar config is backward compatibility; it is not part of the
repeated-vector equivalence theorem.

## 3. Hysteresis well-posedness

For a new vector configuration require

$$
\theta_i^-<\theta_i^+\quad\text{for every }i.
$$

Then the lower and upper predicates cannot both hold and the middle retention interval is
nonempty. If only one bit vector is supplied, the other effective vector is the scalar
broadcast and the same inequality is checked. Legacy scalar-only configuration is left
behaviorally untouched; this is a compatibility decision, not approval of invalid scalar
hysteresis.
The effective scalar counterpart must also be finite whenever a vector path is selected.

## 4. Frozen heterogeneous witness

With $W=0$, old $a_i=.2$, zero refractory/adaptation/input/goal/replay/noise in WAKE,

$$
a_i^+=(1-.18)(.2)+.82\tanh(0)=.164.
$$

For initial $b=(0,1,1)$, upper vector $(.15,.22,.30)$ and lower vector
$(.10,.17,.20)$:

- neuron 0 crosses its upper threshold, so $b_0^+=1$;
- neurons 1 and 2 cross their lower thresholds, so $b_1^+=b_2^+=0$.

Thus the preregistered output is $(1,0,0)$. Separately, with
$s=(.30,.40,.50)$ and $\boldsymbol\vartheta=(.35,.35,.55)$, only neuron 1 is
eligible even though neuron 2 has the largest raw salience. This distinguishes threshold
heterogeneity from ordinary TopK ranking.

## 5. Backend semantics

The current Rust map is parameterized by scalar bit thresholds. Passing a vector config while
continuing to call it with scalar fields would compute a different discrete map. Silent scalar
projection is therefore a P0 semantic error, not an approximation.

Torch fallback for `auto` and explicit-Rust rejection are exact fail-closed behavior. An active
threshold vector alone is different: active selection occurs after the Rust cell call in Python,
and the Rust-returned private count is overwritten. It is therefore admissible on the existing
no-delay Rust cell, provided the final Python mask/count and continuous parity are tested.

## 6. Formal status before implementation

- equations (T1)--(T4): **[정의]**;
- scalar broadcast identity: **[조건부 정리]**;
- concrete three-neuron outcome: **[산출]**;
- runtime wiring and snapshot/backend behavior: **[미검증 구현 명제]**;
- physiological thresholds, learning and anatomy: **[미완성]**.

Pre-implementation status is `READY_FOR_AUDIT`.
