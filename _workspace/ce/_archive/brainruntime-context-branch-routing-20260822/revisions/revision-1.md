# Revision 1 — remove the dead extra relay

Status: COMPLETE

Frozen seed `97501` under Revision 0 passed every preflight receipt:
five block maps had four edges and rank four, both context products had minimum
singular value `0.7071067691`, masks had the declared budget, stores and delay
ring were empty, and the selected-hidden ratio was approximately one.

Nevertheless every arm had `Y` norm `0` and accuracy `0`.  Tick tracing showed
the correct hidden block peak at about `0.249245`, while the extra shared relay
peaked at only `0.059073` and never became an eligible emitting source.  Thus
the third hop could not reach `Y` under the already-frozen heterogeneous
thresholds.

Revision 1 removes only $R$:

$$
S_c(k)\xrightarrow{L}H_c(k)\xrightarrow{L}Y(k).
$$

Both $Y\leftarrow H_0$ and $Y\leftarrow H_1$ maps remain open and identical
for both contexts; only $H_c\leftarrow S_c$ changes.  Dimension becomes `20`,
mask budget becomes `3m=12`, union support becomes `4m=16`, and the derived
readout call changes from `3(L+1)=9` to `2(L+1)=6`.  Thresholds, decoder
cosine/margin, delay, local eligibility parameters, payload width, and seed
lists do not change.
