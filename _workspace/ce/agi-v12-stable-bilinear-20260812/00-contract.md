# V12 stable bilinear local/cloud contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v11-strong-ood-20260812`

V12 freezes a 20-state salience-gated local/cloud memory with bilinear policy, 53 trainable
parameters, retention cap `0.995`, salience threshold `0.40`, sharpness `30`, and 200 training
epochs. V11 Elman-3, Elman-20, and GRU-20 remain unchanged controls.

Development uses 16 fresh seeds. If and only if it passes, confirmation uses 32 fresh seeds.
Across ID/noise/horizon/combined, V12 must have accuracy at least `0.95`, beat Elman-3 with positive
LCB, and be noninferior to GRU-20 with LCB at least `-0.01`. Every trained state transition must
retain `q<0.995`; integrity must be zero. Same-run tuning is forbidden after development opens.
