# V11 learned-comparator and OOD contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v10-local-cloud-confirmation-20260812`

## Question

Does confirmed V10 remain competitive when compared with learned recurrent models receiving the
same raw 20-dimensional sequence, and when evaluation noise and horizon shift?

## Frozen development design

- 16 fresh seed blocks; 256 ID train and 256 examples per evaluation panel.
- Panels: ID `(T=4,sigma=0.04)`, noise `(4,0.08)`, horizon `(8,0.04)`, combined `(8,0.08)`.
- V10: frozen transition, train-only ridge.
- Learned controls: 20-hidden Elman RNN, 20-hidden GRU, and 3-hidden compute-matched Elman RNN.
- Full-batch Adam, 100 epochs, learning rate `0.01`, weight decay `0.0001`, gradient clip `1.0`;
  no validation selection or early stopping.
- Accuracy and Brier score are both primary reports.

## Gates

1. On every panel, V10 accuracy minus the stronger of Elman-20/GRU-20 has seed-bootstrap LCB
   at least `-0.02` (noninferiority).
2. On every panel, V10 exceeds compute-matched Elman-3 with LCB above `0`.
3. V10 accuracy is at least `0.60` on ID and at least `0.55` on each OOD panel.
4. No seed overlap, target leak, nonfinite result, training nondeterminism, or hash mismatch.

Any failed primary gate is STOP. Parameter count, state size, inference MAC estimate, train time,
and accuracy are reported separately. No same-run tuning is allowed.
