# Estimand and budget ledger

Status: COMPLETE

For panel $p$, seed-paired noninferiority is

$$
D_{s,p}=A^{V10}_{s,p}-\max(A^{RNN20}_{s,p},A^{GRU20}_{s,p}).
$$

The compute-matched contrast is $C_{s,p}=A^{V10}_{s,p}-A^{RNN3}_{s,p}$. Bootstrap units are seed
blocks. Approximate recurrent multiply counts per tick are declared before execution:

- V10: 76;
- Elman-$h$: $20h+h^2+h$ including scalar readout;
- GRU-$h$: $3(20h+h^2+h)$ including three gates and scalar readout.

Thus Elman-3 uses 72 and is the compute-matched comparator. This ledger is not a hardware latency
identity. V10's contraction theorem is unchanged; learned RNN/GRU stability is not assumed.
