# Focused validation

Status: COMPLETE

## Command

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\measurement-record-one-way-compatibility-20260825\artifacts\verify_one_way_record.py'
```

## Result

All registered checks met tolerance $10^{-12}$. The projective instrument maps $|0\rangle$ and $|1\rangle$ to record distributions $(1,0)$ and $(0,1)$, while scalar effects $0.3I$, $0.7I$ give the constant distribution $(0.3,0.7)$ for all tested states. The Frobenius distance of $P_0$ from its closest scalar form $(\operatorname{tr}P_0/2)I$ is

$$
\|P_0-I/2\|_F=0.707106781186548.
$$

The finite-duration controlled pointer unitary is unitary to numerical tolerance; its induced Kraus operators are $M_0=P_0$, $M_1=-iP_1$ and satisfy completeness exactly within tolerance.

This is a channel-algebra certificate, not evidence for a physical external 0D sector or cosmology.
