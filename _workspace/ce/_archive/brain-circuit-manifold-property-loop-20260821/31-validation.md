# Validation

Status: COMPLETE

Verdict: PROPERTY_PASS

## Final focused command

```powershell
.codex\hooks\python.cmd python _workspace\ce\brain-circuit-manifold-property-loop-20260821\artifacts\a6_property_witness.py
```

Exit code: `0`.

## Frozen artifacts

| artifact | SHA-256 |
|---|---|
| final source | `c67f1f790c291f622db6362f31eeb58b9b7e4bc147d7c8d99d08f48fe511074d` |
| final result | `f6130d58cfe5e8d20b6ea9987c467e473ad9c4a183c4c54f19678e3d1d8a89bd` |
| contract | `6322e6045b87f4fc7cd5d2bee29f1ceb800808de9dc1d2b5fcc0604926434fb4` |
| initial source, preserved | `79d0ef045a0bbad460ae77aa07735bb7b75d431a7fe58457259205327274eda5` |
| initial result, preserved | `3120bc215328f65633dd2fbdc14564bc0d5edb122fd535761942b5fd1665f4c5` |

strict JSON parse with nonfinite-token rejection: PASS. Source self-hash and contract
self-hash in the receipt match the files.

## Property readout

| gate | observed worst case | frozen gate | status |
|---|---:|---:|---|
| passive tangent error | `6.84498e-12` | `<=5e-7` | PASS, 8/8 |
| total `dot J` error | `3.74461e-12` | `<=2e-6` | PASS, 8/8 |
| total `dot g` error | `1.81835e-12` | `<=3e-6` | PASS, 8/8 |
| metric covariance error | `2.30347e-16` | `<=1e-10` | PASS, 8/8 |
| generalized-eigen error | `7.75001e-13` | `<=1e-10` | PASS, 8/8 |
| log-volume-ratio error | `1.14518e-12` | `<=1e-10` | PASS, 8/8 |
| bad untransformed-`G` error, minimum | `1.53962e-3` | `>=1e-5` | KILLED, 8/8 |
| Gramian symmetry error | `3.54054e-17` | `<=1e-12` | PASS, 8/8 |
| smallest Gramian eigenvalue | `1.42614e-2` | PSD floor | PASS, 8/8 |
| control target residual | `8.25567e-16` | `<=1e-9` | PASS, 8/8 |
| energy identity error | `1.72842e-15` | `<=1e-8` | PASS, 8/8 |
| chart energy error | `7.77156e-16` | `<=1e-10` | PASS, 8/8 |
| `dot E` error | `2.24316e-11` | `<=5e-6` | PASS, 8/8 |

모든 passive Jacobian과 pre/post metric은 operational full rank였다. 모든 augmented
Gramian도 full rank였고 최대 condition number는 `20.7726`, inverse-response 허용
상한 `1e8` 안이었다. 최대 path weight는 `0.462277<0.47`, 최대
`||C_Gamma||_infinity`는 `0.293805<0.48`이다.

## Killing controls

- exact `J=diag(1,0)`: `EXACT_RANK_DEFICIENT`, ridge 없음.
- `B=0`, nonzero target: `UNREACHABLE`, energy `Infinity`.
- rank-one map, orthogonal target: `UNREACHABLE`, energy `Infinity`.
- singular values `(1,1e-12)`: operational rank 1, inverse derivative 금지.
- state-dependent efficacy full tangent error: `7.34953e-12`; omitted
  `partial p/partial xi` error: `6.06370e-2`.
- full two-step circuit response error: `4.35997e-12`; omitted `dot p` error:
  `2.21099e-3`.

## Interpretation

수식 revision은 필요하지 않았다. 최초 사후 감사의 실패는 formula가 아니라 PASS에
연결되지 않은 implementation gates였고 Revision 1에서 닫혔다. 이 validation은
`MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED`만 지지한다.
