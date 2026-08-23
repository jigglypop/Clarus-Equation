# M0/M1 development and confirmation validation

Status: IN_PROGRESS

## Development evidence

The independent development unit is one seeded circuit, seeds `97201..97216`.

### M0 capacity ceiling

| Rank | Complete GO | Mean clean | Mean corrupt | Minimum gain | Minimum control advantage |
|---:|---:|---:|---:|---:|---:|
| 1 | 0/16 | 0.270833 | 0.187500 | 0.088228 | 0.000000 |
| 2 | 8/16 | 0.854167 | 0.708333 | 0.191040 | 0.666667 |
| 4 | 16/16 | 1.000000 | 1.000000 | 0.281244 | 1.000000 |
| full | 16/16 | 1.000000 | 1.000000 | 0.281244 | 1.000000 |

The frozen minimum sufficient development rank is therefore 4. All deleted-cue abstention and
dense/sparse parity audits passed 16/16 for every rank. This is supervised capacity, not acquisition.

### M1 fixed-clock delayed three-factor route

Final development result: 16/16 complete GO.

- mean clean, corrupt, deleted, and unknown scores: `1.0` each;
- minimum attractor cosine gain: `0.690751`;
- minimum advantage over the strongest of six controls: `1.0`;
- minimum applied association contrast: `2.226257`;
- minimum target-shuffled contrast margin: `3.285421`;
- inter-phase reset, equal schedule, zero-store rollout, snapshot parity, dense/sparse parity, and
  finite-state audits: 16/16.

The v1 STOP, reset revision result, diagnostic score distribution, and final frozen result are kept
as separate artifacts. Development success does not authorize a biological consolidation claim.

## Frozen files and artifacts

- M0 final development: `artifacts/m0-development-results-v2-frozen.json`
  (`result_sha256=418efd5a066061f8d5254b8ac3e534ed542a3a4a97e7ad82887a3722debbb7b7`).
- M1 final development: `artifacts/m1-development-results-v3-frozen.json`
  (`result_sha256=5df9c1e5aa41e73e042adc16178b1b7f970d4e08e8b6b6cbe00f5f23cc55b84a`).
- Source hash shared by both: `be708bac30bb4e7e681990f838159e70efb9ed36061cef602771e86c8248c27a`.
- Confirmation lock: `artifacts/m0-m1-confirmation-freeze.json`.

## Confirmation

The one-shot confirmation run used seeds `99201..99232` against the frozen source and manifest.
No fitting, threshold change, route addition, or endpoint substitution occurred. The output is
`artifacts/m0-m1-confirmation-results.json` with
`result_sha256=c9bc90b9172f3f0915615665eaca64212eaf5dd12ef86ac19324846f81bfa155`.

### M0 confirmation

| Rank | Complete GO | Mean clean | Mean corrupt | Minimum gain | Minimum control advantage |
|---:|---:|---:|---:|---:|---:|
| 1 | 0/32 | 0.270833 | 0.166667 | 0.089645 | 0.000000 |
| 2 | 15/32 | 0.875000 | 0.687500 | 0.187240 | 0.666667 |
| 4 | 32/32 | 1.000000 | 1.000000 | 0.281323 | 1.000000 |
| full | 32/32 | 1.000000 | 1.000000 | 0.281323 | 1.000000 |

Rank 4 is again the first stable passing rank. This confirms a supervised structural capacity
threshold only.

### M1 confirmation

M1 passed 32/32 complete binding gates.

- clean, corrupt, deleted, and unknown means: `1.0` each;
- attractor cosine gain: mean `0.733442`, minimum `0.653667`;
- control advantage: minimum `1.0`;
- applied association contrast: mean `2.356287`, minimum `2.049822`;
- target-shuffled contrast margin: mean `3.522610`, minimum `3.105409`;
- minimum known-target cosine `0.467714`, maximum deleted-cue cosine `0.179505`, preserving the
  frozen `0.20` abstention boundary;
- reset, equal schedule, zero-store, snapshot, dense/sparse, and finite-state audits: 32/32.

Every adverse control had clean accuracy `0.0` on every circuit: zero clock, sign-flipped clock,
time reversal, eligibility reset, no replay, and target shuffle. This confirms synthetic
fixed-clock delayed-three-factor **binding acquisition**. Factorized held-out transfer has not yet
been run for M1 and is not included in this verdict. No biological consolidation claim is allowed.
