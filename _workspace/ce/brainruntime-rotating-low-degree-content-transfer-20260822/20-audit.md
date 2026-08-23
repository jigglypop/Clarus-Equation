# BA-TR28 pre-execution audit

Gate: PASS.

- The learner API contains only raw cues and observed content vectors.
- The compiler adds only the current cue, arrived coordinates, current weight,
  and response coordinates.
- No finite chart/class table, factor name, cell coordinate, seed, target,
  decoder, reward, endpoint, store, held-out content, or remap enters either
  API.
- Every rotating fold is overdetermined for six degree-two features.
- The affine and exact-lookup controls use the same observed rows.
- Query-only $\delta$ is recorded as a formal nonidentifiability witness, not
  a learned STOP claim.
- Execution may proceed through calibration only; development remains sealed.
