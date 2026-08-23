# First execution receipt

Status: `P2_APPARATUS`

The first invocation stopped before constructing or evaluating any H-A--H-G
quantity. `source_receipt()` pointed to the nonexistent path
`reality_stone/core/src/engine/kernel.rs`; the frozen source is actually
`reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs`, as already
recorded correctly in `10-sources.md`.

- preserved script SHA-256:
  `4f598977359d0eb6c9488c8d825e2f4aeef3151e98a1d5ef5edb67a55852dd1b`
- exit code: `1`
- exception: `FileNotFoundError`
- failed operation: provenance hashing, before formula/test evaluation
- allowed change: replace only the two incorrect kernel path literals
- forbidden changes retained: equations, fixture, directions, steps,
  tolerances, pass gates, and claim ceiling

Traceback terminal frame:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'C:\\Users\\dongh\\OneDrive\\Desktop\\Clarus-Equation\\reality_stone\\core\\src\\engine\\kernel.rs'
```
