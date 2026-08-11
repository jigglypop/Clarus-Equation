# V9 memory development pre-run audit

Status: COMPLETE

Gate: PASS

## Locked object

- preregistration: `artifacts/preregistration.json`
- SHA-256: `B34C22FB2ABD40D2DA5F7DE2E58C22EE1325758B033B5BD163C9C6881C655E03`
- schema: `clarus.v9-memory-prereg.v1`
- status: `LOCKED_BEFORE_RESULTS`
- development result existed at audit: `False`

## Source locks

| Source | SHA-256 |
|---|---|
| `nested_scc_memory_benchmark.py` | `CD1F1846A7EAB5D45BEFCE0209E42A87E6373A6CFF2A42D9AEA936463E46BBCE` |
| `nested_scc_tower.py` | `854673216E5FEACA5FF0E3619DA63B789B15BDFE35994DE539F61C4EAE83A717` |
| `adaptive_scc_tower_controller.py` | `89C49B8D47ECC67A78AF8D4D6AC2160D383312DC171D24EFCC975F415CEB5D33` |

The verifier reconstructed the canonical preregistration from live sources and obtained exact
content equality.

## Design audit

- Development seeds are exactly `0..255`; confirmation seeds `10000..10255` remain forbidden.
- There are six distinct named arms. Candidate and lesions use separate controller instances.
- `UpperReset` resets every upper state and cuts both message directions for its consumed
  decision update. `CrossScaleCut` cuts every cross-level message on every episode update.
- Candidate prediction receives observations only; target labels are compared outside the arm.
- The strongest comparator is chosen by the frozen deterministic rule over stateless, level0,
  and monolithic arms.
- State scalar count and estimated MAC are reported separately and are not called parameter
  capacity.
- GO is one conjunction of five frozen gates. There is no secondary route or threshold tuning.
- The result writer refuses overwrite. A second development invocation cannot replace the
  preserved artifact.

## Pre-seed validation

- V9 related focused suite: `202 passed`.
- Preregistration mutation and source-hash mismatch fail closed.
- Missing authorization, existing result, and early confirmation fail closed.
- No scored development or confirmation seed was generated during these tests.

## Authorization

Exactly one development invocation writing
`artifacts/development-result.json` is authorized. Confirmation remains blocked unless the
preserved development result is `GO` and a separate post-development audit explicitly passes.
