# Chapter 9 split-docs consistency audit

- passed: `True`
- required ok: `True`
- stale ok: `True`

| file | kind | phrase | present |
|---|---|---|---|
| `01_개요와공통식.md` | `required` | `toy positive` | `True` |
| `01_개요와공통식.md` | `required` | `empirical gate pending` | `True` |
| `01_개요와공통식.md` | `required` | `Evolution ladder closure package` | `True` |
| `01_개요와공통식.md` | `required` | `external requirements readiness` | `True` |
| `01_개요와공통식.md` | `stale` | `아직 gate 없음` | `False` |
| `01_개요와공통식.md` | `stale` | `not tested` | `False` |
| `02_c_elegans.md` | `required` | `trial-behavior boundary audit` | `True` |
| `02_c_elegans.md` | `required` | `data-boundary` | `True` |
| `03_drosophila.md` | `required` | `trial-dynamics boundary audit` | `True` |
| `03_drosophila.md` | `required` | `data-boundary` | `True` |
| `04_zebrafish.md` | `required` | `continuous boundary final audit` | `True` |
| `04_zebrafish.md` | `required` | `timestamp-certified alignment` | `True` |
