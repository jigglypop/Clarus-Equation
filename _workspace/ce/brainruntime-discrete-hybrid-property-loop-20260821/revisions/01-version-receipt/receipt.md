# Version-receipt correction

Status: `P2_APPARATUS`

The stable mathematical witness passed, but the result recorded
`reality_stone_version="UNAVAILABLE"` because this source-tree execution has no
installed distribution metadata. The imported package itself exposes
`reality_stone.__version__="0.2.10"`, matching `reality_stone/pyproject.toml`.

- preserved passing script SHA-256:
  `2878bca53478dfbbc966f6b6189d7cfddd3611c1aec561061957560c4f847462`
- preserved passing result SHA-256:
  `52ed6b73fa4fdc94132027ed7a5ce1dbb39de85aaa6460469a43028d46bc1d20`
- allowed change: add the imported package `__version__` as a metadata fallback
  and record which version source was used
- forbidden changes retained: every equation, fixture, direction, seed/step,
  tolerance, gate, source hash, and claim ceiling

This correction changes provenance fields only; all H-A--H-G quantities must
reproduce exactly.
