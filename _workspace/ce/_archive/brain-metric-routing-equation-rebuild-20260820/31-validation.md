# Validation

Status: COMPLETE

## Machine checks

- `.codex\hooks\run.cmd check ... lanes`: `OK lanes`.
- `tests/test_dimensionless.py`: `15 passed in 0.36s`.
- `reality_stone/python/reality_stone/clarus/dimensionless.py`: exit code `0`.
- `git diff --check -- _workspace/ce/brain-metric-routing-equation-rebuild-20260820`: clean before the final audit write.

Python checks used the policy-accepted system CPython `3.11.9` with bytecode and pytest cache disabled because the uv-managed interpreter is blocked by Windows application control on this host. No package was installed and no scientific stage was executed.

## Human-independent checks

- geometry/Fisher verifier: final `PASS`, no P0/P1.
- routing/identifiability verifier: final `PASS`, no P0/P1.
- formal status auditor: final `PASS`, no P0/P1.

## Canonical-document checks

- `docs/6_뇌/11_리만계량_라우팅_논문.md`: stable content audit PASS after two typographic fixes.
- `docs/6_뇌/00_읽기지도.md`: $\mathcal B=(G,R)$ priority and synthetic-global-state boundary PASS.
- mojibake marker scan: none.
- banned `\[` / `\(` delimiter scan in the revised paper: none.
- four local reproduction links under `_workspace/ce/_archive/`: all exist.
- focused `git diff --check`: clean; only the existing Windows LF-to-CRLF warning was emitted.

## Scope

The existing dimensionless test registry does not symbolically parse the new neural equations. The explicit unit derivation in `artifacts/dimensionless-audit.md` is therefore the equation-specific audit; the 15-test run is a regression check for the existing CE dimensionless machinery.
