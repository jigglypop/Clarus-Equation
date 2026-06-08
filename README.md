# Reality Stone

This repository is now centered on `reality_stone`.

The former top-level `clarus` package has been moved into the Reality Stone
Python package as:

```python
import reality_stone.clarus
```

## Layout

```text
reality_stone/
  python/reality_stone/          Reality Stone Python API
  python/reality_stone/clarus/   Clarus runtime, CE ops, agents, and bridges
  src/                           Reality Stone Rust/PyO3 backend
  tests/                         Reality Stone regression tests
  examples/unified_clarus_demo.py

tests/
  test_unified_example.py        smoke test for the single example
```

## Run The Example

```powershell
.\.venv\Scripts\python.exe -B reality_stone\examples\unified_clarus_demo.py
```

The example exercises the unified path:

- imports `reality_stone`
- uses `reality_stone.clarus.runtime.BrainRuntime`
- runs `MetricAttention`
- runs the unified Riemannian fallback bridge

## Quick Checks

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
```

## Native Build Note

The root `pyproject.toml` is the unified checkout entrypoint and targets the
optional Clarus core extension at `reality_stone.clarus._rust`. The vendored
Reality Stone native extension remains available from `reality_stone/pyproject.toml`
as `reality_stone._rust`; Python fallbacks keep the unified package importable
when either native extension is absent.
