# Reality Stone

Reality Stone is the vendored geometry backend used by Clarus Equation.

This copy is kept lean for integration: runtime code, Rust bindings, Python
fallbacks, and tests are retained; upstream repository metadata, build outputs,
experiments, and generated documentation are intentionally omitted.

## Layout

```text
reality_stone/
  src/                    Rust core and PyO3 bindings
  python/reality_stone/   Python API and fallback implementations
  python/reality_stone/clarus/
                          Clarus runtime and CE modules
  examples/               Single unified Clarus/Reality Stone demo
  tests/                  Rust and Python regression tests
  Cargo.toml              Rust crate metadata
  pyproject.toml          Python/maturin package metadata
```

## Python Usage

```python
import reality_stone as rs
from reality_stone.clarus.runtime import BrainRuntime

status = (rs.__version__, rs._has_rust_ext, rs._has_cuda)
```

When the compiled Rust extension is unavailable, `python/reality_stone/_rust.py`
and `python/reality_stone/_fallback.py` provide compatibility paths so Clarus can
still import and run CPU fallback flows.

## Validation

From the repository root:

```powershell
$env:PYTHONPATH = "reality_stone/python"
.\.venv\Scripts\python.exe -m pytest -q reality_stone\tests\layer reality_stone\tests\test_unified_riemannian.py reality_stone\tests\llm\test_metric_attention.py reality_stone\tests\llm\test_metric_router.py reality_stone\tests\api\test_pipeline_api.py
cargo test --manifest-path reality_stone\Cargo.toml --no-default-features
.\.venv\Scripts\python.exe -B reality_stone\examples\unified_clarus_demo.py
```

## Native Build Note

This nested package metadata builds the optional Reality Stone extension as
`reality_stone._rust`. The repository-root `pyproject.toml` builds the optional
Clarus extension as `reality_stone.clarus._rust` for the unified checkout.
Both paths have Python fallbacks, so tests and the unified demo do not require a
native build.
