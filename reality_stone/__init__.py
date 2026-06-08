"""Checkout-time shim for the Reality Stone Python package.

The source package lives in ``reality_stone/python/reality_stone`` so maturin
can build it cleanly. This shim makes ``import reality_stone`` work directly
from the repository root as well.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_INNER = _ROOT / "python" / "reality_stone"
_INNER_INIT = _INNER / "__init__.py"

if not _INNER_INIT.exists():
    raise ImportError(f"Reality Stone Python package not found at {_INNER_INIT}")

__path__ = [str(_INNER), str(_ROOT)]
if __spec__ is not None and __spec__.submodule_search_locations is not None:
    __spec__.submodule_search_locations[:] = __path__

__file__ = str(_INNER_INIT)

with _INNER_INIT.open("r", encoding="utf-8") as _f:
    _code = compile(_f.read(), __file__, "exec")
exec(_code, globals(), globals())

del Path, _ROOT, _INNER, _INNER_INIT, _f, _code
