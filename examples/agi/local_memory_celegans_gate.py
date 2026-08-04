"""Run the leakage-resistant local temporal-memory gate."""

from __future__ import annotations

import importlib
import importlib.machinery
from pathlib import Path
import sys
import types


def _source_tree_main():
    """Load the NumPy/SciPy gate without importing the torch-heavy package root."""

    source_root = Path(__file__).resolve().parents[2] / "reality_stone/python/reality_stone"
    package_paths = {
        "reality_stone": source_root,
        "reality_stone.clarus": source_root / "clarus",
    }
    for package_name, package_path in package_paths.items():
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package
    sys.modules["reality_stone"].clarus = sys.modules["reality_stone.clarus"]
    return importlib.import_module("reality_stone.clarus.local_memory").main


try:
    from reality_stone.clarus.local_memory import main
except ModuleNotFoundError as error:
    if error.name not in {"reality_stone", "torch"}:
        raise
    main = _source_tree_main()


if __name__ == "__main__":
    raise SystemExit(main())
