"""book의 파이썬 코드 조각 실행 검사.

book/ 아래 모든 마크다운에서 ```python 블록을 뽑아 블록마다 새 이름공간에서 실행한다.
저장소 루트를 작업 디렉터리로 두고 실행한다(일부 조각은 사전 등록 파일을 읽는다).

실행(저장소 루트): python -B book/tools/run_snippets.py [장 폴더 이름의 일부 ...]
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def main(filters: list[str]) -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    files = sorted(p for p in BOOK.rglob("*.md") if not filters or any(f in str(p) for f in filters))
    total = failed = 0
    for path in files:
        for i, code in enumerate(BLOCK.findall(path.read_text(encoding="utf-8"))):
            total += 1
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"{path.name}#{i}", "exec"), {"__name__": "__snippet__"})
            except Exception:
                failed += 1
                print(f"FAIL {path.relative_to(BOOK)} 블록 {i}")
                print(traceback.format_exc(limit=2))
    print(f"{total - failed}/{total} 블록 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
