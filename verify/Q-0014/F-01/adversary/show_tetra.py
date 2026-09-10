import json
from pathlib import Path

source = Path(__file__).with_name("degree3_curvature_v2.json")
d = json.loads(source.read_text(encoding="utf-8"))
print(json.dumps(d["tetrahedron"], ensure_ascii=False, indent=2))
