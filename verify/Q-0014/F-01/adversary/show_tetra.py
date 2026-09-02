import json
from pathlib import Path
d = json.loads(Path(r"c:/dev/ce/Clarus-Equation/verify/Q-0014/F-01/adversary/degree3_curvature_v2.json").read_text(encoding="utf-8"))
print(json.dumps(d["tetrahedron"], ensure_ascii=False, indent=2))
