import json
from pathlib import Path
H = Path(__file__).resolve().parent
txt = (H / "hook_rerun.json").read_text(encoding="utf-8")
i = txt.find("{")
d = json.loads(txt[i:])
det = d["details"]
print("n_checks", len(det), "symbolic", d.get("symbolic"), "numeric", d.get("numeric"))
print("all_pass", all(x.get("numeric") == "pass" for x in det))
old = json.loads((H.parent / "hook_result.json").read_text(encoding="utf-8"))
print("same_as_recorded", [x.get("index") for x in det] == [x.get("index") for x in old["details"]],
      len(old["details"]))
