import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / "a4_kill_audit.json").read_text(encoding="utf-8"))
print("eps_star", R["eps_star"])
print(json.dumps(R["physics_mc_mini_binary"], indent=1))
print("windows ok:", R["all_windows_match_card"], "alts outside:", R["all_f02_alternatives_outside"])
for k, v in R["windows"].items():
    print("  %-34s prereg=%-8s win=%s alt=%-7s margin=%.2f halfwidths" % (
        k, v["prereg"], v["card_window"], v["f02_alternative"], v["alt_margin_in_halfwidths"]))
