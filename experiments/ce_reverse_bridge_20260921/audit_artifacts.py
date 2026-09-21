"""Verify this batch's source/result linkage and new chapter links, then write a receipt."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


def main():
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    chapter_dir = root / "paper" / "후속연구_기록과_상태선택"
    chapters = [next(chapter_dir.glob(f"{number}_*.md")) for number in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
    receipt = {"scope": "new CE-RB1 through CE-RB12 artifacts only", "full_thread_goal_complete": False,
               "result_sets": [], "chapter_links": [], "sha256": {}}

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    total, combined = 0, []
    for name, source, first, last in [
        ("results.json", "verify_reverse.py", 1, 21),
        ("results_empirical.json", "verify_empirical.py", 22, 32),
        ("results_branches.json", "verify_branches.py", 33, 40),
        ("results_states.json", "verify_states.py", 41, 48),
        ("results_muon.json", "verify_muon.py", 49, 50),
        ("results_geometry.json", "verify_geometry.py", 51, 58),
        ("results_curvature.json", "verify_curvature.py", 59, 64),
        ("results_realtime.json", "verify_realtime.py", 65, 70),
        ("results_stability.json", "verify_stability.py", 71, 76),
        ("results_uv.json", "verify_uv.py", 77, 82),
        ("results_transport.json", "verify_transport.py", 83, 89),
        ("results_records.json", "verify_records.py", 90, 96),
        ("results_identifiability.json", "verify_identifiability.py", 97, 103),
    ]:
        result = json.loads((here / name).read_text(encoding="utf-8"))
        assert result["source_sha256"] == sha(here / source), (name, "stale source hash")
        if "helper_sha256" in result:
            assert result["helper_sha256"] == sha(here / "verify_reverse.py"), "stale helper"
        assert result["number_of_checks"] == len(result["checks"])
        assert result["all_passed"] and all(row["passed"] for row in result["checks"])
        expected = [f"R{i:02d}" for i in range(first, last+1)]
        assert result["claim_ids"] == expected
        assert sorted({row["claim"] for row in result["checks"]}, key=lambda value: int(value[1:])) == expected
        total += result["number_of_checks"]
        combined.extend(expected)
        receipt["result_sets"].append({"file": name, "count": result["number_of_checks"],
                                       "claims": expected})

    chapter_ids = []
    for chapter in chapters:
        body = chapter.read_text(encoding="utf-8")
        chapter_ids.extend(re.findall(r"^## \d+\.\d+ (R\d+)\b", body, re.M))
        assert body.count("\\[") == body.count("\\]"), (chapter.name, "display math")
        assert "\ufffd" not in body, (chapter.name, "replacement character")
        # Display equations such as \right](n+u) are not Markdown links.
        link_body = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$", "", body, flags=re.S)
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)\n]+)\)", link_body):
            target = match.group(1)
            if re.match(r"[a-zA-Z]+:", target) or target.startswith("#"):
                continue
            path = (chapter.parent / unquote(target.split("#")[0])).resolve()
            assert path.is_file(), (chapter.name, "missing local target", target)
            receipt["chapter_links"].append({"chapter": chapter.name, "target": target,
                                               "exists": True, "target_sha256": sha(path)})
    assert chapter_ids == combined, (chapter_ids, combined)

    sources = list(here.glob("*.py"))+[here / "requirements.txt"]+chapters
    sources += [here / name for name in ["results.json", "results_empirical.json", "results_branches.json", "results_states.json", "results_muon.json", "results_geometry.json", "results_curvature.json", "results_realtime.json", "results_stability.json", "results_uv.json", "results_transport.json", "results_records.json", "results_identifiability.json"]]
    sources += [root / "paper" / "검증_원장" / "20260921_경험식_역증명_대응.md"]
    navigation = [root / "README.md", root / "paper" / "README.md", root / "paper" / "00_읽기_지도.md",
                  root / "paper" / "CE_통합_논문.md", root / "paper" / "진전_원장.md", chapter_dir / "README.md"]
    for path in navigation:
        body = path.read_text(encoding="utf-8")
        marker = f"{len(combined)}개 주장 그룹·{total}개"
        assert marker in body, (path.name, "stale current summary count", marker)
    sources += navigation
    for path in sources:
        receipt["sha256"][path.relative_to(root).as_posix()] = sha(path)
    receipt["total_checks"] = total
    receipt["claim_ids"] = combined
    receipt["all_artifact_checks_passed"] = True
    (here / "artifact_audit.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n",
                                              encoding="utf-8")
    print(f"AUDIT PASS: {len(combined)} claims, {total} checks, {len(receipt['chapter_links'])} local links")


if __name__ == "__main__":
    main()
