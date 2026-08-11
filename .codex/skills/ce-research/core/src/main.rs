use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::SystemTime,
};

use serde_json::{Value, json};

const CONTRACT: &str = "00-contract.md";
const LANES: &[&str] = &["10-sources.md", "11-math.md", "12-routes.md"];
const AUDIT: &str = "20-audit.md";
const BUILD: &[&str] = &["30-implementation.md", "31-validation.md"];
const FINAL: &str = "40-final-report.md";
const ACTIVE: &str = ".active-run";
const NAG: &str = ".nag";
const REVISION_LIMIT: usize = 2;

#[derive(PartialEq)]
enum FileStatus {
    Complete,
    Skipped,
    Abandoned,
    Incomplete,
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let result = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["hook", event] => hook(event),
        ["init", dir] => init(Path::new(dir)),
        ["check", dir, stage] => check(Path::new(dir), stage),
        ["revise", dir, role] => revise(Path::new(dir), role),
        ["gc", ws] => gc(Path::new(ws)),
        _ => Err(concat!(
            "usage: ce-research-core hook <route|stop> | init <run-dir> | ",
            "check <run-dir> <contract|lanes|gate|build|final> | ",
            "revise <run-dir> <role> | gc <workspace-dir>"
        )
        .into()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

// ---------- hooks ----------

fn hook(event: &str) -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    // A UTF-8 BOM sneaks in when stdin is piped through PowerShell/.NET.
    let payload: Value =
        serde_json::from_str(input.trim_start_matches('\u{feff}')).unwrap_or(Value::Null);
    match event {
        "route" => {
            let prompt = payload["prompt"].as_str().unwrap_or("").to_lowercase();
            if is_ce_task(&prompt) {
                let out = json!({"hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "Route narrowly: full CE research -> $ce-research; proof/status -> $ce-closure-gate; dimensions -> $ce-dimensionless; tests -> $ce-validate; docs -> $ce-doc-write; guard benchmarks -> $clarus-guard-bench. Load only the selected skill and keep output concise."
                }});
                println!("{out}");
            }
        }
        "stop" => {
            if payload["stop_hook_active"].as_bool().unwrap_or(false) {
                println!("{{}}");
                return Ok(());
            }
            let cwd = payload["cwd"].as_str().unwrap_or(".");
            if let Some(reason) = stop_reason(Path::new(cwd)) {
                let out = json!({"decision": "block", "reason": reason});
                println!("{out}");
            } else {
                println!("{{}}");
            }
        }
        _ => return Err(format!("unknown hook event: {event}")),
    }
    Ok(())
}

fn is_ce_task(prompt: &str) -> bool {
    ["clarus", "axium", "reality_stone", "부트스트랩", "경로적분"]
        .iter()
        .any(|term| prompt.contains(term))
        || prompt
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| word == "ce")
}

// Block the stop at most once per work burst: if the run advanced since the
// last nag, nag again; otherwise let the stop through (honor system resumes).
fn stop_reason(cwd: &Path) -> Option<String> {
    let ws = cwd.join("_workspace/ce");
    let pointer = ws.join(ACTIVE);
    let run_rel = read_text(&pointer)?.trim().to_string();
    let run = if Path::new(&run_rel).is_absolute() {
        PathBuf::from(&run_rel)
    } else {
        cwd.join(&run_rel)
    };
    match file_status(&run.join(FINAL)) {
        FileStatus::Complete | FileStatus::Abandoned => {
            let _ = fs::remove_file(&pointer);
            return None;
        }
        _ => {}
    }
    let nag = run.join(NAG);
    let nagged_at = mtime(&nag);
    if let Some(nagged_at) = nagged_at
        && latest_md_mtime(&run).is_none_or(|m| m <= nagged_at)
    {
        return None;
    }
    let _ = fs::write(&nag, "");
    Some(format!(
        "CE run {run_rel} is incomplete. Either finish the remaining stages and pass \
         `check {run_rel} final`, or write `Status: ABANDONED` in {run_rel}/{FINAL} \
         with a one-line reason. Leave a `CE_RUN={run_rel}` line in the final message."
    ))
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn latest_md_mtime(dir: &Path) -> Option<SystemTime> {
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
        .filter_map(|e| mtime(&e.path()))
        .max()
}

// ---------- run lifecycle ----------

fn init(dir: &Path) -> Result<(), String> {
    for sub in ["artifacts", "revisions"] {
        fs::create_dir_all(dir.join(sub)).map_err(|e| e.to_string())?;
    }
    let contract = dir.join(CONTRACT);
    if !contract.exists() {
        fs::write(
            &contract,
            "# Research contract\n\nStatus: IN_PROGRESS\n\nPREDECESSOR: none\n",
        )
        .map_err(|e| e.to_string())?;
    }
    if let Some(ws) = dir.parent() {
        fs::write(ws.join(ACTIVE), dir.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        for name in incomplete_siblings(ws, dir) {
            println!("REUSE? incomplete run exists: {name}");
        }
    }
    println!("{}", dir.display());
    Ok(())
}

fn incomplete_siblings(ws: &Path, current: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(ws) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| e.path().is_dir() && e.path() != current)
        .filter(|e| !e.file_name().to_string_lossy().starts_with('_'))
        .filter(|e| {
            !matches!(
                file_status(&e.path().join(FINAL)),
                FileStatus::Complete | FileStatus::Abandoned
            )
        })
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------- stage checks ----------

fn check(dir: &Path, stage: &str) -> Result<(), String> {
    let order = ["contract", "lanes", "gate", "build", "final"];
    let depth = order
        .iter()
        .position(|s| *s == stage)
        .ok_or_else(|| format!("unknown stage: {stage}"))?;
    let mut problems = Vec::new();

    require(dir, CONTRACT, false, &mut problems);
    if depth >= 1 {
        for name in LANES {
            require(dir, name, true, &mut problems);
        }
    }
    if depth >= 2 {
        require(dir, AUDIT, false, &mut problems);
        match gate_verdict(&dir.join(AUDIT)) {
            Some("PASS") => {}
            Some(other) => problems.push(format!("{AUDIT}: Gate is {other}, need PASS")),
            None => problems.push(format!("{AUDIT}: missing `Gate: PASS` line")),
        }
    }
    if depth >= 3 {
        for name in BUILD {
            require(dir, name, true, &mut problems);
        }
    }
    if depth >= 4 {
        require(dir, FINAL, false, &mut problems);
        for stray in stray_root_md(dir) {
            problems.push(format!("{stray}: stray root file, move into artifacts/"));
        }
    }

    if problems.is_empty() {
        if stage == "final"
            && let Some(ws) = dir.parent()
        {
            let pointer = ws.join(ACTIVE);
            if fs::read_to_string(&pointer)
                .is_ok_and(|p| dir.ends_with(p.trim()) || Path::new(p.trim()) == dir)
            {
                let _ = fs::remove_file(&pointer);
            }
            let _ = fs::remove_file(dir.join(NAG));
        }
        println!("OK {stage}");
        Ok(())
    } else {
        Err(format!("incomplete {stage}:\n  {}", problems.join("\n  ")))
    }
}

// The run root holds exactly the eight numbered stage files; everything
// else (preregs, per-loop notes, validation scratch) belongs in artifacts/.
fn stray_root_md(dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut stray: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| {
            name != CONTRACT
                && name != AUDIT
                && name != FINAL
                && !LANES.contains(&name.as_str())
                && !BUILD.contains(&name.as_str())
        })
        .collect();
    stray.sort();
    stray
}

fn require(dir: &Path, name: &str, skippable: bool, problems: &mut Vec<String>) {
    match file_status(&dir.join(name)) {
        FileStatus::Complete => {}
        FileStatus::Skipped if skippable => {}
        FileStatus::Skipped => problems.push(format!("{name}: SKIPPED not allowed here")),
        FileStatus::Abandoned => problems.push(format!("{name}: marked ABANDONED")),
        FileStatus::Incomplete => problems.push(format!("{name}: missing or not COMPLETE")),
    }
}

// Strips the UTF-8 BOM that PowerShell 5.1 prepends when writing files.
fn read_text(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|t| t.trim_start_matches('\u{feff}').to_string())
}

fn file_status(path: &Path) -> FileStatus {
    let Some(text) = read_text(path) else {
        return FileStatus::Incomplete;
    };
    for line in text.lines() {
        let line = line.trim();
        if line == "Status: COMPLETE" {
            return FileStatus::Complete;
        }
        if line.starts_with("Status: SKIPPED") {
            return FileStatus::Skipped;
        }
        if line.starts_with("Status: ABANDONED") {
            return FileStatus::Abandoned;
        }
    }
    FileStatus::Incomplete
}

fn gate_verdict(path: &Path) -> Option<&'static str> {
    let text = read_text(path)?;
    text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("Gate:")?.trim();
        ["PASS", "REVISE", "BLOCKED"]
            .into_iter()
            .find(|v| rest.starts_with(v))
    })
}

// ---------- revision budget ----------

fn revise(dir: &Path, role: &str) -> Result<(), String> {
    let log = dir.join("revisions/log");
    let text = fs::read_to_string(&log).unwrap_or_default();
    let used = text.lines().filter(|line| line.trim() == role).count();
    if used >= REVISION_LIMIT {
        return Err(format!(
            "revision limit ({REVISION_LIMIT}) reached for {role}: mark the finding BLOCKED in \
             {AUDIT} and record it in {FINAL} instead of revising again"
        ));
    }
    fs::create_dir_all(dir.join("revisions")).map_err(|e| e.to_string())?;
    fs::write(&log, format!("{text}{role}\n")).map_err(|e| e.to_string())?;
    println!("OK revision {}/{REVISION_LIMIT} for {role}", used + 1);
    Ok(())
}

// ---------- garbage collection ----------

fn gc(ws: &Path) -> Result<(), String> {
    let archive = ws.join("_archive");
    let entries = fs::read_dir(ws).map_err(|e| e.to_string())?;
    let mut archived = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !path.is_dir() || name.starts_with('_') || name.starts_with('.') {
            continue;
        }
        match file_status(&path.join(FINAL)) {
            FileStatus::Complete | FileStatus::Abandoned => {
                fs::create_dir_all(&archive).map_err(|e| e.to_string())?;
                let _ = fs::remove_file(path.join(NAG));
                fs::rename(&path, archive.join(&name)).map_err(|e| e.to_string())?;
                println!("ARCHIVED {name}");
                archived.push(name);
            }
            _ => println!("STALE {name} (incomplete; finish, resume, or mark ABANDONED)"),
        }
    }
    let pointer = ws.join(ACTIVE);
    if let Ok(active) = fs::read_to_string(&pointer)
        && archived.iter().any(|name| active.trim().ends_with(name))
    {
        let _ = fs::remove_file(&pointer);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = env::temp_dir().join("ce-core-test").join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn routes_on_ce_tokens_only() {
        assert!(is_ce_task("clarus-eq research"));
        assert!(is_ce_task("ce 검증"));
        assert!(is_ce_task("ce-research run"));
        assert!(is_ce_task("ce의 상수"));
        assert!(!is_ce_task("git force-push please"));
        assert!(!is_ce_task("reduce-latency work"));
    }

    #[test]
    fn gate_requires_pass() {
        let dir = tmp("gate");
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\n").unwrap();
        for name in LANES {
            fs::write(dir.join(name), "Status: SKIPPED (no observational claims)\n").unwrap();
        }
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: BLOCKED\n").unwrap();
        assert!(check(&dir, "gate").is_err());
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: PASS\n").unwrap();
        assert!(check(&dir, "gate").is_ok());
    }

    #[test]
    fn final_checks_whole_chain_and_clears_pointer() {
        let ws = tmp("chain");
        let dir = ws.join("run-a");
        fs::create_dir_all(&dir).unwrap();
        fs::write(ws.join(ACTIVE), "run-a").unwrap();
        fs::write(dir.join(FINAL), "Status: COMPLETE\n").unwrap();
        assert!(check(&dir, "final").is_err(), "final alone must not pass");
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\n").unwrap();
        for name in LANES {
            fs::write(dir.join(name), "Status: COMPLETE\n").unwrap();
        }
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: PASS\n").unwrap();
        for name in BUILD {
            fs::write(dir.join(name), "Status: SKIPPED (no code change)\n").unwrap();
        }
        fs::write(dir.join("loop8-prereg.md"), "scratch\n").unwrap();
        let err = check(&dir, "final").unwrap_err();
        assert!(err.contains("loop8-prereg.md"), "stray root md must block final");
        fs::create_dir_all(dir.join("artifacts")).unwrap();
        fs::rename(dir.join("loop8-prereg.md"), dir.join("artifacts/loop8-prereg.md")).unwrap();
        assert!(check(&dir, "final").is_ok());
        assert!(!ws.join(ACTIVE).exists());
    }

    #[test]
    fn revision_budget_is_enforced() {
        let dir = tmp("revise");
        assert!(revise(&dir, "math-verifier").is_ok());
        assert!(revise(&dir, "math-verifier").is_ok());
        assert!(revise(&dir, "math-verifier").is_err());
        assert!(revise(&dir, "impl-engineer").is_ok());
    }

    #[test]
    fn gc_archives_complete_runs() {
        let ws = tmp("gc");
        let done = ws.join("run-done");
        let open = ws.join("run-open");
        fs::create_dir_all(&done).unwrap();
        fs::create_dir_all(&open).unwrap();
        fs::write(done.join(FINAL), "Status: COMPLETE\n").unwrap();
        fs::write(ws.join(ACTIVE), "run-done").unwrap();
        gc(&ws).unwrap();
        assert!(ws.join("_archive/run-done").exists());
        assert!(open.exists());
        assert!(!ws.join(ACTIVE).exists());
    }

    #[test]
    fn stop_nags_once_per_burst() {
        let cwd = tmp("stop");
        let ws = cwd.join("_workspace/ce");
        let run = ws.join("run-x");
        fs::create_dir_all(&run).unwrap();
        fs::write(ws.join(ACTIVE), "_workspace/ce/run-x").unwrap();
        fs::write(run.join(FINAL), "Status: IN_PROGRESS\n").unwrap();
        assert!(stop_reason(&cwd).is_some(), "first stop must nag");
        assert!(stop_reason(&cwd).is_none(), "second stop passes");
        fs::write(run.join(FINAL), "Status: ABANDONED (superseded)\n").unwrap();
        assert!(stop_reason(&cwd).is_none());
        assert!(!ws.join(ACTIVE).exists(), "abandon clears the pointer");
    }
}
