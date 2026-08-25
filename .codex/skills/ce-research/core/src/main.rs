use std::{
    env, fs,
    io::{self, Read},
    path::{Component, Path, PathBuf},
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
const CURRENT_ROUTE: &str = "revisions/current-route";
const PIVOT_LOG: &str = "revisions/pivots.log";
const REVISION_LIMIT: usize = 2;
const CONTROL_FIELDS: &[&str] = &[
    "Objective-ID",
    "Failure-Equation",
    "Minimal-Assumptions",
    "Counterexample",
    "First-Failing-Line",
    "Failure-Type",
    "Removed-Claims",
    "Preserved-Objective",
    "Regression-Test",
];
const ROUTE_FIELDS: &[&str] = &[
    "Objective-ID",
    "Structural-Class",
    "Changed-Structure",
    "Preserved-Objective",
    "New-Degrees-of-Freedom",
    "Parameter-Accounting",
    "Conservation-Law",
    "Dimension-Check",
    "Stability-Condition",
    "Prior-Negative-Control",
    "Falsifier",
];

#[derive(PartialEq)]
enum FileStatus {
    Complete,
    Skipped,
    Abandoned,
    Incomplete,
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["hook", event] => hook(event),
        ["init", dir] => init(Path::new(dir), false),
        ["init", dir, "--independent"] => init(Path::new(dir), true),
        ["status", dir] => status(Path::new(dir)),
        ["check", dir, stage] => check(Path::new(dir), stage),
        ["revise", dir, role] => revise(Path::new(dir), role),
        ["pivot", dir, route, control] => pivot(Path::new(dir), route, Path::new(control)),
        ["gc", ws] => gc(Path::new(ws)),
        _ => Err(concat!(
            "usage: ce-research-core hook <route|stop> | init <run-dir> [--independent] | ",
            "status <run-dir> | check <run-dir> <contract|lanes|gate|build|final> | ",
            "revise <run-dir> <role> | ",
            "pivot <run-dir> <route-id> <artifacts/negative-controls/file.md> | ",
            "gc <workspace-dir>"
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
                    "additionalContext": "Route narrowly: simple question/one-line fix -> answer directly, no run, no skill; full CE research -> $ce-research; proof/status -> $ce-closure-gate; dimensions -> $ce-dimensionless; tests -> $ce-validate; docs -> $ce-doc-write; guard benchmarks -> $clarus-guard-bench. Load only the selected skill. Use `hooks/run status <run-dir>` instead of rereading stage files."
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

fn init(dir: &Path, independent: bool) -> Result<(), String> {
    if !dir.exists()
        && let Some(ws) = dir.parent()
    {
        let siblings = incomplete_siblings(ws, dir);
        if !siblings.is_empty() && !independent {
            return Err(format!(
                "REUSE required before creating a new run: incomplete run(s): {}. Reuse/close them, or rerun with --independent only after the new-parent admission gate is documented",
                siblings.join(", ")
            ));
        }
    }
    for sub in ["artifacts", "revisions"] {
        fs::create_dir_all(dir.join(sub)).map_err(|e| e.to_string())?;
    }
    let route = dir.join(CURRENT_ROUTE);
    if !route.exists() {
        fs::write(&route, "R0\n").map_err(|e| e.to_string())?;
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

// ---------- status ----------

fn current_route(dir: &Path) -> String {
    fs::read_to_string(dir.join(CURRENT_ROUTE))
        .ok()
        .map(|route| route.trim().to_owned())
        .filter(|route| !route.is_empty())
        .unwrap_or_else(|| "R0".into())
}

fn revision_entry(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some((route, role)) = line.split_once('\t') {
        return Some((route.trim().into(), role.trim().into()));
    }
    // Legacy logs predate structural routes and belong to the original route.
    Some(("R0".into(), line.into()))
}

// One-screen run overview so the model never rereads stage files just to
// learn where it is. Output is line-oriented for easy quoting.
fn status(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("no such run: {}", dir.display()));
    }
    let mut names: Vec<&str> = vec![CONTRACT];
    names.extend_from_slice(LANES);
    names.push(AUDIT);
    names.extend_from_slice(BUILD);
    names.push(FINAL);
    for name in names {
        let label = match file_status(&dir.join(name)) {
            FileStatus::Complete => "COMPLETE",
            FileStatus::Skipped => "SKIPPED",
            FileStatus::Abandoned => "ABANDONED",
            FileStatus::Incomplete if dir.join(name).exists() => "IN_PROGRESS",
            FileStatus::Incomplete => "MISSING",
        };
        println!("{name}: {label}");
    }
    println!("Gate: {}", gate_verdict(&dir.join(AUDIT)).unwrap_or("none"));
    println!("Active route: {}", current_route(dir));
    let log = fs::read_to_string(dir.join("revisions/log")).unwrap_or_default();
    let mut roles: Vec<String> = log
        .lines()
        .filter_map(revision_entry)
        .map(|(route, role)| format!("{route}:{role}"))
        .collect();
    roles.sort_unstable();
    let mut summary = Vec::new();
    for chunk in roles.chunk_by(|a, b| a == b) {
        summary.push(format!("{} {}/{REVISION_LIMIT}", chunk[0], chunk.len()));
    }
    println!(
        "Revisions: {}",
        if summary.is_empty() {
            "none".into()
        } else {
            summary.join(", ")
        }
    );
    let pivots = fs::read_to_string(dir.join(PIVOT_LOG)).unwrap_or_default();
    println!(
        "Pivots: {}",
        if pivots.trim().is_empty() {
            "none".into()
        } else {
            pivots.lines().map(str::trim).collect::<Vec<_>>().join(", ")
        }
    );
    Ok(())
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
            // A blocked scientific claim must still be closed with a complete
            // negative report.  It may never unlock the implementation gate.
            Some("BLOCKED") if stage == "final" => {}
            Some(other) => problems.push(format!("{AUDIT}: Gate is {other}, need PASS")),
            None => problems.push(format!("{AUDIT}: missing `Gate: PASS` line")),
        }
    }
    if depth >= 3 {
        for name in BUILD {
            require(dir, name, true, &mut problems);
        }
        if stage == "final" && gate_verdict(&dir.join(AUDIT)) == Some("BLOCKED") {
            for name in BUILD {
                if file_status(&dir.join(name)) != FileStatus::Skipped {
                    problems.push(format!(
                        "{name}: blocked final requires an explicit SKIPPED reason"
                    ));
                }
            }
        }
    }
    if depth >= 4 {
        require(dir, FINAL, false, &mut problems);
        problems.extend(validate_pivot_ledger(dir));
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
    if role.trim().is_empty() || role.contains(['\t', '\n', '\r']) {
        return Err("role must be a non-empty single-line identifier".into());
    }
    let route = current_route(dir);
    let log = dir.join("revisions/log");
    let text = fs::read_to_string(&log).unwrap_or_default();
    let used = text
        .lines()
        .filter_map(revision_entry)
        .filter(|(logged_route, logged_role)| logged_route == &route && logged_role == role)
        .count();
    if used >= REVISION_LIMIT {
        return Err(format!(
            "local revision limit ({REVISION_LIMIT}) reached for {role} on route {route}: \
             preserve the failed equation in artifacts/negative-controls, register a \
             structurally different route with `pivot`, and keep the research objective OPEN. \
             Do not narrow the objective merely because this equation failed"
        ));
    }
    fs::create_dir_all(dir.join("revisions")).map_err(|e| e.to_string())?;
    fs::write(&log, format!("{text}{route}\t{role}\n")).map_err(|e| e.to_string())?;
    println!(
        "OK revision {}/{REVISION_LIMIT} for {role} on route {route}",
        used + 1
    );
    Ok(())
}

fn valid_route_id(route: &str) -> bool {
    !route.is_empty()
        && route.len() <= 64
        && route
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn valid_control_path(path: &Path) -> bool {
    if path.is_absolute() || path.extension().is_none_or(|ext| ext != "md") {
        return false;
    }
    let parts: Vec<_> = path.components().collect();
    parts.len() >= 3
        && parts
            .iter()
            .all(|part| matches!(part, Component::Normal(_)))
        && parts[0].as_os_str() == "artifacts"
        && parts[1].as_os_str() == "negative-controls"
}

fn field_value<'a>(text: &'a str, field: &str) -> Option<&'a str> {
    let prefix = format!("{field}:");
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(&prefix).map(str::trim))
        .filter(|value| !value.is_empty())
}

fn require_fields(text: &str, fields: &[&str], label: &str) -> Result<(), String> {
    let missing: Vec<_> = fields
        .iter()
        .filter(|field| field_value(text, field).is_none())
        .copied()
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{label} is missing non-empty field(s): {}",
            missing.join(", ")
        ))
    }
}

fn route_block(routes: &str, route: &str) -> Option<String> {
    let marker = format!("Route-ID: {route}");
    let mut found = false;
    let mut block = String::new();
    for line in routes.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Route-ID:") {
            if found {
                break;
            }
            if trimmed == marker {
                found = true;
            }
        }
        if found {
            block.push_str(line);
            block.push('\n');
        }
    }
    found.then_some(block)
}

fn normalized_rel(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn validate_pivot_material(dir: &Path, route: &str, control: &Path) -> Result<(), String> {
    let control_full = dir.join(control);
    let control_text = fs::read_to_string(&control_full).map_err(|_| {
        format!(
            "negative control is missing or not UTF-8 text: {}",
            control_full.display()
        )
    })?;
    require_fields(
        &control_text,
        CONTROL_FIELDS,
        "negative-control certificate",
    )?;

    let routes = fs::read_to_string(dir.join("12-routes.md"))
        .map_err(|_| "12-routes.md is missing or not UTF-8 text".to_string())?;
    let block = route_block(&routes, route)
        .ok_or_else(|| format!("12-routes.md must declare `Route-ID: {route}` before pivot"))?;
    require_fields(&block, ROUTE_FIELDS, &format!("route manifest {route}"))?;

    let class = field_value(&block, "Structural-Class").unwrap_or_default();
    if !matches!(
        class,
        "action-state" | "boundary-source" | "micro-macro" | "observable-readout"
    ) {
        return Err(format!(
            "route manifest {route}: Structural-Class must be action-state, boundary-source, micro-macro, or observable-readout"
        ));
    }
    let control_objective = field_value(&control_text, "Objective-ID").unwrap_or_default();
    let route_objective = field_value(&block, "Objective-ID").unwrap_or_default();
    if control_objective != route_objective {
        return Err(format!(
            "route manifest {route}: Objective-ID must match its negative control"
        ));
    }
    let declared_control = field_value(&block, "Prior-Negative-Control")
        .unwrap_or_default()
        .replace('\\', "/");
    if declared_control != normalized_rel(control) {
        return Err(format!(
            "route manifest {route}: Prior-Negative-Control must equal {}",
            normalized_rel(control)
        ));
    }
    Ok(())
}

fn validate_pivot_ledger(dir: &Path) -> Vec<String> {
    let log_path = dir.join(PIVOT_LOG);
    let Ok(log) = fs::read_to_string(&log_path) else {
        return Vec::new();
    };
    let mut problems = Vec::new();
    for line in log.lines().filter(|line| !line.trim().is_empty()) {
        let Some((route, control)) = line.split_once('\t') else {
            problems.push(format!("{}: malformed entry `{line}`", PIVOT_LOG));
            continue;
        };
        if let Err(error) = validate_pivot_material(dir, route, Path::new(control)) {
            problems.push(format!("{}: {error}", PIVOT_LOG));
        }
    }
    problems
}

fn pivot(dir: &Path, route: &str, control: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("no such run: {}", dir.display()));
    }
    if !valid_route_id(route) {
        return Err(
            "route-id must contain only ASCII letters, digits, '-' or '_' and be <=64 bytes".into(),
        );
    }
    if route == current_route(dir) {
        return Err(format!("route {route} is already active"));
    }
    if !valid_control_path(control) {
        return Err(
            "negative control must be a relative artifacts/negative-controls/*.md path".into(),
        );
    }
    validate_pivot_material(dir, route, control)?;
    let log_path = dir.join(PIVOT_LOG);
    let log = fs::read_to_string(&log_path).unwrap_or_default();
    if log.lines().any(|line| {
        line.split_once('\t')
            .is_some_and(|(logged_route, _)| logged_route == route)
    }) {
        return Err(format!("route {route} was already registered"));
    }
    fs::create_dir_all(dir.join("revisions")).map_err(|e| e.to_string())?;
    fs::write(
        &log_path,
        format!("{log}{route}\t{}\n", normalized_rel(control)),
    )
    .map_err(|e| e.to_string())?;
    fs::write(dir.join(CURRENT_ROUTE), format!("{route}\n")).map_err(|e| e.to_string())?;
    println!(
        "OK pivot to {route}; negative control preserved at {}. Parent research objective remains OPEN",
        control.display()
    );
    Ok(())
}

// ---------- garbage collection ----------

fn reference_text_candidate(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return true;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "md" | "txt"
            | "json"
            | "jsonl"
            | "yaml"
            | "yml"
            | "toml"
            | "rs"
            | "py"
            | "ps1"
            | "cmd"
            | "bat"
            | "sh"
            | "tex"
            | "csv"
            | "html"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "ini"
            | "cfg"
            | "lock"
    )
}

fn encoded_needles(needles: &[String]) -> Vec<Vec<u8>> {
    let mut encoded = Vec::new();
    for needle in needles {
        encoded.push(needle.as_bytes().to_vec());
        let mut little = Vec::new();
        let mut big = Vec::new();
        for unit in needle.encode_utf16() {
            little.extend_from_slice(&unit.to_le_bytes());
            big.extend_from_slice(&unit.to_be_bytes());
        }
        encoded.push(little);
        encoded.push(big);
    }
    encoded
}

fn file_contains_reference(path: &Path, needles: &[Vec<u8>]) -> bool {
    if !reference_text_candidate(path) {
        return false;
    }
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let overlap = needles
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .saturating_sub(1);
    let mut carry = Vec::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let Ok(read) = file.read(&mut buffer) else {
            return false;
        };
        if read == 0 {
            return false;
        }
        carry.extend_from_slice(&buffer[..read]);
        if needles
            .iter()
            .any(|needle| carry.windows(needle.len()).any(|window| window == needle))
        {
            return true;
        }
        if carry.len() > overlap {
            carry.drain(..carry.len() - overlap);
        }
    }
}

fn scan_reference_tree(path: &Path, needles: &[Vec<u8>], hits: &mut Vec<PathBuf>) {
    if path.is_file() {
        if file_contains_reference(path, needles) {
            hits.push(path.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        scan_reference_tree(&entry.path(), needles, hits);
    }
}

fn run_references(ws: &Path, run: &Path, name: &str) -> Vec<PathBuf> {
    let Some(workspace) = ws.parent() else {
        return Vec::new();
    };
    let Some(repo) = workspace.parent() else {
        return Vec::new();
    };
    if ws.file_name().is_none_or(|part| part != "ce")
        || workspace
            .file_name()
            .is_none_or(|part| part != "_workspace")
    {
        return Vec::new();
    }
    let absolute = run.to_string_lossy();
    let needles = vec![
        name.to_string(),
        format!("_workspace/ce/{name}"),
        format!("_workspace\\ce\\{name}"),
        format!("../{name}"),
        format!("..\\{name}"),
        absolute.to_string(),
        absolute.replace('\\', "/"),
    ];
    let needles = encoded_needles(&needles);
    let mut hits = Vec::new();
    for root in [repo.join("docs"), repo.join(".codex")] {
        scan_reference_tree(&root, &needles, &mut hits);
    }
    if let Ok(entries) = fs::read_dir(repo) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                scan_reference_tree(&path, &needles, &mut hits);
            }
        }
    }
    if let Ok(entries) = fs::read_dir(ws) {
        for entry in entries.flatten() {
            let path = entry.path();
            let entry_name = entry.file_name().to_string_lossy().into_owned();
            if path == run || entry_name.starts_with('_') || entry_name.starts_with('.') {
                continue;
            }
            scan_reference_tree(&path, &needles, &mut hits);
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

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
        // Runs whose artifacts freeze absolute/relative paths (sha-locked
        // prereg chains) must never be moved: a `.pin` marker keeps them live.
        let pin = path.join(".pin");
        if pin.exists() {
            let reason = fs::read_to_string(&pin).unwrap_or_default();
            if reason.trim().is_empty() {
                println!("INVALID_PIN {name} (empty .pin; preserved until a reason is recorded)");
            } else {
                println!("PINNED {name} (frozen-path run; not archived — see .pin)");
            }
            continue;
        }
        let references = run_references(ws, &path, &name);
        if !references.is_empty() {
            println!(
                "REFERENCED {name} (not archived): {}",
                references
                    .iter()
                    .map(|reference| reference.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            continue;
        }
        match file_status(&path.join(FINAL)) {
            FileStatus::Complete | FileStatus::Abandoned => {
                fs::create_dir_all(&archive).map_err(|e| e.to_string())?;
                let target = archive.join(&name);
                if target.exists() {
                    println!("COLLISION {name} (archive target exists; not moved)");
                    continue;
                }
                let _ = fs::remove_file(path.join(NAG));
                fs::rename(&path, &target).map_err(|e| e.to_string())?;
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

    fn write_pivot_material(dir: &Path, route: &str, control: &str) {
        fs::create_dir_all(dir.join("artifacts/negative-controls")).unwrap();
        fs::write(
            dir.join(control),
            concat!(
                "Objective-ID: objective-1\n",
                "Failure-Equation: equation-r0\n",
                "Minimal-Assumptions: assumption-a\n",
                "Counterexample: witness-x\n",
                "First-Failing-Line: line-4\n",
                "Failure-Type: conservation\n",
                "Removed-Claims: child-claim\n",
                "Preserved-Objective: objective-1\n",
                "Regression-Test: test-r0\n",
            ),
        )
        .unwrap();
        fs::write(
            dir.join("12-routes.md"),
            format!(
                concat!(
                    "Status: COMPLETE\n\n",
                    "Route-ID: {route}\n",
                    "Objective-ID: objective-1\n",
                    "Structural-Class: action-state\n",
                    "Changed-Structure: replace the action variable\n",
                    "Preserved-Objective: objective-1\n",
                    "New-Degrees-of-Freedom: field-r\n",
                    "Parameter-Accounting: one declared input\n",
                    "Conservation-Law: total current conserved\n",
                    "Dimension-Check: every term dimension four\n",
                    "Stability-Condition: positive Hessian\n",
                    "Prior-Negative-Control: {control}\n",
                    "Falsifier: negative Hessian\n",
                ),
                route = route,
                control = control,
            ),
        )
        .unwrap();
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
            fs::write(
                dir.join(name),
                "Status: SKIPPED (no observational claims)\n",
            )
            .unwrap();
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
        assert!(
            err.contains("loop8-prereg.md"),
            "stray root md must block final"
        );
        fs::create_dir_all(dir.join("artifacts")).unwrap();
        fs::rename(
            dir.join("loop8-prereg.md"),
            dir.join("artifacts/loop8-prereg.md"),
        )
        .unwrap();
        assert!(check(&dir, "final").is_ok());
        assert!(!ws.join(ACTIVE).exists());
    }

    #[test]
    fn blocked_final_closes_only_with_skipped_build() {
        let ws = tmp("blocked-final");
        let dir = ws.join("run-blocked");
        fs::create_dir_all(&dir).unwrap();
        fs::write(ws.join(ACTIVE), "run-blocked").unwrap();
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\n").unwrap();
        for name in LANES {
            fs::write(dir.join(name), "Status: COMPLETE\n").unwrap();
        }
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: BLOCKED\n").unwrap();
        fs::write(dir.join(FINAL), "Status: COMPLETE\n").unwrap();
        fs::write(dir.join(BUILD[0]), "Status: COMPLETE\n").unwrap();
        fs::write(dir.join(BUILD[1]), "Status: SKIPPED (blocked)\n").unwrap();
        assert!(
            check(&dir, "gate").is_err(),
            "blocked must not unlock build"
        );
        assert!(
            check(&dir, "final").is_err(),
            "blocked build must be skipped"
        );
        fs::write(dir.join(BUILD[0]), "Status: SKIPPED (blocked)\n").unwrap();
        assert!(check(&dir, "final").is_ok());
        assert!(!ws.join(ACTIVE).exists());
    }

    #[test]
    fn status_reports_stages_and_revisions() {
        assert!(status(Path::new("definitely-missing-run")).is_err());
        let dir = tmp("status");
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\n").unwrap();
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: REVISE\n").unwrap();
        fs::create_dir_all(dir.join("revisions")).unwrap();
        fs::write(dir.join("revisions/log"), "math-verifier\nmath-verifier\n").unwrap();
        assert!(status(&dir).is_ok());
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
    fn init_refuses_partial_creation_when_an_incomplete_run_exists() {
        let ws = tmp("init-reuse");
        fs::create_dir_all(ws.join("run-open")).unwrap();
        let proposed = ws.join("run-new");
        let error = init(&proposed, false).unwrap_err();
        assert!(error.contains("REUSE required"));
        assert!(
            !proposed.exists(),
            "refused init must not leave a directory"
        );
        assert!(init(&proposed, true).is_ok());
    }

    #[test]
    fn structural_pivot_preserves_control_and_resets_route_local_budget() {
        let dir = tmp("pivot");
        fs::create_dir_all(dir.join("revisions")).unwrap();
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\nCLAIM: keep-open\n").unwrap();
        let contract_before = fs::read(dir.join(CONTRACT)).unwrap();
        write_pivot_material(
            &dir,
            "R1-action",
            "artifacts/negative-controls/failed-r0.md",
        );
        assert!(revise(&dir, "math-verifier").is_ok());
        assert!(revise(&dir, "math-verifier").is_ok());
        let capped = revise(&dir, "math-verifier").unwrap_err();
        assert!(capped.contains("structurally different route"));
        pivot(
            &dir,
            "R1-action",
            Path::new("artifacts/negative-controls/failed-r0.md"),
        )
        .unwrap();
        assert_eq!(current_route(&dir), "R1-action");
        assert!(revise(&dir, "math-verifier").is_ok());
        assert!(revise(&dir, "math-verifier").is_ok());
        assert!(revise(&dir, "math-verifier").is_err());
        assert_eq!(fs::read(dir.join(CONTRACT)).unwrap(), contract_before);
        assert!(
            fs::read_to_string(dir.join(PIVOT_LOG))
                .unwrap()
                .contains("R1-action\tartifacts/negative-controls/failed-r0.md")
        );
    }

    #[test]
    fn pivot_rejects_unregistered_or_missing_negative_control() {
        let dir = tmp("pivot-reject");
        fs::create_dir_all(dir.join("artifacts/negative-controls")).unwrap();
        fs::write(dir.join("12-routes.md"), "Route-ID: R1\n").unwrap();
        assert!(
            pivot(
                &dir,
                "../bad",
                Path::new("artifacts/negative-controls/missing.md")
            )
            .is_err()
        );
        assert!(
            pivot(
                &dir,
                "R1",
                Path::new("artifacts/negative-controls/missing.md")
            )
            .is_err()
        );
        fs::write(
            dir.join("artifacts/negative-controls/control.md"),
            "negative control\n",
        )
        .unwrap();
        assert!(
            pivot(
                &dir,
                "R2",
                Path::new("artifacts/negative-controls/control.md")
            )
            .is_err(),
            "route must be declared before pivot"
        );
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
    fn gc_preserves_referenced_pinned_and_colliding_runs() {
        let root = tmp("gc-safe");
        let ws = root.join("_workspace/ce");
        let referenced = ws.join("run-referenced");
        let unreferenced = ws.join("run-unreferenced");
        let pinned = ws.join("run-pinned");
        let colliding = ws.join("run-colliding");
        let predecessor = ws.join("run-predecessor");
        let successor = ws.join("run-successor");
        for dir in [
            &referenced,
            &unreferenced,
            &pinned,
            &colliding,
            &predecessor,
        ] {
            fs::create_dir_all(dir).unwrap();
            fs::write(dir.join(FINAL), "Status: COMPLETE\n").unwrap();
        }
        fs::create_dir_all(&successor).unwrap();
        fs::write(
            successor.join(CONTRACT),
            "Status: IN_PROGRESS\nPREDECESSOR: ../run-predecessor\n",
        )
        .unwrap();
        fs::write(pinned.join(".pin"), "canonical reference\n").unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(
            root.join("docs/reference.md"),
            "See _workspace/ce/run-referenced/40-final-report.md\n",
        )
        .unwrap();
        fs::create_dir_all(ws.join("_archive/run-colliding")).unwrap();
        gc(&ws).unwrap();
        assert!(referenced.exists());
        assert!(pinned.exists());
        assert!(colliding.exists());
        assert!(
            predecessor.exists(),
            "live-run references must be preserved"
        );
        assert!(ws.join("_archive/run-unreferenced").exists());
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
