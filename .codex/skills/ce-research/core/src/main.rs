use std::{
    collections::HashSet,
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
const RESULT_LEDGER: &str = "35-result-ledger.md";
const FINAL: &str = "40-final-report.md";
const ACTIVE: &str = ".active-run";
const ACTIVE_EPOCH: &str = ".active-epoch";
const EPOCHS: &str = "artifacts/epochs";
const COUNTEREXAMPLE: &str = "counterexample.json";
const PORTFOLIO: &str = "portfolio.json";
const NAG: &str = ".nag";
const REVISION_LIMIT: usize = 3;

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
        ["init", "--new-contract", dir] => init(Path::new(dir), true),
        ["status", dir] => status(Path::new(dir)),
        ["check", dir, stage] => check(Path::new(dir), stage),
        ["revise", dir, role] => revise(Path::new(dir), role),
        ["counterexample", dir, id] => counterexample(Path::new(dir), id),
        ["pivot", dir, counterexample_id, route_id] => {
            pivot(Path::new(dir), counterexample_id, route_id)
        }
        ["gc", ws] => gc(Path::new(ws)),
        _ => Err(concat!(
            "usage: ce-research-core hook <route|stop> | ",
            "init [--new-contract] <run-dir> | ",
            "status <run-dir> | check <run-dir> <contract|lanes|gate|build|final> | ",
            "revise <run-dir> <role> | counterexample <run-dir> <cex-id> | ",
            "pivot <run-dir> <cex-id> <route-id> | gc <workspace-dir>"
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
                    "additionalContext": "Route narrowly: simple question/one-line fix -> answer directly, no run, no skill; full CE research -> $ce-research; proof/status -> $ce-closure-gate; dimensions -> $ce-dimensionless; tests -> $ce-validate; ledgers -> $ce-ledger-write; papers/lectures -> $ce-paper-write. Load only the selected skill. Use `hooks/run status <run-dir>` instead of rereading stage files."
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
    if let Some(problem) = active_epoch_problem(&run) {
        let nag = run.join(NAG);
        let nagged_at = mtime(&nag);
        if let Some(nagged_at) = nagged_at
            && latest_md_mtime(&run).is_none_or(|m| m <= nagged_at)
        {
            return None;
        }
        let _ = fs::write(&nag, "");
        return Some(format!(
            "CE run {run_rel} has an unfinished counterexample/pivot epoch: {problem}. \
             Continue inside the same CE_RUN; do not create a successor workspace. \
             Leave a `CE_RUN={run_rel}` line in the final message."
        ));
    }
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

fn init(dir: &Path, allow_new_contract: bool) -> Result<(), String> {
    // Re-entering the active run is safe. Creating or switching to a different
    // run while one is unfinished requires an explicit contract-change flag;
    // importantly, this gate runs before creating any directory or pointer.
    if let Some(ws) = dir.parent() {
        if !allow_new_contract
            && let Some(active) = active_incomplete_run(ws)
            && !same_existing_path(&active, dir)
        {
            return Err(format!(
                "REUSE_REQUIRED: active CE run is unfinished: {}. Resume that CE_RUN; \
                 recovery, recalculation, figures, audits, and manuscript revision stay in that \
                 run. Only a genuinely changed scientific contract may use \
                 `init --new-contract <run-dir>`.",
                active.display()
            ));
        }
        if !dir.exists() {
            let candidates = incomplete_siblings(ws, dir);
            if !candidates.is_empty() && !allow_new_contract {
                return Err(format!(
                    "REUSE_REQUIRED: unfinished CE run(s) exist: {}. Resume the applicable CE_RUN; \
                     recovery, recalculation, figures, audits, and manuscript revision stay in that \
                     run. Only a genuinely changed scientific contract may use \
                     `init --new-contract <run-dir>`.",
                    candidates.join(", ")
                ));
            }
        }
    }
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
    }
    println!("{}", dir.display());
    Ok(())
}

struct ActiveEpoch {
    counterexample_id: String,
    route_id: Option<String>,
}

fn active_incomplete_run(ws: &Path) -> Option<PathBuf> {
    let raw = read_text(&ws.join(ACTIVE))?;
    let stored = PathBuf::from(raw.trim());
    let candidate = if stored.is_absolute() || stored.exists() {
        stored
    } else {
        ws.join(stored)
    };
    if candidate.is_dir() && !run_is_terminal(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn same_existing_path(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn incomplete_siblings(ws: &Path, current: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(ws) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| e.path().is_dir() && e.path() != current)
        .filter(|e| !e.file_name().to_string_lossy().starts_with('_'))
        .filter(|e| !run_is_terminal(&e.path()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------- counterexample -> mechanism portfolio -> pivot ----------

fn counterexample(dir: &Path, id: &str) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("run이 없습니다: {}", dir.display()));
    }
    validate_id(id)?;
    if dir.join(ACTIVE_EPOCH).exists() {
        return Err(format!(
            "{}에 활성 epoch가 있습니다. 같은 자리에서 완료·통합한 뒤 다음 epoch를 여십시오",
            dir.display()
        ));
    }
    let epoch = dir.join(EPOCHS).join(id);
    if epoch.exists() {
        return Err(format!("반례 epoch가 이미 있습니다: {}", epoch.display()));
    }
    fs::create_dir_all(&epoch).map_err(|e| e.to_string())?;

    let record = json!({
        "schema": "ce-counterexample-v1",
        "id": id,
        "status": "IN_PROGRESS",
        "kind": "",
        "parent_claim": "",
        "witnesses": [],
        "violated_assumption": "",
        "preserved_result": "",
        "claim_disposition": "SUSPENDED"
    });
    let portfolio = json!({
        "schema": "ce-mechanism-portfolio-v1",
        "counterexample_id": id,
        "status": "IN_PROGRESS",
        "selected_route": null,
        "routes": []
    });
    write_json(&epoch.join(COUNTEREXAMPLE), &record)?;
    write_json(&epoch.join(PORTFOLIO), &portfolio)?;
    write_json(
        &dir.join(ACTIVE_EPOCH),
        &json!({
            "schema": "ce-active-epoch-v1",
            "counterexample_id": id,
            "route_id": null
        }),
    )?;
    set_active_run(dir)?;
    println!(
        "반례 epoch 열림: {}. {}를 잠그고, {}에 구조적으로 다른 경로를 3개 이상 등록한 뒤 pivot하십시오",
        id,
        epoch.join(COUNTEREXAMPLE).display(),
        epoch.join(PORTFOLIO).display()
    );
    Ok(())
}

fn pivot(dir: &Path, counterexample_id: &str, route_id: &str) -> Result<(), String> {
    validate_id(counterexample_id)?;
    validate_id(route_id)?;
    let active = active_epoch(dir)?
        .ok_or_else(|| format!("{}에 활성 반례 epoch가 없습니다", dir.display()))?;
    if active.counterexample_id != counterexample_id {
        return Err(format!(
            "활성 반례는 {}이며 {counterexample_id}가 아닙니다",
            active.counterexample_id
        ));
    }
    if active.route_id.is_some() {
        return Err(
            "이미 활성 pivot 경로가 있습니다. 새 폴더를 만들지 말고 그 자리에서 완료하십시오"
                .into(),
        );
    }

    let epoch = dir.join(EPOCHS).join(counterexample_id);
    let record = read_json(&epoch.join(COUNTEREXAMPLE))?;
    validate_counterexample(&record, counterexample_id)?;
    let portfolio = read_json(&epoch.join(PORTFOLIO))?;
    let selected = validate_portfolio(&portfolio, counterexample_id, route_id)?;

    let route_dir = epoch.join("pivots").join(route_id);
    if route_dir.exists() {
        return Err(format!(
            "pivot 경로가 이미 있습니다: {}",
            route_dir.display()
        ));
    }
    fs::create_dir_all(&route_dir).map_err(|e| e.to_string())?;
    write_json(&route_dir.join("route.json"), &selected)?;
    let fingerprint = required_string(&selected, "structural_fingerprint")?;
    let change = required_string(&selected, "structural_change")?;
    let changed_term = required_string(&selected, "changed_term")?;
    let prediction = required_string(&selected, "prediction")?;
    let kill_condition = required_string(&selected, "kill_condition")?;
    fs::write(
        route_dir.join("contract.md"),
        format!(
            "# 구조 피벗 계약: {route_id}\n\nStatus: IN_PROGRESS\n\n\
             반례 식별자: `{counterexample_id}`\n\n\
             구조 지문: `{fingerprint}`\n\n\
             구조 변경 종류: `{change}`\n\n\
             바뀌는 항: {changed_term}\n\n\
             판별 예측: {prediction}\n\n\
             중단 조건: {kill_condition}\n"
        ),
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        route_dir.join("report.md"),
        format!("# 구조 피벗 결과: {route_id}\n\nStatus: IN_PROGRESS\n"),
    )
    .map_err(|e| e.to_string())?;
    write_json(
        &dir.join(ACTIVE_EPOCH),
        &json!({
            "schema": "ce-active-epoch-v1",
            "counterexample_id": counterexample_id,
            "route_id": route_id
        }),
    )?;
    set_active_run(dir)?;
    println!(
        "피벗 경로 열림: {counterexample_id}/{route_id}, 위치 {}",
        route_dir.display()
    );
    Ok(())
}

fn validate_counterexample(record: &Value, id: &str) -> Result<(), String> {
    if required_string(record, "schema")? != "ce-counterexample-v1" {
        return Err("counterexample.json: unsupported schema".into());
    }
    if required_string(record, "id")? != id {
        return Err("counterexample.json: id does not match active epoch".into());
    }
    if required_string(record, "status")? != "COUNTEREXAMPLE_LOCKED" {
        return Err("counterexample.json: status must be COUNTEREXAMPLE_LOCKED".into());
    }
    let kind = required_string(record, "kind")?;
    if !["COMPLETE_COUNTEREXAMPLE", "EMPIRICAL_CONTRADICTION"].contains(&kind) {
        return Err(
            "counterexample.json: kind must be COMPLETE_COUNTEREXAMPLE or EMPIRICAL_CONTRADICTION"
                .into(),
        );
    }
    for field in ["parent_claim", "violated_assumption", "preserved_result"] {
        required_string(record, field)?;
    }
    let disposition = required_string(record, "claim_disposition")?;
    if !["SUSPENDED", "RETRACTED"].contains(&disposition) {
        return Err("counterexample.json: claim_disposition must be SUSPENDED or RETRACTED".into());
    }
    let witnesses = record["witnesses"]
        .as_array()
        .filter(|items| {
            !items.is_empty()
                && items
                    .iter()
                    .all(|item| item.as_str().is_some_and(|text| !text.trim().is_empty()))
        })
        .ok_or_else(|| {
            "counterexample.json: witnesses must be a non-empty string array".to_string()
        })?;
    if witnesses.is_empty() {
        return Err("counterexample.json: missing witness".into());
    }
    Ok(())
}

fn validate_portfolio(
    portfolio: &Value,
    counterexample_id: &str,
    route_id: &str,
) -> Result<Value, String> {
    if required_string(portfolio, "schema")? != "ce-mechanism-portfolio-v1" {
        return Err("portfolio.json: unsupported schema".into());
    }
    if required_string(portfolio, "counterexample_id")? != counterexample_id {
        return Err("portfolio.json: counterexample_id mismatch".into());
    }
    if required_string(portfolio, "status")? != "PIVOT_READY" {
        return Err("portfolio.json: status must be PIVOT_READY".into());
    }
    if required_string(portfolio, "selected_route")? != route_id {
        return Err("portfolio.json: selected_route does not match requested route".into());
    }
    let routes = portfolio["routes"]
        .as_array()
        .filter(|routes| routes.len() >= 3)
        .ok_or_else(|| {
            "portfolio.json: at least three structural routes are required".to_string()
        })?;
    let mut ids = HashSet::new();
    let mut fingerprints = HashSet::new();
    let mut selected = None;
    for route in routes {
        let id = required_string(route, "id")?;
        validate_id(id)?;
        if !ids.insert(id.to_string()) {
            return Err(format!("portfolio.json: duplicate route id {id}"));
        }
        let fingerprint = required_string(route, "structural_fingerprint")?;
        if !fingerprints.insert(fingerprint.to_string()) {
            return Err(format!(
                "portfolio.json: duplicate structural fingerprint {fingerprint}"
            ));
        }
        let change = required_string(route, "structural_change")?;
        if !["state", "interaction", "measurement", "intervention"].contains(&change) {
            return Err(format!(
                "portfolio.json: route {id} structural_change must be state, interaction, measurement, or intervention"
            ));
        }
        let changed_term = required_string(route, "changed_term")?;
        if ["threshold", "seed", "endpoint", "decoder"]
            .contains(&changed_term.trim().to_lowercase().as_str())
        {
            return Err(format!(
                "portfolio.json: route {id} changes only {changed_term}; tuning is not a structural pivot"
            ));
        }
        for field in [
            "equation",
            "prediction",
            "adverse_control",
            "data_split",
            "kill_condition",
            "dependency",
        ] {
            required_string(route, field)?;
        }
        let split = required_string(route, "data_split")?.to_lowercase();
        if split.contains("reuse_opened_confirmation")
            || split.contains("reuse opened confirmation")
        {
            return Err(format!(
                "portfolio.json: route {id} reuses opened confirmation data"
            ));
        }
        if id == route_id {
            selected = Some(route.clone());
        }
    }
    selected.ok_or_else(|| format!("portfolio.json: selected route {route_id} not found"))
}

fn active_epoch(dir: &Path) -> Result<Option<ActiveEpoch>, String> {
    let path = dir.join(ACTIVE_EPOCH);
    if !path.exists() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    if required_string(&value, "schema")? != "ce-active-epoch-v1" {
        return Err(".active-epoch: unsupported schema".into());
    }
    let counterexample_id = required_string(&value, "counterexample_id")?.to_string();
    validate_id(&counterexample_id)?;
    let route_id = value["route_id"].as_str().map(str::to_string);
    if let Some(route_id) = &route_id {
        validate_id(route_id)?;
    }
    Ok(Some(ActiveEpoch {
        counterexample_id,
        route_id,
    }))
}

fn active_epoch_problem(dir: &Path) -> Option<String> {
    let active = match active_epoch(dir) {
        Ok(Some(active)) => active,
        Ok(None) => return None,
        Err(error) => return Some(error),
    };
    let Some(route_id) = active.route_id else {
        return Some(format!(
            "counterexample {} needs a locked record, three-route mechanism portfolio, and pivot",
            active.counterexample_id
        ));
    };
    let report = dir
        .join(EPOCHS)
        .join(&active.counterexample_id)
        .join("pivots")
        .join(&route_id)
        .join("report.md");
    if !matches!(
        file_status(&report),
        FileStatus::Complete | FileStatus::Abandoned
    ) {
        return Some(format!(
            "pivot {}/{} is still IN_PROGRESS ({})",
            active.counterexample_id,
            route_id,
            report.display()
        ));
    }
    let final_text = read_text(&dir.join(FINAL)).unwrap_or_default();
    if !final_text.contains(&active.counterexample_id) || !final_text.contains(&route_id) {
        return Some(format!(
            "pivot {}/{} is terminal but not integrated into {FINAL}",
            active.counterexample_id, route_id
        ));
    }
    None
}

fn run_is_terminal(dir: &Path) -> bool {
    active_epoch_problem(dir).is_none()
        && matches!(
            file_status(&dir.join(FINAL)),
            FileStatus::Complete | FileStatus::Abandoned
        )
}

fn set_active_run(dir: &Path) -> Result<(), String> {
    let ws = dir
        .parent()
        .ok_or_else(|| format!("run has no workspace parent: {}", dir.display()))?;
    fs::write(ws.join(ACTIVE), dir.to_string_lossy().as_bytes()).map_err(|e| e.to_string())
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > 80
        || !id
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
        || id.starts_with('-')
        || id.ends_with('-')
    {
        return Err(format!(
            "invalid id `{id}`: use 1-80 lowercase ASCII letters, digits, and interior hyphens"
        ));
    }
    Ok(())
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value[field]
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing non-empty string field `{field}`"))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = read_text(path).ok_or_else(|| format!("missing JSON: {}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    text.push('\n');
    fs::write(path, text).map_err(|e| e.to_string())
}

// ---------- status ----------

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
    let log = fs::read_to_string(dir.join("revisions/log")).unwrap_or_default();
    let mut roles: Vec<&str> = log
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
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
    match active_epoch(&dir) {
        Ok(Some(epoch)) => println!(
            "Epoch: {}/{}",
            epoch.counterexample_id,
            epoch.route_id.as_deref().unwrap_or("PORTFOLIO_REQUIRED")
        ),
        Ok(None) => println!("Epoch: none"),
        Err(error) => println!("Epoch: INVALID ({error})"),
    }
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
        if file_status(&dir.join(FINAL)) == FileStatus::Complete {
            validate_final_handoff(dir, &mut problems);
        }
        for stray in stray_root_md(dir) {
            problems.push(format!("{stray}: stray root file, move into artifacts/"));
        }
        if let Some(problem) = active_epoch_problem(dir) {
            problems.push(format!("active epoch: {problem}"));
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
            let _ = fs::remove_file(dir.join(ACTIVE_EPOCH));
        }
        println!("OK {stage}");
        Ok(())
    } else {
        Err(format!("incomplete {stage}:\n  {}", problems.join("\n  ")))
    }
}

// The workspace keeps a compact handoff, while a narrative manuscript is an
// ordered chapter assembly under docs/. This gate prevents run folders from
// accumulating duplicate papers and verifies the assembled manuscript rather
// than forcing every chapter into one oversized file.
fn validate_final_handoff(dir: &Path, problems: &mut Vec<String>) {
    let handoff = dir.join(FINAL);
    let Some(text) = read_text(&handoff) else {
        return;
    };
    let handoff_chars = text.chars().filter(|c| !c.is_whitespace()).count();
    if handoff_chars > 8_000 {
        problems.push(format!(
            "{FINAL}: run 인계 기록이 너무 깁니다({handoff_chars}자). 논문 서사는 docs/ 조립 논문의 해당 장으로 옮기십시오"
        ));
    }

    let pointers: Vec<String> = text
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("DOCS_PAPER:")
                .map(|value| value.trim().trim_matches('`').to_string())
                .filter(|value| !value.is_empty())
        })
        .collect();
    if pointers.len() != 1 {
        problems.push(format!(
            "{FINAL}: `DOCS_PAPER: docs/<분야>/<논문>/00_논문목차.md` 연결은 정확히 하나여야 합니다"
        ));
        return;
    }
    let relative = &pointers[0];
    let relative_path = Path::new(&relative);
    let components: Vec<_> = relative_path.components().collect();
    if relative_path.is_absolute()
        || components.len() != 4
        || components[0].as_os_str() != "docs"
        || !components
            .iter()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
        || relative_path.file_name().and_then(|value| value.to_str()) != Some("00_논문목차.md")
    {
        problems.push(format!(
            "{FINAL}: DOCS_PAPER는 docs/ 내부 논문 폴더의 00_ 목차 Markdown이어야 합니다: {relative}"
        ));
        return;
    }
    let Some(repo_root) = dir
        .ancestors()
        .find(|ancestor| ancestor.join("docs").is_dir())
    else {
        problems.push(format!("{FINAL}: 저장소 docs/ 루트를 찾지 못했습니다"));
        return;
    };
    let paper = repo_root.join(relative_path);
    if !paper.is_file() {
        problems.push(format!(
            "{FINAL}: 연결한 docs 정본이 없습니다: {}",
            paper.display()
        ));
        return;
    }
    let docs_root = repo_root.join("docs");
    match (fs::canonicalize(&docs_root), fs::canonicalize(&paper)) {
        (Ok(canonical_docs), Ok(canonical_paper))
            if canonical_paper.starts_with(&canonical_docs) => {}
        (Ok(_), Ok(_)) => {
            problems.push(format!(
                "{FINAL}: DOCS_PAPER가 docs/ 밖의 실제 경로를 가리킵니다: {relative}"
            ));
            return;
        }
        _ => {
            problems.push(format!(
                "{FINAL}: DOCS_PAPER 실제 경로를 확인하지 못했습니다: {relative}"
            ));
            return;
        }
    }
    validate_paper_assembly(&paper, problems);
}

fn validate_paper_assembly(entry: &Path, problems: &mut Vec<String>) {
    let Some(entry_text) = read_text(entry) else {
        return;
    };
    let links = assembly_chapter_links(&entry_text);
    if links.len() < 2 {
        problems.push(format!(
            "{}: `## 논문 조립 순서`에 장 Markdown 링크가 둘 이상 필요합니다",
            entry.display()
        ));
    }

    let Some(parent) = entry.parent() else {
        problems.push(format!("{}: 논문 폴더가 없습니다", entry.display()));
        return;
    };
    let Ok(canonical_parent) = fs::canonicalize(parent) else {
        problems.push(format!(
            "{}: 논문 폴더의 실제 경로를 확인하지 못했습니다",
            entry.display()
        ));
        return;
    };
    let mut chapters_text = String::new();
    let mut seen_paths = HashSet::new();
    let mut seen_bodies = HashSet::new();
    let mut expected_order = 1_u32;

    for (list_order, link) in links {
        let target = link
            .split('#')
            .next()
            .unwrap_or_default()
            .trim()
            .trim_matches(|c| c == '<' || c == '>');
        let target_path = Path::new(target);
        let target_components: Vec<_> = target_path.components().collect();
        let valid_local = !target_path.is_absolute()
            && target_components.len() == 1
            && matches!(target_components[0], std::path::Component::Normal(_))
            && target_path.extension().and_then(|value| value.to_str()) == Some("md")
            && target_path.file_name() != entry.file_name();
        if !valid_local {
            problems.push(format!(
                "{}: 장 링크는 같은 논문 폴더의 Markdown이어야 합니다: {target}",
                entry.display()
            ));
            continue;
        }
        let target_name = target_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string();
        if !seen_paths.insert(target_name.clone()) {
            problems.push(format!("{}: 중복 장 링크입니다: {target}", entry.display()));
            continue;
        }
        let Some(order) = chapter_order(target) else {
            problems.push(format!(
                "{}: 장 파일은 `01_...md` 같은 두 자리 순번이 필요합니다: {target}",
                entry.display()
            ));
            continue;
        };
        if list_order != expected_order || order != expected_order {
            problems.push(format!(
                "{}: 목차와 장 파일 순번은 01부터 빠짐없이 일치해야 합니다: 목록 {list_order}, 파일 {target}", entry.display()
            ));
        }
        expected_order += 1;

        let chapter = parent.join(target_path);
        let Some(chapter_text) = read_text(&chapter) else {
            problems.push(format!(
                "{}: 연결한 장 파일이 없습니다: {target}",
                entry.display()
            ));
            continue;
        };
        match fs::canonicalize(&chapter) {
            Ok(actual) if actual.starts_with(&canonical_parent) => {}
            Ok(_) => {
                problems.push(format!(
                    "{}: 장 링크가 논문 폴더 밖을 가리킵니다: {target}",
                    entry.display()
                ));
                continue;
            }
            Err(_) => {
                problems.push(format!(
                    "{}: 장의 실제 경로를 확인하지 못했습니다: {target}",
                    entry.display()
                ));
                continue;
            }
        }
        let chapter_chars = chapter_text.chars().filter(|c| !c.is_whitespace()).count();
        if chapter_chars < 200 {
            problems.push(format!(
                "{}: 장이 너무 짧습니다(공백 제외 {chapter_chars}자)",
                chapter.display()
            ));
        }
        let normalized = chapter_text.trim().to_string();
        if !seen_bodies.insert(normalized) {
            problems.push(format!(
                "{}: 다른 장과 본문이 완전히 중복됩니다",
                chapter.display()
            ));
        }
        chapters_text.push('\n');
        chapters_text.push_str(&chapter_text);
    }

    if let Ok(entries) = fs::read_dir(parent) {
        for file in entries.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if file.path().extension().and_then(|value| value.to_str()) == Some("md")
                && chapter_order(&name).is_some()
                && !seen_paths.contains(&name)
            {
                problems.push(format!(
                    "{}: 목차에 연결되지 않은 번호 장 파일입니다: {name}",
                    entry.display()
                ));
            }
        }
    }

    validate_paper_text(entry, &entry_text, &chapters_text, problems);
}

fn assembly_chapter_links(text: &str) -> Vec<(u32, String)> {
    let mut in_order = false;
    let mut links = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let heading = trimmed.trim_start_matches('#').trim().to_lowercase();
            if heading.contains("논문 조립 순서") || heading.contains("paper assembly order")
            {
                in_order = true;
                continue;
            }
            if in_order {
                break;
            }
        }
        if !in_order {
            continue;
        }
        let Some(dot) = trimmed.find('.') else {
            continue;
        };
        let Ok(order) = trimmed[..dot].parse::<u32>() else {
            continue;
        };
        let item = trimmed[dot + 1..].trim_start();
        if !item.starts_with('[') {
            continue;
        }
        let Some(open) = item.find("](") else {
            continue;
        };
        let after = &item[open + 2..];
        if let Some(close) = after.find(')') {
            links.push((order, after[..close].trim().to_string()));
        }
    }
    links
}

fn chapter_order(target: &str) -> Option<u32> {
    let name = Path::new(target).file_name()?.to_str()?;
    let prefix = name.split('_').next()?;
    if prefix.len() != 2 || !prefix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let order = prefix.parse().ok()?;
    (order > 0).then_some(order)
}

fn validate_paper_text(
    path: &Path,
    entry_text: &str,
    chapters_text: &str,
    problems: &mut Vec<String>,
) {
    let compact_chars = entry_text
        .chars()
        .chain(chapters_text.chars())
        .filter(|c| !c.is_whitespace())
        .count();
    if compact_chars < 1_200 {
        problems.push(format!(
            "{}: 조립 논문이 너무 짧습니다(공백 제외 {compact_chars}자)",
            path.display()
        ));
    }

    if !has_heading(entry_text, &["초록", "abstract"]) {
        problems.push(format!(
            "{}: 목차 입구에 초록 절이 없습니다",
            path.display()
        ));
    }

    let required = [
        (
            "introduction/background/question",
            &["서론", "배경", "연구 질문", "introduction", "background"][..],
        ),
        (
            "methods/data/definitions",
            &[
                "방법",
                "자료",
                "데이터",
                "전처리",
                "측정",
                "정의",
                "method",
                "material",
                "data",
                "definition",
            ][..],
        ),
        (
            "results/derivation/theorem",
            &[
                "결과",
                "산출",
                "유도",
                "정리",
                "증명",
                "result",
                "derivation",
                "theorem",
            ][..],
        ),
        (
            "discussion/interpretation",
            &["논의", "해석", "discussion", "interpretation"][..],
        ),
        (
            "limitations/falsifier/open problems",
            &[
                "한계",
                "미완성",
                "반증",
                "남은 문제",
                "limitation",
                "falsif",
                "open problem",
            ][..],
        ),
        (
            "reproducibility/data-and-code availability",
            &[
                "재현",
                "자료와 코드",
                "코드 이용",
                "reproduc",
                "availability",
            ][..],
        ),
        (
            "references",
            &["참고문헌", "참조", "references", "bibliography"][..],
        ),
    ];
    for (label, aliases) in required {
        if !has_heading(chapters_text, aliases) {
            problems.push(format!(
                "{}: 조립 논문 필수 절이 없습니다: {label}",
                path.display()
            ));
        }
    }
}

fn has_heading(text: &str, aliases: &[&str]) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        if !trimmed.starts_with('#') {
            return false;
        }
        let heading = trimmed.trim_start_matches('#').trim().to_lowercase();
        aliases.iter().any(|alias| heading.contains(alias))
    })
}

// The run root holds the eight core numbered stage files and, when empirical
// results need a frozen narrative source, one optional result ledger.
// Everything else belongs in artifacts/.
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
                && name != RESULT_LEDGER
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
        ["PASS", "REVISE", "PIVOT", "BLOCKED"]
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
            "{role}의 국소 수리 한도({REVISION_LIMIT}회)를 소진했습니다. 주장 축소를 해결로 세지 마십시오. \
             T 잔차나 검증된 반례라면 `counterexample <run-dir> <cex-id>`로 실패식을 잠그고, \
             구조적으로 다른 기전 3개를 만든 뒤 같은 CE_RUN에서 `pivot`하십시오. `BLOCKED`는 \
             명시적 no-go, 필수 외부 자료 부재, 또는 기전 포트폴리오 전체 소진에만 씁니다."
        ));
    }
    fs::create_dir_all(dir.join("revisions")).map_err(|e| e.to_string())?;
    fs::write(&log, format!("{text}{role}\n")).map_err(|e| e.to_string())?;
    println!("국소 수리 기록: {role} {}/{REVISION_LIMIT}", used + 1);
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
        // Runs whose artifacts freeze absolute/relative paths (sha-locked
        // prereg chains) must never be moved: a `.pin` marker keeps them live.
        if path.join(".pin").exists() {
            println!("PINNED {name} (frozen-path run; not archived — see .pin)");
            continue;
        }
        if path.join(ACTIVE_EPOCH).exists() {
            println!("ACTIVE_EPOCH {name} (continue the pivot in place; not archived)");
            continue;
        }
        if has_external_reference(ws, &path, &name) {
            println!("REFERENCED {name} (inbound provenance link; not archived)");
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
    if !archived.is_empty() {
        let index_path = archive.join("INDEX.tsv");
        let mut index =
            fs::read_to_string(&index_path).unwrap_or_else(|_| "run\tarchived_by\n".to_string());
        for name in &archived {
            index.push_str(&format!("{name}\tce-research-core-gc\n"));
        }
        fs::write(index_path, index).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn has_external_reference(ws: &Path, candidate: &Path, name: &str) -> bool {
    let scan_root = if ws.file_name().is_some_and(|part| part == "ce")
        && ws
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|part| part == "_workspace")
    {
        ws.parent().and_then(Path::parent).unwrap_or(ws)
    } else {
        ws
    };
    tree_contains_reference(scan_root, candidate, name)
}

fn tree_contains_reference(root: &Path, candidate: &Path, needle: &str) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == candidate || path.starts_with(candidate) {
            continue;
        }
        if path.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if [
                ".git",
                "_archive",
                "target",
                ".ce-research-target",
                "node_modules",
            ]
            .contains(&name.as_ref())
            {
                continue;
            }
            if tree_contains_reference(&path, candidate, needle) {
                return true;
            }
            continue;
        }
        let extension = path
            .extension()
            .and_then(|part| part.to_str())
            .unwrap_or("");
        if ![
            "md", "json", "txt", "toml", "yaml", "yml", "py", "rs", "sh", "ps1", "cmd",
        ]
        .contains(&extension)
        {
            continue;
        }
        if fs::metadata(&path).is_ok_and(|metadata| metadata.len() <= 5_000_000)
            && fs::read_to_string(&path).is_ok_and(|text| text.contains(needle))
        {
            return true;
        }
    }
    false
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

    fn valid_handoff(paper_name: &str, integration: &str) -> String {
        format!(
            "# 연구 run 인계\n\nStatus: COMPLETE\n\n\
             DOCS_PAPER: docs/뇌/{paper_name}/00_논문목차.md\n\n## 결론\n계산·감사 결과를 정본에 반영했다.\n\n\
             ## epoch 통합\n{integration}\n"
        )
    }

    fn write_docs_paper(root: &Path, paper_name: &str) {
        let paper = root.join("docs/뇌").join(paper_name);
        fs::create_dir_all(&paper).unwrap();
        let detail = "연구 질문과 근거, 계산 절차, 수치 해석, 주장 경계를 독자가 독립적으로 따라갈 수 있게 설명한다. ".repeat(8);
        fs::write(
            paper.join("00_논문목차.md"),
            format!(
                "# 상세 연구 논문\n\nStatus: COMPLETE\n\n## 초록\n{detail}\n\n\
                 ## 논문 조립 순서\n\n1. [질문과 방법](01_질문과방법.md)\n\
                 2. [결과와 논의](02_결과와논의.md)\n\
                 3. [한계와 재현](03_한계와재현.md)\n\n참고 설명의 [외부 출처](https://example.com)는 장 링크가 아니다.\n"
            ),
        )
        .unwrap();
        fs::write(
            paper.join("01_질문과방법.md"),
            format!("# 질문과 방법\n\n## 서론과 연구 질문\n{detail}\n\n## 자료와 방법\n{detail}\n"),
        )
        .unwrap();
        fs::write(
            paper.join("02_결과와논의.md"),
            format!("# 결과와 논의\n\n## 결과\n{detail}\n\n## 논의와 해석\n{detail}\n"),
        )
        .unwrap();
        fs::write(
            paper.join("03_한계와재현.md"),
            format!("# 한계와 재현\n\n## 한계와 반증 조건\n{detail}\n\n## 재현성\n{detail}\n\n## 참고문헌\n{detail}\n"),
        )
        .unwrap();
    }

    fn locked_counterexample(id: &str) -> Value {
        json!({
            "schema": "ce-counterexample-v1",
            "id": id,
            "status": "COUNTEREXAMPLE_LOCKED",
            "kind": "EMPIRICAL_CONTRADICTION",
            "parent_claim": "the effect is reference-invariant",
            "witnesses": ["35-result-ledger.md#reference-sensitivity"],
            "violated_assumption": "clinical and local-bipolar readouts identify the same local source",
            "preserved_result": "the clinical endpoint remains a descriptive observation",
            "claim_disposition": "SUSPENDED"
        })
    }

    fn mechanism_route(id: &str, fingerprint: &str, change: &str, changed_term: &str) -> Value {
        json!({
            "id": id,
            "structural_fingerprint": fingerprint,
            "structural_change": change,
            "changed_term": changed_term,
            "equation": "y(t)=H[x(t),u(t)]",
            "prediction": "a held-out participant shows the preregistered directional response",
            "adverse_control": "phase permutation and wrong-contact control",
            "data_split": "development participants plus sealed held-out participant",
            "kill_condition": "the matched adverse control performs equally well",
            "dependency": "source event metadata"
        })
    }

    fn ready_portfolio(counterexample_id: &str, selected: &str) -> Value {
        json!({
            "schema": "ce-mechanism-portfolio-v1",
            "counterexample_id": counterexample_id,
            "status": "PIVOT_READY",
            "selected_route": selected,
            "routes": [
                mechanism_route(selected, "phase-dose-response-v1", "interaction", "phase-coupling term"),
                mechanism_route("directed-transfer", "directed-kernel-v1", "state", "directed transfer kernel"),
                mechanism_route("montage-decomposition", "montage-physics-v1", "measurement", "reference observation operator")
            ]
        })
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
        write_docs_paper(&ws, "paper");
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
        fs::write(dir.join(FINAL), valid_handoff("paper", "통합 완료")).unwrap();
        fs::write(
            dir.join(RESULT_LEDGER),
            "# Frozen result ledger\n\nStatus: FROZEN\n",
        )
        .unwrap();
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
        write_docs_paper(&ws, "negative-paper");
        fs::write(ws.join(ACTIVE), "run-blocked").unwrap();
        fs::write(dir.join(CONTRACT), "Status: COMPLETE\n").unwrap();
        for name in LANES {
            fs::write(dir.join(name), "Status: COMPLETE\n").unwrap();
        }
        fs::write(dir.join(AUDIT), "Status: COMPLETE\nGate: BLOCKED\n").unwrap();
        fs::write(
            dir.join(FINAL),
            valid_handoff("negative-paper", "음성 결과 통합 완료"),
        )
        .unwrap();
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
    fn final_requires_ordered_chapters_under_docs() {
        let root = tmp("final-paper");
        let dir = root.join("run");
        fs::create_dir_all(&dir).unwrap();
        let report = dir.join(FINAL);
        fs::write(&report, "# 결과\n\nStatus: COMPLETE\n").unwrap();
        let mut problems = Vec::new();
        validate_final_handoff(&dir, &mut problems);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("DOCS_PAPER")),
            "인계 기록은 docs 정본을 가리켜야 한다"
        );

        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(root.join("docs/paper.md"), "# 단일 파일 논문\n").unwrap();
        fs::write(
            &report,
            "# 인계\n\nStatus: COMPLETE\n\nDOCS_PAPER: docs/paper.md\n",
        )
        .unwrap();
        problems.clear();
        validate_final_handoff(&dir, &mut problems);
        assert!(
            problems.iter().any(|problem| problem.contains("00_ 목차")),
            "단일 루트 논문 대신 장 조립 목차를 요구해야 한다"
        );

        fs::create_dir_all(root.join("docs/뇌/bad-paper")).unwrap();
        fs::write(
            root.join("docs/뇌/bad-paper/00_논문목차.md"),
            "# 잘못된 목차\n\n## 초록\n요약\n\n## 논문 조립 순서\n\n1. [밖의 장](../01_밖.md)\n2. [없는 장](02_없음.md)\n",
        )
        .unwrap();
        fs::write(
            &report,
            "# 인계\n\nStatus: COMPLETE\n\nDOCS_PAPER: docs/뇌/bad-paper/00_논문목차.md\n",
        )
        .unwrap();
        problems.clear();
        validate_final_handoff(&dir, &mut problems);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("같은 논문 폴더")),
            "목차 폴더 밖의 장 링크를 거부해야 한다"
        );

        write_docs_paper(&root, "paper");
        fs::write(&report, valid_handoff("paper", "통합 완료")).unwrap();
        problems.clear();
        validate_final_handoff(&dir, &mut problems);
        assert!(problems.is_empty(), "유효한 docs 조립 논문: {problems:?}");

        fs::write(
            root.join("docs/뇌/paper/04_남은사본.md"),
            "# 목차에 없는 오래된 장\n\n이 파일은 연결되지 않은 사본이다. ".repeat(20),
        )
        .unwrap();
        problems.clear();
        validate_final_handoff(&dir, &mut problems);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("연결되지 않은 번호 장")),
            "목차에 없는 번호 장 사본을 거부해야 한다"
        );
        fs::remove_file(root.join("docs/뇌/paper/04_남은사본.md")).unwrap();

        fs::write(
            &report,
            format!(
                "{}\nDOCS_PAPER: docs/뇌/paper/00_논문목차.md\n",
                valid_handoff("paper", "통합 완료")
            ),
        )
        .unwrap();
        problems.clear();
        validate_final_handoff(&dir, &mut problems);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("정확히 하나")),
            "상충하는 DOCS_PAPER 포인터를 거부해야 한다"
        );
    }

    #[test]
    fn init_reuses_unfinished_run_before_creating_another() {
        let ws = tmp("init-reuse");
        let existing = ws.join("existing-run");
        let other_existing = ws.join("other-existing-run");
        let requested = ws.join("new-run");
        fs::create_dir_all(&existing).unwrap();
        fs::create_dir_all(&other_existing).unwrap();
        fs::write(existing.join(CONTRACT), "Status: IN_PROGRESS\n").unwrap();
        fs::write(other_existing.join(CONTRACT), "Status: IN_PROGRESS\n").unwrap();
        fs::write(ws.join(ACTIVE), existing.to_string_lossy().as_bytes()).unwrap();

        let err = init(&other_existing, false).unwrap_err();
        assert!(err.contains("REUSE_REQUIRED"));
        let err = init(&requested, false).unwrap_err();
        assert!(err.contains("REUSE_REQUIRED"));
        assert!(!requested.exists(), "reuse gate must run before mkdir");
        assert!(
            fs::read_to_string(ws.join(ACTIVE))
                .unwrap()
                .contains("existing-run"),
            "blocked init must preserve the active pointer"
        );

        assert!(
            init(&existing, false).is_ok(),
            "resuming the same run is allowed"
        );
        assert!(
            init(&requested, true).is_ok(),
            "explicit new contract is allowed"
        );
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
        assert!(revise(&dir, "math-verifier").is_ok());
        let err = revise(&dir, "math-verifier").unwrap_err();
        assert!(err.contains("counterexample"));
        assert!(err.contains("같은 CE_RUN"));
        assert!(revise(&dir, "impl-engineer").is_ok());
    }

    #[test]
    fn counterexample_and_pivot_stay_in_one_run() {
        let ws = tmp("pivot");
        let dir = ws.join("durable-program");
        fs::create_dir_all(&dir).unwrap();
        let counterexample_id = "reference-sensitivity";
        let selected = "phase-dose-response";

        counterexample(&dir, counterexample_id).unwrap();
        let top_level_dirs = fs::read_dir(&ws)
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .count();
        assert_eq!(
            top_level_dirs, 1,
            "counterexample must not create a sibling run"
        );
        assert!(active_epoch_problem(&dir).unwrap().contains("three-route"));
        assert!(pivot(&dir, counterexample_id, selected).is_err());

        let epoch = dir.join(EPOCHS).join(counterexample_id);
        write_json(
            &epoch.join(COUNTEREXAMPLE),
            &locked_counterexample(counterexample_id),
        )
        .unwrap();
        let mut invalid = ready_portfolio(counterexample_id, selected);
        invalid["routes"][0]["structural_change"] = Value::String("tuning".into());
        invalid["routes"][0]["changed_term"] = Value::String("endpoint".into());
        write_json(&epoch.join(PORTFOLIO), &invalid).unwrap();
        let err = pivot(&dir, counterexample_id, selected).unwrap_err();
        assert!(err.contains("structural_change") || err.contains("tuning"));

        write_json(
            &epoch.join(PORTFOLIO),
            &ready_portfolio(counterexample_id, selected),
        )
        .unwrap();
        pivot(&dir, counterexample_id, selected).unwrap();
        let route = epoch.join("pivots").join(selected);
        assert!(route.join("contract.md").exists());
        assert!(route.join("report.md").exists());
        assert!(active_epoch_problem(&dir).unwrap().contains("IN_PROGRESS"));

        fs::write(route.join("report.md"), "Status: COMPLETE\n").unwrap();
        write_docs_paper(&ws, "pivot-paper");
        fs::write(
            dir.join(FINAL),
            valid_handoff(
                "pivot-paper",
                &format!("{counterexample_id}\n{selected}\n통합 완료"),
            ),
        )
        .unwrap();
        assert!(active_epoch_problem(&dir).is_none());
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
        assert!(ws.join("_archive/INDEX.tsv").exists());
        assert!(open.exists());
        assert!(!ws.join(ACTIVE).exists());
    }

    #[test]
    fn gc_preserves_runs_with_inbound_references() {
        let ws = tmp("gc-referenced");
        let done = ws.join("run-done");
        let referrer = ws.join("run-referrer");
        fs::create_dir_all(&done).unwrap();
        fs::create_dir_all(&referrer).unwrap();
        fs::write(done.join(FINAL), "Status: COMPLETE\n").unwrap();
        fs::write(
            referrer.join(CONTRACT),
            "Status: IN_PROGRESS\nPREDECESSOR: run-done\n",
        )
        .unwrap();
        gc(&ws).unwrap();
        assert!(
            done.exists(),
            "referenced evidence must stay at its frozen path"
        );
        assert!(!ws.join("_archive/run-done").exists());
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
