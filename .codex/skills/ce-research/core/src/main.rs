use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

const FILES: &[(&str, &str)] = &[
    ("00-contract.md", "Research contract"),
    ("10-sources.md", "Sources"),
    ("11-math.md", "Mathematical verification"),
    ("12-routes.md", "Alternative routes"),
    ("20-audit.md", "Status audit"),
    ("30-implementation.md", "Implementation"),
    ("31-validation.md", "Validation"),
    ("40-final-report.md", "Final report"),
];

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let result = match args.as_slice() {
        [cmd, event] if cmd == "hook" => hook(event),
        [cmd, dir] if cmd == "init" => init(Path::new(dir)),
        [cmd, dir, stage] if cmd == "check" => check(Path::new(dir), stage),
        _ => Err("usage: ce-research-core hook <route|stop> | init <run-dir> | check <run-dir> <lanes|gate|build|final>".into()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn hook(event: &str) -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    match event {
        "route" => {
            let prompt = string_field(&input, "prompt")
                .unwrap_or_default()
                .to_lowercase();
            if is_ce_task(&prompt) {
                println!(
                    "{}",
                    r#"{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"Route narrowly: full CE research -> $ce-research; proof/status -> $ce-closure-gate; dimensions -> $ce-dimensionless; tests -> $ce-validate; docs -> $ce-doc-write; guard benchmarks -> $clarus-guard-bench. Load only the selected skill and keep output concise."}}"#
                );
            }
        }
        "stop" => {
            let active = bool_field(&input, "stop_hook_active");
            let message = string_field(&input, "last_assistant_message").unwrap_or_default();
            if !active && let Some(run) = marker(&message) {
                let cwd = string_field(&input, "cwd").unwrap_or_else(|| ".".into());
                let dir = if Path::new(run).is_absolute() {
                    PathBuf::from(run)
                } else {
                    Path::new(&cwd).join(run)
                };
                if !complete(&dir.join("40-final-report.md")) {
                    println!(
                        "{}",
                        r#"{"decision":"block","reason":"CE run is incomplete. Finish or mark 40-final-report.md COMPLETE, then return the same CE_RUN marker."}"#
                    );
                    return Ok(());
                }
            }
            println!("{{}}");
        }
        _ => return Err(format!("unknown hook event: {event}")),
    }
    Ok(())
}

fn is_ce_task(prompt: &str) -> bool {
    [
        "clarus",
        "clarus-eq",
        "clarus-equation",
        "ce-",
        "axium",
        "reality_stone",
        "부트스트랩",
        "경로적분",
    ]
    .iter()
    .any(|term| prompt.contains(term))
        || prompt.contains("ce의")
        || prompt
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| word == "ce")
}

fn string_field(json: &str, key: &str) -> Option<String> {
    let mut chars = json
        .get(json.find(&format!("\"{key}\""))? + key.len() + 2..)?
        .chars();
    while chars.next()? != ':' {}
    while chars.clone().next()?.is_whitespace() {
        chars.next();
    }
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                _ => return None,
            },
            _ => out.push(c),
        }
    }
    None
}

fn bool_field(json: &str, key: &str) -> bool {
    json.get(json.find(&format!("\"{key}\"")).unwrap_or(json.len()) + key.len() + 2..)
        .and_then(|tail| tail.split_once(':'))
        .is_some_and(|(_, value)| value.trim_start().starts_with("true"))
}

fn marker(message: &str) -> Option<&str> {
    message
        .lines()
        .find_map(|line| line.trim().strip_prefix("CE_RUN="))
        .filter(|s| !s.is_empty())
}

fn init(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir.join("artifacts")).map_err(|e| e.to_string())?;
    fs::create_dir_all(dir.join("revisions")).map_err(|e| e.to_string())?;
    for (name, title) in FILES {
        let path = dir.join(name);
        if !path.exists() {
            fs::write(path, format!("# {title}\n\nStatus: IN_PROGRESS\n"))
                .map_err(|e| e.to_string())?;
        }
    }
    println!("{}", dir.display());
    Ok(())
}

fn check(dir: &Path, stage: &str) -> Result<(), String> {
    let names: &[&str] = match stage {
        "lanes" => &["10-sources.md", "11-math.md", "12-routes.md"],
        "gate" => &["20-audit.md"],
        "build" => &["30-implementation.md", "31-validation.md"],
        "final" => &["40-final-report.md"],
        _ => return Err(format!("unknown stage: {stage}")),
    };
    let missing: Vec<_> = names
        .iter()
        .filter(|name| !complete(&dir.join(name)))
        .copied()
        .collect();
    if missing.is_empty() {
        println!("OK {stage}");
        Ok(())
    } else {
        Err(format!("incomplete {stage}: {}", missing.join(", ")))
    }
}

fn complete(path: &Path) -> bool {
    fs::read_to_string(path)
        .is_ok_and(|text| text.lines().any(|line| line.trim() == "Status: COMPLETE"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_routes() {
        let json = r#"{"prompt":"Clarus-EQ \uac80\uc99d","stop_hook_active":true}"#;
        assert_eq!(
            string_field(json, "prompt").as_deref(),
            Some("Clarus-EQ 검증")
        );
        assert!(bool_field(json, "stop_hook_active"));
        assert!(is_ce_task("clarus-eq research"));
        assert!(is_ce_task("ce 검증"));
    }

    #[test]
    fn finds_marker() {
        assert_eq!(
            marker("done\nCE_RUN=_workspace/ce/x"),
            Some("_workspace/ce/x")
        );
    }
}
