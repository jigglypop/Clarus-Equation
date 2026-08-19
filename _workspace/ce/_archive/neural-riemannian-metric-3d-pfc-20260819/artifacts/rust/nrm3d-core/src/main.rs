use nrm3d_core::{FixtureReport, config_hash, gate_a_fixtures};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::Write,
    path::{Component, Path, PathBuf},
    process::Command,
};

const SCHEMA: &str = "nrm3d-gate-a-r6-release-final6-v1";
const ARTIFACT: &str = "artifacts/gate-a-fixtures-r6-release-final6.json";
const MANIFEST: &str = "artifacts/gate-a-fixtures-r6-release-final6.json.manifest.json";
const ORACLE: &str = "artifacts/oracle-r6-release-final6.json";
const PRODUCER: &str = "artifacts/bin/nrm3d-core-gate-a-r6-release-final6.exe";
const VALIDATOR: &str = "artifacts/bin/validate-gate-a-r6-release-final6.exe";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    oracle: String,
    symmetric_exp_log_relative: f64,
    curved_origin_scalar: f64,
    curved_origin_abs: f64,
    atol: f64,
    rtol: f64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleScalar {
    field: String,
    rust: f64,
    reference: f64,
    absolute_residual: f64,
    tolerance: f64,
    pass: bool,
}
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct GateArtifact {
    schema: String,
    scope: String,
    config_hash: String,
    oracle_path: String,
    oracle_blake3: String,
    oracle_scalars: Vec<OracleScalar>,
    fixture_count: usize,
    fixtures: Vec<FixtureReport>,
    all_pass: bool,
}
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct FileRecord {
    path: String,
    bytes: u64,
    blake3: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    artifact_path: String,
    payload_blake3: String,
    files: Vec<FileRecord>,
    compiler_vv: String,
    host_target: String,
    release_profile: String,
    build_command: Vec<String>,
    producer_command: Vec<String>,
    validator_command: Vec<String>,
    threads: usize,
}

fn b3(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(blake3::hash(&fs::read(path)?).to_hex().to_string())
}
fn finite(v: f64) -> bool {
    v.is_finite()
}
fn artifact_finite(v: &GateArtifact) -> bool {
    v.fixtures
        .iter()
        .all(|x| finite(x.value) && finite(x.threshold))
        && v.oracle_scalars.iter().all(|x| {
            finite(x.rust)
                && finite(x.reference)
                && finite(x.absolute_residual)
                && finite(x.tolerance)
        })
}
fn checked_json<T: Serialize>(v: &T) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let b = serde_json::to_vec_pretty(v)?;
    if b.windows(3).any(|w| w == b"NaN") || b.windows(8).any(|w| w == b"Infinity") {
        Err("refusing nonfinite JSON".into())
    } else {
        Ok(b)
    }
}
fn create_only(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn artifact_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?)
}
fn checked_under(root: &Path, raw: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let p = PathBuf::from(raw);
    if p.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err("path must be run-relative".into());
    };
    let full = root.join(p);
    let parent = full.parent().ok_or("path parent")?.canonicalize()?;
    if !parent.starts_with(root) {
        return Err("path escaped frozen artifacts root".into());
    };
    Ok(parent.join(full.file_name().ok_or("path name")?))
}
fn relative_run_path(run: &Path, path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let c = path.canonicalize()?;
    if !c.starts_with(run) {
        return Err("manifest path escaped run".into());
    };
    let r = c
        .strip_prefix(run)?
        .to_str()
        .ok_or("non UTF-8 path")?
        .replace('\\', "/");
    if r.is_empty() || r.contains("../") {
        return Err("invalid manifest path".into());
    };
    Ok(r)
}
fn record(run: &Path, path: &Path) -> Result<FileRecord, Box<dyn std::error::Error>> {
    Ok(FileRecord {
        path: relative_run_path(run, path)?,
        bytes: fs::metadata(path)?.len(),
        blake3: b3(path)?,
    })
}
fn scalar(
    field: &str,
    rust: f64,
    reference: f64,
    atol: f64,
    rtol: f64,
) -> Result<OracleScalar, Box<dyn std::error::Error>> {
    if ![rust, reference, atol, rtol].into_iter().all(finite) || atol < 0.0 || rtol < 0.0 {
        return Err("nonfinite oracle scalar".into());
    };
    let absolute_residual = (rust - reference).abs();
    let tolerance = atol + rtol * reference.abs();
    Ok(OracleScalar {
        field: field.into(),
        rust,
        reference,
        absolute_residual,
        tolerance,
        pass: absolute_residual <= tolerance,
    })
}
fn required_files(run: &Path, artifacts: &Path, oracle: &Path) -> Vec<PathBuf> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        run.join("00-contract.md"),
        run.join("11-math.md"),
        run.join("12-routes.md"),
        run.join("20-audit.md"),
        crate_root.join("Cargo.toml"),
        crate_root.join("Cargo.lock"),
        crate_root.join("src/lib.rs"),
        crate_root.join("src/main.rs"),
        crate_root.join("src/bin/validate_gate_a.rs"),
        artifacts.join("reference_oracle.py"),
        oracle.to_path_buf(),
        artifacts.join("gate-a-CURRENT-v1.json"),
        run.join(PRODUCER),
        run.join(VALIDATOR),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("print-config-hash") => println!("{}", config_hash()),
        Some("--fixtures") => println!("{}", serde_json::to_string_pretty(&gate_a_fixtures()?)?),
        Some("gate-a") => {
            if args.len() != 5 || args[2] != ARTIFACT || args[3] != "--oracle" || args[4] != ORACLE
            {
                return Err(format!("usage: gate-a {ARTIFACT} --oracle {ORACLE}").into());
            }
            let artifacts = artifact_root()?;
            let run = artifacts.parent().ok_or("run root")?.canonicalize()?;
            let output = checked_under(&run, &args[2])?;
            let oracle_path = checked_under(&run, &args[4])?;
            let oracle: Oracle = serde_json::from_slice(&fs::read(&oracle_path)?)?;
            if oracle.oracle != "numpy.linalg.eigh float64"
                || ![
                    oracle.symmetric_exp_log_relative,
                    oracle.curved_origin_scalar,
                    oracle.curved_origin_abs,
                    oracle.atol,
                    oracle.rtol,
                ]
                .into_iter()
                .all(finite)
                || oracle.atol < 0.0
                || oracle.rtol < 0.0
                || oracle.curved_origin_abs != oracle.curved_origin_scalar.abs()
            {
                return Err("invalid oracle identity or values".into());
            }
            let fixtures = gate_a_fixtures()?;
            let exp = fixtures
                .iter()
                .find(|x| x.name == "symmetric_exp_log")
                .ok_or("missing exp fixture")?
                .value;
            let curved = fixtures
                .iter()
                .find(|x| x.name == "F-CURVED_origin_scalar")
                .ok_or("missing curved fixture")?
                .value;
            let oracle_scalars = vec![
                scalar(
                    "symmetric_exp_log_relative",
                    exp,
                    oracle.symmetric_exp_log_relative,
                    oracle.atol,
                    oracle.rtol,
                )?,
                scalar(
                    "curved_origin_scalar",
                    curved,
                    oracle.curved_origin_scalar,
                    oracle.atol,
                    oracle.rtol,
                )?,
            ];
            let artifact = GateArtifact {
                schema: SCHEMA.into(),
                scope: "Gate A numerical fixtures only; scientific outcomes refused".into(),
                config_hash: config_hash(),
                oracle_path: relative_run_path(&run, &oracle_path)?,
                oracle_blake3: b3(&oracle_path)?,
                fixture_count: fixtures.len(),
                all_pass: fixtures.iter().all(|x| x.pass) && oracle_scalars.iter().all(|x| x.pass),
                fixtures,
                oracle_scalars,
            };
            if !artifact_finite(&artifact) || !artifact.all_pass {
                return Err("refusing invalid Gate A artifact".into());
            }
            let payload = checked_json(&artifact)?;
            let files = required_files(&run, &artifacts, &oracle_path);
            if files.iter().any(|p| !p.is_file()) {
                return Err("missing frozen manifest input".into());
            }
            let manifest = Manifest {
                schema: SCHEMA.into(),
                artifact_path: ARTIFACT.into(),
                payload_blake3: blake3::hash(&payload).to_hex().to_string(),
                files: files
                    .iter()
                    .map(|path| record(&run, path))
                    .collect::<Result<_, _>>()?,
                compiler_vv: String::from_utf8(Command::new("rustc").arg("-vV").output()?.stdout)?,
                host_target: env::var("HOST")
                    .unwrap_or_else(|_| format!("{}-pc-windows-msvc", env::consts::ARCH)),
                release_profile: "EXECUTION_PROFILE_RELEASE".into(),
                build_command: vec![
                    "cargo".into(),
                    "build".into(),
                    "--release".into(),
                    "--locked".into(),
                ],
                producer_command: vec![
                    PRODUCER.replace('/', "\\"),
                    "gate-a".into(),
                    ARTIFACT.into(),
                    "--oracle".into(),
                    ORACLE.into(),
                ],
                validator_command: vec![VALIDATOR.into(), ARTIFACT.into(), MANIFEST.into()],
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            };
            let manifest_path = PathBuf::from(format!("{}.manifest.json", output.display()));
            create_only(&output, &payload)?;
            create_only(&manifest_path, &checked_json(&manifest)?)?;
            println!(
                "created {} and {} fixtures={}",
                output.display(),
                manifest_path.display(),
                artifact.fixture_count
            );
        }
        Some("synthetic") | Some("pfc") | Some("outcomes") => return Err(
            "refused: scientific outcomes remain sealed pending a separate stable-code lock token"
                .into(),
        ),
        _ => return Err("usage: nrm3d-core <print-config-hash|--fixtures|gate-a ...>".into()),
    }
    Ok(())
}
