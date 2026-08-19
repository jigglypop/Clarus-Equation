use nrm3d_core::{config_hash, gate_a_fixtures};
use serde::Deserialize;
use std::{
    env, fs,
    path::{Component, Path, PathBuf},
};

const SCHEMA: &str = "nrm3d-gate-a-r6-release-final6-v1";
const ARTIFACT: &str = "artifacts/gate-a-fixtures-r6-release-final6.json";
const MANIFEST: &str = "artifacts/gate-a-fixtures-r6-release-final6.json.manifest.json";
const ORACLE: &str = "artifacts/oracle-r6-release-final6.json";
const PRODUCER: &str = "artifacts/bin/nrm3d-core-gate-a-r6-release-final6.exe";
const VALIDATOR: &str = "artifacts/bin/validate-gate-a-r6-release-final6.exe";
const REQUIRED_FILES: &[&str] = &[
    "00-contract.md",
    "11-math.md",
    "12-routes.md",
    "20-audit.md",
    "artifacts/rust/nrm3d-core/Cargo.toml",
    "artifacts/rust/nrm3d-core/Cargo.lock",
    "artifacts/rust/nrm3d-core/src/lib.rs",
    "artifacts/rust/nrm3d-core/src/main.rs",
    "artifacts/rust/nrm3d-core/src/bin/validate_gate_a.rs",
    "artifacts/reference_oracle.py",
    ORACLE,
    "artifacts/gate-a-CURRENT-v1.json",
    PRODUCER,
    VALIDATOR,
];
const SUPERSEDES: &[&str] = &[
    "gate-a-fixtures.json",
    "gate-a-fixtures-r1.json",
    "gate-a-fixtures-r2.json",
    "gate-a-fixtures-r3.json",
    "gate-a-fixtures-r4.json",
    "gate-a-fixtures-r4.manifest.json",
    "gate-a-fixtures-r4-final.json",
    "gate-a-fixtures-r4-final.manifest.json",
    "gate-a-fixtures-r4-final2.json",
    "gate-a-fixtures-r4-final2.manifest.json",
    "gate-a-fixtures-r5.json",
    "gate-a-fixtures-r5.manifest.json",
    "gate-a-fixtures-r5-final2.json",
    "gate-a-fixtures-r5-final2.manifest.json",
    "oracle-r4.json",
    "oracle-r4-final.json",
    "oracle-r5.json",
    "oracle-r5-final2.json",
    "gate-a-fixtures-r4-SUPERSEDED.md",
    "gate-a-r5-SUPERSEDED.md",
    "oracle-r6.json",
    "gate-a-fixtures-r6-debug.json",
    "gate-a-fixtures-r6-debug.json.manifest.json",
    "gate-a-fixtures-r6-debug-final2.json",
    "gate-a-fixtures-r6-debug-final2.json.manifest.json",
    "gate-a-fixtures-r6-debug-final3.json",
    "gate-a-fixtures-r6-debug-final3.json.manifest.json",
    "gate-a-fixtures-r6-release-final4.json",
    "gate-a-fixtures-r6-release-final4.json.manifest.json",
    "bin/nrm3d-core-gate-a-r6.exe",
    "bin/validate-gate-a-r6.exe",
    "bin/nrm3d-core-gate-a-r6-final2.exe",
    "bin/validate-gate-a-r6-final2.exe",
    "bin/nrm3d-core-gate-a-r6-debug.exe",
    "bin/validate-gate-a-r6-debug.exe",
    "bin/nrm3d-core-gate-a-r6-debug-final2.exe",
    "bin/validate-gate-a-r6-debug-final2.exe",
    "bin/nrm3d-core-gate-a-r6-debug-final3.exe",
    "bin/validate-gate-a-r6-debug-final3.exe",
    "bin/nrm3d-core-gate-a-r6-release-final3.exe",
    "bin/validate-gate-a-r6-release-final3.exe",
    "bin/nrm3d-core-gate-a-r6-release-final4.exe",
    "bin/validate-gate-a-r6-release-final4.exe",
    "gate-a-CURRENT-v1.json",
    "oracle-r6-release-final5.json",
    "gate-a-fixtures-r6-release-final5.json",
    "gate-a-fixtures-r6-release-final5.json.manifest.json",
    "bin/nrm3d-core-gate-a-r6-release-final5.exe",
    "bin/validate-gate-a-r6-release-final5.exe",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    oracle: String,
    symmetric_exp_log_relative: f64,
    curved_origin_scalar: f64,
    curved_origin_abs: f64,
    atol: f64,
    rtol: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleScalar {
    field: String,
    rust: f64,
    reference: f64,
    absolute_residual: f64,
    tolerance: f64,
    pass: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    value: f64,
    threshold: f64,
    pass: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema: String,
    scope: String,
    config_hash: String,
    oracle_path: String,
    oracle_blake3: String,
    oracle_scalars: Vec<OracleScalar>,
    fixture_count: usize,
    fixtures: Vec<Fixture>,
    all_pass: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    path: String,
    bytes: u64,
    blake3: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    artifact_path: String,
    payload_blake3: String,
    files: Vec<Record>,
    compiler_vv: String,
    host_target: String,
    release_profile: String,
    build_command: Vec<String>,
    producer_command: Vec<String>,
    validator_command: Vec<String>,
    threads: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Current {
    schema: String,
    current_artifact: String,
    current_manifest: String,
    execution_profile: String,
    release_execution: String,
    authorization: String,
    status: String,
    supersedes: Vec<String>,
}

fn fail<T>(message: &str) -> Result<T, Box<dyn std::error::Error>> {
    Err(message.into())
}
fn b3(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(blake3::hash(&fs::read(path)?).to_hex().to_string())
}
fn finite(value: f64) -> bool {
    value.is_finite()
}
fn root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?)
}
fn resolve(run: &Path, raw: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = PathBuf::from(raw);
    if path.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return fail("path traversal");
    };
    let joined = run.join(path);
    let parent = joined.parent().ok_or("path parent")?.canonicalize()?;
    if !parent.starts_with(run) {
        return fail("reparse escape");
    };
    Ok(parent.join(joined.file_name().ok_or("path name")?))
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14_f64.max(b.abs() * 1e-12)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != ARTIFACT || args[2] != MANIFEST {
        return fail("usage: validate-gate-a <fixed artifact> <fixed manifest>");
    }
    let run = root()?;
    let artifact_path = resolve(&run, &args[1])?;
    let manifest_path = resolve(&run, &args[2])?;
    let artifact_bytes = fs::read(&artifact_path)?;
    let artifact: Artifact = serde_json::from_slice(&artifact_bytes)?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if artifact.schema != SCHEMA
        || manifest.schema != SCHEMA
        || artifact.scope != "Gate A numerical fixtures only; scientific outcomes refused"
        || artifact.config_hash != config_hash()
    {
        return fail("artifact identity mismatch");
    }
    if manifest.artifact_path != ARTIFACT
        || manifest.payload_blake3 != blake3::hash(&artifact_bytes).to_hex().to_string()
    {
        return fail("payload declaration mismatch");
    }
    let expected = gate_a_fixtures()?;
    if artifact.fixture_count != expected.len() || artifact.fixtures.len() != expected.len() {
        return fail("fixture universe count mismatch");
    }
    for (actual, expected) in artifact.fixtures.iter().zip(expected.iter()) {
        if actual.name != expected.name
            || !finite(actual.value)
            || !finite(actual.threshold)
            || !close(actual.value, expected.value)
            || actual.threshold != expected.threshold
            || actual.pass != expected.pass
        {
            return fail("fixture fresh recomputation mismatch");
        }
    }
    if !artifact.all_pass || !artifact.fixtures.iter().all(|fixture| fixture.pass) {
        return fail("fixture pass mismatch");
    }
    if artifact.oracle_path != ORACLE {
        return fail("oracle path mismatch");
    }
    let oracle_path = resolve(&run, &artifact.oracle_path)?;
    if artifact.oracle_blake3 != b3(&oracle_path)? {
        return fail("oracle source hash mismatch");
    }
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
        return fail("oracle identity/value mismatch");
    }
    let names = ["symmetric_exp_log_relative", "curved_origin_scalar"];
    let references = [
        oracle.symmetric_exp_log_relative,
        oracle.curved_origin_scalar,
    ];
    if artifact.oracle_scalars.len() != names.len() {
        return fail("oracle scalar universe");
    }
    for ((actual, name), reference) in artifact.oracle_scalars.iter().zip(names).zip(references) {
        let residual = (actual.rust - reference).abs();
        let tolerance = oracle.atol + oracle.rtol * reference.abs();
        if actual.field != name
            || ![
                actual.rust,
                actual.reference,
                actual.absolute_residual,
                actual.tolerance,
            ]
            .into_iter()
            .all(finite)
            || !close(actual.reference, reference)
            || !close(actual.absolute_residual, residual)
            || !close(actual.tolerance, tolerance)
            || actual.pass != (residual <= tolerance)
        {
            return fail("oracle scalar residual/tolerance mismatch");
        }
    }
    if !artifact.oracle_scalars.iter().all(|scalar| scalar.pass) {
        return fail("oracle scalar pass mismatch");
    }
    if manifest.files.len() != REQUIRED_FILES.len() {
        return fail("strict manifest universe mismatch");
    }
    for (record, expected_path) in manifest.files.iter().zip(REQUIRED_FILES) {
        if record.path != *expected_path {
            return fail("manifest file order/path mismatch");
        };
        let path = resolve(&run, &record.path)?;
        if !path.is_file()
            || fs::metadata(&path)?.len() != record.bytes
            || b3(&path)? != record.blake3
        {
            return fail("manifest file record mismatch");
        }
    }
    if manifest.compiler_vv.is_empty()
        || manifest.host_target.is_empty()
        || manifest.release_profile != "EXECUTION_PROFILE_RELEASE"
        || manifest.build_command != vec!["cargo", "build", "--release", "--locked"]
        || manifest.producer_command
            != vec![
                PRODUCER.replace('/', "\\"),
                "gate-a".into(),
                ARTIFACT.into(),
                "--oracle".into(),
                ORACLE.into(),
            ]
        || manifest.validator_command
            != vec![
                String::from(VALIDATOR),
                String::from(ARTIFACT),
                String::from(MANIFEST),
            ]
        || manifest.threads == 0
    {
        return fail("provenance mismatch");
    }
    let current: Current = serde_json::from_slice(&fs::read(resolve(
        &run,
        "artifacts/gate-a-CURRENT-v1.json",
    )?)?)?;
    if current.schema != "nrm3d-gate-a-current-final6-v1"
        || current.current_artifact != ARTIFACT
        || current.current_manifest != MANIFEST
        || current.execution_profile != "EXECUTION_PROFILE_RELEASE"
        || current.release_execution != "RELEASE_EXECUTION_VERIFIED_PRINT_CONFIG_HASH"
        || current.authorization != "GATE_A_ONLY_NOT_GATE_B_UNLOCKING_PENDING_EXTERNAL_AUDIT"
        || current.status != "GATE_A_NUMERICAL_PASS_NOT_GATE_B"
        || current
            .supersedes
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != SUPERSEDES
    {
        return fail("current declaration mismatch");
    }
    println!(
        "PASS fixtures={} artifact_blake3={}",
        expected.len(),
        manifest.payload_blake3
    );
    Ok(())
}
