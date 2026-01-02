use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::thread;
use std::time::Duration;

// IBM Quantum Platform API (qiskit_ibm_runtime 방식)
const API_BASE_URL: &str = "https://api.quantum-computing.ibm.com/runtime";

#[derive(Debug, Serialize, Deserialize)]
struct JobResponse {
    id: String,
    status: String,
    #[serde(default)]
    program_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobResult {
    pub status: String,
    pub quasi_dists: Option<Vec<Value>>,
    pub counts: Option<Vec<Value>>,
}

pub struct IbmClient {
    api_key: String,
    client: Client,
}

impl IbmClient {
    pub fn new(api_key: &str) -> Self {
        // reqwest는 기본적으로 시스템 프록시를 사용함
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| Client::new());

        IbmClient {
            api_key: api_key.to_string(),
            client,
        }
    }

    /// Python의 QiskitRuntimeService처럼 인증 불필요, API Key를 Bearer Token으로 직접 사용
    pub fn authenticate(&mut self) -> Result<(), Box<dyn Error>> {
        // QiskitRuntimeService는 별도 인증 없이 바로 API Key 사용
        println!("[SFE-Rust] IBM Quantum Runtime 준비 완료 (API Key 검증 생략)");
        Ok(())
    }

    /// QASM 생성기
    pub fn generate_qasm(&self, pulse_sequence: &[f64], duration_dt: usize) -> String {
        let mut qasm =
            String::from("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nbit[1] c;\nqubit[1] q;\n");
        qasm.push_str("reset q[0];\nh q[0];\n");
        let mut last_t = 0;
        for &t in pulse_sequence {
            let current_t = (t * duration_dt as f64).round() as i32;
            let delay = current_t - last_t;
            if delay > 0 {
                qasm.push_str(&format!("delay[{delay}dt] q[0];\n"));
            }
            qasm.push_str("x q[0];\n");
            last_t = current_t;
        }
        let final_delay = duration_dt as i32 - last_t;
        if final_delay > 0 {
            qasm.push_str(&format!("delay[{final_delay}dt] q[0];\n"));
        }
        qasm.push_str("h q[0];\nmeasure q[0] -> c[0];\n");
        qasm
    }

    /// 작업 제출 (Python QiskitRuntimeService.run() 방식)
    pub fn submit_sfe_job(&mut self, pulse_sequence: &[f64]) -> Result<String, Box<dyn Error>> {
        println!("[SFE-Rust] 작업 페이로드 구성 중...");
        let durations = [0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000];
        let mut circuits = Vec::new();
        for &d in &durations {
            circuits.push(self.generate_qasm(pulse_sequence, d));
        }

        // Runtime Sampler Primitive 호출
        let url = format!("{API_BASE_URL}/jobs");
        let payload = json!({
            "program_id": "sampler",
            "params": {
                "circuits": circuits,
                "run_options": { "shots": 1024 }
            },
            "backend": "ibm_fez"
        });

        println!("[SFE-Rust] POST {url} (Backend: ibm_fez)");

        // Python QiskitRuntimeService처럼 API Key를 Authorization Header에 직접
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()?;

        if resp.status().is_success() {
            let job_data: JobResponse = resp.json()?;
            println!("[SFE-Rust] 작업 제출 성공! Job ID: {}", job_data.id);
            Ok(job_data.id)
        } else {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            Err(format!("작업 제출 실패 [{status}]: {text}").into())
        }
    }

    /// 결과 대기 및 수신 (Polling)
    pub fn wait_for_result(&mut self, job_id: &str) -> Result<JobResult, Box<dyn Error>> {
        let url = format!("{API_BASE_URL}/jobs/{job_id}");
        println!("[SFE-Rust] 결과 대기 중 (Polling Job {job_id})...");

        loop {
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()?;

            if !resp.status().is_success() {
                return Err(format!("Polling 실패: {}", resp.status()).into());
            }

            let job_status: Value = resp.json()?;
            let state = job_status["status"].as_str().unwrap_or("Unknown");

            match state {
                "Completed" => {
                    println!("\n[SFE-Rust] 작업 완료! 결과 다운로드 중...");
                    let result_url = format!("{API_BASE_URL}/jobs/{job_id}/results");
                    let res_resp = self
                        .client
                        .get(&result_url)
                        .header("Authorization", format!("Bearer {}", self.api_key))
                        .send()?;

                    let raw_result: Value = res_resp.json()?;
                    let quasi_dists = raw_result["quasi_dists"].as_array().cloned();

                    return Ok(JobResult {
                        status: "Completed".to_string(),
                        quasi_dists,
                        counts: None,
                    });
                }
                "Failed" | "Cancelled" | "Error" => {
                    return Err(format!("작업 중단됨: {state}").into());
                }
                _ => {
                    print!(".");
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                    thread::sleep(Duration::from_secs(5));
                }
            }
        }
    }
}
