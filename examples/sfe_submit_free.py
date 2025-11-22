from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile

API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

print("=== SFE Quantum Hardware Execution (Free Plan) ===", flush=True)

# 1. 연결
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)
backend = service.least_busy(operational=True, simulator=False)
print(f"Backend: {backend.name}", flush=True)

# 2. SFE 회로
qc = QuantumCircuit(1, 1)
qc.h(0)
sfe_timings = [0.12, 0.35, 0.58, 0.82, 0.95]
for t in sfe_timings:
    qc.delay(int(t * 100), 0)
    qc.x(0)
qc.measure(0, 0)

# 3. 트랜스파일 & 제출 (무료 플랜 방식)
print("Transpiling...", flush=True)
transpiled = transpile(qc, backend, optimization_level=1)

print("Submitting job...", flush=True)
job = backend.run(transpiled, shots=1024)

print(f"\n*** JOB SUBMITTED ***", flush=True)
print(f"Job ID: {job.job_id()}", flush=True)
print(f"URL: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)

