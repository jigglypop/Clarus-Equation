from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV1 as Sampler
from qiskit import QuantumCircuit, transpile

API_KEY = "OtWVkDQxb4Q6KCOukPKq6n61VU9Cpr09oCR4i3zhrmzm"

print("=== SFE -> IBM Quantum ===", flush=True)

# 연결
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)
backend = service.backend("ibm_fez")
print(f"Target: {backend.name}", flush=True)

# SFE 회로
qc = QuantumCircuit(1, 1)
qc.h(0)
for t in [0.12, 0.35, 0.58, 0.82, 0.95]:
    qc.delay(int(t * 100), 0)
    qc.x(0)
qc.measure(0, 0)

# 트랜스파일
print("Transpiling...", flush=True)
transpiled = transpile(qc, backend, optimization_level=1)

# 제출 (V1 Sampler)
print("Submitting...", flush=True)
sampler = Sampler(backend=backend)
job = sampler.run(transpiled, shots=1024)

print(f"\n*** JOB SUBMITTED ***", flush=True)
print(f"Job ID: {job.job_id()}", flush=True)
print(f"URL: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)

