from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
import numpy as np
from examples.ibm_env import load_ibm_api_key

print("=== IBM Fez Noise Characterization (T1/T2) ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
backend = service.backend("ibm_fez")
print(f"Target: {backend.name}", flush=True)

# 2. T1/T2 회로 생성 (간단 버전)
circuits = []

# T1 측정: |1> 상태로 만들고 시간 지연 후 측정
delays = [0, 500, 1000, 1500, 2000, 2500, 3000] # dt 단위 (약 500dt ~ 100ns)
for d in delays:
    qc = QuantumCircuit(1, 1)
    qc.x(0)            # Excitation
    qc.delay(d, 0)     # Wait
    qc.measure(0, 0)   # Measure
    circuits.append(qc)

# 3. 트랜스파일
print(f"Transpiling {len(circuits)} circuits...", flush=True)
transpiled = transpile(circuits, backend, optimization_level=1)

# 4. 제출
print("Submitting batch job...", flush=True)
sampler = Sampler(mode=backend)
job = sampler.run(transpiled, shots=1024)

print(f"\n*** NOISE CHECK SUBMITTED ***", flush=True)
print(f"Job ID: {job.job_id()}", flush=True)
print(f"URL: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)

