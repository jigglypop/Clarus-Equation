from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile

API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

print("=== SFE Quantum (Primitives) ===", flush=True)

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)
backend = service.least_busy(operational=True, simulator=False)
print(f"Backend: {backend.name}", flush=True)

qc = QuantumCircuit(1, 1)
qc.h(0)
for t in [0.12, 0.35, 0.58, 0.82, 0.95]:
    qc.delay(int(t * 100), 0)
    qc.x(0)
qc.measure(0, 0)

print("Transpiling...", flush=True)
transpiled = transpile(qc, backend, optimization_level=1)

print("Submitting...", flush=True)
sampler = Sampler(backend=backend)
job = sampler.run(transpiled, shots=1024)

print(f"\n*** SUCCESS ***", flush=True)
print(f"Job ID: {job.job_id()}", flush=True)
print(f"URL: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)

