from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

print("=== SFE Quantum Hardware Execution ===", flush=True)

# 1. 연결
print("Connecting...", flush=True)
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)
backend = service.least_busy(operational=True, simulator=False)
print(f"Selected: {backend.name}", flush=True)

# 2. SFE 회로 구성
print("Building SFE circuit...", flush=True)
qc = QuantumCircuit(1, 1)
qc.h(0)  # 초기 중첩 상태

# SFE 최적화 펄스 시퀀스 (예시: 5개 타이밍)
sfe_timings = [0.12, 0.35, 0.58, 0.82, 0.95]
for t in sfe_timings:
    qc.delay(int(t * 100), 0)
    qc.x(0)

qc.measure(0, 0)

# 3. 트랜스파일
print("Transpiling...", flush=True)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)

# 4. 제출
print("Submitting job...", flush=True)
with Session(backend=backend) as session:
    sampler = Sampler(session=session)
    job = sampler.run([isa_qc], shots=1024)
    
    print(f"\n*** JOB SUBMITTED ***", flush=True)
    print(f"Job ID: {job.job_id()}", flush=True)
    print(f"URL: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)
    print(f"Backend: {backend.name}", flush=True)

