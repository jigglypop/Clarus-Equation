import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ibm_env import load_ibm_api_key

print("=== Quick SFE Test (8 second duration) ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
backend = service.backend("ibm_fez")
print(f"Backend: {backend.name}", flush=True)

# 8초 = 8,000,000 us = 8e9 ns
# dt = 0.222ns for Fez, so 8s = ~36,000,000,000 dt (too long)
# Actually let's use 8000 dt which is about 1.8us (reasonable)
duration_dt = 8000

def build_circuit(seq, name):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    last_t = 0
    for t_ratio in seq:
        curr_t = int(t_ratio * duration_dt)
        delay = curr_t - last_t
        if delay > 0:
            qc.delay(delay, 0)
        qc.x(0)
        last_t = curr_t
    if duration_dt - last_t > 0:
        qc.delay(duration_dt - last_t, 0)
    qc.h(0)
    qc.measure(0, 0)
    qc.name = name
    return qc

# CPMG-8 vs SFE-8
cpmg_seq = [(i - 0.5)/8.0 for i in range(1, 9)]
sfe_seq = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]  # SFE optimized

circuits = [
    build_circuit(cpmg_seq, "CPMG_8"),
    build_circuit(sfe_seq, "SFE_8")
]

print(f"Submitting 2 circuits (CPMG vs SFE) with duration={duration_dt} dt", flush=True)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuits = pm.run(circuits)

sampler = Sampler(mode=backend)
job = sampler.run(isa_circuits, shots=1024)

print(f"JOB ID: {job.job_id()}", flush=True)
print(f"Monitor: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)
print("Job submitted. Check results later with fetch script.", flush=True)
exit(0)

print("\n=== RESULTS ===", flush=True)
for i, pub_result in enumerate(result):
    name = "CPMG_8" if i == 0 else "SFE_8"
    data = pub_result.data
    counts = None
    if hasattr(data, 'meas'):
        counts = data.meas.get_counts()
    else:
        for attr in dir(data):
            if not attr.startswith('_'):
                try:
                    meas_data = getattr(data, attr)
                    if hasattr(meas_data, 'get_counts'):
                        counts = meas_data.get_counts()
                        break
                except:
                    pass
    
    if counts:
        total = sum(counts.values())
        p0 = counts.get("0", 0) / total
        p1 = counts.get("1", 0) / total
        print(f"{name}: P(0)={p0:.4f}, P(1)={p1:.4f} [Error={p1:.4f}]", flush=True)

# Compare
if len(result) >= 2:
    print("\n--- Comparison ---", flush=True)
    err_cpmg = result[0].data.meas.get_counts().get("1", 0) / 1024
    err_sfe = result[1].data.meas.get_counts().get("1", 0) / 1024
    reduction = (err_cpmg - err_sfe) / err_cpmg * 100 if err_cpmg > 0.01 else 0
    winner = "SFE" if err_sfe < err_cpmg else "CPMG"
    print(f"Error CPMG: {err_cpmg:.4f}", flush=True)
    print(f"Error SFE:  {err_sfe:.4f}", flush=True)
    print(f"Reduction:  {reduction:+.2f}%", flush=True)
    print(f"Winner:     {winner}", flush=True)

