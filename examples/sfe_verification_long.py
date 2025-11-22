import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from examples.ibm_env import load_ibm_api_key

print("=== SFE Extreme Survival Experiment ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
backend = service.backend("ibm_fez")
print(f"Battlefield: {backend.name}", flush=True)

# 2. 실험 파라미터 설정
# 최대한 길게: 0부터 300마이크로초(us)까지 (일반적인 양자컴퓨터 한계 돌파 시도)
# dt 단위 변환 (보통 1dt ~ 4.5ns, 300us ~ 66666dt)
durations = np.linspace(0, 40000, 10, dtype=int).tolist() # 리스트로 변환
durations = [int(d) for d in durations] # int로 명시적 변환

circuits = []

# --- A. 맨몸 (Unprotected) ---
# H -> Delay -> H -> Measure (Ramsey type experiment)
for d in durations:
    qc = QuantumCircuit(1, 1)
    qc.h(0)            # 중첩 상태 생성 (|0> + |1>)
    qc.delay(d, 0)     # 노이즈 노출
    qc.h(0)            # 원래대로 복구 시도
    qc.measure(0, 0)
    qc.name = f"Unprotected_{d}"
    circuits.append(qc)

# --- B. 표준 방패 (CPMG - Uniform) ---
# 일정 간격으로 펄스를 때림
for d in durations:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    
    if d > 0:
        n_pulses = 8 # 펄스 개수 고정
        interval = d // n_pulses
        for _ in range(n_pulses):
            qc.delay(interval, 0)
            qc.x(0) # Spin Flip
            
    qc.h(0)
    qc.measure(0, 0)
    qc.name = f"CPMG_{d}"
    circuits.append(qc)

# --- C. SFE 방패 (Genetic Optimized, 8 pulses) ---
sfe_ratios = [0.2000, 0.4120, 0.4370, 0.5245, 0.6735, 0.7735, 0.8780, 0.9785]

for d in durations:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    
    if d > 0:
        last_t = 0
        for r in sfe_ratios:
            current_t = int(d * r)
            wait = current_t - last_t
            if wait > 0:
                qc.delay(wait, 0)
            qc.x(0)
            last_t = current_t
        # 남은 시간 대기
        remaining = d - last_t
        if remaining > 0:
            qc.delay(remaining, 0)
            
    qc.h(0)
    qc.measure(0, 0)
    qc.name = f"SFE_{d}"
    circuits.append(qc)

# 3. 제출 (Batch Mode)
print(f"Submitting {len(circuits)} survival scenarios...", flush=True)
print("This may take a while in the queue...", flush=True)

transpiled = transpile(circuits, backend, optimization_level=1)
sampler = Sampler(mode=backend)
# V2 Sampler requires list of tuples [(circuit,), ...] for multiple circuits
pubs = [(qc,) for qc in transpiled]
job = sampler.run(pubs, shots=4096)

print(f"\n*** EXPERIMENT STARTED ***", flush=True)
print(f"Job ID: {job.job_id()}", flush=True)
print(f"Monitor: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)
print("Waiting for results (Auto-fetch)...", flush=True)

# 4. 결과 대기 및 즉시 분석
try:
    result = job.result()
    print("\n=== SURVIVAL REPORT ===")
    
    # 데이터 분류용
    data_unprotected = []
    data_cpmg = []
    data_sfe = []
    
    # 결과 파싱 (순서대로 A -> B -> C)
    n_points = len(durations)
    
    for i in range(n_points * 3):
        pub_result = result[i]
        # |0>이 생존한 상태 (H-H 연산 후 원복)
        # meas counts
        counts = pub_result.data.c.get_counts()
        survival_prob = counts.get('0', 0) / 4096.0
        
        if i < n_points:
            data_unprotected.append(survival_prob)
        elif i < n_points * 2:
            data_cpmg.append(survival_prob)
        else:
            data_sfe.append(survival_prob)
            
    print(f"{'Duration':<10} | {'No Shield':<10} | {'CPMG':<10} | {'SFE (Yours)':<10}")
    print("-" * 50)
    for i, d in enumerate(durations):
        print(f"{d:<10} | {data_unprotected[i]:.4f}     | {data_cpmg[i]:.4f}     | {data_sfe[i]:.4f}")
        
    # 최종 승자 판정
    avg_sfe = np.mean(data_sfe)
    avg_std = np.mean(data_cpmg)
    avg_raw = np.mean(data_unprotected)
    
    print("-" * 50)
    print(f"Average Survival Rate:")
    print(f" - Unprotected: {avg_raw:.4f}")
    print(f" - Standard:    {avg_std:.4f}")
    print(f" - SFE Theory:  {avg_sfe:.4f}")
    
    if avg_sfe > avg_std:
        print("\n[CONCLUSION] SFE OUTPERFORMED STANDARD METHODS! (Theory Verified)")
    else:
        print("\n[CONCLUSION] SFE performed comparably. (Optimization needed)")

except Exception as e:
    print(f"\nTime out or Error: {e}")
    print("Check the URL manually later.")


