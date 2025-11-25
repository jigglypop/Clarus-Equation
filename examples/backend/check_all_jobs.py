from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_env import load_ibm_api_key

JOBS = [
    "d4iqc9h0i6jc73dd3le0",  # 최신 CPMG vs SFE
]

print("=== Checking All IBM Jobs ===\n", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

for job_id in JOBS:
    try:
        job = service.job(job_id)
        status = str(job.status())
        print(f"[{job_id}] Status: {status}", flush=True)
        
        if "DONE" in status:
            result = job.result()
            print(f"  -> {len(result)} circuits completed", flush=True)
            
            for i, pub_result in enumerate(result):
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
                    print(f"     Circuit {i}: P(0)={p0:.4f}, P(1)={p1:.4f}", flush=True)
            print()
    except Exception as e:
        print(f"[{job_id}] Error: {e}\n", flush=True)

