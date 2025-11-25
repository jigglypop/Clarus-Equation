from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_env import load_ibm_api_key

JOBS_TO_CANCEL = [
    "d4iq85p0i6jc73dd3hkg",  # RUNNING
    "d4iqasqv0j9c73e2klmg",  # QUEUED
    "d4iqbas3tdfc73dmp7vg",  # QUEUED
]

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

for job_id in JOBS_TO_CANCEL:
    try:
        job = service.job(job_id)
        job.cancel()
        print(f"[{job_id}] CANCELLED", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)

