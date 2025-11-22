from qiskit_ibm_runtime import QiskitRuntimeService
from examples.ibm_env import load_ibm_api_key
JOB_ID = "d4h1h6h2bisc73a414c0"

print(f"=== Fetching Data for Job {JOB_ID} ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

# 2. 작업 조회
try:
    job = service.job(JOB_ID)
    status = job.status()
    print(f"Job Status: {status}", flush=True)
    
    if status == "DONE" or str(status) == "JobStatus.DONE":
        print("Downloading results...", flush=True)
        result = job.result()
        
        # 결과 파싱 (SamplerV2 결과 구조)
        # PubResult -> DataBin -> BitArray
        print("\n--- NOISE DATA (Counts for |1> state) ---", flush=True)
        for i, pub_result in enumerate(result):
            # pub_result.data.c (c는 classical register 이름, 보통 c 혹은 meas)
            # 여기서는 회로에서 measure가 있으므로 보통 'c' 또는 'meas'
            # SamplerV2는 bitstring을 반환하지 않고 counts dict를 줄 수도 있음
            
            # 데이터 구조 확인용 출력
            meas_data = pub_result.data.c  # 기본 레지스터 이름 가정
            counts = meas_data.get_counts()
            
            delay_val = i * 500 # 아까 500dt 간격으로 설정함
            p1 = counts.get('1', 0) / 1024.0 # |1> 확률
            print(f"Delay {delay_val:4}dt: P(|1>) = {p1:.4f}", flush=True)
            
    elif status in ["QUEUED", "RUNNING", "JobStatus.QUEUED", "JobStatus.RUNNING"]:
        print("Job is still running. Please wait a moment and try again.", flush=True)
        print(f"Queue Position: {job.queue_position() if hasattr(job, 'queue_position') else 'Unknown'}", flush=True)
    else:
        print(f"Job ended with status: {status}", flush=True)
        if hasattr(job, "error_message"):
            print(f"Error: {job.error_message()}", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)

