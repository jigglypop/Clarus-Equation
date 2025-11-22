import sys
import traceback

print("STEP 1: Starting script...", flush=True)

try:
    print("STEP 2: Importing Qiskit...", flush=True)
    from qiskit_ibm_runtime import QiskitRuntimeService
    print("STEP 3: Import successful!", flush=True)

    API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

    print("STEP 4: Attempting connection...", flush=True)
    
    # channel 인자를 제거하고 token만 전달 (자동 감지)
    # 또는 에러 메시지대로 channel="ibm_quantum" 시도 (최신 버전 대응)
    try:
        service = QiskitRuntimeService(token=API_KEY)
    except Exception:
        # 만약 자동 감지가 실패하면 명시적으로 지정
        service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)
    
    print("STEP 5: Connection SUCCESS!", flush=True)
    # 계정 정보 출력 (민감 정보 마스킹)
    account = service.active_account()
    masked_token = account.get('token', '****')[:5] + "..."
    print(f"STEP 6: Logged in as {account.get('instance', 'User')} (Token: {masked_token})", flush=True)
    
    # 사용 가능한 백엔드 하나만 출력
    print("STEP 7: Fetching backends...", flush=True)
    backends = service.backends(min_num_qubits=1)
    if backends:
        print(f"STEP 8: Found backend: {backends[0].name}", flush=True)
    else:
        print("STEP 8: No backends found (Check account status)", flush=True)

except Exception:
    print("CRITICAL ERROR OCCURRED:", flush=True)
    traceback.print_exc()
    sys.exit(1)
