from qiskit_ibm_runtime import QiskitRuntimeService

API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

print("STEP 1: Saving account...", flush=True)
try:
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=API_KEY, overwrite=True)
    print("Account saved successfully!", flush=True)
except Exception as e:
    print(f"Save failed: {e}", flush=True)
    print("Trying without channel...", flush=True)
    try:
        QiskitRuntimeService.save_account(token=API_KEY, overwrite=True)
        print("Saved without channel!", flush=True)
    except Exception as e2:
        print(f"Still failed: {e2}", flush=True)

print("\nSTEP 2: Connecting with saved account...", flush=True)
try:
    service = QiskitRuntimeService()
    print("CONNECTED!", flush=True)
    print(f"Account: {service.active_account()}", flush=True)
    
    backends = service.backends()
    print(f"\nBackends: {len(backends)}", flush=True)
    for b in backends[:5]:
        print(f" - {b.name}", flush=True)
        
except Exception as e:
    print(f"Connection failed: {e}", flush=True)

