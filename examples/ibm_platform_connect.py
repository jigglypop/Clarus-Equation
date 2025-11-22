from qiskit_ibm_runtime import QiskitRuntimeService

API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw"

print("Saving with ibm_quantum_platform...", flush=True)
try:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", 
        token=API_KEY, 
        overwrite=True
    )
    print("Saved!", flush=True)
    
    service = QiskitRuntimeService()
    print("CONNECTED!", flush=True)
    print(service.active_account(), flush=True)
    
    backends = service.backends()
    print(f"{len(backends)} backends found", flush=True)
    if backends:
        for b in backends[:3]:
            print(f" - {b.name}", flush=True)
    
except Exception as e:
    print(f"FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()

