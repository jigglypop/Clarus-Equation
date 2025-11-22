from qiskit_ibm_runtime import QiskitRuntimeService

API_KEY = "OtWVkDQxb4Q6KCOukPKq6n61VU9Cpr09oCR4i3zhrmzm"

print("Connecting...", flush=True)
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)

print("CONNECTED!", flush=True)
print(service.active_account(), flush=True)

backends = service.backends()
print(f"\n{len(backends)} backends available:", flush=True)
for b in backends:
    print(f" - {b.name}", flush=True)

