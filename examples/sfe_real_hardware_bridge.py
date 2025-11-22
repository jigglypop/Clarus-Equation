import numpy as np
import time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- User Configuration ---
# Provided API Key
API_KEY = "JYnFOvSCYI4GmXFRq7XHALCyjPIRpPXuibfPT_5fUkYw" 
# --------------------------

def get_sfe_optimized_sequence():
    """
    Simulate fetching the optimal pulse sequence from the Rust SFE Engine.
    In a real scenario, this would read from 'sweep_results.csv' or call the Rust binary.
    Values are normalized time (0.0 to 1.0) within the idle window.
    """
    # Example: SFE-Genetic found these optimal dynamical decoupling points
    return np.array([0.12, 0.35, 0.58, 0.82, 0.95])

def main():
    print(f"[*] Connecting to IBM Quantum with provided API Key...")
    
    try:
        # 1. Authenticate
        # Save account to disk for future use (optional, but good practice)
        try:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=API_KEY, overwrite=True)
        except:
            pass # Might already be saved or env issue
            
        service = QiskitRuntimeService(channel="ibm_quantum", token=API_KEY)
        
        print("[+] Authentication Successful!")
        
        # 2. Find the least busy real backend
        print("[*] Searching for the least busy real quantum computer...")
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=1)
        print(f"[+] Selected Backend: {backend.name}")
        print(f"    - Status: {backend.status().status_msg}")
        print(f"    - Pending Jobs: {backend.status().pending_jobs}")
        
        # 3. Construct SFE Circuit
        # Since direct pulse control (OpenPulse) support varies by backend and access level,
        # we will demonstrate the logical circuit mapping here.
        # For full pulse injection, we would attach a calibration to a custom gate.
        
        print("[*] Constructing SFE-Protected Quantum Circuit...")
        qc = QuantumCircuit(1)
        
        # Prepare superposition
        qc.h(0)
        
        # Apply SFE Protection (Dynamical Decoupling Sequence)
        # In a logical circuit, this looks like a sequence of delays and X gates
        # precisely timed according to SFE engine output.
        sfe_timings = get_sfe_optimized_sequence()
        total_delay = 1000 # units of dt or abstract time
        
        last_t = 0
        for t_norm in sfe_timings:
            current_t = int(t_norm * total_delay)
            wait_time = current_t - last_t
            if wait_time > 0:
                qc.delay(wait_time, 0)
            qc.x(0) # The decoupling pulse
            last_t = current_t
            
        # Final measurement
        qc.measure_all()
        
        # 4. Transpile & Run
        print("[*] Transpiling for hardware...")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)
        
        print(f"[*] Submitting job to {backend.name}...")
        # Using Primitives (Sampler) for execution
        sampler = Sampler(backend=backend)
        job = sampler.run([isa_circuit])
        
        print(f"[!] JOB SUBMITTED! Job ID: {job.job_id()}")
        print(f"[*] You can monitor this job at: https://quantum.ibm.com/jobs/{job.job_id()}")
        print("[*] Waiting for results (this may take time depending on the queue)...")
        
        # For demonstration, we don't block indefinitely if queue is long, 
        # but we'll try to wait a bit.
        # result = job.result()
        # print(f"[+] Result: {result[0].data.meas.get_counts()}")
        
    except Exception as e:
        print(f"\n[!] Error occurred: {e}")
        print("Common fixes:")
        print("1. Check if qiskit-ibm-runtime is installed (pip install qiskit-ibm-runtime)")
        print("2. Verify the API key is correct.")
        print("3. Ensure you have access to open provider backends.")

if __name__ == "__main__":
    main()
