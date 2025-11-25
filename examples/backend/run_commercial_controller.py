import sys
import os
import numpy as np
import subprocess
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

sys.path.append(os.getcwd())
from sfe.controller import SFEAdaptiveController
from examples.backend.ibm_env import load_ibm_api_key
from examples.backend.rust_controller_launcher import parse_rust_output

def build_circuit_from_seq(seq, duration_dt, name):
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
    qc.name = f"{name}_T{duration_dt}"
    return qc

def run_live_commercial_controller():
    print("============================================================")
    print("   SFE Commercial Controller: Live Backend Connection       ")
    print("============================================================")

    # 1. IBM Backend 연결 및 진단 데이터 확보
    try:
        api_key = load_ibm_api_key()
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
        backend = service.backend("ibm_fez")
        print(f"[Link] Connected to Backend: {backend.name}")
        
        # 최신 보정 데이터 가져오기 (Qubit 0 기준)
        props = backend.properties()
        t1_us = props.t1(0) * 1e6
        t2_us = props.t2(0) * 1e6
        gate_err = 1e-3 # Default fallback
        try:
            gate_err = props.gate_error('sx', 0)
        except:
            pass
            
        print(f"[Live Data] Qubit 0 -> T1={t1_us:.1f}us, T2={t2_us:.1f}us, GateErr={gate_err:.1e}")
        
    except Exception as e:
        print(f"[Error] Failed to fetch backend data: {e}")
        return

    controller = SFEAdaptiveController(verbose=True)
    
    spec = controller.diagnose(t1_us, t2_us)
    
    strategy = controller.select_best_strategy(spec)
    
    if strategy.mode == "PASSIVE":
        print("[Stop] Controller decided NO ACTION is needed.")
        return

    rust_seq = None
    try:
        rust_exe = os.path.join("sfe_core", "target", "release", "sfe_engine.exe")
        result = subprocess.run(
            [
                rust_exe,
                "run-controller",
                "--t1",
                str(t1_us),
                "--t2",
                str(t2_us),
                "--gate-err",
                str(gate_err),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        if result.returncode == 0:
            rust_seq = parse_rust_output(result.stdout)
    except Exception:
        rust_seq = None

    print(f"\n[Execution] Generaring circuits for strategy: {strategy.mode}")
    
    durations_dt = [0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000]
    
    cpmg_seq = [(i - 0.5)/8.0 for i in range(1, 9)] # Standard CPMG-8
    sfe_seq = rust_seq if rust_seq else strategy.pulse_sequence
    
    circuits = []
    for d in durations_dt:
        circuits.append(build_circuit_from_seq(cpmg_seq, d, "CPMG_8"))
        circuits.append(build_circuit_from_seq(sfe_seq, d, f"SFE_{strategy.mode}_8"))
        
    print(f"[*] Submitting batch: {len(circuits)} circuits (CPMG vs SFE_{strategy.mode})")
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    
    sampler = Sampler(mode=backend)
    job = sampler.run([(c,) for c in isa_circuits], shots=1024)
    
    print(f"[!] JOB LAUNCHED: {job.job_id()}")
    print(f"[*] Monitor: https://quantum.ibm.com/jobs/{job.job_id()}")
    print("-" * 60)
    print("Controller has successfully deployed the optimal strategy to the cloud.")
    print("-" * 60)

if __name__ == "__main__":
    run_live_commercial_controller()

