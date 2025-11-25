from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_env import load_ibm_api_key

JOB_ID = "d4ir7910i6jc73dd4i70"
SHOTS = 1024

print(f"=== Fetching CPMG vs SFE Results for Job {JOB_ID} ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

try:
    job = service.job(JOB_ID)
    status = job.status()
    print(f"Job Status: {status}", flush=True)

    if status == "DONE" or str(status) == "JobStatus.DONE":
        print("Downloading results...", flush=True)
        result = job.result()

        # Duration sweep: 0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000
        durations = [0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000]
        
        print("\n" + "="*70, flush=True)
        print(f"{'Duration(dt)':<12} | {'CPMG P(0)':<12} {'CPMG P(1)':<12} | {'SFE P(0)':<12} {'SFE P(1)':<12}", flush=True)
        print("="*70, flush=True)

        cpmg_results = []
        sfe_results = []
        
        for i, pub_result in enumerate(result):
            data = pub_result.data
            
            # Get counts
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
                
                # Even index = CPMG, Odd index = SFE
                if i % 2 == 0:
                    cpmg_results.append((p0, p1))
                else:
                    sfe_results.append((p0, p1))

        # Print comparison table
        for idx, dur in enumerate(durations):
            if idx < len(cpmg_results) and idx < len(sfe_results):
                c_p0, c_p1 = cpmg_results[idx]
                s_p0, s_p1 = sfe_results[idx]
                print(f"{dur:<12} | {c_p0:<12.4f} {c_p1:<12.4f} | {s_p0:<12.4f} {s_p1:<12.4f}", flush=True)

        print("="*70, flush=True)
        
        # Analysis
        print("\n--- Analysis: Error Rate Comparison ---", flush=True)
        print(f"{'Duration':<10} | {'Err CPMG':<10} {'Err SFE':<10} | {'Reduction':<10} {'Winner':<8}", flush=True)
        print("-"*60, flush=True)
        
        total_reduction = 0
        sfe_wins = 0
        valid_count = 0
        
        for idx, dur in enumerate(durations):
            if idx < len(cpmg_results) and idx < len(sfe_results):
                # Error = P(1) (should be 0 for perfect coherence after H-delay-H)
                err_cpmg = cpmg_results[idx][1]
                err_sfe = sfe_results[idx][1]
                
                if err_cpmg > 0.01:
                    reduction = (err_cpmg - err_sfe) / err_cpmg * 100
                    total_reduction += reduction
                    valid_count += 1
                else:
                    reduction = 0.0
                
                winner = "SFE" if err_sfe < err_cpmg else "CPMG" if err_cpmg < err_sfe else "TIE"
                if winner == "SFE":
                    sfe_wins += 1
                    
                print(f"{dur:<10} | {err_cpmg:<10.4f} {err_sfe:<10.4f} | {reduction:>+8.2f}%  {winner:<8}", flush=True)
        
        print("-"*60, flush=True)
        avg_reduction = total_reduction / valid_count if valid_count > 0 else 0
        print(f"\nSFE Win Rate: {sfe_wins}/{len(durations)} ({sfe_wins/len(durations)*100:.1f}%)", flush=True)
        print(f"Average Error Reduction: {avg_reduction:+.2f}%", flush=True)

    elif status in ["QUEUED", "RUNNING", "JobStatus.QUEUED", "JobStatus.RUNNING"]:
        print("Job is still running. Please wait...", flush=True)
        try:
            pos = job.queue_position()
            print(f"Queue Position: {pos}", flush=True)
        except:
            pass
    else:
        print(f"Job ended with status: {status}", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
