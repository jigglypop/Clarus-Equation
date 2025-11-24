from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_env import load_ibm_api_key

JOB_ID = "d4hkuqscdebc73f26ueg"
SHOTS = 1024

print(f"=== Fetching Data for Job {JOB_ID} (CPMG_8 vs SFE_OPT_8, 10-duration sweep) ===", flush=True)

api_key = load_ibm_api_key()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)

try:
    job = service.job(JOB_ID)
    status = job.status()
    print(f"Job Status: {status}", flush=True)

    if status == "DONE" or str(status) == "JobStatus.DONE":
        print("Downloading results...", flush=True)
        result = job.result()

        durations = [0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000]
        labels = []
        for d in durations:
            labels.append(f"CPMG_8_T{d}")
            labels.append(f"SFE_OPT_8_T{d}")
        print("\n--- Survival Probabilities P(|0>) ---", flush=True)

        stats = []
        for i, pub_result in enumerate(result):
            label = labels[i] if i < len(labels) else f"Circuit_{i}"
            meas_data = pub_result.data.c
            counts = meas_data.get_counts()
            p0 = counts.get("0", 0) / SHOTS
            p1 = counts.get("1", 0) / SHOTS
            print(f"{label:14s} | P(0) = {p0:.4f}, P(1) = {p1:.4f}", flush=True)
            stats.append((label, p0, p1))

        pairs = {}
        for label, p0, p1 in stats:
            parts = label.split("_")
            dur = ""
            if len(parts) >= 3 and parts[-1].startswith("T"):
                dur = parts[-1][1:]
            kind = "CPMG" if "CPMG" in label else "SFE"
            k = dur if dur else "NA"
            if k not in pairs:
                pairs[k] = {}
            pairs[k][kind] = (p0, p1)

        d_list = [3, 5, 7]
        for dur, entry in pairs.items():
            if "CPMG" in entry and "SFE" in entry:
                p1_cpmg = entry["CPMG"][1]
                p1_sfe = entry["SFE"][1]
                if p1_sfe <= 0.0:
                    continue
                r = p1_cpmg / p1_sfe
                print(f"\n[duration {dur}] p_phy_CPMG={p1_cpmg:.4f}, p_phy_SFE={p1_sfe:.4f}, r={r:.3f}", flush=True)
                for d in d_list:
                    gain = r ** ((d + 1) / 2.0)
                    print(f"  d={d}: P_L(CPMG)/P_L(SFE) ≈ {gain:.2f}", flush=True)

    elif status in ["QUEUED", "RUNNING", "JobStatus.QUEUED", "JobStatus.RUNNING"]:
        print("Job is still running. Please wait a moment and try again.", flush=True)
        pos = job.queue_position() if hasattr(job, "queue_position") else "Unknown"
        print(f"Queue Position: {pos}", flush=True)
    else:
        print(f"Job ended with status: {status}", flush=True)
        if hasattr(job, "error_message"):
            print(f"Error: {job.error_message()}", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)