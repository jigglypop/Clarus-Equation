import numpy as np
from scipy.optimize import curve_fit


def exp_model(t, a, t1, b):
    return a * np.exp(-t / t1) + b


def main():
    delays = np.array([0, 500, 1000, 1500, 2000, 2500, 3000], dtype=float)
    p1 = np.array([0.9834, 0.9473, 0.9014, 0.8701, 0.8340, 0.8184, 0.7891], dtype=float)
    guess = (1.0, 2000.0, 0.0)
    popt, _ = curve_fit(exp_model, delays, p1, p0=guess, maxfev=10000)
    a, t1, b = popt
    print("=== Fitted T1 model ===", flush=True)
    print(f"A = {a:.4f}", flush=True)
    print(f"T1 (dt units) = {t1:.2f}", flush=True)
    print(f"B = {b:.4f}", flush=True)
    t_max = delays[-1]
    print(f"T1 / t_max = {t1 / t_max:.3f}", flush=True)
    s_target = 2.0 * p1[-1] - 1.0
    print(f"Effective coherence at t_max (from P1) ≈ {s_target:.4f}", flush=True)
    print()
    print("이 값들을 사용해서 noise.rs 의 alpha, scale 을 수동으로 조정할 수 있습니다.", flush=True)


if __name__ == "__main__":
    main()


