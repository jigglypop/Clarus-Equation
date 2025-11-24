import math
import os
import itertools

M_MU_MEV = 105.66
M_E_MEV = 0.511
M_P_MEV = 938.27
ALPHA_EM = 1.0 / 137.036


def coeff_r2(m_lep_mev, m_phi_mev):
    reduced_mass = (m_lep_mev * M_P_MEV) / (m_lep_mev + M_P_MEV)
    denom = (2.0 * reduced_mass * ALPHA_EM + m_phi_mev) ** 2
    return (m_lep_mev * M_P_MEV) / denom


def coeff_g2(m_phi_mev):
    return (M_MU_MEV * M_MU_MEV) / (16.0 * math.pi * math.pi * m_phi_mev * m_phi_mev)


def solve3(a, b):
    a = [list(row) for row in a]
    x = list(b)
    n = 3
    for i in range(n):
        pivot = i
        max_val = abs(a[i][i])
        for j in range(i + 1, n):
            v = abs(a[j][i])
            if v > max_val:
                max_val = v
                pivot = j
        if max_val < 1e-18:
            return None
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            x[i], x[pivot] = x[pivot], x[i]
        diag = a[i][i]
        for k in range(i, n):
            a[i][k] /= diag
        x[i] /= diag
        for r in range(n):
            if r != i:
                factor = a[r][i]
                for c in range(i, n):
                    a[r][c] -= factor * a[i][c]
                x[r] -= factor * x[i]
    return x


def solve_multimode(masses, delta_r2_target, delta_a_target):
    c_e = [coeff_r2(M_E_MEV, m) for m in masses]
    c_mu = [coeff_r2(M_MU_MEV, m) for m in masses]
    d = [coeff_g2(m) for m in masses]
    a = [
        [c_e[0], c_e[1], c_e[2]],
        [c_mu[0], c_mu[1], c_mu[2]],
        [d[0], d[1], d[2]],
    ]
    b = [0.0, delta_r2_target, delta_a_target]
    k = solve3(a, b)
    if k is None:
        return None
    delta_r2_e = sum(k[i] * c_e[i] for i in range(3))
    delta_r2_mu = sum(k[i] * c_mu[i] for i in range(3))
    delta_a_mu = sum(k[i] * d[i] for i in range(3))
    res0 = delta_r2_e
    res1 = (delta_r2_mu - delta_r2_target) / delta_r2_target if delta_r2_target != 0 else 0.0
    res2 = (delta_a_mu - delta_a_target) / delta_a_target if delta_a_target != 0 else 0.0
    return {
        "kappa_sq": k,
        "delta_r2_e": delta_r2_e,
        "delta_r2_mu": delta_r2_mu,
        "delta_a_mu": delta_a_mu,
        "residuals": (res0, res1, res2),
    }


def scan_grid():
    # m1: 1 MeV ~ 10 GeV까지 넓은 범위 (대표 질량대 샘플링)
    m1_list = [1.0, 5.0, 10.0, 17.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    # m2, m3: 1 ~ 10000 MeV, 50 MeV 스텝 (아주 넓은 범위)
    m2_list = [float(i) for i in range(1, 10001, 50)]
    m3_list = [float(i) for i in range(1, 10001, 50)]
    delta_r2_target = 0.04 * 0.04
    delta_a_target = 2.51e-9
    results = []
    total = 0
    physical = 0
    for m1, m2, m3 in itertools.product(m1_list, m2_list, m3_list):
        total += 1
        masses = [m1, m2, m3]
        sol = solve_multimode(masses, delta_r2_target, delta_a_target)
        if sol is None:
            status = "singular"
            results.append((m1, m2, m3, status, None))
            continue
        kappa_sq = sol["kappa_sq"]
        res0, res1, res2 = sol["residuals"]
        all_pos = all(ks > 0.0 for ks in kappa_sq)
        good_r = abs(res1) < 0.1
        good_a = abs(res2) < 0.1
        if all_pos and good_r and good_a:
            status = "physical_ok"
            physical += 1
        elif all_pos:
            status = "physical_bad_fit"
        else:
            status = "non_physical"
        results.append((m1, m2, m3, status, sol))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multimode_scan.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("m1_mev,m2_mev,m3_mev,status,kap1_sq,kap2_sq,kap3_sq,delta_r2_e,delta_r2_mu,delta_a_mu,res0,res1,res2\n")
        for m1, m2, m3, status, sol in results:
            if sol is None:
                f.write(f"{m1:.3f},{m2:.3f},{m3:.3f},{status},,,,,,,,\n")
            else:
                k1, k2, k3 = sol["kappa_sq"]
                d_e = sol["delta_r2_e"]
                d_mu = sol["delta_r2_mu"]
                d_a = sol["delta_a_mu"]
                r0, r1, r2 = sol["residuals"]
                f.write(
                    f"{m1:.3f},{m2:.3f},{m3:.3f},{status},"
                    f"{k1:.6e},{k2:.6e},{k3:.6e},"
                    f"{d_e:.6e},{d_mu:.6e},{d_a:.6e},"
                    f"{r0:.3e},{r1:.3e},{r2:.3e}\n"
                )
    print("총 조합 수:", total)
    print("완전히 물리적인 해(κ_A^2>0, 오차<10%):", physical)
    if physical == 0:
        print("양수 κ_A^2를 모두 만족하는 영역은 현재 그리드에서는 발견되지 않았습니다.")
    else:
        print("양수 κ_A^2 영역이 발견되었습니다.")
    print("결과 저장:", out_path)


if __name__ == "__main__":
    scan_grid()


