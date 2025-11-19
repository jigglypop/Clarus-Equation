import numpy as np

# 토이 파워 스펙트럼 (연속, ΛCDM 대략 모양)
def P_toy(k, A=1.0, n_s=0.96, k_star=0.5):
    k = np.asarray(k)
    return np.where(k < k_star,
                    A * (k / k_star)**n_s,
                    A * (k / k_star)**(n_s - 4.0))

# 억압장 고주파 추가 감쇠
def f_sup(k, k_star, sigma):
    k = np.asarray(k)
    return np.where(k < k_star, 1.0, (k_star / k)**sigma)

# 적분 구간 및 스케일 설정
k_clus = 0.5   # 은하단 스케일 ~ 1/(몇 Mpc)
k_gal  = 50.0  # 은하 스케일 ~ 1/(몇 x 10 kpc)

k_min = 1e-4
k_max = 1e3
N = 4000
k = np.logspace(np.log10(k_min), np.log10(k_max), N)
dk = np.gradient(k)

Pk = P_toy(k)

# 우주론 파라미터 (이미 고정된 값들)
delta_gal = 1e8
delta_clus = 1e4
q = 0.675 / 0.315  # ≈ 2.143

for sigma in [1.0, 2.0, 3.0]:
    fs = f_sup(k, k_clus, sigma)

    mask_large = (k < k_clus)
    mask_mid   = (k >= k_clus) & (k < k_gal)
    mask_small = (k >= k_gal)

    A3 = np.sum(k**2 * Pk * mask_large * dk)
    A2 = np.sum(k**2 * Pk * mask_mid   * dk)
    A1 = np.sum(k**2 * Pk * fs * mask_small * dk)

    r1 = A1 / A3  # ≈ C1/C3 에 비례
    r2 = A2 / A3  # ≈ C2/C3 에 비례

    R_gal  = 1.0 + delta_gal  * q * r1
    R_clus = 1.0 + delta_clus * q * r2

    print("sigma =", sigma)
    print("  A1, A2, A3 =", A1, A2, A3)
    print("  r1 = A1/A3 =", r1)
    print("  r2 = A2/A3 =", r2)
    print("  R_gal - 1  =", R_gal - 1.0)
    print("  R_clus - 1 =", R_clus - 1.0)
    print("  ratio (clus/gal) =", (R_clus - 1.0) / (R_gal - 1.0) if (R_gal-1.0)!=0 else np.nan)
    print('-'*60)


def w_large(k, k_clus):
    return np.exp(-0.5 * (k / k_clus) ** 2)


def w_mid(k, k_clus, k_gal):
    k_mid = np.sqrt(k_clus * k_gal)
    sigma_k = 0.3 * k_mid
    return np.exp(-0.5 * ((k - k_mid) / sigma_k) ** 2)


def run_with_windows():
    for sigma in [1.0, 2.0, 3.0]:
        fs = f_sup(k, k_clus, sigma)
        wl = w_large(k, k_clus)
        wm = w_mid(k, k_clus, k_gal)
        w_small = np.clip(1.0 - wl - wm, 0.0, 1.0)

        A3w = np.sum(k**2 * Pk * wl * dk)
        A2w = np.sum(k**2 * Pk * wm * dk)
        A1w = np.sum(k**2 * Pk * fs * w_small * dk)

        r1w = A1w / A3w
        r2w = A2w / A3w

        R_gal_w = 1.0 + delta_gal * q * r1w
        R_clus_w = 1.0 + delta_clus * q * r2w

        print("with windows: sigma =", sigma)
        print("  A1w, A2w, A3w =", A1w, A2w, A3w)
        print("  r1w = A1w/A3w =", r1w)
        print("  r2w = A2w/A3w =", r2w)
        print("  R_gal_w - 1  =", R_gal_w - 1.0)
        print("  R_clus_w - 1 =", R_clus_w - 1.0)
        print("  ratio_w (clus/gal) =", (R_clus_w - 1.0) / (R_gal_w - 1.0) if (R_gal_w-1.0)!=0 else np.nan)
        print('='*60)


def j0(z):
    z = np.asarray(z)
    return np.where(z == 0.0, 1.0, np.sin(z) / z)


def run_with_j0():
    R_gal = 0.01
    R_clus = 1.0
    for sigma in [1.0, 2.0, 3.0]:
        fs = f_sup(k, k_clus, sigma)
        wl = w_large(k, k_clus)
        wm = w_mid(k, k_clus, k_gal)
        w_small = np.clip(1.0 - wl - wm, 0.0, 1.0)
        W = wl + wm + fs * w_small
        C0 = np.sum(k**2 * Pk * W * dk)
        C_gal = np.sum(k**2 * Pk * W * j0(k * R_gal) * dk)
        C_clus = np.sum(k**2 * Pk * W * j0(k * R_clus) * dk)
        r1j = C_gal / C0
        r2j = C_clus / C0
        R_gal_j = 1.0 + delta_gal * q * r1j
        R_clus_j = 1.0 + delta_clus * q * r2j
        print("with j0: sigma =", sigma)
        print("  C0, C_gal, C_clus =", C0, C_gal, C_clus)
        print("  r1j = C_gal/C0 =", r1j)
        print("  r2j = C_clus/C0 =", r2j)
        print("  R_gal_j - 1  =", R_gal_j - 1.0)
        print("  R_clus_j - 1 =", R_clus_j - 1.0)
        print("  ratio_j (clus/gal) =", (R_clus_j - 1.0) / (R_gal_j - 1.0) if (R_gal_j-1.0)!=0 else np.nan)
        print('*'*60)


def W_hd(k, k_hd, n_hd):
    return 1.0 / (1.0 + (k / k_hd) ** (2 * n_hd))


def W_rg(k, k0, p_rg, m_rg):
    return 1.0 / (1.0 + (k / k0) ** p_rg) ** m_rg


def run_general_scan():
    k_hd = k_gal
    k0 = k_clus
    combos = []
    for n_hd in [2, 3]:
        for p_rg in [1, 2]:
            for m_rg in [1, 2, 3]:
                combos.append((n_hd, p_rg, m_rg))
    for n_hd, p_rg, m_rg in combos:
        Whd = W_hd(k, k_hd, n_hd)
        Wrg = W_rg(k, k0, p_rg, m_rg)
        Wtot = Whd * Wrg
        mask_large = (k < k_clus)
        mask_mid = (k >= k_clus) & (k < k_gal)
        mask_small = (k >= k_gal)
        A3g = np.sum(k**2 * Pk * Wtot * mask_large * dk)
        A2g = np.sum(k**2 * Pk * Wtot * mask_mid * dk)
        A1g = np.sum(k**2 * Pk * Wtot * mask_small * dk)
        if A3g == 0.0:
            continue
        r1g = A1g / A3g
        r2g = A2g / A3g
        R_gal_g = 1.0 + delta_gal * q * r1g
        R_clus_g = 1.0 + delta_clus * q * r2g
        print("general W: n_hd =", n_hd, "p_rg =", p_rg, "m_rg =", m_rg)
        print("  A1g, A2g, A3g =", A1g, A2g, A3g)
        print("  r1g = A1g/A3g =", r1g)
        print("  r2g = A2g/A3g =", r2g)
        print("  R_gal_g - 1  =", R_gal_g - 1.0)
        print("  R_clus_g - 1 =", R_clus_g - 1.0)
        print("  ratio_g (clus/gal) =", (R_clus_g - 1.0) / (R_gal_g - 1.0) if (R_gal_g-1.0)!=0 else np.nan)
        print('#'*60)


if __name__ == "__main__":
    run_with_windows()
    run_with_j0()
    run_general_scan()