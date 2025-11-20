import numpy as np


# 간단한 2-모드 억압장 모형
# 모드 0: 배경(진공/암흑에너지 유사)
# 모드 1: 요동(입자/암흑물질 유사)

H = 0.05
m0 = 0.5
m1 = 0.5
g = 0.5
L4 = 0.5  # 진공 에너지 스케일


def rhs(t, y):
    """y = (Phi0, Phi0dot, Phi1, Phi1dot)에 대한 우변"""
    Phi0, Phi0dot, Phi1, Phi1dot = y
    V0p = m0 ** 2 * Phi0
    Phi0ddot = -3.0 * H * Phi0dot - V0p - g * Phi1 ** 2
    omega1_sq = m1 ** 2 + 2.0 * g * Phi0
    Phi1ddot = -3.0 * H * Phi1dot - omega1_sq * Phi1
    return np.array([Phi0dot, Phi0ddot, Phi1dot, Phi1ddot])


def energy_pressure(y):
    """각 모드의 에너지 밀도와 압력 계산"""
    Phi0, Phi0dot, Phi1, Phi1dot = y
    V0 = L4 + 0.5 * m0 ** 2 * Phi0 ** 2
    rho0 = 0.5 * Phi0dot ** 2 + V0
    p0 = 0.5 * Phi0dot ** 2 - V0
    omega1_sq = m1 ** 2 + 2.0 * g * Phi0
    rho1 = 0.5 * Phi1dot ** 2 + 0.5 * omega1_sq * Phi1 ** 2
    p1 = 0.5 * Phi1dot ** 2 - 0.5 * omega1_sq * Phi1 ** 2
    return rho0, p0, rho1, p1


def integrate():
    # 초기 조건: 배경에 에너지 집중, 요동은 거의 0
    Phi0_init = 2.0
    Phi0dot_init = 0.0
    Phi1_init = 1e-3
    Phi1dot_init = 0.0
    y = np.array([Phi0_init, Phi0dot_init, Phi1_init, Phi1dot_init])

    t_max = 200.0
    dt = 0.001
    N = int(t_max / dt)

    ts = np.zeros(N)
    rho0s = np.zeros(N)
    rho1s = np.zeros(N)
    p0s = np.zeros(N)
    p1s = np.zeros(N)
    rhots = np.zeros(N)

    # 4차 Runge–Kutta 적분
    for i in range(N):
        t = i * dt
        ts[i] = t
        rho0, p0, rho1, p1 = energy_pressure(y)
        rho0s[i] = rho0
        rho1s[i] = rho1
        p0s[i] = p0
        p1s[i] = p1
        rhots[i] = rho0 + rho1
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 에너지 교환 항 Q 계산
    rho0dot = np.gradient(rho0s, dt)
    rho1dot = np.gradient(rho1s, dt)
    Q0 = -(rho0dot + 3.0 * H * (rho0s + p0s))
    Q1 = (rho1dot + 3.0 * H * (rho1s + p1s))

    return ts, rho0s, rho1s, rhots, Q0, Q1


def main():
    ts, rho0s, rho1s, rhots, Q0, Q1 = integrate()

    print("=== 에너지 분해 ===")
    print(f"rho0(0) = {rho0s[0]:.6f}, rho1(0) = {rho1s[0]:.6e}")
    print(f"rho0(T) = {rho0s[-1]:.6f}, rho1(T) = {rho1s[-1]:.6f}")
    print(f"rho_tot(0) = {rhots[0]:.6f}, rho_tot(T) = {rhots[-1]:.6f}")

    print("\n=== 에너지 교환 항 검증 ===")
    print(f"<Q0+Q1> ≈ {np.mean(Q0 + Q1):.3e}")
    print(f"max|Q0+Q1| ≈ {np.max(np.abs(Q0 + Q1)):.3e}")
    print(f"<Q0> ≈ {np.mean(Q0):.3e}")
    print(f"<Q1> ≈ {np.mean(Q1):.3e}")

    # 진공 → 입자 방향 에너지 흐름이 있었는지 대략적인 판단
    grew = np.any(rho1s > rho1s[0] * 10)
    print("\n요동 모드 에너지 증폭 여부 (배경→요동 전환 시그널):", grew)


if __name__ == "__main__":
    main()


