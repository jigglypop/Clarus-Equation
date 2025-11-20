
import math

def integrate(func, a, b, n=1000):
    h = (b - a) / n
    s = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        s += func(a + i * h)
    return s * h

def calculate_g2_contribution(g, m_phi, m_mu):
    # g is the Yukawa coupling constant
    r = (m_phi / m_mu)**2
    
    def integrand(x):
        num = (1-x)**2 * (1+x)
        den = (1-x)**2 + x*r
        if den == 0: return 0
        return num/den
    
    val = integrate(integrand, 0, 1)
    
    delta_a = (g**2 / (8 * math.pi**2)) * val
    return delta_a

# Constants
m_mu_MeV = 105.66
m_e_MeV = 0.511
epsilon = 0.37

# Scenario 1: f_phi approx v_EW / sqrt(N)
f_phi_A = 71.0 * 1000 # MeV
f_phi_B = 142.0 * 1000 # MeV

# Coupling derivation: g = epsilon * m / f_phi
g_mu_A = epsilon * m_mu_MeV / f_phi_A
g_mu_B = epsilon * m_mu_MeV / f_phi_B

print(f"Coupling g_mu (f=71GeV): {g_mu_A:.4e}")

# Predicted Masses (from 9B.3.2)
m_phi_2 = 46.0 # MeV

# Calculate g-2
da_A = calculate_g2_contribution(g_mu_A, m_phi_2, m_mu_MeV)
da_B = calculate_g2_contribution(g_mu_B, m_phi_2, m_mu_MeV)

print(f"Predicted g-2 (f=71GeV, m=46MeV): {da_A:.4e}")
print(f"Predicted g-2 (f=142GeV, m=46MeV): {da_B:.4e}")

# Target
target = 2.49e-9
print(f"Target Value: {target:.4e}")

# Check ratio
print(f"Ratio A/Target: {da_A/target:.2f}")

# Perfect f_phi
f_perfect = f_phi_A * math.sqrt(da_A / target)
print(f"Perfect f_phi for match: {f_perfect/1000:.2f} GeV")
