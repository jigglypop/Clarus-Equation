
import math

# 1. Natural Constants (SI Units) - Level 1 Inputs
G = 6.67430e-11      # m^3 kg^-1 s^-2
c = 2.99792458e8     # m s^-1
hbar = 1.0545718e-34 # J s
mp = 1.6726219e-27   # kg
me = 9.10938356e-31  # kg
alpha_em = 7.297352569e-3 # Fine structure constant

# Planck Mass
M_P = math.sqrt(hbar * c / G)
print(f"Planck Mass: {M_P:.4e} kg")

# 2. Derived Alpha (Holographic Scaling) - Level 2 Inputs
# Hypothesis: alpha ~ (mp / M_P)^(2/3)
# This represents the surface-to-volume scaling of information density 
# from Planck scale to Baryon scale.
scaling_factor = (mp / M_P)**(2/3)
# A geometric factor might be needed. Let's try to match 2.3e-13.
# scaling_factor is approx 1.8e-13.
# Let's assume a factor of order 1, e.g., related to spin or pi.
# 4/3? 
geometric_factor = 1.25 # Trial to match close to 2.3e-13 if needed, but let's stick to pure scaling first.

alpha_dimless = scaling_factor * 1.0
# The previous text mentioned 2.3e-13.
# 1.8e-13 is close. Let's see the result.

# Alpha SI
# alpha_si = alpha_dimless * sqrt(G/c)
alpha_si = alpha_dimless * math.sqrt(G / c)

print(f"\n--- Derived Constants (Holographic Hypothesis) ---")
print(f"Mass Ratio (mp/MP): {mp/M_P:.4e}")
print(f"Scaling Factor (ratio^(2/3)): {scaling_factor:.4e}")
print(f"alpha_dimless: {alpha_dimless:.4e}")
print(f"alpha_si: {alpha_si:.4e} kg^-1/2")

# 3. Cosmological Parameters (Observational) - Level 3 Inputs
H0_km_s_Mpc = 67.4 # Planck 2018
H0 = H0_km_s_Mpc * 1000 / (3.08567758e22) # Convert to s^-1
Omega_m = 0.315

rho_c = 3 * H0**2 / (8 * math.pi * G)
rho_m_bar = Omega_m * rho_c

print(f"\n--- Cosmological Inputs (Level 3) ---")
print(f"rho_c: {rho_c:.4e} kg/m^3")
print(f"rho_m_bar: {rho_m_bar:.4e} kg/m^3")


# 4. Fixed Point Iteration for Lambda and Rho_Phi (Chapter 23) - Level 4
# Formula: lambda^4 = (3 * c^2) / (8 * pi * G * alpha^2 * rho_m_bar^2 * C(X))
C_X = 0.46 

# Initial guess
lambda_val = c / H0 
tolerance = 1e-6
max_iter = 100

print(f"\n--- Fixed Point Iteration ---")
for i in range(max_iter):
    numerator = 3 * c**2
    denominator = 8 * math.pi * G * alpha_si**2 * rho_m_bar**2 * C_X
    lambda_new = (numerator / denominator)**(0.25)
    
    if abs(lambda_new - lambda_val) / lambda_val < tolerance:
        lambda_val = lambda_new
        print(f"Converged in {i+1} iterations.")
        break
    lambda_val = lambda_new

print(f"Derived lambda*: {lambda_val:.4e} m")
print(f"Hubble radius (c/H0): {c/H0:.4e} m")
print(f"Ratio lambda/R_H: {lambda_val/(c/H0):.4f}")

# Calculate Rho_Phi
rho_phi = alpha_si**2 * rho_m_bar**2 * lambda_val**2 * C_X
print(f"Derived rho_phi: {rho_phi:.4e} kg/m^3")

# Calculate Omega_Phi (Theory)
Omega_phi_theory = rho_phi / rho_c
print(f"\n--- Results (Level 5 Verification) ---")
print(f"Omega_Phi (Theory): {Omega_phi_theory:.4f}")

# Compare with Observation
Omega_lambda_obs = 0.685 
print(f"Omega_Lambda (Obs): {Omega_lambda_obs}")
error = abs(Omega_phi_theory - Omega_lambda_obs) / Omega_lambda_obs * 100
print(f"Error: {error:.2f}%")

# 5. Derived Epsilon
epsilon_theory = 2 * Omega_phi_theory - 1
print(f"epsilon (Theory): {epsilon_theory:.4f}")
