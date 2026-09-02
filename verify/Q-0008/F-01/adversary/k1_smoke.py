import sys, inspect
sys.path.insert(0, r"C:/dev/ce/Clarus-Equation")
from examples.physics.urbantke_shape_matching_rg import centered_shape_fluctuation_scaling as f
print("signature:", inspect.signature(f.__wrapped__))
r = f(sample_sizes=(4,8), trial_count=3, perturbation=0.05, seed=20260902)
print("K1 kwargs accepted; smoke fitted_power =", round(r.fitted_power,4))
