import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the compilation target directory to path if needed, or assume installed
# For development, one might need to point to target/release/deps or similar
# Here we assume 'maturin develop' was run and sfe_core is in python path.

try:
    import sfe_core
except ImportError:
    print("Error: sfe_core module not found. Please build with 'maturin develop' in sfe_core directory.")
    sys.exit(1)

def calculate_curvature(field):
    """Calculate total curvature (integral of (nabla^2 phi)^2)."""
    # Laplacian in 1D: phi[i-1] - 2phi[i] + phi[i+1]
    laplacian = np.zeros_like(field)
    laplacian[1:-1] = field[:-2] - 2*field[1:-1] + field[2:]
    # Boundary (clamped)
    laplacian[0] = 0 
    laplacian[-1] = 0
    return np.sum(laplacian**2)

def run_simulation(alpha2_value, steps=1000):
    size = 200
    engine = sfe_core.PyQSFEngine(size)
    engine.set_curvature_suppression(alpha2_value)
    
    # Run for some steps
    for _ in range(steps):
        engine.step()
        
    return np.array(engine.get_field())

def main():
    print("=== SFE Master Action Verification: Curvature Suppression ===")
    
    # Case 1: Standard Physics (alpha2 = 0)
    print("Running Standard Physics (alpha2=0)...")
    field_std = run_simulation(0.0)
    curv_std = calculate_curvature(field_std)
    print(f"Standard Final Curvature: {curv_std:.4f}")
    
    # Case 2: SFE Master Action (alpha2 = 5.0)
    # Note: alpha2 coefficient in action implies suppression of curvature
    print("Running SFE Master Action (alpha2=5.0)...")
    field_sfe = run_simulation(5.0)
    curv_sfe = calculate_curvature(field_sfe)
    print(f"SFE Final Curvature: {curv_sfe:.4f}")
    
    ratio = curv_std / curv_sfe if curv_sfe > 0 else float('inf')
    print(f"Curvature Suppression Ratio: {ratio:.2f}")
    
    if ratio > 1.5:
        print("SUCCESS: SFE Master Action effectively suppresses curvature fluctuations.")
    else:
        print("FAIL: Insufficient suppression. Tuning required.")

    # Simple visualization data export
    np.save("field_std.npy", field_std)
    np.save("field_sfe.npy", field_sfe)
    print("Saved field states to .npy files for visualization.")

if __name__ == "__main__":
    main()

