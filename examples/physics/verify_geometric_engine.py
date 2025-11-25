import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_geometric_simulation():
    print("=== Rust SFE Geometric Engine Verification ===")
    
    # 1. Run Geometric Dynamics
    # Find executable path safely
    exe_path = os.path.abspath(os.path.join("sfe_core", "target", "release", "sfe_engine.exe"))
    if not os.path.exists(exe_path):
        # Try without .exe extension if on non-Windows (though this script assumes Windows env mostly)
        exe_path = os.path.abspath(os.path.join("sfe_core", "target", "release", "sfe_engine"))
        
    if not os.path.exists(exe_path):
        print(f"Error: Executable not found at {exe_path}")
        print("Building Rust engine first...")
        subprocess.run(["cargo", "build", "--release"], cwd="sfe_core", check=True)
    
    cmd_dynamics = [
        exe_path,
        "geometric-dynamics",
        "--dim", "2",
        "--steps", "1000",
        "--output", "geometric_output.csv"
    ]
    
    print(f"Running: {cmd_dynamics}")
    # cwd should be sfe_core so that output csv is saved there
    subprocess.run(cmd_dynamics, cwd="sfe_core", check=True)
    
    # 2. Run SCQE Simulation
    cmd_scqe = [
        exe_path,
        "scqe-simulation",
        "--steps", "1000",
        "--output", "scqe_result.csv"
    ]
    
    print(f"Running: {cmd_scqe}")
    subprocess.run(cmd_scqe, cwd="sfe_core", check=True)
    
    # 3. Visualize Results
    df_geo = pd.read_csv("sfe_core/geometric_output.csv")
    df_scqe = pd.read_csv("sfe_core/scqe_result.csv")
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trajectory in 2D Plane (Geometric Dynamics)
    ax[0, 0].plot(df_geo['X0'], df_geo['X1'], label='Particle Trajectory')
    ax[0, 0].set_title('Trajectory under Suppression Field (R-driven)')
    ax[0, 0].set_xlabel('X0')
    ax[0, 0].set_ylabel('X1')
    ax[0, 0].grid(True)
    ax[0, 0].legend()
    
    # Plot 2: Curvature R(x) vs Time
    ax[0, 1].plot(df_geo['Step'], df_geo['R'], color='red', label='Curvature R(x)')
    ax[0, 1].set_title('Riemann Curvature Evolution')
    ax[0, 1].set_xlabel('Step')
    ax[0, 1].set_ylabel('R(x)')
    ax[0, 1].grid(True)
    
    # Plot 3: Suppression Factor e^-R
    ax[1, 0].plot(df_geo['Step'], df_geo['Suppression'], color='green', label='Suppression Factor')
    ax[1, 0].axhline(y=0.90, color='r', linestyle='--', label='Maginot Line (90%)') # [NEW] Maginot Line
    ax[1, 0].set_title('Suppression Factor (e^-R)')
    ax[1, 0].set_xlabel('Step')
    ax[1, 0].set_ylabel('Factor (0-1)')
    ax[1, 0].grid(True)
    ax[1, 0].legend()
    
    # Plot 4: SCQE Energy Density (New)
    if 'EnergyDensity' in df_scqe.columns:
        ax[1, 1].plot(df_scqe['Step'], df_scqe['EnergyDensity'], color='blue', label='Suppression Energy Density')
        ax[1, 1].set_title('SFE Total Energy Density (rho_SFE)')
        ax[1, 1].set_xlabel('Step')
        ax[1, 1].set_ylabel('Density')
    else:
        # Fallback for backward compatibility
        ax[1, 1].plot(df_scqe['Step'], df_scqe['Suppression'], color='purple', label='With Feedback')
        ax[1, 1].set_title('SCQE Feedback Effect (Survival Rate)')
        ax[1, 1].set_xlabel('Step')
        ax[1, 1].set_ylabel('Suppression Factor')
        
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    # [NEW] Check pass/fail
    min_suppression = df_scqe['Suppression'].min()
    print(f"\n[Verification Result]")
    print(f"  Minimum Suppression Factor: {min_suppression:.4f}")
    if min_suppression >= 0.90:
        print("  Status: PASS (Maintained > 90%)")
    else:
        print("  Status: FAIL (Dropped below 90%)")
    
    plt.tight_layout()
    plt.savefig("examples/results/geometric_verification.png")
    print("Results saved to examples/results/geometric_verification.png")

if __name__ == "__main__":
    run_geometric_simulation()
