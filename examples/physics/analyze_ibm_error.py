import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_error_rates():
    # Load sweep results
    try:
        df = pd.read_csv('examples/results/ibm_fez_sweep.csv')
        
        # The CSV columns are: PulseCount,NoiseAmp,UDD_Score,SFE_Score,Improvement_Pct
        # Error Rate = 1.0 - Score. If Score < 0, it means decoherence is complete (Error ~ 1.0)
        # We clamp scores to [0, 1] for physical meaning
        
        df['UDD_Score_Clipped'] = df['UDD_Score'].clip(0, 1)
        df['SFE_Score_Clipped'] = df['SFE_Score'].clip(0, 1)
        
        df['Error_UDD'] = 1.0 - df['UDD_Score_Clipped']
        df['Error_SFE'] = 1.0 - df['SFE_Score_Clipped']
        
        # Calculate Reduction Rate: (Error_UDD - Error_SFE) / Error_UDD
        # Avoid division by zero
        df['Reduction_Rate'] = 0.0
        mask = df['Error_UDD'] > 1e-6
        df.loc[mask, 'Reduction_Rate'] = (df.loc[mask, 'Error_UDD'] - df.loc[mask, 'Error_SFE']) / df.loc[mask, 'Error_UDD'] * 100.0
        
        avg_reduction = df['Reduction_Rate'].mean()
        max_reduction = df['Reduction_Rate'].max()
        
        print("\n=== IBM Fez Simulation Error Analysis ===")
        print(f"Average Error Reduction: {avg_reduction:.2f}%")
        print(f"Maximum Error Reduction: {max_reduction:.2f}%")
        
        # Select noise level 0.15 for detailed view
        target_noise = 0.15
        subset = df[df['NoiseAmp'] == target_noise]
        
        if subset.empty:
             # Fallback to median noise if 0.15 not found
             noise_levels = df['NoiseAmp'].unique()
             target_noise = noise_levels[len(noise_levels)//2]
             subset = df[df['NoiseAmp'] == target_noise]

        print(f"\nDetailed Stats at Noise Amp = {target_noise}:")
        print(subset[['PulseCount', 'Error_UDD', 'Error_SFE', 'Reduction_Rate']].to_string(index=False))
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Error Rates at target noise
        plt.subplot(1, 2, 1)
        plt.plot(subset['PulseCount'], subset['Error_UDD'], 'o-', label='UDD (Baseline)')
        plt.plot(subset['PulseCount'], subset['Error_SFE'], 's-', label='SFE (Optimized)')
        plt.title(f"Logical Error Rate (Noise={target_noise})")
        plt.xlabel("Pulse Count")
        plt.ylabel("Error Rate (1 - Coherence)")
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Heatmap of Reduction Rate
        plt.subplot(1, 2, 2)
        pivot_table = df.pivot(index='NoiseAmp', columns='PulseCount', values='Reduction_Rate')
        sns.heatmap(pivot_table, annot=False, cmap="viridis", cbar_kws={'label': 'Error Reduction (%)'})
        plt.title("Error Reduction Heatmap")
        plt.tight_layout()
        
        plt.savefig('examples/results/error_reduction_analysis.png')
        print("\nPlot saved to examples/results/error_reduction_analysis.png")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_error_rates()
