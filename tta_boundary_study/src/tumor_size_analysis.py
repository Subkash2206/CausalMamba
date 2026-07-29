import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from src.datasets.brats_dataset import BraTSDataset
import os

def main():
    print("Loading BraTS Validation Dataset to calculate tumor volumes...")
    ds = BraTSDataset('brats2021', split='val')
    
    volumes = []
    # Just grab the ground truth masks to count voxels
    for i in range(len(ds)):
        _, mask = ds[i]
        volumes.append(mask.sum().item())
        
    print("Loading evaluation results...")
    df = pd.read_csv('results/brats_results.csv')
    df['tumor_volume'] = volumes
    
    # Calculate HD95 delta (Negative = TTA reduced distance = Improvement)
    df['hd95_delta'] = df['tta_hd95'] - df['baseline_hd95']
    
    # Drop cases where HD95 is NaN (empty masks/predictions)
    clean_df = df.dropna(subset=['hd95_delta', 'tumor_volume'])
    
    # Calculate correlation
    pearson_corr, p_val = pearsonr(clean_df['tumor_volume'], clean_df['hd95_delta'])
    spearman_corr, s_p_val = spearmanr(clean_df['tumor_volume'], clean_df['hd95_delta'])
    
    print(f"\n{'='*55}")
    print(f"Tumor Size vs HD95 Delta Correlation")
    print(f"{'='*55}")
    print(f"Pearson r:  {pearson_corr:.4f} (p-value: {p_val:.4e})")
    print(f"Spearman ρ: {spearman_corr:.4f} (p-value: {s_p_val:.4e})")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(clean_df['tumor_volume'], clean_df['hd95_delta'], alpha=0.6, color='#533483', edgecolors='k')
    plt.axhline(0, color='red', linestyle='--', label='No Change (Delta = 0)')
    
    # Trendline
    z = np.polyfit(clean_df['tumor_volume'], clean_df['hd95_delta'], 1)
    p = np.poly1d(z)
    plt.plot(clean_df['tumor_volume'], p(clean_df['tumor_volume']), "k--", alpha=0.7, label='Trendline')
    
    plt.title('Effect of TTA on HD95 by Tumor Volume\n(Negative HD95 Delta = TTA Improved Outliers)')
    plt.xlabel('Tumor Volume (Voxel Count)')
    plt.ylabel('HD95 Delta (TTA - Baseline)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/tumor_size_correlation.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved correlation plot to results/figures/tumor_size_correlation.png")

if __name__ == '__main__':
    main()
