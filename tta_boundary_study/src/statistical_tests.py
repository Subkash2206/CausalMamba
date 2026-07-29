import pandas as pd
from scipy.stats import ttest_rel
import os

def analyze_results(csv_path, experiment_name):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}")
    print(f"Statistical Analysis: {experiment_name}")
    print(f"{'='*50}")
    
    metrics = [
        ('Dice', 'baseline_dice', 'tta_dice'),
        ('Boundary F1', 'baseline_bf1', 'tta_bf1'),
        ('HD95', 'baseline_hd95', 'tta_hd95')
    ]
    
    for name, base_col, tta_col in metrics:
        # Drop NaNs (crucial for HD95)
        clean_df = df.dropna(subset=[base_col, tta_col])
        base_vals = clean_df[base_col]
        tta_vals = clean_df[tta_col]
        
        # Paired T-Test
        t_stat, p_val = ttest_rel(base_vals, tta_vals)
        
        delta_mean = (tta_vals - base_vals).mean()
        delta_std = (tta_vals - base_vals).std()
        
        # Significance markers
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"{name}:")
        print(f"  Delta:   {delta_mean:+.4f} ± {delta_std:.4f}")
        print(f"  P-Value: {p_val:.4e} ({sig})\n")

def main():
    analyze_results('results/cvc_results.csv', 'CVC-ClinicDB (UNet - Pretrained)')
    analyze_results('results/swin_cvc_results.csv', 'CVC-ClinicDB (SwinUNETR - From Scratch)')
    analyze_results('results/brats_results.csv', 'BraTS 2021 (SegResNet - Pretrained)')

if __name__ == '__main__':
    main()
