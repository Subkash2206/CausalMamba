import pandas as pd
from scipy.stats import ttest_rel
import os

def run_ablation_stats(csv_path, experiment_name):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"\n{'='*65}")
    print(f"Ablation Statistical Analysis: {experiment_name}")
    print(f"{'='*65}")
    
    metrics = ['dice', 'bf1', 'hd95']
    views = [('2-View', 'tta2'), ('4-View', 'tta4'), ('8-View', 'tta8')]
    
    for m in metrics:
        print(f"\n--- Metric: {m.upper()} ---")
        for view_name, v_col in views:
            base_col = f'base_{m}'
            tta_col = f'{v_col}_{m}'
            
            clean_df = df.dropna(subset=[base_col, tta_col])
            base_vals = clean_df[base_col]
            tta_vals = clean_df[tta_col]
            
            if len(clean_df) < 2:
                continue
                
            t_stat, p_val = ttest_rel(base_vals, tta_vals)
            delta_mean = (tta_vals - base_vals).mean()
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns "
            
            print(f"  {view_name}: Delta = {delta_mean:+.4f} | p-value = {p_val:.4e} ({sig})")

def main():
    run_ablation_stats('results/ablation_cvc_unet.csv', 'CVC-ClinicDB (UNet - SOTA)')
    run_ablation_stats('results/ablation_cvc_swin.csv', 'CVC-ClinicDB (SwinUNETR - Weak)')
    run_ablation_stats('results/ablation_brats.csv', 'BraTS 2021 (SegResNet - SOTA)')

if __name__ == '__main__':
    main()
