import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_ablation_means(csv_path, metric):
    df = pd.read_csv(csv_path)
    # Extract base, tta2, tta4, tta8 means for the specific metric
    # HD95 needs dropna()
    if metric == 'hd95':
        return [df[f'base_{metric}'].dropna().mean(), df[f'tta2_{metric}'].dropna().mean(), 
                df[f'tta4_{metric}'].dropna().mean(), df[f'tta8_{metric}'].dropna().mean()]
    else:
        return [df[f'base_{metric}'].mean(), df[f'tta2_{metric}'].mean(), 
                df[f'tta4_{metric}'].mean(), df[f'tta8_{metric}'].mean()]

def plot_ablation_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    views = ['1 (Base)', '2-View', '4-View', '8-View']
    
    models = [
        ('CVC UNet', 'results/ablation_cvc_unet.csv', '#1f77b4', 'o', '-'),
        ('CVC SwinUNETR', 'results/ablation_cvc_swin.csv', '#ff7f0e', 's', '--'),
        ('BraTS SegResNet', 'results/ablation_brats.csv', '#2ca02c', '^', '-.')
    ]
    
    metrics = [('dice', 'Volumetric Overlap (Dice) ↑'), 
               ('bf1', 'Boundary Precision (BF1) ↑'), 
               ('hd95', 'Hausdorff Distance (HD95) ↓')]
               
    for i, (m_key, m_label) in enumerate(metrics):
        ax = axes[i]
        for name, path, color, marker, ls in models:
            if not os.path.exists(path): continue
            y_vals = load_ablation_means(path, m_key)
            ax.plot(views, y_vals, label=name, color=color, marker=marker, markersize=8, linestyle=ls, linewidth=2)
            
        ax.set_title(m_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of TTA Views', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        if i == 0:
            ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/figures/fig_ablation_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/fig_ablation_curves.png', dpi=300, bbox_inches='tight')

def plot_hd95_boxplots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    experiments = [
        ('results/cvc_results.csv', 'CVC-ClinicDB\nUNet (Pretrained)', axes[0]),
        ('results/swin_cvc_results.csv', 'CVC-ClinicDB\nSwinUNETR (From Scratch)', axes[1]),
        ('results/brats_results.csv', 'BraTS 2021\nSegResNet (Pretrained)', axes[2])
    ]
    
    for path, title, ax in experiments:
        if not os.path.exists(path): continue
        df = pd.read_csv(path).dropna(subset=['baseline_hd95', 'tta_hd95'])
        
        data = [df['baseline_hd95'], df['tta_hd95']]
        
        # Create boxplot
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, 
                        boxprops=dict(facecolor='#e6e6fa', color='#1a1a2e'),
                        medianprops=dict(color='#e94560', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#a3a3c2', markersize=4, alpha=0.5))
        
        # Color the TTA box differently
        bp['boxes'][1].set_facecolor('#d4f1f4')
        
        ax.set_xticklabels(['Baseline', 'TTA (8-View)'], fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('HD95 (Lower is Better)', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/figures/fig_hd95_boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/fig_hd95_boxplots.png', dpi=300, bbox_inches='tight')

def main():
    os.makedirs('results/figures', exist_ok=True)
    print("Generating Ablation Line Curves...")
    plot_ablation_curves()
    print("Generating HD95 Outlier Boxplots...")
    plot_hd95_boxplots()
    print("Done! Check results/figures/ for publication-ready PDFs and PNGs.")

if __name__ == '__main__':
    main()
