import torch
import numpy as np
from src.datasets.brats_dataset import BraTSDataset
from monai.networks.nets import SegResNet
from src.metrics.boundary_metrics import dice_score, boundary_f1

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegResNet(in_channels=4, out_channels=1, init_filters=16).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/segresnet_brats_best.pth', map_location=DEVICE))
    model.eval()
    
    ds = BraTSDataset('brats2021', split='val')
    
    print(f"{'Patient':<10} | {'Dice':<8} | {'BF1 (Thick=2)':<15} | {'BF1 (Thick=1)':<15}")
    print("-" * 60)
    
    with torch.no_grad():
        # Test the first 5 patients
        for i in range(5):
            img, mask = ds[i]
            mask_np = mask[0].numpy()
            
            # Skip empty masks
            if mask_np.sum() == 0:
                continue
                
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred_np = torch.sigmoid(logits)[0, 0].cpu().numpy()
            
            dice = dice_score(pred_np, mask_np)
            bf1_t2 = boundary_f1(pred_np, mask_np, thickness=2)
            bf1_t1 = boundary_f1(pred_np, mask_np, thickness=1)
            
            print(f"{i:<10} | {dice:.4f}   | {bf1_t2:.4f}          | {bf1_t1:.4f}")

if __name__ == '__main__':
    main()
