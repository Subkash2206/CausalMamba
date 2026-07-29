import torch
from monai.networks.nets import SwinUNETR

def get_swin_unetr():
    model = SwinUNETR(
        in_channels=3,
        out_channels=1,
        spatial_dims=2
    )
    return model
