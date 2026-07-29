import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.getcwd())

from models.vmunet.vmunet import VMUNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if device.type != 'cuda':
        print('Warning: CUDA not found. Mamba models usually REQUIRE CUDA for selective_scan.')

    # 1. Initialize model
    print('Initializing VMUNet...')
    model = VMUNet().to(device)
    
    # 2. Load checkpoint
    ckpt_path = 'best-ckpt/best-vmunet-isic18.pth'
    print(f'Loading checkpoint from {ckpt_path}...')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully.')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
    
    model.eval()

    # 3. Register forward hooks
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                features[name] = output[0].detach().cpu()
            else:
                features[name] = output.detach().cpu()
        return hook

    print('Registering hooks on VSSBlock layers...')
    hook_count = 0
    for name, module in model.named_modules():
        if 'VSSBlock' in str(type(module)):
            module.register_forward_hook(get_hook(f'hook_{hook_count:02d}_{name}'))
            hook_count += 1
    print(f'Registered {hook_count} hooks.')

    # 4. Load and preprocess image
    img_path = 'data/isic18/train/images/ISIC_0000000.jpg'
    print(f'Loading image from {img_path}...')
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'Error loading image: {e}')
        return
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    print(f'Input tensor shape: {input_tensor.shape}')

    # 5. Inference
    print('Running inference...')
    with torch.no_grad():
        output = model(input_tensor)
    print(f'Model output shape: {output.shape}')

    # 6. Print captured features
    print('\nCaptured Feature Maps (from VSSBlocks):')
    print('-' * 80)
    print(f'{"Hook ID":<10} | {"Layer Name":<50} | {"Shape":<20}')
    print('-' * 80)
    for key in sorted(features.keys()):
        parts = key.split('_')
        hook_id = parts[1]
        layer_name = '_'.join(parts[2:])
        shape_str = str(list(features[key].shape))
        print(f'{hook_id:<10} | {layer_name:<50} | {shape_str:<20}')
    print('-' * 80)

if __name__ == '__main__':
    main()
